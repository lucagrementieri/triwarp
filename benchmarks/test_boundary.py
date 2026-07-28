"""
Benchmarks for ``triwarp.boundary.boundary_loops``.

Axis: **loops**. Boundary extraction is not driven by mesh size -- ``sphere_med`` has 81 920 faces
and no boundary at all, and costs less than the 1 024-face ``rim_short``. What it is driven by is
the *shape* of the boundary, along two independent directions that the axis separates:

- **loop length** decides the ranking work. ``rim_long``'s two rims of 65 536 vertices are the
  asymptotic case: a per-vertex successor walk is O(L^2) on a loop of length L, while the Wyllie
  pointer jumping the implementation uses is O(L log L), one kernel launch per round.
- **loop count** decides the host work. Each loop used to cost a slice of a read-back offset table
  plus its own ``wp.clone``, so ``holes_many``'s 512 three-vertex loops were *more* expensive than
  ``rim_long``'s two enormous ones despite carrying a quarter of the boundary vertices.

Measured medians (RTX 5090, ``--device=cuda``): **0.35 / 2.4 / 4.1 ms** for no boundary / two long
rims / 512 short loops. The spread over an unchanged face count is the point, and it is what said
the per-loop host sequence was the thing to batch rather than the ranking: this group read
0.38 / 3.4 / 12.2 ms -- a 32x spread with the *wrong* end on top -- before ``boundary_loops`` became
a slicing wrapper over the packed ``boundary_loops_batched``, which extracts every loop in one pass
and hands back views instead of ``k`` clones.

References
----------
**trimesh**'s ``Trimesh.outline()`` is the equivalent and is rebuilt inside the timed callable,
since it caches its internals on the mesh. **libigl**'s ``boundary_loop`` returns only the longest
loop, so it is doing strictly less work on ``holes_many`` -- noted rather than corrected, because
the alternative is not comparing against it at all.

**open3d** has no boundary-loop extraction: it can report *which* edges are boundary edges
(``get_non_manifold_edges(allow_boundary_edges=False)``) but never orders them into loops, which is
the whole cost of this function.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw


@pytest.mark.benchmark(group="boundary_loops")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_boundary_loops(bench_case: BenchCase) -> None:
    """Ranking plus batched extraction, across no boundary / two long rims / many short loops."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loops = bench_case.run(lambda: tw.boundary.boundary_loops(vertices, faces))
        assert isinstance(loops, list)
    elif bench_case.kind == "trimesh":  # rebuild inside: trimesh caches outline internals
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).outline())
        assert result is not None
    else:  # igl returns only the longest loop
        faces = bench_case.faces_np
        bench_case.run(lambda: igl.boundary_loop(faces))


@pytest.mark.benchmark(group="boundary_edges")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_boundary_edges(bench_case: BenchCase) -> None:
    """
    The unordered predecessor of ``boundary_loops``: the edge sort without the ranking.

    Subtracting this group from ``boundary_loops`` separates the two costs, which is what says
    whether a regression is in the sort (scales with faces) or in the loop extraction (scales with
    loop count). It reads flat at ~0.4-0.5 ms across the whole axis, so everything above it in
    ``boundary_loops`` is ranking and extraction.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        edges = bench_case.run(lambda: tw.boundary.boundary_edges(vertices, faces))
        assert edges.ndim == 2
    else:  # the pure trimesh.grouping path, not a cached Trimesh property
        faces_np = bench_case.faces_np

        def boundary_tm() -> np.ndarray:
            edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
            return edges_np[tm.grouping.group_rows(edges_np, require_count=1)]

        assert bench_case.run(boundary_tm).ndim == 2
