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

**pymeshlab** is in the same position and appears in the ``boundary_edges`` group only:
``compute_selection_from_mesh_border`` marks the boundary *vertices* as a bool selection array,
which is the same find-the-boundary pass stopped one step short of an edge list -- and it has
nothing that orders the result into loops either. It is read-only apart from the selection bit, so
it shares the MeshSet.
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
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pymeshlab")
def test_boundary_edges(bench_case: BenchCase) -> None:
    """
    The unordered predecessor of ``boundary_loops``: the edge sort without the ranking.

    Subtracting this group from ``boundary_loops`` separates the two costs, which is what says
    whether a regression is in the sort (scales with faces) or in the loop extraction (scales with
    loop count). It reads flat at ~0.4-0.5 ms across the whole axis, so everything above it in
    ``boundary_loops`` is ranking and extraction.

    ``igl.boundary_facets`` is the one reference here that returns the same *thing* triwarp does --
    an ``(n_boundary, 2)`` edge list, plus the incident face and corner indices as second and third
    returns, which is strictly more than triwarp's two columns. It is the right row for this group
    and not for ``boundary_loops``, where ``igl.boundary_loop`` returns only the longest loop.
    """
    if bench_case.kind == "igl":
        faces_np = bench_case.faces_np
        edges_igl, _face_igl, _corner_igl = bench_case.run(lambda: igl.boundary_facets(faces_np))
        assert edges_igl.ndim == 2
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        edges = bench_case.run(lambda: tw.boundary.boundary_edges(vertices, faces))
        assert edges.ndim == 2
    elif bench_case.kind == "pymeshlab":
        # ``compute_selection_from_mesh_border`` marks the boundary *vertices* rather than returning
        # the edge pairs, so it does the same find-the-boundary pass and stops one step earlier.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.compute_selection_from_mesh_border)
        assert meshset_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
    else:  # the pure trimesh.grouping path, not a cached Trimesh property
        faces_np = bench_case.faces_np

        def boundary_tm() -> np.ndarray:
            edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)
            return edges_np[tm.grouping.group_rows(edges_np, require_count=1)]

        assert bench_case.run(boundary_tm).ndim == 2


@pytest.mark.benchmark(group="ears")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp", "igl")
def test_ears(bench_case: BenchCase) -> None:
    """
    Ear triangles -- two boundary edges each -- as ``(face, opposite corner)`` pairs.

    ``hole_filling`` uses these to recognise a rim that closes with a single triangle, so the axis
    is the boundary shape rather than the mesh size, like the rest of the module. ``igl.ears``
    returns the identical pair of arrays, which is unusual enough to note: this is one of the few
    groups where triwarp and igl agree on the *output convention* and not merely the quantity, so
    the parity assert in ``tests/test_boundary.py`` needs only a row sort.
    """
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        ears, opposite = bench_case.run(lambda: tw.boundary.ears(faces))
        assert ears.shape == opposite.shape
        return
    faces_np = bench_case.faces_np
    ears_igl, opposite_igl = bench_case.run(lambda: igl.ears(faces_np))
    assert ears_igl.shape == opposite_igl.shape
