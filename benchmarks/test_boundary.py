"""
Benchmarks for ``triwarp.boundary.boundary_loops``.

Scan-mesh holes are short loops, so the registry meshes mostly measure the surrounding
edge/grouping pipeline. The synthetic open-cylinder case (two rims of 2^16 vertices each) is
the asymptotic demonstration: the pre-fix per-vertex successor walk is O(L^2) total on a loop
of length L, while pointer-jumping list ranking is O(L log L).
"""

from __future__ import annotations

import igl
import pytest
import trimesh as tm
from conftest import RIM_LONG, BenchCase

import triwarp as tw


@pytest.mark.benchmark(group="boundary_loops")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_boundary_loops(bench_case: BenchCase) -> None:
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


@pytest.mark.benchmark(group="boundary_loops_long")
@pytest.mark.benchmeshes("synthetic_cylinder")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_boundary_loops_long_boundary(bench_case: BenchCase) -> None:
    """Synthetic long-boundary case: two rims of 2^16 vertices each."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loops = bench_case.run(lambda: tw.boundary.boundary_loops(vertices, faces))
        assert len(loops) == 2
        assert all(int(loop.shape[0]) == RIM_LONG for loop in loops)
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).outline())
        assert result is not None
    else:
        faces = bench_case.faces_np
        loop = bench_case.run(lambda: igl.boundary_loop(faces))
        assert int(loop.shape[0]) == RIM_LONG
