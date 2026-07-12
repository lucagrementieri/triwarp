"""
Benchmarks for ``triwarp.boundary.boundary_loops``.

Scan-mesh holes are short loops, so the registry meshes mostly measure the surrounding
edge/grouping pipeline. The synthetic open-cylinder case (two rims of 2^16 vertices each) is
the asymptotic demonstration: the pre-fix per-vertex successor walk is O(L^2) total on a loop
of length L, while pointer-jumping list ranking is O(L log L).
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase

import triwarp as tw

_RIM = 1 << 16


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


def _open_cylinder(rim: int) -> tuple[np.ndarray, np.ndarray]:
    """Open tube with two boundary rims of ``rim`` vertices each (2*rim triangles)."""
    angle = 2.0 * np.pi * np.arange(rim) / rim
    ring = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(rim)))
    vertices = np.vstack((ring, ring + np.array([0.0, 0.0, 1.0]))).astype(np.float32)
    j = np.arange(rim)
    k = (j + 1) % rim
    lower = np.column_stack((j, k, j + rim))
    upper = np.column_stack((j + rim, k, k + rim))
    return vertices, np.vstack((lower, upper)).astype(np.int32)


@pytest.mark.benchmark(group="boundary_loops_long")
@pytest.mark.parametrize("mesh_name", ["synthetic_cylinder"])  # label only (group-by needs it)
def test_boundary_loops_long_boundary(benchmark, mesh_name: str) -> None:
    """Synthetic long-boundary case (triwarp only, not registry-mesh-parametrized)."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    vertices_np, faces_np = _open_cylinder(_RIM)
    vertices = wp.array(vertices_np, dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)

    def target() -> list:
        loops = tw.boundary.boundary_loops(vertices, faces)
        if device.startswith("cuda"):
            wp.synchronize_device(device)
        return loops

    loops = benchmark.pedantic(target, rounds=5, warmup_rounds=1, iterations=1)
    assert len(loops) == 2
    assert all(int(loop.shape[0]) == _RIM for loop in loops)
