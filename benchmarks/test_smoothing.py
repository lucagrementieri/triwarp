"""
Benchmarks for ``triwarp.smoothing.filter_mut_dif_laplacian``.

This is the iterative filter whose loop synced a full-array host sum per iteration before the
device-mean fix. The Laplacian operator is precomputed outside the timed callable so the
timing isolates the iteration loop. The trimesh reference is capped at ``bunny``: its CPU
loop takes tens of seconds on ``dragon``.

**open3d**'s ``filter_smooth_laplacian`` runs the same number of uniform-weight Laplacian
iterations, so it is the reference for the ``novol`` case. It has no volume-constraint variant
(``filter_smooth_taubin`` alternates two Laplacian passes to limit shrinkage, which is a different
scheme), so the ``vol`` case stays triwarp/trimesh only. Open3D returns a new mesh, so the shared
mesh is reusable across rounds.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_ITERATIONS = 10

_operator_cache: dict[tuple[str, str], wps.BsrMatrix[wp.float32]] = {}


def _laplacian_operator(bench_case: BenchCase) -> wps.BsrMatrix[wp.float32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_cache:
        _operator_cache[key] = tw.laplacian.laplacian(bench_case.vertices_wp, bench_case.faces_wp)
    return _operator_cache[key]


@pytest.mark.benchmark(group="filter_mut_dif_laplacian")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("volume_constraint", [False, True], ids=["novol", "vol"])
def test_filter_mut_dif_laplacian(bench_case: BenchCase, volume_constraint: bool) -> None:
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "open3d":
        if volume_constraint:
            pytest.skip("open3d has no volume-constrained Laplacian smoother")
        mesh_o3d = bench_case.mesh_o3d
        smoothed = bench_case.run(
            lambda: mesh_o3d.filter_smooth_laplacian(number_of_iterations=_ITERATIONS)
        )
        assert len(smoothed.vertices) == bench_case.n_vertices
        return
    if bench_case.kind == "triwarp" and bench_case.device == "cpu" and volume_constraint:
        # Native abort inside the volume-constraint path on the CPU device (Warp 1.15);
        # under investigation alongside the device-mean fix.
        pytest.skip("volume-constraint path aborts on the CPU device")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: tw.smoothing.filter_mut_dif_laplacian(
                vertices,
                faces,
                iterations=_ITERATIONS,
                volume_constraint=volume_constraint,
                laplacian_operator=operator,
            )
        )
        assert result.shape == vertices.shape
    else:  # trimesh mutates the mesh in place: rebuild it inside the timed callable
        skip_larger_than(bench_case, "bunny", "trimesh CPU loop takes tens of seconds on dragon")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            # trimesh's reference normalizes vertex normals via a scipy-sparse divide that
            # hits 1/0 on degenerate (zero-length) normals; the result is finite and
            # unused, so silence the third-party warning rather than let it leak.
            with np.errstate(divide="ignore", invalid="ignore"):
                tm.smoothing.filter_mut_dif_laplacian(
                    mesh, iterations=_ITERATIONS, volume_constraint=volume_constraint
                )
            return mesh

        result = bench_case.run(run)
        assert result.vertices.shape == vertices.shape
