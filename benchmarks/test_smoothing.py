"""
Benchmarks for ``triwarp.smoothing.filter_mut_dif_laplacian``.

This is the iterative filter whose loop synced a full-array host sum per iteration before the
device-mean fix. The Laplacian operator is precomputed outside the timed callable so the
timing isolates the iteration loop. The trimesh reference is capped at ``bunny``: its CPU
loop takes tens of seconds on ``dragon``.
"""

from __future__ import annotations

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
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("volume_constraint", [False, True], ids=["novol", "vol"])
def test_filter_mut_dif_laplacian(bench_case: BenchCase, volume_constraint: bool) -> None:
    skip_larger_than(bench_case, "dragon")
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
            tm.smoothing.filter_mut_dif_laplacian(
                mesh, iterations=_ITERATIONS, volume_constraint=volume_constraint
            )
            return mesh

        result = bench_case.run(run)
        assert result.vertices.shape == vertices.shape
