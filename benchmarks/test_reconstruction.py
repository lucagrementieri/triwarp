"""
Benchmarks for ``triwarp.reconstruction``.

The point cloud is a registry mesh's own vertices with area-weighted vertex normals — deterministic
(no sampling RNG), consistently oriented, and it scales with the mesh. Both are precomputed and
cached: they are the *input*, not part of the operation being timed.

There is no CPU reference in the harness ``LIBRARIES`` registry for these algorithms (the test
suite compares against ``open3d`` / ``pymeshlab`` / ``meshlib``, none of which the benchmark harness
registers), so every case here is ``triwarp``-only.

Sizing notes measured on an RTX 5090 before the baseline was captured:

- ``ball_pivoting`` uses ``1.5 * mean_edge`` as its radius. A larger multiple (2.5x) overflows the
  ``4 * n + 16`` triangle budget and raises, so it is not benchmarked; that overflow is a
  pre-existing robustness issue, not a benchmark artefact.
- ``screened_poisson`` in ``dense`` mode is dominated by the ``2^depth`` cubed node grid, not by the
  point count — it costs the same on ``bunny_decimated`` and ``bunny``. The ``adaptive`` mode does
  scale with the cloud. Both are timed at the default ``depth=8``.

Everything is capped at ``bunny``: point-cloud triangulation and ball pivoting are superlinear in
the cloud size, and at ``dragon`` (871k points) they would dominate the whole suite. This is a
deliberate coverage gap — the optimizations these benchmarks gate are host-sync and launch-overhead
fixes, which show up at these sizes.
"""

from __future__ import annotations

from typing import Literal

import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# MeshLib's default neighbour count for local fan triangulation.
_NUM_NEIGHBOURS = 18

# Ball radius as a multiple of the mean edge length (a proxy for point spacing).
_BPA_RADIUS_FRACTION = 1.5

_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}


def _skip_cg_on_cpu(bench_case: BenchCase) -> None:
    """Skip cases whose solver path needs CUDA (``warp.optim.linear.cg`` is NaN on CPU)."""
    assert bench_case.device is not None
    if wp.get_device(bench_case.device).is_cpu:
        pytest.skip("warp.optim.linear.cg returns NaN on the CPU device in Warp 1.14-1.15")


def _normals(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Area-weighted vertex normals for the point cloud, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _normals_cache[key] = tw.vertices.area_weighted_vertex_normals(
            int(vertices.shape[0]), vertices, faces
        )
    return _normals_cache[key]


@pytest.mark.benchmark(group="triangulate_point_cloud")
@pytest.mark.benchlibs("triwarp")
def test_triangulate_point_cloud(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "local triangulation above bunny dominates the suite")
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.triangulate_point_cloud(
            points, normals, num_neighbours=_NUM_NEIGHBOURS
        )
    )
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="ball_pivoting")
@pytest.mark.benchlibs("triwarp")
def test_ball_pivoting(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "ball pivoting above bunny dominates the suite")
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    radius = _BPA_RADIUS_FRACTION * bench_case.mean_edge
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.ball_pivoting(points, normals, radius=radius)
    )
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="screened_poisson")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("method", ["dense", "adaptive"])
def test_screened_poisson(bench_case: BenchCase, method: Literal["dense", "adaptive"]) -> None:
    skip_larger_than(bench_case, "bunny", "screened Poisson above bunny dominates the suite")
    _skip_cg_on_cpu(bench_case)
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.screened_poisson(points, normals, method=method)
    )
    assert int(faces.shape[0]) > 0
