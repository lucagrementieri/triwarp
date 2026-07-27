"""
Benchmarks for ``triwarp.reconstruction``.

The point cloud is a registry mesh's own vertices with area-weighted vertex normals — deterministic
(no sampling RNG), consistently oriented, and it scales with the mesh. Both are precomputed and
cached: they are the *input*, not part of the operation being timed.

**open3d** implements the same two published algorithms and is the reference for both:
``create_from_point_cloud_ball_pivoting`` (Bernardini's BPA, given the identical radius) and
``create_from_point_cloud_poisson`` (Kazhdan's screened Poisson, given the identical octree depth).
Both take an oriented ``PointCloud``; open3d gets the same points and computes its own area-weighted
vertex normals, which is the same quantity triwarp's
[`area_weighted_vertex_normals`][triwarp.vertices.area_weighted_vertex_normals] produces.

``triangulate_point_cloud`` has no open3d counterpart: it is a port of MeshLib's local-fan
triangulation, and open3d's nearest analogue (``create_from_point_cloud_alpha_shape``) is a
different algorithm solving the problem a different way, so timing them against each other would
compare algorithm choices rather than implementations.

What the open3d comparison showed when it was added (medians, RTX 5090, ``depth=8``,
``radius = 1.5 * mean_edge``):

- ``screened_poisson`` is a clear win — 134 ms (dense) against open3d's 2.0-2.3 s, so **15-25x
  faster** on both meshes.
- ``ball_pivoting`` used to be the outlier of this module at 580 ms / 1085 ms against open3d's
  100 ms / 445 ms. Rebuilding it around a **persistent front** — an edge hash table plus a
  compacted boundary-edge list, mutated in place by ``commit_triangles`` and never re-derived from
  the triangle soup — took it to **121 ms / 214 ms**, i.e. from 2.4x slower to **2.1x faster** on
  ``bunny``. Three things paid for that: retiring provably-dead front edges (96% of all pivots in
  a run were re-searches of edges already known to be impossible), caching each front edge's best
  candidate so the ~73% that lose the vertex claim each wave re-validate in O(1), and dropping the
  per-wave ``edges_unique`` + sort + three readbacks the front rebuild needed.

  It also reconstructs a **much better surface**: 1.97 faces per referenced vertex against 3.08,
  and 1.7% boundary edges against 23.6%. The old per-wave rebuild was letting colliding fronts
  triangulate a neighbourhood in overlapping layers, which cleanup then tore back into open
  patches. The run-to-run spread collapsed with it (148 ms StdDev -> 3 ms).

  What is left is ``pivot_front_edges`` (94% of kernel time) and it is *occupancy*-bound, not
  throughput-bound: a wave has a few thousand live front edges, so a few thousand threads do
  serial, dependent hash-grid work on a device with 350k thread slots. ``bunny_decimated`` is
  still 1.19x behind open3d for exactly that reason — it is too small to fill the GPU — while
  ``bunny``, four times larger, is 2.1x ahead. Going further means parallelising *within* an
  edge's candidate search (a warp per edge), not shaving the wave loop.

Sizing notes measured on an RTX 5090 before the baseline was captured:

- ``ball_pivoting`` uses ``1.5 * mean_edge`` as its radius. A larger multiple used to overflow the
  ``4 * n + 16`` triangle budget and raise; the budget now doubles on demand (rehashing the edge
  table and rebuilding the front from it), so 2.5x completes. It is still not benchmarked, because
  a ball that wide searches a much larger neighbourhood per pivot and measures a different thing.
- ``screened_poisson`` in ``dense`` mode is dominated by the ``2^depth`` cubed node grid, not by the
  point count — it costs the same on ``bunny_decimated`` and ``bunny``. The ``adaptive`` mode does
  scale with the cloud. Both are timed at the default ``depth=8``.

Everything is capped at ``bunny``: point-cloud triangulation and ball pivoting are superlinear in
the cloud size, and at ``dragon`` (871k points) they would dominate the whole suite. This is a
deliberate coverage gap — the optimizations these benchmarks gate are host-sync and launch-overhead
fixes, which show up at these sizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

if TYPE_CHECKING:
    import open3d as o3d

# MeshLib's default neighbour count for local fan triangulation.
_NUM_NEIGHBOURS = 18

# Ball radius as a multiple of the mean edge length (a proxy for point spacing).
_BPA_RADIUS_FRACTION = 1.5

# Octree depth of ``triwarp.reconstruction.screened_poisson``, mirrored on the open3d side.
_POISSON_DEPTH = 8

_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_cloud_o3d_cache: dict[str, o3d.geometry.PointCloud] = {}


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


def _cloud_o3d(bench_case: BenchCase) -> o3d.geometry.PointCloud:
    """
    Oriented open3d point cloud: the mesh vertices with area-weighted vertex normals.

    The same input triwarp gets, with the normals computed by open3d's own
    ``compute_vertex_normals`` (also area-weighted). Cached — it is the input, not the operation.
    """
    if bench_case.mesh_name not in _cloud_o3d_cache:
        import open3d as o3d

        mesh_o3d = bench_case.mesh_o3d
        mesh_o3d.compute_vertex_normals()
        cloud = o3d.geometry.PointCloud(mesh_o3d.vertices)
        cloud.normals = mesh_o3d.vertex_normals
        _cloud_o3d_cache[bench_case.mesh_name] = cloud
    return _cloud_o3d_cache[bench_case.mesh_name]


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
@pytest.mark.benchlibs("triwarp", "open3d")
def test_ball_pivoting(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "ball pivoting above bunny dominates the suite")
    radius = _BPA_RADIUS_FRACTION * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        points, normals = bench_case.vertices_wp, _normals(bench_case)
        _vertices, faces = bench_case.run(
            lambda: tw.reconstruction.ball_pivoting(points, normals, radius=radius)
        )
        assert int(faces.shape[0]) > 0
    else:  # open3d's BPA takes a radius list; give it the single identical radius
        import open3d as o3d

        cloud = _cloud_o3d(bench_case)
        radii = o3d.utility.DoubleVector([radius])
        mesh_bpa = bench_case.run(
            lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
        )
        assert len(mesh_bpa.triangles) > 0


@pytest.mark.benchmark(group="screened_poisson")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("method", ["dense", "adaptive"])
def test_screened_poisson(bench_case: BenchCase, method: Literal["dense", "adaptive"]) -> None:
    skip_larger_than(bench_case, "bunny", "screened Poisson above bunny dominates the suite")
    if bench_case.kind == "open3d":
        if method != "dense":
            pytest.skip("open3d has a single screened-Poisson path; timed once under 'dense'")
        import open3d as o3d

        cloud = _cloud_o3d(bench_case)
        mesh_poisson, _density = bench_case.run(
            lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=_POISSON_DEPTH
            )
        )
        assert len(mesh_poisson.triangles) > 0
        return
    _skip_cg_on_cpu(bench_case)
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.screened_poisson(points, normals, method=method)
    )
    assert int(faces.shape[0]) > 0
