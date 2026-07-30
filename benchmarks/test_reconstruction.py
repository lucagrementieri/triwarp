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

**pymeshlab** is the third implementation of both, and the reason it is worth a row is that in each
case it wraps *the same upstream code as one of the other two*:
``generate_surface_reconstruction_ball_pivoting`` is VCGlib's original BPA (Bernardini was
co-authored out of that lab) and ``generate_surface_reconstruction_screened_poisson`` is Kazhdan's
own implementation, which is also what open3d wraps. So the pymeshlab-versus-open3d gap on the
Poisson row is two wrappers over one solver, and only triwarp's is a different program. Its BPA is
given the identical absolute radius via ``PureValue`` (the ``0%`` default autoguesses one, which
would compare two different parameters) with ``clustering=0`` to disable the merge-nearby-vertices
step triwarp does not do. Both filters push their output as a new layer without touching the cloud,
so the cloud MeshSet is cached; ``set_current_mesh(0)`` inside the callable restores the layer the
previous round's push moved away from.

**Its cloud is not quite the same cloud**, and that is forced rather than chosen. Screened Poisson
rejects a point set carrying *any* null normal outright -- ``Failed to apply filter: Filter requires
correct per vertex normals`` -- and every scan mesh has unreferenced vertices, whose area-weighted
normal is exactly zero: **47 of bunny_decimated's 8 171 and 1 113 of bunny's 35 947**. So the
pymeshlab cloud drops those points, leaving it 0.6% / 3.1% smaller than the one triwarp and open3d
reconstruct from. The alternative, ``preclean=True``, moves the same cleaning *inside* the timed
filter, which is worse: it puts a pass triwarp does not run into the measured region.

``generate_surface_reconstruction_vcg`` was tried as a *fourth* algorithm and **rejected**: it
returns **zero faces** on this input at every voxel size probed (``PureValue`` 0.02 / 0.05 / 0.10),
reporting ``Mesh Saved 'plymcout.ply': 0 vertices, 0 faces``. It reconstructs from all *visible*
layers through a temporary ``.vmi`` file and did not produce geometry from a single oriented cloud;
a row that silently measures a no-op is worse than no row.

``triangulate_point_cloud`` has no open3d or pymeshlab counterpart: it is a port of MeshLib's
local-fan triangulation, and the nearest analogues (open3d's
``create_from_point_cloud_alpha_shape``, MeshLab's ``generate_alpha_shape``) are a different
algorithm solving the problem a different way, so timing them against each other would compare
algorithm choices rather than implementations.

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

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

if TYPE_CHECKING:
    import open3d as o3d

# MeshLib's default neighbour count for local fan triangulation.
_NUM_NEIGHBOURS = 18

# Ball radius as a multiple of the mean edge length (a proxy for point spacing).
_BPA_RADIUS_FRACTION = 1.5

# Octree depths of ``triwarp.reconstruction.screened_poisson``, mirrored on the open3d side. This
# is the module's dominant knob by a wide margin: ``dense`` mode is a ``2^depth`` cubed node grid,
# so each step is ~8x the nodes and the point count is almost secondary.
_POISSON_DEPTHS = [7, 9]

# A depth-9 dense solve and a ball-pivoting front are both seconds a call.
_HEAVY_ROUNDS = 3
_POISSON_DEPTH = 8

_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_cloud_o3d_cache: dict[str, o3d.geometry.PointCloud] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}


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


def _cloud_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the oriented point cloud as a face-less pymeshlab mesh, cached per mesh.

    Both reconstruction filters *push* their output onto the MeshSet rather than editing mesh 0, so
    the cloud itself is never touched -- but ``current_mesh()`` moves to the new layer, so the
    filter is always called on the cached set with mesh 0 still current from the previous round's
    push. ``set_current_mesh(0)`` restores that, and is part of the timed callable because it is the
    cost of driving the MeshSet API rather than of the reconstruction.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        # ``_normals`` goes through ``bench_case.vertices_wp``, which needs a Warp device -- and a
        # pymeshlab case has none. trimesh's ``vertex_normals`` is area-weighted too, which is the
        # same argument the open3d branch makes for using open3d's own.
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        normals_np = np.asarray(mesh_tm.vertex_normals, dtype=np.float64)
        # Screened Poisson *rejects* a cloud carrying any null normal outright ("Filter requires
        # correct per vertex normals"), and every scan mesh has unreferenced vertices, whose
        # area-weighted normal is exactly zero -- 47 of bunny_decimated's 8 171 and 1 113 of bunny's
        # 35 947. Dropping them here rather than passing ``preclean=True`` keeps the cleaning out of
        # the timed region; the cost is that the reference reconstructs from 0.6% / 3.1% fewer
        # points than triwarp and open3d do, which is recorded in the module docstring.
        keep_np = np.linalg.norm(normals_np, axis=1) > 0.0
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(
            ml.Mesh(
                vertex_matrix=np.ascontiguousarray(
                    bench_case.vertices_np[keep_np], dtype=np.float64
                ),
                v_normals_matrix=np.ascontiguousarray(normals_np[keep_np], dtype=np.float64),
            )
        )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    meshset_pml = _cloud_pml_cache[bench_case.mesh_name]
    meshset_pml.set_current_mesh(0)
    return meshset_pml


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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
def test_ball_pivoting(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "ball pivoting above bunny dominates the suite")
    radius = _BPA_RADIUS_FRACTION * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        # VCGlib's original BPA, given the identical absolute radius via ``PureValue`` (its default
        # ``0%`` autoguesses one, which would compare two different algorithms' parameters).
        # ``clustering=0`` disables the merge-nearby-vertices step triwarp does not do.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.generate_surface_reconstruction_ball_pivoting(
                ballradius=ml.PureValue(radius), clustering=0.0
            ),
            rounds=_HEAVY_ROUNDS,
        )
        return
    if bench_case.kind == "triwarp":
        points, normals = bench_case.vertices_wp, _normals(bench_case)
        _vertices, faces = bench_case.run(
            lambda: tw.reconstruction.ball_pivoting(points, normals, radius=radius),
            rounds=_HEAVY_ROUNDS,
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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
@pytest.mark.parametrize("depth", _POISSON_DEPTHS)
@pytest.mark.parametrize("method", ["dense", "adaptive"])
def test_screened_poisson(
    bench_case: BenchCase, method: Literal["dense", "adaptive"], depth: int
) -> None:
    skip_larger_than(bench_case, "bunny", "screened Poisson above bunny dominates the suite")
    if bench_case.kind == "pymeshlab":
        if method != "dense":
            pytest.skip("MeshLab has a single screened-Poisson path; timed once under 'dense'")
        # Kazhdan's own code, the same implementation open3d wraps -- so its row prices MeshLab's
        # wrapper against open3d's over an identical solver, and only triwarp's is a different one.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.generate_surface_reconstruction_screened_poisson(depth=depth),
            rounds=_HEAVY_ROUNDS,
        )
        return
    if bench_case.kind == "open3d":
        if method != "dense":
            pytest.skip("open3d has a single screened-Poisson path; timed once under 'dense'")
        import open3d as o3d

        cloud = _cloud_o3d(bench_case)
        mesh_poisson, _density = bench_case.run(
            lambda: o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
        )
        assert len(mesh_poisson.triangles) > 0
        return
    _skip_cg_on_cpu(bench_case)
    points, normals = bench_case.vertices_wp, _normals(bench_case)
    _vertices, faces = bench_case.run(
        lambda: tw.reconstruction.screened_poisson(points, normals, depth=depth, method=method),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(faces.shape[0]) > 0


# Cell sizes for the resampling group, as a fraction of the bbox diagonal. 2% is MeshLab's own
# default; 1% is eight times the field evaluation, which is what makes the pair worth having.
_RESAMPLE_CELL_FRACTIONS = [0.02, 0.01]


@pytest.mark.benchmark(group="resample_uniform")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("cell_fraction", _RESAMPLE_CELL_FRACTIONS)
def test_resample_uniform(bench_case: BenchCase, cell_fraction: float) -> None:
    """
    Sample the signed distance field on a grid and march it: the one **cubic** group in the suite.

    Both rows get the identical absolute cell size, derived from the mesh's own bounding-box
    diagonal on the host so neither side computes its own. The pair is a slope check: halving the
    cell is 8x the lattice on both sides, and on ``bunny`` triwarp measures **9.9 to 13.5 ms** for
    it -- a 1.4x rise against an 8x lattice, so the field evaluation is *not* what dominates and the
    fixed marching and cleanup passes are. MeshLab rises 206 to 292 ms over the same step, also
    sublinear. Read a regression here as the slope steepening rather than the absolute number
    moving.

    MeshLab's ``offset`` parameter is passed as ``PureValue(0.0)``: as a ``PercentageValue`` it runs
    from full erosion at 0% to full dilation at 100%, so its own 50% default is the *zero* offset
    and ``PercentageValue(0)`` would erode the mesh away (measured: a unit sphere down to radius
    0.30).
    """
    diagonal = float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    voxel_size = cell_fraction * diagonal
    if bench_case.kind == "pymeshlab":
        # ``generate_*`` pushes a new mesh onto the set, so the MeshSet is rebuilt per round.
        skip_larger_than(bench_case, "bunny", "MeshLab marches the whole lattice on one core")
        new_meshset_pml = bench_case.new_meshset_pml

        def resample_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.generate_resampled_uniform_mesh(
                cellsize=ml.PureValue(voxel_size), offset=ml.PureValue(0.0)
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(resample_pml, rounds=_HEAVY_ROUNDS) > 0
        return
    skip_larger_than(bench_case, "bunny", "a 1% lattice over dragon is 8 GB of field")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _out_vertices, out_faces = bench_case.run(
        lambda: tw.reconstruction.resample_uniform(vertices, faces, voxel_size=voxel_size),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(out_faces.shape[0]) > 0
