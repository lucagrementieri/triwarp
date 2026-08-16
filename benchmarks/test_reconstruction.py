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

  What is left is ``pivot_front_edges`` (90.8% of kernel time) and it is *occupancy*-bound, not
  throughput-bound: a wave has a few hundred live front edges, so a few hundred threads do
  serial, dependent hash-grid work on a device with 350k thread slots. ``bunny_decimated`` is
  still behind for exactly that reason — it is too small to fill the GPU — while ``bunny``,
  four times larger, fares much better.

  **A warp per edge was the next step, and it landed.** The wave cost used to be nearly flat in the
  front size (0.425 ms at a front under 64 against 1.636 at 1024-4096 — a 250x range of work for
  4.3x of cost), which is the signature of a device left idle: a wave has only a few hundred live
  edges. The per-edge work is ~215 point tests (the outer query enumerates 87.6 candidates, 59.0
  pass the prefilter, 8.93 run the acceptance test, each walking a second ball of 14.2 points).

  What blocked the obvious fix was the *index*, not the kernel: Warp's hash grid exposes only a
  sequential per-thread iterator with no per-cell entry point, so its walk — **70-73%** of a query's
  cost — cannot be split across lanes. Warp's **BVH** can be, through ``tile_bvh_query_aabb``, and
  that turned out to be 2.4-8.9x over the hash grid at the front sizes a wave actually has. (A
  *serial* BVH walk is 1.8x **slower** than the hash grid, so the win is the cooperation, not the
  tree; and a hand-rolled hashed cell grid would be a further 2.2-5.5x that this kernel's own
  ceiling cannot spend.)

  So ``pivot_front_edges`` now runs one block per front edge, one warp per block, walking the BVH
  cooperatively, with the empty-ball test left as a per-lane serial hash-grid query — at that point
  each lane is testing a different ball, so there is nothing to cooperate on, and it gets its
  speedup from running 32-way concurrently instead. Measured back to back in one session:
  **116.7 -> 25.9 ms on ``bunny_decimated`` and 203.4 -> 41.6 ms on ``bunny`` (4.5x / 4.9x)**,
  taking the group from 3.78x behind pymeshlab to **1.34x ahead**, and from 1.54x behind to
  **3.65x ahead**.

Sizing notes measured on an RTX 5090 before the baseline was captured:

- ``ball_pivoting`` uses ``1.5 * mean_edge`` as its radius. A larger multiple used to overflow the
  ``4 * n + 16`` triangle budget and raise; the budget now doubles on demand (rehashing the edge
  table and rebuilding the front from it), so 2.5x completes. It is still not benchmarked, because
  a ball that wide searches a much larger neighbourhood per pivot and measures a different thing.
- ``screened_poisson`` in ``dense`` mode is dominated by the ``2^depth`` cubed node grid, not by the
  point count — it costs the same on ``bunny_decimated`` and ``bunny``. The ``adaptive`` mode does
  scale with the cloud. Both are timed at the default ``depth=8``.

Everything is capped at ``bunny``, and the cap is load-bearing rather than tidy: point-cloud
triangulation and ball pivoting are superlinear in the cloud size, and the CPU screened-Poisson
references are worse than superlinear in practice -- ``test_screened_poisson[dragon-open3d-*]`` was
measured running **93 minutes without completing a single round** before it was killed, which is
more wall clock than the other 40 benchmark modules combined. See the comment on that cap. This is
a deliberate coverage gap -- the optimizations these benchmarks gate are host-sync and
launch-overhead fixes, which show up at these sizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from conftest import BenchCase, BenchLibrary, skip_larger_than

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


# Point counts for the planar Delaunay sweep. It is a *sequential* incremental insertion followed
# by a parallel flip loop, so the axis worth showing is the point count rather than any mesh size.
_DELAUNAY_POINTS = [2_000, 20_000]

_delaunay_np_cache: dict[int, np.ndarray] = {}


def _delaunay_points_np(n_points: int) -> np.ndarray:
    """``(n, 2)`` uniform planar cloud, cached per size -- the *input*, not part of the work."""
    if n_points not in _delaunay_np_cache:
        _delaunay_np_cache[n_points] = np.ascontiguousarray(
            np.random.default_rng(0).random((n_points, 2)), dtype=np.float64
        )
    return _delaunay_np_cache[n_points]


@pytest.mark.benchmark(group="delaunay_triangulation")
@pytest.mark.benchlibs("triwarp", "scipy", "pyvista")
@pytest.mark.parametrize("n_points", _DELAUNAY_POINTS)
def test_delaunay_triangulation(bench_lib: BenchLibrary, n_points: int) -> None:
    """
    Planar Delaunay triangulation, on the point-count axis -- the module's one mesh-free group.

    It takes ``bench_lib`` rather than ``bench_case`` for the reason ``test_creation.py``'s groups
    do: there is no input mesh, so the work is sized by a plain ``parametrize`` (the same 2-D cloud
    both references see, built once per size outside the timed region).

    !!! note "This row was the suite's largest loss, and both halves of it are now compiled"
        **The seed.** 370 ms against scipy's 45.8 at 20 000 points, of which 337.8 ms (91.3 %) was
        the host-side seed -- a pure-Python sweep doing ~460 000 ``_orient2d`` calls (n insertions
        x a ~23-vertex hull boundary) while the device did 2.33 ms of work. Porting that sweep to a
        single-thread Warp **CPU** kernel took the row to 36.2 ms. The CPU device is not a
        concession: the identical sweep measures 352 ms in Python, 93 ms in one **CUDA** thread and
        1.40 ms in one CPU thread, so a single GPU thread is the wrong tool by a factor of 66.

        **The flip loop**, which then dominated the small row. Its cost was not the launch count
        the first reading blamed: at 962 us of host time per pass, 442 went to
        [`face_adjacency`][triwarp.adjacency.face_adjacency] and 186 to a *second* radix sort of
        the same edge keys ``face_adjacency`` had already sorted internally, against ~1.2 ms of
        device work for the whole 37-pass loop. Since a flip leaves the vertex, face and
        interior-edge counts alone, that whole working set is invariant and is now built once into
        fixed buffers by one sort and five launches. **36.1 -> 18.7 ms (1.93x)** at 20 000 points,
        a 2.4x win over scipy, and **18.7 -> 7.2 ms (2.61x)** at 2 000. See
        [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation].

    The three implementations answer the same question by different means: triwarp seeds a
    sequential lexicographic incremental triangulation and then drives its **parallel** edge-flip
    loop to the empty-circumcircle fixed point, scipy calls Qhull, and VTK runs
    ``vtkDelaunay2D``'s serial insertion. The answers agree on the interior and differ only in hull
    slivers -- pyvista's 374 triangles are a strict *subset* of triwarp's 384 on a 200-point cloud
    (``tests/test_reconstruction.py``), so read the counts as well as the clock.
    """
    points_np = _delaunay_points_np(n_points)
    if bench_lib.kind == "pyvista":
        cloud_pv = pv.PolyData(np.column_stack([points_np, np.zeros(n_points)]).astype(np.float64))
        triangulated_pv = bench_lib.run(cloud_pv.delaunay_2d)
        assert triangulated_pv.n_faces > 0
        return
    if bench_lib.kind == "scipy":
        from scipy.spatial import Delaunay

        triangulated_sp = bench_lib.run(lambda: Delaunay(points_np))
        assert triangulated_sp.simplices.shape[1] == 3
        return
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=bench_lib.device
    )
    faces = bench_lib.run(lambda: tw.reconstruction.delaunay_triangulation(points_wp))
    assert int(faces.shape[0]) % 3 == 0
    assert int(faces.shape[0]) > 0


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
        #
        # ``clustering`` stays at MeshLab's default 20%: at ``0`` the filter reconstructs
        # **nothing** -- measured 0 faces against 1 277 at the default, on the same cloud and the
        # same radius -- and returns in 9.6 ms against 2.7 ms, so this row previously timed a
        # failure and read *slower* for it. The clustering fraction is a seed-triangle spacing
        # floor, not an optional post-pass. At the default the two agree: 1 277 faces against
        # triwarp's 1 280, asserted in
        # tests/test_reconstruction.py::test_ball_pivoting_matches_pymeshlab.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.generate_surface_reconstruction_ball_pivoting(
                ballradius=ml.PureValue(radius), clustering=20.0
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
    # Do not lift this cap. open3d's CPU ``create_from_point_cloud_poisson`` is 7.5 s per call at
    # depth 9 on ``bunny``'s 35 947 points; ``dragon`` has 437 645, and the ``[dragon-open3d-*]``
    # rows were measured running for **93 minutes without completing a single round** (GPU idle,
    # 42 cores saturated) before being killed. That one parametrization is worth more wall clock
    # than the other 40 benchmark modules combined, and what it measures is open3d, not triwarp.
    skip_larger_than(
        bench_case, "bunny", "screened Poisson above bunny dominates the suite (93 min at dragon)"
    )
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
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab")
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

    **libigl's ``offset_surface`` is the same operation at ``isolevel=0``** -- sample the signed
    distance field on a grid, march it -- and its ``signed_distance_type`` is given
    ``PSEUDONORMAL``, the mode ``tests/test_proximity.py`` establishes agrees with triwarp's default
    sign to 8e-8. Its resolution parameter ``s`` is a *cell count along the longest axis*, not a
    length, so it receives ``round(longest_extent / voxel_size)`` -- the named transform that puts
    all three rows on the identical lattice. In-harness on ``bunny_decimated`` it reads 29.7 and
    40.3 ms over the cell pair and on ``bunny`` 117 and 119, so it is sublinear in the lattice like
    the other two and needs no cap of its own beyond the group's.
    """
    diagonal = float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    voxel_size = cell_fraction * diagonal
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the field is evaluated on one core")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        extent = float((vertices_np.max(axis=0) - vertices_np.min(axis=0)).max())
        resolution = max(2, round(extent / voxel_size))
        vertices_igl, faces_igl = bench_case.run(
            lambda: igl.offset_surface(
                vertices_np, faces_np, 0.0, resolution, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            )[:2],
            rounds=_HEAVY_ROUNDS,
        )
        assert faces_igl.shape[0] > 0
        assert vertices_igl.shape[1] == 3
        return
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
