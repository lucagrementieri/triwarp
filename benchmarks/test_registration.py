"""
Benchmarks for ``triwarp.registration``.

Times the Procrustes fit and the ICP variants against a source cloud built from the mesh itself:
20k subsampled vertices pushed through a fixed rotation plus a 2%-of-diagonal translation and
0.2%-of-diagonal Gaussian noise. Deriving the source from the target means the correspondence
problem is well posed on every mesh, and the same perturbation goes to every library, so none of
them gets an easier problem.

Axis: **scale** for the ICP groups, and the scan sweep for ``procrustes``. ICP has two cost drivers
and the target's face count is only half of one of them: the per-iteration nearest-neighbour search
scales with the target, but the *number* of iterations scales with the initial misalignment, and
that is what a caller actually varies. So the ICP groups take the clean three-point ``scale`` axis
rather than the scan registry -- the largest scan meshes were adding ten minutes of wall clock for a
ratio the axis already establishes -- and ``icp_convergence`` sweeps the starting angle to measure
the other driver directly. ``procrustes`` keeps the scan sweep: it has no search at all, so it is a
pure throughput case and cheap everywhere.

References
----------
libigl exposes no Python binding for ``iterative_closest_point``, so **open3d** is the reference
here: ``open3d.pipelines.registration`` is what triwarp's point-to-plane path is ported from
(``TransformationEstimationPointToPlane`` with an optional ``RobustKernel``), and it is the only
CPU library in the test group that implements the point-to-plane metric at all.

* Procrustes — ``trimesh.registration.procrustes`` and open3d's
  ``TransformationEstimationPointToPoint.compute_transformation``, which is the same Kabsch fit on
  an explicit correspondence set.
* Point-to-point ICP — ``trimesh.registration.icp`` (cKDTree) and open3d's ``registration_icp``
  (KDTreeFlann).
* Point-to-plane ICP — open3d only, on a point-cloud target so both sides consume the *same*
  per-vertex normals (computed once with trimesh from the shared float64 source). The robust variant
  has no reference and is timed for triwarp alone.
* Mesh-target ICP — **pymeshlab**'s ``compute_matrix_by_icp_between_meshes``, which correspondences
  against the reference *mesh* rather than a point cloud and so is the equivalent of ``icp`` rather
  than of ``icp_point_cloud``. It is the only reference that group has. One constraint shapes it:
  **both layers must carry faces** -- a face-less source raises ``Failed to apply filter`` -- so its
  source is the whole mesh under the same rotation, translation and noise the point sample gets,
  with ``samplenum`` matched to triwarp's point count so both minimize over the same number of
  correspondences.

Both sides are pinned to exactly ``_ICP_ITERATIONS`` iterations — triwarp with ``threshold=-inf``
and open3d with ``relative_fitness=relative_rmse=0`` — otherwise a library that early-exits after
three iterations would look fast for the wrong reason. ``max_correspondence_distance`` is set to the
bbox diagonal so open3d rejects nothing, matching triwarp's ``max_distance=None`` default.

What is inside the timed callable
---------------------------------
Everything the public function does, including the spatial index build: triwarp's ``wp.Mesh`` BVH
for a mesh target, and open3d's ``KDTreeFlann`` for a point-cloud target. triwarp's public functions
take raw buffers, so there is no way to hoist that without benchmarking something other than the
API. Point-cloud construction (``Vector3dVector`` copies) and the shared vertex normals *are* setup
and cached outside the timed region.

The nearest-neighbour search dominates every ICP number here; the 6x6 solve and the tiled reductions
are a small fraction of the total, so the number to read for a kernel change is the before/after
delta on a *fixed* mesh, not the absolute time. Two things the absolute times do say, both measured
on ``bunny`` (35 947 target points, 20 000 source points, 10 iterations):

* The **mesh** target is still the faster of the two — 4.0 ms versus 7.9 ms — because it rides
  Warp's built-in ``wp.mesh_query_point_no_sign`` and never touches the k-NN path at all.
* The **point-cloud** target used to be 48x slower than that (191 ms), because
  [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] searched the whole cloud on every
  call: with ``max_radius=inf`` it clamped the query cube to the scene diagonal, so the BVH pruned
  nothing. It now deepens iteratively from a density estimate, and the target's BVH, bounds and
  radius are hoisted out of the loop, which moved point-cloud ICP from ~10x *slower* than open3d's
  serial ``KDTreeFlann`` to ~2.3x faster (7.9 ms versus 18.0 ms). The residual gap to the mesh
  target is per-iteration ``procrustes`` latency, not the search — which is why
  ``icp_point_to_plane``, whose iteration has no ``procrustes`` call, lands at 5.6 ms.
"""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_SEED = 42
_N_POINTS = 20_000
_ICP_ITERATIONS = 10

# Misalignments applied to the source cloud, in degrees. ICP's cost is the *actual* iteration
# count, not ``max_iterations``: it exits early once the cost improvement drops below ``threshold``,
# so a nearly-aligned pair converges in two or three passes where a badly aligned one runs the full
# schedule. That makes the starting angle the axis, and it is invisible to any mesh-size sweep.
_ROTATION_SWEEP = [5.0, 45.0]
_ROTATION_DEGREES = 5.0
_TRANSLATION_FRACTION = 0.02
_NOISE_FRACTION = 0.002

# Tukey cut-off for the robust point-to-plane fit, in fractions of the bbox diagonal. Passed
# explicitly (rather than letting triwarp derive it from the residual MAD) so triwarp and open3d
# minimize the same objective.
_TUKEY_FRACTION = 0.01

# MeshLab's ICP rebuilds its MeshSet per round on top of running ten serial iterations, so it lands
# in the hundreds of milliseconds where the other libraries are in the tens.
_PML_ROUNDS = 3

_source_np_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
_source_mesh_np_cache: dict[str, np.ndarray] = {}
_source_wp_cache: dict[tuple[str, str, float], wp.array] = {}
_normals_np_cache: dict[str, np.ndarray] = {}
_normals_wp_cache: dict[tuple[str, str], wp.array] = {}
_pcd_cache: dict[tuple[str, str], o3d.geometry.PointCloud] = {}


def _rotation_matrix(degrees: float = _ROTATION_DEGREES) -> np.ndarray:
    """Rodrigues rotation of ``degrees`` about the fixed axis ``(1, 2, 3)``."""
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.deg2rad(degrees)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _diagonal(bench_case: BenchCase) -> float:
    """Bounding-box diagonal — the length scale every parameter below is expressed in."""
    vertices = bench_case.vertices_np
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


def _source_np(
    bench_case: BenchCase, degrees: float = _ROTATION_DEGREES
) -> tuple[np.ndarray, np.ndarray]:
    """``(source_points, target_indices)``: perturbed vertex subsample and the source rows."""
    key = (bench_case.mesh_name, degrees)
    if key not in _source_np_cache:
        rng = np.random.default_rng(_SEED)
        vertices = bench_case.vertices_np
        count = min(_N_POINTS, vertices.shape[0])
        indices = rng.choice(vertices.shape[0], size=count, replace=False)
        diagonal = _diagonal(bench_case)
        offset = _TRANSLATION_FRACTION * diagonal * np.array([1.0, -1.0, 0.5]) / np.sqrt(2.25)
        source = vertices[indices] @ _rotation_matrix(degrees).T + offset
        source += rng.normal(scale=_NOISE_FRACTION * diagonal, size=source.shape)
        _source_np_cache[key] = (np.ascontiguousarray(source), indices)
    return _source_np_cache[key]


def _source_mesh_np(bench_case: BenchCase) -> np.ndarray:
    """
    Apply the same misalignment ``_source_np`` gives its subsample to *every* vertex.

    MeshLab's ICP needs both layers to carry faces, so its source is the whole mesh rather than a
    point sample; keeping the transform and the noise identical is what makes the two rows
    comparable.
    """
    if bench_case.mesh_name not in _source_mesh_np_cache:
        rng = np.random.default_rng(_SEED)
        vertices = bench_case.vertices_np
        diagonal = _diagonal(bench_case)
        offset = _TRANSLATION_FRACTION * diagonal * np.array([1.0, -1.0, 0.5]) / np.sqrt(2.25)
        source = vertices @ _rotation_matrix(_ROTATION_DEGREES).T + offset
        source += rng.normal(scale=_NOISE_FRACTION * diagonal, size=source.shape)
        _source_mesh_np_cache[bench_case.mesh_name] = np.ascontiguousarray(source, dtype=np.float64)
    return _source_mesh_np_cache[bench_case.mesh_name]


def _source_wp(bench_case: BenchCase, degrees: float = _ROTATION_DEGREES) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device), degrees)
    if key not in _source_wp_cache:
        _source_wp_cache[key] = wp.array(
            np.ascontiguousarray(_source_np(bench_case, degrees)[0], dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _source_wp_cache[key]


def _vertex_normals_np(bench_case: BenchCase) -> np.ndarray:
    """
    Area-weighted unit vertex normals, computed once with trimesh from the shared float64 source.

    The point-to-plane comparison hinges on both libraries fitting to the *same* tangent planes, so
    the normals are a shared input rather than something each library estimates for itself.
    """
    name = bench_case.mesh_name
    if name not in _normals_np_cache:
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        # ``np.array`` (not ``ascontiguousarray``): trimesh hands back a read-only ``TrackedArray``
        # view, and open3d's ``Vector3dVector`` rejects a non-writeable buffer.
        _normals_np_cache[name] = np.array(mesh_tm.vertex_normals, dtype=np.float64)
    return _normals_np_cache[name]


def _vertex_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_wp_cache:
        _normals_wp_cache[key] = wp.array(
            np.ascontiguousarray(_vertex_normals_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _normals_wp_cache[key]


def _pcd(bench_case: BenchCase, role: str) -> o3d.geometry.PointCloud:
    """Open3D point cloud for ``role`` (``"source"`` / ``"target"``), built once per mesh."""
    key = (bench_case.mesh_name, role)
    if key not in _pcd_cache:
        if role == "source":
            points = _source_np(bench_case)[0]
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        else:
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
            cloud.normals = o3d.utility.Vector3dVector(_vertex_normals_np(bench_case))
        _pcd_cache[key] = cloud
    return _pcd_cache[key]


def _o3d_criteria() -> o3d.pipelines.registration.ICPConvergenceCriteria:
    """Convergence criteria pinned to a fixed iteration count (no early exit)."""
    return o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.0, relative_rmse=0.0, max_iteration=_ICP_ITERATIONS
    )


@pytest.mark.benchmark(group="procrustes")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_procrustes(bench_case: BenchCase) -> None:
    """Procrustes fit on exact correspondences: tiled reductions plus the SVD kernel."""
    source_np, indices = _source_np(bench_case)
    if bench_case.kind == "triwarp":
        source = _source_wp(bench_case)
        target = wp.array(
            np.ascontiguousarray(bench_case.vertices_np[indices], dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
        matrix, _transformed, _cost = bench_case.run(
            lambda: tw.registration.procrustes(source, target, reflection=False, scale=False)
        )
        assert matrix.shape == (1,)
    elif bench_case.kind == "trimesh":
        target_np = np.ascontiguousarray(bench_case.vertices_np[indices])
        matrix_tm, _transformed, _cost = bench_case.run(
            lambda: tm.registration.procrustes(source_np, target_np, reflection=False, scale=False)
        )
        assert matrix_tm.shape == (4, 4)
    else:  # open3d: the same Kabsch fit, on an explicit index-pair correspondence set
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        )
        source_pcd = _pcd(bench_case, "source")
        target_pcd = _pcd(bench_case, "target")
        pairs = np.column_stack((np.arange(source_np.shape[0]), indices)).astype(np.int32)
        correspondences = o3d.utility.Vector2iVector(pairs)
        matrix_o3d = bench_case.run(
            lambda: estimation.compute_transformation(source_pcd, target_pcd, correspondences)
        )
        assert np.asarray(matrix_o3d).shape == (4, 4)


@pytest.mark.benchmark(group="icp_point_cloud")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_icp_point_cloud(bench_case: BenchCase) -> None:
    """Point-to-point ICP against a point-cloud target: BVH versus cKDTree versus KDTreeFlann."""
    if bench_case.kind == "triwarp":
        source, target = _source_wp(bench_case), bench_case.vertices_wp
        matrix, _transformed, _cost = bench_case.run(
            lambda: tw.registration.icp(
                source, target, max_iterations=_ICP_ITERATIONS, threshold=-math.inf
            )
        )
        assert matrix.shape == (1,)
    elif bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "trimesh icp rebuilds a cKDTree every iteration")
        source_np = _source_np(bench_case)[0]
        target_np = bench_case.vertices_np
        matrix_tm, _transformed, _cost = bench_case.run(
            lambda: tm.registration.icp(
                source_np,
                target_np,
                threshold=-math.inf,
                max_iterations=_ICP_ITERATIONS,
                scale=False,
                reflection=False,
            )
        )
        assert matrix_tm.shape == (4, 4)
    else:
        source_pcd, target_pcd = _pcd(bench_case, "source"), _pcd(bench_case, "target")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        )
        criteria, max_dist = _o3d_criteria(), _diagonal(bench_case)
        result = bench_case.run(
            lambda: o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_dist, np.eye(4), estimation, criteria
            )
        )
        assert np.asarray(result.transformation).shape == (4, 4)


@pytest.mark.benchmark(group="icp_convergence")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("degrees", _ROTATION_SWEEP, ids=["near5deg", "far45deg"])
def test_icp_convergence(bench_case: BenchCase, degrees: float) -> None:
    """
    Point-to-point ICP with early exit enabled, from a near and a far starting pose.

    Every other ICP group here pins ``threshold=-inf`` so all ``max_iterations`` run, which is what
    makes the cross-library comparison fair -- it measures *per-iteration* cost. This group does the
    opposite: it leaves the default threshold in place so the loop exits when it converges, which
    makes the timing report the *iteration count*. That is the number a caller actually pays, and
    it is a function of the initial misalignment, not of the mesh.
    """
    source, target = _source_wp(bench_case, degrees), bench_case.vertices_wp
    matrix, _transformed, _cost = bench_case.run(
        lambda: tw.registration.icp(source, target, max_iterations=_ICP_ITERATIONS)
    )
    assert matrix.shape == (1,)


@pytest.mark.benchmark(group="icp_mesh")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_icp_mesh(bench_case: BenchCase) -> None:
    """Point-to-point ICP against the triangle surface (closest-point-on-mesh correspondences)."""
    skip_larger_than(bench_case, "happy_buddha", "per-iteration mesh queries scale with face count")
    if bench_case.kind == "pymeshlab":
        # ``compute_matrix_by_icp_between_meshes`` correspondences run against the *reference mesh*
        # rather than a point cloud, which is what makes it the equivalent of ``icp`` here rather
        # than of ``icp_point_cloud``. Both layers must carry faces: handing it a face-less source
        # raises ``Failed to apply filter``, so the source is the whole mesh under the same
        # rotation / translation / noise the point sample gets rather than the sample itself.
        # ``samplenum`` is matched to triwarp's point count so both sides minimize over the same
        # number of correspondences. It writes the source layer's transform, so the set is rebuilt.
        source_mesh_np = _source_mesh_np(bench_case)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def icp_pml() -> None:
            meshset_pml = ml.MeshSet()
            faces_i32 = np.ascontiguousarray(faces_np, dtype=np.int32)
            meshset_pml.add_mesh(
                ml.Mesh(np.ascontiguousarray(vertices_np, dtype=np.float64), faces_i32)
            )
            meshset_pml.add_mesh(ml.Mesh(source_mesh_np, faces_i32))
            meshset_pml.compute_matrix_by_icp_between_meshes(
                referencemesh=0, sourcemesh=1, samplenum=min(_N_POINTS, vertices_np.shape[0])
            )

        bench_case.run(icp_pml, rounds=_PML_ROUNDS)
        return
    source, vertices, faces = _source_wp(bench_case), bench_case.vertices_wp, bench_case.faces_wp
    matrix, _transformed, _cost = bench_case.run(
        lambda: tw.registration.icp(
            source, vertices, faces, max_iterations=_ICP_ITERATIONS, threshold=-math.inf
        )
    )
    assert matrix.shape == (1,)


@pytest.mark.benchmark(group="icp_point_to_plane_cloud")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_icp_point_to_plane_cloud(bench_case: BenchCase) -> None:
    """Gauss-Newton point-to-plane ICP against a point-cloud target, on shared vertex normals."""
    if bench_case.kind == "triwarp":
        source, target = _source_wp(bench_case), bench_case.vertices_wp
        normals = _vertex_normals_wp(bench_case)
        matrix, _transformed, _cost = bench_case.run(
            lambda: tw.registration.icp_point_to_plane(
                source,
                target,
                target_normals=normals,
                max_iterations=_ICP_ITERATIONS,
                threshold=-math.inf,
            )
        )
        assert matrix.shape == (1,)
    else:
        source_pcd, target_pcd = _pcd(bench_case, "source"), _pcd(bench_case, "target")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        criteria, max_dist = _o3d_criteria(), _diagonal(bench_case)
        result = bench_case.run(
            lambda: o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_dist, np.eye(4), estimation, criteria
            )
        )
        assert np.asarray(result.transformation).shape == (4, 4)


@pytest.mark.benchmark(group="icp_point_to_plane_tukey")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_icp_point_to_plane_tukey(bench_case: BenchCase) -> None:
    """Robust point-to-plane ICP: triwarp's Tukey weight dispatch against open3d's ``TukeyLoss``."""
    tukey_k = _TUKEY_FRACTION * _diagonal(bench_case)
    if bench_case.kind == "triwarp":
        source, target = _source_wp(bench_case), bench_case.vertices_wp
        normals = _vertex_normals_wp(bench_case)
        matrix, _transformed, _cost = bench_case.run(
            lambda: tw.registration.icp_point_to_plane(
                source,
                target,
                target_normals=normals,
                max_iterations=_ICP_ITERATIONS,
                threshold=-math.inf,
                robust_kernel="tukey",
                robust_scale=tukey_k,
            )
        )
        assert matrix.shape == (1,)
    else:
        source_pcd, target_pcd = _pcd(bench_case, "source"), _pcd(bench_case, "target")
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=tukey_k)
        )
        criteria, max_dist = _o3d_criteria(), _diagonal(bench_case)
        result = bench_case.run(
            lambda: o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_dist, np.eye(4), estimation, criteria
            )
        )
        assert np.asarray(result.transformation).shape == (4, 4)


@pytest.mark.benchmark(group="icp_point_to_plane_mesh")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp")
def test_icp_point_to_plane_mesh(bench_case: BenchCase) -> None:
    """
    Point-to-plane ICP against the triangle surface: closest-face normals, no CPU equivalent.

    Also the only case that exercises the MAD-derived robust scale, since ``robust_scale`` is left
    to default here while the open3d comparison above pins it.
    """
    skip_larger_than(bench_case, "happy_buddha", "per-iteration mesh queries scale with face count")
    source, vertices, faces = _source_wp(bench_case), bench_case.vertices_wp, bench_case.faces_wp
    matrix, _transformed, _cost = bench_case.run(
        lambda: tw.registration.icp_point_to_plane(
            source,
            vertices,
            faces,
            max_iterations=_ICP_ITERATIONS,
            threshold=-math.inf,
            robust_kernel="tukey",
        )
    )
    assert matrix.shape == (1,)
