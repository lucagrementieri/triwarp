"""
Regression tests for ``triwarp.metrics`` Chamfer and Hausdorff metrics.

Point-cloud metrics compare against SciPy (``KDTree`` for Chamfer,
``scipy.spatial.distance.directed_hausdorff`` for Hausdorff). Mesh-surface metrics
compare against libigl's ``igl.point_mesh_squared_distance`` (the primitive behind
``igl::hausdorff``); ``igl.hausdorff`` itself is not exposed in the Python bindings.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

import triwarp as tw
from tests.conversions import (
    points_to_open3d,
    points_to_pymeshlab,
    points_to_warp,
    trimesh_to_meshlib,
    trimesh_to_warp,
)

# igl requires float64 vertices / int64 faces; Warp uses float32 / int32, so mesh-surface
# references diverge from Warp at roughly float32 precision.
_MESH_RTOL = 1e-4
_MESH_ATOL = 1e-5


def _igl_point_mesh_sqr_dist(
    query_np: np.ndarray, vertices_np: np.ndarray, faces_np: np.ndarray
) -> np.ndarray:
    sqr_distances_igl, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(query_np, dtype=np.float64),
        np.ascontiguousarray(vertices_np, dtype=np.float64),
        np.ascontiguousarray(faces_np, dtype=np.int64),
    )
    return np.asarray(sqr_distances_igl, dtype=np.float64)


# ---------------------------------------------------------------------------
# Chamfer: point cloud to point cloud
# ---------------------------------------------------------------------------


def test_chamfer_points_to_points_mean(device: str) -> None:
    rng = np.random.default_rng(0)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = np.mean(distance_xy_np**2) + np.mean(distance_yx_np**2)

    chamfer_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_sum(device: str) -> None:
    rng = np.random.default_rng(1)
    x_np = (rng.random((50, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((70, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = np.sum(distance_xy_np**2) + np.sum(distance_yx_np**2)

    chamfer_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction="sum"
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-4)


def test_chamfer_points_to_points_max(device: str) -> None:
    rng = np.random.default_rng(2)
    x_np = (rng.random((50, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((70, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    distance_xy_np = KDTree(y_np).query(x_np)[0]
    distance_yx_np = KDTree(x_np).query(y_np)[0]
    chamfer_np = max(np.max(distance_xy_np**2), np.max(distance_yx_np**2))

    chamfer_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction="max"
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_single_directional(device: str) -> None:
    rng = np.random.default_rng(3)
    x_np = (rng.random((40, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((90, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    chamfer_np = np.mean(KDTree(y_np).query(x_np)[0] ** 2)

    chamfer_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), single_directional=True
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=1e-5, atol=1e-5)


def test_chamfer_points_to_points_unreduced(device: str) -> None:
    rng = np.random.default_rng(4)
    x_np = (rng.random((40, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((90, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    squared_xy_np = KDTree(y_np).query(x_np)[0] ** 2
    squared_yx_np = KDTree(x_np).query(y_np)[0] ** 2

    forward_wp, backward_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), point_reduction=None
    )
    assert np.allclose(forward_wp.numpy(), squared_xy_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(backward_wp.numpy(), squared_yx_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
@pytest.mark.parity("chamfer_points_to_points", "open3d")
def test_chamfer_points_to_points_matches_open3d(device: str, single_directional: bool) -> None:
    """
    Class B: Open3D returns the per-point nearest distances, not the Chamfer scalar.

    ``PointCloud.compute_point_cloud_distance`` is the *unsquared, unreduced* one-way answer, so the
    named transform is triwarp's own reduction applied to it -- square, then mean, and add the
    reverse direction when ``single_directional=False``. That is exactly what the benchmark's open3d
    branch does, so this asserts the two sides of that row compute the same number.
    """
    rng = np.random.default_rng(21)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    cloud_x_o3d, cloud_y_o3d = points_to_open3d(x_np), points_to_open3d(y_np)
    chamfer_o3d = np.square(
        np.asarray(cloud_x_o3d.compute_point_cloud_distance(cloud_y_o3d))
    ).mean()
    if not single_directional:
        chamfer_o3d += np.square(
            np.asarray(cloud_y_o3d.compute_point_cloud_distance(cloud_x_o3d))
        ).mean()

    chamfer_wp = tw.metrics.chamfer_points_to_points(
        points_to_warp(x_np, device),
        points_to_warp(y_np, device),
        single_directional=single_directional,
    )
    assert np.allclose(chamfer_wp, chamfer_o3d, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Chamfer: mesh to mesh (vertex-to-surface)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
def test_chamfer_mesh_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_a_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    vertices_b_np = vertices_a_np + np.array([0.3, -0.15, 0.2])

    sqr_a_to_b_np = _igl_point_mesh_sqr_dist(vertices_a_np, vertices_b_np, faces_np)
    sqr_b_to_a_np = _igl_point_mesh_sqr_dist(vertices_b_np, vertices_a_np, faces_np)
    chamfer_np = np.mean(sqr_a_to_b_np) + np.mean(sqr_b_to_a_np)

    vertices_b_wp = points_to_warp(vertices_b_np, mesh_wp.device)
    chamfer_wp = tw.metrics.chamfer_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, vertices_b_wp, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


def test_chamfer_mesh_to_mesh_identical_is_zero(icosahedron) -> None:
    _mesh_tm, mesh_wp = icosahedron
    chamfer_wp = tw.metrics.chamfer_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Chamfer: point cloud to mesh (mixed)
# ---------------------------------------------------------------------------

# MeshLib's own float upper bound. ``math.inf`` segfaults in ``findProjections`` rather than
# raising, which is why this is spelled out rather than computed.
_MESHLIB_FLT_MAX = 3.4028234663852886e38


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("chamfer_points_to_mesh", "meshlib")
def test_chamfer_points_to_mesh_forward_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the forward half: ``PointsToMeshProjector`` returns the same squared distances.

    The complement to [`test_chamfer_points_to_mesh`], which composes igl and scipy to reach the
    whole two-sided Chamfer. This one isolates the expensive half -- the cloud-to-surface query --
    and compares it **element-wise** rather than through a mean, which is strictly stronger: a mean
    hides a permutation and a pair of compensating errors, and MeshLib reports one
    ``MeshProjectionResult`` per query point so the correspondence is available. `distSq` is already
    squared, matching triwarp's ``point_reduction=None`` output with no transform.

    Measured 2.4e-07 max absolute difference over 500 points (1.1e-07 relative), which is triwarp's
    float32 vertex storage against MeshLib's float32 -- see the `getNumpyVerts` note in CLAUDE.md
    section 6.

    Three MeshLib call conventions this depends on, each a documented hazard:
    ``updateMeshData`` stores a raw pointer, so ``mesh_ml`` is bound to a name that outlives every
    query; ``upDistLimitSq`` precedes ``loDistLimitSq`` and must be MeshLib's own ``FLT_MAX``
    (``math.inf`` segfaults, and passing ``0.0`` in that slot silently returns all-zero distances --
    which is what a first attempt at this test did); and the AABB tree is built lazily on first
    query, so it is pre-warmed here to keep the comparison independent of call order.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    center = np.asarray(mesh_tm.vertices, dtype=np.float64).mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(500, 3))).astype(np.float32)

    forward_wp = tw.metrics.chamfer_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device),
        mesh_wp.points,
        mesh_wp.indices,
        single_directional=True,
        point_reduction=None,
    ).numpy()

    mesh_ml = trimesh_to_meshlib(mesh_tm)  # must outlive the projector: it stores a raw pointer
    projector_ml = mm.PointsToMeshProjector()
    projector_ml.updateMeshData(mesh_ml)
    points_ml = mm.std_vector_Vector3_float()
    for point in points_np:
        points_ml.append(mm.Vector3f(float(point[0]), float(point[1]), float(point[2])))
    results_ml = mm.std_vector_MeshProjectionResult()
    projector_ml.findProjections(
        results_ml, points_ml, mm.AffineXf3f(), mm.AffineXf3f(), _MESHLIB_FLT_MAX, 0.0
    )
    forward_ml = np.array([result.distSq for result in results_ml], dtype=np.float64)

    # Non-vacuity: an all-zero pair of answers would satisfy the comparison below.
    assert forward_ml.shape == forward_wp.shape
    assert forward_ml.min() > 0.0
    assert np.ptp(forward_ml) > 1e-3
    assert np.allclose(forward_wp, forward_ml, rtol=_MESH_RTOL, atol=_MESH_ATOL)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("chamfer_points_to_mesh", "igl")
def test_chamfer_points_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (a named composition): igl supplies the forward half, scipy the backward one.

    ``igl.point_mesh_squared_distance`` *is* the cloud-to-surface query and is exact, but there is
    no igl cloud-to-cloud counterpart, so the reference Chamfer is assembled from it plus a
    ``KDTree`` ``k=1`` search back from the mesh vertices -- the same two halves triwarp computes,
    in the same order, each from an independent implementation. That composition is why the
    benchmark's igl row is the forward half alone and reads as a lower bound.

    The tolerance is ``_MESH_RTOL`` rather than ``1e-5`` for a stated reason: igl consumes float64
    vertices where Warp carries float32, so the two agree only to float32 precision here.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)

    rng = np.random.default_rng(5)
    center = vertices_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(120, 3))).astype(np.float32)

    forward_sqr_igl = _igl_point_mesh_sqr_dist(points_np, vertices_np, faces_np)
    backward_sqr_np = KDTree(points_np.astype(np.float64)).query(vertices_np)[0] ** 2
    chamfer_np = np.mean(forward_sqr_igl) + np.mean(backward_sqr_np)

    chamfer_wp = tw.metrics.chamfer_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device), mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(chamfer_wp, chamfer_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Hausdorff: point cloud to point cloud
# ---------------------------------------------------------------------------


def test_hausdorff_points_to_points(device: str) -> None:
    rng = np.random.default_rng(6)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    hausdorff_np = max(directed_hausdorff(x_np, y_np)[0], directed_hausdorff(y_np, x_np)[0])

    hausdorff_wp = tw.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=1e-5, atol=1e-5)


def test_hausdorff_points_to_points_single_directional(device: str) -> None:
    rng = np.random.default_rng(7)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    hausdorff_np = directed_hausdorff(x_np, y_np)[0]

    hausdorff_wp = tw.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device), single_directional=True
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("hausdorff_points_to_points", "open3d", "pymeshlab")
def test_hausdorff_points_to_points_matches_open3d_and_pymeshlab(device: str) -> None:
    """
    Class B against both references the benchmark times, each needing a different named transform.

    **Open3D** returns the per-point one-way distances from ``compute_point_cloud_distance``, so the
    transform is the ``max`` reduction plus the second direction -- the symmetric Hausdorff is
    ``max`` over both.

    **MeshLab** returns a *dict of statistics* from ``get_hausdorff_distance``, so the transform is
    the ``"max"`` key, again over both directions. Two further parameters are load-bearing and are
    the same ones the benchmark passes: the filter is one-directional by construction (it samples
    ``sampledmesh`` and searches ``targetmesh``), and its default ``samplenum=8`` would compare
    eight random points rather than the cloud, so ``samplevert=True`` with ``samplenum`` at the full
    count is what makes it sample every point. The clouds go in as **face-less** meshes, since
    MeshLab's sampler would otherwise sample the surface instead of the vertices.
    """
    rng = np.random.default_rng(22)
    x_np = (rng.random((80, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    y_np = (rng.random((60, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)

    cloud_x_o3d, cloud_y_o3d = points_to_open3d(x_np), points_to_open3d(y_np)
    hausdorff_o3d = max(
        np.asarray(cloud_x_o3d.compute_point_cloud_distance(cloud_y_o3d)).max(),
        np.asarray(cloud_y_o3d.compute_point_cloud_distance(cloud_x_o3d)).max(),
    )

    meshset_pml = points_to_pymeshlab(x_np)
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(y_np, dtype=np.float64)))
    hausdorff_pml = max(
        float(
            meshset_pml.get_hausdorff_distance(
                sampledmesh=0, targetmesh=1, samplevert=True, samplenum=x_np.shape[0]
            )["max"]
        ),
        float(
            meshset_pml.get_hausdorff_distance(
                sampledmesh=1, targetmesh=0, samplevert=True, samplenum=y_np.shape[0]
            )["max"]
        ),
    )

    hausdorff_wp = tw.metrics.hausdorff_points_to_points(
        points_to_warp(x_np, device), points_to_warp(y_np, device)
    )
    assert np.allclose(hausdorff_wp, hausdorff_o3d, rtol=1e-5, atol=1e-5)
    assert np.allclose(hausdorff_wp, hausdorff_pml, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Hausdorff: mesh to mesh (direct port of igl::hausdorff)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
def test_hausdorff_mesh_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_a_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    vertices_b_np = vertices_a_np + np.array([0.3, -0.15, 0.2])

    sqr_a_to_b_np = _igl_point_mesh_sqr_dist(vertices_a_np, vertices_b_np, faces_np)
    sqr_b_to_a_np = _igl_point_mesh_sqr_dist(vertices_b_np, vertices_a_np, faces_np)
    hausdorff_np = np.sqrt(max(sqr_a_to_b_np.max(), sqr_b_to_a_np.max()))

    vertices_b_wp = points_to_warp(vertices_b_np, mesh_wp.device)
    hausdorff_wp = tw.metrics.hausdorff_mesh_to_mesh(
        mesh_wp.points, mesh_wp.indices, vertices_b_wp, mesh_wp.indices
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Hausdorff: point cloud to mesh (mixed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_hausdorff_points_to_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)

    rng = np.random.default_rng(8)
    center = vertices_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.6, size=(120, 3))).astype(np.float32)

    forward_np = np.sqrt(_igl_point_mesh_sqr_dist(points_np, vertices_np, faces_np).max())
    backward_np = KDTree(points_np.astype(np.float64)).query(vertices_np)[0].max()
    hausdorff_np = max(forward_np, backward_np)

    hausdorff_wp = tw.metrics.hausdorff_points_to_mesh(
        points_to_warp(points_np, mesh_wp.device), mesh_wp.points, mesh_wp.indices
    )
    assert np.allclose(hausdorff_wp, hausdorff_np, rtol=_MESH_RTOL, atol=_MESH_ATOL)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_chamfer_points_to_points_empty(device: str) -> None:
    x_wp = wp.empty(0, dtype=wp.vec3, device=device)
    y_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    assert tw.metrics.chamfer_points_to_points(x_wp, y_wp) == 0.0


def test_chamfer_points_to_points_empty_unreduced(device: str) -> None:
    x_wp = wp.empty(0, dtype=wp.vec3, device=device)
    y_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    forward_wp, backward_wp = tw.metrics.chamfer_points_to_points(x_wp, y_wp, point_reduction=None)
    assert forward_wp.shape == (0,)
    assert backward_wp.shape == (0,)


def test_hausdorff_points_to_mesh_empty_faces(device: str) -> None:
    points_wp = points_to_warp(np.zeros((5, 3), dtype=np.float32), device)
    vertices_wp = points_to_warp(np.zeros((3, 3), dtype=np.float32), device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.metrics.hausdorff_points_to_mesh(points_wp, vertices_wp, faces_wp) == 0.0


def test_hausdorff_points_to_points_identical_is_zero(device: str) -> None:
    rng = np.random.default_rng(9)
    x_np = (rng.random((30, 3), dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    x_wp = points_to_warp(x_np, device)
    assert tw.metrics.hausdorff_points_to_points(x_wp, x_wp) == 0.0


# ---------------------------------------------------------------------------
# Differentiable Chamfer losses (gradient checks)
# ---------------------------------------------------------------------------
#
# References reimplement the pytorch3d gradient convention in numpy: the
# nearest-neighbor / closest-face assignment is held constant, so the loss is a smooth
# function of the coordinates. Point-cloud gradients use the closed-form derivative of
# the squared-distance chamfer; mesh-surface gradients use central finite differences of
# the numpy point-triangle distance (there is no simple closed form across the
# face/edge/vertex regions). The assignment fed to each reference is the exact one
# triwarp used, isolating the gradient computation from float32 argmin tie-breaking.

_GRAD_RTOL = 1e-4
_GRAD_ATOL = 1e-4
_FD_RTOL = 1e-2
_FD_ATOL = 2e-3


def _nn_indices_np(a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
    """Index in ``b`` of the nearest point to each row of ``a`` (brute force)."""
    squared = ((a_np[:, None, :] - b_np[None, :, :]) ** 2).sum(-1)
    return squared.argmin(1).astype(np.int64)


def _pt_tri_sq_np(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Squared distance from ``p`` to triangle ``(a, b, c)`` (numpy mirror of the kernel)."""
    q, e1, e2 = p - a, b - a, c - a
    e1e1, e1e2, e2e2 = e1 @ e1, e1 @ e2, e2 @ e2
    det = e1e1 * e2e2 - e1e2 * e1e2
    if det > e1e1 * e2e2 * 1e-6:
        e1p, e2p = e1 @ q, e2 @ q
        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det
        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            r = q - s * e1 - t * e2
            return float(r @ r)

    def seg(qq: np.ndarray, ss: np.ndarray, length_sq: float) -> float:
        u = np.clip((qq @ ss) / length_sq, 0.0, 1.0)
        r = qq - u * ss
        return float(r @ r)

    d1 = seg(q, e1, e1e1)
    d2 = seg(q, e2, e2e2)
    d12 = seg(q - e1, e2 - e1, (e2 - e1) @ (e2 - e1))
    return min(d1, d2, d12)


def _surface_sq_np(
    points_np: np.ndarray, verts_np: np.ndarray, faces_np: np.ndarray, face_id_np: np.ndarray
) -> np.ndarray:
    """Per-point squared distance to the assigned triangle ``face_id[i]``."""
    out = np.empty(len(points_np), dtype=np.float64)
    for i in range(len(points_np)):
        f = int(face_id_np[i])
        a = verts_np[faces_np[3 * f + 0]]
        b = verts_np[faces_np[3 * f + 1]]
        c = verts_np[faces_np[3 * f + 2]]
        out[i] = _pt_tri_sq_np(points_np[i], a, b, c)
    return out


def _reduce_np(per_point: np.ndarray, reduction: str) -> float:
    return float(per_point.mean() if reduction == "mean" else per_point.sum())


def _fd_grad(loss_fn, arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite-difference gradient of ``loss_fn`` w.r.t. in-place array ``arr``."""
    grad = np.zeros_like(arr)
    flat = arr.reshape(-1)
    grad_flat = grad.reshape(-1)
    for idx in range(flat.size):
        original = flat[idx]
        flat[idx] = original + eps
        loss_plus = loss_fn()
        flat[idx] = original - eps
        loss_minus = loss_fn()
        flat[idx] = original
        grad_flat[idx] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_points_to_points_loss_grad(
    device: str, reduction: str, single_directional: bool
) -> None:
    rng = np.random.default_rng(20)
    x_np = (rng.random((25, 3)) * 4.0 - 2.0).astype(np.float32).astype(np.float64)
    y_np = (rng.random((18, 3)) * 4.0 - 2.0).astype(np.float32).astype(np.float64)
    n, m = len(x_np), len(y_np)

    nn_xy = _nn_indices_np(x_np, y_np)
    nn_yx = _nn_indices_np(y_np, x_np)

    # Reference loss and closed-form gradient (pytorch3d convention, fixed assignment).
    scale_f = (1.0 / n) if reduction == "mean" else 1.0
    loss_np = _reduce_np(((x_np - y_np[nn_xy]) ** 2).sum(-1), reduction)
    grad_x_np = 2.0 * scale_f * (x_np - y_np[nn_xy])
    grad_y_np = np.zeros_like(y_np)
    np.add.at(grad_y_np, nn_xy, -2.0 * scale_f * (x_np - y_np[nn_xy]))
    if not single_directional:
        scale_b = (1.0 / m) if reduction == "mean" else 1.0
        loss_np += _reduce_np(((y_np - x_np[nn_yx]) ** 2).sum(-1), reduction)
        grad_y_np += 2.0 * scale_b * (y_np - x_np[nn_yx])
        np.add.at(grad_x_np, nn_yx, -2.0 * scale_b * (y_np - x_np[nn_yx]))

    x_wp = wp.array(x_np, dtype=wp.vec3, device=device, requires_grad=True)
    y_wp = wp.array(y_np, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    loss_wp = tw.metrics.chamfer_points_to_points_loss(
        x_wp, y_wp, tape=tape, point_reduction=reduction, single_directional=single_directional
    )
    tape.backward(loss=loss_wp)

    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(x_wp.grad.numpy(), grad_x_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)
    assert np.allclose(y_wp.grad.numpy(), grad_y_np, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_points_to_mesh_loss_grad(
    icosahedron, reduction: str, single_directional: bool
) -> None:
    mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.device
    verts_np = np.asarray(mesh_tm.vertices, dtype=np.float32).astype(np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1)

    rng = np.random.default_rng(21)
    center = verts_np.mean(axis=0)
    points_np = (center + rng.normal(scale=0.5, size=(10, 3))).astype(np.float32).astype(np.float64)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=True)
    verts_wp = wp.array(verts_np, dtype=wp.vec3, device=device, requires_grad=True)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    # Fixed assignments exactly as the loss function computes them.
    face_id = tw.proximity.closest_point_on_mesh(verts_wp, faces_wp, points_wp)[2].numpy()
    nn_vp = tw.neighbors.query_nearest(points_wp, verts_wp, k=1)[0].numpy()

    def evaluate_loss_np() -> float:
        total = _reduce_np(_surface_sq_np(points_np, verts_np, faces_np, face_id), reduction)
        if not single_directional:
            total += _reduce_np(((verts_np - points_np[nn_vp]) ** 2).sum(-1), reduction)
        return total

    loss_np = evaluate_loss_np()
    grad_points_np = _fd_grad(evaluate_loss_np, points_np)
    grad_verts_np = _fd_grad(evaluate_loss_np, verts_np)

    tape = wp.Tape()
    loss_wp = tw.metrics.chamfer_points_to_mesh_loss(
        points_wp,
        verts_wp,
        faces_wp,
        tape=tape,
        point_reduction=reduction,
        single_directional=single_directional,
    )
    tape.backward(loss=loss_wp)

    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(points_wp.grad.numpy(), grad_points_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(verts_wp.grad.numpy(), grad_verts_np, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parity(
    "chamfer_mesh_to_mesh_loss",
    "numpy",
    benchmarked=False,
    reason="the reference is a central finite difference of the same loss, which costs two forward "
    "evaluations per coordinate -- 6n for an n-vertex pair -- so timing it would price the "
    "oracle's own cost rather than a competing autodiff. No installed library differentiates a "
    "mesh-to-mesh Chamfer at all; the gradient values are what is comparable.",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("single_directional", [False, True])
def test_chamfer_mesh_to_mesh_loss_grad(
    icosahedron, reduction: str, single_directional: bool
) -> None:
    mesh_tm, mesh_wp = icosahedron
    device = mesh_wp.device
    verts_a_np = np.asarray(mesh_tm.vertices, dtype=np.float32).astype(np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1)
    verts_b_np = (verts_a_np + np.array([0.3, -0.15, 0.2])).astype(np.float32).astype(np.float64)

    verts_a_wp = wp.array(verts_a_np, dtype=wp.vec3, device=device, requires_grad=True)
    verts_b_wp = wp.array(verts_b_np, dtype=wp.vec3, device=device, requires_grad=True)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    face_id_ab = tw.proximity.closest_point_on_mesh(verts_b_wp, faces_wp, verts_a_wp)[2].numpy()
    face_id_ba = tw.proximity.closest_point_on_mesh(verts_a_wp, faces_wp, verts_b_wp)[2].numpy()

    def evaluate_loss_np() -> float:
        total = _reduce_np(_surface_sq_np(verts_a_np, verts_b_np, faces_np, face_id_ab), reduction)
        if not single_directional:
            total += _reduce_np(
                _surface_sq_np(verts_b_np, verts_a_np, faces_np, face_id_ba), reduction
            )
        return total

    loss_np = evaluate_loss_np()
    grad_a_np = _fd_grad(evaluate_loss_np, verts_a_np)
    grad_b_np = _fd_grad(evaluate_loss_np, verts_b_np)

    tape = wp.Tape()
    loss_wp = tw.metrics.chamfer_mesh_to_mesh_loss(
        verts_a_wp,
        faces_wp,
        verts_b_wp,
        faces_wp,
        tape=tape,
        point_reduction=reduction,
        single_directional=single_directional,
    )
    tape.backward(loss=loss_wp)

    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    assert np.allclose(verts_a_wp.grad.numpy(), grad_a_np, rtol=_FD_RTOL, atol=_FD_ATOL)
    if not single_directional:
        assert np.allclose(verts_b_wp.grad.numpy(), grad_b_np, rtol=_FD_RTOL, atol=_FD_ATOL)


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_chamfer_losses_match_numpy_on_both_devices(kernel_device: str) -> None:
    """
    Pin both chamfer loss reductions on **both** devices against a closed-form NumPy sum.

    ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.17's CPU backend, so the
    block-wide ``wp.tile_sum`` these losses used to perform accumulated one point per 64-point tile
    there and returned a loss roughly 64x too small -- measured 0.43 absolute on this size of cloud.
    Each term now has a lane-free ``*_sliced`` kernel for CPU and keeps the ``*_tiled`` one on CUDA,
    so the parametrization is what covers both: dropping either device leaves a kernel untested.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(91)
    x_np = rng.standard_normal((500, 3))
    y_np = rng.standard_normal((300, 3)) * 0.7
    x_wp = points_to_warp(x_np, kernel_device)
    y_wp = points_to_warp(y_np, kernel_device)

    # Bidirectional mean-reduced squared chamfer, exactly what the default arguments compute.
    squared_np = ((x_np[:, None, :] - y_np[None, :, :]) ** 2).sum(-1)
    loss_np = squared_np.min(1).mean() + squared_np.min(0).mean()
    loss_wp = tw.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    assert np.allclose(loss_wp.numpy()[0], loss_np, rtol=1e-4, atol=1e-4)

    # Single-directional points-to-mesh: the reference is the distance to the nearest triangle,
    # which for a convex mesh sampled outside it is the distance to its surface.
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    surface_loss_wp = tw.metrics.chamfer_points_to_mesh_loss(
        x_wp, mesh_wp.points, mesh_wp.indices, single_directional=True
    )
    _closest_np, distance_np, _face_np = tm_proximity.closest_point(mesh_tm, x_np)
    assert np.allclose(surface_loss_wp.numpy()[0], (distance_np**2).mean(), rtol=1e-3, atol=1e-3)


def test_chamfer_loss_no_tape_has_value_but_no_grad(device: str) -> None:
    rng = np.random.default_rng(22)
    x_np = (rng.random((12, 3)) * 2.0 - 1.0).astype(np.float32)
    y_np = (rng.random((9, 3)) * 2.0 - 1.0).astype(np.float32)
    x_wp = points_to_warp(x_np, device)
    y_wp = points_to_warp(y_np, device)

    loss_wp = tw.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    reduced = tw.metrics.chamfer_points_to_points(x_wp, y_wp)
    assert np.allclose(loss_wp.numpy()[0], reduced, rtol=1e-5, atol=1e-5)


def test_chamfer_loss_rejects_max_reduction(device: str) -> None:
    x_wp = points_to_warp(np.zeros((3, 3), dtype=np.float32), device)
    y_wp = points_to_warp(np.ones((3, 3), dtype=np.float32), device)
    with pytest.raises(ValueError, match="mean"):
        tw.metrics.chamfer_points_to_points_loss(x_wp, y_wp, point_reduction="max")  # type: ignore[arg-type]


def test_chamfer_points_to_points_loss_empty(device: str) -> None:
    x_wp = wp.empty(0, dtype=wp.vec3, device=device)
    y_wp = points_to_warp(np.zeros((4, 3), dtype=np.float32), device)
    loss_wp = tw.metrics.chamfer_points_to_points_loss(x_wp, y_wp)
    assert loss_wp.shape == (1,)
    assert float(loss_wp.numpy()[0]) == 0.0
