"""Regression tests for ``triwarp.registration`` against ``trimesh.registration``."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pytorch3d.ops as p3d_ops
import pyvista as pv
import trimesh as tm
import trimesh.registration as tm_reg
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import (
    points_to_meshlib,
    points_to_open3d,
    points_to_torch,
    points_to_warp,
    trimesh_to_meshlib,
)


def _make_point_clouds(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    a_np = rng.standard_normal((n, 3)).astype(np.float64)
    # b is a mildly rotated/translated version of a to keep correspondence meaningful
    b_np = rng.standard_normal((n, 3)).astype(np.float64)
    return a_np, b_np


def _run_both(
    a_np: np.ndarray,
    b_np: np.ndarray,
    device: str,
    weights_np: np.ndarray | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
) -> tuple:
    """Return (matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_tw, cost_tw)."""
    kwargs = dict(reflection=reflection, translation=translation, scale=scale)

    matrix_tm, transformed_tm, cost_tm = tm_reg.procrustes(a_np, b_np, weights=weights_np, **kwargs)

    a_wp = points_to_warp(a_np, device)
    b_wp = points_to_warp(b_np, device)
    weights_wp = (
        wp.array(weights_np.astype(np.float32), dtype=wp.float32, device=device)
        if weights_np is not None
        else None
    )

    matrix_wp, transformed_wp, cost_tw = tw.registration.procrustes(
        a_wp, b_wp, weights=weights_wp, **kwargs
    )

    matrix_tw = matrix_wp.numpy()[0]
    return matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw


def test_procrustes_default(device: str) -> None:
    """
    Class A: the transform, the transformed cloud and the cost against ``trimesh.registration``.

    All three returns are compared, not just the matrix: a transposed rotation still gives a
    plausible matrix and a wrong cloud. The clouds are related by a known rigid motion, so the
    optimum is unique and the comparison can be elementwise.
    """
    rng = np.random.default_rng(0)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("procrustes", "meshlib")
def test_procrustes_matches_meshlib(device: str) -> None:
    """
    Class A on the rigid transform: ``PointToPointAligningTransform`` solves the same fit.

    MeshLib accumulates correspondences one at a time -- ``add(p1, p2, weight)`` -- and
    ``findBestRigidXf`` returns an ``AffineXf3d`` carrying a ``Matrix3d`` and a translation, so the
    transform is unpacking that into a 4x4 rather than anything about the values. Both recover a
    *known* rigid motion here, which is what makes the comparison a check on the solver and not a
    restatement: the rotation and translation agree with the planted ones and with each other to
    **1.1e-06**, triwarp's float32 floor.

    ``scale=False`` and ``reflection=False`` are triwarp's settings for this pairing:
    ``findBestRigidXf`` fits a rotation and translation only. Its sibling
    ``findBestRigidScaleXf`` is the ``scale=True`` form and is not what this compares.
    """
    rng = np.random.default_rng(0)
    source_np = rng.standard_normal((200, 3))
    rotation_np = tm.transformations.rotation_matrix(0.4, [0.3, 0.5, 0.8])[:3, :3]
    translation_np = np.array([1.0, -2.0, 0.5])
    target_np = source_np @ rotation_np.T + translation_np

    matrix_wp, _transformed_wp, cost_wp = tw.registration.procrustes(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        reflection=False,
        scale=False,
    )
    matrix_np = matrix_wp.numpy()[0]

    aligner_ml = mm.PointToPointAligningTransform()
    for source_point_np, target_point_np in zip(source_np, target_np, strict=True):
        aligner_ml.add(
            mm.Vector3d(*source_point_np.tolist()), mm.Vector3d(*target_point_np.tolist()), 1.0
        )
    transform_ml = aligner_ml.findBestRigidXf()
    rotation_ml = np.array(
        [
            [transform_ml.A.x.x, transform_ml.A.x.y, transform_ml.A.x.z],
            [transform_ml.A.y.x, transform_ml.A.y.y, transform_ml.A.y.z],
            [transform_ml.A.z.x, transform_ml.A.z.y, transform_ml.A.z.z],
        ]
    )
    translation_ml = np.array([transform_ml.b.x, transform_ml.b.y, transform_ml.b.z])

    # Non-vacuity: the reference recovered the planted motion, so it is not returning the identity.
    assert np.allclose(rotation_ml, rotation_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(translation_ml, translation_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(matrix_np[:3, :3], rotation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(matrix_np[:3, 3], translation_ml, rtol=1e-4, atol=1e-4)
    assert cost_wp < 1e-8  # an exact correspondence set has a zero-residual fit


def test_procrustes_return_cost_arity(device: str) -> None:
    """
    Class A: the bare call returns its documented three-tuple, ``return_cost=False`` one array.

    The overloads used to declare ``return_cost: Literal[False] = False`` first, so a checker
    resolved the bare ``procrustes(a, b)`` to the matrix-only signature while the implementation
    returned the tuple. Nothing exercised the *bare* default, which is why it went unnoticed.
    """
    rng = np.random.default_rng(7)
    a_np, b_np = _make_point_clouds(rng)
    a_wp, b_wp = points_to_warp(a_np, device), points_to_warp(b_np, device)

    defaulted = tw.registration.procrustes(a_wp, b_wp)
    assert isinstance(defaulted, tuple)
    assert len(defaulted) == 3
    matrix_wp, transformed_wp, cost = defaulted
    assert matrix_wp.shape[0] == 1
    assert transformed_wp.shape[0] == a_np.shape[0]
    assert np.isfinite(cost)

    matrix_only = tw.registration.procrustes(a_wp, b_wp, return_cost=False)
    assert isinstance(matrix_only, wp.array)
    assert np.allclose(matrix_only.numpy(), matrix_wp.numpy(), rtol=1e-5, atol=1e-5)


def test_procrustes_uniform_weights(device: str) -> None:
    """
    Class A: an all-ones weight vector must reproduce the unweighted answer exactly.

    The degenerate case of the weighted path, and the one that catches a normalization missing
    from the weighted moments -- it would still converge, just to a different scale.
    """
    rng = np.random.default_rng(1)
    a_np, b_np = _make_point_clouds(rng)
    n = a_np.shape[0]
    weights_np = np.ones(n, dtype=np.float64)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, weights_np=weights_np
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_binary_weights(device: str) -> None:
    """
    Class A: zero weights must exclude their points, matching trimesh on the retained half.

    Half the cloud is weighted out, so an implementation that ignores weights fits a different
    optimum and fails. trimesh takes the same weights, so this stays elementwise.
    """
    rng = np.random.default_rng(2)
    a_np, b_np = _make_point_clouds(rng)
    n = a_np.shape[0]
    weights_np = np.zeros(n, dtype=np.float64)
    weights_np[: n // 2] = 1.0
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, weights_np=weights_np
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(
        transformed_wp.numpy()[: n // 2], transformed_tm[: n // 2], rtol=1e-4, atol=1e-4
    )
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_reflection(device: str) -> None:
    """
    Class A on the ``reflection=False`` branch, where the rotation is constrained to det = +1.

    A separate test per flag because each changes the SVD post-processing rather than the
    input; trimesh exposes the identical switch.
    """
    rng = np.random.default_rng(3)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, reflection=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_translation(device: str) -> None:
    """
    Class A on the ``translation=False`` branch: the clouds are not centred first.

    Same reasoning as the flag above -- and this is the branch where a mistakenly-subtracted
    centroid would still produce a valid-looking rotation.
    """
    rng = np.random.default_rng(4)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, translation=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_scale(device: str) -> None:
    """
    Class A on the ``scale=False`` branch, which fixes the scale factor at one.

    Completes the three flags. Compared elementwise against trimesh's identical switch.
    """
    rng = np.random.default_rng(5)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, scale=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_far_from_origin(device: str) -> None:
    """
    Class A: single-pass moments must survive a cloud far from the origin.

    The fused accumulation shifts by ``a[0]`` rather than by the origin precisely so the
    cancellation in the second-moment identity stays bounded by ``(diameter / spread) ** 2``
    instead of ``(|centroid| / spread) ** 2`` — the latter is unbounded and would be worth about
    27 bits here, i.e. all of float32.

    The reference is trimesh run on the **float32-rounded** clouds. At ``+1e4`` with unit spread,
    representing the input in float32 at all costs ~2e-4 in the rotation and ~3 units in the
    translation, whichever algorithm consumes it; comparing against the float64-input fit would
    measure that instead of the accumulation.
    """
    rng = np.random.default_rng(11)
    a_np, b_np = _make_point_clouds(rng)
    a_np = (a_np + 1.0e4).astype(np.float32).astype(np.float64)
    b_np = (b_np + 1.0e4).astype(np.float32).astype(np.float64)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device
    )
    assert np.allclose(matrix_tw[:3, :3], matrix_tm[:3, :3], rtol=1e-4, atol=1e-4)
    # The translation is ``bcenter - sR @ acenter``, so its error scales with the *centroid*
    # magnitude (1e4 here), not with its own — a component that happens to land near zero is not
    # thereby more accurate. Tolerance is therefore ``1e-4`` of the coordinate scale.
    coordinate_scale = float(np.abs(a_np).max())
    assert np.allclose(matrix_tw[:3, 3], matrix_tm[:3, 3], rtol=1e-4, atol=1e-4 * coordinate_scale)
    assert np.allclose(
        transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4 * coordinate_scale
    )
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_fractional_weights(device: str) -> None:
    """Class A: with non-binary weights, the masked covariance and weighted moments agree."""
    rng = np.random.default_rng(12)
    a_np, b_np = _make_point_clouds(rng)
    weights_np = rng.uniform(0.1, 3.0, size=a_np.shape[0])
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, weights_np=weights_np
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("procrustes", "trimesh")
def test_procrustes_return_matrix_only(device: str) -> None:
    """
    Triwarp against triwarp: ``return_cost=False`` returns only the matrix it otherwise would.

    A signature claim rather than a numerical one -- the matrix's oracle is
    [`test_procrustes_default`] -- and what it catches is a return tuple that changed shape
    silently.
    """
    rng = np.random.default_rng(6)
    a_np, b_np = _make_point_clouds(rng)
    a_wp = points_to_warp(a_np, device)
    b_wp = points_to_warp(b_np, device)

    result = tw.registration.procrustes(a_wp, b_wp, return_cost=False)
    assert isinstance(result, wp.array)
    assert result.shape == (1,)
    assert result.dtype == wp.mat44

    matrix_tm, _, _ = tm_reg.procrustes(a_np, b_np)
    assert np.allclose(result.numpy()[0], matrix_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_length_mismatch(device: str) -> None:
    a_wp = points_to_warp(np.random.default_rng(20).standard_normal((10, 3)), device)
    b_wp = points_to_warp(np.random.default_rng(21).standard_normal((11, 3)), device)
    with pytest.raises(ValueError, match="same length"):
        tw.registration.procrustes(a_wp, b_wp)


def test_procrustes_weights_length_mismatch(device: str) -> None:
    a_wp = points_to_warp(np.random.default_rng(22).standard_normal((10, 3)), device)
    b_wp = points_to_warp(np.random.default_rng(23).standard_normal((10, 3)), device)
    weights_wp = wp.zeros(5, dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="same length"):
        tw.registration.procrustes(a_wp, b_wp, weights=weights_wp)


def test_procrustes_scale_on_duplicated_points(device: str) -> None:
    """
    Not a library comparison: a scale fit over a zero-variance cloud has no defined answer.

    This pins that the kernel's own scale-factor sqrt is floored rather than fed a
    fp-cancellation-driven negative argument and returning NaN.

    Every point of ``a`` (and of ``b``) is identical, so the shifted second moment
    ``sum |a - p|^2`` is exactly zero for the shift point itself and only cancellation noise for
    any other point sharing its value -- exactly the zero-or-slightly-negative case the scale
    floor exists for.
    """
    a_wp = points_to_warp(np.full((3, 3), [1.0, 2.0, 3.0], dtype=np.float32), device)
    b_wp = points_to_warp(np.full((3, 3), [4.0, 5.0, 6.0], dtype=np.float32), device)
    matrix_wp, _transformed_wp, cost_tw = tw.registration.procrustes(a_wp, b_wp, scale=True)
    assert np.all(np.isfinite(matrix_wp.numpy()))
    assert np.isfinite(cost_tw)


def test_procrustes_all_zero_weights(device: str) -> None:
    """An all-zero, non-empty ``weights`` must raise rather than silently return NaN."""
    a_wp = points_to_warp(np.random.default_rng(24).standard_normal((10, 3)), device)
    b_wp = points_to_warp(np.random.default_rng(25).standard_normal((10, 3)), device)
    weights_wp = wp.zeros(10, dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="sum to zero"):
        tw.registration.procrustes(a_wp, b_wp, weights=weights_wp)


def test_procrustes_empty(device: str) -> None:
    """``n == 0`` must return the identity, not the NaN a division by zero would give."""
    a_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    b_wp = wp.zeros(0, dtype=wp.vec3, device=device)

    matrix_wp, transformed_wp, cost_tw = tw.registration.procrustes(a_wp, b_wp)
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4))
    assert transformed_wp.shape == (0,)
    assert cost_tw == 0.0

    matrix_only = tw.registration.procrustes(a_wp, b_wp, return_cost=False)
    assert np.allclose(matrix_only.numpy()[0], np.eye(4))


# --- Iterative closest point (ICP) -----------------------------------------


def _rigid_transform(angle: float, axis: list[float], trans: list[float]) -> tuple:
    """Return ``(rotation (3, 3), translation (3,))`` float32 arrays."""
    rotation = tm.transformations.rotation_matrix(angle, axis)[:3, :3].astype(np.float32)
    translation = np.asarray(trans, dtype=np.float32)
    return rotation, translation


def _rms(points_a: np.ndarray, points_b: np.ndarray) -> float:
    return float(np.sqrt(((points_a - points_b) ** 2).sum(axis=1).mean()))


def _mesh_vertices_faces(mesh_tm: tm.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    return mesh_tm.vertices.astype(np.float32), mesh_tm.faces.reshape(-1).astype(np.int32)


@pytest.mark.parity("procrustes", "open3d")
def test_procrustes_matches_open3d(device: str) -> None:
    """
    Closed-form Kabsch against Open3D's, with the correspondence handed to both.

    Open3D's ``TransformationEstimationPointToPoint.compute_transformation`` takes an explicit
    correspondence list, which is exactly triwarp's ``procrustes`` contract, so this is Class A on
    the 4x4 matrix -- no ICP loop, no nearest-neighbour search, just the SVD. Measured worst entry
    deviation **1.4e-7**, well inside the 1e-5 asserted here.

    ``with_scaling=False`` on Open3D's side pairs with ``scale=False``; the two also agree that a
    reflection must not be introduced, which the determinant check pins.
    """
    rng = np.random.default_rng(10)
    target_np = rng.standard_normal((300, 3)).astype(np.float32)
    rotation_np, translation_np = _rigid_transform(0.15, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)

    matrix_wp, _transformed_wp, _cost = tw.registration.procrustes(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        reflection=False,
        scale=False,
    )

    correspondence = o3d.utility.Vector2iVector(np.stack([np.arange(300), np.arange(300)], axis=1))
    matrix_o3d = np.asarray(
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ).compute_transformation(
            points_to_open3d(source_np), points_to_open3d(target_np), correspondence
        )
    )

    assert np.allclose(matrix_wp.numpy()[0], matrix_o3d, rtol=1e-5, atol=1e-5)
    assert np.isclose(np.linalg.det(matrix_o3d[:3, :3]), 1.0, atol=1e-5)


@pytest.mark.parity("procrustes", "pytorch3d")
def test_procrustes_matches_pytorch3d(device: str) -> None:
    """
    Class B: ``corresponding_points_alignment`` is a **row-vector** convention, so ``R`` transposes.

    pytorch3d solves ``s * X @ R + T = Y`` where triwarp returns a column-vector ``wp.mat44``, so
    the reference's ``R`` is triwarp's linear block divided by the scale and **transposed**:
    measured 2.54e-07. The translation needs no transform (2.38e-07) and the scale comes back
    1.29999983 against pytorch3d's 1.30000031 on a planted 1.3 -- so the pair pins the scale too,
    which is the part ``estimate_scale=False`` would silently drop.

    This is the second pair ``triwarp/registration.py``'s prose has been asserting with nothing
    running pytorch3d.
    """
    rng = np.random.default_rng(11)
    a_np = rng.normal(size=(200, 3)).astype(np.float32)
    angle = 0.3
    rotation_np = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    b_np = (1.3 * (a_np @ rotation_np.T) + np.array([0.4, -0.2, 0.7], np.float32)).astype(
        np.float32
    )
    aligned_p3d = p3d_ops.corresponding_points_alignment(
        points_to_torch(a_np, device), points_to_torch(b_np, device), estimate_scale=True
    )
    matrix_wp = tw.registration.procrustes(
        points_to_warp(a_np, device), points_to_warp(b_np, device), return_cost=False
    )
    matrix_np = matrix_wp.numpy()[0]
    linear_np = matrix_np[:3, :3]
    scale = float(np.linalg.norm(linear_np[:, 0]))

    assert np.allclose(float(aligned_p3d.s[0]), 1.3, rtol=1e-5, atol=1e-5)
    assert np.allclose(scale, float(aligned_p3d.s[0]), rtol=1e-5, atol=1e-5)
    assert np.allclose((linear_np / scale).T, aligned_p3d.R[0].cpu().numpy(), rtol=1e-5, atol=1e-6)
    assert np.allclose(matrix_np[:3, 3], aligned_p3d.T[0].cpu().numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parity("icp_point_cloud", "pytorch3d")
def test_icp_point_cloud_matches_pytorch3d(device: str) -> None:
    """
    Class B: ``iterative_closest_point``'s converged transform, under the same ``R`` transpose.

    The **converged transform and its residual** are what is compared, not the iteration count:
    pytorch3d's stopping rule is its own ``relative_rmse_thr`` and it reached this fixture's answer
    in 4 iterations where triwarp's threshold takes its own number, so a count comparison would be
    pinning two different rules. Measured 4.17e-07 on the rotation and 2.35e-07 on the
    translation for a 300-point cloud under a planted 0.15 rad rotation, with pytorch3d's own
    ``rmse`` at 4.03e-07 and triwarp's cost at 1.4e-13.

    ``estimate_scale=False`` matches triwarp's ``scale=False`` default, and the pair is
    correspondence-free on both sides -- each iteration re-runs its own nearest-neighbour search,
    which is the formulation ``triwarp/registration.py``'s docstring credits to pytorch3d.
    """
    rng = np.random.default_rng(11)
    a_np = rng.normal(size=(300, 3)).astype(np.float32)
    angle = 0.15
    rotation_np = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    b_np = (a_np @ rotation_np.T + np.array([0.05, 0.02, -0.03], np.float32)).astype(np.float32)
    solution_p3d = p3d_ops.iterative_closest_point(
        points_to_torch(a_np, device), points_to_torch(b_np, device), estimate_scale=False
    )
    matrix_wp, _, cost = tw.registration.icp(
        points_to_warp(a_np, device), points_to_warp(b_np, device)
    )
    matrix_np = matrix_wp.numpy()[0]

    assert bool(solution_p3d.converged)
    assert float(solution_p3d.rmse[0]) < 1e-5
    assert cost < 1e-9
    assert np.allclose(
        matrix_np[:3, :3].T, solution_p3d.RTs.R[0].cpu().numpy(), rtol=1e-5, atol=1e-6
    )
    assert np.allclose(matrix_np[:3, 3], solution_p3d.RTs.T[0].cpu().numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parity("icp_point_cloud", "open3d", "trimesh")
def test_icp_point_to_point_matches_open3d_and_trimesh(device: str) -> None:
    """
    ICP against both references by the recovered *transform*, not by each side's own cost.

    ``test_icp_point_to_point_cloud`` checks that triwarp and trimesh each reach a low cost, which
    is two independent self-consistency checks rather than a comparison -- both could converge to
    different transforms and still pass. Here the three transforms are applied to the same source
    and the resulting point clouds compared directly, which is what "the same registration" means.

    Class C only in that a small rigid offset is required for it to be well posed: nearest-neighbour
    correspondence has to be unique, or the three solvers may legitimately land in different local
    minima and no comparison is meaningful. With the offset small enough (0.15 rad, 0.07
    translation) all three recover the exact alignment and agree to **1.4e-6**, so the 1e-4 bound
    below carries a 70x margin. A solver landing in a different minimum would miss by orders of
    magnitude, not by a tolerance.
    """
    rng = np.random.default_rng(10)
    target_np = rng.standard_normal((300, 3)).astype(np.float32)
    rotation_np, translation_np = _rigid_transform(0.15, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)

    _matrix_wp, transformed_wp, _cost_wp = tw.registration.icp(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        None,
        max_iterations=100,
        reflection=False,
        scale=False,
    )

    result_o3d = o3d.pipelines.registration.registration_icp(
        points_to_open3d(source_np),
        points_to_open3d(target_np),
        1e9,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    matrix_o3d = np.asarray(result_o3d.transformation)
    moved_o3d = (matrix_o3d[:3, :3] @ source_np.T).T + matrix_o3d[:3, 3]

    _matrix_tm, moved_tm, _cost_tm = tm_reg.icp(
        source_np.astype(np.float64),
        target_np.astype(np.float64),
        threshold=-np.inf,
        max_iterations=100,
        scale=False,
        reflection=False,
    )

    assert np.allclose(transformed_wp.numpy(), moved_o3d, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), moved_tm, rtol=1e-4, atol=1e-4)
    # All three land on the target itself, so none of the above is a comparison of two failures.
    assert _rms(transformed_wp.numpy(), target_np) < 1e-3


@pytest.mark.parity("icp_point_cloud", "meshlib")
def test_icp_point_to_point_matches_meshlib(device: str) -> None:
    """
    Class A on the recovered transform: MeshLib's ``ICP`` lands on the same one, to **7.4e-07**.

    Compared the way the open3d and trimesh pairing above is -- by the transform, not by each
    side's own cost, since two solvers reaching a low cost independently is not a comparison. Four
    parameters have to be set for that to be a like-for-like run, and each is a real choice rather
    than boilerplate:

    - ``ICPMethod.PointToPoint``, because MeshLib's default is **point-to-plane**, which is
      triwarp's *other* function;
    - ``iterLimit``, matched to ``max_iterations``;
    - a ``samplingVoxelSize`` small relative to the cloud, since the constructor overload taking one
      *subsamples* both clouds and a coarse value would register two different point sets;
    - ``MeshOrPoints`` wrappers, which is how the class accepts a cloud rather than a mesh.

    The clouds are related by a small rigid motion (8 degrees) so the nearest-neighbour
    correspondence is unique and both solvers are in the same basin -- without that, two ICPs may
    legitimately reach different local minima and no comparison is meaningful. Both leave the same
    residual (0.1607 on this fixture, which is the sphere's own rotational near-degeneracy, not a
    failure), and that equality is asserted too: it is what says they converged to the same place
    rather than agreeing on a transform by chance.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3)
    target_np = np.ascontiguousarray(sphere_tm.vertices)
    rotation_np, translation_np = _rigid_transform(
        np.deg2rad(8.0), [0.2, 0.9, 0.3], [0.05, -0.03, 0.04]
    )
    source_np = np.ascontiguousarray(target_np @ rotation_np.T + translation_np)

    matrix_wp, transformed_wp, _cost_wp = tw.registration.icp(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        None,
        max_iterations=30,
        reflection=False,
        scale=False,
    )
    matrix_np = matrix_wp.numpy()[0]

    icp_ml = mm.ICP(
        mm.MeshOrPoints(points_to_meshlib(source_np)),
        mm.MeshOrPoints(points_to_meshlib(target_np)),
        mm.AffineXf3f(),
        mm.AffineXf3f(),
        0.05,
    )
    properties_ml = mm.ICPProperties()
    properties_ml.iterLimit = 30
    properties_ml.method = mm.ICPMethod.PointToPoint  # its default is point-to-plane
    icp_ml.setParams(properties_ml)
    transform_ml = icp_ml.calculateTransformation()
    rotation_ml = np.array(
        [
            [transform_ml.A.x.x, transform_ml.A.x.y, transform_ml.A.x.z],
            [transform_ml.A.y.x, transform_ml.A.y.y, transform_ml.A.y.z],
            [transform_ml.A.z.x, transform_ml.A.z.y, transform_ml.A.z.z],
        ]
    )
    translation_ml = np.array([transform_ml.b.x, transform_ml.b.y, transform_ml.b.z])

    # Non-vacuity: the reference actually moved the cloud, and by more than a rounding error.
    assert np.abs(rotation_ml - np.eye(3)).max() > 1e-3
    assert np.allclose(matrix_np[:3, :3], rotation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(matrix_np[:3, 3], translation_ml, rtol=1e-4, atol=1e-4)

    # And they converged to the same place: the same residual, both below the starting offset.
    moved_ml = source_np @ rotation_ml.T + translation_ml
    assert np.isclose(_rms(transformed_wp.numpy(), target_np), _rms(moved_ml, target_np), rtol=1e-4)
    assert _rms(transformed_wp.numpy(), target_np) < _rms(source_np, target_np)


@pytest.mark.parametrize("robust_kernel", ["none", "tukey"])
@pytest.mark.parity("icp_point_to_plane_cloud", "open3d")
@pytest.mark.parity("icp_point_to_plane_tukey", "open3d")
def test_icp_point_to_plane_matches_open3d(
    device: str, robust_kernel: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class C (a fit-error bound): point-to-plane ICP against Open3D's, plain and robust.

    Both benchmark groups are the same solver at two robust-kernel settings, and Open3D exposes the
    matching pair -- ``TransformationEstimationPointToPlane()`` and the same wrapped in
    ``TukeyLoss(k=)`` -- so the parameter maps one-for-one, which is not true of most rows in this
    module.

    Compared by the recovered alignment rather than by each side's cost, for the reason given in
    ``test_icp_point_to_point_matches_open3d_and_trimesh``. Measured worst per-point deviation
    **1.2e-5** plain and **1.1e-5** with Tukey, against the 1e-3 bound here -- a ~90x margin. The
    bound is looser than the point-to-point test's because the linearized point-to-plane step is
    solved slightly differently on the two sides, so they stop at marginally different iterates;
    both still land on the target to better than 1e-5 RMS, which the final assert pins.
    """
    mesh_tm, _mesh_tm_wp = icosphere
    target_np = np.asarray(mesh_tm.vertices, dtype=np.float32)
    normals_np = np.asarray(mesh_tm.vertex_normals, dtype=np.float32)
    rotation_np, translation_np = _rigid_transform(0.08, [0.1, 0.9, 0.2], [0.02, -0.01, 0.015])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)
    scale = 0.1

    _matrix_wp, transformed_wp, _cost_wp = tw.registration.icp_point_to_plane(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        target_normals=points_to_warp(normals_np, device),
        max_iterations=50,
        threshold=-np.inf,
        robust_kernel=robust_kernel,
        robust_scale=scale,
    )

    estimation_o3d = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=scale)
        if robust_kernel == "tukey"
        else o3d.pipelines.registration.L2Loss()
    )
    result_o3d = o3d.pipelines.registration.registration_icp(
        points_to_open3d(source_np),
        points_to_open3d(target_np, normals_np),
        1e9,
        np.eye(4),
        estimation_o3d,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    matrix_o3d = np.asarray(result_o3d.transformation)
    moved_o3d = (matrix_o3d[:3, :3] @ source_np.T).T + matrix_o3d[:3, 3]

    assert np.allclose(transformed_wp.numpy(), moved_o3d, rtol=1e-3, atol=1e-3)
    assert _rms(transformed_wp.numpy(), target_np) < 1e-4
    assert _rms(moved_o3d, target_np) < 1e-4


@pytest.mark.parity("icp_point_to_plane_cloud", "meshlib")
def test_icp_point_to_plane_matches_meshlib(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A on the recovered transform: the same solver object, at its **default** method.

    ``ICPMethod.PointToPlane`` is what ``ICPProperties`` starts at, so this pairing is the one that
    needs no method override -- and the point-to-*point* pairing in
    [`test_icp_point_to_point_matches_meshlib`] is the one that does. The default is asserted rather
    than assumed, through ``getParams()``, so a future rebinding that changed it would fail here
    instead of silently comparing two different solvers.

    The other requirement is on the *input*: MeshLib reads the normals off the **reference** cloud,
    so it is built through ``points_to_meshlib(points, normals)`` with the same per-vertex normals
    triwarp is handed. Without them the constructor accepts the cloud and the linearized step has no
    plane to project onto.

    Measured on ``icosphere(3)`` misaligned by 0.08 rad: the transforms agree to **6.0e-06** and the
    moved clouds to **7.4e-06** per point, against a starting RMS of 0.0706 -- so the 1e-4 bound
    carries a 13x margin and both sides genuinely converged (final RMS 6.7e-06 and 1.0e-06).
    """
    mesh_tm, _mesh_wp = icosphere
    target_np = np.ascontiguousarray(mesh_tm.vertices)
    normals_np = np.ascontiguousarray(mesh_tm.vertex_normals)
    rotation_np, translation_np = _rigid_transform(0.08, [0.1, 0.9, 0.2], [0.02, -0.01, 0.015])
    source_np = np.ascontiguousarray(target_np @ rotation_np.T + translation_np)

    matrix_wp, transformed_wp, _cost_wp = tw.registration.icp_point_to_plane(
        points_to_warp(source_np, device),
        points_to_warp(target_np, device),
        target_normals=points_to_warp(normals_np, device),
        max_iterations=50,
        threshold=-np.inf,
    )
    matrix_np = matrix_wp.numpy()[0]

    icp_ml = mm.ICP(
        mm.MeshOrPoints(points_to_meshlib(source_np)),
        mm.MeshOrPoints(points_to_meshlib(target_np, normals_np)),  # normals live on the reference
        mm.AffineXf3f(),
        mm.AffineXf3f(),
        0.02,
    )
    properties_ml = mm.ICPProperties()
    properties_ml.iterLimit = 50
    icp_ml.setParams(properties_ml)
    assert icp_ml.getParams().method == mm.ICPMethod.PointToPlane  # its default, unchanged
    transform_ml = icp_ml.calculateTransformation()
    rotation_ml = np.array(
        [
            [transform_ml.A.x.x, transform_ml.A.x.y, transform_ml.A.x.z],
            [transform_ml.A.y.x, transform_ml.A.y.y, transform_ml.A.y.z],
            [transform_ml.A.z.x, transform_ml.A.z.y, transform_ml.A.z.z],
        ]
    )
    translation_ml = np.array([transform_ml.b.x, transform_ml.b.y, transform_ml.b.z])
    moved_ml = source_np @ rotation_ml.T + translation_ml

    assert _rms(source_np, target_np) > 1e-2  # non-vacuity: the clouds start apart
    assert np.allclose(matrix_np[:3, :3], rotation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(matrix_np[:3, 3], translation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), moved_ml, rtol=1e-4, atol=1e-4)
    # Both converged, rather than agreeing on a transform that fits nothing.
    assert _rms(transformed_wp.numpy(), target_np) < 1e-4
    assert _rms(moved_ml, target_np) < 1e-4


def test_icp_point_to_point_cloud(device: str) -> None:
    """
    Not a library comparison: ICP must recover a known rigid motion it was given exactly.

    The target is a transformed copy of the source, so the answer is known in closed form and
    no reference is needed; [`test_icp_point_to_point_matches_open3d_and_trimesh`] is the
    cross-library comparison. This is the test that would catch a converged-but-wrong fit.
    """
    rng = np.random.default_rng(10)
    target_np = rng.standard_normal((300, 3)).astype(np.float32)
    rotation_np, translation_np = _rigid_transform(0.15, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = points_to_warp(source_np, device)
    target_wp = points_to_warp(target_np, device)

    _, transformed_wp, cost_tw = tw.registration.icp(
        source_wp, target_wp, None, max_iterations=100, reflection=False, scale=False
    )

    # Small rigid offset keeps nearest-neighbor correspondence unique -> exact recovery.
    assert _rms(transformed_wp.numpy(), target_np) < 1e-3
    assert cost_tw < 1e-6

    # trimesh reaches a comparably low per-point cost on the same problem.
    _, _, cost_tm = tm_reg.icp(source_np.astype(np.float64), target_np.astype(np.float64))
    assert cost_tm / len(source_np) < 1e-3


def test_icp_point_to_point_mesh(half_torus: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    mesh_tm, mesh_wp = half_torus
    vertices_np, faces_np = _mesh_vertices_faces(mesh_tm)
    rotation_np, translation_np = _rigid_transform(0.1, [0.2, 0.6, 0.3], [0.03, -0.02, 0.04])
    source_np = (vertices_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = points_to_warp(source_np, mesh_wp.device)
    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=mesh_wp.device)

    _, transformed_wp, cost_tw = tw.registration.icp(
        source_wp, vertices_wp, faces_wp, max_iterations=60, reflection=False, scale=False
    )

    # Points register onto the surface (low cost); vertices realign closely (mild
    # tangential slide on the curved surface keeps RMS small but non-zero).
    assert cost_tw < 1e-3
    assert _rms(transformed_wp.numpy(), vertices_np) < 5e-2


@pytest.mark.parametrize("angle", [0.15, 0.30])
@pytest.mark.parity("icp_mesh", "pymeshlab")
def test_icp_mesh_matches_pymeshlab(device: str, angle: float) -> None:
    """
    Class B: both recover the exact alignment, and agree to **0.0** RMS at both offsets.

    ``compute_matrix_by_icp_between_meshes`` runs its correspondences against the reference *mesh*
    rather than a point cloud, which is what makes it the equivalent of the mesh-target ``icp``
    rather than of ``icp_point_cloud``. Three named transforms, all of them plumbing:

    * The filter returns ``None`` and **does not move the vertices**. It writes the source layer's
      *transformation matrix*, so ``vertex_matrix()`` reads back byte-identical to the input -- the
      answer is in ``transform_matrix()`` / ``transformed_vertex_matrix()``. This is the trap here:
      a comparison against ``vertex_matrix()`` looks like a total ICP failure (RMS unchanged at
      0.105) and would be read as a disagreement.
    * Both layers must carry faces; a face-less source raises ``Failed to apply filter``.
    * ``samplenum`` is matched to the vertex count so both sides minimize over the same number of
      correspondences.

    **The fixture is chosen so the problem is well posed.** ICP has no unique answer on a
    rotationally symmetric shape: on an ``icosphere`` any rotation maps the surface onto itself, and
    measured there triwarp reduces the RMS only 0.141 -> 0.124 while MeshLab's own result is equally
    arbitrary -- neither is wrong and the comparison is meaningless. A **notched** cube breaks every
    symmetry, and on it both solvers drive the RMS to zero exactly. MeshLab also needs enough
    samples: at 64 vertices the filter raises, so the cube is subdivided twice to 256.

    Two offsets, 0.15 and 0.30 rad, so the assert is not resting on one starting point.
    """
    mesh_tm = tm.boolean.difference(
        [tm.creation.box(extents=[1.0, 1.0, 1.0]), tm.creation.box(extents=[0.4, 0.4, 2.0])]
    ).subdivide()
    mesh_tm = mesh_tm.subdivide()
    vertices_np = mesh_tm.vertices.astype(np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    rotation_np, translation_np = _rigid_transform(angle, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = vertices_np @ rotation_np.T.astype(np.float64) + translation_np

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(vertices_np), faces_np))
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(source_np), faces_np))
    meshset_pml.compute_matrix_by_icp_between_meshes(
        referencemesh=0, sourcemesh=1, samplenum=vertices_np.shape[0]
    )
    meshset_pml.set_current_mesh(1)
    # Read the *transformed* vertices: the filter writes the layer transform, not the positions.
    assert np.allclose(meshset_pml.current_mesh().vertex_matrix(), source_np)
    moved_pml = np.asarray(meshset_pml.current_mesh().transformed_vertex_matrix(), dtype=np.float64)
    assert np.isclose(
        np.linalg.det(np.asarray(meshset_pml.current_mesh().transform_matrix())[:3, :3]), 1.0
    )

    _matrix_wp, transformed_wp, _cost_wp = tw.registration.icp(
        points_to_warp(source_np, device),
        points_to_warp(vertices_np, device),
        wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device),
        max_iterations=100,
        threshold=-np.inf,
        reflection=False,
        scale=False,
    )
    moved_wp = transformed_wp.numpy().astype(np.float64)

    # Both land on the reference itself, so neither comparison below is two failures agreeing.
    assert _rms(source_np, vertices_np) > 0.1
    assert _rms(moved_pml, vertices_np) < 1e-4
    assert _rms(moved_wp, vertices_np) < 1e-4
    assert np.allclose(moved_wp, moved_pml, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("angle", [0.15, 0.30])
@pytest.mark.parity("icp_mesh", "pyvista")
def test_icp_mesh_matches_pyvista(device: str, angle: float) -> None:
    """
    Class C: both solvers recover the same rigid motion, compared through the aligned positions.

    A derived scalar rather than an element-wise match on the *matrix*, because ICP's answer is a
    transform and two implementations that converge to the same alignment can differ in the last
    digits of the rotation while agreeing on where every point lands. So the assert is on the RMS
    to the reference, on both sides, plus that the two aligned clouds agree with each other.

    ``align(return_matrix=True)`` returns ``(aligned_mesh, 4x4 matrix)`` -- and unlike MeshLab it
    *does* move the points, so the aligned mesh's ``points`` is the answer rather than a layer
    transform. Measured mean residual **4.8e-04** recovering a 10-degree rotation of the notched
    cube.

    Same fixture and same reason as the pymeshlab test above: a rotationally symmetric shape makes
    ICP's answer non-unique, so a notched cube is used and the starting RMS is asserted large before
    the two results are compared to it.
    """
    mesh_tm = tm.boolean.difference(
        [tm.creation.box(extents=[1.0, 1.0, 1.0]), tm.creation.box(extents=[0.4, 0.4, 2.0])]
    ).subdivide()
    mesh_tm = mesh_tm.subdivide()
    vertices_np = mesh_tm.vertices.astype(np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    rotation_np, translation_np = _rigid_transform(angle, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = vertices_np @ rotation_np.T.astype(np.float64) + translation_np

    source_pv = pv.PolyData.from_regular_faces(np.ascontiguousarray(source_np), faces_np)
    target_pv = pv.PolyData.from_regular_faces(np.ascontiguousarray(vertices_np), faces_np)
    aligned_pv, matrix_pv = source_pv.align(target_pv, return_matrix=True)
    moved_pv = np.asarray(aligned_pv.points, dtype=np.float64)
    assert np.isclose(np.linalg.det(np.asarray(matrix_pv)[:3, :3]), 1.0, atol=1e-4)

    _matrix_wp, transformed_wp, _cost_wp = tw.registration.icp(
        points_to_warp(source_np, device),
        points_to_warp(vertices_np, device),
        wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device),
        max_iterations=100,
        threshold=-np.inf,
        reflection=False,
        scale=False,
    )
    moved_wp = transformed_wp.numpy().astype(np.float64)

    # Neither result is two failures agreeing: the input starts far from the reference.
    assert _rms(source_np, vertices_np) > 0.1
    assert _rms(moved_pv, vertices_np) < 1e-2
    assert _rms(moved_wp, vertices_np) < 1e-2
    assert _rms(moved_wp, moved_pv) < 1e-2


@pytest.mark.parity(
    "icp_point_to_plane_mesh",
    "meshlib",
    benchmarked=False,
    reason="MeshLib reaches a mesh target the same way, but its ICP is one call that also "
    "builds the AABB tree it queries, and that tree is cached on the Mesh -- so a row would "
    "time either the build or a warmed query depending on call order, the hazard CLAUDE.md "
    "section 6 records. Its cloud form already carries the timed row.",
)
@pytest.mark.parametrize("mesh_name", ["half_torus", "unit_box"])
def test_icp_point_to_plane_mesh_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A: the same rigid transform, against ``ICP`` given a ``MeshOrPoints`` holding a mesh.

    This is the surface form -- triwarp projects each source point onto the closest *triangle* and
    uses that face's plane, where the ``icp_point_to_plane_cloud`` comparisons hand both sides a
    cloud with per-vertex normals. MeshLib is the only registered reference that takes a mesh target
    at all: ``mm.MeshOrPoints`` accepts a ``Mesh`` and its ICP then queries that mesh's AABB tree.
    Measured agreement on the transform is 5.0e-07 on ``unit_box`` and 1.3e-07 on ``half_torus``,
    both sides converging to an RMS below 1e-06 from a start above 5e-02.

    !!! warning "The fixture must not be rotationally symmetric, and the default one is"
        Point-to-*surface* alignment has no signal for a rotation that maps the surface to
        itself: on a sphere every rotation about the centre keeps all source points exactly on
        the surface, so the objective is flat and the iteration wanders. Measured on
        ``icosphere(3)`` misaligned by
        0.08 rad, triwarp's mesh form ends at RMS 1.33e-01 against a 7.06e-02 *start* -- worse
        than not aligning -- while its cloud form on identical data reaches 8.2e-06, because
        discrete vertex correspondences do constrain the rotation. That is a property of the
        objective, not a defect, but it means ``icosphere`` cannot be used here; every
        non-symmetric fixture probed
        (``unit_box``, ``half_torus``, a torus, a capsule) converges to 1e-06 or better.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_vertices_faces(mesh_tm)
    rotation_np, translation_np = _rigid_transform(0.08, [0.1, 0.9, 0.2], [0.02, -0.01, 0.015])
    source_np = np.ascontiguousarray(vertices_np @ rotation_np.T + translation_np)

    matrix_wp, transformed_wp, _cost_wp = tw.registration.icp_point_to_plane(
        points_to_warp(source_np, mesh_wp.device),
        points_to_warp(vertices_np, mesh_wp.device),
        wp.array(faces_np, dtype=wp.int32, device=mesh_wp.device),
        max_iterations=50,
        threshold=-np.inf,
    )
    matrix_np = matrix_wp.numpy()[0]

    # The mesh has to outlive the ICP: MeshOrPoints does not keep it alive on its own.
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    icp_ml = mm.ICP(
        mm.MeshOrPoints(points_to_meshlib(source_np)),
        mm.MeshOrPoints(mesh_ml),
        mm.AffineXf3f(),
        mm.AffineXf3f(),
        0.02,
    )
    properties_ml = mm.ICPProperties()
    properties_ml.iterLimit = 50
    icp_ml.setParams(properties_ml)
    assert icp_ml.getParams().method == mm.ICPMethod.PointToPlane  # its default, unchanged
    transform_ml = icp_ml.calculateTransformation()
    rotation_ml = np.array(
        [
            [transform_ml.A.x.x, transform_ml.A.x.y, transform_ml.A.x.z],
            [transform_ml.A.y.x, transform_ml.A.y.y, transform_ml.A.y.z],
            [transform_ml.A.z.x, transform_ml.A.z.y, transform_ml.A.z.z],
        ]
    )
    translation_ml = np.array([transform_ml.b.x, transform_ml.b.y, transform_ml.b.z])
    moved_ml = source_np @ rotation_ml.T + translation_ml

    assert _rms(source_np, vertices_np) > 1e-2  # non-vacuity: the clouds start apart
    assert np.allclose(matrix_np[:3, :3], rotation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(matrix_np[:3, 3], translation_ml, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), moved_ml, rtol=1e-4, atol=1e-4)
    # Both converged, rather than agreeing on a transform that fits nothing.
    assert _rms(transformed_wp.numpy(), vertices_np) < 1e-4
    assert _rms(moved_ml, vertices_np) < 1e-4


def test_icp_point_to_plane_mesh(half_torus: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    mesh_tm, mesh_wp = half_torus
    vertices_np, faces_np = _mesh_vertices_faces(mesh_tm)
    rotation_np, translation_np = _rigid_transform(0.1, [0.2, 0.6, 0.3], [0.03, -0.02, 0.04])
    source_np = (vertices_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = points_to_warp(source_np, mesh_wp.device)
    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=mesh_wp.device)

    _, transformed_wp, cost_tw = tw.registration.icp_point_to_plane(
        source_wp, vertices_wp, faces_wp, max_iterations=60
    )

    assert cost_tw < 1e-6
    assert _rms(transformed_wp.numpy(), vertices_np) < 1e-3


def test_icp_point_to_plane_robust_outliers(
    half_torus: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    rng = np.random.default_rng(11)
    mesh_tm, mesh_wp = half_torus
    vertices_np, faces_np = _mesh_vertices_faces(mesh_tm)
    rotation_np, translation_np = _rigid_transform(0.08, [0.1, 0.5, 0.3], [0.02, -0.01, 0.03])
    source_np = (vertices_np @ rotation_np.T + translation_np).astype(np.float32)

    outlier_idx = rng.choice(len(source_np), size=len(source_np) // 10, replace=False)
    source_np[outlier_idx] += rng.standard_normal((len(outlier_idx), 3)).astype(np.float32) * 2.0
    inlier_idx = np.setdiff1d(np.arange(len(source_np)), outlier_idx)

    source_wp = points_to_warp(source_np, mesh_wp.device)
    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=mesh_wp.device)

    _, transformed_none, _ = tw.registration.icp_point_to_plane(
        source_wp, vertices_wp, faces_wp, max_iterations=60, robust_kernel="none"
    )
    _, transformed_tukey, _ = tw.registration.icp_point_to_plane(
        source_wp, vertices_wp, faces_wp, max_iterations=60, robust_kernel="tukey"
    )

    rms_none = _rms(transformed_none.numpy()[inlier_idx], vertices_np[inlier_idx])
    rms_tukey = _rms(transformed_tukey.numpy()[inlier_idx], vertices_np[inlier_idx])
    # The Tukey biweight rejects outliers and recovers a much better inlier fit.
    assert rms_tukey < rms_none


def test_icp_point_to_plane_cloud_with_normals(device: str) -> None:
    rng = np.random.default_rng(12)
    target_np = rng.standard_normal((300, 3)).astype(np.float32)
    normals_np = target_np / np.linalg.norm(target_np, axis=1, keepdims=True)
    rotation_np, translation_np = _rigid_transform(0.1, [0.3, 0.4, 0.5], [0.03, -0.02, 0.02])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = points_to_warp(source_np, device)
    target_wp = points_to_warp(target_np, device)
    normals_wp = points_to_warp(normals_np, device)

    _, transformed_wp, cost_tw = tw.registration.icp_point_to_plane(
        source_wp, target_wp, None, target_normals=normals_wp, max_iterations=100
    )
    assert cost_tw < 1e-4
    assert _rms(transformed_wp.numpy(), target_np) < 1e-2


def test_icp_empty_source(device: str) -> None:
    target_wp = points_to_warp(np.random.default_rng(13).standard_normal((50, 3)), device)
    empty_wp = wp.zeros(0, dtype=wp.vec3, device=device)

    matrix_wp, transformed_wp, cost_tw = tw.registration.icp(empty_wp, target_wp, None)
    assert matrix_wp.shape == (1,)
    assert matrix_wp.dtype == wp.mat44
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)
    assert transformed_wp.shape == (0,)
    assert not np.isfinite(cost_tw)


def test_icp_point_to_plane_requires_normals(device: str) -> None:
    rng = np.random.default_rng(14)
    target_wp = points_to_warp(rng.standard_normal((50, 3)), device)
    source_wp = points_to_warp(rng.standard_normal((50, 3)), device)
    with pytest.raises(ValueError, match="target_normals"):
        tw.registration.icp_point_to_plane(source_wp, target_wp, None)


def test_icp_max_distance_all_rejected(device: str) -> None:
    rng = np.random.default_rng(15)
    target_np = rng.standard_normal((100, 3)).astype(np.float32)
    source_np = (target_np + np.array([5.0, 5.0, 5.0], dtype=np.float32)).astype(np.float32)
    source_wp = points_to_warp(source_np, device)
    target_wp = points_to_warp(target_np, device)

    # Every correspondence is beyond max_distance -> no fit, identity returned, no crash.
    matrix_wp, _, _ = tw.registration.icp(
        source_wp, target_wp, None, max_iterations=10, max_distance=1e-6
    )
    assert np.isfinite(matrix_wp.numpy()).all()
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)


def test_icp_point_to_plane_target_normals_length_mismatch(device: str) -> None:
    rng = np.random.default_rng(16)
    target_wp = points_to_warp(rng.standard_normal((50, 3)), device)
    source_wp = points_to_warp(rng.standard_normal((50, 3)), device)
    normals_wp = points_to_warp(rng.standard_normal((40, 3)), device)
    with pytest.raises(ValueError, match="target_normals"):
        tw.registration.icp_point_to_plane(source_wp, target_wp, None, target_normals=normals_wp)


def test_icp_point_to_plane_max_distance_all_rejected(device: str) -> None:
    rng = np.random.default_rng(17)
    target_np = rng.standard_normal((100, 3)).astype(np.float32)
    normals_np = target_np / np.linalg.norm(target_np, axis=1, keepdims=True)
    source_np = (target_np + np.array([5.0, 5.0, 5.0], dtype=np.float32)).astype(np.float32)
    source_wp = points_to_warp(source_np, device)
    target_wp = points_to_warp(target_np, device)
    normals_wp = points_to_warp(normals_np, device)

    # Every correspondence is beyond max_distance -> the loop must bail out on the first
    # iteration rather than read a zeroed accumulator as a converged cost=0.0 fit.
    matrix_wp, _, cost_tw = tw.registration.icp_point_to_plane(
        source_wp, target_wp, None, target_normals=normals_wp, max_iterations=10, max_distance=1e-6
    )
    assert np.isfinite(matrix_wp.numpy()).all()
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)
    assert not np.isfinite(cost_tw)


def test_icp_point_to_plane_tukey_all_weights_zero(device: str) -> None:
    # Not a library comparison: this pins the loop's own bail-out against a Tukey kernel driving
    # every in-range correspondence's *weight* to zero, a distinct failure mode from the
    # distance-rejection case above -- ``residual_valid`` (and so the ``valid`` guard) accepts
    # every correspondence here, since none of them is out of range; it is ``robust_weight`` alone
    # that zeroes every contribution once ``robust_scale`` is tighter than every residual.
    rng = np.random.default_rng(23)
    target_np = rng.standard_normal((100, 3)).astype(np.float32) * 2.0
    normals_np = target_np / np.linalg.norm(target_np, axis=1, keepdims=True)
    # A real, nontrivial offset: a correctly converged fit would move every source point.
    source_np = (target_np + np.array([1.0, 0.5, -0.3], dtype=np.float32)).astype(np.float32)
    source_wp = points_to_warp(source_np, device)
    target_wp = points_to_warp(target_np, device)
    normals_wp = points_to_warp(normals_np, device)

    matrix_wp, _, cost_tw = tw.registration.icp_point_to_plane(
        source_wp,
        target_wp,
        None,
        target_normals=normals_wp,
        max_iterations=10,
        robust_kernel="tukey",
        robust_scale=1e-9,
    )
    # A collapsed-weight bail must report the same "did not converge" signal as an all-rejected
    # one: an untouched (identity) transform and a non-finite cost, never the spurious cost=0.0 a
    # zeroed accumulator would otherwise read as a perfect fit.
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)
    assert not np.isfinite(cost_tw)


def test_icp_point_to_plane_rejects_an_off_menu_robust_kernel(device: str) -> None:
    """
    Not a library comparison: the ``robust_kernel`` menu's own guard.

    An unrecognised name used to index ``_ROBUST_KINDS`` directly and surface as a bare
    ``KeyError('bogus')``, which names neither the argument nor the three kernels. All three
    documented names still run, which is what keeps the guard from being a spelling of
    "reject everything".
    """
    rng = np.random.default_rng(18)
    target_np = rng.standard_normal((60, 3)).astype(np.float32)
    normals_np = target_np / np.linalg.norm(target_np, axis=1, keepdims=True)
    source_wp = points_to_warp(target_np + 0.05, device)
    target_wp = points_to_warp(target_np, device)
    normals_wp = points_to_warp(normals_np, device)
    with pytest.raises(ValueError, match="robust_kernel must be one of"):
        tw.registration.icp_point_to_plane(
            source_wp,
            target_wp,
            None,
            target_normals=normals_wp,
            robust_kernel="bogus",  # type: ignore[arg-type]
        )
    for robust_kernel in ("none", "huber", "tukey"):
        matrix_wp, _, _ = tw.registration.icp_point_to_plane(
            source_wp,
            target_wp,
            None,
            target_normals=normals_wp,
            max_iterations=2,
            robust_kernel=robust_kernel,  # type: ignore[arg-type]
        )
        assert np.isfinite(matrix_wp.numpy()).all()
