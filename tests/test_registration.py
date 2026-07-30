"""Regression tests for ``triwarp.registration`` against ``trimesh.registration``."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import trimesh.registration as tm_reg
import warp as wp

import triwarp as tw


def _make_point_clouds(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    a_np = rng.standard_normal((n, 3)).astype(np.float64)
    # b is a mildly rotated/translated version of a to keep correspondence meaningful
    b_np = rng.standard_normal((n, 3)).astype(np.float64)
    return a_np, b_np


def _to_wp(arr_np: np.ndarray, device: str) -> wp.array:
    return wp.array(arr_np.astype(np.float32), dtype=wp.vec3, device=device)


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

    a_wp = _to_wp(a_np, device)
    b_wp = _to_wp(b_np, device)
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
    rng = np.random.default_rng(0)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_uniform_weights(device: str) -> None:
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
    rng = np.random.default_rng(3)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, reflection=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_translation(device: str) -> None:
    rng = np.random.default_rng(4)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, translation=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_scale(device: str) -> None:
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
    Single-pass moments must survive a cloud far from the origin.

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
    """Non-binary weights: the masked covariance and the weighted moments must agree."""
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
    rng = np.random.default_rng(6)
    a_np, b_np = _make_point_clouds(rng)
    a_wp = _to_wp(a_np, device)
    b_wp = _to_wp(b_np, device)

    result = tw.registration.procrustes(a_wp, b_wp, return_cost=False)
    assert isinstance(result, wp.array)
    assert result.shape == (1,)
    assert result.dtype == wp.mat44

    matrix_tm, _, _ = tm_reg.procrustes(a_np, b_np)
    assert np.allclose(result.numpy()[0], matrix_tm, rtol=1e-4, atol=1e-4)


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


def test_icp_point_to_point_cloud(device: str) -> None:
    rng = np.random.default_rng(10)
    target_np = rng.standard_normal((300, 3)).astype(np.float32)
    rotation_np, translation_np = _rigid_transform(0.15, [0.2, 0.7, 0.1], [0.05, -0.03, 0.04])
    source_np = (target_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = _to_wp(source_np, device)
    target_wp = _to_wp(target_np, device)

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

    source_wp = _to_wp(source_np, mesh_wp.device)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=mesh_wp.device)

    _, transformed_wp, cost_tw = tw.registration.icp(
        source_wp, vertices_wp, faces_wp, max_iterations=60, reflection=False, scale=False
    )

    # Points register onto the surface (low cost); vertices realign closely (mild
    # tangential slide on the curved surface keeps RMS small but non-zero).
    assert cost_tw < 1e-3
    assert _rms(transformed_wp.numpy(), vertices_np) < 5e-2


def test_icp_point_to_plane_mesh(half_torus: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    mesh_tm, mesh_wp = half_torus
    vertices_np, faces_np = _mesh_vertices_faces(mesh_tm)
    rotation_np, translation_np = _rigid_transform(0.1, [0.2, 0.6, 0.3], [0.03, -0.02, 0.04])
    source_np = (vertices_np @ rotation_np.T + translation_np).astype(np.float32)

    source_wp = _to_wp(source_np, mesh_wp.device)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=mesh_wp.device)
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

    source_wp = _to_wp(source_np, mesh_wp.device)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=mesh_wp.device)
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

    source_wp = _to_wp(source_np, device)
    target_wp = _to_wp(target_np, device)
    normals_wp = _to_wp(normals_np, device)

    _, transformed_wp, cost_tw = tw.registration.icp_point_to_plane(
        source_wp, target_wp, None, target_normals=normals_wp, max_iterations=100
    )
    assert cost_tw < 1e-4
    assert _rms(transformed_wp.numpy(), target_np) < 1e-2


def test_icp_empty_source(device: str) -> None:
    target_wp = _to_wp(np.random.default_rng(13).standard_normal((50, 3)), device)
    empty_wp = wp.zeros(0, dtype=wp.vec3, device=device)

    matrix_wp, transformed_wp, cost_tw = tw.registration.icp(empty_wp, target_wp, None)
    assert matrix_wp.shape == (1,)
    assert matrix_wp.dtype == wp.mat44
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)
    assert transformed_wp.shape == (0,)
    assert not np.isfinite(cost_tw)


def test_icp_point_to_plane_requires_normals(device: str) -> None:
    rng = np.random.default_rng(14)
    target_wp = _to_wp(rng.standard_normal((50, 3)), device)
    source_wp = _to_wp(rng.standard_normal((50, 3)), device)
    with pytest.raises(ValueError, match="target_normals"):
        tw.registration.icp_point_to_plane(source_wp, target_wp, None)


def test_icp_max_distance_all_rejected(device: str) -> None:
    rng = np.random.default_rng(15)
    target_np = rng.standard_normal((100, 3)).astype(np.float32)
    source_np = (target_np + np.array([5.0, 5.0, 5.0], dtype=np.float32)).astype(np.float32)
    source_wp = _to_wp(source_np, device)
    target_wp = _to_wp(target_np, device)

    # Every correspondence is beyond max_distance -> no fit, identity returned, no crash.
    matrix_wp, _, _ = tw.registration.icp(
        source_wp, target_wp, None, max_iterations=10, max_distance=1e-6
    )
    assert np.isfinite(matrix_wp.numpy()).all()
    assert np.allclose(matrix_wp.numpy()[0], np.eye(4), atol=1e-6)
