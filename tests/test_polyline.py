from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from trimesh.path import segments as tm_segments
from trimesh.path import traversal as tm_traversal

import triwarp as tw


def _polyline_wp(pts_np: np.ndarray, device: str) -> wp.array:
    return wp.array(np.ascontiguousarray(pts_np.astype(np.float32)), dtype=wp.vec3, device=device)


def _random_open_polyline(seed: int, n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


def _closed_from(pts_np: np.ndarray) -> np.ndarray:
    return np.concatenate([pts_np, pts_np[:1]], axis=0)


# --- NumPy reference implementations (mirroring the source PyTorch logic) ---


def _centroid_np(pts: np.ndarray) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(segments, axis=-1, keepdims=True)
    midpoints = pts[:-1] + segments / 2
    return (midpoints * seg_len).sum(axis=0) / seg_len.sum()


def _normal_np(pts: np.ndarray) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    normal = np.cross(segments[:-1], segments[1:]).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _closed_normal_np(pts: np.ndarray) -> np.ndarray:
    pts = _closed_from(pts)
    segments = np.diff(pts, axis=0)
    normal = np.cross(segments, np.roll(segments, -1, axis=0)).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _angles_np(pts: np.ndarray) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    rolled = np.roll(segments, -1, axis=0)
    cos = np.sum(segments * rolled, axis=-1) / (
        np.linalg.norm(segments, axis=-1) * np.linalg.norm(rolled, axis=-1)
    )
    angles = np.arccos(np.clip(cos, -1.0, 1.0))
    if np.allclose(pts[0], pts[-1]):
        return np.concatenate([angles, angles[:1]])
    angles[-1] = 0.0
    return np.concatenate([angles[-1:], angles])


def _upsample_np(pts: np.ndarray, step: float) -> np.ndarray:
    segments = np.diff(pts, axis=0)
    steps = np.clip(np.linalg.norm(segments, axis=-1) // step, 1, None).astype(np.int64)
    cumulative = np.roll(np.cumsum(steps), 1)
    total = int(cumulative[0])
    cumulative[0] = 0
    indices = np.repeat(np.arange(len(steps)), steps)
    weights = (np.arange(total) - cumulative[indices]) / steps[indices]
    return pts[indices] + weights[:, None] * segments[indices]


def _downsample_np(pts: np.ndarray, step: float) -> np.ndarray:
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    keep = [0]
    last = 0.0
    for i in range(1, len(pts)):
        if cumulative[i] - last >= step:
            keep.append(i)
            last = cumulative[i]
    return pts[keep]


def _resample_np(pts: np.ndarray, num_points: int) -> np.ndarray:
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc = np.linspace(0.0, cumulative[-1], num_points)
    return np.stack([np.interp(arc, cumulative, pts[:, k]) for k in range(3)], axis=1)


def _distance_np(points: np.ndarray, pts: np.ndarray) -> np.ndarray:
    starts = pts[:-1][None]
    segments = np.diff(pts, axis=0)[None]
    seg_sq = (segments**2).sum(axis=-1, keepdims=True).clip(1e-8)
    vectors = points[:, None] - starts
    t = np.clip((vectors * segments).sum(axis=-1, keepdims=True) / seg_sq, 0.0, 1.0)
    closest = starts + t * segments
    return np.linalg.norm(points[:, None] - closest, axis=-1).min(axis=1)


def _radius_np(
    pts: np.ndarray,
    reduction: str,
    center: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> float:
    if center is None:
        center = _centroid_np(pts)
    if normal is None:
        normal = _normal_np(pts)
    else:
        normal = normal / np.linalg.norm(normal)
    projected = pts - normal * np.sum((pts - center) * normal, axis=-1, keepdims=True)
    segments = np.diff(projected, axis=0)
    seg_sq = np.sum(segments**2, axis=-1).clip(1e-8)
    double_areas = -np.sum((projected[:-1] - center) * segments, axis=-1)
    t = np.clip(double_areas / seg_sq, 0.0, 1.0)
    closest = projected[:-1] + t[:, None] * segments
    distances = np.linalg.norm(closest - center, axis=-1)
    return float(getattr(np, reduction)(distances))


# --- open / close ---


def test_open_polyline_drops_duplicate_endpoint(device: str) -> None:
    pts_np = _random_open_polyline(0)
    closed_np = _closed_from(pts_np)
    opened_wp = tw.polyline.open_polyline(_polyline_wp(closed_np, device))
    assert np.allclose(opened_wp.numpy(), pts_np.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_open_polyline_leaves_open_unchanged(device: str) -> None:
    pts_np = _random_open_polyline(1)
    opened_wp = tw.polyline.open_polyline(_polyline_wp(pts_np, device))
    assert np.allclose(opened_wp.numpy(), pts_np.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_close_polyline_appends_first_point(device: str) -> None:
    pts_np = _random_open_polyline(2)
    closed_wp = tw.polyline.close_polyline(_polyline_wp(pts_np, device))
    expected = _closed_from(pts_np).astype(np.float32)
    assert np.allclose(closed_wp.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_close_polyline_leaves_closed_unchanged(device: str) -> None:
    closed_np = _closed_from(_random_open_polyline(3))
    closed_wp = tw.polyline.close_polyline(_polyline_wp(closed_np, device))
    assert np.allclose(closed_wp.numpy(), closed_np.astype(np.float32), rtol=1e-5, atol=1e-5)


# --- length (trimesh oracle) ---


@pytest.mark.parametrize("seed", [10, 11])
def test_polyline_length_matches_trimesh(device: str, seed: int) -> None:
    pts_np = _random_open_polyline(seed)
    segs = np.stack([pts_np[:-1], pts_np[1:]], axis=1)
    length_tm = tm_segments.length(segs, summed=True)
    length_wp = tw.polyline.polyline_length(_polyline_wp(pts_np, device))
    assert np.allclose(length_wp, length_tm, rtol=1e-4, atol=1e-4)


def test_closed_polyline_length_matches_trimesh(device: str) -> None:
    pts_np = _random_open_polyline(12)
    closed_np = _closed_from(pts_np)
    segs = np.stack([closed_np[:-1], closed_np[1:]], axis=1)
    length_tm = tm_segments.length(segs, summed=True)
    length_wp = tw.polyline.closed_polyline_length(_polyline_wp(pts_np, device))
    assert np.allclose(length_wp, length_tm, rtol=1e-4, atol=1e-4)


# --- centroid / normal (NumPy reference) ---


def test_polyline_centroid_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(20)
    centroid_wp = tw.polyline.polyline_centroid(_polyline_wp(pts_np, device))
    assert np.allclose(list(centroid_wp), _centroid_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_normal_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(21)
    normal_wp = tw.polyline.polyline_normal(_polyline_wp(pts_np, device))
    normal_np = _normal_np(pts_np)
    # normal sign is well-defined by the summed cross products; compare directly.
    assert np.allclose(list(normal_wp), normal_np, rtol=1e-4, atol=1e-4)


def test_closed_polyline_normal_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(22)
    normal_wp = tw.polyline.closed_polyline_normal(_polyline_wp(pts_np, device))
    assert np.allclose(list(normal_wp), _closed_normal_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_normal_requires_three_points(device: str) -> None:
    with pytest.raises(ValueError, match="three points"):
        tw.polyline.polyline_normal(_polyline_wp(_random_open_polyline(0, n=2), device))


# --- angles (NumPy reference; trimesh vector_angle cross-check) ---


def test_polyline_angles_open_matches_reference(device: str) -> None:
    pts_np = _random_open_polyline(30)
    angles_wp = tw.polyline.polyline_angles(_polyline_wp(pts_np, device))
    assert np.allclose(angles_wp.numpy(), _angles_np(pts_np), rtol=1e-4, atol=1e-4)


def test_polyline_angles_closed_matches_reference(device: str) -> None:
    closed_np = _closed_from(_random_open_polyline(31))
    angles_wp = tw.polyline.polyline_angles(_polyline_wp(closed_np, device))
    assert np.allclose(angles_wp.numpy(), _angles_np(closed_np), rtol=1e-4, atol=1e-4)


def test_closed_polyline_angles_length_matches_original(device: str) -> None:
    pts_np = _random_open_polyline(32)
    angles_wp = tw.polyline.closed_polyline_angles(_polyline_wp(pts_np, device))
    assert angles_wp.shape[0] == pts_np.shape[0]


# --- distance (NumPy reference) ---


def test_distance_to_polyline_matches_reference(device: str) -> None:
    rng = np.random.default_rng(40)
    pts_np = _random_open_polyline(40)
    points_np = rng.standard_normal((25, 3))
    points_wp = _polyline_wp(points_np, device)
    distances_wp = tw.polyline.distance_to_polyline(points_wp, _polyline_wp(pts_np, device))
    assert np.allclose(distances_wp.numpy(), _distance_np(points_np, pts_np), rtol=1e-4, atol=1e-4)


def test_distance_to_single_point_polyline(device: str) -> None:
    rng = np.random.default_rng(41)
    points_np = rng.standard_normal((10, 3))
    target_np = rng.standard_normal((1, 3))
    distances_wp = tw.polyline.distance_to_polyline(
        _polyline_wp(points_np, device), _polyline_wp(target_np, device)
    )
    expected = np.linalg.norm(points_np - target_np[0], axis=-1)
    assert np.allclose(distances_wp.numpy(), expected, rtol=1e-4, atol=1e-4)


# --- upsample (NumPy reference) ---


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_upsample_polyline_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(50)
    upsampled_wp = tw.polyline.upsample_polyline(_polyline_wp(pts_np, device), step)
    assert np.allclose(upsampled_wp.numpy(), _upsample_np(pts_np, step), rtol=1e-4, atol=1e-4)


def test_upsample_point_count(device: str) -> None:
    pts_np = _random_open_polyline(51)
    step = 0.4
    seg_len = np.linalg.norm(np.diff(pts_np, axis=0), axis=-1)
    expected_count = int(np.clip(seg_len // step, 1, None).astype(np.int64).sum())
    upsampled = tw.polyline.upsample_polyline(_polyline_wp(pts_np, device), step).numpy()
    assert upsampled.shape[0] == expected_count


# --- downsample (NumPy reference) ---


@pytest.mark.parametrize("step", [0.5, 1.5])
def test_downsample_polyline_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(60)
    downsampled_wp = tw.polyline.downsample_polyline(_polyline_wp(pts_np, device), step)
    assert np.allclose(downsampled_wp.numpy(), _downsample_np(pts_np, step), rtol=1e-4, atol=1e-4)


# --- resample (NumPy interp reference + trimesh oracle) ---


@pytest.mark.parametrize("num_points", [5, 50])
def test_resample_polyline_matches_numpy_interp(device: str, num_points: int) -> None:
    pts_np = _random_open_polyline(70)
    resampled_wp = tw.polyline.resample_polyline(_polyline_wp(pts_np, device), num_points)
    assert np.allclose(resampled_wp.numpy(), _resample_np(pts_np, num_points), rtol=1e-4, atol=1e-4)


def test_resample_polyline_matches_trimesh(device: str) -> None:
    pts_np = _random_open_polyline(71)
    num_points = 40
    resampled_tm = tm_traversal.resample_path(pts_np, count=num_points)
    resampled_wp = tw.polyline.resample_polyline(_polyline_wp(pts_np, device), num_points)
    assert np.allclose(resampled_wp.numpy(), resampled_tm, rtol=1e-3, atol=1e-3)


def test_resample_closed_polyline_shape_and_endpoints(device: str) -> None:
    pts_np = _random_open_polyline(72)
    num_points = 16
    resampled_wp = tw.polyline.resample_closed_polyline(_polyline_wp(pts_np, device), num_points)
    resampled = resampled_wp.numpy()
    assert resampled.shape == (num_points, 3)
    # first sample is the closed-polyline start; there is no duplicated closing point
    assert np.allclose(resampled[0], pts_np[0].astype(np.float32), rtol=1e-4, atol=1e-4)


def test_resample_single_point_repeats(device: str) -> None:
    point_np = np.array([[1.0, 2.0, 3.0]])
    resampled_wp = tw.polyline.resample_polyline(_polyline_wp(point_np, device), 5)
    assert np.allclose(resampled_wp.numpy(), np.repeat(point_np, 5, axis=0), rtol=1e-5, atol=1e-5)


# --- radius (NumPy reference) ---


def _planar_circle(n: int, radius: float) -> np.ndarray:
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], axis=1)


@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_polyline_radius_explicit_plane_matches_reference(device: str, reduction: str) -> None:
    pts_np = _planar_circle(24, radius=2.0)
    center_np = np.zeros(3)
    normal_np = np.array([0.0, 0.0, 1.0])
    radius_wp = tw.polyline.polyline_radius(
        _polyline_wp(pts_np, device),
        reduction,
        wp.vec3(*center_np.tolist()),
        wp.vec3(*normal_np.tolist()),
    )
    radius_np = _radius_np(pts_np, reduction, center_np, normal_np)
    assert np.allclose(radius_wp, radius_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_polyline_radius_default_plane_matches_reference(device: str, reduction: str) -> None:
    pts_np = _random_open_polyline(80)
    radius_wp = tw.polyline.polyline_radius(_polyline_wp(pts_np, device), reduction)
    radius_np = _radius_np(pts_np, reduction)
    assert np.allclose(radius_wp, radius_np, rtol=1e-3, atol=1e-3)


def test_polyline_radius_rejects_unknown_reduction(device: str) -> None:
    pts_np = _random_open_polyline(81)
    with pytest.raises(ValueError, match="unsupported reduction"):
        tw.polyline.polyline_radius(_polyline_wp(pts_np, device), "sum")  # type: ignore[arg-type]


# --- new reducers / array helpers ---


@pytest.mark.parametrize("n", [7, 8])
def test_reduce_median_matches_numpy(device: str, n: int) -> None:
    rng = np.random.default_rng(90 + n)
    values_np = rng.standard_normal(n).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    assert np.allclose(tw.reduce.median(values_wp), np.median(values_np), rtol=1e-5, atol=1e-5)


def test_array_allclose_matches_numpy(device: str) -> None:
    rng = np.random.default_rng(95)
    a_np = rng.standard_normal((6, 3)).astype(np.float32)
    a_wp = wp.array(a_np, dtype=wp.vec3, device=device)
    b_close = wp.array(a_np + 1e-9, dtype=wp.vec3, device=device)
    b_far = wp.array(a_np + 1e-2, dtype=wp.vec3, device=device)
    assert tw.array.allclose(a_wp, b_close) == bool(np.allclose(a_np, a_np + 1e-9))
    assert tw.array.allclose(a_wp, b_far) == bool(np.allclose(a_np, a_np + 1e-2))


def test_array_allclose_empty_is_true(device: str) -> None:
    empty = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.array.allclose(empty, empty) is True


# --- edge cases ---


def test_length_short_polyline_is_zero(device: str) -> None:
    single = _polyline_wp(np.array([[0.0, 0.0, 0.0]]), device)
    assert tw.polyline.polyline_length(single) == 0.0


def test_angles_short_polyline_is_zeros(device: str) -> None:
    single = _polyline_wp(np.array([[0.0, 0.0, 0.0]]), device)
    angles = tw.polyline.polyline_angles(single).numpy()
    assert np.array_equal(angles, np.zeros(1, dtype=np.float32))


def test_distance_empty_polyline(device: str) -> None:
    points = _polyline_wp(np.random.default_rng(99).standard_normal((4, 3)), device)
    empty = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.polyline.distance_to_polyline(points, empty).shape[0] == 4
