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


def _endpoint_normals_np(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n1 = np.cross(b, a + c)
    n2 = np.cross(b, a - c)
    plane = n1 if n1 @ n1 >= n2 @ n2 else n2
    if plane @ plane < 1e-6:
        return np.zeros(3), np.zeros(3)
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    nod = unit(np.cross(plane, b))
    return unit(nod + unit(np.cross(plane, a))), unit(nod + unit(np.cross(plane, c)))


def _arc_point_np(
    po: np.ndarray, pd: np.ndarray, no: np.ndarray, nd: np.ndarray, t: float
) -> np.ndarray:
    b = pd - po
    chord = np.linalg.norm(b)
    linear = po + t * b
    if chord < 1e-6:
        return po
    if no @ no < 0.5 or nd @ nd < 0.5:
        return linear
    theta = abs(np.arccos(np.clip(no @ nd, -1.0, 1.0)))
    if theta < 1e-6:
        return linear
    tangent = b / chord
    sign = 1.0 if b @ (nd - no) >= 0.0 else -1.0
    bulge = sign * (no + nd)
    m = bulge - (bulge @ tangent) * tangent
    if m @ m < 1e-6:
        return linear
    m = m / np.linalg.norm(m)
    alpha = 0.5 * theta
    radius = chord / (2.0 * np.sin(alpha))
    center = 0.5 * (po + pd) - radius * np.cos(alpha) * m
    phi = (2.0 * t - 1.0) * alpha
    return center + radius * (np.cos(phi) * m + np.sin(phi) * tangent)


def _smooth_upsample_np(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
    n = len(pts)
    m = n - 1  # distinct vertices when closed (pts[-1] == pts[0])
    segments = np.diff(pts, axis=0)
    steps = np.clip(np.linalg.norm(segments, axis=-1) // step, 1, None).astype(np.int64)
    out = []
    for s in range(n - 1):
        po, pd = pts[s], pts[s + 1]
        no = nd = np.zeros(3)
        if closed:
            a, c = po - pts[(s - 1 + m) % m], pts[(s + 2) % m] - pd
            no, nd = _endpoint_normals_np(a, segments[s], c)
            interior = True
        elif s >= 1 and s + 2 <= n - 1:
            a, c = po - pts[s - 1], pts[s + 2] - pd
            no, nd = _endpoint_normals_np(a, segments[s], c)
            interior = True
        else:
            interior = False
        for k in range(int(steps[s])):
            t = k / steps[s]
            out.append(_arc_point_np(po, pd, no, nd, t) if interior else po + t * (pd - po))
    return np.array(out)


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


# --- smooth upsample (curvature-aware, NumPy reference + circle oracle) ---


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_smooth_upsample_matches_reference(device: str, step: float) -> None:
    pts_np = _random_open_polyline(50)
    smoothed_wp = tw.polyline.smooth_upsample_polyline(_polyline_wp(pts_np, device), step)
    assert np.allclose(
        smoothed_wp.numpy(), _smooth_upsample_np(pts_np, step, closed=False), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("step", [0.3, 0.75])
def test_smooth_upsample_closed_matches_reference(device: str, step: float) -> None:
    pts_np = _closed_from(_random_open_polyline(52))
    smoothed_wp = tw.polyline.smooth_upsample_closed_polyline(_polyline_wp(pts_np, device), step)
    assert np.allclose(
        smoothed_wp.numpy(), _smooth_upsample_np(pts_np, step, closed=True), rtol=1e-4, atol=1e-4
    )


def test_smooth_upsample_same_point_count_as_linear(device: str) -> None:
    pts_np = _random_open_polyline(53)
    step = 0.4
    linear = tw.polyline.upsample_polyline(_polyline_wp(pts_np, device), step).numpy()
    smoothed = tw.polyline.smooth_upsample_polyline(_polyline_wp(pts_np, device), step).numpy()
    assert smoothed.shape == linear.shape


def test_smooth_upsample_straight_line_reduces_to_linear(device: str) -> None:
    # Collinear vertices have zero curvature everywhere, so the arc collapses onto the chord and
    # the result must coincide with plain linear upsampling.
    pts_np = np.stack([np.linspace(0.0, 3.0, 7), np.zeros(7), np.zeros(7)], axis=1)
    step = 0.25
    linear = tw.polyline.upsample_polyline(_polyline_wp(pts_np, device), step).numpy()
    smoothed = tw.polyline.smooth_upsample_polyline(_polyline_wp(pts_np, device), step).numpy()
    assert np.allclose(smoothed, linear, rtol=1e-5, atol=1e-5)


def test_smooth_upsample_preserves_original_vertices(device: str) -> None:
    # Each segment's first sample (t == 0) is its start vertex, so all but the final vertex survive.
    pts_np = _random_open_polyline(54, n=6)
    smoothed = tw.polyline.smooth_upsample_polyline(_polyline_wp(pts_np, device), 0.5).numpy()
    for vertex in pts_np[:-1]:
        assert np.any(np.all(np.isclose(smoothed, vertex, rtol=1e-4, atol=1e-4), axis=1))


def test_smooth_upsample_closed_recovers_circle(device: str) -> None:
    # A regular polygon inscribed in a circle: curvature fitting reconstructs the circumscribed
    # arcs, so every inserted point lies on the circle (a linear upsample would cut inside it).
    radius = 2.0
    polygon_np = _planar_circle(8, radius)
    smoothed = tw.polyline.smooth_upsample_closed_polyline(
        _polyline_wp(polygon_np, device), 0.35
    ).numpy()
    assert np.allclose(np.linalg.norm(smoothed, axis=-1), radius, rtol=1e-3, atol=1e-3)
    # The added points genuinely bulge outward relative to the straight-chord upsample.
    linear = tw.polyline.upsample_closed_polyline(_polyline_wp(polygon_np, device), 0.35).numpy()
    assert np.linalg.norm(linear, axis=-1).min() < radius - 1e-2


def test_smooth_upsample_short_polyline_unchanged(device: str) -> None:
    single = _polyline_wp(np.array([[1.0, 2.0, 3.0]]), device)
    assert tw.polyline.smooth_upsample_polyline(single, 0.5).shape[0] == 1


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


# --- triangulate (ear clipping) ---


def _convex_ngon(n: int, radius: float = 1.5) -> np.ndarray:
    """CCW regular polygon in the xy-plane."""
    angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)], axis=1)


def _l_shape() -> np.ndarray:
    """Return a non-convex (one reflex corner) simple polygon in the xy-plane, CCW."""
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)


def _star(points: int = 5, outer: float = 2.0, inner: float = 0.8) -> np.ndarray:
    """Return a star polygon (alternating reflex corners), CCW, in the xy-plane."""
    angle = np.linspace(0.0, 2 * np.pi, 2 * points, endpoint=False)
    radius = np.where(np.arange(2 * points) % 2 == 0, outer, inner)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), np.zeros(2 * points)], axis=1)


def _rotate_into_3d(pts_xy: np.ndarray, seed: int) -> np.ndarray:
    """Apply a random rotation + translation so the polygon lies in a tilted 3D plane."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return pts_xy @ q.T + rng.standard_normal(3)


def _polygon_area(pts: np.ndarray) -> float:
    """Area of a planar polygon in 3D via the summed cross products (Newell)."""
    rolled = np.roll(pts, -1, axis=0)
    return float(np.linalg.norm(np.cross(pts, rolled).sum(axis=0)) / 2.0)


def _triangle_areas(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=-1) / 2.0


def _assert_valid_triangulation(pts: np.ndarray, faces: np.ndarray) -> None:
    n = pts.shape[0]
    assert faces.shape == (n - 2, 3)
    assert faces.min() >= 0
    assert faces.max() < n
    assert set(faces.ravel().tolist()) == set(range(n))  # no orphan vertices
    tri_areas = _triangle_areas(pts, faces)
    assert np.all(tri_areas > 1e-6)  # no degenerate triangles
    assert np.allclose(tri_areas.sum(), _polygon_area(pts), rtol=1e-5, atol=1e-5)


def test_triangulate_convex_is_fan(device: str) -> None:
    pts_np = _convex_ngon(8)
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    n = pts_np.shape[0]
    expected = np.stack([np.zeros(n - 2), np.arange(1, n - 1), np.arange(2, n)], axis=1)
    assert np.array_equal(faces_wp.numpy(), expected.astype(np.int32))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_l_shape(device: str) -> None:
    pts_np = _l_shape()
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_star(device: str) -> None:
    pts_np = _star(6)
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_tilted_plane(device: str) -> None:
    pts_np = _rotate_into_3d(_star(6), seed=7)
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_clockwise_orientation(device: str) -> None:
    pts_np = _l_shape()[::-1].copy()  # reverse to clockwise
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    _assert_valid_triangulation(pts_np, faces_wp.numpy())


def test_triangulate_closed_input_matches_open(device: str) -> None:
    pts_np = _star(5)
    closed_np = _closed_from(pts_np)
    faces_open = tw.polyline.triangulate(_polyline_wp(pts_np, device)).numpy()
    faces_closed = tw.polyline.triangulate(_polyline_wp(closed_np, device)).numpy()
    assert np.array_equal(faces_open, faces_closed)


def test_triangulate_single_triangle(device: str) -> None:
    pts_np = _convex_ngon(3)
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    assert faces_wp.shape == (1, 3)
    assert set(faces_wp.numpy().ravel().tolist()) == {0, 1, 2}


def test_triangulate_too_few_points(device: str) -> None:
    pts_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    faces_wp = tw.polyline.triangulate(_polyline_wp(pts_np, device))
    assert faces_wp.shape == (0, 3)


# --- simplify (Ramer-Douglas-Peucker, NumPy reference) ---


def _simplify_np(pts_np: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    # Mirrors reference/libigl/include/igl/ramer_douglas_peucker.cpp; returns (S, J).
    n = len(pts_np)
    keep = np.ones(n, dtype=bool)
    stol = tol * tol

    def rec(ixs: int, ixe: int) -> None:
        sdmax, ixc = 0.0, -1
        if ixe - ixs > 1:
            s, d = pts_np[ixs], pts_np[ixe]
            dms = d - s
            sdes = float(dms @ dms)
            for k in range(ixs + 1, ixe):
                p = pts_np[k]
                if sdes <= 1e-7:
                    sd = float((p - s) @ (p - s))
                else:
                    t = -(dms @ (s - p)) / sdes
                    proj = (1.0 - t) * s + t * d
                    sd = float((p - proj) @ (p - proj))
                if sd > sdmax:
                    sdmax, ixc = sd, k
        if sdmax <= stol:
            keep[ixs + 1 : ixe] = False
        else:
            rec(ixs, ixc)
            rec(ixc, ixe)

    if n >= 2:
        rec(0, n - 1)
    indices = np.flatnonzero(keep)
    return pts_np[indices], indices


@pytest.mark.parametrize("tol", [0.1, 0.5])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_simplify_matches_reference(device: str, tol: float, seed: int) -> None:
    pts_np = _random_open_polyline(seed, n=40)
    simplified_wp, indices_wp = tw.polyline.simplify(_polyline_wp(pts_np, device), tol)
    simplified_np, indices_np = _simplify_np(pts_np, tol)
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))
    assert np.allclose(
        simplified_wp.numpy(), simplified_np.astype(np.float32), rtol=1e-4, atol=1e-4
    )


def test_simplify_collinear_collapses_to_endpoints(device: str) -> None:
    pts_np = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)], axis=1)
    simplified_wp, indices_wp = tw.polyline.simplify(_polyline_wp(pts_np, device), 1e-3)
    assert np.array_equal(indices_wp.numpy(), np.array([0, 10], dtype=np.int32))
    assert np.allclose(simplified_wp.numpy(), pts_np[[0, 10]].astype(np.float32))


def test_simplify_preserves_a_sharp_corner(device: str) -> None:
    # A tent: the apex deviates far from the base chord and must be kept.
    pts_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    _, indices_wp = tw.polyline.simplify(_polyline_wp(pts_np, device), 0.1)
    assert np.array_equal(indices_wp.numpy(), np.array([0, 1, 2], dtype=np.int32))


def test_simplify_empty(device: str) -> None:
    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    simplified_wp, indices_wp = tw.polyline.simplify(empty_wp, 0.5)
    assert simplified_wp.shape == (0,)
    assert indices_wp.shape == (0,)


def test_simplify_single_point(device: str) -> None:
    pts_np = np.array([[0.3, -0.4, 1.2]])
    simplified_wp, indices_wp = tw.polyline.simplify(_polyline_wp(pts_np, device), 0.5)
    assert np.array_equal(indices_wp.numpy(), np.array([0], dtype=np.int32))
    assert np.allclose(simplified_wp.numpy(), pts_np.astype(np.float32))


def test_simplify_two_points_kept(device: str) -> None:
    pts_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    _, indices_wp = tw.polyline.simplify(_polyline_wp(pts_np, device), 0.5)
    assert np.array_equal(indices_wp.numpy(), np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize("tol", [0.1, 0.5])
def test_simplify_closed_matches_reference(device: str, tol: float) -> None:
    pts_np = _random_open_polyline(3, n=30)
    closed_np = _closed_from(pts_np)
    simplified_wp, indices_wp = tw.polyline.simplify_closed(_polyline_wp(pts_np, device), tol)
    simplified_np, indices_np = _simplify_np(closed_np, tol)
    assert np.array_equal(indices_wp.numpy(), indices_np.astype(np.int32))
    assert np.allclose(
        simplified_wp.numpy(), simplified_np.astype(np.float32), rtol=1e-4, atol=1e-4
    )
