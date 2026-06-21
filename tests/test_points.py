import numpy as np
import trimesh.points as tm
import warp as wp

import triwarp.points as tw


def test_point_plane_distance(device: str) -> None:
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((50, 3))
    plane_normal_np = rng.standard_normal(3)
    plane_origin_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np, plane_origin_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distances_wp = tw.point_plane_distance(
        points_wp, wp.vec3(*plane_normal_np.tolist()), wp.vec3(*plane_origin_np.tolist())
    )

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)


def test_centroid(device: str) -> None:
    rng = np.random.default_rng(14)
    points_np = rng.standard_normal((128, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    centroid_wp = tw.centroid(points_wp)
    assert np.allclose(centroid_wp.numpy()[0], points_np.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_fit_line(device: str) -> None:
    rng = np.random.default_rng(2)
    # points strongly elongated along a known direction so the major axis
    # is well-defined and robust to the SVD sign convention.
    direction_np = rng.standard_normal(3)
    direction_np /= np.linalg.norm(direction_np)
    t_np = rng.uniform(-10.0, 10.0, size=200)
    points_np = t_np[:, None] * direction_np[None, :] + 0.01 * rng.standard_normal((200, 3))

    axis_tm = tm.major_axis(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    axis_wp = tw.fit_line(points_wp)

    # axis is direction-only: compare up to sign against trimesh and the
    # ground-truth direction.
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), axis_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), direction_np)), 1.0, atol=1e-3)


def test_fit_plane(device: str) -> None:
    rng = np.random.default_rng(3)
    points_np = rng.standard_normal((80, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    # normal is sign-ambiguous: compare up to sign.
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)


def test_fit_line_large(device: str) -> None:
    # n far larger than TILE_1D (64) to exercise the multi-tile reduction path
    # and the remainder branch (5000 = 78 * 64 + 8).
    rng = np.random.default_rng(5)
    direction_np = rng.standard_normal(3)
    direction_np /= np.linalg.norm(direction_np)
    t_np = rng.uniform(-10.0, 10.0, size=5000)
    points_np = t_np[:, None] * direction_np[None, :] + 0.01 * rng.standard_normal((5000, 3))

    axis_tm = tm.major_axis(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    axis_wp = tw.fit_line(points_wp)

    assert np.isclose(np.abs(np.dot(np.array(axis_wp), axis_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(axis_wp), direction_np)), 1.0, atol=1e-3)


def test_fit_plane_large(device: str) -> None:
    # n far larger than TILE_1D (64) to exercise the multi-tile reduction path.
    rng = np.random.default_rng(6)
    points_np = rng.standard_normal((5000, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)


def test_point_plane_distance_no_origin(device: str) -> None:
    rng = np.random.default_rng(1)
    points_np = rng.standard_normal((30, 3))
    plane_normal_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distances_wp = tw.point_plane_distance(points_wp, wp.vec3(*plane_normal_np.tolist()))

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)
