import numpy as np
import pytest
import trimesh.points as tm
import warp as wp

import triwarp.points as tw
import triwarp.proximity as tw_proximity


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic near-uniform points on the unit sphere (unique pairwise distances)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


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


def test_radial_sort(device: str) -> None:
    rng = np.random.default_rng(7)
    n = 256
    # evenly spaced angles so the radial order is unambiguous and float32 cannot
    # flip the order of neighboring points relative to the float64 reference.
    theta_np = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius_np = rng.uniform(0.1, 2.0, n)
    points_np = np.column_stack(
        (np.cos(theta_np) * radius_np, np.sin(theta_np) * radius_np, np.zeros(n))
    )
    points_np = points_np[rng.permutation(n)]
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])

    ordered_tm = tm.radial_sort(points_np, origin=origin_np, normal=normal_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    ordered_wp = tw.radial_sort(
        points_wp, wp.vec3(*origin_np.tolist()), wp.vec3(*normal_np.tolist())
    )

    assert np.allclose(ordered_wp.numpy(), ordered_tm, rtol=1e-5, atol=1e-5)


def test_radial_sort_with_start(device: str) -> None:
    rng = np.random.default_rng(8)
    n = 256
    theta_np = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius_np = rng.uniform(0.1, 2.0, n)
    points_np = np.column_stack(
        (np.cos(theta_np) * radius_np, np.sin(theta_np) * radius_np, np.zeros(n))
    )
    points_np = points_np[rng.permutation(n)]
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])
    start_np = np.array([1.0, 0.0, 0.0])

    ordered_tm = tm.radial_sort(points_np, origin=origin_np, normal=normal_np, start=start_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    ordered_wp = tw.radial_sort(
        points_wp,
        wp.vec3(*origin_np.tolist()),
        wp.vec3(*normal_np.tolist()),
        start=wp.vec3(*start_np.tolist()),
    )

    assert np.allclose(ordered_wp.numpy(), ordered_tm, rtol=1e-5, atol=1e-5)


def test_radial_sort_parallel_start_raises(device: str) -> None:
    points_np = np.zeros((4, 3))
    origin_np = np.array([0.0, 0.0, 0.0])
    normal_np = np.array([0.0, 0.0, 1.0])
    # start parallel to normal is invalid.
    start_np = np.array([0.0, 0.0, 2.0])

    with pytest.raises(ValueError, match=r"must not.*parallel"):
        tm.radial_sort(points_np, origin=origin_np, normal=normal_np, start=start_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match=r"must not.*parallel"):
        tw.radial_sort(
            points_wp,
            wp.vec3(*origin_np.tolist()),
            wp.vec3(*normal_np.tolist()),
            start=wp.vec3(*start_np.tolist()),
        )


def test_estimate_normals_matches_open3d(device: str) -> None:
    o3d = pytest.importorskip("open3d")
    knn = 30
    points_np = _fibonacci_sphere(2000)

    # Open3D reference: PCA normal from the k-nearest neighbourhood (KNN includes self).
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals_o3d = np.asarray(pcd.normals)

    # triwarp: build the same k-neighbourhood (self + knn-1 = knn points total), then PCA.
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_proximity.query_bvh_nearest(points_wp, points_wp, k=knn)
    normals_wp = tw.estimate_normals(points_wp, neighbor_idx_wp)

    # Both estimators fix the smallest-eigenvalue covariance eigenvector but leave the sign
    # free, so compare up to sign. A few points may disagree on KNN ties; require the vast
    # majority to align.
    abs_dots = np.abs(np.einsum("ij,ij->i", normals_wp.numpy(), normals_o3d))
    assert np.mean(abs_dots > 0.99) > 0.98


def test_estimate_normals_orientation(device: str) -> None:
    # Small negative slack: the kernel enforces the sign in float32, so a float64
    # recomputation can dip just below zero at a zero-crossing.
    tol = 1e-5
    points_np = _fibonacci_sphere(1000)
    centroid_np = points_np.mean(axis=0)
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_proximity.query_bvh_nearest(points_wp, points_wp, k=20)

    # Default: outward from the cloud centroid (the reference vector the kernel uses).
    normals_default = tw.estimate_normals(points_wp, neighbor_idx_wp).numpy()
    assert np.all(np.einsum("ij,ij->i", normals_default, points_np - centroid_np) >= -tol)

    # Align with a fixed direction (Open3D orient_normals_to_align_with_direction).
    reference_np = np.array([0.0, 0.0, 1.0])
    normals_dir = tw.estimate_normals(
        points_wp, neighbor_idx_wp, orient_reference=wp.vec3(*reference_np.tolist())
    ).numpy()
    assert np.all(normals_dir @ reference_np >= -tol)

    # Toward a camera at the sphere centre (Open3D orient_normals_towards_camera_location):
    # every normal points inward, i.e. opposite the outward position vector.
    normals_cam = tw.estimate_normals(
        points_wp, neighbor_idx_wp, camera_location=wp.vec3(0.0, 0.0, 0.0)
    ).numpy()
    assert np.all(np.einsum("ij,ij->i", normals_cam, points_np) <= tol)


def test_estimate_normals_mutually_exclusive_orientation(device: str) -> None:
    points_wp = wp.array(_fibonacci_sphere(16).astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_proximity.query_bvh_nearest(points_wp, points_wp, k=8)
    with pytest.raises(ValueError, match=r"at most one"):
        tw.estimate_normals(
            points_wp,
            neighbor_idx_wp,
            orient_reference=wp.vec3(0.0, 0.0, 1.0),
            camera_location=wp.vec3(0.0, 0.0, 0.0),
        )
