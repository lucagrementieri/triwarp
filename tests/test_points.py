import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import trimesh.geometry as tm_geometry
import trimesh.points as tm
import warp as wp

import triwarp.neighbors as tw_neighbors
import triwarp.points as tw
import triwarp.typing as twt
from tests.conversions import points_to_open3d


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic near-uniform points on the unit sphere (unique pairwise distances)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


@pytest.mark.parity("point_plane_distance", "trimesh")
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


def test_gram_matrix(device: str) -> None:
    # 200 = 3 * 64 + 8 exercises the multi-tile reduction and remainder path.
    rng = np.random.default_rng(10)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    gram_np = points_np.T @ points_np
    assert np.allclose(tw.gram_matrix(points_wp).numpy()[0], gram_np, rtol=1e-4, atol=1e-4)


def test_gram_matrix_empty(device: str) -> None:
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert np.allclose(tw.gram_matrix(points_wp).numpy()[0], np.zeros((3, 3)))


@pytest.mark.parity("fit_line", "trimesh")
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


def test_centered_covariance(device: str) -> None:
    rng = np.random.default_rng(11)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    centered_np = points_np - points_np.mean(axis=0)
    scatter_np = centered_np.T @ centered_np
    cov_wp = tw.centered_covariance(points_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


def test_centered_covariance_precomputed_center(device: str) -> None:
    rng = np.random.default_rng(12)
    points_np = rng.standard_normal((150, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    mean_np = points_np.mean(axis=0)
    center_wp = wp.array(mean_np.reshape(1, 3).astype(np.float32), dtype=wp.vec3, device=device)
    centered_np = points_np - mean_np
    scatter_np = centered_np.T @ centered_np
    cov_wp = tw.centered_covariance(points_wp, center=center_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("fit_plane", "trimesh")
def test_fit_plane(device: str) -> None:
    rng = np.random.default_rng(3)
    points_np = rng.standard_normal((80, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    # normal is sign-ambiguous: compare up to sign.
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)


def test_covariance(device: str) -> None:
    rng = np.random.default_rng(13)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    cov_np = np.cov(points_np.T, ddof=1)
    assert np.allclose(tw.covariance(points_wp).numpy()[0], cov_np, rtol=1e-4, atol=1e-4)


def test_covariance_too_few_points_raises(device: str) -> None:
    points_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="ddof"):
        tw.covariance(points_wp)


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


@pytest.mark.parity("radial_sort", "trimesh")
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


@pytest.mark.parity("estimate_normals_knn", "open3d")
def test_estimate_normals_matches_open3d(device: str) -> None:
    knn = 30
    points_np = _fibonacci_sphere(2000)

    # Open3D reference: PCA normal from the k-nearest neighbourhood (KNN includes self).
    pcd = points_to_open3d(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals_o3d = np.asarray(pcd.normals)

    # triwarp: build the same k-neighbourhood (self + knn-1 = knn points total), then PCA.
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=knn)
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
    neighbor_idx_wp, _ = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=20)

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
    neighbor_idx_wp, _ = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=8)
    with pytest.raises(ValueError, match=r"at most one"):
        tw.estimate_normals(
            points_wp,
            neighbor_idx_wp,
            orient_reference=wp.vec3(0.0, 0.0, 1.0),
            camera_location=wp.vec3(0.0, 0.0, 0.0),
        )


def _cloud_with_outliers(seed: int = 3, n_inliers: int = 400, n_outliers: int = 15) -> np.ndarray:
    """Build a tight Gaussian blob plus far stragglers, which land last in the array."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(size=(n_inliers, 3)), rng.normal(scale=6.0, size=(n_outliers, 3))])


def _loop_reference(points_np: np.ndarray, k: int, scale: float = 3.0) -> np.ndarray:
    """LoOP scores (Kriegel et al.) via scipy, the float64 oracle for the Warp implementation."""
    from scipy.spatial import cKDTree
    from scipy.special import erf

    distance_np, idx_np = cKDTree(points_np).query(points_np, k=k)
    sigma_np = np.sqrt((distance_np**2).mean(axis=1))
    plof_np = sigma_np / sigma_np[idx_np].mean(axis=1) - 1.0
    normalizer = scale * np.sqrt((plof_np**2).mean())
    return np.maximum(0.0, erf(plof_np / (normalizer * np.sqrt(2.0))))


def test_outlier_probability_matches_scipy(device: str) -> None:
    k = 32
    points_np = _cloud_with_outliers()
    probability_np = _loop_reference(points_np, k)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(
        points_wp, points_wp, k=k
    )
    probability_wp = tw.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)
    assert np.allclose(probability_wp.numpy(), probability_np, rtol=1e-4, atol=1e-4)


@pytest.mark.parity("outlier_probability", "pymeshlab")
def test_outlier_probability_ranks_the_planted_outliers(device: str) -> None:
    """The 15 far points are the 15 highest-scoring, and pymeshlab's selection is a subset."""
    k = 32
    n_inliers, n_outliers = 400, 15
    points_np = _cloud_with_outliers(n_inliers=n_inliers, n_outliers=n_outliers)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(
        points_wp, points_wp, k=k
    )
    probability_wp = tw.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)
    ranked = np.argsort(-probability_wp.numpy())
    assert set(ranked[:n_outliers].tolist()) == set(range(n_inliers, n_inliers + n_outliers))

    # pymeshlab's ``compute_selection_point_cloud_outliers`` is the same LoOP score under a
    # ``propthreshold``; it differs from this port in how its k-d tree counts the query point, so
    # compare the *sets* rather than the scores. Everything it flags must be a planted outlier, and
    # every point this port flags at MeshLab's own default threshold must be flagged there too.
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(np.ascontiguousarray(points_np, dtype=np.float64)))
    meshset_pml.compute_selection_point_cloud_outliers(propthreshold=0.8, knearest=k)
    selected_pml = np.flatnonzero(meshset_pml.current_mesh().vertex_selection_array())
    assert set(selected_pml.tolist()) <= set(range(n_inliers, n_inliers + n_outliers))
    assert set(np.flatnonzero(probability_wp.numpy() > 0.8).tolist()) <= set(selected_pml.tolist())


def test_outlier_probability_is_scale_invariant(device: str) -> None:
    """``sigma`` scales with the cloud but ``plof`` is a ratio, so the score must not move."""
    points_np = _cloud_with_outliers()
    scores = []
    for factor in (1.0, 100.0):
        points_wp = wp.array((points_np * factor).astype(np.float32), dtype=wp.vec3, device=device)
        neighbor_idx_wp, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(
            points_wp, points_wp, k=32
        )
        scores.append(tw.outlier_probability(neighbor_idx_wp, neighbor_distance_wp).numpy())
    assert np.allclose(scores[0], scores[1], rtol=1e-4, atol=1e-4)


def test_outlier_probability_invalid_scale(device: str) -> None:
    points_wp = wp.array(_fibonacci_sphere(16).astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(
        points_wp, points_wp, k=4
    )
    with pytest.raises(ValueError, match="scale must be positive"):
        tw.outlier_probability(neighbor_idx_wp, neighbor_distance_wp, scale=0.0)


def test_outlier_probability_shape_mismatch(device: str) -> None:
    points_wp = wp.array(_fibonacci_sphere(16).astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=4)
    _, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=5)
    with pytest.raises(ValueError, match="same shape"):
        tw.outlier_probability(neighbor_idx_wp, neighbor_distance_wp)


@pytest.mark.parity("statistical_outlier_mask", "open3d")
def test_statistical_outlier_mask_matches_open3d(device: str) -> None:
    k, std_ratio = 20, 2.0
    points_np = _cloud_with_outliers()

    pcd = points_to_open3d(points_np)
    _kept, keep_indices = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
    outlier_o3d = np.ones(points_np.shape[0], dtype=bool)
    outlier_o3d[np.asarray(keep_indices)] = False

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    _idx, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=k)
    outlier_wp = tw.statistical_outlier_mask(neighbor_distance_wp, std_ratio=std_ratio)
    assert np.array_equal(outlier_wp.numpy().astype(bool), outlier_o3d)


def test_statistical_outlier_mask_empty(device: str) -> None:
    neighbor_distance_wp = twt.empty_float32_2d((0, 8), device=device)
    assert tw.statistical_outlier_mask(neighbor_distance_wp).shape == (0,)


@pytest.mark.parity("vector_angle", "trimesh")
def test_vector_angle(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 64
    vecs_a_np = rng.standard_normal((n, 3))
    vecs_a_np /= np.linalg.norm(vecs_a_np, axis=1, keepdims=True)
    vecs_b_np = rng.standard_normal((n, 3))
    vecs_b_np /= np.linalg.norm(vecs_b_np, axis=1, keepdims=True)

    pairs_np = np.stack([vecs_a_np, vecs_b_np], axis=1)
    angles_tm = tm_geometry.vector_angle(pairs_np)

    vecs_a_wp = wp.array(vecs_a_np.astype(np.float32), dtype=wp.vec3, device=device)
    vecs_b_wp = wp.array(vecs_b_np.astype(np.float32), dtype=wp.vec3, device=device)
    angles_wp = tw.vector_angle(vecs_a_wp, vecs_b_wp)
    assert np.allclose(angles_wp.numpy(), angles_tm, rtol=1e-5, atol=1e-5)


def test_vector_angle_empty(device: str) -> None:
    vecs_a_wp = wp.empty(0, dtype=wp.vec3, device=device)
    vecs_b_wp = wp.empty(0, dtype=wp.vec3, device=device)
    angles_wp = tw.vector_angle(vecs_a_wp, vecs_b_wp)
    assert angles_wp.shape == (0,)
