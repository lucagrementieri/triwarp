import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh.geometry as tm_geometry
import trimesh.points as tm
import warp as wp

import triwarp.neighbors as tw_neighbors
import triwarp.points as tw
import triwarp.typing as twt
from tests.comparisons import assert_same_up_to_sign
from tests.conversions import points_to_open3d, points_to_pymeshlab


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


@pytest.mark.parity("principal_axes", "pyvista")
def test_principal_axes(device: str) -> None:
    """Class A up to the free sign of the first two axes: rows, eigenvalues and centroid."""
    rng = np.random.default_rng(7)
    # A well-separated spectrum, so every axis is individually determined. Offset from the origin
    # too, which is what separates a centred fit from fit_line's uncentred one.
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + 5.0
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)

    rotation_wp, eigenvalues_wp, centroid_wp = tw.principal_axes(points_wp)
    rotation_np = np.array(rotation_wp).reshape(3, 3)

    axes_pv = pv.principal_axes(points_np)
    for row in range(3):
        assert np.isclose(abs(float(np.dot(rotation_np[row], axes_pv[row]))), 1.0, atol=1e-5)

    # A proper, orthonormal rotation -- the determinant is what pins the third row's sign.
    assert np.allclose(rotation_np @ rotation_np.T, np.eye(3), atol=1e-5)
    assert np.isclose(float(np.linalg.det(rotation_np)), 1.0, atol=1e-5)

    # Eigenvalues are of the scatter matrix, so they carry no 1/n.
    scatter_np = np.cov(points_np.T) * (points_np.shape[0] - 1)
    assert np.allclose(
        np.array(eigenvalues_wp), np.linalg.eigvalsh(scatter_np)[::-1], rtol=1e-5, atol=1e-5
    )
    assert np.allclose(np.array(centroid_wp), points_np.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_principal_axes_is_not_fit_line(device: str) -> None:
    """
    Pin the distinction the two functions exist to keep apart.

    ``fit_line`` is ``trimesh.points.major_axis``: a *singular-value-weighted sum of all three*
    right singular vectors of the uncentered matrix. ``principal_axes`` is the leading eigenvector
    of the centred covariance. On an ordinary cloud these are different directions, which is why the
    second was added rather than the first being changed.

    Note what the weighted sum costs: because it mixes all three vectors, its value depends on each
    one's *sign*, and no SVD fixes those. So ``fit_line`` reproduces its own oracle only where one
    singular value dominates -- measured ``|dot|`` 1.000 on a 1000:1 needle against **0.992** on the
    moderate cloud below. That is a property of the quantity, not a defect in either port.
    """
    rng = np.random.default_rng(3)
    moderate_np = rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])
    moderate_wp = wp.array(moderate_np.astype(np.float32), dtype=wp.vec3, device=device)

    leading_np = np.linalg.eigh(np.cov(moderate_np.T))[1][:, -1]
    first_axis_np = np.array(tw.principal_axes(moderate_wp)[0]).reshape(3, 3)[0]

    # principal_axes is the leading eigenvector ...
    assert np.isclose(abs(float(np.dot(first_axis_np, leading_np))), 1.0, atol=1e-4)
    # ... and the weighted major axis is a measurably different direction, on both sides.
    assert abs(float(np.dot(tm.major_axis(moderate_np), leading_np))) < 0.99
    assert abs(float(np.dot(np.array(tw.fit_line(moderate_wp)), leading_np))) < 0.99

    # On a needle every definition coincides, which is why the difference went unnoticed.
    needle_np = (rng.uniform(-10.0, 10.0, 200)[:, None] * np.array([0.3, 0.5, 0.8])) + (
        0.01 * rng.standard_normal((200, 3))
    )
    needle_wp = wp.array(needle_np.astype(np.float32), dtype=wp.vec3, device=device)
    needle_first_np = np.array(tw.principal_axes(needle_wp)[0]).reshape(3, 3)[0]
    assert np.isclose(abs(float(np.dot(needle_first_np, tm.major_axis(needle_np)))), 1.0, atol=1e-3)
    assert np.isclose(
        abs(float(np.dot(np.array(tw.fit_line(needle_wp)), tm.major_axis(needle_np)))),
        1.0,
        atol=1e-3,
    )


def test_principal_axes_degenerate_spectrum(device: str) -> None:
    """A near-equal eigenpair leaves its plane arbitrary; only the separated axis is comparable."""
    rng = np.random.default_rng(5)
    # 1000:1 needle: axes 2 and 3 both sit in the noise, so their split is not determined.
    points_np = (rng.uniform(-10.0, 10.0, 200)[:, None] * np.array([0.3, 0.5, 0.8])) + (
        0.01 * rng.standard_normal((200, 3))
    )
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    rotation_np = np.array(tw.principal_axes(points_wp)[0]).reshape(3, 3)
    axes_pv = pv.principal_axes(points_np)

    assert np.isclose(abs(float(np.dot(rotation_np[0], axes_pv[0]))), 1.0, atol=1e-5)
    # The degenerate pair still spans the same plane, even though the axes within it differ.
    assert np.isclose(
        abs(float(np.dot(rotation_np[0], np.cross(axes_pv[1], axes_pv[2])))), 1.0, atol=1e-4
    )
    assert np.allclose(rotation_np @ rotation_np.T, np.eye(3), atol=1e-5)


def test_principal_axes_empty_and_single(device: str) -> None:
    """An empty cloud gives the identity frame; a single point gives a frame at that point."""
    empty_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    rotation_wp, eigenvalues_wp, centroid_wp = tw.principal_axes(empty_wp)
    assert np.array_equal(np.array(rotation_wp).reshape(3, 3), np.eye(3))
    assert np.array_equal(np.array(eigenvalues_wp), np.zeros(3))
    assert np.array_equal(np.array(centroid_wp), np.zeros(3))

    single_wp = wp.array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    rotation_wp, _eigenvalues_wp, centroid_wp = tw.principal_axes(single_wp)
    assert np.isclose(float(np.linalg.det(np.array(rotation_wp).reshape(3, 3))), 1.0, atol=1e-6)
    assert np.allclose(np.array(centroid_wp), np.array([1.0, 2.0, 3.0]))


@pytest.mark.parity("fit_plane", "trimesh", "pymeshlab")
def test_fit_plane(device: str) -> None:
    """
    Class B against both references, each needing one named transform.

    trimesh's ``plane_fit`` returns the centroid and normal directly. **MeshLab** returns a *dict*,
    so the transform is the ``"fitting_plane_normal"`` key -- the rotation matrix and average error
    it also builds are work triwarp does not do. Two of its quirks are load-bearing and match what
    the benchmark passes: it raises ``Cannot compute rotation: there is no selection`` unless
    something is selected, so ``set_selection_all`` is how it is told to fit *all* the points; and
    it takes a face-less MeshSet, which is what these bare points are. Both references leave the
    normal
    **sign** free (it is a covariance eigenvector), so all three are compared up to sign.
    """
    rng = np.random.default_rng(3)
    points_np = rng.standard_normal((80, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.set_selection_all()
    normal_pml = np.asarray(
        meshset_pml.compute_matrix_by_fitting_to_plane(targetplane="XY plane")[
            "fitting_plane_normal"
        ],
        dtype=np.float64,
    )
    normal_pml /= np.linalg.norm(normal_pml)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    # normal is sign-ambiguous: compare up to sign.
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_pml)), 1.0, atol=1e-4)


@pytest.mark.parity("fit_plane", "pyvista")
def test_fit_plane_normal_matches_pyvista(device: str) -> None:
    """
    Class B, and the transform is a **projection**: only the normal is comparable.

    ``pv.fit_plane_to_points(..., return_meta=True)`` returns ``(plane, centre, normal)`` and its
    ``centre`` is *not* the centroid -- VTK re-centres the returned plane on the middle of the
    fitted patch, measured ``[5.018, 4.734, 5.008]`` against the cloud's own ``[4.784, 4.949,
    4.998]`` on the fixture below. So comparing the origin would fail for a correct implementation;
    the normal is the shared quantity, up to sign, and it agrees to ``|dot| = 1.0000004``.

    Both the centre and the normal come back **float32** even though pyvista stores points in
    float64, which is why the tolerance here is ``1e-4`` rather than tighter.
    """
    rng = np.random.default_rng(7)
    # Nearly planar and off the origin: the normal is well determined, and the centre pyvista
    # returns is not the centroid.
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.05])) + 5.0

    _plane_pv, centre_pv, normal_pv = pv.fit_plane_to_points(points_np, return_meta=True)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.isclose(
        abs(float(np.dot(np.array(normal_wp), np.asarray(normal_pv)))), 1.0, atol=1e-4
    )
    # The origins are different quantities; asserting that keeps the projection above honest.
    assert np.allclose(np.array(centroid_wp), points_np.mean(axis=0), rtol=1e-4, atol=1e-4)
    assert not np.allclose(np.asarray(centre_pv), points_np.mean(axis=0), atol=1e-2)


@pytest.mark.parametrize(
    "normal",
    [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.95, 0.1, 0.2), (0.3, 0.9, 0.1), (0.0, -2.0, 0.0)],
)
def test_plane_basis_is_right_handed_and_orthonormal(normal: tuple[float, float, float]) -> None:
    """
    All four properties its docstring promises, over both branches of the axis choice.

    The implementation picks its seed axis by whether ``|n_x| > 0.9``, so the parametrisation spans
    both: two normals take the ``x`` seed and three the ``y`` seed. A non-unit normal is included
    because the signature says it need not be normalized. Measured residuals at most 3.7e-08.
    """
    u_wp, v_wp = tw.plane_basis(wp.vec3(*normal))

    u_np = np.array(list(u_wp))
    v_np = np.array(list(v_wp))
    unit_np = np.asarray(normal) / np.linalg.norm(normal)

    assert np.allclose(np.linalg.norm(u_np), 1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.linalg.norm(v_np), 1.0, rtol=1e-6, atol=1e-6)
    assert np.allclose([u_np @ v_np, u_np @ unit_np, v_np @ unit_np], 0.0, rtol=1e-6, atol=1e-6)
    # Right-handed: (u, v, n) in that order, so u x v is the normal rather than its negation.
    assert np.allclose(np.cross(u_np, v_np), unit_np, rtol=1e-6, atol=1e-6)


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


@pytest.mark.parity("estimate_normals_knn", "open3d", "pymeshlab")
def test_estimate_normals_matches_open3d(device: str) -> None:
    """
    Class B against both references: the same search-plus-PCA, with the normal's sign left free.

    Both fix the smallest-eigenvalue covariance eigenvector and neither fixes its direction, so the
    named transform on all three sides is ``|dot| == 1``. **MeshLab** additionally needs
    ``smoothiter=0`` -- its default runs a normal-smoothing pass afterwards, which triwarp does not
    do -- and a face-less MeshSet, since ``compute_normal_for_point_clouds`` is for datasets with no
    faces. Both parameters are the ones the benchmark passes.

    MeshLab agrees essentially exactly (worst measured ``|dot|`` 0.9999999), so it gets a hard
    bound; Open3D disagrees on a handful of points because the two break k-nearest *ties*
    differently, which is why its assert is a fraction rather than a per-point bound.
    """
    knn = 30
    points_np = _fibonacci_sphere(2000)

    # Open3D reference: PCA normal from the k-nearest neighbourhood (KNN includes self).
    pcd = points_to_open3d(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals_o3d = np.asarray(pcd.normals)

    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.compute_normal_for_point_clouds(k=knn, smoothiter=0)
    normals_pml = np.asarray(meshset_pml.current_mesh().vertex_normal_matrix())

    # triwarp: build the same k-neighbourhood (self + knn-1 = knn points total), then PCA.
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _ = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=knn)
    normals_wp = tw.estimate_normals(points_wp, neighbor_idx_wp)

    # Both estimators fix the smallest-eigenvalue covariance eigenvector but leave the sign
    # free, so compare up to sign. A few points may disagree on KNN ties; require the vast
    # majority to align.
    abs_dots = np.abs(np.einsum("ij,ij->i", normals_wp.numpy(), normals_o3d))
    assert np.mean(abs_dots > 0.99) > 0.98
    assert_same_up_to_sign(normals_wp.numpy(), normals_pml, atol=1e-5)


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
    neighbor_distance_wp = twt.empty_2d((0, 8), wp.float32, device=device)
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
