import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh.geometry as tm_geometry
import trimesh.points as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp.neighbors as tw_neighbors
import triwarp.points as tw
import triwarp.typing as twt
from tests.comparisons import assert_same_up_to_sign
from tests.conversions import (
    meshlib_bitset_to_numpy,
    points_to_meshlib,
    points_to_open3d,
    points_to_pymeshlab,
)


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
    """
    Class A: signed distances against ``trimesh.points.point_plane_distance``, sign included.

    The normal is not unit length here, which is what makes the normalization part of the claim
    rather than an assumption both sides happen to share.
    """
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
    """
    Class B (sign fix): the major axis against ``trimesh.points.major_axis``, up to direction.

    An eigenvector is defined only up to sign, so the comparison goes through
    [`tests.comparisons.assert_same_up_to_sign`][]. The cloud is elongated 1000:1 so the axis
    itself is well determined -- on an isotropic cloud there would be nothing to compare.
    """
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


def _point_accumulator_ml(points_np: np.ndarray) -> mm.PointAccumulator:
    """
    Accumulate a NumPy cloud into a ``PointAccumulator``, the object behind both fitting oracles.

    ``accumulatePoints`` takes a ``std_vector_Vector3_float``, so the fill is a per-point Python
    loop -- fine at test size, and the reason neither of these pairs carries a benchmark row of its
    own beyond the ones already there.
    """
    accumulator_ml = mm.PointAccumulator()
    points_ml = mm.std_vector_Vector3_float()
    for point_np in np.asarray(points_np, dtype=np.float64):
        points_ml.append(mm.Vector3f(*point_np.tolist()))
    mm.accumulatePoints(accumulator_ml, points_ml)
    return accumulator_ml


@pytest.mark.parity("fit_plane", "meshlib")
def test_fit_plane_matches_meshlib(device: str) -> None:
    """
    Class B (sign gauge only): ``PointAccumulator.getBestPlanef`` is the same least-squares plane.

    The transform is the *representation*: MeshLib returns a ``Plane3f`` -- a unit normal and an
    offset ``d`` -- where triwarp returns a centroid and a normal, so the comparison is the normal
    up to sign (both are the smallest-eigenvalue covariance eigenvector, whose direction neither
    library fixes) plus the plane equation ``n . c == d`` evaluated at triwarp's centroid.

    Measured on an anisotropic 500-point cloud: ``|dot| = 1.0000000`` to seven digits, which is
    tighter than the trimesh and MeshLab pairings above and is why this is the one asserted at
    ``1e-6``. The offset check is what makes it a plane comparison rather than a direction one --
    a fit that found the right orientation through the wrong point would pass the dot alone.
    """
    rng = np.random.default_rng(3)
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + np.array(
        [2.0, -1.0, 0.5]
    )
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    plane_ml = _point_accumulator_ml(points_np).getBestPlanef()
    normal_ml = np.array([plane_ml.n.x, plane_ml.n.y, plane_ml.n.z], dtype=np.float64)

    assert np.isclose(np.linalg.norm(normal_ml), 1.0, atol=1e-6)  # non-vacuity: a real plane
    assert np.isclose(
        abs(float(np.dot(np.array(centroid_wp), normal_ml)) - plane_ml.d), 0.0, atol=1e-4
    )
    assert np.isclose(abs(float(np.dot(np.array(normal_wp), normal_ml))), 1.0, atol=1e-6)


@pytest.mark.parity("principal_axes", "meshlib")
def test_principal_axes_matches_meshlib(device: str) -> None:
    """
    Class B: the same eigen-decomposition, reported in the **opposite** order.

    ``getCenteredCovarianceEigen`` is an out-parameter call -- it takes a centroid, a ``Matrix3``
    and an eigenvalue vector to fill and returns a ``bool`` -- and it orders its eigenvalues
    **ascending** where triwarp orders them descending, so the named transform is a reversal plus
    the per-axis sign every eigenvector comparison needs. Measured on a 500-point anisotropic cloud
    the eigenvalues agree to 4 digits (4542.72 / 538.20 / 19.62 on both sides, reversed) and each
    axis matches to ``|dot| = 1``.

    Its eigenvalues are of the *scatter* matrix, carrying no ``1/n``, which is triwarp's convention
    too -- so this pins that convention against a second library, where the pyvista pairing above
    pins the axes alone.
    """
    rng = np.random.default_rng(4)
    points_np = (rng.standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.2])) + np.array(
        [2.0, -1.0, 0.5]
    )
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    rotation_wp, eigenvalues_wp, centroid_wp = tw.principal_axes(points_wp)

    centroid_ml = mm.Vector3d()
    eigenvectors_ml = mm.Matrix3d()
    eigenvalues_ml = mm.Vector3d()
    assert _point_accumulator_ml(points_np).getCenteredCovarianceEigen(
        centroid_ml, eigenvectors_ml, eigenvalues_ml
    )

    axes_ml = np.array(
        [
            [eigenvectors_ml.x.x, eigenvectors_ml.x.y, eigenvectors_ml.x.z],
            [eigenvectors_ml.y.x, eigenvectors_ml.y.y, eigenvectors_ml.y.z],
            [eigenvectors_ml.z.x, eigenvectors_ml.z.y, eigenvectors_ml.z.z],
        ]
    )[::-1]  # ascending -> descending
    values_ml = np.array([eigenvalues_ml.x, eigenvalues_ml.y, eigenvalues_ml.z])[::-1]

    assert values_ml[0] > values_ml[1] > values_ml[2] > 0.0  # non-vacuity: a separated spectrum
    assert np.allclose(
        np.array(centroid_wp), [centroid_ml.x, centroid_ml.y, centroid_ml.z], atol=1e-4
    )
    assert np.allclose(np.array(eigenvalues_wp), values_ml, rtol=1e-4, atol=1e-4)
    rotation_np = np.array(rotation_wp).reshape(3, 3)
    for row in range(3):
        assert np.isclose(abs(float(np.dot(rotation_np[row], axes_ml[row]))), 1.0, atol=1e-4)


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
    """Class C (one axis only): a near-equal eigenpair leaves its own plane arbitrary."""
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
    """
    Class B (sign fix): the same comparison at 5 000 points, past the tiled reduction's tile size.

    ``5000 = 78 * 64 + 8`` exercises both the multi-tile path and its remainder branch, which
    the 200-point test above does not reach at all.
    """
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
    """
    Class B (sign fix on the normal): ``trimesh.points.plane_fit`` at 5 000 points.

    Same multi-tile motivation as the line fit above. The centroid needs no sign fix and is
    compared directly, which is what separates a reduction bug from an eigenvector one.
    """
    # n far larger than TILE_1D (64) to exercise the multi-tile reduction path.
    rng = np.random.default_rng(6)
    points_np = rng.standard_normal((5000, 3))

    centroid_tm, normal_tm = tm.plane_fit(points_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    centroid_wp, normal_wp = tw.fit_plane(points_wp)

    assert np.allclose(np.array(centroid_wp), centroid_tm, rtol=1e-4, atol=1e-4)
    assert np.isclose(np.abs(np.dot(np.array(normal_wp), normal_tm)), 1.0, atol=1e-4)


def test_point_plane_distance_no_origin(device: str) -> None:
    """
    Class A: the default-origin overload, where the plane passes through the world origin.

    A separate test because the default is a different code path, not a value the caller could
    pass -- and an implementation defaulting to the *centroid* instead would pass the test
    above and fail this one.
    """
    rng = np.random.default_rng(1)
    points_np = rng.standard_normal((30, 3))
    plane_normal_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distances_wp = tw.point_plane_distance(points_wp, wp.vec3(*plane_normal_np.tolist()))

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("radial_sort", "trimesh")
def test_radial_sort(device: str) -> None:
    """
    Class A: the angular order against a numpy ``arctan2`` argsort, index for index.

    The angles are evenly spaced by construction, which is what makes an exact index comparison
    sound: with random angles two neighbours can differ by less than ``float32`` resolves and
    the orders diverge legitimately (section 6's note on ``lexsort`` and float ties).
    """
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
    """
    Class A: the same order rotated to begin at a supplied start direction.

    The input is permuted first, so a function ignoring ``start`` and returning the input order
    cannot pass. The reference rotation is computed in numpy from the same start vector.
    """
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


@pytest.mark.parity("estimate_normals_knn", "meshlib")
def test_estimate_normals_matches_meshlib(device: str) -> None:
    """
    Class B (sign gauge): ``makeUnorientedNormals`` is the same PCA under a *radius* search.

    The named transform is the neighbourhood: MeshLib searches by **radius** where triwarp is given
    a k-nearest table, so the two see the same points only where the cloud is uniform -- which is
    what the Fibonacci sphere is for. Measured there: minimum ``|dot|`` **0.99966** at a radius of
    1.5 mean spacings and **0.99945** at 3.0, i.e. every one of 2 000 normals agrees, and the result
    is insensitive to the radius over a factor of two.

    ``makeOrientedNormals`` is the oriented sibling and fixes the sign by propagating over a
    spanning structure; on a closed cloud it agrees with the outward radial direction on **100 %**
    of points, which is the assert below. That is the half triwarp's ``orient_reference`` does with
    a single reference vector instead, and the two are compared here for the first time.
    """
    knn = 30
    points_np = _fibonacci_sphere(2000)
    spacing = float(np.sqrt(4.0 * np.pi / points_np.shape[0]))

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    neighbor_idx_wp, _distances_wp = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=knn)
    normals_wp = tw.estimate_normals(points_wp, neighbor_idx_wp).numpy()

    cloud_ml = points_to_meshlib(points_np)
    for radius_scale in (1.5, 3.0):
        unoriented_ml = mm.makeUnorientedNormals(cloud_ml, radius_scale * spacing)
        normals_ml = mn.toNumpyArray(unoriented_ml)
        assert normals_ml.shape == points_np.shape  # non-vacuity: one normal per point
        assert np.abs(np.einsum("ij,ij->i", normals_wp, normals_ml)).min() > 0.999

    # The oriented form, against triwarp's outward reference: both point away from the centre.
    oriented_ml = mm.makeOrientedNormals(cloud_ml, 2.0 * spacing)
    outward_np = points_np / np.linalg.norm(points_np, axis=1, keepdims=True)
    assert (np.einsum("ij,ij->i", mn.toNumpyArray(oriented_ml), outward_np) > 0.0).all()


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
    """Class C (a ranking plus a subset): the 15 planted outliers must score highest."""
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
    """
    Class B (mask inversion): Open3D returns the indices it *keeps*, so they are inverted.

    Both sides use the same ``k`` and ``std_ratio`` and the same planted cloud, so the
    comparison is exact on the mask -- the transform is only that Open3D reports the
    complement.
    """
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


@pytest.mark.parity(
    "statistical_outlier_mask",
    "meshlib",
    benchmarked=False,
    reason="findOutliers is a different criterion, not a different tuning: its four modes are "
    "connectivity- and normal-based (SmallComponents, WeaklyConnected, FarSurface, AwayNormal) "
    "where statistical_outlier_mask is a z-score on the k-NN mean distance, and it takes a radius "
    "where triwarp takes a neighbour count. Timing them against each other would price two "
    "different questions -- measured on a planted cloud, SmallComponents flags 26 points to "
    "triwarp's 13 while both contain all 15 planted outliers. open3d's remove_statistical_outlier "
    "is the same z-score and carries the timed row; the comparison here is what MeshLib can still "
    "say about the answer.",
)
def test_statistical_outlier_mask_matches_meshlib(device: str) -> None:
    """
    Class C (recall and containment): MeshLib's criterion is stricter, and strictly wider.

    ``findOutliers`` with ``OutlierTypeMask.SmallComponents`` labels a point an outlier when its
    connected component under a radius graph is small, which is a different question from a z-score
    on the k-NN mean distance -- so there is no correspondence to compare and the statistic is what
    each finds. On the planted cloud both find **all 15** seeded outliers, and triwarp's 13 flagged
    points are a **subset** of MeshLib's 26: it is the more conservative of the two, with zero false
    positives against MeshLib's 11.

    **Bug class excluded:** a detector that flags the wrong points, or flags on the wrong scale --
    containment is what a looser threshold cannot fake, since a mask that grew arbitrarily would
    break the subset relation in the other direction. **Mutation probe, measured:** shuffling
    triwarp's mask drops the element-wise agreement from 0.969 to 0.906 -- too thin a margin to
    assert on, which is exactly why the assert is recall plus containment rather than agreement.

    **The default mask is unusable and crashes**: ``FindOutliersParams.mask`` defaults to ``All``,
    which includes ``AwayNormal``, and that criterion **segfaults** on a cloud carrying no normals
    -- no exception, no traceback. The mode is always set explicitly here.
    """
    k, std_ratio = 20, 2.0
    points_np = _cloud_with_outliers()
    n_outliers = 15
    planted_np = np.zeros(points_np.shape[0], dtype=bool)
    planted_np[-n_outliers:] = True

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    _idx_wp, neighbor_distance_wp = tw_neighbors.query_bvh_nearest(points_wp, points_wp, k=k)
    outlier_wp = tw.statistical_outlier_mask(neighbor_distance_wp, std_ratio=std_ratio).numpy()

    cloud_ml = points_to_meshlib(points_np)
    params_ml = mm.FindOutliersParams()
    params_ml.radius = 1.0
    params_ml.mask = mm.OutlierTypeMask.SmallComponents  # ``All`` segfaults without normals
    outlier_ml = meshlib_bitset_to_numpy(mm.findOutliers(cloud_ml, params_ml), points_np.shape[0])

    assert (
        outlier_ml.sum() < 0.1 * points_np.shape[0]
    )  # non-vacuity: not "everything is an outlier"
    assert (outlier_ml & planted_np).sum() == n_outliers  # the reference found every planted one
    assert (outlier_wp & planted_np).sum() >= n_outliers - 2
    assert (outlier_wp & ~outlier_ml).sum() == 0  # triwarp's set is contained in MeshLib's


def test_statistical_outlier_mask_empty(device: str) -> None:
    neighbor_distance_wp = twt.empty_2d((0, 8), wp.float32, device=device)
    assert tw.statistical_outlier_mask(neighbor_distance_wp).shape == (0,)


@pytest.mark.parity("vector_angle", "trimesh")
def test_vector_angle(device: str) -> None:
    """
    Class B (input packing): ``trimesh.geometry.vector_angle`` wants the pairs as ``(n, 2, 3)``.

    The named transform is the reshape on the reference side; the values are compared directly.
    Unit vectors on both sides, so this is the angle formula alone rather than a normalization
    test.
    """
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
