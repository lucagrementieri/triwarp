"""Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import faces_igl, trimesh_to_pymeshlab, trimesh_to_warp


@pytest.mark.parity("face_normals_and_areas", "trimesh")
def test_face_normals_and_areas(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    normal_wp, area_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(normal_wp.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(area_wp.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_normals_and_areas", "igl", "potpourri3d", "pymeshlab")
def test_face_normals_and_areas_against_the_partial_references(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The three references that each answer *half* of this function, so the benchmark can be read.

    ``face_normals_and_areas`` returns both quantities from one cross product, and the benchmark
    reads these three rows as a floor rather than a fair race because each computes less. That makes
    them no *less* valid as oracles, only partial. All three are class B -- exact transforms, full
    ``1e-5`` tolerance:

    - ``igl.doublearea`` is literally twice the area, so the transform is a factor of two;
    - ``potpourri3d.face_areas`` is the area directly, and is pure numpy rather than
      geometry-central, so it is the weakest of the three as independent evidence;
    - MeshLab's ``compute_normal_per_face`` turns out to check **both** halves, not one.
      ``face_normal_matrix()`` returns the *unnormalised* cross product: measured on this fixture
      its magnitude is ``2 * area`` to within 1e-15 across every face, and its direction matches
      trimesh's unit normals exactly. So dividing by the magnitude gives the normal and halving the
      magnitude gives the area, and this row is a full oracle rather than a partial one. Do not
      "fix" a failure here by normalising triwarp's side -- triwarp's normals are already unit, and
      the magnitude is the area check.

    Uses ``half_torus`` rather than ``icosahedron`` so the areas actually vary across faces: on a
    mesh of 20 congruent triangles a factor-of-two error in one reference and a wrong *constant* in
    triwarp are indistinguishable.
    """
    mesh_tm, mesh_wp = half_torus
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = faces_igl(mesh_tm)

    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    areas_igl = np.asarray(igl.doublearea(vertices_np, faces_np)) / 2.0
    assert np.allclose(areas_wp.numpy(), areas_igl, rtol=1e-5, atol=1e-5)

    areas_pp = pp3d.face_areas(vertices_np, mesh_tm.faces.astype(np.int32))
    assert np.allclose(areas_wp.numpy(), areas_pp, rtol=1e-5, atol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_normal_per_face()
    crosses_pml = meshset_pml.current_mesh().face_normal_matrix()
    magnitudes_pml = np.linalg.norm(crosses_pml, axis=1)
    assert np.allclose(areas_wp.numpy(), magnitudes_pml / 2.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(
        normals_wp.numpy(), crosses_pml / magnitudes_pml[:, None], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity("centroid", "pymeshlab")
def test_centroid_matches_pymeshlab_shell_barycenter(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    The area-weighted centroid against MeshLab's ``shell_barycenter``.

    ``get_geometric_measures`` answers six questions in one call -- which is why
    ``benchmarks/test_triangles.py`` reads its row as an *upper* bound on the centroid alone -- but
    "upper bound" is a statement about cost, not about the value. Indexing the dict is the whole
    transform (class B).

    The distinction that matters here is that the same call also returns ``barycenter``, the plain
    *vertex* mean, and the two differ on any mesh with uneven triangle areas. Asserting against
    ``shell_barycenter`` while checking that ``barycenter`` is measurably different is what makes
    this a real test of the area weighting rather than of a centre of mass in general: on
    ``hemisphere`` they sit about 0.02 apart, some 400x the 1e-5 tolerance.
    """
    mesh_tm, mesh_wp = hemisphere
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()

    centroid_wp = tw.triangles.centroid(mesh_wp.points, mesh_wp.indices)
    centroid_np = np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z])

    assert np.allclose(centroid_np, measures_pml["shell_barycenter"], rtol=1e-5, atol=1e-5)
    # The vertex mean is a different quantity; if it were not, the assert above would be weightless.
    assert not np.allclose(measures_pml["barycenter"], measures_pml["shell_barycenter"], atol=1e-3)


def test_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(angles_wp.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("metric", "filter_metric"),
    [
        ("radius_ratio", "inradius/circumradius"),
        ("area_max_side", "area/max side"),
        ("mean_ratio", "Mean ratio"),
        ("area", "Area"),
    ],
)
@pytest.mark.parity("face_quality", "pymeshlab")
def test_face_quality_against_pymeshlab(
    half_torus: tuple[tm.Trimesh, wp.Mesh], metric: str, filter_metric: str
):
    """The four VCG shape measures, against the filter they were ported from."""
    mesh_tm, mesh_wp = half_torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_aspect_ratio_per_face(metric=filter_metric)
    quality_pml = meshset_pml.current_mesh().face_scalar_array()

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric=metric)
    assert np.allclose(quality_wp.numpy(), quality_pml, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_angles", "trimesh", "igl")
def test_face_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on both references: ``(n_faces, 3)`` interior angles, element-wise, no transform.

    All three libraries index the angle by the corner it sits at -- column ``j`` is the angle at
    ``faces[f, j]`` -- so ``igl.internal_angles``, ``trimesh``'s ``face_angles`` property and
    ``triangles.face_angles`` are directly comparable. That is worth pinning rather than assuming:
    the natural alternative convention indexes an angle by the *edge* opposite it, which is a
    cyclic shift of the row and would still pass a per-face sum check.

    ``half_torus`` rather than ``icosahedron`` because every corner of an icosahedron's faces
    carries the same angle, which is exactly the fixture a shifted row would survive.
    """
    mesh_tm, mesh_wp = half_torus
    angles_igl = igl.internal_angles(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    )

    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(angles_wp.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)
    assert np.allclose(angles_wp.numpy(), angles_igl, rtol=1e-5, atol=1e-5)
    # The row is not merely a permutation: the angle in column j sits at vertex faces[f, j].
    assert np.allclose(angles_wp.numpy().sum(axis=1), np.pi, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_quality", "igl")
def test_face_quality_aspect_ratio_against_igl(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """``aspect_ratio`` is circumradius over twice the inradius, which igl gives as two arrays."""
    mesh_tm, mesh_wp = half_torus
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    aspect_igl = np.asarray(igl.circumradius(vertices_np, faces_np)[0]) / (
        2.0 * np.asarray(igl.inradius(vertices_np, faces_np))
    )

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="aspect_ratio")
    assert np.allclose(quality_wp.numpy(), aspect_igl, rtol=1e-4, atol=1e-5)


def test_face_quality_radius_ratio_equilateral(device: str):
    """An equilateral triangle is the optimum of all three normalized measures."""
    side = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=np.float32
    )
    vertices_wp = wp.array(side, dtype=wp.vec3, device=device)
    faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="radius_ratio").numpy(), 1.0
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="mean_ratio").numpy(), 1.0
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="area_max_side").numpy(),
        np.sqrt(3.0) / 2.0,
    )
    assert np.allclose(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio").numpy(), 1.0
    )


def test_face_quality_degenerate(sliver_patch: tuple[np.ndarray, np.ndarray, wp.array, wp.array]):
    """The sliver reads ``+inf`` under ``aspect_ratio`` and ~0 under the bounded measures."""
    _vertices_np, _faces_np, vertices_wp, faces_wp = sliver_patch
    aspect_wp = tw.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio")
    assert np.isinf(aspect_wp.numpy()[0])
    for metric in ("radius_ratio", "area_max_side", "mean_ratio"):
        quality_wp = tw.triangles.face_quality(vertices_wp, faces_wp, metric=metric)
        assert quality_wp.numpy()[0] < 1e-5


def test_face_quality_unknown_metric(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="unknown metric"):
        tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="skewness")  # type: ignore[arg-type]


def test_nondegenerate(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    nondegenerate_tm = tm.triangles.nondegenerate(mesh_tm.triangles)
    nondegenerate_wp = tw.triangles.nondegenerate(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(nondegenerate_wp.numpy().astype(bool), nondegenerate_tm)


def test_barycentric_to_points(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    barycentric_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    points_tm = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)

    barycentric_wp = wp.array(barycentric_np, dtype=wp.vec3, device=mesh_wp.points.device)
    points_wp = tw.triangles.barycentric_to_points(mesh_wp.points, mesh_wp.indices, barycentric_wp)
    assert np.allclose(points_wp.numpy(), points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["cramer", "cross"])
@pytest.mark.parity("points_to_barycentric", "trimesh", "igl")
def test_points_to_barycentric(hemisphere: tuple[tm.Trimesh, wp.Mesh], method: str):
    """
    Class A on both references, for both solver methods.

    ``igl.barycentric_coordinates(P, A, B, C)`` takes the three corner arrays rather than a mesh, so
    the only difference from a call convention standpoint is that the triangle soup is unpacked into
    three ``(n, 3)`` blocks -- the values are compared with no transform at all.

    The points are generated *from* barycentric weights, so each one lies exactly in its triangle's
    plane and the coordinates are the ones that produced it; a solver that got the plane projection
    wrong rather than the in-plane solve would still pass a "sums to one" check, which is why the
    comparison is against two independent solvers instead.
    """
    mesh_tm, mesh_wp = hemisphere

    barycentric_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    points_np = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)
    barycentric_tm = tm.triangles.points_to_barycentric(mesh_tm.triangles, points_np, method=method)
    barycentric_igl = igl.barycentric_coordinates(
        np.ascontiguousarray(points_np),
        np.ascontiguousarray(mesh_tm.triangles[:, 0]),
        np.ascontiguousarray(mesh_tm.triangles[:, 1]),
        np.ascontiguousarray(mesh_tm.triangles[:, 2]),
    )

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, points_wp, method=method
    )
    assert np.allclose(barycentric_wp.numpy(), barycentric_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(barycentric_wp.numpy(), barycentric_igl, rtol=1e-5, atol=1e-5)


def test_closest_point(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    points_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    closest_points_tm = tm.triangles.closest_point(mesh_tm.triangles, points_np)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    closest_points_wp = tw.triangles.closest_point(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(closest_points_wp.numpy(), closest_points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("centroid", "trimesh")
def test_centroid(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere

    centroid_tm = mesh_tm.centroid

    centroid_wp = tw.triangles.centroid(mesh_wp.points, mesh_wp.indices)
    centroid_wp = np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z])
    assert np.allclose(centroid_wp, centroid_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_centroid_matches_trimesh_on_a_skewed_mesh_on_both_devices(kernel_device: str):
    """
    Pin the area-weighted centroid sum on **both** devices, on a deliberately asymmetric mesh.

    Three things this guards. ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.15's CPU
    backend, so the block-wide ``wp.tile_sum`` this reduction used to perform accumulated one face
    per 64-face tile there. A *symmetric* mesh hides that completely -- the centroid of every 64th
    face of a sphere is still the sphere's centre -- which is why the mesh is stretched and sheared
    first. Measured: the sub-sampled sum was off by 1.1e-2 on this shape and by 4e-8 on the
    unmodified sphere. And the reduction now has two implementations (``centroid_tiled`` on CUDA,
    ``centroid_sliced`` on CPU), so the parametrization is what covers both of them -- running this
    on one device only would leave a whole kernel untested.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=3)
    mesh_tm.vertices[:, 0] *= 6.0
    mesh_tm.vertices[:, 2] += 0.4 * mesh_tm.vertices[:, 1] ** 3
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)

    centroid_wp = tw.triangles.centroid(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z]),
        mesh_tm.centroid,
        rtol=1e-4,
        atol=1e-4,
    )


def test_centroid_empty(device: str):
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.array([], dtype=wp.int32, device=device)
    centroid_wp = tw.triangles.centroid(vertices, faces)
    centroid_wp = np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z])
    assert np.isnan(centroid_wp).all()


def test_volume(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    volume_wp = tw.triangles.volume(mesh_wp.points, mesh_wp.indices)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_volume_inward_normals_negative(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    faces_np = mesh_tm.faces[:, ::-1].reshape(-1).astype(np.int32)
    faces_flipped_wp = wp.array(
        np.ascontiguousarray(faces_np), dtype=wp.int32, device=mesh_wp.device
    )
    volume_wp = tw.triangles.volume(mesh_wp.points, faces_flipped_wp)
    assert np.isclose(volume_wp, -mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_volume_empty(device: str):
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.array([], dtype=wp.int32, device=device)
    assert tw.triangles.volume(vertices, faces) == 0.0


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_centroids", "igl")
def test_face_centroids(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: one barycentre per face, element-wise against ``igl.barycenter``.

    The assert that matters beyond the comparison is the second one: the barycentre must lie *in*
    its own triangle, which the barycentric coordinates ``(1/3, 1/3, 1/3)`` state exactly. A
    function that returned the mesh centroid broadcast, or the first corner, would match neither.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    centroids_igl = igl.barycenter(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    )

    centroids_wp = tw.triangles.face_centroids(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(centroids_wp.numpy(), centroids_igl, rtol=1e-5, atol=1e-5)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, centroids_wp
    )
    assert np.allclose(barycentric_wp.numpy(), 1.0 / 3.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("moments", "igl", "trimesh")
def test_moments(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B on igl (its first moment is un-normalised), class A on trimesh.

    ``igl.moments`` returns ``(m0, m1, m2)`` where ``m1`` is the centre of mass **times the mass**,
    so the named transform is ``m1 / m0``. ``m2`` needs no transform: it is already referred to the
    centre of mass rather than the origin, which was verified against a *translated* mesh rather
    than assumed -- on a mesh centred at the origin the two references coincide and the check would
    be vacuous.

    trimesh answers all three too (``volume`` / ``center_mass`` / ``moment_inertia``), so this is a
    three-way comparison of the same integrals.

    The inertia tolerance is relative to the tensor's own scale: triwarp integrates ``float32``
    positions in ``float64``, so the deviation tracks the positions' precision (measured 3.5e-8
    relative on ``icosahedron``, and 3.6e-15 on an axis-aligned box whose coordinates are exact in
    ``float32``).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)

    volume_igl, first_moment_igl, inertia_igl = igl.moments(vertices_np, faces_igl(mesh_tm))
    volume_wp, center_wp, inertia_wp = tw.triangles.moments(mesh_wp.points, mesh_wp.indices)

    assert np.isclose(volume_wp, volume_igl, rtol=1e-5)
    assert np.isclose(volume_wp, mesh_tm.volume, rtol=1e-5)
    assert np.allclose(
        [center_wp.x, center_wp.y, center_wp.z],
        np.asarray(first_moment_igl) / volume_igl,
        rtol=1e-4,
        atol=1e-5,
    )
    assert np.allclose(
        [center_wp.x, center_wp.y, center_wp.z], mesh_tm.center_mass, rtol=1e-4, atol=1e-5
    )
    scale = float(np.abs(np.asarray(inertia_igl)).max())
    assert np.abs(inertia_wp - np.asarray(inertia_igl)).max() < 1e-5 * scale
    assert np.abs(inertia_wp - mesh_tm.moment_inertia).max() < 1e-5 * scale


def test_moments_center_of_mass_differs_from_the_surface_centroid(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
):
    """
    The two centres are different quantities, which is why both functions exist.

    ``centroid`` is the area-weighted centre of the *shell* and ``moments``' is the volume centre of
    the solid. On a closed uniform sphere they coincide; on anything asymmetric they do not, and
    asserting they differ is what keeps ``moments`` from being a synonym.
    """
    _mesh_tm, mesh_wp = half_torus
    surface_centroid = tw.triangles.centroid(mesh_wp.points, mesh_wp.indices)
    _volume, center_of_mass, _inertia = tw.triangles.moments(mesh_wp.points, mesh_wp.indices)
    assert not np.allclose(
        [surface_centroid.x, surface_centroid.y, surface_centroid.z],
        [center_of_mass.x, center_of_mass.y, center_of_mass.z],
        atol=1e-3,
    )


def test_moments_translation_shifts_only_the_center(device: str):
    """
    Translation moves the centre of mass and leaves the volume and inertia tensor alone.

    The parallel-axis shift is what makes that true, and getting it wrong is invisible on any mesh
    centred at the origin -- which is why this fixture is deliberately offset.
    """
    box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    offset_np = np.array([1.5, -2.0, 0.5])

    def moments_of(vertices_np: np.ndarray):
        vertices_wp = wp.array(
            np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
        )
        faces_wp = wp.array(
            np.ascontiguousarray(box_tm.faces.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        return tw.triangles.moments(vertices_wp, faces_wp)

    volume_a, center_a, inertia_a = moments_of(box_tm.vertices)
    volume_b, center_b, inertia_b = moments_of(box_tm.vertices + offset_np)

    assert np.isclose(volume_a, volume_b, rtol=1e-5)
    assert np.allclose(
        [center_b.x, center_b.y, center_b.z],
        np.array([center_a.x, center_a.y, center_a.z]) + offset_np,
        atol=1e-5,
    )
    assert np.abs(inertia_a - inertia_b).max() < 1e-4 * float(np.abs(inertia_a).max())


def test_moments_empty(device: str):
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    volume, center, inertia = tw.triangles.moments(vertices_wp, faces_wp)
    assert volume == 0.0
    assert np.isnan([center.x, center.y, center.z]).all()
    assert np.array_equal(inertia, np.zeros((3, 3)))
