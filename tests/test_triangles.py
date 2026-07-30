"""Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import faces_igl, trimesh_to_pymeshlab


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
def test_points_to_barycentric(hemisphere: tuple[tm.Trimesh, wp.Mesh], method: str):
    mesh_tm, mesh_wp = hemisphere

    barycentric_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    points_np = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)
    barycentric_tm = tm.triangles.points_to_barycentric(mesh_tm.triangles, points_np, method=method)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, points_wp, method=method
    )
    assert np.allclose(barycentric_wp.numpy(), barycentric_tm, rtol=1e-5, atol=1e-5)


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
