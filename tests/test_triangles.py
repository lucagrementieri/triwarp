"""Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import (
    faces_igl,
    trimesh_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
)


@pytest.mark.parity("face_normals_and_areas", "trimesh")
def test_face_normals_and_areas(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the unit normal and the area per face, both against trimesh.

    The two come out of one cross product, so comparing both is what separates a normalization
    bug from a winding one -- a flipped face has the right area and the wrong normal.
    """
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


@pytest.mark.parity("face_normals_and_areas", "open3d")
def test_face_normals_matches_open3d(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the normals half; the areas reduce to open3d's one total.

    ``compute_triangle_normals`` returns *unit* normals, unlike MeshLab's raw cross product above,
    so no transform is needed. The area half has no per-face open3d counterpart
    (``get_surface_area`` is the total), so the sum is compared as its class-B reduction.
    ``half_torus`` for the same varying-area reason as the partial-references test.
    """
    mesh_tm, mesh_wp = half_torus
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.compute_triangle_normals()
    assert np.allclose(normals_wp.numpy(), np.asarray(mesh_o3d.triangle_normals), atol=1e-5)
    assert np.isclose(float(areas_wp.numpy().sum()), mesh_o3d.get_surface_area(), rtol=1e-5)


@pytest.mark.parity("face_normals_and_areas", "pyvista")
def test_face_normals_and_areas_match_pyvista(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on both halves, from the one reference that answers both -- in two calls.

    VTK splits the cross product across two filters: ``compute_normals`` gives the *unit* cell
    normal (float32, per this module's dtype note) and ``compute_cell_sizes`` the area. The three
    ``compute_normals`` flags matter and are passed explicitly: ``consistent_normals`` and
    ``auto_orient_normals`` would let VTK re-wind the mesh before differentiating it, which would
    compare triwarp's normals against a *different* orientation, and ``split_vertices`` would change
    the point count. ``half_torus`` for the varying-area reason the partial-references test gives.
    """
    mesh_tm, mesh_wp = half_torus
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    mesh_pv = trimesh_to_pyvista(mesh_tm)
    normals_pv = mesh_pv.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    assert np.allclose(
        normals_wp.numpy(), np.asarray(normals_pv.cell_data["Normals"]), rtol=1e-5, atol=1e-5
    )
    sizes_pv = mesh_pv.compute_cell_sizes(length=False, area=True, volume=False)
    assert np.allclose(
        areas_wp.numpy(), np.asarray(sizes_pv.cell_data["Area"]), rtol=1e-5, atol=1e-5
    )


def test_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the three corner angles per face against ``Trimesh.face_angles``, in corner order.

    Column order is part of the claim -- angle ``i`` is at corner ``i`` -- because the
    cotangent Laplacian and the angle defect both index it that way.
    """
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
    """Class B (per-measure naming): the four VCG measures, against the filter they came from."""
    mesh_tm, mesh_wp = half_torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_aspect_ratio_per_face(metric=filter_metric)
    quality_pml = meshset_pml.current_mesh().face_scalar_array()

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric=metric)
    assert np.allclose(quality_wp.numpy(), quality_pml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("metric", "measure", "reciprocal"),
    [
        ("area", "area", False),
        ("aspect_ratio", "radius_ratio", False),
        ("mean_ratio", "shape", False),
        ("mean_ratio", "aspect_frobenius", True),
    ],
)
@pytest.mark.parity("face_quality", "pyvista")
def test_face_quality_against_the_verdict_measures(
    half_torus: tuple[tm.Trimesh, wp.Mesh], metric: str, measure: str, reciprocal: bool
):
    """
    Decode VTK's Verdict measure names onto triwarp's, three class A and one class B.

    The mapping is the trap, not the arithmetic, and the two inversions in it will mislead anyone
    reading pyvista's docs instead of this table:

    - Verdict's ``radius_ratio`` is ``R / (2 r_in)``, which is triwarp's **aspect_ratio**;
    - triwarp's own ``radius_ratio`` is its *reciprocal* (asserted below so the inversion is pinned
      rather than described);
    - ``shape`` is ``4 sqrt(3) A / (a^2 + b^2 + c^2)``, triwarp's **mean_ratio**;
    - ``aspect_frobenius`` is one over that -- the class B row, one named reciprocal -- and
      ``condition`` duplicates it exactly, so it gets no row of its own.

    ``area_max_side`` has no Verdict counterpart at all, and Verdict's ``aspect_ratio``
    (``max_edge / (2 sqrt(3) r_in)``) has no triwarp counterpart; neither is compared.

    Anti-vacuity, which this comparison is unusually exposed to: of ``cell_quality``'s 28 measures
    only 12 are defined on a triangle and the other 16 come back as the constant ``-1.0`` null
    value, while ``distortion`` is a constant ``1.0`` on ordinary input. So the measure is asserted
    to *vary* across faces before it is compared -- on ``half_torus`` it does, by construction.
    """
    mesh_tm, mesh_wp = half_torus
    quality_pv = np.asarray(trimesh_to_pyvista(mesh_tm).cell_quality(measure).cell_data[measure])
    # Neither a null (-1.0) nor a constant: both would pass an allclose against a broken port.
    assert quality_pv.min() > 0.0
    assert np.ptp(quality_pv) > 1e-3

    quality_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric=metric)
    expected_pv = 1.0 / quality_pv if reciprocal else quality_pv
    assert np.allclose(quality_wp.numpy(), expected_pv, rtol=1e-5, atol=1e-5)

    if measure == "radius_ratio":  # triwarp's like-named metric is the other way up
        radius_ratio_wp = tw.triangles.face_quality(
            mesh_wp.points, mesh_wp.indices, metric="radius_ratio"
        )
        assert np.allclose(radius_ratio_wp.numpy() * quality_pv, 1.0, rtol=1e-4, atol=1e-4)


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


@pytest.mark.parity("face_angles", "pyvista")
def test_face_angles_extremes_against_pyvista(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class B: VTK gives the two *extremes* only, in **degrees**.

    ``cell_quality('min_angle')`` and ``('max_angle')`` are the same corner angles this module
    returns, reduced per face and converted -- so the named transform is ``np.degrees`` plus a
    min/max along the row, and the comparison cannot see the third angle or which corner each
    belongs to. That is what keeps the igl / trimesh row above the element-wise oracle.
    """
    mesh_tm, mesh_wp = half_torus
    quality_pv = trimesh_to_pyvista(mesh_tm).cell_quality(["min_angle", "max_angle"])

    angles_np = np.degrees(tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices).numpy())

    assert np.allclose(
        angles_np.min(axis=1), np.asarray(quality_pv.cell_data["min_angle"]), rtol=1e-5, atol=1e-4
    )
    assert np.allclose(
        angles_np.max(axis=1), np.asarray(quality_pv.cell_data["max_angle"]), rtol=1e-5, atol=1e-4
    )
    # Non-vacuous: on a mesh of congruent equilateral faces both columns would read 60 everywhere.
    assert np.ptp(np.asarray(quality_pv.cell_data["min_angle"])) > 1.0


@pytest.mark.parity("face_normals_and_areas", "meshlib")
@pytest.mark.parity(
    "face_centroids",
    "meshlib",
    benchmarked=False,
    reason="MeshLib's triCenter is per *face*, so a batched row would be a Python loop over the "
    "face buffer and would time the loop rather than MeshLib -- 49-67x the batched cost where a "
    "batched form exists at all (section 6). It is a sound correctness oracle at fixture size, "
    "which is what this test uses it as. igl and pyvista carry the timed rows for this group.",
)
@pytest.mark.parity(
    "face_quality",
    "meshlib",
    benchmarked=False,
    reason="triangleAspectRatio is per *face*, same as triCenter above: batching it means a Python "
    "loop, and that row would price the loop. It is the strongest correctness oracle this group "
    "has -- exactly 0.0 difference against metric='aspect_ratio', where pyvista names the same "
    "quantity radius_ratio and igl gives it only as a ratio of two other arrays -- so the "
    "comparison belongs here and the timing stays with igl / pymeshlab / pyvista.",
)
def test_per_face_quantities_match_meshlib(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A for three per-face families at once, all element-wise in face order.

    MeshLib is the fourth independent implementation of these, and it is the one that pins two
    conventions the others leave open. Its ``computePerFaceNormals`` is **normalized** (|n| = 1 to
    6e-08 measured), unlike pymeshlab's ``face_normal_matrix()``, which is the raw cross product at
    magnitude ``2 * area``; and its ``triangleAspectRatio`` is exactly the measure
    [`face_quality`][triwarp.triangles.face_quality] calls ``aspect_ratio`` -- **0.0** difference,
    where pyvista names the same quantity ``radius_ratio`` and igl gives it only as a ratio of two
    other arrays.

    One named transform, the same one ``igl.doublearea`` needs: ``dblArea`` is twice the area.
    Everything else is direct. Three of MeshLib's four entry points here are **per-face** rather
    than batched, so they are looped on the reference side -- fine in a test at this size, and the
    reason ``benchmarks/`` reads those rows as an upper bound (see section 6).

    Testing the three together is deliberate: they come out of the same corner load, so a
    fixture-level disagreement (a converter dropping a vertex, a face buffer reshaped wrong) shows
    up in all three at once and is distinguishable from a real per-quantity bug.
    """
    mesh_tm, mesh_wp = half_torus
    mesh_ml = trimesh_to_meshlib(mesh_tm)
    topology_ml, points_ml = mesh_ml.topology, mesh_ml.points
    n_faces = int(mesh_wp.indices.shape[0]) // 3
    assert topology_ml.numValidFaces() == n_faces > 0  # non-vacuity, and the converter's own check

    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    normals_ml = mn.toNumpyArray(mm.computePerFaceNormals(mesh_ml))
    assert np.allclose(np.linalg.norm(normals_ml, axis=1), 1.0, atol=1e-6)
    assert np.allclose(normals_wp.numpy(), normals_ml, rtol=1e-5, atol=1e-5)

    faces_ml = [mm.FaceId(f) for f in range(n_faces)]
    areas_ml = np.array([mm.dblArea(topology_ml, points_ml, f) / 2.0 for f in faces_ml])
    assert np.allclose(areas_wp.numpy(), areas_ml, rtol=1e-5, atol=1e-5)

    centroids_ml = np.array([[*mm.triCenter(topology_ml, points_ml, f)] for f in faces_ml])
    centroids_wp = tw.triangles.face_centroids(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(centroids_wp.numpy(), centroids_ml, rtol=1e-5, atol=1e-5)

    aspect_ml = np.array([mm.triangleAspectRatio(topology_ml, points_ml, f) for f in faces_ml])
    aspect_wp = tw.triangles.face_quality(mesh_wp.points, mesh_wp.indices, metric="aspect_ratio")
    assert np.ptp(aspect_ml) > 0.1  # non-vacuity: a constant would pass any tolerance
    assert np.allclose(aspect_wp.numpy(), aspect_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("face_quality", "igl")
def test_face_quality_aspect_ratio_against_igl(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """Class B (a derived ratio): igl gives the circumradius and inradius as two arrays."""
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
    """
    Class A on a boolean mask, against ``trimesh.triangles.nondegenerate``.

    All-``True`` on this fixture, which is why the degenerate branch is covered separately by
    the zero-area tests in this file -- a mask that was always ``True`` would pass here alone.
    """
    mesh_tm, mesh_wp = hemisphere
    nondegenerate_tm = tm.triangles.nondegenerate(mesh_tm.triangles)
    nondegenerate_wp = tw.triangles.nondegenerate(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(nondegenerate_wp.numpy().astype(bool), nondegenerate_tm)


def test_barycentric_to_points(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: barycentric-to-Cartesian against trimesh, on random unnormalized coordinates.

    The coordinates are not normalized to sum to one, which is deliberate: both libraries treat
    them as affine weights, and normalizing would hide a divide the other does not do.
    """
    mesh_tm, mesh_wp = hemisphere
    barycentric_np = np.random.default_rng(31).random((mesh_tm.triangles.shape[0], 3))
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

    barycentric_np = np.random.default_rng(32).random((mesh_tm.triangles.shape[0], 3))
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
    """
    Class A: the per-triangle closest point against ``trimesh.triangles.closest_point``.

    Unlike the whole-mesh query in ``test_proximity``, each query is matched to *its own*
    triangle, so there is no tie between faces and the point itself is comparable rather than
    only its distance.
    """
    mesh_tm, mesh_wp = hemisphere
    points_np = np.random.default_rng(33).random((mesh_tm.triangles.shape[0], 3))
    closest_points_tm = tm.triangles.closest_point(mesh_tm.triangles, points_np)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    closest_points_wp = tw.triangles.closest_point(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(closest_points_wp.numpy(), closest_points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_centroids", "igl", "pyvista")
def test_face_centroids(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A on both references: one barycentre per face, element-wise, no transform.

    ``igl.barycenter`` and VTK's ``cell_centers`` both return the corner mean in face order — the
    latter is a *parametric* centre in general, but on a triangle that is the barycentre, which is
    why the row is class A rather than a documented approximation.

    The assert that matters beyond the comparison is the last one: the barycentre must lie *in* its
    own triangle, which the barycentric coordinates ``(1/3, 1/3, 1/3)`` state exactly. A function
    that returned the mesh centroid broadcast, or the first corner, would match neither.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    centroids_igl = igl.barycenter(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), faces_igl(mesh_tm)
    )
    centroids_pv = np.asarray(trimesh_to_pyvista(mesh_tm).cell_centers().points)

    centroids_wp = tw.triangles.face_centroids(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(centroids_wp.numpy(), centroids_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(centroids_wp.numpy(), centroids_pv, rtol=1e-5, atol=1e-5)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, centroids_wp
    )
    assert np.allclose(barycentric_wp.numpy(), 1.0 / 3.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "half_torus"])
def test_face_signed_volumes(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A against the NumPy oracle, element-wise, plus class B on the sum.

    ``dot(v0, cross(v1, v2)) / 6`` per face, which is a one-liner in NumPy and therefore an exact
    oracle rather than an approximate one. The class-B half is that the sum over a closed mesh is
    ``mesh_tm.volume`` -- the transform being the reduction -- and it is the assert that would catch
    a per-face sign error the element-wise comparison could only catch if the oracle had the same
    bug. ``half_torus`` is open, so it is here for the element-wise half only, which is exactly the
    point: the per-face quantity is defined whether or not the surface bounds anything.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    triangles_np = np.asarray(mesh_tm.triangles, dtype=np.float64)
    v0, v1, v2 = triangles_np[:, 0], triangles_np[:, 1], triangles_np[:, 2]
    volumes_np = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0

    volumes_wp = tw.triangles.face_signed_volumes(mesh_wp.points, mesh_wp.indices)

    scale = float(np.abs(volumes_np).max())
    assert np.allclose(volumes_wp.numpy(), volumes_np, rtol=1e-5, atol=1e-5 * scale)
    if mesh_tm.is_watertight:
        assert np.isclose(volumes_wp.numpy().sum(), mesh_tm.volume, rtol=1e-5, atol=1e-5)


def test_face_signed_volumes_apex_shifts_each_face_but_not_the_sum(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
):
    """
    Moving the apex changes every per-face volume and leaves the closed-mesh total alone.

    Both halves are needed. The invariance of the sum is the property ``totals.volume`` relies on
    when it passes no apex at all; the *variance* of the individual entries is what makes ``apex``
    a real parameter rather than a decoration, and it is the axis ``sample.sample_volume`` uses --
    it fans from the surface centroid precisely so that no entry comes out negative.
    """
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    vertices_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    at_origin_np = tw.triangles.face_signed_volumes(vertices_wp, faces_wp).numpy()
    shifted_np = tw.triangles.face_signed_volumes(
        vertices_wp, faces_wp, wp.vec3(2.0, -1.0, 0.5)
    ).numpy()

    assert not np.allclose(at_origin_np, shifted_np, atol=1e-4)
    assert np.isclose(at_origin_np.sum(), shifted_np.sum(), rtol=1e-4)


def test_face_signed_volumes_follows_the_input_dtype(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
):
    """``vec3d`` in, ``float64`` out -- the axis ``smoothing``'s volume constraint needs."""
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    vertices_f64_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), dtype=wp.vec3d, device=device
    )
    vertices_f32_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )

    volumes_f64_wp = tw.triangles.face_signed_volumes(vertices_f64_wp, faces_wp)
    volumes_f32_wp = tw.triangles.face_signed_volumes(vertices_f32_wp, faces_wp)

    assert volumes_f64_wp.dtype is wp.float64
    assert volumes_f32_wp.dtype is wp.float32
    assert np.allclose(volumes_f64_wp.numpy(), volumes_f32_wp.numpy(), rtol=1e-6, atol=1e-7)


def test_face_signed_volumes_empty(device: str):
    vertices_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    faces_wp = wp.array([], dtype=wp.int32, device=device)
    assert tw.triangles.face_signed_volumes(vertices_wp, faces_wp).shape == (0,)
