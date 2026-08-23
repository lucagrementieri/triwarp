"""Regression tests for ``triwarp.curvature`` against ``trimesh.curvature`` (CPU reference)."""

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.comparisons import fraction_within
from tests.conversions import points_to_warp, trimesh_to_pymeshlab


@pytest.mark.parity("principal_curvature", "igl")
def test_principal_curvature(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: curvature values against libigl on an icosahedron, frame-dependent path."""
    mesh_tm, mesh_wp = icosahedron

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, _ = igl.principal_curvature(vertices_np, faces_np, useKring=False)

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # frame_independent=False reproduces igl::principal_curvature's symmetrized shape operator.
    _, _, pv1_wp, pv2_wp = tw.curvature.principal_curvature(
        vertices_wp, faces_wp, frame_independent=False
    )

    assert np.allclose(pv1_wp.numpy(), pv1_igl, atol=1e-3, rtol=1e-3)
    assert np.allclose(pv2_wp.numpy(), pv2_igl, atol=1e-3, rtol=1e-3)


def test_principal_curvature_half_torus(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class B (directions up to sign): values and directions where curvature varies."""
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    pd1_igl, pd2_igl, pv1_igl, pv2_igl, bad_igl = igl.principal_curvature(
        vertices_np, faces_np, useKring=False
    )

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # frame_independent=False reproduces igl::principal_curvature's symmetrized shape operator.
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = tw.curvature.principal_curvature(
        vertices_wp, faces_wp, frame_independent=False
    )

    # Exclude vertices igl marked bad (degenerate) and umbilics where PV1 ~ PV2 (dirs undefined)
    bad = np.array(bad_igl, dtype=np.int32)
    gap = np.abs(pv1_igl - pv2_igl)
    mask = np.ones(len(pv1_igl), dtype=bool)
    if len(bad) > 0:
        mask[bad] = False
    mask[gap < 1e-2] = False

    # Tolerance is relaxed relative to the icosahedron test: float32 input vs libigl float64,
    # plus slight radius difference from avg_edge_length rounding.
    assert np.allclose(pv1_wp.numpy()[mask], pv1_igl[mask], atol=5e-2, rtol=5e-2)
    assert np.allclose(pv2_wp.numpy()[mask], pv2_igl[mask], atol=5e-2, rtol=5e-2)
    # Directions defined up to sign — compare |cos angle| ≈ 1 at non-umbilic vertices
    pd1_dot = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy()[mask], pd1_igl[mask]))
    pd2_dot = np.abs(np.einsum("ij,ij->i", pd2_wp.numpy()[mask], pd2_igl[mask]))
    assert np.allclose(pd1_dot, 1.0, atol=1e-1)
    assert np.allclose(pd2_dot, 1.0, atol=1e-1)


def test_principal_curvature_frame_independent(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (a fraction bound): the frame-independent map stays close to libigl in the bulk.

    ``frame_independent=True`` solves the true generalized eigenproblem (a surface invariant)
    rather than libigl's frame-dependent symmetrized operator. The two formulations share the
    trace of the shape operator, so the mean curvature ``(PV1 + PV2) / 2`` is preserved exactly;
    only the eigenvalue *spread* differs, and only appreciably at high-anisotropy vertices where
    ``PV1 - PV2`` is large. The bulk of vertices therefore stay close to libigl.
    """
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, bad_igl = igl.principal_curvature(vertices_np, faces_np, useKring=False)

    vertices_wp = points_to_warp(vertices_np, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # Default (frame_independent=True): true Weingarten map, independent of the tangent frame.
    _, _, pv1_wp, pv2_wp = tw.curvature.principal_curvature(vertices_wp, faces_wp)
    pv1_indep = pv1_wp.numpy()
    pv2_indep = pv2_wp.numpy()

    # Same masking as the frame-dependent test: drop degenerate and umbilic vertices.
    bad = np.array(bad_igl, dtype=np.int32)
    gap = np.abs(pv1_igl - pv2_igl)
    mask = np.ones(len(pv1_igl), dtype=bool)
    if len(bad) > 0:
        mask[bad] = False
    mask[gap < 1e-2] = False

    # Mean curvature (the shared trace invariant) must match libigl tightly.
    mean_indep = 0.5 * (pv1_indep + pv2_indep)
    mean_igl = 0.5 * (pv1_igl + pv2_igl)
    assert np.allclose(mean_indep[mask], mean_igl[mask], atol=5e-2, rtol=5e-2)

    # The principal values themselves stay close for the vast majority of vertices; genuine
    # divergence is confined to the few highest-anisotropy vertices, which the mask already drops.
    # Class C, so it carries the shuffle probe its helper asks for: both fractions measure
    # **1.0000** over the 544 surviving vertices, and permuting the reference drops them to 0.105
    # and 0.074 -- 9x under the bar, so the threshold is testing the correspondence and not the
    # marginal distributions.
    assert fraction_within(pv1_indep[mask], pv1_igl[mask]) > 0.95
    assert fraction_within(pv2_indep[mask], pv2_igl[mask]) > 0.95


@pytest.mark.parity("principal_curvature", "pymeshlab")
def test_principal_curvature_directions_match_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B on the principal *direction*, which is the only quantity MeshLab exposes comparably.

    ``compute_curvature_principal_directions_per_vertex`` writes two direction matrices and its
    ``vertex_curvature_principal_dir1_matrix()`` entries are **unit vectors** -- measured ``|d1| ==
    1`` at every vertex -- so the curvature *magnitudes* are simply not in them. Its scalar output
    is a mean curvature over a neighbourhood MeshLab derives itself rather than from a radius, and
    it correlates 0.94 with triwarp's with a systematic offset (max deviation 0.82), so it is not a
    value oracle either. The direction is, and the transform is the usual eigenvector sign freedom:
    an eigenvector is defined up to sign, so the comparison is ``|dot| == 1``.

    **Fixture choice is the substance here.** Principal directions are only defined where the two
    principal curvatures differ, so ``torus`` -- whose curvature gap is at minimum 2.63 and median
    3.25 -- is the fixture, and the two obvious alternatives are excluded for measured reasons: on
    an ``icosphere`` every point is umbilic (``k1 == k2``, so any orthonormal tangent pair is a
    valid answer and the agreement reads a meaningless 0.62), and ``half_torus``'s gap falls to
    0.096, where only 54% of vertices reach ``|dot| > 0.99``.

    **Measured, and the mutation probes.** On ``torus`` the worst ``|dot|`` over all 1 024 vertices
    is **0.9997** against a 0.99 bound -- a 33x margin on the deviation from 1. Pairing triwarp's
    first direction with MeshLab's *second* instead collapses it to a mean of 0.058 and only 2.3% of
    vertices above the bound; comparing it against the vertex normal gives a mean of 0.002 and 0%.
    So neither an axis swap nor "return any tangent vector" survives.

    MeshLab's second direction is deliberately **not** asserted: 94.8% of vertices agree to 0.99 but
    the remaining 5% fall to 0.009, i.e. MeshLab and triwarp order the two eigenvectors differently
    at some vertices. That is an ordering convention, and pinning the first direction is the part
    that says the two computed the same shape operator.

    ``autoclean=False`` is load-bearing: the filter defaults to deleting unreferenced vertices,
    which would silently renumber the output against triwarp's.
    """
    mesh_tm, mesh_wp = torus

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_curvature_principal_directions_per_vertex(
        method="Quadric Fitting", autoclean=False
    )
    assert meshset_pml.current_mesh().vertex_number() == mesh_tm.vertices.shape[0]
    direction_pml = np.asarray(
        meshset_pml.current_mesh().vertex_curvature_principal_dir1_matrix(), dtype=np.float64
    )
    direction_pml /= np.linalg.norm(direction_pml, axis=1, keepdims=True)

    direction_wp, _direction2_wp, pv1_wp, pv2_wp = tw.curvature.principal_curvature(
        mesh_wp.points, mesh_wp.indices
    )
    # The fixture must actually have distinct principal curvatures, or the directions are arbitrary.
    assert np.abs(pv1_wp.numpy() - pv2_wp.numpy()).min() > 1.0

    dots_np = np.abs(np.einsum("ij,ij->i", direction_pml, direction_wp.numpy()))
    assert dots_np.min() > 0.99, f"worst |dot| {dots_np.min():.4f}"


@pytest.mark.parity("discrete_gaussian_curvature", "trimesh")
def test_discrete_gaussian_curvature(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the Cohen-Steiner/Morvan ball measure against trimesh's, at the same radius.

    Both sides are fed *trimesh's* face angles, so the comparison isolates the ball integration
    rather than re-testing [`triangles.face_angles`], which has its own oracle. Only four query
    points, which is enough because the measure is local and each one integrates an independent
    1-ring.
    """
    mesh_tm, mesh_wp = hemisphere

    face_angles_tm = mesh_tm.face_angles
    points_tm = mesh_tm.vertices[:4]
    radius = 0.1
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    gauss_curvature_wp = tw.curvature.discrete_gaussian_curvature(
        points_wp, vertices_wp, faces_wp, face_angles_wp, radius
    )
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("mesh_name", "radius"), [("icosahedron", 2.0), ("icosphere", 0.5)])
@pytest.mark.parity("discrete_mean_curvature", "trimesh")
def test_discrete_mean_curvature(
    request: pytest.FixtureRequest, mesh_name: str, radius: float
) -> None:
    """
    Class A: the ball mean-curvature measure against trimesh's, over every vertex.

    **Two fixtures, because either alone tests half of it.** On the ``icosahedron`` the radius of
    2.0 exceeds the mesh, so every query integrates the whole surface -- the case that exercises
    the ball clipping rather than avoiding it. But the icosahedron is *regular*, so that answer is
    one number repeated: measured on trimesh's side, **1 unique value across all 12 vertices, spread
    exactly 0.0**. A comparison of two constant arrays cannot see a permuted result, an off-by-one
    in the gather or a query/vertex index swap -- only a global scale error. The ``icosphere`` at
    0.5 is the per-vertex half: 10 distinct values over its 642 vertices at a spread of 0.0497, and
    the assert below checks that the reference really did vary before comparing to it.

    ``benchmarks/test_curvature.py`` records why pymeshlab cannot be the oracle here (a
    different operator, 0.982 correlation with a 7 % offset).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_tm = mesh_tm.vertices
    mean_curvature_tm = tm.curvature.discrete_mean_curvature_measure(mesh_tm, points_tm, radius)

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    mean_curvature_wp = tw.curvature.discrete_mean_curvature(
        points_wp, vertices_wp, faces_wp, radius
    )
    # Non-vacuous on the curved fixture: a constant reference would pass any per-vertex bug.
    assert mesh_name == "icosahedron" or np.ptp(mean_curvature_tm) > 1e-3
    assert np.allclose(mean_curvature_wp.numpy(), mean_curvature_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not wp.is_cuda_available(),
    reason="needs a second device to make the current device differ from the arrays' device",
)
def test_discrete_gaussian_curvature_ignores_the_current_device(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
):
    """
    Class A: ``discrete_gaussian_curvature`` answers on its inputs' device, not Warp's current one.

    Companion to ``test_vertices.py``'s scatter-wrapper case: this function's ``scatter_offset_sum``
    launch forwarded no ``device=``, which the ordinary tests cannot see because they run with the
    arrays' device already current.
    """
    mesh_tm, mesh_wp = hemisphere
    radius = 0.5
    points_tm = mesh_tm.vertices
    face_angles_tm = mesh_tm.face_angles
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = points_to_warp(points_tm, mesh_wp.device)
    vertices_wp = points_to_warp(mesh_tm.vertices, mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)

    with wp.ScopedDevice("cpu"):
        gauss_curvature_wp = tw.curvature.discrete_gaussian_curvature(
            points_wp, vertices_wp, faces_wp, face_angles_wp, radius
        )

    assert str(gauss_curvature_wp.device) == str(mesh_wp.device)
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)
