"""
Regression tests for ``triwarp.energies``: the quadratic forms assembled from a Laplacian.

Every comparison here is against libigl, which is the only reference that exposes these operators at
all -- ``harmonic_integrated_from_laplacian_and_mass``, ``hessian_energy``,
``curved_hessian_energy``, the Crouzeix-Raviart pair and ``lscm``'s ``Q``. Mirrors the module's
source order: k-harmonic, the two Hessian energies, the Crouzeix-Raviart pair, the LSCM operators.

``vector_area_matrix`` has no binding of its own, so it is derived from the two things that do:
``A = (-repdiag(L, 2) - Q) / 2``. That is a class-B transform, named here so the assert is not
mistaken for a direct one.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import bsr_to_csr

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _mesh_numpy(mesh_tm: tm.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices float64 (n, 3), faces int64 (n_faces, 3))`` for the libigl references."""
    return np.asarray(mesh_tm.vertices, dtype=np.float64), np.asarray(mesh_tm.faces, dtype=np.int64)


def _upload_bsr_float64(
    matrix_sp: sp.spmatrix, device: str | wp.context.Device
) -> wp.sparse.BsrMatrix:
    """Upload a scipy sparse matrix as a float64 1x1-block BSR on ``device``."""
    coo = matrix_sp.tocoo()
    return wp.sparse.bsr_from_triplets(
        coo.shape[0],
        coo.shape[1],
        wp.array(coo.row.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(coo.col.astype(np.int32), dtype=wp.int32, device=device),
        wp.array(coo.data.astype(np.float64), dtype=wp.float64, device=device),
        prune_numerical_zeros=False,
    )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
@pytest.mark.parity("k_harmonic", "igl")
def test_k_harmonic_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, k: int, device: str
) -> None:
    """
    Class A: identical Laplacian and mass on both sides isolate the k-harmonic composition.

    igl's cotangent Laplacian and barycentric mass diagonal are handed to
    ``igl.harmonic_integrated_from_laplacian_and_mass`` and (uploaded unchanged) to
    ``k_harmonic``; the assembled ``Q`` must then agree to assembly rounding at every
    power, including the ``k == 3`` case whose entries igl itself flags as not numerically robust.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    q_igl = igl.harmonic_integrated_from_laplacian_and_mass(laplacian_igl, mass_igl, k).toarray()

    laplacian_wp = _upload_bsr_float64(laplacian_igl, mesh_wp.device)
    mass_wp = wp.array(mass_igl.diagonal(), dtype=wp.float64, device=mesh_wp.device)
    q_wp = bsr_to_csr(tw.energies.k_harmonic(laplacian_wp, mass_wp, k=k)).toarray()

    assert q_wp.shape == q_igl.shape
    scale = np.abs(q_igl).max()
    assert np.allclose(q_wp, q_igl, rtol=1e-9, atol=1e-9 * scale)


def test_k_harmonic_identity_mass_and_power_guard(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    With ``mass=None`` the operator is the plain power ``(-L)^k``, the ``tutte`` flavor.

    Checked against scipy's own sparse product; ``k < 1`` must raise.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = _upload_bsr_float64(laplacian_igl, mesh_wp.device)
    q_wp = bsr_to_csr(tw.energies.k_harmonic(laplacian_wp, k=2)).toarray()
    q_sp = (laplacian_igl @ laplacian_igl).toarray()
    assert np.allclose(q_wp, q_sp, rtol=1e-9, atol=1e-9 * np.abs(q_sp).max())

    with pytest.raises(ValueError, match="k must be >= 1"):
        tw.energies.k_harmonic(laplacian_wp, k=0)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
@pytest.mark.parity("hessian_energy", "igl")
def test_hessian_energy_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the assembled ``(n_vertices, n_vertices)`` matrix.

    igl is handed the float32-rounded vertices triwarp actually computes from, so the comparison
    isolates the operator assembly (the two-ring contraction, the Voronoi mass, the boundary
    kill) from input precision; entries scale as the inverse fourth power of the mesh size, which
    would otherwise let vertex rounding dominate the tolerance. The open fixtures are the ones
    that exercise the killed boundary degrees of freedom.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_wp.points.numpy(), dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_wp.indices.numpy().reshape(-1, 3), dtype=np.int64)

    q_igl = igl.hessian_energy(vertices_np, faces_np).toarray()
    q_wp = bsr_to_csr(tw.energies.hessian_energy(mesh_wp.points, mesh_wp.indices)).toarray()

    assert q_wp.shape == q_igl.shape
    assert np.abs(q_igl).max() > 0.0
    scale = np.abs(q_igl).max()
    assert np.allclose(q_wp, q_igl, rtol=1e-7, atol=1e-7 * scale)


def test_hessian_energy_annihilates_linear_fields_where_biharmonic_does_not(device: str) -> None:
    """
    The natural-boundary property the energy exists for, checked without igl.

    On a **flat** open mesh every affine field ``a + b.x`` has zero Hessian, so it must be exactly
    in the energy's null space — while the clamped biharmonic operator (``k_harmonic`` at
    ``k == 2``) penalizes the same fields at the boundary, which is the distortion Stein et
    al. 2018 diagnose. The contrast is asserted so the null-space check cannot pass vacuously.
    Flatness matters: on the curved fixtures the piecewise-linear Hessian of ``b.x`` is genuinely
    nonzero across bent edges (measured 5.06 against a 9.0 matrix scale on ``icosahedron``, for
    igl and triwarp alike), so this property is only testable on a planar patch.
    """
    vertices_wp, faces_wp = tw.creation.grid(count=(7, 7), device=device)
    vertices_np = np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64)
    n_vertices = len(vertices_np)
    linear_fields = np.c_[np.ones(n_vertices), vertices_np]

    q_hessian = bsr_to_csr(tw.energies.hessian_energy(vertices_wp, faces_wp)).toarray()
    residual_hessian = np.abs(q_hessian @ linear_fields).max()

    laplacian_wp = tw.laplacian.cotmatrix(vertices_wp, faces_wp, dtype=wp.float64)
    mass_wp = tw.laplacian.mass_matrix_entries(vertices_wp, faces_wp, dtype=wp.float64)
    q_biharmonic = bsr_to_csr(tw.energies.k_harmonic(laplacian_wp, mass_wp, k=2)).toarray()
    residual_biharmonic = np.abs(q_biharmonic @ linear_fields).max()

    scale = np.abs(q_hessian).max()
    assert residual_hessian < 1e-9 * scale
    assert residual_biharmonic > 1e-3 * scale


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
@pytest.mark.parity("curved_hessian_energy", "igl")
def test_curved_hessian_energy_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on the assembled ``(n_vertices, n_vertices)`` matrix.

    The per-face contraction of ``D^T Mi (L + K) Mi D`` must agree with igl's chained sparse
    products; igl gets the float32-rounded vertices for the same reason as ``hessian_energy``'s
    test. triwarp numbers and orients its unique edges differently from ``igl::orient_halfedges``
    (min-vertex-first instead of first-occurrence-first), and the agreement here is what shows the
    energy is invariant to that gauge. ``torus`` is the curvature-rich closed case where the
    ``K`` correction actually contributes.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_wp.points.numpy(), dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_wp.indices.numpy().reshape(-1, 3), dtype=np.int64)

    q_igl = igl.curved_hessian_energy(vertices_np, faces_np).toarray()
    q_wp = bsr_to_csr(tw.energies.curved_hessian_energy(mesh_wp.points, mesh_wp.indices)).toarray()

    assert q_wp.shape == q_igl.shape
    assert np.abs(q_igl).max() > 0.0
    scale = np.abs(q_igl).max()
    assert np.allclose(q_wp, q_igl, rtol=1e-7, atol=1e-7 * scale)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus"])
def test_curved_hessian_energy_annihilates_constants(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """Constants have zero energy by construction: every CR gradient row sums to zero."""
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    q_wp = bsr_to_csr(tw.energies.curved_hessian_energy(mesh_wp.points, mesh_wp.indices)).toarray()
    ones = np.ones(q_wp.shape[0])
    assert np.abs(q_wp).max() > 0.0
    assert np.abs(q_wp @ ones).max() < 1e-9 * np.abs(q_wp).max()


def _igl_edge_arguments(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Triwarp's ``edges_unique`` numbering in the ``(E, EMAP)`` layout igl's CR bindings take.

    igl's ``EMAP`` is column-major over ``igl::oriented_facets`` — entry ``c * n_faces + f`` is
    the edge opposite corner ``c`` of face ``f`` — while triwarp's ``inverse`` is row-major over
    halfedges ``(f[s], f[s+1])``, where slot ``s`` spans the edge opposite corner ``(s + 2) % 3``.
    """
    unique_edges_wp, inverse_wp = tw.edges.edges_unique(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )
    inverse_np = inverse_wp.numpy().reshape(-1, 3)
    edge_map_igl = np.concatenate([inverse_np[:, (c + 1) % 3] for c in range(3)], dtype=np.int64)
    return unique_edges_wp.numpy().astype(np.int64), edge_map_igl


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
@pytest.mark.parity("crouzeix_raviart_cotmatrix", "igl")
def test_crouzeix_raviart_cotmatrix_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B; the named transform is the edge numbering, handed *to* igl.

    ``igl.crouzeix_raviart_cotmatrix`` accepts an explicit ``(E, EMAP)``, so feeding it triwarp's
    ``edges_unique`` numbering makes the two ``(n_edges, n_edges)`` matrices directly comparable —
    no row permutation is applied to either side's output.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    edges_igl, edge_map_igl = _igl_edge_arguments(mesh_wp)
    matrix_igl = igl.crouzeix_raviart_cotmatrix(
        vertices_np, faces_np, edges_igl, edge_map_igl
    ).toarray()
    matrix_wp = bsr_to_csr(
        tw.energies.crouzeix_raviart_cotmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()

    assert matrix_wp.shape == matrix_igl.shape == (len(edges_igl), len(edges_igl))
    assert np.allclose(matrix_wp, matrix_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
@pytest.mark.parity("crouzeix_raviart_massmatrix", "igl")
def test_crouzeix_raviart_massmatrix_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B via the same handed-to-igl edge numbering as the cotmatrix test.

    Asserted on the dense form so a stray off-diagonal triplet is visible, exactly like
    ``test_mass_matrix_assembled_matches_igl``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    edges_igl, edge_map_igl = _igl_edge_arguments(mesh_wp)
    matrix_igl = igl.crouzeix_raviart_massmatrix(
        vertices_np, faces_np, edges_igl, edge_map_igl
    ).toarray()
    matrix_wp = bsr_to_csr(
        tw.energies.crouzeix_raviart_massmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()

    assert matrix_wp.shape == matrix_igl.shape
    assert np.allclose(matrix_wp, matrix_igl, rtol=1e-5, atol=1e-5)


def test_crouzeix_raviart_shared_edge_numbering(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Precomputed ``(unique_edges, edge_map)`` must not change the answer; half a pair raises."""
    _mesh_tm, mesh_wp = half_torus
    unique_edges, edge_map = tw.edges.edges_unique(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )
    derived = bsr_to_csr(
        tw.energies.crouzeix_raviart_cotmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()
    supplied = bsr_to_csr(
        tw.energies.crouzeix_raviart_cotmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges, edge_map=edge_map
        )
    ).toarray()
    assert np.array_equal(derived, supplied)

    with pytest.raises(ValueError, match="together"):
        tw.energies.crouzeix_raviart_massmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges
        )


def test_operator_family_empty_mesh(device: str) -> None:
    """Empty face buffers return square zero-nnz operators of the right dimension."""
    vertices_wp = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    hessian = tw.energies.hessian_energy(vertices_wp, faces_wp)
    assert (int(hessian.nrow), int(hessian.ncol)) == (3, 3)
    curved = tw.energies.curved_hessian_energy(vertices_wp, faces_wp)
    assert (int(curved.nrow), int(curved.ncol)) == (3, 3)
    cr_cot = tw.energies.crouzeix_raviart_cotmatrix(vertices_wp, faces_wp)
    assert (int(cr_cot.nrow), int(cr_cot.ncol)) == (0, 0)
    cr_mass = tw.energies.crouzeix_raviart_massmatrix(vertices_wp, faces_wp)
    assert (int(cr_mass.nrow), int(cr_mass.ncol)) == (0, 0)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_lscm_hessian_matches_igl(request, device, mesh_name):
    # No CPU skip: this builds the Hessian only, no conjugate-gradient solve.
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_numpy(mesh_tm)
    n_vertices = int(mesh_wp.points.shape[0])

    # igl.lscm returns (V_uv, Q); its Q equals -repdiag(L, 2) - 2A exactly.
    pins_np = np.array([0, 1], dtype=np.int64)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    _, hessian_igl = igl.lscm(vertices_np, faces_np, pins_np, pins_uv_np)

    hessian_wp = tw.energies.lscm_hessian(mesh_wp.points, mesh_wp.indices)
    hessian_dense = sp.csr_matrix(
        (hessian_wp.values.numpy(), hessian_wp.columns.numpy(), hessian_wp.offsets.numpy()),
        shape=(2 * n_vertices, 2 * n_vertices),
    ).toarray()

    assert np.allclose(hessian_dense, hessian_igl.toarray(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_vector_area_matrix_matches_igl_derived(request, device, mesh_name):
    # The bindings do not expose vector_area_matrix; derive it from A = (-repdiag(L,2) - Q) / 2.
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_numpy(mesh_tm)
    n_vertices = int(mesh_wp.points.shape[0])

    pins_np = np.array([0, 1], dtype=np.int64)
    pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    _, hessian_igl = igl.lscm(vertices_np, faces_np, pins_np, pins_uv_np)
    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    area_igl = (-sp.block_diag([laplacian_igl, laplacian_igl]) - hessian_igl) / 2.0

    area_wp = tw.energies.vector_area_matrix(mesh_wp.points, mesh_wp.indices)
    area_dense = sp.csr_matrix(
        (area_wp.values.numpy(), area_wp.columns.numpy(), area_wp.offsets.numpy()),
        shape=(2 * n_vertices, 2 * n_vertices),
    ).toarray()

    assert np.allclose(area_dense, area_igl.toarray(), rtol=1e-5, atol=1e-5)
