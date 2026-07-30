"""Regression tests for ``triwarp.laplacian`` against igl (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import scipy.sparse as sp
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp

import triwarp as tw
from tests.conversions import bsr_to_dense

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _bsr_to_csr(matrix: wp.sparse.BsrMatrix) -> sp.csr_matrix:
    nrow = int(matrix.nrow)  # pyright: ignore[reportAttributeAccessIssue]
    ncol = int(matrix.ncol)  # pyright: ignore[reportAttributeAccessIssue]
    offsets = matrix.offsets.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    columns = matrix.columns.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    values = matrix.values.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    return sp.csr_matrix((values, columns, offsets), shape=(nrow, ncol))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("cotmatrix_entries", "igl")
def test_cotmatrix_entries(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    cot_entries_igl = igl.cotmatrix_entries(vertices_np, faces_np)
    cot_entries_wp = tw.laplacian.cotmatrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("cotmatrix_entries_intrinsic", "igl")
def test_cotmatrix_entries_intrinsic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    edge_lengths_igl = igl.edge_lengths(vertices_np, faces_np)
    cot_entries_igl = igl.cotmatrix_entries(edge_lengths_igl)
    edge_lengths_wp = wp.array(
        edge_lengths_igl.astype(np.float32), dtype=wp.float32, device=mesh_wp.device
    )
    cot_entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths_wp)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("cotmatrix", "igl")
def test_cotmatrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = _bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))

    assert laplacian_wp.shape == laplacian_igl.shape
    assert np.allclose(laplacian_wp.toarray(), laplacian_igl.toarray(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("cotmatrix", "potpourri3d")
@pytest.mark.parity("mass_matrix_entries", "potpourri3d")
def test_cotmatrix_and_mass_match_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The two operator builds against geometry-central's bindings, which already agree with libigl.

    A third independent implementation of quantities libigl already pins is worth having precisely
    because it is cheap: these are the operators every solver in the library is built on, so a
    regression here surfaces as a wrong answer several modules away.

    ``vertex_areas`` is class A: it is one third of the incident face areas, which is exactly the
    barycentric lumped mass diagonal ``mass_matrix_entries`` returns.

    ``cotan_laplacian`` is class B, and the transform is a **sign flip**. geometry-central builds
    the positive-semidefinite Laplacian while libigl -- and triwarp with it -- builds the negative
    one: measured on ``icosahedron``, ``pp3d.cotan_laplacian`` is ``-igl.cotmatrix`` to 1e-9 entry
    for entry, with a ``+2.887`` diagonal against igl's ``-2.887``. Neither is wrong, but handing
    one to a solver expecting the other flips the sign of every diffusion step, so the negation
    here is the substance of the comparison rather than bookkeeping.

    Read potpourri3d as the *weakest* of the three references rather than the strongest: it
    assembles both of these in vectorized numpy into a scipy COO, not in geometry-central's C++, so
    it is closer to an independent re-derivation of the same formula than to a separate codebase.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)

    cotmatrix_pp = pp3d.cotan_laplacian(vertices_np, faces_np).tocsr()
    cotmatrix_wp = _bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))
    assert cotmatrix_wp.shape == cotmatrix_pp.shape
    assert np.allclose(cotmatrix_wp.toarray(), -cotmatrix_pp.toarray(), rtol=1e-5, atol=1e-5)

    mass_pp = pp3d.vertex_areas(vertices_np, faces_np)
    mass_wp = tw.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(mass_wp.numpy(), mass_pp, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("mass_matrix", "igl")
def test_mass_matrix_assembled_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The *assembled* barycentric mass matrix, not just its diagonal.

    ``test_mass_matrix`` already pins ``mass_matrix_entries`` against ``igl.massmatrix(...)
    .diagonal()``, which is the same numbers; what this adds is the sparse build around them, and
    that is a separate benchmark group for the same reason. Class A on the dense form: the matrix is
    diagonal, so every off-diagonal entry must be zero, and asserting on the full array rather than
    on ``.diagonal()`` is what makes a stray off-diagonal triplet visible.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).tocsr()
    mass_wp = _bsr_to_csr(tw.laplacian.mass_matrix(mesh_wp.points, mesh_wp.indices))

    assert mass_wp.shape == mass_igl.shape
    assert np.allclose(mass_wp.toarray(), mass_igl.toarray(), rtol=1e-5, atol=1e-5)


def test_cotmatrix_null_space(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np)
    ones = np.ones(vertices_np.shape[0], dtype=np.float64)
    assert np.linalg.norm(laplacian_igl @ ones) < 1e-10

    laplacian_wp = _bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))
    ones_wp = np.ones(int(mesh_wp.points.shape[0]), dtype=np.float32)
    assert np.linalg.norm(laplacian_wp @ ones_wp) < 1e-4


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parametrize("equal_weight", [True, False])
@pytest.mark.parity("laplacian_uniform", "trimesh")
@pytest.mark.parity("laplacian_inverse_distance", "trimesh")
def test_laplacian_operator(
    request: pytest.FixtureRequest, mesh_name: str, equal_weight: bool
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    operator_tm = tms.laplacian_calculation(mesh_tm, equal_weight=equal_weight).tocsr()
    operator_wp = _bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=equal_weight)
    )

    assert operator_wp.shape == operator_tm.shape
    assert np.allclose(operator_wp.toarray(), operator_tm.toarray(), rtol=1e-5, atol=1e-5)


def test_laplacian_symmetric_flag(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    # On an open mesh the directed and symmetric adjacencies differ; forcing ``symmetric``
    # overrides the ``not equal_weight`` default, so uniform+symmetric is row-stochastic and
    # symmetric while uniform+directed matches trimesh's (asymmetric) ``edges_to_coo``.
    _, mesh_wp = half_torus
    directed = _bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=True, symmetric=False)
    )
    symmetric = _bsr_to_csr(
        tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=True, symmetric=True)
    )

    directed_dense = directed.toarray()
    symmetric_dense = symmetric.toarray()
    assert not np.allclose(directed_dense, symmetric_dense)
    # Symmetric adjacency has a symmetric sparsity pattern; directed does not (open boundary).
    assert np.array_equal(symmetric_dense != 0.0, (symmetric_dense != 0.0).T)
    assert not np.array_equal(directed_dense != 0.0, (directed_dense != 0.0).T)
    # Both operators are row-stochastic.
    assert np.allclose(symmetric_dense.sum(axis=1), 1.0)
    assert np.allclose(directed_dense.sum(axis=1), 1.0)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("mass_matrix_entries", "igl")
def test_mass_matrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    mass_wp = tw.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(mass_wp.numpy(), mass_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_operators_float64_match_float32(request: pytest.FixtureRequest, mesh_name: str) -> None:
    # Every operator exposes a ``dtype`` parameter for native float64 assembly (used by the
    # linear-system solvers). The float64 build must carry float64 values and agree with the
    # float32 build within float32 precision.
    _, mesh_wp = request.getfixturevalue(mesh_name)
    points, indices = mesh_wp.points, mesh_wp.indices

    cot_entries_f32 = tw.laplacian.cotmatrix_entries(points, indices)
    cot_entries_f64 = tw.laplacian.cotmatrix_entries(points, indices, dtype=wp.float64)
    assert cot_entries_f64.dtype == wp.float64
    assert np.allclose(cot_entries_f64.numpy(), cot_entries_f32.numpy(), rtol=1e-5, atol=1e-5)

    mass_f32 = tw.laplacian.mass_matrix_entries(points, indices)
    mass_f64 = tw.laplacian.mass_matrix_entries(points, indices, dtype=wp.float64)
    assert mass_f64.dtype == wp.float64
    assert np.allclose(mass_f64.numpy(), mass_f32.numpy(), rtol=1e-5, atol=1e-5)

    builders = [
        tw.laplacian.cotmatrix,
        tw.laplacian.laplacian,
        tw.laplacian.uniform_laplacian,
        tw.laplacian.mass_matrix,
    ]
    for builder in builders:
        matrix_f32 = builder(points, indices)
        matrix_f64 = builder(points, indices, dtype=wp.float64)
        assert matrix_f64.values.dtype == wp.float64  # pyright: ignore[reportAttributeAccessIssue]
        dense_f32 = _bsr_to_csr(matrix_f32).toarray()
        dense_f64 = _bsr_to_csr(matrix_f64).toarray()
        assert dense_f64.shape == dense_f32.shape
        assert np.allclose(dense_f64, dense_f32, rtol=1e-5, atol=1e-5)


def test_cotmatrix_entries_intrinsic_float64(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    edge_lengths_igl = igl.edge_lengths(vertices_np, faces_np)
    edge_lengths_wp = wp.array(
        edge_lengths_igl.astype(np.float32), dtype=wp.float32, device=mesh_wp.device
    )
    cot_entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths_wp, dtype=wp.float64)

    assert cot_entries_wp.dtype == wp.float64
    cot_entries_igl = igl.cotmatrix_entries(edge_lengths_igl)
    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


def test_cotmatrix_empty_mesh(device: str) -> None:
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.empty((0, 3), dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    laplacian_wp = _bsr_to_csr(tw.laplacian.cotmatrix(vertices_wp, faces_wp))

    assert laplacian_wp.shape == laplacian_igl.shape == (3, 3)
    assert laplacian_wp.nnz == 0
    assert laplacian_igl.nnz == 0


# --- robust_laplacian / mollify_intrinsic (libigl reference) ---------------------------
@pytest.mark.parametrize("mesh_name", _MESHES)
def test_robust_laplacian_is_unchanged_on_a_clean_mesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    robust = tw.laplacian.robust_laplacian(
        mesh_wp.points, mesh_wp.indices, use_intrinsic_delaunay=False
    )
    plain = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices)

    # Mollification adds nothing when every triangle is already non-degenerate, so with the flips
    # turned off the operator must be the ordinary one up to the intrinsic route's rounding.
    assert np.allclose(
        bsr_to_dense(robust, n_vertices), bsr_to_dense(plain, n_vertices), rtol=1e-4, atol=1e-5
    )


def test_robust_laplacian_keeps_couplings_the_plain_one_drops(sliver_patch: tuple) -> None:
    """
    Mollification's purpose, on a mesh with a triangle too thin to have cotangents.

    Both operators are *finite* -- ``cot_entries_from_l2`` refuses to divide by a zero area, so a
    degenerate face contributes nothing rather than an infinity. Contributing nothing is the only
    finite choice available (a zero-area triangle's angles are 0 or pi), but it is not free: that
    face's edge couplings vanish from the operator, which is what mollification exists to avoid.
    Here vertices 0 and 1 share an edge of the degenerate face and end up **uncoupled** in the
    plain operator while the mollified one couples them.
    """
    vertices_np, _, vertices_wp, faces_wp = sliver_patch
    n_vertices = len(vertices_np)
    plain = bsr_to_dense(tw.laplacian.cotmatrix(vertices_wp, faces_wp), n_vertices)
    robust = bsr_to_dense(
        tw.laplacian.robust_laplacian(vertices_wp, faces_wp, use_intrinsic_delaunay=False),
        n_vertices,
    )

    assert np.isfinite(plain).all()
    assert np.isfinite(robust).all()

    # The coupling the degenerate face should have provided.
    assert plain[0, 1] == 0.0
    assert robust[0, 1] != 0.0
    off_diagonal = ~np.eye(n_vertices, dtype=bool)
    assert (np.abs(plain[off_diagonal]) > 0).sum() < (np.abs(robust[off_diagonal]) > 0).sum()


def test_cotmatrix_entries_are_zero_for_a_zero_area_face(device: str) -> None:
    """
    A collinear triangle yields zero weights, not infinities.

    Regression guard for the division in ``cot_entries_from_l2``: its denominator is
    ``4 * doublearea``, and ``doublearea_from_lengths`` deliberately reports ``0.0`` for a
    degenerate triangle, so an unguarded divide sends the whole assembled operator -- and any
    solve against it -- to NaN off a single bad face.
    """
    collinear_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)

    entries_wp = tw.laplacian.cotmatrix_entries(collinear_wp, faces_wp)
    assert np.array_equal(entries_wp.numpy(), np.zeros((1, 3), dtype=np.float32))
    assert np.isfinite(bsr_to_dense(tw.laplacian.cotmatrix(collinear_wp, faces_wp), 3)).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_robust_laplacian_matches_igl_intrinsic_assembly(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths_wp, delta = tw.laplacian.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)
    assert delta == 0.0  # these fixtures are clean, so the comparison is against the plain lengths

    # igl takes the same (n_faces, 3) opposite-edge-length table, which pins the column order.
    laplacian_igl = igl.cotmatrix_intrinsic(
        np.ascontiguousarray(lengths_wp.numpy(), dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(lengths_wp)
    laplacian_wp = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, cot_entries=entries_wp)

    dense_igl = np.asarray(laplacian_igl.todense())
    assert np.allclose(
        bsr_to_dense(laplacian_wp, len(mesh_tm.vertices)), dense_igl, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
@pytest.mark.parity("robust_laplacian", "igl")
def test_robust_laplacian_matches_igl_intrinsic_delaunay(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)

    # With the flips on, this *is* ``igl::intrinsic_delaunay_cotmatrix`` — the strongest oracle in
    # the module, and one that pins the flip's new-edge-length formula and its winding bookkeeping
    # at once. The intrinsic Delaunay triangulation is unique, so the two implementations need not
    # (and do not) perform the same flips in the same order to agree on the matrix.
    laplacian_igl, _, _ = igl.intrinsic_delaunay_cotmatrix(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    laplacian_wp = tw.laplacian.robust_laplacian(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        bsr_to_dense(laplacian_wp, n_vertices),
        np.asarray(laplacian_igl.todense()),
        rtol=1e-4,
        atol=1e-5,
    )


def test_mollify_intrinsic_is_a_no_op_on_a_clean_mesh(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    original = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices)
    mollified, delta = tw.laplacian.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)

    assert delta == 0.0
    assert np.array_equal(mollified.numpy(), original.numpy())


def test_mollify_intrinsic_restores_the_triangle_inequality(sliver_patch: tuple) -> None:
    _, _, vertices_wp, faces_wp = sliver_patch
    original = tw.edges.face_edge_lengths(vertices_wp, faces_wp).numpy()
    mollified, delta = tw.laplacian.mollify_intrinsic(vertices_wp, faces_wp)

    assert delta > 0.0

    # Every mollified triangle satisfies the inequality strictly; at least one original did not.
    def worst_slack(lengths: np.ndarray) -> np.ndarray:
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        return np.minimum(np.minimum(a + b - c, b + c - a), c + a - b)

    assert worst_slack(original).min() <= 0.0
    assert worst_slack(mollified.numpy()).min() > 0.0
    # One global constant, added to every length: that is what keeps the operator symmetric. The
    # tolerance is loose because ``delta`` here is ~1e-5 against lengths of ~1, so the sum lands at
    # the edge of float32's resolution.
    assert np.allclose(mollified.numpy() - original, delta, rtol=5e-2, atol=1e-9)
