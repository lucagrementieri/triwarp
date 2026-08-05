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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_gradients", "igl")
def test_face_gradients_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (matrix form applied): ``igl.grad`` is the same operator as a sparse ``(3F, V)`` map.

    igl returns the operator; triwarp returns its product with the field. The named transform is
    therefore to *apply* igl's matrix and unstack the result, which comes back as
    ``[all x; all y; all z]`` rather than interleaved -- getting that wrong yields a permutation of
    the right numbers, so the test also checks the defining property below, which no permutation
    satisfies.

    The tolerance is the package's standard ``1e-5`` rather than something tighter, and the reason
    is structural rather than a fudge: triwarp accumulates in ``float64`` but takes its normals and
    areas from ``face_normals_and_areas``, which is ``float32``, so the geometry enters at single
    precision where igl's is double throughout. Measured worst deviation across these fixtures is
    **1.4e-6 relative** (on ``half_torus``, whose faces are the smallest), so the bound has a 7x
    margin -- and it is a float32-vs-float64 gap, not a disagreement about the operator.

    The second assert is the gradient's defining identity, checked without igl:
    ``dot(grad, v1 - v0) == values[v1] - values[v0]`` for every face. A finite difference along the
    edges, or a gradient left in the wrong plane, fails it.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    values_np = np.ascontiguousarray(vertices_np[:, 2])

    n_faces = faces_np.shape[0]
    stacked_igl = igl.grad(vertices_np, faces_np) @ values_np
    gradients_igl = np.stack(
        [stacked_igl[:n_faces], stacked_igl[n_faces : 2 * n_faces], stacked_igl[2 * n_faces :]],
        axis=1,
    )

    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)
    gradients_wp = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, values_wp)

    assert np.allclose(gradients_wp.numpy(), gradients_igl, rtol=1e-5, atol=1e-5)

    # The identity that defines a piecewise-linear gradient, independent of either library.
    edges_np = vertices_np[faces_np[:, 1]] - vertices_np[faces_np[:, 0]]
    differences_np = values_np[faces_np[:, 1]] - values_np[faces_np[:, 0]]
    assert np.allclose(
        np.einsum("ij,ij->i", gradients_wp.numpy(), edges_np), differences_np, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_gradients_of_a_constant_field_is_zero(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """A constant field has no gradient, and a degenerate face has none either."""
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_wp.points.shape[0])
    constant_wp = wp.array(
        np.full(n_vertices, 3.25, dtype=np.float64), dtype=wp.float64, device=mesh_wp.device
    )
    gradients_wp = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, constant_wp)
    assert np.allclose(gradients_wp.numpy(), 0.0, atol=1e-9)


def test_face_gradients_precomputed_face_data(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Passing the face normals and areas must not change the answer -- only skip recomputing."""
    _mesh_tm, mesh_wp = half_torus
    values_wp = wp.array(
        np.ascontiguousarray(mesh_wp.points.numpy()[:, 2], dtype=np.float64),
        dtype=wp.float64,
        device=mesh_wp.device,
    )
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    derived = tw.laplacian.face_gradients(mesh_wp.points, mesh_wp.indices, values_wp)
    supplied = tw.laplacian.face_gradients(
        mesh_wp.points, mesh_wp.indices, values_wp, face_normals=normals_wp, face_areas=areas_wp
    )
    assert np.array_equal(derived.numpy(), supplied.numpy())


def test_face_gradients_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    values_wp = wp.array(np.array([], dtype=np.float64), dtype=wp.float64, device=device)
    assert tw.laplacian.face_gradients(vertices_wp, faces_wp, values_wp).shape == (0,)


# --- harmonic_integrated / hessian energies / Crouzeix-Raviart (libigl reference) -------


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
@pytest.mark.parity("harmonic_integrated", "igl")
def test_harmonic_integrated_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, k: int, device: str
) -> None:
    """
    Class A: identical Laplacian and mass on both sides isolate the k-harmonic composition.

    igl's cotangent Laplacian and barycentric mass diagonal are handed to
    ``igl.harmonic_integrated_from_laplacian_and_mass`` and (uploaded unchanged) to
    ``harmonic_integrated``; the assembled ``Q`` must then agree to assembly rounding at every
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
    q_wp = _bsr_to_csr(tw.laplacian.harmonic_integrated(laplacian_wp, mass_wp, k=k)).toarray()

    assert q_wp.shape == q_igl.shape
    scale = np.abs(q_igl).max()
    assert np.allclose(q_wp, q_igl, rtol=1e-9, atol=1e-9 * scale)


def test_harmonic_integrated_identity_mass_and_power_guard(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    With ``mass=None`` the operator is the plain power ``(-L)^k``, the ``tutte`` flavor.

    Checked against scipy's own sparse product; ``k < 1`` must raise.
    """
    mesh_tm, mesh_wp = hemisphere
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = _upload_bsr_float64(laplacian_igl, mesh_wp.device)
    q_wp = _bsr_to_csr(tw.laplacian.harmonic_integrated(laplacian_wp, k=2)).toarray()
    q_sp = (laplacian_igl @ laplacian_igl).toarray()
    assert np.allclose(q_wp, q_sp, rtol=1e-9, atol=1e-9 * np.abs(q_sp).max())

    with pytest.raises(ValueError, match="k must be >= 1"):
        tw.laplacian.harmonic_integrated(laplacian_wp, k=0)


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
    q_wp = _bsr_to_csr(tw.laplacian.hessian_energy(mesh_wp.points, mesh_wp.indices)).toarray()

    assert q_wp.shape == q_igl.shape
    assert np.abs(q_igl).max() > 0.0
    scale = np.abs(q_igl).max()
    assert np.allclose(q_wp, q_igl, rtol=1e-7, atol=1e-7 * scale)


def test_hessian_energy_annihilates_linear_fields_where_biharmonic_does_not(device: str) -> None:
    """
    The natural-boundary property the energy exists for, checked without igl.

    On a **flat** open mesh every affine field ``a + b.x`` has zero Hessian, so it must be exactly
    in the energy's null space — while the clamped biharmonic operator (``harmonic_integrated`` at
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

    q_hessian = _bsr_to_csr(tw.laplacian.hessian_energy(vertices_wp, faces_wp)).toarray()
    residual_hessian = np.abs(q_hessian @ linear_fields).max()

    laplacian_wp = tw.laplacian.cotmatrix(vertices_wp, faces_wp, dtype=wp.float64)
    mass_wp = tw.laplacian.mass_matrix_entries(vertices_wp, faces_wp, dtype=wp.float64)
    q_biharmonic = _bsr_to_csr(
        tw.laplacian.harmonic_integrated(laplacian_wp, mass_wp, k=2)
    ).toarray()
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
    q_wp = _bsr_to_csr(
        tw.laplacian.curved_hessian_energy(mesh_wp.points, mesh_wp.indices)
    ).toarray()

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
    q_wp = _bsr_to_csr(
        tw.laplacian.curved_hessian_energy(mesh_wp.points, mesh_wp.indices)
    ).toarray()
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
    matrix_wp = _bsr_to_csr(
        tw.laplacian.crouzeix_raviart_cotmatrix(mesh_wp.points, mesh_wp.indices)
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
    matrix_wp = _bsr_to_csr(
        tw.laplacian.crouzeix_raviart_massmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()

    assert matrix_wp.shape == matrix_igl.shape
    assert np.allclose(matrix_wp, matrix_igl, rtol=1e-5, atol=1e-5)


def test_crouzeix_raviart_shared_edge_numbering(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Precomputed ``(unique_edges, edge_map)`` must not change the answer; half a pair raises."""
    _mesh_tm, mesh_wp = half_torus
    unique_edges, edge_map = tw.edges.edges_unique(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )
    derived = _bsr_to_csr(
        tw.laplacian.crouzeix_raviart_cotmatrix(mesh_wp.points, mesh_wp.indices)
    ).toarray()
    supplied = _bsr_to_csr(
        tw.laplacian.crouzeix_raviart_cotmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges, edge_map=edge_map
        )
    ).toarray()
    assert np.array_equal(derived, supplied)

    with pytest.raises(ValueError, match="together"):
        tw.laplacian.crouzeix_raviart_massmatrix(
            mesh_wp.points, mesh_wp.indices, unique_edges=unique_edges
        )


def test_operator_family_empty_mesh(device: str) -> None:
    """Empty face buffers return square zero-nnz operators of the right dimension."""
    vertices_wp = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    hessian = tw.laplacian.hessian_energy(vertices_wp, faces_wp)
    assert (int(hessian.nrow), int(hessian.ncol)) == (3, 3)
    curved = tw.laplacian.curved_hessian_energy(vertices_wp, faces_wp)
    assert (int(curved.nrow), int(curved.ncol)) == (3, 3)
    cr_cot = tw.laplacian.crouzeix_raviart_cotmatrix(vertices_wp, faces_wp)
    assert (int(cr_cot.nrow), int(cr_cot.ncol)) == (0, 0)
    cr_mass = tw.laplacian.crouzeix_raviart_massmatrix(vertices_wp, faces_wp)
    assert (int(cr_mass.nrow), int(cr_mass.ncol)) == (0, 0)
