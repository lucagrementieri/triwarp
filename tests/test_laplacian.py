"""Regression tests for ``triwarp.laplacian`` against igl (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp

import triwarp as tw


def _bsr_to_csr(matrix: wp.sparse.BsrMatrix) -> sp.csr_matrix:
    nrow = int(matrix.nrow)  # pyright: ignore[reportAttributeAccessIssue]
    ncol = int(matrix.ncol)  # pyright: ignore[reportAttributeAccessIssue]
    offsets = matrix.offsets.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    columns = matrix.columns.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    values = matrix.values.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    return sp.csr_matrix((values, columns, offsets), shape=(nrow, ncol))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_cotmatrix_entries(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    cot_entries_igl = igl.cotmatrix_entries(vertices_np, faces_np)
    cot_entries_wp = tw.laplacian.cotmatrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_cotmatrix_entries_intrinsic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    edge_lengths_igl = igl.edge_lengths(vertices_np, faces_np)
    cot_entries_igl = igl.cotmatrix_entries(edge_lengths_igl)
    edge_lengths_wp = wp.array(edge_lengths_igl.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    cot_entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths_wp)

    assert np.allclose(cot_entries_wp.numpy(), cot_entries_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_cotmatrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    laplacian_igl = igl.cotmatrix(vertices_np, faces_np).tocsr()
    laplacian_wp = _bsr_to_csr(tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices))

    assert laplacian_wp.shape == laplacian_igl.shape
    assert np.allclose(laplacian_wp.toarray(), laplacian_igl.toarray(), rtol=1e-5, atol=1e-5)


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
def test_mass_matrix(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    mass_igl = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    mass_wp = tw.laplacian.mass_matrix_entries(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(mass_wp.numpy(), mass_igl, rtol=1e-5, atol=1e-5)


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
