"""
Regression tests for ``triwarp.geometry`` against ``trimesh.geometry`` (CPU reference).
"""

import pytest
import numpy as np
import numpy.typing as npt
import scipy.sparse
import warp as wp

import trimesh as tm
import triwarp as tw


@pytest.mark.parametrize("data", (None, np.arange(1, 10, dtype=np.int32)))
def test_index_sparse(data: npt.NDArray[np.int32] | None, device: str):
    n_rows = 4
    indices = np.array([[0, 1, 2], [0, 3, 1], [1, 2, 3]])

    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device) if data is not None else None

    result_wp = tw.geometry.index_sparse(n_rows, indices_wp, data_wp)
    assert np.array_equal(result_wp.offsets.numpy(), result_np.indptr)
    assert np.array_equal(result_wp.values.numpy(), result_np.data)


def test_index_sparse_repeated_indices(device: str):
    n_rows = 4
    indices = np.array([[0, 1, 0], [3, 3, 1], [1, 2, 3]])
    data = np.ones(9, dtype=np.int32)
    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device) if data is not None else None

    result_wp = tw.geometry.index_sparse(n_rows, indices_wp, data_wp)
    result_csr = scipy.sparse.csr_matrix(
        (result_wp.values.numpy(), result_wp.columns.numpy(), result_wp.offsets.numpy()), shape=result_wp.shape
    )
    assert np.array_equal(result_csr.todense(), result_np.todense())
