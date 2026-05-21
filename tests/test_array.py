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


def test_pack_1d_wp_arrays(device: str):
    parts = [
        wp.array([1, 2, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([4], dtype=wp.int32, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    assert flat.device == device
    assert offsets.device == device
    assert np.array_equal(flat.numpy(), np.array([1, 2, 3, 4], dtype=np.int32))
    assert np.array_equal(offsets.numpy(), np.array([0, 3, 3], dtype=np.int32))


def test_pack_1d_wp_arrays_vec3(device: str):
    parts = [
        wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device),
        wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    got = flat.numpy().reshape(-1, 3)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)
    assert np.array_equal(offsets.numpy(), np.array([0, 2], dtype=np.int32))


def test_pack_1d_wp_arrays_dtype_mismatch(device: str):
    parts = [
        wp.array([1], dtype=wp.int32, device=device),
        wp.array([2.0], dtype=wp.float32, device=device),
    ]
    with pytest.raises(ValueError, match="same dtype"):
        tw.array.pack_1d_arrays(parts)


def test_sort_rows(device: str):
    rng = np.random.default_rng(42)
    data = rng.random(size=(32, 4), dtype=np.float32)
    sorted_data_np = np.sort(data, axis=1)

    data_wp = wp.array(data, dtype=wp.float32, device=device)
    tw.array.sort_rows(data_wp)
    assert np.array_equal(data_wp.numpy(), sorted_data_np)


@pytest.mark.parametrize("data", (None, np.arange(1, 13, dtype=np.int32)))
def test_index_sparse(data: npt.NDArray[np.int32] | None, device: str):
    n_rows = 4
    indices = np.array([[0, 1, 2], [0, 3, 1], [1, 2, 3], [0, 2, 3]])

    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device) if data is not None else None

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp)
    assert result_wp.values.dtype == (data_wp.dtype if data_wp is not None else wp.float32)
    assert np.array_equal(result_wp.offsets.numpy(), result_np.indptr)
    assert np.array_equal(result_wp.values.numpy(), result_np.data)


def test_index_sparse_repeated_indices(device: str):
    n_rows = 4
    indices = np.array([[0, 1, 0], [3, 3, 1], [1, 2, 3]])
    data = np.ones(9, dtype=np.int32)
    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device)

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp, dtype=wp.float64)
    assert result_wp.values.dtype == wp.float64
    result_csr = scipy.sparse.csr_matrix(
        (result_wp.values.numpy(), result_wp.columns.numpy(), result_wp.offsets.numpy()), shape=result_wp.shape
    )
    assert np.array_equal(result_csr.todense(), result_np.todense())
