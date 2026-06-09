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
        (result_wp.values.numpy(), result_wp.columns.numpy(), result_wp.offsets.numpy()),
        shape=result_wp.shape,
    )
    assert np.array_equal(result_csr.todense(), result_np.todense())


def test_isin_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    elements_np = rng.integers(0, 20, size=50, dtype=np.int32)
    test_np = rng.choice(20, size=8, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_2d(device: str) -> None:
    rng = np.random.default_rng(7)
    elements_np = rng.integers(0, 15, size=(12, 3), dtype=np.int32)
    test_np = rng.choice(15, size=5, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_empty_test(device: str) -> None:
    elements_np = np.array([0, 1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.empty(0, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_ref_np = np.zeros_like(elements_np, dtype=bool)
    assert np.array_equal(mask_wp.numpy(), mask_ref_np)


def test_isin_sparse_large_indices(device: str) -> None:
    """Forces sort + binary-search path (max index >> len(test_elements))."""
    elements_np = np.array([1, 1_000_000, 2, 999_999, 3], dtype=np.int32)
    test_np = np.array([1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_flatnonzero(device: str) -> None:
    rng = np.random.default_rng(11)
    mask_np = rng.choice([False, True], size=64, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    indices_ref_np = np.flatnonzero(mask_np).astype(np.int32)
    assert np.array_equal(indices_wp.numpy(), indices_ref_np)


def test_flatnonzero_empty(device: str) -> None:
    mask_wp = wp.array(np.zeros(8, dtype=bool), dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    assert indices_wp.shape == (0,)


def test_vector_angle(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 64
    vecs_a_np = rng.standard_normal((n, 3))
    vecs_a_np /= np.linalg.norm(vecs_a_np, axis=1, keepdims=True)
    vecs_b_np = rng.standard_normal((n, 3))
    vecs_b_np /= np.linalg.norm(vecs_b_np, axis=1, keepdims=True)

    pairs_np = np.stack([vecs_a_np, vecs_b_np], axis=1)
    angles_tm = tm.geometry.vector_angle(pairs_np)

    vecs_a_wp = wp.array(vecs_a_np.astype(np.float32), dtype=wp.vec3, device=device)
    vecs_b_wp = wp.array(vecs_b_np.astype(np.float32), dtype=wp.vec3, device=device)
    angles_wp = tw.array.vector_angle(vecs_a_wp, vecs_b_wp)
    assert np.allclose(angles_wp.numpy(), angles_tm, rtol=1e-5, atol=1e-5)


def test_vector_angle_empty(device: str) -> None:
    vecs_a_wp = wp.empty(0, dtype=wp.vec3, device=device)
    vecs_b_wp = wp.empty(0, dtype=wp.vec3, device=device)
    angles_wp = tw.array.vector_angle(vecs_a_wp, vecs_b_wp)
    assert angles_wp.shape == (0,)
