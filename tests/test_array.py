"""Regression tests for ``triwarp.geometry`` against ``trimesh.geometry`` (CPU reference)."""

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import trimesh as tm
import warp as wp

import triwarp as tw


def test_init_range(device: str) -> None:
    n = 8
    out_wp = tw.array.init_range(n, device)
    assert np.array_equal(out_wp.numpy(), np.arange(n, dtype=np.int32))


def test_init_range_zero(device: str) -> None:
    out_wp = tw.array.init_range(0, device)
    assert out_wp.shape == (0,)


def test_init_range_step(device: str) -> None:
    count, step = 6, 3
    out_wp = tw.array.init_range_step(count, step, device)
    assert np.array_equal(out_wp.numpy(), np.arange(0, count * step, step, dtype=np.int32))


def test_init_range_step_zero_count(device: str) -> None:
    out_wp = tw.array.init_range_step(0, 5, device)
    assert out_wp.shape == (0,)


def test_init_sort_pair_indices(device: str) -> None:
    n, fill = 5, -1
    out_wp = tw.array.init_sort_pair_indices(n, fill, device)
    expected_np = np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1], dtype=np.int32)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_init_sort_pair_indices_zero(device: str) -> None:
    out_wp = tw.array.init_sort_pair_indices(0, -1, device)
    assert out_wp.shape == (0,)


def test_init_repeat_index(device: str) -> None:
    count, repeats = 9, 3
    out_wp = tw.array.init_repeat_index(count, repeats, device)
    expected_np = np.repeat(np.arange(count // repeats, dtype=np.int32), repeats)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_init_repeat_index_zero_count(device: str) -> None:
    out_wp = tw.array.init_repeat_index(0, 4, device)
    assert out_wp.shape == (0,)


def test_append(device: str) -> None:
    arr_wp = wp.array([0, 3, 7], dtype=wp.int32, device=device)
    out_wp = tw.array.append(arr_wp, 12)
    assert np.array_equal(out_wp.numpy(), np.array([0, 3, 7, 12], dtype=np.int32))


def test_append_empty(device: str) -> None:
    arr_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_wp = tw.array.append(arr_wp, 5)
    assert np.array_equal(out_wp.numpy(), np.array([5], dtype=np.int32))


def test_square(device: str) -> None:
    rng = np.random.default_rng(0)
    values_np = (rng.random(64, dtype=np.float32) * 4.0 - 2.0).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    squared_wp = tw.array.square(values_wp)
    assert np.allclose(squared_wp.numpy(), values_np**2, rtol=1e-5, atol=1e-5)


def test_square_empty(device: str) -> None:
    values_wp = wp.empty(0, dtype=wp.float32, device=device)
    squared_wp = tw.array.square(values_wp)
    assert squared_wp.shape == (0,)


def test_concatenate(device: str) -> None:
    parts = [
        wp.array([0, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([7, 12], dtype=wp.int32, device=device),
    ]
    out_wp = tw.array.concatenate(parts)
    assert np.array_equal(out_wp.numpy(), np.array([0, 3, 7, 12], dtype=np.int32))


def test_concatenate_single_returns_input(device: str) -> None:
    arr_wp = wp.array([1, 2], dtype=wp.int32, device=device)
    out_wp = tw.array.concatenate([arr_wp])
    assert out_wp is arr_wp


def test_concatenate_empty_segments(device: str) -> None:
    out_wp = tw.array.concatenate([wp.empty(0, dtype=wp.int32, device=device)])
    assert out_wp.shape == (0,)


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


@pytest.mark.parametrize("data", [None, np.arange(1, 13, dtype=np.int32)])
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


def test_gather_1d(device: str) -> None:
    rng = np.random.default_rng(5)
    values_np = rng.integers(0, 1000, size=32, dtype=np.int32)
    indices_np = rng.integers(0, 32, size=10, dtype=np.int32)

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(values_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), values_np[indices_np])


def test_gather_2d_rows(device: str) -> None:
    rng = np.random.default_rng(6)
    rows_np = rng.integers(0, 1000, size=(20, 2), dtype=np.int32)
    indices_np = rng.integers(0, 20, size=7, dtype=np.int32)

    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)

    assert gathered_wp.shape == (7, 2)
    assert np.array_equal(gathered_wp.numpy(), rows_np[indices_np])


def test_gather_vec3(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((16, 3)).astype(np.float32)
    indices_np = rng.integers(0, 16, size=5, dtype=np.int32)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(points_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), points_np[indices_np])


def test_gather_empty_indices(device: str) -> None:
    rows_wp = wp.array(np.zeros((4, 2), dtype=np.int32), dtype=wp.int32, device=device)
    indices_wp = wp.empty(0, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)
    assert gathered_wp.shape == (0, 2)


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


def test_gram_matrix(device: str) -> None:
    # 200 = 3 * 64 + 8 exercises the multi-tile reduction and remainder path.
    rng = np.random.default_rng(10)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    gram_np = points_np.T @ points_np
    assert np.allclose(tw.array.gram_matrix(points_wp).numpy()[0], gram_np, rtol=1e-4, atol=1e-4)


def test_gram_matrix_empty(device: str) -> None:
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert np.allclose(tw.array.gram_matrix(points_wp).numpy()[0], np.zeros((3, 3)))


def test_centered_covariance(device: str) -> None:
    rng = np.random.default_rng(11)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    centered_np = points_np - points_np.mean(axis=0)
    scatter_np = centered_np.T @ centered_np
    cov_wp = tw.array.centered_covariance(points_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


def test_centered_covariance_precomputed_center(device: str) -> None:
    rng = np.random.default_rng(12)
    points_np = rng.standard_normal((150, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    mean_np = points_np.mean(axis=0)
    center_wp = wp.array(mean_np.reshape(1, 3).astype(np.float32), dtype=wp.vec3, device=device)
    centered_np = points_np - mean_np
    scatter_np = centered_np.T @ centered_np
    cov_wp = tw.array.centered_covariance(points_wp, center=center_wp)
    assert np.allclose(cov_wp.numpy()[0], scatter_np, rtol=1e-4, atol=1e-4)


def test_covariance(device: str) -> None:
    rng = np.random.default_rng(13)
    points_np = rng.standard_normal((200, 3)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    cov_np = np.cov(points_np.T, ddof=1)
    assert np.allclose(tw.array.covariance(points_wp).numpy()[0], cov_np, rtol=1e-4, atol=1e-4)


def test_covariance_too_few_points_raises(device: str) -> None:
    points_wp = wp.zeros(1, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="ddof"):
        tw.array.covariance(points_wp)
