from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels.grouping import VEC3_PACK_PRECISION, VEC3_PACK_SHIFT

group_test_data = (
    (wp.array([1, 3, 2, 3, 4, 4, 7, 5, -1, 5, 5], dtype=wp.int32), 2, wp.array([[1, 3], [4, 5]])),
    (
        wp.array([0, 1, 2, 1, 5, 6, 1, 0, 0, 0, 6, 4, 6], dtype=wp.uint64),
        3,
        wp.array([[1, 3, 6], [5, 10, 12]]),
    ),
    (
        wp.array([-1, 3, 2, -3, 4, 2, -1, 2, 2, 2], dtype=wp.int64),
        4,
        wp.empty((0, 4), dtype=wp.int32),
    ),
)


@pytest.mark.parametrize("values, length, expected", group_test_data)
def test_group(
    device: str, values: wp.array[wp.Int], length: int, expected: twt.Array2dInt32
) -> None:
    values_wp = wp.array(values, dtype=values.dtype, device=device)
    groups_wp = tw.grouping.group(values_wp, length)
    assert np.array_equal(groups_wp.numpy(), expected.numpy())


def test_group_int_rows(device: str) -> None:
    data_np = np.array([[1, 2], [3, 4], [1, 2], [2, 1], [3, 4], [0, 1], [3, 4]], dtype=np.int32)
    length = 2
    groups_np = np.sort(tm.grouping.group_rows(data_np, require_count=length), axis=1)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    groups_wp = tw.grouping.group_int_rows(data_wp, length)
    assert np.array_equal(np.sort(groups_wp.numpy(), axis=1), groups_np)


def test_hash_vector_rows(device: str) -> None:
    rng = np.random.default_rng(17)
    n = 256
    vectors_np = rng.standard_normal((n, 3), dtype=np.float64)
    packed_np = _pack_vec3_np(vectors_np)

    vectors_wp = wp.array(vectors_np, dtype=wp.vec3, device=device)
    packed_wp = tw.grouping.hash_vector_rows(vectors_wp)
    packed = packed_wp.numpy()

    assert np.array_equal(packed, packed_np)

    vectors_double_wp = wp.array(vectors_np, dtype=wp.vec3d, device=device)
    with pytest.raises(ValueError, match="data must be a wp.array\\[wp.vec3\\]"):
        _ = tw.grouping.hash_vector_rows(vectors_double_wp)


def test_hash_indices_rows_valid(device: str) -> None:
    rng = np.random.default_rng(23)
    n_rows, n_cols = 64, 5
    actual_max_index = 17
    max_index = actual_max_index + 3
    indices_np = rng.integers(0, actual_max_index, size=(n_rows, n_cols), dtype=np.int32)
    packed_np = _pack_indices_rows_np(indices_np, max_index)
    packed_default_np = _pack_indices_rows_np(indices_np)

    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    packed_wp = tw.grouping.hash_indices_rows(indices_wp, max_index=max_index)
    packed_default_wp = tw.grouping.hash_indices_rows(indices_wp)
    assert np.array_equal(packed_wp.numpy(), packed_np)
    assert np.array_equal(packed_default_wp.numpy(), packed_default_np)


def test_hash_indices_rows_invalid(device: str) -> None:
    max_index = 8
    indices_wp = wp.array([[0, 1, 2], [-3, 4, 1]], dtype=wp.int32, device=device)

    with pytest.raises(ValueError, match="data must be non-negative, got a minimum of -3"):
        _ = tw.grouping.hash_indices_rows(indices_wp, max_index=max_index)

    indices_oob = wp.array([[0, 1, 2], [3, 8, 1]], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="data must be less than max_index 8, got a maximum of 8"):
        _ = tw.grouping.hash_indices_rows(indices_oob, max_index=max_index)

    with pytest.raises(ValueError, match="max_index must be positive, got 0"):
        _ = tw.grouping.hash_indices_rows(indices_wp, max_index=0)

    indices_ok = wp.array([[0, 1, 2], [3, 4, 5]], dtype=wp.int32, device=device)
    indices_np_ok = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    packed_np = _pack_indices_rows_np(indices_np_ok, max_index)
    packed_wp = tw.grouping.hash_indices_rows(indices_ok, max_index=max_index)
    assert np.array_equal(packed_wp.numpy(), packed_np)


def _pack_vec3_np(vectors_np: np.ndarray) -> np.ndarray:
    if vectors_np.dtype != np.float32:
        vectors_np = vectors_np.astype(np.float32)
    bits = vectors_np.view(np.uint32)
    ix = bits[:, 0].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iy = bits[:, 1].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iz = bits[:, 2].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    return ix | (iy << VEC3_PACK_PRECISION.value) | (iz << (2 * VEC3_PACK_PRECISION.value))


def _pack_indices_rows_np(indices_np: np.ndarray, max_index: int | None = None) -> np.ndarray:
    """CPU reference for ``pack_indices``: mixed-radix sum with wrapping ``uint64`` math."""
    if max_index is None:
        max_index = np.max(indices_np) + 1
    return np.sum(indices_np * np.power(max_index, np.arange(indices_np.shape[1])), axis=1)
