from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import triwarp as tw


def test_unique_1d(device: str):
    data_np = np.array([0, 1, 20, 3, 1, 3, 10, 20], dtype=np.int32)
    unique_np = np.unique(data_np)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    unique_wp = tw.unique.unique_1d(data_wp)
    assert np.array_equal(unique_wp.numpy(), unique_np)


def test_unique_1d_counts(device: str):
    data_np = np.array([20.0, 10.0, 2.0, 3.0, 1.0, 3.0, 10.0, 20.0], dtype=np.float32)
    unique_np, counts_np = np.unique(data_np, return_counts=True)

    data_wp = wp.array(data_np, dtype=wp.float32, device=device)
    unique_wp, counts_wp = tw.unique.unique_1d(data_wp, return_counts=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(counts_wp.numpy(), counts_np)


def test_unique_1d_inverse(device: str):
    data_np = np.array([20, 10, 2, 3, 1, 3, 10, 20], dtype=np.int64)
    unique_np, inverse_np = np.unique(data_np, return_inverse=True)

    data_wp = wp.array(data_np, dtype=wp.int64, device=device)
    unique_wp, inverse_wp = tw.unique.unique_1d(data_wp, return_inverse=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np)


def test_unique_1d_inverse_counts(device: str):
    data_np = np.array([20, 10, 20, 3, 1, 3, 10, 20], dtype=np.uint64)
    unique_np, inverse_np, counts_np = np.unique(data_np, return_inverse=True, return_counts=True)

    data_wp = wp.array(data_np, dtype=wp.uint64, device=device)
    unique_wp, inverse_wp, counts_wp = tw.unique.unique_1d(data_wp, return_inverse=True, return_counts=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np)
    assert np.array_equal(counts_wp.numpy(), counts_np)


reinterpret_cast_test_data = (
    wp.array([-128, -127, -1, 0, 1, 127], dtype=wp.int8),
    wp.array([255, 0, 254, 1], dtype=wp.uint8),
    wp.array([-32768, -32767, 10, 0, -1, 32767, 32765], dtype=wp.int16),
    wp.array([65535, 32, 65534, 0, 1], dtype=wp.uint16),
    wp.array([-2147483648, -2147483647, 10, 0, -1, 2147483647, 2147483645], dtype=wp.int32),
    wp.array([4294967295, 32, 4294967294, 0, 1], dtype=wp.uint32),
    wp.array(
        [-9223372036854775808, -9223372036854775807, 10, 0, -1, 9223372036854775806, 9223372036854775807],
        dtype=wp.int64,
    ),
    wp.array([18446744073709551615, 65535, 18446744073709551614, 131070, 1], dtype=wp.uint64),
    wp.array(
        np.asarray(
            [
                0x0000,  # +0
                0x8000,  # -0
                0x7F80,  # +inf
                0xFF80,  # -inf
                0x7FFF,  # quiet NaN (preserved through float32 widen/narrow on CUDA)
                0x7F7F,  # largest finite
                0xFF7F,  # smallest (most negative) finite
                0x0080,  # smallest subnormal
                0x8100,  # negative subnormal
            ],
            dtype=np.uint16,
        ),
        dtype=wp.bfloat16,
    ),
    wp.array(
        [
            np.finfo(np.float16).min,
            np.finfo(np.float16).max,
            -np.finfo(np.float16).max,
            np.finfo(np.float16).smallest_subnormal,
            -np.finfo(np.float16).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float16,
    ),
    wp.array(
        [
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).smallest_subnormal,
            -np.finfo(np.float32).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float32,
    ),
    wp.array(
        [
            np.finfo(np.float64).min,
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            np.finfo(np.float64).smallest_subnormal,
            -np.finfo(np.float64).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float64,
    ),
)


@pytest.mark.parametrize(
    "data",
    reinterpret_cast_test_data,
    ids=[a.dtype.__name__ for a in reinterpret_cast_test_data],
)
def test_reinterpret_cast_int_reciprocity(device: str, data: wp.array[wp.Scalar]):
    as_int = tw.unique.reinterpret_cast_to_int(data.to(device))
    recovered = tw.unique.reinterpret_cast_from_int(as_int, data.dtype)
    assert np.array_equal(data.numpy(), recovered.numpy(), equal_nan=True)
