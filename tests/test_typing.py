from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import triwarp.typing as twt


def test_ensure_ndim_rejects_1d(device: str) -> None:
    arr = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(TypeError, match="expected 2D array"):
        twt.ensure_ndim(arr, 2)


def test_as_array2d_accepts_2d(device: str) -> None:
    arr = wp.empty((2, 3), dtype=wp.int32, device=device)
    out = twt.as_array2d(arr, wp.int32)
    assert out.ndim == 2
    assert out.dtype == wp.int32


_INT_WP_TO_NUMPY = (
    (wp.int8, np.int8),
    (wp.uint8, np.uint8),
    (wp.int16, np.int16),
    (wp.uint16, np.uint16),
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
)

_FLOAT_DTYPES = (wp.float16, wp.float32, wp.float64)


def test_dtype_max() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert twt.dtype_max(wp_dt) == np.iinfo(np_dtype).max
    for wp_dt in _FLOAT_DTYPES:
        assert np.isposinf(twt.dtype_max(wp_dt))


def test_dtype_min() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert twt.dtype_min(wp_dt) == np.iinfo(np_dtype).min
    for wp_dt in _FLOAT_DTYPES:
        assert np.isneginf(twt.dtype_min(wp_dt))


def test_empty_2d_shape(device: str) -> None:
    arr: twt.Array2dInt32 = twt.empty_2d((0, 2), wp.int32, device=device)
    assert arr.shape == (0, 2)
    assert arr.ndim == 2


def test_dtype_zero_splits_int_and_float_like_python(device: str) -> None:
    """Integer types give a Python ``int`` and float ones a ``float``, not merely ``== 0``."""
    for dtype_wp, _np_dtype in _INT_WP_TO_NUMPY:
        zero = twt.dtype_zero(dtype_wp)
        assert zero == 0
        assert isinstance(zero, int)
        assert not isinstance(zero, bool)
    for dtype_wp in _FLOAT_DTYPES:
        zero = twt.dtype_zero(dtype_wp)
        assert zero == 0.0
        assert isinstance(zero, float)


@pytest.mark.parametrize("dtype_wp", [wp.float32, wp.bool])
def test_empty_3d_shape(device: str, dtype_wp: type) -> None:
    """The rank-3 allocator over both dtypes its overloads admit, on the caller's device."""
    arr = twt.empty_3d((2, 3, 4), dtype_wp, device=device)
    assert arr.shape == (2, 3, 4)
    assert arr.ndim == 3
    assert arr.dtype == dtype_wp
    assert str(arr.device) == device
