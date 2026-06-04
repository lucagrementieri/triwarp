from __future__ import annotations

import pytest
import warp as wp

import triwarp.typing as twt


def test_ensure_ndim_rejects_1d(device: str) -> None:
    arr = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(TypeError, match="expected 2D array"):
        twt.ensure_ndim(arr, 2)


def test_as_array2d_int32_accepts_2d(device: str) -> None:
    arr = wp.empty((2, 3), dtype=wp.int32, device=device)
    out = twt.as_array2d_int32(arr)
    assert out.ndim == 2
    assert out.dtype == wp.int32


def test_empty_int32_2d_shape(device: str) -> None:
    arr: twt.Array2dInt32 = twt.empty_int32_2d((0, 2), device=device)
    assert arr.shape == (0, 2)
    assert arr.ndim == 2
