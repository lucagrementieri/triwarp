from __future__ import annotations

import numpy as np
import warp as wp

import triwarp.reduce as tw_reduce

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


def test_max_for_dtype() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert tw_reduce.max_for_dtype(wp_dt) == np.iinfo(np_dtype).max
    for wp_dt in _FLOAT_DTYPES:
        assert np.isposinf(tw_reduce.max_for_dtype(wp_dt))


def test_min_for_dtype() -> None:
    for wp_dt, np_ic in _INT_WP_TO_NUMPY:
        np_dtype = np.dtype(np_ic)
        assert tw_reduce.min_for_dtype(wp_dt) == np.iinfo(np_dtype).min
    for wp_dt in _FLOAT_DTYPES:
        assert np.isneginf(tw_reduce.min_for_dtype(wp_dt))


def test_min_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    min_np = values_np.min()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    min_wp = tw_reduce.min(values_wp)
    assert np.allclose(min_wp, min_np)


def test_min_2d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 200
    m = 100
    values_np = rng.standard_normal((n, m), dtype=np.float32)
    min_np = values_np.min()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    min_wp = tw_reduce.min(values_wp)
    assert np.allclose(min_wp, min_np)


def test_max_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    max_np = values_np.max()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    max_wp = tw_reduce.max(values_wp)
    assert np.allclose(max_wp, max_np)


def test_max_2d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 200
    m = 100
    values_np = rng.standard_normal((n, m), dtype=np.float32)
    max_np = values_np.max()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    max_wp = tw_reduce.max(values_wp)
    assert np.allclose(max_wp, max_np)


def test_minmax_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    min_np = values_np.min()
    max_np = values_np.max()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    min_wp, max_wp = tw_reduce.minmax(values_wp)
    assert np.allclose(min_wp, min_np)
    assert np.allclose(max_wp, max_np)


def test_minmax_2d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 200
    m = 100
    values_np = rng.standard_normal((n, m), dtype=np.float32)
    min_np = values_np.min()
    max_np = values_np.max()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    min_wp, max_wp = tw_reduce.minmax(values_wp)
    assert np.allclose(min_wp, min_np)
    assert np.allclose(max_wp, max_np)
