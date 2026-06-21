from __future__ import annotations

import numpy as np
import pytest
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


@pytest.mark.parametrize("axis", [0, 1])
def test_min_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got_wp = tw_reduce.min(values_wp, axis=axis)
    exp_np = values_np.min(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


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


@pytest.mark.parametrize("axis", [0, 1])
def test_max_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got_wp = tw_reduce.max(values_wp, axis=axis)
    exp_np = values_np.max(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


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


@pytest.mark.parametrize("axis", [0, 1])
def test_minmax_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.standard_normal((32, 10), dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    got_min_wp, got_max_wp = tw_reduce.minmax(values_wp, axis=axis)
    exp_min_np = values_np.min(axis=axis)
    exp_max_np = values_np.max(axis=axis)
    assert np.allclose(got_min_wp.numpy(), exp_min_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(got_max_wp.numpy(), exp_max_np, rtol=1e-5, atol=1e-5)


def test_any_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    any_wp = tw_reduce.any(mask_wp)
    any_ref_np = np.any(mask_np)
    assert any_wp == any_ref_np


def test_all_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    all_wp = tw_reduce.all(mask_wp)
    all_ref_np = np.all(mask_np)
    assert all_wp == all_ref_np


@pytest.mark.parametrize("axis", [0, 1])
def test_any_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    any_wp = tw_reduce.any(mask_wp, axis=axis)
    any_ref_np = np.any(mask_np, axis=axis)
    assert np.array_equal(any_wp.numpy(), any_ref_np)


def test_any_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        got = tw_reduce.any(mask_wp, axis=None)
        exp = bool(np.any(mask_np))
        assert got == exp, f"any global mismatch: got {got}, exp {exp}"


@pytest.mark.parametrize("axis", [0, 1])
def test_all_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    all_wp = tw_reduce.all(mask_wp, axis=axis)
    all_ref_np = np.all(mask_np, axis=axis)
    assert np.array_equal(all_wp.numpy(), all_ref_np)


def test_all_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        got = tw_reduce.all(mask_wp, axis=None)
        exp = bool(np.all(mask_np))
        assert got == exp, f"all global mismatch: got {got}, exp {exp}"


def test_scalar_reduce_1d_axis_raises(device: str) -> None:
    values_wp = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    for fn in (tw_reduce.min, tw_reduce.max, tw_reduce.minmax, tw_reduce.sum):
        with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
            fn(values_wp, axis=0)


def test_sum_bool_1d_axis_raises(device: str) -> None:
    mask_wp = wp.array([True, False, True], dtype=wp.bool, device=device)
    with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
        tw_reduce.sum(mask_wp, axis=0)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_min_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got = tw_reduce.min(values_wp)
    assert np.allclose(got, values_np.min())


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_max_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.standard_normal(shape).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    got = tw_reduce.max(values_wp)
    assert np.allclose(got, values_np.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_minmax_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.standard_normal(shape).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    got_min, got_max = tw_reduce.minmax(values_wp)
    assert np.allclose(got_min, values_np.min(), rtol=1e-5, atol=1e-5)
    assert np.allclose(got_max, values_np.max(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_any_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert tw_reduce.any(mask_wp) == bool(np.any(mask_np))


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_all_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert tw_reduce.all(mask_wp) == bool(np.all(mask_np))


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_scalar_reduce_partial_tiles_axis(device: str, shape: tuple[int, int], axis: int) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    assert np.array_equal(tw_reduce.min(values_wp, axis=axis).numpy(), values_np.min(axis=axis))
    assert np.array_equal(tw_reduce.max(values_wp, axis=axis).numpy(), values_np.max(axis=axis))
    got_min, got_max = tw_reduce.minmax(values_wp, axis=axis)
    assert np.array_equal(got_min.numpy(), values_np.min(axis=axis))
    assert np.array_equal(got_max.numpy(), values_np.max(axis=axis))
    assert np.array_equal(tw_reduce.sum(values_wp, axis=axis).numpy(), values_np.sum(axis=axis))


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_bool_reduce_partial_tiles_axis(device: str, shape: tuple[int, int], axis: int) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.array_equal(tw_reduce.any(mask_wp, axis=axis).numpy(), np.any(mask_np, axis=axis))
    assert np.array_equal(tw_reduce.all(mask_wp, axis=axis).numpy(), np.all(mask_np, axis=axis))


def test_sum_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.integers(-1000, 1000, (n,), dtype=np.int32)
    sum_np = values_np.sum()

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    sum_wp = tw_reduce.sum(values_wp)
    assert np.allclose(sum_wp, sum_np)


def test_sum_2d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 200
    m = 100
    values_np = rng.standard_normal((n, m), dtype=np.float32)
    sum_np = values_np.sum()
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    sum_wp = tw_reduce.sum(values_wp)
    assert np.allclose(sum_wp, sum_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_sum_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got_wp = tw_reduce.sum(values_wp, axis=axis)
    exp_np = values_np.sum(axis=axis)
    assert np.array_equal(got_wp.numpy(), exp_np)


def test_sum_bool_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(100,), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    sum_wp = tw_reduce.sum(mask_wp)
    sum_np = int(mask_np.sum())
    assert sum_wp == sum_np


@pytest.mark.parametrize("axis", [0, 1])
def test_sum_bool_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    got_wp = tw_reduce.sum(mask_wp, axis=axis)
    exp_np = mask_np.sum(axis=axis).astype(np.int32)
    assert np.array_equal(got_wp.numpy(), exp_np)


def test_sum_bool_2d_global(device: str) -> None:
    rng = np.random.default_rng(42)
    for mask_np in [
        rng.choice([False, True], size=(32, 4), replace=True),
        np.zeros((32, 4), dtype=bool),
        np.ones((32, 4), dtype=bool),
    ]:
        mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
        got = tw_reduce.sum(mask_wp, axis=None)
        exp = int(mask_np.sum())
        assert got == exp, f"sum global mismatch: got {got}, exp {exp}"


def test_weighted_sum_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    n = 100
    values_np = rng.standard_normal(n, dtype=np.float32)
    weights_np = rng.random(n, dtype=np.float32)
    exp_np = float(np.sum(values_np * weights_np))

    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    got_wp = tw_reduce.weighted_sum(values_wp, weights_wp)
    assert np.allclose(got_wp, exp_np, rtol=1e-5, atol=1e-5)


def test_weighted_sum_length_mismatch_raises(device: str) -> None:
    values_wp = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
    weights_wp = wp.array([1.0], dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="equal length"):
        tw_reduce.weighted_sum(values_wp, weights_wp)


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_sum_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    got = tw_reduce.sum(values_wp)
    assert np.allclose(got, values_np.sum())


@pytest.mark.parametrize("shape", [(65,), (9, 9), (65, 10)])
def test_sum_bool_partial_tiles(device: str, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(99)
    mask_np = rng.choice([False, True], size=shape, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert tw_reduce.sum(mask_wp) == int(mask_np.sum())


@pytest.mark.parametrize(
    ("shape", "axis"), [((9, 9), 0), ((9, 9), 1), ((65, 10), 0), ((65, 10), 1)]
)
def test_sum_partial_tiles_axis(device: str, shape: tuple[int, int], axis: int) -> None:
    rng = np.random.default_rng(99)
    values_np = rng.integers(-1000, 1000, shape, dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    assert np.array_equal(tw_reduce.sum(values_wp, axis=axis).numpy(), values_np.sum(axis=axis))


def test_mean_1d_float(device: str) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.standard_normal(100, dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = tw_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


def test_mean_1d_int(device: str) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (100,), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    mean_wp = tw_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


def test_mean_2d(device: str) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.standard_normal((200, 100), dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = tw_reduce.mean(values_wp)
    assert np.allclose(mean_wp, values_np.mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.standard_normal((32, 10), dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=device)
    mean_wp = tw_reduce.mean(values_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), values_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_2d_axis_int(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    values_np = rng.integers(-1000, 1000, (32, 10), dtype=np.int32)
    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    mean_wp = tw_reduce.mean(values_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), values_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


def test_mean_bool_global(device: str) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.allclose(tw_reduce.mean(mask_wp), mask_np.mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_bool_2d_axis(device: str, axis: int) -> None:
    rng = np.random.default_rng(42)
    mask_np = rng.choice([False, True], size=(32, 4), replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    mean_wp = tw_reduce.mean(mask_wp, axis=axis)
    assert np.allclose(mean_wp.numpy(), mask_np.mean(axis=axis), rtol=1e-5, atol=1e-5)


def test_mean_1d_axis_raises(device: str) -> None:
    values_wp = wp.array([1, 2, 3], dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="requires axis=None for a 1D array"):
        tw_reduce.mean(values_wp, axis=0)


def test_mean_vec3_1d(device: str) -> None:
    rng = np.random.default_rng(20)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.vec3, device=device)
    mean_wp = tw_reduce.mean(values_wp)
    assert np.allclose(np.array(mean_wp), values_np.mean(axis=0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [63, 64, 65, 197])
def test_mean_vec3_partial_tiles(device: str, n: int) -> None:
    rng = np.random.default_rng(n)
    values_np = rng.standard_normal((n, 3)).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.vec3, device=device)
    mean_wp = tw_reduce.mean(values_wp)
    assert np.allclose(np.array(mean_wp), values_np.mean(axis=0), rtol=1e-4, atol=1e-4)


def test_mean_vec3_axis_raises(device: str) -> None:
    values_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="axis"):
        tw_reduce.mean(values_wp, axis=0)


def test_mean_vec3_empty_raises(device: str) -> None:
    values_wp = wp.empty(0, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="non-empty"):
        tw_reduce.mean(values_wp)


def test_sum_vec3_1d(device: str) -> None:
    rng = np.random.default_rng(20)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.vec3, device=device)
    sum_wp = tw_reduce.sum(values_wp)
    assert np.allclose(np.array(sum_wp), values_np.sum(axis=0), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [63, 64, 65, 197])
def test_sum_vec3_partial_tiles(device: str, n: int) -> None:
    rng = np.random.default_rng(n)
    values_np = rng.standard_normal((n, 3)).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.vec3, device=device)
    sum_wp = tw_reduce.sum(values_wp)
    assert np.allclose(np.array(sum_wp), values_np.sum(axis=0), rtol=1e-4, atol=1e-4)


def test_sum_vec3_axis_raises(device: str) -> None:
    values_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="axis"):
        tw_reduce.sum(values_wp, axis=0)


def test_sum_vec3_empty_raises(device: str) -> None:
    values_wp = wp.empty(0, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="non-empty"):
        tw_reduce.sum(values_wp)


def test_weighted_sum_vec3_1d(device: str) -> None:
    rng = np.random.default_rng(21)
    values_np = rng.standard_normal((300, 3)).astype(np.float32)
    weights_np = rng.random(300, dtype=np.float32)
    values_wp = wp.array(values_np, dtype=wp.vec3, device=device)
    weights_wp = wp.array(weights_np, dtype=wp.float32, device=device)
    sum_wp = tw_reduce.weighted_sum(values_wp, weights_wp)
    exp_np = (weights_np[:, None] * values_np).sum(axis=0)
    assert np.allclose(np.array(sum_wp), exp_np, rtol=1e-4, atol=1e-4)
