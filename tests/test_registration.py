"""Regression tests for ``triwarp.registration`` against ``trimesh.registration`` (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import trimesh.registration as tm_reg
import triwarp as tw


def _make_point_clouds(
    rng: np.random.Generator, n: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    a_np = rng.standard_normal((n, 3)).astype(np.float64)
    # b is a mildly rotated/translated version of a to keep correspondence meaningful
    b_np = rng.standard_normal((n, 3)).astype(np.float64)
    return a_np, b_np


def _to_wp(arr_np: np.ndarray, device: str) -> wp.array:
    return wp.array(arr_np.astype(np.float32), dtype=wp.vec3, device=device)


def _run_both(
    a_np: np.ndarray,
    b_np: np.ndarray,
    device: str,
    weights_np: np.ndarray | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
) -> tuple:
    """Return (matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_tw, cost_tw)."""
    kwargs = dict(reflection=reflection, translation=translation, scale=scale)

    matrix_tm, transformed_tm, cost_tm = tm_reg.procrustes(
        a_np, b_np, weights=weights_np, **kwargs
    )

    a_wp = _to_wp(a_np, device)
    b_wp = _to_wp(b_np, device)
    weights_wp = (
        wp.array(weights_np.astype(np.float32), dtype=wp.float32, device=device)
        if weights_np is not None
        else None
    )

    matrix_wp, transformed_wp, cost_tw = tw.registration.procrustes(
        a_wp, b_wp, weights=weights_wp, **kwargs
    )

    matrix_tw = matrix_wp.numpy()[0]
    return matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw


def test_procrustes_default(device: str) -> None:
    rng = np.random.default_rng(0)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_uniform_weights(device: str) -> None:
    rng = np.random.default_rng(1)
    a_np, b_np = _make_point_clouds(rng)
    n = a_np.shape[0]
    weights_np = np.ones(n, dtype=np.float64)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, weights_np=weights_np
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_binary_weights(device: str) -> None:
    rng = np.random.default_rng(2)
    a_np, b_np = _make_point_clouds(rng)
    n = a_np.shape[0]
    weights_np = np.zeros(n, dtype=np.float64)
    weights_np[: n // 2] = 1.0
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, weights_np=weights_np
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(
        transformed_wp.numpy()[: n // 2], transformed_tm[: n // 2], rtol=1e-4, atol=1e-4
    )
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_reflection(device: str) -> None:
    rng = np.random.default_rng(3)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, reflection=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_translation(device: str) -> None:
    rng = np.random.default_rng(4)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, translation=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_no_scale(device: str) -> None:
    rng = np.random.default_rng(5)
    a_np, b_np = _make_point_clouds(rng)
    matrix_tm, transformed_tm, cost_tm, matrix_tw, transformed_wp, cost_tw = _run_both(
        a_np, b_np, device, scale=False
    )
    assert np.allclose(matrix_tw, matrix_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(transformed_wp.numpy(), transformed_tm, rtol=1e-4, atol=1e-4)
    assert np.allclose(cost_tw, cost_tm, rtol=1e-4, atol=1e-4)


def test_procrustes_return_matrix_only(device: str) -> None:
    rng = np.random.default_rng(6)
    a_np, b_np = _make_point_clouds(rng)
    a_wp = _to_wp(a_np, device)
    b_wp = _to_wp(b_np, device)

    result = tw.registration.procrustes(a_wp, b_wp, return_cost=False)
    assert isinstance(result, wp.array)
    assert result.shape == (1,)
    assert result.dtype == wp.mat44

    matrix_tm, _, _ = tm_reg.procrustes(a_np, b_np)
    assert np.allclose(result.numpy()[0], matrix_tm, rtol=1e-4, atol=1e-4)
