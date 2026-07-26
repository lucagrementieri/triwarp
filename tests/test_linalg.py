from __future__ import annotations

import numpy as np
import pytest
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt


def _skip_on_cpu(device: str) -> None:
    # Every solve here goes through ``warp.optim.linear.cg``, which returns NaN on the CPU device
    # in Warp 1.14-1.15; ``triwarp.linalg`` raises NotImplementedError there.
    if wp.get_device(device).is_cpu:
        pytest.skip("warp.optim.linear.cg returns NaN on the CPU device in Warp 1.14-1.15.")


def _spd_system(device: str, n: int = 64, n_rhs: int = 3, seed: int = 11):
    """Build an SPD operator with ``n_rhs`` right-hand sides, plus its NumPy form to solve."""
    rng = np.random.default_rng(seed)
    dense_np = rng.standard_normal((n, n))
    dense_np = dense_np @ dense_np.T + n * np.eye(n)
    rows_np, cols_np = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    matrix_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(cols_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(dense_np.ravel()), dtype=wp.float64, device=device),
    )
    rhs_np = rng.standard_normal((n_rhs, n))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    return matrix_wp, twt.as_array2d_float(rhs_wp, dtype=wp.float64), dense_np, rhs_np


def test_solve_spd_columns_matches_numpy(device: str) -> None:
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, twt.as_array2d_float(solution_wp, dtype=wp.float64)
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("check_every", [1, 25, 0])
def test_solve_spd_columns_check_every_is_solution_invariant(device: str, check_every: int) -> None:
    # ``check_every`` only changes how often the residual is tested (``0`` tests it on device via
    # ``wp.capture_while``), never the system being solved, so every setting converges to the same
    # answer. It is a performance knob only.
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp,
        rhs_wp,
        twt.as_array2d_float(solution_wp, dtype=wp.float64),
        check_every=check_every,
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


def test_spd_column_solver_check_every_reused_across_calls(device: str) -> None:
    # The hoisted functor keeps its ``check_every`` across calls and warm-starts from ``solution``.
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = tw.linalg.spd_column_solver(
        matrix_wp, rhs_wp, twt.as_array2d_float(solution_wp, dtype=wp.float64), check_every=0
    )
    solver()
    solver()
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)
