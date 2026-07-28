"""
Benchmarks for ``triwarp.linalg``: the conjugate-gradient solvers behind every smoother and map.

Axis: **quality**. This module has no size story worth telling -- ``nnz`` sets the cost of one
mat-vec and that is arithmetic -- but it has a conditioning story, and conditioning is the whole
reason an iterative solver is unpredictable. ``saddle`` and ``saddle_graded`` are the same mesh with
the same connectivity, the same ``nnz`` and the same right-hand side; only the spacing along one
axis differs, pushing the worst triangle aspect ratio from 1.6 to 4 719. Everything these groups
measure beyond a fixed per-iteration cost is the iteration count that difference buys.

Three knobs are swept on top of it, each isolating a different lever:

* **fixed fraction** -- ``min_quad_with_fixed`` extracts the free-free block, whose ``nnz`` goes as
  ``(n_free / n)^2``, and a larger fixed set is also a better-conditioned system. Pinning 1% of the
  vertices against 50% moves both at once, which is what a caller actually chooses between. The
  returned free-degree-of-freedom count is asserted on, so a run that accidentally pins everything
  -- and therefore skips the solve entirely -- fails rather than reporting a suspiciously good
  time. It is *not* an iteration count: ``min_quad_with_fixed`` returns
  ``(solution, free_map, n_free)`` and never surfaces one.
* **check_every** -- how often ``solve_spd_columns`` tests the residual. ``1`` checks every
  iteration with a host sync each time; ``0`` (now the default) tests every iteration on device via
  ``wp.capture_while``, with no readback at all. Skipping checks trades syncs for possibly-wasted
  iterations, and which side wins is a measurement, not a derivation -- read this sweep together
  with the mesh pair, because on a well-conditioned system the syncs dominate and on a badly
  conditioned one they should not. It is this group that flipped the default: against the former
  ``10`` the on-device row measured 22.8 vs 31.7 ms on ``saddle`` and 104.7 vs 154.1 ms on
  ``saddle_graded``, 28-32 % on both.
* **repeat count** -- ``spd_column_solver`` preallocates its temporaries and batch layout for reuse,
  which is what ``parametrization.arap`` builds outside its iteration loop. One solve against fifty
  is the amortization question: if ``x50`` lands near 50x ``once``, the preallocation is not earning
  its API surface.

Everything runs in **float64** (the operator dtype these entry points require) and is
**CUDA-only**: ``warp.optim.linear.cg`` returns NaN on the Warp CPU backend in 1.14-1.15 and
``triwarp.linalg`` raises ``NotImplementedError`` there, so the ``triwarp-cpu`` variant is skipped.

References
----------
None. These are the raw solver entry points over a ``warp.sparse`` BSR operator, and no reference in
the test group exposes an equivalent: scipy's ``cg`` would be a host solver over a scipy matrix
(timing SciPy's sparse layer, not this one), libigl solves only through its own direct
factorizations with no iterative entry point, and neither trimesh nor open3d has a linear solver at
all. The cross-library conditioning comparison lives in
[`test_parametrization.py`](test_parametrization.py), where libigl's direct LDLT solves the same
systems end to end. These groups are before/after self-comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
import warp.sparse as wps
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt

# Fraction of vertices pinned as Dirichlet boundary conditions.
_FIXED_FRACTIONS = [0.01, 0.5]

# Residual-check cadence: every iteration (a host sync each) against the on-device test.
_CHECK_EVERY = [1, 0]

# Solve counts for the amortization group.
_REPEATS = [1, 50]

# Right-hand sides solved together, matching the two UV columns the parametrization solvers batch.
_N_RHS = 2

_SEED = 23

_operator_cache: dict[tuple[str, str], wps.BsrMatrix] = {}
_fixed_cache: dict[tuple[str, str, float], tuple] = {}
_rhs_cache: dict[tuple[str, str], wp.array] = {}


def _skip_cpu(bench_case: BenchCase) -> None:
    """Skip triwarp-on-CPU: every solver here goes through ``warp.optim.linear.cg``."""
    assert bench_case.device is not None
    if wp.get_device(bench_case.device).is_cpu:
        pytest.skip("warp.optim.linear.cg returns NaN on the CPU device in Warp 1.14-1.15")


def _operator(bench_case: BenchCase) -> wps.BsrMatrix:
    """Float64 cotangent stiffness -- the *input*; its assembly is timed in test_laplacian."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_cache:
        _operator_cache[key] = tw.laplacian.cotmatrix(
            bench_case.vertices_wp, bench_case.faces_wp, dtype=wp.float64
        )
    return _operator_cache[key]


def _fixed(bench_case: BenchCase, fraction: float) -> tuple:
    """``(fixed_mask, fixed_values)`` pinning ``fraction`` of the vertices, at a fixed seed."""
    key = (bench_case.mesh_name, str(bench_case.device), fraction)
    if key not in _fixed_cache:
        rng = np.random.default_rng(_SEED)
        n = bench_case.n_vertices
        mask_np = np.zeros(n, dtype=bool)
        mask_np[rng.choice(n, size=max(1, int(n * fraction)), replace=False)] = True
        values_np = np.tile(bench_case.vertices_np[:, 2], (_N_RHS, 1))
        _fixed_cache[key] = (
            wp.array(mask_np, dtype=wp.bool, device=bench_case.device),
            twt.as_array2d_float(
                wp.array(
                    np.ascontiguousarray(values_np), dtype=wp.float64, device=bench_case.device
                ),
                dtype=wp.float64,
            ),
        )
    return _fixed_cache[key]


def _rhs(bench_case: BenchCase) -> wp.array:
    """``(n_rhs, n)`` float64 right-hand sides taken from the vertex coordinates."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _rhs_cache:
        rhs_np = np.ascontiguousarray(bench_case.vertices_np[:, :_N_RHS].T)
        _rhs_cache[key] = wp.array(rhs_np, dtype=wp.float64, device=bench_case.device)
    return _rhs_cache[key]


@pytest.mark.benchmark(group="min_quad_with_fixed")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_min_quad_with_fixed(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    Dirichlet-constrained quadratic minimization: submatrix extraction plus a batched CG solve.

    Four rows per table. Across the mesh pair the gap is conditioning; across the pin pair it is
    both a smaller free block and a better-conditioned one. The interesting cell is
    ``saddle_graded`` at 1% pinned -- the worst-conditioned, largest free system.
    """
    _skip_cpu(bench_case)
    operator = _operator(bench_case)
    fixed_mask, fixed_values = _fixed(bench_case, fixed_fraction)
    solution, _free_map, n_free = bench_case.run(
        lambda: tw.linalg.min_quad_with_fixed(operator, fixed_mask, fixed_values)
    )
    assert solution.shape[0] == _N_RHS
    assert n_free > 0, "a fully-pinned system returns without solving, so there is nothing to time"


@pytest.mark.benchmark(group="solve_spd_columns")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("check_every", _CHECK_EVERY, ids=["every", "ondevice"])
def test_solve_spd_columns(bench_case: BenchCase, check_every: int) -> None:
    """Residual-check cadence: host syncs traded against possibly-wasted iterations."""
    _skip_cpu(bench_case)
    operator, rhs = _operator(bench_case), _rhs(bench_case)
    solution = wp.zeros_like(rhs)
    solution_2d = twt.as_array2d_float(solution, dtype=wp.float64)
    rhs_2d = twt.as_array2d_float(rhs, dtype=wp.float64)

    def run() -> tuple:
        solution.zero_()
        return tw.linalg.solve_spd_columns(operator, rhs_2d, solution_2d, check_every=check_every)

    # ``check_every=0`` reports its iteration count in a device array (the residual test runs
    # inside ``wp.capture_while``), while a positive cadence reports a host int -- so only the
    # solution is asserted on here, and the iteration count is asserted in the group above.
    assert bench_case.run(run) is not None
    assert solution.shape[0] == _N_RHS


@pytest.mark.benchmark(group="spd_column_solver_amortized")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("repeats", _REPEATS, ids=["once", "x50"])
def test_spd_column_solver_amortized(bench_case: BenchCase, repeats: int) -> None:
    """
    One solve against fifty over the same preallocated state.

    Each call warm-starts from whatever ``solution`` already holds, which is exactly how
    ``parametrization.arap`` drives it -- so the later solves in the ``x50`` row are converging from
    a good guess and should be much cheaper than the first. That is the effect being measured; a
    ``x50`` row near 50x ``once`` would mean the reuse is buying nothing.

    Unlike ``arap``, this group re-solves the *same* right-hand side, so calls 2..50 are essentially
    no-ops -- which makes it the sharpest probe in the suite of per-call solver overhead. It is what
    caught the one regime where the default ``check_every=0`` loses: the conditional-graph loop
    costs ~0.5 ms a call whatever it does, so this row runs 2x slower than at ``check_every=10``
    while every other solver group got faster. See the ``check_every`` notes on
    ``triwarp.linalg.solve_spd_columns``.
    """
    _skip_cpu(bench_case)
    operator, rhs = _operator(bench_case), _rhs(bench_case)
    solution = wp.zeros_like(rhs)
    solver = tw.linalg.spd_column_solver(
        operator,
        twt.as_array2d_float(rhs, dtype=wp.float64),
        twt.as_array2d_float(solution, dtype=wp.float64),
    )

    def run() -> None:
        solution.zero_()
        for _ in range(repeats):
            solver()

    bench_case.run(run, rounds=3)
    assert solution.shape[0] == _N_RHS
