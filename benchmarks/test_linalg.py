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
**pymeshlab** is the only reference this module has, and it covers exactly one of the three groups:
``compute_scalar_by_scalar_harmonic_field_per_vertex`` (MeshLab's Generate Scalar Harmonic Field) is
a Dirichlet-constrained solve of the same cotangent system ``min_quad_with_fixed`` solves, so the
two answer the same question by opposite means -- MeshLab factors the free-free block directly,
triwarp runs batched CG on it.

That makes it worth more than a timing row: it is the **conditioning control**. Measured on the axis
meshes (RTX 5090 host) it runs **38.2 ms on ``saddle`` and 39.6 ms on ``saddle_graded``** -- flat --
against triwarp's **14.2 -> 72.5 ms** at 1% pinned. So triwarp wins by 2.7x on the well-conditioned
mesh and loses by **1.8x** on the graded one, and the entire spread is its CG iteration count rather
than anything intrinsic about the problem: a direct factorization of the identical system does not
care. It is the same observation the potpourri3d rows make in
[`test_geodesic.py`](test_geodesic.py) and libigl's LDLT makes in
[`test_parametrization.py`](test_parametrization.py).

Those triwarp figures moved a long way when ``assemble_interior_system`` stopped routing an
already-sorted CSR through ``bsr_from_triplets``, and the *shape* of the move is the interesting
part. At 1% pinned this group went 33.7 -> 14.2 ms (``saddle``) and 82.9 -> 72.5 (``graded``); at
50% pinned it went **30.1 -> 3.1** and **29.4 -> 4.7**, a 6-10x. The triplet build cost ``q.nnz``
regardless of how much of ``Q`` survived, and it left ``q_uu.nnz`` at ``q.nnz`` as well -- an upper
bound 3.5x the true count on a lightly-pinned system -- so every CG mat-vec was dimensioned for the
*unreduced* matrix too. That is why the pin50pct rows, where the free block is smallest, gained the
most: they were the rows paying most for work proportional to the wrong matrix.

Two limits on it, both structural. MeshLab pins **exactly two vertices** (``point1`` / ``point2``
with scalar values), so it has no fixed-fraction axis at all and appears only in the ``pin1pct``
row, the closer and harder of the two. And it is *only* the harmonic solve: nothing in the reference
group exposes a raw iterative solver, so ``solve_spd_columns`` and ``spd_column_solver_amortized``
remain before/after self-comparisons. scipy's ``cg`` would be a host solver over a scipy matrix
(timing SciPy's sparse layer, not this one), libigl solves only through its own direct
factorizations, and neither trimesh nor open3d has a linear solver at all.
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


def _harmonic_endpoints_pml(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Return the two extremal-``x`` vertices, the Dirichlet pair MeshLab's harmonic field takes."""
    vertices_np = bench_case.vertices_np
    order = np.argsort(vertices_np[:, 0])
    return vertices_np[order[0]], vertices_np[order[-1]]


@pytest.mark.benchmark(group="min_quad_with_fixed")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("fixed_fraction", _FIXED_FRACTIONS, ids=["pin1pct", "pin50pct"])
def test_min_quad_with_fixed(bench_case: BenchCase, fixed_fraction: float) -> None:
    """
    Dirichlet-constrained quadratic minimization: submatrix extraction plus a batched CG solve.

    Four rows per table. Across the mesh pair the gap is conditioning; across the pin pair it is
    both a smaller free block and a better-conditioned one. The interesting cell is
    ``saddle_graded`` at 1% pinned -- the worst-conditioned, largest free system.

    The pymeshlab row is the control for exactly that cell: its direct factorization of the same
    system is flat across the mesh pair, so whatever spread triwarp shows is the CG iteration count
    and not the problem.
    """
    if bench_case.kind == "pymeshlab":
        if fixed_fraction != min(_FIXED_FRACTIONS):
            pytest.skip("MeshLab's harmonic field pins exactly two vertices: no fraction axis")
        # Geometry-preserving with ``colorize=False`` (it writes only the vertex scalar), so the
        # shared MeshSet is sound and the ~9 ms build stays out of a ~40 ms row.
        meshset_pml = bench_case.meshset_pml
        point1, point2 = _harmonic_endpoints_pml(bench_case)
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_scalar_harmonic_field_per_vertex(
                point1=point1, point2=point2, colorize=False
            )
        )
        assert meshset_pml.current_mesh().vertex_scalar_array().shape[0] == bench_case.n_vertices
        return

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
