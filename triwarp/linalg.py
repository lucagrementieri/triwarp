"""
Sparse linear-algebra helpers shared by the parametrization and smoothing solvers.

Two layers:

- [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed] and its two steps
  ([`free_partition`][triwarp.linalg.free_partition],
  [`assemble_interior_system`][triwarp.linalg.assemble_interior_system]) port
  ``igl::min_quad_with_fixed``: minimize a quadratic form with some degrees of freedom pinned, by
  eliminating the fixed rows/columns into a reduced symmetric system.
- [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] and
  [`spd_column_solver`][triwarp.linalg.spd_column_solver] solve one symmetric positive-definite
  operator against several right-hand-side columns in a *single* conjugate-gradient call.

**Why batch the columns.** Solving a ``k``-column system as a Python loop of ``k`` independent
``cg`` calls costs the ``sum`` of the per-column iteration counts, and every CG iteration then
launches ``k`` separate sets of reduction and AXPY kernels.
[`replicated_operator`][triwarp.linalg.replicated_operator] instead presents the *same* operator as
``k`` independent subproblems over one flat ``k * n`` vector: the solver then advances all columns
together and stops on the worst-case residual, so the cost becomes ``max`` of the per-column
iteration counts with one set of vector kernels per iteration. The sparse matrix is never
replicated in memory — the ``matvec`` issues ``k`` ``bsr_mv`` calls against the single operator.

**Whose conjugate gradient.** One column goes to ``warp.optim.linear.cg``; more than one goes to
this module's own ``_BatchedCg``, which runs the same iteration and the same stopping rule. The
split is not a preference — it is the one input for which Warp's reduction degrades: batching is
expressed to Warp as ``batch_offsets``, and that is exactly what makes its dot-product reduction
take a per-column path whose cost grows with the vector length rather than staying flat. Reducing
per column with a real two-stage tree keeps the batching and drops that cost; see ``_BatchedCg``
for the mechanism.

**Determinism.** Build each operator natively at its final dtype in a *single*
``warp.sparse.bsr_from_triplets`` and never recast or rebuild it. A rebuild re-sorts and
duplicate-accumulates an order the CSR already has, which is wasted work; and if the rebuild's
triplet buffers are ever sized off ``BsrMatrix.nnz`` (a stale *capacity*, not the true entry count —
see ``nnz_sync()``) the buffers' tail reaches ``bsr_from_triplets`` uninitialized. Nothing here
recasts an operator, and ``Q_uu`` is assembled as a CSR *directly*, without any triplet build, so it
carries an exact ``nnz``.

**Why Jacobi.** Every solve here preconditions with ``warp.optim.linear.preconditioner(A, "diag")``.
On a cotangent Laplacian a preconditioner costing ``k`` mat-vecs per iteration cuts the iteration
count by only about ``sqrt(k)``, so total work scales as ``k / sqrt(k) = sqrt(k)`` — single-level
preconditioning loses on this operator class, and only a multilevel method escapes it. IC(0) would
help if it were available, but Warp has no sparse triangular solve — ``warp.sparse`` exposes only
``bsr_from_triplets`` and ``warp.optim.linear`` only the Krylov methods and a diagonal
preconditioner — and no parallel substitute for a triangular solve keeps its advantage: a fully
parallel Jacobi-sweep approximation, an exact apply parallelized by graph coloring (whose
uncoalesced access and extra launches eat most of its own iteration win), and natural-ordering
level sets (whose level count swings wildly with the input's vertex numbering) are all worse than
Jacobi in practice. Chebyshev as a single-level preconditioner is a further alternative and is also
worse; Chebyshev as the multigrid V-cycle's *smoother* is a separate question — see
``_MULTIGRID_SWEEPS`` — and neither decline covers the other.

Two further obstacles are specific to this repository. Obtuse triangles give negative cotangent
weights (``triwarp/kernels/laplacian.py``), so ``-L`` is often not the M-matrix that IC(0) existence
requires; and both [`heat_geodesic`][triwarp.heat.heat_geodesic] and
[`heat_signed_distance`][triwarp.heat.heat_signed_distance] solve a ``-L`` with a genuine constant
null space, where IC(0) hits a zero pivot on the last row of every connected component.

**The multilevel option** is [`multigrid_preconditioner`][triwarp.linalg.multigrid_preconditioner]:
smoothed aggregation, the one scheme that breaks the conjugate-gradient iteration count's growth
with problem size rather than paying it down by a constant factor. It is not the default, and the
reason is the *setup* rather than the cycle: building the hierarchy (one aggregation, one power
iteration, a ``bsr_transposed`` and three ``bsr_mm`` per level) has a real fixed cost, so the
V-cycle wins exactly where the solve it replaces is long enough to amortize that setup, and loses
on a well-conditioned or already-fast-converging system. See
[`CG_MULTIGRID_DOMINANCE`][triwarp.linalg.CG_MULTIGRID_DOMINANCE] for how the gate decides which
systems clear that bar.

A GPU sparse direct solver (cuDSS through ``nvmath-python`` and CuPy) was also considered and
declined for the same reason the multilevel preconditioner is gated: its cost is a host-side
symbolic factorization plan that is flat regardless of how well-conditioned the system is, so it
wins exactly on the systems the multigrid gate already routes to a hierarchy and loses everywhere
CG converges quickly. It would also add an optional CUDA-only dependency, and a direct solver is
singular on the empty rows CG tolerates (unreferenced free vertices, which a caller would have to
pin and restore). One idea from that investigation is worth keeping in mind for a future factor-
once, solve-many caller: reusing a factorization plan across solves of one sparsity pattern is
comparatively cheap, so a caller that resolves the same operator repeatedly (as ``arap`` already
does with its own preconditioner) is the shape that would benefit.

Nothing cheap predicts in advance which side of the multigrid-vs-Jacobi line a system falls on,
which is why ``preconditioner="auto"`` uses a capped Jacobi probe rather than a heuristic predictor
of the iteration count; see [`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS].

The aggregation keeps a strength-of-connection threshold (``_MULTIGRID_THETA``) rather than every
off-diagonal, because on a graded (anisotropic) patch keeping every off-diagonal aggregates across
the weak direction and the hierarchy converges far more slowly.

**Routing through ``"auto"`` decides from the operator, not the caller**, and the axis that matters
is conditioning: only a system whose off-diagonal dominance clears
[`CG_MULTIGRID_DOMINANCE`][triwarp.linalg.CG_MULTIGRID_DOMINANCE] benefits from a hierarchy. A plain
Laplacian (``tutte``, `min_quad_with_fixed` on a raw cotangent matrix) and `lscm`'s coupled u/v
system sit below that bar and a forced hierarchy regresses them, where a *squared* operator
(`harmonic` at ``k >= 2``) sits comfortably above it. So
[`harmonic`][triwarp.parametrization.harmonic] passes ``"auto"`` at ``k >= 2`` and ``"diag"`` below,
and
[`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed] keeps ``"diag"`` as its default rather
than becoming a second ``"auto"`` caller.

!!! warning "``harmonic`` at ``k=2`` on a strongly graded patch may not converge under Jacobi"
    A ``k=2`` biharmonic operator on a strongly graded patch can be outside what
    Jacobi-preconditioned conjugate gradient reaches in ``float64`` at all: the iteration can hit
    its cap and return a residual several orders of magnitude above the requested tolerance, with
    nothing but a ``UserWarning`` to say so, producing a visibly wrong UV map. A caller who needs
    that combination should pass a stronger preconditioner and check the warning.

That heat system ``M - tL`` [`heat_geodesic`][triwarp.heat.heat_geodesic] solves needs no
multilevel help of its own: it converges in a small, size-independent number of iterations, because
``t = h**2`` makes it a small perturbation of the mass matrix.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, cast, overload

import numpy as np
import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.kernels import array as kernel_array
from triwarp.kernels import linalg as kernel_linalg
from triwarp.kernels.algorithms import conjugate_gradient as kernel_cg
from triwarp.kernels.algorithms import multigrid as kernel_mg

# Relative residual tolerance used when a caller does not supply one.
CG_TOLERANCE = 1e-10

# Iteration cap as a multiple of the per-column system size, matching the callers' previous
# hand-rolled ``maxiter=10 * n``.
CG_MAXITER_FACTOR = 10

# How often the conjugate-gradient loop tests the residual against the tolerance. ``0`` means
# "every iteration, on device": ``warp.optim.linear`` then drives the loop with ``wp.capture_while``
# and an on-device condition kernel, with no host readback at all. See the ``check_every`` parameter
# docs for the measurements and for the return-type consequence.
CG_CHECK_EVERY = 0

# Batching several iterations per conditional-graph test was tried and removed; do not
# reintroduce it without a *short* solve in the sweep. ``wp.capture_while`` evaluates its condition
# on device at a few microseconds against ~1 us for a replayed launch, so running a run of
# iterations per test amortizes that -- but it also overshoots by up to ``K - 1`` iterations past
# the point the residual crossed, and that overshoot is a fixed number of launches whose *share*
# is set by how long the solve is. Long solves therefore win a little and short ones lose a lot:
# the best cell gained ~13 % while three others lost 28-40 %, and every summary statistic put the
# unbatched loop ahead. A converged iteration here is a pure no-op -- ``cg_step_p`` and
# ``cg_step_x_r_z`` pin a converged column's ``beta`` / ``alpha`` to exactly zero -- so the
# overshoot buys nothing at all, unlike the equivalent batching in a breadth-first level loop,
# where an extra level still does useful work and a small batch is kept.

# Cadence substituted for ``check_every=0`` on a device without conditional CUDA graphs, where Warp
# cannot test the residual on device and would otherwise run every solve to ``maxiter``. Warp's own
# default, and what this module shipped before the device-side check became the default.
CG_CHECK_EVERY_FALLBACK = 10

# Jacobi iterations ``preconditioner="auto"`` runs before it escalates to a multigrid hierarchy, on
# the systems its gate (``CG_MULTIGRID_DOMINANCE`` below) did not already send straight to one. This
# cap is the *fallback* branch and runs unchanged on every system the gate declines.
#
# It is a cap, not a predictor: nothing cheap distinguishes in advance a system that benefits from a
# hierarchy from one that does not, on axes like size, iteration count or the probe's own
# convergence rate, so a system that converges inside the cap never pays for a hierarchy setup and
# runs at exactly Jacobi's speed. That is conservative -- it forgoes the hierarchy's benefit on
# systems that converge just inside the cap -- and deliberately so: escalating early or not at all
# is better than escalating from a mid-range cap, which pays for both the probe and the hierarchy
# setup without benefiting from either. The escalated solve does not meaningfully reuse the probe's
# iterates either, since Jacobi leaves a residual a V-cycle still has to work down from scratch. The
# axis that actually separates the two classes is a property of the operator, not of the probe --
# see ``CG_MULTIGRID_DOMINANCE``.
CG_PROBE_ITERATIONS = 2000

# The gate ``preconditioner="auto"`` applies *before* the probe above: a system that clears it goes
# straight to a multigrid hierarchy and never runs a Jacobi iteration, and one that does not falls
# through to ``CG_PROBE_ITERATIONS`` unchanged. So the gate can only ever *remove* a probe, which is
# what makes it safe: on a system it declines, ``"auto"`` behaves exactly as it did before it
# existed.
#
# The axis is the operator's off-diagonal dominance, not its size: ``max_i sum_{j != i} |A_ij| /
# A_ii``, one kernel and one max-reduction over the assembled system (``_offdiagonal_dominance``).
# Smoothed aggregation's advantage over Jacobi grows with how much weight a row carries off its
# diagonal, so this is the operator property the V-cycle is actually paid for.
CG_MULTIGRID_MIN_UNKNOWNS = 1500

# Free unknowns above which a merely *moderate* dominance is enough to skip the probe -- the
# well-shaped regular meshes, where conjugate gradient's growth in the mesh size is the whole
# problem. Paired with ``CG_MULTIGRID_SIZE_FLOOR`` below, which keeps this branch from firing on an
# operator class it was not calibrated on.
CG_MULTIGRID_LARGE_UNKNOWNS = 7000

# Dominance floor under which the size branch above does **not** fire, however large the system.
#
# The size branch is the operator-class-specific half of the gate: it transfers to well-shaped
# regular meshes but not to every large system. A plain Laplacian's rows nearly sum to zero (a
# uniform-weight parametrization, a raw cotangent system) and does not need a hierarchy at any size
# -- it needs iterations -- while a squared or coupled operator does. Without this floor the size
# branch alone would fire on any large system regardless of conditioning.
CG_MULTIGRID_SIZE_FLOOR = 1.9

# Off-diagonal dominance above which ``"auto"`` skips its probe, given at least
# ``CG_MULTIGRID_MIN_UNKNOWNS`` unknowns.
CG_MULTIGRID_DOMINANCE = 2.15


def min_quad_with_fixed(
    q: wps.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: twt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str = "diag",
) -> tuple[twt.Array2dFloat, wp.array[wp.int32], int]:
    """
    Minimize a quadratic form with pinned degrees of freedom.

    Eliminates the fixed degrees of freedom from ``0.5 x' Q x`` and solves the reduced symmetric
    system ``Q_uu x_u = -Q_ub bc`` for every right-hand-side column at once. Handles both shapes the
    parametrization solvers need: several *independent* columns over ``n_vertices`` unknowns
    (``harmonic`` / ``tutte``, one column per UV coordinate) and a single *coupled* column over
    ``2 * n_vertices`` unknowns (``lscm``).

    Parameters
    ----------
    q
        ``(n_dofs, n_dofs)`` symmetric positive-semi-definite operator, ``float64``.
    fixed_mask
        Length-``n_dofs`` mask: ``True`` marks a pinned degree of freedom.
    fixed_values
        ``(n_rhs, n_dofs)`` prescribed values; only the entries where ``fixed_mask`` is ``True`` are
        read.
    tol
        Relative residual tolerance for the conjugate-gradient solve.
    check_every
        Iterations between residual tests, forwarded to
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]. The default tests on device every
        iteration; see there for the tradeoff. This entry point's own return type does not depend
        on it — the solver's ``(iterations, residual, atol)`` triple is not surfaced here.
    preconditioner
        Forwarded to [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]. ``"diag"`` (the
        default) because the eliminated operator is usually a plain Laplacian, whose rows nearly sum
        to zero and which a V-cycle does not help. The exception is a caller that squares the
        operator, and [`harmonic`][triwarp.parametrization.harmonic] passes ``"auto"`` at ``k >= 2``
        for exactly that reason.

    Returns
    -------
    solution : twt.Array2dFloat
        ``(n_rhs, n_free)`` solved values for the unpinned degrees of freedom.
    free_map : wp.array[wp.int32]
        Length-``n_dofs`` compact remap of the unpinned degrees of freedom.
    n_free : int
        Number of unpinned degrees of freedom.

    Raises
    ------
    RuntimeError
        If ``fixed_mask`` and ``fixed_values`` are not all on one device.

    Notes
    -----
    The elimination is ``igl::min_quad_with_fixed``'s, without its ``Aeq`` linear-equality block --
    no caller here needs one.

    When every degree of freedom is pinned (``n_free == 0``) the prescribed values are the whole
    answer: an empty solution is returned without a solve, so this case also works on the CPU.
    Reconstructing the full field from ``solution`` / ``free_map`` is left to the caller's scatter.

    See Also
    --------
    [`free_partition`][triwarp.linalg.free_partition]
    [`assemble_interior_system`][triwarp.linalg.assemble_interior_system]
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]
    """
    require_same_device(fixed_mask=fixed_mask, fixed_values=fixed_values)
    device = fixed_mask.device
    n_rhs = int(fixed_values.shape[0])
    free_map, n_free = free_partition(fixed_mask)

    solution = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    if n_free == 0:
        return twt.as_array2d(solution, wp.float64), free_map, n_free

    q_uu, rhs = assemble_interior_system(q, fixed_mask, free_map, fixed_values, n_free)
    solve_spd_columns(
        q_uu,
        rhs,
        twt.as_array2d(solution, wp.float64),
        tol=tol,
        check_every=check_every,
        preconditioner=preconditioner,
    )
    return twt.as_array2d(solution, wp.float64), free_map, n_free


def free_partition(fixed_mask: wp.array[wp.bool]) -> tuple[wp.array[wp.int32], int]:
    """
    Compact remap of the *unpinned* degrees of freedom, plus their count.

    Thin inversion of [`mask_to_compact_ranks`][triwarp.array.mask_to_compact_ranks]:
    ``fixed_mask`` marks the constrained degrees of freedom, and the returned map indexes the
    reduced system built over the complement.

    Parameters
    ----------
    fixed_mask
        Length-``n_dofs`` mask: ``True`` marks a pinned degree of freedom.

    Returns
    -------
    free_map : wp.array[wp.int32]
        Length-``n_dofs``; ``free_map[i]`` is the reduced-system index of degree of freedom ``i``,
        meaningful only where ``fixed_mask[i]`` is ``False``.
    n_free : int
        Number of unpinned degrees of freedom.

    See Also
    --------
    [`mask_to_compact_ranks`][triwarp.array.mask_to_compact_ranks]
    """
    return tw.array.mask_to_compact_ranks(fixed_mask, invert=True)


def assemble_interior_system(
    q: wps.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    fixed_values: twt.Array2dFloat,
    n_free: int,
) -> tuple[wps.BsrMatrix[wp.float64], twt.Array2dFloat]:
    """
    Extract the free-free block ``Q_uu`` and the constant right-hand side ``-Q_ub bc``.

    A **CSR-to-CSR** extraction in two launches over the ``n_dofs`` rows of ``q``: the first counts
    each free row's surviving entries (and accumulates the pinned-column contributions into the
    ``n_rhs`` right-hand-side rows, which needs the same row scan), a prefix sum turns those counts
    into row offsets, and the second fills ``columns`` / ``values`` in place, remapped through
    ``free_map``. The result is handed to a ``warp.sparse.BsrMatrix`` directly.

    !!! note "Why not ``bsr_from_triplets``"
        Because the input is *already* a CSR. Emitting ``q.nnz`` triplets and letting
        ``bsr_from_triplets`` sort and duplicate-accumulate them re-derives an order the input
        already had, at real cost for a large operator. Row order comes from the launch index and
        column order from ``q``'s own rows through the monotone ``free_map``, so the extracted
        matrix is sorted by construction.

        It also avoids a second, quieter cost: ``bsr_from_triplets`` leaves ``nnz`` at the
        *triplet* count, which is ``q.nnz`` — an upper bound on the true entry count of a
        lightly-pinned ``Q_uu`` — so every downstream ``bsr_mv`` would be dimensioned for the
        unreduced matrix. The count here is exact.

    Parameters
    ----------
    q
        ``(n_dofs, n_dofs)`` symmetric positive-semi-definite operator, ``float64``.
    fixed_mask
        Length-``n_dofs`` mask: ``True`` marks a pinned degree of freedom.
    free_map
        Compact remap from [`free_partition`][triwarp.linalg.free_partition].
    fixed_values
        ``(n_rhs, n_dofs)`` prescribed values.
    n_free
        Number of unpinned degrees of freedom.

    Returns
    -------
    q_uu : ``warp.sparse.BsrMatrix``
        ``(n_free, n_free)`` reduced operator.
    rhs : twt.Array2dFloat
        ``(n_rhs, n_free)`` constant right-hand side. ``arap`` adds its per-iteration rotation term
        to this buffer rather than reassembling it.

    Raises
    ------
    RuntimeError
        If ``fixed_mask``, ``free_map`` and ``fixed_values`` are not all on one device.

    See Also
    --------
    [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed]
    """
    require_same_device(fixed_mask=fixed_mask, free_map=free_map, fixed_values=fixed_values)
    device = fixed_mask.device
    n_dofs = int(fixed_mask.shape[0])
    n_rhs = int(fixed_values.shape[0])
    # Every free row writes its own count exactly once (``free_map`` is a bijection onto
    # ``[0, n_free)``), so there is nothing to pre-zero.
    counts = wp.empty(n_free, dtype=wp.int32, device=device)
    rhs = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    wp.launch(
        kernel_linalg.interior_row_counts,
        dim=n_dofs,
        inputs=[q.offsets, q.columns, q.values, fixed_mask, free_map, fixed_values, counts, rhs],
        device=device,
    )
    # The total-terminated form *is* the CSR offsets array, and the one host read it costs is what
    # sizes ``columns`` / ``values`` for their final use at allocation time.
    row_offsets, nnz_uu = tw.array.counts_to_offsets(counts, include_total=True)
    columns = wp.empty(nnz_uu, dtype=wp.int32, device=device)
    values = wp.empty(nnz_uu, dtype=wp.float64, device=device)
    wp.launch(
        kernel_linalg.interior_system_csr,
        dim=n_dofs,
        inputs=[q.offsets, q.columns, q.values, fixed_mask, free_map, row_offsets, columns, values],
        device=device,
    )
    # Hand the finished CSR to a compact ``BsrMatrix`` directly. ``notify_nnz_changed`` is Warp's
    # documented entry point for exactly this -- storage metadata assigned from outside
    # ``warp.sparse`` -- and ``bsr_zeros`` leaves ``row_counts`` at ``None``, which is the compact
    # topology these arrays describe.
    q_uu = wps.bsr_zeros(n_free, n_free, wp.float64, device=device)
    q_uu.offsets = row_offsets
    q_uu.columns = columns
    q_uu.values = values
    q_uu.notify_nnz_changed(nnz=nnz_uu)
    return q_uu, twt.as_array2d(rhs, wp.float64)


def solve_spd(
    matrix: wps.BsrMatrix[Any],
    rhs: twt.ArrayNd,
    solution: twt.ArrayNd,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY_FALLBACK,
    preconditioner: wpl.LinearOperator | None = None,
    name: str = "solve_spd",
) -> tuple[int, float, float]:
    """
    Solve one symmetric positive-definite system by preconditioned conjugate gradient.

    The single-right-hand-side form of
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns], and deliberately dtype-agnostic: it
    carries scalar ``float64`` systems as well as the ``wp.mat22d``-block operators the
    tangent-field solvers build, whose right-hand side is an array of ``wp.vec2d``. ``solution`` is
    the initial guess and is overwritten in place, so warm starting is free.

    This exists so the Jacobi preconditioner and the iteration cap are decided in one place rather
    than re-derived at every call site.

    !!! note "``check_every`` defaults to the host-side cadence here"
        Unlike [`solve_spd_columns`][triwarp.linalg.solve_spd_columns], which defaults to
        [`CG_CHECK_EVERY`][triwarp.linalg.CG_CHECK_EVERY] (the device-side test), this defaults to
        [`CG_CHECK_EVERY_FALLBACK`][triwarp.linalg.CG_CHECK_EVERY_FALLBACK] — Warp's own default —
        so the return values stay host scalars. The device-side check trades a host readback for a
        conditional-graph loop, which only pays off when the solve runs long enough to amortize it;
        the short solves that call this (tens of iterations) run slower with it. Pass
        ``check_every=0`` where a solve is known to be long.

    Parameters
    ----------
    matrix
        Symmetric positive-(semi-)definite operator. Scalar or block dtype.
    rhs
        Right-hand side, with as many rows as ``matrix``.
    solution
        Initial guess, overwritten with the result. Same shape and dtype as ``rhs``.
    tol
        Relative residual tolerance.
    maxiter
        Iteration cap. When ``None``, uses
        [`CG_MAXITER_FACTOR`][triwarp.linalg.CG_MAXITER_FACTOR] times the number of rows.
    check_every
        Residual-test cadence; ``0`` tests on device every iteration and returns device arrays.
    preconditioner
        Optional Jacobi preconditioner for ``matrix``, built here when ``None``. Pass one to hoist
        its construction out of a loop that solves against the same operator repeatedly.
    name
        Caller name, used in the non-convergence warning.

    Returns
    -------
    tuple[int, float, float]
        Whatever ``warp.optim.linear.cg`` returns: iteration count, residual and tolerance. Device
        1-element arrays instead of host scalars when ``check_every=0``.

    Raises
    ------
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Warns
    -----
    UserWarning
        When the solve exhausts ``maxiter`` without reaching ``tol``. The returned ``solution`` is
        then whatever the last iterate happened to be, not an answer -- silence here is how a
        diverging solve reaches a caller looking like a converged one. Only detectable when
        ``check_every > 0``; under ``check_every=0`` the counts stay on device and testing them
        would reintroduce the readback that setting exists to avoid.

    See Also
    --------
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]
    [`spd_column_solver`][triwarp.linalg.spd_column_solver]
    """
    require_same_device(rhs=rhs, solution=solution)
    n_rows = int(rhs.shape[0])
    iteration_cap = CG_MAXITER_FACTOR * n_rows if maxiter is None else maxiter
    result = wpl.cg(
        matrix,
        rhs,
        solution,
        tol=tol,
        # ``atol=0.0`` rather than omitted: ``warp.optim.linear``'s own tolerance resolution sets
        # ``atol := tol`` whenever ``atol`` is left ``None``, silently turning this "relative
        # residual tolerance" into an *absolute* floor of the same numeric value -- a right-hand
        # side whose own norm falls below ``tol`` then "converges" at the untouched initial guess
        # in zero iterations, which looks like a correct answer rather than a failure (CLAUDE.md
        # section 12.7). ``_BatchedCg`` below already passes an explicit zero for the same
        # reason.
        atol=0.0,
        maxiter=iteration_cap,
        M=wpl.preconditioner(matrix, "diag") if preconditioner is None else preconditioner,
        check_every=_supported_check_every(check_every),
    )
    _warn_if_not_converged(result, iteration_cap, name)
    return cast("tuple[int, float, float]", result)


def solve_spd_columns(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str = "diag",
) -> tuple[int, float, float]:
    """
    Solve one symmetric positive-definite operator against several right-hand-side columns.

    Single batched conjugate-gradient call: all ``n_rhs`` columns advance together and convergence
    uses the worst-case residual, so the iteration count is the *maximum* over columns rather than
    their sum. ``solution`` is used as the initial guess (warm starting is therefore free) and is
    overwritten in place.

    !!! warning "``check_every=0`` returns device arrays, not host scalars"
        Under the default ``check_every=0`` nothing is ever read back, so the three returned values
        are 1-element **device arrays** — and the second and third are the *squared* residual norm
        and *squared* absolute tolerance, matching ``warp.optim.linear.cg``. Code that formats or
        compares them must call ``.numpy()`` first, which reintroduces the host sync the setting
        exists to avoid. Pass a positive ``check_every`` to get host scalars back. ``solution`` is
        unaffected either way.

        On a device without conditional CUDA graphs there is nowhere to put the device-side test,
        so ``check_every=0`` is replaced by
        [`CG_CHECK_EVERY_FALLBACK`][triwarp.linalg.CG_CHECK_EVERY_FALLBACK] and host scalars come
        back instead. Warp's own behaviour in that case is to run every solve to ``maxiter`` —
        ``CG_MAXITER_FACTOR * n`` iterations of guaranteed waste — so substituting the cadence is
        the only usable reading of the request.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-definite operator, ``float64``.
    rhs
        ``(n_rhs, n)`` right-hand sides, one per row. Must be contiguous.
    solution
        ``(n_rhs, n)`` initial guess, overwritten with the result. Must be contiguous.
    tol
        Relative residual tolerance, as a ratio of the right-hand-side norm.
    maxiter
        Iteration cap. Defaults to ``CG_MAXITER_FACTOR * n``.
    check_every
        How many iterations run between residual tests. ``0`` (the default) tests every iteration
        on device; see Notes for the tradeoff and the warning above for what it does to the return
        type.
    preconditioner
        ``"diag"`` (the default) for the Jacobi preconditioner, ``"multigrid"`` for the
        smoothed-aggregation V-cycle [`multigrid_preconditioner`]
        [triwarp.linalg.multigrid_preconditioner] builds, or ``"auto"`` to let the operator decide.
        ``"auto"`` first tests the assembled system against
        [`CG_MULTIGRID_DOMINANCE`][triwarp.linalg.CG_MULTIGRID_DOMINANCE] and its size floor: a
        system that clears the gate builds the V-cycle outright, and one that does not runs Jacobi
        under [`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS] and escalates only if
        that has not converged. The V-cycle costs a setup pass and pays for itself only where the
        solve dominates the call, so ``"auto"`` is the setting for a caller whose systems vary --
        it cannot regress a solve that was already short, though it can forgo a small win on a
        small ill-conditioned system that the size floor declines. See that function's Notes, and
        both constants.

    Returns
    -------
    tuple[int, float, float]
        ``(iterations, residual_norm, absolute_tolerance)`` on ``warp.optim.linear.cg``'s terms,
        with the residual taken over the worst column. Device arrays rather than host scalars under
        ``check_every=0``; see the warning above. A one-column solve returns Warp's own values and
        a batched one returns the same three from this module's solver.

    Raises
    ------
    ValueError
        If ``preconditioner`` is not one of the three names above. It is rejected rather than
        treated as ``"diag"``, so a misspelling cannot turn into a silently slower solve.
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Warns
    -----
    UserWarning
        When the solve exhausts ``maxiter`` without reaching ``tol``, on the same terms as
        [`solve_spd`][triwarp.linalg.solve_spd]. A batched solve converges on its *worst* column,
        so hitting the cap here means at least one column is unsolved. Only detectable when
        ``check_every > 0``.

    Notes
    -----
    ``check_every`` is a pure performance knob on every path but one — it cannot change the
    converged answer, only how far past the tolerance the solver may overshoot before it notices.
    The exception is the exactly-two-column, ``"diag"``-preconditioned path (this module's internal
    block conjugate gradient): it shares one Krylov subspace across both columns, so a positive
    ``check_every`` can let a batch of iterations apply a real, coupled update to a column that
    already crossed its own tolerance before the next readback catches up. Its device-side default
    (``0``) scales with how much work a single ``cg`` call does, and the regimes disagree, so read
    the one that matches the caller:

    - **Cold single solves** — one ``cg`` call from a zero initial guess, the shape
      ``harmonic`` / ``tutte`` / ``smooth_region`` take. This is the regime the default is set for,
      and it is a clear win there.
    - **Warm-started solves inside an iteration loop** — ``arap``, whose right-hand side changes
      every iteration so each solve still runs tens of CG iterations from the previous answer. The
      default is neutral to mildly positive here.
    - **Repeated near-converged solves over one [`spd_column_solver`]
      [triwarp.linalg.spd_column_solver] state** — the same right-hand side re-solved back to back,
      so every call after the first converges in one or two iterations. Here the device-side check's
      fixed per-call cost dominates a solve that short, so it is a real loss; pass a positive
      ``check_every`` to a state driven that way.
    - **Raising ``check_every`` well above the default is a loss in every regime**: the readback it
      saves is cheap, while the extra iterations it can cause on top of the tolerance are real work
      — the smaller the solve, the worse the trade.

    The *preconditioner* is not a knob worth turning beyond ``"diag"`` / ``"multigrid"`` /
    ``"auto"`` above: IC(0) and Chebyshev both come out a wash or a loss against the Jacobi
    preconditioner used here. See "Why Jacobi" in the [`triwarp.linalg`][triwarp.linalg] module
    documentation.

    The *per-iteration* cost of batching was a separate lever and has been taken: with more than one
    column this runs triwarp's own conjugate gradient rather than ``warp.optim.linear``'s, because
    Warp's reduction degrades on precisely the batched input the worst-case stopping rule needs. See
    "Whose conjugate gradient" in the [`triwarp.linalg`][triwarp.linalg] module documentation.

    See Also
    --------
    [`spd_column_solver`][triwarp.linalg.spd_column_solver]
    [`replicated_operator`][triwarp.linalg.replicated_operator]
    """
    require_same_device(rhs=rhs, solution=solution)
    result = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=maxiter,
        check_every=check_every,
        preconditioner=preconditioner,
        run=True,
        caller="solve_spd_columns",
    )
    _warn_if_not_converged(
        result,
        maxiter if maxiter is not None else CG_MAXITER_FACTOR * int(rhs.shape[1]),
        "solve_spd_columns",
    )
    return result


def spd_column_solver(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    maxiter: int | None = None,
    check_every: int = CG_CHECK_EVERY,
    preconditioner: str = "diag",
) -> wpl.LinearSolverState | _BatchedCg:
    """
    Pre-allocated batched conjugate-gradient state, for repeated solves of one operator.

    Same solve as [`solve_spd_columns`][triwarp.linalg.solve_spd_columns], but the temporary buffers
    and the batch layout are allocated once and the returned state is callable. Build it *outside* a
    per-iteration loop and call it inside: ``solve_spd_columns`` would otherwise reallocate its
    temporaries and rebuild the operator wrappers on every iteration.

    ``rhs`` and ``solution`` are captured at construction, so a loop that overwrites those same
    buffers in place can call the state with no arguments.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-definite operator, ``float64``. Held by the returned state.
    rhs
        ``(n_rhs, n)`` right-hand sides; re-read from this buffer on every call.
    solution
        ``(n_rhs, n)`` initial guess and result; warm-started from its current contents each call.
    tol
        Relative residual tolerance.
    maxiter
        Iteration cap. Defaults to ``CG_MAXITER_FACTOR * n``.
    check_every
        Iterations between residual tests; see
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] for the tradeoff and for what the
        default ``0`` does to the values each call returns.
    preconditioner
        ``"diag"`` or ``"multigrid"``, as in
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]. Built once here and reused by every
        call against this state, which is the shape the V-cycle's setup cost wants. ``"auto"`` is
        **not** accepted: it decides on the first solve -- from that system's operator, and from
        the probe when the operator does not settle it -- and a hoisted state exists to be driven
        many times.

    Returns
    -------
    ``warp.optim.linear.LinearSolverState`` | internal batched state
        Callable solver state: a single-column solve returns Warp's own
        ``LinearSolverState``, and a multi-column one returns this module's own ``_BatchedCg``,
        kept internal because a caller never has to name the difference (see
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]'s "Whose conjugate gradient").
        Substituted operands must match the construction-time shape, dtype, device and batch
        layout. Each call returns what
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] returns, including its
        ``check_every=0`` device arrays.

    Raises
    ------
    ValueError
        If ``preconditioner`` is ``"auto"``, for the reason above, or is not one of the two names
        this accepts.
    RuntimeError
        If ``rhs`` and ``solution`` are not all on one device.

    Examples
    --------
    ```python
    solver = spd_column_solver(q_uu, rhs, solution, tol=1e-10)
    for _ in range(max_iterations):
        ...  # rewrite rhs in place
        solver()
    ```

    See Also
    --------
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]
    """
    require_same_device(rhs=rhs, solution=solution)
    return _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=maxiter,
        check_every=check_every,
        preconditioner=preconditioner,
        run=False,
        caller="spd_column_solver",
    )


@overload
def _cg_columns(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str,
    run: Literal[True],
    caller: str,
) -> tuple[int, float, float]: ...
@overload
def _cg_columns(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str,
    run: Literal[False],
    caller: str,
) -> wpl.LinearSolverState | _BatchedCg: ...
def _cg_columns(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float,
    maxiter: int | None,
    check_every: int,
    preconditioner: str,
    run: bool,
    caller: str,
) -> tuple[int, float, float] | wpl.LinearSolverState | _BatchedCg:
    """
    Batched conjugate gradient over the columns of ``rhs`` -- the shared body of the two solvers.

    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] and
    [`spd_column_solver`][triwarp.linalg.spd_column_solver] build the same replicated operator and
    the same replicated Jacobi preconditioner and differ only in ``run``: the first drives the
    solve, the second hands back the un-run state for a caller to drive repeatedly.
    """
    n_columns, n = int(rhs.shape[0]), int(rhs.shape[1])
    iteration_cap = maxiter if maxiter is not None else CG_MAXITER_FACTOR * n
    if preconditioner == "auto":
        if not run:
            raise ValueError(
                f'{caller} cannot take preconditioner="auto": it decides on the *first* solve, '
                'and a hoisted state is built to be driven many times. Pass "diag" or '
                '"multigrid".'
            )
        return _cg_columns_auto(
            matrix,
            rhs,
            solution,
            tol=tol,
            cap=iteration_cap,
            check_every=check_every,
            caller=caller,
        )
    # Validated rather than left to fall through. Every branch below tests for one name and treats
    # anything else as ``"diag"``, so without this an unrecognised string -- ``"jacobi"``,
    # ``"amg"``, a capitalised ``"Multigrid"`` -- runs a Jacobi solve silently, bit-identically to
    # ``"diag"`` and with nothing but the clock to tell the caller.
    if preconditioner not in ("diag", "multigrid"):
        raise ValueError(
            f'{caller}: unknown preconditioner {preconditioner!r}, expected "diag", "multigrid" '
            'or "auto".'
        )
    if n_columns > 1:
        # ``_BatchedCg`` exists only for this branch; a single column already reaches
        # ``warp.optim.linear``'s fast reduction, because there is nothing to batch. See that
        # class's Notes.
        #
        # A block-CG variant that shares one Krylov subspace across exactly two columns
        # (O'Leary 1980) was built for this branch, shipped, and removed again. It does reduce the
        # iteration count on a *uniform* mesh -- 140 against 182 and 275 against 355 on the two
        # saddle patches -- but its iteration costs 10 launches against this one's 8, so the wall
        # clock is a wash there (1.03x), and on every other system reached in this package it is
        # neutral or a loss: the same operator on a graded mesh runs 4 774 iterations against 2 230
        # (0.39x end to end), ``harmonic[hemisphere]`` 234 against 237 (0.88x), and
        # ``heat.extend_scalar`` returns the identical count on both (1.00x), so that function's
        # measured win belongs entirely to batching its two solves into one call. Best case +3 %
        # against a worst case of -159 % is not a mechanism worth a gate, and the ill-conditioned
        # half is block CG's own documented failure mode -- near-parallel search directions, which
        # a Tikhonov floor bounds but does not deflate. Do not reintroduce it without deflation and
        # a measurement on a graded mesh.
        state = _BatchedCg(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=iteration_cap,
            check_every=_supported_check_every(check_every),
            preconditioner=preconditioner,
        )
        return cast("tuple[int, float, float]", state()) if run else state
    operator = replicated_operator(matrix, n_columns)
    if preconditioner == "multigrid":
        apply_inverse = multigrid_preconditioner(matrix, n_columns)
    else:
        apply_inverse = replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    return cast(
        "tuple[int, float, float] | wpl.LinearSolverState",
        wpl.cg(
            operator,
            rhs.flatten(),
            solution.flatten(),
            tol=tol,
            # See the identical comment in solve_spd (CLAUDE.md section 12.7): an omitted atol
            # here silently becomes atol := tol, an absolute floor of the "relative" tolerance.
            atol=0.0,
            maxiter=iteration_cap,
            M=apply_inverse,
            check_every=_supported_check_every(check_every),
            run=run,
        ),
    )


def _cg_columns_auto(
    matrix: wps.BsrMatrix[wp.float64],
    rhs: twt.Array2dFloat,
    solution: twt.Array2dFloat,
    *,
    tol: float,
    cap: int,
    check_every: int,
    caller: str,
) -> tuple[int, float, float]:
    """
    Run Jacobi under ``CG_PROBE_ITERATIONS``, then escalate if it has not converged.

    A system that finishes inside the probe pays nothing at all for the option -- the probe *is* the
    solve. One that does not is the ill-conditioned kind
    [`multigrid_preconditioner`][triwarp.linalg.multigrid_preconditioner] is for.

    The escalated solve does warm-start from the iterate the probe left in ``solution``, but **do
    not read that as the probe being cheap**: its cost is very nearly additive on top of the
    V-cycle's, because Jacobi leaves a residual whose low-frequency part is exactly what the
    V-cycle then has to work down, so the probe's iterates carry over close to nothing.

    See [`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS] for why this is a cap rather
    than a predictor, and [`CG_MULTIGRID_DOMINANCE`][triwarp.linalg.CG_MULTIGRID_DOMINANCE] for the
    gate that runs first and decides, on the operator alone, whether to skip the probe entirely.
    """
    if _wants_multigrid(matrix):
        return _cg_columns(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=cap,
            check_every=check_every,
            preconditioner="multigrid",
            run=True,
            caller=caller,
        )
    if cap <= CG_PROBE_ITERATIONS:
        return _cg_columns(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=cap,
            check_every=check_every,
            preconditioner="diag",
            run=True,
            caller=caller,
        )
    probe = _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=CG_PROBE_ITERATIONS,
        check_every=check_every,
        preconditioner="diag",
        run=True,
        caller=caller,
    )
    residual, tolerance = _cg_residual_and_tolerance(probe)
    if residual <= tolerance:
        return probe
    # ``cap`` is the caller's total iteration budget, and the probe above already spent
    # ``CG_PROBE_ITERATIONS`` of it (this branch is only reached once ``cap > CG_PROBE_ITERATIONS``,
    # by the early return above) -- so the escalated call, which is a fresh ``wpl.cg`` call with its
    # own iteration counter, gets what remains rather than the full ``cap`` again.
    return _cg_columns(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=cap - CG_PROBE_ITERATIONS,
        check_every=check_every,
        preconditioner="multigrid",
        run=True,
        caller=caller,
    )


def _wants_multigrid(matrix: wps.BsrMatrix[wp.float64]) -> bool:
    """
    Whether ``preconditioner="auto"`` should skip its probe and build a hierarchy outright.

    The gate is a size floor crossed with the operator's off-diagonal dominance -- see
    [`CG_MULTIGRID_DOMINANCE`][triwarp.linalg.CG_MULTIGRID_DOMINANCE]. Declining is always safe: the
    caller then runs the Jacobi probe it would have run anyway.
    """
    n_rows = int(matrix.nrow)
    if n_rows < CG_MULTIGRID_MIN_UNKNOWNS:
        return False
    dominance = _offdiagonal_dominance(matrix)
    if dominance > CG_MULTIGRID_DOMINANCE:
        return True
    return n_rows >= CG_MULTIGRID_LARGE_UNKNOWNS and dominance > CG_MULTIGRID_SIZE_FLOOR


def _offdiagonal_dominance(matrix: wps.BsrMatrix[wp.float64]) -> float:
    """
    ``max_i sum_{j != i} |A_ij| / A_ii`` over the rows of a scalar-block CSR operator.

    One launch and one max-reduction, so the whole thing is cheap relative to the solve it is
    deciding about, essentially independent of the operator's size. The only host readback is the
    reduced scalar. Rows with a non-positive diagonal contribute 0 rather than an infinity -- see
    the kernel.
    """
    n_rows = int(matrix.nrow)
    device = matrix.values.device
    ratios = wp.empty(n_rows, dtype=wp.float64, device=device)
    wp.launch(
        kernel_linalg.offdiagonal_dominance_rows,
        dim=n_rows,
        # ``nnz`` is a stale capacity, but ``offsets``/``columns``/``values`` are indexed through
        # the row offsets here rather than sliced by it, so no count is read off the matrix.
        inputs=[matrix.offsets, matrix.columns, matrix.values, ratios],
        device=device,
    )
    return float(tw.reduce.max(ratios))


def _cg_residual_and_tolerance(result: tuple[Any, ...]) -> tuple[float, float]:
    """
    Unwrap a conjugate-gradient result's worst-column residual norm and tolerance as host floats.

    ``check_every=0`` returns 1-element *device* arrays holding the **squared** residual norm and
    squared absolute tolerance, one entry per column, and a positive cadence returns their square
    roots as host scalars already reduced over the columns -- so the unwrapping differs and the
    quantity does not. Two 8-byte readbacks in the device case, once per solve.
    """
    _iterations, residual, tolerance = result
    if not isinstance(residual, wp.array):
        return float(residual), float(tolerance)
    residual_np, tolerance_np = residual.numpy(), tolerance.numpy()
    worst = int(np.argmax(residual_np / np.maximum(tolerance_np, 1e-300)))
    return math.sqrt(float(residual_np[worst])), math.sqrt(float(tolerance_np[worst]))


class _BatchedCg:
    """
    Jacobi-preconditioned conjugate gradient over the columns of one operator.

    Same iteration as ``warp.optim.linear.cg`` and the same stopping rule -- every column runs
    until *its own* residual is under ``max(atol, tol * ||b_c||)`` -- so it is a drop-in for the
    multi-column path, down to the ``(iterations, residual, tolerance)`` return contract. Callable
    like ``warp.optim.linear``'s solver state: the buffers are allocated once and every call
    re-reads ``rhs`` and warm-starts from whatever ``solution`` currently holds.

    Notes
    -----
    It is here because of what the batching costs inside Warp's solver rather than because of the
    arithmetic. ``replicated_operator`` attaches ``batch_offsets`` to get the per-column stopping
    rule, and that is precisely the input for which Warp's ``TiledDot`` selects a per-column
    reduction path whose cost grows with the vector length, where the reduction it uses for an
    unbatched vector stays flat -- and a CG iteration runs two of these dot products. Reducing per
    column with a real two-stage tree keeps the batching and drops that cost back down.

    Two fusions ride along and are free. The Jacobi apply is an elementwise multiply of the ``r``
    that ``cg_step_x_r_z`` has just written, so it happens in a register rather than in its own
    launch; and the ``rz_old = rz_new`` copy folds into the ``p.Ap`` finalize, which is the one
    point in the iteration after the ``p`` update that last read ``rz_old`` and before the x/r
    update that reads it next.

    **What was tried and is not here.** Solving the columns as separate unbatched ``cg`` calls also
    reaches the tree reduction, but it pays a second copy of every other kernel in the iteration
    and loses. And dropping ``batch_offsets`` to get the tree from a single call is not a tuning
    change at all -- ``alpha`` and ``beta`` would then be global rather than per column, which is CG
    on the block system and a different iteration.

    **Sharing the Krylov subspace across columns was tried and removed.** This class's
    "worst-column stopping rule" batches the *launches* but runs each column's iteration
    mathematically independently, so the iteration counts are unchanged versus separate solves. A
    classical block conjugate gradient (O'Leary 1980) over exactly two columns does reduce them --
    140 against 182 and 275 against 355 on the two uniform saddle patches -- but at 10 launches per
    iteration against this class's 8 the wall clock is a wash (1.03x), and on a *graded* mesh the
    two search directions go nearly parallel and the count goes the other way, 4 774 against 2 230
    (0.39x end to end). See ``_cg_columns`` for the full seven-system measurement and why the
    mechanism is not gated but gone.
    """

    def __init__(
        self,
        matrix: wps.BsrMatrix[wp.float64],
        rhs: twt.Array2dFloat,
        solution: twt.Array2dFloat,
        *,
        tol: float,
        maxiter: int,
        check_every: int,
        preconditioner: str = "diag",
    ) -> None:
        device = matrix.device
        self._device = device
        self._matrix = matrix
        self._n_columns, self._n = int(rhs.shape[0]), int(rhs.shape[1])
        self._tol = float(tol)
        self._maxiter = int(maxiter)
        self._check_every = int(check_every)

        # Every vector the iteration reduces is one this class owns, so the column pitch is ours to
        # choose: pad it to a whole number of tiles and zero the gap, and the dot has no ragged
        # block at all. See ``cg_dot_partials`` for what the ragged block cost when it existed.
        tile = int(kernel_cg.CG_TILE)
        self._blocks = (self._n + tile - 1) // tile
        self._stride = self._blocks * tile
        self._dofs = self._n_columns * self._stride
        # The partials are padded on the same argument, so the finalize's last tile reads zeros
        # rather than the next column's partials; see ``cg_dot_finalize``.
        partial_pitch = ((self._blocks + tile - 1) // tile) * tile

        self._rhs = rhs
        self._solution = solution
        self._solution_flat = solution.flatten()
        # ``wp.zeros`` rather than ``wp.empty``: the pad between each column's ``n`` entries and
        # ``stride`` must read as zero, and it stays zero because every kernel writing there writes
        # a multiple of it.
        self._r = wp.zeros(self._dofs, dtype=wp.float64, device=device)
        self._z = wp.zeros(self._dofs, dtype=wp.float64, device=device)
        self._p = wp.zeros(self._dofs, dtype=wp.float64, device=device)
        self._ap = wp.zeros(self._dofs, dtype=wp.float64, device=device)
        self._partials = wp.zeros(
            (2, self._n_columns, partial_pitch), dtype=wp.float64, device=device
        )
        self._dots = wp.zeros((2, self._n_columns), dtype=wp.float64, device=device)
        # The two rows of ``_dots``, viewed once. ``_dots`` is allocated here and never rebound, so
        # both views are valid for the solver's whole life -- and re-taking one costs ~3 us of
        # ``wp.array.__getitem__`` every time. ``_dot_finalize`` runs once per CG iteration on the
        # host-check path, where that measured **2.00 slices per iteration**; on the CUDA capture
        # path the body is recorded once and replayed, so there it is per *solve* instead.
        self._dots_rz = twt.as_dense(self._dots[0])
        self._dots_carry = twt.as_dense(self._dots[1])
        self._p_dot_ap = wp.zeros((2, self._n_columns), dtype=wp.float64, device=device)
        self._rz_old = wp.zeros(self._n_columns, dtype=wp.float64, device=device)
        self._atol_sq = wp.zeros(self._n_columns, dtype=wp.float64, device=device)
        # The round-loop state array (``kernels/array.py``'s ``LOOP_ROUND`` / ``LOOP_CONDITION``),
        # the iteration count in the first slot. Seeded to ``[0, 1]`` in ``_reset`` -- a zero
        # condition would run no iterations at all, since ``wp.capture_while`` reads it first.
        self._state = wp.zeros(kernel_array.LOOP_STATE_SIZE, dtype=wp.int32, device=device)

        # A multigrid V-cycle cannot be fused into a register the way the Jacobi apply is, so it
        # runs as its own launches over the same padded vectors and the x/r update drops its ``z``
        # write -- one extra launch per iteration, which is the whole cost of un-fusing. The
        # hierarchy is batched over the columns at *this* solver's pitch, so the cycle's elementwise
        # passes stay one launch each rather than one per column.
        self._cycle = None
        if preconditioner == "multigrid":
            hierarchy = _multigrid_hierarchy(matrix, 0)
            if hierarchy is not None:
                self._cycle = _MultigridCycle(
                    *hierarchy, n_columns=self._n_columns, stride=self._stride
                )
        # 1 in the pad, so the fused Jacobi apply there is a no-op on an already-zero residual.
        self._inv_diag = wp.full(self._stride, 1.0, dtype=wp.float64, device=device)
        wp.map(kernel_array.inverse_or_one, wps.bsr_get_diag(matrix), out=self._inv_diag[: self._n])
        # Per-column view of ``r``, built once: ``_initialize`` still seeds and residual-corrects
        # it one column at a time (a one-time cost, unlike the per-iteration matvec in
        # ``_iteration``, which reads the flat buffers directly through ``csr_matvec``). Spans
        # ``n``, not ``stride``, so nothing writes the pad.
        self._r_blocks = self._column_views(self._r)

    def _column_views(self, flat: wp.array[wp.float64]) -> list[wp.array[wp.float64]]:
        """Split a padded flat vector into its ``n_columns`` blocks of ``n`` live entries."""
        return _flat_column_views(flat, self._n_columns, self._n, self._stride)

    def _dot(
        self,
        a: wp.array[wp.float64],
        b0: wp.array[wp.float64],
        b1: wp.array[wp.float64],
        pairs: int,
        out_dots: twt.Array2dFloat64,
        carry: bool = False,
    ) -> None:
        """Per-column dots of ``(a, b0)`` and, when ``pairs == 2``, ``(a, b1)``."""
        wp.launch_tiled(
            kernel_cg.cg_dot_partials,
            dim=(self._n_columns, self._blocks),
            inputs=[a, b0, b1, wp.int32(self._stride), wp.int32(pairs)],
            outputs=[self._partials],
            block_dim=int(kernel_cg.CG_TILE),
            device=self._device,
        )
        self._dot_finalize(pairs, out_dots, carry=carry)

    def _dot_finalize(self, pairs: int, out_dots: twt.Array2dFloat64, carry: bool = False) -> None:
        """
        Second stage of the dot: fold ``self._partials`` per (pair, column).

        Split out of ``_dot`` because the *first* stage has two producers -- the standalone
        ``cg_dot_partials`` launch, and ``cg_step_x_r_z_dot``, which emits the same partials as a
        side effect of writing the ``r`` and ``z`` they are taken over.
        """
        wp.launch_tiled(
            kernel_cg.cg_dot_finalize,
            # ``pairs``, not the kernel's max of 2: the ``r.z`` dot only ever asks for one row, so
            # launching the second row's blocks (which the kernel's own ``p >= pairs`` guard would
            # just return out of) is pure waste.
            dim=(pairs, self._n_columns),
            inputs=[
                self._partials,
                wp.int32(self._blocks),
                wp.int32(pairs),
                wp.int32(1 if carry else 0),
                self._dots_carry,
            ],
            outputs=[out_dots, self._rz_old],
            block_dim=int(kernel_cg.CG_TILE),
            device=self._device,
        )

    def _iteration(self) -> None:
        """One CG step, none of which reads back to the host."""
        # One launch over every column at once rather than ``n_columns`` separate ``bsr_mv`` calls
        # against the same operator -- ``kernels/algorithms/multigrid.py::csr_matvec``'s own
        # measurement (34 launches / 189 us through ``bsr_mv`` against 15 / 85 through this, same
        # answer) is exactly this call shape, and it already reads this solver's own flat
        # column-block layout (``_column_views``' ``column * stride + row`` addressing). This
        # collapses what used to be ``n_columns`` separate matvec launches into exactly one,
        # regardless of ``n_columns``.
        wp.launch(
            kernel_mg.csr_matvec,
            dim=self._n_columns * self._n,
            inputs=[
                wp.int32(self._n),
                wp.int32(self._stride),
                wp.int32(self._stride),
                wp.int32(0),
                wp.float64(1.0),
                self._matrix.offsets,
                self._matrix.columns,
                self._matrix.values,
                self._p,
            ],
            outputs=[self._ap],
            device=self._device,
        )
        # ``carry=True`` performs ``rz_old = rz_new`` here; see ``cg_dot_finalize``.
        self._dot(self._p, self._ap, self._ap, 1, self._p_dot_ap, carry=True)
        step = [
            wp.int32(self._stride),
            wp.int32(self._n),
            self._rz_old,
            self._p_dot_ap,
            self._dots,
            self._atol_sq,
        ]
        if self._cycle is None:
            # Six launches, not seven: the x/r/z update also emits the first stage of its own
            # ``r.r`` / ``r.z`` reduction, so only the finalize is left to launch. A block-level
            # partial needs nothing but its own block's data, which is what makes folding a
            # reduction's first stage into its producer legal.
            tile = int(kernel_cg.CG_TILE)
            wp.launch_tiled(
                kernel_cg.cg_step_x_r_z_dot,
                dim=(self._n_columns, self._stride // tile),
                inputs=[*step, self._inv_diag, self._p, self._ap],
                outputs=[self._solution_flat, self._r, self._z, self._partials],
                block_dim=tile,
                device=self._device,
            )
            self._dot_finalize(2, self._dots)
        else:
            wp.launch(
                kernel_cg.cg_step_x_r,
                dim=self._dofs,
                inputs=[*step, self._p, self._ap],
                outputs=[self._solution_flat, self._r],
                device=self._device,
            )
            self._cycle.apply(self._r, self._z)
            self._dot(self._r, self._r, self._z, 2, self._dots)
        # Seven launches, not eight: the round-loop advance rides in ``cg_step_p``'s thread 0
        # rather than in a ``dim=1`` launch behind it. See that kernel for why no barrier is
        # needed.
        wp.launch(
            kernel_cg.cg_step_p,
            dim=self._dofs,
            inputs=[
                wp.int32(self._stride),
                wp.int32(self._maxiter),
                wp.int32(self._n_columns),
                self._rz_old,
                self._dots,
                self._atol_sq,
                self._z,
            ],
            outputs=[self._p, self._state],
            device=self._device,
        )

    def _initialize(self) -> None:
        """Seed the residual from the caller's operands, then set the tolerances and ``p``."""
        # ``r`` starts as ``b``, which also gives the tolerance its ``||b||`` without a buffer of
        # its own: the pad is zero, so the dot over the padded ``r`` is the dot over ``b``.
        for column in range(self._n_columns):
            wp.copy(self._r_blocks[column], self._rhs[column])
        self._dot(self._r, self._r, self._r, 1, self._p_dot_ap)
        wp.launch(
            kernel_cg.cg_absolute_tolerance,
            dim=self._n_columns,
            inputs=[wp.float64(self._tol * self._tol), wp.float64(0.0), self._p_dot_ap],
            outputs=[self._atol_sq],
            device=self._device,
        )
        # ``r -= A x`` in place, warm-starting from whatever ``solution`` currently holds -- one
        # launch across every column via ``csr_matvec``'s own ``alpha`` rather than ``n_columns``
        # separate ``bsr_mv`` calls (the same conversion as ``_iteration``'s matvec, applied to this
        # one-time setup call). ``x_stride`` is ``n``, not ``self._stride``: ``solution`` is the
        # caller's own unpadded buffer.
        wp.launch(
            kernel_mg.csr_matvec,
            dim=self._n_columns * self._n,
            inputs=[
                wp.int32(self._n),
                wp.int32(self._n),
                wp.int32(self._stride),
                wp.int32(1),
                wp.float64(-1.0),
                self._matrix.offsets,
                self._matrix.columns,
                self._matrix.values,
                self._solution_flat,
            ],
            outputs=[self._r],
            device=self._device,
        )
        if self._cycle is None:
            wp.launch(
                kernel_cg.scaled_diagonal_apply,
                dim=self._dofs,
                inputs=[
                    # ``stride`` twice, so the kernel skips nothing -- *not* because ``n ==
                    # stride`` (it is only equal when ``n`` is a whole number of tiles), but
                    # because running the apply over the pad is a no-op: ``r``'s pad is zero and
                    # ``inv_diag``'s is one, so ``z`` gets zero there, which is what the dots
                    # downstream already assume.
                    wp.int32(self._stride),
                    wp.int32(self._stride),
                    self._inv_diag,
                    wp.float64(1.0),
                    self._r,
                ],
                outputs=[self._z],
                device=self._device,
            )
        else:
            self._cycle.apply(self._r, self._z)
        self._dot(self._r, self._r, self._z, 2, self._dots)
        wp.copy(self._p, self._z)
        wp.copy(self._rz_old, self._dots_carry)
        self._state.assign([0, 1])

    def __call__(self):
        """
        Run the solve, returning ``warp.optim.linear.cg``'s three values on its own terms.

        Device arrays under ``check_every == 0``, host scalars otherwise, exactly as
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] documents.
        """
        self._initialize()
        # The device-side loop needs a conditional CUDA graph, so a CPU-resident system takes the
        # host cadence even where the machine has a GPU -- ``_supported_check_every`` can only see
        # the *machine*, not which device these arrays are on.
        check_every = self._check_every
        if check_every == 0 and not self._device.is_cuda:
            check_every = CG_CHECK_EVERY_FALLBACK
        if check_every > 0:
            self._run_with_host_checks(check_every)
            return (
                int(read_scalar(self._state, 0)),
                math.sqrt(float(self._dots_rz.numpy().max())),
                math.sqrt(float(self._atol_sq.numpy().max())),
            )
        condition = self._state[kernel_array.LOOP_CONDITION_VIEW]
        # One iteration per conditional-graph test, not a batched run of them: see the note above
        # ``CG_CHECK_EVERY`` for the sweep that removed the batching.
        with wp.ScopedCapture(self._device) as capture:
            wp.capture_while(condition, self._iteration)
        wp.capture_launch(capture.graph)
        return self._state[0:1], self._dots_rz, self._atol_sq

    def _run_with_host_checks(self, check_every: int) -> None:
        """
        Drive the loop from the host: issue a block of iterations, then read the residual.

        The block is trimmed against ``maxiter`` so the cap is exact rather than rounded up to the
        next multiple of the cadence -- a caller that reads the returned iteration count against
        the cap it passed is how a non-convergence warning gets raised.
        """
        # ``_atol_sq`` is written once by ``_initialize`` and never touched again, so it is read
        # back on the first check only: the test was taking *two* readbacks per block where one is
        # a loop constant. It is read **after** ``_dots_rz`` rather than before the loop, and the
        # order is the whole point -- the first read of a block drains the queue that block's
        # launches just filled (~0.1 ms) and a second one straight after it is ~0.02 ms, so
        # hoisting it above the loop would buy its own drain and lose on a single-block solve.
        done = 0
        atol_sq_np = None
        while done < self._maxiter:
            block = min(check_every, self._maxiter - done)
            for _ in range(block):
                self._iteration()
            done += block
            residual_np = self._dots_rz.numpy()
            if atol_sq_np is None:
                atol_sq_np = self._atol_sq.numpy()
            if bool((residual_np <= atol_sq_np).all()):
                return


def _flat_column_views(
    flat: wp.array[wp.float64], n_columns: int, n: int, stride: int
) -> list[wp.array[wp.float64]]:
    """Split a padded flat ``n_columns * stride`` vector into its ``n_columns`` blocks of ``n``."""
    return [twt.as_dense(flat[c * stride : c * stride + n]) for c in range(n_columns)]


def replicated_operator(
    matrix: wps.BsrMatrix[wp.float64] | wpl.LinearOperator, n_columns: int
) -> wpl.LinearOperator:
    """
    Present one ``(n, n)`` operator as ``n_columns`` independent subproblems over a flat vector.

    The returned operator acts on length-``n_columns * n`` vectors laid out as ``n_columns``
    contiguous blocks — exactly the memory of a contiguous ``(n_columns, n)`` array — and carries
    the ``batch_offsets`` that make ``warp.optim.linear``'s solvers iterate all blocks together and
    converge on the worst-case residual. ``matrix`` is **not** replicated in memory: the ``matvec``
    applies the single operator to each block in turn.

    Parameters
    ----------
    matrix
        ``(n, n)`` operator: a ``warp.sparse.BsrMatrix``, a 1-D array (read as a diagonal), or an
        existing ``warp.optim.linear.LinearOperator`` such as a preconditioner.
    n_columns
        Number of independent right-hand-side columns.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        Batched operator of shape ``(n_columns * n, n_columns * n)``.

    Notes
    -----
    A block-diagonal *matrix* would give the same batching at ``n_columns`` times the operator
    memory, and 2x2 blocks holding a scalar multiple of the identity would cost four times the
    storage and flops per scalar entry. Replicating only the ``matvec`` avoids both.

    ``batch_offsets`` is attached and ``max_batch_length`` deliberately is not, although this
    function knows the value exactly (every subproblem is ``n`` scalars): passing it to skip
    redundant reduction levels for a known maximum subproblem size made no measurable difference
    here, so it would be one more argument to explain for nothing. ``warp.optim.linear``'s
    ``restart`` option was also considered and is worse: resetting the search direction throws away
    the Krylov space CG has built, costing more iterations rather than buying better-conditioned
    ones.

    See Also
    --------
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]
    """
    base = wpl.aslinearoperator(matrix)
    if n_columns == 1:
        # Nothing to batch: hand back the operator itself rather than wrapping every matvec in a
        # one-iteration Python loop. ``lscm`` solves a single coupled column and would otherwise pay
        # the wrapper cost for no benefit.
        return base
    n = int(base.shape[0])

    # Per-block views, cached by buffer address. A solver calls ``matvec`` once per iteration with
    # the *same* handful of temporaries, so slicing them fresh every time would add thousands of
    # Python-level array constructions to a solve — enough to erase the batching win outright, and
    # enough to keep the iteration from being captured as a CUDA graph. The cache is keyed on the
    # full memory layout, so a different buffer (or a re-slice of one) never aliases a stale view.
    block_views: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[twt.ArrayNd]] = {}

    def blocks(array: twt.ArrayNd) -> list[twt.ArrayNd]:
        key = (array.ptr, tuple(array.shape), tuple(array.strides))
        views = block_views.get(key)
        if views is None:
            views = [
                twt.as_dense(array[column * n : (column + 1) * n]) for column in range(n_columns)
            ]
            block_views[key] = views
        return views

    def matvec(x: twt.ArrayNd, y: twt.ArrayNd, z: twt.ArrayNd, alpha: float, beta: float) -> None:
        x_blocks, y_blocks, z_blocks = blocks(x), blocks(y), blocks(z)
        for column in range(n_columns):
            base.matvec(x_blocks[column], y_blocks[column], z_blocks[column], alpha, beta)

    offsets = wp.array(
        [column * n for column in range(n_columns + 1)], dtype=wp.int32, device=base.device
    )
    total = n_columns * n
    return wpl.LinearOperator(
        (total, total), base.dtype, base.device, matvec, batch_offsets=offsets
    )


def multigrid_preconditioner(
    matrix: wps.BsrMatrix[wp.float64], n_columns: int = 1, *, seed: int = 0
) -> wpl.LinearOperator:
    """
    Smoothed-aggregation multigrid preconditioner for a symmetric positive-semi-definite operator.

    A single-level preconditioner cannot break conjugate gradient's growth in the mesh size -- see
    "Why Jacobi" in the [`triwarp.linalg`][triwarp.linalg] module documentation for the alternatives
    that were tried and rejected -- because cutting the iteration count by ``sqrt(k)`` at ``k``
    mat-vecs per apply leaves the total work growing. A multigrid V-cycle attacks the low-frequency
    error the smoother cannot see, so the iteration count grows much more slowly with the problem
    size than it does under Jacobi.

    The returned operator acts on length-``n_columns * n`` vectors laid out as ``n_columns``
    contiguous blocks, exactly as [`replicated_operator`][triwarp.linalg.replicated_operator] does,
    so it drops into ``M=`` beside a Jacobi preconditioner without any other change.

    Parameters
    ----------
    matrix
        ``(n, n)`` symmetric positive-(semi-)definite operator, ``float64``. Zero diagonal entries
        are allowed: such a row is identically zero for a semi-definite operator, and the whole
        chain -- smoother, coarse solve and all -- leaves those unknowns at zero.
    n_columns
        Number of independent right-hand-side columns the operator will be applied to.
    seed
        Seeds the aggregation's randomized priorities and the power iteration's start vector, so a
        hierarchy is a deterministic function of the operator and this number.

    Returns
    -------
    ``warp.optim.linear.LinearOperator``
        The V-cycle, of shape ``(n_columns * n, n_columns * n)``. A **Jacobi** preconditioner
        instead when the operator does not coarsen -- see Notes.

    Raises
    ------
    ValueError
        If the returned operator's ``matvec`` is called with ``alpha != 1`` or ``beta != 0``. A
        preconditioner apply is always ``z = M x``, and the general form would cost a pass that no
        caller needs.

    Notes
    -----
    Setup is not free and it is not amortized over anything: building the hierarchy costs one
    aggregation, one power iteration, two ``bsr_mm`` and one ``bsr_transposed`` per level. **Ask for
    this where the solve dominates the call**, and pass the same operator's preconditioner into a
    loop rather than rebuilding it -- [`solve_spd`][triwarp.linalg.solve_spd]'s ``preconditioner``
    parameter exists for exactly that.

    **The setup is host cost and it is per *level*, not per unknown, so it cannot be cut by making
    the operator smaller.** A level's cost is dominated by host-side overhead in
    ``warp.sparse.bsr_mm``, which makes device-to-host readbacks to size its output and so costs
    roughly the same regardless of the level's size. That is Warp's, not this package's, so "make
    the hierarchy cheap enough to run unconditionally" is not a lever available here; the reachable
    version is to build **fewer levels**, which is what this module's ``_MULTIGRID_MAX_COARSE`` is
    set for.

    Coarsening stops at 128 rows, or earlier if a level fails to shrink; the coarsest operator is
    then inverted densely on the host, which is exact and is a single launch inside the cycle where
    an iterative coarse solve would be a data-dependent loop. When coarsening stalls while the level
    is still too large to factor, there is no usable hierarchy and this hands back
    ``warp.optim.linear.preconditioner(matrix, "diag")`` rather than a cycle whose coarse solve is a
    guess -- so a caller never has to branch on the operator's shape.

    !!! warning "That fallback is silent, and the strength threshold can trigger it"
        A stalled hierarchy is indistinguishable from a weak one at the call site: the solve simply
        runs at its Jacobi iteration count. ``_MULTIGRID_THETA`` is what decides how easily it
        happens -- raising it makes more off-diagonals weak, and past some threshold an operator can
        stop coarsening entirely and come back at *exactly* the Jacobi count. So an aggregation
        change that "did nothing" should be checked against the level count before it is read as a
        change that did not help.

    See Also
    --------
    [`solve_spd`][triwarp.linalg.solve_spd]
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]
    [`replicated_operator`][triwarp.linalg.replicated_operator]
    """
    base = wpl.aslinearoperator(matrix)
    n = int(matrix.nrow)
    hierarchy = _multigrid_hierarchy(matrix, seed)
    if hierarchy is None:
        return replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    levels, coarse_inverse = hierarchy
    cycle = _MultigridCycle(levels, coarse_inverse, n_columns=n_columns, stride=n)

    def matvec(x: twt.ArrayNd, y: twt.ArrayNd, z: twt.ArrayNd, alpha: float, beta: float) -> None:
        if alpha != 1.0 or beta != 0.0:
            raise ValueError(
                "multigrid_preconditioner's operator only implements z = M x "
                f"(got alpha={alpha}, beta={beta})"
            )
        cycle.apply(x, z)

    total = n_columns * n
    return wpl.LinearOperator((total, total), base.dtype, base.device, matvec)


# ---------------------------------------------------------------------------
# Smoothed-aggregation multigrid preconditioner
# ---------------------------------------------------------------------------

# Rows below which a level is solved exactly instead of coarsened further. The coarse solve is a
# dense pseudo-inverse factored on the host, so this is also the size of that factorization.
#
# The last level is expensive twice over: it costs a level's setup (one aggregation, one power
# iteration, a prolongator, a transpose and two ``bsr_mm``) and then a smoothing sweep, a
# restriction and a prolongation in *every* cycle, where solving that level densely instead is one
# matvec. 384 keeps headroom under ``_MULTIGRID_MAX_DENSE`` and the host ``pinv`` that guard sizes.
_MULTIGRID_MAX_COARSE = 384

# Hard cap on the hierarchy depth, and on the dense coarse solve. Coarsening stops early whenever a
# level fails to shrink by ``1 - _MULTIGRID_MIN_COARSENING``, which is what happens once a level is
# mostly isolated rows -- the least-squares operators here can carry a fair number of them.
_MULTIGRID_MAX_LEVELS = 12
_MULTIGRID_MIN_COARSENING = 0.9
_MULTIGRID_MAX_DENSE = 512

# Damped-Jacobi sweeps per level per half-cycle, and the damping as a multiple of ``1 / rho`` where
# ``rho`` is the spectral radius of ``D^-1 A``. 4/3 is the classical smoothed-aggregation choice and
# is used both for the smoother and for the prolongation smoother.
#
# Two sweeps is close to the fewest cycles needed for the fastest solve across the systems this
# hierarchy is built for; more sweeps reduce the iteration count further but at a cost per cycle
# that offsets the gain, so the curve in solve time is fairly flat around this value.
#
# A Chebyshev smoother was built to try to do better and is refuted -- do not rebuild it. A
# polynomial smoother of the same degree looked attractive on a synthetic system but did not
# transfer to the real ones this hierarchy solves: a solve cycle here is launch-bound rather than
# flop-bound, and a Chebyshev step costs more launches per iteration than a Jacobi sweep, so trading
# launches for a modest iteration-count reduction is the wrong side of this cost model. No single
# Chebyshev degree and interval was best across every system tried, either.
_MULTIGRID_SWEEPS = 2
_MULTIGRID_JACOBI_FACTOR = 4.0 / 3.0

# Power iterations for that spectral radius, and the round cap for the aggregation's independent
# set. The power iteration is unnormalized -- ``rho`` is recovered from the growth over all the
# steps -- so it costs one mat-vec and one elementwise pass per step, and two inner products. Eight
# rather than fifteen: the extra steps move the iteration count negligibly at real setup cost.
_MULTIGRID_POWER_STEPS = 8
_MULTIGRID_MIS_ROUNDS = 32

# Strength-of-connection threshold for the aggregation: an off-diagonal counts as an edge only when
# ``|A_ij| >= theta sqrt(A_ii A_jj)``. ``0.0`` keeps every off-diagonal, which is the aggregation
# this package shipped first and is bit-exactly what a zero threshold reduces to.
#
# A larger threshold removes weak edges from the strength graph, so aggregates are smaller and the
# hierarchy can grow a level -- each of which costs real setup -- while the iteration count falls.
# ``0.05`` sits in a basin where that trade is favourable on the systems that reach the hierarchy
# today: it buys enough of a fall in iteration count to pay back the extra setup, without pushing
# the coarsening into the range where it stalls (which, at a large enough threshold, leaves a level
# unable to coarsen at all and a caller silently falling back to a plain Jacobi preconditioner). The
# optimum is flat rather than sharp, so this should be moved only on evidence spanning multiple
# systems and both the setup and the solve, not a single system's clock.
_MULTIGRID_THETA = 0.05


class _MultigridLevel:
    """One level of the hierarchy: its operator, its smoother, and its link to the next."""

    __slots__ = (
        "ax",
        "b",
        "dim",
        "inverse_diagonal",
        "matvec_dim",
        "n",
        "omega",
        "operator",
        "prolong_dim",
        "prolongator",
        "r",
        "restrict_dim",
        "restrictor",
        "stride",
        "x",
    )

    def __init__(self, operator: wps.BsrMatrix[wp.float64]) -> None:
        self.operator = operator
        self.n = int(operator.nrow)
        self.prolongator = None
        self.restrictor = None
        self.inverse_diagonal = None
        self.omega = 0.0


def _multigrid_hierarchy(
    matrix: wps.BsrMatrix[wp.float64], seed: int
) -> tuple[list[_MultigridLevel], twt.ArrayNd] | None:
    """
    Coarsen ``matrix`` until a level is small enough to invert densely.

    ``None`` when no usable hierarchy exists -- the coarsening stalled above the dense cap -- which
    is the one case [`multigrid_preconditioner`][triwarp.linalg.multigrid_preconditioner] hands back
    a Jacobi preconditioner instead.
    """
    levels: list[_MultigridLevel] = []
    operator = matrix
    while True:
        level = _MultigridLevel(operator)
        levels.append(level)
        if level.n <= _MULTIGRID_MAX_COARSE or len(levels) >= _MULTIGRID_MAX_LEVELS:
            break
        # One extraction per level, read by both consumers: the aggregation's strength test wants
        # ``sqrt(|A_ii|)`` and the smoother wants ``1 / A_ii``. ``wps.bsr_get_diag`` is an
        # allocation *and* a launch, so extracting it once here rather than once per consumer saves
        # a redundant diagonal extraction per level.
        diag = wps.bsr_get_diag(operator)
        label, n_aggregates = _multigrid_aggregate(operator, diag, seed)
        if n_aggregates >= _MULTIGRID_MIN_COARSENING * level.n:
            break
        diagonal = wp.empty(level.n, dtype=wp.float64, device=operator.device)
        wp.map(kernel_array.inverse_or_one, diag, out=diagonal)
        level.inverse_diagonal = diagonal
        level.omega = _MULTIGRID_JACOBI_FACTOR / _multigrid_spectral_radius(
            operator, diagonal, seed
        )
        level.prolongator = _multigrid_prolongator(
            operator, label, n_aggregates, diagonal, level.omega
        )
        level.restrictor = wps.bsr_transposed(level.prolongator)
        operator = _multigrid_prune(
            wps.bsr_mm(level.restrictor, wps.bsr_mm(operator, level.prolongator))
        )
    coarse_inverse = _multigrid_dense_inverse(levels[-1].operator)
    if coarse_inverse is None:
        return None
    return levels, coarse_inverse


def _multigrid_aggregate(
    matrix: wps.BsrMatrix[wp.float64], diagonal: wp.array[wp.float64], seed: int
) -> tuple[wp.array[wp.int32], int]:
    """
    Aggregate label per row, from a distance-2 maximal independent set on the off-diagonal graph.

    The selection is the randomized-priority pattern ``sample.dart_select_minima`` runs, lifted to
    distance 2 by propagating the packed ``(state, priority, index)`` maximum over one-hop
    neighbours *twice* per round -- so no squared graph is built. Roots then spread their label two
    hops, which tiles the graph because the MIS keeps them at least three hops apart.

    Both walks are over the operator's **strong** off-diagonal graph
    (``|A_ij| >= _MULTIGRID_THETA sqrt(A_ii A_jj)``), tested per edge rather than materialized, so
    that a level pays no extra allocation for the filter and ``theta = 0`` is bit-exactly the
    unfiltered aggregation. ``diagonal`` is ``matrix``'s, passed in because the caller has already
    extracted it for the smoother.
    """
    device = matrix.device
    n = int(matrix.nrow)
    offsets, columns, values = matrix.offsets, matrix.columns, matrix.values
    theta = wp.float64(_MULTIGRID_THETA)
    # ``sqrt(|A_ii|)`` per row, so the strength test below is a product rather than a square root
    # per edge. One ``(n,)`` buffer and one map, read by both walks. The ``diagonal`` argument is
    # the caller's -- the hierarchy needs the same extraction for the smoother, and extracting it
    # here as well is what this used to do.
    scaled_diagonal = wp.empty(n, dtype=wp.float64, device=device)
    wp.map(kernel_array.sqrt_abs, diagonal, out=scaled_diagonal)

    priority = wp.empty(n, dtype=wp.uint32, device=device)
    wp.launch(
        kernel_array.random_priorities, dim=n, inputs=[wp.int32(seed), priority], device=device
    )
    state = wp.full(n, int(kernel_mg.MG_UNDECIDED), dtype=wp.int32, device=device)
    next_state = wp.empty(n, dtype=wp.int32, device=device)
    key = wp.empty(n, dtype=wp.int64, device=device)
    next_key = wp.empty(n, dtype=wp.int64, device=device)
    undecided = wp.zeros(1, dtype=wp.int32, device=device)
    for _ in range(_MULTIGRID_MIS_ROUNDS):
        wp.launch(kernel_mg.mis_seed_keys, dim=n, inputs=[state, priority, key], device=device)
        for _ in range(2):
            wp.launch(
                kernel_mg.mis_propagate,
                dim=n,
                inputs=[key, offsets, columns, values, scaled_diagonal, theta, next_key],
                device=device,
            )
            key, next_key = next_key, key
        undecided.zero_()
        wp.launch(
            kernel_mg.mis_decide,
            dim=n,
            inputs=[key, priority, state, next_state, undecided],
            device=device,
        )
        state, next_state = next_state, state
        # One 4-byte read per round, and there are a handful of rounds: the loop cannot be a
        # device-side one because the *number of aggregates* sizes every buffer downstream.
        if int(read_scalar(undecided, 0)) == 0:
            break

    flags = wp.empty(n, dtype=wp.int32, device=device)
    wp.map(kernel_mg.mis_root_flag, state, out=flags)
    scan_pos = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, scan_pos, inclusive=True)
    n_aggregates = int(read_scalar(scan_pos))

    label = wp.empty(n, dtype=wp.int32, device=device)
    next_label = wp.empty(n, dtype=wp.int32, device=device)
    wp.map(kernel_mg.aggregate_label, state, scan_pos, out=label)
    for _ in range(2):
        wp.launch(
            kernel_mg.spread_aggregate_labels,
            dim=n,
            inputs=[label, offsets, columns, values, scaled_diagonal, theta, next_label],
            device=device,
        )
        label, next_label = next_label, label
    return label, n_aggregates


def _multigrid_spectral_radius(
    matrix: wps.BsrMatrix[wp.float64], inverse_diagonal: wp.array[wp.float64], seed: int
) -> float:
    """
    Spectral radius of ``D^-1 A`` by unnormalized power iteration, for the damping factor.

    Normalizing every step would cost a host readback per step; leaving the iterate to grow and
    taking the geometric mean of the growth over all the steps costs **one**, because the start
    vector is a sign vector whose squared norm is exactly ``n``. ``rho`` is around 3 here, so eight
    steps grow the vector by about ``3 ** 8`` and ``float64`` has room to spare. The estimate
    approaches ``rho`` from below, which is the safe side: it makes the damping *smaller* than the
    stability limit rather than larger.

    Everything about the arithmetic here is chosen against the launch count, because the hierarchy
    build is launch-bound. A step is one fused ``power_step`` launch rather than a ``bsr_mv`` plus
    an elementwise scale, and the two buffers are ping-ponged rather than updated in place, which is
    what allows the single kernel. What is left is mostly the single remaining host sync.
    """
    device = matrix.device
    n = int(matrix.nrow)
    x = wp.empty(n, dtype=wp.float64, device=device)
    y = wp.empty(n, dtype=wp.float64, device=device)
    wp.launch(kernel_mg.random_signs, dim=n, inputs=[wp.int32(seed), x], device=device)
    # Exact: every entry of a sign vector is +-1, so its squared norm is exactly n.
    start = float(n)
    for _ in range(_MULTIGRID_POWER_STEPS):
        wp.launch(
            kernel_mg.power_step,
            dim=n,
            inputs=[inverse_diagonal, matrix.offsets, matrix.columns, matrix.values, x, y],
            device=device,
        )
        x, y = y, x
    # The one readback, and the whole point of the loop: how much the iterate grew over ``K`` steps
    # of ``D^-1 A`` is ``rho ** K`` to the accuracy this needs.
    end = float(wp.utils.array_inner(x, x))
    if not (start > 0.0 and end > 0.0 and math.isfinite(end)):
        return 1.0
    return math.sqrt(end / start) ** (1.0 / _MULTIGRID_POWER_STEPS)


def _multigrid_prolongator(
    matrix: wps.BsrMatrix[wp.float64],
    label: wp.array[wp.int32],
    n_aggregates: int,
    inverse_diagonal: wp.array[wp.float64],
    omega: float,
) -> wps.BsrMatrix[wp.float64]:
    """Smoothed prolongator ``(I - omega D^-1 A) P0`` for the piecewise-constant ``P0``."""
    device = matrix.device
    n = int(matrix.nrow)
    sizes = wp.zeros(n_aggregates, dtype=wp.int32, device=device)
    wp.launch(kernel_mg.aggregate_sizes, dim=n, inputs=[label, sizes], device=device)
    rows = wp.empty(n, dtype=wp.int32, device=device)
    columns = wp.empty(n, dtype=wp.int32, device=device)
    values = wp.empty(n, dtype=wp.float64, device=device)
    wp.launch(
        kernel_mg.tentative_prolongator_triplets,
        dim=n,
        inputs=[label, sizes, rows, columns, values],
        device=device,
    )
    # Exactly one triplet per row and no duplicates, so this build's ``nnz`` is exact.
    tentative = wps.bsr_from_triplets(
        n, n_aggregates, rows, columns, values, prune_numerical_zeros=False
    )
    smoothed = wps.bsr_mm(matrix, tentative)
    # Row-scale by ``-omega D^-1`` in place: one pass over the product's values, where a ``bsr_mm``
    # against a diagonal matrix would be a second sparse product.
    wp.launch(
        kernel_mg.scale_rows,
        dim=n,
        inputs=[smoothed.offsets, inverse_diagonal, wp.float64(-omega), smoothed.values],
        device=device,
    )
    return _multigrid_prune(wps.bsr_axpy(smoothed, tentative, alpha=1.0, beta=1.0))


def _multigrid_prune(matrix: wps.BsrMatrix[wp.float64]) -> wps.BsrMatrix[wp.float64]:
    """
    Drop a matrix's explicitly-zero entries, by rebuilding it from its own CSR.

    ``bsr_mm`` returns a structural **superset** of the product, with the extra entries exactly
    zero, and here those zeros are not cosmetic. An explicit zero at ``(i, c)`` makes coarse column
    ``c`` see fine row ``i``, so the Galerkin product inherits every aggregate reachable from it,
    which can blow up the pattern far more than the true product would. Pruning is part of the
    algorithm, not tidying, and it is what holds operator complexity down to a reasonable multiple.

    The extra entries are interspersed in column order rather than trailing reserved capacity, so
    they cannot be dropped by truncating a row -- the columns stay strictly increasing within every
    row and the padding is scattered gaps.

    ``bsr_compress(matrix, prune_numerical_zeros=True)`` is the API for exactly this and is used
    directly here.

    !!! warning "``inplace=False`` does not mean the source is untouched"
        ``wps.bsr_compress(m)`` at the documented default returns **``m`` itself**, pruned in place
        -- ``result is m`` and ``result.values.ptr == m.values.ptr``, with ``m.nnz_sync()`` reduced
        across the call. It returns ``src`` unchanged when there is nothing to prune, too. That is
        safe at both call sites here only because each passes a freshly built temporary
        (``bsr_mm(...)`` / ``bsr_axpy(...)``) that nothing else holds. **A caller that still needs
        the unpruned matrix must copy it first.**
    """
    return cast("wps.BsrMatrix[wp.float64]", wps.bsr_compress(matrix, prune_numerical_zeros=True))


def _multigrid_dense_inverse(matrix: wps.BsrMatrix[wp.float64]) -> twt.ArrayNd | None:
    """
    Pseudo-inverse of the coarsest operator as a dense ``(n, n)`` device array, or ``None``.

    ``None`` when the level is too large to factor, which is what makes the caller fall back to the
    Jacobi preconditioner rather than ship a cycle whose coarse solve is a guess. Rows whose
    diagonal is zero are dropped before the factorization and left as zero in the result: a positive
    semi-definite operator with ``A_ii == 0`` has an identically zero row and column, so those
    unknowns are already solved, and skipping them is what keeps the factorization small when
    coarsening stalls on isolated rows.
    """
    n = int(matrix.nrow)
    nnz = int(matrix.nnz_sync())
    offsets = matrix.offsets.numpy()[: n + 1]
    columns = matrix.columns.numpy()[:nnz]
    values = matrix.values.numpy()[:nnz].astype(np.float64).reshape(nnz)
    dense = np.zeros((n, n), dtype=np.float64)
    # A triplet build coalesces duplicates, so the CSR has one entry per position and a plain
    # scatter is exact -- ``np.add.at`` would be the same answer several times slower.
    dense[np.repeat(np.arange(n), np.diff(offsets)), columns] = values
    active = np.flatnonzero(dense.diagonal() != 0.0)
    if active.size > _MULTIGRID_MAX_DENSE:
        return None
    inverse = np.zeros((n, n), dtype=np.float64)
    if active.size:
        # ``hermitian=True`` factors through ``eigh`` rather than a general SVD, which the operator
        # being symmetric makes exact and considerably cheaper.
        block = np.linalg.pinv(dense[np.ix_(active, active)], rcond=1e-12, hermitian=True)
        inverse[np.ix_(active, active)] = block
    return wp.array(inverse, dtype=wp.float64, device=matrix.device)


class _MultigridCycle:
    """
    One V-cycle of a smoothed-aggregation hierarchy, over ``n_columns`` blocks of a flat vector.

    Batched over the columns: *every* pass, the sparse mat-vecs included, is one launch over the
    whole flat vector. That is not a tidiness choice -- a cycle at these sizes is launch-bound, and
    ``warp.sparse.bsr_mv`` takes one vector, so routing several right-hand sides through it would
    cost one launch per column per mat-vec. A hand-written batched CSR mat-vec
    (``kernels/algorithms/multigrid.csr_matvec``) gives the same answer in about half the launches.

    The vectors of the *top* level are the caller's own, at the caller's column pitch, so the cycle
    adds no copy at its boundary; every level below allocates its own at pitch ``n``. Every buffer
    is built here and the top level's is bound by ``apply``, so ``apply`` issues launches and
    nothing else -- which is what lets the whole preconditioned iteration be captured as one graph.
    """

    def __init__(
        self,
        levels: list[_MultigridLevel],
        coarse_inverse: wp.array[wp.float64],
        *,
        n_columns: int,
        stride: int,
        sweeps: int = _MULTIGRID_SWEEPS,
    ) -> None:
        self._levels = levels
        self._coarse_inverse = coarse_inverse
        self._device = levels[0].operator.device
        self._n_columns = n_columns
        self._sweeps = max(1, int(sweeps))

        for depth, level in enumerate(levels):
            top = depth == 0
            level.stride = stride if top else level.n
            level.dim = n_columns * level.stride
            level.matvec_dim = n_columns * level.n
            level.ax = wp.zeros(level.dim, dtype=wp.float64, device=self._device)
            level.r = wp.zeros(level.dim, dtype=wp.float64, device=self._device)
            if top:
                continue
            level.b = wp.zeros(level.dim, dtype=wp.float64, device=self._device)
            level.x = wp.zeros(level.dim, dtype=wp.float64, device=self._device)
        for depth, level in enumerate(levels[:-1]):
            child = levels[depth + 1]
            level.restrict_dim = n_columns * child.n
            level.prolong_dim = n_columns * level.n

    @property
    def grid(self) -> list[int]:
        """Rows per level, coarsest last -- the hierarchy's shape."""
        return [level.n for level in self._levels]

    def apply(self, source: wp.array[wp.float64], destination: wp.array[wp.float64]) -> None:
        """One V-cycle: ``destination = M^-1 source``, both at the caller's column pitch."""
        top = self._levels[0]
        top.b, top.x = source, destination
        self._cycle(0)

    def _matvec(
        self,
        matrix: wps.BsrMatrix[wp.float64],
        dim: int,
        n_rows: int,
        x_stride: int,
        y_stride: int,
        x: wp.array[wp.float64],
        y: wp.array[wp.float64],
        accumulate: bool = False,
    ) -> None:
        wp.launch(
            kernel_mg.csr_matvec,
            dim=dim,
            inputs=[
                wp.int32(n_rows),
                wp.int32(x_stride),
                wp.int32(y_stride),
                wp.int32(1 if accumulate else 0),
                wp.float64(1.0),
                matrix.offsets,
                matrix.columns,
                matrix.values,
                x,
                y,
            ],
            device=self._device,
        )

    def _smooth(self, level: _MultigridLevel, sweeps: int) -> None:
        for _ in range(sweeps):
            self._matvec(
                level.operator,
                level.matvec_dim,
                level.n,
                level.stride,
                level.stride,
                level.x,
                level.ax,
            )
            wp.launch(
                kernel_mg.jacobi_sweep,
                dim=level.dim,
                inputs=[
                    wp.int32(level.n),
                    wp.int32(level.stride),
                    level.inverse_diagonal,
                    wp.float64(level.omega),
                    level.b,
                    level.ax,
                    level.x,
                ],
                device=self._device,
            )

    def _cycle(self, depth: int) -> None:
        level = self._levels[depth]
        if level.prolongator is None:
            wp.launch(
                kernel_mg.dense_solve,
                dim=level.dim,
                inputs=[
                    wp.int32(level.n),
                    wp.int32(level.stride),
                    self._coarse_inverse,
                    level.b,
                    level.x,
                ],
                device=self._device,
            )
            return
        # The first sweep from a zero initial guess is a *write*, so nothing has to be zeroed and
        # that sweep costs no mat-vec.
        wp.launch(
            kernel_cg.scaled_diagonal_apply,
            dim=level.dim,
            inputs=[
                wp.int32(level.n),
                wp.int32(level.stride),
                level.inverse_diagonal,
                wp.float64(level.omega),
                level.b,
                level.x,
            ],
            device=self._device,
        )
        self._smooth(level, self._sweeps - 1)
        self._matvec(
            level.operator, level.matvec_dim, level.n, level.stride, level.stride, level.x, level.ax
        )
        wp.launch(
            kernel_mg.residual,
            dim=level.dim,
            inputs=[wp.int32(level.n), wp.int32(level.stride), level.b, level.ax, level.r],
            device=self._device,
        )
        child = self._levels[depth + 1]
        self._matvec(
            level.restrictor,
            level.restrict_dim,
            child.n,
            level.stride,
            child.stride,
            level.r,
            child.b,
        )
        self._cycle(depth + 1)
        # Prolong and correct in one launch: ``accumulate`` adds into the fine iterate.
        self._matvec(
            level.prolongator,
            level.prolong_dim,
            level.n,
            child.stride,
            level.stride,
            child.x,
            level.x,
            accumulate=True,
        )
        self._smooth(level, self._sweeps)


def _warn_if_not_converged(result: tuple[int, float, float], iteration_cap: int, name: str) -> None:
    """
    Warn when conjugate gradient stopped because it ran out of iterations, not because it converged.

    Skipped under ``check_every=0``, where the three values are 1-element *device* arrays and
    inspecting them would cost the host sync that setting exists to avoid.

    That early return is also why the message says "residual norm" and not "squared residual": the
    squared form is what the *device* path carries, and this never reaches it. Every value this
    formats has already had its square root taken -- by ``warp.optim.linear.cg`` on the one-column
    path, and by ``_BatchedCg`` on the batched ones.
    """
    iterations, residual, atol = result
    if isinstance(iterations, wp.array):
        return
    if int(iterations) >= iteration_cap and float(residual) > float(atol):
        warnings.warn(
            f"{name}: conjugate gradient hit its {iteration_cap}-iteration cap with residual norm "
            f"{float(residual):.3e} against tolerance {float(atol):.3e}; the result is "
            f"the last iterate, not a solution. The operator is likely ill-conditioned or "
            f"singular — consider triwarp.laplacian.robust_laplacian, or a smaller step.",
            stacklevel=3,
        )


def _supported_check_every(check_every: int) -> int:
    """
    Substitute a host-side cadence for ``check_every=0`` where the device cannot test on device.

    ``warp.optim.linear`` implements ``check_every=0`` with ``wp.capture_while``; without
    conditional CUDA graphs it has nowhere to put the test and runs every solve to ``maxiter``
    instead. Falling back to [`CG_CHECK_EVERY_FALLBACK`][triwarp.linalg.CG_CHECK_EVERY_FALLBACK]
    keeps the request's *meaning* (stop when converged) at the cost of its return type.
    """
    if check_every == 0 and not wp.is_conditional_graph_supported():
        return CG_CHECK_EVERY_FALLBACK
    return check_every
