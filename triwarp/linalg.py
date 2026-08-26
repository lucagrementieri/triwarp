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

**Why batch the columns.** A ``k``-column solve used to be a Python loop of ``k`` independent
``cg`` calls, so its cost was ``sum`` of the per-column iteration counts and every CG iteration
launched ``k`` separate sets of reduction and AXPY kernels.
[`replicated_operator`][triwarp.linalg.replicated_operator] instead presents the *same* operator as
``k`` independent subproblems over one flat ``k * n`` vector: the solver then advances all columns
together and stops on the worst-case residual, so the cost becomes ``max`` of the per-column
iteration counts with one set of vector kernels per iteration. The sparse matrix is never
replicated in memory — the ``matvec`` issues ``k`` ``bsr_mv`` calls against the single operator.

**Whose conjugate gradient.** One column goes to ``warp.optim.linear.cg``; more than one goes to
this module's own ``_BatchedCg``, which runs the same iteration and the same stopping rule. The
split is not a preference — it is the one input for which Warp's reduction degrades. Batching is
expressed to Warp as ``batch_offsets``, and ``batch_offsets`` is exactly what makes its ``TiledDot``
take the *direct batched* path: one block per (column, subproblem), each lane reducing
``n / tile_size`` entries serially, so a dot that is flat in ``n`` for a single column becomes
``O(n)`` for a batched one (measured 4.55 / 9.51 / 18.66 / **66.15** us at n = 4 356 / 17 161 /
40 962 / 163 842, against 5.09-6.06 us unbatched). Two dots per iteration made that 19 us of a
41 us iteration on ``harmonic``'s ``saddle`` system. Reducing per column with a real two-stage tree
keeps the batching and drops the cost: **1.10-1.50x end to end** across ``harmonic`` and
``smooth_region_fixed_rim``, with the answers unchanged. See ``_BatchedCg`` for the full
measurement, including the two things that look like the same idea and are losses.

**Determinism.** Build each operator natively at its final dtype in a *single*
``warp.sparse.bsr_from_triplets`` and never recast or rebuild it. This rule was written when a
rebuild appeared to make ``bsr_mm`` nondeterministic; the real cause was sizing the rebuild's
triplet buffers by ``BsrMatrix.nnz``, which is the *capacity* the matrix was built with rather than
its entry count (``nnz_sync()``), so the buffers' tail reached ``bsr_from_triplets`` uninitialized.
A rebuild sliced to ``nnz_sync()`` is safe, and ``bsr_mm`` itself was never at fault. The rule
survives on cost instead: a second build re-sorts and duplicate-accumulates an order the CSR has,
20.7 ms in [`assemble_interior_system`][triwarp.linalg.assemble_interior_system]'s ``saddle`` case.
Nothing here recasts an operator, and ``Q_uu`` is assembled as a CSR *directly*, without any triplet
build, so it carries an exact ``nnz``.

**Why Jacobi.** Every solve here preconditions with ``warp.optim.linear.preconditioner(A, "diag")``,
and the alternatives were measured and rejected rather than overlooked. On a cotangent Laplacian a
preconditioner costing ``k`` mat-vecs per iteration cuts the iteration count by only about
``sqrt(k)``, so total work scales as ``k / sqrt(k) = sqrt(k)`` — single-level preconditioning loses
on this operator class, and only a multilevel method escapes it. IC(0) is the instructive case: its
quality is real (2.1x to 4.2x fewer iterations under an exact apply), but Warp has no sparse
triangular solve through Warp 1.16 — ``warp.sparse`` exposes only ``bsr_from_triplets`` and
``warp.optim.linear`` only the Krylov methods and a diagonal preconditioner — and no substitute for
one keeps the win. Scored in mat-vec equivalents against
Jacobi at ``tol=1e-8``, on ``-L`` with one degree of freedom pinned:

- a fully parallel apply (``k`` Jacobi sweeps per triangular solve) runs **0.70x to 1.03x**;
- an exact apply parallelized by graph coloring runs **1.02x to 1.26x**, and even that is
  optimistic — it prices only ``nnz`` traffic, ignoring the twelve dependent launches per iteration
  at ``n / 6`` occupancy and the uncoalesced access the color permutation causes;
- natural-ordering level sets are not viable at all: 8 levels on an icosphere against 157 on a
  torus, so throughput would swing with the input's vertex numbering;
- Chebyshev — pure mat-vecs, trivially graph-capturable, no factorization — runs **0.60x to 0.92x**
  at degrees 2, 4 and 8.

Two further obstacles are specific to this repository. Obtuse triangles give negative cotangent
weights (``triwarp/kernels/laplacian.py``), so a noisy sphere carries 16.75 % positive
off-diagonals and ``-L`` is not the M-matrix that IC(0) existence requires; and both
[`heat_geodesic`][triwarp.heat.distance.heat_geodesic] and
[`heat_signed_distance`][triwarp.heat.signed.heat_signed_distance] solve a ``-L`` with a genuine
constant null space, where IC(0) hits a zero pivot on the last row of every connected component.
**And the multilevel option is now here**, as
[`multigrid_preconditioner`][triwarp.linalg.multigrid_preconditioner]: smoothed aggregation, the one
scheme that breaks the ``O(sqrt(n))`` growth rather than paying it down by a constant. On the
least-squares operator [`smooth_region`][triwarp.smoothing.smooth_region] builds -- the
worst-conditioned system this package solves, 6 541 Jacobi iterations at 8 987 unknowns, and 99 % of
a 224 ms call -- the *solve* measures **2.46x on ``bunny``** and 2.55x on the CPU device, at a
**12.5x** reduction in iterations.

It is not the default, and the reason is the *setup*, not the cycle. Building the hierarchy is one
aggregation, one power iteration, a ``bsr_transposed`` and three ``bsr_mm`` per level, and at these
sizes almost every one of those is Warp's fixed per-call cost rather than work: **12-17 ms**, near
flat in the operator. So the V-cycle wins exactly where the solve it replaces is longer than that,
and everywhere else it loses by the setup: ``smooth_region_fixed_rim``, whose graph-Laplacian
Dirichlet system converges in a tenth the iterations, runs **0.48-0.53x**, and ``harmonic`` at
``k=1`` runs **0.43-1.13x**. Cutting that setup is the lever that would make it unconditional, and
the term to cut is ``bsr_mm``'s ~0.84 ms per call.

**Nothing cheap predicts which side of that line a system falls on**, which is why the third mode is
a capped probe and not a heuristic; see
[`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS] for the two predictors that were built
and refuted (size, and extrapolating the probe's own convergence rate).

That lead about anisotropy is now measured rather than open, and the answer split in two. The
aggregation used to keep *every* off-diagonal, which on a graded patch means aggregating across the
weak direction; the strength-of-connection threshold ``|A_ij| >= theta sqrt(A_ii A_jj)`` is now in
(``_MULTIGRID_THETA``, and its comment carries the sweep). It **is** the mechanism the diagnosis
predicted -- on the graded saddle's harmonic system it takes the V-cycle from 349 iterations to
**58** at ``theta = 0.02``, and on the graded ``k=2`` system from *not converging in 20 000* to
7 760 -- but it buys **1.05x** end to end, because the only call site that reaches the hierarchy
today is ``smooth_region``.

**The rest of that family never sees a preconditioner choice at all**, which is the finding worth
carrying forward: [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed] takes the default
``preconditioner="diag"``, so ``harmonic`` / ``tutte`` / ``lscm`` run Jacobi whatever the operator's
conditioning and no threshold can reach them.

**Routing them through ``"auto"`` was measured and is refuted.** End to end on the whole
``harmonic`` call, five reps interleaved, medians:

| system | ``"diag"`` | ``"auto"`` | |
|---|---|---|---|
| ``k=2`` saddle, 17 161 unknowns | 199.1 ms | 119.6 ms | **1.66x** |
| ``k=1`` saddle_small, 4 356 | 8.0 | 7.2 | 1.11x |
| ``k=2`` saddle_small | 53.4 | 53.4 | 1.00x, converges inside the probe |
| ``k=1`` saddle | 11.3 | 11.4 | 0.99x |
| ``k=1`` saddle_graded | 58.5 | 78.9 | **0.74x** |

The graded ``k=1`` row is the refutation: it converges under *both* preconditioners, and ``"auto"``
still loses, because the probe spends its 2 000 Jacobi iterations and then pays the hierarchy setup
on top. One 1.66x win does not buy a 0.74x loss on the row the threshold was supposed to be for, and
nothing cheap separates them -- which is ``CG_PROBE_ITERATIONS``' own conclusion reappearing one
level up.

!!! warning "``harmonic`` at ``k=2`` on a graded patch does not converge, and the fast number is the
    non-answer"
    That sweep's sixth cell reads as a catastrophic 0.12x (4.19 s against 33.8 s) and is the
    opposite. Instrumented on ``saddle_graded`` at ``k=2``, 17 161 unknowns: **``"diag"`` runs
    171 610 iterations, hits its cap and returns a residual of 9.74e+03 against an absolute
    tolerance of 3.72e-01** -- four orders of magnitude out, with nothing but a ``UserWarning`` to
    say so -- while ``"auto"`` converges in 83 408 iterations at 3.707e-01. The two UV maps
    differ by **0.71** on a unit disk. So the pair is not a solver comparison at all: it prices
    a failure against a solve, and the ``k=2`` biharmonic operator on a strongly graded patch
    is outside what Jacobi-preconditioned conjugate gradient reaches in ``float64``. A caller
    who needs that combination should pass a stronger preconditioner and check the warning.

The *target* was sound even though the tool is not: on an RTX 5090
[`heat_geodesic`][triwarp.heat.distance.heat_geodesic] with cached operators measures 8.1, 12.6 and
30.3 ms at 10 242, 40 962 and 163 842 vertices, of which the heat solve is about 2 ms flat and
assembly 1.1 to 2.3 ms — the Poisson solve is 75-90 % of the call. That heat system ``M - tL``
needs no help of its own: 30 iterations at every size, because ``t = h**2`` makes it a small
perturbation of the mass matrix.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar
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

# Cadence substituted for ``check_every=0`` on a device without conditional CUDA graphs, where Warp
# cannot test the residual on device and would otherwise run every solve to ``maxiter``. Warp's own
# default, and what this module shipped before the device-side check became the default.
CG_CHECK_EVERY_FALLBACK = 10

# Jacobi iterations ``preconditioner="auto"`` runs before it escalates to a multigrid hierarchy.
#
# It is a *cap*, not a predictor, and that is a measured retreat rather than a first choice. Nothing
# cheap tells the two cases apart. Size does not: at ~2 000 free unknowns ``smooth_region`` on
# ``bunny_decimated`` takes 1 784 Jacobi iterations and a V-cycle wins 1.88x, while the same
# free-set size on an ``icosphere`` takes 521 and loses 0.69x. Extrapolating the probe's own
# convergence rate does not either -- tried at probe lengths 200, 400 and 600, the estimated
# remaining count of the *losing* systems (2 993 / 4 307 / 6 481) interleaves with the winning
# ones' (2 410 / 4 336 / 5 795) at every length, because what decides the ratio is the V-cycle's own
# iteration count and that is not knowable without building the hierarchy.
#
# So the rule is the conservative one: a system that converges inside the cap never pays for a
# hierarchy and runs at exactly Jacobi's speed. What that costs is the upside on every system that
# converges *just* inside it, and that is a large number rather than a rounding error. The clearest
# case is ``smooth_region`` on ``bunny_decimated``, the 1 784-iteration row in the table below:
# it converges under 2 000, therefore never escalates, and therefore forgoes the 2.29x a V-cycle
# measures on it. That is deliberate rather than an oversight -- the table says what moving the cap
# would cost instead.
#
# **Re-measured on the whole population, and the conclusion holds.** ``smooth_region`` is this
# package's *only* ``"auto"`` caller, in two shapes -- a mesh region and a hole patch -- so 17
# systems spanning both, plus ``fill_smooth`` on three hole meshes, is the whole table rather than a
# sample of it. Whole solve including setup, interleaved, ``min`` of 5, RTX 5090:
#
#     system                n   jac it    cap 2000    1000     500     300     150
#     region icos4 q25     641     151        6.38    6.23    6.25    6.26   17.93
#     region icos5 q25   2 561     521       18.46   18.46   37.23   35.69   31.33
#     region icos5 q50   5 057     997       34.93   34.47   61.70   55.11   50.69
#     region icos6 q25  10 239   1 946       68.54   87.39   72.14   65.48   61.04
#     region icos6 q50  20 353   3 827      170.73  132.32  114.34  108.61  103.69
#     region bunny_dec   2 043   1 784       65.32   63.54   48.04   40.52   36.24
#     region bunny       8 987   6 541      135.61  107.30   93.20   86.82   82.45
#     patch bunny_dec      787     425       15.95   22.38   35.45   31.49   27.27
#     patch bunny 4000   3 970   3 145      131.36   83.44   67.24   60.68   56.06
#     fill_smooth[rim_short]                 73.92   74.26   74.48   93.45   87.10
#     fill_smooth[holes_many]                19.84   20.76   19.54   19.06   19.76
#     TOTAL (all 17)                         848.1   739.0   716.6   653.0   634.5
#
# Three things that table says, none of them guessable:
#
# **Escalate early or not at all.** A mid-range cap is worse than both ends, because it pays the
# probe *and* the hierarchy: ``icos6 q25`` reads 68.54 at 2 000, **87.39** at 1 000 and 61.04 at
# 150. So 1 000 or 1 500 is never the answer -- it is the worst of both.
#
# **Nothing separates the two classes, on any axis tried.** Iteration count interleaves (a
# 244-iteration system wins 1.51x while a 997-iteration one loses 0.79x) and so does Jacobi
# wall-clock (wins from 2.7 ms, losses up to 35.0). A cap *proportional to* ``n`` is refuted
# outright and backwards: a larger system gets a larger cap and therefore escalates **later**,
# taking ``region bunny`` from 136.8 ms to 216.9 at ``alpha = 0.5``.
#
# **The escalated solve does not really reuse the probe's work.** ``region bunny`` measures 135.61
# escalating at 2 000 against 67.29 for multigrid from the start, and 2 000 Jacobi iterations cost
# ~64 ms -- so the probe's iterates carry over close to nothing and its cost is very nearly
# additive. The warm start is real (``solution`` is threaded through) but Jacobi leaves a residual
# a V-cycle has to work down anyway.
#
# 2 000 is kept because it is the only value that regresses **nothing**, which is the property this
# setting was introduced to have. Lowering it to 150 is the best *total* (848 -> 634 ms, 1.34x) and
# would take ``smooth_region[bunny_decimated]`` from 5.5x behind to roughly 3x -- at the price of
# up to 2.8x on small well-conditioned solves, each bounded by one hierarchy setup (+4 to +16 ms).
# That is a policy choice and not a measurement question; the numbers for it are above.
#
# One thing the table retires: the 0.21x ``fill_smooth[holes_many]`` disaster that motivated the cap
# is **not** a cap-value problem. That mesh is flat at 19.0-20.8 ms for every cap from 100 to 2 000
# and only collapses (77.4 ms, 0.26x) under *unconditional* multigrid, because its 3-vertex patches
# converge in far fewer iterations than any cap considered. The row that actually constrains a low
# cap is ``fill_smooth[rim_short]``, whose two 512-vertex rims put it just past 500.
CG_PROBE_ITERATIONS = 2000


def min_quad_with_fixed(
    q: wps.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: twt.Array2dFloat,
    *,
    tol: float = CG_TOLERANCE,
    check_every: int = CG_CHECK_EVERY,
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
        iteration; see there for the measured tradeoff. This entry point's own return type does not
        depend on it — the solver's ``(iterations, residual, atol)`` triple is not surfaced here.

    Returns
    -------
    solution : twt.Array2dFloat
        ``(n_rhs, n_free)`` solved values for the unpinned degrees of freedom.
    free_map : wp.array[wp.int32]
        Length-``n_dofs`` compact remap of the unpinned degrees of freedom.
    n_free : int
        Number of unpinned degrees of freedom.

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
    device = fixed_mask.device
    n_rhs = int(fixed_values.shape[0])
    free_map, n_free = free_partition(fixed_mask)

    solution = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    if n_free == 0:
        return twt.as_array2d(solution, wp.float64), free_map, n_free

    q_uu, rhs = assemble_interior_system(q, fixed_mask, free_map, fixed_values, n_free)
    solve_spd_columns(
        q_uu, rhs, twt.as_array2d(solution, wp.float64), tol=tol, check_every=check_every
    )
    return twt.as_array2d(solution, wp.float64), free_map, n_free


def free_partition(fixed_mask: wp.array[wp.bool]) -> tuple[wp.array[wp.int32], int]:
    """
    Compact remap of the *unpinned* degrees of freedom, plus their count.

    Thin inversion of [`mask_to_index_map`][triwarp.array.mask_to_index_map]: ``fixed_mask`` marks
    the constrained degrees of freedom, and the returned map indexes the reduced system built over
    the complement.

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
    [`mask_to_index_map`][triwarp.array.mask_to_index_map]
    """
    return tw.array.mask_to_index_map(fixed_mask, invert=True)


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
        already had -- measured at **20.7 ms in a single launch, 91 % of all device time** in
        ``min_quad_with_fixed`` on ``benchmarks/test_linalg.py``'s ``saddle`` case, against 1.65 ms
        for the entire CG solve it feeds. Row order comes from the launch index and column order
        from ``q``'s own rows through the monotone ``free_map``, so the extracted matrix is sorted
        by construction.

        It also fixes a second, quieter cost: ``bsr_from_triplets`` leaves ``nnz`` at the *triplet*
        count, which is ``q.nnz`` — an upper bound measured at 3.5x the true entry count of a
        lightly-pinned ``Q_uu`` — so every downstream ``bsr_mv`` was dimensioned for the unreduced
        matrix. The count here is exact.

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

    See Also
    --------
    [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed]
    """
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
    # The total-terminated form *is* the CSR offsets array, and the one host read it costs
    # (~0.1 ms) is what sizes ``columns`` / ``values`` for their final use at allocation time.
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
    matrix: wps.BsrMatrix,
    rhs: wp.array,
    solution: wp.array,
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
        the short solves that call this (tens of iterations) measured slower with it. Pass
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
    n_rows = int(rhs.shape[0])
    iteration_cap = CG_MAXITER_FACTOR * n_rows if maxiter is None else maxiter
    result = wpl.cg(
        matrix,
        rhs,
        solution,
        tol=tol,
        maxiter=iteration_cap,
        M=wpl.preconditioner(matrix, "diag") if preconditioner is None else preconditioner,
        check_every=_supported_check_every(check_every),
    )
    _warn_if_not_converged(result, iteration_cap, name)
    return result


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
        on device; see Notes for the measurements and the warning above for what it does to the
        return type.
    preconditioner
        ``"diag"`` (the default) for the Jacobi preconditioner, ``"multigrid"`` for the
        smoothed-aggregation V-cycle [`multigrid_preconditioner`]
        [triwarp.linalg.multigrid_preconditioner] builds, or ``"auto"`` to run Jacobi under
        [`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS] and escalate to the V-cycle only
        if that has not converged. The V-cycle costs a setup pass and pays for itself only where the
        solve dominates the call, so ``"auto"`` is the setting for a caller whose systems vary --
        it cannot regress a solve that was already short, and gives up the upside on one that is
        only just long enough. See that function's Notes and ``CG_PROBE_ITERATIONS``.

    Returns
    -------
    tuple[int, float, float]
        ``(iterations, residual_norm, absolute_tolerance)`` on ``warp.optim.linear.cg``'s terms,
        with the residual taken over the worst column. Device arrays rather than host scalars under
        ``check_every=0``; see the warning above. A one-column solve returns Warp's own values and
        a batched one returns the same three from this module's solver.

    Warns
    -----
    UserWarning
        When the solve exhausts ``maxiter`` without reaching ``tol``, on the same terms as
        [`solve_spd`][triwarp.linalg.solve_spd]. A batched solve converges on its *worst* column,
        so hitting the cap here means at least one column is unsolved. Only detectable when
        ``check_every > 0``.

    Notes
    -----
    ``check_every`` is a pure performance knob — it cannot change the converged answer, only how far
    past the tolerance the solver may overshoot before it notices. Two regimes have been measured on
    an RTX 5090, and they disagree, so read the one that matches the caller:

    - **Cold single solves** — one ``cg`` call from a zero initial guess, the shape
      ``harmonic`` / ``tutte`` / ``smooth_region`` take. Measured on
      `benchmarks/test_linalg.py`'s ``solve_spd_columns`` group, ``0`` against ``10``: 22.8 vs
      31.7 ms well-conditioned and 104.7 vs 154.1 ms ill-conditioned, **28-32 % faster on both**.
      This is the regime the default is set for.
    - **Warm-started solves inside an iteration loop** — ``arap``, whose right-hand side changes
      every iteration so each solve still runs tens of CG iterations from the previous answer.
      Measured on `benchmarks/test_parametrization.py`, ``0`` is **neutral to positive** (0.90x to
      1.00x): fewer readbacks, and enough iterations for them to matter a little.
    - **Repeated near-converged solves over one [`spd_column_solver`]
      [triwarp.linalg.spd_column_solver] state** — the same right-hand side re-solved back to back,
      so every call after the first converges in one or two iterations. Here ``0`` is a **2x loss**:
      49.6 ms against 24.5 ms for 50 calls on a 2 562-vertex system. The conditional-graph loop
      costs about **0.5 ms per call** regardless of iteration count, which a solve that short cannot
      recover. Pass a positive ``check_every`` to a state driven that way; a single cold call
      through the same state is still 1.2x *faster* at ``0``.
    - **Raising it (25, 50) is a loss** of 0 % to 6 % in every regime. The readback it saves costs
      about 0.1 ms, while the up-to-``check_every - 1`` extra iterations it causes are real work —
      the smaller the solve, the worse the trade.

    So the device-side check scales with how much work a single ``cg`` call does: a large win on
    long solves, a wash on medium ones, and a loss only once the solve is shorter than the
    graph-launch overhead. The earlier "opt-in rather than the default" note recorded only the
    middle regime.

    The *preconditioner* has been measured on the same systems and is not a knob worth turning:
    IC(0) and Chebyshev both come out a wash or a loss against the ``"diag"`` Jacobi used here. See
    "Why Jacobi" in the [`triwarp.linalg`][triwarp.linalg] module documentation for the numbers and
    for the one direction that would pay off.

    The *per-iteration* cost was a separate lever and has been taken: with more than one column this
    runs triwarp's own conjugate gradient rather than ``warp.optim.linear``'s, because Warp's
    reduction degrades on precisely the batched input the worst-case stopping rule needs. Worth
    **1.10-1.50x** end to end. See "Whose conjugate gradient" in the
    [`triwarp.linalg`][triwarp.linalg] module documentation.

    See Also
    --------
    [`spd_column_solver`][triwarp.linalg.spd_column_solver]
    [`replicated_operator`][triwarp.linalg.replicated_operator]
    """
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
) -> wpl.LinearSolverState:
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
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] for the measured tradeoff and for
        what the default ``0`` does to the values each call returns.
    preconditioner
        ``"diag"`` or ``"multigrid"``, as in
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns]. Built once here and reused by every
        call against this state, which is the shape the V-cycle's setup cost wants. ``"auto"`` is
        **not** accepted: its probe decides on the first solve, and a hoisted state exists to be
        driven many times.

    Returns
    -------
    ``warp.optim.linear.LinearSolverState``
        Callable solver state. Substituted operands must match the construction-time shape, dtype,
        device and batch layout. Each call returns what
        [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] returns, including its
        ``check_every=0`` device arrays.

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
):
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
                f'{caller} cannot take preconditioner="auto": the probe decides on the *first* '
                'solve, and a hoisted state is built to be driven many times. Pass "diag" or '
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
    if n_columns > 1:
        # ``_BatchedCg`` exists only for this branch; a single column already reaches
        # ``warp.optim.linear``'s fast reduction, because there is nothing to batch. See that
        # class's Notes for the measurement.
        state = _BatchedCg(
            matrix,
            rhs,
            solution,
            tol=tol,
            maxiter=iteration_cap,
            check_every=_supported_check_every(check_every),
            preconditioner=preconditioner,
        )
        return state() if run else state
    operator = replicated_operator(matrix, n_columns)
    if preconditioner == "multigrid":
        apply_inverse = multigrid_preconditioner(matrix, n_columns)
    else:
        apply_inverse = replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    return wpl.cg(
        operator,
        rhs.flatten(),
        solution.flatten(),
        tol=tol,
        maxiter=iteration_cap,
        M=apply_inverse,
        check_every=_supported_check_every(check_every),
        run=run,
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
):
    """
    Run Jacobi under ``CG_PROBE_ITERATIONS``, then escalate if it has not converged.

    A system that finishes inside the probe pays nothing at all for the option -- the probe *is* the
    solve. One that does not is the ill-conditioned kind
    [`multigrid_preconditioner`][triwarp.linalg.multigrid_preconditioner] is for.

    The escalated solve does warm-start from the iterate the probe left in ``solution``, but **do
    not read that as the probe being cheap**: measured on ``smooth_region``'s ``bunny`` system,
    escalating at 2 000 costs 135.6 ms against 67.3 for a V-cycle from the start, and 2 000 Jacobi
    iterations are ~64 ms of that -- so the probe is very nearly additive and its iterates carry
    over close to nothing. Jacobi leaves a residual whose low-frequency part is exactly what the
    V-cycle then has to work down.

    See [`CG_PROBE_ITERATIONS`][triwarp.linalg.CG_PROBE_ITERATIONS] for why this is a cap rather
    than the rate prediction it started out as.
    """
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


def _cg_residual_and_tolerance(result: tuple) -> tuple[float, float]:
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
    rule, and that is precisely the input for which Warp's ``TiledDot`` selects its **direct
    batched** reduction: one block per (column, subproblem), every lane reducing ``n / tile_size``
    entries serially. Its cost is therefore *O(n)* where the tiled tree it uses for an unbatched
    vector is flat -- measured 4.55 / 9.51 / 18.66 / **66.15** us at n = 4 356 / 17 161 / 40 962 /
    163 842 against 5.09-6.06 us -- and a CG iteration runs two of them.

    Reducing per column with a real two-stage tree keeps the batching and drops that cost. Measured
    back to back against ``warp.optim.linear.cg`` on ``harmonic``'s own interior system, both
    graph-captured: **40.9 -> 28.2 us per iteration at ``saddle`` k=2 (1.45x)**, 37.2 -> 24.6 at
    k=1 (1.51x), 30.6 -> 27.8 on ``saddle_small`` (1.10x, where the vector is short enough that
    Warp's one block per column is not yet starved). The converged answers agree to **1.8e-10** and
    this takes slightly *fewer* iterations (7 453 against 7 460).

    Two fusions ride along and are free. The Jacobi apply is an elementwise multiply of the ``r``
    that ``cg_step_x_r_z`` has just written, so it happens in a register rather than in its own
    launch; and the ``rz_old = rz_new`` copy folds into the ``p.Ap`` finalize, which is the one
    point in the iteration after the ``p`` update that last read ``rz_old`` and before the x/r
    update that reads it next.

    **What was tried and is not here.** Solving the columns as separate unbatched ``cg`` calls also
    reaches the tree reduction, and is a **0.60-0.81x loss**: it pays a second copy of every other
    kernel in the iteration. And dropping ``batch_offsets`` to get the tree from a single call is
    not a tuning change at all -- ``alpha`` and ``beta`` would then be global rather than per
    column, which is CG on the block system and a different iteration.

    The first prototype of this class was a **0.58-0.77x loss** with the reduction folding 8 tiles
    per block: that left 18 blocks on a 170-SM device and paid a ``wp.tile_sum`` per tile, and the
    dot measured 18.2 us where Warp's was 9.5. One tile per block -- 68 blocks per column at
    ``saddle`` -- took the same dot to 3.2 us. The lesson is the recorded one: price the launches
    individually, and capture both arms.
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
        self._p_dot_ap = wp.zeros((2, self._n_columns), dtype=wp.float64, device=device)
        self._rz_old = wp.zeros(self._n_columns, dtype=wp.float64, device=device)
        self._atol_sq = wp.zeros(self._n_columns, dtype=wp.float64, device=device)
        # [iterations, loop condition]; the second element is what ``wp.capture_while`` watches.
        self._state = wp.zeros(2, dtype=wp.int32, device=device)

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
        wp.launch(
            kernel_cg.cg_inverse_diagonal,
            dim=self._n,
            inputs=[wps.bsr_get_diag(matrix)],
            outputs=[self._inv_diag],
            device=device,
        )
        # Per-column views, built once: the matvec is ``n_columns`` ``bsr_mv`` calls against the one
        # operator, and re-slicing them per iteration would add Python to every CG step and keep the
        # loop from being captured. They span ``n``, not ``stride``, so nothing writes the pad.
        self._p_blocks = self._column_views(self._p)
        self._r_blocks = self._column_views(self._r)
        self._ap_blocks = self._column_views(self._ap)

    def _column_views(self, flat: wp.array[wp.float64]) -> list[wp.array[wp.float64]]:
        """Split a padded flat vector into its ``n_columns`` blocks of ``n`` live entries."""
        return [flat[c * self._stride : c * self._stride + self._n] for c in range(self._n_columns)]

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
        tile = int(kernel_cg.CG_TILE)
        wp.launch_tiled(
            kernel_cg.cg_dot_partials,
            dim=(self._n_columns, self._blocks),
            inputs=[a, b0, b1, wp.int32(self._stride), wp.int32(pairs)],
            outputs=[self._partials],
            block_dim=tile,
            device=self._device,
        )
        wp.launch_tiled(
            kernel_cg.cg_dot_finalize,
            dim=(2, self._n_columns),
            inputs=[
                self._partials,
                wp.int32(self._blocks),
                wp.int32(pairs),
                wp.int32(1 if carry else 0),
                self._dots[1],
            ],
            outputs=[out_dots, self._rz_old],
            block_dim=tile,
            device=self._device,
        )

    def _iteration(self) -> None:
        """One CG step: 4 + ``n_columns`` launches, none of which reads back to the host."""
        for column in range(self._n_columns):
            wps.bsr_mv(
                self._matrix, self._p_blocks[column], self._ap_blocks[column], alpha=1.0, beta=0.0
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
            wp.launch(
                kernel_cg.cg_step_x_r_z,
                dim=self._dofs,
                inputs=[*step, self._inv_diag, self._p, self._ap],
                outputs=[self._solution_flat, self._r, self._z],
                device=self._device,
            )
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
        wp.launch(
            kernel_cg.cg_step_p,
            dim=self._dofs,
            inputs=[wp.int32(self._stride), self._rz_old, self._dots, self._atol_sq, self._z],
            outputs=[self._p],
            device=self._device,
        )
        wp.launch(
            kernel_cg.cg_advance_condition,
            dim=1,
            inputs=[wp.int32(self._maxiter), wp.int32(self._n_columns), self._dots, self._atol_sq],
            outputs=[self._state],
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
        # ``r = b - A x`` in place, warm-starting from whatever ``solution`` currently holds.
        for column in range(self._n_columns):
            wps.bsr_mv(
                self._matrix, self._solution[column], self._r_blocks[column], alpha=-1.0, beta=1.0
            )
        if self._cycle is None:
            wp.launch(
                kernel_cg.scaled_diagonal_apply,
                dim=self._dofs,
                inputs=[
                    # ``n == stride``: this state's vectors carry no rows the kernel must skip.
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
        wp.copy(self._rz_old, self._dots[1])
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
                math.sqrt(float(self._dots.numpy()[0].max())),
                math.sqrt(float(self._atol_sq.numpy().max())),
            )
        condition = self._state[1:2]
        with wp.ScopedCapture(self._device) as capture:
            wp.capture_while(condition, self._iteration)
        wp.capture_launch(capture.graph)
        return self._state[0:1], self._dots[0], self._atol_sq

    def _run_with_host_checks(self, check_every: int) -> None:
        """
        Drive the loop from the host: issue a block of iterations, then read the residual.

        The block is trimmed against ``maxiter`` so the cap is exact rather than rounded up to the
        next multiple of the cadence -- a caller that reads the returned iteration count against
        the cap it passed is how a non-convergence warning gets raised.
        """
        done = 0
        while done < self._maxiter:
            block = min(check_every, self._maxiter - done)
            for _ in range(block):
                self._iteration()
            done += block
            if bool((self._dots.numpy()[0] <= self._atol_sq.numpy()).all()):
                return


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
    block_views: dict[tuple[int, tuple[int, ...], tuple[int, ...]], list[wp.array]] = {}

    def blocks(array: wp.array) -> list[wp.array]:
        key = (array.ptr, tuple(array.shape), tuple(array.strides))
        views = block_views.get(key)
        if views is None:
            views = [array[column * n : (column + 1) * n] for column in range(n_columns)]
            block_views[key] = views
        return views

    def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float) -> None:
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
    "Why Jacobi" in the [`triwarp.linalg`][triwarp.linalg] module documentation for the four that
    were measured and rejected -- because cutting the iteration count by ``sqrt(k)`` at ``k``
    mat-vecs per apply leaves the total work growing. A multigrid V-cycle attacks the low-frequency
    error the smoother cannot see, so the count stops growing with ``n``: on the least-squares
    operator [`smooth_region`][triwarp.smoothing.smooth_region] builds, Jacobi-preconditioned
    conjugate gradient takes 1 753 iterations at 2 043 unknowns and 6 521 at 8 987, and this takes
    **288 and 554** -- 6.1x and 11.8x fewer, and the *growth* falls from 3.7x to 1.9x over the same
    4.4x in size.

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
    the operator smaller.** Measured on ``smooth_region``'s systems: 83-84 % host across 284-468
    launches, and a level costs the same wherever it sits in the hierarchy -- 5.70 ms at n = 8 987
    against 5.71 ms at n = 879. The reason is underneath triwarp: ``warp.sparse.bsr_mm`` measures
    **~0.6 ms at every size probed**, n = 128 through n = 65 536 (0.573 / 0.603 / 0.614 / 0.696 ms),
    at 88-91 % host in only 12 launches, because it makes three device-to-host readbacks to size its
    output. That is Warp's, not this package's, so "make the hierarchy cheap enough to run
    unconditionally" is not a lever available here; the reachable version is to build **fewer
    levels**, which is what this module's ``_MULTIGRID_MAX_COARSE`` is set for.

    Coarsening stops at 128 rows, or earlier if a level fails to shrink; the coarsest operator is
    then inverted densely on the host, which is exact and is a single launch inside the cycle where
    an iterative coarse solve would be a data-dependent loop. When coarsening stalls while the level
    is still too large to factor, there is no usable hierarchy and this hands back
    ``warp.optim.linear.preconditioner(matrix, "diag")`` rather than a cycle whose coarse solve is a
    guess -- so a caller never has to branch on the operator's shape.

    !!! warning "That fallback is silent, and the strength threshold can trigger it"
        A stalled hierarchy is indistinguishable from a weak one at the call site: the solve simply
        runs at its Jacobi iteration count. ``_MULTIGRID_THETA`` is what decides how easily it
        happens -- raising it makes more off-diagonals weak, and measured at ``0.25`` several of the
        harmonic operators stop coarsening entirely and come back at *exactly* the Jacobi count. So
        an aggregation change that "did nothing" should be checked against the level count before it
        is read as a change that did not help.

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

    def matvec(x: wp.array, y: wp.array, z: wp.array, alpha: float, beta: float) -> None:
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
# **384 rather than a smaller cap because the last level is the expensive one twice over**: it costs
# a level's setup (one aggregation, one power iteration, a prolongator, a transpose and two
# ``bsr_mm``, all of which are fixed host cost -- ``bsr_mm`` measures ~0.6 ms at *every* size from
# n=128 to n=65 536, 88-91 % host) and then a smoothing sweep, a restriction and a prolongation in
# **every cycle**. Solving that level densely instead is one matvec. Swept over every system that
# reaches a hierarchy -- ``smooth_region`` is the package's only ``"auto"`` caller, in its two
# shapes, a mesh region and a hole patch -- interleaved, ``min`` of 5, whole solve including setup:
#
#     system                  n    128 rows    384 rows          levels 128 -> 384
#     region[bunny]        8 987    84.94 ms    73.98  1.15x    [8987,879,360,306] -> [8987,879,360]
#     region[bunny_dec]    2 043    36.49       28.75  1.27x    [2043,158,27]      -> [2043,158]
#     region[icosphere5]   2 561    29.56       24.68  1.20x    [2561,205,27]      -> [2561,205]
#     region[icosphere4]     641    15.61       15.34  1.02x    [641,59]           -> [641,59]
#     patch[bunny]           994    31.48       22.14  1.42x    [994,165,66]       -> [994,165]
#
# Every system improves or is flat and none regresses, and the gain is mostly in the *solve* rather
# than the setup -- ``bunny_decimated`` moves 10.86 -> 10.35 ms of setup against 25.6 -> 18.4 of
# solve -- which is the per-cycle half above. Tightening ``_MULTIGRID_MIN_COARSENING`` instead was
# measured and is strictly weaker: it drops ``bunny``'s 360 -> 306 level (that one barely shrinks)
# and nothing else, because 158 -> 27 passes any stall test while still costing a cycle.
#
# Values from 384 to 512 measure identically here; 384 keeps headroom under
# ``_MULTIGRID_MAX_DENSE`` and the host ``pinv`` that guard sizes, which is ~4.9 ms at 587 rows.
_MULTIGRID_MAX_COARSE = 384

# Hard cap on the hierarchy depth, and on the dense coarse solve. Coarsening stops early whenever a
# level fails to shrink by ``1 - _MULTIGRID_MIN_COARSENING``, which is what happens once a level is
# mostly isolated rows -- the least-squares operators here carry them (297 of 8 987 on ``bunny``).
_MULTIGRID_MAX_LEVELS = 12
_MULTIGRID_MIN_COARSENING = 0.9
_MULTIGRID_MAX_DENSE = 512

# Damped-Jacobi sweeps per level per half-cycle, and the damping as a multiple of ``1 / rho`` where
# ``rho`` is the spectral radius of ``D^-1 A``. 4/3 is the classical smoothed-aggregation choice and
# is used both for the smoother and for the prolongation smoother.
#
# Two sweeps, measured on ``smooth_region``'s batched three-column solve at 8 987 unknowns: 1 / 2 /
# 3 / 4 sweeps take 735 / 524 / 445 / 400 iterations and 104.0 / 90.8 / 90.6 / 93.1 ms, so the curve
# is flat from 2 to 3 and turns at 4. One is the cheapest cycle and not the cheapest solve.
_MULTIGRID_SWEEPS = 2
_MULTIGRID_JACOBI_FACTOR = 4.0 / 3.0

# Power iterations for that spectral radius, and the round cap for the aggregation's independent
# set. The power iteration is unnormalized -- ``rho`` is recovered from the growth over all the
# steps -- so it costs one mat-vec and one elementwise pass per step, and two inner products.
# Eight rather than fifteen: the extra seven steps move the iteration count by under 1 % and cost
# 1.5-2 ms of setup, which at these sizes is 2 % of the whole solve.
_MULTIGRID_POWER_STEPS = 8
_MULTIGRID_MIS_ROUNDS = 32

# Strength-of-connection threshold for the aggregation: an off-diagonal counts as an edge only when
# ``|A_ij| >= theta sqrt(A_ii A_jj)``. ``0.0`` keeps every off-diagonal, which is the aggregation
# this package shipped first and is bit-exactly what a zero threshold reduces to.
#
# ``0.05`` is the measured minimum on the one row that reaches the hierarchy today,
# ``smoothing.smooth_region`` on ``bunny`` (8 987 unknowns, 163 588 nnz). Swept on that system's
# solve alone, interleaved, five reps, RTX 5090:
#
#     theta  levels  coarse n  iterations   min ms   median ms
#     0.0         3       310         524    91.10       91.51
#     0.02        4       302         400    89.86       93.41
#     0.05        4       306         344    81.77       82.71
#     0.08        4       320         299    82.71       84.09
#     0.15        5       323         233    94.62       95.85
#     0.25        6       400         177   191.80      202.95
#
# The iteration count falls monotonically and the *clock* is a U: every extra level adds setup and
# makes a cycle more expensive, so the two cross at 0.05-0.08. ``0.08`` is statistically tied on
# time with 15 % fewer iterations, which is the value to try first if a caller ever puts a harder
# operator on the hierarchy -- iterations are the quantity that transfers, the clock is not.
#
# Two things to know before moving it. **A large threshold stalls the coarsening silently**: at
# ``0.25`` several operators leave ``_multigrid_hierarchy`` with nothing usable and
# ``multigrid_preconditioner`` hands back Jacobi, which reads as "multigrid did not help" rather
# than as "there was no multigrid". And ``0.0`` keeps every off-diagonal, which is the aggregation
# this package shipped first and is bit-exactly what a zero threshold reduces to -- so the
# unfiltered form is a special case of this one and not a separate path.
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
) -> tuple[list[_MultigridLevel], wp.array] | None:
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
        label, n_aggregates = _multigrid_aggregate(operator, seed)
        if n_aggregates >= _MULTIGRID_MIN_COARSENING * level.n:
            break
        diagonal = wp.empty(level.n, dtype=wp.float64, device=operator.device)
        wp.launch(
            kernel_cg.cg_inverse_diagonal,
            dim=level.n,
            inputs=[wps.bsr_get_diag(operator)],
            outputs=[diagonal],
            device=operator.device,
        )
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
    matrix: wps.BsrMatrix[wp.float64], seed: int
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
    unfiltered aggregation.
    """
    device = matrix.device
    n = int(matrix.nrow)
    offsets, columns, values = matrix.offsets, matrix.columns, matrix.values
    theta = wp.float64(_MULTIGRID_THETA)
    # ``sqrt(|A_ii|)`` per row, so the strength test below is a product rather than a square
    # root per edge. One ``(n,)`` buffer and two launches per level, read by both walks.
    scaled_diagonal = wp.empty(n, dtype=wp.float64, device=device)
    wp.launch(
        kernel_mg.mg_scaled_diagonal,
        dim=n,
        inputs=[wps.bsr_get_diag(matrix)],
        outputs=[scaled_diagonal],
        device=device,
    )

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
    wp.launch(kernel_mg.mis_root_flags, dim=n, inputs=[state, flags], device=device)
    scan_pos = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, scan_pos, inclusive=True)
    n_aggregates = int(read_scalar(scan_pos))

    label = wp.empty(n, dtype=wp.int32, device=device)
    next_label = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_mg.seed_aggregate_labels, dim=n, inputs=[state, scan_pos, label], device=device
    )
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
    build is launch-bound and this used to be a fifth of it. A step is one fused ``power_step``
    launch rather than a ``bsr_mv`` plus an elementwise scale -- an uncaptured ``bsr_mv`` costs
    ~0.1 ms whatever its nnz -- and the two buffers are ping-ponged rather than updated in place,
    which is what allows the single kernel. Measured over two levels of ``bunny``'s hierarchy:
    **2.57 ms with ``bsr_mv`` and two inner products, 0.91 ms fused, 0.76 ms once the start vector
    made its own norm free.** What is left is mostly the single remaining host sync.
    """
    device = matrix.device
    n = int(matrix.nrow)
    x = wp.empty(n, dtype=wp.float64, device=device)
    y = wp.empty(n, dtype=wp.float64, device=device)
    wp.launch(kernel_mg.random_signs, dim=n, inputs=[wp.int32(seed), x], device=device)
    # Exact, not measured: every entry of a sign vector is +-1.
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

    ``bsr_mm`` returns a structural **superset** of the product -- measured 9 590 entries for one
    whose true pattern is 5 578, the extra ones exactly zero -- and here those zeros are not
    cosmetic. An explicit zero at ``(i, c)`` makes coarse column ``c`` see fine row ``i``, so the
    Galerkin product inherits every aggregate reachable from it: **191 181 entries for the 587-row
    coarse operator against a true 4 084**, a 47x pattern blowup out of a 1.7x one. Pruning is part
    of the algorithm, not tidying, and it is what holds operator complexity at 1.02.

    The extra entries are **interspersed in column order, not trailing capacity**, which is worth
    pinning because the two have different causes and only one is a filed bug. Measured over the 639
    rows that carry a zero: **0** of them have their zeros only at the row's end, the columns stay
    strictly increasing within every row, and the padding is a variable per-row gap fill (mean 1.96
    extra entries, max 22). So the product's *pattern* is wider than the true one rather than
    its *count* over-reporting reserved space.

    It has to be a rebuild rather than ``bsr_compress``, which is the API for exactly this and
    **hard-faults**: compressing a ``bsr_mm`` result makes the *next* ``bsr_mm`` die with
    ``CUDA error 700: an illegal memory access`` inside ``wp_free_device_async`` on Warp 1.16.0,
    which is the signature reported as NVIDIA/warp#1769.

    !!! note "Re-probe both halves on the Warp 1.17 upgrade"
        NVIDIA/warp#1769 (*CUDA ``bsr_compress(inplace=True)`` treats trailing capacity as active*)
        is **closed upstream with milestone 1.17.0**, unreleased as of Warp 1.16.0. Two separate
        things to check when it lands, because the fix addresses one of them at most:

        - **The fault.** If ``bsr_compress`` survives a following ``bsr_mm``, this function may
          collapse to one call. Note the issue is filed against ``inplace=True`` while the crash
          here came from the *default* ``inplace=False``, so confirm the exact call before trusting
          it -- and measure, because a rebuild is only 0.46 ms and ``bsr_compress`` was never timed
          cleanly (it faulted downstream of every attempt).
        - **The superset**, which is what makes a prune necessary at all and is a *different*
          behaviour -- see the paragraph above. Nothing in that issue describes it.

        The ceiling on the first is small: ``_multigrid_prune`` is **1.78 ms of the hierarchy's
        15.42 ms** on ``bunny``. The setup's dominant terms are six ``bsr_mm`` calls (4.92 ms) and
        the aggregation's launches (3.53), and 1.17 touches neither.
    """
    device = matrix.device
    n_rows = int(matrix.nrow)
    nnz = int(matrix.nnz_sync())
    if nnz == 0:
        return matrix
    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.segment_owner_labels, dim=n_rows, inputs=[matrix.offsets, rows], device=device
    )
    return wps.bsr_from_triplets(
        n_rows,
        int(matrix.ncol),
        rows,
        matrix.columns[:nnz],
        matrix.values[:nnz],
        prune_numerical_zeros=True,
    )


def _multigrid_dense_inverse(matrix: wps.BsrMatrix[wp.float64]) -> wp.array | None:
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
        # being symmetric makes exact and which is where most of this function's time was: measured
        # 14.8 -> 4.9 ms on a 587-row level.
        block = np.linalg.pinv(dense[np.ix_(active, active)], rcond=1e-12, hermitian=True)
        inverse[np.ix_(active, active)] = block
    return wp.array(inverse, dtype=wp.float64, device=matrix.device)


class _MultigridCycle:
    """
    One V-cycle of a smoothed-aggregation hierarchy, over ``n_columns`` blocks of a flat vector.

    Batched over the columns: *every* pass, the sparse mat-vecs included, is one launch over the
    whole flat vector. That is not a tidiness choice -- a cycle at these sizes is launch-bound, and
    ``warp.sparse.bsr_mv`` takes one vector, so routing three right-hand sides through it costs
    three launches per mat-vec. Measured on ``smooth_region``'s operator with three columns, both
    captured: **34 launches and 189 us per cycle through ``bsr_mv``, 15 and 85 through
    ``kernels/algorithms/multigrid.csr_matvec``**, same answer.

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
    """
    iterations, residual, atol = result
    if isinstance(iterations, wp.array):
        return
    if int(iterations) >= iteration_cap and float(residual) > float(atol):
        warnings.warn(
            f"{name}: conjugate gradient hit its {iteration_cap}-iteration cap with squared "
            f"residual {float(residual):.3e} against tolerance {float(atol):.3e}; the result is "
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
