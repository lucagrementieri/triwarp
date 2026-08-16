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
If this is revisited, the direction is smoothed-aggregation multigrid — the only option that breaks
the ``O(sqrt(n))`` iteration growth — and ``bsr_mm``, ``bsr_transposed`` and ``bsr_mv`` are all
available to build it.

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

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import linalg as kernel_linalg
from triwarp.kernels.algorithms import conjugate_gradient as kernel_cg

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
        )
        return state() if run else state
    operator = replicated_operator(matrix, n_columns)
    preconditioner = replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    return wpl.cg(
        operator,
        rhs.flatten(),
        solution.flatten(),
        tol=tol,
        maxiter=iteration_cap,
        M=preconditioner,
        check_every=_supported_check_every(check_every),
        run=run,
    )


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
        wp.launch(
            kernel_cg.cg_step_x_r_z,
            dim=self._dofs,
            inputs=[
                wp.int32(self._stride),
                wp.int32(self._n),
                self._rz_old,
                self._p_dot_ap,
                self._dots,
                self._atol_sq,
                self._inv_diag,
                self._p,
                self._ap,
            ],
            outputs=[self._solution_flat, self._r, self._z],
            device=self._device,
        )
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
        wp.launch(
            kernel_cg.cg_apply_inverse_diagonal,
            dim=self._dofs,
            inputs=[wp.int32(self._stride), self._inv_diag, self._r],
            outputs=[self._z],
            device=self._device,
        )
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
                int(self._state.numpy()[0]),
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
