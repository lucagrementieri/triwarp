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
  operator against several right-hand-side columns in a *single* conjugate-gradient call, using
  ``warp.optim.linear``'s batched-``LinearOperator`` support.

!!! note
    Every solve here goes through ``warp.optim.linear.cg``, which returns NaN on the Warp CPU
    backend (1.14-1.15), so each entry point that actually reaches a solve raises
    ``NotImplementedError`` on a CPU device. A fully-constrained
    [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed] does no solve and stays CPU-safe.

**Why batch the columns.** A ``k``-column solve used to be a Python loop of ``k`` independent
``cg`` calls, so its cost was ``sum`` of the per-column iteration counts and every CG iteration
launched ``k`` separate sets of reduction and AXPY kernels.
[`replicated_operator`][triwarp.linalg.replicated_operator] instead presents the *same* operator as
``k`` independent subproblems over one flat ``k * n`` vector: ``cg`` then advances all columns
together and stops on the worst-case residual, so the cost becomes ``max`` of the per-column
iteration counts with one set of vector kernels per iteration. The sparse matrix is never
replicated in memory — the ``matvec`` issues ``k`` ``bsr_mv`` calls against the single operator.

**Determinism.** Build each operator natively at its final dtype in a *single*
``warp.sparse.bsr_from_triplets`` and never recast or rebuild it: ``bsr_mm`` is only deterministic
on single-build operators (see ``downloads/issue_report.md``). Nothing here recasts an operator.
"""

from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_cuda
from triwarp.kernels import linalg as kernel_linalg

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
    Minimize a quadratic form with pinned degrees of freedom (``igl::min_quad_with_fixed``).

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
        return twt.as_array2d_float(solution, dtype=wp.float64), free_map, n_free
    require_cuda(device, "min_quad_with_fixed")

    q_uu, rhs = assemble_interior_system(q, fixed_mask, free_map, fixed_values, n_free)
    solve_spd_columns(
        q_uu,
        rhs,
        twt.as_array2d_float(solution, dtype=wp.float64),
        tol=tol,
        check_every=check_every,
    )
    return twt.as_array2d_float(solution, dtype=wp.float64), free_map, n_free


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

    One launch over the ``n_dofs`` rows of ``q`` emits the free-free COO block (remapped through
    ``free_map``) and accumulates the pinned-column contributions into ``n_rhs`` right-hand-side
    rows, followed by a single ``bsr_from_triplets``.

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
    nnz = int(q.nnz)
    out_rows = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_cols = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_vals = wp.zeros(nnz, dtype=wp.float64, device=device)
    rhs = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    wp.launch(
        kernel_linalg.interior_system_triplets,
        dim=n_dofs,
        inputs=[
            q.offsets,
            q.columns,
            q.values,
            fixed_mask,
            free_map,
            fixed_values,
            out_rows,
            out_cols,
            out_vals,
            rhs,
        ],
        device=device,
    )
    q_uu = wps.bsr_from_triplets(
        n_free, n_free, out_rows, out_cols, out_vals, prune_numerical_zeros=False
    )
    return q_uu, twt.as_array2d_float(rhs, dtype=wp.float64)


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
        ``(iterations, residual_norm, absolute_tolerance)`` as returned by
        ``warp.optim.linear.cg``, with the residual taken over the worst column. Device arrays
        rather than host scalars under ``check_every=0``; see the warning above.

    Notes
    -----
    ``check_every`` is a pure performance knob — it cannot change the converged answer, only how far
    past the tolerance the solver may overshoot before it notices. Two regimes have been measured on
    an RTX 5090, and they disagree, so read the one that matches the caller:

    - **Cold single solves** — one ``cg`` call from a zero initial guess, the shape
      ``harmonic`` / ``tutte`` / ``position_verts_smoothly`` take. Measured on
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

    See Also
    --------
    [`spd_column_solver`][triwarp.linalg.spd_column_solver]
    [`replicated_operator`][triwarp.linalg.replicated_operator]
    """
    require_cuda(rhs.device, "solve_spd_columns")
    n_columns, n = int(rhs.shape[0]), int(rhs.shape[1])
    operator = replicated_operator(matrix, n_columns)
    preconditioner = replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    return wpl.cg(
        operator,
        rhs.flatten(),
        solution.flatten(),
        tol=tol,
        maxiter=maxiter if maxiter is not None else CG_MAXITER_FACTOR * n,
        M=preconditioner,
        check_every=_supported_check_every(check_every),
    )


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
    require_cuda(rhs.device, "spd_column_solver")
    n_columns, n = int(rhs.shape[0]), int(rhs.shape[1])
    operator = replicated_operator(matrix, n_columns)
    preconditioner = replicated_operator(wpl.preconditioner(matrix, "diag"), n_columns)
    return wpl.cg(
        operator,
        rhs.flatten(),
        solution.flatten(),
        tol=tol,
        maxiter=maxiter if maxiter is not None else CG_MAXITER_FACTOR * n,
        M=preconditioner,
        check_every=_supported_check_every(check_every),
        run=False,
    )


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
