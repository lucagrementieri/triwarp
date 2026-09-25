"""
Batched Jacobi-preconditioned conjugate gradient over the columns of one operator.

Drives every multi-column SPD solve in ``triwarp/linalg.py`` -- ``harmonic`` / ``tutte`` /
``arap``'s global step, the implicit smoothers, the signed heat solve -- in place of
``warp.optim.linear.cg``'s iteration. It exists for one measured reason: ``replicated_operator``
attaches ``batch_offsets`` so the solve converges on its worst column, and that is exactly the input
for which Warp's ``TiledDot`` takes its **direct batched** reduction, one block per (column,
subproblem). Every lane of that one block then reduces ``n / tile_size`` entries serially, so the
dot costs *O(n)* where the unbatched tiled tree is flat -- and two dots run per iteration, which
made them roughly half of an iteration at a typical system size.

The reduction here is a real two-stage tree that stays per column, so the batching is kept and the
cost is not. The iteration is the Chronopoulos-Gear form (1989): the same iterates as CG in exact
arithmetic, with ``alpha`` recovered from ``r.u`` and ``w.u`` (``u = M^-1 r``, ``w = A u``) and
``s = A p`` carried by recurrence, so a round is one mat-vec that reduces all three of its dots
(``cg_matvec_dots``) and one update (``cg_update``) -- two launches, against standard CG's
mat-vec, ``p.Ap``, update and ``r.z`` with a finalize after each reduction. The Jacobi apply rides
in the update's register, and each block of the update folds the reductions' second stage itself.
Measured against the six-launch standard iteration on the same systems: a round costs ~13 us
against ~17 us replayed, the conditional-graph test being the largest single part of what is left,
with iteration counts equal or within a few rounds on the systems the suite and benchmarks solve.

**A block conjugate gradient (O'Leary 1980) sharing one Krylov subspace across exactly two columns
lived here and was removed.** It advanced both columns against one shared subspace, the scalar
``alpha`` / ``beta`` becoming 2x2 dense matrices solved fresh every iteration. It does cut the
iteration count on a uniform mesh, but its iteration costs more launches than this one's and on
an ill-conditioned system the count goes the other way -- see ``_cg_columns`` in
``triwarp/linalg.py`` for the measurement that retired it.
"""

from typing import Any

import warp as wp

from triwarp.kernels.algorithms.multigrid import csr_row_dot
from triwarp.kernels.array import LOOP_CONDITION, LOOP_ROUND, OverloadTable
from triwarp.kernels.reduce import block_sum

# Lanes per block for both stages of the conjugate-gradient dot product. The partial stage gets one
# block per ``CG_TILE`` entries *of each column*, which is what makes its grid grow with the system
# instead of its serial depth; the finalize stage folds that column's partials with one more block.
# A host-side launch shape only: no kernel reads it.
CG_TILE = 256

# Tiles a round's block spans once a column is too long to fold (see ``cg_layout``): the target
# block count a long column is launched at. Every block pays one block-wide reduction and one
# partial, so a column of millions of entries at one tile a block spends more on those than on its
# own streams -- measured on a 17-million-node Poisson grid, where 66 000 one-tile blocks ran the
# mat-vec with its dots at several times its memory traffic. The block count is the fold width
# section 2.2 and 13.2 of ``.claude/CLAUDE.md`` describe, not the lane count.
CG_TARGET_BLOCKS = 2048


def cg_layout(n: int, fold_max_blocks: int) -> tuple[int, int, bool]:
    """
    Lay one column of ``n`` entries out as ``(span, blocks, fold)``.

    ``span`` is the entries a block covers, ``blocks`` the block count, and ``fold`` whether the
    round's update folds the dots itself.

    A column of up to ``fold_max_blocks`` tiles is launched one tile a block and folds; a longer
    one is launched at whole powers of two of tiles a block, sized so its block count lands near
    ``CG_TARGET_BLOCKS``, and takes the ``cg_coefficients`` launch instead. The column pitch every
    vector is padded to is ``span * blocks``.
    """
    tile = int(CG_TILE)
    tiles = max((n + tile - 1) // tile, 1)
    if tiles <= fold_max_blocks:
        return tile, tiles, True
    per_block = 1
    while tiles > per_block * CG_TARGET_BLOCKS:
        per_block *= 2
    span = per_block * tile
    return span, (n + span - 1) // span, False


@wp.func
def sum_block_partials(row: wp.array[wp.float64], n_blocks: wp.int32, t: wp.int32) -> wp.float64:
    # The block-wide sum of one row of first-stage partials -- ``cg_seed``'s fold of ``||b||^2``,
    # ``cg_fold_column``'s shape for one quantity: the lanes stride the ``n_blocks`` live entries
    # by ``wp.block_dim()`` (section 2.2) and ``block_sum`` folds them. Block-collective.
    acc = wp.float64(0.0)
    for k in range(t, n_blocks, wp.block_dim()):
        acc += row[k]
    return block_sum(acc)


@wp.kernel
def cg_initial(
    n: wp.int32,
    stride: wp.int32,
    span: wp.int32,
    jacobi: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    rhs: wp.array[wp.Float],
    x: wp.array[wp.Float],
    inv_diag: wp.array[wp.Float],
    out_r: wp.array[wp.Float],
    out_u: wp.array[wp.Float],
    out_p: wp.array[wp.Float],
    out_s: wp.array[wp.Float],
    out_partials: wp.array3d[wp.float64],
) -> None:
    # Everything a solve computes before its first round that is per entry, in one launch tiled
    # over ``(n_columns, stride / CG_TILE)`` at ``block_dim=CG_TILE``: the warm-started residual
    # ``r = b - A x`` from the caller's ``b`` and ``x`` (both at pitch ``n``, into the solver's own
    # vectors at the padded pitch ``stride``), ``u = D^-1 r`` under Jacobi (``jacobi`` set; another
    # preconditioner writes ``u`` in its own launches after this one), ``p = s = 0``, and the first
    # stage of ``||b||^2`` into ``out_partials[0]``, which ``cg_seed`` folds into the tolerance. The
    # pad rows are written as zeros, which is what every reduction over them assumes. The lanes
    # stride by ``wp.block_dim()`` for the CPU device (section 2.2).
    #
    # Generic over the vectors' storage precision (see ``cg_update``); the residual is formed and
    # the norm accumulated in ``float64``.
    c, blk, t = wp.tid()
    acc = wp.float64(0.0)
    for k in range(t, span, wp.block_dim()):
        local = blk * span + k
        i = c * stride + local
        b = wp.float64(0.0)
        r = wp.float64(0.0)
        u = wp.float64(0.0)
        if local < n:
            b = wp.float64(rhs[c * n + local])
            r = b - wp.float64(csr_row_dot(local, c * n, offsets, columns, values, x))
            if jacobi != 0:
                u = wp.float64(inv_diag[local]) * r
        out_r[i] = out_r.dtype(r)
        if jacobi != 0:
            out_u[i] = out_u.dtype(u)
        out_p[i] = out_p.dtype(0.0)
        out_s[i] = out_s.dtype(0.0)
        acc += b * b
    total = block_sum(acc)
    if t == 0:
        out_partials[0, c, blk] = total


@wp.kernel
def cg_seed(
    tol_sq: wp.float64,
    atol_sq: wp.float64,
    n_blocks: wp.int32,
    partials: wp.array3d[wp.float64],
    out_atol_sq: wp.array[wp.float64],
    out_gamma_new: wp.array[wp.float64],
    out_alpha_new: wp.array[wp.float64],
    out_iterations: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Everything a solve sets before its first round, in one launch tiled over ``(n_columns,)``:
    # the per-column squared stopping threshold ``max(atol, tol * ||b_c||)^2`` -- matching
    # ``warp.optim.linear``'s convention so the two solvers stop on the same condition -- folded
    # from ``partials[0]``, where ``cg_initial`` has left ``||b_c||^2``'s first stage; the
    # recurrence's seeds, ``gamma = inf`` (so the first round's ``beta`` is zero) and
    # ``alpha = 1``, in the ``*_new`` slots the first ``cg_matvec_dots`` carries from; and the
    # round-loop state at ``[0, 1]`` with a zero count. A zero condition would run no rounds at
    # all, since ``wp.capture_while`` reads it first.
    c, t = wp.tid()
    b_norm_sq = sum_block_partials(partials[0, c], n_blocks, t)
    if t == 0:
        out_atol_sq[c] = wp.max(tol_sq * b_norm_sq, atol_sq)
        out_gamma_new[c] = wp.float64(wp.inf)
        out_alpha_new[c] = wp.float64(1.0)
        if c == 0:
            out_iterations[0] = 0
            out_state[LOOP_ROUND] = 0
            out_state[LOOP_CONDITION] = 1


@wp.kernel
def scaled_diagonal_apply(
    n: wp.int32,
    stride: wp.int32,
    inv_diag: wp.array[wp.float64],
    factor: wp.float64,
    source: wp.array[wp.float64],
    out_destination: wp.array[wp.float64],
) -> None:
    # ``out = factor * D^-1 * source``, over every column of one operator at once.
    #
    # Two callers, which is why this one kernel carries the two extra arguments rather than there
    # being two. The conjugate gradient wants ``z = M^-1 r`` for the *initial* residual only
    # (inside the iteration it is fused into ``cg_update``), at ``factor = 1`` and with no
    # rows to skip.
    # The multigrid cycle wants the power iteration's step and the first Jacobi sweep from a zero
    # initial guess -- the sweep being a *write*, so the cycle never has to zero its working
    # vectors. It was written twice, a week apart, in two files, one of which plugs into the other.
    #
    # ``stride`` is the column pitch and ``n`` the row count; one operator serves every column, so
    # the diagonal is indexed *within* the column rather than across the flat vector. They differ
    # wherever the conjugate-gradient state pads each column out to a whole reduction tile, and the
    # pad is skipped rather than written because that solver reduces over it and needs it zero. That
    # is reachable from both callers, not only the CG one: the multigrid cycle's *top* level reuses
    # the outer CG state's own (possibly padded) stride, so ``n < stride`` there too whenever the
    # system size is not a whole number of tiles -- harmless only because the level's ``b`` is that
    # same CG state's ``r``, whose pad the CG state already keeps zero.
    #
    # Lives here rather than in ``algorithms/multigrid.py`` because the cycle is a
    # ``preconditioner=`` choice of *this* solver: the dependency runs multigrid -> CG, never back.
    t = wp.int32(wp.tid())
    row = t % stride
    if row >= n:
        return
    out_destination[t] = factor * inv_diag[row] * source[t]


@wp.func
def cg_publish_round_dots(
    total: wp.vec3d,
    c: wp.int32,
    blk: wp.int32,
    t: wp.int32,
    gamma_new: wp.array[wp.float64],
    alpha_new: wp.array[wp.float64],
    out_partials: wp.array3d[wp.float64],
    out_gamma_old: wp.array[wp.float64],
    out_alpha_old: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # The tail every round's mat-vec launch shares, whatever applies the operator -- the CSR one
    # below and ``kernels/reconstruction.poisson_cg_matvec_dots``' matrix-free stencil: lane 0
    # publishes the block's ``(r.u, w.u, r.r)`` partials, block ``(c, 0)`` carries column ``c``'s
    # ``gamma`` and ``alpha`` from the slots the previous ``cg_update`` wrote into the ones the next
    # reads -- ``cg_update`` cannot write its own inputs, since every block of it reads them -- and
    # block ``(0, 0)`` advances the round count, which ``cg_update`` reads to test the cap. Nothing
    # in the launch reads what those writes touch.
    if t == 0:
        out_partials[0, c, blk] = total[0]
        out_partials[1, c, blk] = total[1]
        out_partials[2, c, blk] = total[2]
        if blk == 0:
            out_gamma_old[c] = gamma_new[c]
            out_alpha_old[c] = alpha_new[c]
            if c == 0:
                out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1


@wp.func
def cg_round_terms(r: wp.Float, u: wp.Float, w: wp.Float) -> Any:
    # One entry's ``(r.u, w.u, r.r)`` terms, at the storage precision. Shared by every kernel that
    # reduces a round's dots, whatever formed ``w``. A block sums them at that precision too and
    # only the few thousand block partials are folded in ``float64`` (``cg_widen``): on a
    # ``float32`` Poisson grid a ``float64`` sum per entry measured 1.1-1.35x slower on the levels
    # that fit in L2, for the identical true residual (``.claude/CLAUDE.md`` section 16.16).
    return wp.vector(r * u, w * u, r * r)


@wp.func
def cg_widen(total: Any) -> wp.vec3d:
    # A block's ``(r.u, w.u, r.r)`` sum, widened for the ``float64`` fold across blocks.
    return wp.vec3d(wp.float64(total[0]), wp.float64(total[1]), wp.float64(total[2]))


@wp.kernel
def cg_matvec_dots(
    n: wp.int32,
    stride: wp.int32,
    span: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.Float],
    r: wp.array[wp.Float],
    u: wp.array[wp.Float],
    gamma_new: wp.array[wp.float64],
    alpha_new: wp.array[wp.float64],
    out_w: wp.array[wp.Float],
    out_partials: wp.array3d[wp.float64],
    out_gamma_old: wp.array[wp.float64],
    out_alpha_old: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # The first node of a Chronopoulos-Gear iteration: ``w = A u`` for every column at once, and
    # the first stage of its **three** dots -- ``gamma = r.u``, ``delta = w.u``, ``r.r`` -- in one
    # block-wide ``wp.vec3d`` reduction. That is the variant's point: standard CG needs ``p.Ap``
    # before it can update ``r`` and ``r.z`` after, two reductions with a dependency between them,
    # where this recovers ``alpha`` from ``gamma`` and ``delta`` alone (``cg_update``), so an
    # iteration is one mat-vec with its reduction and one update. Launch tiled over
    # ``(n_columns, stride // CG_TILE)`` at ``block_dim=CG_TILE``; the lanes stride by
    # ``wp.block_dim()`` for the CPU device (section 2.2). The pad rows write ``w = 0``, so every
    # vector the dots reduce over keeps a zero pad.
    #
    # ``csr_matvec`` itself is left row-per-thread: it is shared with the multigrid V-cycle at four
    # sites that want no partials, and those launch at the raw row count with no tile padding.
    #
    # Generic over the storage precision like ``cg_update``: the block sums at the storage
    # precision and the partials are ``float64`` (``cg_round_terms``). The carry and the round
    # count are ``cg_publish_round_dots``.
    c, blk, t = wp.tid()
    acc = cg_round_terms(r.dtype(0.0), r.dtype(0.0), r.dtype(0.0))
    for k in range(t, span, wp.block_dim()):
        local = blk * span + k
        i = c * stride + local
        wi = out_w.dtype(0.0)
        if local < n:
            wi = csr_row_dot(local, c * stride, offsets, columns, values, u)
        out_w[i] = wi
        acc += cg_round_terms(r[i], u[i], wi)
    total = cg_widen(block_sum(acc))
    cg_publish_round_dots(
        total,
        c,
        blk,
        t,
        gamma_new,
        alpha_new,
        out_partials,
        out_gamma_old,
        out_alpha_old,
        out_state,
    )


@wp.kernel
def cg_round_dots(
    stride: wp.int32,
    span: wp.int32,
    r: wp.array[wp.Float],
    u: wp.array[wp.Float],
    w: wp.array[wp.Float],
    gamma_new: wp.array[wp.float64],
    alpha_new: wp.array[wp.float64],
    out_partials: wp.array3d[wp.float64],
    out_gamma_old: wp.array[wp.float64],
    out_alpha_old: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # ``cg_matvec_dots`` without the mat-vec, for an operator whose rows are long enough that
    # ``warp.sparse.bsr_mv``'s block-per-row kernel forms ``w = A u`` faster than one lane a row
    # does (``linalg.CG_HEAVY_ROW_ENTRIES``): a lane-per-row mat-vec fused with a block-wide
    # reduction holds every lane of the block until its longest row is done. The pads of ``r``,
    # ``u`` and ``w`` are zero, so reducing over them is a no-op. Launch like ``cg_matvec_dots``.
    c, blk, t = wp.tid()
    acc = cg_round_terms(r.dtype(0.0), r.dtype(0.0), r.dtype(0.0))
    for k in range(t, span, wp.block_dim()):
        i = c * stride + blk * span + k
        acc += cg_round_terms(r[i], u[i], w[i])
    total = cg_widen(block_sum(acc))
    cg_publish_round_dots(
        total,
        c,
        blk,
        t,
        gamma_new,
        alpha_new,
        out_partials,
        out_gamma_old,
        out_alpha_old,
        out_state,
    )


@wp.func
def cg_fold_column(
    c: wp.int32, t: wp.int32, n_blocks: wp.int32, partials: wp.array3d[wp.float64]
) -> wp.vec3d:
    # Column ``c``'s ``(gamma, delta, r.r)``, folded from ``cg_matvec_dots``' partials in one
    # block-wide ``wp.vec3d`` reduction whose order is fixed by the launch shape alone -- so every
    # block that folds the same column gets the bit-identical triple and agrees on ``alpha`` and
    # ``beta``, which is what makes a redundant fold a legal stand-in for the grid-wide barrier Warp
    # does not expose. Block-collective. The lanes stride by ``wp.block_dim()`` (section 2.2).
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(t, n_blocks, wp.block_dim()):
        acc += wp.vec3d(partials[0, c, k], partials[1, c, k], partials[2, c, k])
    return block_sum(acc)


@wp.func
def cg_step_scalars(
    dots: wp.vec3d,
    gamma_old: wp.float64,
    alpha_old: wp.float64,
    atol_sq: wp.float64,
    round_index: wp.int32,
    maxiter: wp.int32,
) -> wp.vec3d:
    # ``(alpha, beta, stepping)`` for one column from its ``(gamma, delta, r.r)``:
    # ``beta = gamma / gamma_old`` and ``alpha = gamma / (delta - beta gamma / alpha_old)``. The
    # triple describes the ``r`` the round starts from, so the stopping test comes first: a column
    # inside its tolerance -- or any column once the round count has passed ``maxiter`` -- takes no
    # step, and ``stepping`` is 0. The first round reads ``gamma_old = inf``, so ``beta`` is zero.
    #
    # ``gamma`` and the denominator share the system's sign and are nonzero on any step a definite
    # system can take (either sign: a negative-definite operator runs the same iteration). A column
    # driven to its round-off floor -- a zero tolerance, which the heat method's reach-bound solves
    # use -- underflows both to exactly zero, and ``0 / 0`` would poison it with NaN. Such a column
    # has converged exactly, and takes no step.
    zero = wp.float64(0.0)
    if dots[2] > atol_sq and round_index <= maxiter and dots[0] != zero:
        beta = dots[0] / gamma_old
        denominator = dots[1] - beta * dots[0] / alpha_old
        if denominator != zero:
            return wp.vec3d(dots[0] / denominator, beta, 1.0)
    return wp.vec3d(0.0, 0.0, 0.0)


@wp.func
def cg_publish_column(
    c: wp.int32,
    dots: wp.vec3d,
    step: wp.vec3d,
    out_gamma_new: wp.array[wp.float64],
    out_alpha_new: wp.array[wp.float64],
    out_dots: wp.array2d[wp.float64],
) -> None:
    # One lane's publication of column ``c``'s round: ``(r.r, gamma)`` into ``out_dots`` (the host
    # checks and the returned residual read row 0), and -- if it stepped -- ``gamma`` and ``alpha``
    # into the ``*_new`` slots, which ``cg_matvec_dots`` carries to the ``*_old`` ones the next
    # round reads (the round's own readers are every block, so it cannot write them in place).
    out_dots[0, c] = dots[2]
    out_dots[1, c] = dots[0]
    if step[2] != wp.float64(0.0):
        out_gamma_new[c] = dots[0]
        out_alpha_new[c] = step[0]


@wp.func
def cg_close_round(
    first_step: wp.vec3d,
    t: wp.int32,
    n_columns: wp.int32,
    n_blocks: wp.int32,
    partials: wp.array3d[wp.float64],
    atol_sq: wp.array[wp.float64],
    round_index: wp.int32,
    maxiter: wp.int32,
    out_iterations: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Called by the one block that owns column 0: counts the round if any column stepped, and lets
    # another run only if one did and the cap leaves room -- the next round's mat-vec is what
    # measures the residual this step produced. Every column's ``r.r`` is needed, and the other
    # columns' blocks publish theirs concurrently, so this block folds them itself; the fold is
    # block-collective, so the whole block runs the loop and only lane 0 writes. Column 0's step is
    # the caller's own.
    stepped = wp.int32(0)
    if first_step[2] != wp.float64(0.0):
        stepped = 1
    for col in range(1, n_columns):
        col_rr = cg_fold_column(col, t, n_blocks, partials)[2]
        if col_rr > atol_sq[col] and round_index <= maxiter:
            stepped = 1
    if t == 0:
        out_iterations[0] = out_iterations[0] + stepped
        keep = wp.int32(0)
        if stepped != 0 and round_index < maxiter:
            keep = 1
        out_state[LOOP_CONDITION] = keep


@wp.kernel
def cg_coefficients(
    n_columns: wp.int32,
    maxiter: wp.int32,
    n_blocks: wp.int32,
    partials: wp.array3d[wp.float64],
    gamma_old: wp.array[wp.float64],
    alpha_old: wp.array[wp.float64],
    atol_sq: wp.array[wp.float64],
    state: wp.array[wp.int32],
    out_coefficients: wp.array2d[wp.float64],
    out_gamma_new: wp.array[wp.float64],
    out_alpha_new: wp.array[wp.float64],
    out_dots: wp.array2d[wp.float64],
    out_iterations: wp.array[wp.int32],
) -> None:
    # The unfolded path's middle node, tiled over ``(n_columns,)``: one block per column folds its
    # partials, derives ``(alpha, beta, stepping)`` into ``out_coefficients[:, c]`` for
    # ``cg_update`` to read, and publishes and closes the round -- everything a folded
    # ``cg_update`` does besides the vectors, done once per column instead of once per block and
    # thread. Taken where a column is longer than ``linalg.CG_FOLD_MAX_BLOCKS`` tiles.
    c, t = wp.tid()
    round_index = state[LOOP_ROUND]
    dots = cg_fold_column(c, t, n_blocks, partials)
    step = cg_step_scalars(dots, gamma_old[c], alpha_old[c], atol_sq[c], round_index, maxiter)
    if t == 0:
        out_coefficients[0, c] = step[0]
        out_coefficients[1, c] = step[1]
        out_coefficients[2, c] = step[2]
        cg_publish_column(c, dots, step, out_gamma_new, out_alpha_new, out_dots)
    if c == 0:
        cg_close_round(
            step,
            t,
            n_columns,
            n_blocks,
            partials,
            atol_sq,
            round_index,
            maxiter,
            out_iterations,
            state,
        )


@wp.kernel
def cg_update(
    stride: wp.int32,
    span: wp.int32,
    n: wp.int32,
    n_columns: wp.int32,
    maxiter: wp.int32,
    fold: wp.int32,
    n_blocks: wp.int32,
    jacobi: wp.int32,
    partials: wp.array3d[wp.float64],
    coefficients: wp.array2d[wp.float64],
    gamma_old: wp.array[wp.float64],
    alpha_old: wp.array[wp.float64],
    atol_sq: wp.array[wp.float64],
    inv_diag: wp.array[wp.Float],
    w: wp.array[wp.Float],
    p: wp.array[wp.Float],
    s: wp.array[wp.Float],
    r: wp.array[wp.Float],
    u: wp.array[wp.Float],
    state: wp.array[wp.int32],
    out_x: wp.array[wp.Float],
    out_scaled: wp.array[wp.Float],
    out_gamma_new: wp.array[wp.float64],
    out_alpha_new: wp.array[wp.float64],
    out_dots: wp.array2d[wp.float64],
    out_iterations: wp.array[wp.int32],
) -> None:
    # The second node: ``beta = gamma / gamma_old``, ``alpha = gamma / (delta - beta gamma /
    # alpha_old)``, then ``p = u + beta p``, ``s = w + beta s`` (so ``s = A p`` by recurrence, with
    # no second mat-vec), ``x += alpha p``, ``r -= alpha s`` and, with ``jacobi`` set,
    # ``out_scaled = D^-1 r`` in the same register. Chronopoulos & Gear (1989): the same iterates as
    # CG in exact arithmetic, one global reduction per iteration instead of two. Under Jacobi
    # ``out_scaled`` *is* ``u``; under the Jacobi-Chebyshev polynomial it is the polynomial's input,
    # whose own scaling launch this replaces, and the polynomial then writes ``u``. A V-cycle writes
    # ``u`` in its own launches with ``jacobi`` unset. Only the ``n`` live rows are written, so
    # ``inv_diag`` is length ``n`` and the pad of ``out_scaled`` stays at its allocation's zero.
    # Launch tiled like ``cg_matvec_dots``.
    #
    # The triple describes the ``r`` this update starts from, so the stopping test comes first: a
    # column whose ``r.r`` is inside its tolerance -- or any column once the round count has passed
    # ``maxiter`` -- takes no step at all and idles while its neighbours finish. The first round
    # reads ``gamma_old = inf``, which makes ``beta`` zero and ``p = u``.
    #
    # ``p``, ``s``, ``r`` and ``u`` are updated in place, each entry by the one lane that owns it.
    #
    # **Where the scalars come from depends on ``fold``.** With it set, every block folds its
    # column's partials and derives ``alpha`` / ``beta`` itself (``cg_fold_column``,
    # ``cg_step_scalars``); block ``(c, 0)`` publishes them (``cg_publish_column``) and block
    # ``(0, 0)`` closes the round (``cg_close_round``). Without it ``cg_coefficients`` has done all
    # of that once per column and this launch only streams the vectors. The split is not only about
    # the fold's ``n_blocks`` squared reads: deriving the scalars costs three ``float64``
    # divisions, which every *thread* of this launch would otherwise repeat, and on a device whose
    # ``float64`` rate is a small fraction of its ``float32`` one that was measured to double the
    # launch at 17 million entries (0.53 -> 1.0 ms) against the same kernel reading constants.
    #
    # **Generic over the vectors' storage precision.** The element updates run at the storage
    # precision, as the in-block dot sums do (``cg_round_terms``); the fold across blocks,
    # ``alpha`` and ``beta`` are ``float64``. On a ``float32`` system that matched an
    # all-``float64`` reduction's true residual to 3-4 digits, capped and converged, because
    # ``float32`` storage is what sets the floor. On ``float64`` storage every conversion is the
    # identity.
    c, blk, t = wp.tid()
    round_index = state[LOOP_ROUND]
    dots = wp.vec3d(0.0, 0.0, 0.0)
    step = wp.vec3d(coefficients[0, c], coefficients[1, c], coefficients[2, c])
    if fold != 0:
        dots = cg_fold_column(c, t, n_blocks, partials)
        step = cg_step_scalars(dots, gamma_old[c], alpha_old[c], atol_sq[c], round_index, maxiter)
    # The scalars are rounded to the storage precision once and the element updates run there:
    # on ``float32`` storage the grid's L2-resident levels are bound by arithmetic rather than
    # bytes, and widening every element to ``float64`` measured slower than Warp's own solver there.
    alpha = p.dtype(step[0])
    beta = p.dtype(step[1])
    if step[2] != wp.float64(0.0):
        for k in range(t, span, wp.block_dim()):
            local = blk * span + k
            i = c * stride + local
            pk = u[i] + beta * p[i]
            sk = w[i] + beta * s[i]
            p[i] = pk
            s[i] = sk
            if local < n:
                slot = c * n + local
                out_x[slot] = out_x[slot] + alpha * pk
            residual = r[i] - alpha * sk
            r[i] = residual
            if jacobi != 0 and local < n:
                out_scaled[i] = inv_diag[local] * residual
    if fold != 0:
        if blk == 0 and t == 0:
            cg_publish_column(c, dots, step, out_gamma_new, out_alpha_new, out_dots)
        if c == 0 and blk == 0:
            cg_close_round(
                step,
                t,
                n_columns,
                n_blocks,
                partials,
                atol_sq,
                round_index,
                maxiter,
                out_iterations,
                state,
            )


# The storage precisions ``_BatchedCg`` solves in: ``float64`` for every mesh operator, ``float32``
# for the ``warp.fem`` Poisson system ``reconstruction`` assembles. The matrix-free Poisson grid
# reaches ``cg_update`` at ``float32`` too, through its own mat-vec.
_CG_DTYPES = (wp.float32, wp.float64)


def _register_overloads() -> None:
    """Instantiate the generic round kernels at every storage precision, keyed by the dtype."""
    global CG_INITIAL, CG_MATVEC_DOTS, CG_ROUND_DOTS, CG_UPDATE
    f64 = wp.float64
    CG_INITIAL = OverloadTable(
        cg_initial,
        {
            d: [wp.int32] * 4
            + [wp.array[wp.int32], wp.array[wp.int32]]
            + [wp.array[d]] * 8
            + [wp.array3d[f64]]
            for d in _CG_DTYPES
        },
    )
    CG_MATVEC_DOTS = OverloadTable(
        cg_matvec_dots,
        {
            d: [wp.int32] * 3
            + [wp.array[wp.int32], wp.array[wp.int32]]
            + [wp.array[d]] * 3
            + [wp.array[f64]] * 2
            + [wp.array[d], wp.array3d[f64], wp.array[f64], wp.array[f64], wp.array[wp.int32]]
            for d in _CG_DTYPES
        },
    )
    CG_ROUND_DOTS = OverloadTable(
        cg_round_dots,
        {
            d: [wp.int32] * 2
            + [wp.array[d]] * 3
            + [wp.array[f64]] * 2
            + [wp.array3d[f64], wp.array[f64], wp.array[f64], wp.array[wp.int32]]
            for d in _CG_DTYPES
        },
    )
    CG_UPDATE = OverloadTable(
        cg_update,
        {
            d: [wp.int32] * 8
            + [wp.array3d[f64], wp.array2d[f64], wp.array[f64], wp.array[f64], wp.array[f64]]
            + [wp.array[d]] * 6
            + [wp.array[wp.int32], wp.array[d], wp.array[d]]
            + [wp.array[f64], wp.array[f64], wp.array2d[f64], wp.array[wp.int32]]
            for d in _CG_DTYPES
        },
    )


_register_overloads()
