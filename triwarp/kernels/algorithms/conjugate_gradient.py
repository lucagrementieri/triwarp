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
cost is not. Two fusions ride along, both free: the Jacobi apply is an elementwise multiply of the
``r`` the x/r update has just written, so it happens in a register there rather than in its own
launch; and CG's ``rz_old = rz_new`` copy folds into the ``p.Ap`` finalize, the one point in the
iteration between the ``p`` update that last read ``rz_old`` and the x/r update that reads it next.

**A block conjugate gradient (O'Leary 1980) sharing one Krylov subspace across exactly two columns
lived here and was removed.** It advanced both columns against one shared subspace, the scalar
``alpha`` / ``beta`` becoming 2x2 dense matrices solved fresh every iteration. It does cut the
iteration count on a uniform mesh, but its iteration costs more launches than this one's and on
an ill-conditioned system the count goes the other way -- see ``_cg_columns`` in
``triwarp/linalg.py`` for the measurement that retired it.
"""

import warp as wp

from triwarp.kernels.array import LOOP_CONDITION, LOOP_ROUND

# Lanes per block for both stages of the conjugate-gradient dot product. The partial stage gets one
# block per ``CG_TILE`` entries *of each column*, which is what makes its grid grow with the system
# instead of its serial depth; the finalize stage folds that column's partials with one more tile.
# Deliberately a bare ``wp.constant(256)`` and not ``wp.constant(wp.int32(256))``: this constant
# also serves as a ``wp.tile_load`` / ``wp.tile_zeros`` ``shape=``, and a tile shape must be a
# plain integer -- the typed spelling fails to parse on Warp 1.17 with an ``AttributeError`` in
# ``cg_dot_finalize``. The cost is that check 17 cannot type the ``//`` below from the constant's
# declaration and has to take it on trust.
CG_TILE = wp.constant(256)


@wp.func
def sum_padded_row(row: wp.array[wp.float64], n_blocks: wp.int32) -> wp.float64:
    # Cooperative tile-fold sum of one padded row -- the fold ``cg_dot_finalize`` below uses.
    # ``row``'s tail past
    # ``n_blocks`` is zero-padded to a whole number of ``CG_TILE``-wide tiles by whichever partials
    # kernel wrote it, so this needs no ragged branch.
    acc = wp.tile_zeros(shape=CG_TILE, dtype=wp.float64)
    for s in range((n_blocks + CG_TILE - 1) // CG_TILE):
        acc += wp.tile_load(row, shape=CG_TILE, offset=s * CG_TILE, storage="register")
    return wp.tile_sum(acc)[0]


@wp.kernel
def cg_dot_partials(
    a: wp.array[wp.float64],
    b0: wp.array[wp.float64],
    b1: wp.array[wp.float64],
    stride: wp.int32,
    pairs: wp.int32,
    out_partials: wp.array3d[wp.float64],
) -> None:
    # Per-column partial dots of ``(a, b0)`` and, when ``pairs == 2``, of ``(a, b1)`` as well.
    # Launch tiled over ``(n_columns, stride / CG_TILE)`` with ``block_dim=CG_TILE``.
    #
    # ``stride`` is the column pitch, which the wrapper pads to a multiple of ``CG_TILE`` and
    # zero-fills past ``n``, so every block here is a whole tile and there is no ragged branch.
    # That padding is not tidiness: the tail used to run a serial dependent loop over ``stride - n``
    # entries, which every lane of that block executed, and the whole launch waits for it -- so the
    # dot's cost swung severalfold on the arithmetic of ``n mod block_dim``, enough to turn a real
    # win on one mesh into a loss on another of nearly the same size.
    #
    # The second pair shares its first operand, which is what CG's r.r and r.z want: one pass over
    # ``r`` answers both. ``pairs`` is warp-uniform, so the branch costs nothing.
    #
    # A self-dot (``a`` and ``b0`` the same array, as in the initial ``r.r``) loads the identical
    # tile twice rather than reusing ``tile_a`` for ``tile_b0`` -- doubling the traffic for that one
    # tile. Not special-cased: it is a per-call setup cost, not per-iteration (the hot per-iteration
    # dots are ``p.Ap`` and ``r.z``, neither a self-dot), so the win would not show up in any
    # measured iteration cost, and the branch to detect aliasing would run on every call regardless.
    c, blk, t = wp.tid()
    offset = c * stride + blk * CG_TILE
    tile_a = wp.tile_load(a, shape=CG_TILE, offset=offset, storage="register")
    tile_b0 = wp.tile_load(b0, shape=CG_TILE, offset=offset, storage="register")
    acc0 = wp.tile_sum(tile_a * tile_b0)[0]
    acc1 = wp.float64(0.0)
    if pairs == 2:
        tile_b1 = wp.tile_load(b1, shape=CG_TILE, offset=offset, storage="register")
        acc1 = wp.tile_sum(tile_a * tile_b1)[0]
    if t == 0:
        out_partials[0, c, blk] = acc0
        if pairs == 2:
            out_partials[1, c, blk] = acc1


@wp.kernel
def cg_dot_finalize(
    partials: wp.array3d[wp.float64],
    n_blocks: wp.int32,
    pairs: wp.int32,
    carry: wp.int32,
    carry_src: wp.array[wp.float64],
    out_dots: wp.array2d[wp.float64],
    out_carry: wp.array[wp.float64],
) -> None:
    # Sum one (pair, column) row of partials cooperatively, via ``sum_padded_row``. Launch tiled
    # over ``(2, n_columns)``. The partial
    # buffer's last axis is padded to a multiple of ``CG_TILE`` and zeroed once at allocation, so
    # the final tile reads zeros past ``n_blocks`` rather than the next column's partials -- the
    # tail masking the first stage needs is not needed again here.
    #
    # ``carry`` folds CG's ``rz_old = rz_new`` copy into this launch, which is legal only for the
    # ``p.Ap`` finalize: it runs after the p update that last read ``rz_old`` and before the x/r
    # update that reads it next. Doing it in the r.z finalize instead would overwrite the value the
    # p update of the *same* iteration still needs.
    p, c, t = wp.tid()
    if p >= pairs:
        return
    total = sum_padded_row(partials[p, c], n_blocks)
    if t == 0:
        out_dots[p, c] = total
        if carry != 0 and p == 0:
            out_carry[c] = carry_src[c]


@wp.kernel
def cg_absolute_tolerance(
    tol_sq: wp.float64,
    atol_sq: wp.float64,
    b_norm_sq: wp.array2d[wp.float64],
    out_atol_sq: wp.array[wp.float64],
) -> None:
    # Per-column squared stopping threshold, ``max(atol, tol * ||b_c||)^2``, matching
    # ``warp.optim.linear``'s convention so the two solvers stop on the same condition.
    c = wp.int32(wp.tid())
    relative = tol_sq * b_norm_sq[0, c]
    out_atol_sq[c] = wp.max(relative, atol_sq)


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
    # being two. The conjugate gradient wants ``z = M^-1 r`` for the *initial* residual only (inside
    # the iteration it is fused into ``cg_step_x_r_z``), at ``factor = 1`` and with no rows to skip.
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
def cg_advance_x_r(
    i: wp.int32,
    c: wp.int32,
    local: wp.int32,
    n: wp.int32,
    rz_old: wp.array[wp.float64],
    p_dot_ap: wp.array2d[wp.float64],
    r_norm_sq: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
    out_r: wp.array[wp.float64],
) -> wp.float64:
    # ``x += alpha p`` and ``r -= alpha Ap`` for one entry, returning the new residual so the caller
    # can precondition it in the same pass when the preconditioner is elementwise.
    #
    # A column already inside its tolerance takes ``alpha = 0`` and therefore stops moving, which is
    # how the batched loop lets a converged column idle while its neighbours finish rather than
    # dividing by a ``p.Ap`` that has gone to zero.
    #
    # ``p``, ``ap`` and ``r`` are the solver's own, at column pitch ``stride``, and their pad is
    # zero and stays zero here. ``out_x`` is the **caller's** buffer at pitch ``n``, which is why it
    # is indexed separately and skipped in the pad -- copying it into a padded buffer instead cost
    # more than the padding saved on short solves.
    alpha = wp.float64(0.0)
    if r_norm_sq[0, c] > atol_sq[c]:
        alpha = rz_old[c] / p_dot_ap[0, c]
    if local < n:
        slot = c * n + local
        out_x[slot] = out_x[slot] + alpha * p[i]
    residual = out_r[i] - alpha * ap[i]
    out_r[i] = residual
    return residual


@wp.kernel
def cg_step_x_r_z(
    stride: wp.int32,
    n: wp.int32,
    rz_old: wp.array[wp.float64],
    p_dot_ap: wp.array2d[wp.float64],
    r_norm_sq: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    inv_diag: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
    out_r: wp.array[wp.float64],
    out_z: wp.array[wp.float64],
) -> None:
    # ``x += alpha p``; ``r -= alpha Ap``; ``z = M^-1 r``, with the **Jacobi** apply fused in --
    # ``inv_diag`` is padded to ``stride`` alongside the vectors. The only difference from
    # ``cg_step_x_r`` below is that fusion, which is available exactly when the preconditioner is an
    # elementwise multiply of the residual this pass has just written.
    i = wp.int32(wp.tid())
    c = i // stride
    local = i % stride
    residual = cg_advance_x_r(
        i, c, local, n, rz_old, p_dot_ap, r_norm_sq, atol_sq, p, ap, out_x, out_r
    )
    out_z[i] = inv_diag[local] * residual


@wp.kernel
def cg_step_x_r_z_dot(
    stride: wp.int32,
    n: wp.int32,
    rz_old: wp.array[wp.float64],
    p_dot_ap: wp.array2d[wp.float64],
    r_norm_sq: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    inv_diag: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
    out_r: wp.array[wp.float64],
    out_z: wp.array[wp.float64],
    out_partials: wp.array3d[wp.float64],
) -> None:
    # ``cg_step_x_r_z`` with the **first stage** of the ``r.r`` / ``r.z`` reduction folded in: the
    # block sums its own slice of the two dots straight into ``out_partials`` instead of a separate
    # ``cg_dot_partials`` launch reading back the ``r`` and ``z`` this kernel has just written.
    # That is CLAUDE.md section 14.10's producer-consumer fusion applied to a reduction's first
    # stage, which is legal precisely because a block-level partial needs only its own block's data
    # -- no barrier beyond the tile reduction itself, and the finalize stage is unchanged.
    #
    # Launch ``wp.launch_tiled(dim=(n_columns, stride // CG_TILE), block_dim=CG_TILE)``: ``stride``
    # is padded to a whole number of ``CG_TILE`` tiles and zero-filled past ``n``, so every block
    # owns exactly one tile and there is no ragged branch -- the same contract ``cg_dot_partials``
    # relies on, and for the same measured reason.
    #
    # **The lanes stride by ``wp.block_dim()``, not by 1.** On CUDA ``block_dim()`` is ``CG_TILE``
    # and the loop runs once per lane, which is the one-element-per-lane shape this wants. On the
    # CPU device ``wp.launch_tiled`` runs a single lane per block and ``wp.block_dim()`` reads 1
    # (section 2.2), so that lane walks the whole tile and the ``wp.tile`` reductions below
    # degenerate to one-element tiles holding its own totals. Writing ``local = blk * CG_TILE + t``
    # instead would leave all but one entry per tile unwritten there.
    c, blk, t = wp.tid()
    acc_rr = wp.float64(0.0)
    acc_rz = wp.float64(0.0)
    for k in range(t, CG_TILE, wp.block_dim()):
        local = blk * CG_TILE + k
        i = c * stride + local
        residual = cg_advance_x_r(
            i, c, local, n, rz_old, p_dot_ap, r_norm_sq, atol_sq, p, ap, out_x, out_r
        )
        z = inv_diag[local] * residual
        out_z[i] = z
        acc_rr += residual * residual
        acc_rz += residual * z
    total_rr = wp.tile_sum(wp.tile(acc_rr))[0]
    total_rz = wp.tile_sum(wp.tile(acc_rz))[0]
    if t == 0:
        out_partials[0, c, blk] = total_rr
        out_partials[1, c, blk] = total_rz


@wp.kernel
def cg_step_x_r(
    stride: wp.int32,
    n: wp.int32,
    rz_old: wp.array[wp.float64],
    p_dot_ap: wp.array2d[wp.float64],
    r_norm_sq: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
    out_r: wp.array[wp.float64],
) -> None:
    # The same x/r update with **no** preconditioner apply, for a preconditioner that is not an
    # elementwise multiply -- a multigrid V-cycle, which is its own sequence of launches and reads
    # the ``r`` this pass leaves behind. One extra launch per iteration against ``cg_step_x_r_z``,
    # and that is the whole cost of un-fusing.
    i = wp.int32(wp.tid())
    c = i // stride
    local = i % stride
    cg_advance_x_r(i, c, local, n, rz_old, p_dot_ap, r_norm_sq, atol_sq, p, ap, out_x, out_r)


@wp.kernel
def cg_step_p(
    stride: wp.int32,
    maxiter: wp.int32,
    n_columns: wp.int32,
    rz_old: wp.array[wp.float64],
    dots: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    z: wp.array[wp.float64],
    out_p: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # ``p = z + beta p``, with ``beta = rz_new / rz_old``. ``dots[0]`` is r.r and ``dots[1]`` is
    # r.z, both written by the finalize that precedes this launch.
    i = wp.int32(wp.tid())
    c = i // stride
    beta = wp.float64(0.0)
    if dots[0, c] > atol_sq[c]:
        beta = dots[1, c] / rz_old[c]
    out_p[i] = z[i] + beta * out_p[i]

    # The round-loop advance rides in thread 0 rather than in a ``dim=1`` launch of its own. It
    # reads ``dots`` and ``atol_sq``, both written by the finalize *before* this launch, and
    # writes ``out_state``, which nothing above reads -- so no barrier is needed and nothing
    # races with the ``out_p`` writes. That removes one of the iteration's eight launches
    # unconditionally, which is worth having because a conjugate-gradient iteration on a small
    # system is launch-bound.
    #
    # ``out_state`` is the round-loop state array (``array.LOOP_ROUND`` / ``LOOP_CONDITION``), the
    # iteration count in the first slot. The loop runs while *any* column is still above its own
    # tolerance -- the worst-case rule a batched multi-column solve needs -- and stops at
    # ``maxiter``.
    if i == 0:
        # Clamped at ``maxiter`` rather than free-running, so the reported count can never exceed
        # the cap a caller passed -- a caller that tests the returned count against its own cap is
        # how a non-convergence warning gets raised. The device-side loop runs one iteration per
        # conditional test, so today the clamp never fires; it is kept because any scheme that
        # issues a *run* of iterations per test brings the overshoot back, and it costs a ``min``.
        round_index = wp.min(out_state[LOOP_ROUND] + 1, maxiter)
        out_state[LOOP_ROUND] = round_index
        keep = wp.int32(0)
        for col in range(n_columns):
            if dots[0, col] > atol_sq[col]:
                keep = 1
        if round_index >= maxiter:
            keep = 0
        out_state[LOOP_CONDITION] = keep
