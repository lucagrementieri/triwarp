"""
Batched Jacobi-preconditioned conjugate gradient over the columns of one operator.

Drives every multi-column SPD solve in ``triwarp/linalg.py`` -- ``harmonic`` / ``tutte`` /
``arap``'s global step, the implicit smoothers, the signed heat solve -- in place of
``warp.optim.linear.cg``'s iteration. It exists for one measured reason: ``replicated_operator``
attaches ``batch_offsets`` so the solve converges on its worst column, and that is exactly the input
for which Warp's ``TiledDot`` takes its **direct batched** reduction, one block per (column,
subproblem). Every lane of that one block then reduces ``n / tile_size`` entries serially, so the
dot costs *O(n)*: measured 4.55 / 9.51 / 18.66 / **66.15** us at n = 4 356 / 17 161 / 40 962 /
163 842, against 5.09-6.06 us flat for the tiled tree Warp uses when there is nothing to batch. Two
dots run per iteration, which is why they were 19 us of a 41 us iteration at ``harmonic``'s size.

The reduction here is a real two-stage tree that stays per column, so the batching is kept and the
cost is not. Two fusions ride along, both free: the Jacobi apply is an elementwise multiply of the
``r`` the x/r update has just written, so it happens in a register there rather than in its own
launch; and CG's ``rz_old = rz_new`` copy folds into the ``p.Ap`` finalize, the one point in the
iteration between the ``p`` update that last read ``rz_old`` and the x/r update that reads it next.
"""

import warp as wp

# Lanes per block for both stages of the conjugate-gradient dot product. The partial stage gets one
# block per ``CG_TILE`` entries *of each column*, which is what makes its grid grow with the system
# instead of its serial depth; the finalize stage folds that column's partials with one more tile.
CG_TILE = wp.constant(256)


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
    # entries, which every lane of that block executed, and the whole launch waits for it. Measured
    # at ``block_dim=256``: a 9-entry tail cost 2.32 us and a 129-entry tail **11.03 us**, so the
    # dot's cost swung 4.7x on the arithmetic of ``n mod 256`` -- and that alone turned a 1.5x win
    # on ``saddle`` into a 0.82x loss on ``hemisphere``, at nearly the same size.
    #
    # The second pair shares its first operand, which is what CG's r.r and r.z want: one pass over
    # ``r`` answers both. ``pairs`` is warp-uniform, so the branch costs nothing.
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
    # Sum one (pair, column) row of partials cooperatively. Launch tiled over ``(2, n_columns)``.
    # The partial buffer's last axis is padded to a multiple of ``CG_TILE`` and zeroed once at
    # allocation, so the final tile reads zeros past ``n_blocks`` rather than the next column's
    # partials -- the tail masking the first stage needs is not needed again here.
    #
    # ``carry`` folds CG's ``rz_old = rz_new`` copy into this launch, which is legal only for the
    # ``p.Ap`` finalize: it runs after the p update that last read ``rz_old`` and before the x/r
    # update that reads it next. Doing it in the r.z finalize instead would overwrite the value the
    # p update of the *same* iteration still needs.
    p, c, t = wp.tid()
    if p >= pairs:
        return
    row = partials[p, c]
    acc = wp.tile_zeros(shape=CG_TILE, dtype=wp.float64)
    for s in range((n_blocks + CG_TILE - 1) / CG_TILE):
        acc += wp.tile_load(row, shape=CG_TILE, offset=s * CG_TILE, storage="register")
    total = wp.tile_sum(acc)[0]
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
def cg_inverse_diagonal(diag: wp.array[wp.float64], out_inv_diag: wp.array[wp.float64]) -> None:
    # Jacobi preconditioner. A zero diagonal entry maps to 1 rather than to infinity, which is what
    # ``warp.optim.linear.preconditioner(m, "diag")`` does: such a row contributes nothing and must
    # not poison the whole vector with a NaN.
    i = wp.int32(wp.tid())
    value = diag[i]
    out_inv_diag[i] = wp.where(value != wp.float64(0.0), wp.float64(1.0) / value, wp.float64(1.0))


@wp.kernel
def cg_apply_inverse_diagonal(
    stride: wp.int32,
    inv_diag: wp.array[wp.float64],
    values: wp.array[wp.float64],
    out_preconditioned: wp.array[wp.float64],
) -> None:
    # ``z = M^-1 r`` for the *initial* residual only; inside the iteration this is fused into
    # ``cg_step_x_r_z``. One operator serves every column, so the diagonal is indexed within the
    # column rather than across the flat vector -- and is padded to ``stride`` alongside it.
    i = wp.int32(wp.tid())
    out_preconditioned[i] = inv_diag[i - (i // stride) * stride] * values[i]


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
    local = i - c * stride
    residual = cg_advance_x_r(
        i, c, local, n, rz_old, p_dot_ap, r_norm_sq, atol_sq, p, ap, out_x, out_r
    )
    out_z[i] = inv_diag[local] * residual


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
    local = i - c * stride
    cg_advance_x_r(i, c, local, n, rz_old, p_dot_ap, r_norm_sq, atol_sq, p, ap, out_x, out_r)


@wp.kernel
def cg_step_p(
    stride: wp.int32,
    rz_old: wp.array[wp.float64],
    dots: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    z: wp.array[wp.float64],
    out_p: wp.array[wp.float64],
) -> None:
    # ``p = z + beta p``, with ``beta = rz_new / rz_old``. ``dots[0]`` is r.r and ``dots[1]`` is
    # r.z, both written by the finalize that precedes this launch.
    i = wp.int32(wp.tid())
    c = i // stride
    beta = wp.float64(0.0)
    if dots[0, c] > atol_sq[c]:
        beta = dots[1, c] / rz_old[c]
    out_p[i] = z[i] + beta * out_p[i]


@wp.kernel
def cg_advance_condition(
    maxiter: wp.int32,
    n_columns: wp.int32,
    r_norm_sq: wp.array2d[wp.float64],
    atol_sq: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # ``out_state`` is ``[iterations, condition]``. The loop runs while *any* column is still above
    # its own tolerance -- the worst-case rule the batching exists for -- and stops at ``maxiter``.
    # Launch with ``dim=1``: the column count is the right-hand-side count, a handful.
    out_state[0] = out_state[0] + 1
    keep = wp.int32(0)
    for c in range(n_columns):
        if r_norm_sq[0, c] > atol_sq[c]:
            keep = 1
    if out_state[0] >= maxiter:
        keep = 0
    out_state[1] = keep
