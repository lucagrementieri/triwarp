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

**The second half of this module is a different iteration, not a faster reduction: classical block
conjugate gradient (O'Leary 1980) sharing one Krylov subspace across exactly two columns.** Where
the kernels above batch two columns' *launches* while each column's search direction stays its own,
``triwarp/linalg.py``'s ``_BlockCg2`` uses these kernels to advance both columns against one shared
subspace -- the scalar ``alpha`` / ``beta`` of a two-column ``_BatchedCg`` iteration become 2x2
dense matrices here, solved fresh every iteration. See ``_BlockCg2`` for the measured
iteration-count win, the gate that decides when it applies, and the near-rank-deficiency guard's
own limits.
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
    for s in range((n_blocks + CG_TILE - 1) // CG_TILE):
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
    # only where the conjugate-gradient state pads each column out to a whole reduction tile, and
    # the pad is skipped rather than written because that solver reduces over it and needs it zero.
    # A caller with no padding passes ``n == stride``, which makes the guard unreachable.
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
    # ``out_state`` is the round-loop state array (``array.LOOP_ROUND`` / ``LOOP_CONDITION``), the
    # iteration count in the first slot. The loop runs while *any* column is still above its own
    # tolerance -- the worst-case rule the batching exists for -- and stops at ``maxiter``.
    # Launch with ``dim=1``: the column count is the right-hand-side count, a handful.
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1
    keep = wp.int32(0)
    for c in range(n_columns):
        if r_norm_sq[0, c] > atol_sq[c]:
            keep = 1
    if out_state[LOOP_ROUND] >= maxiter:
        keep = 0
    out_state[LOOP_CONDITION] = keep


# -------------------------------------------------------------------------------------------
# Block conjugate gradient (s = 2): a shared Krylov subspace across exactly two columns.
# -------------------------------------------------------------------------------------------

# Relative Tikhonov floor added to a 2x2 Gram matrix's diagonal before it is solved, guarding the
# near-rank-deficiency breakdown block CG is known for (two directions going nearly parallel) --
# see ``_BlockCg2`` in ``triwarp/linalg.py`` for what this guard does and does not recover.
BLOCK_CG_REG_EPS = wp.constant(wp.float64(1e-10))


@wp.func
def solve_sym2x2(
    g00: wp.float64,
    g01: wp.float64,
    g11: wp.float64,
    rhs00: wp.float64,
    rhs01: wp.float64,
    rhs11: wp.float64,
) -> tuple[wp.float64, wp.float64, wp.float64, wp.float64]:
    # Solve ``G X = RHS`` for the dense 2x2 ``X``, where ``G`` and ``RHS`` are symmetric 2x2
    # matrices carried as their three independent entries (``[00, 01, 11]``, ``01 == 10``) but
    # ``X`` is returned in full (``a00, a01, a10, a11``) because a product of two symmetric
    # matrices is not symmetric in general.
    #
    # ``G`` is a Gram matrix (``P^T A P`` or ``R^T Z``) and therefore positive semi-definite; it is
    # singular exactly when the block's two directions have gone linearly dependent -- the
    # classical block-CG breakdown. The guard here is a Tikhonov floor on the diagonal,
    # proportional to the trace: negligible relative error in the well-conditioned case (the trace
    # is the matrix's own scale) and enough to keep the solve finite in the near-singular one. It
    # is not a deflation: a genuinely rank-deficient block degrades toward the regularized system's
    # answer rather than recovering the two-column convergence rate, which is why this stays "a
    # guard against breakdown" rather than "a fix for it" -- see ``_BlockCg2``'s Notes.
    trace = g00 + g11
    reg = BLOCK_CG_REG_EPS * trace
    rg00 = g00 + reg
    rg11 = g11 + reg
    det = rg00 * rg11 - g01 * g01
    det_safe = wp.max(det, wp.float64(1e-300))
    inv_det = wp.float64(1.0) / det_safe
    inv00 = rg11 * inv_det
    inv01 = -g01 * inv_det
    inv11 = rg00 * inv_det
    a00 = inv00 * rhs00 + inv01 * rhs01
    a01 = inv00 * rhs01 + inv01 * rhs11
    a10 = inv01 * rhs00 + inv11 * rhs01
    a11 = inv01 * rhs01 + inv11 * rhs11
    return a00, a01, a10, a11


@wp.kernel
def block_cg_gram3_partials(
    a: wp.array[wp.float64],
    b: wp.array[wp.float64],
    stride: wp.int32,
    out_partials: wp.array2d[wp.float64],
) -> None:
    # Per-tile partial sums of the symmetric 2x2 Gram of ``(a0, a1)`` against ``(b0, b1)``, both
    # stacked as one flat ``2 * stride`` array (column 0 at ``[0, stride)``, column 1 at
    # ``[stride, 2 * stride)``). Launch tiled over ``(stride / CG_TILE,)`` with
    # ``block_dim=CG_TILE``; ``stride`` is padded to a whole number of tiles and zero-filled past
    # ``n``, so there is no ragged tail (see ``cg_dot_partials`` for the cost of one).
    #
    # Used twice: once for ``P^T A P`` (``a = p``, ``b = Ap``) and once, at setup only, for the
    # initial ``R^T Z``. Only three of the four entries are independent -- ``a0.b1 == a1.b0`` when
    # ``a`` and ``b`` are related through a symmetric operator or a diagonal preconditioner, both of
    # which hold at every call site in this module.
    blk, t = wp.tid()
    offset = blk * CG_TILE
    tile_a0 = wp.tile_load(a, shape=CG_TILE, offset=offset, storage="register")
    tile_a1 = wp.tile_load(a, shape=CG_TILE, offset=stride + offset, storage="register")
    tile_b0 = wp.tile_load(b, shape=CG_TILE, offset=offset, storage="register")
    tile_b1 = wp.tile_load(b, shape=CG_TILE, offset=stride + offset, storage="register")
    s00 = wp.tile_sum(tile_a0 * tile_b0)[0]
    s01 = wp.tile_sum(tile_a0 * tile_b1)[0]
    s11 = wp.tile_sum(tile_a1 * tile_b1)[0]
    if t == 0:
        out_partials[0, blk] = s00
        out_partials[1, blk] = s01
        out_partials[2, blk] = s11


@wp.kernel
def block_cg_gram3_finalize(
    partials: wp.array2d[wp.float64], n_blocks: wp.int32, out_g: wp.array[wp.float64]
) -> None:
    # Sum one of the three Gram rows cooperatively. Launch tiled over ``(3,)`` with
    # ``block_dim=CG_TILE``; the partial buffer's block axis is padded to a multiple of ``CG_TILE``
    # and zeroed once at allocation (see ``cg_dot_finalize``).
    row, t = wp.tid()
    values = partials[row]
    acc = wp.tile_zeros(shape=CG_TILE, dtype=wp.float64)
    for s in range((n_blocks + CG_TILE - 1) // CG_TILE):
        acc += wp.tile_load(values, shape=CG_TILE, offset=s * CG_TILE, storage="register")
    total = wp.tile_sum(acc)[0]
    if t == 0:
        out_g[row] = total


@wp.kernel
def block_cg_gram5_partials(
    r: wp.array[wp.float64],
    z: wp.array[wp.float64],
    stride: wp.int32,
    out_partials: wp.array2d[wp.float64],
) -> None:
    # Per-tile partials of the five scalars a post-update block CG iteration needs in one pass over
    # ``r``/``z``: the two per-column true-residual self-dots (``r0.r0``, ``r1.r1``, the
    # convergence quantity) and the symmetric ``R^T Z`` Gram's three independent entries (``r0.z0``,
    # ``r0.z1``, ``r1.z1``, the next iteration's ``beta`` right-hand side). One pass rather than two
    # separate reductions, since all four tiles are already resident. Also (ab)used at setup with
    # ``z = r`` to get the initial ``||b||`` alone -- the three Gram entries that call produces are
    # discarded.
    blk, t = wp.tid()
    offset = blk * CG_TILE
    tile_r0 = wp.tile_load(r, shape=CG_TILE, offset=offset, storage="register")
    tile_r1 = wp.tile_load(r, shape=CG_TILE, offset=stride + offset, storage="register")
    tile_z0 = wp.tile_load(z, shape=CG_TILE, offset=offset, storage="register")
    tile_z1 = wp.tile_load(z, shape=CG_TILE, offset=stride + offset, storage="register")
    rr0 = wp.tile_sum(tile_r0 * tile_r0)[0]
    rr1 = wp.tile_sum(tile_r1 * tile_r1)[0]
    rz00 = wp.tile_sum(tile_r0 * tile_z0)[0]
    rz01 = wp.tile_sum(tile_r0 * tile_z1)[0]
    rz11 = wp.tile_sum(tile_r1 * tile_z1)[0]
    if t == 0:
        out_partials[0, blk] = rr0
        out_partials[1, blk] = rr1
        out_partials[2, blk] = rz00
        out_partials[3, blk] = rz01
        out_partials[4, blk] = rz11


@wp.kernel
def block_cg_gram5_finalize(
    partials: wp.array2d[wp.float64], n_blocks: wp.int32, out_g5: wp.array[wp.float64]
) -> None:
    # Sum one of the five rows ``block_cg_gram5_partials`` wrote. Launch tiled over ``(5,)`` with
    # ``block_dim=CG_TILE``.
    row, t = wp.tid()
    values = partials[row]
    acc = wp.tile_zeros(shape=CG_TILE, dtype=wp.float64)
    for s in range((n_blocks + CG_TILE - 1) // CG_TILE):
        acc += wp.tile_load(values, shape=CG_TILE, offset=s * CG_TILE, storage="register")
    total = wp.tile_sum(acc)[0]
    if t == 0:
        out_g5[row] = total


@wp.kernel
def block_cg_absolute_tolerance(
    tol_sq: wp.float64, b_norm_sq: wp.array[wp.float64], out_atol_sq: wp.array[wp.float64]
) -> None:
    # Per-column squared stopping threshold ``(tol * ||b_c||)^2``, exactly two columns. Matches
    # ``cg_absolute_tolerance``'s convention with the absolute-floor term dropped: every call site
    # in this module passes it as ``0.0``, so carrying it here would be dead weight.
    c = wp.int32(wp.tid())
    out_atol_sq[c] = tol_sq * b_norm_sq[c]


@wp.kernel
def block_cg_solve_alpha(
    g: wp.array[wp.float64], rz_old: wp.array[wp.float64], out_alpha: wp.array[wp.float64]
) -> None:
    # ``alpha = (P^T A P)^-1 (R^T Z)_old``, both 2x2. Launch with ``dim=1``: the whole solve is nine
    # flops on scalars already in global memory, dwarfed by its own launch, so it is not folded into
    # the finalize kernel that produced ``g`` -- that kernel's three outputs are still spread across
    # different thread blocks with no barrier between them (Warp exposes no grid-wide barrier; see
    # CLAUDE.md section 12.2), so they are only all visible to a *subsequent* launch.
    a00, a01, a10, a11 = solve_sym2x2(g[0], g[1], g[2], rz_old[0], rz_old[1], rz_old[2])
    out_alpha[0] = a00
    out_alpha[1] = a01
    out_alpha[2] = a10
    out_alpha[3] = a11


@wp.kernel
def block_cg_solve_beta(
    rz_old: wp.array[wp.float64],
    g5: wp.array[wp.float64],
    out_beta: wp.array[wp.float64],
    out_rz_old: wp.array[wp.float64],
) -> None:
    # ``beta = (R^T Z)_old^-1 (R^T Z)_new``, then carry ``(R^T Z)_new`` forward as next iteration's
    # ``rz_old`` -- folding CG's ``rz_old = rz_new`` copy into this launch the same way
    # ``cg_dot_finalize``'s ``carry`` argument does for the single-column iteration.
    rz_new0 = g5[2]
    rz_new1 = g5[3]
    rz_new2 = g5[4]
    b00, b01, b10, b11 = solve_sym2x2(rz_old[0], rz_old[1], rz_old[2], rz_new0, rz_new1, rz_new2)
    out_beta[0] = b00
    out_beta[1] = b01
    out_beta[2] = b10
    out_beta[3] = b11
    out_rz_old[0] = rz_new0
    out_rz_old[1] = rz_new1
    out_rz_old[2] = rz_new2


@wp.kernel
def block_cg_step_x_r_z(
    stride: wp.int32,
    n: wp.int32,
    alpha: wp.array[wp.float64],
    inv_diag: wp.array[wp.float64],
    p: wp.array[wp.float64],
    ap: wp.array[wp.float64],
    out_x: wp.array2d[wp.float64],
    out_r: wp.array[wp.float64],
    out_z: wp.array[wp.float64],
) -> None:
    # ``X += P alpha``; ``R -= (AP) alpha``; ``Z = M^-1 R`` (Jacobi fused in), for exactly two
    # columns. Unlike ``cg_step_x_r_z`` above, one thread here owns one *row* rather than one
    # (column, row) pair -- the update mixes both columns of that row, so it has to. Launch with
    # ``dim=stride``: half ``_BatchedCg``'s per-iteration thread count for the same system, each
    # thread doing twice the work.
    #
    # ``out_x`` is the caller's own ``(2, n)`` buffer, unpadded; ``out_r`` doubles as the read of
    # the previous residual and the write of the new one, exactly as ``cg_advance_x_r`` does -- no
    # other thread touches this row's two entries, so the read-before-write is safe.
    local = wp.int32(wp.tid())
    p0 = p[local]
    p1 = p[stride + local]
    ap0 = ap[local]
    ap1 = ap[stride + local]
    a00 = alpha[0]
    a01 = alpha[1]
    a10 = alpha[2]
    a11 = alpha[3]
    if local < n:
        out_x[0, local] = out_x[0, local] + a00 * p0 + a10 * p1
        out_x[1, local] = out_x[1, local] + a01 * p0 + a11 * p1
    r0_old = out_r[local]
    r1_old = out_r[stride + local]
    r0_new = r0_old - (a00 * ap0 + a10 * ap1)
    r1_new = r1_old - (a01 * ap0 + a11 * ap1)
    out_r[local] = r0_new
    out_r[stride + local] = r1_new
    d = inv_diag[local]
    out_z[local] = d * r0_new
    out_z[stride + local] = d * r1_new


@wp.kernel
def block_cg_step_p(
    stride: wp.int32,
    beta: wp.array[wp.float64],
    z: wp.array[wp.float64],
    out_p: wp.array[wp.float64],
) -> None:
    # ``P = Z + P beta``, mixing both columns per row; see ``block_cg_step_x_r_z`` for why one
    # thread owns a whole row here rather than one column.
    local = wp.int32(wp.tid())
    p0_old = out_p[local]
    p1_old = out_p[stride + local]
    z0 = z[local]
    z1 = z[stride + local]
    b00 = beta[0]
    b01 = beta[1]
    b10 = beta[2]
    b11 = beta[3]
    out_p[local] = z0 + b00 * p0_old + b10 * p1_old
    out_p[stride + local] = z1 + b01 * p0_old + b11 * p1_old


@wp.kernel
def block_cg_advance_condition(
    maxiter: wp.int32,
    residual_sq: wp.array[wp.float64],
    atol_sq: wp.array[wp.float64],
    out_state: wp.array[wp.int32],
) -> None:
    # The two-column worst-case stopping rule, written directly rather than through
    # ``cg_advance_condition``'s generic ``n_columns`` loop: this solver only ever has two.
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1
    keep = wp.int32(0)
    if residual_sq[0] > atol_sq[0]:
        keep = 1
    if residual_sq[1] > atol_sq[1]:
        keep = 1
    if out_state[LOOP_ROUND] >= maxiter:
        keep = 0
    out_state[LOOP_CONDITION] = keep
