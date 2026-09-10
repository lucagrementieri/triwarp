"""
Smoothed-aggregation algebraic multigrid: the setup passes and the V-cycle.

**Why a hierarchy at all.** Jacobi-preconditioned conjugate gradient's iteration count grows with
the mesh: on the least-squares operator ``smoothing.smooth_region`` builds it measures 1 753
iterations at 2 043 unknowns and 6 521 at 8 987 -- 3.7x the count for 4.4x the size. A multigrid
V-cycle attacks the low-frequency error the smoother cannot see, so the count stops growing; the
whole item is whether one cycle costs less than the iterations it removes, which is a *launch*
question on this hardware and not an arithmetic one. See ``triwarp.linalg`` for the measurement.

**Aggregation is the only part that is not a library call**, and it is a parallel maximal
independent set, which this package already runs twice on device (``sample.dart_select_minima``'s
randomized-priority selection and ``remesh``'s hashed-key independent set). The roots of a
**distance-2** MIS on the operator's off-diagonal graph are at least three hops apart, so their
one-rings are disjoint and every remaining node is within two hops of exactly one candidate root;
spreading the root's label two hops therefore tiles the graph into aggregates. Distance-2 is reached
without building the squared graph: propagating the lexicographic maximum of
``(state, priority, index)`` over one-hop neighbours **twice** gives every node the maximum over its
two-hop ball, which is the Bell/Dalton/Olson formulation.

Measured against ``pyamg``'s serial ``standard_aggregation`` on the same operator, same smoother,
same cycle and same coarse solve -- the comparison that decides whether the parallel aggregation
gives anything up: **288 iterations against 269** at 2 043 unknowns and **554 against 495** at
8 987, i.e. 7-12 % more. That is the price of the parallelism and it is small.

The rest is ``warp.sparse``: the tentative prolongator is one entry per row (the constant
near-nullspace vector, normalized per aggregate), the smoothed prolongator is
``P = (I - w D^-1 A) P0`` through one ``bsr_mm`` and one ``bsr_axpy``, and the coarse operator is
the Galerkin product ``P^T A P``. Operator complexity comes out at **1.02-1.03**, so the coarse
levels are nearly free and a cycle's cost is its fine level.

The per-level inverse diagonal is ``array.inverse_or_one`` mapped over the operator's diagonal,
not a copy of the conjugate gradient's: the quantity is the same one the Jacobi preconditioner
needs, down to mapping a zero diagonal to 1 rather than to infinity -- which the least-squares
operators here rely on, since they carry empty rows (297 of 8 987 on ``bunny``) for free vertices
no equation reaches. The strength test's ``sqrt(|A_ii|)`` is ``array.sqrt_abs`` over the same
diagonal.
"""

import warp as wp

# Node states for the distance-2 maximal independent set. The encoding is ordered rather than
# arbitrary: a root must win any maximum (it vetoes every node in its two-hop ball) and an excluded
# node must lose to every undecided one (it can no longer veto anything).
MG_EXCLUDED = wp.constant(wp.int32(0))
MG_UNDECIDED = wp.constant(wp.int32(1))
MG_ROOT = wp.constant(wp.int32(2))

# Layout of the packed comparison key: state in bits 60-61, priority in bits 32-59, index in bits
# 0-31. Two hops of a plain integer ``max`` then implement the lexicographic order, and the key
# stays positive so a signed ``wp.int64`` compares correctly. 28 bits of priority is plenty -- the
# index breaks any tie, which is what makes the aggregation independent of thread order.
_MG_STATE_SHIFT = wp.constant(wp.int64(60))
_MG_PRIORITY_SHIFT = wp.constant(wp.int64(32))
_MG_PRIORITY_MASK = wp.constant(wp.uint32(0x0FFFFFFF))

MG_UNAGGREGATED = wp.constant(wp.int32(-1))


@wp.func
def mg_is_strong(
    value: wp.float64,
    scaled_diagonal_row: wp.float64,
    scaled_diagonal_column: wp.float64,
    theta: wp.float64,
) -> wp.bool:
    """Whether ``A_ij`` is a strong connection: ``|A_ij| >= theta * sqrt(A_ii * A_jj)``."""
    # ``scaled_diagonal`` carries ``sqrt(|A_ii|)``, so the product is the geometric mean and no
    # square root runs per edge. At ``theta = 0`` this is unconditionally true -- including for an
    # explicit zero, since ``0 >= 0`` -- which is what makes the unfiltered aggregation the exact
    # ``theta = 0`` case of this one rather than a separate code path.
    return wp.abs(value) >= theta * scaled_diagonal_row * scaled_diagonal_column


@wp.func
def mis_key(state: wp.int32, priority: wp.uint32, index: wp.int32) -> wp.int64:
    masked = priority & _MG_PRIORITY_MASK
    return (
        (wp.int64(state) << _MG_STATE_SHIFT)
        | (wp.int64(masked) << _MG_PRIORITY_SHIFT)
        | wp.int64(index)
    )


@wp.kernel
def mis_seed_keys(
    state: wp.array[wp.int32], priority: wp.array[wp.uint32], out_key: wp.array[wp.int64]
) -> None:
    i = wp.int32(wp.tid())
    out_key[i] = mis_key(state[i], priority[i], i)


@wp.kernel
def mis_propagate(
    key: wp.array[wp.int64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    out_key: wp.array[wp.int64],
) -> None:
    # One hop of the lexicographic maximum over the operator's *strong off-diagonal* graph -- the
    # diagonal is skipped because a node is not its own neighbour, and the node's own key is folded
    # in separately so the reduction is over the closed neighbourhood. Two launches of this give the
    # maximum over the two-hop ball, which is the distance-2 test without a squared graph.
    #
    # The strength test is applied here rather than by materializing a filtered graph, so a level
    # pays no extra allocation and ``theta = 0`` is bit-exactly the unfiltered aggregation.
    #
    # Reads ``key`` and writes a second buffer, so the caller swaps rather than synchronizing.
    i = wp.int32(wp.tid())
    best = key[i]
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if j != i and mg_is_strong(values[k], scaled_diagonal[i], scaled_diagonal[j], theta):
            best = wp.max(best, key[j])
    out_key[i] = best


@wp.kernel
def mis_decide(
    reduced_key: wp.array[wp.int64],
    priority: wp.array[wp.uint32],
    state: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
    out_undecided: wp.array[wp.int32],
) -> None:
    # One round's verdict for every still-undecided node: a root anywhere in its two-hop ball
    # excludes it, and otherwise being the maximum of that ball makes it a root. Anything else waits
    # for the next round, and ``out_undecided`` is what tells the host whether there is one.
    i = wp.int32(wp.tid())
    current = state[i]
    if current != MG_UNDECIDED:
        out_state[i] = current
        return
    best = reduced_key[i]
    if wp.int32(best >> _MG_STATE_SHIFT) == MG_ROOT:
        out_state[i] = MG_EXCLUDED
        return
    if best == mis_key(current, priority[i], i):
        out_state[i] = MG_ROOT
        return
    out_state[i] = MG_UNDECIDED
    wp.atomic_add(out_undecided, 0, wp.int32(1))


@wp.func
def mis_root_flag(state: wp.int32) -> wp.int32:
    """Whether ``state`` claims an aggregate, as the 0/1 flag ``wp.utils.array_scan`` wants."""
    # An inclusive scan of these numbers the aggregates consecutively and its last element is the
    # aggregate count.
    #
    # The test is "not excluded" rather than "is a root" so that a node still undecided when the
    # round cap is reached becomes an aggregate of its own instead of an unaggregated hole. The
    # selection normally settles well inside the cap and the two readings then coincide.
    #
    # A ``@wp.func`` rather than a kernel because the wrapper maps it (CLAUDE.md section 4). It
    # returns the flag directly rather than composing ``array.not_equal`` with an
    # ``array_cast(bool -> int32)``, which would be two device passes and a second buffer.
    return wp.where(state != MG_EXCLUDED, wp.int32(1), wp.int32(0))


@wp.func
def aggregate_label(state: wp.int32, scan_pos: wp.int32) -> wp.int32:
    # An excluded node has no aggregate; everything else takes the (0-based) index its inclusive
    # scan position names. A `wp.map` target (CLAUDE.md section 4) rather than a kernel: the body
    # is one indexed assignment reading only `state[i]` / `scan_pos[i]`, the elementwise-map scan's
    # own definition of a trivial kernel. Left un-hoisted (no `return_kernel=True`) at its one call
    # site inside `linalg._multigrid_aggregate`'s per-level loop: hoisting would need a dummy
    # int32 array allocated before the loop just to seed the kernel factory, or threading the
    # cached kernel object through the function's signature, for an ~11 us/level saving against a
    # setup section already measured at 9-18 ms (CLAUDE.md section 13) -- not where that cost lives.
    return wp.where(state != MG_EXCLUDED, scan_pos - wp.int32(1), MG_UNAGGREGATED)


@wp.kernel
def spread_aggregate_labels(
    label: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    scaled_diagonal: wp.array[wp.float64],
    theta: wp.float64,
    out_label: wp.array[wp.int32],
) -> None:
    # An unlabelled node adopts a neighbour's aggregate, largest id winning so the choice does not
    # depend on thread order. Two launches cover the two hops the MIS guarantees are enough.
    #
    # The spread walks the same *strong* graph the independent set was selected on, which is what
    # keeps the tiling consistent: every excluded node has a root within two strong hops precisely
    # because the exclusion came from a strong-graph propagation.
    i = wp.int32(wp.tid())
    best = label[i]
    if best != MG_UNAGGREGATED:
        out_label[i] = best
        return
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        if j != i and mg_is_strong(values[k], scaled_diagonal[i], scaled_diagonal[j], theta):
            best = wp.max(best, label[j])
    out_label[i] = best


@wp.kernel
def aggregate_sizes(label: wp.array[wp.int32], out_sizes: wp.array[wp.int32]) -> None:
    i = wp.int32(wp.tid())
    wp.atomic_add(out_sizes, label[i], wp.int32(1))


@wp.kernel
def tentative_prolongator_triplets(
    label: wp.array[wp.int32],
    sizes: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # The tentative prolongator: one entry per row, carrying the constant near-nullspace vector
    # restricted to the node's aggregate and normalized so every column has unit norm. Exactly one
    # triplet per row and no duplicates, so the build's ``nnz`` is exact.
    i = wp.int32(wp.tid())
    aggregate = label[i]
    out_rows[i] = i
    out_columns[i] = aggregate
    out_values[i] = wp.float64(1.0) / wp.sqrt(wp.float64(sizes[aggregate]))


@wp.kernel
def scale_rows(
    offsets: wp.array[wp.int32],
    row_scale: wp.array[wp.float64],
    factor: wp.float64,
    values: wp.array[wp.float64],
) -> None:
    # Row-scale a matrix in place: the prolongation smoother needs ``-w D^-1 (A P0)``, and scaling
    # the product's values is one pass where ``bsr_mm`` against a diagonal matrix would be another
    # sparse product. ``values`` is both the input and the result.
    i = wp.int32(wp.tid())
    scale = factor * row_scale[i]
    for k in range(offsets[i], offsets[i + 1]):
        values[k] = values[k] * scale


@wp.func
def csr_row_dot(
    row: wp.int32,
    x_offset: wp.int32,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
) -> wp.float64:
    # One CSR row against one column of ``x``. Shared by the cycle's mat-vec and the power
    # iteration's step below, which differ only in what they do with the result -- and the row bound
    # comes from ``offsets`` alone, so neither depends on the matrix's ``nnz`` field being fresh.
    total = wp.float64(0.0)
    for k in range(offsets[row], offsets[row + 1]):
        total += values[k] * x[x_offset + columns[k]]
    return total


@wp.kernel
def csr_matvec(
    n_rows: wp.int32,
    x_stride: wp.int32,
    y_stride: wp.int32,
    accumulate: wp.int32,
    alpha: wp.float64,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
) -> None:
    # ``y = alpha * A x`` (or ``y += alpha * A x``) for **every column at once**, which is the whole
    # reason this exists rather than ``warp.sparse.bsr_mv``: that takes one vector, so a cycle over
    # three right-hand sides pays three launches per mat-vec, and the V-cycle is launch-bound at
    # these sizes. Measured on ``smooth_region``'s operator, three columns, captured: **34 launches
    # and 189 us per cycle through ``bsr_mv`` against 15 and 85 through this**, with the same
    # answer.
    #
    # ``x_stride`` and ``y_stride`` differ whenever the operator is rectangular -- the restriction
    # reads at the fine pitch and writes at the coarse one -- and either may exceed its own row
    # count, since the top level's vectors carry the conjugate-gradient state's tile padding.
    # ``accumulate`` is warp-uniform and folds the prolongation's correction into the same kernel.
    # ``alpha`` mirrors ``bsr_mv``'s own scale argument -- every caller here passes a compile-time
    # constant (``1.0`` or ``-1.0``), so it costs one multiply per row rather than a second kernel.
    t = wp.int32(wp.tid())
    column = t // n_rows
    row = t % n_rows
    total = alpha * csr_row_dot(row, column * x_stride, offsets, columns, values, x)
    slot = column * y_stride + row
    if accumulate != wp.int32(0):
        out_y[slot] += total
    else:
        out_y[slot] = total


@wp.kernel
def power_step(
    inv_diag: wp.array[wp.float64],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
) -> None:
    # ``y = D^-1 A x``: one whole step of the power iteration that estimates the spectral radius, in
    # one launch. It differs from ``csr_matvec`` above only in folding the diagonal scale in and in
    # being single-column -- and that fusion is the point, because the setup is launch-bound: the
    # step used to be a ``bsr_mv`` plus a ``scaled_diagonal_apply``, and an uncaptured ``bsr_mv``
    # costs ~0.1 ms whatever its nnz. Writing a second buffer rather than updating ``x`` in place is
    # what lets it be one launch; the caller swaps the two.
    i = wp.int32(wp.tid())
    out_y[i] = inv_diag[i] * csr_row_dot(i, wp.int32(0), offsets, columns, values, x)


@wp.kernel
def random_signs(seed: wp.int32, out_x: wp.array[wp.float64]) -> None:
    # Start vector for the power iteration that estimates the spectral radius of ``D^-1 A``.
    #
    # Random rather than constant, because on a Laplacian-like operator the dominant eigenvector is
    # the highest-frequency mode and a constant vector is nearly orthogonal to it. And *signs*
    # rather than uniform values, because then the norm is exactly ``sqrt(n)`` and the caller needs
    # one host readback for the whole estimate instead of two -- which at these sizes is most of
    # what the estimate costs.
    #
    # Deliberately *not* folded into ``array.random_priorities``, which has the same shape over
    # ``wp.randu``: that one draws a total order on the elements, this one draws a start vector
    # whose norm is known in closed form. Same tokens, different quantities.
    i = wp.int32(wp.tid())
    out_x[i] = wp.where(
        wp.randi(wp.rand_init(seed, i)) < wp.int32(0), wp.float64(-1.0), wp.float64(1.0)
    )


@wp.kernel
def jacobi_sweep(
    n: wp.int32,
    stride: wp.int32,
    inv_diag: wp.array[wp.float64],
    omega: wp.float64,
    rhs: wp.array[wp.float64],
    operator_x: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # One damped-Jacobi sweep, ``x += w D^-1 (b - A x)``, with ``A x`` already in hand. ``out_x`` is
    # both the input and the result, which is what the sweep means.
    #
    # ``row = t % stride`` then ``if row >= n: return`` is the padded-row guard this file's cycle
    # kernels share; what the pad is and why it must stay unwritten is written out once, on
    # ``algorithms/conjugate_gradient.scaled_diagonal_apply``, which owns the padding contract. Not
    # factored into a ``@wp.func``: the caller still needs the flat ``t`` for its own indexing, so
    # a helper returning ``-1`` for the pad renames the three lines rather than removing any.
    t = wp.int32(wp.tid())
    row = t % stride
    if row >= n:
        return
    out_x[t] += omega * inv_diag[row] * (rhs[t] - operator_x[t])


@wp.kernel
def residual(
    n: wp.int32,
    stride: wp.int32,
    rhs: wp.array[wp.float64],
    operator_x: wp.array[wp.float64],
    out_residual: wp.array[wp.float64],
) -> None:
    t = wp.int32(wp.tid())
    row = t % stride
    if row >= n:
        return
    out_residual[t] = rhs[t] - operator_x[t]


@wp.kernel
def dense_solve(
    n: wp.int32,
    stride: wp.int32,
    inverse: wp.array2d[wp.float64],
    rhs: wp.array[wp.float64],
    out_x: wp.array[wp.float64],
) -> None:
    # The coarsest level, solved against a pseudo-inverse factored on the host at setup: a few dozen
    # unknowns, where a dense row dot is cheaper than any iteration and -- unlike an iteration whose
    # count depends on the data -- it is a single launch inside the captured cycle. The *pseudo*
    # inverse because the coarse operator inherits the fine one's null space.
    t = wp.int32(wp.tid())
    column = t // stride
    row = t % stride
    if row >= n:
        return
    total = wp.float64(0.0)
    for k in range(n):
        total += inverse[row, k] * rhs[column * stride + k]
    out_x[t] = total
