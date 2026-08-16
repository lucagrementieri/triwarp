"""
Kernels for the shared sparse linear-algebra layer (``triwarp/linalg.py``).

The **reduced-system assembly** used by every fixed-value quadratic solve (``harmonic`` / ``tutte``
/ ``lscm`` / ``arap``). It is a **CSR-to-CSR** extraction in two passes -- count the surviving
entries per free row, then fill them -- rather than a COO emission handed to
``warp.sparse.bsr_from_triplets``. The input rows come out of a CSR, so they are already row-major,
already column-sorted and already duplicate-free, and ``free_map`` is monotone, so the extracted
row is sorted by construction: the sort and duplicate-accumulation a triplet build performs are pure
waste here. Measured on ``benchmarks/test_linalg.py``'s ``saddle`` case, that build was **20.7 ms in
a single launch, 91 % of all device time** in ``min_quad_with_fixed``.

The module's other half, the batched conjugate-gradient iteration that solves the system this
assembles, lives in ``triwarp.kernels.algorithms.conjugate_gradient``.

!!! note
    ``smoothing``'s ``dirichlet_system_triplets`` / ``laplacian_ls_triplets`` are deliberately
    *not* merged in here. They look similar but solve a different problem: they **build** the
    operator ``A = D - W`` from a weight CSR (plus a stabilizer, with a reserved diagonal slot per
    row), whereas these kernels **extract** the free-free block of an operator that already exists.
    Routing them through here would force an extra full matrix build.
"""

import warp as wp


@wp.kernel
def interior_row_counts(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    fixed_values: wp.array2d[wp.float64],
    out_counts: wp.array[wp.int32],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Pass 1 of the free-free extraction: one thread per row ``i`` of the operator ``Q`` (positive
    # semi-definite), counting the entries of that row whose column also survives. The right-hand
    # side rides along in the same launch because it needs the identical row scan and depends on
    # nothing this pass produces: fixed-column contributions move to the right-hand side as
    # ``Q_uu x_u = -Q_ub bc``, generalized to ``n_rhs`` columns (``fixed_values`` is
    # ``(n_rhs, n_dofs)``, ``out_rhs`` is ``(n_rhs, n_free)``). Assembled in float64: the biharmonic
    # (k > 1) operator squares the Laplacian condition number, beyond float32 CG's reach; LSCM's
    # coupled u/v system is likewise ill-conditioned.
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        return
    ri = free_map[i]
    start = offsets[i]
    end = offsets[i + 1]
    kept = wp.int32(0)
    for e in range(start, end):
        if not fixed_mask[columns[e]]:
            kept += 1
    # A free row every one of whose columns is pinned counts 0 and becomes an *empty* CSR row, not a
    # dropped one -- the scan of these counts is what keeps the row offsets in step.
    out_counts[ri] = kept
    # The thread owns row ``ri`` of ``out_rhs`` exclusively, so a single register accumulator and
    # write suffice per right-hand-side column.
    #
    # The nest re-reads the row once per right-hand side, which looks like an ``n_rhs``-fold read
    # amplification worth inverting (scan the row once into an ``n_rhs``-wide register vector).
    # Measured before building it, and it is not: this kernel is launched **once** per
    # ``min_quad_with_fixed``, one of 15 launches in a 1.6 ms solve, and ``n_rhs`` is **1** for the
    # default call -- so the inner loop runs a single iteration and there is nothing to invert. The
    # callers that pass more are ``lscm`` (2) and a vector-valued ``harmonic`` (3), where the whole
    # pass is still one launch against a conjugate-gradient solve whose iteration count is what the
    # benchmark's 12.6x spread between 50%- and 1%-pinned actually measures. A register vector would
    # also need a compile-time ``MAX_RHS`` cap, which is a new documented limitation bought for
    # nothing.
    for c in range(fixed_values.shape[0]):
        acc = wp.float64(0.0)
        for e in range(start, end):
            j = columns[e]
            if fixed_mask[j]:
                acc -= values[e] * fixed_values[c, j]
        out_rhs[c, ri] = acc


@wp.kernel
def interior_system_csr(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    out_offsets: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
    out_values: wp.array[wp.float64],
) -> None:
    # Pass 2: one thread per row ``i`` of ``Q``, writing that row's surviving entries into the slots
    # ``out_offsets`` reserved for it. ``out_offsets`` is the exclusive scan of
    # ``interior_row_counts``' output, so the destination range is exactly the right size and no two
    # threads overlap. Column order is inherited from ``Q``'s row and ``free_map`` is monotone
    # non-decreasing, so the emitted row is column-sorted by construction -- which is the whole
    # reason this can skip a triplet sort.
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        return
    ri = free_map[i]
    slot = out_offsets[ri]
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if not fixed_mask[j]:
            out_columns[slot] = free_map[j]
            out_values[slot] = values[e]
            slot += 1
