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

from typing import Any

import warp as wp

# Warp's own Householder QR. From ``warp._src.fem.linalg`` rather than the public
# ``warp.fem.linalg``: both bind the same two ``@wp.func``s, which inline here and trigger no fem
# codegen, but the public path executes ``warp/fem/__init__.py`` and eagerly loads the whole fem
# package -- measured on **Warp 1.16** at 0.24-0.29 s against 0.008-0.010 s, and 1.49 s against
# 1.18 s for ``import triwarp`` end to end. ``kernels/reduce.py`` reaches into ``warp._src`` on the
# same terms.
from warp._src.fem.linalg import householder_qr_decomposition, solve_triangular


@wp.func
def solve_normal_equations(matrix: Any, rhs: Any):
    """
    Solve a small dense symmetric system ``A x = b`` by Householder QR, at any rank.

    Returns ``(solution, ok)``; ``ok`` is False when the system is singular to ``1e-14``, in which
    case the returned vector is ``rhs`` unchanged. Reporting rather than raising, because the caller
    is a kernel: a per-vertex least-squares fit that fails on a degenerate 1-ring has to fall back,
    not abort the launch.

    ``|R[k, k]|`` is the norm of column k after the preceding reflections -- the QR analogue of the
    partial-pivot magnitude a Gaussian elimination would test, to within a ``sqrt(n)`` factor -- so
    a single threshold carries across ranks.

    **The rank is nowhere in this function, and that is the point.** It was written twice, as a 5x5
    for ``curvature``'s quadric fit and a 6x6 for ``smoothing``'s area-equalizing solve, because the
    singularity test was a ``for k in range(5)`` / ``range(6)`` loop and a generic matrix has no
    readable rank in kernel scope -- ``r.shape[0]`` is a ``WarpCodegenAttributeError`` at parse time
    on Warp 1.16. ``wp.min(wp.abs(wp.get_diag(r)))`` asks the identical question ("is some
    diagonal below tolerance") with no loop and no rank, which is what let the two collapse into
    one. Verified against ``numpy.linalg.solve`` at both ranks: max abs error 4.163e-17 at 5 and
    5.551e-17 at 6, with the singular case reporting ``ok=False`` at both.

    The two predicates were also compared directly, 810 finite matrices per rank -- 200 well
    conditioned, 200 near-singular spanning fourteen orders of magnitude of conditioning, and one
    exactly rank-deficient per column -- and they agree on **every** one. They part on exactly two
    inputs, and in the safe direction: a matrix carrying a ``nan`` or an ``inf`` entry passed the
    old loop (``wp.abs(nan) < tol`` is False, so no iteration rejected it) and now reports
    ``ok=False``. So a degenerate 1-ring that used to yield a ``nan`` fit silently now takes the
    caller's fallback, which is what both callers already do for a singular system.
    """
    q, r = householder_qr_decomposition(matrix)
    if wp.min(wp.abs(wp.get_diag(r))) < wp.float64(1e-14):
        return rhs, False
    return solve_triangular(r, wp.transpose(q) * rhs), True


@wp.func
def free_row(fixed_mask: wp.array[wp.bool], free_map: wp.array[wp.int32], i: wp.int32) -> wp.int32:
    # The compact row index degree of freedom ``i`` occupies in the reduced system, or ``-1`` when
    # it is pinned and has no row at all. The sentinel rather than an early return, because a
    # ``@wp.func`` cannot return for its caller.
    #
    # This does not shorten the three lines it replaces -- ``if ri < 0: return`` costs what
    # ``if fixed_mask[i]: return`` cost. What it buys is that the free/fixed/compact-index
    # convention is written down *once*, in the module that owns the elimination, instead of being
    # re-inferred at seven sites across four modules: ``free_map`` is defined only where
    # ``fixed_mask`` is False, it is monotone non-decreasing over the free indices (which is what
    # lets ``interior_row_entries`` skip a triplet sort), and reading it at a pinned index gives a
    # stale or out-of-range value rather than an error.
    if fixed_mask[i]:
        return wp.int32(-1)
    return free_map[i]


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
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
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
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    slot = out_offsets[ri]
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if not fixed_mask[j]:
            out_columns[slot] = free_map[j]
            out_values[slot] = values[e]
            slot += 1
