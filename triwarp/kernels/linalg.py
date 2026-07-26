"""
Kernels for the shared sparse linear-algebra layer (``triwarp/linalg.py``).

Holds the reduced-system assembly used by every fixed-value quadratic solve
(``harmonic`` / ``tutte`` / ``lscm`` / ``arap``).

!!! note
    ``smoothing``'s ``dirichlet_system_triplets`` / ``laplacian_ls_triplets`` are deliberately
    *not* merged in here. They look similar but solve a different problem: they **build** the
    operator ``A = D - W`` from a weight CSR (plus a stabilizer, with a reserved diagonal slot per
    row), whereas ``interior_system_triplets`` **extracts** the free-free block of an operator that
    already exists. Routing them through this kernel would force an extra full matrix build.
"""

import warp as wp


@wp.kernel
def interior_system_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    fixed_values: wp.array2d[wp.float64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # One thread per row ``i`` of the operator ``Q`` (positive semi-definite). Emit the free-free
    # block into COO triplets (remapped to the compact free index) and move fixed-column
    # contributions to the right-hand side: ``Q_uu x_u = -Q_ub bc``, generalized to ``n_rhs``
    # columns (``fixed_values`` is ``(n_rhs, n_dofs)``, ``out_rhs`` is ``(n_rhs, n_free)``). Fixed
    # rows leave their pre-zeroed output slots untouched. Assembled in float64: the biharmonic
    # (k > 1) operator squares the Laplacian condition number, beyond float32 CG's reach; LSCM's
    # coupled u/v system is likewise ill-conditioned.
    i = int(wp.tid())
    if fixed_mask[i]:
        return
    ri = free_map[i]
    start = offsets[i]
    end = offsets[i + 1]
    # Pass 1: free-free triplets into slot ``e``; fixed columns leave the pre-zeroed slot as-is.
    for e in range(start, end):
        j = columns[e]
        if not fixed_mask[j]:
            out_rows[e] = ri
            out_cols[e] = free_map[j]
            out_vals[e] = values[e]
    # Pass 2: per right-hand-side column, accumulate the fixed-column contributions. The thread owns
    # row ``ri`` of ``out_rhs`` exclusively, so a single register accumulator and write suffice.
    for c in range(fixed_values.shape[0]):
        acc = wp.float64(0.0)
        for e in range(start, end):
            j = columns[e]
            if fixed_mask[j]:
                acc -= values[e] * fixed_values[c, j]
        out_rhs[c, ri] = acc
