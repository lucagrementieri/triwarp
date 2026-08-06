# warp.sparse API

Source: https://nvidia.github.io/warp/stable/api_reference/warp_sparse.html (Warp 1.16.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Block-sparse (BSR/CSR) matrix support. Import via `from warp.sparse import ...` or `wp.sparse.<name>`.

## Classes
- `BsrMatrix` — untyped base class for BSR and CSR matrices.
- `bsr_matrix_t` — typed BSR matrix class.
- `bsr_mm_work_arrays` — persists temporary work buffers across matrix-matrix multiply calls.
- `bsr_axpy_work_arrays` — persists temporary work buffers across addition calls.

## `BsrMatrix` members
Fields `nrow` / `ncol` / `nnz` / `offsets` / `row_counts` / `columns` / `values`; properties
`shape`, `dtype`, `device`, `scalar_type`, `scalar_values`, `block_shape`, `block_size`,
`requires_grad`.
- `nnz_sync()` — host-sync the block count after a topology change (a device readback).
- `notify_nnz_changed(nnz=None, nnz_capacity=None)` — declare a new block count without a readback,
  for when the caller already knows it (used by `triwarp.linalg`).
- `copy_nnz_async()` — **deprecated in 1.16**; use `notify_nnz_changed()` instead.
- `status_sync()` / `status_message()` / `clear_status()` — read/clear the status code below.
- `uncompress_rows(out=None)` / `transpose()`.

## Status Codes (new in 1.15)
- `BSR_STATUS_SUCCESS` — operation completed successfully.
- `BSR_STATUS_ROW_CAPACITY_EXCEEDED` — a topology-changing operation exceeded the padded row capacity.

## Matrix Construction
- `bsr_zeros(shape, row_capacity=...) -> BsrMatrix` — empty BSR/CSR matrix of given shape; `row_capacity` (1.15) reserves padded per-row block storage.
- `bsr_identity(n) -> BsrMatrix` — square identity matrix.
- `bsr_diag(block_value) -> BsrMatrix` — block-diagonal matrix from a block value or array.
- `bsr_from_triplets(rows, cols, values, shape) -> BsrMatrix` — build from COO triplets.
- `bsr_copy(A) -> BsrMatrix` — copy, optionally changing scalar type.
- `bsr_compress(src, prune_numerical_zeros=True, inplace=False, topology=None) -> BsrMatrix` — (1.15) sort/coalesce active blocks and compact storage; `topology="padded"` keeps reserved row capacity. **Caution:** calling it on a matrix rebuilt from its own CSR via a second `bsr_from_triplets` makes the next `bsr_mm` crash with an illegal memory access — see `issue_report.md`.

## Matrix Operations
- `bsr_mv(A, x, y, alpha, beta)` — sparse matrix-vector product with scaling.
- `bsr_mm(x, y, z, alpha, beta) -> BsrMatrix` — sparse matrix-matrix multiply with scaling.
- `bsr_axpy(x, y, alpha, beta) -> BsrMatrix` — sparse matrix addition with scaling.
- `bsr_scale(x, alpha) -> BsrMatrix` — scale matrix by scalar.
- `bsr_assign(dest, src)` — copy contents from src to dest.

## Matrix Transformation
- `bsr_transposed(A) -> BsrMatrix` — return transposed copy.
- `bsr_set_transpose(dest, src)` — assign transpose to destination.

## Diagonal Operations
- `bsr_get_diag(A) -> array` — array of diagonal blocks.
- `bsr_set_diag(A, blocks)` — set matrix as block-diagonal.
- `bsr_set_identity(A)` — set matrix as identity.

## Matrix Initialization
- `bsr_set_zero(A, shape)` — set to zero, optionally resizing.
- `bsr_set_from_triplets(A, rows, cols, values)` — fill from COO triplets.

## Utility Functions
- `bsr_block_index(A, row, col) -> int` — block index at block-coords, or -1.
- `bsr_row_index(A, block) -> int` — row index containing a block, or -1.
