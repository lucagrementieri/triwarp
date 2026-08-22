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
  **`nnz` is a stale upper bound, `nnz_sync()` is the count.** After `bsr_from_triplets` the `nnz`
  field still holds the *triplet capacity* it was handed, duplicates included — measured 8400 against
  a true 4516 on a duplicate-emitting Laplacian build, and 3.4x on `laplacian.cotmatrix`. Size any
  buffer, slice or launch dim off `nnz_sync()` (or `offsets[nrow]`); sizing off `nnz` leaves a tail
  that is never written, and `bsr_from_triplets` will read it back as triplets. Out-of-range garbage
  indices are dropped silently, but garbage landing in range accumulates into a real entry.
  `nnz_sync()` **repairs the `nnz` cache in place** (15 360 → 4 482 on the same object) and nothing
  else does — `bsr_mv`, `values.numpy()` and `offsets.numpy()` all leave it stale — so a `.nnz` read
  is correct or not depending on whether earlier code happened to sync that matrix.
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
- `bsr_compress(src, prune_numerical_zeros=True, inplace=False, topology=None) -> BsrMatrix` — (1.15) sort/coalesce active blocks and compact storage; `topology="padded"` keeps reserved row capacity. **Caution:** calling it on a matrix rebuilt from its own CSR via a second `bsr_from_triplets` makes the next `bsr_mm` crash with an illegal memory access (first surfacing in `scan_device`, `warp/native/scan.cu:109`). Confirmed upstream as a CUDA bug, tracked as NVIDIA/warp#1769 — unrelated to the `nnz`-capacity trap above. **That issue is now closed upstream with milestone 1.17.0** (*CUDA `bsr_compress(inplace=True)` treats trailing capacity as active*), unreleased as of Warp 1.16.0, so re-probe on the upgrade — the same fault reproduces from a `bsr_mm` result and from the *default* `inplace=False`, neither of which the issue title covers, so confirm the exact call rather than assuming the fix reaches it (`triwarp.linalg._multigrid_prune` carries the workaround and the numbers).

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
