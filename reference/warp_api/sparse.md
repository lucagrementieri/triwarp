# warp.sparse API

Source: https://nvidia.github.io/warp/stable/api_reference/warp_sparse.html (Warp 1.14.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Block-sparse (BSR/CSR) matrix support. Import via `from warp.sparse import ...` or `wp.sparse.<name>`.

## Classes
- `BsrMatrix` — untyped base class for BSR and CSR matrices.
- `bsr_matrix_t` — typed BSR matrix class.
- `bsr_mm_work_arrays` — persists temporary work buffers across matrix-matrix multiply calls.
- `bsr_axpy_work_arrays` — persists temporary work buffers across addition calls.

## Matrix Construction
- `bsr_zeros(shape) -> BsrMatrix` — empty BSR/CSR matrix of given shape.
- `bsr_identity(n) -> BsrMatrix` — square identity matrix.
- `bsr_diag(block_value) -> BsrMatrix` — block-diagonal matrix from a block value or array.
- `bsr_from_triplets(rows, cols, values, shape) -> BsrMatrix` — build from COO triplets.
- `bsr_copy(A) -> BsrMatrix` — copy, optionally changing scalar type.

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
