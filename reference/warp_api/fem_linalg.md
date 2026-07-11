# warp.fem.linalg API

Source: https://nvidia.github.io/warp/stable/api_reference/warp_fem_linalg.html (Warp 1.15.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Linear-algebra utilities from the FEM module. Called as `wp.fem.linalg.<name>(...)`. Most are usable inside kernels / `@wp.func`.

Signatures below are introspected from the installed package (the docs page lists names only).

## API
- `array_axpy(x: array, y: array, alpha: float = 1.0, beta: float = 1.0)` — compute `y = alpha*x + beta*y`.
- `generalized_inner(x: vec, y: vec)` — generalized inner product.
- `generalized_outer(x: vec, y: vec)` — generalized outer product.
- `householder_make_hessenberg(A: mat)` — transform a square matrix to Hessenberg form via Householder reflections.
- `householder_qr_decomposition(A: mat)` — QR decomposition of a square matrix via Householder reflections.
- `inverse_qr(A: mat)` — inverse of a square matrix using QR factorization.
- `skew_part(x: mat)` — skew part of a 3x3 tensor as the corresponding rotation vector.
- `solve_triangular(R: mat, b: vec)` — solve `R x = b` for an upper triangular `R`.
- `spherical_part(x: mat)` — spherical part of a square tensor.
- `symmetric_eigenvalues_qr(A: mat, tol)` — eigenvalues/eigenvectors of a symmetric matrix via the QR algorithm.
- `symmetric_part(x: mat)` — symmetric part of a square tensor.
- `tridiagonal_symmetric_eigenvalues_qr(D, L, Q, tol)` — eigenvalues/eigenvectors of a symmetric tridiagonal matrix via implicit-Wilkinson-shift QR.
