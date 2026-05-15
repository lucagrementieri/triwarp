"""Mesh connectivity helpers on NVIDIA Warp."""

import warp as wp
import warp.sparse as wps

from typing import Union  # pyright: ignore[reportDeprecated]


def index_sparse(
    n_rows: int,
    indices: wp.array2d[wp.int32],
    data: Union[wp.array[wp.Scalar], None] = None,  # pyright: ignore[reportDeprecated]
    dtype: type[wp.Scalar] | None = None,
    *,
    prune_numerical_zeros: bool = True,
) -> wps.BsrMatrix[wp.Scalar]:
    """
    Build a sparse matrix indicating which row indices (e.g. vertices) appear in which columns (e.g. faces).

    This mirrors ``trimesh.geometry.index_sparse``, but returns a :class:`warp.sparse.BsrMatrix` in 1x1 BSR
    (CSR) form instead of ``scipy.sparse.coo_matrix``.

    Parameters
    ----------
    n_rows
        Number of matrix rows (e.g. vertex count). Matrix shape is ``(n_rows, len(indices))``.
    indices
        Integer array of shape ``(m, d)`` — typically ``mesh.faces`` with three vertex indices per face.
    data
        Optional 1-D array of length ``m * d``. If omitted, ``wp.ones`` is used; see ``dtype``.
    dtype
        Scalar type for ``wp.ones`` when ``data`` is ``None`` (defaults to ``wp.float32`` if ``dtype`` is
        ``None``). When ``data`` and ``dtype`` are provided, the values of the matrix are cast to ``dtype``.
    prune_numerical_zeros
        Forwarded to :func:`warp.sparse.bsr_from_triplets`.

    Returns
    -------
    warp.sparse.BsrMatrix
        Sparse matrix with shape ``(n_rows, len(indices))`` and 1x1 blocks.
    """
    prune_numerical_zeros = prune_numerical_zeros and data is not None
    if data is None:
        data = wp.ones(indices.size, dtype=dtype if dtype is not None else wp.float32)
    else:
        if data.size != indices.size:
            raise ValueError("data must have the same size as indices, got {} and {}".format(data.size, indices.size))
        if dtype is not None and data.dtype != dtype:
            casted_data = wp.empty(data.shape, dtype=dtype)
            wp.utils.array_cast(data, casted_data)
            data = casted_data

    n_cols, n_repeats = indices.shape
    cols = wp.array([c for c in range(n_cols) for _ in range(n_repeats)], dtype=wp.int32)
    return wps.bsr_from_triplets(
        n_rows, indices.shape[0], indices.flatten(), cols, data, prune_numerical_zeros=prune_numerical_zeros
    )
