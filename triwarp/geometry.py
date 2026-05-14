"""Mesh connectivity helpers on NVIDIA Warp."""

import warp as wp
import numpy as np
import warp.sparse as wps

from typing import Union  # pyright: ignore[reportDeprecated]


def index_sparse(
    n_rows: int,
    indices: wp.array2d[wp.int32],
    data: Union[wp.array[wp.Scalar], None] = None,  # pyright: ignore[reportDeprecated]
    dtype: wp.Scalar = wp.float32,
    *,
    prune_numerical_zeros: bool = True,
) -> wps.BsrMatrix[wp.Scalar]:
    """
    Build a sparse matrix indicating which row indices (e.g. vertices) appear in which columns (e.g. faces).

    This mirrors ``trimesh.geometry.index_sparse``, but returns a :class:`warp.sparse.BsrMatrix` in 1x1 BSR
    (CSR) form instead of ``scipy.sparse.coo_matrix``.

    The ``dtype`` argument is used only when ``data`` is ``None``, selecting the scalar type for the
    default ``wp.ones`` fill; it is ignored when ``data`` is provided.

    Parameters
    ----------
    n_rows
        Number of matrix rows (e.g. vertex count). Matrix shape is ``(n_rows, len(indices))``.
    indices
        Integer array of shape ``(m, d)`` — typically ``mesh.faces`` with three vertex indices per face.
    data
        Optional 1-D array of length ``m * d``. If omitted, entries come from ``wp.ones(...)``; see ``dtype``.
    dtype
        Scalar type for ``wp.ones`` when ``data`` is ``None``; has no effect when ``data`` is supplied.
    prune_numerical_zeros
        Forwarded to :func:`warp.sparse.bsr_from_triplets`.

    Returns
    -------
    warp.sparse.BsrMatrix
        Sparse matrix with shape ``(n_rows, len(indices))`` and 1x1 blocks.
    """
    prune_numerical_zeros = prune_numerical_zeros and data is not None
    if data is None:
        data = wp.ones(indices.size, dtype=dtype)
    else:
        if data.size != indices.size:
            raise ValueError("data must have the same size as indices, got {} and {}".format(data.size, indices.size))

    n_cols, n_repeats = indices.shape
    cols = wp.array(np.repeat(np.arange(n_cols), n_repeats), dtype=wp.int32)
    return wps.bsr_from_triplets(
        n_rows, indices.shape[0], indices.flatten(), cols, data, prune_numerical_zeros=prune_numerical_zeros
    )
