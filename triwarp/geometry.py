"""Mesh connectivity helpers on NVIDIA Warp."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp
import warp.sparse as wps

from typing import Union, Any  # pyright: ignore[reportDeprecated]


def pack_1d_arrays(arrays: Sequence[wp.array[Any]]) -> tuple[wp.array[Any], wp.array[wp.int32]]:  # pyright: ignore[reportExplicitAny]
    """
    Concatenate several 1-D :class:`warp.array` instances into one buffer plus CSR-style offsets.

    Each input segment ``i`` occupies ``flat[offsets[i] : offsets[i + 1]]``. This is the usual
    packed representation for variable-length per-item lists on the device (no nested arrays).

    Parameters
    ----------
    arrays
        Non-empty sequence of 1-D arrays sharing the same ``dtype`` and ``device``.

    Returns
    -------
    flat
        1-D array of length ``sum(a.size for a in arrays)``, same ``dtype`` and ``device`` as the inputs.
    offsets
        Length ``len(arrays) + 1``, ``dtype`` ``wp.int32``, same ``device`` as the inputs.
        ``offsets[0] == 0`` and ``offsets[-1] == flat.size``.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, ranks differ from one, or ``dtype`` / ``device`` are inconsistent.
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")

    dtype = arrays[0].dtype
    device = arrays[0].device
    sizes = []
    offsets = [0]
    for i, arr in enumerate(arrays):
        if arr.dtype != dtype:
            raise ValueError(
                "all arrays must have the same dtype, got {} and {} at index {}".format(dtype, arr.dtype, i)
            )
        if arr.device != device:
            raise ValueError(
                "all arrays must live on the same device, got {!r} and {!r} at index {}".format(device, arr.device, i)
            )
        sizes.append(int(arr.size))
        offsets.append(offsets[-1] + sizes[-1])

    total = offsets.pop()
    flat = wp.empty(total, dtype=dtype, device=device)
    for array, offset in zip(arrays, offsets, strict=True):
        array_length = int(array.size)
        if array_length > 0:
            wp.copy(flat, array, dest_offset=offset, src_offset=0, count=array_length)

    return flat, wp.array(offsets, dtype=wp.int32, device=device)


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
