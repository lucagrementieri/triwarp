"""Mesh connectivity helpers on NVIDIA Warp."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp
import warp.sparse as wps

from typing import Union  # pyright: ignore[reportDeprecated]

import triwarp.typing as twt
import triwarp as tw
from triwarp.kernels import array as kernel_array

# Use a direct-index membership table when max(value)+1 is at most this multiple of |test_elements|.
_ISIN_MASK_SIZE_FACTOR = 8


def pack_1d_arrays(
    arrays: Sequence[wp.array[wp.Scalar]],
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
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
                "all arrays must have the same dtype, got {} and {} at index {}".format(
                    dtype, arr.dtype, i
                )
            )
        if arr.device != device:
            raise ValueError(
                "all arrays must live on the same device, got {!r} and {!r} at index {}".format(
                    device, arr.device, i
                )
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


def sort_rows(data: twt.Array2dInt32 | twt.Array2dFloat32) -> None:
    n = data.size
    data_buffer = wp.empty(n * 2, dtype=data.dtype, device=data.device)
    wp.copy(data_buffer, data, count=n)
    indices_buffer = wp.array(list(range(n)) + [-1] * n, dtype=wp.int32, device=data.device)
    segment_start_indices = wp.array(
        list(range(0, n + 1, data.shape[1])), dtype=wp.int32, device=data.device
    )
    wp.utils.segmented_sort_pairs(
        data_buffer, indices_buffer, n, segment_start_indices=segment_start_indices
    )
    wp.copy(data, data_buffer, count=n)


def index_sparse(
    n_rows: int,
    indices: twt.Array2dInt32,
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
            raise ValueError(
                "data must have the same size as indices, got {} and {}".format(
                    data.size, indices.size
                )
            )
        if dtype is not None and data.dtype != dtype:
            casted_data = wp.empty(data.shape, dtype=dtype)
            wp.utils.array_cast(data, casted_data)
            data = casted_data

    n_cols, n_repeats = indices.shape
    cols = wp.array([c for c in range(n_cols) for _ in range(n_repeats)], dtype=wp.int32)
    return wps.bsr_from_triplets(
        n_rows,
        indices.shape[0],
        indices.flatten(),
        cols,
        data,
        prune_numerical_zeros=prune_numerical_zeros,
    )


def isin(
    elements: twt.Array1dInt32 | twt.Array2dInt32, test_elements: twt.Array1dInt32
) -> wp.array[wp.bool]:
    """
    Test whether each element appears in ``test_elements`` (``numpy.isin`` for ``int32``).

    When ``max(elements, test_elements) + 1`` is modest relative to ``len(test_elements)``,
    membership is implemented with a boolean lookup table (fast for dense mesh indices).
    Otherwise ``test_elements`` is sorted and each query uses binary search (bounded memory
    when values are sparse in ``int32``).

    Parameters
    ----------
    elements
        Rank-1 or rank-2 ``wp.int32`` array on the target device.
    test_elements
        1D ``wp.int32`` array of values to test membership against, on the same device.

    Returns
    -------
    wp.array[wp.bool]
        Boolean array with the same shape as ``elements``. All ``False`` when either
        input is empty.

    Raises
    ------
    ValueError
        If ``elements`` and ``test_elements`` live on different devices.
    """
    device = elements.device
    if test_elements.device != device:
        raise ValueError(
            f"test_elements must live on the same device as elements, got {test_elements.device} and {device}"
        )

    k = int(test_elements.shape[0])
    if k == 0 or int(elements.size) == 0:
        return wp.zeros(elements.shape, dtype=wp.bool, device=device)

    if int(elements.ndim) > 1:
        twt.ensure_ndim(elements, 2, dtype=wp.int32)
    elements_flat = elements.flatten() if int(elements.ndim) > 1 else elements

    max_index = int(max(tw.reduce.max(elements), tw.reduce.max(test_elements)) + 1)
    if max_index <= _ISIN_MASK_SIZE_FACTOR * k:
        out_flat = _isin_lookup_mask(elements_flat, test_elements, max_index)
    else:
        out_flat = _isin_lookup_sorted(elements_flat, test_elements)

    if int(elements.ndim) > 1:
        return out_flat.reshape(elements.shape)
    return out_flat


def _sorted_int32_copy(values: wp.array[wp.int32]) -> wp.array[wp.int32]:
    n = int(values.shape[0])
    device = values.device
    if n <= 1:
        sorted_wp = wp.empty(n, dtype=wp.int32, device=device)
        if n == 1:
            wp.copy(sorted_wp, values)
        return sorted_wp
    keys_wp = wp.empty(2 * n, dtype=wp.int32, device=device)
    wp.copy(keys_wp, values, count=n)
    indices_wp = wp.array(list(range(n)) + [n] * n, dtype=wp.int32, device=device)
    wp.utils.radix_sort_pairs(keys_wp, indices_wp, count=n)
    sorted_wp = wp.empty(n, dtype=wp.int32, device=device)
    wp.copy(sorted_wp, keys_wp, count=n)
    return sorted_wp


def _isin_lookup_mask(
    elements_flat: wp.array[wp.int32], test_elements: wp.array[wp.int32], max_index: int
) -> wp.array[wp.bool]:
    k = int(test_elements.shape[0])
    device = elements_flat.device
    membership_wp = wp.zeros(max_index, dtype=wp.bool, device=device)
    wp.launch(
        kernel_array.mark_membership_mask,
        dim=k,
        inputs=[test_elements, membership_wp],
        device=device,
    )
    out_wp = wp.empty(elements_flat.shape, dtype=wp.bool, device=device)
    wp.launch(
        kernel_array.isin_lookup_mask,
        dim=int(elements_flat.shape[0]),
        inputs=[elements_flat, membership_wp, out_wp],
        device=device,
    )
    return out_wp


def _isin_lookup_sorted(
    elements_flat: wp.array[wp.int32], test_elements: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    device = elements_flat.device
    sorted_test_wp = _sorted_int32_copy(test_elements)
    out_wp = wp.empty(elements_flat.shape, dtype=wp.bool, device=device)
    wp.launch(
        kernel_array.isin_lookup_sorted,
        dim=int(elements_flat.shape[0]),
        inputs=[elements_flat, sorted_test_wp, out_wp],
        device=device,
    )
    return out_wp


def flatnonzero(mask: wp.array[wp.bool]) -> wp.array[wp.int32]:
    """
    Indices of ``True`` entries in a 1D boolean mask (``numpy.flatnonzero``).

    Parameters
    ----------
    mask
        Length-``n`` ``wp.bool`` array on the target device.

    Returns
    -------
    wp.array[wp.int32]
        Selected indices on ``mask.device``. Empty when no entries are ``True``.

    Raises
    ------
    ValueError
        If ``mask`` is not rank-1.
    """
    if int(mask.ndim) != 1:
        raise ValueError(f"flatnonzero requires a 1D mask, got ndim={mask.ndim}")

    device = mask.device
    n = int(mask.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    flags = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_cast(mask, flags)

    exclusive = wp.empty(n, dtype=wp.int32, device=device)
    inclusive = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=exclusive, inclusive=False)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    n_out = int(inclusive.numpy()[-1])

    if n_out == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    out_indices = wp.empty(n_out, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.scatter_compact_indices,
        dim=n,
        inputs=[mask, exclusive, out_indices],
        device=device,
    )
    return out_indices


def vector_angle(a: wp.array[wp.vec3], b: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between pairs of unit vectors.

    For each index ``i``, computes ``abs(arccos(clip(dot(a[i], b[i]), -1, 1)))``.
    Matches :func:`trimesh.geometry.vector_angle` on stacked pairs.

    Parameters
    ----------
    a
        Length-``n`` unit vectors on the target device.
    b
        Length-``n`` unit vectors on the same device as ``a``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` unsigned angles in radians on ``a.device``. Empty when ``n == 0``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` live on different devices or have different lengths.

    See Also
    --------
    :func:`trimesh.geometry.vector_angle`
    """
    device = a.device
    if b.device != device:
        raise ValueError(f"a and b must live on the same device, got {device} and {b.device}")

    n = int(a.shape[0])
    if n != int(b.shape[0]):
        raise ValueError(f"a and b must have the same length, got {n} and {b.shape[0]}")

    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_angles = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(kernel_array.vector_angle, dim=n, inputs=[a, b, out_angles], device=device)
    return out_angles
