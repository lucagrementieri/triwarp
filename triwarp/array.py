"""Mesh connectivity helpers on NVIDIA Warp."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import reduce as kernel_reduce

DType = TypeVar("DType")

# Use a direct-index membership table when max(value)+1 is at most this multiple of |test_elements|.
_ISIN_MASK_SIZE_FACTOR = 8


def _ensure_int_dtype(dtype: type) -> type[wp.Int]:
    if not wp.types.type_is_int(dtype):
        raise TypeError(f"dtype must be a Warp integer type, got {dtype!r}")
    return dtype


def _check_int_fits(dtype: type[wp.Int], value: int, name: str) -> None:
    vmin = tw.reduce.min_for_dtype(dtype)
    vmax = tw.reduce.max_for_dtype(dtype)
    if value < vmin or value > vmax:
        raise ValueError(f"{name}={value} is out of range for {dtype} [{vmin}, {vmax}]")


def _int_scalar(dtype: type[wp.Int], value: int) -> wp.Int:
    _check_int_fits(dtype, value, "value")
    return dtype(value)


def init_range(
    n: int,
    device: str,
    *,
    dtype: type[wp.Int] = wp.int32,
) -> wp.array:
    """Fill ``out[i] = i`` for ``i`` in ``[0, n)``."""
    dtype = _ensure_int_dtype(dtype)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > 0:
        _check_int_fits(dtype, n - 1, "n")
    out = wp.empty(n, dtype=dtype, device=device)
    if n > 0:
        wp.launch(kernel_array.init_range, dim=n, inputs=[out], device=device)
    return out


def init_range_step(
    count: int,
    step: int,
    device: str,
    *,
    dtype: type[wp.Int] = wp.int32,
) -> wp.array:
    """Fill ``out[i] = i * step`` (``numpy.arange(0, count * step, step)``)."""
    dtype = _ensure_int_dtype(dtype)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if count > 0:
        _check_int_fits(dtype, (count - 1) * step, "count * step")
        _check_int_fits(dtype, step, "step")
    out = wp.empty(count, dtype=dtype, device=device)
    if count > 0:
        wp.launch(
            kernel_array.init_range_step,
            dim=count,
            inputs=[out, _int_scalar(dtype, step)],
            device=device,
        )
    return out


def init_sort_pair_indices(
    n: int,
    fill_value: int,
    device: str,
    *,
    dtype: type[wp.Int] = wp.int32,
) -> wp.array:
    """Fill ``[0, 1, ..., n-1, fill_value, ..., fill_value]`` (length ``2 * n``)."""
    dtype = _ensure_int_dtype(dtype)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > 0:
        _check_int_fits(dtype, n - 1, "n")
    _check_int_fits(dtype, fill_value, "fill_value")
    out = wp.empty(2 * n, dtype=dtype, device=device)
    if n > 0:
        wp.launch(
            kernel_array.init_sort_pair_indices,
            dim=2 * n,
            inputs=[out, _int_scalar(dtype, n), _int_scalar(dtype, fill_value)],
            device=device,
        )
    return out


def init_repeat_index(
    count: int,
    repeats: int,
    device: str,
    *,
    dtype: type[wp.Int] = wp.int32,
) -> wp.array:
    """Fill ``out[i] = i // repeats`` (repeat each index ``repeats`` times)."""
    dtype = _ensure_int_dtype(dtype)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if count > 0:
        _check_int_fits(dtype, (count - 1) // repeats, "count // repeats")
    out = wp.empty(count, dtype=dtype, device=device)
    if count > 0:
        wp.launch(
            kernel_array.init_repeat_index,
            dim=count,
            inputs=[out, _int_scalar(dtype, repeats)],
            device=device,
        )
    return out


def append(arr: wp.array[wp.Scalar], value: wp.Scalar) -> wp.array[wp.Scalar]:
    """
    Return a new 1-D array with ``value`` appended after ``arr``.

    Allocates ``len(arr) + 1`` elements initialized to ``value``, then copies
    ``arr`` into the prefix with a single ``wp.copy``.
    """
    n = int(arr.shape[0])
    out = wp.full(n + 1, value, dtype=arr.dtype, device=arr.device)
    if n > 0:
        wp.copy(out, arr, dest_offset=0, src_offset=0, count=n)
    return out


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
        1-D array of length ``sum(a.size for a in arrays)``, same ``dtype`` and ``device`` as
        the inputs.
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
                f"all arrays must have the same dtype, got {dtype} and {arr.dtype} at index {i}"
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


def concatenate(arrays: Sequence[wp.array[DType]]) -> wp.array[DType]:
    """
    Concatenate 1-D :class:`warp.array` instances in order (``numpy.concatenate``).

    Parameters
    ----------
    arrays
        Non-empty sequence of rank-1 arrays sharing the same ``dtype`` and ``device``.
        Empty segments are allowed.

    Returns
    -------
    wp.array
        Contiguous 1-D array of length ``sum(a.size for a in arrays)`` on the input
        device. When ``arrays`` has a single element, that array is returned without
        copying.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, any input is not rank-1, or ``dtype`` / ``device`` differ.
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")
    if len(arrays) == 1:
        arr = arrays[0]
        if int(arr.ndim) != 1:
            raise ValueError(f"concatenate requires rank-1 arrays, got ndim={arr.ndim}")
        return arr

    dtype = arrays[0].dtype
    device = arrays[0].device
    total = 0
    for i, arr in enumerate(arrays):
        if int(arr.ndim) != 1:
            raise ValueError(f"concatenate requires rank-1 arrays, got ndim={arr.ndim} at index {i}")
        if arr.dtype != dtype:
            raise ValueError(
                f"all arrays must have the same dtype, got {dtype} and {arr.dtype} at index {i}"
            )
        total += int(arr.shape[0])

    if total == 0:
        return wp.empty(0, dtype=dtype, device=device)

    out = wp.empty(total, dtype=dtype, device=device)
    dest = 0
    for arr in arrays:
        n = int(arr.shape[0])
        if n > 0:
            wp.copy(out, arr, dest_offset=dest, count=n)
            dest += n
    return out


def sort_rows(data: twt.Array2dInt32 | twt.Array2dFloat32) -> None:
    n = data.size
    data_buffer = wp.empty(n * 2, dtype=data.dtype, device=data.device)
    wp.copy(data_buffer, data, count=n)
    indices_buffer = init_sort_pair_indices(n, -1, data.device)
    n_cols = int(data.shape[1])
    segment_start_indices = init_range_step(n // n_cols + 1, n_cols, data.device)
    wp.utils.segmented_sort_pairs(
        data_buffer, indices_buffer, n, segment_start_indices=segment_start_indices
    )
    wp.copy(data, data_buffer, count=n)


def index_sparse(
    n_rows: int,
    indices: twt.Array2dInt32,
    data: wp.array[wp.Scalar] | None = None,
    dtype: type[wp.Scalar] | None = None,
    *,
    prune_numerical_zeros: bool = True,
) -> wps.BsrMatrix[wp.Scalar]:
    """
    Build a sparse row/column incidence matrix from flat index columns.

    This mirrors ``trimesh.geometry.index_sparse``, but returns a :class:`warp.sparse.BsrMatrix`
    in 1x1 BSR (CSR) form instead of ``scipy.sparse.coo_matrix``.

    Parameters
    ----------
    n_rows
        Number of matrix rows (e.g. vertex count). Matrix shape is ``(n_rows, len(indices))``.
    indices
        Integer array of shape ``(m, d)`` — typically ``mesh.faces`` with three vertex indices
        per face.
    data
        Optional 1-D array of length ``m * d``. If omitted, ``wp.ones`` is used; see ``dtype``.
    dtype
        Scalar type for ``wp.ones`` when ``data`` is ``None`` (defaults to ``wp.float32`` if
        ``dtype`` is ``None``). When ``data`` and ``dtype`` are provided, the values of the
        matrix are cast to ``dtype``.
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
                f"data must have the same size as indices, got {data.size} and {indices.size}"
            )
        if dtype is not None and data.dtype != dtype:
            casted_data = wp.empty(data.shape, dtype=dtype)
            wp.utils.array_cast(data, casted_data)
            data = casted_data

    n_cols, n_repeats = indices.shape
    cols = init_repeat_index(n_cols * n_repeats, n_repeats, indices.device)
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
    indices_wp = init_sort_pair_indices(n, n, device)
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
    Return indices of ``True`` entries in a 1D boolean mask (``numpy.flatnonzero``).

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


def gather(src: wp.array[DType], indices: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Dense copy of ``src`` gathered along its first axis by ``indices`` (``numpy.take``).

    Warp's ``src[indices]`` fancy indexing yields a :class:`warp.indexedarray` view; this
    materializes a contiguous :class:`warp.array` (performing the copy) so callers get a real
    array supporting ``.reshape`` and a stable return type. Works for rank-1 sources
    (``src[indices]``) and rank-2 row gather (``src[indices, :]``), with any scalar or vector
    ``dtype``.

    Parameters
    ----------
    src
        Rank-1 or rank-2 ``wp.array`` on the target device.
    indices
        1D ``wp.int32`` array of indices into the first axis of ``src``, on the same device.

    Returns
    -------
    wp.array
        Contiguous gathered copy on ``src.device`` with shape
        ``(len(indices), *src.shape[1:])`` and the same ``dtype`` as ``src``. Empty along the
        first axis when ``indices`` is empty.
    """
    k = int(indices.shape[0])
    out_shape = (k, *(int(dim) for dim in src.shape[1:]))
    out = wp.empty(out_shape, dtype=src.dtype, device=src.device)
    if k > 0:
        wp.copy(out, src[indices])
    return out


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
    n = int(a.shape[0])
    if n != int(b.shape[0]):
        raise ValueError(f"a and b must have the same length, got {n} and {b.shape[0]}")

    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_angles = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(kernel_array.vector_angle, dim=n, inputs=[a, b, out_angles], device=device)
    return out_angles


def gram_matrix(points: wp.array[wp.vec3]) -> wp.array[wp.mat33]:
    """
    Uncentered Gram (scatter) matrix ``G = sum_k outer(x_k, x_k)``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the ``3x3`` Gram matrix on
        ``points.device``. All-zeros when ``points`` is empty.
    """
    device = points.device
    n = int(points.shape[0])
    out = wp.zeros(1, dtype=wp.mat33, device=device)
    if n == 0:
        return out
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_array.gram_matrix,
        dim=[n_tiles],
        inputs=[points, out],
        block_dim=TILE_1D,
        device=device,
    )
    return out


def centered_covariance(
    points: wp.array[wp.vec3], center: wp.array[wp.vec3] | None = None
) -> wp.array[wp.mat33]:
    """
    Centered scatter matrix ``C = sum_k outer(x_k - mu, x_k - mu)`` (no ``1/n``).

    Centering happens inside the outer-product loop (rather than via the
    ``sum(x x^T) - n mu mu^T`` identity) to avoid float32 catastrophic
    cancellation.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    center
        Optional precomputed centroid as a ``(1,)`` ``wp.vec3`` device array.
        When ``None`` it is computed on-device as ``sum(points) / n``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the centered ``3x3`` scatter matrix
        on ``points.device``. All-zeros when ``points`` is empty.
    """
    device = points.device
    n = int(points.shape[0])
    out = wp.zeros(1, dtype=wp.mat33, device=device)
    if n == 0:
        return out
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    if center is None:
        center = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch_tiled(
            kernel_reduce.sum_vec3_1d_tiled,
            dim=[n_tiles],
            inputs=[points, center],
            block_dim=TILE_1D,
            device=device,
        )
        wp.launch(kernel_array.divide, dim=1, inputs=[center, wp.float32(n)], device=device)
    wp.launch_tiled(
        kernel_array.centered_covariance,
        dim=[n_tiles],
        inputs=[points, center, out],
        block_dim=TILE_1D,
        device=device,
    )
    return out


def covariance(points: wp.array[wp.vec3], ddof: int = 1) -> wp.array[wp.mat33]:
    """
    Sample covariance matrix ``(1 / (n - ddof)) sum_k outer(x_k - mu, x_k - mu)``.

    Matches ``numpy.cov(points.T, ddof=ddof)`` for the default ``ddof=1``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    ddof
        Delta degrees of freedom; the divisor is ``n - ddof``. Defaults to ``1``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the ``3x3`` covariance matrix on
        ``points.device``.

    Raises
    ------
    ValueError
        If ``n - ddof <= 0``.
    """
    device = points.device
    n = int(points.shape[0])
    if n - ddof <= 0:
        raise ValueError(f"covariance requires n > ddof, got n={n}, ddof={ddof}")
    out = centered_covariance(points)
    wp.launch(kernel_array.divide, dim=1, inputs=[out, wp.float32(n - ddof)], device=device)
    return out
