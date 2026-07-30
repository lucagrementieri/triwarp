"""NumPy-style structural and elementwise operations on Warp arrays (init, gather, sort, masks)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import scatter as kernel_scatter

DType = TypeVar("DType")

# Use a direct-index membership table when max(value)+1 is at most this multiple of |test_elements|.
_ISIN_MASK_SIZE_FACTOR = 8

# Row width up to which [`sort_rows`][triwarp.array.sort_rows] uses a per-row insertion sort instead
# of a segmented radix sort. Comfortably above every in-library row width (edges 2, corners 3).
SORT_ROWS_INSERTION_MAX_COLS = 8


def init_range(n: int, device: str, *, dtype: type[wp.Int] = wp.int32) -> wp.array:
    """
    Fill ``out[i] = i`` for ``i`` in ``[0, n)`` (``numpy.arange``).

    Parameters
    ----------
    n
        Number of elements; must be non-negative.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result. Must be able to represent ``n - 1``.

    Returns
    -------
    wp.array
        Length-``n`` array of consecutive indices on ``device``.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``n - 1`` does not fit in ``dtype``.
    """
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
    count: int, step: int, device: str, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``out[i] = i * step`` (``numpy.arange(0, count * step, step)``).

    Parameters
    ----------
    count
        Number of elements; must be non-negative.
    step
        Stride between consecutive values; must be non-negative.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result. Must be able to represent ``(count - 1) * step``.

    Returns
    -------
    wp.array
        Length-``count`` array on ``device``.

    Raises
    ------
    ValueError
        If ``count`` or ``step`` is negative, or the largest value does not fit in ``dtype``.
    """
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
    n: int, fill_value: int, device: str, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``[0, 1, ..., n-1, fill_value, ..., fill_value]`` (length ``2 * n``).

    The payload buffer ``warp.utils.radix_sort_pairs`` wants: the first half seeded with the
    identity permutation, the second half (its scratch) filled with a padding value.

    Parameters
    ----------
    n
        Number of real entries; the result has length ``2 * n``.
    fill_value
        Padding written into the upper half.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result.

    Returns
    -------
    wp.array
        Length-``2 * n`` array on ``device``.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``n - 1`` / ``fill_value`` does not fit in ``dtype``.

    See Also
    --------
    [`sort_pairs`][triwarp.array.sort_pairs]
    """
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
    count: int, repeats: int, device: str, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``out[i] = i // repeats`` (``numpy.repeat`` of an index range).

    Parameters
    ----------
    count
        Number of elements; must be non-negative.
    repeats
        How many consecutive entries share an index; must be positive.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result.

    Returns
    -------
    wp.array
        Length-``count`` array on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is negative, ``repeats`` is not positive, or the largest value does not fit
        in ``dtype``.
    """
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
    Concatenate several 1-D ``warp.array`` instances into one buffer plus per-segment offsets.

    Segment ``i`` starts at ``offsets[i]`` in ``flat``; its length is ``arrays[i].size``, so it
    occupies ``flat[offsets[i] : offsets[i] + arrays[i].size]``. This is the usual packed
    representation for variable-length per-item lists on the device (no nested arrays).

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
        Length ``len(arrays)`` (the start offset of each segment, an exclusive scan of the
        segment sizes), ``dtype`` ``wp.int32``, same ``device`` as the inputs. ``offsets[0] == 0``.
        This is *not* a trailing-sentinel CSR array — there is no ``offsets[-1] == flat.size``
        terminator, so the last segment's length must be taken from ``arrays[-1].size`` (or
        ``flat.size - offsets[-1]``).

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
    Concatenate 1-D ``warp.array`` instances in order (``numpy.concatenate``).

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
            raise ValueError(
                f"concatenate requires rank-1 arrays, got ndim={arr.ndim} at index {i}"
            )
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


def allclose(
    a: wp.array[wp.float32] | wp.array[wp.vec3],
    b: wp.array[wp.float32] | wp.array[wp.vec3],
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """
    Test whether two arrays are element-wise equal within a tolerance (``numpy.allclose``).

    Reduces ``|a - b| <= atol + rtol * |b|`` (element-wise, and component-wise for ``wp.vec3``)
    to a single Python ``bool`` on-device, without copying either array to the host. Matches the
    asymmetric ``numpy.allclose`` / ``torch.allclose`` tolerance convention.

    Parameters
    ----------
    a
        Length-``n`` ``wp.float32`` or ``wp.vec3`` array on the target device.
    b
        Array of the same length and dtype as ``a``, on the same device.
    rtol
        Relative tolerance. Defaults to ``1e-05``.
    atol
        Absolute tolerance. Defaults to ``1e-08``.

    Returns
    -------
    bool
        ``True`` when every element (every component, for ``wp.vec3``) is within tolerance.
        ``True`` for empty inputs, following the ``numpy.allclose`` convention.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or dtypes.
    """
    if a.dtype != b.dtype:
        raise ValueError(f"allclose requires matching dtypes, got {a.dtype} and {b.dtype}")
    n = int(a.shape[0])
    if n != int(b.shape[0]):
        raise ValueError(f"allclose requires equal lengths, got {n} and {b.shape[0]}")
    if n == 0:
        return True

    device = a.device
    mask = wp.empty(n, dtype=wp.bool, device=device)
    is_close = kernel_array.is_close_vec3 if a.dtype == wp.vec3 else kernel_array.is_close_scalar
    wp.map(is_close, a, b, wp.float32(rtol), wp.float32(atol), out=mask)
    return bool(tw.reduce.all(mask))


def sort_pairs(
    keys: wp.array[wp.Scalar], *, fill_value: int = -1
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
    """
    Ascending radix sort of ``keys``, returning the sorted keys and their original positions.

    Wraps ``warp.utils.radix_sort_pairs``, which needs double-width scratch for both the keys and
    the payload; this allocates that scratch, seeds the payload with ``0..n-1`` and hands back
    length-``n`` views of the sorted prefixes.

    Parameters
    ----------
    keys
        Length-``n`` sort keys (any radix-sortable scalar dtype).
    fill_value
        Padding written into the upper half of the payload buffer, where the sort's scratch lives.
        Only matters to callers that read past ``n``.

    Returns
    -------
    sorted_keys : wp.array
        Length-``n`` view of the ascending keys.
    order : wp.array[wp.int32]
        Length-``n`` view of the original index of each sorted key.

    Notes
    -----
    Both results are **views** into the scratch buffers, kept alive by the returned arrays. Clone
    them if they must outlive the caller's frame alongside another sort.
    """
    device = keys.device
    n = int(keys.shape[0])
    if n == 0:
        return keys, wp.empty(0, dtype=wp.int32, device=device)
    keys_buffer = wp.empty(2 * n, dtype=keys.dtype, device=device)
    wp.copy(keys_buffer, keys, count=n)
    order_buffer = init_sort_pair_indices(n, fill_value, device)
    wp.utils.radix_sort_pairs(keys_buffer, order_buffer, count=n)
    return keys_buffer[:n], order_buffer[:n]


def sort_rows(data: twt.Array2dInt32 | twt.Array2dFloat32) -> None:
    """
    Sort each row of a 2D array independently, in place, ascending.

    Each row is treated as its own radix-sort segment, so rows are reordered internally
    but their relative row order is unaffected.

    Parameters
    ----------
    data
        ``(n, w)`` device array sorted in place, row by row.

    Notes
    -----
    Rows no wider than ``SORT_ROWS_INSERTION_MAX_COLS`` are sorted by a per-row insertion sort (one
    thread per row); wider rows fall back to a segmented radix sort. The narrow path is not a
    micro-optimization: ``segmented_sort_pairs`` pays a fixed cost per *segment*, so sorting a
    million two-element rows with it cost ~183 ms against ~0.1 ms for the compare-and-swap the width
    actually needs. Every in-library caller sorts vertex pairs or triangle corners.
    """
    n = data.size
    n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
    if n_rows == 0 or n_cols < 2:
        return
    if n_cols <= SORT_ROWS_INSERTION_MAX_COLS:
        wp.launch(kernel_array.sort_rows_insertion, dim=n_rows, inputs=[data], device=data.device)
        return

    data_buffer = wp.empty(n * 2, dtype=data.dtype, device=data.device)
    wp.copy(data_buffer, data, count=n)
    indices_buffer = init_sort_pair_indices(n, -1, data.device)
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

    This mirrors ``trimesh.geometry.index_sparse``, but returns a ``warp.sparse.BsrMatrix``
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
        Forwarded to ``warp.sparse.bsr_from_triplets``.

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


def _sorted_copy(values: wp.array[DType]) -> wp.array[DType]:
    """
    Ascending-sorted copy of a 1D scalar array.

    ``sort_pairs`` returns a *view* into its own scratch and this outlives the caller's frame, so
    the keys are cloned. The order payload is discarded, which is why the padding value it seeds
    does not matter here.
    """
    if int(values.shape[0]) <= 1:
        return values
    return wp.clone(sort_pairs(values)[0])


def _isin_lookup_mask(
    elements_flat: wp.array[wp.int32], test_elements: wp.array[wp.int32], max_index: int
) -> wp.array[wp.bool]:
    k = int(test_elements.shape[0])
    device = elements_flat.device
    membership_wp = wp.zeros(max_index, dtype=wp.bool, device=device)
    wp.launch(
        kernel_scatter.mark_membership_mask,
        dim=k,
        inputs=[test_elements, wp.int32(max_index), membership_wp],
        device=device,
    )
    # ``membership_wp[elements_flat]`` gathers the boolean membership flag per element.
    return gather(membership_wp, elements_flat)


def _isin_lookup_sorted(
    elements_flat: wp.array[wp.int32], test_elements: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    device = elements_flat.device
    sorted_test_wp = _sorted_copy(test_elements)
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

    # Inclusive scan: the total is its last element, so one 4-byte tail read sizes the output
    # (the scatter kernel derives each exclusive position as inclusive[i] - 1).
    inclusive = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    n_out = int(inclusive[n - 1 :].numpy()[0])

    if n_out == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    out_indices = wp.empty(n_out, dtype=wp.int32, device=device)
    wp.launch(
        kernel_scatter.scatter_index_where,
        dim=n,
        inputs=[mask, inclusive, out_indices],
        device=device,
    )
    return out_indices


def gather(src: wp.array[DType], indices: wp.array[wp.int32]) -> wp.array[DType]:
    """
    Dense copy of ``src`` gathered along its first axis by ``indices`` (``numpy.take``).

    Warp's ``src[indices]`` fancy indexing yields a ``warp.indexedarray`` view; this
    materializes a contiguous ``warp.array`` (performing the copy) so callers get a real
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


def indices_to_mask(
    indices: wp.array[wp.int32], n: int, *, device: wp.DeviceLike = None
) -> wp.array[wp.bool]:
    """
    Boolean membership mask of length ``n`` marking each value in ``indices`` as ``True``.

    Wraps the ``mark_membership_mask`` scatter kernel: every ``indices[i]`` sets
    ``out_mask[indices[i]] = True``. The inverse of [`flatnonzero`][triwarp.array.flatnonzero].

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of values in ``[0, n)`` to mark. May be empty.
    n
        Length of the returned mask.
    device
        Target Warp device. Defaults to ``indices.device``.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask, all ``False`` except at positions named by ``indices``.
    """
    device = device if device is not None else indices.device
    mask = wp.zeros(n, dtype=wp.bool, device=device)
    k = int(indices.shape[0])
    if k > 0:
        wp.launch(
            kernel_scatter.mark_membership_mask,
            dim=k,
            inputs=[indices, wp.int32(n), mask],
            device=device,
        )
    return mask


def mask_to_index_map(
    mask: wp.array[wp.bool], *, invert: bool = False
) -> tuple[wp.array[wp.int32], int]:
    """
    Compact index map over the ``True`` entries of a boolean mask, plus their count.

    Parameters
    ----------
    mask
        Length-``n`` ``wp.bool`` array.
    invert
        When ``True``, map the ``False`` entries instead. Useful for a free/fixed degree-of-freedom
        partition, where the mask marks the *constrained* entries and the compact map is wanted over
        the unconstrained complement (see [`free_partition`][triwarp.linalg.free_partition]).

    Returns
    -------
    index_map : wp.array[wp.int32]
        Length-``n`` array on ``mask.device``: an exclusive scan of the (optionally inverted) mask,
        so ``index_map[i]`` is the compact 0-based rank of element ``i`` among the selected
        entries at or before it (meaningful only where element ``i`` is itself selected).
    count : int
        Total number of selected entries in ``mask``.
    """
    device = mask.device
    n = int(mask.shape[0])
    if n == 0:
        return wp.zeros(0, dtype=wp.int32, device=device), 0
    flags = wp.empty(n, dtype=wp.int32, device=device)
    if invert:
        wp.map(kernel_array.complement_flag, mask, out=flags)
    else:
        wp.utils.array_cast(mask, flags)
    return counts_to_offsets(flags)


def counts_to_offsets(counts: wp.array[wp.int32]) -> tuple[wp.array[wp.int32], int]:
    """
    Exclusive prefix sum of ``counts``, plus their total.

    The CSR-building step that turns per-element counts into row starts. Done in **one** scan pass
    and one 4-byte host read: the scan runs *inclusive* into the tail of an ``n + 1`` buffer whose
    leading zero is already in place, which makes the first ``n`` entries the exclusive sum and the
    last entry the total. The obvious spelling — one exclusive scan for the offsets and a second
    inclusive scan (or a ``reduce.sum``) for the total — costs a second full pass over ``counts``,
    and reading the total as ``inclusive.numpy()[-1]`` copies the whole array to the host to look at
    one element of it.

    Parameters
    ----------
    counts
        Length-``n`` ``wp.int32`` per-element counts.

    Returns
    -------
    offsets : wp.array[wp.int32]
        Length-``n`` exclusive prefix sum, a **view** into an ``n + 1`` buffer that ``total`` keeps
        alive. Element ``i`` owns ``[offsets[i], offsets[i] + counts[i])``.
    total : int
        Sum of ``counts``.

    Notes
    -----
    Two offsets conventions coexist in this package: the length-``n`` form returned here, with the
    total implicit, and a length-``n + 1`` form that stores it (``halfedge.vertex_one_rings``,
    ``tracing.trace_geodesic_from_vertex``). This builds the latter internally, so a caller that
    wants it can be given the whole buffer instead.

    See Also
    --------
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`mask_to_index_map`][triwarp.array.mask_to_index_map]
    """
    n = int(counts.shape[0])
    device = counts.device
    if n == 0:
        return wp.zeros(0, dtype=wp.int32, device=device), 0
    # The leading zero from ``wp.zeros`` is the first exclusive offset; the inclusive scan fills the
    # rest, so ``buffer[n]`` is the total and ``buffer[:n]`` the exclusive offsets.
    buffer = wp.zeros(n + 1, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=buffer[1:], inclusive=True)
    return buffer[:n], int(buffer[n:].numpy()[0])


def remap_indices(indices: wp.array[wp.int32], remap: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Remap an index buffer through a lookup table, skipping negative (sentinel) entries.

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of indices into ``remap`` (e.g. a flat face buffer). Negative
        entries are passed through unchanged.
    remap
        1D ``wp.int32`` lookup table (e.g. old-to-new vertex index map).

    Returns
    -------
    wp.array[wp.int32]
        Length ``len(indices)`` array on ``indices.device`` with ``out[i] = remap[indices[i]]``
        for non-negative ``indices[i]``, and ``out[i] = indices[i]`` otherwise.
    """
    n = int(indices.shape[0])
    device = indices.device
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)
    out = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.gather_1d_skip_negative, dim=n, inputs=[indices, remap, out], device=device
    )
    return out


def trim_to_count(
    counter: wp.array[wp.int32], *buffers: wp.array[DType]
) -> tuple[int, list[wp.array[DType]]]:
    """
    Trim atomic-append output buffers to the number of elements actually written.

    A kernel that emits an unpredictable number of results cannot size its output ahead of
    time. The usual pattern is to over-allocate the output buffers to a safe upper bound and
    have each thread claim its slots with ``wp.atomic_add`` on a shared length-1 ``counter``,
    writing into ``buffer[slot]``. After the launch, only the first ``n_out`` slots hold valid
    data and the tail is uninitialized, but ``n_out`` is known only on the device.

    This finalizes that pattern: it reads the counter back to the host once, then copies the
    valid prefix ``buffer[:n_out]`` of each over-allocated buffer into a freshly allocated,
    exact-size array. Pass every buffer filled by the same counter in one call so they are all
    trimmed to a consistent length.

    Parameters
    ----------
    counter
        Length-1 ``wp.int32`` array holding the final atomic-append count on the target device.
    *buffers
        Over-allocated output buffers to trim, all indexed along their first axis by the same
        counter. Any rank and ``dtype``; trailing dimensions are preserved.

    Returns
    -------
    n_out : int
        The counter value: the number of valid leading elements in each buffer.
    trimmed : list[wp.array]
        One contiguous ``(n_out, *buffer.shape[1:])`` copy per input buffer, in order, each on
        its buffer's device.
    """
    n_out = int(counter[:1].numpy()[0])
    trimmed = []
    for buffer in buffers:
        out_shape = (n_out, *(int(dim) for dim in buffer.shape[1:]))
        out = wp.empty(out_shape, dtype=buffer.dtype, device=buffer.device)
        if n_out > 0:
            wp.copy(out, buffer[:n_out])
        trimmed.append(out)
    return n_out, trimmed


def square(values: wp.array[wp.Scalar]) -> wp.array[wp.Scalar]:
    """
    Element-wise square of a scalar array.

    Computes ``out[i] = values[i] ** 2`` on the device, returning a freshly
    allocated array of the same dtype and length.

    Parameters
    ----------
    values
        Length-``n`` scalar Warp array (e.g. ``wp.float32`` or ``wp.int32``).

    Returns
    -------
    wp.array[wp.Scalar]
        Length-``n`` squared values on ``values.device``. Empty when ``n == 0``.
    """
    device = values.device
    n = int(values.shape[0])
    out = wp.empty(n, dtype=values.dtype, device=device)
    if n == 0:
        return out
    wp.map(kernel_array.square_scalar, values, out=out)
    return out


def clamp(
    values: wp.array[wp.Scalar], minimum: wp.Scalar, maximum: wp.Scalar
) -> wp.array[wp.Scalar]:
    """
    Element-wise clamp of a scalar array to ``[minimum, maximum]``.

    Computes ``out[i] = min(max(values[i], minimum), maximum)`` on the device, returning a freshly
    allocated array of the same dtype and length. The counterpart of MeshLab's
    ``apply_scalar_clamping_per_vertex`` for a per-vertex field, though nothing here is mesh-aware.

    Parameters
    ----------
    values
        Length-``n`` scalar Warp array.
    minimum
        Lower bound, in ``values.dtype``.
    maximum
        Upper bound, in ``values.dtype``. Must not be below ``minimum``; when it is, ``wp.clamp``
        returns ``maximum`` for every element rather than raising.

    Returns
    -------
    wp.array[wp.Scalar]
        Length-``n`` clamped values on ``values.device``. Empty when ``n == 0``.

    See Also
    --------
    [`square`][triwarp.array.square]
    [`triwarp.smoothing.saturate_scalar_gradient`][triwarp.smoothing.saturate_scalar_gradient]
    """
    device = values.device
    n = int(values.shape[0])
    out = wp.empty(n, dtype=values.dtype, device=device)
    if n == 0:
        return out
    wp.map(wp.clamp, values, minimum, maximum, out=out)
    return out


def sortable_dtype(dtype: type[wp.Scalar]) -> type[wp.Scalar]:
    """
    Same-width dtype that ``warp.utils.radix_sort_pairs`` accepts, preserving ``dtype``'s order.

    The single widening rule for every radix sort in this package. Sorting a sub-32-bit dtype is
    not supported by Warp, and sorting the reinterpreted *bits* of a float or an unsigned value
    gets the order wrong, so callers ask here rather than widening ad hoc.

    The hash table works in one common signed-integer key space (see
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]), which is fine for equality but wrong for
    ordering: negative floats have descending bit patterns, and a ``uint64`` with its top bit set
    reads as a negative ``int64``. Warp 1.15 sorts ``uint32`` / ``uint64`` / ``float64`` keys
    directly, so the sort is done in this dtype instead of on the reinterpreted bits.
    """
    wide = wp.types.type_size_in_bytes(dtype) > 4
    if wp.types.type_is_float(dtype):
        return wp.float64 if wide else wp.float32
    if dtype.__name__.lower().startswith("u"):
        return wp.uint64 if wide else wp.uint32
    return wp.int64 if wide else wp.int32


def bitcast_to_int(
    data: wp.array[wp.Scalar], count: int | None = None
) -> wp.array[wp.int32] | wp.array[wp.int64]:
    """
    Reinterpret an array's underlying bits as a same-width signed integer dtype.

    32-bit-or-narrower dtypes (``wp.int32``, ``wp.uint32``, ``wp.float32``, and narrower) are
    reinterpreted as ``wp.int32``; wider dtypes (``wp.int64``, ``wp.uint64``, ``wp.float64``) as
    ``wp.int64``. Narrower-than-32-bit floating point values are first upcast to ``wp.float32``
    (a numeric cast, not a bit reinterpretation) so every dtype narrower than 32 bits shares one
    ``wp.int32`` key space. Used by the hashing/uniqueness machinery
    ([`unique_1d`][triwarp.grouping.unique_1d]) to give arbitrary scalar dtypes a common sortable,
    hashable integer key.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` of any scalar dtype.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized (over-allocation for in-place radix-sort scratch).

    Returns
    -------
    wp.array[wp.int32] | wp.array[wp.int64]
        Bit-reinterpreted (or, for sub-32-bit floats, upcast-then-reinterpreted) copy of length
        ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_from_int`][triwarp.array.bitcast_from_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits > 32:
        reinterpreted = wp.empty(count, dtype=wp.int64, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    reinterpreted = wp.empty(count, dtype=wp.int32, device=data.device)
    if wp.types.type_is_float(data.dtype):
        src = data
        if n_bits < 32:
            src = wp.empty(copy_count, dtype=wp.float32, device=data.device)
            wp.utils.array_cast(data, src, count=copy_count)
        wp.copy(reinterpreted, src, count=copy_count)
    else:
        wp.utils.array_cast(data, reinterpreted, count=copy_count)
    return reinterpreted


def bitcast_from_int(
    data: wp.array[wp.int32] | wp.array[wp.int64], dtype: type[wp.Scalar], count: int | None = None
) -> wp.array[wp.Scalar]:
    """
    Inverse of [`bitcast_to_int`][triwarp.array.bitcast_to_int]: recover the original dtype.

    Reinterprets (same-width) or upcasts-then-reinterprets (narrower target) the bits produced
    by ``bitcast_to_int`` back into ``dtype``.

    Parameters
    ----------
    data
        Rank-1 ``wp.array[wp.int32]`` or ``wp.array[wp.int64]``, typically the output of
        [`bitcast_to_int`][triwarp.array.bitcast_to_int].
    dtype
        Target scalar dtype to reinterpret ``data`` as.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized.

    Returns
    -------
    wp.array[wp.Scalar]
        Array of dtype ``dtype`` and length ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n_target_bits = wp.types.type_size_in_bytes(dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits == n_target_bits:
        reinterpreted = wp.empty(count, dtype=dtype, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    if wp.types.type_is_float(dtype) and n_bits > n_target_bits:
        wide_dtype = getattr(wp, f"float{n_bits}")
        wide = wp.empty(count, dtype=wide_dtype, device=data.device)
        wp.copy(wide, data, count=copy_count)
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(wide, reinterpreted_casted, count=copy_count)
    else:
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(data, reinterpreted_casted, count=copy_count)
    return reinterpreted_casted


# ---------------------------------------------------------------------------
# Private cross-cutting helpers (used by a single external caller today; promote to public
# only with demonstrated cross-module demand).
# ---------------------------------------------------------------------------


def _as_vec3d(vertices: wp.array[wp.vec3]) -> wp.array[wp.vec3d]:
    """Widen a ``wp.vec3`` array to ``wp.vec3d`` (used by ``smoothing``'s float64 solves)."""
    out = wp.empty(int(vertices.shape[0]), dtype=wp.vec3d, device=vertices.device)
    wp.map(kernel_array.to_vec3d, vertices, out=out)
    return out


def _as_vec3(positions: wp.array[wp.vec3d]) -> wp.array[wp.vec3]:
    """Narrow a ``wp.vec3d`` array back to ``wp.vec3`` (used by ``smoothing``'s float64 solves)."""
    out = wp.empty(int(positions.shape[0]), dtype=wp.vec3, device=positions.device)
    wp.map(kernel_array.to_vec3, positions, out=out)
    return out


def _ensure_int_dtype(dtype: type) -> type[wp.Int]:
    if not wp.types.type_is_int(dtype):
        raise TypeError(f"dtype must be a Warp integer type, got {dtype!r}")
    return dtype


def _check_int_fits(dtype: type[wp.Int], value: int, name: str) -> None:
    vmin = twt.dtype_min(dtype)
    vmax = twt.dtype_max(dtype)
    if value < vmin or value > vmax:
        raise ValueError(f"{name}={value} is out of range for {dtype} [{vmin}, {vmax}]")


def _int_scalar(dtype: type[wp.Int], value: int) -> wp.Int:
    _check_int_fits(dtype, value, "value")
    return dtype(value)
