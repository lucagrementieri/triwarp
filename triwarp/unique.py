"""1D unique values (``numpy.unique``-style) for Warp rank-1 scalar arrays."""

from __future__ import annotations

from typing import Literal, TypeVar, overload

import warp as wp
import triwarp as tw
from triwarp.kernels import unique as kernel_unique
from triwarp.kernels import array as kernel_array

Scalar = TypeVar("Scalar", bound=wp.Scalar)


# TODO: do a faster unique for 64-bit types
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[False] = False, return_counts: Literal[False] = False
) -> wp.array[Scalar]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: bool = False, return_counts: bool = False
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find sorted unique elements of a 1D Warp array (``numpy.unique`` subset).

    Works for any Warp scalar dtype (signed and unsigned integers, floating-point
    types, etc.). Uses stable ``warp.utils.radix_sort_pairs`` on sortable integer
    keys (signed integers are remapped so radix order matches two's-complement
    order; floats use total-order bit keys). ``return_inverse`` maps each input
    position to the index of its value in the sorted unique output.
    ``return_index`` is not supported.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` with any scalar ``dtype``.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique : wp.array
        Sorted unique values (same ``dtype`` and ``device`` as ``data``).
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique`` such
        that conceptually ``unique[inverse[i]] == data[i]`` (float ``NaN`` matches
        NumPy: all ``NaN`` values share one slot).
    counts : wp.array[wp.int32], optional
        Present if ``return_counts=True``. Number of occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-1 or length is ``>= 2**31``.
    """
    if int(data.ndim) != 1:
        raise ValueError(f"unique_1d expects a rank-1 array, got ndim={data.ndim}")

    device = data.device
    n = int(data.shape[0])
    if n >= (1 << 31):
        raise ValueError(
            f"unique_1d requires length < 2**31 because radix sort value indices are int32, got length {n}"
        )

    if n == 0:
        empty_unique = wp.empty(0, dtype=data.dtype, device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique, inverse=empty_i32 if return_inverse else None, counts=empty_i32 if return_counts else None
        )

    indices_buffer = wp.array(list(range(n)) + [n] * n, dtype=wp.int32, device=device)
    data_buffer = reinterpret_cast_to_int(data, 2 * n)
    wp.utils.radix_sort_pairs(data_buffer, indices_buffer, count=n)
    sorted_data = wp.empty(n, dtype=data_buffer.dtype, device=device)
    wp.copy(sorted_data, data_buffer, count=n)

    unique_counts = None
    unique_inverse = None
    inverse_buffer = None

    if wp.types.types_equal(data_buffer.dtype, wp.int32):
        unique_values_int = wp.empty(n, dtype=wp.int32, device=device)
        unique_counts_buffer = wp.empty(n, dtype=wp.int32, device=device)
        n_unique = wp.utils.runlength_encode(sorted_data, unique_values_int, run_lengths=unique_counts_buffer)
        if return_counts:
            unique_counts = wp.empty(n_unique, dtype=wp.int32, device=device)
            wp.copy(unique_counts, unique_counts_buffer, count=n_unique)
        if return_inverse:
            counts_list = unique_counts_buffer.list()
            inverse_buffer = wp.array(
                [i for i in range(n_unique) for _ in range(counts_list[i])] + [-1] * n, dtype=wp.int32, device=device
            )
    else:
        unique_start_mask = wp.empty(n, dtype=wp.int32, device=device)
        wp.launch(kernel_unique.mark_run_starts, dim=n, inputs=[sorted_data, unique_start_mask], device=device)
        indices = wp.empty(n, dtype=wp.int32, device=device)
        wp.utils.array_scan(unique_start_mask, indices, inclusive=True)
        wp.launch(kernel_array.sub, dim=n, inputs=[indices, wp.int32(1)], device=device)
        n_unique = tw.reduce.max(indices) + 1
        unique_values_int = wp.empty(n, dtype=sorted_data.dtype, device=device)
        wp.launch(
            kernel_unique.scatter_from_masked_indices,
            dim=n,
            inputs=[sorted_data, unique_start_mask, indices, unique_values_int],
            device=device,
        )
        if return_counts:
            unique_counts_buffer = wp.empty(n, dtype=wp.int32, device=device)
            _ = wp.utils.runlength_encode(indices, run_values=unique_start_mask, run_lengths=unique_counts_buffer)
            unique_counts = wp.empty(n_unique, dtype=wp.int32, device=device)
            wp.copy(unique_counts, unique_counts_buffer, count=n_unique)
        if return_inverse:
            inverse_buffer = wp.empty(2 * n, dtype=wp.int32, device=device)
            wp.copy(inverse_buffer, indices, count=n)
    unique_values = reinterpret_cast_from_int(unique_values_int, data.dtype, count=n_unique)

    if return_inverse:
        assert inverse_buffer is not None
        wp.utils.radix_sort_pairs(indices_buffer, inverse_buffer, n)
        unique_inverse = wp.empty(n, dtype=wp.int32, device=device)
        wp.copy(unique_inverse, inverse_buffer, count=n)

    return _pack_unique_result(unique_values, inverse=unique_inverse, counts=unique_counts)


def _pack_unique_result(
    unique: wp.array[Scalar], *, inverse: wp.array[wp.int32] | None = None, counts: wp.array[wp.int32] | None = None
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    if inverse is not None and counts is not None:
        return unique, inverse, counts
    if inverse is not None:
        return unique, inverse
    if counts is not None:
        return unique, counts
    return unique


def reinterpret_cast_to_int(
    data: wp.array[wp.Scalar], count: int | None = None
) -> wp.array[wp.int32] | wp.array[wp.int64]:
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


def reinterpret_cast_from_int(
    data: wp.array[wp.int32] | wp.array[wp.int64], dtype: type[Scalar], count: int | None = None
) -> wp.array[Scalar]:
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
