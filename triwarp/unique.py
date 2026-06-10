"""1D unique values (``numpy.unique``-style) for Warp rank-1 scalar arrays."""

from __future__ import annotations

import math
from typing import Literal, TypeVar, overload

import warp as wp

import triwarp as tw
from triwarp.array import init_range
from triwarp.kernels import array as kernel_array
from triwarp.kernels import unique as kernel_unique

Scalar = TypeVar("Scalar", bound=wp.Scalar)


@overload
def unique_1d(
    data: wp.array[Scalar],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
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

    Uses an open-addressing hash table: O(n) inserts plus an O(n_unique log n_unique) sort of only
    the unique values.

    ``return_inverse`` maps each input position to the index of its value in the sorted unique
    output.
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

    if n == 0:
        empty_unique = wp.empty(0, dtype=data.dtype, device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )
    if n > (1 << 30):
        raise ValueError(f"unique_1d requires length <= 2**30, got length {n}")

    log2_capacity = max(3, math.ceil(math.log2(n) + 1))
    mask = wp.int32((1 << log2_capacity) - 1)

    data_int = reinterpret_cast_to_int(data, n)
    return _unique_hash(data_int, data.dtype, n, mask, return_inverse, return_counts)


def _unique_hash(
    data_int: wp.array[wp.int32] | wp.array[wp.int64],
    original_dtype: type[Scalar],
    n: int,
    mask: wp.int32,
    return_inverse: bool,
    return_counts: bool,
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    cap = int(mask) + 1
    key_dtype = data_int.dtype
    device = data_int.device

    # Phase 1: parallel insert into open-addressing hash table (slot_key 0 = empty).
    slot_key = wp.zeros(cap, dtype=key_dtype, device=device)
    slot_counts = wp.zeros(cap, dtype=wp.int32, device=device)
    wp.launch(
        kernel_unique.hash_insert,
        dim=n,
        inputs=[data_int, slot_key, slot_counts, mask],
        device=device,
    )

    # Phase 2: mark occupied slots, prefix-scan to get compact positions.
    occ_mask = wp.zeros(cap, dtype=wp.int32, device=device)
    wp.launch(kernel_unique.mark_occupied, dim=cap, inputs=[slot_key, occ_mask], device=device)
    scan_pos = wp.empty(cap, dtype=wp.int32, device=device)
    wp.utils.array_scan(occ_mask, scan_pos, inclusive=True)
    wp.launch(kernel_array.sub, dim=cap, inputs=[scan_pos, wp.int32(1)], device=device)
    n_unique = int(tw.reduce.max(scan_pos)) + 1

    # Phase 3: compact unique keys and their occurrence counts.
    keys_compact = wp.empty(n_unique, dtype=key_dtype, device=device)
    cnts_compact = wp.empty(n_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_unique.compact_from_table,
        dim=cap,
        inputs=[slot_key, slot_counts, occ_mask, scan_pos, keys_compact, cnts_compact],
        device=device,
    )

    # Phase 4: sort only the n_unique keys (typically n_unique << n).
    keys_buf = wp.empty(2 * n_unique, dtype=key_dtype, device=device)
    wp.copy(keys_buf, keys_compact, count=n_unique)
    perm_buf = init_range(2 * n_unique, device)
    wp.utils.radix_sort_pairs(keys_buf, perm_buf, count=n_unique)

    unique_values = reinterpret_cast_from_int(keys_buf, original_dtype, count=n_unique)

    unique_counts = None
    if return_counts:
        sort_perm = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.copy(sort_perm, perm_buf, count=n_unique)
        unique_counts = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.copy(unique_counts, cnts_compact[sort_perm])

    unique_inverse = None
    if return_inverse:
        sorted_dense = wp.empty(n_unique, dtype=key_dtype, device=device)
        wp.copy(sorted_dense, keys_buf, count=n_unique)
        unique_inverse = wp.empty(n, dtype=wp.int32, device=device)
        wp.launch(
            kernel_array.map_sorted_inverse,
            dim=n,
            inputs=[data_int, sorted_dense, unique_inverse],
            device=device,
        )

    return _pack_unique_result(unique_values, inverse=unique_inverse, counts=unique_counts)


def _pack_unique_result(
    unique: wp.array[Scalar],
    *,
    inverse: wp.array[wp.int32] | None = None,
    counts: wp.array[wp.int32] | None = None,
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
