"""1D unique values (``numpy.unique``-style) for Warp rank-1 scalar arrays."""

import warp as wp
from typing import Literal, overload


@overload
def unique_1d(
    data: wp.array[wp.int32],
    *,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
) -> wp.array[wp.int32]: ...
@overload
def unique_1d(
    data: wp.array[wp.float32],
    *,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[False] = ...,
) -> wp.array[wp.float32]: ...
@overload
def unique_1d(
    data: wp.array[wp.int32],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[wp.float32],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = ...,
) -> tuple[wp.array[wp.float32], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[wp.int32],
    *,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[wp.float32],
    *,
    return_inverse: Literal[False] = ...,
    return_counts: Literal[True],
) -> tuple[wp.array[wp.float32], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[wp.int32],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[True],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[wp.float32],
    *,
    return_inverse: Literal[True],
    return_counts: Literal[True],
) -> tuple[wp.array[wp.float32], wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_1d(
    data: wp.array[wp.int32] | wp.array[wp.float32],
    *,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> (
    wp.array[wp.int32]
    | wp.array[wp.float32]
    | tuple[wp.array[wp.int32], wp.array[wp.int32]]
    | tuple[wp.array[wp.float32], wp.array[wp.int32]]
    | tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
    | tuple[wp.array[wp.float32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find sorted unique elements of a 1D Warp array (``numpy.unique`` subset).

    Uses stable ``warp.utils.radix_sort_pairs`` on sortable integer keys (signed
    integers are remapped so radix order matches two's-complement order; floats
    use total-order bit keys). ``return_inverse`` maps each input position to the
    index of its value in the sorted unique output. ``return_index`` is not
    supported.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` of ``wp.int32`` or ``wp.float32``.
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
        If ``data`` is not rank-1, dtype is unsupported, or length is ``>= 2**31``.
    """
    if int(data.ndim) != 1:
        raise ValueError(f"unique_1d expects a rank-1 array, got ndim={data.ndim}")
    if data.dtype not in (wp.int32, wp.float32):
        raise ValueError(f"unique_1d dtype must be one of (wp.int32, wp.float32), got {data.dtype}")

    n = int(data.shape[0])
    if n >= (1 << 31):
        raise ValueError(
            f"unique_1d requires length < 2**31 because radix sort value indices are int32, got length {n}"
        )

    if n == 0:
        empty_unique = wp.empty(0, dtype=data.dtype, device=data.device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=data.device)
        if not return_inverse and not return_counts:
            return empty_unique
        if return_inverse and return_counts:
            return empty_unique, empty_i32, empty_i32
        if return_inverse:
            return empty_unique, empty_i32
        return empty_unique, empty_i32

    # Sort data and indices
    indices_buffer = wp.array(list(range(n)) + [-1] * n, dtype=wp.int32, device=data.device)
    data_buffer = wp.empty(indices_buffer.shape, dtype=data.dtype, device=data.device)
    wp.copy(data_buffer, data)
    wp.utils.radix_sort_pairs(data_buffer, indices_buffer, n)
    data_buffer_int32 = wp.empty(n, dtype=wp.int32, device=data.device)
    wp.copy(data_buffer_int32, data_buffer, count=n)

    unique_values_int32 = wp.empty(n, dtype=wp.int32, device=data.device)
    unique_counts = wp.empty(n, dtype=wp.int32, device=data.device)

    n_unique = wp.utils.runlength_encode(data_buffer_int32, unique_values_int32, run_lengths=unique_counts)
    unique_values = wp.empty(n_unique, dtype=data.dtype, device=data.device)
    wp.copy(unique_values, unique_values_int32, count=n_unique)

    if not return_inverse:
        if return_counts:
            return unique_values, unique_counts
        return unique_values

    if return_inverse and return_counts:
        return unique_values, unique_counts, unique_counts
    return unique_values, unique_counts
