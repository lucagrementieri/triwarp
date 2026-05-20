import warp as wp

from triwarp.kernels import grouping as kernel_grouping
import triwarp as tw


def group(values: wp.array[wp.Int], length: int) -> wp.array2d[wp.int32]:
    n = int(values.shape[0])
    values_buffer = tw.unique.reinterpret_cast_to_int(values, 2 * n)
    indices_buffer = wp.array(list(range(n)) + [-1] * n, dtype=wp.int32, device=values.device)
    wp.utils.radix_sort_pairs(values_buffer, indices_buffer, count=n)

    counter = wp.zeros(1, dtype=wp.int32, device=values.device)
    groups_buffer = wp.empty((n, length), dtype=wp.int32, device=values.device)
    wp.launch(
        kernel_grouping.group_sorted_fixed_length,
        dim=n - length + 1,
        inputs=[values_buffer, indices_buffer, length, counter, groups_buffer],
        device=values.device,
    )
    n_groups = counter.numpy().item()
    if n_groups == 0:
        return wp.empty((0, length), dtype=wp.int32, device=values.device)
    groups = wp.empty((n_groups, length), dtype=wp.int32, device=values.device)
    wp.copy(groups, groups_buffer, count=n_groups * length)
    return groups


def hash_vector_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]:
    """
    Pack each ``wp.vec3`` row into a single ``uint64`` key.

    Each coordinate is interpreted as ``float32`` bits, right-shifted by 11 to drop low
    mantissa bits, then concatenated into one key per row for bucketing or
    near-duplicate grouping.

    Parameters
    ----------
    data
        ``(n,)`` device array of ``wp.vec3`` values.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    See Also
    --------
    hash_indices_rows
    """
    if data.dtype != wp.vec3:
        raise ValueError(f"data must be a wp.array[wp.vec3], got wp.array[{data.dtype}]")
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(kernel_grouping.pack_vec3, dim=data.shape[0], inputs=[data, hashes], device=data.device)
    return hashes


def hash_indices_rows(data: wp.array2d[wp.int32], max_index: int | None = None) -> wp.array[wp.uint64]:
    """
    Pack each row of non-negative ``int32`` values into a single ``uint64`` key.

    Each row is treated as digits in base ``max_index`` (or ``max(data)+1`` when
    ``max_index`` is omitted) and combined into one integer key per row. A global
    min/max over ``data`` (via ``triwarp.reduce.minmax``) validates the range and
    supplies the radix when it is inferred.

    Parameters
    ----------
    data
        ``(n, w)`` device array of ``int32``; every entry must be non-negative. If
        ``max_index`` is given, every entry must satisfy ``entry < max_index``.
    max_index
        Optional exclusive upper bound on entries and radix for packing; must be
        positive when provided. If ``None``, set to ``max(data) + 1`` after validation.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    See Also
    --------
    hash_vector_rows
    """
    if data.dtype != wp.int32:
        raise ValueError(f"data must be a wp.array2d[wp.int32], got wp.array2d[{data.dtype}]")
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    min_data, max_data = tw.reduce.minmax(data)
    if min_data < 0:
        raise ValueError(f"data must be non-negative, got a minimum of {min_data}")
    if max_index is not None and max_data >= max_index:
        raise ValueError(f"data must be less than max_index {max_index}, got a maximum of {max_data}")
    if max_index is None:
        max_index = max_data + 1
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(
        kernel_grouping.pack_indices, dim=data.shape[0], inputs=[data, wp.uint64(max_index), hashes], device=data.device
    )
    return hashes
