import warp as wp

from triwarp.kernels import grouping as kernel_grouping
import triwarp as tw


# TODO: probably implement with stream compaction instead, probably general function usable also for unique
def group(values: wp.array[wp.Int], length: int) -> wp.array2d[wp.int32]:
    """
    Return index groups of exactly ``length`` entries that share the same value.

    ``values`` are radix-sorted with their original indices; each output row lists
    ``length`` indices whose corresponding entries are equal and form a run of
    precisely that size (runs shorter or longer than ``length`` are omitted). This
    matches :func:`trimesh.grouping.group` with ``min_len == max_len == length`` and
    :func:`trimesh.grouping.group_rows` with ``require_count=length``.

    Parameters
    ----------
    values
        ``(n,)`` device array of integer keys (any ``wp.Int`` dtype).
    length
        Required run length; each returned group contains exactly this many indices.

    Returns
    -------
    wp.array2d[wp.int32]
        ``(g, length)`` array on ``values.device`` where ``g`` is the number of
        groups found. Empty when no run has exactly ``length`` equal neighbors.

    See Also
    --------
    group_int_rows
    hash_indices_rows
    """
    n = int(values.shape[0])
    values_buffer = tw.unique.reinterpret_cast_to_int(values, 2 * n)
    indices_buffer = wp.array(list(range(n)) + [-1] * n, dtype=wp.int32, device=values.device)
    wp.utils.radix_sort_pairs(values_buffer, indices_buffer, count=n)

    counter = wp.zeros(1, dtype=wp.int32, device=values.device)
    groups_buffer = wp.empty((n, length), dtype=wp.int32, device=values.device)
    wp.launch(
        kernel_grouping.group_sorted_fixed_length,
        dim=n - length + 1,
        inputs=[values_buffer, indices_buffer, counter, groups_buffer],
        device=values.device,
    )
    n_groups = counter.numpy().item()
    groups = wp.empty((n_groups, length), dtype=wp.int32, device=values.device)
    if n_groups > 0:
        wp.copy(groups, groups_buffer, count=n_groups * length)
    return groups


def group_int_rows(data: wp.array2d[wp.Int], length: int, max_value: int | None = None) -> wp.array2d[wp.int32]:
    """
    Return index groups of exactly ``length`` rows that are identical.

    Each row is hashed with :func:`hash_indices_rows`, then :func:`group` finds runs
    of ``length`` equal keys in sorted order. For example, ``[[1, 2], [3, 4], [1, 2]]``
    with ``length=2`` yields one group ``[[0, 2]]`` (same as
    :func:`trimesh.grouping.group_rows` with ``require_count=2``).

    Parameters
    ----------
    data
        ``(n, w)`` device array of non-negative ``int32`` row values.
    length
        Required number of duplicate rows per group.
    max_value
        Optional exclusive upper bound on entries and radix for row hashing; passed
        through to :func:`hash_indices_rows` as ``max_index``. If ``None``, inferred
        from ``max(data) + 1``.

    Returns
    -------
    wp.array2d[wp.int32]
        ``(g, length)`` array on ``data.device`` with original row indices per group.

    See Also
    --------
    group
    hash_indices_rows
    """
    hashed_rows = hash_indices_rows(data, max_value)
    return group(hashed_rows, length)


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
