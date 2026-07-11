import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import init_sort_pair_indices
from triwarp.kernels import grouping as kernel_grouping


# TODO: probably implement with stream compaction instead, probably general function
# usable also for unique
def group(values: wp.array[wp.Int], length: int) -> twt.Array2dInt32:
    """
    Return index groups of exactly ``length`` entries that share the same value.

    ``values`` are radix-sorted with their original indices; each output row lists
    ``length`` indices whose corresponding entries are equal and form a run of
    precisely that size (runs shorter or longer than ``length`` are omitted). This
    matches [`trimesh.grouping.group`][] with ``min_len == max_len == length`` and
    [`trimesh.grouping.group_rows`][] with ``require_count=length``.

    Parameters
    ----------
    values
        ``(n,)`` device array of integer keys (any ``wp.Int`` dtype).
    length
        Required run length; each returned group contains exactly this many indices.

    Returns
    -------
    twt.Array2dInt32
        ``(g, length)`` array on ``values.device`` where ``g`` is the number of
        groups found. Empty when no run has exactly ``length`` equal neighbors.

    See Also
    --------
    [`group_int_rows`][triwarp.grouping.group_int_rows]
    """
    n = int(values.shape[0])
    sort_dtype = values.dtype if wp.types.type_size_in_bytes(values.dtype) >= 4 else wp.int32
    values_buffer = wp.empty(2 * n, dtype=sort_dtype, device=values.device)
    if sort_dtype == values.dtype:
        wp.copy(values_buffer, values, count=n)
    else:
        wp.utils.array_cast(values, values_buffer, count=n)
    indices_buffer = init_sort_pair_indices(n, -1, values.device)
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
    return twt.as_array2d_int32(groups)


def group_int_rows(
    data: twt.Array2dInt, length: int, max_value: int | None = None
) -> twt.Array2dInt32:
    """
    Return index groups of exactly ``length`` rows that are identical.

    Each row is hashed with [`hash_indices_rows`][triwarp.unique.hash_indices_rows], then
    [`group`][triwarp.grouping.group] finds runs of ``length`` equal keys in sorted order. For
    example, ``[[1, 2], [3, 4], [1, 2]]`` with ``length=2`` yields one group ``[[0, 2]]`` (same as
    [`trimesh.grouping.group_rows`][] with ``require_count=2``).

    Parameters
    ----------
    data
        ``(n, w)`` device array of non-negative ``int32`` row values.
    length
        Required number of duplicate rows per group.
    max_value
        Optional exclusive upper bound on entries and radix for row hashing; passed
        through to [`hash_indices_rows`][triwarp.unique.hash_indices_rows] as ``max_index``. If
        ``None``, inferred
        from ``max(data) + 1``.

    Returns
    -------
    twt.Array2dInt32
        ``(g, length)`` array on ``data.device`` with original row indices per group.

    See Also
    --------
    [`group`][triwarp.grouping.group]
    """
    twt.ensure_ndim(data, 2)
    hashed_rows = tw.unique.hash_indices_rows(data, max_value)
    return group(hashed_rows, length)
