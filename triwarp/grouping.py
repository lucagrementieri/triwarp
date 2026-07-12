import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import init_sort_pair_indices
from triwarp.kernels import grouping as kernel_grouping


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
    device = values.device
    if n < length or length <= 0:
        return twt.as_array2d_int32(twt.empty_int32_2d((0, max(length, 0)), device=device))

    sort_dtype = values.dtype if wp.types.type_size_in_bytes(values.dtype) >= 4 else wp.int32
    values_buffer = wp.empty(2 * n, dtype=sort_dtype, device=device)
    if sort_dtype == values.dtype:
        wp.copy(values_buffer, values, count=n)
    else:
        wp.utils.array_cast(values, values_buffer, count=n)
    indices_buffer = init_sort_pair_indices(n, -1, device)
    wp.utils.radix_sort_pairs(values_buffer, indices_buffer, count=n)

    # Scan + scatter compaction: mark run starts, compact them with flatnonzero, then emit one
    # right-sized row per group (deterministic ascending-value order, no atomic counter and no
    # (n, length) over-allocation).
    is_start = wp.empty(n, dtype=wp.bool, device=device)
    wp.launch(
        kernel_grouping.mark_group_starts,
        dim=n,
        inputs=[values_buffer, wp.int32(n), wp.int32(length), is_start],
        device=device,
    )
    starts = tw.array.flatnonzero(is_start)
    n_groups = int(starts.shape[0])
    if n_groups == 0:
        return twt.as_array2d_int32(twt.empty_int32_2d((0, length), device=device))
    groups = twt.empty_int32_2d((n_groups, length), device=device)
    wp.launch(
        kernel_grouping.emit_groups,
        dim=n_groups,
        inputs=[starts, indices_buffer, groups],
        device=device,
    )
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
