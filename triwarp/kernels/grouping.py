import warp as wp


@wp.kernel
def mark_group_starts(
    sorted_values: wp.array[wp.Int], n: wp.int32, length: wp.int32, out_is_start: wp.array[wp.bool]
) -> None:
    # Flag positions that start a run of exactly ``length`` equal values in the sorted buffer
    # (which may be over-allocated radix-sort scratch; only the first ``n`` entries are data).
    tid = int(wp.tid())
    is_start = True
    if tid + int(length) > int(n):
        is_start = False  # run would extend past the data
    elif tid != 0 and sorted_values[tid] == sorted_values[tid - 1]:
        is_start = False  # not the start of a run
    elif sorted_values[tid] != sorted_values[tid + int(length) - 1]:
        is_start = False  # run shorter than ``length``
    elif tid + int(length) < int(n) and sorted_values[tid] == sorted_values[tid + int(length)]:
        is_start = False  # run longer than ``length``
    out_is_start[tid] = is_start


@wp.kernel
def emit_groups(
    starts: wp.array[wp.int32], indices: wp.array[wp.int32], out_groups: wp.array2d[wp.int32]
) -> None:
    g = int(wp.tid())
    start = starts[g]
    for j in range(out_groups.shape[1]):
        out_groups[g, j] = indices[start + j]
