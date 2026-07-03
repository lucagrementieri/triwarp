import warp as wp


@wp.kernel
def group_sorted_fixed_length(
    sorted_values: wp.array[wp.Int],
    indices: wp.array[wp.int32],
    out_counter: wp.array[wp.int32],
    out_groups: wp.array2d[wp.int32],
) -> None:
    tid = wp.tid()
    length = out_groups.shape[1]
    n = out_groups.shape[0]

    # Prevent out-of-bounds indexing
    if tid + length > n:
        return

    # Check if the index is a start of a run
    if (tid != 0) and (sorted_values[tid] == sorted_values[tid - 1]):
        return
    # Check if the run spans at least 'length'
    if sorted_values[tid] != sorted_values[tid + length - 1]:
        return
    # Check if the run exceeds 'length'
    if (tid + length < n) and (sorted_values[tid] == sorted_values[tid + length]):
        return

    idx = wp.atomic_add(out_counter, 0, 1)
    for j in range(length):
        out_groups[idx, j] = indices[tid + j]
