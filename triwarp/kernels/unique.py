import warp as wp


# TODO: maybe where or select
@wp.kernel
def mark_run_starts(values: wp.array[wp.Scalar], out_starts: wp.array[wp.int32]) -> None:
    i = int(wp.tid())
    if i == 0 or values[i] != values[i - 1]:
        out_starts[i] = wp.int32(1)
    else:
        out_starts[i] = wp.int32(0)


@wp.kernel
def scatter_from_masked_indices(
    values: wp.array[wp.Scalar], mask: wp.array[wp.int32], indices: wp.array[wp.int32], out_values: wp.array[wp.Scalar]
) -> None:
    i = int(wp.tid())
    if mask[i] == 1:
        index = indices[i]
        out_values[index] = values[i]
