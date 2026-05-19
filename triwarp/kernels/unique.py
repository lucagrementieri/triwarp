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
def scatter_unique_from_run_starts(
    values: wp.array[wp.Scalar],
    starts: wp.array[wp.int32],
    shifted_indices: wp.array[wp.int32],
    out_unique: wp.array[wp.Scalar],
) -> None:
    i = int(wp.tid())
    if starts[i] == 1:
        unique_index = shifted_indices[i] - wp.int32(1)
        out_unique[unique_index] = values[i]
