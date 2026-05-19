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
    if starts[i] == wp.int32(1):
        unique_index = shifted_indices[i] - wp.int32(1)
        out_unique[unique_index] = values[i]


@wp.kernel
def compute_run_lengths(
    flags: wp.array[wp.int32],
    inclusive_scan: wp.array[wp.int32],
    n: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    if flags[i] != wp.int32(1):
        return

    slot = inclusive_scan[i] - wp.int32(1)
    j = i + 1
    n_int = int(n)
    while j < n_int and flags[j] == wp.int32(0):
        j += 1
    out_counts[slot] = wp.int32(j - i)


@wp.kernel
def sorted_unique_index(inclusive_scan: wp.array[wp.int32], out_inverse: wp.array[wp.int32]) -> None:
    i = int(wp.tid())
    out_inverse[i] = inclusive_scan[i] - wp.int32(1)


@wp.kernel
def gather_by_indices(
    values: wp.array[wp.uint64], indices: wp.array[wp.int32], out_sorted: wp.array[wp.uint64]
) -> None:
    i = int(wp.tid())
    out_sorted[i] = values[indices[i]]
