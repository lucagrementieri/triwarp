import warp as wp


@wp.kernel
def mark_label_starts(sorted_labels: wp.array[wp.int32], out_is_start: wp.array[wp.bool]) -> None:
    # Segment boundaries of a label-sorted array: position 0 and every label change.
    i = int(wp.tid())
    out_is_start[i] = i == 0 or sorted_labels[i] != sorted_labels[i - 1]
