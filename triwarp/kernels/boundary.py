import warp as wp


@wp.kernel
def find_ears(
    edge_boundary: wp.array[wp.bool],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    f = int(wp.tid())
    base = f * 3
    b0 = edge_boundary[base]
    b1 = edge_boundary[base + 1]
    b2 = edge_boundary[base + 2]
    n = wp.int32(b0) + wp.int32(b1) + wp.int32(b2)
    if n == 2:
        slot = wp.atomic_add(out_count, 0, 1)
        out_ear[slot] = wp.int32(f)
        if not b0:
            out_ear_opp[slot] = wp.int32(0)
        elif not b1:
            out_ear_opp[slot] = wp.int32(1)
        else:
            out_ear_opp[slot] = wp.int32(2)
