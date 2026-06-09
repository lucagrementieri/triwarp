import warp as wp


@wp.kernel
def offset_copy_int32(
    src: wp.array[wp.int32], offset: wp.int32, dest_offset: wp.int32, out: wp.array[wp.int32]
) -> None:
    tid = int(wp.tid())
    out[dest_offset + tid] = src[tid] + offset
