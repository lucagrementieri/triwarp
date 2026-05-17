import warp as wp

VEC3_PACK_PRECISION = wp.constant(wp.uint64(64 // 3))
VEC3_PACK_SHIFT = wp.constant(wp.uint32(11))


@wp.kernel
def pack_vec3(vectors: wp.array[wp.vec3], out_packed: wp.array[wp.uint64]) -> None:
    tid = int(wp.tid())
    vector = vectors[tid]

    # 1. Bit-cast float32 to uint32 to look at raw bits
    # 2. Shift right by 11 bits to discard the lower mantissa bits
    ix = wp.cast(vector[0], wp.uint32) >> VEC3_PACK_SHIFT
    iy = wp.cast(vector[1], wp.uint32) >> VEC3_PACK_SHIFT
    iz = wp.cast(vector[2], wp.uint32) >> VEC3_PACK_SHIFT

    # 3. Explicitly promote components to uint64 before shifting.
    # This avoids 32-bit integer overflow during the large left-shifts (<< 21 and << 42)
    packed_val = wp.uint64(ix) | (
        (wp.uint64(iy) << VEC3_PACK_PRECISION) | (wp.uint64(iz) << (VEC3_PACK_PRECISION + VEC3_PACK_PRECISION))
    )

    out_packed[tid] = packed_val
