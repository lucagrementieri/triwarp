import warp as wp

VEC3_PACK_PRECISION = wp.constant(wp.uint64(64 // 3))
VEC3_PACK_SHIFT = wp.constant(wp.uint32(11))


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
    packed_value = wp.uint64(ix) | (
        (wp.uint64(iy) << VEC3_PACK_PRECISION) | (wp.uint64(iz) << (VEC3_PACK_PRECISION + VEC3_PACK_PRECISION))
    )

    out_packed[tid] = packed_value


@wp.kernel
def pack_indices(indices: wp.array2d[wp.int32], max_index: wp.uint64, out_packed: wp.array[wp.uint64]) -> None:
    tid = int(wp.tid())
    indices_row = indices[tid]
    packed_value = wp.uint64(0)
    power = wp.uint64(1)
    for i in range(indices_row.shape[0]):
        digit = wp.uint64(wp.uint32(indices_row[i]))
        packed_value = packed_value + digit * power
        power = power * max_index

    out_packed[tid] = packed_value
