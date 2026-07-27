import warp as wp

from triwarp.kernels import array as kernel_array


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


HASH_MULT_U64 = wp.constant(wp.uint64(11400714819323198485))  # 0x9e3779b97f4a7c15

VEC3_PACK_PRECISION = wp.constant(wp.uint64(64 // 3))
VEC3_PACK_SHIFT = wp.constant(wp.uint32(11))


# ---------------------------------------------------------------------------
# Shared hash helpers
# All hash kernels receive `mask: wp.int32` = capacity - 1 rather than capacity
# itself.  For capacity <= 2^31, mask = capacity - 1 <= INT32_MAX, which is always
# a valid non-negative int32.  Slot indices h = hash(...) & mask satisfy
# h <= mask <= INT32_MAX, so h is also safely non-negative as int32.
# ---------------------------------------------------------------------------


@wp.func
def hash_slot(key: wp.Int, mask: wp.int32) -> wp.int32:
    """Fibonacci hash; mask = capacity-1, capacity must be power-of-2."""
    # wp.cast is same-size only; use constructors for cross-size conversion.
    h = wp.uint64(wp.int64(key)) * HASH_MULT_U64
    return wp.int32(h & wp.uint64(mask))


@wp.func
def next_slot(h: wp.int32, mask: wp.int32) -> wp.int32:
    """Linear-probe to the next slot, wrapping with the power-of-2 mask."""
    return wp.cast((wp.cast(h, wp.uint32) + wp.uint32(1)) & wp.cast(mask, wp.uint32), wp.int32)


@wp.func
def empty_key(key: wp.Int) -> wp.Int:
    """Typed zero matching ``key`` (for empty-slot sentinel and comparisons)."""
    return key - key


@wp.func
def encode_key(key: wp.Int) -> wp.Int:
    """Map keys to table slots; 0 is reserved as the empty sentinel."""
    return -~key


@wp.func
def decode_key(stored: wp.Int) -> wp.Int:
    zero = stored - stored
    return stored + (~zero)


# ---------------------------------------------------------------------------
# Generic hash table kernels
# Slot layout: slot_key (Int, 0 = empty) + slot_counts (int32).
# Insertion uses atomic_cas on slot_key so publication is a single atomic
# operation — no separate lock/ready state and no cross-thread visibility races.
# ---------------------------------------------------------------------------


@wp.func
def hash_find_or_insert(key: wp.Int, slot_key: wp.array[wp.Int], mask: wp.int32) -> wp.int32:
    """
    Slot holding ``key``, inserting it first if it is not there yet.

    Publication is the single ``atomic_cas`` that claims an empty slot, so there is no separate
    lock or ready state and no cross-thread visibility race. The probe cannot terminate on a
    **full** table, so callers must size the table above the number of distinct keys.
    """
    empty = empty_key(key)
    encoded = encode_key(key)
    h = hash_slot(key, mask)
    while True:
        prev = wp.atomic_cas(slot_key, h, empty, encoded)
        if prev == empty or prev == encoded:
            break
        h = next_slot(h, mask)
    return h


@wp.func
def hash_find(key: wp.Int, slot_key: wp.array[wp.Int], mask: wp.int32) -> wp.int32:
    """
    Slot holding ``key``, or ``-1`` when it is absent.

    Read-only, so it terminates at the first empty slot in the probe chain even on a full table.
    """
    empty = empty_key(key)
    encoded = encode_key(key)
    h = hash_slot(key, mask)
    # No boolean accumulator: a bare ``True`` / ``False`` is a *constant* in kernel scope and Warp
    # refuses to let a dynamic loop mutate one. The probe state itself carries the answer.
    stored = slot_key[h]
    while stored != empty and stored != encoded:
        h = next_slot(h, mask)
        stored = slot_key[h]
    if stored != encoded:
        h = wp.int32(-1)
    return h


@wp.func
def hash_contains(key: wp.Int, slot_key: wp.array[wp.Int], mask: wp.int32) -> bool:
    """Whether ``key`` is in the table, without inserting it."""
    return hash_find(key, slot_key, mask) >= 0


@wp.kernel
def hash_insert(
    data: wp.array[wp.Int],
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    mask: wp.int32,
) -> None:
    i = int(wp.tid())
    wp.atomic_add(slot_counts, hash_find_or_insert(data[i], slot_key, mask), wp.int32(1))


@wp.kernel
def mark_occupied(slot_key: wp.array[wp.Int], out_mask: wp.array[wp.int32]) -> None:
    h = int(wp.tid())
    if slot_key[h] != empty_key(slot_key[h]):
        out_mask[h] = wp.int32(1)


@wp.kernel
def compact_from_table(
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    occ_mask: wp.array[wp.int32],
    scan_pos: wp.array[wp.int32],
    out_keys: wp.array[wp.Int],
    out_counts: wp.array[wp.int32],
) -> None:
    h = int(wp.tid())
    if occ_mask[h] == wp.int32(1):
        pos = scan_pos[h]
        out_keys[pos] = decode_key(slot_key[h])
        out_counts[pos] = slot_counts[h]


@wp.func
def pack_vec3(vector: wp.vec3) -> wp.uint64:
    # 1. Bit-cast float32 to uint32 to look at raw bits
    # 2. Shift right by 11 bits to discard the lower mantissa bits
    ix = wp.cast(vector[0], wp.uint32) >> VEC3_PACK_SHIFT
    iy = wp.cast(vector[1], wp.uint32) >> VEC3_PACK_SHIFT
    iz = wp.cast(vector[2], wp.uint32) >> VEC3_PACK_SHIFT

    # 3. Explicitly promote components to uint64 before shifting.
    # This avoids 32-bit integer overflow during the large left-shifts (<< 21 and << 42)
    return wp.uint64(ix) | (
        (wp.uint64(iy) << VEC3_PACK_PRECISION)
        | (wp.uint64(iz) << (VEC3_PACK_PRECISION + VEC3_PACK_PRECISION))
    )


@wp.func
def pack_edge_key(u: wp.int32, v: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of undirected edge (u, v); matches ``pack_indices`` for a sorted 2-index row."""
    lo = wp.uint64(wp.uint32(wp.min(u, v)))
    hi = wp.uint64(wp.uint32(wp.max(u, v)))
    return lo + hi * base


@wp.kernel
def pack_indices(
    indices: wp.array2d[wp.int32], max_index: wp.uint64, out_packed: wp.array[wp.uint64]
) -> None:
    tid = int(wp.tid())
    indices_row = indices[tid]
    packed_value = wp.uint64(0)
    power = wp.uint64(1)
    for i in range(indices_row.shape[0]):
        digit = wp.uint64(wp.uint32(indices_row[i]))
        packed_value = packed_value + digit * power
        power = power * max_index

    out_packed[tid] = packed_value


@wp.kernel
def round_vec3_scaled(
    vertices: wp.array[wp.vec3], inv_epsilon: wp.float32, out_rounded: wp.array2d[wp.int32]
) -> None:
    tid = int(wp.tid())
    v = vertices[tid] * inv_epsilon
    out_rounded[tid, 0] = wp.int32(wp.round(v[0]))
    out_rounded[tid, 1] = wp.int32(wp.round(v[1]))
    out_rounded[tid, 2] = wp.int32(wp.round(v[2]))


@wp.kernel
def sort_face_indices(faces: wp.array2d[wp.int32], out_sorted: wp.array2d[wp.int32]) -> None:
    tid = int(wp.tid())
    i0 = faces[tid, 0]
    i1 = faces[tid, 1]
    i2 = faces[tid, 2]
    s0, s1, s2 = kernel_array.sort3(i0, i1, i2)
    out_sorted[tid, 0] = wp.int32(s0)
    out_sorted[tid, 1] = wp.int32(s1)
    out_sorted[tid, 2] = wp.int32(s2)
