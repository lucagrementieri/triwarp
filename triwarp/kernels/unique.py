import warp as wp

HASH_MULT_U64 = wp.constant(wp.uint64(11400714819323198485))  # 0x9e3779b97f4a7c15


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


@wp.kernel
def hash_insert(
    data: wp.array[wp.Int],
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    mask: wp.int32,
) -> None:
    i = int(wp.tid())
    key = data[i]
    empty = empty_key(key)
    encoded = encode_key(key)
    h = hash_slot(key, mask)
    while True:
        prev = wp.atomic_cas(slot_key, h, empty, encoded)
        if prev == empty or prev == encoded:
            wp.atomic_add(slot_counts, h, wp.int32(1))
            break
        h = next_slot(h, mask)


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
