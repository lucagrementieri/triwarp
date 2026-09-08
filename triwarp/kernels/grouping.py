import warp as wp

from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import OverloadTable


@wp.kernel
def scatter_first_occurrence(inverse: wp.array[wp.int32], out_first: wp.array[wp.int32]) -> None:
    # Smallest index mapping to each class. ``out_first`` must be pre-filled with a sentinel at
    # least ``inverse.shape[0]``, so a class with no member keeps it.
    i = wp.int32(wp.tid())
    wp.atomic_min(out_first, inverse[i], i)


@wp.func
def sorted_run_start(sorted_values: wp.array[wp.Int], i: wp.int32) -> wp.bool:
    # Does position ``i`` begin a run of equal values? The one test every run-length grouping in
    # the package shares, and each of its three callers adds a different condition on top:
    # ``mark_group_starts`` requires the run to be exactly ``length`` long, ``remesh``'s
    # ``mark_edge_pair_starts`` specialises that to two, and its ``mark_unique_edge_starts`` wants
    # every run whatever its length. Bounding the *data* is the caller's job too -- the buffer is
    # usually over-allocated radix-sort scratch.
    if i == 0:
        return True
    return sorted_values[i] != sorted_values[i - 1]


@wp.kernel
def mark_group_starts(
    sorted_values: wp.array[wp.Int], n: wp.int32, length: wp.int32, out_is_start: wp.array[wp.bool]
) -> None:
    # Flag positions that start a run of exactly ``length`` equal values in the sorted buffer
    # (which may be over-allocated radix-sort scratch; only the first ``n`` entries are data).
    tid = wp.int32(wp.tid())
    is_start = True
    if tid + length > n:
        is_start = False  # run would extend past the data
    elif not sorted_run_start(sorted_values, tid):
        is_start = False  # not the start of a run
    elif sorted_values[tid] != sorted_values[tid + length - 1]:
        is_start = False  # run shorter than ``length``
    elif tid + length < n and sorted_values[tid] == sorted_values[tid + length]:
        is_start = False  # run longer than ``length``
    out_is_start[tid] = is_start


@wp.kernel
def emit_groups(
    starts: wp.array[wp.int32], indices: wp.array[wp.int32], out_groups: wp.array2d[wp.int32]
) -> None:
    g = wp.int32(wp.tid())
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


@wp.kernel
def hash_insert(
    data: wp.array[wp.Int],
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    mask: wp.int32,
    out_occupied: wp.array[wp.int32],
) -> None:
    # `encode_key` reserves 0 as the empty-slot sentinel, so exactly one key -- the one that encodes
    # to 0, i.e. -1 -- can never be published in the table: its CAS would look like an untouched
    # slot. Since no bijection on the full integer range can avoid mapping *something* onto the
    # sentinel, that key gets a dedicated slot one past the end of the table instead. Callers must
    # therefore allocate `mask + 2` slots, and occupancy is stamped here rather than read back off
    # `slot_key`, whose reserved slot keeps an untouched 0 and would otherwise look empty --
    # `decode_key` turns that 0 straight back into -1, so compaction needs no special case either.
    #
    # `out_occupied` is a zero-filled 0/1 array, the dtype `wp.utils.array_scan` wants, so the scan
    # of it gives the compaction's write positions directly. Every thread landing in a slot stores
    # the same 1, which is why the unsynchronized store is benign; writing it here rather than in a
    # second pass over the whole `mask + 2` table saves that pass -- measured 30 us of a 313 us
    # `unique_1d(100k)`, where the table is 2.6x the input.
    i = wp.int32(wp.tid())
    key = data[i]
    slot = mask + 1
    if encode_key(key) != empty_key(key):
        slot = hash_find_or_insert(key, slot_key, mask)
    wp.atomic_add(slot_counts, slot, wp.int32(1))
    out_occupied[slot] = wp.int32(1)


@wp.kernel
def compact_from_table(
    slot_key: wp.array[wp.Int],
    slot_counts: wp.array[wp.int32],
    occupied: wp.array[wp.int32],
    scan_pos: wp.array[wp.int32],
    out_keys: wp.array[wp.Int],
    out_counts: wp.array[wp.int32],
    out_perm: wp.array[wp.int32],
) -> None:
    # ``scan_pos`` is the *inclusive* scan of ``occupied``, so an occupied slot's compact position
    # is one less. Taking the -1 here rather than in a pass over the whole table is the other half
    # of the saving described in ``hash_insert``, and it leaves the scan's last element reading
    # ``n_unique`` outright.
    #
    # ``out_perm`` is the identity permutation the radix sort pairs with the keys. Writing it here
    # replaces an ``arange`` launch of its own, measured at 30 us -- as much as the sort it feeds.
    # Only the leading ``n_unique`` entries are written; the rest of the buffer is the sort's
    # double-buffer scratch, which it fills before reading.
    h = wp.int32(wp.tid())
    if occupied[h] == wp.int32(1):
        pos = scan_pos[h] - wp.int32(1)
        out_keys[pos] = decode_key(slot_key[h])
        out_counts[pos] = slot_counts[h]
        out_perm[pos] = pos


@wp.func
def bucket_float32(value: wp.float32) -> wp.uint32:
    # Bit-cast float32 to uint32 and drop the low 11 mantissa bits, so each key names a bucket
    # roughly 2^-12 wide *relative* to the value's magnitude.
    #
    # The sign bit is the most significant bit and survives the shift, so opposite signs never
    # share a bucket. That is what you want everywhere except at zero, where IEEE-754 has two
    # representations that compare equal: -0.0 has to fold onto +0.0 or a point sitting exactly on
    # an axis will not match itself. A revolved sphere's pole is the standard way to hit this,
    # since ``cos(theta) * 0.0`` is -0.0 for half the slices.
    if value == 0.0:
        return wp.uint32(0)
    return wp.cast(value, wp.uint32) >> VEC3_PACK_SHIFT


@wp.func
def pack_vec3(vector: wp.vec3) -> wp.uint64:
    ix = bucket_float32(vector[0])
    iy = bucket_float32(vector[1])
    iz = bucket_float32(vector[2])

    # Promote each component to uint64 before shifting, to avoid 32-bit overflow in the large
    # left-shifts (<< 21 and << 42).
    return wp.uint64(ix) | (
        (wp.uint64(iy) << VEC3_PACK_PRECISION)
        | (wp.uint64(iz) << (VEC3_PACK_PRECISION + VEC3_PACK_PRECISION))
    )


@wp.kernel
def pack_directed_index_keys(
    indices: wp.array2d[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # ``pack_indices`` for exactly two columns, taking the radix directly instead of inferring it.
    # Row order is preserved, so ``(a, b)`` and ``(b, a)`` get different keys -- which is the point
    # wherever a directed edge has to be told from its twin.
    i = wp.int32(wp.tid())
    out_keys[i] = kernel_array.pack_directed_key(indices[i, 0], indices[i, 1], base)


@wp.kernel
def pack_undirected_edge_keys(
    edges: wp.array2d[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # Deliberately not ``pack_indices``: that packs a row in the order it is given, and these keys
    # are compared against ``array.pack_edge_key``, which sorts. A caller's reversed row would
    # hash to something no halfedge can produce, so the edge would be silently missed.
    i = wp.int32(wp.tid())
    out_keys[i] = kernel_array.pack_edge_key(edges[i, 0], edges[i, 1], base)


@wp.kernel
def pack_indices(
    indices: wp.array2d[wp.int32], max_index: wp.uint64, out_packed: wp.array[wp.uint64]
) -> None:
    tid = wp.int32(wp.tid())
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
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    inv_epsilon: wp.float32,
    out_rounded: wp.array2d[wp.int32],
) -> None:
    # Snapping relative to `origin` -- the data's own minimum corner -- rather than to the
    # coordinate origin does two things. The cell indices come out non-negative, which the row
    # packing needs since it treats a row as digits in a positive radix. And the product stays the
    # size of the data's *extent* instead of its distance from zero: float32 carries about 7 digits,
    # so scaling a coordinate near 100 by 1e6 has already quantised away the low bits.
    tid = wp.int32(wp.tid())
    v = (vertices[tid] - origin) * inv_epsilon
    out_rounded[tid, 0] = wp.int32(wp.round(v[0]))
    out_rounded[tid, 1] = wp.int32(wp.round(v[1]))
    out_rounded[tid, 2] = wp.int32(wp.round(v[2]))


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured over the suite: 7 overloads created across **9** module loads.
#
# These take the caller's *key* dtype, and the two sets differ because the two call paths do.
# ``mark_group_starts`` is reached from ``grouping.group``, which widens through
# ``twt.sortable_dtype`` and is handed ``wp.uint64`` row hashes by ``hash_indices_rows`` -- so it
# needs the full unsigned surface ``triwarp.array``'s search kernels see. The open-addressing pair
# does not: their only caller is ``grouping._unique_hash``, whose ``data_int`` parameter is typed
# ``wp.array[wp.int32] | wp.array[wp.int64]`` because ``array.bitcast_to_int`` reinterprets every
# key into one *signed* space before the table sees it. The unsigned rows were unreachable, and
# section 2.5's rule is to register what the wrapper's dispatch can reach.
_KEY_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)
_TABLE_DTYPES = (wp.int32, wp.int64)


# The concrete handles keyed by the caller's key dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable]. Measured end to end on this module's own
# consumers, interleaved and min-of-12 on an RTX 5090: ``unique_1d(200k, return_inverse=True)``
# 305.0 -> 266.4 us (**1.15x**), ``unique_rows`` over an icosphere(6) edge table 612.4 -> 548.9
# (1.12x), ``edges_unique`` 683.7 -> 647.8 (1.055x), ``group(200k, 2)`` 273.9 -> 261.6 (1.047x) --
# roughly 12 us per generic launch removed, and this path issues two or three of them.
MARK_GROUP_STARTS: OverloadTable
HASH_INSERT: OverloadTable
COMPACT_FROM_TABLE: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MARK_GROUP_STARTS, HASH_INSERT, COMPACT_FROM_TABLE
    MARK_GROUP_STARTS = OverloadTable(
        mark_group_starts,
        {d: [wp.array[d], wp.int32, wp.int32, wp.array[wp.bool]] for d in _KEY_DTYPES},
    )
    HASH_INSERT = OverloadTable(
        hash_insert,
        {
            d: [wp.array[d], wp.array[d], wp.array[wp.int32], wp.int32, wp.array[wp.int32]]
            for d in _TABLE_DTYPES
        },
    )
    # ``slot_key`` and ``out_keys`` carry the key dtype; every count/offset buffer is int32.
    COMPACT_FROM_TABLE = OverloadTable(
        compact_from_table,
        {
            d: [
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.int32],
            ]
            for d in _TABLE_DTYPES
        },
    )


_register_overloads()
