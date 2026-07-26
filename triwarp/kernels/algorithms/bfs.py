"""
Breadth-first search over a CSR adjacency graph.

Two traversal cores:

- :func:`single_source_bfs` — one Warp thread walks the whole graph from a single source using a
  dense ``dist`` array as the visited marker (``dist[node] == -1`` ⇒ unvisited). Run at ``dim=1`` it
  reproduces ``scipy.sparse.csgraph.breadth_first_order`` order/parents/distances exactly when the
  CSR columns are sorted ascending (as produced by :func:`triwarp.graph.edges_to_csr`).
- :func:`per_source_bfs_collect` — one thread per source over caller-provided global-memory
  scratch rows (a FIFO queue whose emitted prefix is the discovery order, an open-addressing
  visited hash set, and a small nearest-fallback pool). With a finite ``radius`` it enqueues only
  neighbors within ``radius`` of the center (and backfills the nearest out-of-ball vertices up to
  ``min_count``) — the geodesic-ball query. Its nearest-fallback pool scans with the shared
  ``wp.ref`` argmin/argmax helpers, so any kernel calling it must be decorated
  ``@wp.kernel(enable_backward=False)`` (see :mod:`triwarp.kernels.array`).
"""

import warp as wp

from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels.array import update_argmax, update_argmin

# Per-source scratch capacities (rows of the wrapper-allocated global-memory pools).
# ``_PER_SOURCE_MAX_NEIGHBORS`` caps the queue — and therefore the collected set — as before;
# the visited hash row is power-of-two sized with a 3/4 load-factor fill bound; the extras pool
# only needs to hold the nearest out-of-ball frontier for the ``min_count`` backfill.
_PER_SOURCE_MAX_NEIGHBORS = 512
_VISITED_HASH_CAPACITY = 1024
_VISITED_MAX_FILL = 768
_EXTRAS_CAPACITY = 64


@wp.func
def bfs_visited_insert(
    visited: wp.array[wp.int32],
    mask: wp.int32,
    value: wp.int32,
    count: wp.int32,
    out_overflow: wp.array[wp.int32],
) -> tuple[wp.bool, wp.int32]:
    """
    Insert ``value`` into the open-addressing ``visited`` row (empty slots hold ``-1``).

    Returns ``(is_new, new_count)``: ``is_new`` is ``False`` when the value was already present.
    Beyond the load-factor fill bound the insert is dropped (counted in ``out_overflow``) and the
    value reads as new, mirroring the old full-buffer behavior where dropped nodes could be
    revisited.
    """
    slot = kernel_grouping.hash_slot(value, mask)
    while True:
        stored = visited[slot]
        if stored == value:
            return False, count
        if stored == wp.int32(-1):
            if count >= _VISITED_MAX_FILL:
                wp.atomic_add(out_overflow, 0, 1)
                return True, count
            visited[slot] = value
            return True, count + 1
        slot = kernel_grouping.next_slot(slot, mask)


@wp.func
def bfs_extras_push_nearest(
    ext_dist: wp.array[wp.float32],
    ext_idx: wp.array[wp.int32],
    distance: wp.float32,
    neighbor: wp.int32,
    count: wp.int32,
) -> wp.int32:
    """Keep the ``cap`` nearest candidates: append, or replace the farthest kept one."""
    cap = ext_dist.shape[0]
    if count < cap:
        ext_dist[count] = distance
        ext_idx[count] = neighbor
        return count + 1
    farthest = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    farthest_distance = ext_dist[0]
    for k in range(1, cap):
        update_argmax(farthest_distance, farthest, ext_dist[k], k)
    if distance < farthest_distance:
        ext_dist[farthest] = distance
        ext_idx[farthest] = neighbor
    return count


@wp.func
def bfs_extras_pop_nearest(
    ext_dist: wp.array[wp.float32], ext_idx: wp.array[wp.int32], count: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Remove and return the nearest candidate (swap-remove); caller ensures ``count > 0``."""
    best = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    best_distance = ext_dist[0]
    for k in range(1, count):
        update_argmin(best_distance, best, ext_dist[k], k)
    nearest = ext_idx[best]
    last = count - 1
    ext_dist[best] = ext_dist[last]
    ext_idx[best] = ext_idx[last]
    return nearest, last


@wp.func
def per_source_bfs_collect(
    i: wp.int32,
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    queue: wp.array[wp.int32],
    visited: wp.array[wp.int32],
    ext_dist: wp.array[wp.float32],
    ext_idx: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> wp.int32:
    """
    BFS over the CSR edge graph from source ``i``; returns the collected count.

    Traverses ``adj_offsets``/``adj_columns`` with a FIFO ``queue`` and an O(1) open-addressing
    ``visited`` hash row — power-of-two length, pre-filled with ``-1`` by the caller before the
    launch. When ``radius`` is finite the traversal is *geodesic* (libigl ``getSphere``): a
    neighbor is enqueued only when within Euclidean ``radius`` of the center, and out-of-ball
    neighbors feed a nearest fallback (``ext_dist``/``ext_idx``) drained to ``min_count``. When
    ``radius`` is ``+inf`` the geometric predicate is disabled (``vertices`` is never read, so a
    length-1 placeholder is fine); pass ``min_count = 0`` so the fallback never engages.

    On return the collected set *is* ``queue[:count]`` in BFS-then-backfill order (drained
    fallback candidates are appended to the queue), so ``count == q_tail <= queue capacity``
    always and the caller gathers results straight from its queue row — no second traversal.
    Exceeding the queue or visited capacity increments ``out_overflow`` and drops the surplus.
    """
    use_geometry = not wp.isinf(radius)

    queue_cap = queue.shape[0]
    mask = visited.shape[0] - 1

    center = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_geometry:
        center = vertices[i]

    visited_n = wp.int32(0)
    is_new = wp.bool(True)
    is_new, visited_n = bfs_visited_insert(visited, mask, i, visited_n, out_overflow)
    queue[0] = i
    q_head = wp.int32(0)
    q_tail = wp.int32(1)
    ext_n = wp.int32(0)
    collected = wp.int32(0)

    while q_head < q_tail:
        current = queue[q_head]
        q_head += wp.int32(1)
        collected += wp.int32(1)

        start = adj_offsets[current]
        end = adj_offsets[current + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            is_new, visited_n = bfs_visited_insert(visited, mask, neighbor, visited_n, out_overflow)
            if not is_new:
                continue
            distance = wp.float32(0.0)
            if use_geometry:
                distance = wp.length(vertices[neighbor] - center)
            if distance < radius:
                if q_tail < queue_cap:
                    queue[q_tail] = neighbor
                    q_tail += wp.int32(1)
                else:
                    wp.atomic_add(out_overflow, 0, 1)
            elif collected < min_count:
                ext_n = bfs_extras_push_nearest(ext_dist, ext_idx, distance, neighbor, ext_n)

    # Drained candidates are appended to the queue (not re-expanded from it) so the queue prefix
    # stays the complete collected set. The main loop exits with collected == q_tail, so with the
    # default min_count << queue capacity the drain never sees a full queue; only a pathological
    # min_count > capacity can overflow here, dropping the surplus like the main phase does.
    while ext_n > wp.int32(0) and collected < min_count:
        cand = wp.int32(0)
        cand, ext_n = bfs_extras_pop_nearest(ext_dist, ext_idx, ext_n)

        if q_tail < queue_cap:
            queue[q_tail] = cand
            q_tail += wp.int32(1)
            collected += wp.int32(1)
        else:
            wp.atomic_add(out_overflow, 0, 1)
            continue

        start = adj_offsets[cand]
        end = adj_offsets[cand + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            is_new, visited_n = bfs_visited_insert(visited, mask, neighbor, visited_n, out_overflow)
            if not is_new:
                continue
            distance = wp.float32(0.0)
            if use_geometry:
                distance = wp.length(vertices[neighbor] - center)
            ext_n = bfs_extras_push_nearest(ext_dist, ext_idx, distance, neighbor, ext_n)

    return collected


@wp.func
def single_source_bfs(
    source: wp.int32,
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_dist: wp.array[wp.int32],
) -> wp.int32:
    """
    Run a serial single-source BFS over the CSR edge graph; return the reachable-node count.

    ``out_dist`` doubles as the visited marker (caller pre-fills it and ``out_parent`` with ``-1``).
    ``out_order`` (size N) also serves as the FIFO: in serial BFS its prefix *is* the queue, so the
    first ``count`` entries are the discovery order. Each node is enqueued at most once, so the
    size-N buffers never overflow. Visiting neighbors in CSR (ascending) column order reproduces
    ``scipy.sparse.csgraph.breadth_first_order``.
    """
    head = wp.int32(0)
    tail = wp.int32(1)
    out_dist[source] = wp.int32(0)
    out_order[0] = source

    current = wp.int32(0)
    neighbor = wp.int32(0)
    k = wp.int32(0)
    while head < tail:
        current = out_order[head]
        head += wp.int32(1)
        start = adj_offsets[current]
        end = adj_offsets[current + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            if out_dist[neighbor] == wp.int32(-1):
                out_dist[neighbor] = out_dist[current] + wp.int32(1)
                out_parent[neighbor] = current
                out_order[tail] = neighbor
                tail += wp.int32(1)
    return tail


@wp.kernel
def single_source_bfs_kernel(
    source: wp.int32,
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_order: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_dist: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # Launched at dim=1: a single thread runs the whole serial traversal for exact-order match.
    _ = int(wp.tid())
    out_count[0] = single_source_bfs(
        source, adj_offsets, adj_columns, out_order, out_parent, out_dist
    )


@wp.kernel
def bfs_seed(source: wp.int32, out_order: wp.array[wp.int32], out_dist: wp.array[wp.int32]) -> None:
    # dim=1: place the source at order slot 0 with distance 0.
    _ = int(wp.tid())
    out_order[0] = source
    out_dist[source] = wp.int32(0)


# The level-synchronous frontier loop runs entirely on device (``wp.capture_while``): the
# frontier bounds live in a 4-int32 ``state`` array [frontier start, frontier end, level, loop
# condition], every kernel is launched at a fixed ``dim=node_count`` with an early-exit guard on
# the device-side frontier size, and the per-level prefix sum is a capture-safe fixed-buffer
# scan (``wp.utils.array_scan`` allocates CUB temp storage internally, which conditional graph
# bodies reject). ``BFS_SCAN_BLOCK`` also sizes the wrapper's padded scan buffers.
_STATE_START = wp.constant(wp.int32(0))
_STATE_TAIL = wp.constant(wp.int32(1))
_STATE_LEVEL = wp.constant(wp.int32(2))
_STATE_COND = wp.constant(wp.int32(3))
BFS_SCAN_BLOCK = 256


@wp.kernel
def bfs_expand_claim(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    order: wp.array[wp.int32],
    state: wp.array[wp.int32],
    dist: wp.array[wp.int32],
    out_claim_rank: wp.array[wp.int32],
) -> None:
    # Level-synchronous expansion: thread r (the dequeue rank of its frontier node) claims each
    # unvisited neighbor v via atomic_min on the rank — the earliest-dequeued parent wins,
    # matching scipy's first-discoverer predecessor. ``out_claim_rank`` is initialized to
    # INT32_MAX once and never reset: a node claimed at level L is always finalized at level L
    # (its dist is set by the scatter), so the ``dist[v] == -1`` guard in the count/scatter
    # kernels rejects any stale rank from an earlier level.
    r = int(wp.tid())
    if r >= state[_STATE_TAIL] - state[_STATE_START]:
        return
    u = order[state[_STATE_START] + r]
    for k in range(adj_offsets[u], adj_offsets[u + 1]):
        v = adj_columns[k]
        if dist[v] == wp.int32(-1):
            wp.atomic_min(out_claim_rank, v, wp.int32(r))


@wp.kernel
def bfs_count_claims(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    order: wp.array[wp.int32],
    state: wp.array[wp.int32],
    dist: wp.array[wp.int32],
    claim_rank: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Thread r counts the neighbors it owns this level: still unvisited (stale-rank guard, see
    # bfs_expand_claim) and won by rank r — the atomic_min winner is unique per node. Stale
    # ``out_counts`` entries beyond the frontier size are harmless: an inclusive scan's prefix
    # only depends on the prefix, and offsets past the frontier are never read.
    r = int(wp.tid())
    if r >= state[_STATE_TAIL] - state[_STATE_START]:
        return
    u = order[state[_STATE_START] + r]
    count = wp.int32(0)
    for k in range(adj_offsets[u], adj_offsets[u + 1]):
        v = adj_columns[k]
        if dist[v] == wp.int32(-1) and claim_rank[v] == wp.int32(r):
            count += wp.int32(1)
    out_counts[r] = count


@wp.kernel
def bfs_scan_blocks(
    values: wp.array[wp.int32], out_scanned: wp.array[wp.int32], out_block_sums: wp.array[wp.int32]
) -> None:
    # Capture-safe scan, pass 1: per-block inclusive scan (arrays are padded to a multiple of
    # BFS_SCAN_BLOCK; padding garbage never reaches a read prefix). Thread 0 exports the block
    # total for pass 2.
    i, t = wp.tid()
    offset = i * BFS_SCAN_BLOCK
    tile = wp.tile_load(values, shape=BFS_SCAN_BLOCK, offset=offset, storage="register")
    scanned = wp.tile_scan_inclusive(tile)
    wp.tile_store(out_scanned, scanned, offset=offset)
    if t == 0:
        out_block_sums[i] = scanned[BFS_SCAN_BLOCK - 1]


@wp.kernel
def bfs_scan_block_sums(out_block_sums: wp.array[wp.int32]) -> None:
    # Capture-safe scan, pass 2 (dim=1): serial inclusive scan of the few thousand block sums.
    _ = int(wp.tid())
    n = out_block_sums.shape[0]
    total = wp.int32(0)
    for k in range(n):
        total += out_block_sums[k]
        out_block_sums[k] = total


@wp.kernel
def bfs_add_block_offsets(block_sums: wp.array[wp.int32], out_scanned: wp.array[wp.int32]) -> None:
    # Capture-safe scan, pass 3: add the preceding blocks' total to each element.
    idx = int(wp.tid())
    b = wp.int32(idx) / wp.int32(BFS_SCAN_BLOCK)
    if b > wp.int32(0):
        out_scanned[idx] += block_sums[b - 1]


@wp.kernel
def bfs_scatter_claims(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    state: wp.array[wp.int32],
    claim_rank: wp.array[wp.int32],
    offsets_inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_dist: wp.array[wp.int32],
) -> None:
    # Emits the level in exact scipy FIFO order without a sort: segments are rank-major (thread
    # r's segment starts at the exclusive-scan value offsets_inclusive[r - 1]) and each segment
    # fills in ascending CSR column order — (parent dequeue rank, ascending node id), exactly
    # the order the former int64 claim-key radix sort produced. ``order`` is read as the current
    # frontier and written with the next level's nodes. ``out_dist`` doubles as the visited
    # marker read by the ownership guard: only the unique rank winner writes dist[v], so its
    # racy read by non-owners cannot flip their guard (they fail claim_rank[v] == r).
    r = int(wp.tid())
    start = state[_STATE_START]
    tail = state[_STATE_TAIL]
    if r >= tail - start:
        return
    u = order[start + r]
    level = state[_STATE_LEVEL]
    slot = tail
    if r > 0:
        slot += offsets_inclusive[r - 1]
    for k in range(adj_offsets[u], adj_offsets[u + 1]):
        v = adj_columns[k]
        if out_dist[v] == wp.int32(-1) and claim_rank[v] == wp.int32(r):
            order[slot] = v
            out_parent[v] = u
            out_dist[v] = level
            slot += wp.int32(1)


@wp.kernel
def bfs_update_state(offsets_inclusive: wp.array[wp.int32], out_state: wp.array[wp.int32]) -> None:
    # dim=1, last op of each level: advance the frontier window to the freshly scattered
    # segment, bump the level, and keep looping while the level emitted anything.
    _ = int(wp.tid())
    frontier_size = out_state[_STATE_TAIL] - out_state[_STATE_START]
    count = wp.int32(0)
    if frontier_size > 0:
        count = offsets_inclusive[frontier_size - 1]
    out_state[_STATE_START] = out_state[_STATE_TAIL]
    out_state[_STATE_TAIL] = out_state[_STATE_TAIL] + count
    out_state[_STATE_LEVEL] = out_state[_STATE_LEVEL] + wp.int32(1)
    out_state[_STATE_COND] = wp.where(count > wp.int32(0), wp.int32(1), wp.int32(0))
