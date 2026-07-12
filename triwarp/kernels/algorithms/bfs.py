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
  ``min_count``) — the geodesic-ball query.
"""

import warp as wp

from triwarp.constants import INT64_MAX_CONSTANT
from triwarp.kernels import unique as kernel_unique

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
    slot = kernel_unique.hash_slot(value, mask)
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
        slot = kernel_unique.next_slot(slot, mask)


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
    farthest = wp.int32(0)
    farthest_distance = ext_dist[0]
    for k in range(1, cap):
        if ext_dist[k] > farthest_distance:
            farthest_distance = ext_dist[k]
            farthest = k
    if distance < farthest_distance:
        ext_dist[farthest] = distance
        ext_idx[farthest] = neighbor
    return count


@wp.func
def bfs_extras_pop_nearest(
    ext_dist: wp.array[wp.float32], ext_idx: wp.array[wp.int32], count: wp.int32
) -> tuple[wp.int32, wp.int32]:
    """Remove and return the nearest candidate (swap-remove); caller ensures ``count > 0``."""
    best = wp.int32(0)
    best_distance = ext_dist[0]
    for k in range(1, count):
        if ext_dist[k] < best_distance:
            best_distance = ext_dist[k]
            best = k
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
    write: wp.bool,
    base: wp.int32,
    out_flat: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> wp.int32:
    """
    BFS over the CSR edge graph from source ``i``; returns the collected count.

    Traverses ``adj_offsets``/``adj_columns`` with a FIFO ``queue`` (whose emitted prefix is the
    discovery order) and an O(1) open-addressing ``visited`` hash row — power-of-two length,
    pre-filled with ``-1`` by the caller before the launch. When ``radius`` is finite the
    traversal is *geodesic* (libigl ``getSphere``): a neighbor is enqueued only when within
    Euclidean ``radius`` of the center, and out-of-ball neighbors feed a nearest fallback
    (``ext_dist``/``ext_idx``) drained to ``min_count``. When ``radius`` is ``+inf`` the
    geometric predicate is disabled (``vertices`` is never read, so a length-1 placeholder is
    fine); pass ``min_count = 0`` so the fallback never engages. When ``write`` is true,
    collected vertices are emitted to ``out_flat[base + pos]`` in BFS order. Exceeding the queue
    or visited capacity increments ``out_overflow``.
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
        if write:
            out_flat[base + collected] = current
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

    while ext_n > wp.int32(0) and collected < min_count:
        cand = wp.int32(0)
        cand, ext_n = bfs_extras_pop_nearest(ext_dist, ext_idx, ext_n)

        if write:
            out_flat[base + collected] = cand
        collected += wp.int32(1)

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


@wp.kernel
def bfs_expand(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    frontier: wp.array[wp.int32],
    node_count: wp.int64,
    dist: wp.array[wp.int32],
    claim_key: wp.array[wp.int64],
    out_candidates: wp.array[wp.int32],
    out_candidate_count: wp.array[wp.int32],
) -> None:
    # Level-synchronous expansion reproducing scipy's FIFO discovery order: thread r (the
    # dequeue rank of its frontier node) claims each unvisited neighbor v with the composite
    # key r * n + v. atomic_min keeps the earliest-dequeued parent; only the first claimer
    # (sentinel seen) appends v to the candidate list, so each node appears exactly once.
    r = int(wp.tid())
    u = frontier[r]
    for k in range(adj_offsets[u], adj_offsets[u + 1]):
        v = adj_columns[k]
        if dist[v] == wp.int32(-1):
            key = wp.int64(r) * node_count + wp.int64(v)
            previous = wp.atomic_min(claim_key, v, key)
            if previous == INT64_MAX_CONSTANT:
                slot = wp.atomic_add(out_candidate_count, 0, 1)
                out_candidates[slot] = v


@wp.kernel
def bfs_gather_claim_keys(
    candidates: wp.array[wp.int32], claim_key: wp.array[wp.int64], out_keys: wp.array[wp.int64]
) -> None:
    i = int(wp.tid())
    out_keys[i] = claim_key[candidates[i]]


@wp.kernel
def bfs_finalize_level(
    frontier_sorted: wp.array[wp.int32],
    claim_key: wp.array[wp.int64],
    prev_frontier: wp.array[wp.int32],
    node_count: wp.int64,
    level: wp.int32,
    order_base: wp.int32,
    out_order: wp.array[wp.int32],
    out_parent: wp.array[wp.int32],
    out_dist: wp.array[wp.int32],
) -> None:
    # Candidates sorted by claim key = (parent dequeue rank, ascending node id) = exact scipy
    # FIFO discovery order given ascending CSR columns; the min-rank claimer is scipy's
    # first-discoverer predecessor.
    i = int(wp.tid())
    v = frontier_sorted[i]
    rank = wp.int32((claim_key[v] - wp.int64(v)) / node_count)
    out_dist[v] = level
    out_parent[v] = prev_frontier[rank]
    out_order[order_base + i] = v
