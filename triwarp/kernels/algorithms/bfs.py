"""
Breadth-first search over a CSR adjacency graph.

Two traversal cores share the leaf helpers in :mod:`triwarp.kernels.array`:

- :func:`single_source_bfs` — one Warp thread walks the whole graph from a single source using a
  dense ``dist`` array as the visited marker (``dist[node] == -1`` ⇒ unvisited). Run at ``dim=1`` it
  reproduces ``scipy.sparse.csgraph.breadth_first_order`` order/parents/distances exactly when the
  CSR columns are sorted ascending (as produced by :func:`triwarp.graph.edges_to_csr`).
- :func:`per_source_bfs_collect` — one thread per source, fixed-capacity per-thread sorted
  ``visited`` scratch. Relocated from the geodesic-ball query, it keeps an optional geometric
  predicate: with a
  finite ``radius`` it enqueues only neighbors within ``radius`` of the center (and backfills the
  nearest out-of-ball vertices up to ``min_count``); with ``radius = +inf`` (and ``min_count = 0``)
  the predicate is disabled and it becomes a pure topological reachable-set BFS, which the
  multi-source kernels here use.
"""

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, INT32_MAX_CONSTANT
from triwarp.kernels import array as kernel_array

# Fixed per-thread scratch capacity for the per-source BFS (queue, visited, and extras buffers).
# Local kernel arrays need a compile-time-constant shape; a source whose reachable set exceeds this
# is clamped and the wrapper warns. 512 comfortably covers observed neighborhoods (~272 on a folded
# half-torus).
_PER_SOURCE_MAX_NEIGHBORS = 512


@wp.func
def bfs_sorted_insert_unique(
    arr: wp.array[wp.int32], value: wp.int32, count: wp.int32, out_overflow: wp.array[wp.int32]
) -> wp.int32:
    """Insert ``value`` into the sorted prefix of ``arr`` (tail filled with max-int sentinels)."""
    if count >= arr.shape[0]:
        wp.atomic_add(out_overflow, 0, 1)
        return count
    slot = kernel_array.binary_search_index(arr, value)
    kernel_array.array_shift_insert(arr, value, slot)
    return count + 1


@wp.func
def _bfs_extras_push(
    ext_dist: wp.array[wp.float32],
    ext_idx: wp.array[wp.int32],
    distance: wp.float32,
    neighbor: wp.int32,
    count: wp.int32,
) -> wp.int32:
    """Insert ``(distance, neighbor)`` keeping ``ext_dist`` ascending (distance-only ordering)."""
    cap = ext_dist.shape[0]
    slot = kernel_array.binary_search_index(ext_dist, distance)
    if slot >= cap:
        return count  # farther than every kept extra and the buffer is full — drop it
    kernel_array.array_shift_insert(ext_dist, distance, slot)
    kernel_array.array_shift_insert(ext_idx, neighbor, slot)
    if count < cap:
        return count + 1
    return count


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

    Traverses ``adj_offsets``/``adj_columns`` (FIFO ``queue`` + sorted ``visited`` with binary
    search). When ``radius`` is finite the traversal is *geodesic* (libigl ``getSphere``): a
    neighbor is enqueued only when within Euclidean ``radius`` of ``vertices[i]``, and out-of-ball
    neighbors feed a nearest fallback (``ext_dist``/``ext_idx``) drained to ``min_count``. When
    ``radius`` is ``+inf`` the geometric predicate is disabled (``vertices`` is never read, so a
    length-1 placeholder is fine) and this is a pure topological reachable-set BFS; pass
    ``min_count = 0`` so the fallback never engages. When ``write`` is true, collected vertices are
    emitted to ``out_flat[base + pos]`` in BFS order. ``queue``, ``visited`` and the extras buffers
    are caller-allocated fixed-capacity scratch; exceeding capacity increments ``out_overflow``.
    """
    use_geometry = not wp.isinf(radius)

    visited_cap = visited.shape[0]
    queue_cap = queue.shape[0]
    extras_cap = ext_dist.shape[0]

    # Fill the unused tail with the largest representable values so real entries always sort before
    # them and ``binary_search_index`` holds across the whole buffer (sentinels shift off the end).
    for k in range(visited_cap):
        visited[k] = INT32_MAX_CONSTANT
    for k in range(extras_cap):
        ext_dist[k] = FLOAT32_INF_CONSTANT
        ext_idx[k] = wp.int32(-1)

    center = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_geometry:
        center = vertices[i]

    visited[0] = i
    visited_n = wp.int32(1)
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
            if kernel_array.binary_search_sorted_contains(visited, neighbor):
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
                ext_n = _bfs_extras_push(ext_dist, ext_idx, distance, neighbor, ext_n)
            visited_n = bfs_sorted_insert_unique(visited, neighbor, visited_n, out_overflow)

    while ext_n > wp.int32(0) and collected < min_count:
        cand = ext_idx[0]
        for k in range(extras_cap - 1):
            ext_dist[k] = ext_dist[k + 1]
            ext_idx[k] = ext_idx[k + 1]
        ext_dist[extras_cap - 1] = FLOAT32_INF_CONSTANT
        ext_idx[extras_cap - 1] = wp.int32(-1)
        ext_n -= wp.int32(1)

        if write:
            out_flat[base + collected] = cand
        collected += wp.int32(1)

        start = adj_offsets[cand]
        end = adj_offsets[cand + 1]
        for k in range(start, end):
            neighbor = adj_columns[k]
            if kernel_array.binary_search_sorted_contains(visited, neighbor):
                continue
            distance = wp.float32(0.0)
            if use_geometry:
                distance = wp.length(vertices[neighbor] - center)
            ext_n = _bfs_extras_push(ext_dist, ext_idx, distance, neighbor, ext_n)
            visited_n = bfs_sorted_insert_unique(visited, neighbor, visited_n, out_overflow)

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
def multi_source_bfs_count(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    t = int(wp.tid())
    queue = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    vertices = wp.zeros(shape=1, dtype=wp.vec3)
    dummy = wp.zeros(shape=1, dtype=wp.int32)
    out_counts[t] = per_source_bfs_collect(
        sources[t],
        vertices,
        adj_offsets,
        adj_columns,
        FLOAT32_INF_CONSTANT,
        wp.int32(0),
        queue,
        visited,
        ext_dist,
        ext_idx,
        False,
        wp.int32(0),
        dummy,
        out_overflow,
    )


@wp.kernel
def multi_source_bfs_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_neighbors: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    t = int(wp.tid())
    queue = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=_PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    vertices = wp.zeros(shape=1, dtype=wp.vec3)
    per_source_bfs_collect(
        sources[t],
        vertices,
        adj_offsets,
        adj_columns,
        FLOAT32_INF_CONSTANT,
        wp.int32(0),
        queue,
        visited,
        ext_dist,
        ext_idx,
        True,
        offsets[t],
        out_neighbors,
        out_overflow,
    )
