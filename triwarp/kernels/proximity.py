import warp as wp

from triwarp.constants import (
    FLOAT32_INF_CONSTANT,
    INT32_MAX_CONSTANT,
    TOLERANCE_MERGE_CONSTANT,
    TOLERANCE_PLANAR_CONSTANT,
)
from triwarp.kernels import array as kernel_array


@wp.kernel
def aabb_bounds(
    points: wp.array[wp.vec3], out_min: wp.array[wp.float32], out_max: wp.array[wp.float32]
) -> None:
    tid = wp.tid()
    p = points[tid]

    # Atomic operations for component-wise reduction
    wp.atomic_min(out_min, 0, p[0])
    wp.atomic_min(out_min, 1, p[1])
    wp.atomic_min(out_min, 2, p[2])

    wp.atomic_max(out_max, 0, p[0])
    wp.atomic_max(out_max, 1, p[1])
    wp.atomic_max(out_max, 2, p[2])


@wp.kernel
def query_bvh_aabb_count(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    h = half_extent
    lower = wp.vec3(q[0] - h, q[1] - h, q[2] - h)
    upper = wp.vec3(q[0] + h, q[1] + h, q[2] + h)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        c = c + 1
    out_counts[tid] = c


@wp.kernel
def query_bvh_aabb_neighbors(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    h = half_extent
    lower = wp.vec3(q[0] - h, q[1] - h, q[2] - h)
    upper = wp.vec3(q[0] + h, q[1] + h, q[2] + h)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    primitive_idx = wp.int32(0)
    w = int(offsets[tid])
    while wp.bvh_query_next(query, primitive_idx):
        out_indices[w] = primitive_idx
        w = w + 1


@wp.kernel
def query_mesh_aabb_bounds_count(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    lower = query_lower[tid]
    upper = query_upper[tid]
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_idx = wp.int32(0)
    c = wp.int32(0)
    max_hits_i = int(max_hits)
    while wp.mesh_query_aabb_next(query, face_idx) and c < max_hits_i:
        c = c + 1
    out_counts[tid] = c


@wp.kernel
def query_mesh_aabb_bounds_neighbors(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    lower = query_lower[tid]
    upper = query_upper[tid]
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_idx = wp.int32(0)
    w = int(offsets[tid])
    max_hits_i = int(max_hits)
    hits = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face_idx) and hits < max_hits_i:
        out_indices[w] = face_idx
        w = w + 1
        hits = hits + 1


@wp.kernel
def query_hashgrid_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    r = radius
    query = wp.hash_grid_query(grid_id, q, r)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        if wp.length(points[j] - q) <= r:
            c = c + 1
    out_neighbor_counts[tid] = c


@wp.kernel
def query_hashgrid_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    r = radius
    query = wp.hash_grid_query(grid_id, q, r)
    point_idx = wp.int32(0)
    w = int(offsets[tid])
    while wp.hash_grid_query_next(query, point_idx):
        d = wp.length(points[point_idx] - q)
        if d <= r:
            out_indices[w] = point_idx
            out_distances[w] = d
            w = w + 1


@wp.kernel
def query_bvh_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    r = radius
    lower = wp.vec3(q[0] - r, q[1] - r, q[2] - r)
    upper = wp.vec3(q[0] + r, q[1] + r, q[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        if wp.length(points[j] - q) <= r:
            c = c + 1
    out_neighbor_counts[tid] = c


@wp.kernel
def query_bvh_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    r = radius
    lower = wp.vec3(q[0] - r, q[1] - r, q[2] - r)
    upper = wp.vec3(q[0] + r, q[1] + r, q[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    point_idx = wp.int32(0)
    w = int(offsets[tid])
    while wp.bvh_query_next(query, point_idx):
        d = wp.length(points[point_idx] - q)
        if d <= r:
            out_indices[w] = point_idx
            out_distances[w] = d
            w = w + 1


@wp.kernel
def geodesic_ball_reference_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_reference: wp.array[wp.int32],
) -> None:
    """Lowest-indexed edge neighbor per vertex (libigl ``adjacency_list[i][0]``); self if alone."""
    i = int(wp.tid())
    start = int(adj_offsets[i])
    end = int(adj_offsets[i + 1])
    if start == end:
        out_reference[i] = i
        return
    minimum = adj_columns[start]
    for k in range(start + 1, end):
        if adj_columns[k] < minimum:
            minimum = adj_columns[k]
    out_reference[i] = minimum


@wp.func
def _geodesic_sorted_insert_unique(
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
def _geodesic_extras_push(
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
def _geodesic_ball_collect(
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
    Geodesic-ball BFS for vertex ``i`` (libigl ``getSphere``); returns the collected count.

    Traverses the mesh edge graph (CSR ``adj_offsets``/``adj_columns``), enqueueing a neighbor only
    when it lies within Euclidean ``radius`` of vertex ``i``. Out-of-ball neighbors feed a nearest
    fallback (``ext_dist``/``ext_idx``) drained until ``min_count`` is reached. When ``write`` is
    true, collected vertices are emitted to ``out_flat[base + pos]`` in BFS order. ``queue``,
    ``visited`` and the extras buffers are caller-allocated fixed-capacity scratch sized to the
    neighbor limit; exceeding it increments ``out_overflow`` so the caller can warn.
    """
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
            distance = wp.length(vertices[neighbor] - center)
            if distance < radius:
                if q_tail < queue_cap:
                    queue[q_tail] = neighbor
                    q_tail += wp.int32(1)
                else:
                    wp.atomic_add(out_overflow, 0, 1)
            elif collected < min_count:
                ext_n = _geodesic_extras_push(ext_dist, ext_idx, distance, neighbor, ext_n)
            visited_n = _geodesic_sorted_insert_unique(visited, neighbor, visited_n, out_overflow)

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
            distance = wp.length(vertices[neighbor] - center)
            ext_n = _geodesic_extras_push(ext_dist, ext_idx, distance, neighbor, ext_n)
            visited_n = _geodesic_sorted_insert_unique(visited, neighbor, visited_n, out_overflow)

    return collected


# Fixed per-thread scratch capacity for the BFS (queue, visited, and extras buffers). Local kernel
# arrays need a compile-time-constant shape; a vertex whose neighborhood exceeds this is clamped and
# the wrapper warns. 512 comfortably covers observed neighborhoods (~272 on a folded half-torus).
_GEODESIC_MAX_NEIGHBORS = 512


@wp.kernel
def query_geodesic_ball_count(
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    out_counts: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    queue = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    dummy = wp.zeros(shape=1, dtype=wp.int32)
    out_counts[i] = _geodesic_ball_collect(
        i,
        vertices,
        adj_offsets,
        adj_columns,
        radius,
        min_count,
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
def query_geodesic_ball_neighbors(
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    offsets: wp.array[wp.int32],
    out_neighbors: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    queue = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=_GEODESIC_MAX_NEIGHBORS, dtype=wp.int32)
    _geodesic_ball_collect(
        i,
        vertices,
        adj_offsets,
        adj_columns,
        radius,
        min_count,
        queue,
        visited,
        ext_dist,
        ext_idx,
        True,
        offsets[i],
        out_neighbors,
        out_overflow,
    )


@wp.kernel
def query_bvh_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    k: wp.int32,
    radius: wp.float32,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    r = radius
    lower = wp.vec3(q[0] - r, q[1] - r, q[2] - r)
    upper = wp.vec3(q[0] + r, q[1] + r, q[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    point_index = wp.int32(0)

    while wp.bvh_query_next(query, point_index):
        d = wp.length(points[point_index] - q)

        if d > radius:
            continue

        if d >= out_distances[tid, k - 1]:
            continue

        if k == 1:
            out_indices[tid, 0] = point_index
            out_distances[tid, 0] = d
            continue

        slot = kernel_array.binary_search_index(out_distances[tid], d)
        kernel_array.array_shift_insert(out_distances[tid], d, slot)
        kernel_array.array_shift_insert(out_indices[tid], point_index, slot)


@wp.kernel
def query_hashgrid_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    k: wp.int32,
    radius: wp.float32,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]

    query = wp.hash_grid_query(grid_id, q, radius)
    point_index = wp.int32(-1)

    while wp.hash_grid_query_next(query, point_index):
        d = wp.length(points[point_index] - q)

        if d > radius:
            continue

        if d >= out_distances[tid, k - 1]:
            continue

        if k == 1:
            out_indices[tid, 0] = point_index
            out_distances[tid, 0] = d
            continue

        slot = kernel_array.binary_search_index(out_distances[tid], d)
        kernel_array.array_shift_insert(out_distances[tid], d, slot)
        kernel_array.array_shift_insert(out_indices[tid], point_index, slot)


@wp.kernel
def closest_point_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_face: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out_closest[tid] = closest
        out_distance[tid] = wp.length(p - closest)
        out_face[tid] = query.face
    else:
        out_closest[tid] = p
        out_distance[tid] = max_dist
        out_face[tid] = wp.int32(-1)


@wp.func
def point_strictly_inside_aabb(p: wp.vec3, mesh_min: wp.vec3, mesh_max: wp.vec3) -> bool:
    return (
        p[0] > mesh_min[0]
        and p[1] > mesh_min[1]
        and p[2] > mesh_min[2]
        and p[0] < mesh_max[0]
        and p[1] < mesh_max[1]
        and p[2] < mesh_max[2]
    )


@wp.kernel
def contains_points_sign_parity(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    mesh_min: wp.vec3,
    mesh_max: wp.vec3,
    out_contains: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    p = points[tid]

    if not point_strictly_inside_aabb(p, mesh_min, mesh_max):
        out_contains[tid] = False
        return

    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_contains[tid] = query.result and query.sign < wp.float32(0.0)


@wp.kernel
def signed_distance_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    out_distance: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    if not query.result:
        out_distance[tid] = max_dist
        return
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    dist = wp.length(p - closest)
    if dist <= TOLERANCE_MERGE_CONSTANT:
        out_distance[tid] = dist
    else:
        out_distance[tid] = query.sign * dist


@wp.kernel
def init_sphere_radii(
    mesh_vertices: wp.array[wp.vec3],
    n_vertices: wp.int32,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.float32],
    out_radii: wp.array[wp.float32],
    out_not_converged: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    p = points[tid]
    n = normals[tid]
    d = distances[tid]

    if not wp.isinf(d):
        out_radii[tid] = d * wp.float32(0.5)
        out_not_converged[tid] = True
        return

    max_proj = wp.float32(-1e38)
    best_v = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    found = wp.bool(False)

    for v_idx in range(n_vertices):
        v = mesh_vertices[v_idx]
        proj = wp.dot(v - p, n)
        if proj > max_proj:
            max_proj = proj
            best_v = v
            found = True

    if not found or max_proj < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        return

    diff = best_v - p
    denom = wp.float32(2.0) * wp.dot(diff, n)
    if wp.abs(denom) < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        return

    out_radii[tid] = wp.dot(diff, diff) / denom
    out_not_converged[tid] = True


@wp.kernel
def compute_sphere_centers(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    radii: wp.array[wp.float32],
    out_centers: wp.array[wp.vec3],
) -> None:
    tid = wp.tid()
    r = radii[tid]
    if wp.isinf(r) or wp.isnan(r):
        out_centers[tid] = wp.vec3(wp.nan, wp.nan, wp.nan)
    else:
        out_centers[tid] = points[tid] + normals[tid] * r


@wp.kernel
def step_sphere_shrink(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    n_points: wp.array[wp.vec3],
    n_dists: wp.array[wp.float32],
    centers: wp.array[wp.vec3],
    old_radii: wp.array[wp.float32],
    convergence_threshold: wp.float32,
    not_converged: wp.array[wp.bool],
    out_radii: wp.array[wp.float32],
    out_centers: wp.array[wp.vec3],
    out_not_converged: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    if not not_converged[tid]:
        return

    p = points[tid]
    center = centers[tid]
    dist_to_start = wp.length(center - p)

    if wp.abs(n_dists[tid] - dist_to_start) < TOLERANCE_PLANAR_CONSTANT:
        out_not_converged[tid] = False
        return

    diff = n_points[tid] - p
    denom = wp.float32(2.0) * wp.dot(diff, normals[tid])
    if wp.abs(denom) < TOLERANCE_PLANAR_CONSTANT:
        out_not_converged[tid] = False
        return

    new_r = wp.dot(diff, diff) / denom
    out_radii[tid] = new_r
    out_centers[tid] = p + normals[tid] * new_r

    if old_radii[tid] - new_r < convergence_threshold:
        out_not_converged[tid] = False
