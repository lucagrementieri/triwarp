import warp as wp

from triwarp.kernels import array as kernel_array


@wp.func
def bvh_aabb_collect(
    bvh_id: wp.uint64,
    q: wp.vec3,
    half_extent: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) the BVH hits around ``q``.
    h = half_extent
    lower = q - wp.vec3(h)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(h)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        if write:
            out_indices[base + c] = j
        c = c + 1
    return c


@wp.kernel
def query_bvh_aabb_count(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    # ``out_counts`` doubles as the (never written) emit target of the counting pass.
    out_counts[tid] = bvh_aabb_collect(
        bvh_id, queries[tid], half_extent, wp.bool(False), wp.int32(0), out_counts
    )


@wp.kernel
def query_bvh_aabb_neighbors(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    bvh_aabb_collect(bvh_id, queries[tid], half_extent, wp.bool(True), offsets[tid], out_indices)


@wp.func
def hashgrid_ball_collect(
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) points within ``radius``.
    query = wp.hash_grid_query(grid_id, q, radius)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        d = wp.length(points[j] - q)
        if d <= radius:
            if write:
                out_indices[base + c] = j
                out_distances[base + c] = d
            c = c + 1
    return c


@wp.kernel
def query_hashgrid_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    dummy_dist = wp.zeros(shape=1, dtype=wp.float32)
    out_neighbor_counts[tid] = hashgrid_ball_collect(
        points,
        grid_id,
        queries[tid],
        radius,
        wp.bool(False),
        wp.int32(0),
        out_neighbor_counts,
        dummy_dist,
    )


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
    hashgrid_ball_collect(
        points,
        grid_id,
        queries[tid],
        radius,
        wp.bool(True),
        offsets[tid],
        out_indices,
        out_distances,
    )


@wp.func
def bvh_ball_collect(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) points within ``radius``.
    lower = q - wp.vec3(radius)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(radius)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        d = wp.length(points[j] - q)
        if d <= radius:
            if write:
                out_indices[base + c] = j
                out_distances[base + c] = d
            c = c + 1
    return c


@wp.kernel
def query_bvh_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    dummy_dist = wp.zeros(shape=1, dtype=wp.float32)
    out_neighbor_counts[tid] = bvh_ball_collect(
        points,
        bvh_id,
        queries[tid],
        radius,
        wp.bool(False),
        wp.int32(0),
        out_neighbor_counts,
        dummy_dist,
    )


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
    bvh_ball_collect(
        points,
        bvh_id,
        queries[tid],
        radius,
        wp.bool(True),
        offsets[tid],
        out_indices,
        out_distances,
    )


@wp.func
def knn_sorted_insert(
    point_index: wp.int32,
    d: wp.float32,
    k: wp.int32,
    radius: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> None:
    # Insert ``(point_index, d)`` into the ascending k-nearest rows, dropping the current worst.
    if d > radius:
        return
    if d >= out_distances_row[k - 1]:
        return
    if k == 1:
        out_indices_row[0] = point_index
        out_distances_row[0] = d
        return
    slot = kernel_array.binary_search_index(out_distances_row, d)
    kernel_array.array_shift_insert(out_distances_row, d, slot)
    kernel_array.array_shift_insert(out_indices_row, point_index, slot)


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
    lower = q - wp.vec3(r)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    point_index = wp.int32(0)

    while wp.bvh_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, radius, out_indices[tid], out_distances[tid])


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
        knn_sorted_insert(point_index, d, k, radius, out_indices[tid], out_distances[tid])
