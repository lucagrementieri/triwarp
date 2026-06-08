import warp as wp
from triwarp.kernels import array as kernel_array


@wp.kernel
def aabb_bounds(
    points: wp.array[wp.vec3],
    out_min: wp.array[wp.float32],
    out_max: wp.array[wp.float32],
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
def query_ball_count(
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
    j = int(0)
    c = int(0)
    while wp.bvh_query_next(query, j):
        if wp.length(points[j] - q) <= r:
            c = c + 1
    out_neighbor_counts[tid] = c


# TODO: Maybe use hashgrid
@wp.kernel
def query_ball_neighbors(
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
    point_idx = int(0)
    w = int(offsets[tid])
    while wp.bvh_query_next(query, point_idx):
        d = wp.length(points[point_idx] - q)
        if d <= r:
            out_indices[w] = point_idx
            out_distances[w] = d
            w = w + 1


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
    j = int(0)
    c = int(0)
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
    primitive_idx = int(0)
    w = int(offsets[tid])
    while wp.bvh_query_next(query, primitive_idx):
        out_indices[w] = primitive_idx
        w = w + 1


@wp.kernel
def query_nearest_neighbors(
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
    point_index = int(-1)

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
