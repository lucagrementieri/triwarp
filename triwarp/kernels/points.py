import warp as wp


@wp.kernel
def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = int(wp.tid())
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
    tid = int(wp.tid())
    q = queries[tid]
    r = radius
    lower = wp.vec3(q[0] - r, q[1] - r, q[2] - r)
    upper = wp.vec3(q[0] + r, q[1] + r, q[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = int(0)
    w = int(offsets[tid])
    while wp.bvh_query_next(query, j):
        d = wp.length(points[j] - q)
        if d <= r:
            out_indices[w] = j
            out_distances[w] = d
            w = w + 1


@wp.kernel
def count_close_pairs(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    pair_count: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    p = points[i]
    r = radius
    lower = wp.vec3(p[0] - r, p[1] - r, p[2] - r)
    upper = wp.vec3(p[0] + r, p[1] + r, p[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    j = int(0)
    c = int(0)
    r2 = r * r
    while wp.bvh_query_next(query, j):
        if j > i:
            q = points[j]
            d = q - p
            if wp.dot(d, d) <= r2:
                c = c + 1
    pair_count[i] = c


@wp.kernel
def fill_close_pairs(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    pairs_a: wp.array[wp.int32],
    pairs_b: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    p = points[i]
    r = radius
    lower = wp.vec3(p[0] - r, p[1] - r, p[2] - r)
    upper = wp.vec3(p[0] + r, p[1] + r, p[2] + r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    j = int(0)
    w = int(offsets[i])
    r2 = r * r
    while wp.bvh_query_next(query, j):
        if j > i:
            q = points[j]
            d = q - p
            if wp.dot(d, d) <= r2:
                pairs_a[w] = i
                pairs_b[w] = j
                w = w + 1
