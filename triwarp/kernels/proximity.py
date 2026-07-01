import warp as wp

from triwarp.constants import TILE_1D, TOLERANCE_MERGE_CONSTANT, TOLERANCE_PLANAR_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.algorithms import bfs as kernel_bfs


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
    queue = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    dummy = wp.zeros(shape=1, dtype=wp.int32)
    out_counts[i] = kernel_bfs.per_source_bfs_collect(
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
    queue = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    visited = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    ext_dist = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.float32)
    ext_idx = wp.zeros(shape=kernel_bfs._PER_SOURCE_MAX_NEIGHBORS, dtype=wp.int32)
    kernel_bfs.per_source_bfs_collect(
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
def solid_angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.float32:
    """Signed solid angle subtended by triangle (a, b, c) at point p (``igl::solid_angle``)."""
    v0 = a - p
    v1 = b - p
    v2 = c - p
    vl0 = wp.length(v0)
    vl1 = wp.length(v1)
    vl2 = wp.length(v2)
    detf = (
        v0[0] * v1[1] * v2[2]
        + v1[0] * v2[1] * v0[2]
        + v2[0] * v0[1] * v1[2]
        - v2[0] * v1[1] * v0[2]
        - v1[0] * v0[1] * v2[2]
        - v0[0] * v2[1] * v1[2]
    )
    dp0 = wp.dot(v1, v2)
    dp1 = wp.dot(v2, v0)
    dp2 = wp.dot(v0, v1)
    denom = vl0 * vl1 * vl2 + dp0 * vl0 + dp1 * vl1 + dp2 * vl2
    return wp.atan2(detf, denom) / (wp.float32(2.0) * wp.PI)


@wp.func
def solid_angle_at_face(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: int,
    p: wp.vec3,
) -> wp.float32:
    face_indices = faces[f * 3 : (f + 1) * 3]
    i0 = int(face_indices[0])
    i1 = int(face_indices[1])
    i2 = int(face_indices[2])
    return solid_angle(vertices[i0], vertices[i1], vertices[i2], p)


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
def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q = int(wp.tid())
    p = query_points[q]
    w = wp.float32(0.0)
    n_f = int(n_faces)
    for f in range(n_f):
        face_indices = faces[f * 3 : (f + 1) * 3]
        i0 = int(face_indices[0])
        i1 = int(face_indices[1])
        i2 = int(face_indices[2])
        w = w + solid_angle(vertices[i0], vertices[i1], vertices[i2], p)
    out_winding[q] = w


@wp.kernel
def winding_number_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q, tile_i, t = wp.tid()
    n_f = int(n_faces)
    face_offset = int(tile_i) * TILE_1D
    if face_offset >= n_f:
        return

    remaining = n_f - face_offset
    count = remaining
    if count > TILE_1D:
        count = TILE_1D

    p = query_points[int(q)]
    face_idx = face_offset + int(t)
    contrib = wp.float32(0.0)
    if int(t) < count:
        contrib = solid_angle_at_face(vertices, faces, face_idx, p)

    tile = wp.tile(contrib)
    tile_sum = wp.tile_sum(tile)
    if t == 0:
        wp.tile_atomic_add(out_winding, tile_sum, (int(q),))


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
