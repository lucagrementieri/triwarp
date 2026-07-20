import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, TILE_1D


@wp.kernel
def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    out_projections: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    normal = face_normals[face_adjacency[tid, 0]]
    origin = vertices[face_adjacency_edges[tid, 0]]
    vid_other = face_adjacency_unshared[tid, 1]
    vector_other = vertices[vid_other] - origin
    out_projections[tid] = wp.dot(vector_other, normal)


@wp.kernel
def hull_support_extremes(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    n_points: wp.int32,
    out_best_max: wp.array[wp.float32],
    out_best_min: wp.array[wp.float32],
) -> None:
    k, tile_i, t = wp.tid()
    n_p = int(n_points)
    point_offset = int(tile_i) * TILE_1D
    if point_offset >= n_p:
        return

    remaining = n_p - point_offset
    count = remaining
    if count > TILE_1D:
        count = TILE_1D

    direction = directions[int(k)]
    idx = point_offset + int(t)
    # Out-of-range lanes must not win either reduction, so seed max with -inf and
    # min with +inf; valid lanes overwrite both with their signed distance.
    contrib_max = -FLOAT32_INF_CONSTANT
    contrib_min = FLOAT32_INF_CONSTANT
    if int(t) < count:
        distance = wp.dot(direction, points[idx])
        contrib_max = distance
        contrib_min = distance

    tile_max = wp.tile_max(wp.tile(contrib_max))[0]
    tile_min = wp.tile_min(wp.tile(contrib_min))[0]
    if t == 0:
        wp.atomic_max(out_best_max, int(k), tile_max)
        wp.atomic_min(out_best_min, int(k), tile_min)


@wp.kernel
def mark_hull_support(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    n_points: wp.int32,
    out_mask: wp.array[wp.bool],
) -> None:
    k, tile_i, t = wp.tid()
    n_p = int(n_points)
    point_offset = int(tile_i) * TILE_1D
    if point_offset >= n_p:
        return

    idx = point_offset + int(t)
    if idx < n_p:
        # A hemisphere direction n covers both +n (max, supports the vertex farthest
        # along n) and -n (min, supports the vertex farthest along -n).
        distance = wp.dot(directions[int(k)], points[idx])
        # Slack scales with the per-direction support extent so the test is
        # scale-invariant and stays above the float32 dot-product noise floor.
        slack = tolerance * (best_max[int(k)] - best_min[int(k)])
        if distance >= best_max[int(k)] - slack or distance <= best_min[int(k)] + slack:
            out_mask[idx] = True
