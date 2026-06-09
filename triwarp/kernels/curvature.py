import warp as wp

from triwarp.kernels import array as kernel_array


@wp.func
def line_ball_intersection_segment(
    start_point: wp.vec3, end_point: wp.vec3, center: wp.vec3, radius: wp.float32
) -> wp.float32:
    L = end_point - start_point
    oc = start_point - center
    r = radius
    ldotl = wp.dot(L, L)
    ldotoc = wp.dot(L, oc)
    ocdotoc = wp.dot(oc, oc)
    discrim = ldotoc * ldotoc - ldotl * (ocdotoc - r * r)

    if discrim <= wp.float32(0.0):
        return wp.float32(0.0)

    sqrt_discrim = wp.sqrt(discrim)
    d1 = (-ldotoc - sqrt_discrim) / ldotl
    d2 = (-ldotoc + sqrt_discrim) / ldotl

    d1 = wp.clamp(d1, wp.float32(0.0), wp.float32(1.0))
    d2 = wp.clamp(d2, wp.float32(0.0), wp.float32(1.0))

    return (d2 - d1) * wp.sqrt(ldotl)


@wp.kernel
def edge_aabb_from_endpoints(
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    tid = int(wp.tid())
    v0 = vertices[face_adjacency_edges[tid, 0]]
    v1 = vertices[face_adjacency_edges[tid, 1]]
    out_lower[tid] = wp.vec3(wp.min(v0[0], v1[0]), wp.min(v0[1], v1[1]), wp.min(v0[2], v1[2]))
    out_upper[tid] = wp.vec3(wp.max(v0[0], v1[0]), wp.max(v0[1], v1[1]), wp.max(v0[2], v1[2]))


@wp.kernel
def accumulate_mean_curvature(
    queries: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    angles: wp.array[wp.float32],
    convex: wp.array[wp.bool],
    candidate_edge_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    radius: wp.float32,
    out_mean_curvature: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    edge_idx = candidate_edge_indices[tid]
    query_idx = kernel_array.binary_search_index(offsets, wp.int32(tid)) - wp.int32(1)

    e0 = face_adjacency_edges[edge_idx, 0]
    e1 = face_adjacency_edges[edge_idx, 1]
    start_point = vertices[e0]
    end_point = vertices[e1]
    center = queries[query_idx]

    length = line_ball_intersection_segment(start_point, end_point, center, radius)
    angle = angles[edge_idx]
    sign = wp.float32(1.0)
    if not convex[edge_idx]:
        sign = wp.float32(-1.0)

    wp.atomic_add(out_mean_curvature, query_idx, length * angle * sign * wp.float32(0.5))
