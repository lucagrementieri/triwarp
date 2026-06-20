import warp as wp


@wp.kernel
def point_plane_distance(
    points: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    w = points[tid] - plane_origin
    out_distances[tid] = wp.dot(plane_normal, w) / wp.length(plane_normal)
