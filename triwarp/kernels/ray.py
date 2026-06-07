import warp as wp


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

    if not (
        p[0] > mesh_min[0]
        and p[1] > mesh_min[1]
        and p[2] > mesh_min[2]
        and p[0] < mesh_max[0]
        and p[1] < mesh_max[1]
        and p[2] < mesh_max[2]
    ):
        out_contains[tid] = False
        return

    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_contains[tid] = query.result and query.sign < 0.0


@wp.kernel
def intersects_first(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_triangle_index: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    direction = wp.normalize(ray_directions[tid])
    query = wp.mesh_query_ray(mesh_id, ray_origins[tid], direction, max_t)
    if query.result:
        out_triangle_index[tid] = query.face
    else:
        out_triangle_index[tid] = wp.int32(-1)
