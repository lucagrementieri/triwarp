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


@wp.func
def ray_query_first(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: wp.float32,
) -> tuple[wp.bool, wp.int32, wp.vec3]:
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_t)
    if query.result:
        return True, query.face, origin + direction * query.t
    return False, wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)


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
    _hit, face, _location = ray_query_first(mesh_id, ray_origins[tid], direction, max_t)
    out_triangle_index[tid] = face


@wp.kernel
def intersects_first_detail(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_triangle_index: wp.array[wp.int32],
    out_locations: wp.array[wp.vec3],
) -> None:
    tid = wp.tid()
    direction = wp.normalize(ray_directions[tid])
    _hit, face, location = ray_query_first(mesh_id, ray_origins[tid], direction, max_t)
    out_triangle_index[tid] = face
    out_locations[tid] = location


@wp.kernel
def face_hit_mask(
    triangle_index: wp.array[wp.int32],
    out_hit: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    out_hit[tid] = triangle_index[tid] >= 0


@wp.kernel
def intersects_any(
    mesh_id: wp.uint64,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    max_t: wp.float32,
    out_hit: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    direction = wp.normalize(ray_directions[tid])
    out_hit[tid] = wp.mesh_query_ray_anyhit(mesh_id, ray_origins[tid], direction, max_t)
