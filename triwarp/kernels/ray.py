import warp as wp


@wp.func
def ray_query_first(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.bool, wp.int32, wp.vec3]:
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_t)
    if query.result:
        return True, query.face, origin + direction * query.t
    return False, wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)


@wp.func
def first_hit(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.int32, wp.vec3]:
    # First-hit face index (-1 on miss) and location; ``direction`` need not be unit length.
    _hit, face, location = ray_query_first(mesh_id, origin, wp.normalize(direction), max_t)
    return face, location


@wp.func
def any_hit(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> wp.bool:
    return wp.mesh_query_ray_anyhit(mesh_id, origin, wp.normalize(direction), max_t)


@wp.func
def is_hit(triangle_index: wp.int32) -> wp.bool:
    return triangle_index >= 0


@wp.func
def longest_ray_distance(
    mesh_id: wp.uint64,
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: wp.float32,
    planar_tol: wp.float32,
) -> wp.float32:
    # ``direction`` need not be unit length; the offset walk below assumes a unit ray.
    unit_direction = wp.normalize(direction)
    t_offset = wp.float32(0.0)
    cur_origin = origin
    for _i in range(64):
        remaining = max_t - t_offset
        if remaining <= wp.float32(0.0):
            break
        query = wp.mesh_query_ray(mesh_id, cur_origin, unit_direction, remaining)
        if not query.result:
            break
        dist = t_offset + query.t
        if dist > planar_tol:
            return dist
        t_offset = t_offset + query.t + planar_tol
        cur_origin = origin + unit_direction * t_offset
        if t_offset >= max_t:
            break
    return wp.inf
