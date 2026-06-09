import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels import array as kernel_array

CASE_NONE = wp.constant(wp.int32(0))
CASE_BASIC = wp.constant(wp.int32(1))
CASE_ONE_VERTEX = wp.constant(wp.int32(2))
CASE_ONE_EDGE = wp.constant(wp.int32(3))


@wp.func
def triangle_case_code(s0: wp.int32, s1: wp.int32, s2: wp.int32) -> wp.int32:
    sa, sb, sc = kernel_array.sort3(s0, s1, s2)
    coded = wp.int32(14) + (sa << 3) + (sb << 2) + (sc << 1)
    if coded == wp.int32(4) or coded == wp.int32(12):
        return CASE_BASIC
    if coded == wp.int32(8):
        return CASE_ONE_VERTEX
    if coded == wp.int32(16):
        return CASE_ONE_EDGE
    return CASE_NONE


@wp.func
def plane_with_line(
    plane_origin: wp.vec3, plane_normal: wp.vec3, p0: wp.vec3, p1: wp.vec3, line_segments: wp.bool
) -> tuple[wp.vec3, wp.bool]:
    line_dir = wp.normalize(p1 - p0)
    n = wp.normalize(plane_normal)
    t = wp.dot(n, plane_origin - p0)
    b = wp.dot(n, line_dir)
    valid = wp.abs(b) > TOLERANCE_ZERO_CONSTANT
    if line_segments:
        test = wp.dot(n, plane_origin - p1)
        different_sides = wp.sign(t) != wp.sign(test)
        nonzero = (wp.abs(t) > TOLERANCE_ZERO_CONSTANT) or (wp.abs(test) > TOLERANCE_ZERO_CONSTANT)
        valid = valid and different_sides and nonzero
    if valid:
        d = t / b
        return p0 + line_dir * d, wp.bool(True)
    return wp.vec3(0.0, 0.0, 0.0), wp.bool(False)


@wp.func
def find_unique_sign_vertex(s0: wp.int32, s1: wp.int32, s2: wp.int32) -> wp.int32:
    if s0 != s1 and s0 != s2:
        return wp.int32(0)
    if s1 != s0 and s1 != s2:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def vertex_at(local_index: wp.int32, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    if local_index == wp.int32(0):
        return v0
    if local_index == wp.int32(1):
        return v1
    return v2


@wp.func
def mesh_with_plane_segment_for_face(
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    s0: wp.int32,
    s1: wp.int32,
    s2: wp.int32,
) -> tuple[wp.bool, wp.vec3, wp.vec3]:
    case_code = triangle_case_code(s0, s1, s2)

    if case_code == CASE_BASIC:
        unique_i = find_unique_sign_vertex(s0, s1, s2)
        other_a = (unique_i + wp.int32(1)) % wp.int32(3)
        other_b = (unique_i + wp.int32(2)) % wp.int32(3)
        unique_v = vertex_at(unique_i, v0, v1, v2)
        va = vertex_at(other_a, v0, v1, v2)
        vb = vertex_at(other_b, v0, v1, v2)
        p_a, valid_a = plane_with_line(plane_origin, plane_normal, unique_v, va, False)
        p_b, valid_b = plane_with_line(plane_origin, plane_normal, unique_v, vb, False)
        if valid_a and valid_b:
            return True, p_a, p_b
        return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    if case_code == CASE_ONE_VERTEX:
        on_plane_i = wp.int32(0)
        if s0 == wp.int32(0):
            on_plane_i = wp.int32(0)
        elif s1 == wp.int32(0):
            on_plane_i = wp.int32(1)
        else:
            on_plane_i = wp.int32(2)
        other_a = (on_plane_i + wp.int32(1)) % wp.int32(3)
        other_b = (on_plane_i + wp.int32(2)) % wp.int32(3)
        on_plane_v = vertex_at(on_plane_i, v0, v1, v2)
        va = vertex_at(other_a, v0, v1, v2)
        vb = vertex_at(other_b, v0, v1, v2)
        hit, valid = plane_with_line(plane_origin, plane_normal, va, vb, False)
        if valid:
            return True, on_plane_v, hit
        return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    if case_code == CASE_ONE_EDGE:
        i0 = wp.int32(0)
        i1 = wp.int32(1)
        if s0 == wp.int32(0) and s1 == wp.int32(0):
            i0 = wp.int32(0)
            i1 = wp.int32(1)
        elif s0 == wp.int32(0) and s2 == wp.int32(0):
            i0 = wp.int32(0)
            i1 = wp.int32(2)
        else:
            i0 = wp.int32(1)
            i1 = wp.int32(2)
        return True, vertex_at(i0, v0, v1, v2), vertex_at(i1, v0, v1, v2)

    return False, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def vertex_plane_dots(
    vertices: wp.array[wp.vec3], plane_origin: wp.vec3, plane_normal: wp.vec3, out_dots: wp.array[wp.float32]
) -> None:
    tid = wp.tid()
    out_dots[tid] = wp.dot(vertices[tid] - plane_origin, plane_normal)


@wp.kernel
def mesh_with_plane_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    out_valid: wp.array[wp.bool],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    f = wp.tid()
    i0 = faces[f * 3]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    s0 = kernel_array.tolerance_sign(vertex_dots[i0])
    s1 = kernel_array.tolerance_sign(vertex_dots[i1])
    s2 = kernel_array.tolerance_sign(vertex_dots[i2])
    valid, p0, p1 = mesh_with_plane_segment_for_face(plane_origin, plane_normal, v0, v1, v2, s0, s1, s2)
    out_valid[f] = valid
    out_segments[f, 0] = p0
    out_segments[f, 1] = p1


@wp.kernel
def segments_with_plane(
    start_points: wp.array[wp.vec3],
    end_points: wp.array[wp.vec3],
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    line_segments: wp.bool,
    out_intersections: wp.array[wp.vec3],
    out_valid: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    hit, valid = plane_with_line(plane_origin, plane_normal, start_points[tid], end_points[tid], line_segments)
    out_intersections[tid] = hit
    out_valid[tid] = valid


@wp.func
def vec3_get(v: wp.vec3, i: wp.int32) -> wp.float32:
    if i == wp.int32(0):
        return v[0]
    if i == wp.int32(1):
        return v[1]
    return v[2]


@wp.func
def vec3_equal(a: wp.vec3, b: wp.vec3) -> wp.bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def vec3_argsort(a: wp.vec3) -> tuple[wp.int32, wp.int32, wp.int32]:
    xy = a[0] <= a[1]
    yz = a[1] <= a[2]
    xz = a[0] <= a[2]
    if xy:
        if yz:
            return wp.int32(0), wp.int32(1), wp.int32(2)
        if xz:
            return wp.int32(0), wp.int32(2), wp.int32(1)
        return wp.int32(2), wp.int32(0), wp.int32(1)
    if xz:
        return wp.int32(1), wp.int32(0), wp.int32(2)
    if yz:
        return wp.int32(1), wp.int32(2), wp.int32(0)
    return wp.int32(2), wp.int32(1), wp.int32(0)


@wp.func
def interval_intersect(a: wp.vec2, b: wp.vec2) -> wp.vec2:
    return wp.vec2(wp.max(a[0], b[0]), wp.min(a[1], b[1]))


@wp.func
def intersection_line_coordinate(
    start1: wp.vec3, direction1: wp.vec3, start2: wp.vec3, direction2: wp.vec3
) -> wp.float32:
    minors = wp.cross(direction1, direction2)
    order_x, order_y, order_z = vec3_argsort(minors)
    minor_x = vec3_get(minors, order_x)
    minor_z = vec3_get(minors, order_z)
    i = order_z
    if minor_z >= -minor_x:
        i = order_z
    else:
        i = order_x
    numerator = vec3_get(wp.cross(start2 - start1, direction2), i)
    denominator = vec3_get(minors, i)
    return numerator / denominator


@wp.func
def triangle_aabb(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    lower = wp.vec3(
        wp.min(v0[0], wp.min(v1[0], v2[0])), wp.min(v0[1], wp.min(v1[1], v2[1])), wp.min(v0[2], wp.min(v1[2], v2[2]))
    )
    upper = wp.vec3(
        wp.max(v0[0], wp.max(v1[0], v2[0])), wp.max(v0[1], wp.max(v1[1], v2[1])), wp.max(v0[2], wp.max(v1[2], v2[2]))
    )
    return lower, upper


@wp.func
def triangles_share_vertex(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3) -> wp.bool:
    return (
        vec3_equal(a0, b0)
        or vec3_equal(a0, b1)
        or vec3_equal(a0, b2)
        or vec3_equal(a1, b0)
        or vec3_equal(a1, b1)
        or vec3_equal(a1, b2)
        or vec3_equal(a2, b0)
        or vec3_equal(a2, b1)
        or vec3_equal(a2, b2)
    )


@wp.func
def axis_interval_projection(axis: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> wp.vec2:
    p0 = wp.dot(axis, v0)
    p1 = wp.dot(axis, v1)
    p2 = wp.dot(axis, v2)
    return wp.vec2(wp.min(p0, wp.min(p1, p2)), wp.max(p0, wp.max(p1, p2)))


@wp.func
def overlaps_along_axis(
    axis: wp.vec3, a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> wp.bool:
    interval_a = axis_interval_projection(axis, a0, a1, a2)
    interval_b = axis_interval_projection(axis, b0, b1, b2)
    return interval_a[0] <= interval_b[1] and interval_a[1] >= interval_b[0]


@wp.func
def triangles_intersect_sat(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3) -> wp.bool:
    edge0 = a1 - a0
    edge1 = a2 - a0
    edge2 = a2 - a1
    normal = wp.cross(edge0, edge1)

    other_edge0 = b1 - b0
    other_edge1 = b2 - b0
    other_edge2 = b2 - b1
    other_normal = wp.cross(other_edge0, other_edge1)

    if vec3_equal(normal, other_normal):
        return False

    axis0 = normal
    axis1 = other_normal
    axis2 = wp.cross(edge0, other_edge0)
    axis3 = wp.cross(edge0, other_edge1)
    axis4 = wp.cross(edge0, other_edge2)
    axis5 = wp.cross(edge1, other_edge0)
    axis6 = wp.cross(edge1, other_edge1)
    axis7 = wp.cross(edge1, other_edge2)
    axis8 = wp.cross(edge2, other_edge0)
    axis9 = wp.cross(edge2, other_edge1)
    axis10 = wp.cross(edge2, other_edge2)

    if not overlaps_along_axis(axis0, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis1, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis2, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis3, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis4, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis5, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis6, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis7, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis8, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis9, a0, a1, a2, b0, b1, b2):
        return False
    if not overlaps_along_axis(axis10, a0, a1, a2, b0, b1, b2):
        return False
    return True


@wp.func
def triangle_intersection_segment(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> tuple[wp.bool, wp.vec3, wp.vec3]:
    edge0 = a1 - a0
    edge1 = a2 - a0
    normal = wp.normalize(wp.cross(edge0, edge1))

    other_edge0 = b1 - b0
    other_edge1 = b2 - b0
    other_normal = wp.normalize(wp.cross(other_edge0, other_edge1))

    proj1 = wp.vec3(wp.dot(other_normal, a0 - b0), wp.dot(other_normal, a1 - b0), wp.dot(other_normal, a2 - b0))
    proj2 = wp.vec3(wp.dot(normal, b0 - a0), wp.dot(normal, b1 - a0), wp.dot(normal, b2 - a0))

    order1_x, order1_y, order1_z = vec3_argsort(proj1)
    proj1a = vec3_get(proj1, order1_x)
    proj1c = vec3_get(proj1, order1_z)
    proj1b = vec3_get(proj1, order1_y)

    va_min = vertex_at(order1_x, a0, a1, a2)
    va_max = vertex_at(order1_z, a0, a1, a2)
    line_origin = (va_min * proj1c - va_max * proj1a) / (proj1c - proj1a)
    line_direction = wp.cross(normal, other_normal)

    edge_direction = vertex_at(order1_x if proj1b >= wp.float32(0.0) else order1_z, a0, a1, a2) - vertex_at(
        order1_y, a0, a1, a2
    )
    t2 = intersection_line_coordinate(line_origin, line_direction, vertex_at(order1_y, a0, a1, a2), edge_direction)

    if t2 > wp.float32(0.0):
        interval = wp.vec2(wp.float32(0.0), t2)
    else:
        interval = wp.vec2(t2, wp.float32(0.0))

    order2_x, order2_y, order2_z = vec3_argsort(proj2)
    edge_direction = vertex_at(order2_z, b0, b1, b2) - vertex_at(order2_x, b0, b1, b2)
    s1 = intersection_line_coordinate(line_origin, line_direction, vertex_at(order2_x, b0, b1, b2), edge_direction)
    edge_direction = vertex_at(
        order2_x if vec3_get(proj2, order2_y) >= wp.float32(0.0) else order2_z, b0, b1, b2
    ) - vertex_at(order2_y, b0, b1, b2)
    s2 = intersection_line_coordinate(line_origin, line_direction, vertex_at(order2_y, b0, b1, b2), edge_direction)

    if s1 <= s2:
        other_interval = wp.vec2(s1, s2)
    else:
        other_interval = wp.vec2(s2, s1)

    interval = interval_intersect(interval, other_interval)

    p0 = line_origin + interval[0] * line_direction
    p1 = line_origin + interval[1] * line_direction
    return True, p0, p1


@wp.func
def face_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    base = face_index * wp.int32(3)
    i0 = faces[base]
    i1 = faces[base + wp.int32(1)]
    i2 = faces[base + wp.int32(2)]
    return vertices[i0], vertices[i1], vertices[i2]


@wp.kernel
def face_aabb_bounds(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lower: wp.array[wp.vec3], out_upper: wp.array[wp.vec3]
) -> None:
    f = wp.tid()
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(f))
    lower, upper = triangle_aabb(v0, v1, v2)
    out_lower[f] = lower
    out_upper[f] = upper


@wp.kernel
def expand_query_target_pairs(
    offsets: wp.array[wp.int32],
    hit_counts: wp.array[wp.int32],
    target_indices: wp.array[wp.int32],
    out_pairs: wp.array2d[wp.int32],
) -> None:
    q = wp.tid()
    start = int(offsets[q])
    count = int(hit_counts[q])
    i = int(0)
    w = start
    while i < count:
        out_pairs[w, 0] = wp.int32(q)
        out_pairs[w, 1] = target_indices[w]
        w = w + 1
        i = i + 1


@wp.kernel
def filter_intersecting_pairs(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    pairs: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    qa, qb, qc = face_vertices(query_vertices, query_faces, pairs[tid, 0])
    ta, tb, tc = face_vertices(target_vertices, target_faces, pairs[tid, 1])
    if triangles_share_vertex(qa, qb, qc, ta, tb, tc):
        out_valid[tid] = False
        return
    out_valid[tid] = triangles_intersect_sat(qa, qb, qc, ta, tb, tc)


@wp.kernel
def triangle_pair_segments(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    pairs: wp.array2d[wp.int32],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    tid = wp.tid()
    qa, qb, qc = face_vertices(query_vertices, query_faces, pairs[tid, 0])
    ta, tb, tc = face_vertices(target_vertices, target_faces, pairs[tid, 1])
    valid, p0, p1 = triangle_intersection_segment(qa, qb, qc, ta, tb, tc)
    if valid:
        out_segments[tid, 0] = p0
        out_segments[tid, 1] = p1


@wp.kernel
def segment_nondegenerate(segments: wp.array2d[wp.vec3], out_valid: wp.array[wp.bool]) -> None:
    tid = wp.tid()
    p0 = segments[tid, 0]
    p1 = segments[tid, 1]
    out_valid[tid] = wp.length(p1 - p0) > TOLERANCE_MERGE_CONSTANT
