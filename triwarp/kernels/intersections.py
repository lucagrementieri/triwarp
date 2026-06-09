import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
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
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    p0: wp.vec3,
    p1: wp.vec3,
    line_segments: wp.bool,
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
    vertices: wp.array[wp.vec3],
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    out_dots: wp.array[wp.float32],
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
