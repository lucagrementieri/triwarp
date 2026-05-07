import warp as wp

TOLERANCE_MERGE = 1e-8
TOLERANCE_ZERO = 1e-12


@wp.func
def triangle_cross(
    in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]
) -> wp.vec3:
    v0 = in_vertices[in_face[0]]
    v1 = in_vertices[in_face[1]]
    v2 = in_vertices[in_face[2]]
    e0 = v1 - v0
    e1 = v2 - v0
    return wp.cast(wp.cross(e0, e1), wp.vec3)


@wp.func
def triangle_edges(
    in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    e0 = wp.vec3(*(in_vertices[in_face[1]] - in_vertices[in_face[0]]))
    e1 = wp.vec3(*(in_vertices[in_face[2]] - in_vertices[in_face[0]]))
    e2 = wp.vec3(*(in_vertices[in_face[2]] - in_vertices[in_face[1]]))
    return e0, e1, e2


@wp.func
def face_normals_and_area(
    in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]
) -> tuple[wp.vec3, wp.float32]:
    cross = triangle_cross(in_vertices, in_face)
    norm = wp.length(cross)
    if norm > TOLERANCE_ZERO:
        normal = cross / norm
    area = 0.5 * norm
    return normal, area


@wp.kernel
def face_normals_and_areas(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
    out_areas: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    normal, area = face_normals_and_area(in_vertices, in_faces[f * 3 : (f + 1) * 3])
    out_normals[f] = normal
    out_areas[f] = area


@wp.kernel
def angles(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    out_angles: wp.array[wp.vec3],
) -> None:
    f = int(wp.tid())
    edges = triangle_edges(in_vertices, in_faces[f * 3 : (f + 1) * 3])

    u = wp.normalize(edges[0])
    v = wp.normalize(edges[1])
    w = wp.normalize(edges[2])

    out_angles[f][0] = wp.acos(wp.clamp(wp.dot(u, v), -1.0, 1.0))
    out_angles[f][1] = wp.acos(wp.clamp(wp.dot(-u, w), -1.0, 1.0))
    out_angles[f][2] = wp.pi - out_angles[f][0] - out_angles[f][1]

    degen = (
        (out_angles[f][0] < TOLERANCE_MERGE)
        or (out_angles[f][1] < TOLERANCE_MERGE)
        or (out_angles[f][2] < TOLERANCE_MERGE)
    )
    if degen:
        out_angles[f] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def nondegenerate(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    out_nondegenerate: wp.array[wp.bool],
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(in_vertices, triangle_face)
    _, area = face_normals_and_area(in_vertices, triangle_face)
    length_e0 = wp.length(e0)
    length_e1 = wp.length(e1)
    height_e0 = 2.0 * area / length_e0
    height_e1 = 2.0 * area / length_e1
    out_nondegenerate[f] = (
        (height_e0 > TOLERANCE_MERGE)
        and (height_e1 > TOLERANCE_MERGE)
        and (length_e0 > TOLERANCE_MERGE)
        and (length_e1 > TOLERANCE_MERGE)
    )


@wp.kernel
def barycentric_to_points(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    in_barycentric: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    barycentric = in_barycentric[f]
    s = barycentric[0] + barycentric[1] + barycentric[2]
    barycentric = barycentric / s
    out_points[f] = (
        in_vertices[triangle_face[0]] * barycentric[0]
        + in_vertices[triangle_face[1]] * barycentric[1]
        + in_vertices[triangle_face[2]] * barycentric[2]
    )


@wp.kernel
def points_to_barycentric_cramer(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    in_points: wp.array[wp.vec3],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(in_vertices, triangle_face)
    w = in_points[f] - in_vertices[triangle_face[0]]
    dot00 = wp.length_sq(e0)
    dot01 = wp.dot(e0, e1)
    dot02 = wp.dot(e0, w)
    dot11 = wp.length_sq(e1)
    dot12 = wp.dot(e1, w)
    inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)
    out_barycentric[f][2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
    out_barycentric[f][1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
    out_barycentric[f][0] = 1.0 - out_barycentric[f][1] - out_barycentric[f][2]


@wp.kernel
def points_to_barycentric_cross(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    in_points: wp.array[wp.vec3],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(in_vertices, triangle_face)
    w = in_points[f] - in_vertices[triangle_face[0]]
    n = wp.cross(e0, e1)
    inverse_denominator = 1.0 / wp.length_sq(n)
    out_barycentric[f][2] = wp.dot(wp.cross(e0, w), n) * inverse_denominator
    out_barycentric[f][1] = wp.dot(wp.cross(w, e1), n) * inverse_denominator
    out_barycentric[f][0] = 1.0 - out_barycentric[f][1] - out_barycentric[f][2]


@wp.kernel
def closest_point(
    in_vertices: wp.array[wp.vec3],
    in_faces: wp.array[wp.int32],
    in_points: wp.array[wp.vec3],
    out_closest: wp.array[wp.vec3],
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    ab, ac, bc = triangle_edges(in_vertices, triangle_face)

    # check if P is in vertex region outside A
    ap = in_points[f] - in_vertices[triangle_face[0]]
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    is_a = d1 < 0.0 and d2 < 0.0
    if is_a:
        out_closest[f] = in_vertices[triangle_face[0]]
        return

    # check if P in vertex region outside B
    bp = in_points[f] - in_vertices[triangle_face[1]]
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    is_b = d3 > -TOLERANCE_ZERO and d4 <= d3
    if is_b:
        out_closest[f] = in_vertices[triangle_face[1]]
        return

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = vc < TOLERANCE_ZERO and d1 > -TOLERANCE_ZERO and d3 < TOLERANCE_ZERO
    if is_ab:
        v = d1 / (d1 - d3)
        out_closest[f] = in_vertices[triangle_face[0]] + v * ab
        return

    # check if P in vertex region outside C
    cp = in_points[f] - in_vertices[triangle_face[2]]
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    is_c = d6 > -TOLERANCE_ZERO and d5 <= d6
    if is_c:
        out_closest[f] = in_vertices[triangle_face[2]]
        return

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = vb < TOLERANCE_ZERO and d2 > -TOLERANCE_ZERO and d6 < TOLERANCE_ZERO
    if is_ac:
        w = d2 / (d2 - d6)
        out_closest[f] = in_vertices[triangle_face[0]] + w * ac
        return

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = (
        va < TOLERANCE_ZERO
        and (d4 - d3) > -TOLERANCE_ZERO
        and (d5 - d6) > -TOLERANCE_ZERO
    )
    if is_bc:
        d43 = d4 - d3
        w = d43 / (d43 + (d5 - d6))
        out_closest[f] = in_vertices[triangle_face[1]] + w * bc
        return

    # any remaining points must be inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    out_closest[f] = in_vertices[triangle_face[0]] + ab * v + ac * w
