import warp as wp

TOLERANCE_MERGE = 1e-8


@wp.func
def triangle_cross(in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]) -> wp.vec3:
    v0 = in_vertices[in_face[0]]
    v1 = in_vertices[in_face[1]]
    v2 = in_vertices[in_face[2]]
    e0 = v1 - v0
    e1 = v2 - v0
    return wp.cast(wp.cross(e0, e1), wp.vec3)


@wp.func
def triangle_edges(in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    e0 = wp.vec3(*(in_vertices[in_face[1]] - in_vertices[in_face[0]]))
    e1 = wp.vec3(*(in_vertices[in_face[2]] - in_vertices[in_face[0]]))
    e2 = wp.vec3(*(in_vertices[in_face[2]] - in_vertices[in_face[1]]))
    return e0, e1, e2


@wp.func
def triangle_area(in_vertices: wp.array[wp.vec3], in_face: wp.array[wp.int32]) -> wp.float32:
    cross = triangle_cross(in_vertices, in_face)
    return 0.5 * wp.length(cross)


@wp.kernel
def face_areas(in_vertices: wp.array[wp.vec3], in_faces: wp.array[wp.int32], out_areas: wp.array[wp.float32]) -> None:
    f = int(wp.tid())
    out_areas[f] = triangle_area(in_vertices, in_faces[f * 3 : (f + 1) * 3])


@wp.kernel
def face_normals(in_vertices: wp.array[wp.vec3], in_faces: wp.array[wp.int32], out_normals: wp.array[wp.vec3]) -> None:
    f = int(wp.tid())
    cross = triangle_cross(in_vertices, in_faces[f * 3 : (f + 1) * 3])
    out_normals[f] = wp.normalize(cross)


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
    in_vertices: wp.array[wp.vec3], in_faces: wp.array[wp.int32], out_nondegenerate: wp.array[wp.bool]
) -> None:
    f = int(wp.tid())
    triangle_face = in_faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(in_vertices, triangle_face)
    area = triangle_area(in_vertices, triangle_face)
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
