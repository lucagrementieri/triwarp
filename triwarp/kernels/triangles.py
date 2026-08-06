from typing import Any

import warp as wp

from triwarp.constants import PI, TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import to_vec3d
from triwarp.kernels.predicates import triangle_aspect_ratio

# ``face_quality`` metric selectors. Passed as a warp-uniform kernel argument so all four share one
# compiled module (a ``wp.Function`` cannot be a kernel argument -- see AGENTS.md section 4).
QUALITY_ASPECT_RATIO = wp.constant(wp.int32(0))  # circumradius / (2 * inradius), 1 .. +inf
QUALITY_RADIUS_RATIO = wp.constant(wp.int32(1))  # VCG QualityRadii, 0 .. 1
QUALITY_AREA_MAX_SIDE = wp.constant(wp.int32(2))  # 2 * area / longest_side^2, 0 .. sqrt(3)/2
QUALITY_MEAN_RATIO = wp.constant(wp.int32(3))  # 4 * sqrt(3) * area / (a^2 + b^2 + c^2), 0 .. 1
QUALITY_AREA = wp.constant(wp.int32(4))  # plain triangle area


@wp.func
def face_vertices(vertices: wp.array[Any], faces: wp.array[wp.int32], face_index: wp.int32):
    """
    Load the three per-corner values of face ``face_index`` from a flat index buffer.

    Generic over the value dtype: works for positions (``wp.vec3``/``wp.vec3d``/``wp.vec2``)
    as well as per-vertex scalar fields.
    """
    base = face_index * wp.int32(3)
    i0 = faces[base]
    i1 = faces[base + wp.int32(1)]
    i2 = faces[base + wp.int32(2)]
    return vertices[i0], vertices[i1], vertices[i2]


@wp.func
def face_vertices_vec3d(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_index: wp.int32
) -> tuple[wp.vec3d, wp.vec3d, wp.vec3d]:
    """Load the three corners of face ``face_index`` promoted to ``wp.vec3d``."""
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return to_vec3d(v0), to_vec3d(v1), to_vec3d(v2)


@wp.kernel
def face_signed_volumes(
    vertices: wp.array[Any], faces: wp.array[wp.int32], center: Any, out_volumes: wp.array[wp.Float]
) -> None:
    # Signed volume of the tetrahedron (center, v0, v1, v2); the sum over faces is the mesh volume.
    fi = int(wp.tid())
    p0, p1, p2 = face_vertices(vertices, faces, wp.int32(fi))
    d = wp.dot(p0 - center, wp.cross(p1 - center, p2 - center))
    out_volumes[fi] = d / type(d)(6.0)


@wp.func
def triangle_cross(vertices: wp.array[wp.vec3], face: wp.array[wp.int32]) -> wp.vec3:
    v0 = vertices[face[0]]
    v1 = vertices[face[1]]
    v2 = vertices[face[2]]
    e0 = v1 - v0
    e1 = v2 - v0
    return wp.cast(wp.cross(e0, e1), wp.vec3)


@wp.func
def triangle_edges(
    vertices: wp.array[wp.vec3], face: wp.array[wp.int32]
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    e0 = wp.vec3(*(vertices[face[1]] - vertices[face[0]]))
    e1 = wp.vec3(*(vertices[face[2]] - vertices[face[0]]))
    e2 = wp.vec3(*(vertices[face[2]] - vertices[face[1]]))
    return e0, e1, e2


@wp.func
def face_normals_and_area(
    vertices: wp.array[wp.vec3], face: wp.array[wp.int32]
) -> tuple[wp.vec3, wp.float32]:
    normal = triangle_cross(vertices, face)
    norm = wp.length(normal)
    if norm > TOLERANCE_ZERO_CONSTANT:
        normal = normal / norm
    area = 0.5 * norm
    return normal, area


@wp.kernel
def face_normals_and_areas(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
    out_areas: wp.array[wp.float32],
) -> None:
    f = wp.tid()
    normal, area = face_normals_and_area(vertices, faces[f * 3 : (f + 1) * 3])
    out_normals[f] = normal
    out_areas[f] = area


@wp.kernel
def angles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_angles: wp.array2d[wp.float32]
) -> None:
    f = wp.tid()
    edges = triangle_edges(vertices, faces[f * 3 : (f + 1) * 3])

    u = wp.normalize(edges[0])
    v = wp.normalize(edges[1])
    w = wp.normalize(edges[2])

    # wp.acos auto-clamps its argument to [-1, 1], so no explicit wp.clamp is needed.
    out_angles[f, 0] = wp.acos(wp.dot(u, v))
    out_angles[f, 1] = wp.acos(wp.dot(-u, w))
    out_angles[f, 2] = PI - out_angles[f, 0] - out_angles[f, 1]

    degen = (
        (out_angles[f][0] < TOLERANCE_MERGE_CONSTANT)
        or (out_angles[f][1] < TOLERANCE_MERGE_CONSTANT)
        or (out_angles[f][2] < TOLERANCE_MERGE_CONSTANT)
    )
    if degen:
        out_angles[f, 0] = 0.0
        out_angles[f, 1] = 0.0
        out_angles[f, 2] = 0.0


@wp.func
def triangle_radius_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``QualityRadii`` ("inradius/circumradius"): the ratio of the two radii, rescaled so an
    # equilateral triangle reads 1 (the bare geometric ratio is 1/2 there). Symmetric in the three
    # side lengths; zero for a degenerate triangle.
    bc = wp.length(c - b)
    ca = wp.length(a - c)
    ab = wp.length(b - a)
    product = ab * ca * bc
    if product <= type(product)(0.0):
        return type(product)(0.0)
    return (ab + ca - bc) * (bc + ab - ca) * (ca + bc - ab) / product


@wp.func
def triangle_area_max_side(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``Quality`` ("area/max side"): twice the area over the longest side squared, so it is
    # scale-invariant despite the name. ``sqrt(3)/2`` for an equilateral triangle, 0 for a
    # degenerate one.
    ab = b - a
    ac = c - a
    longest_sq = wp.max(wp.max(wp.length_sq(ab), wp.length_sq(ac)), wp.length_sq(wp.sub(c, b)))
    if longest_sq <= type(longest_sq)(0.0):
        return type(longest_sq)(0.0)
    return wp.length(wp.cross(ab, ac)) / longest_sq


@wp.func
def triangle_mean_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # VCG ``QualityMeanRatio``: ``4 * sqrt(3) * area / (a^2 + b^2 + c^2)`` -- 1 for an equilateral
    # triangle, 0 for a degenerate one.
    ab = b - a
    ac = c - a
    sum_sq = wp.length_sq(ab) + wp.length_sq(ac) + wp.length_sq(wp.sub(c, b))
    if sum_sq <= type(sum_sq)(0.0):
        return type(sum_sq)(0.0)
    return type(sum_sq)(2.0) * wp.sqrt(type(sum_sq)(3.0)) * wp.length(wp.cross(ab, ac)) / sum_sq


@wp.func
def triangle_quality(a: Any, b: Any, c: Any, metric: wp.int32) -> wp.Float:
    # Warp-uniform dispatch over the ``QUALITY_*`` selectors: one compiled module for all five
    # metrics, since a ``wp.Function`` cannot cross the ``wp.launch`` boundary.
    if metric == QUALITY_ASPECT_RATIO:
        return triangle_aspect_ratio(a, b, c)
    if metric == QUALITY_RADIUS_RATIO:
        return triangle_radius_ratio(a, b, c)
    if metric == QUALITY_AREA_MAX_SIDE:
        return triangle_area_max_side(a, b, c)
    if metric == QUALITY_MEAN_RATIO:
        return triangle_mean_ratio(a, b, c)
    return type(a[0])(0.5) * wp.length(wp.cross(b - a, c - a))


@wp.kernel
def face_quality(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: wp.int32,
    out_quality: wp.array[wp.float32],
) -> None:
    f = wp.tid()
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(f))
    out_quality[f] = triangle_quality(v0, v1, v2, metric)


@wp.kernel
def nondegenerate(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_nondegenerate: wp.array[wp.bool]
) -> None:
    f = wp.tid()
    triangle_face = faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(vertices, triangle_face)
    _, area = face_normals_and_area(vertices, triangle_face)
    length_e0 = wp.length(e0)
    length_e1 = wp.length(e1)
    height_e0 = 2.0 * area / length_e0
    height_e1 = 2.0 * area / length_e1
    out_nondegenerate[f] = (
        (height_e0 > TOLERANCE_MERGE_CONSTANT)
        and (height_e1 > TOLERANCE_MERGE_CONSTANT)
        and (length_e0 > TOLERANCE_MERGE_CONSTANT)
        and (length_e1 > TOLERANCE_MERGE_CONSTANT)
    )


@wp.kernel
def barycentric_to_points(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
) -> None:
    f = wp.tid()
    triangle_face = faces[f * 3 : (f + 1) * 3]
    face_barycentric = barycentric[f]
    s = face_barycentric[0] + face_barycentric[1] + face_barycentric[2]
    face_barycentric = face_barycentric / s
    out_points[f] = (
        vertices[triangle_face[0]] * face_barycentric[0]
        + vertices[triangle_face[1]] * face_barycentric[1]
        + vertices[triangle_face[2]] * face_barycentric[2]
    )


@wp.func
def point_barycentric_cramer(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, point: wp.vec3) -> wp.vec3:
    # Barycentric coordinates of ``point`` projected into the plane of triangle (v0, v1, v2),
    # by Cramer's rule on the 2x2 Gram system of the two edge vectors. A degenerate triangle makes
    # the determinant zero and the result infinite, which is the caller's cue rather than this
    # function's business (the kernel form has always behaved that way).
    e0 = v1 - v0
    e1 = v2 - v0
    w = point - v0
    dot00 = wp.length_sq(e0)
    dot01 = wp.dot(e0, e1)
    dot02 = wp.dot(e0, w)
    dot11 = wp.length_sq(e1)
    dot12 = wp.dot(e1, w)
    inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)
    v = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
    w2 = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
    return wp.vec3(1.0 - v - w2, v, w2)


@wp.kernel
def points_to_barycentric_cramer(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    f = wp.tid()
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(f))
    out_barycentric[f] = point_barycentric_cramer(v0, v1, v2, points[f])


@wp.kernel
def points_to_barycentric_cross(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_barycentric: wp.array[wp.vec3],
) -> None:
    f = wp.tid()
    triangle_face = faces[f * 3 : (f + 1) * 3]
    e0, e1, _ = triangle_edges(vertices, triangle_face)
    w = points[f] - vertices[triangle_face[0]]
    n = wp.cross(e0, e1)
    inverse_denominator = 1.0 / wp.length_sq(n)
    out_barycentric[f][2] = wp.dot(wp.cross(e0, w), n) * inverse_denominator
    out_barycentric[f][1] = wp.dot(wp.cross(w, e1), n) * inverse_denominator
    out_barycentric[f][0] = 1.0 - out_barycentric[f][1] - out_barycentric[f][2]


@wp.kernel
def closest_point(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_closest: wp.array[wp.vec3],
) -> None:
    f = wp.tid()
    triangle_face = faces[f * 3 : (f + 1) * 3]
    ab, ac, bc = triangle_edges(vertices, triangle_face)

    # check if P is in vertex region outside A
    ap = points[f] - vertices[triangle_face[0]]
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    is_a = d1 < 0.0 and d2 < 0.0
    if is_a:
        out_closest[f] = vertices[triangle_face[0]]
        return

    # check if P in vertex region outside B
    bp = points[f] - vertices[triangle_face[1]]
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    is_b = d3 > -TOLERANCE_ZERO_CONSTANT and d4 <= d3
    if is_b:
        out_closest[f] = vertices[triangle_face[1]]
        return

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = (
        vc < TOLERANCE_ZERO_CONSTANT
        and d1 > -TOLERANCE_ZERO_CONSTANT
        and d3 < TOLERANCE_ZERO_CONSTANT
    )
    if is_ab:
        v = d1 / (d1 - d3)
        out_closest[f] = vertices[triangle_face[0]] + v * ab
        return

    # check if P in vertex region outside C
    cp = points[f] - vertices[triangle_face[2]]
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    is_c = d6 > -TOLERANCE_ZERO_CONSTANT and d5 <= d6
    if is_c:
        out_closest[f] = vertices[triangle_face[2]]
        return

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (
        vb < TOLERANCE_ZERO_CONSTANT
        and d2 > -TOLERANCE_ZERO_CONSTANT
        and d6 < TOLERANCE_ZERO_CONSTANT
    )
    if is_ac:
        w = d2 / (d2 - d6)
        out_closest[f] = vertices[triangle_face[0]] + w * ac
        return

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = (
        va < TOLERANCE_ZERO_CONSTANT
        and (d4 - d3) > -TOLERANCE_ZERO_CONSTANT
        and (d5 - d6) > -TOLERANCE_ZERO_CONSTANT
    )
    if is_bc:
        d43 = d4 - d3
        w = d43 / (d43 + (d5 - d6))
        out_closest[f] = vertices[triangle_face[1]] + w * bc
        return

    # any remaining points must be inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    out_closest[f] = vertices[triangle_face[0]] + ab * v + ac * w


@wp.func
def face_centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32) -> wp.vec3:
    # Barycentre of face ``f``: the mean of its three corners.
    return (
        vertices[faces[f * 3 + 0]] + vertices[faces[f * 3 + 1]] + vertices[faces[f * 3 + 2]]
    ) / 3.0


@wp.kernel
def face_centroids(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_centroids: wp.array[wp.vec3]
) -> None:
    f = int(wp.tid())
    out_centroids[f] = face_centroid(vertices, faces, wp.int32(f))


@wp.func
def face_gradient(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    f: wp.int32,
) -> wp.vec3d:
    # Gradient of a per-vertex scalar field inside face ``f``, in the face's plane:
    #   grad = 1/(2A) * sum_k values_k * (n x e_k^opp),   e_k^opp the CCW edge opposite corner k.
    #
    # Accumulated in float64. The fields this serves (diffused heat, geodesic distance) decay
    # exponentially and a float32 sum of the cross products loses the far field, so the geometry is
    # promoted rather than the result being widened after the fact.
    #
    # A degenerate face contributes nothing and returns the zero vector.
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    n = to_vec3d(normals[f])
    area = wp.float64(areas[f])

    grad = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if area > wp.float64(0.0):
        grad = (
            values[i0] * wp.cross(n, v2 - v1)
            + values[i1] * wp.cross(n, v0 - v2)
            + values[i2] * wp.cross(n, v1 - v0)
        ) / (wp.float64(2.0) * area)
    return grad


@wp.func
def face_unit_gradient(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    f: wp.int32,
) -> wp.vec3d:
    # The direction of ``face_gradient``. ``normalize`` returns the zero vector for a zero-length
    # gradient (Warp's ``kEps`` is 0), so a degenerate face and a constant field both give zero.
    return wp.normalize(face_gradient(vertices, faces, normals, areas, values, f))


@wp.kernel
def face_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    out_gradients: wp.array[wp.vec3d],
) -> None:
    f = int(wp.tid())
    out_gradients[f] = face_gradient(vertices, faces, normals, areas, values, wp.int32(f))
