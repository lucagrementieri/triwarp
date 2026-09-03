import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels import triangles as kernel_triangles
from triwarp.kernels.array import OverloadTable, declare_map_signatures, map_probe, map_probe_single
from triwarp.kernels.predicates import triangles_intersect

SLICE_SIGN_INSIDE = wp.constant(wp.int32(-1))
SLICE_SIGN_OUTSIDE = wp.constant(wp.int32(1))
SLICE_SIGN_ON_PLANE = wp.constant(wp.int32(0))
EDGE_DENOM_EPSILON = wp.constant(wp.float32(1e-12))

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
    plane_normal: wp.vec3, plane_origin: wp.vec3, p0: wp.vec3, p1: wp.vec3, line_segments: wp.bool
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
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
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
        p_a, valid_a = plane_with_line(plane_normal, plane_origin, unique_v, va, False)
        p_b, valid_b = plane_with_line(plane_normal, plane_origin, unique_v, vb, False)
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
        hit, valid = plane_with_line(plane_normal, plane_origin, va, vb, False)
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
def mesh_with_plane_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    out_valid: wp.array[wp.bool],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    f = wp.int32(wp.tid())
    i0 = faces[f * 3]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    s0 = kernel_array.sign_with_tolerance(vertex_dots[i0], TOLERANCE_MERGE_CONSTANT)
    s1 = kernel_array.sign_with_tolerance(vertex_dots[i1], TOLERANCE_MERGE_CONSTANT)
    s2 = kernel_array.sign_with_tolerance(vertex_dots[i2], TOLERANCE_MERGE_CONSTANT)
    valid, p0, p1 = mesh_with_plane_segment_for_face(
        plane_normal, plane_origin, v0, v1, v2, s0, s1, s2
    )
    out_valid[f] = valid
    out_segments[f, 0] = p0
    out_segments[f, 1] = p1


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
    order_x, _order_y, order_z = vec3_argsort(minors)
    minor_x = minors[order_x]
    minor_z = minors[order_z]
    i = order_z
    if minor_z >= -minor_x:
        i = order_z
    else:
        i = order_x
    offset_minors = wp.cross(start2 - start1, direction2)
    numerator = offset_minors[i]
    denominator = minors[i]
    return numerator / denominator


@wp.func
def triangles_share_vertex(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> wp.bool:
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
def triangle_intersection_segment(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, b0: wp.vec3, b1: wp.vec3, b2: wp.vec3
) -> tuple[wp.bool, wp.vec3, wp.vec3]:
    edge0 = a1 - a0
    edge1 = a2 - a0
    normal = wp.normalize(wp.cross(edge0, edge1))

    other_edge0 = b1 - b0
    other_edge1 = b2 - b0
    other_normal = wp.normalize(wp.cross(other_edge0, other_edge1))

    proj1 = wp.vec3(
        wp.dot(other_normal, a0 - b0), wp.dot(other_normal, a1 - b0), wp.dot(other_normal, a2 - b0)
    )
    proj2 = wp.vec3(wp.dot(normal, b0 - a0), wp.dot(normal, b1 - a0), wp.dot(normal, b2 - a0))

    order1_x, order1_y, order1_z = vec3_argsort(proj1)
    proj1a = proj1[order1_x]
    proj1c = proj1[order1_z]
    proj1b = proj1[order1_y]

    va_min = vertex_at(order1_x, a0, a1, a2)
    va_max = vertex_at(order1_z, a0, a1, a2)
    line_origin = (va_min * proj1c - va_max * proj1a) / (proj1c - proj1a)
    line_direction = wp.cross(normal, other_normal)

    edge_direction = vertex_at(
        wp.where(proj1b >= wp.float32(0.0), order1_x, order1_z), a0, a1, a2
    ) - vertex_at(order1_y, a0, a1, a2)
    t2 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order1_y, a0, a1, a2), edge_direction
    )

    if t2 > wp.float32(0.0):
        interval = wp.vec2(wp.float32(0.0), t2)
    else:
        interval = wp.vec2(t2, wp.float32(0.0))

    order2_x, order2_y, order2_z = vec3_argsort(proj2)
    edge_direction = vertex_at(order2_z, b0, b1, b2) - vertex_at(order2_x, b0, b1, b2)
    s1 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order2_x, b0, b1, b2), edge_direction
    )
    edge_direction = vertex_at(
        wp.where(proj2[order2_y] >= wp.float32(0.0), order2_x, order2_z), b0, b1, b2
    ) - vertex_at(order2_y, b0, b1, b2)
    s2 = intersection_line_coordinate(
        line_origin, line_direction, vertex_at(order2_y, b0, b1, b2), edge_direction
    )

    if s1 <= s2:
        other_interval = wp.vec2(s1, s2)
    else:
        other_interval = wp.vec2(s2, s1)

    interval = interval_intersect(interval, other_interval)

    p0 = line_origin + interval[0] * line_direction
    p1 = line_origin + interval[1] * line_direction
    return True, p0, p1


@wp.kernel
def expand_query_target_pairs(
    offsets: wp.array[wp.int32],
    hit_counts: wp.array[wp.int32],
    target_indices: wp.array[wp.int32],
    out_pairs: wp.array2d[wp.int32],
) -> None:
    q = wp.int32(wp.tid())
    start = offsets[q]
    count = hit_counts[q]
    i = wp.int32(0)
    w = start
    while i < count:
        out_pairs[w, 0] = q
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
    tid = wp.int32(wp.tid())
    qa, qb, qc = kernel_triangles.face_vertices(query_vertices, query_faces, pairs[tid, 0])
    ta, tb, tc = kernel_triangles.face_vertices(target_vertices, target_faces, pairs[tid, 1])
    if triangles_share_vertex(qa, qb, qc, ta, tb, tc):
        out_valid[tid] = False
        return
    out_valid[tid] = triangles_intersect(qa, qb, qc, ta, tb, tc)


@wp.kernel
def swap_pair_columns(pairs: wp.array2d[wp.int32], out_pairs: wp.array2d[wp.int32]) -> None:
    # Put a colliding pair back in the caller's (a, b) order. The broad phase queries the *larger*
    # mesh's faces against the smaller one's BVH, so which input is the query depends on the face
    # counts and the pair columns come out in that order rather than the caller's.
    i = wp.int32(wp.tid())
    out_pairs[i, 0] = pairs[i, 1]
    out_pairs[i, 1] = pairs[i, 0]


@wp.kernel
def mark_pair_masks(
    pairs: wp.array2d[wp.int32], out_mask_a: wp.array[wp.bool], out_mask_b: wp.array[wp.bool]
) -> None:
    # One mask per mesh from the pair list. Written as a kernel rather than two
    # ``scatter.mark_membership_mask`` calls over ``pairs[:, k]`` because such a column is a
    # *strided* view, and Warp's Python-scope gather reads an index buffer as if contiguous
    # (CLAUDE.md section 4) -- it would silently mark the wrong faces.
    i = wp.int32(wp.tid())
    out_mask_a[pairs[i, 0]] = True
    out_mask_b[pairs[i, 1]] = True


@wp.kernel
def triangle_pair_segments(
    query_vertices: wp.array[wp.vec3],
    query_faces: wp.array[wp.int32],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32],
    pairs: wp.array2d[wp.int32],
    out_segments: wp.array2d[wp.vec3],
) -> None:
    tid = wp.int32(wp.tid())
    qa, qb, qc = kernel_triangles.face_vertices(query_vertices, query_faces, pairs[tid, 0])
    ta, tb, tc = kernel_triangles.face_vertices(target_vertices, target_faces, pairs[tid, 1])
    valid, p0, p1 = triangle_intersection_segment(qa, qb, qc, ta, tb, tc)
    if valid:
        out_segments[tid, 0] = p0
        out_segments[tid, 1] = p1


@wp.kernel
def segment_nondegenerate(segments: wp.array2d[wp.vec3], out_valid: wp.array[wp.bool]) -> None:
    tid = wp.int32(wp.tid())
    p0 = segments[tid, 0]
    p1 = segments[tid, 1]
    out_valid[tid] = wp.length(p1 - p0) > TOLERANCE_MERGE_CONSTANT


@wp.func
def find_corner_with_sign(s0: wp.int32, s1: wp.int32, s2: wp.int32, sign: wp.int32) -> wp.int32:
    # The corner carrying ``sign``, for the two cut cases where exactly one does. Corner 2 is the
    # fallthrough rather than a third test: the caller has already established that one of the three
    # matches.
    if s0 == sign:
        return wp.int32(0)
    if s1 == sign:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def edge_level_crossing(
    origin: wp.vec3, dest: wp.vec3, value_origin: wp.float32, value_dest: wp.float32
) -> wp.vec3:
    # Where the field crosses zero along one edge, from the two endpoint values. For a plane's
    # signed distance this is ``edge_plane_intersection``'s algebra with the dot products already
    # taken: ``dot(o - a, n) / dot(b - a, n)`` is ``va / (va - vb)``.
    denominator = value_origin - value_dest
    if denominator == wp.float32(0.0):
        denominator = EDGE_DENOM_EPSILON
    return origin + (dest - origin) * (value_origin / denominator)


# How a face meets the level set. The three kept classes are numbered in the order the fused
# compaction below lays them out, so a class is also its block index plus one. ``ON_PLANE`` is a
# provisional answer that ``resolve_on_plane_faces`` turns into ``INSIDE`` or ``DROP``.
SLICE_CLASS_DROP = wp.constant(wp.int32(0))
SLICE_CLASS_INSIDE = wp.constant(wp.int32(1))
SLICE_CLASS_CUT_QUAD = wp.constant(wp.int32(2))
SLICE_CLASS_CUT_TRI = wp.constant(wp.int32(3))
SLICE_CLASS_ON_PLANE = wp.constant(wp.int32(4))

# Kept classes, i.e. the number of blocks the flag buffer carries.
SLICE_CLASSES = 3


@wp.func
def face_level_set_signs(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    f: wp.int32,
    out_signs: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32]:
    # One face's three level-set signs, stored, plus the two sums every classifier decides on.
    #
    # ``SLICE_SIGN_INSIDE`` (-1) is the ``>= isovalue`` side, so the tolerance sign is negated and a
    # value on the level set counts as positive. ``signs_sum`` separates "all one side" from "cut"
    # and ``signs_asum`` counts the corners strictly off the plane; between them they name every
    # case both classifiers below distinguish.
    #
    # Shared by ``classify_faces_for_slice`` and ``classify_faces_for_split``, which differ only in
    # the 3-way vs 4-way decision they map these two sums onto -- that is the genuine variation, and
    # the nine statements above it were a copy. Kept here rather than in ``kernels/predicates.py``
    # because it writes an output array: that module holds pure predicates, and both callers of this
    # one are in this file.
    i0, i1, i2 = kernel_triangles.corner_triple(faces, f)
    s0 = -kernel_array.sign_with_tolerance(vertex_dots[i0], TOLERANCE_MERGE_CONSTANT)
    s1 = -kernel_array.sign_with_tolerance(vertex_dots[i1], TOLERANCE_MERGE_CONSTANT)
    s2 = -kernel_array.sign_with_tolerance(vertex_dots[i2], TOLERANCE_MERGE_CONSTANT)
    out_signs[f, 0] = s0
    out_signs[f, 1] = s1
    out_signs[f, 2] = s2
    return s0 + s1 + s2, wp.abs(s0) + wp.abs(s1) + wp.abs(s2)


@wp.kernel
def classify_faces_for_slice(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_classes: wp.array[wp.int32],
    out_signs: wp.array2d[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    signs_sum, signs_asum = face_level_set_signs(faces, vertex_dots, f, out_signs)

    # A face lying in the plane also satisfies the "wholly inside" test (both sums are zero), so
    # it has to be tested first -- its side is decided from its normal, not from its vertices.
    face_class = SLICE_CLASS_DROP
    if signs_asum == wp.int32(0):
        face_class = SLICE_CLASS_ON_PLANE
    elif signs_sum == -signs_asum:
        face_class = SLICE_CLASS_INSIDE
    elif signs_asum >= wp.int32(2) and wp.abs(signs_sum) <= wp.int32(1):
        if signs_sum < wp.int32(0):
            face_class = SLICE_CLASS_CUT_QUAD
        else:
            face_class = SLICE_CLASS_CUT_TRI
    out_classes[f] = face_class


@wp.kernel
def resolve_on_plane_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    out_classes: wp.array[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    if out_classes[f] != SLICE_CLASS_ON_PLANE:
        return
    normal, area = kernel_triangles.face_normals_and_area(vertices, faces, f)
    if area <= TOLERANCE_ZERO_CONSTANT:
        out_classes[f] = SLICE_CLASS_DROP
        return
    if wp.dot(normal, plane_normal) < wp.float32(0.0):
        out_classes[f] = SLICE_CLASS_INSIDE
    else:
        out_classes[f] = SLICE_CLASS_DROP


@wp.kernel
def slice_class_flags(
    classes: wp.array[wp.int32],
    n_faces: wp.int32,
    n_classes: wp.int32,
    out_flags: wp.array[wp.int32],
) -> None:
    # Selection flags for all three kept classes at once, as three ``n_faces``-long blocks of one
    # buffer. Scanning that buffer once compacts the three classes into one index array with the
    # blocks contiguous, so the whole partition costs one scan and one host readback rather than
    # three of each. Every thread writes all three of its slots, which is what keeps the buffer
    # from needing a memset first. ``n_classes`` is an argument rather than a module constant so
    # that the both-sides split, which keeps four classes where the clip keeps three, shares it.
    f = wp.int32(wp.tid())
    face_class = classes[f]
    for block in range(n_classes):
        # ``wp.int32(0)``, not ``0``: the loop bound is now an argument rather than a module
        # constant, so the loop is dynamic and a bare literal is a constant Warp refuses to
        # mutate inside one.
        flag = wp.int32(0)
        if face_class == block + 1:
            flag = wp.int32(1)
        out_flags[block * n_faces + f] = flag


@wp.kernel
def slice_class_counts(
    inclusive: wp.array[wp.int32],
    n_faces: wp.int32,
    n_classes: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    # Per-class counts from the block ends of the inclusive scan: block ``b``'s own count is its
    # running total minus the previous block's. One launch so the host reads a few bytes once.
    previous = wp.int32(0)  # dynamic loop below: a bare literal would be a constant (see above)
    for block in range(n_classes):
        total = inclusive[(block + 1) * n_faces - 1]
        out_counts[block] = total - previous
        previous = total


@wp.kernel
def scatter_slice_class(
    flags: wp.array[wp.int32],
    inclusive: wp.array[wp.int32],
    n_faces: wp.int32,
    out_indices: wp.array[wp.int32],
) -> None:
    # ``scatter.scatter_index_where`` over the blocked flag buffer, except that the value written is
    # the *face* index rather than the flag index, so each block comes out addressing faces.
    t = wp.int32(wp.tid())
    if flags[t] != 0:
        out_indices[inclusive[t] - 1] = t % n_faces


@wp.func
def canonical_edge_crossing(
    vertices: wp.array[wp.vec3], values: wp.array[wp.float32], i: wp.int32, j: wp.int32
) -> wp.vec3:
    # Always interpolate from the lower-numbered endpoint, so the two faces sharing a cut edge
    # evaluate the identical expression and land on **bitwise equal** points. That is what lets
    # ``clip_mesh_with_field(cap=True)`` weld the section rim exactly rather than by tolerance.
    a = wp.min(i, j)
    b = wp.max(i, j)
    return edge_level_crossing(vertices[a], vertices[b], values[a], values[b])


@wp.kernel
def edge_level_crossings(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_points: wp.array2d[wp.vec3],
) -> None:
    # The three edge crossings of one cut face, from the per-vertex field the classifier already
    # signed -- so the plane case reuses the dot products rather than recomputing them per edge.
    tid = wp.int32(wp.tid())
    face_index = face_indices[tid]
    i0, i1, i2 = kernel_triangles.corner_triple(faces, face_index)
    out_points[tid, 0] = canonical_edge_crossing(vertices, vertex_values, i0, i1)
    out_points[tid, 1] = canonical_edge_crossing(vertices, vertex_values, i1, i2)
    out_points[tid, 2] = canonical_edge_crossing(vertices, vertex_values, i2, i0)


@wp.func
def emit_cut_vertices(
    edge_points: wp.array2d[wp.vec3],
    cut: wp.int32,
    edge_0: wp.int32,
    edge_1: wp.int32,
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
) -> tuple[wp.int32, wp.int32]:
    # Write the two new vertices cut face ``cut`` contributes, and return their indices in the
    # *concatenated* buffer, where the new block starts at ``vertex_base``.
    #
    # Every cut face emits exactly two, whichever side is alone in sign: a quad cut splits into two
    # triangles and a corner cut into one, but both are bounded by the same two edge crossings. So
    # the slot is a function of the thread alone and neither kernel needs a counter -- which is the
    # invariant this function exists to state, and the reason ``2 * cut`` is now computed once
    # rather than seventeen times across the two callers.
    slot = wp.int32(2) * cut
    out_new_verts[slot] = edge_points[cut, edge_0]
    out_new_verts[slot + wp.int32(1)] = edge_points[cut, edge_1]
    return vertex_base + slot, vertex_base + slot + wp.int32(1)


@wp.kernel
def emit_quad_cut(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    edge_points: wp.array2d[wp.vec3],
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    face_index = face_indices[tid]
    base = face_index * wp.int32(3)
    s0, s1, s2 = kernel_triangles.row_triple(face_signs, face_index)
    outside = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_OUTSIDE)
    inside_a = (outside + wp.int32(1)) % wp.int32(3)
    inside_b = (outside + wp.int32(2)) % wp.int32(3)
    v_a = faces[base + inside_a]
    v_b = faces[base + inside_b]
    edge_a = (outside + wp.int32(2)) % wp.int32(3)
    edge_b = outside
    new_i0, new_i1 = emit_cut_vertices(edge_points, tid, edge_a, edge_b, vertex_base, out_new_verts)
    # The quad becomes two triangles, in the same pair of rows the two vertices went into.
    row = wp.int32(2) * tid
    out_new_faces[row, 0] = v_a
    out_new_faces[row, 1] = v_b
    out_new_faces[row, 2] = new_i0
    out_new_faces[row + wp.int32(1), 0] = new_i0
    out_new_faces[row + wp.int32(1), 1] = new_i1
    out_new_faces[row + wp.int32(1), 2] = v_a


@wp.kernel
def emit_tri_cut(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    edge_points: wp.array2d[wp.vec3],
    vertex_base: wp.int32,
    out_new_verts: wp.array[wp.vec3],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    face_index = face_indices[tid]
    base = face_index * wp.int32(3)
    s0, s1, s2 = kernel_triangles.row_triple(face_signs, face_index)
    inside = find_corner_with_sign(s0, s1, s2, SLICE_SIGN_INSIDE)
    v_inside = faces[base + inside]
    edge_0 = inside
    edge_1 = (inside + wp.int32(2)) % wp.int32(3)
    new_i0, new_i1 = emit_cut_vertices(edge_points, tid, edge_0, edge_1, vertex_base, out_new_verts)
    # The corner stays one triangle, so this kernel writes one face row per cut, not two.
    out_new_faces[tid, 0] = v_inside
    out_new_faces[tid, 1] = new_i0
    out_new_faces[tid, 2] = new_i1


# Face classes for the both-sides split. Numbered from 1 contiguously because
# ``slice_class_flags`` maps class ``c`` to block ``c - 1``; class 0 is "not selected", which this
# taxonomy never needs since a split keeps every face.
SPLIT_CLASS_POSITIVE = wp.constant(wp.int32(1))
SPLIT_CLASS_NEGATIVE = wp.constant(wp.int32(2))
SPLIT_CLASS_CUT_EDGES = wp.constant(wp.int32(3))
SPLIT_CLASS_CUT_CORNER = wp.constant(wp.int32(4))
SPLIT_CLASSES = 4


@wp.kernel
def classify_faces_for_split(
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_classes: wp.array[wp.int32],
    out_signs: wp.array2d[wp.int32],
) -> None:
    # Four classes rather than the clip's three, because a split keeps both sides and so has to
    # tell the two *uncut* sides apart -- and because a face with one corner exactly on the level
    # set splits into two triangles, not three. The sign convention is ``face_level_set_signs``'.
    f = wp.int32(wp.tid())
    signs_sum, signs_asum = face_level_set_signs(faces, vertex_dots, f, out_signs)
    if signs_sum == -signs_asum:
        # Every corner on the positive side, or all three exactly on the level set -- which counts
        # as positive, so the face is kept whole rather than cut along itself.
        out_classes[f] = SPLIT_CLASS_POSITIVE
    elif signs_sum == signs_asum:
        out_classes[f] = SPLIT_CLASS_NEGATIVE
    elif signs_asum == wp.int32(3):
        out_classes[f] = SPLIT_CLASS_CUT_EDGES
    else:
        # One corner exactly on the level set and the other two on opposite sides: a single edge
        # crossing, joined to that corner.
        out_classes[f] = SPLIT_CLASS_CUT_CORNER


@wp.func
def split_lone_corner(s0: wp.int32, s1: wp.int32, s2: wp.int32) -> wp.int32:
    # The corner alone in sign, which is the apex of the one-triangle side of an edge-to-edge cut.
    if s1 == s2:
        return wp.int32(0)
    if s0 == s2:
        return wp.int32(1)
    return wp.int32(2)


@wp.kernel
def level_set_edge_vertices(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    crossed_edge_indices: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
) -> None:
    # One crossing point per crossed *edge*, not per crossed face-corner. That is what makes the
    # split watertight: both faces sharing the edge address the same new vertex, where a per-face
    # crossing would leave two coincident copies and a seam of loose edges (which is exactly why
    # ``clip_mesh_with_field(cap=True)`` has to weld before it can fill).
    i = wp.int32(wp.tid())
    e = crossed_edge_indices[i]
    out_points[i] = canonical_edge_crossing(
        vertices, vertex_dots, unique_edges[e, 0], unique_edges[e, 1]
    )


@wp.func
def split_edge_vertex(
    halfedge_edges: wp.array[wp.int32],
    edge_vertex_rank: wp.array[wp.int32],
    vertex_base: wp.int32,
    halfedge: wp.int32,
) -> wp.int32:
    # The new vertex index sitting on a face's edge ``k``, which is halfedge ``3 * f + k``.
    return vertex_base + edge_vertex_rank[halfedge_edges[halfedge]]


@wp.kernel
def emit_split_cut_edges(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    halfedge_edges: wp.array[wp.int32],
    edge_vertex_rank: wp.array[wp.int32],
    vertex_base: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # The generic cut: the level set enters through one edge and leaves through another, so the face
    # becomes a corner triangle plus a quad -- three triangles sharing the two crossing vertices.
    # Windings match ``emit_tri_cut`` and ``emit_quad_cut``, which emit these same triangles one
    # side at a time; the difference is that both sides are kept here.
    tid = wp.int32(wp.tid())
    face_index = face_indices[tid]
    base = face_index * wp.int32(3)
    s0, s1, s2 = kernel_triangles.row_triple(face_signs, face_index)
    lone = split_lone_corner(s0, s1, s2)
    next_corner = (lone + wp.int32(1)) % wp.int32(3)
    last_corner = (lone + wp.int32(2)) % wp.int32(3)
    # ``p0`` on the edge leaving the lone corner, ``p1`` on the edge arriving at it.
    p0 = split_edge_vertex(halfedge_edges, edge_vertex_rank, vertex_base, base + lone)
    p1 = split_edge_vertex(halfedge_edges, edge_vertex_rank, vertex_base, base + last_corner)
    v_next = faces[base + next_corner]
    v_last = faces[base + last_corner]

    row = wp.int32(3) * tid
    out_new_faces[row, 0] = faces[base + lone]
    out_new_faces[row, 1] = p0
    out_new_faces[row, 2] = p1
    out_new_faces[row + wp.int32(1), 0] = v_next
    out_new_faces[row + wp.int32(1), 1] = v_last
    out_new_faces[row + wp.int32(1), 2] = p1
    out_new_faces[row + wp.int32(2), 0] = p1
    out_new_faces[row + wp.int32(2), 1] = p0
    out_new_faces[row + wp.int32(2), 2] = v_next

    lone_is_positive = face_signs[face_index, lone] == SLICE_SIGN_INSIDE
    out_positive[row] = lone_is_positive
    out_positive[row + wp.int32(1)] = not lone_is_positive
    out_positive[row + wp.int32(2)] = not lone_is_positive


@wp.kernel
def emit_split_cut_corner(
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    face_signs: wp.array2d[wp.int32],
    halfedge_edges: wp.array[wp.int32],
    edge_vertex_rank: wp.array[wp.int32],
    vertex_base: wp.int32,
    out_new_faces: wp.array2d[wp.int32],
    out_positive: wp.array[wp.bool],
) -> None:
    # One corner sits exactly on the level set, so the cut runs from it to the single crossing on
    # the opposite edge: two triangles, no quad. Emitting three the other kernel's way would put a
    # zero-area sliver in the output, which is the only reason this class exists separately.
    tid = wp.int32(wp.tid())
    face_index = face_indices[tid]
    base = face_index * wp.int32(3)
    _s0, s1, s2 = kernel_triangles.row_triple(face_signs, face_index)
    on_level = wp.int32(0)  # corner 0 by elimination: exactly one of the three is zero here
    if s1 == wp.int32(0):
        on_level = wp.int32(1)
    elif s2 == wp.int32(0):
        on_level = wp.int32(2)
    next_corner = (on_level + wp.int32(1)) % wp.int32(3)
    last_corner = (on_level + wp.int32(2)) % wp.int32(3)
    crossing = split_edge_vertex(halfedge_edges, edge_vertex_rank, vertex_base, base + next_corner)

    row = wp.int32(2) * tid
    out_new_faces[row, 0] = faces[base + on_level]
    out_new_faces[row, 1] = faces[base + next_corner]
    out_new_faces[row, 2] = crossing
    out_new_faces[row + wp.int32(1), 0] = faces[base + on_level]
    out_new_faces[row + wp.int32(1), 1] = crossing
    out_new_faces[row + wp.int32(1), 2] = faces[base + last_corner]
    out_positive[row] = face_signs[face_index, next_corner] == SLICE_SIGN_INSIDE
    out_positive[row + wp.int32(1)] = face_signs[face_index, last_corner] == SLICE_SIGN_INSIDE


@wp.kernel
def plane_crossed_edge_mask(
    unique_edges: wp.array2d[wp.int32],
    vertex_dots: wp.array[wp.float32],
    tolerance: wp.float32,
    out_crossed: wp.array[wp.bool],
) -> None:
    # An edge needs a new vertex only when the plane passes through its *interior*: an endpoint
    # already in the plane (within ``tolerance``) serves as the crossing itself, so splitting there
    # would emit a duplicate. Strict opposite signs is therefore the condition, and it is also why
    # at most two of a triangle's three edges can ever be flagged -- two of three vertices always
    # share a sign, so their edge is never crossed and ``emit_size_faces``' 3-split branch is
    # unreachable from here.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    sign_a = kernel_array.sign_with_tolerance(vertex_dots[a], tolerance)
    sign_b = kernel_array.sign_with_tolerance(vertex_dots[b], tolerance)
    out_crossed[e] = sign_a * sign_b < wp.int32(0)


@wp.kernel
def plane_edge_crossing_points(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    crossed: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    out_points: wp.array[wp.vec3],
) -> None:
    # ``remesh.fill_edge_midpoints`` with the plane crossing in place of the midpoint. One point per
    # *unique edge* rather than per cut face, which is what makes the split crack-free where
    # ``_clip_with_vertex_field`` is cracked: the two faces sharing the edge read one index.
    e = wp.int32(wp.tid())
    if crossed[e]:
        out_points[offsets[e]] = canonical_edge_crossing(
            vertices, vertex_dots, unique_edges[e, 0], unique_edges[e, 1]
        )


@wp.kernel
def label_faces_by_plane_side(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane_normal: wp.vec3,
    tolerance: wp.float32,
    out_above: wp.array[wp.bool],
) -> None:
    # After the split no face straddles the plane, so the *largest-magnitude* vertex dot decides the
    # side for the whole face. Reading the extremum rather than a sum or a centroid keeps a sliver
    # face -- two crossing vertices at dot 0 and one real vertex just off the plane -- on the side
    # its real vertex is on, where a centroid would divide the offset by three and a sum would let
    # two rounding-level zeros outvote it.
    f = wp.int32(wp.tid())
    extreme = wp.float32(0.0)
    for corner in range(3):
        value = vertex_dots[faces[f * 3 + corner]]
        if wp.abs(value) > wp.abs(extreme):
            extreme = value

    side = kernel_array.sign_with_tolerance(extreme, tolerance)
    if side != wp.int32(0):
        out_above[f] = side > wp.int32(0)
        return

    # The face lies *in* the plane, so its vertices give no answer. Decide from its own normal, the
    # same tie-break ``resolve_on_plane_faces`` applies, so that the ``above`` block of this split
    # holds exactly the faces ``slice_mesh_with_plane`` keeps.
    normal, area = kernel_triangles.face_normals_and_area(vertices, faces, f)
    if area <= TOLERANCE_ZERO_CONSTANT:
        out_above[f] = True
    else:
        out_above[f] = wp.dot(normal, plane_normal) < wp.float32(0.0)


@wp.func
def crossing_point(
    value_from: wp.Float, value_to: wp.Float, point_from: wp.vec3, point_to: wp.vec3
) -> wp.vec3:
    # Linear crossing of the zero level set along one edge. The caller only passes edges whose
    # endpoints straddle zero with the ``>= 0`` convention below, so the denominator is never zero.
    t = wp.float32(value_from / (value_from - value_to))
    return point_from + t * (point_to - point_from)


@wp.kernel
def marching_triangles_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.Float],
    edge_ids: wp.array[wp.int32],
    out_valid: wp.array[wp.bool],
    out_segments: wp.array2d[wp.vec3],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per face. ``values`` is the field with the isovalue already subtracted, and a value
    # of exactly zero counts as positive, so every cut face has exactly one vertex alone in sign and
    # yields exactly one segment: the two edges incident to that vertex are the crossed ones.
    f = wp.int32(wp.tid())
    i0, i1, i2 = kernel_triangles.corner_triple(faces, f)
    d0 = values[i0]
    d1 = values[i1]
    d2 = values[i2]
    p0 = d0 >= values.dtype(0.0)
    p1 = d1 >= values.dtype(0.0)
    p2 = d2 >= values.dtype(0.0)

    if p0 == p1 and p1 == p2:
        out_valid[f] = False
        return

    # Local index of the vertex whose sign differs from the other two.
    lone = wp.int32(0)
    if p1 != p0 and p1 != p2:
        lone = wp.int32(1)
    elif p2 != p0 and p2 != p1:
        lone = wp.int32(2)
    next_index = (lone + wp.int32(1)) % wp.int32(3)
    prev_index = (lone + wp.int32(2)) % wp.int32(3)

    vertex_lone = faces[f * 3 + lone]
    vertex_next = faces[f * 3 + next_index]
    vertex_prev = faces[f * 3 + prev_index]
    value_lone = values[vertex_lone]

    point_next = crossing_point(
        value_lone, values[vertex_next], vertices[vertex_lone], vertices[vertex_next]
    )
    point_prev = crossing_point(
        value_lone, values[vertex_prev], vertices[vertex_lone], vertices[vertex_prev]
    )
    # Halfedge ``3f + k`` spans local corners ``k`` and ``k + 1``, so the edge from ``lone`` to the
    # next corner is halfedge ``lone`` and the edge from the previous corner to ``lone`` is halfedge
    # ``prev_index``; their unique-edge ids are what stitches segments into curves.
    edge_next = edge_ids[f * 3 + lone]
    edge_prev = edge_ids[f * 3 + prev_index]

    # Orient the segment so the region where the field exceeds the isovalue lies to its left, with
    # the face normal as up. That makes the crossing shared by two faces an outgoing endpoint of one
    # and an incoming endpoint of the other, which is what lets the curves be linked at all.
    out_valid[f] = True
    lone_is_positive = p0
    if lone == wp.int32(1):
        lone_is_positive = p1
    elif lone == wp.int32(2):
        lone_is_positive = p2

    if lone_is_positive:
        out_segments[f, 0] = point_next
        out_segments[f, 1] = point_prev
        out_edges[f, 0] = edge_next
        out_edges[f, 1] = edge_prev
    else:
        out_segments[f, 0] = point_prev
        out_segments[f, 1] = point_next
        out_edges[f, 0] = edge_prev
        out_edges[f, 1] = edge_next


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured at 2 overloads across **4** module loads, on a 15-kernel module.
#
# Only the scalar field being contoured is generic: ``marching_triangles`` accepts a ``wp.float32``
# or ``wp.float64`` per-vertex field (the heat solvers produce the latter), while the geometry it
# writes stays ``wp.vec3``.
# The concrete handle keyed by the field dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable].
MARCHING_TRIANGLES_SEGMENTS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global MARCHING_TRIANGLES_SEGMENTS
    MARCHING_TRIANGLES_SEGMENTS = OverloadTable(
        marching_triangles_segments,
        {
            d: [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[d],
                wp.array[wp.int32],
                wp.array[wp.bool],
                wp.array2d[wp.vec3],
                wp.array2d[wp.int32],
            ]
            for d in (wp.float32, wp.float64)
        },
    )


_register_overloads()


@wp.func
def is_positive_split_class(face_class: wp.int32) -> wp.bool:
    """Side label for an *uncut* face, so the no-crossing path needs no second classifier."""
    return face_class == SPLIT_CLASS_POSITIVE


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (
                plane_with_line,
                (wp.vec3(), wp.vec3(), dense(wp.vec3), dense(wp.vec3), wp.bool(True)),
                [wp.vec3, wp.bool],
            ),
            (
                plane_with_line,
                (wp.vec3(), wp.vec3(), single(wp.vec3), single(wp.vec3), wp.bool(True)),
                [wp.vec3, wp.bool],
            ),
        ]
    )


_declare_map_kernels()
