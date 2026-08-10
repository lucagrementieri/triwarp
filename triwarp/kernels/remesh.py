import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import binary_search_sorted_contains, to_vec2d, to_vec3, to_vec3d
from triwarp.kernels.grouping import hash_slot, pack_edge_key
from triwarp.kernels.predicates import (
    delone_metrics,
    dihedral_angle,
    is_unfold_quadrangle_convex,
    mincircle_diameter_sq,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    triangle_normal,
)
from triwarp.kernels.triangles import face_vertices_vec3d, triangle_quality
from triwarp.kernels.voxels import voxel_cell

# Delaunay / Delone edge-flip constants (ported from MRMeshDelone.cpp). The flip predicate
# runs in float64: MeshLib deliberately widens to double because circumcircle diameters of
# near-degenerate triangles have too large a rounding error in float32 (infinite flip loops).
DELONE_CRITICAL_DOT = wp.constant(wp.float64(-0.9))
DELONE_EPS = wp.constant(wp.float64(1e-7))
NO_ANGLE_CHANGE_LIMIT = wp.constant(wp.float64(6.283185307179586))  # 2*pi (NoAngleChangeLimit)
F32_LARGE = wp.constant(wp.float32(3.0e38))  # "disabled gate" sentinel (~FLT_MAX)

# Loop subdivision stencil weights. The even-vertex relaxation uses Warren's beta rather than Loop's
# original trigonometric weight, which is the choice ``igl::loop`` makes; see `loop_even_positions`.
LOOP_ODD_ENDPOINT = wp.constant(wp.float32(3.0 / 8.0))
LOOP_ODD_OPPOSITE = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BOUNDARY_SELF = wp.constant(wp.float32(3.0 / 4.0))
LOOP_BOUNDARY_NEIGHBOR = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BETA_VALENCE_3 = wp.constant(wp.float32(3.0 / 16.0))
LOOP_BETA_NUMERATOR = wp.constant(wp.float32(3.0 / 8.0))


@wp.func
def edge_midpoint(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.vec3:
    v0 = vertices[unique_edges[e, 0]]
    v1 = vertices[unique_edges[e, 1]]
    return wp.lerp(v0, v1, wp.float32(0.5))


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = int(wp.tid())
    out_midpoints[k] = edge_midpoint(vertices, unique_edges, wp.int32(k))


@wp.func
def split_face_four(fv: wp.vec3i, mv: wp.vec3i) -> tuple[wp.vec3i, wp.vec3i, wp.vec3i, wp.vec3i]:
    # 1 -> 4 loop-subdivision template: three corner triangles, then the central triangle.
    t0 = wp.vec3i(fv[0], mv[0], mv[2])
    t1 = wp.vec3i(mv[0], fv[1], mv[1])
    t2 = wp.vec3i(mv[2], mv[1], fv[2])
    t3 = wp.vec3i(mv[0], mv[1], mv[2])
    return t0, t1, t2, t3


@wp.kernel
def subdivide_faces(
    faces: wp.array[wp.int32], mid_idx: wp.array2d[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = int(wp.tid())
    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(mid_idx[f, 0], mid_idx[f, 1], mid_idx[f, 2])
    t0, t1, t2, t3 = split_face_four(fv, mv)
    base = f * 12
    out_faces[base + 0] = t0[0]
    out_faces[base + 1] = t0[1]
    out_faces[base + 2] = t0[2]
    out_faces[base + 3] = t1[0]
    out_faces[base + 4] = t1[1]
    out_faces[base + 5] = t1[2]
    out_faces[base + 6] = t2[0]
    out_faces[base + 7] = t2[1]
    out_faces[base + 8] = t2[2]
    out_faces[base + 9] = t3[0]
    out_faces[base + 10] = t3[1]
    out_faces[base + 11] = t3[2]


@wp.kernel
def loop_edge_opposites(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    out_opposite_sum: wp.array[wp.vec3],
    out_face_count: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many faces use it, and the sum of the vertices opposite it in each.
    # Corner ``j`` of face ``f`` spans ``(fv[j], fv[j + 1])`` and its opposite vertex is
    # ``fv[j + 2]``, so one pass over the faces gathers both halves of the Loop odd-vertex stencil.
    f = int(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        wp.atomic_add(out_opposite_sum, e, vertices[faces[f * 3 + (j + 2) % 3]])
        wp.atomic_add(out_face_count, e, 1)


@wp.kernel
def loop_vertex_rings(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    out_valence: wp.array[wp.int32],
    out_ring_sum: wp.array[wp.vec3],
    out_boundary_count: wp.array[wp.int32],
    out_boundary_sum: wp.array[wp.vec3],
) -> None:
    # Per vertex: its valence and 1-ring position sum, plus the same two restricted to boundary
    # edges. Driven by the *unique* edge list rather than by the faces, so the valence is the number
    # of distinct neighbours on any input -- the count a per-face pass would have to deduplicate
    # (each neighbour appears twice around an interior vertex but once at a boundary).
    e = int(wp.tid())
    v0 = unique_edges[e, 0]
    v1 = unique_edges[e, 1]
    p0 = vertices[v0]
    p1 = vertices[v1]
    wp.atomic_add(out_valence, v0, 1)
    wp.atomic_add(out_valence, v1, 1)
    wp.atomic_add(out_ring_sum, v0, p1)
    wp.atomic_add(out_ring_sum, v1, p0)
    if edge_face_count[e] == 1:
        wp.atomic_add(out_boundary_count, v0, 1)
        wp.atomic_add(out_boundary_count, v1, 1)
        wp.atomic_add(out_boundary_sum, v0, p1)
        wp.atomic_add(out_boundary_sum, v1, p0)


@wp.kernel
def loop_odd_positions(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    edge_opposite_sum: wp.array[wp.vec3],
    edge_face_count: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Loop's odd (edge) vertices: 3/8 on each endpoint and 1/8 on each of the two opposite vertices.
    e = int(wp.tid())
    endpoints = vertices[unique_edges[e, 0]] + vertices[unique_edges[e, 1]]
    if edge_face_count[e] == 2:
        out_positions[e] = LOOP_ODD_ENDPOINT * endpoints + LOOP_ODD_OPPOSITE * edge_opposite_sum[e]
    else:
        # A boundary edge (one face) or a non-manifold one (three or more): the interior stencil
        # needs exactly two opposite vertices, so both take the midpoint rule instead.
        out_positions[e] = wp.float32(0.5) * endpoints


@wp.kernel
def loop_even_positions(
    vertices: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    ring_sum: wp.array[wp.vec3],
    boundary_count: wp.array[wp.int32],
    boundary_sum: wp.array[wp.vec3],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Loop's even (original) vertices, relaxed towards their 1-ring. Warren's beta -- 3/16 at
    # valence 3 and 3/(8n) above it -- which is the variant ``igl::loop`` uses, not Loop's original
    # trigonometric weight.
    v = int(wp.tid())
    position = vertices[v]
    n = valence[v]
    out_positions[v] = position  # the fallbacks below leave the vertex where it is
    if boundary_count[v] == 2:
        # Boundary vertex: 3/4 of itself, 1/8 of each neighbour along the boundary. Its interior
        # neighbours do not enter, which is what keeps a shared boundary curve identical on both
        # sides of a seam.
        out_positions[v] = LOOP_BOUNDARY_SELF * position + LOOP_BOUNDARY_NEIGHBOR * boundary_sum[v]
    elif boundary_count[v] == 0 and n > 0:
        beta = LOOP_BETA_VALENCE_3
        if n != 3:
            beta = LOOP_BETA_NUMERATOR / wp.float32(n)
        out_positions[v] = (wp.float32(1.0) - wp.float32(n) * beta) * position + beta * ring_sum[v]
    # Anything else keeps the position written above: an isolated vertex with no edges, or a
    # non-manifold boundary vertex where one or three-plus boundary edges meet and neither stencil
    # is defined.


@wp.kernel
def build_midpoint_index(
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_offset: wp.int32,
    out_midpoint_idx: wp.array[wp.int32],
) -> None:
    e = int(wp.tid())
    if long_mask[e]:
        out_midpoint_idx[e] = vertex_offset + offsets[e]
    else:
        out_midpoint_idx[e] = wp.int32(-1)


@wp.kernel
def fill_edge_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_mid: wp.array[wp.vec3],
) -> None:
    e = int(wp.tid())
    if long_mask[e]:
        out_mid[offsets[e]] = edge_midpoint(vertices, unique_edges, wp.int32(e))


@wp.func
def _write_tri(
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
    slot: wp.int32,
    tri: wp.vec3i,
    valid: wp.bool,
    src: wp.int32,
) -> None:
    out_faces[slot, 0] = tri[0]
    out_faces[slot, 1] = tri[1]
    out_faces[slot, 2] = tri[2]
    out_valid[slot] = valid
    out_slot_index[slot] = src


@wp.kernel
def emit_size_faces(
    faces: wp.array[wp.int32],
    face_mid: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    index_in: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
) -> None:
    f = int(wp.tid())
    src = index_in[f]

    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(face_mid[f, 0], face_mid[f, 1], face_mid[f, 2])

    s0 = wp.where(mv[0] >= 0, wp.int32(1), wp.int32(0))
    s1 = wp.where(mv[1] >= 0, wp.int32(1), wp.int32(0))
    s2 = wp.where(mv[2] >= 0, wp.int32(1), wp.int32(0))
    count = s0 + s1 + s2

    # Four output triangle slots; unused slots are marked invalid.
    t0 = wp.vec3i(0, 0, 0)
    t1 = wp.vec3i(0, 0, 0)
    t2 = wp.vec3i(0, 0, 0)
    t3 = wp.vec3i(0, 0, 0)
    n0 = False
    n1 = False
    n2 = False
    n3 = False

    if count == 0:
        # No split edges: the face passes through unchanged.
        t0 = fv
        n0 = True
    elif count == 1:
        # Rotate so the split edge is (a, b); fan its midpoint p to the
        # opposite corner c as [a, p, c], [p, b, c].
        j = wp.int32(0)
        if s1 == 1:
            j = 1
        if s2 == 1:
            j = 2
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        t0 = wp.vec3i(a, p, c)
        n0 = True
        t1 = wp.vec3i(p, b, c)
        n1 = True
    elif count == 2:
        # Rotate so the unsplit edge is (c, a); emit corner triangle [p, b, q]
        # plus the quad (a, p, q, c) cut along its shorter diagonal.
        u = wp.int32(0)
        if s1 == 0:
            u = 1
        if s2 == 0:
            u = 2
        j = (u + 1) % 3
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        q = mv[(j + 1) % 3]
        t0 = wp.vec3i(p, b, q)
        n0 = True
        d_aq = wp.length_sq(vertices[a] - vertices[q])
        d_pc = wp.length_sq(vertices[p] - vertices[c])
        if d_aq <= d_pc:
            t1 = wp.vec3i(a, p, q)
            t2 = wp.vec3i(a, q, c)
        else:
            t1 = wp.vec3i(a, p, c)
            t2 = wp.vec3i(p, q, c)
        n1 = True
        n2 = True
    else:
        # Three split edges: the regular 1 -> 4 split (matches subdivide).
        t0, t1, t2, t3 = split_face_four(fv, mv)
        n0 = True
        n1 = True
        n2 = True
        n3 = True

    base = f * 4
    _write_tri(out_faces, out_valid, out_slot_index, base + 0, t0, n0, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 1, t1, n1, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 2, t2, n2, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 3, t3, n3, src)


# ---------------------------------------------------------------------------
# Region-restricted subdivision helpers
# ---------------------------------------------------------------------------


@wp.kernel
def mark_region_edges(
    region_flags: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_edge_in_region: wp.array[wp.bool],
) -> None:
    # A unique edge is in/on the region boundary if at least one of its incident faces is in
    # the region (MeshLib isInnerOrBdEdge, subdivideBorder default). Benign write race: every
    # thread writing the same slot writes True.
    i = int(wp.tid())
    if region_flags[i // 3] != 0:
        out_edge_in_region[inverse[i]] = wp.bool(True)


@wp.func
def long_region_edge(length: wp.float32, max_edge: wp.float32, in_region: wp.bool) -> wp.bool:
    return in_region and length > max_edge


# ---------------------------------------------------------------------------
# Float64 Delone edge-flip predicate (ported from MRMeshDelone.cpp / MRTriMath.h /
# MRReducePath.cpp). Computed in double precision, matching MeshLib.
# ---------------------------------------------------------------------------


@wp.func
def _segments_dist_sq_d(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d) -> wp.float64:
    # Squared distance between segments [p1,q1] and [p2,q2] (Ericson, clamped closest points).
    eps = wp.float64(1e-30)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    aa = wp.length_sq(d1)
    ee = wp.length_sq(d2)
    f = wp.dot(d2, r)
    s = wp.float64(0.0)
    t = wp.float64(0.0)
    if aa <= eps and ee <= eps:
        return wp.length_sq(p1 - p2)
    if aa <= eps:
        t = wp.clamp(f / ee, wp.float64(0.0), wp.float64(1.0))
    else:
        cc = wp.dot(d1, r)
        if ee <= eps:
            s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
        else:
            bb = wp.dot(d1, d2)
            denom = aa * ee - bb * bb
            if denom != wp.float64(0.0):
                s = wp.clamp((bb * f - cc * ee) / denom, wp.float64(0.0), wp.float64(1.0))
            t = (bb * s + f) / ee
            if t < wp.float64(0.0):
                t = wp.float64(0.0)
                s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
            elif t > wp.float64(1.0):
                t = wp.float64(1.0)
                s = wp.clamp((bb - cc) / aa, wp.float64(0.0), wp.float64(1.0))
    cp1 = p1 + s * d1
    cp2 = p2 + t * d2
    return wp.length_sq(cp1 - cp2)


@wp.func
def _check_delone_quadrangle_d(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, max_angle_change: wp.float64
) -> wp.bool:
    # Returns True to KEEP the current diagonal (a-c), False to flip to (b-d). Exact port of
    # checkDeloneQuadrangle(Vector3d, ...).
    n_abc = triangle_normal(a, b, c)
    n_acd = triangle_normal(a, c, d)
    old_pocket = wp.dot(n_abc, n_acd) < DELONE_CRITICAL_DOT

    n_abd = triangle_normal(a, b, d)
    n_dbc = triangle_normal(d, b, c)
    new_pocket = wp.dot(n_abd, n_dbc) < DELONE_CRITICAL_DOT

    if old_pocket != new_pocket:
        return new_pocket

    if old_pocket:
        metric_ac = wp.max(mincircle_diameter_sq(a, c, d), mincircle_diameter_sq(c, a, b))
        metric_bd = wp.max(mincircle_diameter_sq(b, d, a), mincircle_diameter_sq(d, b, c))
        return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)

    if max_angle_change < NO_ANGLE_CHANGE_LIMIT:
        old_angle = dihedral_angle(n_abd, n_dbc, d - b)
        new_angle = dihedral_angle(n_abc, n_acd, a - c)
        if wp.abs(old_angle - new_angle) > max_angle_change:
            return True

    metric_ac, metric_bd = delone_metrics(a, b, c, d)

    if wp.isinf(metric_ac):
        if wp.isinf(metric_bd):
            return wp.length_sq(a - c) <= wp.length_sq(b - d)
        return False
    return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)


# ---------------------------------------------------------------------------
# 2D orientation / incircle predicate (for delaunay_triangulation)
# ---------------------------------------------------------------------------


@wp.func
def _incircle_d(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d, d: wp.vec2d) -> wp.float64:
    # Positive iff d is inside the circumcircle of CCW triangle (a, b, c). The 3x3 determinant is
    # kept expanded in components (as in ``predicates.orient2d``) because the term order is what
    # makes the sign reliable near cocircularity; only the squared radii go through ``length_sq``.
    ad = a - d
    bd = b - d
    cd = c - d
    a2 = wp.length_sq(ad)
    b2 = wp.length_sq(bd)
    c2 = wp.length_sq(cd)
    return (
        ad[0] * (bd[1] * c2 - b2 * cd[1])
        - ad[1] * (bd[0] * c2 - b2 * cd[0])
        + a2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


# ---------------------------------------------------------------------------
# Shared parallel edge-flip core
# ---------------------------------------------------------------------------


@wp.func
def _resolve_flip_quad(
    faces: wp.array[wp.int32], f0: wp.int32, u: wp.int32, v: wp.int32, d0: wp.int32, d1: wp.int32
) -> wp.vec4i:
    # Orient the flip quad so f0 traverses a->c (its apex d0 is the left apex "d"); the other
    # face's apex d1 is the right apex "b". Returns (a, b, c, d); a<0 marks inconsistent winding.
    a = wp.int32(-1)
    c = wp.int32(-1)
    for k in range(3):
        va = faces[f0 * 3 + k]
        vb = faces[f0 * 3 + (k + 1) % 3]
        if va == u and vb == v:
            a = u
            c = v
        if va == v and vb == u:
            a = v
            c = u
    return wp.vec4i(a, d1, c, d0)


@wp.func
def _resolve_flip_quad_guarded(
    faces: wp.array[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    k: wp.int32,
    f0: wp.int32,
    out_quad: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # Shared flip-candidate preamble: reject missing apexes, inconsistent winding, b == d, and
    # flips that would duplicate an existing edge. Returns the unpacked ``(a, b, c, d)`` with
    # a < 0 when not flippable, so every candidate kernel opens with the same two lines; out_quad[k]
    # is written only for valid quads (claim/commit read quad[k] only when the caller has set
    # out_flip[k], which stays False for rejected/non-flipped edges).
    a = wp.int32(-1)
    b = wp.int32(-1)
    c = wp.int32(-1)
    d = wp.int32(-1)
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 >= 0 and d1 >= 0:
        quad = _resolve_flip_quad(faces, f0, adjacency_edges[k, 0], adjacency_edges[k, 1], d0, d1)
        if (
            quad[0] >= 0
            and quad[1] != quad[3]
            and not binary_search_sorted_contains(
                sorted_edge_keys, pack_edge_key(quad[1], quad[3], key_base)
            )
        ):
            a = quad[0]
            b = quad[1]
            c = quad[2]
            d = quad[3]
            out_quad[k, 0] = a
            out_quad[k, 1] = b
            out_quad[k, 2] = c
            out_quad[k, 3] = d
    return a, b, c, d


@wp.kernel
def delone_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    max_angle_change: wp.float32,
    max_deviation_sq: wp.float32,
    critical_aspect: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = int(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if region_flags[f0] == 0 or region_flags[f1] == 0:
        return
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    if a < 0:
        return
    ap = to_vec3d(vertices[a])
    bp = to_vec3d(vertices[b])
    cp = to_vec3d(vertices[c])
    dp = to_vec3d(vertices[d])
    if max_deviation_sq < F32_LARGE:
        if _segments_dist_sq_d(ap, cp, bp, dp) > wp.float64(max_deviation_sq):
            return
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    angle = wp.float64(max_angle_change)
    if critical_aspect < F32_LARGE and angle < NO_ANGLE_CHANGE_LIMIT:
        max_aspect = wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp))
        if max_aspect > wp.float64(critical_aspect):
            angle = NO_ANGLE_CHANGE_LIMIT
    out_flip[k] = not _check_delone_quadrangle_d(ap, bp, cp, dp, angle)


@wp.kernel
def incircle_flip_candidates(
    points: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = int(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    if a < 0:
        return
    ap = to_vec2d(points[a])
    bp = to_vec2d(points[b])
    cp = to_vec2d(points[c])
    dp = to_vec2d(points[d])
    # Post-flip triangles (a, b, d) and (d, b, c) must both be positively oriented (convex quad).
    if orient2d(ap, bp, dp) <= wp.float64(0.0) or orient2d(dp, bp, cp) <= wp.float64(0.0):
        return
    # f0 = (a, c, d) is CCW; flip iff the opposite apex b lies inside its circumcircle.
    out_flip[k] = _incircle_d(ap, cp, dp, bp) > wp.float64(0.0)


@wp.kernel
def claim_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_face_claim: wp.array[wp.int32],
    out_edge_claim: wp.array[wp.int32],
) -> None:
    k = int(wp.tid())
    if not flip[k]:
        return
    wp.atomic_min(out_face_claim, adjacency[k, 0], k)
    wp.atomic_min(out_face_claim, adjacency[k, 1], k)
    slot = hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
    wp.atomic_min(out_edge_claim, slot, k)


@wp.kernel
def commit_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    face_claim: wp.array[wp.int32],
    edge_claim: wp.array[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_faces: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    k = int(wp.tid())
    if not flip[k]:
        return
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if face_claim[f0] != k or face_claim[f1] != k:
        return
    slot = hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
    if edge_claim[slot] != k:
        return
    a = quad[k, 0]
    b = quad[k, 1]
    c = quad[k, 2]
    d = quad[k, 3]
    # New diagonal b-d: faces become (a, b, d) and (c, d, b), preserving winding.
    out_faces[f0 * 3 + 0] = a
    out_faces[f0 * 3 + 1] = b
    out_faces[f0 * 3 + 2] = d
    out_faces[f1 * 3 + 0] = c
    out_faces[f1 * 3 + 1] = d
    out_faces[f1 * 3 + 2] = b
    wp.atomic_add(out_count, 0, 1)


# ===========================================================================
# Isotropic explicit remeshing (Botsch-Kobbelt split/collapse/flip/smooth/reproject)
#
# Split reuses subdivide_to_size. The kernels below add: feature/boundary classification
# (per-vertex FREE/CREASE/CORNER codes), a parallel edge-collapse primitive with full 1-ring
# locking + link-condition guard, valence-driven edge flips, tangential Laplacian smoothing, and
# reprojection of free vertices onto the original surface.
# ===========================================================================
FREE_VERTEX = wp.constant(wp.int32(0))
CREASE_VERTEX = wp.constant(wp.int32(1))
CORNER_VERTEX = wp.constant(wp.int32(2))


@wp.kernel
def scatter_edge_endpoint_counts(
    edges: wp.array2d(dtype=wp.int32), out_count: wp.array(dtype=wp.int32)
) -> None:
    # Add 1 to the per-vertex counter for both endpoints of every listed edge.
    e = int(wp.tid())
    wp.atomic_add(out_count, edges[e, 0], 1)
    wp.atomic_add(out_count, edges[e, 1], 1)


@wp.kernel
def scatter_feature_endpoint_counts(
    adjacency_edges: wp.array2d(dtype=wp.int32),
    angles: wp.array(dtype=wp.float32),
    feature_angle: wp.float32,
    out_count: wp.array(dtype=wp.int32),
) -> None:
    # Add 1 to both endpoints of every interior edge sharper than feature_angle.
    k = int(wp.tid())
    if angles[k] > feature_angle:
        wp.atomic_add(out_count, adjacency_edges[k, 0], 1)
        wp.atomic_add(out_count, adjacency_edges[k, 1], 1)


@wp.func
def finalize_vertex_codes(feature_count: wp.int32) -> wp.int32:
    # 0 feature edges -> FREE; exactly 2 -> CREASE (on a smooth feature/boundary line);
    # anything else (1 = feature endpoint, >=3 = junction) -> CORNER (frozen).
    code = CORNER_VERTEX
    if feature_count == 0:
        code = FREE_VERTEX
    elif feature_count == 2:
        code = CREASE_VERTEX
    return code


@wp.func
def csr_common_neighbor_count(
    offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), a: wp.int32, b: wp.int32
) -> wp.int32:
    # Number of vertices adjacent to both a and b (two nested scans; degrees are tiny).
    count = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
    for i in range(offsets[a], offsets[a + 1]):
        w = columns[i]
        for j in range(offsets[b], offsets[b + 1]):
            if columns[j] == w:
                count += 1
    return count


@wp.kernel(enable_backward=False)
def collapse_candidates(
    unique_edges: wp.array2d(dtype=wp.int32),
    lengths: wp.array(dtype=wp.float32),
    vertices: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int32),
    edge_face_count: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    low: wp.float32,
    high: wp.float32,
    out_survivor: wp.array(dtype=wp.int32),
    out_removed: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3),
) -> None:
    k = int(wp.tid())
    out_survivor[k] = -1
    if lengths[k] >= low:
        return
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    cu = codes[u]
    cv = codes[v]
    is_boundary = edge_face_count[k] == 1

    # Choose the surviving vertex and its target position (features/corners stay put).
    s = u
    r = v
    p = wp.lerp(vertices[u], vertices[v], 0.5)
    reject = False
    if cu == CORNER_VERTEX and cv == CORNER_VERTEX:
        reject = True
    elif cu >= CREASE_VERTEX and cv >= CREASE_VERTEX:
        # Two feature vertices: only collapse along a boundary edge (both plain creases).
        if is_boundary and cu == CREASE_VERTEX and cv == CREASE_VERTEX:
            s = u
            r = v
            p = wp.lerp(vertices[u], vertices[v], 0.5)
        else:
            reject = True
    elif cu >= CREASE_VERTEX:
        s = u
        r = v
        p = vertices[u]
    elif cv >= CREASE_VERTEX:
        s = v
        r = u
        p = vertices[v]
    if reject:
        return

    # Link condition: exactly 2 shared neighbours for an interior edge, 1 for a boundary edge.
    required = 2
    if is_boundary:
        required = 1
    if csr_common_neighbor_count(offsets, columns, u, v) != required:
        return

    # Anti-oscillation: reject if the collapse would create an edge longer than the high band.
    for i in range(offsets[r], offsets[r + 1]):
        w = columns[i]
        if w != s and wp.length(p - vertices[w]) > high:
            return

    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = p


@wp.kernel(enable_backward=False)
def claim_collapses(
    survivor: wp.array(dtype=wp.int32),
    removed: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    out_claim: wp.array(dtype=wp.int32),
) -> None:
    # Lock the full closed 1-ring of both endpoints (min edge id wins), so committed
    # collapses have disjoint neighbourhoods and stay independent.
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    key = k
    r = removed[k]
    wp.atomic_min(out_claim, s, key)
    wp.atomic_min(out_claim, r, key)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_claim, columns[i], key)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_claim, columns[i], key)


@wp.kernel(enable_backward=False)
def commit_collapses(
    survivor: wp.array(dtype=wp.int32),
    removed: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    claim: wp.array(dtype=wp.int32),
    out_remap: wp.array(dtype=wp.int32),
    out_positions: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.int32),
) -> None:
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    key = k
    r = removed[k]
    won = True
    if claim[s] != key or claim[r] != key:
        won = False
    for i in range(offsets[s], offsets[s + 1]):
        if claim[columns[i]] != key:
            won = False
    for i in range(offsets[r], offsets[r + 1]):
        if claim[columns[i]] != key:
            won = False
    if not won:
        return
    out_remap[r] = s
    out_positions[s] = pos[k]
    wp.atomic_add(out_count, 0, 1)


@wp.kernel
def mark_distinct_faces(
    faces: wp.array(dtype=wp.int32), out_valid: wp.array(dtype=wp.bool)
) -> None:
    # A face survives a collapse remap only if its three vertex indices are still distinct.
    f = int(wp.tid())
    a = faces[f * 3 + 0]
    b = faces[f * 3 + 1]
    c = faces[f * 3 + 2]
    out_valid[f] = a != b and b != c and a != c


@wp.kernel
def accumulate_vertex_valence(
    unique_edges: wp.array2d(dtype=wp.int32), out_valence: wp.array(dtype=wp.int32)
) -> None:
    e = int(wp.tid())
    wp.atomic_add(out_valence, unique_edges[e, 0], 1)
    wp.atomic_add(out_valence, unique_edges[e, 1], 1)


@wp.kernel
def valence_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    valence: wp.array[wp.int32],
    boundary_vertex: wp.array[wp.bool],
    feature_angle: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = int(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    # Never flip a feature edge (sharp dihedral between the two incident faces).
    n0 = triangle_normal(
        vertices[faces[f0 * 3 + 0]], vertices[faces[f0 * 3 + 1]], vertices[faces[f0 * 3 + 2]]
    )
    n1 = triangle_normal(
        vertices[faces[f1 * 3 + 0]], vertices[faces[f1 * 3 + 1]], vertices[faces[f1 * 3 + 2]]
    )
    if wp.acos(wp.dot(n0, n1)) > feature_angle:  # wp.acos auto-clamps to [-1, 1]
        return
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    if a < 0:
        return
    ap = to_vec3d(vertices[a])
    bp = to_vec3d(vertices[b])
    cp = to_vec3d(vertices[c])
    dp = to_vec3d(vertices[d])
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    # Shape guard. Convexity makes the flip *legal* but says nothing about the shape of what it
    # produces, and the valence objective below is blind to geometry: on a graded mesh it will
    # happily turn two slivers into two worse ones, which in float32 lands on exactly-zero area.
    # (Measured on ``saddle_graded``: the swap stage alone produced 3 992 zero-area faces out of
    # 92 100, and none survive this guard. ``delone_flip_candidates`` has its own deviation and
    # aspect gates; this is the equivalent for the valence objective.)
    #
    # ``triangle_aspect_ratio`` is circumradius / 2 * inradius and returns +inf for a degenerate
    # triangle, so the two tests below read as "never create a degenerate triangle" and "never make
    # the worse of the pair worse". Post-flip faces are (a, b, d) and (c, d, b) -- see commit_flips.
    aspect_after = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(cp, dp, bp))
    if not wp.isfinite(aspect_after):
        return
    if aspect_after > wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp)):
        return
    # Target valence: 4 on the boundary, 6 in the interior.
    ta = wp.where(boundary_vertex[a], 4, 6)
    tb = wp.where(boundary_vertex[b], 4, 6)
    tc = wp.where(boundary_vertex[c], 4, 6)
    td = wp.where(boundary_vertex[d], 4, 6)
    va = valence[a]
    vb = valence[b]
    vc = valence[c]
    vd = valence[d]
    before = (
        (va - ta) * (va - ta)
        + (vb - tb) * (vb - tb)
        + (vc - tc) * (vc - tc)
        + (vd - td) * (vd - td)
    )
    after = (
        (va - 1 - ta) * (va - 1 - ta)
        + (vb + 1 - tb) * (vb + 1 - tb)
        + (vc - 1 - tc) * (vc - 1 - tc)
        + (vd + 1 - td) * (vd + 1 - td)
    )
    out_flip[k] = after < before


@wp.kernel
def accumulate_one_ring(
    unique_edges: wp.array2d(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    out_sum: wp.array(dtype=wp.vec3),
    out_degree: wp.array(dtype=wp.int32),
) -> None:
    # Unweighted one-ring centroid. Note this is *not* the area-equalizing relaxation that
    # Botsch-Kobbelt specify: on a regular graded grid every vertex already sits at the plain
    # average of its neighbours, so this smoother is at a fixed point and cannot equalize the
    # sampling. Area-weighting it takes the 99th-percentile aspect ratio on such a patch from 352
    # to 20, but also makes ``is_watertight`` fail on ``cave_cube`` through a self-intersection at
    # *every* step size down to lam=0.1, so it needs a fold guard first. See
    # ``tests/test_remesh.py::test_remesh_emits_no_degenerate_faces``.
    e = int(wp.tid())
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    wp.atomic_add(out_sum, u, vertices[v])
    wp.atomic_add(out_degree, u, 1)
    wp.atomic_add(out_sum, v, vertices[u])
    wp.atomic_add(out_degree, v, 1)


@wp.func
def tangential_smooth_step(
    vertex: wp.vec3,
    code: wp.int32,
    normal: wp.vec3,
    ring_sum: wp.vec3,
    degree: wp.int32,
    lam: wp.float32,
) -> wp.vec3:
    # Move a free vertex toward its one-ring centroid, but only within the tangent plane, so the
    # surface is smoothed without being shrunk. Pinned vertices and isolated ones stay put.
    p = vertex
    if code != FREE_VERTEX or degree == 0:
        return p
    centroid = ring_sum / float(degree)
    delta = centroid - p
    tangential = project_out_normal(delta, normal)
    return p + lam * tangential


@wp.func
def reproject_vertices(
    vertex: wp.vec3, code: wp.int32, mesh_id: wp.uint64, max_dist: wp.float32
) -> wp.vec3:
    # Snap a free vertex back onto the closest point of the original surface, undoing the drift the
    # smoothing pass introduces. Pinned vertices and failed queries keep their position.
    if code != FREE_VERTEX:
        return vertex
    query = wp.mesh_query_point_no_sign(mesh_id, vertex, max_dist)
    if query.result:
        return wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    return vertex


@wp.func
def local_corner(faces: wp.array[wp.int32], f: wp.int32, vertex: wp.int32) -> wp.int32:
    # Which corner of face ``f`` holds ``vertex``, or -1. Needed because an edge-length table is
    # indexed by *corner*, while the flip machinery speaks in vertex indices.
    for k in range(3):
        if faces[f * 3 + k] == vertex:
            return k
    return wp.int32(-1)


@wp.func
def law_of_cosines_angle(adjacent_a: wp.float32, adjacent_b: wp.float32, opposite: wp.float32):
    # Angle between the two adjacent sides of a triangle, from its three side lengths alone. Every
    # geometric quantity the intrinsic flip needs comes through here -- no vertex position does.
    denominator = 2.0 * adjacent_a * adjacent_b
    if denominator <= TOLERANCE_ZERO_CONSTANT:
        return wp.float32(0.0)
    cosine = (adjacent_a * adjacent_a + adjacent_b * adjacent_b - opposite * opposite) / denominator
    return wp.acos(wp.clamp(cosine, -1.0, 1.0))


@wp.func
def edge_lengths_at(
    edge_lengths: wp.array2d[wp.float32],
    faces: wp.array[wp.int32],
    f: wp.int32,
    opposite_vertex: wp.int32,
) -> wp.float32:
    # The length of the edge of ``f`` that faces ``opposite_vertex``.
    corner = local_corner(faces, f, opposite_vertex)
    if corner < 0:
        return wp.float32(0.0)
    return edge_lengths[f, corner]


@wp.kernel
def intrinsic_delaunay_candidates(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
    out_new_length: wp.array[wp.float32],
) -> None:
    # Mark the interior edges that violate the local Delaunay condition, and measure what the
    # flipped edge would be -- both from edge lengths only, which is what makes the retriangulation
    # intrinsic: no vertex moves, so the *surface* is unchanged and only its triangulation improves.
    k = int(wp.tid())
    out_flip[k] = False
    out_new_length[k] = 0.0
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    first, _apex1, second, apex0 = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if first < wp.int32(0):
        return

    corner_f0_first = local_corner(faces, f0, first)
    corner_f0_second = local_corner(faces, f0, second)
    corner_f0_apex = local_corner(faces, f0, apex0)
    corner_f1_first = local_corner(faces, f1, first)
    corner_f1_second = local_corner(faces, f1, second)
    if corner_f0_first < 0 or corner_f0_second < 0 or corner_f0_apex < 0:
        return
    if corner_f1_first < 0 or corner_f1_second < 0:
        return

    shared = edge_lengths[f0, corner_f0_apex]
    first_apex0 = edge_lengths[f0, corner_f0_second]
    second_apex0 = edge_lengths[f0, corner_f0_first]
    first_apex1 = edge_lengths[f1, corner_f1_second]
    second_apex1 = edge_lengths[f1, corner_f1_first]

    # The Delaunay test: the two angles facing the shared edge sum past a straight angle exactly
    # when the edge's cotangent weight would go negative.
    angle0 = law_of_cosines_angle(first_apex0, second_apex0, shared)
    angle1 = law_of_cosines_angle(first_apex1, second_apex1, shared)
    if angle0 + angle1 <= wp.PI:
        return

    # Unfold both triangles about the shared edge and measure the other diagonal. The wedge angles
    # at ``first`` add because the two triangles lie on opposite sides of the shared edge.
    wedge0 = law_of_cosines_angle(shared, first_apex0, second_apex0)
    wedge1 = law_of_cosines_angle(shared, first_apex1, second_apex1)
    total = wedge0 + wedge1
    flipped = (
        first_apex0 * first_apex0
        + first_apex1 * first_apex1
        - (2.0 * first_apex0 * first_apex1 * wp.cos(total))
    )
    if flipped <= TOLERANCE_ZERO_CONSTANT:
        return
    out_new_length[k] = wp.sqrt(flipped)
    out_flip[k] = True


@wp.kernel
def update_flipped_lengths(
    faces: wp.array[wp.int32],
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    new_length: wp.array[wp.float32],
    face_claim: wp.array[wp.int32],
    edge_claim: wp.array[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_edge_lengths: wp.array2d[wp.float32],
) -> None:
    # Rewrite the two faces' edge-length rows for the flips that won their claims, reading the old
    # rows first. This must run *before* the connectivity rewrite, which is what still knows which
    # corner holds which vertex; a committed flip owns both its faces exclusively, so reading and
    # writing the same rows here is race-free.
    k = int(wp.tid())
    if not flip[k]:
        return
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if face_claim[f0] != k or face_claim[f1] != k:
        return
    if edge_claim[hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)] != k:
        return

    first = quad[k, 0]
    second = quad[k, 2]
    diagonal = new_length[k]

    first_apex0 = edge_lengths_at(out_edge_lengths, faces, f0, second)
    second_apex0 = edge_lengths_at(out_edge_lengths, faces, f0, first)
    first_apex1 = edge_lengths_at(out_edge_lengths, faces, f1, second)
    second_apex1 = edge_lengths_at(out_edge_lengths, faces, f1, first)

    # ``commit_flips`` rewrites f0 as (first, apex1, apex0) and f1 as (second, apex0, apex1); each
    # column holds the edge opposite that corner.
    out_edge_lengths[f0, 0] = diagonal
    out_edge_lengths[f0, 1] = first_apex0
    out_edge_lengths[f0, 2] = first_apex1
    out_edge_lengths[f1, 0] = diagonal
    out_edge_lengths[f1, 1] = second_apex1
    out_edge_lengths[f1, 2] = second_apex0


@wp.func
def voxel_size_inverse(voxel_size: wp.float32) -> wp.float32:
    # Reciprocal cell width, so the two ``cluster_*`` kernels can take the width itself (which they
    # also need for the cell centre) without the caller passing both.
    return 1.0 / voxel_size


@wp.kernel
def cluster_accumulate(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_sum: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    v = int(wp.tid())
    wp.atomic_add(out_sum, labels[v], vertices[v])
    wp.atomic_add(out_count, labels[v], 1)


@wp.kernel
def cluster_min_center_distance(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    voxel_size: wp.float32,
    out_min_distance: wp.array[wp.float32],
) -> None:
    # Pass 1 of the "closest to the cell centre" representative: the winning *distance* per cluster.
    # Split from the index pick so both passes use 32-bit atomics only; the two together are
    # deterministic because pass 2 breaks ties by lowest vertex index.
    v = int(wp.tid())
    cell = voxel_cell(vertices[v], origin, voxel_size_inverse(voxel_size))
    center = origin + wp.vec3(
        (wp.float32(cell[0]) + 0.5) * voxel_size,
        (wp.float32(cell[1]) + 0.5) * voxel_size,
        (wp.float32(cell[2]) + 0.5) * voxel_size,
    )
    wp.atomic_min(out_min_distance, labels[v], wp.length_sq(vertices[v] - center))


@wp.kernel
def cluster_pick_closest(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    voxel_size: wp.float32,
    min_distance: wp.array[wp.float32],
    out_representative: wp.array[wp.int32],
) -> None:
    # Pass 2: whichever vertices tie for their cluster's winning distance, the lowest index wins.
    v = int(wp.tid())
    cell = voxel_cell(vertices[v], origin, voxel_size_inverse(voxel_size))
    center = origin + wp.vec3(
        (wp.float32(cell[0]) + 0.5) * voxel_size,
        (wp.float32(cell[1]) + 0.5) * voxel_size,
        (wp.float32(cell[2]) + 0.5) * voxel_size,
    )
    if wp.length_sq(vertices[v] - center) <= min_distance[labels[v]]:
        wp.atomic_min(out_representative, labels[v], v)


@wp.kernel
def faces_with_distinct_indices(faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]) -> None:
    # A face survives vertex clustering only if its three corners landed in three different cells.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    out_mask[f] = i0 != i1 and i1 != i2 and i0 != i2


# Objective for ``objective_flip_candidates``. A warp-uniform kernel argument rather than a
# ``wp.Function``, so both predicates share one compiled module (AGENTS.md section 4).
OBJECTIVE_PLANARITY = wp.constant(wp.int32(0))  # improve triangle shape on a near-planar quad
OBJECTIVE_CURVATURE = wp.constant(wp.int32(1))  # pick whichever diagonal bends the surface less
OBJECTIVE_T_VERTEX = wp.constant(wp.int32(2))  # break up a sliver whose apex sits on the far edge

# Relative margin a flip must beat the current diagonal by. Without it a quad whose two diagonals
# score equally (every quad of a regular grid) flips back and forth forever, one pass each way.
OBJECTIVE_EPS = wp.constant(wp.float32(1e-6))


@wp.kernel
def objective_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    objective: wp.int32,
    metric: wp.int32,
    planar_cos: wp.float32,
    aspect_threshold: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    # Quad convention (shared with ``delone_flip_candidates``): the current diagonal is a-c, with
    # faces (a, b, c) and (a, c, d); the flip replaces it with b-d, giving (a, b, d) and (d, b, c).
    k = int(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if region_flags[f0] == 0 or region_flags[f1] == 0:
        return
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    if a < 0:
        return
    ap = vertices[a]
    bp = vertices[b]
    cp = vertices[c]
    dp = vertices[d]

    # A non-convex quad has no valid flip: the new diagonal would fall outside it.
    if not is_unfold_quadrangle_convex(to_vec3d(ap), to_vec3d(bp), to_vec3d(cp), to_vec3d(dp)):
        return

    if objective == OBJECTIVE_T_VERTEX:
        # A T-vertex shows up as a sliver: one apex sits (nearly) on the opposite edge, which drives
        # the circumradius-to-inradius ratio through the roof. Flip only when the sliver is *that*
        # bad and the flip actually improves it, so a merely thin triangle is left alone.
        old_worst = wp.max(triangle_aspect_ratio(ap, bp, cp), triangle_aspect_ratio(ap, cp, dp))
        if not (old_worst > aspect_threshold):  # also excludes a NaN ratio
            return
        new_worst = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(dp, bp, cp))
        out_flip[k] = new_worst < old_worst
        return

    normal_abc = triangle_normal(ap, bp, cp)
    normal_acd = triangle_normal(ap, cp, dp)
    normal_abd = triangle_normal(ap, bp, dp)
    normal_dbc = triangle_normal(dp, bp, cp)

    if objective == OBJECTIVE_PLANARITY:
        # Only rewrite a quad that is flat enough for the rewrite not to change the surface. The
        # gate is on the *cosine* of the dihedral so the kernel needs no inverse trigonometry.
        if wp.dot(normal_abc, normal_acd) < planar_cos:
            return
        old_score = wp.min(
            triangle_quality(ap, bp, cp, metric), triangle_quality(ap, cp, dp, metric)
        )
        new_score = wp.min(
            triangle_quality(ap, bp, dp, metric), triangle_quality(dp, bp, cp, metric)
        )
        out_flip[k] = new_score > old_score * (1.0 + OBJECTIVE_EPS)
        return

    # Curvature: keep whichever diagonal leaves the two triangles closer to coplanar. Unlike the
    # planarity objective this deliberately *does* change the surface -- that is the point.
    old_bend = wp.abs(dihedral_angle(normal_abc, normal_acd, cp - ap))
    new_bend = wp.abs(dihedral_angle(normal_abd, normal_dbc, dp - bp))
    out_flip[k] = new_bend < old_bend * (1.0 - OBJECTIVE_EPS)


# ---------------------------------------------------------------------------
# Quadric error metric (Garland-Heckbert) decimation
# ---------------------------------------------------------------------------

# Quadrics are accumulated in float64. That is not caution: the entries are sums of ``area * d^2``
# with ``d`` an absolute plane offset, so on a mesh whose coordinates are far from the origin they
# span many orders of magnitude and a float32 accumulation loses the small ones -- which are exactly
# the terms that distinguish two candidate collapses. libigl and MeshLab both use double here.

# Below this determinant (relative to the quadric's own scale) the 3x3 system is treated as singular
# and the optimum falls back to the edge midpoint: a planar neighbourhood has a whole plane of
# equally good positions and picking one by inversion amplifies noise.
QUADRIC_SINGULAR_EPS = wp.constant(wp.float64(1e-12))

# A collapse is rejected when it would turn an incident face's normal by more than this. Zero would
# allow a face to become exactly degenerate; 0.2 (~78 degrees) still permits real simplification of
# a curved region while refusing an outright fold.
COLLAPSE_MIN_NORMAL_DOT = wp.constant(wp.float32(0.2))


@wp.func
def plane_quadric(normal: wp.vec3d, offset: wp.float64, weight: wp.float64) -> wp.mat44d:
    # Garland-Heckbert fundamental quadric of the plane ``dot(normal, x) + offset = 0``, scaled by
    # ``weight``. Laid out so that ``[p, 1]^T Q [p, 1]`` is the weighted squared distance to the
    # plane: the leading 3x3 block is ``n n^T``, the last row and column are ``offset * n``, and the
    # corner is ``offset^2``.
    a = weight * normal[0]
    b = weight * normal[1]
    c = weight * normal[2]
    d = weight * offset
    return wp.mat44d(
        a * normal[0],
        a * normal[1],
        a * normal[2],
        a * offset,
        b * normal[0],
        b * normal[1],
        b * normal[2],
        b * offset,
        c * normal[0],
        c * normal[1],
        c * normal[2],
        c * offset,
        d * normal[0],
        d * normal[1],
        d * normal[2],
        d * offset,
    )


@wp.func
def quadric_error(quadric: wp.mat44d, p: wp.vec3d) -> wp.float64:
    # ``[p, 1]^T Q [p, 1]``: the accumulated squared distance from ``p`` to every plane folded into
    # ``Q``. Clamped at zero, since a float64 sum of positive-semidefinite terms can still land a
    # hair below it and a negative "error" would sort ahead of every real candidate.
    homogeneous = wp.vec4d(p[0], p[1], p[2], wp.float64(1.0))
    return wp.max(wp.float64(0.0), wp.dot(homogeneous, quadric * homogeneous))


@wp.func
def quadric_optimum(quadric: wp.mat44d, fallback: wp.vec3d) -> wp.vec3d:
    # Position minimizing the quadric: solve ``A p = -b`` for the leading 3x3 block ``A`` and the
    # last column ``b``. ``fallback`` (the edge midpoint) is returned when ``A`` is singular
    # relative to its own scale, which is the planar case -- there the minimum is a whole plane and
    # inverting a near-singular matrix would place the vertex arbitrarily far away.
    a = wp.mat33d(
        quadric[0, 0],
        quadric[0, 1],
        quadric[0, 2],
        quadric[1, 0],
        quadric[1, 1],
        quadric[1, 2],
        quadric[2, 0],
        quadric[2, 1],
        quadric[2, 2],
    )
    scale = wp.abs(quadric[0, 0]) + wp.abs(quadric[1, 1]) + wp.abs(quadric[2, 2])
    if scale <= wp.float64(0.0):
        return fallback
    if wp.abs(wp.determinant(a)) <= QUADRIC_SINGULAR_EPS * scale * scale * scale:
        return fallback
    b = wp.vec3d(quadric[0, 3], quadric[1, 3], quadric[2, 3])
    return -(wp.inverse(a) * b)


@wp.kernel
def accumulate_face_quadrics(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_quadrics: wp.array[wp.mat44d]
) -> None:
    # Area-weighted plane quadric of each face, scattered onto its three corners. Area weighting is
    # Garland-Heckbert's: a large triangle constrains its vertices more than a sliver does.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    cross = wp.cross(v1 - v0, v2 - v0)
    double_area = wp.length(cross)
    if double_area <= wp.float64(0.0):
        return
    normal = cross / double_area
    quadric = plane_quadric(normal, -wp.dot(normal, v0), double_area * wp.float64(0.5))
    for k in range(3):
        wp.atomic_add(out_quadrics, faces[f * 3 + k], quadric)


@wp.func
def collapse_flips_normal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    moved: wp.int32,
    partner: wp.int32,
    target: wp.vec3,
) -> wp.bool:
    # Would moving ``moved`` to ``target`` (and welding it onto ``partner``) invert any face it
    # still belongs to? The two faces containing *both* endpoints vanish in the collapse and are
    # skipped; every other incident face keeps its other two corners and must keep its orientation.
    #
    # This is the guard that separates a usable decimator from one that produces self-intersecting
    # geometry, and it is why the vertex-face CSR is built at all.
    for slot in range(vertex_face_offsets[moved], vertex_face_offsets[moved + 1]):
        f = vertex_faces[slot]
        i0 = faces[f * 3 + 0]
        i1 = faces[f * 3 + 1]
        i2 = faces[f * 3 + 2]
        if i0 == partner or i1 == partner or i2 == partner:
            continue
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        before = wp.cross(p1 - p0, p2 - p0)
        if i0 == moved:
            p0 = target
        elif i1 == moved:
            p1 = target
        else:
            p2 = target
        after = wp.cross(p1 - p0, p2 - p0)
        before_length = wp.length(before)
        after_length = wp.length(after)
        if before_length <= 0.0:
            continue  # already degenerate: nothing to invert
        if after_length <= 0.0:
            return True  # the collapse would flatten it outright
        if wp.dot(before / before_length, after / after_length) < COLLAPSE_MIN_NORMAL_DOT:
            return True
    return False


@wp.kernel
def quadric_collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    quadrics: wp.array[wp.mat44d],
    codes: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_cost: wp.array[wp.float32],
) -> None:
    # Garland-Heckbert candidate: the cost of collapsing this edge and where its survivor lands.
    # ``out_cost`` is left at +inf for a rejected edge, so the caller's cost sort puts every
    # rejection past every candidate and the budget cut never picks one up.
    k = int(wp.tid())
    out_survivor[k] = -1
    out_cost[k] = wp.inf
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    cu = codes[u]
    cv = codes[v]
    is_boundary = edge_face_count[k] == 1

    # Feature handling mirrors ``collapse_candidates``: a corner never moves, a crease only
    # collapses along its own feature, and otherwise the quadric chooses the position freely.
    s = u
    r = v
    free_position = wp.bool(True)
    if cu == CORNER_VERTEX and cv == CORNER_VERTEX:
        return
    if cu >= CREASE_VERTEX and cv >= CREASE_VERTEX:
        if not (is_boundary and cu == CREASE_VERTEX and cv == CREASE_VERTEX):
            return
    elif cu >= CREASE_VERTEX:
        s = u
        r = v
        free_position = wp.bool(False)
    elif cv >= CREASE_VERTEX:
        s = v
        r = u
        free_position = wp.bool(False)

    # Link condition: exactly 2 shared neighbours for an interior edge, 1 for a boundary edge.
    required = 2
    if is_boundary:
        required = 1
    if csr_common_neighbor_count(offsets, columns, u, v) != required:
        return

    quadric = quadrics[u] + quadrics[v]
    midpoint = (to_vec3d(vertices[u]) + to_vec3d(vertices[v])) * wp.float64(0.5)
    optimum = midpoint
    if free_position:
        optimum = quadric_optimum(quadric, midpoint)
    else:
        optimum = to_vec3d(vertices[s])
    target = to_vec3(optimum)

    if collapse_flips_normal(
        vertices, faces, vertex_face_offsets, vertex_faces, r, s, target
    ) or collapse_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, s, r, target):
        return

    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = target
    out_cost[k] = wp.float32(quadric_error(quadric, optimum))


@wp.kernel
def assign_collapse_priority(
    order: wp.array[wp.int32],
    budget: wp.int32,
    out_priority: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
) -> None:
    # Turn the cost ranking into the key the claim/commit pass locks with, and drop everything past
    # the pass budget. Ranking by cost rather than by edge index is the whole difference between a
    # quadric decimation and a shortest-edge one: the cheapest collapse must win a contested ring.
    i = int(wp.tid())
    k = order[i]
    if i < int(budget):
        out_priority[k] = i
        return
    out_priority[k] = INT32_MAX_CONSTANT
    out_survivor[k] = -1


@wp.func
def scramble_index(index: wp.int32) -> wp.int32:
    # Spatially incoherent lock key for the independent-set pass, from the candidate's own index.
    #
    # This is the load-bearing detail of the whole parallel selection. ``edges_unique`` orders edges
    # lexicographically by endpoint index, which on any structured mesh is *spatially monotone* --
    # and a monotone key field has essentially one local minimum, so a min-key lock commits a single
    # collapse per pass however many candidates there are. Measured on ``saddle_graded``: locking by
    # raw edge index yields exactly **1** winner out of 51 546 candidates, and locking by quadric
    # cost yields 23 (the cost field is smoothly graded there, so it is monotone too). Hashing the
    # index breaks the correlation and restores the expected ~candidates/valence winners.
    #
    # Murmur-style 32-bit finalizer; the top bit is cleared so the key stays a non-negative int32
    # and ``INT32_MAX`` remains usable as the unclaimed sentinel.
    x = wp.uint32(index)
    x = (x ^ (x >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
    x = (x ^ (x >> wp.uint32(15))) * wp.uint32(0x846CA68B)
    x = x ^ (x >> wp.uint32(16))
    return wp.int32(x & wp.uint32(0x7FFFFFFF))


@wp.kernel(enable_backward=False)
def claim_collapse_key(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_min_key: wp.array[wp.int32],
) -> None:
    # Pass 1 of 3: the winning (smallest scrambled) key over the closed 1-rings of both endpoints.
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    key = scramble_index(wp.int32(k))
    r = removed[k]
    wp.atomic_min(out_min_key, s, key)
    wp.atomic_min(out_min_key, r, key)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_min_key, columns[i], key)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_min_key, columns[i], key)


@wp.func
def wins_key_everywhere(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    min_key: wp.array[wp.int32],
    s: wp.int32,
    r: wp.int32,
    key: wp.int32,
) -> wp.bool:
    # Does ``key`` win at every vertex of the two closed 1-rings? ``min_key`` is a minimum over
    # candidates including this one, so the test is equality rather than ``<=``.
    if min_key[s] != key or min_key[r] != key:
        return False
    for i in range(offsets[s], offsets[s + 1]):
        if min_key[columns[i]] != key:
            return False
    for i in range(offsets[r], offsets[r + 1]):
        if min_key[columns[i]] != key:
            return False
    return True


@wp.kernel(enable_backward=False)
def claim_collapse_index(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    min_key: wp.array[wp.int32],
    out_claim: wp.array[wp.int32],
) -> None:
    # Pass 2 of 3: two candidates whose scrambled keys collide would both believe they won, which
    # would break independence -- unlikely at 2^31 keys, but a corrupted mesh when it happens. Among
    # the key winners in a neighbourhood the lowest edge index takes it.
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    if not wins_key_everywhere(offsets, columns, min_key, s, r, scramble_index(wp.int32(k))):
        return
    wp.atomic_min(out_claim, s, k)
    wp.atomic_min(out_claim, r, k)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_claim, columns[i], k)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_claim, columns[i], k)


@wp.kernel(enable_backward=False)
def mark_collapse_winners(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    min_key: wp.array[wp.int32],
    claim: wp.array[wp.int32],
    cost: wp.array[wp.float32],
    out_survivor: wp.array[wp.int32],
    out_cost: wp.array[wp.float32],
) -> None:
    # Pass 3 of 3: the win test, kept separate from the commit so the caller can apply its per-pass
    # budget *after* the independent set is known. Trimming members from an independent set keeps it
    # independent; trimming the candidate list beforehand would change which set is found.
    k = int(wp.tid())
    out_cost[k] = wp.inf
    s = survivor[k]
    if s < 0:
        out_survivor[k] = -1
        return
    r = removed[k]
    won = wins_key_everywhere(offsets, columns, min_key, s, r, scramble_index(wp.int32(k)))
    if claim[s] != k or claim[r] != k:
        won = False
    for i in range(offsets[s], offsets[s + 1]):
        if claim[columns[i]] != k:
            won = False
    for i in range(offsets[r], offsets[r + 1]):
        if claim[columns[i]] != k:
            won = False
    if not won:
        out_survivor[k] = -1
        return
    out_survivor[k] = s
    out_cost[k] = cost[k]


@wp.kernel(enable_backward=False)
def commit_selected_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    target_pos: wp.array[wp.vec3],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Apply an already-independent set: no claim test, because ``mark_collapse_winners`` established
    # independence and the budget cut only ever *removes* members from it.
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    out_remap[removed[k]] = s
    out_positions[s] = target_pos[k]
    wp.atomic_add(out_count, 0, 1)


@wp.kernel(enable_backward=False)
def lock_collapse_neighborhoods(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_locked: wp.array[wp.int32],
) -> None:
    # Mark the closed 1-rings of both endpoints of every committed collapse, so a later
    # independent-set round in the *same* pass can be told which candidates the commit invalidated
    # (see ``drop_locked_candidates``). Plain stores rather than atomics: every write is the same
    # value.
    k = int(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    out_locked[s] = 1
    out_locked[r] = 1
    for i in range(offsets[s], offsets[s + 1]):
        out_locked[columns[i]] = 1
    for i in range(offsets[r], offsets[r + 1]):
        out_locked[columns[i]] = 1


@wp.kernel(enable_backward=False)
def drop_locked_candidates(
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
) -> None:
    # Restore the pass's candidate list for another independent-set round, retiring every candidate
    # whose two closed 1-rings touch an already-collapsed neighbourhood.
    #
    # That disjointness is exactly what makes reusing the pass's scoring legal. A candidate whose
    # closed 1-rings miss every locked vertex has *no incident face* holding a collapsed endpoint,
    # so its endpoints' quadrics, its cost, its target position, its link condition and its
    # normal-flip veto are all still the ones the scoring pass computed. Fail that test and the
    # candidate must wait for the next geometry rebuild.
    k = int(wp.tid())
    out_survivor[k] = -1
    s = candidates[k]
    if s < 0:
        return
    r = removed[k]
    if locked[s] != 0 or locked[r] != 0:
        return
    for i in range(offsets[s], offsets[s + 1]):
        if locked[columns[i]] != 0:
            return
    for i in range(offsets[r], offsets[r + 1]):
        if locked[columns[i]] != 0:
            return
    out_survivor[k] = s
