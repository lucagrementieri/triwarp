import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import binary_search_sorted_contains, to_vec2d, to_vec3d
from triwarp.kernels.grouping import hash_slot, pack_edge_key
from triwarp.kernels.predicates import (
    circumcircle_diameter_sq,
    dihedral_angle,
    is_unfold_quadrangle_convex,
    mincircle_diameter_sq,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    triangle_normal,
)

# Delaunay / Delone edge-flip constants (ported from MRMeshDelone.cpp). The flip predicate
# runs in float64: MeshLib deliberately widens to double because circumcircle diameters of
# near-degenerate triangles have too large a rounding error in float32 (infinite flip loops).
DELONE_CRITICAL_DOT = wp.constant(wp.float64(-0.9))
DELONE_EPS = wp.constant(wp.float64(1e-7))
NO_ANGLE_CHANGE_LIMIT = wp.constant(wp.float64(6.283185307179586))  # 2*pi (NoAngleChangeLimit)
F32_LARGE = wp.constant(wp.float32(3.0e38))  # "disabled gate" sentinel (~FLT_MAX)


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

    metric_ac = wp.max(circumcircle_diameter_sq(a, c, d), circumcircle_diameter_sq(c, a, b))
    metric_bd = wp.max(circumcircle_diameter_sq(b, d, a), circumcircle_diameter_sq(d, b, c))

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
) -> wp.vec4i:
    # Shared flip-candidate preamble: reject missing apexes, inconsistent winding, b == d, and
    # flips that would duplicate an existing edge. Returns (a, b, c, d) with a < 0 when not
    # flippable; out_quad[k] is written only for valid quads (claim/commit read quad[k] only
    # when the caller has set out_flip[k], which stays False for rejected/non-flipped edges).
    invalid = wp.vec4i(-1, -1, -1, -1)
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 < 0 or d1 < 0:
        return invalid
    quad = _resolve_flip_quad(faces, f0, adjacency_edges[k, 0], adjacency_edges[k, 1], d0, d1)
    if quad[0] < 0 or quad[1] == quad[3]:
        return invalid
    if binary_search_sorted_contains(sorted_edge_keys, pack_edge_key(quad[1], quad[3], key_base)):
        return invalid
    out_quad[k, 0] = quad[0]
    out_quad[k, 1] = quad[1]
    out_quad[k, 2] = quad[2]
    out_quad[k, 3] = quad[3]
    return quad


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
    quad = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    a = quad[0]
    b = quad[1]
    c = quad[2]
    d = quad[3]
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
    quad = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    a = quad[0]
    b = quad[1]
    c = quad[2]
    d = quad[3]
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


@wp.kernel
def finalize_vertex_codes(
    feature_count: wp.array(dtype=wp.int32), out_code: wp.array(dtype=wp.int32)
) -> None:
    # 0 feature edges -> FREE; exactly 2 -> CREASE (on a smooth feature/boundary line);
    # anything else (1 = feature endpoint, >=3 = junction) -> CORNER (frozen).
    v = int(wp.tid())
    count = feature_count[v]
    code = CORNER_VERTEX
    if count == 0:
        code = FREE_VERTEX
    elif count == 2:
        code = CREASE_VERTEX
    out_code[v] = code


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
    out_survivor: wp.array(dtype=wp.int32),
    out_removed: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    out_claim: wp.array(dtype=wp.int32),
) -> None:
    # Lock the full closed 1-ring of both endpoints (min edge id wins), so committed
    # collapses have disjoint neighbourhoods and stay independent.
    k = int(wp.tid())
    s = out_survivor[k]
    if s < 0:
        return
    r = out_removed[k]
    wp.atomic_min(out_claim, s, k)
    wp.atomic_min(out_claim, r, k)
    for i in range(offsets[s], offsets[s + 1]):
        wp.atomic_min(out_claim, columns[i], k)
    for i in range(offsets[r], offsets[r + 1]):
        wp.atomic_min(out_claim, columns[i], k)


@wp.kernel(enable_backward=False)
def commit_collapses(
    out_survivor: wp.array(dtype=wp.int32),
    out_removed: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=wp.int32),
    columns: wp.array(dtype=wp.int32),
    claim: wp.array(dtype=wp.int32),
    out_remap: wp.array(dtype=wp.int32),
    out_positions: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.int32),
) -> None:
    k = int(wp.tid())
    s = out_survivor[k]
    if s < 0:
        return
    r = out_removed[k]
    won = True
    if claim[s] != k or claim[r] != k:
        won = False
    for i in range(offsets[s], offsets[s + 1]):
        if claim[columns[i]] != k:
            won = False
    for i in range(offsets[r], offsets[r + 1]):
        if claim[columns[i]] != k:
            won = False
    if not won:
        return
    out_remap[r] = s
    out_positions[s] = out_pos[k]
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
    quad = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, wp.int32(k), f0, out_quad
    )
    a = quad[0]
    b = quad[1]
    c = quad[2]
    d = quad[3]
    if a < 0:
        return
    if not is_unfold_quadrangle_convex(
        to_vec3d(vertices[a]), to_vec3d(vertices[b]), to_vec3d(vertices[c]), to_vec3d(vertices[d])
    ):
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
    e = int(wp.tid())
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    wp.atomic_add(out_sum, u, vertices[v])
    wp.atomic_add(out_degree, u, 1)
    wp.atomic_add(out_sum, v, vertices[u])
    wp.atomic_add(out_degree, v, 1)


@wp.kernel
def tangential_smooth_step(
    vertices: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3),
    ring_sum: wp.array(dtype=wp.vec3),
    degree: wp.array(dtype=wp.int32),
    lam: wp.float32,
    out_positions: wp.array(dtype=wp.vec3),
) -> None:
    i = int(wp.tid())
    p = vertices[i]
    out_positions[i] = p
    if codes[i] != FREE_VERTEX or degree[i] == 0:
        return
    centroid = ring_sum[i] / float(degree[i])
    delta = centroid - p
    n = normals[i]
    tangential = project_out_normal(delta, n)
    out_positions[i] = p + lam * tangential


@wp.kernel(enable_backward=False)
def reproject_vertices(
    mesh_id: wp.uint64,
    codes: wp.array(dtype=wp.int32),
    vertices: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    out_positions: wp.array(dtype=wp.vec3),
) -> None:
    i = int(wp.tid())
    p = vertices[i]
    out_positions[i] = p
    if codes[i] != FREE_VERTEX:
        return
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    if query.result:
        out_positions[i] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)


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
    quad = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if quad[0] < wp.int32(0):
        return
    first = quad[0]
    second = quad[2]
    apex0 = quad[3]

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
