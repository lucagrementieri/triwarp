import warp as wp

from triwarp.kernels.unique import hash_slot

# Delaunay / Delone edge-flip constants (ported from MRMeshDelone.cpp). The flip predicate
# runs in float64: MeshLib deliberately widens to double because circumcircle diameters of
# near-degenerate triangles have too large a rounding error in float32 (infinite flip loops).
DELONE_CRITICAL_DOT = wp.constant(wp.float64(-0.9))
DELONE_EPS = wp.constant(wp.float64(1e-7))
NO_ANGLE_CHANGE_LIMIT = wp.constant(wp.float64(6.283185307179586))  # 2*pi (NoAngleChangeLimit)
F64_INF = wp.constant(wp.float64(1.0e308))
F32_LARGE = wp.constant(wp.float32(3.0e38))  # "disabled gate" sentinel (~FLT_MAX)


@wp.func
def edge_midpoint(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.vec3:
    v0 = vertices[unique_edges[e, 0]]
    v1 = vertices[unique_edges[e, 1]]
    return (v0 + v1) * wp.float32(0.5)


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = int(wp.tid())
    out_midpoints[k] = edge_midpoint(vertices, unique_edges, wp.int32(k))


@wp.kernel
def build_mid_idx(
    inverse: wp.array[wp.int32], vertex_offset: wp.int32, out_mid_idx: wp.array2d[wp.int32]
) -> None:
    f = int(wp.tid())
    out_mid_idx[f, 0] = inverse[f * 3 + 0] + vertex_offset
    out_mid_idx[f, 1] = inverse[f * 3 + 1] + vertex_offset
    out_mid_idx[f, 2] = inverse[f * 3 + 2] + vertex_offset


@wp.kernel
def subdivide_faces(
    faces: wp.array[wp.int32], mid_idx: wp.array2d[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = int(wp.tid())
    v0 = faces[f * 3 + 0]
    v1 = faces[f * 3 + 1]
    v2 = faces[f * 3 + 2]
    m0 = mid_idx[f, 0]
    m1 = mid_idx[f, 1]
    m2 = mid_idx[f, 2]
    base = f * 12
    # (v0, m0, m2)
    out_faces[base + 0] = v0
    out_faces[base + 1] = m0
    out_faces[base + 2] = m2
    # (m0, v1, m1)
    out_faces[base + 3] = m0
    out_faces[base + 4] = v1
    out_faces[base + 5] = m1
    # (m2, m1, v2)
    out_faces[base + 6] = m2
    out_faces[base + 7] = m1
    out_faces[base + 8] = v2
    # (m0, m1, m2)
    out_faces[base + 9] = m0
    out_faces[base + 10] = m1
    out_faces[base + 11] = m2


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


@wp.kernel
def build_face_mid(
    inverse: wp.array[wp.int32],
    midpoint_idx: wp.array[wp.int32],
    out_face_mid: wp.array2d[wp.int32],
) -> None:
    f = int(wp.tid())
    out_face_mid[f, 0] = midpoint_idx[inverse[f * 3 + 0]]
    out_face_mid[f, 1] = midpoint_idx[inverse[f * 3 + 1]]
    out_face_mid[f, 2] = midpoint_idx[inverse[f * 3 + 2]]


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

    s0 = wp.int32(0)
    s1 = wp.int32(0)
    s2 = wp.int32(0)
    if mv[0] >= 0:
        s0 = 1
    if mv[1] >= 0:
        s1 = 1
    if mv[2] >= 0:
        s2 = 1
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
        m0 = mv[0]
        m1 = mv[1]
        m2 = mv[2]
        t0 = wp.vec3i(fv[0], m0, m2)
        t1 = wp.vec3i(m0, fv[1], m1)
        t2 = wp.vec3i(m2, m1, fv[2])
        t3 = wp.vec3i(m0, m1, m2)
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
def _to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _to_vec2d(v: wp.vec2) -> wp.vec2d:
    return wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))


@wp.func
def _normal_d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.vec3d:
    n = wp.cross(b - a, c - a)
    length = wp.length(n)
    if length <= wp.float64(0.0):
        return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    return n / length


@wp.func
def _circumcircle_diameter_sq_d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    ab = wp.length_sq(b - a)
    ca = wp.length_sq(a - c)
    bc = wp.length_sq(c - b)
    if ab <= wp.float64(0.0):
        return ca
    if ca <= wp.float64(0.0):
        return bc
    if bc <= wp.float64(0.0):
        return ab
    f = wp.length_sq(wp.cross(b - a, c - a))
    if f <= wp.float64(0.0):
        return F64_INF
    return ab * ca * bc / f


@wp.func
def _mincircle_diameter_sq_d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    ab = wp.length_sq(b - a)
    ca = wp.length_sq(a - c)
    bc = wp.length_sq(c - b)
    if ca >= bc + ab:
        return ca
    if bc >= ab + ca:
        return bc
    if ab >= ca + bc:
        return ab
    f = wp.length_sq(wp.cross(b - a, c - a))
    if f <= wp.float64(0.0):
        return F64_INF
    return ab * ca * bc / f


@wp.func
def _dihedral_angle_d(left_n: wp.vec3d, right_n: wp.vec3d, edge_vec: wp.vec3d) -> wp.float64:
    edge_dir = wp.normalize(edge_vec)
    s = wp.dot(edge_dir, wp.cross(left_n, right_n))
    co = wp.dot(left_n, right_n)
    return wp.atan2(s, co)


@wp.func
def _triangle_aspect_ratio_d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d) -> wp.float64:
    bc = wp.length(c - b)
    ca = wp.length(a - c)
    ab = wp.length(b - a)
    half = (bc + ca + ab) / wp.float64(2.0)
    den = wp.float64(8.0) * (half - bc) * (half - ca) * (half - ab)
    if den <= wp.float64(0.0):
        return F64_INF
    return bc * ca * ab / den


@wp.func
def _cross2_d(u: wp.vec2d, v: wp.vec2d) -> wp.float64:
    return u[0] * v[1] - u[1] * v[0]


@wp.func
def _unfold_on_plane_d(b: wp.vec3d, c: wp.vec3d, d: wp.vec2d, to_left: wp.bool) -> wp.vec2d:
    dot_bc = wp.dot(b, c)
    crs_bc = wp.length(wp.cross(b, c))
    dd = wp.dot(d, d)
    if dd <= wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    o = wp.vec2d(-d[1], d[0])
    if not to_left:
        o = wp.vec2d(d[1], -d[0])
    return (dot_bc * d + crs_bc * o) / dd


@wp.func
def _line_isect_d(b: wp.vec2d, c: wp.vec2d, d: wp.vec2d) -> wp.float64:
    c1 = _cross2_d(d, c)
    c2 = _cross2_d(c - b, d - b)
    if c1 == wp.float64(0.0) and c2 == wp.float64(0.0):
        bb = wp.dot(b, b)
        if bb == wp.float64(0.0):
            return wp.float64(0.0)
        return (wp.dot(c, b) + wp.dot(d, b)) / (wp.float64(2.0) * bb)
    cc = c1 + c2
    if cc == wp.float64(0.0):
        return wp.float64(0.0)
    return c1 / cc


@wp.func
def _is_unfold_quad_convex_d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d) -> wp.bool:
    # Ports isUnfoldQuadrangleConvex(a,b,c,d): unfold triangles ABC/ACD into a plane and
    # test where the shortest B->D path crosses diagonal AC. Convex iff strictly interior.
    vec_b = b - a
    vec_c = c - a
    vec_d = d - a
    unfold_b = wp.vec2d(wp.length(vec_b), wp.float64(0.0))
    unfold_c = _unfold_on_plane_d(vec_b, vec_c, unfold_b, wp.bool(True))
    unfold_d = _unfold_on_plane_d(vec_c, vec_d, unfold_c, wp.bool(True))
    x = wp.clamp(_line_isect_d(unfold_c, unfold_b, unfold_d), wp.float64(0.0), wp.float64(1.0))
    return x > wp.float64(0.0) and x < wp.float64(1.0)


@wp.func
def _segments_dist_sq_d(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d) -> wp.float64:
    # Squared distance between segments [p1,q1] and [p2,q2] (Ericson, clamped closest points).
    eps = wp.float64(1e-30)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    aa = wp.dot(d1, d1)
    ee = wp.dot(d2, d2)
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
    n_abc = _normal_d(a, b, c)
    n_acd = _normal_d(a, c, d)
    old_pocket = wp.dot(n_abc, n_acd) < DELONE_CRITICAL_DOT

    n_abd = _normal_d(a, b, d)
    n_dbc = _normal_d(d, b, c)
    new_pocket = wp.dot(n_abd, n_dbc) < DELONE_CRITICAL_DOT

    if old_pocket != new_pocket:
        return new_pocket

    if old_pocket:
        metric_ac = wp.max(_mincircle_diameter_sq_d(a, c, d), _mincircle_diameter_sq_d(c, a, b))
        metric_bd = wp.max(_mincircle_diameter_sq_d(b, d, a), _mincircle_diameter_sq_d(d, b, c))
        return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)

    if max_angle_change < NO_ANGLE_CHANGE_LIMIT:
        old_angle = _dihedral_angle_d(n_abd, n_dbc, d - b)
        new_angle = _dihedral_angle_d(n_abc, n_acd, a - c)
        if wp.abs(old_angle - new_angle) > max_angle_change:
            return True

    metric_ac = wp.max(_circumcircle_diameter_sq_d(a, c, d), _circumcircle_diameter_sq_d(c, a, b))
    metric_bd = wp.max(_circumcircle_diameter_sq_d(b, d, a), _circumcircle_diameter_sq_d(d, b, c))

    if metric_ac >= F64_INF:
        if metric_bd >= F64_INF:
            return wp.length_sq(a - c) <= wp.length_sq(b - d)
        return False
    return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)


# ---------------------------------------------------------------------------
# 2D orientation / incircle predicate (for delaunay_triangulation)
# ---------------------------------------------------------------------------


@wp.func
def _orient2d_d(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d) -> wp.float64:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@wp.func
def _incircle_d(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d, d: wp.vec2d) -> wp.float64:
    # Positive iff d is inside the circumcircle of CCW triangle (a, b, c).
    ax = a[0] - d[0]
    ay = a[1] - d[1]
    bx = b[0] - d[0]
    by = b[1] - d[1]
    cx = c[0] - d[0]
    cy = c[1] - d[1]
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    return ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx)


# ---------------------------------------------------------------------------
# Shared parallel edge-flip core
# ---------------------------------------------------------------------------


@wp.func
def _edge_key(u: wp.int32, v: wp.int32, base: wp.uint64) -> wp.uint64:
    # Matches kernels.unique.pack_indices for a sorted 2-index row: min + max * base.
    lo = wp.uint64(wp.uint32(wp.min(u, v)))
    hi = wp.uint64(wp.uint32(wp.max(u, v)))
    return lo + hi * base


@wp.func
def _edge_exists(sorted_keys: wp.array[wp.uint64], n: wp.int32, key: wp.uint64) -> wp.bool:
    lo = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    hi = int(n)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < n:
        return sorted_keys[lo] == key
    return False


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


@wp.kernel
def delone_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    n_keys: wp.int32,
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
    u = adjacency_edges[k, 0]
    v = adjacency_edges[k, 1]
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 < 0 or d1 < 0:
        return
    quad = _resolve_flip_quad(faces, f0, u, v, d0, d1)
    a = quad[0]
    b = quad[1]
    c = quad[2]
    d = quad[3]
    if a < 0:
        return
    if b == d:
        return
    if _edge_exists(sorted_edge_keys, n_keys, _edge_key(b, d, key_base)):
        return
    out_quad[k, 0] = a
    out_quad[k, 1] = b
    out_quad[k, 2] = c
    out_quad[k, 3] = d
    ap = _to_vec3d(vertices[a])
    bp = _to_vec3d(vertices[b])
    cp = _to_vec3d(vertices[c])
    dp = _to_vec3d(vertices[d])
    if max_deviation_sq < F32_LARGE:
        if _segments_dist_sq_d(ap, cp, bp, dp) > wp.float64(max_deviation_sq):
            return
    if not _is_unfold_quad_convex_d(ap, bp, cp, dp):
        return
    angle = wp.float64(max_angle_change)
    if critical_aspect < F32_LARGE and angle < NO_ANGLE_CHANGE_LIMIT:
        max_aspect = wp.max(
            _triangle_aspect_ratio_d(ap, cp, dp), _triangle_aspect_ratio_d(cp, ap, bp)
        )
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
    n_keys: wp.int32,
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = int(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    u = adjacency_edges[k, 0]
    v = adjacency_edges[k, 1]
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 < 0 or d1 < 0:
        return
    quad = _resolve_flip_quad(faces, f0, u, v, d0, d1)
    a = quad[0]
    b = quad[1]
    c = quad[2]
    d = quad[3]
    if a < 0 or b == d:
        return
    if _edge_exists(sorted_edge_keys, n_keys, _edge_key(b, d, key_base)):
        return
    ap = _to_vec2d(points[a])
    bp = _to_vec2d(points[b])
    cp = _to_vec2d(points[c])
    dp = _to_vec2d(points[d])
    # Post-flip triangles (a, b, d) and (d, b, c) must both be positively oriented (convex quad).
    if _orient2d_d(ap, bp, dp) <= wp.float64(0.0) or _orient2d_d(dp, bp, cp) <= wp.float64(0.0):
        return
    out_quad[k, 0] = a
    out_quad[k, 1] = b
    out_quad[k, 2] = c
    out_quad[k, 3] = d
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
    slot = hash_slot(_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
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
    slot = hash_slot(_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
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
