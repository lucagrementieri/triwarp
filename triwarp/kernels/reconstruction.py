"""
Kernels for point-cloud surface reconstruction (ported from MeshLib).

The heart is
[`build_local_triangulations`][triwarp.kernels.reconstruction.build_local_triangulations], a
per-point fan triangulation that mirrors MeshLib's
``TriangulationHelpers::buildLocalTriangulation``
(``reference/MeshLib/source/MRMesh/MRPointCloudTriangulationHelpers.cpp``). Every geometry helper
below is a direct port of the corresponding ``MRTriMath.h`` / ``MRReducePath`` primitive.
"""

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT

# Compile-time upper bound on the per-point fan size (neighbours kept for one center).
# Per-thread scratch arrays are sized to this; the runtime ``max_neighbours`` must not exceed it.
MAX_NEIGHBOURS = 64

# MeshLib: faces with aspect ratio above this are treated as degenerate and removed first.
CRITICAL_ASPECT_RATIO = wp.constant(wp.float32(1e3))
# MeshLib filterNeighbors: drop a neighbour whose oriented normal opposes the center normal.
NORMAL_FILTER_DOT = wp.constant(wp.float32(-0.3))
TWO_PI = wp.constant(wp.float32(6.283185307179586))
PI = wp.constant(wp.float32(3.141592653589793))


# --------------------------------------------------------------------------------------
# Geometry primitives (ports of MRTriMath.h / MRReducePath)
# --------------------------------------------------------------------------------------
@wp.func
def circumcircle_diameter_sq(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # MRTriMath.h: squared diameter of triangle ABC circumcircle.
    ab = wp.dot(b - a, b - a)
    ca = wp.dot(a - c, a - c)
    bc = wp.dot(c - b, c - b)
    if ab <= 0.0:
        return ca
    if ca <= 0.0:
        return bc
    if bc <= 0.0:
        return ab
    cr = wp.cross(b - a, c - a)
    f = wp.dot(cr, cr)
    if f <= 0.0:
        return FLOAT32_INF_CONSTANT
    return ab * ca * bc / f


@wp.func
def triangle_aspect_ratio(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # MRTriMath.h: circum-radius over twice in-radius; large for slivers.
    bc = wp.length(c - b)
    ca = wp.length(a - c)
    ab = wp.length(b - a)
    half_perimeter = (bc + ca + ab) / 2.0
    den = 8.0 * (half_perimeter - bc) * (half_perimeter - ca) * (half_perimeter - ab)
    if den <= 0.0:
        return FLOAT32_INF_CONSTANT
    return bc * ca * ab / den


@wp.func
def delone_flip_profit_sq(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3
) -> wp.float32:
    # MRPointCloudTriangulationHelpers.cpp: profit of flipping diagonal AC to BD.
    metric_ac = wp.max(circumcircle_diameter_sq(a, c, d), circumcircle_diameter_sq(c, a, b))
    metric_bd = wp.max(circumcircle_diameter_sq(b, d, a), circumcircle_diameter_sq(d, b, c))
    return metric_ac - metric_bd


@wp.func
def vec_angle(a: wp.vec3, b: wp.vec3) -> wp.float32:
    # MRVector3.h angle(a, b) = atan2(|cross|, dot); robust unsigned angle in [0, pi].
    return wp.atan2(wp.length(wp.cross(a, b)), wp.dot(a, b))


@wp.func
def tris_angle_profit(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, crit_ang: wp.float32
) -> wp.float32:
    # MRPointCloudTriangulationHelpers.cpp: dihedral angle across edge AC minus critical angle.
    ac = c - a
    ab = b - a
    ad = d - a
    dir_abc = wp.cross(ab, ac)
    dir_acd = wp.cross(ac, ad)
    return vec_angle(dir_abc, dir_acd) - crit_ang


@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> wp.float32:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def unfold_on_plane(b: wp.vec3, c: wp.vec3, d: wp.vec2, to_left: bool) -> wp.vec2:
    # MRReducePath.cpp unfoldOnPlane: place c in the plane relative to already-placed d.
    dot_bc = wp.dot(b, c)
    crs_bc = wp.length(wp.cross(b, c))
    dd = wp.dot(d, d)
    if dd <= 0.0:
        return wp.vec2(0.0, 0.0)
    if to_left:
        o = wp.vec2(-d[1], d[0])
    else:
        o = wp.vec2(d[1], -d[0])
    return (dot_bc * d + crs_bc * o) / dd


@wp.func
def line_isect(b: wp.vec2, c: wp.vec2, d: wp.vec2) -> wp.float32:
    # MRReducePath.cpp lineIsect: parameter where segment 0-B meets line C-D.
    c1 = cross2(d, c)
    c2 = cross2(c - b, d - b)
    if c1 == 0.0 and c2 == 0.0:
        bb = wp.dot(b, b)
        if bb == 0.0:
            return 0.0
        return (wp.dot(c, b) + wp.dot(d, b)) / (2.0 * bb)
    cc = c1 + c2
    if cc == 0.0:
        return 0.0
    return c1 / cc


@wp.func
def is_unfold_quadrangle_convex(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3
) -> bool:
    # MRReducePath: unfold triangles ABC and ACD into a plane; convex iff the B-D path
    # crosses edge AC strictly between A and C.
    vec_b = b - a
    vec_c = c - a
    vec_d = d - a
    unfold_b = wp.vec2(wp.length(vec_b), 0.0)
    unfold_c = unfold_on_plane(vec_b, vec_c, unfold_b, True)
    unfold_d = unfold_on_plane(vec_c, vec_d, unfold_c, True)
    x = line_isect(unfold_c, unfold_b, unfold_d)
    return x > 0.0 and x < 1.0


# --------------------------------------------------------------------------------------
# Fan cyclic navigation over the compacted local neighbour array
# --------------------------------------------------------------------------------------
@wp.func
def cycle_next(nbr: wp.array(dtype=wp.int32), m: wp.int32, i: wp.int32) -> wp.int32:
    j = i
    for _ in range(m):
        j = j + 1
        if j >= m:
            j = 0
        if nbr[j] >= 0:
            return j
    return i


@wp.func
def cycle_prev(nbr: wp.array(dtype=wp.int32), m: wp.int32, i: wp.int32) -> wp.int32:
    j = i
    for _ in range(m):
        j = j - 1
        if j < 0:
            j = m - 1
        if nbr[j] >= 0:
            return j
    return i


# --------------------------------------------------------------------------------------
# Fan edge-removal weight (port of FanOptimizer::calcQueueElement_ / updateBorderQueueElement_)
# --------------------------------------------------------------------------------------
@wp.func
def edge_removal_weight(
    center: wp.int32,
    i: wp.int32,
    m: wp.int32,
    border: wp.int32,
    crit_angle: wp.float32,
    normalizer_sq: wp.float32,
    nbr: wp.array(dtype=wp.int32),
    ang: wp.array(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
) -> wp.vec2:
    # Returns (weight, stable) with stable encoded as result[1] > 0.5.
    stable = wp.vec2(0.0, 1.0)
    prev = cycle_prev(nbr, m, i)
    nxt = cycle_next(nbr, m, i)

    a = points[center]
    n_center = normals[center]

    # --- boundary-edge handling (updateBorderQueueElement_) ---
    if border >= 0 and (nbr[i] == border or nbr[prev] == border):
        next_el = nbr[prev] == border
        if next_el:
            prev_ind = i
            next_ind = nxt
            other_id = nxt
        else:
            prev_ind = prev
            next_ind = i
            other_id = prev
        length_sq = wp.dot(a - points[nbr[i]], a - points[nbr[i]])
        other_length_sq = wp.dot(a - points[nbr[other_id]], a - points[nbr[other_id]])
        if length_sq < other_length_sq:
            return stable
        bb = points[nbr[prev_ind]]
        cc = points[nbr[next_ind]]
        if triangle_aspect_ratio(a, bb, cc) <= CRITICAL_ASPECT_RATIO:
            return stable
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # --- interior-edge handling (calcQueueElement_) ---
    dif_angle = ang[nxt] - ang[prev]
    if dif_angle < 0.0:
        dif_angle += TWO_PI
    if dif_angle > PI:
        return stable  # removing this edge would leave an angle > pi

    av = center
    bv = nbr[nxt]
    cv = nbr[i]
    dv = nbr[prev]
    b = points[bv]
    c = points[cv]
    d = points[dv]

    ac_length_sq = wp.dot(a - c, a - c)
    if (
        ac_length_sq > wp.dot(b - a, b - a)
        and triangle_aspect_ratio(a, b, c) > CRITICAL_ASPECT_RATIO
    ) or (
        ac_length_sq > wp.dot(d - a, d - a)
        and triangle_aspect_ratio(a, c, d) > CRITICAL_ASPECT_RATIO
    ):
        # degenerate triangle, longest edge -> remove as fast as possible
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # flip possibility: trusted normals allow a flip across a fold, else require convexity
    flip = False
    if wp.dot(n_center, normals[cv]) < 0.0:
        flip = True
    else:
        flip = is_unfold_quadrangle_convex(a, b, c, d)
    if not flip:
        return stable

    delone_prof = delone_flip_profit_sq(a, b, c, d)
    if delone_prof == 0.0 and wp.min(av, cv) > wp.min(bv, dv):
        delone_prof = -1.0
    angle_prof = tris_angle_profit(a, b, c, d, crit_angle)
    if delone_prof < 0.0 and angle_prof <= 0.0:
        return stable

    weight = 0.0
    if delone_prof > 0.0:
        weight += delone_prof / normalizer_sq
    if angle_prof > 0.0:
        weight += angle_prof

    norm_val = wp.length(c - a)
    if norm_val == 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    plane_dist = wp.abs(wp.dot(n_center, c - a))
    weight += plane_dist / norm_val

    # trusted-normal agreement bonuses
    c_norm = normals[cv]
    weight += 5.0 * (1.0 - wp.dot(n_center, c_norm))
    tri_norm = wp.normalize(wp.cross(b - a, c - a) + wp.cross(c - a, d - a))
    tri_norm_weight = wp.dot(tri_norm, c_norm)
    if tri_norm_weight < 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    weight += 5.0 * (1.0 - tri_norm_weight)

    return wp.vec2(weight, 0.0)


# --------------------------------------------------------------------------------------
# Local fan triangulation (port of buildLocalTriangulation + FanOptimizer)
# --------------------------------------------------------------------------------------
@wp.kernel
def build_local_triangulations(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    neighbor_idx: wp.array2d(dtype=wp.int32),
    neighbor_dist: wp.array2d(dtype=wp.float32),
    radius: wp.float32,
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    out_tris: wp.array3d(dtype=wp.int32),
    out_valid: wp.array2d(dtype=wp.bool),
) -> None:
    v = int(wp.tid())
    k = neighbor_idx.shape[1]
    cap = out_tris.shape[1]

    a = points[v]
    n_center = normals[v]

    nbr = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.int32)
    ang = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.float32)

    # --- gather + filter neighbours ---
    m = 0
    for i in range(k):
        if m >= MAX_NEIGHBOURS:
            break
        nb = neighbor_idx[v, i]
        if nb < 0 or nb == v:
            continue
        if radius > 0.0 and neighbor_dist[v, i] > radius:
            continue
        pn = points[nb]
        if pn[0] == a[0] and pn[1] == a[1] and pn[2] == a[2]:
            continue  # coincident with center
        if wp.dot(n_center, normals[nb]) < NORMAL_FILTER_DOT:
            continue
        nbr[m] = nb
        m += 1

    if m < 2:
        return

    # --- tangent-plane basis (project neighbours onto plane through center) ---
    base = wp.vec3(0.0, 0.0, 0.0)
    normalizer_sq = 0.0
    for i in range(m):
        d = points[nbr[i]] - a
        pv = d - wp.dot(n_center, d) * n_center
        if normalizer_sq <= 0.0:
            base = pv
            normalizer_sq = wp.dot(pv, pv)
    if normalizer_sq <= 0.0:
        normalizer_sq = 1.0
    if wp.dot(base, base) > 0.0:
        base = wp.normalize(base)

    # --- polar angle of each neighbour around the center in the tangent plane ---
    for i in range(m):
        d = points[nbr[i]] - a
        pv = d - wp.dot(n_center, d) * n_center
        if wp.dot(pv, pv) > 0.0:
            vec = wp.normalize(pv)
        else:
            vec = base
        cp = wp.cross(vec, base)
        s = 1.0
        if wp.dot(cp, n_center) < 0.0:
            s = -1.0
        ang[i] = wp.atan2(s * wp.length(cp), wp.dot(vec, base))

    # --- sort neighbours by angle (selection sort; m <= MAX_NEIGHBOURS) ---
    for i in range(m):
        mn = i
        for j in range(i + 1, m):
            if ang[j] < ang[mn]:
                mn = j
        if mn != i:
            ta = ang[i]
            ang[i] = ang[mn]
            ang[mn] = ta
            tn = nbr[i]
            nbr[i] = nbr[mn]
            nbr[mn] = tn

    # --- boundary detection: first angular gap wider than boundary_angle ---
    border = -1
    for i in range(m):
        if i + 1 < m:
            diff = ang[i + 1] - ang[i]
        else:
            diff = ang[0] + TWO_PI - ang[i]
        if diff > boundary_angle:
            border = nbr[i]
            break

    # --- greedy fan optimisation (linear-scan replacement of the priority queue) ---
    current = m
    for _step in range(m):
        best_w = -FLOAT32_INF_CONSTANT
        best_pos = -1
        for i in range(m):
            if nbr[i] < 0:
                continue
            res = edge_removal_weight(
                v, i, m, border, crit_angle, normalizer_sq, nbr, ang, points, normals
            )
            if res[1] > 0.5:
                continue  # stable, cannot remove
            if res[0] > best_w:
                best_w = res[0]
                best_pos = i
        if best_pos < 0:
            break
        old = nbr[best_pos]
        prv = cycle_prev(nbr, m, best_pos)
        nbr[best_pos] = -1
        current -= 1
        if old == border:
            border = nbr[prv]
        if current < 2:
            for i in range(m):
                nbr[i] = -1
            break

    # --- emit fan triangles between consecutive surviving neighbours ---
    slot = 0
    for i in range(m):
        if nbr[i] < 0:
            continue
        if border >= 0 and nbr[i] == border:
            continue  # boundary gap: no triangle here
        nxt = cycle_next(nbr, m, i)
        if nxt == i:
            continue
        bidx = nbr[i]
        cidx = nbr[nxt]
        pb = points[bidx]
        pc = points[cidx]
        # orient so the face normal agrees with the (trusted) center normal
        if wp.dot(wp.cross(pb - a, pc - a), n_center) < 0.0:
            tmp = bidx
            bidx = cidx
            cidx = tmp
        if slot < cap:
            out_tris[v, slot, 0] = v
            out_tris[v, slot, 1] = bidx
            out_tris[v, slot, 2] = cidx
            out_valid[v, slot] = True
            slot += 1


# --------------------------------------------------------------------------------------
# Canonical (sorted) triangle key + oriented copy, for repeated-triangle dedup
# --------------------------------------------------------------------------------------
@wp.kernel
def canonicalize_triangles(
    tris: wp.array2d(dtype=wp.int32),
    out_sorted: wp.array2d(dtype=wp.int32),
) -> None:
    t = int(wp.tid())
    i = tris[t, 0]
    j = tris[t, 1]
    k = tris[t, 2]
    # sort the three indices ascending (unoriented key)
    if i > j:
        tmp = i
        i = j
        j = tmp
    if j > k:
        tmp = j
        j = k
        k = tmp
    if i > j:
        tmp = i
        i = j
        j = tmp
    out_sorted[t, 0] = i
    out_sorted[t, 1] = j
    out_sorted[t, 2] = k


@wp.kernel
def copy_first_column(
    groups: wp.array2d(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
) -> None:
    t = int(wp.tid())
    out[t] = groups[t, 0]
