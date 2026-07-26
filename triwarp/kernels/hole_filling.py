import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import update_argmin
from triwarp.kernels.array import wrap_index as _wrap
from triwarp.kernels.predicates import circumcircle_diameter_sq, triangle_aspect_ratio

# Big-but-finite penalty (MeshLib ``BadTriangulationMetric``): lets the DP keep a bad triangulation
# rather than break entirely, while staying below ``float`` precision limits when summed.
BAD_METRIC = wp.constant(wp.float32(1e10))

# Fill-metric selectors for ``triangle_fill_metric`` / ``fill_edge_term`` / ``fill_dp_span``.
METRIC_PLANE_NORMALIZED = wp.constant(wp.int32(0))
METRIC_MIN_AREA = wp.constant(wp.int32(1))
METRIC_CIRCUMSCRIBED = wp.constant(wp.int32(2))
METRIC_PLANE = wp.constant(wp.int32(3))
METRIC_MIN_TRI_ANGLE = wp.constant(wp.int32(4))
METRIC_EDGE_LENGTH = wp.constant(wp.int32(5))
METRIC_UNIVERSAL = wp.constant(wp.int32(6))
METRIC_MAX_DIHEDRAL = wp.constant(wp.int32(7))
METRIC_COMPLEX_FILL = wp.constant(wp.int32(8))

# How per-triangle / per-edge terms accumulate across the triangulation.
COMBINE_SUM = wp.constant(wp.int32(0))
COMBINE_MAX = wp.constant(wp.int32(1))

# sin of 60 degrees = sqrt(3)/2, the maximal possible minimal-angle sine (equilateral triangle).
MAX_MIN_ANGLE_SIN = wp.constant(wp.float32(0.86602540378443864676))


@wp.func
def loop_size(
    loop_starts: wp.array[wp.int32], total: wp.int32, n_loops: wp.int32, i: wp.int32
) -> wp.int32:
    # ``loop_starts`` is the exclusive scan of the loop sizes; the last loop ends at ``total``.
    end = total
    if i + 1 < n_loops:
        end = loop_starts[i + 1]
    return end - loop_starts[i]


@wp.kernel
def fan_faces(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    # This loop contributes s - 2 fan triangles; earlier loops occupy o - 2 * ell of them.
    base = o - 2 * ell
    for k in range(1, s - 1):
        t = base + (k - 1)
        out_faces[3 * t + 0] = flat_loops[o]
        out_faces[3 * t + 1] = flat_loops[o + k + 1]
        out_faces[3 * t + 2] = flat_loops[o + k]


@wp.kernel
def cone_faces(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    n_vertices: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    apex = n_vertices + ell
    # This loop contributes s cone triangles; the cone base equals o (scan of the loop sizes).
    for j in range(s):
        t = o + j
        nxt = o + (j + 1) % s
        out_faces[3 * t + 0] = apex
        out_faces[3 * t + 1] = flat_loops[nxt]
        out_faces[3 * t + 2] = flat_loops[o + j]


@wp.kernel
def loop_centroids(
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    out_centroids: wp.array[wp.vec3],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(s):
        acc = acc + vertices[flat_loops[o + j]]
    out_centroids[ell] = acc * (1.0 / wp.float32(s))


# --- Boundary-to-boundary zippering (``triangulate_boundaries`` / ``stitch``) ---------------


@wp.func
def searchsorted_right(edge: wp.array[wp.int32], n: wp.int32, value: wp.int32) -> wp.int32:
    # Number of entries in the non-decreasing ``edge[0:n]`` that are ``<= value``
    # (``numpy.searchsorted(..., side="right")``), via binary search.
    lo = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    hi = int(n)
    while lo < hi:
        mid = (lo + hi) // 2
        if edge[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def cyclic_gather(
    src: wp.array[wp.int32],
    n: wp.int32,
    shift: wp.int32,
    flip: wp.bool,
    value_offset: wp.int32,
    out_gathered: wp.array[wp.int32],
) -> None:
    # out[i] = src[wrap(index)] + value_offset with index = n-1-i (flip) or i+shift (roll);
    # covers loop reversal, cyclic rolls, and the roll-plus-vertex-offset variant in one kernel.
    i = int(wp.tid())
    j = i + shift
    if flip:
        j = n - 1 - i
    out_gathered[i] = src[_wrap(j, n)] + value_offset


@wp.kernel
def boundary_perimeters(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    n_a: wp.int32,
    out_perimeters: wp.array2d[wp.float32],
) -> None:
    i, j = wp.tid()
    edge_start = a_pos[i]
    edge_end = a_pos[_wrap(i + 1, n_a)]
    b = b_pos[j]
    out_perimeters[i, j] = wp.length(edge_start - b) + wp.length(edge_end - b)


@wp.kernel(enable_backward=False)
def row_argmin(
    perimeters: wp.array2d[wp.float32],
    m_b: wp.int32,
    out_col: wp.array[wp.int32],
    out_val: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    best_col = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    best_val = perimeters[i, 0]
    for j in range(1, m_b):
        update_argmin(best_val, best_col, perimeters[i, j], j)
    out_col[i] = best_col
    out_val[i] = best_val


@wp.kernel(enable_backward=False)
def global_argmin(
    col_min: wp.array[wp.int32],
    val_min: wp.array[wp.float32],
    n_a: wp.int32,
    out_shift: wp.array[wp.int32],
) -> None:
    # Single-thread reduction: out_shift = (shift_a, shift_b).
    best_row = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    best_val = val_min[0]
    for i in range(1, n_a):
        update_argmin(best_val, best_row, val_min[i], i)
    out_shift[0] = best_row
    out_shift[1] = col_min[best_row]


@wp.kernel
def rolled_edge_map(
    col_min: wp.array[wp.int32],
    shift_a: wp.int32,
    shift_b: wp.int32,
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Per-edge B vertex after rolling both loops so the global-min pair is first
    # (``argmin(roll(roll(perimeters, -shift_a, 0), -shift_b, 1), axis=1)``).
    i = int(wp.tid())
    out_edge[i] = _wrap(col_min[_wrap(i + shift_a, n_a)] - shift_b, m_b)


@wp.kernel(enable_backward=False)
def resolve_corrections(
    perimeters: wp.array2d[wp.float32],
    unsorted_indices: wp.array[wp.int32],
    next_indices: wp.array[wp.int32],
    n_corrections: wp.int32,
    row_roll: wp.int32,
    col_roll: wp.int32,
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Single-thread sequential correction: force ``out_edge`` non-decreasing by re-picking, for
    # each unsorted edge, the B vertex minimizing the perimeter within the bracket of its stable
    # neighbours. ``out_edge`` has length ``n_a + 1`` with the sentinel ``out_edge[n_a] == m_b``.
    # ``perimeters`` is the unrolled matrix, indexed through the running ``row_roll``/``col_roll``.
    for k in range(n_corrections):
        idx = unsorted_indices[k]
        lo = out_edge[idx - 1]
        hi = out_edge[next_indices[k]]
        if hi > m_b - 1:
            hi = m_b - 1
        row = _wrap(idx + row_roll, n_a)
        best_col = lo
        best_val = perimeters[row, _wrap(lo + col_roll, m_b)]
        for c in range(lo + 1, hi + 1):
            update_argmin(best_val, best_col, perimeters[row, _wrap(c + col_roll, m_b)], c)
        out_edge[idx] = best_col


@wp.kernel
def bridge_a_faces(
    roll_loop_a: wp.array[wp.int32],
    roll_loop_b: wp.array[wp.int32],
    edge: wp.array[wp.int32],
    n_a: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    out_faces[3 * i + 0] = roll_loop_a[i]
    out_faces[3 * i + 1] = roll_loop_a[_wrap(i + 1, n_a)]
    out_faces[3 * i + 2] = roll_loop_b[edge[i]]


@wp.kernel
def bridge_b_faces(
    roll_loop_a: wp.array[wp.int32],
    roll_loop_b: wp.array[wp.int32],
    edge: wp.array[wp.int32],
    n_a: wp.int32,
    m_b: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    j = int(wp.tid())
    apex = roll_loop_a[searchsorted_right(edge, n_a, j) % n_a]
    # The B edge is reversed (``fliplr``) so the bridge winding matches mesh B's faces.
    out_faces[3 * j + 0] = roll_loop_b[_wrap(j + 1, m_b)]
    out_faces[3 * j + 1] = roll_loop_b[j]
    out_faces[3 * j + 2] = apex


# --- Minimum-weight hole triangulation (Liepa interval DP, port of MRMeshFillHole.cpp) --------


@wp.func
def circumcircle_diameter(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # Diameter (not squared) of triangle ABC's circumcircle; +inf when degenerate, which the
    # square root preserves.
    return wp.sqrt(circumcircle_diameter_sq(a, b, c))


@wp.func
def triangle_double_area(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # Twice the triangle area (MeshLib ``dblArea``); the min-area fallback metric.
    return wp.length(wp.cross(b - a, c - a))


@wp.func
def min_triangle_angle_sin(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # sin of smallest angle = shortest edge / circumcircle diameter (MeshLib minTriangleAngleSin).
    ab = wp.length(b - a)
    ca = wp.length(a - c)
    bc = wp.length(c - b)
    if ab <= 0.0 or ca <= 0.0 or bc <= 0.0:
        return 0.0
    f = wp.length(wp.cross(b - a, c - a))
    return f * wp.min(wp.vec3(ab, ca, bc)) / (ab * ca * bc)


@wp.func
def dihedral_angle(left_norm: wp.vec3, right_norm: wp.vec3, edge_vec: wp.vec3) -> wp.float32:
    # Signed dihedral angle from two (possibly un-normalized) face normals and the shared edge
    # (MeshLib ``dihedralAngle`` = atan2(sin, cos)); atan2 is invariant to the common |L||R| scale.
    edge_dir = wp.normalize(edge_vec)
    sin_a = wp.dot(edge_dir, wp.cross(left_norm, right_norm))
    cos_a = wp.dot(left_norm, right_norm)
    return wp.atan2(sin_a, cos_a)


@wp.func
def triangle_fill_metric(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    plane_normal: wp.vec3,
    char_area: wp.float32,
    metric_id: wp.int32,
) -> wp.float32:
    # Per-triangle term of each fill metric (0 for the purely edge-based metrics).
    if metric_id == METRIC_MIN_AREA:
        return triangle_double_area(a, b, c)
    if metric_id == METRIC_CIRCUMSCRIBED:
        return circumcircle_diameter(a, b, c)
    if metric_id == METRIC_MIN_TRI_ANGLE:
        return wp.exp(25.0 * (MAX_MIN_ANGLE_SIN - min_triangle_angle_sin(a, b, c)))
    if metric_id == METRIC_EDGE_LENGTH or metric_id == METRIC_MAX_DIHEDRAL:
        return 0.0
    if metric_id == METRIC_UNIVERSAL:
        return circumcircle_diameter(a, b, c)
    if metric_id == METRIC_COMPLEX_FILL:
        aspect_ratio = triangle_aspect_ratio(a, b, c)
        if aspect_ratio > BAD_METRIC:
            return BAD_METRIC
        # 1e2 == MeshLib's empirical ``TriangleAreaModifier``; char_area == 1 / maxEdgeLengthSq.
        return aspect_ratio + 100.0 * triangle_double_area(a, b, c) * char_area
    if metric_id == METRIC_PLANE:
        if wp.dot(plane_normal, wp.cross(b - a, c - a)) < 0.0:
            return BAD_METRIC
        return circumcircle_diameter(a, b, c)
    # METRIC_PLANE_NORMALIZED (default, MeshLib getPlaneNormalizedFillMetric): penalize triangles
    # flipped or tilted > 60 degrees off the hole plane, and thin slivers.
    face_norm = wp.cross(b - a, c - a)
    face_dbl_area_sq = wp.length_sq(face_norm)
    if face_dbl_area_sq == 0.0:
        return BAD_METRIC
    dot_res = wp.dot(plane_normal, face_norm)
    if dot_res < 0.0 or dot_res * dot_res * 4.0 < face_dbl_area_sq:
        return BAD_METRIC
    aspect_ratio = triangle_aspect_ratio(a, b, c)
    if aspect_ratio > BAD_METRIC:
        return BAD_METRIC
    return circumcircle_diameter(a, b, c) * aspect_ratio


@wp.func
def fill_edge_term(
    a: wp.vec3, b: wp.vec3, lft: wp.vec3, rgt: wp.vec3, metric_id: wp.int32
) -> wp.float32:
    # Per-edge term for edge a->b, with opposite apexes lft (left) and rgt (right); 0 for the
    # triangle-only metrics. Ports the ``edgeMetric`` lambdas in MeshLib ``MRMeshMetrics.cpp``.
    if metric_id == METRIC_EDGE_LENGTH:
        return wp.length(b - a)
    ab = b - a
    if metric_id == METRIC_UNIVERSAL:
        norm_l = wp.cross(lft - a, ab)
        norm_r = wp.cross(ab, rgt - a)
        dbl_area = wp.length(norm_l) + wp.length(norm_r)
        return wp.sqrt(dbl_area) * wp.exp(5.0 * wp.abs(dihedral_angle(norm_l, norm_r, ab)))
    if metric_id == METRIC_MAX_DIHEDRAL:
        norm_l = wp.cross(lft - a, ab)
        norm_r = wp.cross(ab, rgt - a)
        return wp.abs(dihedral_angle(norm_l, norm_r, ab))
    if metric_id == METRIC_COMPLEX_FILL:
        norm_a = wp.cross(rgt - b, -ab)
        norm_c = wp.cross(lft - a, ab)
        denom = wp.length(norm_a) * wp.length(norm_c)
        if denom == 0.0:
            return BAD_METRIC
        cos_ac = wp.dot(norm_a, norm_c) / denom
        if cos_ac <= -1.0:
            return BAD_METRIC
        t = (1.0 - cos_ac) / (1.0 + cos_ac)
        return t * t * t * t
    return 0.0


@wp.func
def combine_metric(accumulated: wp.float32, term: wp.float32, combine_id: wp.int32) -> wp.float32:
    if combine_id == COMBINE_MAX:
        return wp.max(accumulated, term)
    return accumulated + term


@wp.kernel
def closed_edge_sq_lengths(
    loop_pos: wp.array[wp.vec3], n: wp.int32, out_sq: wp.array[wp.float32]
) -> None:
    # Squared length of each closed-loop rim edge ``loop[i] -> loop[(i + 1) % n]``; the max feeds
    # the ``char_area`` (1 / maxEdgeLengthSq) scale of the complex-fill metric.
    i = int(wp.tid())
    d = loop_pos[_wrap(i + 1, n)] - loop_pos[i]
    out_sq[i] = wp.length_sq(d)


@wp.kernel
def init_dp_base(dp: wp.array2d[wp.float32], prev: wp.array2d[wp.int32], b: wp.int32) -> None:
    # dp[i, j] = min metric to triangulate the sub-polygon on chord (i, j) and the arc i..j;
    # adjacent spans (rim edges) cost 0, everything else starts unfilled.
    i, j = wp.tid()
    prev[i, j] = -1
    if j == i + 1:
        dp[i, j] = 0.0
    else:
        dp[i, j] = BAD_METRIC


@wp.kernel(enable_backward=False)
def fill_dp_span(
    loop_pos: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    forbidden: wp.array2d[wp.int32],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_area: wp.float32,
    metric_id: wp.int32,
    combine_id: wp.int32,
    smooth_bd: wp.int32,
    span: wp.int32,
    b: wp.int32,
    dp: wp.array2d[wp.float32],
    prev: wp.array2d[wp.int32],
) -> None:
    # One thread per span-``span`` interval (i, j = i + span); reads only strictly smaller spans, so
    # successive launches (span = 2, 3, ...) are the DP barriers. Adds per-edge (dihedral) terms for
    # the interior chords (i, k) / (k, j) using the neighbouring sub-interval's apex, and the
    # boundary (smoothBd) rim edges using the existing face's opposite vertex.
    i = int(wp.tid())
    j = i + span
    if forbidden[i, j] != 0:
        # Interior chord would duplicate an existing mesh edge (non-manifold) — leave it unfilled.
        dp[i, j] = BAD_METRIC
        prev[i, j] = -1
        return
    a_pos = loop_pos[i]
    c_pos = loop_pos[j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = int(-1)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    for k in range(i + 1, j):
        k_pos = loop_pos[k]
        tri = triangle_fill_metric(a_pos, k_pos, c_pos, plane_normal, char_area, metric_id)
        val = combine_metric(dp[i, k], dp[k, j], combine_id)
        val = combine_metric(val, tri, combine_id)

        # Edge (i, k): neighbour on the other side is the sub-triangulation apex, or (rim edge) the
        # existing face's opposite vertex.
        if k > i + 1:
            if prev[i, k] >= 0:
                e = fill_edge_term(a_pos, k_pos, loop_pos[prev[i, k]], c_pos, metric_id)
                val = combine_metric(val, e, combine_id)
        elif smooth_bd != 0 and rim_opp_valid[i] != 0:
            e = fill_edge_term(a_pos, k_pos, rim_opp_pos[i], c_pos, metric_id)
            val = combine_metric(val, e, combine_id)

        # Edge (k, j).
        if j > k + 1:
            if prev[k, j] >= 0:
                e = fill_edge_term(k_pos, c_pos, loop_pos[prev[k, j]], a_pos, metric_id)
                val = combine_metric(val, e, combine_id)
        elif smooth_bd != 0 and rim_opp_valid[k] != 0:
            e = fill_edge_term(k_pos, c_pos, rim_opp_pos[k], a_pos, metric_id)
            val = combine_metric(val, e, combine_id)

        # Closing rim edge (loop[b-1] -> loop[0]) is the base of the whole loop and has no parent,
        # so its boundary term is added here for the top interval only.
        if is_top and smooth_bd != 0 and rim_opp_valid[b - 1] != 0:
            e = fill_edge_term(a_pos, c_pos, rim_opp_pos[b - 1], k_pos, metric_id)
            val = combine_metric(val, e, combine_id)

        update_argmin(best_val, best_k, val, k)
    dp[i, j] = best_val
    prev[i, j] = best_k


# --- Metric-based two-hole stitching (grid DP, port of MRMeshFillHole.cpp stitchHoles) ---------

# Stitch-metric selectors for ``stitch_triangle_metric`` / ``stitch_edge_metric`` / stitch DP.
METRIC_COMPLEX_STITCH = wp.constant(wp.int32(0))
METRIC_EDGE_LENGTH_STITCH = wp.constant(wp.int32(1))
METRIC_VERTICAL_STITCH = wp.constant(wp.int32(2))

# came-from directions stored per grid cell.
CAME_NONE = wp.constant(wp.int32(-1))
CAME_A = wp.constant(wp.int32(0))  # reached (i, j) by advancing loop A: from (i-1, j)
CAME_B = wp.constant(wp.int32(1))  # reached (i, j) by advancing loop B: from (i, j-1)


@wp.func
def stitch_triangle_metric(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, metric_id: wp.int32
) -> wp.float32:
    # Per-band-triangle term of each stitch metric (MeshLib arg order preserved).
    if metric_id == METRIC_EDGE_LENGTH_STITCH:
        return wp.length(c - a)
    if metric_id == METRIC_VERTICAL_STITCH:
        ab = b - a
        ac = c - a
        bc = c - b
        norm = wp.cross(ab, ac)
        pp = wp.abs(wp.dot(up, norm))
        sides = wp.length_sq(ab) + wp.length_sq(ac) + wp.length_sq(bc)
        return wp.length_sq(norm) + 100.0 * pp * pp + 0.5 * sides * sides
    # METRIC_COMPLEX_STITCH: proportional to triangle aspect ratio (1e-2 as aspect is unbounded).
    return (triangle_aspect_ratio(a, b, c) - 1.0) * 0.01


@wp.func
def stitch_edge_metric(a: wp.vec3, b: wp.vec3, lft: wp.vec3, rgt: wp.vec3) -> wp.float32:
    # complex_stitch edge term: (1 - cos dihedral) * 1e4 between the two triangles sharing edge a-b
    # (MeshLib getComplexStitchMetric edgeMetric). Normals are unit here (unlike the fill dihedral).
    ab = b - a
    norm_l = wp.normalize(wp.cross(lft - a, ab))
    norm_r = wp.normalize(wp.cross(ab, rgt - a))
    return (1.0 - wp.dot(norm_l, norm_r)) * 1.0e4


@wp.func
def stitch_prev_apex(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    came: wp.array2d[wp.int32],
    n_a: wp.int32,
    n_b: wp.int32,
    i: wp.int32,
    j: wp.int32,
) -> wp.vec3:
    # Apex of the band triangle already placed at cell (i, j) — the vertex advanced to reach it
    # (MeshLib ``cOp``). Only called when came[i, j] is valid.
    if came[i, j] == CAME_A:
        return a_pos[(i - 1) % n_a]
    return b_pos[(j - 1) % n_b]


@wp.kernel
def pair_sq_distances(
    a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3], out_dist: wp.array2d[wp.float32]
) -> None:
    # Squared distance between every rim-A vertex ``i`` and rim-B vertex ``j``; the global argmin
    # (reused ``row_argmin`` + ``global_argmin``) is MeshLib's aligned start pair for stitchHoles.
    i, j = wp.tid()
    d = a_pos[i] - b_pos[j]
    out_dist[i, j] = wp.length_sq(d)


@wp.kernel
def set_dp_origin(dp: wp.array2d[wp.float32]) -> None:
    # Seed the stitch grid DP: the empty band consuming 0 edges of either loop costs nothing.
    dp[0, 0] = 0.0


@wp.kernel(enable_backward=False)
def stitch_dp_diag(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    a_opp: wp.array[wp.vec3],
    a_opp_valid: wp.array[wp.int32],
    b_opp: wp.array[wp.vec3],
    b_opp_valid: wp.array[wp.int32],
    up: wp.vec3,
    metric_id: wp.int32,
    n_a: wp.int32,
    n_b: wp.int32,
    diag: wp.int32,
    dp: wp.array2d[wp.float32],
    came: wp.array2d[wp.int32],
) -> None:
    # One thread per cell (i, j) on anti-diagonal ``diag = i + j``; each reads only the previous
    # diagonal, so launching diag = 1, 2, ... in order are the DP barriers. dp[i, j] = min cost of
    # the band consuming i edges of A and j edges of B from the aligned start (cell (0, 0)).
    i_lo = wp.max(0, diag - n_b)
    i = i_lo + int(wp.tid())
    j = diag - i
    if i > n_a or j < 0 or j > n_b or (i == 0 and j == 0):
        return
    # Never let a full ring come from one loop before touching the other.
    if (i == n_a and j == 0) or (j == n_b and i == 0):
        dp[i, j] = BAD_METRIC
        came[i, j] = CAME_NONE
        return

    complex_edge = metric_id == METRIC_COMPLEX_STITCH
    best = FLOAT32_INF_CONSTANT
    best_came = CAME_NONE

    # Advance loop A: new triangle (a[i-1], b[j], a[i]) (MeshLib addALoop arg order).
    if i >= 1 and dp[i - 1, j] < BAD_METRIC:
        a_prev = a_pos[(i - 1) % n_a]
        a_cur = a_pos[i % n_a]
        b_cur = b_pos[j % n_b]
        w = dp[i - 1, j] + stitch_triangle_metric(a_prev, b_cur, a_cur, up, metric_id)
        if complex_edge:
            if came[i - 1, j] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, came, n_a, n_b, i - 1, j)
                w = w + stitch_edge_metric(a_prev, b_cur, c_op, a_cur)
            if a_opp_valid[(i - 1) % n_a] != 0:
                w = w + stitch_edge_metric(a_cur, a_prev, a_opp[(i - 1) % n_a], b_cur)
        update_argmin(best, best_came, w, CAME_A)

    # Advance loop B: new triangle (a[i], b[j-1], b[j]).
    if j >= 1 and dp[i, j - 1] < BAD_METRIC:
        a_cur = a_pos[i % n_a]
        b_prev = b_pos[(j - 1) % n_b]
        b_cur = b_pos[j % n_b]
        w = dp[i, j - 1] + stitch_triangle_metric(a_cur, b_prev, b_cur, up, metric_id)
        if complex_edge:
            if came[i, j - 1] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, came, n_a, n_b, i, j - 1)
                w = w + stitch_edge_metric(a_cur, b_prev, c_op, b_cur)
            if b_opp_valid[j % n_b] != 0:
                w = w + stitch_edge_metric(b_prev, b_cur, b_opp[j % n_b], a_cur)
        update_argmin(best, best_came, w, CAME_B)

    dp[i, j] = best
    came[i, j] = best_came


@wp.kernel
def edge_third_vertex(faces: wp.array[wp.int32], out_third: wp.array[wp.int32]) -> None:
    # Third vertex per faces_to_edges row: edge k of face f is (v_k, v_{k+1}), third is v_{k+2}.
    r = int(wp.tid())
    f = r // 3
    k = r % 3
    out_third[r] = faces[f * 3 + (k + 2) % 3]


@wp.kernel
def rim_opposite_from_table(
    loop: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.uint64],
    sorted_rows: wp.array[wp.int32],
    thirds: wp.array[wp.int32],
    max_index: wp.uint64,
    out_positions: wp.array[wp.vec3],
    out_valid: wp.array[wp.int32],
) -> None:
    # Per rim edge (loop[i], loop[i+1]): probe the sorted packed-edge table; a single hit means
    # exactly one adjacent existing face, whose third vertex blends the fill dihedral metrics
    # into the surface (host dict semantics of the former _rim_opposite).
    i = int(wp.tid())
    b = loop.shape[0]
    u = loop[i]
    v = loop[(i + 1) % b]
    lo_v = wp.min(u, v)
    hi_v = wp.max(u, v)
    key = wp.uint64(wp.uint32(lo_v)) + wp.uint64(wp.uint32(hi_v)) * max_index
    lo_idx = kernel_array.binary_search_index_left(sorted_keys, key)
    hi_idx = kernel_array.binary_search_index(sorted_keys, key)
    out_positions[i] = wp.vec3(0.0, 0.0, 0.0)
    out_valid[i] = wp.int32(0)
    if hi_idx - lo_idx == 1:
        out_positions[i] = vertices[thirds[sorted_rows[lo_idx]]]
        out_valid[i] = wp.int32(1)


@wp.kernel
def scatter_loop_positions(loop: wp.array[wp.int32], out_position: wp.array[wp.int32]) -> None:
    # Highest position wins on duplicate loop vertices, matching the last-write-wins dict the
    # host implementation built (dict insertion followed ascending positions).
    i = int(wp.tid())
    wp.atomic_max(out_position, loop[i], wp.int32(i))


@wp.kernel
def mark_forbidden_chords(
    edges_sorted: wp.array2d[wp.int32],
    position: wp.array[wp.int32],
    b: wp.int32,
    out_mask: wp.array2d[wp.int32],
) -> None:
    # Chords (non-adjacent loop positions) that already exist as mesh edges are forbidden
    # (MeshLib MultipleEdgesResolveMode::Simple). Idempotent writes: no atomics needed.
    e = int(wp.tid())
    pu = position[edges_sorted[e, 0]]
    pv = position[edges_sorted[e, 1]]
    if pu < 0 or pv < 0:
        return
    lo = wp.min(pu, pv)
    hi = wp.max(pu, pv)
    if hi - lo >= 2 and hi - lo <= int(b) - 2:
        out_mask[lo, hi] = wp.int32(1)
        out_mask[hi, lo] = wp.int32(1)


@wp.kernel
def clear_loop_positions(loop: wp.array[wp.int32], out_position: wp.array[wp.int32]) -> None:
    # Reset the touched slots so the (n_vertices,) scratch is reusable across loops.
    i = int(wp.tid())
    out_position[loop[i]] = wp.int32(-1)
