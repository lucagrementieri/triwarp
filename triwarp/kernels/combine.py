"""Kernels for assembling meshes from parts and joining them at a boundary (``triwarp.combine``)."""

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import update_argmin
from triwarp.kernels.array import wrap_index as _wrap
from triwarp.kernels.predicates import triangle_aspect_ratio

# Sentinel standing in for "this band is unusable", finite so it can be compared and accumulated.
# The same value ``kernels.hole_filling`` uses for the single-hole DP, kept separate rather than
# imported: the two dynamic programs are independent and neither reads the other's tables.
BAD_METRIC = wp.constant(wp.float32(1e10))


@wp.kernel
def offset_packed_faces(
    piece_starts: wp.array[wp.int32], vertex_offsets: wp.array[wp.int32], faces: wp.array[wp.int32]
) -> None:
    # Renumber a packed face buffer in place: every index of piece ``p`` shifts by that piece's
    # cumulative vertex offset. One launch over all indices, so the *number of pieces* costs
    # nothing here — the owning piece is found by an upper-bound search over its start offsets
    # (``piece_starts[0]`` is 0, so the search never returns 0). Pieces contributing no faces share
    # a start with their successor; the upper bound skips them, which is the right answer.
    i = int(wp.tid())
    piece = kernel_array.binary_search_index(piece_starts, i) - 1
    faces[i] = faces[i] + vertex_offsets[piece]


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
