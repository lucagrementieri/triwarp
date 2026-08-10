import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, INT32_MAX_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import update_argmin
from triwarp.kernels.array import wrap_index as _wrap
from triwarp.kernels.predicates import (
    circumcircle_diameter_sq,
    dihedral_angle,
    triangle_aspect_ratio,
)

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


@wp.func
def char_area_from_max(max_edge_sq: wp.float32) -> wp.float32:
    # MeshLib's ``char_area`` scale for the complex-fill metric: 1 / maxEdgeLengthSq, or 1 for a
    # fully degenerate rim.
    if max_edge_sq > 0.0:
        return 1.0 / max_edge_sq
    return 1.0


@wp.kernel
def loop_rim_metrics(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_max_edge_sq: wp.array[wp.float32],
    out_normal: wp.array[wp.vec3],
) -> None:
    # One thread per rim vertex of every loop at once. Each thread owns the rim edge leaving its
    # vertex and folds it into its loop's two per-loop scalars: the longest edge (segmented max,
    # the ``char_area`` scale) and the Newell normal sum (segmented sum, the hole plane). The
    # per-loop equivalents are ``tw.reduce.max`` over a gathered rim and
    # ``tw.polyline.polyline_normal``, each of which costs a host synchronization per loop.
    t = int(wp.tid())
    ell = loop_id[t]
    o = loop_starts[ell]
    b = loop_sizes[ell]
    a = vertices[flat_loops[t]]
    c = vertices[flat_loops[o + _wrap(t - o + 1, b)]]
    wp.atomic_max(out_max_edge_sq, ell, wp.length_sq(c - a))
    wp.atomic_add(out_normal, ell, wp.cross(a, c))


@wp.kernel
def loop_perimeters(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_perimeter: wp.array[wp.float32],
) -> None:
    # Segmented ``polyline_length(closed=True)``: the arc length of every loop in one launch, so
    # ``preserve_largest_hole`` costs one readback instead of two per loop.
    t = int(wp.tid())
    ell = loop_id[t]
    o = loop_starts[ell]
    b = loop_sizes[ell]
    a = vertices[flat_loops[t]]
    c = vertices[flat_loops[o + _wrap(t - o + 1, b)]]
    wp.atomic_add(out_perimeter, ell, wp.length(c - a))


@wp.kernel
def init_dp_base(
    loop_sizes: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_dp: wp.array[wp.float32],
    out_prev: wp.array[wp.int32],
) -> None:
    # dp[i, j] = min metric to triangulate the sub-polygon on chord (i, j) and the arc i..j;
    # adjacent spans (rim edges) cost 0, everything else starts unfilled. The ragged table packs
    # every loop's ``B x B`` block end to end, so loop ``ell`` owns
    # ``out_dp[dp_offsets[ell] + i * B + j]``. Launched over ``(n_loops, max_B)`` with one thread
    # per row: rows past a loop's own ``B`` exit immediately.
    ell, i = wp.tid()
    if active[ell] == 0:
        return
    b = loop_sizes[ell]
    if i >= b:
        return
    row = dp_offsets[ell] + i * b
    for j in range(b):
        out_prev[row + j] = -1
        if j == i + 1:
            out_dp[row + j] = 0.0
        else:
            out_dp[row + j] = BAD_METRIC


# Lanes per block of ``fill_dp_span_tiled``: one block owns one interval and its lanes stride the
# apex loop. Short spans leave lanes idle, which costs nothing the one-thread-per-interval kernel
# was not already wasting on its ~1 024-wide grid.
#
# **Measured flat between 32 and 128** on ``rim_short`` (two 512-vertex loops, medians of 15 over
# two runs each): 19.0/20.1 at 32, 20.5/35.4 at 64, 19.6/23.9 at 128, against 24.0/36.0 at 256 —
# i.e. within the noise band up to 128 and a regression past it, so the apex loop is not what the
# call is bound by. 32 wins the tie on cost per barrier: a single-warp block takes the
# ``warp_count == 1`` fast path in Warp's ``tile_reduce_impl`` (a ballot plus a warp shuffle, no
# cross-warp shared-memory round trip), measured at 126 ns per ``tile_min`` against 325 ns at 64
# and 369 ns at 128.
HOLE_DP_BLOCK = 32


@wp.func
def apex_cost(
    loop_pos: wp.array[wp.vec3],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
    metric_id: wp.int32,
    combine_id: wp.int32,
    smooth_bd: wp.int32,
    o: wp.int32,
    b: wp.int32,
    base: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    is_top: wp.bool,
    a_pos: wp.vec3,
    c_pos: wp.vec3,
    plane_normal: wp.vec3,
    char_area: wp.float32,
) -> wp.float32:
    # Metric of triangulating the interval (i, j) with apex ``k``: the two sub-intervals' costs,
    # this triangle's term, and the per-edge (dihedral) terms for the interior chords (i, k) /
    # (k, j) taken at the neighbouring sub-interval's apex — or, for a rim edge, at the existing
    # face's opposite vertex when ``smooth_bd`` is set.
    k_pos = loop_pos[o + k]
    tri = triangle_fill_metric(a_pos, k_pos, c_pos, plane_normal, char_area, metric_id)
    val = combine_metric(dp[base + i * b + k], dp[base + k * b + j], combine_id)
    val = combine_metric(val, tri, combine_id)

    if k > i + 1:
        if prev[base + i * b + k] >= 0:
            e = fill_edge_term(a_pos, k_pos, loop_pos[o + prev[base + i * b + k]], c_pos, metric_id)
            val = combine_metric(val, e, combine_id)
    elif smooth_bd != 0 and rim_opp_valid[o + i] != 0:
        e = fill_edge_term(a_pos, k_pos, rim_opp_pos[o + i], c_pos, metric_id)
        val = combine_metric(val, e, combine_id)

    if j > k + 1:
        if prev[base + k * b + j] >= 0:
            e = fill_edge_term(k_pos, c_pos, loop_pos[o + prev[base + k * b + j]], a_pos, metric_id)
            val = combine_metric(val, e, combine_id)
    elif smooth_bd != 0 and rim_opp_valid[o + k] != 0:
        e = fill_edge_term(k_pos, c_pos, rim_opp_pos[o + k], a_pos, metric_id)
        val = combine_metric(val, e, combine_id)

    # Closing rim edge (loop[b-1] -> loop[0]) is the base of the whole loop and has no parent, so
    # its boundary term is added here for the top interval only.
    if is_top and smooth_bd != 0 and rim_opp_valid[o + b - 1] != 0:
        e = fill_edge_term(a_pos, c_pos, rim_opp_pos[o + b - 1], k_pos, metric_id)
        val = combine_metric(val, e, combine_id)
    return val


@wp.kernel(enable_backward=False)
def fill_dp_span(
    loop_pos: wp.array[wp.vec3],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    active: wp.array[wp.int32],
    plane_normals: wp.array[wp.vec3],
    forbidden: wp.array[wp.int32],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_areas: wp.array[wp.float32],
    metric_id: wp.int32,
    combine_id: wp.int32,
    smooth_bd: wp.int32,
    span: wp.int32,
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
) -> None:
    # One thread per span-``span`` interval (i, j = i + span) of every loop at once; reads only
    # strictly smaller spans, so successive launches (span = 2, 3, ...) are the DP barriers.
    #
    # Launched over ``(n_loops, max_B - span)``: threads whose loop is shorter than the current
    # span exit at once, so a mesh whose loops differ wildly in length wastes some of the grid, but
    # the launch *count* is ``max_B - 1`` for the whole mesh instead of ``B - 1`` per loop.
    #
    # This is the **CPU** engine and the tie-break reference; CUDA runs
    # ``fill_dp_span_tiled``, which must agree with it apex for apex (see
    # ``tests/test_holes.py::test_fill_dp_span_tiled_matches_serial``).
    ell, i = wp.tid()
    if active[ell] == 0:
        return
    b = loop_sizes[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = dp_offsets[ell]
    if forbidden[base + i * b + j] != 0:
        # Interior chord would duplicate an existing mesh edge (non-manifold) — leave it unfilled.
        dp[base + i * b + j] = BAD_METRIC
        prev[base + i * b + j] = -1
        return
    o = loop_starts[ell]
    plane_normal = plane_normals[ell]
    char_area = char_areas[ell]
    a_pos = loop_pos[o + i]
    c_pos = loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = int(-1)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    for k in range(i + 1, j):
        val = apex_cost(
            loop_pos,
            rim_opp_pos,
            rim_opp_valid,
            dp,
            prev,
            metric_id,
            combine_id,
            smooth_bd,
            o,
            b,
            base,
            i,
            j,
            k,
            is_top,
            a_pos,
            c_pos,
            plane_normal,
            char_area,
        )
        update_argmin(best_val, best_k, val, k)
    dp[base + i * b + j] = best_val
    prev[base + i * b + j] = best_k


@wp.kernel(enable_backward=False)
def fill_dp_span_tiled(
    loop_pos: wp.array[wp.vec3],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    active: wp.array[wp.int32],
    plane_normals: wp.array[wp.vec3],
    forbidden: wp.array[wp.int32],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_areas: wp.array[wp.float32],
    metric_id: wp.int32,
    combine_id: wp.int32,
    smooth_bd: wp.int32,
    span: wp.int32,
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
) -> None:
    # One *block* per span-``span`` interval, its ``HOLE_DP_BLOCK`` lanes striding the apex loop.
    # Same DP, same launch count, ``HOLE_DP_BLOCK`` times the parallelism: the serial kernel above
    # puts at most ``n_loops * (max_B - span)`` threads on the machine, which for a single long
    # boundary is a few hundred out of a few hundred thousand.
    #
    # **The tie-break is the contract, not the cost.** ``update_argmin`` takes the *smallest* apex
    # ``k`` at equal cost, and that choice decides the emitted triangles, so a differently-tied
    # reduction is a valid, equal-cost, *different* filling — which every metric/count test in the
    # suite passes. The two-stage reduction below reproduces it exactly and without any float
    # bit-packing: the block minimum of the cost, then the block minimum of ``k`` over just the
    # lanes that attained it. A lane's own ``update_argmin`` already holds the smallest ``k`` at its
    # own minimum, so the pair is (min cost, min k attaining it) — which is what an ascending strict
    # ``<`` scan returns. Lanes with no apex, and an all-non-finite interval, both leave
    # ``(inf, -1)`` and agree with the serial kernel there too.
    ell, i, t = wp.tid()
    # Every guard below is warp-uniform (it reads only ``ell``, ``i`` and ``span``), so the whole
    # block returns together and the tile reductions never run in divergent control flow.
    if active[ell] == 0:
        return
    b = loop_sizes[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = dp_offsets[ell]
    if forbidden[base + i * b + j] != 0:
        if t == 0:
            dp[base + i * b + j] = BAD_METRIC
            prev[base + i * b + j] = -1
        return
    o = loop_starts[ell]
    plane_normal = plane_normals[ell]
    char_area = char_areas[ell]
    a_pos = loop_pos[o + i]
    c_pos = loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = int(-1)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    for k in range(i + 1 + t, j, HOLE_DP_BLOCK):
        val = apex_cost(
            loop_pos,
            rim_opp_pos,
            rim_opp_valid,
            dp,
            prev,
            metric_id,
            combine_id,
            smooth_bd,
            o,
            b,
            base,
            i,
            j,
            k,
            is_top,
            a_pos,
            c_pos,
            plane_normal,
            char_area,
        )
        update_argmin(best_val, best_k, val, k)
    block_val = wp.tile_min(wp.tile(best_val))[0]
    attained = wp.where(best_val == block_val, best_k, INT32_MAX_CONSTANT)
    block_k = wp.tile_min(wp.tile(attained))[0]
    if t == 0:
        dp[base + i * b + j] = block_val
        prev[base + i * b + j] = block_k


@wp.kernel
def flag_bad_triangulations(
    loop_sizes: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    dp: wp.array[wp.float32],
    out_retry: wp.array[wp.int32],
) -> None:
    # Per-loop min-area retry mask: the whole-loop interval is dp[0, B - 1]. Testing it on device
    # replaces copying every loop's ``B x B`` table to the host to read one scalar out of it.
    ell = int(wp.tid())
    top = dp[dp_offsets[ell] + loop_sizes[ell] - 1]
    out_retry[ell] = wp.where(top >= BAD_METRIC, wp.int32(1), wp.int32(0))


@wp.kernel
def edge_third_vertex(faces: wp.array[wp.int32], out_third: wp.array[wp.int32]) -> None:
    # Third vertex per faces_to_edges row: edge k of face f is (v_k, v_{k+1}), third is v_{k+2}.
    r = int(wp.tid())
    f = r // 3
    k = r % 3
    out_third[r] = faces[f * 3 + (k + 2) % 3]


@wp.kernel
def rim_opposite_from_table(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    sorted_keys: wp.array[wp.uint64],
    sorted_rows: wp.array[wp.int32],
    thirds: wp.array[wp.int32],
    max_index: wp.uint64,
    out_positions: wp.array[wp.vec3],
    out_valid: wp.array[wp.int32],
) -> None:
    # Per rim edge (loop[i], loop[i+1]) of every loop at once: probe the sorted packed-edge table;
    # a single hit means exactly one adjacent existing face, whose third vertex blends the fill
    # dihedral metrics into the surface (host dict semantics of the former _rim_opposite).
    t = int(wp.tid())
    ell = loop_id[t]
    o = loop_starts[ell]
    u = flat_loops[t]
    v = flat_loops[o + _wrap(t - o + 1, loop_sizes[ell])]
    lo_v = wp.min(u, v)
    hi_v = wp.max(u, v)
    key = wp.uint64(wp.uint32(lo_v)) + wp.uint64(wp.uint32(hi_v)) * max_index
    lo_idx = kernel_array.binary_search_index_left(sorted_keys, key)
    hi_idx = kernel_array.binary_search_index(sorted_keys, key)
    out_positions[t] = wp.vec3(0.0, 0.0, 0.0)
    out_valid[t] = wp.int32(0)
    if hi_idx - lo_idx == 1:
        out_positions[t] = vertices[thirds[sorted_rows[lo_idx]]]
        out_valid[t] = wp.int32(1)


@wp.kernel
def scatter_loop_positions(
    flat_loops: wp.array[wp.int32], out_flat_slot: wp.array[wp.int32]
) -> None:
    # Where each mesh vertex sits in the packed loop buffer, for the chord test below. The value
    # stored is the *flat* index, which is globally unique and increases with position inside a
    # loop — so the highest position still wins on a duplicated loop vertex (matching the
    # last-write-wins dict the host implementation built), and a pinch vertex shared by two loops
    # resolves to the later loop rather than needing the scratch cleared between loops.
    t = int(wp.tid())
    wp.atomic_max(out_flat_slot, flat_loops[t], wp.int32(t))


@wp.kernel
def mark_forbidden_chords(
    edges_sorted: wp.array2d[wp.int32],
    flat_slot: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    dp_offsets: wp.array[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # Chords (non-adjacent loop positions) that already exist as mesh edges are forbidden
    # (MeshLib MultipleEdgesResolveMode::Simple). Idempotent writes: no atomics needed. One pass
    # over the whole mesh marks the masks of *all* loops, because ``flat_slot`` distinguishes them:
    # an edge whose endpoints land in two different loops is not a chord of either.
    e = int(wp.tid())
    tu = flat_slot[edges_sorted[e, 0]]
    tv = flat_slot[edges_sorted[e, 1]]
    if tu < 0 or tv < 0:
        return
    ell = loop_id[tu]
    if ell != loop_id[tv]:
        return
    b = loop_sizes[ell]
    o = loop_starts[ell]
    lo = wp.min(tu, tv) - o
    hi = wp.max(tu, tv) - o
    if hi - lo >= 2 and hi - lo <= b - 2:
        base = dp_offsets[ell]
        out_mask[base + lo * b + hi] = wp.int32(1)
        out_mask[base + hi * b + lo] = wp.int32(1)
