"""Kernels for closing boundary holes and stitching two rims (``triwarp.holes``)."""

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.array import (
    declare_map_signatures,
    loop_next_slot,
    map_probe,
    map_probe_single,
    pack_nearest_key,
    tile_argmin,
    update_argmin,
)
from triwarp.kernels.array import wrap_index as _wrap
from triwarp.kernels.predicates import (
    circumcircle_diameter,
    dihedral_angle,
    project_out_normal,
    side_lengths,
    triangle_aspect_ratio,
    triangle_double_area,
)
from triwarp.kernels.triangles import corner_triple

# Big-but-finite penalty for a triangulation the metric rejects: lets the DP keep a bad
# triangulation rather than break entirely, while staying below ``float`` precision limits
# when summed.
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


# One convention for a loop's extent, and this is it: ``loop_starts[ell]`` and
# ``loop_sizes[ell]``, both uploaded once by ``holes._PackedLoops``. The kernels below used to
# split -- three of them re-derived the size from ``loop_starts``, a ``total`` and an ``n_loops``
# through a ``loop_size`` helper, while the rest read the size array directly -- which meant one
# file answered the same question two ways and a new kernel could pick a third.


@wp.kernel
def fan_faces(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    ell = wp.int32(wp.tid())
    o = loop_starts[ell]
    s = loop_sizes[ell]
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
    loop_sizes: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    ell = wp.int32(wp.tid())
    o = loop_starts[ell]
    s = loop_sizes[ell]
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
    loop_sizes: wp.array[wp.int32],
    out_centroids: wp.array[wp.vec3],
) -> None:
    ell = wp.int32(wp.tid())
    o = loop_starts[ell]
    s = loop_sizes[ell]
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(s):
        acc = acc + vertices[flat_loops[o + j]]
    out_centroids[ell] = acc * (1.0 / wp.float32(s))


@wp.func
def min_triangle_angle_sin(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    # sin of the smallest angle = shortest edge / circumcircle diameter.
    bc, ca, ab = side_lengths(a, b, c)
    if ab <= 0.0 or ca <= 0.0 or bc <= 0.0:
        return 0.0
    f = triangle_double_area(a, b, c)
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
        # 1e2 is the empirical area weight; char_area == 1 / maxEdgeLengthSq.
        return aspect_ratio + 100.0 * triangle_double_area(a, b, c) * char_area
    if metric_id == METRIC_PLANE:
        if wp.dot(plane_normal, wp.cross(b - a, c - a)) < 0.0:
            return BAD_METRIC
        return circumcircle_diameter(a, b, c)
    # METRIC_PLANE_NORMALIZED (the default): penalize triangles
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
    # triangle-only metrics.
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
    # Characteristic-area scale for the complex-fill metric: 1 / maxEdgeLengthSq, or 1 for a
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
    t = wp.int32(wp.tid())
    ell = loop_id[t]
    a = vertices[flat_loops[t]]
    c = vertices[flat_loops[loop_next_slot(loop_id, loop_starts, loop_sizes, t)]]
    wp.atomic_max(out_max_edge_sq, ell, wp.length_sq(c - a))
    wp.atomic_add(out_normal, ell, wp.cross(a, c))


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
# **Two values, and the choice is an occupancy question rather than a rim-length one.** The grid is
# ``(n_loops, max_size - span)``, so a *wide* block only pays when that grid on its own would
# starve the device -- which is CLAUDE.md section 3's block-per-item rule in a second place, and
# getting it backwards costs 12 %. Measured on an RTX 5090 / Warp 1.17, interleaved, ``min`` of
# 4-5, with the emitted face buffer **identical at every block size in every row**:
#
# | case | ``n_loops`` | ``max_size`` | 32 -> 128 |
# |---|---|---|---|
# | one wavy rim | 1 | 128 | **0.93x** |
# | one wavy rim | 1 | 197 | 1.00 |
# | one wavy rim | 1 | 380 | 1.06 |
# | one wavy rim | 1 | 590 | 1.17 |
# | one wavy rim | 1 | 1024 | **1.45** |
# | ``rim_short`` (the benchmarked shape) | 2 | 512 | **1.13** |
# | capped tubes | 1 | 512 | 1.01 |
# | capped tubes | 8 | 512 | 1.03 |
# | capped tubes | 32 | 512 | **0.93** |
# | capped tubes | 128 | 512 | **0.89** |
# | ``happy_buddha``, a scattered 1/5 region | 99 | 3025 | **0.88** |
# | ``holes_many`` | 512 | 3 | 1.02 |
#
# So the wide block is worth 1.13-1.45x at one or two long rims and **loses 0.88-0.93x from ~32
# rims up**, whatever the rim length -- the 99-rim ``happy_buddha`` row has the *longest* spans here
# and is the worst loss, which is what rules out keying on ``max_size``. Hence both conditions in
# ``hole_dp_block``: a long rim for the lanes to have work, and few enough rims that the grid needs
# them. Within the winning corner the size of the win is mesh-dependent (1.01x on a capped tube
# against 1.13x on ``rim_short`` at the same shape), so treat the table as a floor, not a formula.
#
# Two notes on method, each of which reversed a conclusion here. The previous reading of this knob,
# **"measured flat between 32 and 128"** (19.0/20.1 at 32 against 19.6/23.9 at 128 on
# ``rim_short``), was taken on Warp 1.16 and no longer holds on 1.17 -- re-probe a tuning constant
# after an upgrade rather than trusting the comment. And a first version of this cut keyed on
# ``max_size`` alone; it was caught by an A/B on a *scattered* deleted region, not by the benchmark
# rows, so a many-rim case belongs in any future sweep of it. The old tie-break reason still stands
# on its own terms -- a single-warp block takes the ``warp_count == 1`` fast path in Warp's
# ``tile_reduce_impl``, 126 ns per ``tile_min`` against 325 at 64 and 369 at 128 -- it is simply
# outweighed when the grid is narrow and the apex loop long.
HOLE_DP_BLOCK = 32
HOLE_DP_BLOCK_LONG = 128
# Longest rim at or above which the wide block can pay (measured flat for both at exactly 256)...
HOLE_DP_LONG_RIM = 256
# ...and the rim count above which the grid no longer needs it. Between 8 and 32 in the table above;
# placed at the low end because the losses past it are consistent and the wins below it are not.
HOLE_DP_WIDE_GRID_LOOPS = 4


def hole_dp_block(max_size: int, n_loops: int) -> int:
    """
    Lanes per block for a DP sweep over ``n_loops`` rims whose longest is ``max_size``.

    Wide only when the rim is long enough to give the lanes work *and* there are few enough rims
    that the ``(n_loops, max_size - span)`` grid would otherwise starve the device. See the table
    above for why both conditions are needed.
    """
    if max_size >= HOLE_DP_LONG_RIM and n_loops <= HOLE_DP_WIDE_GRID_LOOPS:
        return HOLE_DP_BLOCK_LONG
    return HOLE_DP_BLOCK


@wp.struct
class HoleFillTables:
    """
    The hole-filling DP's invariant inputs, bundled so the per-span launches carry one argument.

    ``holes._fill_dp`` launches ``fill_dp_span`` once per span -- ``max_B - 2`` times for the whole
    mesh -- and every one of these thirteen values is the same on every launch. A ``wp.launch``
    argument costs ~1.0 us of host time, linearly and on both devices (measured over 2-28
    arguments: 15 us at 2, 41 us at 28), so a 16-argument kernel launched ~510 times on a
    512-edge rim spent milliseconds marshalling constants. Only ``span`` and the two in-place DP
    tables stay as arguments, because those are what a launch is actually about.

    Measured on an RTX 5090, the two spellings of the span loop interleaved in one process over the
    same tables (median/min): **1.56x/1.54x** at 2 loops x 512 (``rim_short``'s shape, 510
    launches), **1.82x/1.85x** at 64 x 32 (``holes_many``'s, 30 launches), 1.85x/1.93x at 2 x 128
    and 1.19x/1.17x at 1 x 1024. The saving works out at 9.5-11.9 us per launch against the 12 us
    the twelve dropped arguments predict.

    A cross-*session* before/after had read this as a 1.36x win on the long rims and an 8% *loss* on
    the short ones; the loss was drift in the surrounding work, which is what interleaving in one
    clock state is for.

    Build it ONCE in the wrapper and reuse it: construction costs ~2.6 us, which would give most
    of the saving back if it were done per launch.
    """

    loop_pos: wp.array[wp.vec3]
    loop_starts: wp.array[wp.int32]
    loop_sizes: wp.array[wp.int32]
    dp_offsets: wp.array[wp.int32]
    active: wp.array[wp.int32]
    plane_normals: wp.array[wp.vec3]
    forbidden: wp.array[wp.int32]
    rim_opp_pos: wp.array[wp.vec3]
    rim_opp_valid: wp.array[wp.int32]
    char_areas: wp.array[wp.float32]
    metric_id: wp.int32
    combine_id: wp.int32
    smooth_bd: wp.int32


@wp.func
def apex_cost(
    tables: HoleFillTables,
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
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
    # face's opposite vertex when ``tables.smooth_bd`` is set.
    k_pos = tables.loop_pos[o + k]
    tri = triangle_fill_metric(a_pos, k_pos, c_pos, plane_normal, char_area, tables.metric_id)
    val = combine_metric(dp[base + i * b + k], dp[base + k * b + j], tables.combine_id)
    val = combine_metric(val, tri, tables.combine_id)

    if k > i + 1:
        if prev[base + i * b + k] >= 0:
            e = fill_edge_term(
                a_pos, k_pos, tables.loop_pos[o + prev[base + i * b + k]], c_pos, tables.metric_id
            )
            val = combine_metric(val, e, tables.combine_id)
    elif tables.smooth_bd != 0 and tables.rim_opp_valid[o + i] != 0:
        e = fill_edge_term(a_pos, k_pos, tables.rim_opp_pos[o + i], c_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)

    if j > k + 1:
        if prev[base + k * b + j] >= 0:
            e = fill_edge_term(
                k_pos, c_pos, tables.loop_pos[o + prev[base + k * b + j]], a_pos, tables.metric_id
            )
            val = combine_metric(val, e, tables.combine_id)
    elif tables.smooth_bd != 0 and tables.rim_opp_valid[o + k] != 0:
        e = fill_edge_term(k_pos, c_pos, tables.rim_opp_pos[o + k], a_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)

    # Closing rim edge (loop[b-1] -> loop[0]) is the base of the whole loop and has no parent, so
    # its boundary term is added here for the top interval only.
    if is_top and tables.smooth_bd != 0 and tables.rim_opp_valid[o + b - 1] != 0:
        e = fill_edge_term(a_pos, c_pos, tables.rim_opp_pos[o + b - 1], k_pos, tables.metric_id)
        val = combine_metric(val, e, tables.combine_id)
    return val


@wp.kernel(enable_backward=False)
def fill_dp_span(
    tables: HoleFillTables, span: wp.int32, dp: wp.array[wp.float32], prev: wp.array[wp.int32]
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
    #
    # The prologue it shares with the tiled kernel is deliberately not extracted. Three of its
    # lines are early ``return``s and a ``@wp.func`` cannot return from its caller, so hoisting it
    # would mean a validity flag plus restructured guards in both kernels -- more code at the call
    # sites than it removes. What is left after the guards is five independent loads that read
    # fine where they are; grouping them would name a bag, not a quantity. The genuinely shared
    # arithmetic already moved: ``apex_cost`` went from 19 parameters to 14 when the tables became
    # ``HoleFillTables``.
    ell, i = wp.tid()
    if tables.active[ell] == 0:
        return
    b = tables.loop_sizes[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = tables.dp_offsets[ell]
    if tables.forbidden[base + i * b + j] != 0:
        # Interior chord would duplicate an existing mesh edge (non-manifold) — leave it unfilled.
        dp[base + i * b + j] = BAD_METRIC
        prev[base + i * b + j] = -1
        return
    o = tables.loop_starts[ell]
    plane_normal = tables.plane_normals[ell]
    char_area = tables.char_areas[ell]
    a_pos = tables.loop_pos[o + i]
    c_pos = tables.loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = wp.int32(-1)
    for k in range(i + 1, j):
        val = apex_cost(
            tables, dp, prev, o, b, base, i, j, k, is_top, a_pos, c_pos, plane_normal, char_area
        )
        update_argmin(best_val, best_k, val, k)
    dp[base + i * b + j] = best_val
    prev[base + i * b + j] = best_k


@wp.kernel(enable_backward=False)
def fill_dp_span_tiled(
    tables: HoleFillTables, span: wp.int32, dp: wp.array[wp.float32], prev: wp.array[wp.int32]
) -> None:
    # One *block* per span-``span`` interval, its lanes striding the apex loop. Same DP, same launch
    # count, ``block_dim`` times the parallelism: the serial kernel above puts at most
    # ``n_loops * (max_B - span)`` threads on the machine, which for a single long boundary is a few
    # hundred out of a few hundred thousand.
    #
    # **The stride is ``wp.block_dim()``, not ``HOLE_DP_BLOCK``, and that is what makes this kernel
    # portable.** On CUDA they agree, because the launch passes whichever of the two constants
    # ``hole_dp_block`` picked -- which is a second reason the stride must be the runtime value and
    # not the constant, since the constant is now only one of the two it could have been launched
    # with. They differ on the CPU device, where ``wp.launch_tiled`` runs exactly one lane
    # per block through Warp 1.17 and ``wp.block_dim()`` reads 1. With the constant, lane 0 was the
    # only lane running and it stepped by 32, so the DP minimized over every 32nd apex and returned
    # a valid-looking, equal-count, *wrong* triangulation -- measured on ``_star_tube``, 42 of 44
    # triangles differed from the serial engine. With the runtime value the single CPU lane strides
    # by 1, covers every apex, and the two tile reductions below degenerate to one-element tiles
    # that return that lane's own answer. Byte-identical to ``fill_dp_span`` on both devices.
    #
    # Measured on an RTX 5090, Warp 1.16, a 400-vertex loop, three alternating pairs: the runtime
    # stride costs nothing (min 10.07 / 10.25 / 10.19 ms against 9.98 / 10.23 / 9.77 for the
    # constant; medians 10.38 against 10.60). The loop body is an ``apex_cost`` call, so there was
    # never much for a compile-time step to unroll.
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
    if tables.active[ell] == 0:
        return
    b = tables.loop_sizes[ell]
    if span >= b or i >= b - span:
        return
    j = i + span
    base = tables.dp_offsets[ell]
    if tables.forbidden[base + i * b + j] != 0:
        if t == 0:
            dp[base + i * b + j] = BAD_METRIC
            prev[base + i * b + j] = -1
        return
    o = tables.loop_starts[ell]
    plane_normal = tables.plane_normals[ell]
    char_area = tables.char_areas[ell]
    a_pos = tables.loop_pos[o + i]
    c_pos = tables.loop_pos[o + j]
    is_top = i == 0 and j == b - 1
    best_val = FLOAT32_INF_CONSTANT
    best_k = wp.int32(-1)
    for k in range(i + 1 + t, j, wp.block_dim()):
        val = apex_cost(
            tables, dp, prev, o, b, base, i, j, k, is_top, a_pos, c_pos, plane_normal, char_area
        )
        update_argmin(best_val, best_k, val, k)
    block_val, block_k = tile_argmin(best_val, best_k)
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
    ell = wp.int32(wp.tid())
    top = dp[dp_offsets[ell] + loop_sizes[ell] - 1]
    out_retry[ell] = wp.where(top >= BAD_METRIC, wp.int32(1), wp.int32(0))


@wp.kernel
def edge_third_vertex(faces: wp.array[wp.int32], out_third: wp.array[wp.int32]) -> None:
    # Third vertex per faces_to_edges row: edge k of face f is (v_k, v_{k+1}), third is v_{k+2}.
    r = wp.int32(wp.tid())
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
    t = wp.int32(wp.tid())
    u = flat_loops[t]
    v = flat_loops[loop_next_slot(loop_id, loop_starts, loop_sizes, t)]
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
    t = wp.int32(wp.tid())
    wp.atomic_max(out_flat_slot, flat_loops[t], t)


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
    # outright, rather than re-routed. Idempotent writes: no atomics needed. One pass
    # over the whole mesh marks the masks of *all* loops, because ``flat_slot`` distinguishes them:
    # an edge whose endpoints land in two different loops is not a chord of either.
    e = wp.int32(wp.tid())
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


# --- stitching two boundary loops -------------------------------------------------------------


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
    i = wp.int32(wp.tid())
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
    i = wp.int32(wp.tid())
    best_col = wp.int32(0)
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
    #
    # One lane walking ``n_a`` elements looks like the antipattern it usually is, and folding it
    # into ``row_argmin`` -- which would reduce each row's winner into a packed
    # ``pack_nearest_key`` atomic and drop this launch entirely -- was measured and **declined**.
    # On an RTX 5090, two facing fan disks, min of 7, this kernel costs 0.023 / 0.042 / 0.108 ms
    # at rims of 100 / 1 000 / 4 000 against 1.54 / 7.24 / 26.6 ms for the whole ``stitch_loops``
    # call: **1.51 % / 0.58 % / 0.41 %**. It is launch-dominated rather than loop-dominated (a
    # bare launch is ~32 us, CLAUDE.md section 13), so the serial walk is not what is being paid
    # for, and the share *falls* with rim size -- the saving would be largest exactly where the
    # call is already cheap. The whole alignment path -- this plus ``row_argmin`` plus
    # ``boundary_perimeters`` -- is 4.5 % at 100 and 2.3 % at 4 000.
    best_row = wp.int32(0)
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
    i = wp.int32(wp.tid())
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
    i = wp.int32(wp.tid())
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
    j = wp.int32(wp.tid())
    # ``edge`` is non-decreasing, so the upper bound over its first ``n_a`` entries is the count of
    # entries at or below ``j`` -- the A-loop vertex this B-loop vertex fans to. A result of ``n_a``
    # wraps to the first.
    apex = roll_loop_a[kernel_array.binary_search_index(edge[:n_a], j) % n_a]
    # The B edge is reversed (``fliplr``) so the bridge winding matches mesh B's faces.
    out_faces[3 * j + 0] = roll_loop_b[_wrap(j + 1, m_b)]
    out_faces[3 * j + 1] = roll_loop_b[j]
    out_faces[3 * j + 2] = apex


# --- Minimum-weight hole triangulation (the Liepa/Klincsek interval DP) -----------------------


# --- Metric-based two-hole stitching (the same DP over a grid rather than an interval) ---------

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
    # Per-band-triangle term of each stitch metric.
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
    # Normals are unit here, unlike the fill dihedral.
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
    # Apex of the band triangle already placed at cell (i, j) -- the vertex advanced to reach it.
    # Only called when came[i, j] is valid.
    if came[i, j] == CAME_A:
        return a_pos[(i - 1) % n_a]
    return b_pos[(j - 1) % n_b]


@wp.kernel
def pair_sq_distances(
    a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3], out_dist: wp.array2d[wp.float32]
) -> None:
    # Squared distance between every rim-A vertex ``i`` and rim-B vertex ``j``; the global argmin
    # (reused ``row_argmin`` + ``global_argmin``) is the aligned start pair the band grows from.
    i, j = wp.tid()
    d = a_pos[i] - b_pos[j]
    out_dist[i, j] = wp.length_sq(d)


@wp.kernel
def set_dp_origin(out_dp: wp.array2d[wp.float32]) -> None:
    # Seed the stitch grid DP: the empty band consuming 0 edges of either loop costs nothing.
    out_dp[0, 0] = 0.0


@wp.struct
class StitchTables:
    """
    The stitch DP's invariant inputs, bundled so the per-diagonal launches carry one argument.

    ``holes._stitch_halves`` launches ``stitch_dp_diag`` once per anti-diagonal --
    ``n_a + n_b`` times, so 400 launches for two 200-vertex rims -- and every one of these ten
    values is the same on every launch. Only ``diag`` and the two in-place DP tables vary. This is
    ``HoleFillTables``' argument exactly, in the same file and on the same shape of loop; see that
    struct's docstring for the measured per-argument cost model and for why the struct must be
    built **once**, outside the loop.

    The measurement for *this* kernel is recorded at its launch site in ``holes._stitch_halves``.

    Not graph capture: recording a graph costs at least what issuing the launches costs, so capture
    pays only on a sequence that is *replayed*, and this loop runs once per call. Measured at 400
    launches, capture-and-replay-once is a 0.84x loss where a bundle is a 1.88x win.
    """

    a_pos: wp.array[wp.vec3]
    b_pos: wp.array[wp.vec3]
    a_opp: wp.array[wp.vec3]
    a_opp_valid: wp.array[wp.int32]
    b_opp: wp.array[wp.vec3]
    b_opp_valid: wp.array[wp.int32]
    up: wp.vec3
    metric_id: wp.int32
    n_a: wp.int32
    n_b: wp.int32


@wp.kernel(enable_backward=False)
def stitch_dp_diag(
    tables: StitchTables,
    diag: wp.int32,
    out_dp: wp.array2d[wp.float32],
    out_came: wp.array2d[wp.int32],
) -> None:
    # One thread per cell (i, j) on anti-diagonal ``diag = i + j``; each reads only the previous
    # diagonal, so launching diag = 1, 2, ... in order are the DP barriers. dp[i, j] = min cost of
    # the band consuming i edges of A and j edges of B from the aligned start (cell (0, 0)).
    a_pos = tables.a_pos
    b_pos = tables.b_pos
    up = tables.up
    metric_id = tables.metric_id
    n_a = tables.n_a
    n_b = tables.n_b
    i_lo = wp.max(0, diag - n_b)
    i = i_lo + wp.int32(wp.tid())
    j = diag - i
    if i > n_a or j < 0 or j > n_b or (i == 0 and j == 0):
        return
    # Never let a full ring come from one loop before touching the other.
    if (i == n_a and j == 0) or (j == n_b and i == 0):
        out_dp[i, j] = BAD_METRIC
        out_came[i, j] = CAME_NONE
        return

    complex_edge = metric_id == METRIC_COMPLEX_STITCH
    best = FLOAT32_INF_CONSTANT
    best_came = CAME_NONE

    # Advance loop A: new triangle (a[i-1], b[j], a[i]).
    if i >= 1 and out_dp[i - 1, j] < BAD_METRIC:
        a_prev = a_pos[(i - 1) % n_a]
        a_cur = a_pos[i % n_a]
        b_cur = b_pos[j % n_b]
        w = out_dp[i - 1, j] + stitch_triangle_metric(a_prev, b_cur, a_cur, up, metric_id)
        if complex_edge:
            if out_came[i - 1, j] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, out_came, n_a, n_b, i - 1, j)
                w = w + stitch_edge_metric(a_prev, b_cur, c_op, a_cur)
            if tables.a_opp_valid[(i - 1) % n_a] != 0:
                w = w + stitch_edge_metric(a_cur, a_prev, tables.a_opp[(i - 1) % n_a], b_cur)
        update_argmin(best, best_came, w, CAME_A)

    # Advance loop B: new triangle (a[i], b[j-1], b[j]).
    if j >= 1 and out_dp[i, j - 1] < BAD_METRIC:
        a_cur = a_pos[i % n_a]
        b_prev = b_pos[(j - 1) % n_b]
        b_cur = b_pos[j % n_b]
        w = out_dp[i, j - 1] + stitch_triangle_metric(a_cur, b_prev, b_cur, up, metric_id)
        if complex_edge:
            if out_came[i, j - 1] != CAME_NONE:
                c_op = stitch_prev_apex(a_pos, b_pos, out_came, n_a, n_b, i, j - 1)
                w = w + stitch_edge_metric(a_cur, b_prev, c_op, b_cur)
            # ``rim_opposite_from_table`` keys slot k on the rim edge (loop[k], loop[k + 1]), so
            # the edge (b[j - 1], b[j]) introduced by this step is slot j - 1, not j -- the same
            # convention the A branch uses above.
            if tables.b_opp_valid[(j - 1) % n_b] != 0:
                w = w + stitch_edge_metric(b_prev, b_cur, tables.b_opp[(j - 1) % n_b], a_cur)
        update_argmin(best, best_came, w, CAME_B)

    out_dp[i, j] = best
    out_came[i, j] = best_came


@wp.kernel
def mark_loops_with_chords(
    unique_edges: wp.array2d[wp.int32],
    loop_of_vertex: wp.array[wp.int32],
    position_in_loop: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    out_has_chord: wp.array[wp.bool],
) -> None:
    # A *chord* is a mesh edge joining two vertices of one boundary loop that are not neighbours
    # along it. A min-weight fill triangulates over the loop's own vertices, so it can propose that
    # chord as a fill edge -- and the mesh already has one, which makes the result non-manifold.
    # That is the hazard ``fill_min_weight(resolve_multiple_edges=True)`` works around after the
    # fact; naming it per loop lets a caller decide before filling.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    loop = loop_of_vertex[a]
    if loop < 0 or loop_of_vertex[b] != loop:
        return
    size = loop_sizes[loop]
    gap = position_in_loop[a] - position_in_loop[b]
    if gap < 0:
        gap = -gap
    if gap != 1 and gap != size - 1:
        out_has_chord[loop] = True


@wp.kernel
def project_loop_to_plane(
    vertices: wp.array[wp.vec3],
    loop_vertices: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origins: wp.array[wp.vec3],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Each rim vertex's orthogonal projection onto its loop's plane. The *ring* of these is what the
    # rim is bridged to, so the extension is exactly the ruled surface between the two. The origin
    # is per loop rather than global because a bottom plane is fitted to each rim separately.
    #
    # The shared predicate takes the offset from the origin and returns it, so the origin is
    # subtracted and added back -- one more subtract than writing the projection out, and a
    # cancellation the direct form does not have. Not bit-identical to it, and deliberately so: the
    # disagreement is **3.0e-08 relative** on the extension's own output and stays there whatever
    # the model's scale or distance from the origin (probed at unit scale, at 1e3 and 1e5 away, and
    # at scale 100), i.e. it sits at float32 epsilon rather than growing. Face buffers are
    # unchanged, so nothing topological turns on it, and the off-plane residual is a wash between
    # the two forms -- neither is the more accurate one.
    i = wp.int32(wp.tid())
    point = vertices[loop_vertices[i]]
    origin = plane_origins[loop_id[i]]
    out_positions[i] = origin + project_out_normal(point - origin, plane_normal)


@wp.kernel
def loop_extreme_projection(
    vertices: wp.array[wp.vec3],
    loop_vertices: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    direction: wp.vec3,
    out_extremes: wp.array[wp.float32],
) -> None:
    # The smallest ``dot(position, direction)`` over each rim, which is where that rim's bottom
    # plane sits. ``out_extremes`` arrives pre-filled with +infinity.
    i = wp.int32(wp.tid())
    wp.atomic_min(out_extremes, loop_id[i], wp.dot(vertices[loop_vertices[i]], direction))


@wp.func
def plane_origin_from_extreme(
    extreme: wp.float32, direction: wp.vec3, extension: wp.float32
) -> wp.vec3:
    # A point on the plane through the rim's extreme vertex, pushed ``extension`` further along
    # ``-direction``. Only its component along ``direction`` matters to the projection.
    return direction * (extreme - extension)


@wp.kernel
def bridge_loop_to_ring(
    loop_vertices: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    ring_base: wp.int32,
    out_faces: wp.array2d[wp.int32],
) -> None:
    # Two triangles per rim edge, joining it to the corresponding edge of the projected ring. The
    # rim runs with the surface on its left, so the quad ``(a, b, b', a')`` is wound the other way
    # round to keep the extension's outward side the same as the mesh's.
    t = wp.int32(wp.tid())
    next_slot = loop_next_slot(loop_id, loop_starts, loop_sizes, t)
    a = loop_vertices[t]
    b = loop_vertices[next_slot]
    projected_a = ring_base + t
    projected_b = ring_base + next_slot

    row = wp.int32(2) * t
    out_faces[row, 0] = a
    out_faces[row, 1] = projected_b
    out_faces[row, 2] = b
    out_faces[row + wp.int32(1), 0] = a
    out_faces[row + wp.int32(1), 1] = projected_a
    out_faces[row + wp.int32(1), 2] = projected_b


@wp.kernel
def directed_edge_opposites(
    faces: wp.array[wp.int32], edges: wp.array2d[wp.int32], out_opposites: wp.array[wp.int32]
) -> None:
    # For each queried directed edge ``(u, v)``, the third corner of the one face that winds
    # ``u -> v``. A boundary edge occurs in exactly one face, so at most one thread writes each
    # slot and the scatter needs no atomic; ``out_opposites`` arrives filled with -1, which is what
    # survives when the edge is not a directed edge of the mesh at all.
    #
    # The corners go into a ``wp.vec3i`` rather than staying the tuple ``corner_triple`` returns,
    # because ``k`` is a *runtime* index and a tuple cannot be subscripted by one in kernel scope.
    # A vector can (verified on Warp 1.17), which is what keeps this off the flat-slice spelling
    # ``faces[f * 3 : (f + 1) * 3]`` that the rest of the tree no longer uses: the rule is not "no
    # slices", it is "no slice where an index form exists", and here one does.
    f, q = wp.tid()
    u = edges[q, 0]
    v = edges[q, 1]
    c0, c1, c2 = corner_triple(faces, f)
    corner = wp.vec3i(c0, c1, c2)
    for k in range(3):
        if corner[k] == u and corner[_wrap(k + 1, 3)] == v:
            out_opposites[q] = corner[_wrap(k + 2, 3)]


@wp.kernel
def reduce_closest_cross_label_pair(
    vertices: wp.array[wp.vec3],
    members: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    max_distance_sq: wp.float32,
    out_best: wp.array[wp.int64],
    out_partner: wp.array[wp.int32],
) -> None:
    # The globally closest pair of ``members`` carrying **different** labels, reduced into one
    # ``int64`` by ``pack_nearest_key`` -- "smallest distance, lowest index on a tie", so the answer
    # is deterministic whatever the thread order. Thread ``i`` finds its own nearest cross-label
    # partner and records it in ``out_partner[i]``, so the winning *pair* is recoverable from the
    # key (whose low half is the query index) plus one lookup.
    #
    # The scan is exhaustive: every thread walks the whole member list. That is ``O(B^2)`` for ``B``
    # members and it is the deliberate choice, because the members here are *boundary* vertices of a
    # mesh with more than one component -- a single-component mesh has no cross-label pair and the
    # caller never launches this -- so ``B`` is split across the components that exist. Accelerating
    # it means a structure per round and a label predicate inside the query; nothing has measured a
    # need for that yet.
    i = wp.int32(wp.tid())
    n = members.shape[0]
    position = vertices[members[i]]
    label = labels[i]
    best_sq = FLOAT32_INF_CONSTANT
    best = wp.int32(-1)
    for j in range(n):
        if labels[j] == label:
            continue
        distance_sq = wp.length_sq(vertices[members[j]] - position)
        if distance_sq < best_sq:
            best_sq = distance_sq
            best = j
    out_partner[i] = best
    if best >= 0 and best_sq <= max_distance_sq:
        wp.atomic_min(out_best, 0, pack_nearest_key(wp.sqrt(best_sq), i))


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
            (char_area_from_max, (dense(wp.float32),), wp.float32),
            (char_area_from_max, (single(wp.float32),), wp.float32),
            (plane_origin_from_extreme, (dense(wp.float32), wp.vec3(), wp.float32(1)), wp.vec3),
            (plane_origin_from_extreme, (single(wp.float32), wp.vec3(), wp.float32(1)), wp.vec3),
        ]
    )


_declare_map_kernels()
