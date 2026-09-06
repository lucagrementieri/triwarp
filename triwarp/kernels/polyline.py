import warp as wp

from triwarp.kernels.array import (
    LOOP_CONDITION,
    LOOP_ROUND,
    binary_search_index,
    cross2,
    declare_map_signatures,
    lowbias32,
    map_probe,
    map_probe_single,
    wrap_index,
)
from triwarp.kernels.points import plane_basis
from triwarp.kernels.predicates import (
    closest_point_on_segment,
    orient2d,
    point_to_segment_distance,
    project_out_normal,
    vector_angle,
)
from triwarp.kernels.reduce import ITEMS_PER_BLOCK_1D, tile_chunk


@wp.func
def segment_displacement(polyline: wp.array[wp.vec3], i: wp.int32) -> wp.vec3:
    """Displacement vector ``polyline[i + 1] - polyline[i]`` of segment ``i``."""
    return polyline[i + 1] - polyline[i]


@wp.func
def project_point_to_plane(p: wp.vec3, origin: wp.vec3, unit_normal: wp.vec3) -> wp.vec3:
    """Orthogonal projection of ``p`` onto the plane through ``origin`` with ``unit_normal``."""
    return p - unit_normal * wp.dot(p - origin, unit_normal)


@wp.func
def line_squared_distance(p: wp.vec3, s: wp.vec3, d: wp.vec3, seg_sq_len: wp.float32) -> wp.float32:
    """
    Squared perpendicular distance from ``p`` to the infinite line ``s -> d``.

    Mirrors ``igl::project_to_line`` with an **unclamped** parameter ``t`` (distance to the
    line, not the segment). ``seg_sq_len`` is the precomputed ``dot(d - s, d - s)``.
    """
    dms = d - s
    smp = s - p
    t = -wp.dot(dms, smp) / seg_sq_len
    return wp.length_sq(p - wp.lerp(s, d, t))


@wp.func
def segment_length(start: wp.vec3, end: wp.vec3) -> wp.float32:
    return wp.length(end - start)


@wp.func
def segment_midpoint_and_length(start: wp.vec3, end: wp.vec3) -> tuple[wp.vec3, wp.float32]:
    return wp.lerp(start, end, 0.5), wp.length(end - start)


@wp.kernel
def accumulate_newell_normal(polyline: wp.array[wp.vec3], out_normal: wp.array[wp.vec3]) -> None:
    # Newell's method sums cross products of consecutive vertices (position vectors),
    # cross(V_i, V_{i + 1}), over the ``n - 1`` pairs of a *closed* polyline whose last vertex
    # duplicates its first -- which folds the wrap-around edge into this same sum, and is what
    # ``polyline_normal`` appends before launching.
    #
    # This kernel and the two below are the lane-strided single-slot reduction: launched with
    # ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``, one block per
    # ``ITEMS_PER_BLOCK_1D`` elements, lanes striding that block's own chunk by
    # ``wp.block_dim()``, one atomic per block. All three were one ``wp.atomic_add`` per thread to
    # a *constant* slot, which serializes the entire reduction on one address. Interleaved A/B in
    # one session on an RTX 5090, min of three alternating pairs, and the values against a float64
    # reference of the same sum:
    #
    #   n                          4 096    65 536   262 144
    #   newell_normal    atomic    0.0187   0.2482   0.9880  ms
    #                    tiled     0.0142   0.0127   0.0138  ms   -> 1.32x / 19.6x / 71.8x
    #   turning_angle    atomic    0.0113   0.0843   0.3319  ms
    #                    tiled     0.0135   0.0141   0.0121  ms   -> 0.84x / 5.97x / 27.4x
    #   loop_frame       atomic    0.0310   0.4390   1.6174  ms
    #                    tiled     0.0163   0.0150   0.0155  ms   -> 1.90x / 29.3x / 104x
    #
    # ``rim_long`` is 1 << 16 = 65 536 vertices per rim and is the ``polyline`` group's asymptotic
    # fixture, so the middle column is the benchmarked point. The 4 096 turning-angle row is a
    # small loss and is the occupancy trade: four blocks on 170 SMs.
    #
    # The reduction is also **three orders of magnitude more accurate**, which is the opposite of
    # what a different summation order usually costs -- at n = 262 144 the tree sits at 2.6e-07 /
    # 3.5e-07 / 4.7e-07 relative against float64 where the serialized atomics sit at 1.4e-03 /
    # 5.3e-06 / 1.4e-03.
    #
    # **No ``prefers_tiled_reduction`` branch.** The lanes partition a chunk the block already
    # owns and take their stride from ``wp.block_dim()``, which section 3 of
    # ``.claude/CLAUDE.md`` says is correct on both devices: on CPU ``wp.block_dim()`` reads 1,
    # lane 0 walks the whole chunk, and the one-element tile holds that chunk's true total.
    # Measured with ``CUDA_VISIBLE_DEVICES=""``: 0.79-1.00x on the clock, and the same accuracy
    # win (4.3e-07 against 2.6e-04 for the Newell normal at n = 65 536).
    #
    # A ``wp.vec3`` accumulator is summed component-wise because ``wp.tile(wp.vec3)`` does not
    # parse (verified on Warp 1.17: ``Error while parsing function``), which is the same reason
    # ``measures.centroid_tiled`` takes three ``wp.tile_sum`` calls.
    #
    # Only an *unconditional* atomic belongs in this shape. A compaction cursor, a
    # "did anything change" flag or a rare-event counter -- ``count_reflex`` two hundred lines
    # down, ``boundary.find_ears``, ``remesh.commit_flips`` -- contends in proportion to its
    # **hits** rather than to the launch, so converting one would add a block reduction to a
    # kernel that atomically adds a handful of times. Leave those alone.
    #
    # The four-statement prologue below (``wp.tid()``, ``tile_chunk``, the guard, the clamp) is
    # repeated in all four converted kernels and the statement-run scan pairs them; it is **not**
    # extractable and the reason is structural rather than a judgement call. ``wp.tid()`` may only
    # be called from a ``@wp.kernel`` (section 1), and the third statement is an early ``return``
    # that a ``@wp.func`` cannot perform for its caller -- so a helper would have to return a
    # validity flag and every call site would regain the guard it was meant to lose. Same verdict
    # and same reason as ``holes.fill_dp_span``'s written decline. What *is* shared is the number
    # the four of them have to agree on, and that is named: ``reduce.ITEMS_PER_BLOCK_1D``, read by
    # ``blocks_1d`` at the launch and by ``tile_chunk`` here.
    chunk, lane = wp.tid()
    n_pairs = polyline.shape[0] - 1
    offset, remaining = tile_chunk(n_pairs, chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    local = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        local += wp.cross(polyline[i], polyline[i + 1])
    sum_x = wp.tile_sum(wp.tile(local[0]))[0]
    sum_y = wp.tile_sum(wp.tile(local[1]))[0]
    sum_z = wp.tile_sum(wp.tile(local[2]))[0]
    if lane == 0:
        wp.atomic_add(out_normal, 0, wp.vec3(sum_x, sum_y, sum_z))


@wp.kernel
def cyclic_segment_angles(polyline: wp.array[wp.vec3], out_angles: wp.array[wp.float32]) -> None:
    # dim == n_points - 1: angle between segment i and the cyclically next segment.
    i = wp.int32(wp.tid())
    n_segments = polyline.shape[0] - 1
    s0 = segment_displacement(polyline, i)
    s1 = segment_displacement(polyline, (i + 1) % n_segments)
    # ``vector_angle`` is scale-free, so the segments go in unnormalized.
    out_angles[i] = vector_angle(s0, s1)


@wp.kernel
def distance_to_segments(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], out_distances: wp.array[wp.float32]
) -> None:
    tid = wp.int32(wp.tid())
    p = points[tid]
    n_segments = polyline.shape[0] - 1
    best = point_to_segment_distance(polyline[0], polyline[1], p)
    for i in range(1, n_segments):
        best = wp.min(best, point_to_segment_distance(polyline[i], polyline[i + 1], p))
    out_distances[tid] = best


@wp.kernel
def distance_to_first_point(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], out_distances: wp.array[wp.float32]
) -> None:
    tid = wp.int32(wp.tid())
    out_distances[tid] = wp.length(points[tid] - polyline[0])


@wp.kernel
def segment_step_counts(
    polyline: wp.array[wp.vec3], step_size: wp.float32, out_steps: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    length = wp.length(segment_displacement(polyline, i))
    out_steps[i] = wp.max(wp.int32(wp.floor(length / step_size)), wp.int32(1))


@wp.func
def segment_parameter(
    offsets: wp.array[wp.int32], steps: wp.array[wp.int32], j: wp.int32
) -> tuple[wp.int32, wp.float32]:
    # Which segment output sample ``j`` belongs to, and its parameter in ``[0, 1)`` along that
    # segment. ``offsets`` is the exclusive scan of ``steps``, so the containing segment is the last
    # offset not past ``j`` -- ``binary_search_index`` returns the first strictly greater, hence the
    # ``- 1``.
    segment = binary_search_index(offsets, j) - 1
    k = j - offsets[segment]
    return segment, wp.float32(k) / wp.float32(steps[segment])


@wp.kernel
def upsample_gather(
    polyline: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    j = wp.int32(wp.tid())
    segment, weight = segment_parameter(offsets, steps, j)
    out_points[j] = wp.lerp(polyline[segment], polyline[segment + 1], weight)


CURVATURE_EPS = wp.constant(wp.float32(1.0e-6))


@wp.func
def plane_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """
    Return the normal of the plane approximately containing segment vectors ``a``, ``b``, ``c``.

    Returns whichever of ``b x (a + c)`` and ``b x (a - c)`` has the larger magnitude, which stays
    well-defined when ``a`` and ``c`` are nearly parallel or anti-parallel.
    """
    n1 = wp.cross(b, a + c)
    n2 = wp.cross(b, a - c)
    if wp.length_sq(n1) >= wp.length_sq(n2):
        return n1
    return n2


@wp.func
def endpoint_normals(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """
    In-plane unit normals at the two ends of segment ``b`` bracketed by neighbours ``a``, ``c``.

    Rotate each segment 90 degrees within the fitted plane (``plane_normal``) and average the edge
    normal with each neighbour's normal. Returns two zero vectors when the segments are (nearly)
    collinear, signalling the caller to fall back to a
    straight chord.
    """
    normal = plane_normal(a, b, c)
    if wp.length_sq(normal) < CURVATURE_EPS:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
    nod = wp.normalize(wp.cross(normal, b))
    no = wp.normalize(nod + wp.normalize(wp.cross(normal, a)))
    nd = wp.normalize(nod + wp.normalize(wp.cross(normal, c)))
    return no, nd


@wp.func
def arc_point(po: wp.vec3, pd: wp.vec3, no: wp.vec3, nd: wp.vec3, t: wp.float32) -> wp.vec3:
    """
    Point at parameter ``t`` in ``[0, 1]`` along the circular arc from ``po`` to ``pd``.

    The arc is the one whose unit end-normals are ``no`` and ``nd``; its midpoint offset from the
    chord is the ``(|chord| / 2) * tan(theta / 4)`` sagitta, generalised here to every ``t`` for
    multi-point subdivision. Degenerate inputs (zero-length chord, collinear neighbours
    signalled by zero normals, straight/near-straight arc, or a cusp) collapse to the straight
    chord ``po + t * (pd - po)``, so ``t == 0`` always returns ``po`` exactly.
    """
    b = pd - po
    chord = wp.length(b)
    linear = wp.lerp(po, pd, t)
    if chord < CURVATURE_EPS:
        return po
    # Zero end-normals are the collinear sentinel from endpoint_normals; unit normals have norm 1.
    if wp.length_sq(no) < 0.5 or wp.length_sq(nd) < 0.5:
        return linear
    theta = vector_angle(no, nd)
    if theta < CURVATURE_EPS:
        return linear
    tangent = wp.normalize(b)
    # wp.sign is -1 below zero and +1 otherwise, matching the guard this replaces.
    bulge = wp.sign(wp.dot(b, nd - no)) * (no + nd)
    m = project_out_normal(bulge, tangent)  # bulge direction, orthogonalised against the chord
    if wp.length_sq(m) < CURVATURE_EPS:
        return linear
    m = wp.normalize(m)
    alpha = 0.5 * theta
    radius = chord / (2.0 * wp.sin(alpha))
    center = 0.5 * (po + pd) - radius * wp.cos(alpha) * m
    phi = (2.0 * t - 1.0) * alpha
    return center + radius * (wp.cos(phi) * m + wp.sin(phi) * tangent)


@wp.kernel
def smooth_upsample_gather(
    polyline: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    closed: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    j = wp.int32(wp.tid())
    segment, t = segment_parameter(offsets, steps, j)
    n = polyline.shape[0]
    po = polyline[segment]
    pd = polyline[segment + 1]
    # Locate the vertices bracketing this segment; interior segments fit a curvature arc, boundary
    # segments of an open polyline (missing a neighbour) stay linear.
    has_neighbours = 0
    prev_index = 0
    next_index = 0
    if closed == 1:
        m = n - 1  # distinct vertices: polyline[n - 1] duplicates polyline[0]
        prev_index = (segment - 1 + m) % m
        next_index = (segment + 2) % m
        has_neighbours = 1
    elif segment >= 1 and segment + 2 <= n - 1:
        prev_index = segment - 1
        next_index = segment + 2
        has_neighbours = 1
    if has_neighbours == 0:
        out_points[j] = wp.lerp(po, pd, t)
        return
    no, nd = endpoint_normals(po - polyline[prev_index], pd - po, polyline[next_index] - pd)
    out_points[j] = arc_point(po, pd, no, nd, t)


@wp.kernel
def greedy_downsample_mask(
    cumulative_lengths: wp.array[wp.float32], step_size: wp.float32, out_keep: wp.array[wp.bool]
) -> None:
    # Single-thread greedy walk (dim == 1): the selection is sequential because each kept point
    # moves the reference the next one is measured from.
    #
    # **Kept for short polylines only**, and it is `polyline_downsample`'s
    # ``_DOWNSAMPLE_DOUBLING_FROM`` that decides. The walk is ~60 ns a point, so it is the cheapest
    # thing available until the point count pays for the ``2 log2(n) + 1`` launches the parallel
    # form below costs; the numbers and the crossover are on that constant.
    n = cumulative_lengths.shape[0]
    out_keep[0] = True
    last = cumulative_lengths[0]
    for i in range(1, n):
        if cumulative_lengths[i] - last >= step_size:
            out_keep[i] = True
            last = cumulative_lengths[i]


@wp.kernel
def greedy_successors(
    cumulative_lengths: wp.array[wp.float32],
    step_size: wp.float32,
    out_successor: wp.array[wp.int32],
) -> None:
    # The greedy walk's step function, for every point at once: ``out_successor[i]`` is the point
    # the walk would keep next *if* it had just kept ``i``, or ``n`` when the polyline ends first.
    # The walk is then the orbit of 0 under this map, which ``spread_reached`` below enumerates in
    # ``log2(n)`` rounds instead of ``n`` steps.
    #
    # A hand-written lower bound rather than ``array.binary_search_index_left`` because the
    # predicate has to be **the serial kernel's, character for character**: ``cum[mid] - base`` and
    # ``cum[mid] - step`` are not the same test in float32, so searching on a shifted key would
    # move the accepted set at the boundary. It is monotone in ``mid`` because ``cum`` is
    # non-decreasing, which is what makes the search valid at all.
    i = wp.int32(wp.tid())
    n = cumulative_lengths.shape[0]
    base = cumulative_lengths[i]
    lo = i + 1
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if cumulative_lengths[mid] - base >= step_size:
            hi = mid
        else:
            lo = mid + 1
    out_successor[i] = lo


@wp.kernel
def square_successors(successor: wp.array[wp.int32], out_successor: wp.array[wp.int32]) -> None:
    # One pointer-doubling round: ``succ^(2k)`` from ``succ^k``. ``n`` is the absorbing state (the
    # walk has run off the end) and stays absorbing.
    #
    # Ping-ponged rather than written in place, and that is load-bearing: in place a thread could
    # read a slot another thread had already doubled, giving ``succ^(a + b)`` for uncontrolled
    # ``a``, ``b`` -- which breaks the round count's guarantee below.
    i = wp.int32(wp.tid())
    n = successor.shape[0]
    j = successor[i]
    out_successor[i] = wp.where(j >= n, n, successor[j])


@wp.kernel
def spread_reached(
    successor: wp.array[wp.int32], reached: wp.array[wp.bool], out_keep: wp.array[wp.bool]
) -> None:
    # One round of doubling the *reached set*: given ``successor`` holding ``succ^(2^k)`` and
    # ``reached`` holding ``{succ^t(0) : t < 2^k}``, mark ``succ^(t + 2^k)(0)`` for each of them, so
    # the set covers ``t < 2^(k + 1)``. ``ceil(log2(n + 1))`` rounds therefore cover the whole
    # orbit, whatever its length -- the walk advances by at least ``step_size`` each time, so the
    # orbit is at most ``n`` long.
    #
    # ``reached`` and ``out_keep`` are **the same buffer**, updated in place, and unlike the
    # doubling above that is safe *and* deliberate. Every write is ``True``, so a lost update is
    # impossible; a thread that happens to see a mark written this round propagates one extra hop,
    # which can only mark another point of the same orbit (``succ`` of an orbit point is one).
    # So intermediate rounds are nondeterministic in *which* extra points they mark and the final
    # answer is not, because the round count alone guarantees completeness.
    i = wp.int32(wp.tid())
    if reached[i]:
        j = successor[i]
        if j < successor.shape[0]:
            out_keep[j] = True


RDP_LINE_EPS = wp.constant(wp.float32(1.0e-7))  # libigl FLOAT_EPS: degenerate-segment threshold
RDP_SETTLED = wp.constant(wp.int32(-1))  # ``span_lo`` sentinel: this point's fate is decided


@wp.func
def rdp_chord_squared_distance(
    polyline: wp.array[wp.vec3], i: wp.int32, lo: wp.int32, hi: wp.int32
) -> wp.float32:
    """Squared distance from ``polyline[i]`` to the chord ``polyline[lo] -> polyline[hi]``."""
    start = polyline[lo]
    end = polyline[hi]
    seg_sq_len = wp.length_sq(end - start)
    if seg_sq_len <= RDP_LINE_EPS:
        return wp.length_sq(polyline[i] - start)  # degenerate chord: distance to the shared point
    return line_squared_distance(polyline[i], start, end, seg_sq_len)


@wp.kernel
def rdp_seed_spans(
    out_span_lo: wp.array[wp.int32], out_span_hi: wp.array[wp.int32], out_keep: wp.array[wp.bool]
) -> None:
    # Ramer-Douglas-Peucker, level-synchronous: one round per level of the recursion tree instead
    # of one thread walking the whole tree. Round 0 puts every interior point in the single span
    # ``(0, n - 1)``; the two endpoints are kept unconditionally and never belong to a span.
    #
    # A span is identified by its **left endpoint**, and that is the whole reason there is no span
    # list to build or compact: the open spans at any level partition the polyline, so their left
    # endpoints are distinct and index a plain ``(n,)`` accumulator directly.
    i = wp.int32(wp.tid())
    n = out_span_lo.shape[0]
    if i == 0 or i == n - 1:
        out_span_lo[i] = RDP_SETTLED
        out_span_hi[i] = RDP_SETTLED
        out_keep[i] = True
    else:
        out_span_lo[i] = 0
        out_span_hi[i] = n - 1
        out_keep[i] = False


@wp.kernel
def rdp_begin_round(
    out_state: wp.array[wp.int32],
    out_span_max: wp.array[wp.float32],
    out_span_argmax: wp.array[wp.int32],
) -> None:
    # Round, pass 1 of 4: arm the per-span accumulators and clear the loop condition. Kept as its
    # own launch rather than folded into the split kernel (which knows each child span's slot)
    # because a ping-ponged pair of accumulators cannot be swapped inside a captured graph -- the
    # buffers are baked in at capture time. One extra ``dim=n`` launch per round buys the readback.
    i = wp.int32(wp.tid())
    if i == 0:
        out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1
        out_state[LOOP_CONDITION] = 0
    out_span_max[i] = -1.0
    out_span_argmax[i] = out_span_max.shape[0]  # past every valid index, so atomic_min always wins


@wp.kernel
def rdp_span_max(
    polyline: wp.array[wp.vec3],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    out_squared_distances: wp.array[wp.float32],
    out_span_max: wp.array[wp.float32],
) -> None:
    # Round, pass 2 of 4: every unsettled point measures itself against its span's chord and
    # max-reduces into the span's slot. One thread per *point* rather than per span, so a round
    # costs the same whatever shape the level has -- which is what makes the depth, and not the
    # span sizes, the cost model.
    i = wp.int32(wp.tid())
    lo = span_lo[i]
    if lo >= 0:
        squared_distance = rdp_chord_squared_distance(polyline, i, lo, span_hi[i])
        out_squared_distances[i] = squared_distance
        wp.atomic_max(out_span_max, lo, squared_distance)


@wp.kernel
def rdp_span_argmax(
    span_lo: wp.array[wp.int32],
    squared_distances: wp.array[wp.float32],
    span_max: wp.array[wp.float32],
    out_span_argmax: wp.array[wp.int32],
) -> None:
    # Round, pass 3 of 4: recover *which* point won. Reducing the index with ``atomic_min`` over
    # every point holding the span's maximum keeps the lowest such index, which is exactly what the
    # strict '>' argmax of the recursive form kept (Eigen maxCoeff, and libigl's tie convention).
    #
    # A packed ``(bits, ~index)`` int64 key would fold this into pass 2 -- ``wp.atomic_max`` on
    # ``wp.int64`` works on both devices -- and is not used, because the float comparison here is
    # against a value this same expression produced, so it is exact without reinterpreting bits.
    i = wp.int32(wp.tid())
    lo = span_lo[i]
    if lo >= 0 and squared_distances[i] >= span_max[lo]:
        wp.atomic_min(out_span_argmax, lo, i)


@wp.kernel
def rdp_split_spans(
    squared_tolerance: wp.float32,
    span_max: wp.array[wp.float32],
    span_argmax: wp.array[wp.int32],
    span_lo: wp.array[wp.int32],
    span_hi: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_keep: wp.array[wp.bool],
) -> None:
    # Round, pass 4 of 4: split or settle. ``span_lo`` / ``span_hi`` are the point's span and are
    # rewritten in place to its child span, so they are neither an input nor the answer; ``state``
    # is the round loop's own condition, raised whenever a point survives into the next level.
    #
    # The keep set is identical to the recursive form's by construction: there, everything starts
    # kept and a span within tolerance drops its interior; here, nothing starts kept and every
    # split point is kept. Both leave exactly the endpoints and the split points, because the
    # terminal spans partition the polyline. The loop terminates because a child span is strictly
    # narrower than its parent and a span two wide holds a single point, which settles either way.
    i = wp.int32(wp.tid())
    lo = span_lo[i]
    if lo < 0:
        return
    hi = span_hi[i]
    split = span_argmax[lo]
    # ``split <= lo or split >= hi`` catches an unresolved argmax -- ``split`` still at the sentinel
    # ``rdp_begin_round`` armed -- and settling the span turns what would be a read past the end of
    # the polyline on the next round into a dropped interior.
    #
    # It is **defensive and measured to be unreachable**, which is worth saying because the obvious
    # reason to expect otherwise is wrong: a non-finite coordinate does *not* produce it, because
    # ``wp.atomic_max`` does not propagate ``NaN`` (measured ``atomic_max(-1, NaN, 25) == 25.0`` on
    # both devices). So either some point wrote a real maximum, and that same point then satisfies
    # the ``>=`` in ``rdp_span_argmax`` and resolves the index; or every interior distance was
    # ``NaN``, the accumulator keeps the ``-1.0`` it was armed with, and the tolerance test above
    # settles the span first. Seven non-finite shapes -- interior, endpoint, all-``NaN``, ``inf`` --
    # give byte-identical answers with the two comparisons deleted.
    if span_max[lo] <= squared_tolerance or split <= lo or split >= hi:
        span_lo[i] = RDP_SETTLED  # the whole span is within tolerance, so its interior drops
        return
    if i == split:
        out_keep[i] = True
        span_lo[i] = RDP_SETTLED
        return
    if i < split:
        span_hi[i] = split  # ``lo < i < split``, so the child span is never degenerate
    else:
        span_lo[i] = split
    # A plain store, not an atomic: one address, one value, nothing to serialize (the rule is
    # on ``array.LOOP_CONDITION``, which also says why the array must be zero-initialized).
    state[LOOP_CONDITION] = 1


@wp.kernel
def broadcast_first_point(polyline: wp.array[wp.vec3], out_points: wp.array[wp.vec3]) -> None:
    j = wp.int32(wp.tid())
    out_points[j] = polyline[0]


@wp.kernel
def resample_interp(
    polyline: wp.array[wp.vec3],
    cumulative_lengths: wp.array[wp.float32],
    num_points: wp.int32,
    out_points: wp.array[wp.vec3],
) -> None:
    # Linear interpolation at evenly spaced arc lengths, mimicking numpy.interp:
    # constant (clamped) extrapolation at the endpoints.
    j = wp.int32(wp.tid())
    n = polyline.shape[0]
    total = cumulative_lengths[n - 1]
    x = wp.float32(0.0)
    if num_points > 1:
        x = wp.float32(j) / wp.float32(num_points - 1) * total
    hi = binary_search_index(cumulative_lengths, x)
    if hi == 0:
        out_points[j] = polyline[0]
    elif hi >= n:
        out_points[j] = polyline[n - 1]
    else:
        denominator = cumulative_lengths[hi] - cumulative_lengths[hi - 1]
        t = wp.float32(0.0)
        if denominator > 0.0:
            t = (x - cumulative_lengths[hi - 1]) / denominator
        out_points[j] = wp.lerp(polyline[hi - 1], polyline[hi], t)


@wp.func
def radius_segment_distances(
    start: wp.vec3, end: wp.vec3, center: wp.vec3, normal: wp.vec3
) -> wp.float32:
    # In-plane distance from ``center`` to one projected segment. Mapped over a shifted pair of
    # views (``polyline[:-1]``, ``polyline[1:]``), so the two endpoints arrive as separate inputs
    # rather than being indexed as ``i`` and ``i + 1``.
    #
    # ``normal`` is normalized here rather than at Python scope to keep the arithmetic identical to
    # the kernel this replaced.
    unit_normal = wp.normalize(normal)
    a = project_point_to_plane(start, center, unit_normal)
    b = project_point_to_plane(end, center, unit_normal)
    return wp.length(closest_point_on_segment(a, b, center) - center)


# --- polygon triangulation (parallel ear clipping); port of libigl ear_clipping.cpp ---


@wp.func
def point_in_triangle(a: wp.vec2, b: wp.vec2, c: wp.vec2, p: wp.vec2) -> wp.bool:
    """
    Whether ``p`` lies inside or on the boundary of the CCW triangle ``(a, b, c)``.

    Boundary inclusion matters for the ear test: a (reflex) vertex lying exactly on a candidate
    ear's cutting diagonal must block that ear, otherwise a degenerate/overlapping triangle is
    emitted.
    """
    return orient2d(a, b, p) >= 0 and orient2d(b, c, p) >= 0 and orient2d(c, a, p) >= 0


@wp.func
def is_ear_at(
    points2d: wp.array[wp.vec2],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    i: wp.int32,
    n: wp.int32,
) -> wp.bool:
    # Corner (a, i, b) is an ear iff it is strictly convex and no other active vertex lies
    # strictly inside triangle (a, i, b). Equivalent to libigl's edge-intersection walk for a
    # simple polygon, but simpler to evaluate in parallel per corner.
    a = left[i]
    b = right[i]
    if a == b or a == i or b == i:
        return False
    pa = points2d[a]
    pi = points2d[i]
    pb = points2d[b]
    if orient2d(pa, pi, pb) <= 0:
        return False
    # Walk the remaining ring from R[b] up to a, skipping the ear's own vertices.
    j = right[b]
    while j != a:
        if active[j] == 1 and j != i and point_in_triangle(pa, pi, pb, points2d[j]):
            return False
        j = right[j]
    return True


@wp.func
def project_to_plane_2d(point: wp.vec3, center: wp.vec3, u: wp.vec3, v: wp.vec3) -> wp.vec2:
    d = point - center
    return wp.vec2(wp.dot(d, u), wp.dot(d, v))


@wp.kernel
def accumulate_loop_frame(
    polyline: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
    out_weighted_midpoint: wp.array[wp.vec3],
    out_length: wp.array[wp.float32],
) -> None:
    # Over an *open* loop of n distinct vertices. One pass replaces the three separate reductions
    # ``polyline_triangulate``'s prologue used to run, each of which ended in a host readback
    # because the next one consumed its Python-scope result.
    #
    # Newell's normal is cyclic -- element i takes the edge (i, (i + 1) % n), so the wrap-around
    # edge is element n - 1 and no closing vertex has to be appended first. The length-weighted
    # centroid deliberately is *not* cyclic: it runs over the n - 1 open segments, which is what
    # ``polyline_centroid`` (``closed=False``) computes and what this function has always used.
    #
    # Lane-strided single-slot reduction -- see ``accumulate_newell_normal`` for the shape and why
    # no device branch is needed. This is the largest of the three wins, at 1.90x / 29.3x / 104x
    # for n = 4 096 / 65 536 / 262 144, because it carried *three* unconditional atomics per
    # element and so three times the contention.
    chunk, lane = wp.tid()
    n = polyline.shape[0]
    offset, remaining = tile_chunk(n, chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    normal = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    weighted = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    length_total = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        start = polyline[i]
        normal += wp.cross(start, polyline[wrap_index(i + 1, n)])
        if i + 1 < n:
            midpoint, length = segment_midpoint_and_length(start, polyline[i + 1])
            weighted += midpoint * length
            length_total += length
    normal_x = wp.tile_sum(wp.tile(normal[0]))[0]
    normal_y = wp.tile_sum(wp.tile(normal[1]))[0]
    normal_z = wp.tile_sum(wp.tile(normal[2]))[0]
    weighted_x = wp.tile_sum(wp.tile(weighted[0]))[0]
    weighted_y = wp.tile_sum(wp.tile(weighted[1]))[0]
    weighted_z = wp.tile_sum(wp.tile(weighted[2]))[0]
    total_length = wp.tile_sum(wp.tile(length_total))[0]
    if lane == 0:
        wp.atomic_add(out_normal, 0, wp.vec3(normal_x, normal_y, normal_z))
        wp.atomic_add(out_weighted_midpoint, 0, wp.vec3(weighted_x, weighted_y, weighted_z))
        wp.atomic_add(out_length, 0, total_length)


@wp.kernel
def finalize_loop_frame(
    normal: wp.array[wp.vec3],
    weighted_midpoint: wp.array[wp.vec3],
    total_length: wp.array[wp.float32],
    out_frame: wp.array[wp.vec3],
) -> None:
    # Single thread: turn the three accumulators into the plane frame, on device. ``out_frame`` is
    # ``[center, u, v]``, which ``project_polyline_to_plane`` reads directly -- so the frame never
    # crosses to the host at all.
    u, v = plane_basis(normal[0])
    out_frame[0] = weighted_midpoint[0] / total_length[0]
    out_frame[1] = u
    out_frame[2] = v


@wp.kernel
def project_polyline_to_plane(
    polyline: wp.array[wp.vec3], frame: wp.array[wp.vec3], out_points2d: wp.array[wp.vec2]
) -> None:
    # dim == n. The device-frame counterpart of mapping ``project_to_plane_2d`` over host-scope
    # ``wp.vec3`` uniforms.
    i = wp.int32(wp.tid())
    out_points2d[i] = project_to_plane_2d(polyline[i], frame[0], frame[1], frame[2])


@wp.kernel
def accumulate_turning_angle(points2d: wp.array[wp.vec2], out_total: wp.array[wp.float32]) -> None:
    # Cyclic signed exterior angle at each vertex; the sum's sign gives the loop orientation.
    # Lane-strided single-slot reduction -- see ``accumulate_newell_normal`` for the shape, its
    # measured table (0.84x / 5.97x / 27.4x at n = 4 096 / 65 536 / 262 144) and why no device
    # branch is needed.
    chunk, lane = wp.tid()
    n = points2d.shape[0]
    offset, remaining = tile_chunk(n, chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    local = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        current = points2d[i]
        nxt = points2d[(i + 1) % n]
        after = points2d[(i + 2) % n]
        d1 = nxt - current
        d2 = after - nxt
        local += wp.atan2(cross2(d1, d2), wp.dot(d1, d2))
    total = wp.tile_sum(wp.tile(local))[0]
    if lane == 0:
        wp.atomic_add(out_total, 0, total)


@wp.kernel
def orient_ccw(points2d: wp.array[wp.vec2], turning_angle: wp.array[wp.float32]) -> None:
    # Mirror the y-axis to flip a clockwise loop to counter-clockwise (replaces libigl's row
    # reversal); the convex/ear tests assume CCW orientation. The sign test reads the accumulated
    # turning angle *on device*, so the caller does not have to synchronize between the two: on the
    # convex fast path that pair of readbacks was the entire cost of the call.
    if turning_angle[0] >= 0.0:
        return
    i = wp.int32(wp.tid())
    p = points2d[i]
    points2d[i] = wp.vec2(p[0], -p[1])


@wp.kernel
def count_reflex(points2d: wp.array[wp.vec2], out_count: wp.array[wp.int32]) -> None:
    # Pre-clip the ring is trivial, so use direct cyclic neighbours. Convex polygon <=> 0 reflex.
    i = wp.int32(wp.tid())
    n = points2d.shape[0]
    prev = points2d[wrap_index(i - 1, n)]
    cur = points2d[i]
    nxt = points2d[(i + 1) % n]
    if orient2d(prev, cur, nxt) < 0:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def fan_triangulate(out_faces: wp.array2d[wp.int32]) -> None:
    # Convex fast-path: fan from vertex 0. dim == n - 2.
    k = wp.int32(wp.tid())
    out_faces[k, 0] = wp.int32(0)
    out_faces[k, 1] = k + 1
    out_faces[k, 2] = k + 2


@wp.kernel
def init_ring(
    left: wp.array[wp.int32], right: wp.array[wp.int32], active: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    n = left.shape[0]
    left[i] = wrap_index(i - 1, n)
    right[i] = (i + 1) % n
    active[i] = wp.int32(1)


@wp.kernel
def compute_ears(
    points2d: wp.array[wp.vec2],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_is_ear: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    n = points2d.shape[0]
    if active[i] == 0:
        out_is_ear[i] = wp.int32(0)
        return
    if is_ear_at(points2d, left, right, active, i, n):
        out_is_ear[i] = wp.int32(1)
    else:
        out_is_ear[i] = wp.int32(0)


@wp.func
def ear_priority(i: wp.int32) -> wp.uint32:
    # An effectively random but perfectly deterministic order on the ring indices -- see
    # ``array.lowbias32`` for why a hash and not the index itself, and for the bijectivity
    # ``ear_outranks`` leans on.
    return lowbias32(wp.uint32(i))


@wp.func
def ear_outranks(a: wp.int32, b: wp.int32) -> wp.bool:
    # Strict total order on ring indices. The hash is injective, so the index tiebreak below never
    # fires; it is there so the order stays total if the mixer is ever changed.
    key_a = ear_priority(a)
    key_b = ear_priority(b)
    if key_a != key_b:
        return key_a < key_b
    return a < b


@wp.kernel
def select_independent(
    is_ear: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    out_selected: wp.array[wp.int32],
) -> None:
    # Select ear i iff it outranks every ear within ring-distance 2. This keeps chosen ears >= 3
    # apart, so their clip footprints {L[i], i, R[i]} are disjoint and can be clipped concurrently.
    # The globally top-ranked ear is always selected, guaranteeing progress.
    #
    # Rank is a *hash* of the ring index rather than the index itself, which is what makes the
    # round count logarithmic. Under the raw index, a ring whose ears alternate (any star polygon)
    # lets the ear at i - 2 suppress the ear at i for every i, so exactly one ear is clipped per
    # round and the clipper runs its full ``n``-round cap. Comparing by an effectively random key
    # instead makes this the textbook maximal-independent-set rule, which retires a constant
    # fraction of the ears per round.
    i = wp.int32(wp.tid())
    out_selected[i] = wp.int32(0)
    if is_ear[i] == 0:
        return
    ll = left[left[i]]
    left_i = left[i]
    r = right[i]
    rr = right[right[i]]
    if is_ear[ll] == 1 and ear_outranks(ll, i):
        return
    if is_ear[left_i] == 1 and ear_outranks(left_i, i):
        return
    if is_ear[r] == 1 and ear_outranks(r, i):
        return
    if is_ear[rr] == 1 and ear_outranks(rr, i):
        return
    out_selected[i] = wp.int32(1)


@wp.kernel
def clip_selected(
    selected: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    active: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    i = wp.int32(wp.tid())
    if selected[i] == 0:
        return
    a = left[i]
    b = right[i]
    slot = wp.atomic_add(out_count, 0, 1)
    out_faces[slot, 0] = a
    out_faces[slot, 1] = i
    out_faces[slot, 2] = b
    active[i] = wp.int32(0)
    right[a] = b
    left[b] = a


@wp.kernel
def ear_loop_continue(
    count: wp.array[wp.int32], target: wp.int32, max_rounds: wp.int32, out_state: wp.array[wp.int32]
) -> None:
    # Ear-clipping loop control, kept on device so ``wp.capture_while`` can drive the rounds without
    # a readback each time; the slot table is ``array.LOOP_ROUND`` / ``LOOP_CONDITION``.
    # The round cap is what stops a degenerate or self-intersecting loop that never retires an ear
    # -- the same bound the host-driven form got from iterating ``range(n)``.
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + 1
    if count[0] < target and out_state[LOOP_ROUND] < max_rounds:
        out_state[LOOP_CONDITION] = wp.int32(1)
    else:
        out_state[LOOP_CONDITION] = wp.int32(0)


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    # Every op here forks on the **length-1** axis alone, and it is the whole table's shape: the
    # three of them are mapped over the shifted pair ``polyline[:-1]`` / ``polyline[1:]``, so a
    # two-point polyline hands them one-element views where every longer one hands them dense
    # arrays. That is section 3's "the fork axis is not only the dtype", and a two-point polyline
    # is an ordinary input rather than a corner: ``polyline_centroid`` and ``polyline_radius``
    # accept one, their guard being ``n_segments < 1``.
    #
    # ``segment_midpoint_and_length`` and ``radius_segment_distances`` were missing from this list
    # and the cost is what section 15 predicts. Measured on the CPU device by walking an 8-point
    # polyline and then a 2-point one through ``polyline_centroid`` / ``polyline_radius`` /
    # ``polyline_length`` with Warp's debug log on: ``Module hash changed, recompiling`` for
    # ``map_segment_midpoint_and_length`` (**1 620 ms**) and ``map_radius_segment_distances``
    # (**196 ms**), while ``map_segment_length`` -- the one that was declared -- held. Three ops on
    # one call path, the declared one steady and the two undeclared ones rebuilding, is the
    # controlled version of that finding.
    dense, single = map_probe, map_probe_single
    center, normal = wp.vec3(), wp.vec3()
    declare_map_signatures(
        [
            (segment_length, (dense(wp.vec3), dense(wp.vec3)), wp.float32),
            (segment_length, (single(wp.vec3), single(wp.vec3)), wp.float32),
            (segment_midpoint_and_length, (dense(wp.vec3), dense(wp.vec3)), (wp.vec3, wp.float32)),
            (
                segment_midpoint_and_length,
                (single(wp.vec3), single(wp.vec3)),
                (wp.vec3, wp.float32),
            ),
            (
                radius_segment_distances,
                (dense(wp.vec3), dense(wp.vec3), center, normal),
                wp.float32,
            ),
            (
                radius_segment_distances,
                (single(wp.vec3), single(wp.vec3), center, normal),
                wp.float32,
            ),
        ]
    )


_declare_map_kernels()
