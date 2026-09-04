"""
Open and closed 3D polyline operations.

**The ``polyline_`` prefix.** Where a name carries the module's own token, the token comes
**first** -- ``polyline_length``, ``polyline_open``, ``polyline_resample``,
``polyline_triangulate`` -- so the whole family sorts and completes together. The measures used to
take the prefix while the operations took it as a suffix, which put one convention in two places
for no reason.

A function that does *not* need the token does not gain one:
[`is_closed`][triwarp.polyline.is_closed] and
[`cumulative_arc_length`][triwarp.polyline.cumulative_arc_length] have an unambiguous subject
already, and [`triangulate_polygon`][triwarp.polyline.triangulate_polygon] takes a 2D polygon
rather than a polyline, so the token there would be wrong rather than redundant.
[`polyline_point_distance`][triwarp.polyline.polyline_point_distance] keeps both nouns because the
prefix alone would lose "from what"; it is spelled to match
[`points.point_plane_distance`][triwarp.points.point_plane_distance].

**The ``closed=`` convention.** Ten of these functions -- length, centroid, radius, angles,
point-to-curve distance, and the five resampling operations -- take a keyword-only
``closed: bool = False``. Passing ``closed=True`` appends the closing edge back to the first point
if it is absent ([`polyline_close`][triwarp.polyline.polyline_close]) and then does the open
computation on that, so the seam is treated like any other segment. Three of the ten additionally
drop the duplicated seam point on the way out, so their result is a clean cyclic ring with one entry
per *original* point rather than one extra:
[`polyline_resample`][triwarp.polyline.polyline_resample],
[`polyline_angles`][triwarp.polyline.polyline_angles] and
[`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample].

The keyword is not always needed. [`polyline_angles`][triwarp.polyline.polyline_angles] and
[`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample] already detect an
*explicitly* closed input -- one whose last point equals its first -- with
[`is_closed`][triwarp.polyline.is_closed]; ``closed=True`` is for the common case of a loop stored
without that duplicate, which is the form
[`boundary_loops`][triwarp.boundary.boundary_loops] returns.
"""

from __future__ import annotations

import math
from typing import Literal

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import polyline as kernel_polyline
from triwarp.kernels import reduce as kernel_reduce

# Point count from which [`polyline_downsample`][triwarp.polyline.polyline_downsample] stops
# walking its greedy selection serially and pointer-doubles it instead -- **on the CUDA device
# only**. The doubling costs ``2 ceil(log2(n + 1)) + 1`` launches, which pays off once the polyline
# is long enough that a serial walk's linear cost exceeds it; the two masks are verified
# **byte-identical** at every size.
#
# On the CPU device the doubling loses at every size, because it does ``n log n`` work where the
# serial form does ``n``, on a backend that runs a launch grid as one serial loop -- there is no GPU
# win being paid for, so the branch takes the device too.
_DOWNSAMPLE_DOUBLING_FROM = 8192


def is_closed(polyline: wp.array[wp.vec3]) -> bool:
    """
    Whether a polyline is closed (its first and last points coincide).

    The endpoint comparison runs on-device via [`allclose`][triwarp.array.allclose], so no array
    is copied to the host.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    bool
        ``True`` when the polyline has at least two points and its first and last points are
        equal within the default ``allclose`` tolerance; ``False`` otherwise.

    See Also
    --------
    [`polyline_open`][triwarp.polyline.polyline_open]
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    n = int(polyline.shape[0])
    return n >= 2 and tw.array.allclose(polyline[0:1], polyline[n - 1 : n])


def polyline_open(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Open a polyline by dropping the last point when it duplicates the first.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.vec3]
        The input unchanged when it has fewer than two points or is already open;
        otherwise a length ``n - 1`` view without the duplicated closing point.

    See Also
    --------
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    n = int(polyline.shape[0])
    if not is_closed(polyline):
        return polyline
    return polyline[0 : n - 1]


def polyline_close(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Close a polyline by appending the first point when it is not already the last.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.vec3]
        The input unchanged when it has fewer than two points or is already closed;
        otherwise a length ``n + 1`` array with the first point appended.

    See Also
    --------
    [`polyline_open`][triwarp.polyline.polyline_open]
    """
    n = int(polyline.shape[0])
    if n < 2 or is_closed(polyline):
        return polyline
    return tw.array.concatenate([polyline, polyline[0:1]])


def polyline_length(polyline: wp.array[wp.vec3], *, closed: bool = False) -> float:
    """
    Total arc length of a polyline (sum of segment lengths).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge counts toward the total.

    Returns
    -------
    float
        The summed segment length, ``0.0`` for fewer than two points.

    See Also
    --------
    [`cumulative_arc_length`][triwarp.polyline.cumulative_arc_length]
        The same lengths, unreduced.
    """
    if closed:
        polyline = polyline_close(polyline)
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments < 1:
        return 0.0
    lengths = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(kernel_polyline.segment_length, polyline[:-1], polyline[1:], out=lengths)
    return float(tw.reduce.sum(lengths))


def polyline_centroid(polyline: wp.array[wp.vec3], *, closed: bool = False) -> wp.vec3:
    """
    Segment-length-weighted centroid of a polyline.

    Each segment contributes its midpoint weighted by its length, so the result is invariant to
    how densely the polyline is sampled (unlike the plain mean of the vertices).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        segment contributes its midpoint and length like any other.

    Returns
    -------
    wp.vec3
        The weighted centroid on the host.

    Raises
    ------
    ValueError
        If the polyline has fewer than two points.

    See Also
    --------
    [`polyline_normal`][triwarp.polyline.polyline_normal]
    [`polyline_radius`][triwarp.polyline.polyline_radius]
        Both default their plane to this centroid.
    """
    if closed:
        polyline = polyline_close(polyline)
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments < 1:
        raise ValueError("polyline_centroid requires at least two points")
    midpoints = wp.empty(n_segments, dtype=wp.vec3, device=device)
    lengths = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(
        kernel_polyline.segment_midpoint_and_length,
        polyline[:-1],
        polyline[1:],
        out=[midpoints, lengths],
    )
    weighted = tw.reduce.weighted_sum(midpoints, lengths)
    return weighted / float(tw.reduce.sum(lengths))


def polyline_normal(polyline: wp.array[wp.vec3]) -> wp.vec3:
    """
    Average unit normal of a 3D polyline via Newell's method.

    The polyline is treated as a closed loop (its closing edge is added if absent), then the
    cross products of consecutive vertices, ``cross(V_i, V_{i + 1})``, are summed and normalized.
    This is Newell's method: the summed cross products of consecutive position vectors around a
    closed loop equal the area-weighted, translation-invariant plane normal. The open-chain
    variant (omitting the closing edge) is origin-dependent and not computed.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    wp.vec3
        The unit normal on the host.

    Raises
    ------
    ValueError
        If the polyline has fewer than three points.
    """
    polyline = polyline_close(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    # The closed polyline's last vertex duplicates its first, so it has n - 1 distinct vertices;
    # a non-degenerate loop normal needs at least three of them.
    if n < 4:
        raise ValueError("polyline_normal requires at least three points")
    out_normal = wp.zeros(1, dtype=wp.vec3, device=device)
    # The polyline is closed (last vertex duplicates the first), so summing cross(V_i, V_{i + 1})
    # over the n - 1 consecutive pairs includes the wrap-around edge — full Newell's method.
    wp.launch_tiled(
        kernel_polyline.accumulate_newell_normal,
        dim=kernel_reduce.blocks_1d(n - 1),
        inputs=[polyline, out_normal],
        block_dim=TILE_1D,
        device=device,
    )
    wp.map(wp.normalize, out_normal, out=out_normal)
    return out_normal.list()[0]


def polyline_point_distance(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], *, closed: bool = False
) -> wp.array[wp.float32]:
    """
    Minimum distance from each query point to the nearest segment of a polyline.

    Parameters
    ----------
    points
        ``(n,)`` query points as ``wp.vec3``.
    polyline
        ``(m,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge is a candidate segment like any other.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` minimum distances on ``points.device``. Every entry is ``inf`` when
        ``polyline`` is empty, there being no segment to measure against -- the same
        ``inf``-on-miss convention
        [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] uses.
    """
    if closed:
        polyline = polyline_close(polyline)
    device = points.device
    n_points = int(points.shape[0])
    m = int(polyline.shape[0])
    if m == 0:
        return wp.full(n_points, float("inf"), dtype=wp.float32, device=device)
    out_distances = wp.empty(n_points, dtype=wp.float32, device=device)
    if n_points == 0:
        return out_distances
    if m == 1:
        wp.launch(
            kernel_polyline.distance_to_first_point,
            dim=n_points,
            inputs=[points, polyline, out_distances],
            device=device,
        )
        return out_distances
    wp.launch(
        kernel_polyline.distance_to_segments,
        dim=n_points,
        inputs=[points, polyline, out_distances],
        device=device,
    )
    return out_distances


def polyline_upsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size.

    Each segment is split into ``max(floor(length / step_size), 1)`` equal pieces. The final
    endpoint of the polyline is not emitted (matching the source implementation).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), and the
        duplicated closing point is not emitted, so the result is a cyclic ring.

    Returns
    -------
    wp.array[wp.vec3]
        The upsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample]
        The same subdivision with the new points placed on a fitted arc instead of the chord.
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
    [`polyline_resample`][triwarp.polyline.polyline_resample]
    """
    if closed:
        polyline = polyline_close(polyline)
    return _upsample(polyline, step_size, kernel_polyline.upsample_gather, [])


def polyline_smooth_upsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size, following local curvature.

    Like [`polyline_upsample`][triwarp.polyline.polyline_upsample], each segment is split into
    ``max(floor(length / step_size), 1)`` pieces and the final endpoint is not emitted. Unlike it,
    the inserted points are placed on a circular arc fitted to the segment's endpoint tangents
    (estimated from the two bracketing neighbour vertices) rather than on the straight chord, so a
    coarsely sampled curve is refined smoothly. Curvature-aware placement is usually stated for the
    edge midpoint alone; here it is generalised to every interpolation parameter.

    The first and last segments of an open polyline have no bracketing neighbour and are subdivided
    linearly; collinear neighbours likewise reduce to the straight chord. Original vertices are
    preserved exactly, since each segment's first sample coincides with its start vertex.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Target spacing between consecutive output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). Every segment
        including the seam is then treated as interior, so neighbour tangents wrap cyclically and
        the whole loop is smoothed rather than only its middle.

    Returns
    -------
    wp.array[wp.vec3]
        The curvature-aware upsampled polyline. The input is returned unchanged for fewer than
        two points.

    See Also
    --------
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
        The straight-chord version, which is what this reduces to on the end segments.
    """
    if closed:
        # Every segment including the seam becomes interior, so neighbour tangents wrap cyclically
        # and the duplicated closing point is dropped -- a clean cyclic ring.
        polyline = polyline_close(polyline)
    gather = kernel_polyline.smooth_upsample_gather
    return _upsample(polyline, step_size, gather, [wp.int32(closed)])


def _upsample(
    polyline: wp.array[wp.vec3],
    step_size: float,
    gather_kernel: wp.Kernel,
    extra_inputs: list[wp.int32],
) -> wp.array[wp.vec3]:
    """
    Split every segment into ``max(floor(length / step_size), 1)`` pieces and gather the samples.

    The shared body of [`polyline_upsample`][triwarp.polyline.polyline_upsample] and
    [`polyline_smooth_upsample`][triwarp.polyline.polyline_smooth_upsample], which differ only in
    the gather kernel that places each sample -- on the chord or on a fitted arc -- and in the
    extra arguments that kernel takes. Both have already applied their own ``closed`` handling.
    """
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments < 1:
        return polyline

    steps = wp.empty(n_segments, dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.segment_step_counts,
        dim=n_segments,
        inputs=[polyline, wp.float32(step_size), steps],
        device=device,
    )
    offsets, total = tw.array.counts_to_offsets(steps)

    out_points = wp.empty(total, dtype=wp.vec3, device=device)
    wp.launch(
        gather_kernel,
        dim=total,
        inputs=[polyline, offsets, steps, *extra_inputs, out_points],
        device=device,
    )
    return out_points


def cumulative_arc_length(polyline: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Cumulative arc length from the first vertex to each vertex of an open polyline.

    The underlying arc-length parametrization behind
    [`polyline_downsample`][triwarp.polyline.polyline_downsample] and
    [`polyline_resample`][triwarp.polyline.polyline_resample]; exposed directly for callers
    doing custom resampling along the polyline.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` array on ``polyline.device``. Entry ``0`` is ``0.0``; entry ``i`` is the
        summed length of segments ``0..i-1``.

    See Also
    --------
    [`polyline_length`][triwarp.polyline.polyline_length]
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
    [`polyline_resample`][triwarp.polyline.polyline_resample]
    """
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments <= 0:
        # No segment to sum: a single vertex is at arc length 0, an empty polyline has no entry.
        # The guard is required, not defensive -- ``polyline[:-1]`` below would be a zero-length
        # slice, which Warp rejects outright.
        return wp.zeros(max(int(polyline.shape[0]), 0), dtype=wp.float32, device=device)

    lengths = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(kernel_polyline.segment_length, polyline[:-1], polyline[1:], out=lengths)
    # The leading zero of the n + 1 buffer is the first cumulative length, and the inclusive scan
    # fills the rest -- the same one-allocation idiom as ``array.counts_to_offsets``, in float32.
    # Scanning into a separate buffer and concatenating a zero in front costs three allocations
    # and two more copy launches for the identical answer.
    cumulative = wp.zeros(n_segments + 1, dtype=wp.float32, device=device)
    wp.utils.array_scan(lengths, out_array=cumulative[1:], inclusive=True)
    return cumulative


def polyline_downsample(
    polyline: wp.array[wp.vec3], step_size: float, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Downsample a polyline to a minimum arc-length spacing between kept points.

    Greedily keeps the first point, then each subsequent point at least ``step_size`` of arc
    length beyond the previously kept point.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Minimum arc-length distance between kept points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        edge counts toward the spacing.

    Returns
    -------
    wp.array[wp.vec3]
        The downsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`polyline_simplify`][triwarp.polyline.polyline_simplify]
        Drops points by *shape* error rather than by spacing.
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
    """
    if closed:
        polyline = polyline_close(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 2:
        return polyline

    cumulative = cumulative_arc_length(polyline)
    keep_mask = wp.zeros(n, dtype=wp.bool, device=device)
    if wp.get_device(device).is_cuda and n >= _DOWNSAMPLE_DOUBLING_FROM:
        _greedy_downsample_doubling(cumulative, step_size, keep_mask)
    else:
        wp.launch(
            kernel_polyline.greedy_downsample_mask,
            dim=1,
            inputs=[cumulative, wp.float32(step_size), keep_mask],
            device=device,
        )
    return tw.array.gather(polyline, tw.array.flatnonzero(keep_mask))


def _greedy_downsample_doubling(
    cumulative: wp.array[wp.float32], step_size: float, out_keep: wp.array[wp.bool]
) -> None:
    """
    Mark the greedy walk's kept points by pointer-doubling its step function.

    The kept set is the orbit of point 0 under "the next point at least ``step_size`` further
    along", so building that step function for every point at once
    ([`greedy_successors`][triwarp.kernels.polyline.greedy_successors]) turns an ``n``-step walk
    into ``ceil(log2(n + 1))`` rounds of squaring it. The answer is the serial walk's, exactly and
    not approximately: the successor search evaluates the same float32 comparison the walk does, so
    the two masks agree bit for bit -- verified over 27 shapes including exact ties and heavily
    clustered spacing.
    """
    device = cumulative.device
    n = int(cumulative.shape[0])
    successor = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.greedy_successors,
        dim=n,
        inputs=[cumulative, wp.float32(step_size), successor],
        device=device,
    )
    out_keep[:1].fill_(True)  # the walk always keeps the first point; the caller zeroed the rest
    squared = wp.empty(n, dtype=wp.int32, device=device)
    for _ in range(max(1, math.ceil(math.log2(n + 1)))):
        wp.launch(
            kernel_polyline.spread_reached,
            dim=n,
            inputs=[successor, out_keep, out_keep],
            device=device,
        )
        wp.launch(
            kernel_polyline.square_successors, dim=n, inputs=[successor, squared], device=device
        )
        successor, squared = squared, successor


def polyline_simplify(
    polyline: wp.array[wp.vec3], tol: float, *, closed: bool = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    Drops interior vertices whose perpendicular distance to the chord spanning a kept sub-range is
    at most ``tol``; the first and last vertices are always retained. The recursion is evaluated
    **level-synchronously** rather than depth-first -- one round of four ``dim=n`` launches per
    level of the split tree, so the cost is the tree's *depth* (about ``log2(n)`` on a mesh
    boundary loop) rather than one thread's walk of the whole tree. The accepted set is identical
    either way, since breadth-first and depth-first evaluation of the same recursion accept the
    same points.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    tol
        Maximum Euclidean distance allowed between a dropped vertex and the retained chord.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). ``indices``
        then refer to the *closed* polyline, so the shared start/end vertex is preserved at both
        ends.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(simplified, indices)`` on ``polyline.device``: the ``(m,)`` retained vertices and the
        ``(m,)`` sorted indices into the input such that ``polyline[indices] == simplified``. An
        empty input yields two empty arrays; a single point is returned unchanged with
        ``indices == [0]``.

    Notes
    -----
    The round loop runs **on device**, driven by ``wp.capture_while`` exactly as
    [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]'s ear rounds are, so the whole
    simplification costs one graph launch and no readback at all. The accepted index set is
    identical to a single-thread recursive evaluation of the same recursion, since breadth-first and
    depth-first evaluation accept the same points.

    **The depth is bounded by the accepted count, not by ``n``, and that is why there is no round
    cap and no serial fallback here.** Every root-to-leaf path of the split tree accepts one point
    per level, so ``rounds <= kept + 1`` -- a deep tree is precisely an input that accepts most of
    its points, which is also the input the recursion does the most work on. A spiral's depth tracks
    its *turn count* rather than ``n``, while a power curve, a geometric staircase and a decaying
    sawtooth are all shallower than a boundary loop.

    This is deliberately **one algorithm on both devices, with no serial fallback and no
    host-driven loop for small inputs**: the fixed cost of graph capture, the keep-mask compaction
    and the round loop itself all scale with the tree depth rather than with ``n``, so a size-gated
    fallback would add a second implementation (and the test that its accepted set agrees with this
    one's) without a reliable win. On the CPU backend a ``dim=n`` launch runs as a single lane, so
    each round costs ``O(n)`` sequential work rather than the host's ``O(n log n)`` recursive total
    -- the device is the target, so this is recorded rather than branched on.

    See Also
    --------
    [`polyline_downsample`][triwarp.polyline.polyline_downsample]
        Drops points by *spacing* rather than by shape error.
    """
    if closed:
        polyline = polyline_close(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    span_lo = wp.empty(n, dtype=wp.int32, device=device)
    span_hi = wp.empty(n, dtype=wp.int32, device=device)
    keep_mask = wp.empty(n, dtype=wp.bool, device=device)
    wp.launch(
        kernel_polyline.rdp_seed_spans, dim=n, inputs=[span_lo, span_hi, keep_mask], device=device
    )
    if n > 2:  # fewer than three points have no interior to drop, and no thread 0 to clear `state`
        span_max = wp.empty(n, dtype=wp.float32, device=device)
        span_argmax = wp.empty(n, dtype=wp.int32, device=device)
        squared_distances = wp.empty(n, dtype=wp.float32, device=device)
        # state = [levels run, loop condition], both written on device so the round loop needs no
        # readback -- ``polyline_triangulate``'s ear rounds are driven the same way.
        state = wp.array([0, 1], dtype=wp.int32, device=device)  # see array.LOOP_ROUND
        squared_tolerance = wp.float32(tol * tol)

        def split_round() -> None:
            wp.launch(
                kernel_polyline.rdp_begin_round,
                dim=n,
                inputs=[state, span_max, span_argmax],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_span_max,
                dim=n,
                inputs=[polyline, span_lo, span_hi, squared_distances, span_max],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_span_argmax,
                dim=n,
                inputs=[span_lo, squared_distances, span_max, span_argmax],
                device=device,
            )
            wp.launch(
                kernel_polyline.rdp_split_spans,
                dim=n,
                inputs=[
                    squared_tolerance,
                    span_max,
                    span_argmax,
                    span_lo,
                    span_hi,
                    state,
                    keep_mask,
                ],
                device=device,
            )

        condition = state[kernel_array.LOOP_CONDITION : kernel_array.LOOP_CONDITION + 1]
        # Graph capture needs a CUDA stream, so the CPU device takes the direct-execution branch
        # even when the driver supports conditional nodes.
        if wp.get_device(device).is_cuda and wp.is_conditional_graph_supported():
            with wp.ScopedCapture(device) as capture:
                wp.capture_while(condition, split_round)
            wp.capture_launch(capture.graph)
        else:
            wp.capture_while(condition, split_round)

    indices = tw.array.flatnonzero(keep_mask)
    return tw.array.gather(polyline, indices), indices


def polyline_resample(
    polyline: wp.array[wp.vec3], num_points: int, *, closed: bool = False
) -> wp.array[wp.vec3]:
    """
    Resample a polyline to a fixed number of points evenly spaced by arc length.

    Points are sampled at ``num_points`` arc lengths evenly spanning ``[0, total_length]`` and
    linearly interpolated between the bracketing vertices (matching ``numpy.interp`` semantics).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    num_points
        Number of output points.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). The result has
        ``num_points`` *distinct* points: the duplicated seam is dropped.

    Returns
    -------
    wp.array[wp.vec3]
        Length ``num_points`` resampled polyline. An empty input is returned unchanged; a single
        point is repeated ``num_points`` times.

    See Also
    --------
    [`polyline_upsample`][triwarp.polyline.polyline_upsample]
        Targets a step size instead of a point count.
    """
    if closed:
        # ``num_points + 1`` samples of the closed polyline, minus the duplicated seam point, so the
        # result has ``num_points`` *distinct* points and is a clean cyclic ring.
        return polyline_resample(polyline_close(polyline), num_points + 1)[0:num_points]
    device = polyline.device
    n = int(polyline.shape[0])
    if n == 0:
        return polyline
    out_points = wp.empty(num_points, dtype=wp.vec3, device=device)
    if n == 1:
        wp.launch(
            kernel_polyline.broadcast_first_point,
            dim=num_points,
            inputs=[polyline, out_points],
            device=device,
        )
        return out_points

    cumulative = cumulative_arc_length(polyline)
    wp.launch(
        kernel_polyline.resample_interp,
        dim=num_points,
        inputs=[polyline, cumulative, wp.int32(num_points), out_points],
        device=device,
    )
    return out_points


def polyline_radius(
    polyline: wp.array[wp.vec3],
    reduction: Literal["min", "max", "mean", "median"] = "min",
    center: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
    *,
    closed: bool = False,
) -> float:
    """
    Radius of a polyline projected onto the plane through ``center`` with the given ``normal``.

    Each segment is projected onto the plane and its closest point to ``center`` is found; the
    per-segment distances to ``center`` are then reduced. With ``reduction="min"`` this is the
    inner radius (nearest point, not vertex, on the polyline).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    reduction
        Reduction over the per-segment radial distances: ``"min"``, ``"max"``, ``"mean"``, or
        ``"median"``. Defaults to ``"min"``.
    center
        Plane origin. Defaults to [`polyline_centroid`][triwarp.polyline.polyline_centroid].
    normal
        Plane normal (need not be unit). Defaults to
        [`polyline_normal`][triwarp.polyline.polyline_normal].

        These two are the circle's frame, not a cutting plane, so they deliberately keep their own
        names and their own order rather than the ``(plane_normal, plane_origin)`` convention every
        plane argument in the package follows -- both are keyword-defaulted here, and calling them
        a plane would misdescribe the geometry.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]), so the closing
        segment contributes a radial distance like any other, and the default ``center`` /
        ``normal`` are the closed polyline's.

    Returns
    -------
    float
        The reduced radius.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of the supported values, or the polyline has fewer than
        two points.

    See Also
    --------
    [`polyline_centroid`][triwarp.polyline.polyline_centroid]
    [`polyline_normal`][triwarp.polyline.polyline_normal]
        The two defaults for the plane.
    [`median`][triwarp.reduce.median]
    """
    if closed:
        polyline = polyline_close(polyline)
    if reduction not in ("min", "max", "mean", "median"):
        raise ValueError(f"unsupported reduction {reduction!r}")
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments < 1:
        raise ValueError("polyline_radius requires at least two points")

    if center is None:
        center = polyline_centroid(polyline)
    if normal is None:
        normal = polyline_normal(polyline)

    distances = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(
        kernel_polyline.radius_segment_distances,
        polyline[:-1],
        polyline[1:],
        center,
        normal,
        out=distances,
    )
    if reduction == "min":
        return float(tw.reduce.min(distances))
    if reduction == "max":
        return float(tw.reduce.max(distances))
    if reduction == "mean":
        return float(tw.reduce.mean(distances))
    return tw.reduce.median(distances)


def polyline_angles(polyline: wp.array[wp.vec3], *, closed: bool = False) -> wp.array[wp.float32]:
    """
    Angles between consecutive segments at each vertex of a polyline.

    Returns one angle per point. For an open polyline the two endpoints get an angle of ``0``;
    for a closed polyline (first point equal to last) the turning angles wrap around.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    closed
        When ``True``, treat the polyline as a loop: the closing edge back to the first point is
        added if absent (see [`polyline_close`][triwarp.polyline.polyline_close]). The result
        still has one angle per *original* point, and the turning angles wrap around.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` angles in radians on ``polyline.device``. All zeros for fewer than two points.

    See Also
    --------
    [`is_closed`][triwarp.polyline.is_closed]
        What decides the wrap-around when ``closed`` is left ``False``.
    """
    if closed:
        # One angle per *original* point: closing appends a duplicate of the first, whose angle is
        # the first's, so the tail is dropped rather than returned twice.
        n_original = int(polyline.shape[0])
        return polyline_angles(polyline_close(polyline))[0:n_original]
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 2:
        return wp.zeros(n, dtype=wp.float32, device=device)

    n_segments = n - 1
    raw = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.launch(
        kernel_polyline.cyclic_segment_angles, dim=n_segments, inputs=[polyline, raw], device=device
    )
    if is_closed(polyline):
        return tw.array.concatenate([raw, raw[0:1]])
    zero = wp.zeros(1, dtype=wp.float32, device=device)
    return tw.array.concatenate([zero, raw[0 : n_segments - 1], zero])


def polyline_triangulate(polyline: wp.array[wp.vec3]) -> twt.Array2dInt32:
    """
    Triangulate the simple planar polygon bounded by a closed 3D polyline (ear clipping).

    The polyline is treated as the boundary of a simple polygon, which is filled with triangles
    whose vertices are the polyline vertices themselves — no new (Steiner) points are introduced.
    A simple ``n``-gon yields ``n - 2`` non-overlapping triangles that cover the polygon. The loop
    is first projected onto its best-fit plane (via
    [`polyline_normal`][triwarp.polyline.polyline_normal]) so any planar loop works, not only
    ones lying in the ``xy`` plane.

    The implementation is a GPU-parallel port of ``ear_clipping.cpp`` from libigl: convex polygons
    use a single fan, while non-convex polygons clip a maximal independent set of ears per round
    until the polygon is exhausted. Returned faces are consistently wound counter-clockwise with
    respect to the loop's turning direction; the exact set of triangles may differ from a
    sequential ear clip, but every triangulation of a simple polygon has ``n - 2`` faces.

    Every launch in the round loop is ``dim=n``, so the cost is set by the **round count**, and the
    round count by how many ears the independent-set rule can retire at once. Competing ears are
    ranked by a bijective hash of their ring index rather than by the index itself, which is what
    keeps that logarithmic: under the raw index an alternating star lets the ear at ``i - 2``
    suppress the ear at ``i`` for every ``i``, so one ear is clipped per round and the loop runs its
    full ``n``-round cap, where the hash keeps the round count near ``log2(n)``.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. A duplicated closing point is dropped via
        [`polyline_open`][triwarp.polyline.polyline_open].

    Returns
    -------
    twt.Array2dInt32
        ``(m, 3)`` triangle vertex indices into ``polyline`` on ``polyline.device``. ``m`` is
        ``n - 2`` for a simple polygon; a degenerate or self-intersecting loop may yield fewer
        (a partial triangulation). Empty ``(0, 3)`` for fewer than three points.

    Notes
    -----
    The round loop runs **on device**, driven by ``wp.capture_while`` over a device-side condition
    exactly as [`bfs`][triwarp.graph.bfs] drives its levels, so the whole clip costs one graph
    launch and one readback (the final face count) rather than a readback per round.

    **The prologue is fused.** The plane frame
    ([`polyline_normal`][triwarp.polyline.polyline_normal], its internal
    [`polyline_close`][triwarp.polyline.polyline_close] closure test, and
    [`polyline_centroid`][triwarp.polyline.polyline_centroid]) is built as one accumulation pass,
    one single-thread finalize and one projection, living in device memory and never crossing to
    the host -- rather than as three separate host-scope reductions each ending in its own
    readback. Only two readbacks are left in the whole function, both structural:
    ``polyline_open``'s [`is_closed`][triwarp.polyline.is_closed], which decides ``n`` and
    therefore every launch dimension, and the reflex count that selects the convex fan fast path.

    When conditional graph nodes are unavailable (CPU, or a CUDA driver below 12.4)
    ``wp.capture_while`` executes the same loop directly with one pinned 4-byte readback per round,
    which is the behaviour this loop had throughout. Verified equal: the CPU fallback and the
    captured CUDA loop produce the same triangulation up to row order on a 64-point star, and that
    row order was never stable on CUDA either — ``clip_selected`` appends through an atomic.

    See Also
    --------
    [`polyline_normal`][triwarp.polyline.polyline_normal]
    [`polyline_close`][triwarp.polyline.polyline_close]
    """
    polyline = polyline_open(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 3:
        return twt.empty_2d((0, 3), wp.int32, device=device)

    # The plane frame is built and consumed entirely on device: one accumulation pass, one
    # single-thread finalize, one projection. The three host-scope reductions this replaces
    # (``polyline_normal``, its internal ``polyline_close`` closure test, and
    # ``polyline_centroid``) each ended in a readback because the next one consumed its result,
    # and that prologue was flat in ``n`` -- the whole of this function's fixed cost at small
    # loops. ``frame`` is ``[center, u, v]``.
    normal = wp.zeros(1, dtype=wp.vec3, device=device)
    weighted_midpoint = wp.zeros(1, dtype=wp.vec3, device=device)
    total_length = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.accumulate_loop_frame,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[polyline, normal, weighted_midpoint, total_length],
        block_dim=TILE_1D,
        device=device,
    )
    frame = wp.empty(3, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_polyline.finalize_loop_frame,
        dim=1,
        inputs=[normal, weighted_midpoint, total_length, frame],
        device=device,
    )
    points2d = wp.empty(n, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_polyline.project_polyline_to_plane,
        dim=n,
        inputs=[polyline, frame, points2d],
        device=device,
    )

    # Orientation is fixed up on device (``orient_ccw`` reads the accumulated angle itself), so the
    # reflex count below is the only readback before the convex fast path returns.
    total = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch_tiled(
        kernel_polyline.accumulate_turning_angle,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[points2d, total],
        block_dim=TILE_1D,
        device=device,
    )
    wp.launch(kernel_polyline.orient_ccw, dim=n, inputs=[points2d, total], device=device)

    out_faces = twt.empty_2d((n - 2, 3), wp.int32, device=device)
    out_count = wp.zeros(1, dtype=wp.int32, device=device)

    reflex = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_polyline.count_reflex, dim=n, inputs=[points2d, reflex], device=device)
    if int(read_scalar(reflex, 0)) == 0:
        wp.launch(kernel_polyline.fan_triangulate, dim=n - 2, inputs=[out_faces], device=device)
        return twt.as_array2d(out_faces, wp.int32)

    left = wp.empty(n, dtype=wp.int32, device=device)
    right = wp.empty(n, dtype=wp.int32, device=device)
    active = wp.empty(n, dtype=wp.int32, device=device)
    is_ear = wp.empty(n, dtype=wp.int32, device=device)
    selected = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(kernel_polyline.init_ring, dim=n, inputs=[left, right, active], device=device)

    # state = [rounds run, loop condition], both written by ``ear_loop_continue``.
    state = wp.array([0, 1], dtype=wp.int32, device=device)  # see array.LOOP_ROUND

    def clip_round() -> None:
        wp.launch(
            kernel_polyline.compute_ears,
            dim=n,
            inputs=[points2d, left, right, active, is_ear],
            device=device,
        )
        wp.launch(
            kernel_polyline.select_independent,
            dim=n,
            inputs=[is_ear, left, right, selected],
            device=device,
        )
        wp.launch(
            kernel_polyline.clip_selected,
            dim=n,
            inputs=[selected, left, right, active, out_faces, out_count],
            device=device,
        )
        wp.launch(
            kernel_polyline.ear_loop_continue,
            dim=1,
            inputs=[out_count, wp.int32(n - 2), wp.int32(n), state],
            device=device,
        )

    condition = state[kernel_array.LOOP_CONDITION : kernel_array.LOOP_CONDITION + 1]
    # Graph capture needs a CUDA stream, so the CPU device takes the direct-execution branch even
    # when the driver supports conditional nodes.
    if wp.get_device(device).is_cuda and wp.is_conditional_graph_supported():
        with wp.ScopedCapture(device) as capture:
            wp.capture_while(condition, clip_round)
        wp.capture_launch(capture.graph)
    else:
        wp.capture_while(condition, clip_round)

    count = int(read_scalar(out_count, 0))
    return twt.as_array2d(out_faces[0:count], wp.int32)


def triangulate_polygon(polygon: wp.array[wp.vec2]) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """
    Triangulate a simple 2D polygon by ear clipping, adding no new vertices.

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped.

    Returns
    -------
    ring : wp.array[wp.vec2]
        The input ring with any repeated closing point removed; the vertices ``faces`` indexes.
    faces : wp.array[wp.int32]
        Length-``3 * (n - 2)`` flat triangle index buffer into ``ring``.

    Notes
    -----
    Differs from [`trimesh.creation.triangulate_polygon`][] in three ways, all of which follow from
    replacing the CPU polygon libraries (``mapbox_earcut`` / ``manifold3d`` / ``triangle``) with
    triwarp's own GPU ear clipper, [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]:

    - The input is a ``wp.vec2`` ring, not a ``shapely.geometry.Polygon``, and **interior rings
      (holes) are not supported**.
    - No Steiner points are ever inserted, which is trimesh's ``force_vertices=True`` contract
      rather than its default.
    - There is no ``engine`` selection, and ``faces`` is flat rather than ``(m, 3)``, matching the
      face layout used throughout triwarp.

    The ring must be a simple (non self-intersecting) polygon. A degenerate ring yields a partial
    triangulation with fewer than ``n - 2`` triangles rather than raising.

    See Also
    --------
    [`polyline_triangulate`][triwarp.polyline.polyline_triangulate]
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`trimesh.creation.triangulate_polygon`][]
    """
    twt.ensure_ndim(polygon, 1, dtype=wp.vec2)
    device = polygon.device
    n = int(polygon.shape[0])
    if n < 3:
        return polygon, wp.empty(0, dtype=wp.int32, device=device)

    lifted = wp.empty(n, dtype=wp.vec3, device=device)
    wp.map(kernel_array.lift_vec2, polygon, wp.float32(0.0), out=lifted)
    opened = polyline_open(lifted)
    faces = polyline_triangulate(opened).reshape((-1,))
    # polyline_open only ever drops a repeated final point, so the matching 2D ring is a prefix.
    return polygon[: int(opened.shape[0])].contiguous(), faces
