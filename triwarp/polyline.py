"""Open and closed 3D polyline operations on NVIDIA Warp."""

from __future__ import annotations

from typing import Literal

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import polyline as kernel_polyline


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
    [`open_polyline`][triwarp.polyline.open_polyline]
    [`close_polyline`][triwarp.polyline.close_polyline]
    """
    n = int(polyline.shape[0])
    return n >= 2 and tw.array.allclose(polyline[0:1], polyline[n - 1 : n])


def open_polyline(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
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
    [`close_polyline`][triwarp.polyline.close_polyline]
    """
    n = int(polyline.shape[0])
    if not is_closed(polyline):
        return polyline
    return polyline[0 : n - 1]


def close_polyline(polyline: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
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
    [`open_polyline`][triwarp.polyline.open_polyline]
    """
    n = int(polyline.shape[0])
    if n < 2 or is_closed(polyline):
        return polyline
    return tw.array.concatenate([polyline, polyline[0:1]])


def polyline_length(polyline: wp.array[wp.vec3]) -> float:
    """
    Total arc length of a polyline (sum of segment lengths).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    float
        The summed segment length, ``0.0`` for fewer than two points.

    See Also
    --------
    [`closed_polyline_length`][triwarp.polyline.closed_polyline_length]
    """
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    if n_segments < 1:
        return 0.0
    lengths = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(kernel_polyline.segment_length, polyline[:-1], polyline[1:], out=lengths)
    return float(tw.reduce.sum(lengths))


def closed_polyline_length(polyline: wp.array[wp.vec3]) -> float:
    """
    Total arc length of a closed polyline (closing edge included).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    float
        The summed segment length of the closed polyline.

    See Also
    --------
    [`polyline_length`][triwarp.polyline.polyline_length]
    """
    return polyline_length(close_polyline(polyline))


def polyline_centroid(polyline: wp.array[wp.vec3]) -> wp.vec3:
    """
    Segment-length-weighted centroid of a polyline.

    Each segment contributes its midpoint weighted by its length, so the result is invariant to
    how densely the polyline is sampled (unlike the plain mean of the vertices).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

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
    [`closed_polyline_centroid`][triwarp.polyline.closed_polyline_centroid]
    """
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


def closed_polyline_centroid(polyline: wp.array[wp.vec3]) -> wp.vec3:
    """
    Segment-length-weighted centroid of a closed polyline (closing edge included).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    wp.vec3
        The weighted centroid on the host.

    See Also
    --------
    [`polyline_centroid`][triwarp.polyline.polyline_centroid]
    """
    return polyline_centroid(close_polyline(polyline))


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
    polyline = close_polyline(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    # The closed polyline's last vertex duplicates its first, so it has n - 1 distinct vertices;
    # a non-degenerate loop normal needs at least three of them.
    if n < 4:
        raise ValueError("polyline_normal requires at least three points")
    out_normal = wp.zeros(1, dtype=wp.vec3, device=device)
    # The polyline is closed (last vertex duplicates the first), so summing cross(V_i, V_{i + 1})
    # over the n - 1 consecutive pairs includes the wrap-around edge — full Newell's method.
    wp.launch(
        kernel_polyline.accumulate_newell_normal,
        dim=n - 1,
        inputs=[polyline, out_normal],
        device=device,
    )
    wp.map(wp.normalize, out_normal, out=out_normal)
    return wp.vec3(*out_normal.numpy()[0].tolist())


def distance_to_polyline(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3]
) -> wp.array[wp.float32]:
    """
    Minimum distance from each query point to the nearest segment of a polyline.

    Parameters
    ----------
    points
        ``(n,)`` query points as ``wp.vec3``.
    polyline
        ``(m,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` minimum distances on ``points.device``.

    See Also
    --------
    [`distance_to_closed_polyline`][triwarp.polyline.distance_to_closed_polyline]
    """
    device = points.device
    n_points = int(points.shape[0])
    m = int(polyline.shape[0])
    out_distances = wp.empty(n_points, dtype=wp.float32, device=device)
    if m == 0 or n_points == 0:
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


def distance_to_closed_polyline(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3]
) -> wp.array[wp.float32]:
    """
    Minimum distance from each query point to a closed 3D polyline.

    Parameters
    ----------
    points
        ``(n,)`` query points as ``wp.vec3``.
    polyline
        ``(m,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` minimum distances on ``points.device``.

    See Also
    --------
    [`distance_to_polyline`][triwarp.polyline.distance_to_polyline]
    """
    return distance_to_polyline(points, close_polyline(polyline))


def upsample_polyline(polyline: wp.array[wp.vec3], step_size: float) -> wp.array[wp.vec3]:
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

    Returns
    -------
    wp.array[wp.vec3]
        The upsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`upsample_closed_polyline`][triwarp.polyline.upsample_closed_polyline]
    [`downsample_polyline`][triwarp.polyline.downsample_polyline]
    [`resample_polyline`][triwarp.polyline.resample_polyline]
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
    offsets = wp.empty(n_segments, dtype=wp.int32, device=device)
    inclusive = wp.empty(n_segments, dtype=wp.int32, device=device)
    wp.utils.array_scan(steps, out_array=offsets, inclusive=False)
    wp.utils.array_scan(steps, out_array=inclusive, inclusive=True)
    total = int(inclusive.numpy()[-1])

    out_points = wp.empty(total, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_polyline.upsample_gather,
        dim=total,
        inputs=[polyline, offsets, steps, out_points],
        device=device,
    )
    return out_points


def upsample_closed_polyline(polyline: wp.array[wp.vec3], step_size: float) -> wp.array[wp.vec3]:
    """
    Upsample a closed polyline to an approximately uniform step size.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    step_size
        Target spacing between consecutive output points.

    Returns
    -------
    wp.array[wp.vec3]
        The upsampled closed polyline.

    See Also
    --------
    [`upsample_polyline`][triwarp.polyline.upsample_polyline]
    """
    return upsample_polyline(close_polyline(polyline), step_size)


def _smooth_upsample(
    polyline: wp.array[wp.vec3], step_size: float, closed: bool
) -> wp.array[wp.vec3]:
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
    offsets = wp.empty(n_segments, dtype=wp.int32, device=device)
    inclusive = wp.empty(n_segments, dtype=wp.int32, device=device)
    wp.utils.array_scan(steps, out_array=offsets, inclusive=False)
    wp.utils.array_scan(steps, out_array=inclusive, inclusive=True)
    total = int(inclusive.numpy()[-1])

    out_points = wp.empty(total, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_polyline.smooth_upsample_gather,
        dim=total,
        inputs=[polyline, offsets, steps, wp.int32(closed), out_points],
        device=device,
    )
    return out_points


def smooth_upsample_polyline(polyline: wp.array[wp.vec3], step_size: float) -> wp.array[wp.vec3]:
    """
    Upsample a polyline to an approximately uniform step size, following local curvature.

    Like [`upsample_polyline`][triwarp.polyline.upsample_polyline], each segment is split into
    ``max(floor(length / step_size), 1)`` pieces and the final endpoint is not emitted. Unlike it,
    the inserted points are placed on a circular arc fitted to the segment's endpoint tangents
    (estimated from the two bracketing neighbour vertices) rather than on the straight chord, so a
    coarsely sampled curve is refined smoothly. This is a port of the ``useCurvature`` vertex
    placement in MeshLib ``MRPolylineSubdivide.cpp``, generalised from the edge midpoint to every
    interpolation parameter.

    The first and last segments have no bracketing neighbour and are subdivided linearly (matching
    MeshLib, which applies curvature only to interior edges); collinear neighbours likewise reduce
    to the straight chord. Original vertices are preserved exactly, since each segment's first
    sample coincides with its start vertex.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    step_size
        Target spacing between consecutive output points.

    Returns
    -------
    wp.array[wp.vec3]
        The curvature-aware upsampled polyline. The input is returned unchanged for fewer than
        two points.

    See Also
    --------
    [`smooth_upsample_closed_polyline`][triwarp.polyline.smooth_upsample_closed_polyline]
    [`upsample_polyline`][triwarp.polyline.upsample_polyline]
    """
    return _smooth_upsample(polyline, step_size, closed=False)


def smooth_upsample_closed_polyline(
    polyline: wp.array[wp.vec3], step_size: float
) -> wp.array[wp.vec3]:
    """
    Upsample a closed polyline to an approximately uniform step size, following local curvature.

    The closing edge is added if absent, and every segment — including the seam — is treated as
    interior, so neighbour tangents wrap cyclically and the whole loop is smoothed. The duplicated
    closing point is not emitted, yielding a clean cyclic ring.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    step_size
        Target spacing between consecutive output points.

    Returns
    -------
    wp.array[wp.vec3]
        The curvature-aware upsampled closed polyline.

    See Also
    --------
    [`smooth_upsample_polyline`][triwarp.polyline.smooth_upsample_polyline]
    [`upsample_closed_polyline`][triwarp.polyline.upsample_closed_polyline]
    """
    return _smooth_upsample(close_polyline(polyline), step_size, closed=True)


def cumulative_arc_length(polyline: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Cumulative arc length from the first vertex to each vertex of an open polyline.

    The underlying arc-length parametrization behind
    [`downsample_polyline`][triwarp.polyline.downsample_polyline] and
    [`resample_polyline`][triwarp.polyline.resample_polyline]; exposed directly for callers
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
    [`downsample_polyline`][triwarp.polyline.downsample_polyline]
    [`resample_polyline`][triwarp.polyline.resample_polyline]
    """
    device = polyline.device
    n_segments = int(polyline.shape[0]) - 1
    lengths = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.map(kernel_polyline.segment_length, polyline[:-1], polyline[1:], out=lengths)
    inclusive = wp.empty(n_segments, dtype=wp.float32, device=device)
    wp.utils.array_scan(lengths, out_array=inclusive, inclusive=True)
    return tw.array.concatenate([wp.zeros(1, dtype=wp.float32, device=device), inclusive])


def downsample_polyline(polyline: wp.array[wp.vec3], step_size: float) -> wp.array[wp.vec3]:
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

    Returns
    -------
    wp.array[wp.vec3]
        The downsampled polyline. The input is returned unchanged for fewer than two points.

    See Also
    --------
    [`downsample_closed_polyline`][triwarp.polyline.downsample_closed_polyline]
    [`upsample_polyline`][triwarp.polyline.upsample_polyline]
    """
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 2:
        return polyline

    cumulative = cumulative_arc_length(polyline)
    keep_mask = wp.zeros(n, dtype=wp.bool, device=device)
    wp.launch(
        kernel_polyline.greedy_downsample_mask,
        dim=1,
        inputs=[cumulative, wp.float32(step_size), keep_mask],
        device=device,
    )
    return tw.array.gather(polyline, tw.array.flatnonzero(keep_mask))


def downsample_closed_polyline(polyline: wp.array[wp.vec3], step_size: float) -> wp.array[wp.vec3]:
    """
    Downsample a closed polyline to a minimum arc-length spacing between kept points.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    step_size
        Minimum arc-length distance between kept points.

    Returns
    -------
    wp.array[wp.vec3]
        The downsampled closed polyline.

    See Also
    --------
    [`downsample_polyline`][triwarp.polyline.downsample_polyline]
    """
    return downsample_polyline(close_polyline(polyline), step_size)


def simplify_polyline(
    polyline: wp.array[wp.vec3], tol: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    Recursively drops interior vertices whose perpendicular distance to the chord spanning a
    kept sub-range is at most ``tol``; the first and last vertices are always retained. This is
    a Warp port of ``ramer_douglas_peucker`` from libigl, evaluated on-device by a single-thread
    stack-based kernel (Warp forbids recursion).

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.
    tol
        Maximum Euclidean distance allowed between a dropped vertex and the retained chord.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(simplified, indices)`` on ``polyline.device``: the ``(m,)`` retained vertices and the
        ``(m,)`` sorted indices into the input such that ``polyline[indices] == simplified``. An
        empty input yields two empty arrays; a single point is returned unchanged with
        ``indices == [0]``.

    See Also
    --------
    [`simplify_closed_polyline`][triwarp.polyline.simplify_closed_polyline]
    [`downsample_polyline`][triwarp.polyline.downsample_polyline]
    """
    device = polyline.device
    n = int(polyline.shape[0])
    # Everything is kept until the recursion drops it, and filling that in parallel here keeps the
    # serial kernel's only serial work the part that has to be.
    keep_mask = wp.full(n, True, dtype=wp.bool, device=device)
    # Scratch stack of interleaved (ixs, ixe) ranges; max(2 * n, 2) keeps n == 0 in bounds.
    stack = wp.empty(max(2 * n, 2), dtype=wp.int32, device=device)
    wp.launch(
        kernel_polyline.rdp_keep_mask,
        dim=1,
        inputs=[polyline, wp.float32(tol * tol), stack, keep_mask],
        device=device,
    )
    indices = tw.array.flatnonzero(keep_mask)
    return tw.array.gather(polyline, indices), indices


def simplify_closed_polyline(
    polyline: wp.array[wp.vec3], tol: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Simplify a closed polyline with the Ramer-Douglas-Peucker algorithm.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    tol
        Maximum Euclidean distance allowed between a dropped vertex and the retained chord.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(simplified, indices)`` on ``polyline.device``. ``indices`` refer to the *closed*
        polyline (the input with its closing point appended), so the shared start/end vertex is
        preserved at both ends.

    See Also
    --------
    [`simplify_polyline`][triwarp.polyline.simplify_polyline]
    [`close_polyline`][triwarp.polyline.close_polyline]
    """
    return simplify_polyline(close_polyline(polyline), tol)


def resample_polyline(polyline: wp.array[wp.vec3], num_points: int) -> wp.array[wp.vec3]:
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

    Returns
    -------
    wp.array[wp.vec3]
        Length ``num_points`` resampled polyline. An empty input is returned unchanged; a single
        point is repeated ``num_points`` times.

    See Also
    --------
    [`resample_closed_polyline`][triwarp.polyline.resample_closed_polyline]
    [`upsample_polyline`][triwarp.polyline.upsample_polyline]
    """
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


def resample_closed_polyline(polyline: wp.array[wp.vec3], num_points: int) -> wp.array[wp.vec3]:
    """
    Resample a closed polyline to a fixed number of points evenly spaced by arc length.

    Resamples the closed polyline to ``num_points + 1`` points and drops the duplicated closing
    point, so the returned polyline has ``num_points`` distinct points.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    num_points
        Number of output points.

    Returns
    -------
    wp.array[wp.vec3]
        Length ``num_points`` resampled closed polyline.

    See Also
    --------
    [`resample_polyline`][triwarp.polyline.resample_polyline]
    """
    resampled = resample_polyline(close_polyline(polyline), num_points + 1)
    return resampled[0:num_points]


def polyline_radius(
    polyline: wp.array[wp.vec3],
    reduction: Literal["min", "max", "mean", "median"] = "min",
    center: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
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
    [`closed_polyline_radius`][triwarp.polyline.closed_polyline_radius]
    [`median`][triwarp.reduce.median]
    """
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
    wp.launch(
        kernel_polyline.radius_segment_distances,
        dim=n_segments,
        inputs=[polyline, center, normal, distances],
        device=device,
    )
    if reduction == "min":
        return float(tw.reduce.min(distances))
    if reduction == "max":
        return float(tw.reduce.max(distances))
    if reduction == "mean":
        return float(tw.reduce.mean(distances))
    return tw.reduce.median(distances)


def closed_polyline_radius(
    polyline: wp.array[wp.vec3],
    reduction: Literal["min", "max", "mean", "median"] = "min",
    center: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
) -> float:
    """
    Radius of a closed polyline projected onto the plane through ``center`` with ``normal``.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.
    reduction
        Reduction over the per-segment radial distances. Defaults to ``"min"``.
    center
        Plane origin. Defaults to the closed-polyline centroid.
    normal
        Plane normal. Defaults to the closed-polyline normal.

    Returns
    -------
    float
        The reduced radius.

    See Also
    --------
    [`polyline_radius`][triwarp.polyline.polyline_radius]
    """
    return polyline_radius(close_polyline(polyline), reduction, center, normal)


def polyline_angles(polyline: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Angles between consecutive segments at each vertex of a polyline.

    Returns one angle per point. For an open polyline the two endpoints get an angle of ``0``;
    for a closed polyline (first point equal to last) the turning angles wrap around.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` angles in radians on ``polyline.device``. All zeros for fewer than two points.

    See Also
    --------
    [`closed_polyline_angles`][triwarp.polyline.closed_polyline_angles]
    """
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


def closed_polyline_angles(polyline: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Angles between consecutive segments at each vertex of a closed polyline.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. The closing edge is added if absent.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` angles in radians on ``polyline.device`` (one per original point).

    See Also
    --------
    [`polyline_angles`][triwarp.polyline.polyline_angles]
    """
    n_original = int(polyline.shape[0])
    angles = polyline_angles(close_polyline(polyline))
    return angles[0:n_original]


def triangulate_polyline(polyline: wp.array[wp.vec3]) -> twt.Array2dInt32:
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
    full ``n``-round cap — measured at 141 ms for a 1 024-point star, against 4.6 ms and 30 rounds
    under the hash.

    Parameters
    ----------
    polyline
        ``(n,)`` polyline vertices as ``wp.vec3``. A duplicated closing point is dropped via
        [`open_polyline`][triwarp.polyline.open_polyline].

    Returns
    -------
    twt.Array2dInt32
        ``(m, 3)`` triangle vertex indices into ``polyline`` on ``polyline.device``. ``m`` is
        ``n - 2`` for a simple polygon; a degenerate or self-intersecting loop may yield fewer
        (a partial triangulation). Empty ``(0, 3)`` for fewer than three points.

    Notes
    -----
    The per-round convergence readback is 45 us of each ~150 us round (measured, RTX 5090), so
    replacing it with a ``wp.capture_while`` loop as [`triwarp.graph.bfs`][triwarp.graph.bfs] does
    would have to cost less than that per round to pay. It does not: conditional-graph iteration is
    recorded at ~0.25 ms an iteration elsewhere in this package, five times the readback it would
    remove. The loop stays host-driven.

    See Also
    --------
    [`polyline_normal`][triwarp.polyline.polyline_normal]
    [`close_polyline`][triwarp.polyline.close_polyline]
    """
    polyline = open_polyline(polyline)
    device = polyline.device
    n = int(polyline.shape[0])
    if n < 3:
        return twt.empty_int32_2d((0, 3), device=device)

    u, v = tw.points.plane_basis(polyline_normal(polyline))
    center = polyline_centroid(polyline)
    points2d = wp.empty(n, dtype=wp.vec2, device=device)
    wp.map(kernel_polyline.project_to_plane_2d, polyline, center, u, v, out=points2d)

    # Orientation is fixed up on device (``orient_ccw`` reads the accumulated angle itself), so the
    # reflex count below is the only readback before the convex fast path returns.
    total = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_polyline.accumulate_turning_angle, dim=n, inputs=[points2d, total], device=device
    )
    wp.launch(kernel_polyline.orient_ccw, dim=n, inputs=[points2d, total], device=device)

    out_faces = twt.empty_int32_2d((n - 2, 3), device=device)
    out_count = wp.zeros(1, dtype=wp.int32, device=device)

    reflex = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(kernel_polyline.count_reflex, dim=n, inputs=[points2d, reflex], device=device)
    if int(reflex.numpy()[0]) == 0:
        wp.launch(kernel_polyline.fan_triangulate, dim=n - 2, inputs=[out_faces], device=device)
        return twt.as_array2d_int32(out_faces)

    left = wp.empty(n, dtype=wp.int32, device=device)
    right = wp.empty(n, dtype=wp.int32, device=device)
    active = wp.empty(n, dtype=wp.int32, device=device)
    is_ear = wp.empty(n, dtype=wp.int32, device=device)
    selected = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(kernel_polyline.init_ring, dim=n, inputs=[left, right, active], device=device)

    for _ in range(n):
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
        if int(out_count.numpy()[0]) >= n - 2:
            break

    count = int(out_count.numpy()[0])
    return twt.as_array2d_int32(out_faces[0:count])
