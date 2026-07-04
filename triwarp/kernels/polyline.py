import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import binary_search_index, vector_angle_vec


@wp.func
def segment_displacement(polyline: wp.array[wp.vec3], i: wp.int32) -> wp.vec3:
    """Displacement vector ``polyline[i + 1] - polyline[i]`` of segment ``i``."""
    return polyline[i + 1] - polyline[i]


@wp.func
def segment_coordinate(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.float32:
    """Clamped projection parameter of ``p`` onto segment ``a -> b`` in ``[0, 1]``."""
    ab = b - a
    length_sq = wp.max(wp.dot(ab, ab), TOLERANCE_MERGE_CONSTANT)
    return wp.clamp(wp.dot(p - a, ab) / length_sq, 0.0, 1.0)


@wp.func
def closest_point_on_segment(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.vec3:
    """Point on segment ``a -> b`` closest to ``p``."""
    t = segment_coordinate(a, b, p)
    return a + t * (b - a)


@wp.func
def point_to_segment_distance(a: wp.vec3, b: wp.vec3, p: wp.vec3) -> wp.float32:
    """Euclidean distance from ``p`` to the closest point on segment ``a -> b``."""
    return wp.length(p - closest_point_on_segment(a, b, p))


@wp.func
def project_point_to_plane(p: wp.vec3, origin: wp.vec3, unit_normal: wp.vec3) -> wp.vec3:
    """Orthogonal projection of ``p`` onto the plane through ``origin`` with ``unit_normal``."""
    return p - unit_normal * wp.dot(p - origin, unit_normal)


@wp.func
def line_squared_distance(
    p: wp.vec3, s: wp.vec3, d: wp.vec3, seg_sq_len: wp.float32
) -> wp.float32:
    """Squared perpendicular distance from ``p`` to the infinite line ``s -> d``.

    Mirrors ``igl::project_to_line`` with an **unclamped** parameter ``t`` (distance to the
    line, not the segment). ``seg_sq_len`` is the precomputed ``dot(d - s, d - s)``.
    """
    dms = d - s
    smp = s - p
    t = -wp.dot(dms, smp) / seg_sq_len
    proj = (1.0 - t) * s + t * d
    diff = p - proj
    return wp.dot(diff, diff)


@wp.kernel
def segment_lengths(polyline: wp.array[wp.vec3], out_lengths: wp.array[wp.float32]) -> None:
    i = int(wp.tid())
    out_lengths[i] = wp.length(segment_displacement(polyline, i))


@wp.kernel
def segment_midpoints_and_lengths(
    polyline: wp.array[wp.vec3],
    out_midpoints: wp.array[wp.vec3],
    out_lengths: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    displacement = segment_displacement(polyline, i)
    out_midpoints[i] = polyline[i] + 0.5 * displacement
    out_lengths[i] = wp.length(displacement)


@wp.kernel
def accumulate_newell_normal(polyline: wp.array[wp.vec3], out_normal: wp.array[wp.vec3]) -> None:
    # dim == n_points - 2: consecutive segment pairs (i, i + 1), no wrap-around.
    i = int(wp.tid())
    s0 = segment_displacement(polyline, i)
    s1 = segment_displacement(polyline, i + 1)
    wp.atomic_add(out_normal, 0, wp.cross(s0, s1))


@wp.kernel
def accumulate_newell_normal_closed(
    polyline: wp.array[wp.vec3], out_normal: wp.array[wp.vec3]
) -> None:
    # dim == n_points - 1: cyclic segment pairs (closing edge included), for a closed polyline.
    i = int(wp.tid())
    n_segments = polyline.shape[0] - 1
    s0 = segment_displacement(polyline, i)
    s1 = segment_displacement(polyline, (i + 1) % n_segments)
    wp.atomic_add(out_normal, 0, wp.cross(s0, s1))


@wp.kernel
def cyclic_segment_angles(polyline: wp.array[wp.vec3], out_angles: wp.array[wp.float32]) -> None:
    # dim == n_points - 1: angle between segment i and the cyclically next segment.
    i = int(wp.tid())
    n_segments = polyline.shape[0] - 1
    s0 = segment_displacement(polyline, i)
    s1 = segment_displacement(polyline, (i + 1) % n_segments)
    out_angles[i] = vector_angle_vec(wp.normalize(s0), wp.normalize(s1))


@wp.kernel
def distance_to_segments(
    points: wp.array[wp.vec3], polyline: wp.array[wp.vec3], out_distances: wp.array[wp.float32]
) -> None:
    tid = int(wp.tid())
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
    tid = int(wp.tid())
    out_distances[tid] = wp.length(points[tid] - polyline[0])


@wp.kernel
def segment_step_counts(
    polyline: wp.array[wp.vec3], step_size: wp.float32, out_steps: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    length = wp.length(segment_displacement(polyline, i))
    out_steps[i] = wp.max(wp.int32(wp.floor(length / step_size)), wp.int32(1))


@wp.kernel
def upsample_gather(
    polyline: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    j = int(wp.tid())
    segment = binary_search_index(offsets, j) - 1
    k = j - offsets[segment]
    weight = wp.float32(k) / wp.float32(steps[segment])
    out_points[j] = polyline[segment] + weight * segment_displacement(polyline, segment)


@wp.kernel
def greedy_downsample_mask(
    cumulative_lengths: wp.array[wp.float32], step_size: wp.float32, out_keep: wp.array[wp.bool]
) -> None:
    # Single-thread greedy walk (dim == 1): the selection is inherently sequential.
    n = cumulative_lengths.shape[0]
    out_keep[0] = True
    last = cumulative_lengths[0]
    for i in range(1, n):
        if cumulative_lengths[i] - last >= step_size:
            out_keep[i] = True
            last = cumulative_lengths[i]


RDP_LINE_EPS = wp.constant(wp.float32(1.0e-7))  # libigl FLOAT_EPS: degenerate-segment threshold


@wp.kernel
def rdp_keep_mask(
    polyline: wp.array[wp.vec3],
    stol: wp.float32,
    stack: wp.array[wp.int32],
    out_keep: wp.array[wp.bool],
) -> None:
    # Single thread (dim == 1): iterative Ramer-Douglas-Peucker over an explicit stack of
    # (ixs, ixe) index ranges, since Warp forbids recursion. ``stack`` is scratch holding the
    # ranges interleaved; its size must be >= max(2 * n, 2). The first and last vertices are
    # always kept; interior vertices closer than sqrt(stol) to their bracketing chord are dropped.
    n = polyline.shape[0]
    for i in range(n):
        out_keep[i] = True
    stack[0] = 0
    stack[1] = n - 1
    top = int(1)  # number of (ixs, ixe) pairs currently on the stack
    while top > 0:
        top -= 1
        ixs = stack[2 * top + 0]
        ixe = stack[2 * top + 1]
        sdmax = float(0.0)
        ixc = int(-1)
        if ixe - ixs > 1:
            seg = polyline[ixe] - polyline[ixs]
            sdes = wp.dot(seg, seg)
            for k in range(ixs + 1, ixe):
                sd = float(0.0)  # initialize before branching (variable scope rule)
                if sdes <= RDP_LINE_EPS:
                    dvec = polyline[k] - polyline[ixs]
                    sd = wp.dot(dvec, dvec)
                else:
                    sd = line_squared_distance(polyline[k], polyline[ixs], polyline[ixe], sdes)
                if sd > sdmax:  # strict '>' keeps the first argmax, matching Eigen maxCoeff
                    sdmax = sd
                    ixc = k
        if sdmax <= stol:
            for k in range(ixs + 1, ixe):  # empty range when there are no interior points
                out_keep[k] = False
        else:
            stack[2 * top + 0] = ixs
            stack[2 * top + 1] = ixc
            top += 1
            stack[2 * top + 0] = ixc
            stack[2 * top + 1] = ixe
            top += 1


@wp.kernel
def broadcast_first_point(
    polyline: wp.array[wp.vec3], out_points: wp.array[wp.vec3]
) -> None:
    j = int(wp.tid())
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
    j = int(wp.tid())
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
        out_points[j] = polyline[hi - 1] + t * (polyline[hi] - polyline[hi - 1])


@wp.kernel
def radius_segment_distances(
    polyline: wp.array[wp.vec3],
    center: wp.vec3,
    normal: wp.vec3,
    out_distances: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    unit_normal = wp.normalize(normal)
    a = project_point_to_plane(polyline[i], center, unit_normal)
    b = project_point_to_plane(polyline[i + 1], center, unit_normal)
    out_distances[i] = wp.length(closest_point_on_segment(a, b, center) - center)


# --- polygon triangulation (parallel ear clipping); port of libigl ear_clipping.cpp ---


@wp.func
def orient2d(a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.int32:
    """Sign of the 2D cross product ``(b - a) x (c - a)``: ``+1`` CCW, ``-1`` CW, ``0`` collinear."""
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if det > TOLERANCE_ZERO_CONSTANT:
        return wp.int32(1)
    if det < -TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1)
    return wp.int32(0)


@wp.func
def point_in_triangle(a: wp.vec2, b: wp.vec2, c: wp.vec2, p: wp.vec2) -> wp.bool:
    """Whether ``p`` lies inside or on the boundary of the CCW triangle ``(a, b, c)``.

    Boundary inclusion matters for the ear test: a (reflex) vertex lying exactly on a candidate
    ear's cutting diagonal must block that ear, otherwise a degenerate/overlapping triangle is
    emitted.
    """
    return (
        orient2d(a, b, p) >= 0 and orient2d(b, c, p) >= 0 and orient2d(c, a, p) >= 0
    )


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


@wp.kernel
def project_to_plane_2d(
    polyline: wp.array[wp.vec3],
    center: wp.vec3,
    u: wp.vec3,
    v: wp.vec3,
    out_points2d: wp.array[wp.vec2],
) -> None:
    i = int(wp.tid())
    d = polyline[i] - center
    out_points2d[i] = wp.vec2(wp.dot(d, u), wp.dot(d, v))


@wp.kernel
def accumulate_turning_angle(
    points2d: wp.array[wp.vec2], out_total: wp.array[wp.float32]
) -> None:
    # Cyclic signed exterior angle at each vertex; the sum's sign gives the loop orientation.
    i = int(wp.tid())
    n = points2d.shape[0]
    current = points2d[i]
    nxt = points2d[(i + 1) % n]
    after = points2d[(i + 2) % n]
    d1 = nxt - current
    d2 = after - nxt
    angle = wp.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1[0] * d2[0] + d1[1] * d2[1])
    wp.atomic_add(out_total, 0, angle)


@wp.kernel
def orient_ccw(points2d: wp.array[wp.vec2]) -> None:
    # Mirror the y-axis to flip a clockwise loop to counter-clockwise (replaces libigl's row
    # reversal); the convex/ear tests assume CCW orientation.
    i = int(wp.tid())
    p = points2d[i]
    points2d[i] = wp.vec2(p[0], -p[1])


@wp.kernel
def count_reflex(points2d: wp.array[wp.vec2], out_count: wp.array[wp.int32]) -> None:
    # Pre-clip the ring is trivial, so use direct cyclic neighbours. Convex polygon <=> 0 reflex.
    i = int(wp.tid())
    n = points2d.shape[0]
    prev = points2d[(i - 1 + n) % n]
    cur = points2d[i]
    nxt = points2d[(i + 1) % n]
    if orient2d(prev, cur, nxt) < 0:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def fan_triangulate(out_faces: wp.array2d[wp.int32]) -> None:
    # Convex fast-path: fan from vertex 0. dim == n - 2.
    k = int(wp.tid())
    out_faces[k, 0] = wp.int32(0)
    out_faces[k, 1] = k + 1
    out_faces[k, 2] = k + 2


@wp.kernel
def init_ring(
    left: wp.array[wp.int32], right: wp.array[wp.int32], active: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    n = left.shape[0]
    left[i] = (i - 1 + n) % n
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
    i = int(wp.tid())
    n = points2d.shape[0]
    if active[i] == 0:
        out_is_ear[i] = wp.int32(0)
        return
    if is_ear_at(points2d, left, right, active, i, n):
        out_is_ear[i] = wp.int32(1)
    else:
        out_is_ear[i] = wp.int32(0)


@wp.kernel
def select_independent(
    is_ear: wp.array[wp.int32],
    left: wp.array[wp.int32],
    right: wp.array[wp.int32],
    out_selected: wp.array[wp.int32],
) -> None:
    # Select ear i iff it has the smallest index among ears within ring-distance 2. This keeps
    # chosen ears >= 3 apart, so their clip footprints {L[i], i, R[i]} are disjoint and can be
    # clipped concurrently. The global-min-index ear is always selected, guaranteeing progress.
    i = int(wp.tid())
    out_selected[i] = wp.int32(0)
    if is_ear[i] == 0:
        return
    ll = left[left[i]]
    l = left[i]
    r = right[i]
    rr = right[right[i]]
    if is_ear[ll] == 1 and ll < i:
        return
    if is_ear[l] == 1 and l < i:
        return
    if is_ear[r] == 1 and r < i:
        return
    if is_ear[rr] == 1 and rr < i:
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
    i = int(wp.tid())
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
