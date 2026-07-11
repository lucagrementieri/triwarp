import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT, TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import (
    binary_search_index,
    cross2,
    update_argmax,
    vector_angle_vec,
    wrap_index,
)


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
def line_squared_distance(p: wp.vec3, s: wp.vec3, d: wp.vec3, seg_sq_len: wp.float32) -> wp.float32:
    """
    Squared perpendicular distance from ``p`` to the infinite line ``s -> d``.

    Mirrors ``igl::project_to_line`` with an **unclamped** parameter ``t`` (distance to the
    line, not the segment). ``seg_sq_len`` is the precomputed ``dot(d - s, d - s)``.
    """
    dms = d - s
    smp = s - p
    t = -wp.dot(dms, smp) / seg_sq_len
    proj = (1.0 - t) * s + t * d
    diff = p - proj
    return wp.dot(diff, diff)


@wp.func
def segment_length(start: wp.vec3, end: wp.vec3) -> wp.float32:
    return wp.length(end - start)


@wp.func
def segment_midpoint_and_length(start: wp.vec3, end: wp.vec3) -> tuple[wp.vec3, wp.float32]:
    displacement = end - start
    return start + 0.5 * displacement, wp.length(displacement)


@wp.kernel
def accumulate_newell_normal(polyline: wp.array[wp.vec3], out_normal: wp.array[wp.vec3]) -> None:
    # dim == n_points - 1: Newell's method sums cross products of consecutive vertices
    # (position vectors), cross(V_i, V_{i + 1}). Appending the closing vertex before launch
    # (as polyline_normal does) folds the wrap-around edge into this same sum.
    i = int(wp.tid())
    wp.atomic_add(out_normal, 0, wp.cross(polyline[i], polyline[i + 1]))


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


CURVATURE_EPS = wp.constant(wp.float32(1.0e-6))


@wp.func
def plane_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """
    Return the normal of the plane approximately containing segment vectors ``a``, ``b``, ``c``.

    Port of ``getPlaneNormal`` from MeshLib ``MRPolylineSubdivide.cpp``: returns whichever of
    ``b x (a + c)`` and ``b x (a - c)`` has the larger magnitude, staying well-defined when ``a``
    and ``c`` are nearly parallel or anti-parallel.
    """
    n1 = wp.cross(b, a + c)
    n2 = wp.cross(b, a - c)
    if wp.dot(n1, n1) >= wp.dot(n2, n2):
        return n1
    return n2


@wp.func
def endpoint_normals(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """
    In-plane unit normals at the two ends of segment ``b`` bracketed by neighbours ``a``, ``c``.

    Mirrors MeshLib's ``no``/``nd``: rotate each segment 90 degrees within the fitted plane
    (``plane_normal``) and average the edge normal with each neighbour's normal. Returns two
    zero vectors when the segments are (nearly) collinear, signalling the caller to fall back to a
    straight chord.
    """
    normal = plane_normal(a, b, c)
    if wp.dot(normal, normal) < CURVATURE_EPS:
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
    chord equals MeshLib's ``(|chord| / 2) * tan(theta / 4)`` sagitta, generalised here to every
    ``t`` for multi-point subdivision. Degenerate inputs (zero-length chord, collinear neighbours
    signalled by zero normals, straight/near-straight arc, or a cusp) collapse to the straight
    chord ``po + t * (pd - po)``, so ``t == 0`` always returns ``po`` exactly.
    """
    b = pd - po
    chord = wp.length(b)
    linear = po + t * b
    if chord < CURVATURE_EPS:
        return po
    # Zero end-normals are the collinear sentinel from endpoint_normals; unit normals have norm 1.
    if wp.dot(no, no) < 0.5 or wp.dot(nd, nd) < 0.5:
        return linear
    theta = vector_angle_vec(no, nd)
    if theta < CURVATURE_EPS:
        return linear
    tangent = b / chord
    sign = 1.0
    if wp.dot(b, nd - no) < 0.0:
        sign = -1.0
    bulge = sign * (no + nd)
    m = bulge - wp.dot(bulge, tangent) * tangent  # bulge direction, orthogonalised against chord
    if wp.dot(m, m) < CURVATURE_EPS:
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
    j = int(wp.tid())
    segment = binary_search_index(offsets, j) - 1
    k = j - offsets[segment]
    t = wp.float32(k) / wp.float32(steps[segment])
    n = polyline.shape[0]
    po = polyline[segment]
    pd = polyline[segment + 1]
    # Locate the vertices bracketing this segment; interior segments fit a curvature arc, boundary
    # segments of an open polyline (missing a neighbour) stay linear, matching MeshLib.
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
        out_points[j] = po + t * (pd - po)
        return
    no, nd = endpoint_normals(po - polyline[prev_index], pd - po, polyline[next_index] - pd)
    out_points[j] = arc_point(po, pd, no, nd, t)


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


@wp.kernel(enable_backward=False)
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
    # int()/float() declare mutable Warp dynamic variables; bare literals are compile-time
    # constants that get folded (freezing the loop). See noqa: UP018/RUF046 below.
    top = int(1)  # noqa: UP018, RUF046 — number of (ixs, ixe) pairs currently on the stack
    while top > 0:
        top -= 1
        ixs = stack[2 * top + 0]
        ixe = stack[2 * top + 1]
        sdmax = float(0.0)  # noqa: UP018 — mutable Warp dynamic variable
        ixc = int(-1)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
        if ixe - ixs > 1:
            seg = polyline[ixe] - polyline[ixs]
            sdes = wp.dot(seg, seg)
            for k in range(ixs + 1, ixe):
                sd = float(0.0)  # noqa: UP018 — mutable; initialize before branching
                if sdes <= RDP_LINE_EPS:
                    dvec = polyline[k] - polyline[ixs]
                    sd = wp.dot(dvec, dvec)
                else:
                    sd = line_squared_distance(polyline[k], polyline[ixs], polyline[ixe], sdes)
                # strict '>' inside update_argmax keeps the first argmax (Eigen maxCoeff)
                update_argmax(sdmax, ixc, sd, k)
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
def broadcast_first_point(polyline: wp.array[wp.vec3], out_points: wp.array[wp.vec3]) -> None:
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
    """
    Sign of the 2D cross product ``(b - a) x (c - a)``.

    ``+1`` CCW, ``-1`` CW, ``0`` collinear.
    """
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if det > TOLERANCE_ZERO_CONSTANT:
        return wp.int32(1)
    if det < -TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1)
    return wp.int32(0)


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
def accumulate_turning_angle(points2d: wp.array[wp.vec2], out_total: wp.array[wp.float32]) -> None:
    # Cyclic signed exterior angle at each vertex; the sum's sign gives the loop orientation.
    i = int(wp.tid())
    n = points2d.shape[0]
    current = points2d[i]
    nxt = points2d[(i + 1) % n]
    after = points2d[(i + 2) % n]
    d1 = nxt - current
    d2 = after - nxt
    angle = wp.atan2(cross2(d1, d2), wp.dot(d1, d2))
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
    prev = points2d[wrap_index(i - 1, n)]
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
    left_i = left[i]
    r = right[i]
    rr = right[right[i]]
    if is_ear[ll] == 1 and ll < i:
        return
    if is_ear[left_i] == 1 and left_i < i:
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
