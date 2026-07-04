import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT
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
