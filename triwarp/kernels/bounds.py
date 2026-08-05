import math

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, TILE_1D

# Super-Fibonacci spiral constants [Alexa 2022]: the two irrational strides whose phase pair
# equidistributes over SO(3). Held as reciprocals, and multiplied rather than divided by, so the
# candidate set is bit-comparable with ``igl::super_fibonacci``'s.
TWO_PI_F64 = wp.constant(wp.float64(2.0 * math.pi))
SUPER_FIBONACCI_RSQRT2 = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
SUPER_FIBONACCI_RPSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))


@wp.kernel
def aabb_corners(points: wp.array[wp.vec3], out_corners: wp.array[wp.float32]) -> None:
    # Both corners of the axis-aligned bounding box, in one launch into one buffer.
    #
    # ``out_corners`` is ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]``, *negating* the upper
    # corner so a single ``wp.full(6, inf)`` initializes both and every update is an
    # ``atomic_min``. The alternative — separate min and max buffers — needs two allocations, two
    # fills and two readbacks, and at this size the reduction is entirely host-latency-bound.
    #
    # One thread per ``TILE_1D`` points, so the atomics see a few hundred contenders per address
    # rather than one per point.
    chunk = int(wp.tid())
    offset = chunk * TILE_1D
    remaining = points.shape[0] - offset
    if remaining <= 0:
        return
    count = wp.min(remaining, TILE_1D)

    lower = points[offset]
    upper = points[offset]
    for k in range(1, count):
        p = points[offset + k]
        lower = wp.min(lower, p)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, p)

    for c in range(3):
        wp.atomic_min(out_corners, c, lower[c])
        wp.atomic_min(out_corners, 3 + c, -upper[c])


@wp.kernel
def oriented_box_candidate_axes(n_rotations: wp.int32, out_axes: wp.array[wp.mat33]) -> None:
    # Candidate box orientations, as world -> box frames whose *rows* are the box axes.
    #
    # The set is the Super-Fibonacci spiral [Alexa 2022] — the same low-discrepancy sampling of
    # SO(3) ``igl::oriented_bounding_box`` searches — with the identity as the **last** candidate,
    # so a returned box can never be worse than the axis-aligned one and ``n_rotations = 1`` reduces
    # to exactly ``aabb_corners``.
    #
    # The phase math runs in float64 and only the resulting quaternion is narrowed. The arguments
    # reach ``2 * pi * n_rotations`` (~6e4 radians at igl's default), where float32 argument
    # reduction has already lost four digits of the angle and the low-discrepancy property with it.
    i = int(wp.tid())
    n_spiral = n_rotations - 1
    if i >= n_spiral:
        out_axes[i] = wp.identity(n=3, dtype=wp.float32)
        return

    s = wp.float64(i) + wp.float64(0.5)
    phase = TWO_PI_F64 * s
    alpha = phase * SUPER_FIBONACCI_RSQRT2
    beta = phase * SUPER_FIBONACCI_RPSI
    height = s / wp.float64(n_spiral)
    radius = wp.sqrt(height)
    radius_conjugate = wp.sqrt(wp.float64(1.0) - height)
    rotation = wp.quat_to_matrix(
        wp.quat(
            wp.float32(radius * wp.sin(alpha)),
            wp.float32(radius * wp.cos(alpha)),
            wp.float32(radius_conjugate * wp.sin(beta)),
            wp.float32(radius_conjugate * wp.cos(beta)),
        )
    )
    # The quaternion names a box -> world rotation; the extent reduction wants world -> box.
    out_axes[i] = wp.transpose(rotation)


@wp.kernel
def oriented_box_extents(
    points: wp.array[wp.vec3],
    axes: wp.array[wp.mat33],
    n_slices: wp.int32,
    out_corners: wp.array[wp.float32],
) -> None:
    # Extent of the cloud in every candidate frame: six slots per candidate, packed
    # ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]`` exactly as ``aabb_corners`` packs its one
    # box, so a single ``wp.full(inf)`` seeds both ends and every update is an ``atomic_min``.
    #
    # Strided slice rather than a contiguous chunk, and lane-free, for the same two reasons as
    # ``kernels/convex.hull_support_extremes``: consecutive threads read consecutive points so the
    # loads coalesce, and ``wp.launch_tiled`` runs one lane per block on Warp 1.15's CPU device,
    # which silently reduces a single point per tile.
    k, j = wp.tid()
    n_points = int(points.shape[0])
    frame = axes[int(k)]

    lower = wp.vec3(FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT)
    upper = wp.vec3(-FLOAT32_INF_CONSTANT, -FLOAT32_INF_CONSTANT, -FLOAT32_INF_CONSTANT)
    for i in range(int(j), n_points, int(n_slices)):
        local = frame * points[i]
        lower = wp.min(lower, local)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, local)

    # A slice past the end of the cloud contributes nothing.
    if upper[0] > -FLOAT32_INF_CONSTANT:
        base = int(k) * 6
        for c in range(3):
            wp.atomic_min(out_corners, base + c, lower[c])
            wp.atomic_min(out_corners, base + 3 + c, -upper[c])
