import warp as wp

from triwarp.constants import TILE_1D


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
