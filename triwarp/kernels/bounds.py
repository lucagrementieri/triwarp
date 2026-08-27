import math

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT

# Super-Fibonacci spiral constants [Alexa 2022]: the two irrational strides whose phase pair
# equidistributes over SO(3). Held as reciprocals, and multiplied rather than divided by, so the
# candidate set is bit-comparable with ``igl::super_fibonacci``'s.
TWO_PI_F64 = wp.constant(wp.float64(2.0 * math.pi))
SUPER_FIBONACCI_RSQRT2 = wp.constant(wp.float64(1.0 / math.sqrt(2.0)))
SUPER_FIBONACCI_RPSI = wp.constant(wp.float64(1.0 / 1.533751168755204288118041))


@wp.kernel
def oriented_box_candidate_axes(n_rotations: wp.int32, out_axes: wp.array[wp.mat33]) -> None:
    # Candidate box orientations, as world -> box frames whose *rows* are the box axes.
    #
    # The set is the Super-Fibonacci spiral [Alexa 2022] — the same low-discrepancy sampling of
    # SO(3) ``igl::oriented_bounding_box`` searches — with the identity as the **last** candidate,
    # so a returned box can never be worse than the axis-aligned one and ``n_rotations = 1`` reduces
    # to exactly the axis-aligned reduction.
    #
    # The phase math runs in float64 and only the resulting quaternion is narrowed. The arguments
    # reach ``2 * pi * n_rotations`` (~6e4 radians at igl's default), where float32 argument
    # reduction has already lost four digits of the angle and the low-discrepancy property with it.
    i = wp.int32(wp.tid())
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
def oriented_box_refine_axes(
    chains: wp.array[wp.mat33],
    angle_scale: wp.float64,
    count_per_chain: wp.int32,
    out_axes: wp.array[wp.mat33],
) -> None:
    # One trust-region ball of perturbed frames per chain: the Super-Fibonacci sample of SO(3),
    # geodesically shrunk toward the identity (each rotation angle scaled by ``sigma / pi``),
    # composed onto the chain's base frame in place of the host einsum + 18 KB upload the
    # refinement loop used to pay per round. Same float64 phase math as
    # ``oriented_box_candidate_axes`` above; the last delta of every chain is the identity, which
    # re-scores the base and keeps each chain monotone.
    i = wp.int32(wp.tid())
    count = count_per_chain
    chain = i // count
    p = i - chain * count
    base = chains[chain]
    if p == count - 1:
        out_axes[i] = base
        return

    s = wp.float64(p) + wp.float64(0.5)
    phase = TWO_PI_F64 * s
    alpha = phase * SUPER_FIBONACCI_RSQRT2
    beta = phase * SUPER_FIBONACCI_RPSI
    height = s / wp.float64(count - 1)
    radius = wp.sqrt(height)
    radius_conjugate = wp.sqrt(wp.float64(1.0) - height)
    qx = radius * wp.sin(alpha)
    qy = radius * wp.cos(alpha)
    qz = radius_conjugate * wp.sin(beta)
    qw = radius_conjugate * wp.cos(beta)
    if qw < wp.float64(0.0):  # same rotation, angle in [0, pi]
        qx = -qx
        qy = -qy
        qz = -qz
        qw = -qw
    angle = wp.float64(2.0) * wp.acos(qw)
    axis_norm = wp.sqrt(qx * qx + qy * qy + qz * qz)
    ax = wp.float64(1.0)
    ay = wp.float64(0.0)
    az = wp.float64(0.0)
    if axis_norm > wp.float64(1e-12):
        ax = qx / axis_norm
        ay = qy / axis_norm
        az = qz / axis_norm
    shrunk_half = wp.float64(0.5) * angle * angle_scale
    sin_half = wp.sin(shrunk_half)
    delta = wp.quat_to_matrix(
        wp.quat(
            wp.float32(ax * sin_half),
            wp.float32(ay * sin_half),
            wp.float32(az * sin_half),
            wp.float32(wp.cos(shrunk_half)),
        )
    )
    out_axes[i] = delta * base


@wp.kernel
def oriented_box_extents(
    points: wp.array[wp.vec3],
    axes: wp.array[wp.mat33],
    n_slices: wp.int32,
    out_corners: wp.array[wp.float32],
) -> None:
    # Extent of the cloud in every candidate frame: six slots per candidate, packed
    # ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]`` exactly as ``kernels/reduce.py``'s
    # ``minmax_vec3_chunked`` packs its one box, so a single ``wp.full(inf)`` seeds both ends
    # and every update is an ``atomic_min``.
    #
    # Strided slice rather than a contiguous chunk, and lane-free, for the same two reasons as
    # ``kernels/points.py::hull_support_extremes``: consecutive threads read consecutive points so
    # the loads coalesce, and the threads partition the **outer** work -- the cloud -- rather than a
    # sequence one block owns, so there is no ``wp.block_dim()`` to stride by and a
    # ``wp.tile(...)`` reduction cannot be reached without changing the launch. See
    # ``.claude/CLAUDE.md`` section 3 for the rule and ``kernels/visibility.py::obscurance`` for a
    # lane-parallel kernel on the other side of it.
    k, j = wp.tid()
    n_points = points.shape[0]
    frame = axes[k]

    lower = wp.vec3(FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT, FLOAT32_INF_CONSTANT)
    upper = wp.vec3(-FLOAT32_INF_CONSTANT, -FLOAT32_INF_CONSTANT, -FLOAT32_INF_CONSTANT)
    for i in range(j, n_points, n_slices):
        local = frame * points[i]
        lower = wp.min(lower, local)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, local)

    # A slice past the end of the cloud contributes nothing.
    if upper[0] > -FLOAT32_INF_CONSTANT:
        base = k * 6
        for c in range(3):
            wp.atomic_min(out_corners, base + c, lower[c])
            wp.atomic_min(out_corners, base + 3 + c, -upper[c])


@wp.kernel
def packed_box_diagonals(corners: wp.array[wp.float32], out_diagonal: wp.array[wp.float32]) -> None:
    # Decode one axis-aligned box per six ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]`` slots
    # into its diagonal length. The shared reader for that packing, which
    # [`oriented_box_extents`][triwarp.kernels.bounds.oriented_box_extents] and
    # [`scatter_group_bounds`][triwarp.kernels.scatter.scatter_group_bounds] both write; launch over
    # the box count.
    #
    # A box nothing accumulated into still holds the ``+inf`` seed in both halves, so its extent
    # comes out negative. That reads as **zero** rather than as ``nan``, which is what lets a caller
    # threshold the whole array uniformly instead of masking the empty slots first.
    box = wp.int32(wp.tid())
    base = box * 6
    extent = wp.vec3(
        -corners[base + 3] - corners[base],
        -corners[base + 4] - corners[base + 1],
        -corners[base + 5] - corners[base + 2],
    )
    if extent[0] < wp.float32(0.0):
        out_diagonal[box] = wp.float32(0.0)
    else:
        out_diagonal[box] = wp.length(extent)
