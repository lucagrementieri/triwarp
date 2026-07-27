"""Axis-aligned bounding boxes: query, diagonal, and union."""

from __future__ import annotations

import math

import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels import bounds as kernel_bounds


def aabb_bounds(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Axis-aligned bounding box of ``points`` (component-wise min / max).

    The reduction runs on ``points.device`` in ``float32``: one chunked kernel writes both
    corners into a single six-element buffer, which is then read back once. That is deliberately
    *not* the generic [`minmax`][triwarp.reduce.minmax] path — this is called on the hot path of
    every k-NN query, where it is entirely host-latency-bound, and ``minmax`` needs two
    allocations, two fills and two readbacks for the same answer.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(min_bound, max_bound)`` with ``min_bound[i] ≤ p[i] ≤ max_bound[i]`` for every
        point ``p`` and axis ``i``. If ``n == 0``, ``min_bound`` is ``(+inf, …)`` and
        ``max_bound`` is ``(-inf, …)``.

    See Also
    --------
    [`aabb_diagonal`][triwarp.bounds.aabb_diagonal]
    [`aabb_union`][triwarp.bounds.aabb_union]
    """
    n = int(points.shape[0])
    if n == 0:
        return (wp.vec3(math.inf, math.inf, math.inf), wp.vec3(-math.inf, -math.inf, -math.inf))
    # One allocation, one launch, one readback. This is a pure reduction on the hot path of every
    # k-NN query, so it is entirely host-latency-bound at any realistic size — the generic
    # [`minmax`][triwarp.reduce.minmax] path costs two allocations, two fills and two readbacks
    # for the same answer, which measured ~2x slower.
    corners = wp.full(6, math.inf, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_bounds.aabb_corners,
        dim=(n + TILE_1D - 1) // TILE_1D,
        inputs=[points, corners],
        device=points.device,
    )
    corners_np = corners.numpy()
    # Slots 3..5 hold the *negated* upper corner; see the kernel.
    return (wp.vec3(*corners_np[:3].tolist()), wp.vec3(*(-corners_np[3:]).tolist()))


def aabb_diagonal(min_bound: wp.vec3, max_bound: wp.vec3) -> float:
    """
    Diagonal length of an axis-aligned bounding box.

    A standard scale heuristic used to derive query radii and search-distance defaults from a
    mesh or point cloud's extent.

    Parameters
    ----------
    min_bound
        Minimum corner, as returned by [`aabb_bounds`][triwarp.bounds.aabb_bounds].
    max_bound
        Maximum corner, as returned by [`aabb_bounds`][triwarp.bounds.aabb_bounds].

    Returns
    -------
    float
        ``|max_bound - min_bound|``.

    See Also
    --------
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    [`aabb_union`][triwarp.bounds.aabb_union]
    """
    return float(wp.length(max_bound - min_bound))


def aabb_union(
    a_min: wp.vec3, a_max: wp.vec3, b_min: wp.vec3, b_max: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """
    Smallest axis-aligned bounding box enclosing two given boxes.

    Parameters
    ----------
    a_min, a_max
        Min/max corners of the first box.
    b_min, b_max
        Min/max corners of the second box.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(min_bound, max_bound)`` of the box enclosing both inputs.

    See Also
    --------
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    [`aabb_diagonal`][triwarp.bounds.aabb_diagonal]
    """
    combined_min = wp.vec3(
        min(a_min[0], b_min[0]), min(a_min[1], b_min[1]), min(a_min[2], b_min[2])
    )
    combined_max = wp.vec3(
        max(a_max[0], b_max[0]), max(a_max[1], b_max[1]), max(a_max[2], b_max[2])
    )
    return combined_min, combined_max
