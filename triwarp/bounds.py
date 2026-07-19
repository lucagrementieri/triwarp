"""Axis-aligned bounding boxes: query, diagonal, and union."""

from __future__ import annotations

import math

import warp as wp

import triwarp as tw
import triwarp.typing as twt


def aabb_bounds(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Axis-aligned bounding box of ``points`` (component-wise min / max).

    The reduction runs on ``points.device`` in ``float32`` as a tiled per-column min/max
    over a zero-copy ``(n, 3)`` scalar view of the ``wp.vec3`` buffer (see
    [`minmax`][triwarp.reduce.minmax]).

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``. Must be contiguous (any freshly allocated or
        ``wp.Mesh``-owned buffer is).

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
    if int(points.shape[0]) == 0:
        return (wp.vec3(math.inf, math.inf, math.inf), wp.vec3(-math.inf, -math.inf, -math.inf))
    components = twt.as_array2d_float32(points.view(wp.float32))
    min_wp, max_wp = tw.reduce.minmax(components, axis=0)
    min_np = min_wp.numpy()
    max_np = max_wp.numpy()
    return (
        wp.vec3(float(min_np[0]), float(min_np[1]), float(min_np[2])),
        wp.vec3(float(max_np[0]), float(max_np[1]), float(max_np[2])),
    )


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
