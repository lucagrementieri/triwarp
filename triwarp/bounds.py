"""Bounding boxes: the axis-aligned box, its diagonal and union, and the oriented box."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels import bounds as kernel_bounds

# Points reduced per thread by the per-candidate extent reduction in
# [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]. Long, and one value for both
# devices, because the *candidate* dimension already fills the device -- this is the per-query
# regime [`ITEMS_PER_SLICE_CUDA`][triwarp.constants.ITEMS_PER_SLICE_CUDA]'s note describes, not the
# global-reduction one.
#
# Swept 32-8192 over three regimes (bunny's 35 947 points at 1 024 and at 10 000 rotations, and a
# 2 000-point cloud at 1 024). The curve is shallow and its optimum drifts with the work: the short
# end pays atomic contention (32 gives 1 124 slices and costs 1.4-2.7x), the long end serializes the
# point loop (8192 costs 1.6-4.6x), and in between everything from 256 to 1024 sits within 1.25x of
# the best reading in every regime. Larger candidate counts want longer slices, small clouds shorter
# ones; 256 is the compromise, and its worst case is the 10 000-rotation row at 1.24x.
ITEMS_PER_CANDIDATE_SLICE = 256


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


def oriented_bounding_box(
    points: wp.array[wp.vec3],
    rotations: int = 4096,
    objective: Literal["volume", "surface_area", "diagonal"] = "volume",
) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """
    Smallest bounding box over a sampled set of orientations, and its frame.

    The box is searched, not solved: ``rotations`` candidate orientations are scored by the extent
    of the rotated cloud and the best one is returned. The candidate set is the **Super-Fibonacci
    spiral** [Alexa 2022], a low-discrepancy sampling of ``SO(3)``, with the identity appended as
    the last candidate -- so the answer is never worse than
    [`aabb_bounds`][triwarp.bounds.aabb_bounds], and ``rotations=1`` reproduces it exactly. That is
    the same algorithm and the same candidate set ``igl.oriented_bounding_box`` uses, which makes
    the two directly comparable; unlike igl's, the search over candidates runs in parallel.

    Cost is ``rotations * len(points)`` point transforms, so both factors matter; pass the convex
    hull rather than a dense cloud when one is at hand. How much the extra candidates buy depends on
    the *shape* rather than on the count alone -- see the note below -- so the guidance is to raise
    ``rotations`` for polyhedral input and leave it alone for smooth surfaces.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    rotations
        Number of candidate orientations to score, ``>= 1``. The default is 1.75x cheaper than igl's
        10 000 (0.56 ms against 0.99 on ``bunny``) and halves the excess volume of a 1 024-candidate
        search on every fixture measured.
    objective
        Which box quantity to minimize: ``"volume"``, ``"surface_area"``, or ``"diagonal"`` (the
        squared diagonal length, which has the same minimizer as the length).

    Returns
    -------
    rotation : wp.mat33
        World-to-box frame, i.e. its **rows** are the box axes in world coordinates. A world point
        ``p`` has box coordinates ``rotation * p``, and a box corner maps back with
        ``wp.transpose(rotation) * corner``. ``rotation`` is a proper rotation (orthonormal,
        determinant ``+1``); ``igl.oriented_bounding_box`` returns its transpose, since igl applies
        the matrix on the right of a row vector.
    min_bound, max_bound
        Extent of ``points`` along the box axes, in box coordinates -- so the box is
        ``{transpose(rotation) * q : min_bound <= q <= max_bound}`` and its side lengths are
        ``max_bound - min_bound``. For ``n == 0`` these are ``(+inf, …)`` and ``(-inf, …)`` and
        ``rotation`` is the identity, matching
        [`aabb_bounds`][triwarp.bounds.aabb_bounds].

    Raises
    ------
    ValueError
        If ``rotations < 1``, or ``objective`` is not one of the three named above.

    Notes
    -----
    Two host readbacks: the ``(rotations, 6)`` extent table, whose objective and ``argmin`` are
    ``O(rotations)`` host arithmetic over a buffer far too small to be worth a device pass, and the
    36 bytes of the winning frame. Everything else -- the candidate frames and the extent reduction
    -- stays on the device, which is why the candidate quaternions are generated in a kernel rather
    than with NumPy: at igl's 10 000 rotations the host-side spiral alone measures 1.1 ms, more than
    the entire call costs on a 35k-point cloud.

    A sampled box is not the minimum-volume box, and how far off it is depends on the shape and not
    only on ``rotations``: a smooth surface's optimum is a broad basin in ``SO(3)`` where a
    polyhedron's is an isolated point. Measured against ``trimesh.bounds.oriented_bounds`` on
    anisotropically stretched, tilted fixtures, this returns 1.3% / 0.3% / 0.1% more volume than
    trimesh on a subdivided sphere at 512 / 4 096 / 32 768 candidates, but 35% / 13% / 5.2% more on
    a **cube**, whose four-fold symmetry leaves a single exact orientation to hit.

    Neither library bounds the other, so ``trimesh.bounds.oriented_bounds`` is not an oracle for the
    minimum: on a stretched icosahedron this returns 1.2% *less* volume than trimesh at 32 768
    candidates, because trimesh searches only orientations flush with a convex-hull face and that
    restriction can miss the optimum (locally refining the frame this returns reaches 2.1% below
    trimesh's).

    See Also
    --------
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    [`trimesh.bounds.oriented_bounds`][]
    ``igl.oriented_bounding_box``
    """
    if rotations < 1:
        raise ValueError(f"rotations must be >= 1, got {rotations}")
    if objective not in ("volume", "surface_area", "diagonal"):
        raise ValueError(
            f'objective must be "volume", "surface_area" or "diagonal", got {objective!r}'
        )

    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    n = int(points.shape[0])
    if n == 0:
        return (
            identity,
            wp.vec3(math.inf, math.inf, math.inf),
            wp.vec3(-math.inf, -math.inf, -math.inf),
        )

    device = points.device
    axes = wp.empty(rotations, dtype=wp.mat33, device=device)
    wp.launch(
        kernel_bounds.oriented_box_candidate_axes,
        dim=rotations,
        inputs=[rotations, axes],
        device=device,
    )

    n_slices = max(1, (n + ITEMS_PER_CANDIDATE_SLICE - 1) // ITEMS_PER_CANDIDATE_SLICE)
    corners = wp.full(6 * rotations, math.inf, dtype=wp.float32, device=device)
    wp.launch(
        kernel_bounds.oriented_box_extents,
        dim=(rotations, n_slices),
        inputs=[points, axes, n_slices, corners],
        device=device,
    )

    # Readback 1 of 2: the extent table. Scoring and the argmin are O(rotations) on at most a few
    # hundred kilobytes, so a device pass would cost more in launches than the arithmetic is worth.
    corners_np = corners.numpy().reshape(rotations, 6)
    # Slots 3..5 hold the *negated* upper corner; see the kernel.
    lower_np, upper_np = corners_np[:, :3], -corners_np[:, 3:]
    sides_np = upper_np - lower_np
    if objective == "volume":
        loss_np = sides_np.prod(axis=1)
    elif objective == "surface_area":
        rolled_np = np.roll(sides_np, 1, axis=1)
        loss_np = 2.0 * (sides_np * rolled_np).sum(axis=1)
    else:
        loss_np = np.square(sides_np).sum(axis=1)
    best = int(loss_np.argmin())

    # Readback 2 of 2: the winning frame alone, 36 bytes off a contiguous one-element slice.
    rotation_np = axes[best : best + 1].numpy()[0]
    return (
        wp.mat33(*rotation_np.ravel().tolist()),
        wp.vec3(*lower_np[best].tolist()),
        wp.vec3(*upper_np[best].tolist()),
    )
