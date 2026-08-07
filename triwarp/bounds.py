"""
Bounding boxes: the axis-aligned box, its diagonal and union, and the oriented box.

[`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal] is the one entry point here that
exists for another module's sake rather than its own: it is the default search radius every
mesh query in the package derives from its input's extent.
"""

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


def enclosing_diagonal(points: wp.array[wp.vec3], other: wp.array[wp.vec3] | None = None) -> float:
    """
    Diagonal of the axis-aligned box enclosing one or two point sets.

    The default search radius for every mesh query in the package: no closest point, ray hit or
    tangent sphere can be further away than the diagonal of a box containing both the surface and
    the query points, so it is the smallest bound that is guaranteed not to cut an answer off. Pass
    ``other`` whenever the queries may lie outside the mesh's own box, which is the usual case --
    with ``points`` alone a query far outside would get a radius too short to reach the surface.

    Parameters
    ----------
    points
        ``(n,)`` positions as ``wp.vec3``, typically a mesh's vertices.
    other
        Optional second ``(m,)`` set, typically the query points. ``None`` or empty measures
        ``points`` alone.

    Returns
    -------
    float
        ``|max_bound - min_bound|`` of the union box. ``inf`` when ``points`` is empty, since an
        empty box has infinite negative extent -- callers guard on the point count first.

    See Also
    --------
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    [`aabb_union`][triwarp.bounds.aabb_union]
    [`aabb_diagonal`][triwarp.bounds.aabb_diagonal]
        The same length from corners the caller already holds.
    """
    points_min, points_max = aabb_bounds(points)
    if other is None or int(other.shape[0]) == 0:
        return aabb_diagonal(points_min, points_max)
    other_min, other_max = aabb_bounds(other)
    union_min, union_max = aabb_union(points_min, points_max, other_min, other_max)
    return aabb_diagonal(union_min, union_max)


def oriented_bounding_box(
    points: wp.array[wp.vec3],
    rotations: int = 4096,
    objective: Literal["volume", "surface_area", "diagonal"] = "volume",
    refine_iterations: int = 8,
) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """
    Smallest bounding box over sampled orientations, sharpened by local refinement, and its frame.

    The box is searched, not solved, in two phases. The **global phase** scores ``rotations``
    candidate orientations from the **Super-Fibonacci spiral** [Alexa 2022], a low-discrepancy
    sampling of ``SO(3)``, with the identity appended as the last candidate -- the same candidate
    set ``igl.oriented_bounding_box`` searches, scored in parallel. The **refinement phase** then
    takes the best-scoring candidates from up to four mutually distant basins and runs
    ``refine_iterations`` trust-region rounds around each: every round scores a low-discrepancy
    ball of perturbed frames whose angular radius starts at the global grid's covering radius and
    halves per round, keeping each chain's best. Refinement is monotone (each chain re-scores its
    own base), so the answer is never worse than the global phase's and never worse than
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]; with ``refine_iterations=0`` and ``rotations=1``
    it reproduces the axis-aligned box exactly.

    Cost is ``rotations * len(points)`` point transforms for the global phase plus
    ``refine_iterations * 512 * len(points)`` for refinement; pass the convex hull rather than a
    dense cloud when one is at hand. Refinement is what makes the *default* polyhedron-safe: it
    recovers the flat-flush orientation a sampled grid can only land near (measured on a tilted
    cube whose exact minimum is 6.0: 6.31 sampled at 32 768 candidates against **6.0013** refined
    at the default 4 096 — from 5.2% over the true box to 0.02%).

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    rotations
        Number of global candidates to score, ``>= 1``. The refinement default makes raising this
        past a few thousand pointless: the global phase only needs to land *inside* the optimum's
        basin, and refinement does the rest.
    objective
        Which box quantity to minimize: ``"volume"``, ``"surface_area"``, or ``"diagonal"`` (the
        squared diagonal length, which has the same minimizer as the length).
    refine_iterations
        Trust-region rounds after the global phase, ``>= 0``. ``0`` disables refinement and returns
        the pure sampled answer, which is what makes the result element-wise comparable to
        ``igl.oriented_bounding_box``'s identical candidate set (the parity tests pass ``0`` for
        exactly that reason). Each round costs one 512-frame launch and one small readback --
        measured back to back on a 36k cloud, 0.45 ms sampled against 3.3-5.8 ms at the default
        eight rounds, i.e. ~0.4-0.7 ms of host-device latency per round; the quality gain past
        eight rounds is under 0.05% on every fixture measured.

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
        If ``rotations < 1``, ``refine_iterations < 0``, or ``objective`` is not one of the three
        named above.

    Notes
    -----
    Host traffic: the ``(rotations, 6)`` extent table (its objective and ``argmin`` are
    ``O(rotations)`` host arithmetic over a buffer far too small to be worth a device pass), the
    36 bytes of the winning frame, and one ``(512, 6)`` table per refinement round. The refinement
    frames are composed on the host -- 512 small matrix products per round is microseconds -- and
    uploaded, so the extent kernel is the only device work either phase does.

    **The result is a converged local minimum, not a certified global one.** Certifying the true
    minimum-volume box requires the exact-arithmetic search over the convex hull's face and edge
    events (O'Rourke's rotating-calipers family -- what ``trimesh.bounds.oriented_bounds`` and
    open3d's ``get_minimal_oriented_bounding_box`` approximate over qhull output), and triwarp
    deliberately has no exact convex hull, on the GPU or off it. What the sampling-plus-refinement
    search gives up is the *certificate*, not (measurably) the volume: the global phase lands in
    the optimum's basin whenever the grid's covering radius resolves it, refinement converges
    within the basin, and against both hull-based references on every fixture probed
    (tilted/stretched icosahedron, cube shell, hemisphere, half torus) the refined default **ties
    or beats each of them** -- including the exact 6.0 on the cube that sampling alone misses by
    5.2%. A pathological cloud whose optimum basin is narrower than the covering radius of
    ``rotations`` samples can still hide its box from this search; raise ``rotations`` if the
    input is a near-symmetric polyhedron far from any sampled orientation.

    Neither hull-based reference is an oracle for the minimum either: on a stretched icosahedron
    the refined search returns **less** volume than both (the hull-face-flush restriction misses
    optima whose box touches only edges and vertices), which is why the regression tests compare
    within measured bands rather than one-sidedly.

    See Also
    --------
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    [`trimesh.bounds.oriented_bounds`][]
    ``igl.oriented_bounding_box``
    """
    if rotations < 1:
        raise ValueError(f"rotations must be >= 1, got {rotations}")
    if refine_iterations < 0:
        raise ValueError(f"refine_iterations must be >= 0, got {refine_iterations}")
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
    lower_np, upper_np = _scored_extents(points, axes, n_slices)
    loss_np = _objective_losses(upper_np - lower_np, objective)
    best = int(loss_np.argmin())

    if refine_iterations == 0:
        # The winning frame alone, 36 bytes off a contiguous one-element slice, so the pure
        # sampled path stays bit-comparable with the device-generated candidate set.
        rotation_np = axes[best : best + 1].numpy()[0]
        return (
            wp.mat33(*rotation_np.ravel().tolist()),
            wp.vec3(*lower_np[best].tolist()),
            wp.vec3(*upper_np[best].tolist()),
        )

    frame_np, lower_best, upper_best = _refine_box(
        points, rotations, objective, refine_iterations, loss_np, n_slices
    )
    return (
        wp.mat33(*frame_np.ravel().tolist()),
        wp.vec3(*lower_best.tolist()),
        wp.vec3(*upper_best.tolist()),
    )


# Refinement geometry: up to four chains cover distinct basins of the sampled landscape (a
# near-symmetric shape has several near-tied optima), and 128 frames per chain per round keeps a
# full refinement at the cost of one extra 512-candidate global pass per round.
_REFINE_CHAINS = 4
_REFINE_CANDIDATES = 128


def _refine_box(
    points: wp.array[wp.vec3],
    rotations: int,
    objective: str,
    refine_iterations: int,
    loss_np: np.ndarray,
    n_slices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trust-region refinement of the sampled winner(s): shrink a low-discrepancy ball per round.

    Chains start from the best sampled candidates of mutually distant basins (greedy spread over
    the top of the loss table). Each round composes ``delta @ base`` perturbations on the host --
    512 small matrix products, microseconds -- scores them with the same extent kernel as the
    global phase, and keeps each chain's best; the last delta is the identity, which is what makes
    the refinement monotone per chain.
    """
    device = points.device
    order = np.argsort(loss_np)
    chains_np = _spread_chain_frames(order, rotations)
    n_chains = chains_np.shape[0]
    chain_loss = np.full(n_chains, np.inf)
    chain_lower = np.zeros((n_chains, 3))
    chain_upper = np.zeros((n_chains, 3))

    # Start at the covering radius of the global grid: the sampled winner is at most about this
    # far from its basin's optimum, and each round halves the radius.
    sigma = 2.0 * (math.pi**2 / max(rotations, 2)) ** (1.0 / 3.0)
    total = n_chains * _REFINE_CANDIDATES
    for _ in range(refine_iterations):
        deltas_np = _ball_rotations(_REFINE_CANDIDATES, sigma)
        axes_np = np.einsum("pij,cjk->cpik", deltas_np, chains_np).reshape(total, 3, 3)
        axes = wp.array(
            np.ascontiguousarray(axes_np, dtype=np.float32), dtype=wp.mat33, device=device
        )
        lower_np, upper_np = _scored_extents(points, axes, n_slices)
        round_loss = _objective_losses(upper_np - lower_np, objective).reshape(
            n_chains, _REFINE_CANDIDATES
        )
        for chain in range(n_chains):
            best = int(round_loss[chain].argmin())
            if round_loss[chain, best] < chain_loss[chain]:
                chain_loss[chain] = round_loss[chain, best]
                row = chain * _REFINE_CANDIDATES + best
                chains_np[chain] = axes_np[row]
                chain_lower[chain] = lower_np[row]
                chain_upper[chain] = upper_np[row]
        # 0.4 rather than 0.5: a flat-flush optimum is a *kink*, so the volume error is linear in
        # the final angular resolution. Swept at eight rounds on the four tilted fixtures: 0.4
        # reads 6.0013 on the cube shell against 0.5's 6.0040 (exact minimum 6.0) and 975.29 on
        # the half torus against 979.73, at identical cost; the 127-frame ball resolves ~sigma/5
        # per round, so shrinking by 0.4 never outruns what a round can see.
        sigma *= 0.4

    winner = int(chain_loss.argmin())
    return chains_np[winner], chain_lower[winner], chain_upper[winner]


def _scored_extents(
    points: wp.array[wp.vec3], axes: wp.array, n_slices: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extent of the cloud in every candidate frame, read back as ``(lower, upper)`` tables."""
    n_axes = int(axes.shape[0])
    corners = wp.full(6 * n_axes, math.inf, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_bounds.oriented_box_extents,
        dim=(n_axes, n_slices),
        inputs=[points, axes, n_slices, corners],
        device=points.device,
    )
    # One readback per scored batch; the objective and argmin are O(n_axes) host arithmetic over a
    # buffer far too small to be worth a device pass. Slots 3..5 hold the *negated* upper corner;
    # see the kernel.
    corners_np = corners.numpy().reshape(n_axes, 6)
    return corners_np[:, :3], -corners_np[:, 3:]


def _objective_losses(sides_np: np.ndarray, objective: str) -> np.ndarray:
    """Score each candidate's box sides under the requested objective."""
    if objective == "volume":
        return sides_np.prod(axis=1)
    if objective == "surface_area":
        return 2.0 * (sides_np * np.roll(sides_np, 1, axis=1)).sum(axis=1)
    return np.square(sides_np).sum(axis=1)


def _spread_chain_frames(order: np.ndarray, rotations: int) -> np.ndarray:
    """
    Pick the refinement chains' starting frames: the loss table's head, greedily spread.

    Adjacent spiral candidates score adjacently, so the top of the table is usually one basin
    sampled several times; the greedy pass keeps only heads at least 0.2 rad apart so the chains
    explore *different* basins. Falls back to the plain head when the spread exhausts the window.
    """
    head = order[: min(32, rotations)]
    head_frames = _spiral_frames(head, rotations)
    picked = [0]
    for i in range(1, head_frames.shape[0]):
        if len(picked) == _REFINE_CHAINS:
            break
        candidate = head_frames[i]
        # Geodesic distance via the relative rotation's angle.
        far = True
        for j in picked:
            relative = candidate @ head_frames[j].T
            angle = math.acos(min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) * 0.5)))
            if angle < 0.2:
                far = False
                break
        if far:
            picked.append(i)
    for i in range(1, head_frames.shape[0]):
        if len(picked) == _REFINE_CHAINS:
            break
        if i not in picked:
            picked.append(i)
    return np.ascontiguousarray(head_frames[picked], dtype=np.float64)


def _spiral_frames(indices: np.ndarray, rotations: int) -> np.ndarray:
    """
    Reproduce ``oriented_box_candidate_axes`` on the host for a handful of indices.

    Same float64 phase math and the same world-to-box transpose as the kernel; recomputing beats
    reading the frames back one 36-byte gather at a time.
    """
    n_spiral = rotations - 1
    quats = np.zeros((len(indices), 4))
    quats[:, 3] = 1.0
    spiral = np.asarray(indices) < n_spiral
    s = np.asarray(indices, dtype=np.float64)[spiral] + 0.5
    phase = 2.0 * math.pi * s
    alpha = phase / math.sqrt(2.0)
    beta = phase / 1.533751168755204288118041
    height = s / float(max(n_spiral, 1))
    radius = np.sqrt(height)
    radius_conjugate = np.sqrt(1.0 - height)
    quats[spiral] = np.stack(
        [
            radius * np.sin(alpha),
            radius * np.cos(alpha),
            radius_conjugate * np.sin(beta),
            radius_conjugate * np.cos(beta),
        ],
        axis=1,
    )
    # World -> box frames: the transpose of the rotation each quaternion names.
    return _quat_matrices(quats).transpose(0, 2, 1)


def _ball_rotations(count: int, sigma: float) -> np.ndarray:
    """
    Low-discrepancy rotations within angular radius ``sigma``, the identity last.

    The Super-Fibonacci sample of the whole group, geodesically shrunk toward the identity: each
    quaternion's rotation angle is rescaled by ``sigma / pi``, which keeps the sample's spread
    while confining it to the trust region.
    """
    s = np.arange(count - 1, dtype=np.float64) + 0.5
    phase = 2.0 * math.pi * s
    alpha = phase / math.sqrt(2.0)
    beta = phase / 1.533751168755204288118041
    height = s / float(count - 1)
    radius = np.sqrt(height)
    radius_conjugate = np.sqrt(1.0 - height)
    quats = np.stack(
        [
            radius * np.sin(alpha),
            radius * np.cos(alpha),
            radius_conjugate * np.sin(beta),
            radius_conjugate * np.cos(beta),
        ],
        axis=1,
    )
    quats[quats[:, 3] < 0.0] *= -1.0  # same rotation, angle in [0, pi]
    angle = 2.0 * np.arccos(np.clip(quats[:, 3], -1.0, 1.0))
    axis_norm = np.linalg.norm(quats[:, :3], axis=1)
    safe = axis_norm > 1e-12
    axes = np.zeros((count - 1, 3))
    axes[safe] = quats[safe, :3] / axis_norm[safe, None]
    axes[~safe, 0] = 1.0
    shrunk_half = 0.5 * angle * (sigma / math.pi)
    shrunk = np.concatenate(
        [axes * np.sin(shrunk_half)[:, None], np.cos(shrunk_half)[:, None]], axis=1
    )
    deltas = np.empty((count, 3, 3))
    deltas[: count - 1] = _quat_matrices(shrunk)
    deltas[count - 1] = np.eye(3)  # re-scores the base: what makes each chain monotone
    return deltas


def _quat_matrices(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices from ``(x, y, z, w)`` quaternions, matching ``wp.quat_to_matrix``."""
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.stack(
        [
            np.stack(
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], 1
            ),
            np.stack(
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], 1
            ),
            np.stack(
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], 1
            ),
        ],
        axis=1,
    )
