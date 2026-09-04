"""
Bounding boxes: building them, and selecting or cropping with them.

Two halves. The **constructors** -- [`aabb`][triwarp.bounds.aabb],
[`aabb_union`][triwarp.bounds.aabb_union], [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
and [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] -- reduce a cloud to a box. The
**queries** -- [`points_in_aabb`][triwarp.bounds.points_in_aabb],
[`points_in_obb`][triwarp.bounds.points_in_obb] and the
[`crop_points`][triwarp.bounds.crop_points] / [`crop_mesh`][triwarp.bounds.crop_mesh] pair -- run
the other way, testing points against a box someone already has. The two halves compose directly:
every constructor's return is a query's argument, in that layout, which is why an oriented box is a
``(rotation, min_bound, max_bound)`` triple here rather than an object.

[`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal] is the one entry point here that
exists for another module's sake rather than its own: it is the default search radius every
mesh query in the package derives from its input's extent.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import triwarp as tw
from triwarp._device import read_scalar
from triwarp.kernels import bounds as kernel_bounds
from triwarp.kernels import predicates as kernel_predicates

# Points reduced per thread by the per-candidate extent reduction in
# [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]. Long, and one value for both
# devices, because the *candidate* dimension already fills the device -- this is the per-query
# regime [`ITEMS_PER_SLICE_CUDA`][triwarp.constants.ITEMS_PER_SLICE_CUDA]'s note describes, not the
# global-reduction one.
ITEMS_PER_CANDIDATE_SLICE = 256

# Cloud size at which [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] first runs
# [`convex_superset_mask`][triwarp.points.convex_superset_mask] and searches only the survivors.
# The mask keeps every convex-hull vertex and the extent reduction is decided by hull vertices.
CONVEX_PREFILTER_MIN_POINTS = 100_000


def aabb(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Axis-aligned bounding box of ``points`` (component-wise min / max).

    The reduction is [`minmax`][triwarp.reduce.minmax]'s ``wp.vec3`` path: one chunked kernel
    writes both corners into a single six-element buffer, read back once — it is called on the
    hot path of every k-NN query, where it is entirely host-latency-bound, so nothing beyond
    that single launch and readback is spent. This wrapper contributes only the empty-input
    ``(+inf, -inf)`` convention, where ``minmax`` raises.

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
    [`aabb_union`][triwarp.bounds.aabb_union]
    [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The other direction: which points a box already in hand contains.
    [`triwarp.reduce.minmax`][triwarp.reduce.minmax]
    """
    if int(points.shape[0]) == 0:
        return (wp.vec3(math.inf, math.inf, math.inf), wp.vec3(-math.inf, -math.inf, -math.inf))
    return tw.reduce.minmax(points)


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
    [`aabb`][triwarp.bounds.aabb]
    [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal]
    """
    return wp.min(a_min, b_min), wp.max(a_max, b_max)


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
    [`aabb`][triwarp.bounds.aabb]
    [`aabb_union`][triwarp.bounds.aabb_union]
    """
    lower, upper = aabb(points)
    if other is not None and int(other.shape[0]) > 0:
        other_lower, other_upper = aabb(other)
        lower, upper = aabb_union(lower, upper, other_lower, other_upper)
    return float(wp.length(upper - lower))


def points_in_aabb(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.int32]:
    """
    Return the indices of the points inside an axis-aligned box, boundary included.

    The bounded selection primitive, and the one that needs no spatial index at all: every point is
    tested independently against six planes, so the cost is one pass over the cloud whatever the
    box holds. A tree would only help if the *box* were the thing being searched for -- see
    [`triwarp.neighbors.query_bvh_box`][triwarp.neighbors.query_bvh_box] for that direction.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box, as [`aabb`][triwarp.bounds.aabb] returns them. An empty box
        (any ``min_bound[i] > max_bound[i]``) selects nothing.

    Returns
    -------
    wp.array[wp.int32]
        Ascending indices into ``points`` on ``points.device``. Empty when nothing is inside.

    Examples
    --------
    ```python
    inside = tw.bounds.points_in_aabb(pts, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    ```

    Notes
    -----
    Containment is **inclusive**: a point exactly on a face, edge or corner of the box is selected.
    A point with a ``nan`` coordinate is *not* selected, and neither is one at an infinity. Both
    conventions were measured against the one reference that answers this question rather than
    chosen -- points placed on both corners, on an edge midpoint and on a face centre are all
    selected there, and the three ``nan`` rows are not -- and both differ from
    [`half_space_mask`][triwarp.points.half_space_mask], whose test is strict.

    See Also
    --------
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask]
        The same selection as a boolean mask, for a caller that wants to combine it.
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        The oriented counterpart.
    [`crop_points`][triwarp.bounds.crop_points]
        This plus the gather, when the points themselves are wanted.
    [`aabb`][triwarp.bounds.aabb]
    """
    return tw.array.flatnonzero(points_in_aabb_mask(points, min_bound, max_bound))


def points_in_aabb_mask(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.bool]:
    """
    Flag the points inside an axis-aligned box, boundary included.

    The mask form of [`points_in_aabb`][triwarp.bounds.points_in_aabb], for a caller that wants to
    intersect the selection with another predicate before compacting it -- combining masks costs
    one ``wp.map`` where combining index lists costs a set operation.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box, as [`aabb`][triwarp.bounds.aabb] returns them.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**, the same sense
        as [`triwarp.points.half_space_mask`][triwarp.points.half_space_mask].

    See Also
    --------
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The index form, and where the boundary and ``nan`` conventions are documented.
    [`points_in_obb_mask`][triwarp.bounds.points_in_obb_mask]
    [`triwarp.array.flatnonzero`][triwarp.array.flatnonzero]
    """
    out_mask = wp.empty(int(points.shape[0]), dtype=wp.bool, device=points.device)
    if int(points.shape[0]) == 0:
        return out_mask

    wp.map(kernel_predicates.is_in_aabb, points, min_bound, max_bound, out=out_mask)
    return out_mask


def points_in_obb(
    points: wp.array[wp.vec3], rotation: wp.mat33, min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.int32]:
    """
    Return the indices of the points inside an oriented box, boundary included.

    Takes the box exactly as [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]
    returns it, so the two compose without a transpose or a corner rebuild.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    rotation
        World-to-box frame as ``wp.mat33``, i.e. its **rows** are the box axes in world
        coordinates. A point's box coordinates are ``rotation * p``.
    min_bound, max_bound
        Extent of the box along its own axes, in box coordinates.

    Returns
    -------
    wp.array[wp.int32]
        Ascending indices into ``points`` on ``points.device``. Empty when nothing is inside.

    Examples
    --------
    ```python
    rotation, lower, upper = tw.bounds.oriented_bounding_box(v, 256)
    inside = tw.bounds.points_in_obb(v, rotation, lower, upper)
    ```

    Notes
    -----
    One rotated point per thread, then the identical six comparisons
    [`points_in_aabb`][triwarp.bounds.points_in_aabb] makes -- the two share one predicate, so the
    inclusive boundary and the ``nan`` exclusion documented there hold here too, and the two
    functions cannot drift apart on either. Passing the identity as ``rotation`` reproduces the
    axis-aligned answer exactly (measured, on points sitting on the corners and on ``nan`` rows).

    Being a proper rotation, the frame's inverse is its transpose, so nothing here inverts a
    matrix; a general affine box would, and is not what this takes.

    See Also
    --------
    [`points_in_obb_mask`][triwarp.bounds.points_in_obb_mask]
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The axis-aligned counterpart, and the shared conventions.
    [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]
        Where the ``(rotation, min_bound, max_bound)`` triple comes from.
    """
    return tw.array.flatnonzero(points_in_obb_mask(points, rotation, min_bound, max_bound))


def points_in_obb_mask(
    points: wp.array[wp.vec3], rotation: wp.mat33, min_bound: wp.vec3, max_bound: wp.vec3
) -> wp.array[wp.bool]:
    """
    Flag the points inside an oriented box, boundary included.

    The mask form of [`points_in_obb`][triwarp.bounds.points_in_obb]; see
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask] for why the mask form exists.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes
        ([`points_in_obb`][triwarp.bounds.points_in_obb] documents the convention).
    min_bound, max_bound
        Extent of the box along its own axes, in box coordinates.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**.

    See Also
    --------
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        The index form, and where the conventions are documented.
    [`points_in_aabb_mask`][triwarp.bounds.points_in_aabb_mask]
    """
    out_mask = wp.empty(int(points.shape[0]), dtype=wp.bool, device=points.device)
    if int(points.shape[0]) == 0:
        return out_mask

    wp.map(kernel_predicates.is_in_obb, points, rotation, min_bound, max_bound, out=out_mask)
    return out_mask


def crop_points(
    points: wp.array[wp.vec3],
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    *,
    rotation: wp.mat33 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the points inside a box, returning the survivors and where they came from.

    Parameters
    ----------
    points
        ``(n,)`` positions in world space as ``wp.vec3``.
    min_bound, max_bound
        Opposite corners of the box: in world coordinates when ``rotation`` is ``None``, in box
        coordinates otherwise.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes. ``None`` -- the default --
        crops to the **axis-aligned** box, which is the same test with one transform fewer rather
        than a different rule; pass the frame
        [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] returns to crop to an
        oriented one.

    Returns
    -------
    kept : wp.array[wp.vec3]
        The points inside the box, in ascending index order, on ``points.device``.
    indices : wp.array[wp.int32]
        Their indices in ``points``, so a caller can crop a parallel per-point attribute with
        [`triwarp.array.gather`][triwarp.array.gather] instead of re-running the test. This is the
        second return rather than an option because the compaction has already computed it.

    Examples
    --------
    ```python
    kept, indices = tw.bounds.crop_points(pts, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    ```

    See Also
    --------
    [`crop_mesh`][triwarp.bounds.crop_mesh]
        The mesh counterpart, which must also decide what to do with a straddling face.
    [`points_in_aabb`][triwarp.bounds.points_in_aabb]
        The selection alone, when the points are not needed.
    [`triwarp.array.gather`][triwarp.array.gather]
    """
    indices = tw.array.flatnonzero(_box_mask(points, min_bound, max_bound, rotation))
    return tw.array.gather(points, indices), indices


def crop_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    *,
    rotation: wp.mat33 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the faces whose three vertices all lie inside a box, reindexed from zero.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    min_bound, max_bound
        Opposite corners of the box: in world coordinates when ``rotation`` is ``None``, in box
        coordinates otherwise.
    rotation
        World-to-box frame as ``wp.mat33``, rows being the box axes; ``None`` crops to the
        axis-aligned box. Same argument as
        [`crop_points`][triwarp.bounds.crop_points]'s.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``, carrying only the vertices
        the kept faces reference.

    Examples
    --------
    ```python
    rotation, lower, upper = tw.bounds.oriented_bounding_box(v, 256)
    sub_v, sub_f = tw.bounds.crop_mesh(v, f, lower, upper, rotation=rotation)
    ```

    Notes
    -----
    A face straddling the box boundary is **dropped**, not clipped: the result is a subset of the
    input's triangles, so the crop never introduces a vertex the input did not have and the cut
    edge is ragged at the triangle scale. That is what makes the operation a selection rather than
    a boolean -- for a flush cut, intersect the mesh with the box as a solid instead -- and it is
    the rule the one reference that answers this question applies, verified face for face on four
    fixtures rather than assumed.

    To keep the straddling faces instead, take the mask and pass it on directly, which is the
    ``face_mode="any"`` rule:

    ```python
    mask = tw.bounds.points_in_aabb_mask(v, wp.vec3(-1.0, -1.0, 0.0), wp.vec3(1.0, 1.0, 3.0))
    sub_v, sub_f = tw.selection.submesh_from_vertex_mask(v, f, mask, face_mode="any")
    ```

    See Also
    --------
    [`crop_points`][triwarp.bounds.crop_points]
    [`triwarp.selection.submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]
        The face selection this delegates to, where ``face_mode`` is exposed.
    """
    vertex_mask = _box_mask(vertices, min_bound, max_bound, rotation)
    return tw.selection.submesh_from_vertex_mask(vertices, faces, vertex_mask, face_mode="all")


def _box_mask(
    points: wp.array[wp.vec3], min_bound: wp.vec3, max_bound: wp.vec3, rotation: wp.mat33 | None
) -> wp.array[wp.bool]:
    """Dispatch the two crop entry points onto the axis-aligned or the oriented predicate."""
    if rotation is None:
        return points_in_aabb_mask(points, min_bound, max_bound)
    return points_in_obb_mask(points, rotation, min_bound, max_bound)


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
    sampling of ``SO(3)``, with the identity appended as the last candidate, scored in parallel.
    The **refinement phase** then takes the best-scoring candidates from up to four mutually distant
    basins and runs ``refine_iterations`` trust-region rounds around each: every round scores a
    low-discrepancy ball of perturbed frames whose angular radius starts at the global grid's
    covering radius and halves per round, keeping each chain's best. Refinement is monotone (each
    chain re-scores its own base), so the answer is never worse than the global phase's and never
    worse than [`aabb`][triwarp.bounds.aabb]; with ``refine_iterations=0`` and ``rotations=1`` it
    reproduces the axis-aligned box exactly.

    Cost is ``rotations * len(points)`` point transforms for the global phase plus
    ``refine_iterations * 512 * len(points)`` for refinement — but past
    ``CONVEX_PREFILTER_MIN_POINTS`` a
    [`convex_superset_mask`][triwarp.points.convex_superset_mask] prefilter first drops every
    point that provably cannot touch the box, so both phases run on the few hull-candidate
    survivors and the cost stops growing with the cloud (the box is identical: the mask keeps
    every hull vertex and the extents are order-independent reductions over them).

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
        the pure sampled answer.

    Returns
    -------
    rotation : wp.mat33
        World-to-box frame, i.e. its **rows** are the box axes in world coordinates. A world point
        ``p`` has box coordinates ``rotation * p``, and a box corner maps back with
        ``wp.transpose(rotation) * corner``. ``rotation`` is a proper rotation (orthonormal,
        determinant ``+1``).
    min_bound, max_bound
        Extent of ``points`` along the box axes, in box coordinates -- so the box is
        ``{transpose(rotation) * q : min_bound <= q <= max_bound}`` and its side lengths are
        ``max_bound - min_bound``. For ``n == 0`` these are ``(+inf, …)`` and ``(-inf, …)`` and
        ``rotation`` is the identity, matching
        [`aabb`][triwarp.bounds.aabb].

    Raises
    ------
    ValueError
        If ``rotations < 1``, ``refine_iterations < 0``, or ``objective`` is not one of the three
        named above.

    Notes
    -----
    Host traffic: the ``(rotations, 6)`` extent table (its objective and ``argmin`` are
    ``O(rotations)`` host arithmetic over a buffer far too small to be worth a device pass), the
    36 bytes of the winning frame, and one ``(512, 6)`` table per refinement round. The
    refinement frames are generated and composed **on the device** (the host quaternion math,
    einsum and per-round upload they replace measured 63% of every round), so the chains live on
    the device and the extent tables are the only per-round traffic.

    **The result is a converged local minimum, not a certified global one.** Certifying the true
    minimum-volume box requires the exact-arithmetic search over the convex hull's face and edge
    events (O'Rourke's rotating-calipers family), and triwarp deliberately has no exact convex hull.
    What the sampling-plus-refinement search gives up is the *certificate*, not (measurably) the
    volume: the global phase lands in the optimum's basin whenever the grid's covering radius
    resolves it, refinement converges within the basin, and against both hull-based references on
    every fixture probed (tilted/stretched icosahedron, cube shell, hemisphere, half torus) the
    refined default **ties or beats each of them**.
    A pathological cloud whose optimum basin is narrower than the covering radius of ``rotations``
    samples can still hide its box from this search; raise ``rotations`` if the input is a
    near-symmetric polyhedron far from any sampled orientation.

    Neither hull-based reference is an oracle for the minimum either: on a stretched icosahedron
    the refined search returns **less** volume than both (the hull-face-flush restriction misses
    optima whose box touches only edges and vertices), which is why the regression tests compare
    within measured bands rather than one-sidedly.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
    [`points_in_obb`][triwarp.bounds.points_in_obb]
        Takes this triple as returned, to select the points a box contains.
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

    n = int(points.shape[0])
    if n == 0:
        return (
            wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            wp.vec3(math.inf, math.inf, math.inf),
            wp.vec3(-math.inf, -math.inf, -math.inf),
        )

    if n >= CONVEX_PREFILTER_MIN_POINTS:
        # Identical box, decided at the threshold above: only hull vertices can touch an
        # enclosing box, the mask keeps all of them, and min/max extents do not care about the
        # discarded interior points.
        mask = tw.points.convex_superset_mask(points)
        points = tw.array.gather(points, tw.array.flatnonzero(mask))
        n = int(points.shape[0])

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
        # The winning frame alone, 36 bytes through the shared readback scratch, so the pure
        # sampled path stays bit-comparable with the device-generated candidate set.
        rotation_np = read_scalar(axes, best)
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
    the top of the loss table) and live on the device. The same extent kernel as the global phase
    scores them, the per-chain argmin stays host arithmetic over the already-read-back table, and
    an improved chain is refreshed with a 36-byte device copy; the last delta is the identity,
    which is what makes the refinement monotone per chain.
    """
    device = points.device
    order = np.argsort(loss_np)
    chains_np = _spread_chain_frames(order, rotations)
    n_chains = chains_np.shape[0]
    chain_loss = np.full(n_chains, np.inf)
    chain_lower = np.zeros((n_chains, 3))
    chain_upper = np.zeros((n_chains, 3))

    chains = wp.array(
        np.ascontiguousarray(chains_np, dtype=np.float32), dtype=wp.mat33, device=device
    )
    total = n_chains * _REFINE_CANDIDATES
    axes = wp.empty(total, dtype=wp.mat33, device=device)

    # Start at the covering radius of the global grid: the sampled winner is at most about this
    # far from its basin's optimum, and each round halves the radius.
    sigma = 2.0 * (math.pi**2 / max(rotations, 2)) ** (1.0 / 3.0)
    for _ in range(refine_iterations):
        wp.launch(
            kernel_bounds.oriented_box_refine_axes,
            dim=total,
            inputs=[chains, wp.float64(sigma / math.pi), wp.int32(_REFINE_CANDIDATES), axes],
            device=device,
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
                wp.copy(chains, axes, dest_offset=chain, src_offset=row, count=1)
                chain_lower[chain] = lower_np[row]
                chain_upper[chain] = upper_np[row]
        # 0.4 rather than 0.5: a flat-flush optimum is a *kink*, so the volume error is linear in
        # the final angular resolution. Swept at eight rounds on the four tilted fixtures: 0.4
        # reads 6.0013 on the cube shell against 0.5's 6.0040 (exact minimum 6.0) and 975.29 on
        # the half torus against 979.73, at identical cost; the 127-frame ball resolves ~sigma/5
        # per round, so shrinking by 0.4 never outruns what a round can see.
        sigma *= 0.4

    winner = int(chain_loss.argmin())
    frame_np = read_scalar(chains, winner).astype(np.float64)
    return frame_np, chain_lower[winner], chain_upper[winner]


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

    Runs once per [`oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] call (before the
    refinement loop, not inside it) on at most 32 candidates picking `_REFINE_CHAINS`, so it is
    Python-call-overhead bound rather than compute bound.
    """
    head = order[: min(32, rotations)]
    head_frames = _spiral_frames(head, rotations)
    flat = head_frames.reshape(head_frames.shape[0], 9)
    # angle < 0.2 rad  <=>  trace > 1 + 2*cos(0.2), since acos is decreasing and every trace here
    # already lies in the valid [-1, 3] range for a rotation-matrix pair -- no clip needed.
    trace_threshold = 1.0 + 2.0 * math.cos(0.2)
    picked = [0]
    for i in range(1, flat.shape[0]):
        if len(picked) == _REFINE_CHAINS:
            break
        candidate = flat[i]
        far = True
        for j in picked:
            if candidate @ flat[j] > trace_threshold:
                far = False
                break
        if far:
            picked.append(i)
    for i in range(1, flat.shape[0]):
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
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]).transpose(2, 1, 0)
