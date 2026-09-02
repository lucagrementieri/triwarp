"""
Point-set acceleration structures (BVH/HashGrid) and raw neighbor queries.

**The accelerator is a keyword, not a function name.** There are two questions here -- "everything
within ``r``" and "the ``k`` nearest" -- and each is one function:
[`query_ball`][triwarp.neighbors.query_ball] (with
[`query_ball_count`][triwarp.neighbors.query_ball_count] and
[`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets] for the count alone and the
flat CSR form) and [`query_nearest`][triwarp.neighbors.query_nearest]. Which broad phase runs is a
[`QueryBackend`][triwarp.neighbors.QueryBackend] keyword -- ``"hashgrid"`` or ``"bvh"`` -- or is
inferred from a prebuilt structure passed as ``accelerator``. Both backends are **exact and return
the same answer**; the choice is a cost one, and ``query_nearest``'s docstring carries the measured
guidance. The kernel side was already one
warp-uniform kernel branching on an ``ACCEL_*`` selector, so only the Python layer had doubled.

The BVH-only queries keep the structure in their names, because naming it is informative rather
than redundant there -- a hash grid has no box query:
[`query_bvh_aabb_with_offsets`][triwarp.neighbors.query_bvh_aabb_with_offsets] and
[`query_bvh_box`][triwarp.neighbors.query_bvh_box].

Also home to [`geodesic_ball`][triwarp.neighbors.geodesic_ball], the surface-aware counterpart to
the spatial ball queries here: it returns the same CSR ``(indices, offsets)`` shape but walks the
mesh edge graph, so it excludes vertices that are close in space yet across a fold of the surface.

[`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance] is the one derived
quantity rather than a raw query: the per-point distance to the closest other point, which is what a
cloud's scale is normally estimated from.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar
from triwarp.kernels import neighbors as kernel_neighbors
from triwarp.kernels import points as kernel_points
from triwarp.kernels.algorithms import bfs as kernel_bfs

# An axis counts towards a point cloud's effective dimension when its extent is at least this
# fraction of the largest one. Below that the cloud is flat (or collinear) along that axis and the
# volume-based density estimate in [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
# would invert a (near-)zero volume.
_FLAT_AXIS_FRACTION = 1e-6

# Cost of one hash-grid cell probe, expressed in linear-scan point tests. Sets where
# [`query_nearest`][triwarp.neighbors.query_nearest] stops widening its cell walk
# under ``backend="hashgrid"``
# and scans exactly instead; see ``_knn_widest_grid_radius``.
_CELL_PROBE_POINTS = 600


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int = 4) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over ``points`` for radius queries.

    Each leaf stores the same geometry as ``points`` (degenerate bounds via a clone),
    matching the broad-phase pattern used by
    [`query_ball`][triwarp.neighbors.query_ball] and
    [`query_nearest`][triwarp.neighbors.query_nearest].

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to ``warp.Bvh``.

    Returns
    -------
    warp.Bvh
        BVH suited for the ``query_ball*`` and ``query_nearest`` family; pass it as their
        ``accelerator``, which selects ``backend="bvh"`` by its type.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    return wp.Bvh(points, points, leaf_size=leaf_size)


def hashgrid_from_points(
    points: wp.array[wp.vec3], radius: float, grid_bins: int = 128
) -> wp.HashGrid:
    """
    Build a 3D hash grid over ``points`` for radius queries.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    radius
        Cell size passed to ``warp.HashGrid.build`` and used by
        the ``query_ball*`` and ``query_nearest`` kernels under ``backend="hashgrid"``.
    grid_bins
        Resolution of the hash grid along each axis.

    Returns
    -------
    warp.HashGrid
        Hash grid suited for the ``query_ball*`` and ``query_nearest`` family,
        and related kernels. The cell width is recorded on the returned object as
        ``cell_width``, which
        [`query_nearest`][triwarp.neighbors.query_nearest] reads back to size its
        search (``warp.HashGrid`` itself does not keep it).

    See Also
    --------
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    n = int(points.shape[0])
    grid = wp.HashGrid(grid_bins, grid_bins, grid_bins, device=points.device)
    grid.reserve(n)
    grid.build(points, radius)
    grid.cell_width = float(radius)
    return grid


def bvh_from_bounds(
    lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], leaf_size: int = 4
) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over axis-aligned bounds.

    Each primitive ``i`` is represented by ``lower[i]`` and ``upper[i]`` corner
    positions, suitable for
    [`query_bvh_aabb_with_offsets`][triwarp.neighbors.query_bvh_aabb_with_offsets]
    broad-phase intersection tests.

    Parameters
    ----------
    lower
        ``(n,)`` minimum corner of each bound as ``wp.vec3``.
    upper
        ``(n,)`` maximum corner of each bound as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to ``warp.Bvh``.

    Returns
    -------
    warp.Bvh
        BVH suited for AABB intersection queries.
    """
    return wp.Bvh(lower, upper, leaf_size=leaf_size)


def query_bvh_aabb_with_offsets(
    bvh: wp.Bvh, queries: wp.array[wp.vec3], half_extent: float, *, include_total: bool = False
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level BVH AABB query: primitive indices in one flat buffer plus offsets.

    For each query center ``q``, tests intersection of the query cube
    ``[q - h, q + h]`` against every primitive bound in ``bvh``. Unlike
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets],
    there is no narrow-phase distance filter;
    every broad-phase hit is returned.

    Parameters
    ----------
    bvh
        Pre-built BVH from [`bvh_from_bounds`][triwarp.neighbors.bvh_from_bounds]
        or [`bvh_from_points`][triwarp.neighbors.bvh_from_points].
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    half_extent
        Half side length of the axis-aligned query cube along each axis.
    include_total
        Return ``offsets`` in the length-``m + 1`` CSR form whose trailing element is the total hit
        count, instead of the length-``m`` form. Same meaning as on the ball queries.

    Returns
    -------
    candidate_indices_flat, offsets
        ``offsets`` has length ``m`` (or ``m + 1`` with ``include_total``) and is the exclusive
        prefix sum of per-query hit counts. Query ``k`` owns
        ``candidate_indices_flat[offsets[k] : offsets[k+1]]``. In the length-``m`` form
        ``offsets[m]`` is understood as ``candidate_indices_flat.shape[0]``, so a Python-scope
        caller iterating the queries wants ``include_total=True`` rather than appending it.

    Notes
    -----
    The default is the length-``m`` form because this function's only in-repo consumer is a kernel
    that recovers the owning query with ``kernels.array.binary_search_index`` and never addresses
    ``offsets[m]``. Both forms are views into one ``m + 1`` scan buffer
    ([`counts_to_offsets`][triwarp.array.counts_to_offsets]), so the keyword costs no allocation.

    See Also
    --------
    [`query_bvh_box`][triwarp.neighbors.query_bvh_box]
        The same query with a **per-query** box instead of one cube size for every query.
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
        The same packing and the same keyword, for a ball query with a narrow-phase filter.
    """
    device = queries.device
    m = int(queries.shape[0])

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.zeros(1 if include_total else 0, dtype=wp.int32, device=device),
        )

    hit_counts = wp.empty(m, dtype=wp.int32, device=device)
    # ``wp.uint64(bvh.id)`` explicitly: unlike ``wp.launch``, ``wp.map`` infers a bare Python int
    # scalar's dtype as ``wp.int32`` rather than matching the mapped @wp.func's declared parameter
    # type, and a mismatched dtype is a codegen-time TypeError, not a silent truncation (probed
    # this session, both here and at ``aabb_count_in_bounds`` below).
    wp.map(
        kernel_neighbors.aabb_count_in_box,
        wp.uint64(bvh.id),
        queries,
        wp.float32(half_extent),
        out=hit_counts,
    )

    # One ``m + 1`` scan buffer serves both forms: the length-``m`` one is a view of its prefix.
    segment_bounds, total_hits = tw.array.counts_to_offsets(hit_counts, include_total=True)
    offsets = segment_bounds if include_total else segment_bounds[:m]
    if total_hits == 0:
        return wp.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_neighbors.query_bvh_aabb_neighbors,
        dim=m,
        inputs=[
            queries,
            bvh.id,
            wp.float32(half_extent),
            segment_bounds[:m],
            candidate_indices_flat,
        ],
        device=device,
    )

    return candidate_indices_flat, offsets


def query_bvh_box(
    bvh: wp.Bvh,
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    *,
    include_total: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Primitives overlapping one axis-aligned box **per query**, in the same CSR packing.

    The box counterpart of the ball queries, and the per-query generalization of
    [`query_bvh_aabb_with_offsets`][triwarp.neighbors.query_bvh_aabb_with_offsets]: each query
    carries its own ``(lower, upper)`` corners rather than sharing one cube size. On a BVH built by
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points], whose leaf bounds are degenerate, a hit
    means the point is **inside** the box, so the answer is exact and no narrow phase is needed; on
    one built by [`bvh_from_bounds`][triwarp.neighbors.bvh_from_bounds] it is box-versus-box
    overlap, like every other broad phase here.

    Parameters
    ----------
    bvh
        Pre-built BVH over points or bounds.
    query_lower, query_upper
        ``(m,)`` ``wp.vec3`` corners of the query boxes, one pair per query.
    include_total
        Return ``offsets`` in the length-``m + 1`` CSR form whose trailing element is the total hit
        count, instead of the length-``m`` form. Same meaning as on the ball queries.

    Returns
    -------
    candidate_indices_flat, offsets
        Query ``k`` owns ``candidate_indices_flat[offsets[k] : offsets[k + 1]]``, with ``offsets``
        the exclusive prefix sum of per-query hit counts.

    Raises
    ------
    ValueError
        If ``query_lower`` and ``query_upper`` do not have the same length.

    Notes
    -----
    The test is **inclusive** on every face: a point exactly on a box face is inside it (measured on
    Warp 1.17, both the lower and the upper face). A box with any ``upper < lower`` component
    matches nothing, and that is not checked -- the check would cost a host readback per call
    (CLAUDE.md section 13) to reject a caller error whose answer is already empty.

    There is no list-returning sibling, so the name carries no ``_with_offsets`` suffix: this
    query's consumers are kernels that recover the owning query from ``offsets``, and a Python list
    of per-query arrays would add ``O(m)`` host slicing to a query whose whole point is that it is
    batched.

    See Also
    --------
    [`query_bvh_aabb_with_offsets`][triwarp.neighbors.query_bvh_aabb_with_offsets]
        One cube size for every query, which needs no corner buffers at all.
    [`triwarp.points.half_space_mask`][triwarp.points.half_space_mask]
        The unbounded counterpart: selection by one plane rather than by a box.
    """
    device = query_lower.device
    m = int(query_lower.shape[0])
    if int(query_upper.shape[0]) != m:
        raise ValueError("query_lower and query_upper must have the same length")

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.zeros(1 if include_total else 0, dtype=wp.int32, device=device),
        )

    hit_counts = wp.empty(m, dtype=wp.int32, device=device)
    # ``wp.uint64(bvh.id)`` explicitly -- see the same cast in ``query_bvh_aabb_with_offsets``
    # above.
    wp.map(
        kernel_neighbors.aabb_count_in_bounds,
        wp.uint64(bvh.id),
        query_lower,
        query_upper,
        out=hit_counts,
    )

    segment_bounds, total_hits = tw.array.counts_to_offsets(hit_counts, include_total=True)
    offsets = segment_bounds if include_total else segment_bounds[:m]
    if total_hits == 0:
        return wp.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_neighbors.query_bvh_box_neighbors,
        dim=m,
        inputs=[query_lower, query_upper, bvh.id, segment_bounds[:m], candidate_indices_flat],
        device=device,
    )

    return candidate_indices_flat, offsets


# Which broad phase a query runs. Public and named because it appears in four public signatures and
# in the benchmark group names; both values are exact and return the same answer, so this is a cost
# choice -- see ``query_nearest``'s docstring for the measured guidance.
QueryBackend = Literal["hashgrid", "bvh"]


@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    grid_bins: int = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]: ...
@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    grid_bins: int = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int = 128,
    leaf_size: int = 4,
    return_sorted: bool = False,
) -> (
    tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]
    | tuple[wp.array[wp.int32], wp.array[wp.float32]]
):
    """
    Find all data points within distance ``r`` of each query center (per-query arrays).

    Same exact search as [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``;
    only the spatial index differs, and which index is a keyword rather than a function name (see
    ``backend``). High-level wrapper around
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets].

    Unlike SciPy's object array of lists, multi-query results are two Python lists of length ``m``,
    each element a rank-1 ``wp.array`` for that query. A single ``wp.vec3`` query returns one
    ``(indices, distances)`` pair directly (not wrapped in lists). This clones each query's segment
    out of the internal flat buffer; for one flat buffer plus offsets on device, call
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets] instead.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator
        A prebuilt ``warp.HashGrid`` or ``warp.Bvh`` over ``points``, to reuse across queries. It
        selects the backend by its own type, so ``backend`` is redundant when this is given and
        raises if it names the other one.
    backend
        Which broad phase to use when ``accelerator`` is ``None``: ``"hashgrid"`` (the default)
        enumerates the cells overlapping the query cube, ``"bvh"`` descends an AABB tree. The
        narrow phase and the answer are identical -- this is a cost choice, not a semantic one.
    grid_bins
        Grid resolution when building a hash grid. Ignored under ``backend="bvh"`` and whenever
        ``accelerator`` is given.
    leaf_size
        Maximum primitives per leaf when building a BVH. Ignored under ``backend="hashgrid"`` and
        whenever ``accelerator`` is given.
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance. If ``False``,
        order follows the broad phase's traversal (undefined ordering).

    Returns
    -------
    neighbor_indices, neighbor_distances
        If ``queries`` has ``m`` rows: ``list[wp.array[wp.int32]]`` and
        ``list[wp.array[wp.float32]]``, each of length ``m``. Element ``k`` lists neighbors
        of ``queries[k]`` (indices into ``points`` and distances ``‖points[i] - q‖₂``).

        If ``queries`` is a single ``wp.vec3``: two rank-1 arrays (possibly length 0), not lists.

        Empty ``points`` yields empty neighbor arrays and per-query empty slices; duplicate
        neighbors are not produced.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries; here
    sorting only occurs when ``return_sorted=True``, and sorts by distance, not by index. Ball
    boundaries use ``float32`` arithmetic; extremely tight radii near representable limits may
    disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets]
        The flat CSR form, without cloning a segment per query.
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
        The counts alone, when the neighbors themselves are not wanted.
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_ball_with_offsets(
        points,
        queries,
        r,
        accelerator=accelerator,
        backend=backend,
        grid_bins=grid_bins,
        leaf_size=leaf_size,
        return_sorted=return_sorted,
    )
    neighbor_indices = tw.array.split(neighbor_indices_flat, offsets, copy=True)
    neighbor_distances = tw.array.split(neighbor_distances_flat, offsets, copy=True)

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int = 128,
    leaf_size: int = 4,
) -> wp.array[wp.int32]:
    """
    Count neighbors of each query within Euclidean distance ``r``.

    For each query center ``q``, returns how many entries ``p`` in ``points`` satisfy
    ``‖p - q‖₂ ≤ r``. This matches [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2``,
    ``eps=0``, and ``return_length=True`` (exact search; only the spatial index differs).

    The narrow phase keeps points with ``float32`` Euclidean distance at most ``r`` whichever
    broad phase ran, so the count does not depend on ``backend``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator, backend, grid_bins, leaf_size
        As in [`query_ball`][triwarp.neighbors.query_ball].

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array whose ``k``-th element is the neighbor count for
        ``queries[k]``. If ``n == 0``, returns zeros.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    kind, accelerator = _resolve_accelerator(accelerator, backend)
    device = points.device
    n = int(points.shape[0])
    m = int(queries.shape[0])
    if n == 0:
        return wp.zeros(m, dtype=wp.int32, device=device)

    neighbor_counts = wp.empty(m, dtype=wp.int32, device=device)
    if kind == "hashgrid":
        if accelerator is None:
            accelerator = hashgrid_from_points(points, r, grid_bins)
        accel_selector = kernel_neighbors.ACCEL_HASHGRID
    else:
        if accelerator is None:
            accelerator = bvh_from_points(points, leaf_size)
        accel_selector = kernel_neighbors.ACCEL_BVH

    wp.launch(
        kernel_neighbors.query_ball_count,
        dim=m,
        inputs=[points, queries, accel_selector, accelerator.id, wp.float32(r), neighbor_counts],
        device=device,
    )
    return neighbor_counts


def query_ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    grid_bins: int = 128,
    leaf_size: int = 4,
    return_sorted: bool = False,
    include_total: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as [`query_ball`][triwarp.neighbors.query_ball] (broad phase out to ``r``,
    ``float32`` test ``‖points[i] - q‖₂ ≤ r``). Semantics match
    [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Prefer [`query_ball`][triwarp.neighbors.query_ball] for a Python list of one array per query;
    use this when you want a single flat buffer on device (e.g. fused downstream kernels) and
    CSR-style boundaries without cloning each segment.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    accelerator, backend, grid_bins, leaf_size
        As in [`query_ball`][triwarp.neighbors.query_ball].
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
    include_total
        Return ``offsets`` in the length-``m + 1`` CSR form, whose trailing element is the total
        neighbor count, instead of the length-``m`` form. Free -- the terminator is built either
        way, since the ``return_sorted`` path needs it as segment bounds.

    Returns
    -------
    neighbor_indices_flat, neighbor_distances_flat, offsets
        Three rank-1 arrays. Let ``m = queries.shape[0]`` after any ``wp.vec3`` wrap.

        ``offsets`` has length ``m`` (or ``m + 1`` with ``include_total``) and is the exclusive
        prefix sum of per-query neighbor counts (same layout as
        ``wp.utils.array_scan(..., inclusive=False)``): query ``k`` owns
        ``neighbor_indices_flat[offsets[k] : offsets[k+1]]`` where ``offsets[m]`` is
        ``neighbor_indices_flat.shape[0]`` (the total neighbor count), stored only in the
        CSR form.

        ``neighbor_indices_flat`` and ``neighbor_distances_flat`` have that total length
        and list point indices and distances ``‖points[i] - q‖₂`` in parallel. Empty
        ``points`` still returns zero ``offsets``; empty neighbor sets yield
        length-0 flat arrays and zero ``offsets``.

    Raises
    ------
    ValueError
        If ``backend`` is neither name, or contradicts the type of ``accelerator``.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries; here
    sorting only occurs when ``return_sorted=True``, and sorts by distance, not by index. Ball
    boundaries use ``float32`` arithmetic; extremely tight radii near representable limits may
    disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_ball`][triwarp.neighbors.query_ball]
    [`query_ball_count`][triwarp.neighbors.query_ball_count]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    kind, resolved = _resolve_accelerator(accelerator, backend)
    if kind == "hashgrid":
        return _ball_with_offsets(
            points,
            queries,
            r,
            build_accelerator=lambda: (
                resolved if resolved is not None else hashgrid_from_points(points, r, grid_bins)
            ),
            count_fn=lambda pts, qrs, radius, built: query_ball_count(
                pts, qrs, radius, accelerator=built
            ),
            accel=kernel_neighbors.ACCEL_HASHGRID,
            include_total=include_total,
            return_sorted=return_sorted,
        )
    return _ball_with_offsets(
        points,
        queries,
        r,
        build_accelerator=lambda: (
            resolved if resolved is not None else bvh_from_points(points, leaf_size)
        ),
        count_fn=lambda pts, qrs, radius, built: query_ball_count(
            pts, qrs, radius, accelerator=built
        ),
        accel=kernel_neighbors.ACCEL_BVH,
        include_total=include_total,
        return_sorted=return_sorted,
    )


def _ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    build_accelerator: Callable[[], wp.HashGrid | wp.Bvh],
    count_fn: Callable[[wp.array[wp.vec3], wp.array[wp.vec3], float, Any], wp.array[wp.int32]],
    accel: wp.int32,
    include_total: bool,
    return_sorted: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Count, scan, gather and optionally sort one ball query -- the body both accelerators share.

    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets] and
    [`query_ball_with_offsets`][triwarp.neighbors.query_ball_with_offsets] differ only in
    which structure they build, which counting pass they run and which traversal the one shared
    neighbour kernel takes (``accel``, one of ``kernels.neighbors.ACCEL_*``); everything else --
    the empty guards, the CSR scan, the ``2x`` sort scratch and the trailing compaction -- is
    identical. The accelerator is built lazily so an empty query never pays for one.
    """
    device = points.device

    if isinstance(queries, wp.vec3):
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    if int(points.shape[0]) == 0 or m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.zeros(m + 1 if include_total else m, dtype=wp.int32, device=device),
        )

    accelerator = build_accelerator()
    neighbor_counts = count_fn(points, queries, r, accelerator)
    # The total-terminated CSR form is the segment-bounds array ``segmented_sort_pairs``
    # wants below; the length-``m`` form is a view of its prefix, so both come out of the one
    # scan buffer.
    segment_bounds, total_neighbors = tw.array.counts_to_offsets(
        neighbor_counts, include_total=True
    )
    offsets = segment_bounds if include_total else segment_bounds[:m]
    if total_neighbors == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            offsets,
        )

    flat_len = total_neighbors * (2 if return_sorted else 1)
    neighbor_indices_flat = wp.empty(flat_len, dtype=wp.int32, device=device)
    neighbor_distances_flat = wp.empty(flat_len, dtype=wp.float32, device=device)
    wp.launch(
        kernel_neighbors.query_ball_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            accel,
            accelerator.id,
            wp.float32(r),
            segment_bounds,
            neighbor_indices_flat,
            neighbor_distances_flat,
        ],
        device=device,
    )

    if return_sorted:
        wp.utils.segmented_sort_pairs(
            neighbor_distances_flat, neighbor_indices_flat, total_neighbors, segment_bounds
        )
    return (
        wp.clone(neighbor_indices_flat[:total_neighbors]),
        wp.clone(neighbor_distances_flat[:total_neighbors]),
        offsets,
    )


def knn_initial_radius(
    points: wp.array[wp.vec3], k: int, *, bounds: tuple[wp.vec3, wp.vec3] | None = None
) -> float:
    """
    Radius at which a ``k``-nearest search is expected to succeed on the first try.

    Inverts a uniform-density model of ``points``: the smallest ball expected to hold ``k`` of
    ``n`` points is the one whose volume is ``k / n`` of the bounding box's. This is the default
    ``initial_radius`` of [`query_nearest`][triwarp.neighbors.query_nearest] and
    [`query_nearest`][triwarp.neighbors.query_nearest], which deepen from it
    until each row certifies itself, so the value affects **speed only** and never the result.

    A cloud that is flat or collinear has a (near-)zero box volume, which the 3-D formula would
    invert into a useless radius, so the effective dimension ``d`` is taken to be the number of
    axes whose extent is non-negligible against the largest, and the matching formula is used:
    ``d = 3`` gives ``(3 f V / 4pi)^(1/3)``, ``d = 2`` gives ``(f A / pi)^(1/2)`` and ``d = 1``
    gives ``f L / 2``, with fill fraction ``f = k / n``.

    Parameters
    ----------
    points
        ``(n,)`` data points as ``wp.vec3``.
    k
        Number of neighbors the search will ask for; must be ``>= 1``.
    bounds
        Optional ``(min_bound, max_bound)`` from
        [`aabb`][triwarp.bounds.aabb]. Pass it to reuse a reduction you already ran;
        otherwise it is computed here (one device reduction plus one readback).

    Returns
    -------
    float
        The estimated radius, or ``math.inf`` when no finite estimate is meaningful — ``n == 0``,
        ``k >= n`` (every point is a neighbor) or a degenerate box (all points coincident). The
        query kernels read ``math.inf`` as "one complete scan", which is the right answer in each
        of those cases.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    [`aabb`][triwarp.bounds.aabb]
    """
    n = int(points.shape[0])
    if n == 0 or k >= n:
        return math.inf

    if bounds is None:
        bounds = tw.bounds.aabb(points)
    min_bound, max_bound = bounds
    extents = sorted((float(max_bound[axis] - min_bound[axis]) for axis in range(3)), reverse=True)
    if extents[0] <= 0.0:
        return math.inf

    fill = k / n
    dimension = sum(extent > _FLAT_AXIS_FRACTION * extents[0] for extent in extents)
    if dimension == 3:
        volume = extents[0] * extents[1] * extents[2]
        return (3.0 * fill * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    if dimension == 2:
        return math.sqrt(fill * extents[0] * extents[1] / math.pi)
    return 0.5 * fill * extents[0]


@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array1dInt32, twt.Array1dFloat32]: ...
@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
@overload
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = ...,
    backend: QueryBackend | None = ...,
    max_radius: float = ...,
    grid_bins: int = ...,
    leaf_size: int = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    accelerator: wp.HashGrid | wp.Bvh | None = None,
    backend: QueryBackend | None = None,
    max_radius: float = math.inf,
    grid_bins: int = 128,
    leaf_size: int = 4,
    initial_radius: float | None = None,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points.

    Iterative deepening: each query grows a search radius from ``initial_radius`` until its ``k``-th
    neighbour is closer than the radius, which certifies the row -- so the answer is exact and does
    not depend on ``backend``. The candidate row is register-resident for ``k <= 64``.

    !!! note "Which backend to use"
        ``"hashgrid"`` (the default) is the faster broad phase when the query scale matches the
        cloud's density, because ``initial_radius`` sets the cell width: the walk visits
        ``(2 ceil(r / cell) + 1) ** 3`` cells, so a radius **f** times too large costs ``f ** 3``,
        and the search falls back to an exact linear scan once ``r`` outgrows a few cells. That also
        makes it sensitive to a cloud whose density is not uniform, and to queries drawn from a
        different distribution than the points -- a translated query cloud measured **87x** worse.

        ``"bvh"`` has no cell width to get wrong, so it is the one to reach for when the query scale
        is unknown, the cloud is non-uniform, or the queries sit outside it. Its own cost grows with
        ``k`` faster than the grid's -- measured **7x** from ``k=1`` to ``k=30`` -- so at ``k=1`` on
        a matched cloud the grid wins and at large ``k`` on an awkward one the tree does.

        Both are exact; this is a cost choice only. Reuse the structure across calls by passing it
        as ``accelerator`` when several queries share one cloud.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``.
    k
        Number of neighbours per query, ``>= 1``.
    accelerator
        A prebuilt ``warp.HashGrid`` or ``warp.Bvh`` over ``points``, to reuse across queries. It
        selects the backend by its own type, so ``backend`` is redundant when this is given and
        raises if it names the other one. A hash grid passed here keeps its own ``cell_width``
        rather than one derived from ``initial_radius``.
    backend
        ``"hashgrid"`` (the default) or ``"bvh"``, when ``accelerator`` is ``None``. See the note
        above.
    max_radius
        Stop deepening past this distance and leave the remaining slots unfilled (index ``-1``,
        distance ``inf``). Defaults to unbounded.
    grid_bins
        Grid resolution when building a hash grid. Ignored under ``backend="bvh"`` and whenever
        ``accelerator`` is given.
    leaf_size
        Maximum primitives per leaf when building a BVH. Ignored under ``backend="hashgrid"`` and
        whenever ``accelerator`` is given.
    initial_radius
        First search radius. Defaults to
        [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius], which estimates it from the
        *cloud's* density -- so pass it explicitly when the queries are at a different scale, since
        under ``"hashgrid"`` this also sets the cell width.
    bounds
        ``(min_bound, max_bound)`` of ``points``, to skip
        [`triwarp.bounds.aabb`][triwarp.bounds.aabb] and its readback.

    Returns
    -------
    neighbor_indices, neighbor_distances
        ``(m, k)`` ``wp.int32`` indices into ``points`` and ``(m, k)`` ``wp.float32`` distances,
        each row sorted by increasing distance. A slot no neighbour was found for holds ``-1`` and
        ``inf``. For a single ``wp.vec3`` query the two are rank-1 of length ``k``.

    Raises
    ------
    ValueError
        If ``k < 1``, ``max_radius < 0``, ``initial_radius < 0``, or ``backend`` is neither name or
        contradicts the type of ``accelerator``.

    See Also
    --------
    [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
        The default first radius, and the one parameter worth passing by hand.
    [`query_ball`][triwarp.neighbors.query_ball]
        A fixed radius rather than a fixed count.
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance]
        The ``k=2`` self-query, which is the cloud's spacing.
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query`][]
    """
    kind, resolved = _resolve_accelerator(accelerator, backend)
    _validate_nearest(k, max_radius, initial_radius)

    device = points.device
    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    m = int(queries.shape[0])
    n = int(points.shape[0])

    if m == 0:
        return (
            twt.empty_2d((0, k), wp.int32, device=device),
            twt.empty_2d((0, k), wp.float32, device=device),
        )
    if n == 0:
        return _empty_nearest(m, k, single_query, device)

    if bounds is None:
        bounds = tw.bounds.aabb(points)
    min_bound, max_bound = bounds
    if initial_radius is None:
        initial_radius = knn_initial_radius(points, k, bounds=bounds)

    if kind == "bvh":
        bvh = resolved if resolved is not None else bvh_from_points(points, leaf_size)
        kernel = kernel_neighbors.bvh_nearest_kernel(k)
        accel_id = bvh.id
        # The BVH follows an unbounded radius, so it needs no linear-scan cutover argument.
        widest: list[wp.float32] = []
    else:
        if resolved is None:
            cell_size = _knn_cell_size(initial_radius, min_bound, max_bound, grid_bins)
            grid = hashgrid_from_points(points, cell_size, grid_bins)
        else:
            grid = resolved
            cell_size = float(getattr(grid, "cell_width", initial_radius))
        kernel = kernel_neighbors.hashgrid_nearest_kernel(k)
        accel_id = grid.id
        widest = [wp.float32(_knn_widest_grid_radius(cell_size, n))]

    # ``wp.empty``, not ``wp.full``: every row is written in full by the kernel, so pre-filling
    # here would be two wasted launches.
    neighbor_indices = twt.empty_2d((m, k), wp.int32, device=device)
    neighbor_distances = twt.empty_2d((m, k), wp.float32, device=device)
    wp.launch(
        kernel,
        dim=m,
        inputs=[
            points,
            queries,
            accel_id,
            wp.int32(k),
            wp.float32(max_radius),
            wp.float32(initial_radius),
            *widest,
            min_bound,
            max_bound,
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )
    return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)


def _resolve_accelerator(
    accelerator: wp.HashGrid | wp.Bvh | None, backend: QueryBackend | None
) -> tuple[QueryBackend, wp.HashGrid | wp.Bvh | None]:
    """
    Settle ``backend`` against ``accelerator``, the one new failure mode the merged API has.

    ``backend`` is ``None`` rather than ``"hashgrid"`` in the signatures so that "not passed" is
    distinguishable from "passed as hashgrid": a caller who hands over a ``wp.Bvh`` and nothing else
    must not be told it contradicts a default they never wrote. The *effective* default is still
    ``"hashgrid"``, and it applies only when no accelerator is given.

    A prebuilt accelerator already knows what it is, so it wins the dispatch -- but a ``backend``
    that names the other one is a mistake in the call rather than a preference to be silently
    dropped, and it raises.
    """
    if accelerator is None:
        if backend is None:
            return "hashgrid", None
        if backend not in ("hashgrid", "bvh"):
            raise ValueError(f'backend must be "hashgrid" or "bvh", got {backend!r}')
        return backend, None

    inferred: QueryBackend = "hashgrid" if isinstance(accelerator, wp.HashGrid) else "bvh"
    if backend is not None and backend != inferred:
        raise ValueError(
            f"backend={backend!r} contradicts the accelerator passed, which is a "
            f"{type(accelerator).__name__} ({inferred!r}). Pass one or the other."
        )
    return inferred, accelerator


def query_weighted_nearest(
    points: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    queries: wp.array[wp.vec3],
    *,
    max_weight: float | None = None,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]:
    """
    Nearest site under the weighted distance ``|p - q| - w(p)``.

    Each site carries a radius, and the winner is the one whose *surface* is closest rather than
    whose centre is -- the additively weighted (Apollonius) nearest-neighbour query, which is what
    picks the influencing site when the sites have different scales: a sphere set, a level-of-detail
    cluster, a set of samples with per-sample confidence. Plain
    [`query_nearest`][triwarp.neighbors.query_nearest] is the ``w = 0`` case, and the two
    genuinely differ -- on a random 40-site cloud, 1 query in 6 had a different winner.

    Parameters
    ----------
    points
        ``(n,)`` site positions as ``wp.vec3``.
    weights
        ``(n,)`` per-site weights, subtracted from the distance. Larger wins ties of distance; may
        be negative, which pushes a site away.
    queries
        ``(m,)`` query positions as ``wp.vec3``.
    max_weight
        An **upper bound** on ``weights``, which is what makes the search prunable. ``None`` reduces
        ``weights`` on the device and reads the maximum back (one readback, ~0.1 ms), so pass it
        when the bound is already known -- a radius cap, or a previous call's reduction.
    bvh
        A ``wp.Bvh`` already built over ``points``, to spare the build.
    leaf_size
        Maximum primitives per BVH leaf when one is built here.

    Returns
    -------
    index, weighted_distance
        ``(m,)`` winning site per query and its ``|p - q| - w(p)``, which is **negative** wherever a
        query lies inside a site's radius. A query with no site at all (an empty cloud) reports
        ``-1`` and ``inf``.

    Raises
    ------
    ValueError
        If ``weights`` does not have one entry per point.

    Notes
    -----
    ``max_weight`` is trusted, not checked: a bound *smaller* than some weight can silently prune
    the true winner, and verifying it would cost the very reduction the parameter exists to avoid.
    The default is therefore the safe one.

    Exactness is the same argument the k-NN queries use, with the weight folded in: a site outside
    the cube of half-extent ``r`` is farther than ``r``, so it scores worse than ``r - max_weight``,
    and a best score at or below that bound cannot be beaten. Unlike a plain nearest query this
    means the search radius must exceed the answer's distance *by the weight range*, so a wide
    weight distribution costs more scans than a narrow one.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
        The unweighted query, and the ``k > 1`` form. This one answers ``k = 1`` only, because no
        caller has needed more.
    """
    device = points.device
    n = int(points.shape[0])
    m = int(queries.shape[0])
    if int(weights.shape[0]) != n:
        raise ValueError("weights must have one entry per point")

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
        )
    if n == 0:
        return (
            wp.full(m, -1, dtype=wp.int32, device=device),
            wp.full(m, float("inf"), dtype=wp.float32, device=device),
        )

    if bvh is None:
        bvh = bvh_from_points(points, leaf_size=leaf_size)
    if max_weight is None:
        max_weight = float(cast(float, tw.reduce.max(weights)))
    min_bound, max_bound = tw.bounds.aabb(points)
    # First radius: the mean spacing's own estimate plus the weight bound, since a query cannot be
    # certified below it however close its winner is.
    initial_radius = knn_initial_radius(points, 1, bounds=(min_bound, max_bound)) + max(
        max_weight, 0.0
    )

    out_indices = wp.empty(m, dtype=wp.int32, device=device)
    out_distances = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_neighbors.query_weighted_nearest_neighbors,
        dim=m,
        inputs=[
            points,
            weights,
            queries,
            bvh.id,
            wp.float32(max_weight),
            wp.float32(initial_radius),
            min_bound,
            max_bound,
            out_indices,
            out_distances,
        ],
        device=device,
    )
    return out_indices, out_distances


def nearest_neighbor_distance(points: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Distance from each point to the closest *other* point of the same cloud.

    The standard scale estimate for an unstructured cloud, and what a sampling-driven parameter is
    normally derived from: the mean of this array is the cloud's mean spacing, which is how
    [`ball_pivoting`][triwarp.reconstruction.ball_pivoting] guesses its ball radius and how
    [`screened_poisson`][triwarp.reconstruction.screened_poisson] caps its octree depth. A
    self-query for two neighbours, keeping the second — the first is the point itself, at distance
    zero.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` distances on ``points.device``. Zero where two points coincide exactly.

    Notes
    -----
    A cloud of fewer than two points has no answer, and this reports ``inf`` for every entry —
    the value [`query_nearest`][triwarp.neighbors.query_nearest] already uses for a slot it
    could not fill. Open3D's ``compute_nearest_neighbor_distance`` reports ``0.0`` in that one
    case; on any cloud of two or more points the two agree, because every point then has a nearest
    neighbour. The distances are ``float32`` (``wp.length``), so they can differ from a
    ``float64`` reference in the last digits.

    See Also
    --------
    [`query_nearest`][triwarp.neighbors.query_nearest]
        The underlying query; call it directly to reuse a BVH across several queries of one cloud.
    [`triwarp.points.farthest_point_sample`][triwarp.points.farthest_point_sample]
    """
    device = points.device
    n = int(points.shape[0])
    if n < 2:
        return wp.full(n, wp.float32(math.inf), dtype=wp.float32, device=device)

    _indices, distances = query_nearest(points, points, k=2, backend="bvh")
    # Column 1 of the ``(n, 2)`` table, which is a *strided* view -- so it is copied into a dense
    # buffer rather than returned, both because callers expect a plain ``wp.array`` and because a
    # strided array is the shape that silently corrupts a downstream Python-scope gather.
    nearest = wp.empty(n, dtype=wp.float32, device=device)
    wp.copy(nearest, distances[:, 1])
    return nearest


def closest_pair(points: wp.array[wp.vec3]) -> tuple[int, int, float]:
    """
    Find the two closest points of a cloud, and the distance between them.

    The global minimum of
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance], with the partner
    recovered -- so it is the same ``k=2`` self-query, reduced instead of returned.
    Answers "does this cloud contain a near-duplicate, and where" in one call, which is the question
    a tolerance for
    [`triwarp.points.point_duplicate_mask`][triwarp.points.point_duplicate_mask] is normally chosen
    from.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``, ``n >= 2``.

    Returns
    -------
    index_a, index_b, distance
        The two point indices — ``index_a < n``, ``index_b`` its nearest neighbour — and their
        Euclidean distance, all as Python scalars.

    Raises
    ------
    ValueError
        If ``points`` holds fewer than two points, which have no pair.

    Notes
    -----
    Ties are broken toward the **lowest** ``index_a``: the reduction is a ``min`` over one ``int64``
    per point holding the distance in the high half and the index in the low, so a shorter distance
    wins and an equal distance defers to the smaller index. Exact duplicates therefore report the
    first duplicated point at distance ``0.0``.

    Two host readbacks, both unavoidable and both single-element: the reduction's result, and
    ``index_b`` from the query table. Everything up to them stays on the device, so this does not
    move the ``(n, 2)`` table across the bus.

    See Also
    --------
    [`nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance]
        The per-point form, when every distance is wanted rather than the smallest.
    [`triwarp.points.point_duplicate_mask`][triwarp.points.point_duplicate_mask]
        Exact coincidence rather than proximity, and a mask rather than one pair.
    """
    n = int(points.shape[0])
    if n < 2:
        raise ValueError("closest_pair needs at least two points")

    indices, distances = query_nearest(points, points, k=2, backend="bvh")
    keys = wp.empty(n, dtype=wp.int64, device=points.device)
    wp.launch(
        kernel_points.nearest_pair_keys, dim=n, inputs=[distances, keys], device=points.device
    )

    key = int(cast(int, tw.reduce.min(keys)))
    index_a = key & 0xFFFFFFFF
    # The high half is the distance's own float32 bits, so it decodes on the host for free rather
    # than costing a third readback.
    distance = float(np.array([key >> 32], dtype=np.uint32).view(np.float32)[0])
    index_b = int(read_scalar(indices.flatten(), 2 * index_a + 1))
    return index_a, index_b, distance


def geodesic_ball(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], radius: float, min_count: int = 6
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Per-vertex geodesic-ball neighborhoods, and the reference neighbor that frames each one.

    For each vertex this is a breadth-first traversal of the mesh edge graph, enqueueing a neighbor
    only when it lies within Euclidean ``radius`` of the center — i.e. the connected component of
    the center within the radius ball. This is a *geodesic* ball rather than a pure Euclidean one,
    so it excludes vertices that are spatially close but lie across a fold of the surface (e.g. the
    opposite wall of a torus tube), which a Euclidean hash-grid query would wrongly include and
    which corrupts the quadric fit. When fewer than ``min_count`` vertices are reachable, the
    nearest out-of-ball vertices are appended (libigl's ``extra_candidates`` path).

    Also returns the per-vertex reference neighbor used to build the tangent frame: the
    lowest-indexed edge neighbor, matching libigl's ``adjacency_list[i][0]``. libigl's symmetrized
    shape operator is frame-dependent, so reproducing its principal values (
    [`principal_curvature`][triwarp.curvature.principal_curvature] with ``frame_independent=False``)
    requires this exact frame; the default frame-independent computation does not depend on it.
    Isolated vertices reference themselves.

    The traversal runs entirely on device. Vertex adjacency is built as a CSR graph via
    [`edges_unique`][triwarp.edges.edges_unique] +
    [`edges_to_csr`][triwarp.graph.edges_to_csr], then a single-pass BFS collects each ball into
    its per-source queue row (the queue prefix *is* the result) and a scan + gather compacts the
    rows into the CSR neighbor buffer. Each source uses fixed-capacity scratch of
    ``_PER_SOURCE_MAX_NEIGHBORS`` neighbors (``triwarp.kernels.algorithms.bfs``, currently 512);
    if a vertex collects more than that the surplus is dropped and a warning is emitted.

    !!! note

        Distances and tie-breaking are computed in ``float32`` (set-equivalent to the libigl
        reference; borderline ties between equidistant neighbors may resolve differently but leave
        the order-independent quadric fit unchanged).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions as ``wp.vec3``.
    faces
        Length-``3 * n_faces`` flat triangle index buffer as ``wp.int32``.
    radius
        Geodesic-ball radius in world units.
    min_count
        Minimum neighbors per vertex; the nearest out-of-ball vertices backfill any shortfall.

    Returns
    -------
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
        ``(neighbor_indices, offsets, reference_neighbors)``. ``offsets`` is the
        length-``n_vertices`` exclusive prefix sum of per-vertex neighbor counts (CSR starts);
        vertex ``i`` owns ``neighbor_indices[offsets[i] : offsets[i + 1]]`` with ``offsets[n]``
        implied as the total. ``reference_neighbors`` has length ``n_vertices``.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return (
            empty,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n)
    adjacency = tw.graph.edges_to_csr(n, unique_edges)
    adj_offsets = adjacency.offsets
    adj_columns = adjacency.columns

    reference_neighbors = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_neighbors.geodesic_ball_reference_neighbors,
        dim=n,
        inputs=[adj_offsets, adj_columns, reference_neighbors],
        device=device,
    )

    # Per-source scratch lives in shared global-memory pools sized for one chunk of sources
    # (queue rows, an open-addressing visited row pre-filled with -1 per launch, and a small
    # nearest-fallback pool) instead of ~8 KB of per-thread local arrays.
    chunk = min(n, 1 << 15)
    queue_pool = wp.empty(
        (chunk, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS), dtype=wp.int32, device=device
    )
    visited_pool = wp.empty(
        (chunk, kernel_bfs._VISITED_HASH_CAPACITY), dtype=wp.int32, device=device
    )
    ext_dist_pool = twt.empty_2d((chunk, kernel_bfs._EXTRAS_CAPACITY), wp.float32, device=device)
    ext_idx_pool = twt.empty_2d((chunk, kernel_bfs._EXTRAS_CAPACITY), wp.int32, device=device)

    overflow = wp.zeros(1, dtype=wp.int32, device=device)
    counts = wp.empty(n, dtype=wp.int32, device=device)
    local_offsets = wp.empty(chunk, dtype=wp.int32, device=device)
    chunk_total_buf = wp.empty(1, dtype=wp.int32, device=device)
    chunk_flats: list[wp.array[wp.int32]] = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        visited_pool.fill_(-1)
        wp.launch(
            kernel_neighbors.query_geodesic_ball_collect,
            dim=m,
            inputs=[
                vertices,
                adj_offsets,
                adj_columns,
                wp.float32(radius),
                wp.int32(min_count),
                wp.int32(start),
                queue_pool,
                visited_pool,
                ext_dist_pool,
                ext_idx_pool,
                counts,
                overflow,
            ],
            device=device,
        )
        # Gather this chunk's queue rows before the next chunk reuses the pools: chunk-local
        # exclusive scan of counts, one 4-byte readback for the chunk total, then a coalesced
        # 2D copy into the chunk's flat buffer.
        wp.utils.array_scan(counts[start : start + m], out_array=local_offsets[:m], inclusive=False)
        wp.map(
            wp.add, local_offsets[m - 1 : m], counts[start + m - 1 : start + m], out=chunk_total_buf
        )
        chunk_total = int(read_scalar(chunk_total_buf, 0))
        flat_chunk = wp.empty(chunk_total, dtype=wp.int32, device=device)
        if chunk_total > 0:
            wp.launch(
                kernel_neighbors.gather_queue_rows,
                dim=(m, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS),
                inputs=[queue_pool, counts, local_offsets, wp.int32(start), flat_chunk],
                device=device,
            )
        chunk_flats.append(flat_chunk)

    n_overflow = int(read_scalar(overflow, 0))
    if n_overflow > 0:
        warnings.warn(
            f"geodesic_ball: {n_overflow} neighborhood capacity breaches "
            f"(fixed cap {kernel_bfs._PER_SOURCE_MAX_NEIGHBORS}); surplus neighbors dropped.",
            stacklevel=2,
        )

    offsets = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=offsets, inclusive=False)

    if len(chunk_flats) == 1:
        # Single chunk (n <= chunk): the chunk buffer already is the global CSR neighbor buffer.
        return chunk_flats[0], offsets, reference_neighbors

    # Chunk order equals ascending source order, so concatenation lines up with the global scan.
    # Same per-segment ``wp.copy`` loop either way -- that is the packing floor -- one call for it.
    return tw.array.concatenate(chunk_flats), offsets, reference_neighbors


def _knn_cell_size(
    initial_radius: float, min_bound: wp.vec3, max_bound: wp.vec3, grid_bins: int
) -> float:
    """Hash-grid cell width for a k-NN search starting at ``initial_radius``."""
    extent = max(float(max_bound[axis] - min_bound[axis]) for axis in range(3))
    if extent <= 0.0:
        # Every point is at the same position, so any positive width buckets them together.
        return 1.0
    # Lower bound ``extent / grid_bins`` keeps the cloud inside one period of the spatial hash;
    # upper bound ``extent`` keeps a huge ``initial_radius`` (``k >= n`` gives ``inf``) finite.
    return min(max(initial_radius, extent / grid_bins), extent)


def _knn_widest_grid_radius(cell_size: float, n: int) -> float:
    """
    Radius past which an exact linear scan beats widening the hash-grid walk.

    ``wp.hash_grid_query`` visits ``(2 ceil(r / cell) + 1) ** 3`` cells, and each visit is a hash
    plus two dependent, uncoalesced global loads. The linear scan it falls back to is the opposite:
    every thread in a warp reads the *same* ``points[j]``, so it streams out of L2 as a broadcast.
    Measured on an RTX 5090, one cell probe costs on the order of ``_CELL_PROBE_POINTS`` point
    tests, which makes the break-even span grow as ``n ** (1/3)`` — one cell of slack on ``bunny``
    (36k points), four on ``dragon`` (438k). A fixed span gets one of those two badly wrong.
    """
    span = 0.5 * ((n / _CELL_PROBE_POINTS) ** (1.0 / 3.0) - 1.0)
    # Never below one cell: the 3x3x3 walk is what the grid was built for and is cheap at any n.
    return max(span, 1.0) * cell_size


def _validate_nearest(k: int, max_radius: float, initial_radius: float | None) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")
    if initial_radius is not None and initial_radius < 0:
        raise ValueError("initial_radius must be >= 0")


def _empty_nearest(
    m: int, k: int, single_query: bool, device: wp.DeviceLike
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """Build the result rows for an empty point cloud: every slot unfilled."""
    neighbor_indices = wp.full((m, k), wp.int32(-1), dtype=wp.int32, device=device)
    neighbor_distances = wp.full((m, k), math.inf, dtype=wp.float32, device=device)
    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return twt.as_array2d(neighbor_indices, wp.int32), twt.as_array2d(
        neighbor_distances, wp.float32
    )


def _shape_nearest(
    neighbor_indices: wp.array[wp.int32],
    neighbor_distances: wp.array[wp.float32],
    k: int,
    single_query: bool,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """Collapse the ``(m, k)`` result to the rank the caller's ``queries`` / ``k`` imply."""
    if k == 1:
        return (
            cast(twt.Array1dInt32, neighbor_indices.reshape(-1)),
            cast(twt.Array1dFloat32, neighbor_distances.reshape(-1)),
        )
    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return twt.as_array2d(neighbor_indices, wp.int32), twt.as_array2d(
        neighbor_distances, wp.float32
    )
