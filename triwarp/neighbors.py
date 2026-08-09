"""
Point-set acceleration structures (BVH/HashGrid) and raw neighbor queries.

Also home to [`geodesic_ball`][triwarp.neighbors.geodesic_ball], the surface-aware counterpart to
the spatial ball queries here: it returns the same CSR ``(indices, offsets)`` shape but walks the
mesh edge graph, so it excludes vertices that are close in space yet across a fold of the surface.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import neighbors as kernel_neighbors
from triwarp.kernels.algorithms import bfs as kernel_bfs

# An axis counts towards a point cloud's effective dimension when its extent is at least this
# fraction of the largest one. Below that the cloud is flat (or collinear) along that axis and the
# volume-based density estimate in [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
# would invert a (near-)zero volume.
_FLAT_AXIS_FRACTION = 1e-6

# Cost of one hash-grid cell probe, expressed in linear-scan point tests. Sets where
# [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest] stops widening its cell walk
# and scans exactly instead; see ``_knn_widest_grid_radius``.
_CELL_PROBE_POINTS = 600


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int = 4) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over ``points`` for radius queries.

    Each leaf stores the same geometry as ``points`` (degenerate bounds via a clone),
    matching the broad-phase pattern used by
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball] and
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest].

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to ``warp.Bvh``.

    Returns
    -------
    warp.Bvh
        BVH suited for ``query_bvh_ball*`` and ``query_bvh_nearest``.

    See Also
    --------
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
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
        ``query_hashgrid_ball*`` and ``query_hashgrid_nearest`` kernels.
    grid_bins
        Resolution of the hash grid along each axis.

    Returns
    -------
    warp.HashGrid
        Hash grid suited for ``query_hashgrid_ball*``, ``query_hashgrid_nearest``,
        and related kernels. The cell width is recorded on the returned object as
        ``cell_width``, which
        [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest] reads back to size its
        search (``warp.HashGrid`` itself does not keep it).

    See Also
    --------
    [`query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count]
    [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball]
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]
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
    bvh: wp.Bvh, queries: wp.array[wp.vec3], half_extent: float
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level BVH AABB query: primitive indices in one flat buffer plus offsets.

    For each query center ``q``, tests intersection of the query cube
    ``[q - h, q + h]`` against every primitive bound in ``bvh``. Unlike
    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets],
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

    Returns
    -------
    candidate_indices_flat, offsets
        ``offsets`` has length ``m`` and is the exclusive prefix sum of per-query
        hit counts. Query ``k`` owns
        ``candidate_indices_flat[offsets[k] : offsets[k+1]]`` where ``offsets[m]``
        is understood as ``candidate_indices_flat.shape[0]``.
    """
    device = queries.device
    m = int(queries.shape[0])

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    hit_counts = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_neighbors.query_bvh_aabb_count,
        dim=m,
        inputs=[queries, bvh.id, wp.float32(half_extent), hit_counts],
        device=device,
    )

    offsets, total_hits = tw.array.counts_to_offsets(hit_counts)
    if total_hits == 0:
        return wp.empty(0, dtype=wp.int32, device=device), offsets

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_neighbors.query_bvh_aabb_neighbors,
        dim=m,
        inputs=[queries, bvh.id, wp.float32(half_extent), offsets, candidate_indices_flat],
        device=device,
    )

    return candidate_indices_flat, offsets


@overload
def query_hashgrid_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    grid: wp.HashGrid | None = ...,
    grid_bins: int = ...,
    return_sorted: bool = ...,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]: ...
@overload
def query_hashgrid_ball(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    r: float,
    *,
    grid: wp.HashGrid | None = ...,
    grid_bins: int = ...,
    return_sorted: bool = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
def query_hashgrid_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    grid: wp.HashGrid | None = None,
    grid_bins: int = 128,
    return_sorted: bool = False,
) -> (
    tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]
    | tuple[wp.array[wp.int32], wp.array[wp.float32]]
):
    """
    Find all data points within distance ``r`` of each query center (per-query arrays).

    High-level wrapper around
    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets]:
    hash-grid
    broad-phase and ``float32`` distance test, same SciPy semantics as
    [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Unlike SciPy's object array of lists, multi-query results are two Python lists of
    length ``m``, each element a rank-1 ``wp.array`` for that query. A single ``wp.vec3``
    query returns one ``(indices, distances)`` pair directly (not wrapped in lists). This
    clones each query's segment out of the internal flat buffer; for one flat buffer plus
    offsets on device, call
    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets]
    instead.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    grid
        Optional pre-built hash grid from ``points``. If ``None``, built via
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points].
    grid_bins
        Grid resolution when constructing ``grid`` (ignored if ``grid`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
        If ``False``, order follows grid traversal (undefined ordering).

    Returns
    -------
    neighbor_indices, neighbor_distances
        If ``queries`` has ``m`` rows: ``list[wp.array[wp.int32]]`` and
        ``list[wp.array[wp.float32]]``, each of length ``m``. Element ``k`` lists neighbors
        of ``queries[k]`` (indices into ``points`` and distances ``‖points[i] - q‖₂``).

        If ``queries`` is a single ``wp.vec3``: two rank-1 arrays (possibly length 0), not
        lists.

        Empty ``points`` yields empty neighbor arrays and per-query empty slices; duplicate
        neighbors are not produced.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries;
    here sorting only occurs when ``return_sorted=True``, and sorts by distance, not by
    index. Ball boundaries use ``float32`` arithmetic; extremely tight radii near representable
    limits may disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets]
    [`query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_hashgrid_ball_with_offsets(
        points, queries, r, grid=grid, grid_bins=grid_bins, return_sorted=return_sorted
    )
    neighbor_indices = tw.array.split(neighbor_indices_flat, offsets, copy=True)
    neighbor_distances = tw.array.split(neighbor_distances_flat, offsets, copy=True)

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


def query_hashgrid_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    grid: wp.HashGrid | None = None,
    grid_bins: int = 128,
) -> wp.array[wp.int32]:
    """
    Count neighbors of each query within Euclidean distance ``r``.

    For each query center ``q``, returns how many entries ``p`` in ``points`` satisfy
    ``‖p - q‖₂ ≤ r``. This matches [`scipy.spatial.KDTree.query_ball_point`][] with
    ``p=2``, ``eps=0``, and ``return_length=True`` (exact search; only the spatial
    index differs).

    Broad-phase traversal uses ``warp.HashGrid`` with ``wp.hash_grid_query`` out
    to ``r``; narrow-phase keeps points with Euclidean distance at most ``r`` (``float32``).

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    grid
        Optional pre-built hash grid from ``points``. If ``None``, built via
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points].
    grid_bins
        Grid resolution when constructing ``grid`` (ignored if ``grid`` is provided).

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array whose ``k``-th element is the neighbor count for
        ``queries[k]``. If ``n == 0``, returns zeros.

    See Also
    --------
    [`query_bvh_ball_count`][triwarp.neighbors.query_bvh_ball_count]
    [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    device = points.device
    n = int(points.shape[0])
    m = int(queries.shape[0])
    if n == 0:
        return wp.zeros(m, dtype=wp.int32, device=device)

    neighbor_counts = wp.empty(m, dtype=wp.int32, device=device)

    if grid is None:
        grid = hashgrid_from_points(points, r, grid_bins)

    wp.launch(
        kernel_neighbors.query_hashgrid_ball_count,
        dim=m,
        inputs=[points, queries, grid.id, wp.float32(r), neighbor_counts],
        device=device,
    )

    return neighbor_counts


def query_hashgrid_ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    grid: wp.HashGrid | None = None,
    grid_bins: int = 128,
    return_sorted: bool = False,
    include_total: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball]
    (hash-grid broad-phase out to ``r``,
    ``float32`` test ``‖points[i] - q‖₂ ≤ r``). Semantics match
    [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Prefer [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball] for a Python
    list of one array per query; use this
    when you want a single flat buffer on device (e.g. fused downstream kernels) and
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
    grid
        Optional pre-built hash grid from ``points``. If ``None``, built via
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points].
    grid_bins
        Grid resolution when constructing ``grid`` (ignored if ``grid`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
        If ``False``, order follows grid traversal (undefined ordering).
    include_total
        Return ``offsets`` in the length-``m + 1`` CSR form, whose trailing element is the total
        neighbor count, instead of the length-``m`` form. Free — the terminator is built either
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

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries;
    here sorting only occurs when ``return_sorted=True``, and sorts by distance, not by
    index. Ball boundaries use ``float32`` arithmetic; extremely tight radii near representable
    limits may disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball]
    [`query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    return _ball_with_offsets(
        points,
        queries,
        r,
        build_accelerator=lambda: (
            grid if grid is not None else hashgrid_from_points(points, r, grid_bins)
        ),
        count_fn=lambda pts, qrs, radius, accelerator: query_hashgrid_ball_count(
            pts, qrs, radius, grid=accelerator
        ),
        neighbors_kernel=kernel_neighbors.query_hashgrid_ball_neighbors,
        include_total=include_total,
        return_sorted=return_sorted,
    )


@overload
def query_bvh_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    bvh: wp.Bvh | None = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]: ...
@overload
def query_bvh_ball(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    r: float,
    *,
    bvh: wp.Bvh | None = ...,
    leaf_size: int = ...,
    return_sorted: bool = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
def query_bvh_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
) -> (
    tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]
    | tuple[wp.array[wp.int32], wp.array[wp.float32]]
):
    """
    Find all data points within distance ``r`` of each query center (BVH backend).

    High-level wrapper around
    [`query_bvh_ball_with_offsets`][triwarp.neighbors.query_bvh_ball_with_offsets]. Same SciPy
    semantics as [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    bvh
        Optional pre-built BVH from ``points``. If ``None``, built via
        [`bvh_from_points`][triwarp.neighbors.bvh_from_points].
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.

    Returns
    -------
    neighbor_indices, neighbor_distances
        Per-query neighbor lists; see
        [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball].

    See Also
    --------
    [`query_hashgrid_ball`][triwarp.neighbors.query_hashgrid_ball]
    [`query_bvh_ball_with_offsets`][triwarp.neighbors.query_bvh_ball_with_offsets]
    [`query_bvh_ball_count`][triwarp.neighbors.query_bvh_ball_count]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_bvh_ball_with_offsets(
        points, queries, r, bvh=bvh, leaf_size=leaf_size, return_sorted=return_sorted
    )
    neighbor_indices = tw.array.split(neighbor_indices_flat, offsets, copy=True)
    neighbor_distances = tw.array.split(neighbor_distances_flat, offsets, copy=True)

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


def query_bvh_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
) -> wp.array[wp.int32]:
    """
    Count neighbors of each query within Euclidean distance ``r`` (BVH backend).

    Same semantics as
    [`query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count];
    broad-phase uses
    ``wp.bvh_query_aabb`` over the cube ``[q ± r]``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query centers stored as ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    bvh
        Optional pre-built BVH from ``points``. If ``None``, built via
        [`bvh_from_points`][triwarp.neighbors.bvh_from_points].
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array of per-query neighbor counts.

    See Also
    --------
    [`query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count]
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    device = points.device
    n = int(points.shape[0])
    m = int(queries.shape[0])
    if n == 0:
        return wp.zeros(m, dtype=wp.int32, device=device)

    neighbor_counts = wp.empty(m, dtype=wp.int32, device=device)

    if bvh is None:
        bvh = bvh_from_points(points, leaf_size)

    wp.launch(
        kernel_neighbors.query_bvh_ball_count,
        dim=m,
        inputs=[points, queries, bvh.id, wp.float32(r), neighbor_counts],
        device=device,
    )

    return neighbor_counts


def query_bvh_ball_with_offsets(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
    include_total: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level BVH ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]. Semantics match
    [`scipy.spatial.KDTree.query_ball_point`][] with ``p=2`` and ``eps=0``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``.
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    bvh
        Optional pre-built BVH from ``points``. If ``None``, built via
        [`bvh_from_points`][triwarp.neighbors.bvh_from_points].
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
    include_total
        Return ``offsets`` in the length-``m + 1`` CSR form whose trailing element is the total
        neighbor count, instead of the length-``m`` form.

    Returns
    -------
    neighbor_indices_flat, neighbor_distances_flat, offsets
        CSR-style flat buffers; see
        [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets].

    See Also
    --------
    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets]
    [`query_bvh_ball`][triwarp.neighbors.query_bvh_ball]
    [`query_bvh_ball_count`][triwarp.neighbors.query_bvh_ball_count]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query_ball_point`][]
    """
    return _ball_with_offsets(
        points,
        queries,
        r,
        build_accelerator=lambda: bvh if bvh is not None else bvh_from_points(points, leaf_size),
        count_fn=lambda pts, qrs, radius, accelerator: query_bvh_ball_count(
            pts, qrs, radius, bvh=accelerator
        ),
        neighbors_kernel=kernel_neighbors.query_bvh_ball_neighbors,
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
    neighbors_kernel: wp.Kernel,
    include_total: bool,
    return_sorted: bool,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Count, scan, gather and optionally sort one ball query -- the body both accelerators share.

    [`query_hashgrid_ball_with_offsets`][triwarp.neighbors.query_hashgrid_ball_with_offsets] and
    [`query_bvh_ball_with_offsets`][triwarp.neighbors.query_bvh_ball_with_offsets] differ only in
    which structure they build, which counting pass they run and which neighbour kernel they
    launch; everything else -- the empty guards, the CSR scan, the ``2x`` sort scratch and the
    trailing compaction -- is identical. The accelerator is built lazily so an empty query never
    pays for one.
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
        neighbors_kernel,
        dim=m,
        inputs=[
            points,
            queries,
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
    ``initial_radius`` of [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] and
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest], which deepen from it
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
        [`aabb_bounds`][triwarp.bounds.aabb_bounds]. Pass it to reuse a reduction you already ran;
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
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]
    [`aabb_bounds`][triwarp.bounds.aabb_bounds]
    """
    n = int(points.shape[0])
    if n == 0 or k >= n:
        return math.inf

    if bounds is None:
        bounds = tw.bounds.aabb_bounds(points)
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
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
    bvh: wp.Bvh | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
    bvh: wp.Bvh | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
    bvh: wp.Bvh | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    max_radius: float = math.inf,
    leaf_size: int = 4,
    bvh: wp.Bvh | None = None,
    initial_radius: float | None = None,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points in Euclidean distance (``p=2``).

    Distances are ``float32`` via ``wp.length``, so they can differ from a ``float64`` reference
    on the same coordinates. For each query, at most ``k`` neighbors with distance
    ``<= max_radius`` are kept; unused slots stay at distance ``inf`` and index ``-1`` (e.g. when
    there are fewer than ``k`` points within that radius, or when ``n == 0``).

    Implementation: a BVH over ``points`` is built via
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points], then each query **deepens
    iteratively** inside a single kernel — no host round trips. A ``wp.bvh_query_aabb`` scan of the
    cube ``[q - r, q + r]`` enumerates every point within Euclidean distance ``r``, so if the row's
    ``k``-th distance comes back ``<= r`` the row *is* the exact answer and the search stops. A
    full row that reaches past the cube is re-scanned once at exactly that distance, which is
    guaranteed to certify; a row that found fewer than ``k`` points grows geometrically instead.
    The last attempt always runs at the per-query radius that provably covers the whole point
    cloud, so the result is exact however the growth went.

    The candidate row itself is held **in registers** for ``k <= 64``, in a kernel generated per
    row-size bucket (``kernels.neighbors.KNN_ROW_BUCKETS``; the row keeps the bucket's worth of
    nearest points and returns the first ``k``). That is what makes a larger ``k`` affordable — the
    row, not the geometry, is the cost of a k-NN scan, and a global-memory row pays a shift per
    accepted candidate: at ``k=32`` on ``bunny``, 225 candidates are enumerated per query against
    2 305 shifted row elements. Measured against that global-memory row on 20 000 ``bunny``
    queries: 1.09x at ``k=1``, 1.5x at ``k=7``, 2.9x at ``k=30``, 7.8x at ``k=64``. Past ``k=64`` a
    register row would spill, so there is no bucket for it: the search falls back to the
    global-memory kernel and the super-linear growth in ``k`` resumes.

    !!! note "``initial_radius`` is a performance knob, not a filter"
        Only ``max_radius`` restricts *which* neighbors are returned. ``initial_radius`` sets
        where the deepening starts, so any non-negative value — including ``0`` and ``math.inf``
        — yields byte-identical output, just at a different speed.

    Parameters
    ----------
    points
        ``n`` data points as ``wp.array[wp.vec3]``.
    queries
        ``m`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3`` (treated as ``m=1``).
    k
        Number of neighbors per query; must be ``>= 1``.
    max_radius
        Ignore point--query pairs whose distance is strictly greater than this bound (stored as
        ``float32``). Must be ``>= 0``.
    leaf_size
        Maximum primitives per BVH leaf when constructing the tree (ignored if ``bvh`` is given).
    bvh
        Optional pre-built BVH over ``points``, from
        [`bvh_from_points`][triwarp.neighbors.bvh_from_points]. Pass it to hoist the build out of
        a loop that queries the same cloud repeatedly.
    initial_radius
        Cube half-extent the search starts from; must be ``>= 0``. Defaults to
        [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius] on ``points``, which is one
        extra reduction — pass it explicitly to hoist that out of a loop as well.
    bounds
        Optional ``(min_bound, max_bound)`` of ``points`` from
        [`aabb_bounds`][triwarp.bounds.aabb_bounds]. Together with ``bvh`` and ``initial_radius``
        this removes every per-call host synchronisation, which is what makes a fixed target cloud
        free to re-query in a loop (see [`icp`][triwarp.registration.icp]).

    Returns
    -------
    indices, distances
        Pair of 2-D arrays with shape ``(m, k)``, row ``q`` listing neighbors for ``queries[q]``
        in non-decreasing distance order (when ``k > 1`` and neighbors exist in that row).

        * If ``k == 1``, both arrays are reshaped to length ``m`` (one index and one distance
          per query).
        * If ``queries`` was a single ``wp.vec3`` and ``k > 1``, returns two length-``k`` 1D
          arrays (the sole query row). If ``k == 1``, returns two length-1 1D arrays.
        * If ``m == 0``, returns empty ``(0, k)`` arrays.
        * If ``n == 0`` but ``m > 0``, returns the pre-filled ``(m, k)`` arrays of ``inf`` and
          ``-1`` (or the corresponding 1D slices for a single ``wp.vec3`` query).

    Raises
    ------
    ValueError
        If ``k < 1``, ``max_radius < 0`` or ``initial_radius < 0``.

    See Also
    --------
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]
    [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
    [`bvh_from_points`][triwarp.neighbors.bvh_from_points]
    [`scipy.spatial.KDTree.query`][]
    """
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
        bounds = tw.bounds.aabb_bounds(points)
    min_bound, max_bound = bounds
    if initial_radius is None:
        initial_radius = knn_initial_radius(points, k, bounds=bounds)
    if bvh is None:
        bvh = bvh_from_points(points, leaf_size)

    # ``wp.empty``, not ``wp.full``: every row is written in full by the kernel, so pre-filling
    # here would be two wasted launches.
    neighbor_indices = twt.empty_2d((m, k), wp.int32, device=device)
    neighbor_distances = twt.empty_2d((m, k), wp.float32, device=device)
    wp.launch(
        kernel_neighbors.bvh_nearest_kernel(k),
        dim=m,
        inputs=[
            points,
            queries,
            bvh.id,
            wp.int32(k),
            wp.float32(max_radius),
            wp.float32(initial_radius),
            min_bound,
            max_bound,
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )
    return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)


@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
    grid: wp.HashGrid | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
    grid: wp.HashGrid | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
    grid: wp.HashGrid | None = ...,
    initial_radius: float | None = ...,
    bounds: tuple[wp.vec3, wp.vec3] | None = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    max_radius: float = math.inf,
    grid_bins: int = 128,
    grid: wp.HashGrid | None = None,
    initial_radius: float | None = None,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points (HashGrid backend).

    Same semantics and the same iterative deepening as
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] — including the register-resident
    candidate row for ``k <= 64``; broad-phase uses ``warp.HashGrid`` with ``wp.hash_grid_query``,
    which enumerates every cell overlapping the query cube, so the same "``k``-th distance at most
    ``r`` certifies the row" argument holds.

    Two things differ from the BVH backend, both because a grid cannot follow an unbounded radius:
    ``wp.hash_grid_query`` visits ``(2 ceil(r / cell) + 1) ** 3`` cells, so the search falls back
    to an **exact linear scan** over ``points`` once ``r`` outgrows a few cells — that is the same
    per-row cost the old whole-cloud query paid for every row, so it is never a regression. And the
    cell width is clamped to at least ``max_extent / grid_bins``, which also removes hash aliasing
    (``hash_grid_index`` takes ``x % dim_x``, so a cloud spanning more than ``grid_bins`` cells per
    axis folds distant world cells into one bucket).

    !!! note "Which backend to use"
        Measured on an RTX 5090, this one is ~1.6x faster than
        [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] when the queries sit on or near
        the cloud, which is the usual case and why the distance metrics in ``triwarp.distance``
        use it. Prefer the BVH when the queries may be *far* from a *large* cloud: a BVH descent
        degrades gracefully with the search radius, whereas this backend hands off to the linear
        scan and pays ``O(n)`` per row (on ``dragon``'s 438k points, 3x slower).

    Parameters
    ----------
    points
        ``n`` data points as ``wp.array[wp.vec3]``.
    queries
        ``m`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``.
    k
        Number of neighbors per query; must be ``>= 1``.
    max_radius
        Ignore point--query pairs whose distance is strictly greater than this bound.
        Must be ``>= 0``.
    grid_bins
        Resolution of the hash grid along each axis when constructing the grid (ignored if
        ``grid`` is given).
    grid
        Optional pre-built hash grid over ``points``. Build it with
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points] so its cell width comes
        along; a grid from anywhere else is assumed to use ``initial_radius`` as its cell width,
        which only affects when the linear-scan fallback kicks in.
    initial_radius
        Cube half-extent the search starts from; must be ``>= 0``. Defaults to
        [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius] on ``points``. Affects speed
        only, never the result.
    bounds
        Optional ``(min_bound, max_bound)`` of ``points`` from
        [`aabb_bounds`][triwarp.bounds.aabb_bounds]; pass it alongside ``grid`` and
        ``initial_radius`` to make a repeated query on a fixed cloud synchronisation-free.

    Returns
    -------
    indices, distances
        Same layout rules as [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest].

    Raises
    ------
    ValueError
        If ``k < 1``, ``max_radius < 0`` or ``initial_radius < 0``.

    See Also
    --------
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
    [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius]
    [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]
    [`scipy.spatial.KDTree.query`][]
    """
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
        bounds = tw.bounds.aabb_bounds(points)
    min_bound, max_bound = bounds
    if initial_radius is None:
        initial_radius = knn_initial_radius(points, k, bounds=bounds)
    if grid is None:
        cell_size = _knn_cell_size(initial_radius, min_bound, max_bound, grid_bins)
        grid = hashgrid_from_points(points, cell_size, grid_bins)
    else:
        cell_size = float(getattr(grid, "cell_width", initial_radius))
    widest = _knn_widest_grid_radius(cell_size, n)

    neighbor_indices = twt.empty_2d((m, k), wp.int32, device=device)
    neighbor_distances = twt.empty_2d((m, k), wp.float32, device=device)
    wp.launch(
        kernel_neighbors.hashgrid_nearest_kernel(k),
        dim=m,
        inputs=[
            points,
            queries,
            grid.id,
            wp.int32(k),
            wp.float32(max_radius),
            wp.float32(initial_radius),
            wp.float32(widest),
            min_bound,
            max_bound,
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )
    return _shape_nearest(neighbor_indices, neighbor_distances, k, single_query)


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
        chunk_total = int(chunk_total_buf.numpy()[0])
        flat_chunk = wp.empty(chunk_total, dtype=wp.int32, device=device)
        if chunk_total > 0:
            wp.launch(
                kernel_neighbors.gather_queue_rows,
                dim=(m, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS),
                inputs=[queue_pool, counts, local_offsets, wp.int32(start), flat_chunk],
                device=device,
            )
        chunk_flats.append(flat_chunk)

    n_overflow = int(overflow.numpy()[0])
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
