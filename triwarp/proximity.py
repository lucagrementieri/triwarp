"""
Proximity queries: point clouds, BVH/HashGrid neighbors, and mesh point queries.

Mesh point queries (:func:`closest_point_on_mesh`, :func:`signed_distance_on_mesh`)
and :func:`triwarp.ray.contains_points` follow Warp's SDF sign convention: outside
positive, inside negative. Trimesh ``signed_distance`` uses the opposite sign.
"""

from __future__ import annotations

import math
from typing import Literal, cast, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import proximity as kernel_proximity
from triwarp.triangles import face_normals_and_areas


def aabb_bounds(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Axis-aligned bounding box of ``points`` (component-wise min / max).

    The reduction runs on ``points.device`` in ``float32`` via atomic min/max per axis.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(min_bound, max_bound)`` with ``min_bound[i] ≤ p[i] ≤ max_bound[i]`` for every
        point ``p`` and axis ``i``. If ``n == 0``, ``min_bound`` is ``(+inf, …)`` and
        ``max_bound`` is ``(-inf, …)`` (initial reduction buffers unchanged).
    """
    out_min = wp.full(3, math.inf, dtype=wp.float32, device=points.device)
    out_max = wp.full(3, -math.inf, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_proximity.aabb_bounds,
        dim=points.shape[0],
        inputs=[points, out_min, out_max],
        device=points.device,
    )
    out_min = out_min.list()
    out_max = out_max.list()
    min_bound = wp.vec3(out_min[0], out_min[1], out_min[2])
    max_bound = wp.vec3(out_max[0], out_max[1], out_max[2])
    return min_bound, max_bound


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int = 4) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over ``points`` for radius queries.

    Each leaf stores the same geometry as ``points`` (degenerate bounds via a clone),
    matching the broad-phase pattern used by :func:`query_bvh_ball` and
    :func:`query_bvh_nearest`.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to :class:`warp.Bvh`.

    Returns
    -------
    warp.Bvh
        BVH suited for ``query_bvh_ball*`` and ``query_bvh_nearest``.

    See Also
    --------
    query_bvh_ball
    query_bvh_nearest
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
        Cell size passed to :meth:`warp.HashGrid.build` and used by
        ``query_hashgrid_ball*`` and ``query_hashgrid_nearest`` kernels.
    grid_bins
        Resolution of the hash grid along each axis.

    Returns
    -------
    warp.HashGrid
        Hash grid suited for ``query_hashgrid_ball*``, ``query_hashgrid_nearest``,
        and related kernels.

    See Also
    --------
    query_hashgrid_ball_count
    query_hashgrid_ball
    query_hashgrid_nearest
    """
    n = int(points.shape[0])
    grid = wp.HashGrid(grid_bins, grid_bins, grid_bins, device=points.device)
    grid.reserve(n)
    grid.build(points, radius)
    return grid


def bvh_from_bounds(
    lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], leaf_size: int = 4
) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over axis-aligned bounds.

    Each primitive ``i`` is represented by ``lower[i]`` and ``upper[i]`` corner
    positions, suitable for :func:`query_bvh_aabb_with_offsets` broad-phase
    intersection tests.

    Parameters
    ----------
    lower
        ``(n,)`` minimum corner of each bound as ``wp.vec3``.
    upper
        ``(n,)`` maximum corner of each bound as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to :class:`warp.Bvh`.

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
    :func:`query_hashgrid_ball_with_offsets`, there is no narrow-phase distance filter;
    every broad-phase hit is returned.

    Parameters
    ----------
    bvh
        Pre-built BVH from :func:`bvh_from_bounds` or :func:`bvh_from_points`.
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
        kernel_proximity.query_bvh_aabb_count,
        dim=m,
        inputs=[queries, bvh.id, wp.float32(half_extent), hit_counts],
        device=device,
    )

    total_hits = int(hit_counts.numpy().sum())
    if total_hits == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
        )

    offsets = wp.empty(m, dtype=wp.int32, device=device)
    wp.utils.array_scan(hit_counts, out_array=offsets, inclusive=False)

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.query_bvh_aabb_neighbors,
        dim=m,
        inputs=[queries, bvh.id, wp.float32(half_extent), offsets, candidate_indices_flat],
        device=device,
    )

    return candidate_indices_flat, offsets


def query_mesh_aabb_bounds_with_offsets(
    mesh: wp.Mesh,
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    *,
    max_hits: int = 16,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level mesh AABB query with per-query axis-aligned bounds.

    For each query primitive ``k``, tests intersection of ``[query_lower[k],
    query_upper[k]]`` against every triangle in ``mesh`` via ``wp.mesh_query_aabb``.
    At most ``max_hits`` candidate face indices are recorded per query.

    Requires the default Warp mesh BVH backend; ``bvh_constructor="cubql"`` meshes
    do not support AABB queries.

    Returns
    -------
    candidate_indices_flat, offsets, hit_counts
        ``offsets`` is the exclusive prefix sum of per-query hit counts.
        Query ``k`` owns ``candidate_indices_flat[offsets[k] : offsets[k] + hit_counts[k]]``.
    """
    device = query_lower.device
    m = int(query_lower.shape[0])
    if int(query_upper.shape[0]) != m:
        raise ValueError("query_lower and query_upper must have the same length")
    if max_hits < 1:
        raise ValueError("max_hits must be >= 1")

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    hit_counts = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.query_mesh_aabb_bounds_count,
        dim=m,
        inputs=[query_lower, query_upper, mesh.id, wp.int32(max_hits), hit_counts],
        device=device,
    )

    total_hits = int(hit_counts.numpy().sum())
    if total_hits == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
            hit_counts,
        )

    offsets = wp.empty(m, dtype=wp.int32, device=device)
    wp.utils.array_scan(hit_counts, out_array=offsets, inclusive=False)

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.query_mesh_aabb_bounds_neighbors,
        dim=m,
        inputs=[
            query_lower,
            query_upper,
            mesh.id,
            wp.int32(max_hits),
            offsets,
            candidate_indices_flat,
        ],
        device=device,
    )

    return candidate_indices_flat, offsets, hit_counts


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
    ``‖p - q‖₂ ≤ r``. This matches :meth:`scipy.spatial.KDTree.query_ball_point` with
    ``p=2``, ``eps=0``, and ``return_length=True`` (exact search; only the spatial
    index differs).

    Broad-phase traversal uses :class:`warp.HashGrid` with ``wp.hash_grid_query`` out
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
        :func:`hashgrid_from_points`.
    grid_bins
        Grid resolution when constructing ``grid`` (ignored if ``grid`` is provided).

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array whose ``k``-th element is the neighbor count for
        ``queries[k]``. If ``n == 0``, returns zeros.

    See Also
    --------
    query_bvh_ball_count
    query_hashgrid_ball
    hashgrid_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
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
        kernel_proximity.query_hashgrid_ball_count,
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
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as :func:`query_hashgrid_ball` (hash-grid broad-phase out to ``r``,
    ``float32`` test ``‖points[i] - q‖₂ ≤ r``). Semantics match
    :meth:`scipy.spatial.KDTree.query_ball_point` with ``p=2`` and ``eps=0``.

    Prefer :func:`query_hashgrid_ball` for a Python list of one array per query; use this
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
        :func:`hashgrid_from_points`.
    grid_bins
        Grid resolution when constructing ``grid`` (ignored if ``grid`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
        If ``False``, order follows grid traversal (undefined ordering).

    Returns
    -------
    neighbor_indices_flat, neighbor_distances_flat, offsets
        Three rank-1 arrays. Let ``m = queries.shape[0]`` after any ``wp.vec3`` wrap.

        ``offsets`` has length ``m`` and is the exclusive prefix sum of per-query neighbor
        counts (same layout as ``wp.utils.array_scan(..., inclusive=False)``): query ``k``
        owns ``neighbor_indices_flat[offsets[k] : offsets[k+1]]`` where ``offsets[m]`` is
        understood as ``neighbor_indices_flat.shape[0]`` (the total neighbor count).

        ``neighbor_indices_flat`` and ``neighbor_distances_flat`` have that total length
        and list point indices and distances ``‖points[i] - q‖₂`` in parallel. Empty
        ``points`` still returns length-``m`` zero ``offsets``; empty neighbor sets yield
        length-0 flat arrays and zero ``offsets``.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries;
    here sorting only occurs when ``return_sorted=True``, and sorts by distance, not by
    index. Ball boundaries use ``float32`` arithmetic; extremely tight radii near representable
    limits may disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    query_hashgrid_ball
    query_hashgrid_ball_count
    hashgrid_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
    """
    device = points.device

    if isinstance(queries, wp.vec3):
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    n: int = int(points.shape[0])
    if n == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
        )

    if grid is None:
        grid = hashgrid_from_points(points, r, grid_bins)

    neighbor_counts = query_hashgrid_ball_count(points, queries, r, grid=grid)
    total_neighbors = int(neighbor_counts.numpy().sum())

    if total_neighbors == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
        )

    offsets = wp.empty(m, dtype=wp.int32, device=device)
    wp.utils.array_scan(neighbor_counts, out_array=offsets, inclusive=False)

    flat_len = total_neighbors * (2 if return_sorted else 1)
    neighbor_indices_flat = wp.empty(flat_len, dtype=wp.int32, device=device)
    neighbor_distances_flat = wp.empty(flat_len, dtype=wp.float32, device=device)
    wp.launch(
        kernel_proximity.query_hashgrid_ball_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            grid.id,
            wp.float32(r),
            offsets,
            neighbor_indices_flat,
            neighbor_distances_flat,
        ],
        device=device,
    )

    if return_sorted:
        segment_bounds = wp.empty(m + 1, dtype=wp.int32, device=device)
        wp.copy(segment_bounds, offsets, dest_offset=0, src_offset=0, count=m)
        wp.copy(
            segment_bounds,
            wp.array([total_neighbors], dtype=wp.int32, device=device),
            dest_offset=m,
            src_offset=0,
            count=1,
        )
        wp.utils.segmented_sort_pairs(
            neighbor_distances_flat, neighbor_indices_flat, total_neighbors, segment_bounds
        )
    return (
        wp.clone(neighbor_indices_flat[:total_neighbors]),
        wp.clone(neighbor_distances_flat[:total_neighbors]),
        offsets,
    )


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

    High-level wrapper around :func:`query_hashgrid_ball_with_offsets`: hash-grid
    broad-phase and ``float32`` distance test, same SciPy semantics as
    :meth:`scipy.spatial.KDTree.query_ball_point` with ``p=2`` and ``eps=0``.

    Unlike SciPy's object array of lists, multi-query results are two Python lists of
    length ``m``, each element a rank-1 ``wp.array`` for that query. A single ``wp.vec3``
    query returns one ``(indices, distances)`` pair directly (not wrapped in lists). This
    clones each query's segment out of the internal flat buffer; for one flat buffer plus
    offsets on device, call :func:`query_hashgrid_ball_with_offsets` instead.

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
        :func:`hashgrid_from_points`.
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
    query_hashgrid_ball_with_offsets
    query_hashgrid_ball_count
    hashgrid_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_hashgrid_ball_with_offsets(
        points, queries, r, grid=grid, grid_bins=grid_bins, return_sorted=return_sorted
    )
    neighbor_indices: list[wp.array[wp.int32]] = []
    neighbor_distances: list[wp.array[wp.float32]] = []

    offsets_list = offsets.list()
    for k in range(m):
        start = offsets_list[k]
        end = offsets_list[k + 1] if k < m - 1 else neighbor_indices_flat.shape[0]
        if end - start > 0:
            neighbor_indices.append(wp.clone(neighbor_indices_flat[start:end]))
            neighbor_distances.append(wp.clone(neighbor_distances_flat[start:end]))
        else:
            neighbor_indices.append(wp.empty(0, dtype=wp.int32, device=device))
            neighbor_distances.append(wp.empty(0, dtype=wp.float32, device=device))

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

    Same semantics as :func:`query_hashgrid_ball_count`; broad-phase uses
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
        :func:`bvh_from_points`.
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).

    Returns
    -------
    wp.array[wp.int32]
        Length-``m`` device array of per-query neighbor counts.

    See Also
    --------
    query_hashgrid_ball_count
    query_bvh_ball
    bvh_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
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
        kernel_proximity.query_bvh_ball_count,
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
) -> tuple[wp.array[wp.int32], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Low-level BVH ball query: neighbors in one concatenated pair plus per-query offsets.

    Same geometry as :func:`query_bvh_ball`. Semantics match
    :meth:`scipy.spatial.KDTree.query_ball_point` with ``p=2`` and ``eps=0``.

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
        :func:`bvh_from_points`.
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.

    Returns
    -------
    neighbor_indices_flat, neighbor_distances_flat, offsets
        CSR-style flat buffers; see :func:`query_hashgrid_ball_with_offsets`.

    See Also
    --------
    query_hashgrid_ball_with_offsets
    query_bvh_ball
    query_bvh_ball_count
    bvh_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
    """
    device = points.device

    if isinstance(queries, wp.vec3):
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    n: int = int(points.shape[0])
    if n == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
        )

    if bvh is None:
        bvh = bvh_from_points(points, leaf_size)

    neighbor_counts = query_bvh_ball_count(points, queries, r, bvh=bvh)
    total_neighbors = int(neighbor_counts.numpy().sum())

    if total_neighbors == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.zeros(m, dtype=wp.int32, device=device),
        )

    offsets = wp.empty(m, dtype=wp.int32, device=device)
    wp.utils.array_scan(neighbor_counts, out_array=offsets, inclusive=False)

    flat_len = total_neighbors * (2 if return_sorted else 1)
    neighbor_indices_flat = wp.empty(flat_len, dtype=wp.int32, device=device)
    neighbor_distances_flat = wp.empty(flat_len, dtype=wp.float32, device=device)
    wp.launch(
        kernel_proximity.query_bvh_ball_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            bvh.id,
            wp.float32(r),
            offsets,
            neighbor_indices_flat,
            neighbor_distances_flat,
        ],
        device=device,
    )

    if return_sorted:
        segment_bounds = wp.empty(m + 1, dtype=wp.int32, device=device)
        wp.copy(segment_bounds, offsets, dest_offset=0, src_offset=0, count=m)
        wp.copy(
            segment_bounds,
            wp.array([total_neighbors], dtype=wp.int32, device=device),
            dest_offset=m,
            src_offset=0,
            count=1,
        )
        wp.utils.segmented_sort_pairs(
            neighbor_distances_flat, neighbor_indices_flat, total_neighbors, segment_bounds
        )
    return (
        wp.clone(neighbor_indices_flat[:total_neighbors]),
        wp.clone(neighbor_distances_flat[:total_neighbors]),
        offsets,
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

    High-level wrapper around :func:`query_bvh_ball_with_offsets`. Same SciPy semantics
    as :meth:`scipy.spatial.KDTree.query_ball_point` with ``p=2`` and ``eps=0``.

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
        :func:`bvh_from_points`.
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.

    Returns
    -------
    neighbor_indices, neighbor_distances
        Per-query neighbor lists; see :func:`query_hashgrid_ball`.

    See Also
    --------
    query_hashgrid_ball
    query_bvh_ball_with_offsets
    query_bvh_ball_count
    bvh_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    neighbor_indices_flat, neighbor_distances_flat, offsets = query_bvh_ball_with_offsets(
        points, queries, r, bvh=bvh, leaf_size=leaf_size, return_sorted=return_sorted
    )
    neighbor_indices: list[wp.array[wp.int32]] = []
    neighbor_distances: list[wp.array[wp.float32]] = []

    offsets_list = offsets.list()
    for k in range(m):
        start = offsets_list[k]
        end = offsets_list[k + 1] if k < m - 1 else neighbor_indices_flat.shape[0]
        if end - start > 0:
            neighbor_indices.append(wp.clone(neighbor_indices_flat[start:end]))
            neighbor_distances.append(wp.clone(neighbor_distances_flat[start:end]))
        else:
            neighbor_indices.append(wp.empty(0, dtype=wp.int32, device=device))
            neighbor_distances.append(wp.empty(0, dtype=wp.float32, device=device))

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


@overload
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    max_radius: float = ...,
    leaf_size: int = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_bvh_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    max_radius: float = math.inf,
    leaf_size: int = 4,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points in Euclidean distance (``p=2``).

    Distances are ``float32`` via ``wp.length``, so they can differ from a ``float64`` reference
    on the same coordinates. For each query, at most ``k`` neighbors with distance ``<=`` the
    effective radius are kept; unused slots stay at distance ``inf`` and index ``-1`` (e.g. when
    there are fewer than ``k`` points within that radius, or when ``n == 0``).

    Implementation: a BVH over ``points`` is built via :func:`bvh_from_points`. Each query
    runs ``wp.bvh_query_aabb`` over the cube ``[q - r, q + r]`` with the (possibly clamped)
    radius below and updates a per-query sorted list of the ``k`` smallest distances
    (binary search + shift insert in the kernel when ``k > 1``; a single running minimum when
    ``k == 1``).

    The radius passed to the query is ``min(max_radius, max(diagonal, 1e-12))``, where
    ``diagonal`` is the length of the axis-aligned box that contains both ``points`` and
    ``queries``. So a user ``max_radius`` larger than that span is capped; ``math.inf`` means
    "use the scene diagonal".

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
        Maximum primitives per BVH leaf when constructing the tree.

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
        If ``k < 1`` or ``max_radius < 0``.

    See Also
    --------
    query_hashgrid_nearest
    :meth:`scipy.spatial.KDTree.query`
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")

    device = points.device
    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    m = int(queries.shape[0])
    n = int(points.shape[0])

    if m == 0:
        return (
            twt.empty_int32_2d((0, k), device=device),
            twt.empty_float32_2d((0, k), device=device),
        )

    neighbor_indices = wp.full((m, k), wp.int32(-1), dtype=wp.int32, device=device)
    neighbor_distances = wp.full((m, k), math.inf, dtype=wp.float32, device=device)
    if n == 0:
        if single_query:
            return neighbor_indices[0], neighbor_distances[0]
        return twt.as_array2d_int32(neighbor_indices), twt.as_array2d_float32(neighbor_distances)

    min_points_bound, max_points_bound = aabb_bounds(points)
    min_queries_bound, max_queries_bound = aabb_bounds(queries)
    min_bound = wp.vec3(
        min(min_points_bound.x, min_queries_bound.x),
        min(min_points_bound.y, min_queries_bound.y),
        min(min_points_bound.z, min_queries_bound.z),
    )
    max_bound = wp.vec3(
        max(max_points_bound.x, max_queries_bound.x),
        max(max_points_bound.y, max_queries_bound.y),
        max(max_points_bound.z, max_queries_bound.z),
    )
    diagonal = wp.length(max_bound - min_bound)
    max_radius = min(max_radius, max(diagonal, 1e-12))

    bvh = bvh_from_points(points, leaf_size)

    wp.launch(
        kernel_proximity.query_bvh_nearest_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            bvh.id,
            wp.int32(k),
            wp.float32(max_radius),
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )

    if k == 1:
        neighbor_indices = neighbor_indices.reshape(-1)
        neighbor_distances = neighbor_distances.reshape(-1)

    if single_query and k > 1:
        return neighbor_indices[0], neighbor_distances[0]
    if k == 1:
        return cast(twt.Array1dInt32, neighbor_indices), cast(
            twt.Array1dFloat32, neighbor_distances
        )
    return twt.as_array2d_int32(neighbor_indices), twt.as_array2d_float32(neighbor_distances)


@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    k: int,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: Literal[1] = 1,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
@overload
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    k: int,
    *,
    max_radius: float = ...,
    grid_bins: int = ...,
) -> tuple[twt.Array2dInt32, twt.Array2dFloat32]: ...
def query_hashgrid_nearest(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    max_radius: float = math.inf,
    grid_bins: int = 128,
) -> tuple[twt.Array2dInt32 | twt.Array1dInt32, twt.Array2dFloat32 | twt.Array1dFloat32]:
    """
    For each query center, find the ``k`` nearest data points (HashGrid backend).

    Same semantics as :func:`query_bvh_nearest`. Broad-phase uses
    :class:`warp.HashGrid` with ``wp.hash_grid_query`` out to the effective radius.

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
        Resolution of the hash grid along each axis when constructing the grid.

    Returns
    -------
    indices, distances
        Same layout rules as :func:`query_bvh_nearest`.

    Raises
    ------
    ValueError
        If ``k < 1`` or ``max_radius < 0``.

    See Also
    --------
    query_bvh_nearest
    hashgrid_from_points
    :meth:`scipy.spatial.KDTree.query`
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")

    device = points.device
    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)

    m = int(queries.shape[0])
    n = int(points.shape[0])

    if m == 0:
        return (
            twt.empty_int32_2d((0, k), device=device),
            twt.empty_float32_2d((0, k), device=device),
        )

    neighbor_indices = wp.full((m, k), wp.int32(-1), dtype=wp.int32, device=device)
    neighbor_distances = wp.full((m, k), math.inf, dtype=wp.float32, device=device)
    if n == 0:
        if single_query:
            return neighbor_indices[0], neighbor_distances[0]
        return twt.as_array2d_int32(neighbor_indices), twt.as_array2d_float32(neighbor_distances)

    min_points_bound, max_points_bound = aabb_bounds(points)
    min_queries_bound, max_queries_bound = aabb_bounds(queries)
    min_bound = wp.vec3(
        min(min_points_bound.x, min_queries_bound.x),
        min(min_points_bound.y, min_queries_bound.y),
        min(min_points_bound.z, min_queries_bound.z),
    )
    max_bound = wp.vec3(
        max(max_points_bound.x, max_queries_bound.x),
        max(max_points_bound.y, max_queries_bound.y),
        max(max_points_bound.z, max_queries_bound.z),
    )
    diagonal = wp.length(max_bound - min_bound)
    max_radius = min(max_radius, max(diagonal, 1e-12))

    grid = hashgrid_from_points(points, max_radius, grid_bins)

    wp.launch(
        kernel_proximity.query_hashgrid_nearest_neighbors,
        dim=m,
        inputs=[
            points,
            queries,
            grid.id,
            wp.int32(k),
            wp.float32(max_radius),
            neighbor_indices,
            neighbor_distances,
        ],
        device=device,
    )

    if k == 1:
        neighbor_indices = neighbor_indices.reshape(-1)
        neighbor_distances = neighbor_distances.reshape(-1)

    if single_query and k > 1:
        return neighbor_indices[0], neighbor_distances[0]
    if k == 1:
        return cast(twt.Array1dInt32, neighbor_indices), cast(
            twt.Array1dFloat32, neighbor_distances
        )
    return twt.as_array2d_int32(neighbor_indices), twt.as_array2d_float32(neighbor_distances)


def _aabb_diagonal(min_bound: wp.vec3, max_bound: wp.vec3) -> float:
    return float(wp.length(max_bound - min_bound))


def _combined_aabb(
    a_min: wp.vec3, a_max: wp.vec3, b_min: wp.vec3, b_max: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    combined_min = wp.vec3(
        min(a_min[0], b_min[0]), min(a_min[1], b_min[1]), min(a_min[2], b_min[2])
    )
    combined_max = wp.vec3(
        max(a_max[0], b_max[0]), max(a_max[1], b_max[1]), max(a_max[2], b_max[2])
    )
    return combined_min, combined_max


def mesh_query_max_dist(mesh_points: wp.array[wp.vec3]) -> float:
    """Diagonal of the axis-aligned bounding box of ``mesh_points``."""
    mesh_min, mesh_max = aabb_bounds(mesh_points)
    return _aabb_diagonal(mesh_min, mesh_max)


def default_mesh_query_max_dist(
    mesh_points: wp.array[wp.vec3], query_points: wp.array[wp.vec3]
) -> float:
    """Diagonal of the AABB enclosing ``mesh_points`` and ``query_points``."""
    mesh_min, mesh_max = aabb_bounds(mesh_points)
    if int(query_points.shape[0]) == 0:
        return _aabb_diagonal(mesh_min, mesh_max)
    query_min, query_max = aabb_bounds(query_points)
    combined_min, combined_max = _combined_aabb(mesh_min, mesh_max, query_min, query_max)
    return _aabb_diagonal(combined_min, combined_max)


def closest_point_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    For each query point, find the closest point on any triangle of the mesh.

    Uses ``wp.mesh_query_point_no_sign`` via ``wp.Mesh``. Distances are unsigned
    Euclidean lengths in ``float32``.

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search radius per query. Faces farther than this are ignored.
        When ``None``, derived from the axis-aligned box enclosing mesh
        vertices and query points.

    Returns
    -------
    closest
        ``(m, 3)`` closest point on the mesh surface for each query.
    distance
        ``(m,)`` unsigned distance from each query to its closest surface point.
    triangle_id
        ``(m,)`` index of the triangle containing each closest point, or ``-1``
        when no face lies within ``max_dist``.
    """
    device = vertices.device
    m = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if m == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    if n_faces == 0:
        nan_closest_np = np.full((m, 3), np.nan, dtype=np.float32)
        out_closest = wp.array(nan_closest_np, dtype=wp.vec3, device=device)
        out_distance = wp.full(m, float("inf"), dtype=wp.float32, device=device)
        out_face = wp.full(m, -1, dtype=wp.int32, device=device)
        return out_closest, out_distance, out_face

    mesh = wp.Mesh(points=wp.clone(vertices), indices=wp.clone(faces))
    if max_dist is None:
        max_dist = default_mesh_query_max_dist(mesh.points, points)

    out_closest = wp.empty(m, dtype=wp.vec3, device=device)
    out_distance = wp.empty(m, dtype=wp.float32, device=device)
    out_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.closest_point_on_mesh,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), out_closest, out_distance, out_face],
        device=device,
    )
    return out_closest, out_distance, out_face


def normals_at_closest_faces(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, max_dist: float | None = None
) -> wp.array[wp.vec3]:
    """
    Return unit face normals at the closest mesh triangle for each query point.

    For each position in ``points``, runs an unsigned closest-point query on
    ``mesh`` and returns the normal of the hit triangle. When no face lies
    within ``max_dist``, the corresponding output is undefined (same as the
    underlying ``wp.mesh_query_point_no_sign`` miss case).

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` query positions as ``wp.vec3``.
    max_dist
        Maximum search radius per query. When ``None``, derived from the
        axis-aligned box enclosing mesh vertices and query points.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` face normals at the closest triangle for each query.
    """
    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.vec3, device=device)

    if max_dist is None:
        max_dist = default_mesh_query_max_dist(mesh.points, points)

    out_closest = wp.empty(m, dtype=wp.vec3, device=device)
    out_dist = wp.empty(m, dtype=wp.float32, device=device)
    out_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.closest_point_on_mesh,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), out_closest, out_dist, out_face],
        device=device,
    )
    all_face_normals, _ = face_normals_and_areas(mesh.points, mesh.indices)
    normals = wp.empty(m, dtype=wp.vec3, device=device)
    wp.copy(normals, all_face_normals[out_face])
    return normals


def signed_distance_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    n_sample: int = 5,
    perturbation_scale: float = 0.1,
) -> wp.array[wp.float32]:
    """
    Signed distance from each query point to a triangle mesh (Warp SDF convention).

    Uses ``wp.mesh_query_point_sign_parity`` via ``wp.Mesh``. Distances follow
    Warp's signed-distance field convention:

    * Points **outside** the mesh have **positive** distance.
    * Points **inside** have **negative** distance.
    * Points within :data:`triwarp.constants.TOLERANCE_MERGE` of the surface
      return positive unsigned distance.

    Trimesh ``signed_distance`` uses the opposite sign; negate its output to compare.
    See also :func:`triwarp.ray.contains_points` (inside iff signed distance is
    negative, except on the on-surface tolerance band).

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search radius per query. When ``None``, derived from the
        axis-aligned box enclosing mesh vertices and query points.
    n_sample
        Perturbed rays for parity voting (off-triangle sign branch).
    perturbation_scale
        Uniform perturbation scale for parity rays.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` signed distances in ``float32``.
    """
    device = vertices.device
    m = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)
    if n_faces == 0:
        return wp.full(m, float("inf"), dtype=wp.float32, device=device)

    mesh = wp.Mesh(points=wp.clone(vertices), indices=wp.clone(faces))
    if max_dist is None:
        max_dist = default_mesh_query_max_dist(mesh.points, points)
    out_distance = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_proximity.signed_distance_on_mesh,
        dim=m,
        inputs=[
            mesh.id,
            points,
            wp.float32(max_dist),
            wp.int32(n_sample),
            wp.float32(perturbation_scale),
            out_distance,
        ],
        device=device,
    )
    return out_distance


def max_tangent_sphere(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    inwards: bool = True,
    normals: wp.array[wp.vec3] | None = None,
    threshold: float = 1e-6,
    max_iter: int = 100,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    Find the center and radius of the sphere tangent to the mesh at each point.

    Implements the shrinking-sphere algorithm (Inui et al. 2016): iteratively
    finds the largest sphere tangent to the mesh at ``points`` with no
    non-tangential intersections.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` surface points as ``wp.vec3``.
    inwards
        If ``True``, sphere grows inward (into the mesh interior). If ``False``,
        grows outward.
    normals
        ``(m,)`` unit surface normals at ``points``. If ``None``, computed from
        the closest triangle.
    threshold
        Convergence threshold as a fraction of the scene diagonal.
    max_iter
        Maximum number of shrink iterations.

    Returns
    -------
    centers
        ``(m,)`` sphere center positions as ``wp.vec3``.
    radii
        ``(m,)`` sphere radii as ``float32``. ``inf`` when the sphere is
        unbounded.
    """
    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
        )

    if normals is None:
        normals = normals_at_closest_faces(mesh, points)

    ray_dirs: wp.array[wp.vec3] = -normals if inwards else normals

    max_t = default_mesh_query_max_dist(mesh.points, points)
    distances = tw.ray.longest_ray(mesh, points, ray_dirs, max_t=max_t)

    n_verts = int(mesh.points.shape[0])
    radii = wp.empty(m, dtype=wp.float32, device=device)
    not_converged = wp.empty(m, dtype=wp.bool, device=device)
    wp.launch(
        kernel_proximity.init_sphere_radii,
        dim=m,
        inputs=[mesh.points, wp.int32(n_verts), points, ray_dirs, distances, radii, not_converged],
        device=device,
    )

    centers = wp.empty(m, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_proximity.compute_sphere_centers,
        dim=m,
        inputs=[points, ray_dirs, radii, centers],
        device=device,
    )

    mesh_min, mesh_max = aabb_bounds(mesh.points)
    D = float(wp.length(mesh_max - mesh_min))
    convergence_threshold = wp.float32(threshold * D)

    n_iter = 0
    while n_iter < max_iter:
        count_wp = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_proximity.count_true, dim=m, inputs=[not_converged, count_wp], device=device
        )
        if int(count_wp.numpy()[0]) == 0:
            break

        n_pts_wp = wp.empty(m, dtype=wp.vec3, device=device)
        n_dists_wp = wp.empty(m, dtype=wp.float32, device=device)
        n_face_wp = wp.empty(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_proximity.closest_point_on_mesh,
            dim=m,
            inputs=[mesh.id, centers, wp.float32(max_t), n_pts_wp, n_dists_wp, n_face_wp],
            device=device,
        )

        new_radii = wp.clone(radii)
        new_centers = wp.clone(centers)
        new_nc = wp.clone(not_converged)
        wp.launch(
            kernel_proximity.step_sphere_shrink,
            dim=m,
            inputs=[
                points,
                ray_dirs,
                n_pts_wp,
                n_dists_wp,
                centers,
                radii,
                convergence_threshold,
                not_converged,
                new_radii,
                new_centers,
                new_nc,
            ],
            device=device,
        )
        radii = new_radii
        centers = new_centers
        not_converged = new_nc
        n_iter += 1

    return centers, radii


def thickness(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    exterior: bool = False,
    normals: wp.array[wp.vec3] | None = None,
    method: Literal["max_sphere", "ray"] = "max_sphere",
) -> wp.array[wp.float32]:
    """
    Thickness of the mesh at each point.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` surface points as ``wp.vec3``.
    exterior
        If ``True``, compute exterior thickness (reach). If ``False``, interior.
    normals
        ``(m,)`` unit surface normals. If ``None``, computed automatically.
    method
        ``"max_sphere"`` (default) or ``"ray"``.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` thickness values. ``inf`` for unbounded.
    """
    if method == "max_sphere":
        _centers, radii = max_tangent_sphere(mesh, points, inwards=not exterior, normals=normals)
        return radii * wp.float32(2.0)

    elif method == "ray":
        if normals is None:
            normals = normals_at_closest_faces(mesh, points)

        ray_dirs = normals if exterior else -normals
        max_t = default_mesh_query_max_dist(mesh.points, points)
        return tw.ray.longest_ray(mesh, points, ray_dirs, max_t=max_t)

    else:
        raise ValueError('Invalid method, use "max_sphere" or "ray"')
