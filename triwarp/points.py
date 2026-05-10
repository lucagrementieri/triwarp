from __future__ import annotations

import math
from typing import Literal, overload

import warp as wp

from triwarp.kernels import points as kernel_points


def aabb_bounds(points: wp.array[wp.vec3]) -> float:
    out_min = wp.full(3, math.inf, dtype=wp.float32, device=points.device)
    out_max = wp.full(3, -math.inf, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_points.aabb_bounds,
        dim=points.shape[0],
        inputs=[points, out_min, out_max],
        device=points.device,
    )
    out_min = out_min.list()
    out_max = out_max.list()
    min_bound = wp.vec3(out_min[0], out_min[1], out_min[2])
    max_bound = wp.vec3(out_max[0], out_max[1], out_max[2])
    return min_bound, max_bound


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int) -> wp.Bvh:
    """
    Build a bounding-volume hierarchy over ``points`` for radius queries.

    Each leaf stores the same geometry as ``points`` (degenerate bounds via a clone),
    matching the broad-phase pattern used by :func:`remove_close` and the
    ``query_ball*`` helpers.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3``.
    leaf_size
        Maximum primitives per leaf; forwarded to :class:`warp.Bvh`.

    Returns
    -------
    warp.Bvh
        BVH suited for ``query_ball_count``, ``query_ball``, and related kernels.

    See Also
    --------
    query_ball_count
    query_ball
    query
    remove_close
    """
    return wp.Bvh(points, points, leaf_size=leaf_size)


def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
) -> wp.array[wp.int32]:
    """
    Count neighbors of each query within Euclidean distance ``r``.

    For each query center ``q``, returns how many entries ``p`` in ``points`` satisfy
    ``‖p - q‖₂ ≤ r``. This matches :meth:`scipy.spatial.KDTree.query_ball_point` with
    ``p=2``, ``eps=0``, and ``return_length=True`` (exact search; only the spatial
    index differs).

    Broad-phase traversal uses an axis-aligned cube ``[q - r, q + r]`` against the BVH;
    narrow-phase keeps points with Euclidean distance at most ``r`` (``float32``), the same
    geometric radius as :func:`remove_close` (which tests ``‖·‖² ≤ r²`` in the kernel).

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
        Length-``m`` device array whose ``k``-th element is the neighbor count for
        ``queries[k]``. If ``n == 0``, returns zeros.

    See Also
    --------
    query_ball
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
        kernel_points.query_ball_count,
        dim=m,
        inputs=[points, queries, bvh.id, wp.float32(r), neighbor_counts],
        device=device,
    )

    return neighbor_counts


@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
) -> tuple[list[wp.array[wp.int32]], list[wp.array[wp.float32]]]: ...
@overload
def query_ball(
    points: wp.array[wp.vec3],
    queries: wp.vec3,
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
    return_sorted: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.float32]]: ...
def query_ball(
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
    Find all data points within distance ``r`` of each query center.

    For each query ``q``, returns indices ``i`` such that ``‖points[i] − q‖₂ ≤ r`` and
    the corresponding distances. Semantics match :meth:`scipy.spatial.KDTree.query_ball_point`
    with ``p=2`` and ``eps=0`` (Minkowski-2, exact). Unlike SciPy, results are returned as
    separate Warp arrays per query (or a single pair for one query), not Python lists inside
    an object array.

    Uses the same BVH broad-phase cube ``[center ± r]`` as :func:`remove_close` and a
    ``float32`` Euclidean distance check ``‖points[i] - q‖₂ ≤ r``.

    Parameters
    ----------
    points
        ``(n, 3)`` data points stored as ``wp.vec3``.
    queries
        Either ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or a single ``wp.vec3``
        (treated as one query).
    r
        Inclusion radius; cast to ``float32`` in kernels (non-negative).
    bvh
        Optional pre-built BVH from ``points``. If ``None``, built via
        :func:`bvh_from_points`.
    leaf_size
        Leaf size when constructing ``bvh`` (ignored if ``bvh`` is provided).
    return_sorted
        If ``True``, neighbors within each query are ordered by increasing distance.
        If ``False``, order follows tree traversal (undefined ordering).

    Returns
    -------
    neighbor_indices, neighbor_distances
        If ``queries`` is an array with ``m`` rows: two lists of length ``m``. Element ``k``
        holds parallel rank-1 ``wp.array`` objects (dtype ``wp.int32`` and ``wp.float32``)
        listing neighbor indices into ``points`` and their distances ``‖points[i] − q‖₂``
        for ``queries[k]``.

        If ``queries`` is a single ``wp.vec3``: returns one pair ``(indices, distances)``
        as two rank-1 arrays (possibly length 0).

        Empty inputs yield empty arrays; duplicate neighbors are not produced.

    Notes
    -----
    SciPy may sort indices when ``return_sorted`` is left default on multi-point queries;
    here sorting only occurs when ``return_sorted=True``, and sorts by distance, not by
    index. Ball boundaries use ``float32`` arithmetic; extremely tight radii near representable
    limits may disagree slightly with pure ``float64`` SciPy runs.

    See Also
    --------
    query_ball_count
    bvh_from_points
    :meth:`scipy.spatial.KDTree.query_ball_point`
    """
    device = points.device

    single_query = isinstance(queries, wp.vec3)
    if single_query:
        queries = wp.array([queries], dtype=wp.vec3, device=device)
    m = int(queries.shape[0])

    n: int = int(points.shape[0])
    if n == 0:
        return (
            (
                wp.empty(0, dtype=wp.int32, device=device),
                wp.empty(0, dtype=wp.float32, device=device),
            )
            if single_query
            else (
                [wp.empty(0, dtype=wp.int32, device=device) for _ in range(m)],
                [wp.empty(0, dtype=wp.float32, device=device) for _ in range(m)],
            )
        )

    if bvh is None:
        bvh = bvh_from_points(points, leaf_size)

    neighbor_counts = query_ball_count(points, queries, r, bvh=bvh)
    total_neighbors = int(neighbor_counts.numpy().sum())

    if total_neighbors == 0:
        return (
            (
                wp.empty(0, dtype=wp.int32, device=device),
                wp.empty(0, dtype=wp.float32, device=device),
            )
            if single_query
            else (
                [wp.empty(0, dtype=wp.int32, device=device) for _ in range(m)],
                [wp.empty(0, dtype=wp.float32, device=device) for _ in range(m)],
            )
        )

    offsets = wp.empty(m, dtype=wp.int32, device=device)
    wp.utils.array_scan(neighbor_counts, out_array=offsets, inclusive=False)

    flat_len = total_neighbors * (2 if return_sorted else 1)
    neighbor_indices_flat = wp.empty(flat_len, dtype=wp.int32, device=device)
    neighbor_distances_flat = wp.empty(flat_len, dtype=wp.float32, device=device)
    wp.launch(
        kernel_points.query_ball_neighbors,
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
            neighbor_distances_flat,
            neighbor_indices_flat,
            total_neighbors,
            segment_bounds,
        )

    neighbor_indices: list[wp.array[wp.int32]] = []
    neighbor_distances: list[wp.array[wp.float32]] = []

    offsets_list = offsets.list()
    for k in range(m):
        start = offsets_list[k]
        end = offsets_list[k + 1] if k < m - 1 else total_neighbors
        neighbor_indices.append(wp.clone(neighbor_indices_flat[start:end]))
        neighbor_distances.append(wp.clone(neighbor_distances_flat[start:end]))

    if single_query:
        return neighbor_indices[0], neighbor_distances[0]
    return neighbor_indices, neighbor_distances


def query(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3] | wp.vec3,
    k: int = 1,
    *,
    max_radius: float = math.inf,
    grid_bins: int = 128,
) -> tuple[wp.array2d[wp.int32], wp.array2d[wp.float32]]:  # TODO fix types for k =1
    """
    For each query center, find the ``k`` nearest points in Euclidean distance.

    Semantics follow :meth:`scipy.spatial.KDTree.query` with ``p=2`` and ``eps=0`` (exact
    Minkowski-2). Distances use ``float32`` (``wp.length``), so results may differ slightly
    from a ``float64`` SciPy tree on the same inputs. Missing neighbors (too few points,
    or ``distance_upper_bound`` too tight) use ``inf`` distance and a sentinel index: ``n``
    when ``n > 0``, otherwise ``0``, matching SciPy.

    Neighbor search uses a :class:`warp.HashGrid` spatial hash: points are bucketed with
    cell size matching the search radius, and each query scans candidates within that
    radius (from ``distance_upper_bound`` when finite, otherwise within the bounding box
    of ``points`` and ``queries``). ``k`` nearest candidates are maintained per query in
    sorted order (same semantics as before).

    Parameters
    ----------
    points
        ``(n, 3)`` data points as ``wp.vec3``.
    queries
        ``(m, 3)`` query centers as ``wp.array[wp.vec3]``, or one ``wp.vec3``.
    k
        Number of neighbors (``1 <= k <= 32``).
    max_radius
        Only neighbors within this Euclidean distance are considered; forwarded as
        ``float32``. Use ``math.inf`` for no bound.

    Returns
    -------
    distances, indices
        ``wp.array[wp.float32]`` and ``wp.array[wp.int32]`` with the same length:

        * ``queries`` array, ``k > 1``: length ``m * k`` (row-major: query ``q``,
          neighbor ``t`` at ``distances[q * k + t]``), sorted by increasing distance per query.
        * ``queries`` array, ``k == 1``: length ``m`` (nearest distance / index per query).
        * Single ``wp.vec3``: length ``k``.

        If ``m == 0``, returns empty arrays. If ``n == 0``, all distances are ``inf`` and
        indices are ``0`` (SciPy behavior), with the same shapes as when ``n > 0``.

    Raises
    ------
    ValueError
        If ``k`` is out of range.

    See Also
    --------
    query_ball
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
            wp.empty((0, k), dtype=wp.int32, device=device),
            wp.empty((0, k), dtype=wp.float32, device=device),
        )

    neighbor_indices = wp.full((m, k), wp.int32(-1), dtype=wp.int32, device=device)
    neighbor_distances = wp.full((m, k), math.inf, dtype=wp.float32, device=device)
    if n == 0:
        if single_query:
            return neighbor_indices[0], neighbor_distances[0]
        return neighbor_indices, neighbor_distances

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

    grid = wp.HashGrid(grid_bins, grid_bins, grid_bins, device=device)
    grid.reserve(n)
    grid.build(points, max_radius)

    wp.launch(
        kernel_points.query_nearest_neighbors,
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
    return neighbor_indices, neighbor_distances
