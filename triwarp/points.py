from __future__ import annotations

from typing import overload

import warp as wp

from triwarp.kernels import points as kernel_points


def bvh_from_points(points: wp.array[wp.vec3], leaf_size: int) -> wp.Bvh:
    return wp.Bvh(points, wp.clone(points), leaf_size=leaf_size)


def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    r: float,
    *,
    bvh: wp.Bvh | None = None,
    leaf_size: int = 4,
) -> wp.array[wp.int32]:
    """
    Indices of ``points`` within Euclidean distance ``r`` of each query center,
    analogous to :meth:`scipy.spatial.KDTree.query_ball_point` (``p=2``, exact).

    Uses the same BVH broad-phase cube ``[center ± r]`` and ``float32`` narrow-phase
    distance test as :func:`remove_close`.

    Parameters
    ----------
    points
        ``(n, 3)`` points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query points stored as ``wp.vec3``.
    r
        Query radius (cast to ``float32`` in kernels).
    bvh
        Optional BVH built from ``points``.
    leaf_size
        Passed to :class:`warp.Bvh`.

    Returns
    -------
    ``(m,)`` array of neighbor counts.
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
    Indices of ``points`` within Euclidean distance ``r`` of each query center,
    analogous to :meth:`scipy.spatial.KDTree.query_ball_point` (``p=2``, exact).

    Uses the same BVH broad-phase cube ``[center ± r]`` and ``float32`` narrow-phase
    distance test as :func:`remove_close`.

    Parameters
    ----------
    points
        ``(n, 3)`` points stored as ``wp.vec3``.
    queries
        ``(m, 3)`` query points stored as ``wp.vec3``.
    r
        Query radius (cast to ``float32`` in kernels).
    bvh
        Optional BVH built from ``points``.
    leaf_size
        Passed to :class:`warp.Bvh`.
    return_sorted
        If True, each neighbor list is sorted by ascending distance. If False, traversal order.
        Note that the returned indices are not sorted by ascending index as in the SciPy implementation.

    Returns
    -------
    List of length ``m`` with ``(k,)`` array of neighbor indices for each query.
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
        print(
            "Indices flat",
            neighbor_indices_flat.numpy(),
            "Distances flat",
            neighbor_distances_flat.numpy(),
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


"""
def _mask_from_close_pairs(pairs: np.ndarray, n: int) -> np.ndarray:
    ""Trimesh-compatible greedy mask from unique pairs (i, j) with i < j.""
    if pairs.size == 0:
        return np.ones(n, dtype=bool)
    count = np.bincount(pairs.ravel(), minlength=n)
    column = count[pairs].argmax(axis=1)
    highest = pairs.ravel()[column + 2 * np.arange(len(column), dtype=np.intp)]
    mask = np.ones(n, dtype=bool)
    mask[highest] = False
    return mask


def remove_close(
    points: wp.array[wp.vec3],
    radius: float,
    *,
    leaf_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    ""
    Return a subset of 3-D points where no two kept points have Euclidean distance
    at most ``radius``, using the same greedy rule as :func:`trimesh.points.remove_close`.

    Broad-phase search uses a Warp :class:`warp.Bvh` over degenerate point bounds; each
    query is an axis-aligned cube of half-extent ``radius``. Narrow-phase filters with
    squared distance in ``float32``.

    Parameters
    ----------
    points
        ``(n, 3)`` positions as ``wp.vec3`` (typically ``float32`` components).
    radius
        Maximum distance defining close pairs (passed as ``float32`` to kernels).
    leaf_size
        BVH leaf size (see Warp ``Bvh`` documentation).

    Returns
    -------
    culled
        ``(m, 3)`` float array of kept points (same dtype as ``points.numpy()``).
    mask
        ``(n,)`` boolean mask into the original ordering.
    ""
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.ones(0, dtype=bool)

    bvh = bvh_from_points(points, leaf_size)

    pair_count = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_points.count_close_pairs,
        dim=n,
        inputs=[points, bvh.id, wp.float32(radius), pair_count],
        device=device,
    )

    counts = pair_count.numpy()
    total_pairs = int(counts.sum())
    if total_pairs == 0:
        mask = np.ones(n, dtype=bool)
        pts_np = points.numpy()
        return pts_np, mask

    offsets_np = np.zeros(n, dtype=np.int32)
    if n > 1:
        offsets_np[1:] = np.cumsum(counts[:-1])
    offsets = wp.array(offsets_np, dtype=wp.int32, device=device)

    pairs_a = wp.empty(total_pairs, dtype=wp.int32, device=device)
    pairs_b = wp.empty(total_pairs, dtype=wp.int32, device=device)
    wp.launch(
        kernel_points.fill_close_pairs,
        dim=n,
        inputs=[points, bvh.id, wp.float32(radius), offsets, pairs_a, pairs_b],
        device=device,
    )

    pairs = np.column_stack((pairs_a.numpy(), pairs_b.numpy()))
    mask = _mask_from_close_pairs(pairs, n)
    pts_np = points.numpy()
    return pts_np[mask], mask
"""
