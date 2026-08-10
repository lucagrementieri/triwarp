import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.algorithms import bfs as kernel_bfs

# Iterative-deepening k-nearest search. A scan at cube half-extent ``r`` enumerates every point
# within Euclidean distance ``r`` (Chebyshev distance never exceeds Euclidean), so a row whose
# k-th distance is at most ``r`` is provably the exact k-NN and the loop can stop.
#
# The loop is a bounded ``for``, never a ``while``: with a NaN query every comparison is false and
# ``r * RADIUS_GROWTH`` stays NaN, which would hang the device. The last attempt is forced to the
# complete radius, which makes termination unconditional and the result exact regardless.
#
# The attempt budget is generous because growth is geometric: the doublings before the successful
# scan sum to less than that scan costs, so a spare attempt is nearly free — whereas running *out*
# of attempts forces the complete scan, which is the whole-cloud brute force this change exists to
# avoid. 16 doublings cover a query 3 x 10^4 spacings away from the cloud.
MAX_SEARCH_ATTEMPTS = wp.constant(wp.int32(16))
RADIUS_GROWTH = wp.constant(wp.float32(2.0))

# Candidate-row sizes the register-resident k-NN kernels are generated for. The row is held in a
# ``wp.types.vector(length=K)`` value type, i.e. in registers, so ``K`` must be a compile-time
# constant and one kernel exists per bucket. A query takes the smallest bucket that fits its ``k``,
# keeps the ``K`` nearest (a superset of the ``k`` nearest) and writes out only the first ``k``
# slots; a ``k`` past the largest bucket falls back to the global-memory row kernels.
#
# Measured on an RTX 5090 against the global-memory row, bunny / 20 000 queries: 1.09x at k=1,
# 1.5x at k=7, 1.9x at k=16, 2.9x at k=30, 7.8x at k=64. 64 is the last bucket because that is
# where the curve turns, not because the gain runs out: a 96-wide row spills (2 x K registers) and
# drops back to 1.66x, so a bucket past 64 would buy little and no in-repo caller asks for one.
#
# The buckets are not free: they cost ``2 x len(KNN_ROW_BUCKETS)`` generated kernels, which take
# this module's cold-cache compile from 3.9 s to ~12 s (once per Warp version and arch) and its
# warm per-process load from 2.4 ms to ~5 ms.
KNN_ROW_BUCKETS = (1, 4, 8, 16, 32, 64)


@wp.func
def bvh_aabb_collect(
    bvh_id: wp.uint64,
    q: wp.vec3,
    half_extent: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) the BVH hits around ``q``.
    h = half_extent
    lower = q - wp.vec3(h)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(h)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        if write:
            out_indices[base + c] = j
        c = c + 1
    return c


@wp.kernel
def query_bvh_aabb_count(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    # ``out_counts`` doubles as the (never written) emit target of the counting pass.
    out_counts[tid] = bvh_aabb_collect(
        bvh_id, queries[tid], half_extent, wp.bool(False), wp.int32(0), out_counts
    )


@wp.kernel
def query_bvh_aabb_neighbors(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    half_extent: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    bvh_aabb_collect(bvh_id, queries[tid], half_extent, wp.bool(True), offsets[tid], out_indices)


@wp.func
def hashgrid_ball_collect(
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) points within ``radius``.
    query = wp.hash_grid_query(grid_id, q, radius)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        d = wp.length(points[j] - q)
        if d <= radius:
            if write:
                out_indices[base + c] = j
                out_distances[base + c] = d
            c = c + 1
    return c


@wp.kernel
def query_hashgrid_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    dummy_dist = wp.zeros(shape=1, dtype=wp.float32)
    out_neighbor_counts[tid] = hashgrid_ball_collect(
        points,
        grid_id,
        queries[tid],
        radius,
        wp.bool(False),
        wp.int32(0),
        out_neighbor_counts,
        dummy_dist,
    )


@wp.kernel
def query_hashgrid_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    hashgrid_ball_collect(
        points,
        grid_id,
        queries[tid],
        radius,
        wp.bool(True),
        offsets[tid],
        out_indices,
        out_distances,
    )


@wp.func
def bvh_ball_collect(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) points within ``radius``.
    lower = q - wp.vec3(radius)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(radius)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        d = wp.length(points[j] - q)
        if d <= radius:
            if write:
                out_indices[base + c] = j
                out_distances[base + c] = d
            c = c + 1
    return c


@wp.kernel
def query_bvh_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    dummy_dist = wp.zeros(shape=1, dtype=wp.float32)
    out_neighbor_counts[tid] = bvh_ball_collect(
        points,
        bvh_id,
        queries[tid],
        radius,
        wp.bool(False),
        wp.int32(0),
        out_neighbor_counts,
        dummy_dist,
    )


@wp.kernel
def query_bvh_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    bvh_ball_collect(
        points,
        bvh_id,
        queries[tid],
        radius,
        wp.bool(True),
        offsets[tid],
        out_indices,
        out_distances,
    )


@wp.func
def knn_sorted_insert(
    point_index: wp.int32,
    d: wp.float32,
    k: wp.int32,
    radius: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> None:
    # Insert ``(point_index, d)`` into the ascending k-nearest rows, dropping the current worst.
    if d > radius:
        return
    if d >= out_distances_row[k - 1]:
        return
    if k == 1:
        out_indices_row[0] = point_index
        out_distances_row[0] = d
        return
    slot = kernel_array.binary_search_index(out_distances_row, d)
    kernel_array.array_shift_insert(out_distances_row, d, slot)
    kernel_array.array_shift_insert(out_indices_row, point_index, slot)


@wp.func
def knn_reset_row(
    k: wp.int32, out_indices_row: wp.array[wp.int32], out_distances_row: wp.array[wp.float32]
) -> None:
    # Every scan starts from an empty row. ``knn_sorted_insert`` does not deduplicate, so a
    # re-scan over a wider radius would otherwise insert each already-found point a second time.
    for i in range(k):
        out_indices_row[i] = wp.int32(-1)
        out_distances_row[i] = FLOAT32_INF_CONSTANT


@wp.func
def complete_radius(q: wp.vec3, min_bound: wp.vec3, max_bound: wp.vec3) -> wp.float32:
    # Smallest cube half-extent about ``q`` that contains the whole point bounding box, i.e. the
    # radius at which a scan is provably complete. Per-query, so it is tighter than a global
    # diagonal, and unbounded for a query far outside the box (which is what keeps that case exact).
    lower = q - min_bound
    upper = max_bound - q
    r = wp.max(lower[0], upper[0])
    r = wp.max(r, wp.max(lower[1], upper[1]))
    return wp.max(r, wp.max(lower[2], upper[2]))


@wp.func
def knn_bvh_scan(
    points: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    r: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> wp.float32:
    # Refill the row from the cube ``[q +/- r]`` and return the k-th best distance (``inf`` when
    # fewer than ``k`` points were accepted). Acceptance stays ``d <= max_radius``; ``r`` bounds
    # only the enumeration.
    knn_reset_row(k, out_indices_row, out_distances_row)
    lower = q - wp.vec3(r)  # wp.vec3(scalar) broadcasts the scalar to every component
    upper = q + wp.vec3(r)
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root=-1)
    point_index = wp.int32(0)
    while wp.bvh_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)
    return out_distances_row[k - 1]


@wp.kernel
def query_bvh_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    k: wp.int32,
    max_radius: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    out_indices_row = out_indices[tid]
    out_distances_row = out_distances[tid]

    r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
    r = wp.min(initial_radius, r_hard)
    # Exactly one ``wp.bvh_query_aabb`` call site in this kernel: ``bvh_query`` declares
    # ``__shared__ int stack[32 * WP_TILE_BLOCK_DIM]`` (32 KB at block_dim=256), so a second
    # textual call site would ask for 64 KB and fail to compile.
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        if attempt == MAX_SEARCH_ATTEMPTS - 1:
            r = r_hard  # forced-complete final attempt: exact whatever the growth did
        worst = knn_bvh_scan(
            points, bvh_id, q, k, max_radius, r, out_indices_row, out_distances_row
        )
        if worst <= r:
            break  # every point outside the cube is farther than the k-th best: certified exact
        if r >= r_hard:
            break  # the scan was already complete, so the row is final
        if worst < FLOAT32_INF_CONSTANT:
            # The row is full but reaches past the cube. Re-scanning at exactly ``worst`` is
            # guaranteed to certify, so this costs at most one more pass.
            r = wp.min(worst, r_hard)
        else:
            r = wp.min(r * RADIUS_GROWTH, r_hard)


# ---------------------------------------------------------------------------------------------
# Register-row k-NN kernel factories.
#
# ``query_bvh_nearest_neighbors`` above keeps its candidate row in the *output* arrays, so every
# accepted candidate pays a binary search plus two shift passes over global memory, and the row is
# re-zeroed there on every deepening attempt. Measured on bunny at k=32: 225 candidates enumerated
# per query against 2 305 shifted row elements, i.e. the row traffic — not the geometry — is the
# call. Holding the row in a ``wp.types.vector(length=K)`` value type puts it in registers, which
# is why these kernels exist and why ``K`` has to be a compile-time constant.
#
# The insert is written out inline rather than factored into a ``@wp.func``: passing a vector to a
# helper takes its address, which forces the row back into local memory and gives up the whole
# point. A ``wp.zeros(shape=K)`` stack array has the same problem — measured a **2x loss** against
# the global row, because nothing promotes it to registers.
#
# The contract the copies must hold is the *distance* row, not the tie-break. Like the shipped
# kernel's ``binary_search_index`` (``searchsorted(side="right")``), the carry below walks past
# equals before displacing, and ``placed`` then shifts the rest of the row down mechanically — a
# plain stable insertion, which is why it matches the global row index-for-index. But that match is
# incidental and ``placed`` is **not** load-bearing: without it the carry skips a slot holding an
# equal distance and displaces further down instead, which permutes *which* of several equidistant
# points fills a slot and leaves every distance bit-identical (measured — deleting the flag from all
# three carries passes the tie test at every bucket on both backends). Callers are told exactly that
# much; ``query_bvh_nearest``'s docstring declares the identity of a tied neighbour unspecified.
#
# So the guard on an edit here is a *tied* fixture, not a second implementation to diff against:
# ``tests/test_neighbors.py::test_query_nearest_ties`` runs every ``KNN_ROW_BUCKETS`` size against
# ``scipy.spatial.KDTree`` on an integer lattice, where a query has dozens of exactly tied
# neighbours and a carry that mishandles the run keeps a farther point. Verified as a live gate:
# shortening the shift chain by one slot fails all 12 cases. Edit the carry and run it.
# ---------------------------------------------------------------------------------------------


def _bvh_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_bvh_nearest_neighbors``."""
    vec_distances = wp.types.vector(length=row_size, dtype=wp.float32)
    vec_indices = wp.types.vector(length=row_size, dtype=wp.int32)

    def _kernel(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        bvh_id: wp.uint64,
        k: wp.int32,
        max_radius: wp.float32,
        initial_radius: wp.float32,
        min_bound: wp.vec3,
        max_bound: wp.vec3,
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        tid = wp.tid()
        q = queries[tid]
        row_indices = vec_indices()
        row_distances = vec_distances()

        r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
        r = wp.min(initial_radius, r_hard)
        # Exactly one ``wp.bvh_query_aabb`` call site, for the 32 KB shared-memory reason given on
        # ``query_bvh_nearest_neighbors``.
        for attempt in range(MAX_SEARCH_ATTEMPTS):
            if attempt == MAX_SEARCH_ATTEMPTS - 1:
                r = r_hard  # forced-complete final attempt: exact whatever the growth did
            for slot in range(row_size):
                row_indices[slot] = wp.int32(-1)
                row_distances[slot] = FLOAT32_INF_CONSTANT
            query = wp.bvh_query_aabb(bvh_id, q - wp.vec3(r), q + wp.vec3(r), root=-1)
            point_index = wp.int32(0)
            while wp.bvh_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            # The k-th distance certifies the scan. Read through an unrolled compare rather than
            # ``row_distances[k - 1]``: a runtime index into a vector spills it to local memory.
            worst = FLOAT32_INF_CONSTANT
            for slot in range(row_size):
                if slot == k - 1:
                    worst = row_distances[slot]
            if worst <= r:
                break  # every point outside the cube is farther than the k-th best: exact
            if r >= r_hard:
                break  # the scan was already complete, so the row is final
            if worst < FLOAT32_INF_CONSTANT:
                # The row is full but reaches past the cube. Re-scanning at exactly ``worst`` is
                # guaranteed to certify, so this costs at most one more pass.
                r = wp.min(worst, r_hard)
            else:
                r = wp.min(r * RADIUS_GROWTH, r_hard)

        for slot in range(row_size):
            if slot < k:
                out_indices[tid, slot] = row_indices[slot]
                out_distances[tid, slot] = row_distances[slot]

    _kernel.__name__ = name
    _kernel.__qualname__ = name
    return wp.kernel(_kernel)


_BVH_NEAREST_ROW_KERNELS = {
    row_size: _bvh_nearest_row_kernel(row_size, f"query_bvh_nearest_neighbors_row{row_size}")
    for row_size in KNN_ROW_BUCKETS
}


def bvh_nearest_kernel(k: int) -> wp.Kernel:
    """Register-row BVH k-NN kernel for the smallest bucket fitting ``k``, else global-row."""
    for row_size in KNN_ROW_BUCKETS:
        if k <= row_size:
            return _BVH_NEAREST_ROW_KERNELS[row_size]
    return query_bvh_nearest_neighbors


@wp.func
def knn_hashgrid_scan(
    points: wp.array[wp.vec3],
    grid_id: wp.uint64,
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    r: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> wp.float32:
    # Hash-grid twin of ``knn_bvh_scan``; ``wp.hash_grid_query`` enumerates every cell overlapping
    # ``[q +/- r]``, so the same "k-th distance <= r certifies" argument applies.
    knn_reset_row(k, out_indices_row, out_distances_row)
    query = wp.hash_grid_query(grid_id, q, r)
    point_index = wp.int32(-1)
    while wp.hash_grid_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)
    return out_distances_row[k - 1]


@wp.func
def knn_linear_scan(
    points: wp.array[wp.vec3],
    q: wp.vec3,
    k: wp.int32,
    max_radius: wp.float32,
    out_indices_row: wp.array[wp.int32],
    out_distances_row: wp.array[wp.float32],
) -> None:
    # Exact fallback for the grid path once the radius outgrows the cell width. This is the same
    # per-row cost the diagonal-radius query used to pay for *every* row, so it is never a
    # regression against the previous behaviour.
    knn_reset_row(k, out_indices_row, out_distances_row)
    for point_index in range(points.shape[0]):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)


@wp.kernel
def query_hashgrid_nearest_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    grid_id: wp.uint64,
    k: wp.int32,
    max_radius: wp.float32,
    initial_radius: wp.float32,
    widest: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_indices: wp.array2d[wp.int32],
    out_distances: wp.array2d[wp.float32],
) -> None:
    tid = wp.tid()
    q = queries[tid]
    out_indices_row = out_indices[tid]
    out_distances_row = out_distances[tid]

    r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
    r = wp.min(initial_radius, r_hard)
    for _attempt in range(MAX_SEARCH_ATTEMPTS):
        if not r <= widest:
            # Past ``widest`` a cell walk costs more than touching every point (and a NaN query
            # lands here too, which is what bounds this loop). Finish exactly instead.
            break
        worst = knn_hashgrid_scan(
            points, grid_id, q, k, max_radius, r, out_indices_row, out_distances_row
        )
        if worst <= r:
            return  # certified exact
        if r >= r_hard:
            return  # the scan was already complete
        if worst < FLOAT32_INF_CONSTANT:
            r = wp.min(worst, r_hard)
        else:
            r = wp.min(r * RADIUS_GROWTH, r_hard)
    knn_linear_scan(points, q, k, max_radius, out_indices_row, out_distances_row)


def _hashgrid_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_hashgrid_nearest_neighbors``."""
    vec_distances = wp.types.vector(length=row_size, dtype=wp.float32)
    vec_indices = wp.types.vector(length=row_size, dtype=wp.int32)

    def _kernel(
        points: wp.array[wp.vec3],
        queries: wp.array[wp.vec3],
        grid_id: wp.uint64,
        k: wp.int32,
        max_radius: wp.float32,
        initial_radius: wp.float32,
        widest: wp.float32,
        min_bound: wp.vec3,
        max_bound: wp.vec3,
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        tid = wp.tid()
        q = queries[tid]
        row_indices = vec_indices()
        row_distances = vec_distances()

        r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
        r = wp.min(initial_radius, r_hard)
        certified = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
        for _attempt in range(MAX_SEARCH_ATTEMPTS):
            if not r <= widest:
                # Past ``widest`` a cell walk costs more than touching every point (and a NaN
                # query lands here too, which is what bounds this loop). Finish exactly instead.
                break
            for slot in range(row_size):
                row_indices[slot] = wp.int32(-1)
                row_distances[slot] = FLOAT32_INF_CONSTANT
            query = wp.hash_grid_query(grid_id, q, r)
            point_index = wp.int32(-1)
            while wp.hash_grid_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            worst = FLOAT32_INF_CONSTANT  # unrolled k-th read, as in the BVH twin
            for slot in range(row_size):
                if slot == k - 1:
                    worst = row_distances[slot]
            if worst <= r:
                certified = 1  # certified exact
                break
            if r >= r_hard:
                certified = 1  # the scan was already complete
                break
            if worst < FLOAT32_INF_CONSTANT:
                r = wp.min(worst, r_hard)
            else:
                r = wp.min(r * RADIUS_GROWTH, r_hard)

        if certified == 0:
            # Exact fallback once the radius outgrows the cell width or the attempt budget runs
            # out — the register twin of ``knn_linear_scan``.
            for slot in range(row_size):
                row_indices[slot] = wp.int32(-1)
                row_distances[slot] = FLOAT32_INF_CONSTANT
            for point_index in range(points.shape[0]):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index

        for slot in range(row_size):
            if slot < k:
                out_indices[tid, slot] = row_indices[slot]
                out_distances[tid, slot] = row_distances[slot]

    _kernel.__name__ = name
    _kernel.__qualname__ = name
    return wp.kernel(_kernel)


_HASHGRID_NEAREST_ROW_KERNELS = {
    row_size: _hashgrid_nearest_row_kernel(
        row_size, f"query_hashgrid_nearest_neighbors_row{row_size}"
    )
    for row_size in KNN_ROW_BUCKETS
}


def hashgrid_nearest_kernel(k: int) -> wp.Kernel:
    """Register-row grid k-NN kernel for the smallest bucket fitting ``k``, else global-row."""
    for row_size in KNN_ROW_BUCKETS:
        if k <= row_size:
            return _HASHGRID_NEAREST_ROW_KERNELS[row_size]
    return query_hashgrid_nearest_neighbors


@wp.kernel
def geodesic_ball_reference_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_reference: wp.array[wp.int32],
) -> None:
    """Lowest-indexed edge neighbor per vertex (libigl ``adjacency_list[i][0]``); self if alone."""
    i = int(wp.tid())
    start = int(adj_offsets[i])
    end = int(adj_offsets[i + 1])
    if start == end:
        out_reference[i] = i
        return
    minimum = adj_columns[start]
    for k in range(start + 1, end):
        if adj_columns[k] < minimum:
            minimum = adj_columns[k]
    out_reference[i] = minimum


@wp.kernel(enable_backward=False)
def query_geodesic_ball_collect(
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    chunk_start: wp.int32,
    queue_pool: wp.array2d[wp.int32],
    visited_pool: wp.array2d[wp.int32],
    ext_dist_pool: wp.array2d[wp.float32],
    ext_idx_pool: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    # Scratch lives in wrapper-allocated global-memory pools (one row per thread of the current
    # chunk) instead of ~8 KB of per-thread local arrays; the wrapper pre-fills the visited pool
    # with -1 before each launch. Single pass: after this kernel the thread's queue row holds
    # the collected set (``queue_pool[t][:out_counts[chunk_start + t]]``) ready to gather.
    t = int(wp.tid())
    i = int(chunk_start) + t
    out_counts[i] = kernel_bfs.per_source_bfs_collect(
        wp.int32(i),
        vertices,
        adj_offsets,
        adj_columns,
        radius,
        min_count,
        queue_pool[t],
        visited_pool[t],
        ext_dist_pool[t],
        ext_idx_pool[t],
        out_overflow,
    )


@wp.kernel
def gather_queue_rows(
    queue_pool: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
    local_offsets: wp.array[wp.int32],
    chunk_start: wp.int32,
    out_flat: wp.array[wp.int32],
) -> None:
    # Compact the chunk's queue rows into its flat CSR buffer. Adjacent j threads read one
    # queue row and write one out_flat segment contiguously (coalesced on both sides).
    # ``counts`` is the global per-source array (indexed at chunk_start + t); ``local_offsets``
    # is the chunk-local exclusive scan of this chunk's counts.
    t, j = wp.tid()
    if j >= counts[int(chunk_start) + t]:
        return
    out_flat[local_offsets[t] + j] = queue_pool[t, j]
