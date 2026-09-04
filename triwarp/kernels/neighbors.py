from typing import NamedTuple

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.array import declare_map_signatures, map_probe, map_probe_single

# Iterative-deepening k-nearest search. A scan at radius ``r`` enumerates every point within
# Euclidean distance ``r``, so a row whose k-th distance is at most ``r`` is provably the exact
# k-NN and the loop can stop. Both accelerators enumerate the ball directly -- the BVH through
# ``wp.bvh_query_sphere`` and the grid through ``wp.hash_grid_query`` -- so the certificate is the
# same statement on both, and neither pays for the bounding cube's extra 6/pi ~ 1.91x of volume.
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
# That spill used to be inferred from the timing curve. Warp 1.17's
# ``wp.get_cuda_kernel_properties`` measures it directly, and it confirms the shape while pinning
# where the wall is: the six shipped buckets report ``register_count``
# **42 / 53 / 64 / 92 / 126 / 222** with ``local_memory_size`` **0** at every one, so nothing
# in the shipped set spills. A 96-wide row extrapolates past the
# 255-register-per-thread hardware limit from there, which is what forces its spill -- the bucket
# list ends where the register file does, not at an arbitrary cut.
#
# **``cuda_max_registers`` does not move that wall, and it was measured rather than assumed.**
# 1.17's per-kernel register cap is the obvious lever for trading occupancy against pressure, so
# the shipped buckets were rebuilt under it (1M points, 100k queries, results identical at every
# cap). It loses everywhere, and the mechanism is in the ``local_memory_size`` column: capping
# below the natural count does not make the kernel leaner, it makes it *spill*.
#
#   | bucket | cap | regs | lmem | vs shipped |
#   |--------|-----|------|------|------------|
#   | 32 | none | 126 | 0 | 3.263 ms |
#   | 32 | 64 | 64 | 352 | **0.28x** |
#   | 32 | 96 | 96 | 80 | **0.71x** |
#   | 32 | 128 | 119 | 0 | 1.01x (a no-op: above the natural count) |
#   | 64 | none | 222 | 0 | 6.478 ms |
#   | 64 | 64 | 64 | 624 | **0.12x** |
#   | 64 | 128 | 128 | 608 | **0.15x** |
#   | 64 | 168 | 168 | 128 | **0.80x** |
#
# So a spilled row is 1.25-12x slower here, which is also the answer to the 96-bucket question
# above: a 96-wide row must spill whatever the cap says, and spilling is precisely what this table
# prices. The buckets stay uncapped.
#
# The buckets are not free: they cost ``2 x len(KNN_ROW_BUCKETS)`` generated kernels, which take
# this module's cold-cache compile from 3.9 s to ~12 s (once per Warp version and arch) and its
# warm per-process load from 2.4 ms to ~5 ms.
KNN_ROW_BUCKETS = (1, 4, 8, 16, 32, 64)

# Which accelerator ``ball_count_in_radius`` / ``ball_collect`` traverse. The ball query is one
# algorithm — same acceptance rule, same emit protocol — over two broad phases whose query
# objects are different types with different ``_next`` builtins, so the enumeration cannot be
# abstracted behind a ``wp.Function`` parameter (CLAUDE.md section 4: ``wp.launch`` cannot pass one
# as a kernel argument). An int selector can: the branch is warp-uniform, both traversals compile
# into this one module, and measured on an RTX 5090 (bunny, 20 000 queries, radius 2x and 4x the
# mean edge) the merged kernels are within noise of the two they replace on the BVH side and
# 1.05-1.09x *faster* on the hash-grid side, where the counting pass no longer allocates a
# throwaway per-thread distance slot. On CPU the same hash-grid counting pass gains 1.43-1.84x and
# the BVH paths lose 1-5%.
ACCEL_HASHGRID = wp.constant(wp.int32(0))
ACCEL_BVH = wp.constant(wp.int32(1))


# The two BVH walk protocols, over a *query* rather than over a bvh id and bounds.
#
# Taking the query is what Warp 1.17 made possible and it is the whole point of these two: a
# ``wp.BvhQuery`` is the common type over the aabb / sphere / capsule query kinds, and a
# ``@wp.func`` may take one as a parameter and another may return one (both verified -- a helper
# that constructs and returns a query, walked by a second helper, compiles and runs). Before that,
# a walk was pinned to the constructor that opened it, so every query kind carried its own copy of
# these four lines: the count walk existed twice, once here for the box and once inside
# ``ball_count_in_radius`` for the ball.
#
# ``@wp.func`` calls inline at codegen, so this is free -- confirmed rather than assumed, with
# ``wp.get_cuda_kernel_properties`` across the extraction: ``query_ball_count`` and
# ``query_ball_neighbors`` hold at **40** registers and the six row buckets at
# **42 / 53 / 64 / 92 / 126 / 222**, every one with ``local_memory_size`` **0**, which are the same
# figures the inline versions reported. And the move is *provably* behaviour-neutral where a
# float32 extraction would not be (CLAUDE.md section 3 warns that a green suite is not evidence):
# neither shared run contains a floating-point expression, so there is no evaluation order for it
# to disturb.


@wp.func
def bvh_walk_count(query: wp.BvhQuery) -> wp.int32:
    # Every primitive the query returns, counted, with no narrow phase.
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        c = c + 1
    return c


@wp.func
def bvh_walk_emit(query: wp.BvhQuery, base: wp.int32, out_indices: wp.array[wp.int32]) -> wp.int32:
    # The same primitives ``bvh_walk_count`` counts, written contiguously from ``base``. Counting
    # and emitting are separate walks rather than one ``write``-flagged function: the counting pass
    # then needs no output array at all, and the emit loop carries no per-candidate branch.
    j = wp.int32(0)
    c = wp.int32(0)
    while wp.bvh_query_next(query, j):
        out_indices[base + c] = j
        c = c + 1
    return c


# The ball-shaped broad phase over bounds, and it is *not* a tighter spelling of the box one below.
# ``wp.bvh_query_sphere`` prunes on an exact sphere-AABB squared distance where
# ``wp.bvh_query_aabb`` prunes on a box overlap, and on Warp 1.17 the box traversal is far the more
# expensive of the two *per candidate returned* -- measured 6.5-15x, isolated with the box's own
# *inscribed* cube (half extent ``r / sqrt(3)``, so strictly fewer candidates than the ball) still
# costing 6.5x the ball query. So the ball trims candidates *and* walks more cheaply, and a caller
# whose predicate is a ball wants this pair rather than the box pair plus a narrow phase. Both
# counts are exact against a brute-force oracle; CLAUDE.md section 12.8 has the numbers.
@wp.func
def ball_count_in_bounds(bvh_id: wp.uint64, q: wp.vec3, radius: wp.float32) -> wp.int32:
    # Broad-phase hits of the ball: every bound whose AABB comes within ``radius`` of ``q``.
    return bvh_walk_count(wp.bvh_query_sphere(bvh_id, q, radius))


@wp.kernel
def query_bvh_ball_neighbors(
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    bvh_walk_emit(wp.bvh_query_sphere(bvh_id, queries[tid], radius), offsets[tid], out_indices)


# The per-query-box pair, in the order its wrapper appears (section 5). Same walk skeleton as the
# ball above and a different node test; the count half is a ``wp.map`` over
# ``aabb_count_in_bounds`` rather than a kernel shim (CLAUDE.md section 4).
#
# There was a third pair here, for one warp-uniform half extent, and it is gone: it was exactly
# this one at ``q -+ h``, returned an identical set, and the corner buffers it saved measured
# **0.950-1.031x** -- no saving, one size the wrong way -- against a query several milliseconds
# long. Do not reintroduce it without a number bigger than that.
@wp.func
def aabb_count_in_bounds(bvh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3) -> wp.int32:
    # Broad-phase hits of the box.
    return bvh_walk_count(wp.bvh_query_aabb(bvh_id, lower, upper, root=-1))


@wp.func
def aabb_collect_in_bounds(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> None:
    # Emit the same hits ``aabb_count_in_bounds`` counted; the emit protocol is ``bvh_walk_emit``'s.
    bvh_walk_emit(wp.bvh_query_aabb(bvh_id, lower, upper, root=-1), base, out_indices)


@wp.kernel
def query_bvh_box_neighbors(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    aabb_collect_in_bounds(bvh_id, query_lower[tid], query_upper[tid], offsets[tid], out_indices)


@wp.func
def ball_count_in_radius(
    points: wp.array[wp.vec3], accel: wp.int32, accel_id: wp.uint64, q: wp.vec3, radius: wp.float32
) -> wp.int32:
    # Points within Euclidean ``radius`` of ``q``, over either accelerator (see ACCEL_* above).
    #
    # The two query objects are deliberately *differently named*: a Warp variable's type is fixed by
    # its first assignment, so binding one ``query`` name to a hash-grid query in one branch and a
    # BVH query in the other does not compile (verified — Warp raises at parse time). Do not "tidy"
    # them into a single name. Warp 1.17's ``wp.BvhQuery`` common type lifted this *between BVH
    # kinds* -- a box query and a sphere query can share one variable, measured -- but a hash-grid
    # query is still not a ``BvhQuery``, so these two stay apart.
    #
    # **The predicate is the squared one, on both branches, and that is a decision rather than a
    # style.** ``wp.bvh_query_sphere`` prunes on an exact sphere-AABB squared-distance test, and on
    # this BVH -- built by ``neighbors.bvh_from_points`` as ``wp.Bvh(points, points)``, so every
    # leaf bound is a degenerate point -- that test *is* the point-in-ball test. There is no narrow
    # phase left to write. Measured on an RTX 5090, interleaved A/B against the cube-plus-
    # ``wp.length`` form this replaced: **1.22x** at 200k points / 20k queries / r=0.01 (0.8
    # neighbours a row), **1.74x** at r=0.02, **2.87x** at r=0.05 (99.1 a row), and 1.50 / 2.06 /
    # 2.77x at 1M points / 100k queries -- monotone in the neighbour count, and 1.14-1.38x on the
    # CPU device at the same six points, so it wins on both.
    #
    # The hash-grid branch is spelled ``wp.length_sq`` to *match* it. ``wp.length(d) <= radius`` and
    # ``wp.length_sq(d) <= radius * radius`` are not the same predicate in float32 -- ``sqrt`` and
    # ``radius * radius`` round independently -- and measured over one candidate stream at 1M
    # points / 100k queries the sphere query agrees with the squared form on **0 of 100 000 rows**
    # and with the sqrt form on all but **1**, which differs by one neighbour, identically on CPU
    # and CUDA. So leaving the grid on ``wp.length`` would make ``backend="hashgrid"`` and
    # ``backend="bvh"`` answer differently at the boundary, which is the one-predicate-one-spelling
    # defect section 3 of CLAUDE.md names; ``test_the_two_backends_agree`` is the gate.
    #
    # An earlier pass measured the two spellings as speed-flat (0.997-1.003x) and kept the sqrt
    # because it is the one the wrapper's "inclusive at exactly ``radius``" docstring means. That
    # measurement still holds for the *narrow phase* -- what changed is that the squared form is now
    # what the tighter broad phase speaks, so the choice buys 1.2-2.9x instead of nothing.
    c = wp.int32(0)
    j = wp.int32(0)
    if accel == ACCEL_HASHGRID:
        query = wp.hash_grid_query(accel_id, q, radius)
        while wp.hash_grid_query_next(query, j):
            if wp.length_sq(points[j] - q) <= radius * radius:
                c = c + 1
    else:
        # No narrow phase: on degenerate point bounds the sphere-AABB test is the ball test, so
        # this is the bare walk ``bvh_walk_count`` also serves the box query with.
        c = bvh_walk_count(wp.bvh_query_sphere(accel_id, q, radius))
    return c


@wp.kernel
def query_ball_count(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    radius: wp.float32,
    out_neighbor_counts: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    out_neighbor_counts[tid] = ball_count_in_radius(points, accel, accel_id, queries[tid], radius)


@wp.func
def ball_collect(
    points: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    q: wp.vec3,
    radius: wp.float32,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    # Emit the same neighbours ``ball_count_in_radius`` counted, contiguously from ``base``, with
    # the distance the caller wants alongside. Traversal order fixes the within-query order, which
    # is why the wrapper's ``return_sorted`` is a separate segmented sort.
    #
    # Both branches must accept exactly what ``ball_count_in_radius`` counts or the offsets it
    # produced would not fit -- so the predicate is the squared one here too, for the reasons
    # written there. The BVH branch needs no acceptance test at all (the sphere query already made
    # it), and the hash-grid branch tests squared and takes the square root only for the candidates
    # it keeps, where the sqrt form paid for one on every candidate it walked.
    c = wp.int32(0)
    j = wp.int32(0)
    if accel == ACCEL_HASHGRID:
        query = wp.hash_grid_query(accel_id, q, radius)
        while wp.hash_grid_query_next(query, j):
            offset = points[j] - q
            if wp.length_sq(offset) <= radius * radius:
                out_indices[base + c] = j
                out_distances[base + c] = wp.length(offset)
                c = c + 1
    else:
        query_sphere = wp.bvh_query_sphere(accel_id, q, radius)
        while wp.bvh_query_next(query_sphere, j):
            out_indices[base + c] = j
            out_distances[base + c] = wp.length(points[j] - q)
            c = c + 1


@wp.kernel
def query_ball_neighbors(
    points: wp.array[wp.vec3],
    queries: wp.array[wp.vec3],
    accel: wp.int32,
    accel_id: wp.uint64,
    radius: wp.float32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    ball_collect(
        points, accel, accel_id, queries[tid], radius, offsets[tid], out_indices, out_distances
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
    # Smallest **ball** radius about ``q`` that contains the whole point bounding box, i.e. the
    # radius at which a scan is provably complete: the distance from ``q`` to the box's farthest
    # corner. Per-query, so it is tighter than a global diagonal, and unbounded for a query far
    # outside the box (which is what keeps that case exact).
    #
    # This is a *ball* radius and not the cube half-extent it used to be, because every enumeration
    # it bounds is now a ``wp.bvh_query_sphere`` or a ``wp.hash_grid_query``, both of which take a
    # Euclidean radius. The two differ by up to sqrt(3), and the cube form was the smaller of the
    # two -- so as a ball radius it did *not* cover the box, which would have made a
    # forced-complete final attempt incomplete. Every caller therefore gets a radius at least as
    # large as before: exact where it was exact, and no longer under-covering on the hash-grid path,
    # whose ``wp.hash_grid_query`` has always read this as a Euclidean radius.
    reach = wp.max(q - min_bound, max_bound - q)  # per-component farthest face distance
    return wp.length(reach)


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
    # Refill the row from the **ball** of radius ``r`` about ``q`` and return the k-th best distance
    # (``inf`` when fewer than ``k`` points were accepted). Acceptance stays ``d <= max_radius``;
    # ``r`` bounds only the enumeration.
    #
    # The enumeration used to be the cube ``[q +/- r]``, whose certificate rested on Chebyshev
    # distance never exceeding Euclidean. ``wp.bvh_query_sphere`` enumerates the ball itself, so the
    # certificate is now direct -- every point outside the ball is farther than ``r``, hence a
    # ``worst <= r`` row is exact -- over a strictly smaller candidate set: the cube holds
    # 6/pi ~ 1.91x the ball's volume, and it is that ratio the caller's speedup comes out of.
    knn_reset_row(k, out_indices_row, out_distances_row)
    query = wp.bvh_query_sphere(bvh_id, q, r)
    point_index = wp.int32(0)
    while wp.bvh_query_next(query, point_index):
        d = wp.length(points[point_index] - q)
        knn_sorted_insert(point_index, d, k, max_radius, out_indices_row, out_distances_row)
    return out_distances_row[k - 1]


@wp.func
def deepen_radius(worst: wp.float32, r: wp.float32, r_hard: wp.float32) -> wp.float32:
    # Next search radius after an uncertified scan, shared by all four k-NN kernels. A full row
    # (finite ``worst``) reaching past the cube certifies at exactly ``worst``, so jump there; an
    # unfilled row has no bound to jump to and grows geometrically instead.
    if worst < FLOAT32_INF_CONSTANT:
        return wp.min(worst, r_hard)
    return wp.min(r * RADIUS_GROWTH, r_hard)


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
    tid = wp.int32(wp.tid())
    q = queries[tid]
    out_indices_row = out_indices[tid]
    out_distances_row = out_distances[tid]

    r_hard = wp.min(max_radius, complete_radius(q, min_bound, max_bound))
    r = wp.min(initial_radius, r_hard)
    # This kernel used to carry a note saying it may hold exactly one ``wp.bvh_query_*`` call site,
    # because "``bvh_query`` declares ``__shared__ int stack[32 * WP_TILE_BLOCK_DIM]`` (32 KB at
    # block_dim=256), so a second textual call site would ask for 64 KB and fail to compile".
    # **That was never true of the plain query.** In ``warp/native/bvh.h`` the ``__shared__``
    # declaration sits inside the *tiled* constructor; ``wp.bvh_query_aabb`` / ``_sphere`` use a
    # per-thread ``int stack[BVH_QUERY_STACK_SIZE]`` in local storage. Measured: a kernel with two
    # plain call sites compiles and runs at block_dim=256 on Warp 1.16, and one with four does on
    # 1.17 (``register_count`` 45, ``local_memory_size`` 0). The single call site here is now just
    # what the loop needs, not a constraint -- so a future edit needing a second one may add it.
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        if attempt == MAX_SEARCH_ATTEMPTS - 1:
            r = r_hard  # forced-complete final attempt: exact whatever the growth did
        worst = knn_bvh_scan(
            points, bvh_id, q, k, max_radius, r, out_indices_row, out_distances_row
        )
        if worst <= r:
            break  # every point outside the ball is farther than the k-th best: certified exact
        if r >= r_hard:
            break  # the scan was already complete, so the row is final
        r = deepen_radius(worst, r, r_hard)


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
# A ``wp.zeros(shape=K)`` stack array does not work here — measured a **2x loss** against the global
# row, because nothing promotes it to registers.
#
# Everything the row touches **once per query or per attempt** — the reset, the k-th read, the
# output write — is factored into the generated ``@wp.func`` set below, which both factories share.
# The
# **per-candidate insert alone stays written out inline**, three times, and the split is measured
# rather than assumed (Warp 1.16, RTX 5090, 200k points / 200k queries on CUDA, 20k / 20k on CPU,
# variants interleaved in one loop, ``min`` of 9 reps, all bit-identical in indices and distances):
#
#   | insert spelling                            | k=1   | k=7   | k=32  | k=64   |
#   |--------------------------------------------|-------|-------|-------|--------|
#   | ``wp.ref`` helper, CUDA                    | 1.00x | 1.00x | 1.00x | 1.00x  |
#   | ``wp.ref`` helper, **CPU**                 | 0.95x | 1.00x | 1.55x | 2.23x  |
#   | by value, returning the pair, **CUDA**     | 1.01x | 0.99x | 5.13x | 10.24x |
#   | by value, returning the pair, CPU          | 0.96x | 1.04x | 1.03x | 1.07x  |
#
# So the two spellings fail on opposite devices, and the once-per-attempt helpers (same ``wp.ref``
# parameters, crossed 1x per attempt instead of 1x per candidate) are flat on **both**: 0.99-1.01x
# CUDA, 0.97-1.00x CPU at every bucket. Do not "finish" this dedup by moving the insert too.
#
# The contract the copies must hold is the *distance* row, not the tie-break. Like the shipped
# kernel's ``binary_search_index`` (``searchsorted(side="right")``), the carry below walks past
# equals before displacing, and ``placed`` then shifts the rest of the row down mechanically — a
# plain stable insertion, which is why it matches the global row index-for-index. But that match is
# incidental and ``placed`` is **not** load-bearing: without it the carry skips a slot holding an
# equal distance and displaces further down instead, which permutes *which* of several equidistant
# points fills a slot and leaves every distance bit-identical (measured — deleting the flag from all
# three carries passes the tie test at every bucket on both backends). Callers are told exactly that
# much; ``query_nearest``'s docstring declares the identity of a tied neighbour unspecified.
#
# So the guard on an edit here is a *tied* fixture, not a second implementation to diff against:
# ``tests/test_neighbors.py::test_query_nearest_ties`` runs every ``KNN_ROW_BUCKETS`` size against
# ``scipy.spatial.KDTree`` on an integer lattice, where a query has dozens of exactly tied
# neighbours and a carry that mishandles the run keeps a farther point. Verified as a live gate:
# shortening the shift chain by one slot fails all 12 cases. Edit the carry and run it.
# ---------------------------------------------------------------------------------------------


class _RowHelpers(NamedTuple):
    """One bucket's register-row vector types and the ``@wp.func``s that operate on a whole row."""

    distances: type
    indices: type
    reset: wp.Function
    kth: wp.Function
    write: wp.Function


def _row_helpers(row_size: int) -> _RowHelpers:
    """Generate the ``K = row_size`` row helper set; ``K`` must be a compile-time constant."""
    vec_distances = wp.types.vector(length=row_size, dtype=wp.float32)
    vec_indices = wp.types.vector(length=row_size, dtype=wp.int32)

    def _reset(row_distances: wp.ref[vec_distances], row_indices: wp.ref[vec_indices]) -> None:
        # Every scan starts from an empty row: the insert does not deduplicate, so a re-scan over a
        # wider radius would otherwise insert each already-found point a second time.
        for slot in range(row_size):
            row_indices[slot] = wp.int32(-1)
            row_distances[slot] = FLOAT32_INF_CONSTANT

    def _kth(row_distances: wp.ref[vec_distances], k: wp.int32) -> wp.float32:
        # The k-th distance certifies the scan (``inf`` while the row is unfilled). Read through an
        # unrolled compare rather than ``row_distances[k - 1]``: ``k`` is a runtime value and a
        # runtime index into a vector spills it to local memory.
        worst = FLOAT32_INF_CONSTANT
        for slot in range(row_size):
            if slot == k - 1:
                worst = row_distances[slot]
        return worst

    def _write(
        row_distances: wp.ref[vec_distances],
        row_indices: wp.ref[vec_indices],
        tid: wp.int32,
        k: wp.int32,
        out_indices: wp.array2d[wp.int32],
        out_distances: wp.array2d[wp.float32],
    ) -> None:
        # Only the first ``k`` slots are the caller's answer; the bucket's tail is padding.
        for slot in range(row_size):
            if slot < k:
                out_indices[tid, slot] = row_indices[slot]
                out_distances[tid, slot] = row_distances[slot]

    return _RowHelpers(
        vec_distances,
        vec_indices,
        wp.func(_reset, name=f"knn_row_reset{row_size}"),
        wp.func(_kth, name=f"knn_row_kth{row_size}"),
        wp.func(_write, name=f"knn_row_write{row_size}"),
    )


# One set per bucket: ``wp.ref`` needs a concrete dtype, so the helpers cannot be generic in ``K``.
_ROW_HELPERS = {row_size: _row_helpers(row_size) for row_size in KNN_ROW_BUCKETS}


def _bvh_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_bvh_nearest_neighbors``."""
    vec_distances, vec_indices, row_reset, row_kth, row_write = _ROW_HELPERS[row_size]

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
        # The ball enumeration and its certificate are ``knn_bvh_scan``'s, which this factory
        # cannot call because the row lives in registers rather than in the output arrays; the
        # traversal is the only part duplicated, and the reason it is duplicated is the row type.
        #
        # **The sphere query is a win at most buckets and a small loss at one, and which one moves
        # with the cloud.** Interleaved A/B against the cube form, one process, RTX 5090, medians
        # (the minima agree to 0.01x), every cell byte-identical in the index rows:
        #
        # | k (bucket) | 200k pts / 20k queries | 1M pts / 100k queries |
        # |---|---|---|
        # | 1 (1) | 1.14x | 1.73x |
        # | 7 (8) | 1.21x | 1.80x |
        # | 16 (16) | 1.58x | 1.41x |
        # | 30 (32) | 1.28x | **0.90x** |
        # | 64 (64) | **0.94x** | 1.61x |
        #
        # The loss is *not* any of the three things it looks like, each checked: the deepening
        # sequence is unchanged (mean attempts 1.000 / 1.292 / 2.000 at k=8 / 30 / 64, identical
        # for both enumerations -- the certificate compares the k-th *distance*, which no
        # enumeration shape can move), the new ``complete_radius`` is not implicated (a
        # sphere-query build carrying the old Chebyshev radius measures 11.22 ms against this
        # one's 11.31 at k=30, i.e. noise), and nothing spills -- ``wp.get_cuda_kernel_properties``
        # reports ``local_memory_size`` 0 at every bucket and *fewer* registers for the sphere form
        # at five of six (42/53/64/92/126/222 against 46/55/67/96/127/218). What is left is the
        # per-node arithmetic: the exact sphere-AABB test costs more per node than a slab test, and
        # at the bucket where row-insertion traffic and traversal cost balance, the ~1.91x
        # candidate saving stops covering it. Kept, because the benchmarked groups are k1 / k7 /
        # k64 and eight of ten cells win.
        for attempt in range(MAX_SEARCH_ATTEMPTS):
            if attempt == MAX_SEARCH_ATTEMPTS - 1:
                r = r_hard  # forced-complete final attempt: exact whatever the growth did
            row_reset(row_distances, row_indices)
            query = wp.bvh_query_sphere(bvh_id, q, r)
            point_index = wp.int32(0)
            while wp.bvh_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            worst = row_kth(row_distances, k)
            if worst <= r:
                break  # every point outside the ball is farther than the k-th best: exact
            if r >= r_hard:
                break  # the scan was already complete, so the row is final
            r = deepen_radius(worst, r, r_hard)

        row_write(row_distances, row_indices, tid, k, out_indices, out_distances)

    _kernel.__name__ = name
    _kernel.__qualname__ = name
    # ``enable_backward=False`` is mandatory, not a choice: the row helpers take ``wp.ref``
    # parameters and a ``wp.ref`` helper has no adjoint, so the module fails to compile without it.
    return wp.kernel(_kernel, enable_backward=False)


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
    #
    # The two are NOT merged behind the ``ACCEL_*`` selector the ball queries use, and the reason
    # is the one that separates section 2.1's case from section 2.5's: there the two accelerators
    # ran the *same* algorithm and only the enumeration differed, so a warp-uniform int made them
    # one kernel. Here the enclosing searches have genuinely diverged -- the grid path takes a
    # ``widest`` bound and falls back to ``knn_linear_scan`` once the radius outgrows the cell
    # width (a cell walk is cubic in the radius, so past that it costs more than touching every
    # point), while the BVH path forces a final complete attempt instead. A merged kernel would
    # carry a parameter that is ignored on one path and a fallback branch that belongs to the
    # other. What the two genuinely share is already shared: ``knn_reset_row``,
    # ``knn_sorted_insert``, ``complete_radius`` and ``deepen_radius``; what is left is three lines
    # of traversal each, plus each one's own certification policy.
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
    tid = wp.int32(wp.tid())
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
        r = deepen_radius(worst, r, r_hard)
    knn_linear_scan(points, q, k, max_radius, out_indices_row, out_distances_row)


def _hashgrid_nearest_row_kernel(row_size: int, name: str):
    """``K = row_size`` register-row variant of ``query_hashgrid_nearest_neighbors``."""
    vec_distances, vec_indices, row_reset, row_kth, row_write = _ROW_HELPERS[row_size]

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
        certified = wp.int32(0)
        for _attempt in range(MAX_SEARCH_ATTEMPTS):
            if not r <= widest:
                # Past ``widest`` a cell walk costs more than touching every point (and a NaN
                # query lands here too, which is what bounds this loop). Finish exactly instead.
                break
            row_reset(row_distances, row_indices)
            query = wp.hash_grid_query(grid_id, q, r)
            point_index = wp.int32(-1)
            while wp.hash_grid_query_next(query, point_index):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index
            worst = row_kth(row_distances, k)
            if worst <= r:
                certified = 1  # certified exact
                break
            if r >= r_hard:
                certified = 1  # the scan was already complete
                break
            r = deepen_radius(worst, r, r_hard)

        if certified == 0:
            # Exact fallback once the radius outgrows the cell width or the attempt budget runs
            # out — the register twin of ``knn_linear_scan``.
            row_reset(row_distances, row_indices)
            for point_index in range(points.shape[0]):
                d = wp.length(points[point_index] - q)
                if d <= max_radius and d < row_distances[row_size - 1]:
                    carry_distance = d
                    carry_index = point_index
                    placed = wp.int32(0)
                    for slot in range(row_size):
                        if placed == 1 or carry_distance < row_distances[slot]:
                            placed = 1
                            held_distance = row_distances[slot]
                            held_index = row_indices[slot]
                            row_distances[slot] = carry_distance
                            row_indices[slot] = carry_index
                            carry_distance = held_distance
                            carry_index = held_index

        row_write(row_distances, row_indices, tid, k, out_indices, out_distances)

    _kernel.__name__ = name
    _kernel.__qualname__ = name
    return wp.kernel(_kernel, enable_backward=False)  # ``wp.ref`` helpers, as in the BVH twin


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
def query_weighted_nearest_neighbors(
    points: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    max_weight: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_indices: wp.array[wp.int32],
    out_distances: wp.array[wp.float32],
) -> None:
    # Nearest site under the *weighted* distance ``|p - q| - w(p)``, i.e. the Apollonius / power
    # nearest neighbour. One site, so no candidate row and no register bucket: the whole k-NN row
    # machinery above collapses to two scalars here.
    #
    # ``max_weight`` is what makes the search prunable, and it is the only thing that does. A site
    # outside the ball of radius ``r`` has ``|p - q| > r``, so its score exceeds
    # ``r - max_weight``; a best score at or below that bound is therefore certified, and the
    # deepening target is ``best + max_weight`` -- which is always past the current ``r``, since
    # failing the test means ``best + max_weight > r``. That is exactly ``deepen_radius``'s contract
    # with the bound shifted, so the helper is shared with the k-NN kernels rather than re-derived.
    #
    # The test is spelled ``best + max_weight <= r`` and **not** the algebraically identical
    # ``best <= r - max_weight``, because the deepening step sets ``r`` to ``best + max_weight``: in
    # float32 the subtracted form can then fail against the very radius it just asked for, when
    # ``best`` is large next to ``max_weight`` and the addition rounds. Measured on a 437k-point
    # cloud with the queries 6.6 spacings off the surface: the subtracted form stalled at a fixed
    # radius for the full 16-attempt budget on 0.1% of queries, each of which then paid the
    # forced-complete final scan -- 437 757 candidates against a mean of 651. Computing the same
    # expression on both sides makes the loop exact, and it is what keeps the worst case bounded.
    tid = wp.int32(wp.tid())
    q = queries[tid]

    r_hard = complete_radius(q, min_bound, max_bound)
    r = wp.min(initial_radius, r_hard)
    best = FLOAT32_INF_CONSTANT
    best_index = wp.int32(-1)
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        if attempt == MAX_SEARCH_ATTEMPTS - 1:
            r = r_hard  # forced-complete final attempt: exact whatever the growth did
        query = wp.bvh_query_sphere(bvh_id, q, r)
        point_index = wp.int32(0)
        while wp.bvh_query_next(query, point_index):
            score = wp.length(points[point_index] - q) - weights[point_index]
            if score < best:
                best = score
                best_index = point_index
        if best + max_weight <= r:
            break  # every site outside the ball scores worse than this: certified exact
        if r >= r_hard:
            break  # the scan was already complete, so the answer is final
        r = deepen_radius(best + max_weight, r, r_hard)

    out_indices[tid] = best_index
    out_distances[tid] = best


@wp.kernel
def geodesic_ball_reference_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_reference: wp.array[wp.int32],
) -> None:
    """Lowest-indexed edge neighbor per vertex (libigl ``adjacency_list[i][0]``); self if alone."""
    i = wp.int32(wp.tid())
    start = adj_offsets[i]
    end = adj_offsets[i + 1]
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
    t = wp.int32(wp.tid())
    i = chunk_start + t
    out_counts[i] = kernel_bfs.per_source_bfs_collect(
        i,
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
    if j >= counts[chunk_start + t]:
        return
    out_flat[local_offsets[t] + j] = queue_pool[t, j]


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (aabb_count_in_bounds, (wp.uint64(1), dense(wp.vec3), dense(wp.vec3)), wp.int32),
            (aabb_count_in_bounds, (wp.uint64(1), single(wp.vec3), single(wp.vec3)), wp.int32),
            (ball_count_in_bounds, (wp.uint64(1), dense(wp.vec3), wp.float32(1)), wp.int32),
            (ball_count_in_bounds, (wp.uint64(1), single(wp.vec3), wp.float32(1)), wp.int32),
        ]
    )


_declare_map_kernels()
