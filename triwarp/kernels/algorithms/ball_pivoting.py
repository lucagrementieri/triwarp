"""
Kernels for wave-parallel ball-pivoting surface reconstruction.

Ports the geometry of Open3D's ``SurfaceReconstructionBallPivoting.cpp`` (``ComputeBallCenter``,
``IsCompatible``, ``FindCandidateVertex``, ``TryTriangleSeed``) to Warp. Instead of the serial
advancing front, every wave pivots a conflict-free independent set of front edges at once and
commits them through a two-phase vertex claim (``wp.atomic_min`` priority on all three vertices),
so at least the globally-lowest-priority triangle always commits and the loop cannot livelock.

The priority is ``proposal_key``, a packing of the proposal's own source edge with a salt drawn from
the wave counter, which is what makes a run **reproducible**: *which* triangles get committed no
longer depends on the order threads reach an atomic, only the order they are written down in does
(see below). The salt is what keeps that from also being *slow* -- a key fixed for the whole run
starves the same front edges wave after wave, which measured 2.2x the waves. See
that function for why the key is unique within a wave, and ``commit_triangles`` for the one other
place arrival order used to leak in (the triangle-budget check). Measured on an irregularly sampled
torus, four runs of one build now commit the same 3 269 triangles with the same winding, where
before they produced 2 980 / 3 036 / 3 042 / 3 089 faces.

Two things this deliberately does *not* pin, so compare a run as a **set** of triangles rather than
buffer to buffer. ``CNT_FACE`` is still a ``wp.atomic_add``, so the face buffer's row order remains
arrival-ordered; determinizing it would mean sorting the whole buffer for a property no caller has
asked for. And ``reconstruction.ball_pivoting``'s cleanup tail loses the winding again, because
``repair.make_winding_consistent`` seeds each connected component from an arbitrary face — measured
on the same fixture, and a defect in that module rather than this one.

Persistent state
----------------
Everything the algorithm needs lives in device buffers that survive the whole run, so a wave costs
launches only — no allocations, no host readbacks, and nothing is re-derived from scratch:

* **Edge table.** An open-addressed hash keyed by the undirected edge (``edge_key``), holding the
  incident-face count, the single incident face's directed orientation (``edge_src`` / ``edge_tgt``
  / ``edge_opp``) while the edge is on the boundary, a retirement flag and a cached best candidate.
  ``commit_triangles`` is its only mutator. This replaces re-deriving the front from the whole
  triangle soup each wave with ``edges_unique`` + a sorted interior-key array, and turns the
  manifold guard from a binary search into one hash probe.
* **Front list.** Edge-table slots of the boundary edges, compacted by ``pivot_front_edges`` into a
  second buffer each wave (the caller ping-pongs them) and extended by ``commit_triangles`` with
  the boundary edges the new triangles create.
* **Per-point state.** ``point_used`` and ``boundary_degree`` are maintained incrementally by
  ``commit_triangles``; a point is available as a pivot candidate iff it is unused or still has an
  incident boundary edge.
* **Counters.** One ``int32`` array (see the ``CNT_*`` slots) carries the face count, the front and
  proposal counts, the seeding/continue/done flags and the wave number, so the wave loop's control
  flow is device-resident and can run inside a captured graph.

Border edges
------------
``pivot_front_edges`` retires a front edge whose search found no candidate (Open3D's
``BallPivotingEdgeType::Border``) and never looks at it again. Measured on ``bunny``, 96% of all
pivots in a run were re-tests of edges already known to be dead, rising to 100% in the tail.

Retiring them is **output-preserving, not an approximation**, because every rejection in the
candidate loop is monotone in the wave index:

* the geometric tests (``compute_ball_center``, ``is_compatible``, ``ball_is_empty``, the crease
  and clustering guards) depend only on the fixed points, normals and radius;
* availability is sticky-false — a vertex that is used and has no incident boundary edge can never
  gain another triangle, since seeding needs an unused point and pivoting needs an available
  candidate, so it can never become available again;
* the interior-edge guard only ever tightens: faces are appended and never removed, so an edge's
  face count only grows.

So the set of valid candidates for a given front edge shrinks monotonically. Two consequences are
used here: an edge with no candidate can be retired, and — the other half of the same theorem — a
**cached** best candidate is still the argmin as long as it is still valid, so an edge that found
one but lost the vertex claim (most of them, every wave) is re-checked in O(1) instead of
re-searched over its whole neighbourhood. An edge is retired only when the *search* failed, never
when it merely lost the claim.
"""

import warp as wp

from triwarp.constants import UINT64_MAX_CONSTANT
from triwarp.kernels.array import pack_edge_key, tile_argmin
from triwarp.kernels.grouping import hash_find, hash_find_or_insert
from triwarp.kernels.predicates import dihedral_angle, triangle_normal

# Per-thread neighbour scratch for the seed search (Open3D re-scans the KNN result twice).
MAX_SEED_NEIGHBORS = 64
# Relative slack on the empty-ball test: absorbs float32 round-off so the three defining points
# (exactly on the ball in exact arithmetic) do not spuriously read as "inside".
BALL_EPS = wp.constant(wp.float32(1e-4))
TWO_PI = wp.constant(2.0 * wp.PI)

# ``counters`` slots. Keeping the wave loop's whole control state on device is what lets the loop
# body run without a host synchronisation, and ultimately inside a captured CUDA graph.
CNT_FACE = wp.constant(0)  # committed triangles so far
CNT_PREV_FACE = wp.constant(1)  # ... as of the start of this wave (the progress test)
CNT_FRONT = wp.constant(2)  # entries in the incoming front list
CNT_NEXT_FRONT = wp.constant(3)  # entries written to the outgoing front list
CNT_LIVE = wp.constant(4)  # of those, how many were live (drives host-side compaction)
CNT_PROPOSAL = wp.constant(5)  # triangles proposed this wave
CNT_SEEDING = wp.constant(6)  # this wave seeds orphans instead of pivoting
CNT_CONTINUE = wp.constant(7)  # the wave loop's condition
CNT_DONE = wp.constant(8)  # the loop finished for good (as opposed to pausing to grow)
CNT_WAVE = wp.constant(9)  # waves executed
CNT_GROW = wp.constant(10)  # the triangle budget is exhausted; hand back to the host
CNT_OVERFLOW = wp.constant(11)  # a fixed-capacity scatter ran out of room this wave
BPA_COUNTERS = 12

# Lanes per front edge in the cooperative pivot search. One warp: the block-cooperative BVH walk
# hands one candidate per lane per step, and 32 covers the ~88 candidates an edge enumerates in
# three steps while keeping the tile reductions on Warp's single-warp fast path.
BPA_PIVOT_BLOCK = 32

EDGE_LIVE = wp.constant(0)
EDGE_RETIRED = wp.constant(1)


@wp.struct
class BpaEdgeTable:
    """
    The open-addressed edge table, as one kernel argument instead of nine.

    Every wave launches ``pivot_front_edges`` (27 arguments before this, 19 after) and
    ``commit_triangles`` (20, now 13), and both carried the whole table; seven functions here
    thread it. Measured on a 35k-point cloud, one run issues **1412 launches carrying 15 442
    arguments** before the bundle and 11 658 after, at a measured ~1.0 us of host time per
    ``wp.launch`` argument.

    Measured directly, both wave shapes interleaved in one process with trivial kernel bodies so
    only marshalling is timed: over 235 waves, **15.12 -> 11.32 ms (min), 1.34x**, which is
    **1.08 us per dropped argument** and agrees with the independent per-argument law.

    It had to be measured that way, because at the time BPA was **run-to-run nondeterministic**:
    three runs of one build on the same cloud produced 44 179 / 44 243 / 44 208 faces with
    1412 / 1412 / 1364 launches, so the end-to-end row was not timing the same reconstruction
    twice. ``proposal_key`` has since removed that, and a run now commits the same triangle set
    every time — but the isolated measurement is still the right one for a launch-path change,
    since the end-to-end row is 300+ ms of pivot search around ~11 ms of marshalling.

    Bound once by ``_BpaState`` and rebound only when ``grow`` reallocates (``_bind_edge_table``
    is called from ``_allocate_budget``, which is the only place the arrays are replaced).
    ``rehash_edges`` still takes the old and new arrays loose, because it is the one kernel that
    sees two tables at once.
    """

    key: wp.array[wp.uint64]
    count: wp.array[wp.int32]
    src: wp.array[wp.int32]
    tgt: wp.array[wp.int32]
    opp: wp.array[wp.int32]
    state: wp.array[wp.int32]
    cand: wp.array[wp.int32]
    mask: wp.int32
    key_base: wp.uint64


@wp.func
def compute_ball_center(
    v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, normal_sum: wp.vec3, radius: wp.float32
) -> wp.vec3:
    # Center of the radius-``radius`` ball touching all three points, on the ``normal_sum`` side.
    # Returns a sentinel of (inf, inf, inf) when the ball does not exist (points too far apart or
    # nearly collinear), which the caller treats as failure.
    fail = wp.vec3(wp.inf, wp.inf, wp.inf)
    c = wp.length_sq(v2 - v1)
    b = wp.length_sq(v1 - v3)
    a = wp.length_sq(v3 - v2)
    alpha = a * (b + c - a)
    beta = b * (a + c - b)
    gamma = c * (a + b - c)
    abg = alpha + beta + gamma
    if abg < 1e-16:
        return fail
    circ_center = (alpha * v1 + beta * v2 + gamma * v3) / abg
    circ_radius2 = a * b * c
    sa = wp.sqrt(a)
    sb = wp.sqrt(b)
    sc = wp.sqrt(c)
    denom = (sa + sb + sc) * (sb + sc - sa) * (sc + sa - sb) * (sa + sb - sc)
    if denom <= 0.0:
        return fail
    circ_radius2 = circ_radius2 / denom
    height = radius * radius - circ_radius2
    if height < 0.0:
        return fail
    tr_norm = triangle_normal(v1, v2, v3)
    pt_norm = wp.normalize(normal_sum)
    if wp.dot(tr_norm, pt_norm) < 0.0:
        tr_norm = -tr_norm
    return circ_center + wp.sqrt(height) * tr_norm


@wp.func
def is_compatible(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, na: wp.vec3, nb: wp.vec3, nc: wp.vec3
) -> wp.bool:
    # The triangle normal must agree (within tolerance) with all three oriented point normals.
    normal = triangle_normal(a, b, c)
    if wp.dot(normal, na) < -1e-16:
        normal = -normal
    return (
        wp.dot(normal, na) > -1e-16 and wp.dot(normal, nb) > -1e-16 and wp.dot(normal, nc) > -1e-16
    )


@wp.func
def ball_is_empty(
    grid_id: wp.uint64,
    points: wp.array[wp.vec3],
    center: wp.vec3,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
) -> wp.bool:
    # True when no point other than the three defining ones lies strictly inside the ball.
    #
    # Compared squared, matching ``candidate_is_viable``'s clustering test. The two spellings are
    # not the same predicate in float32 -- the ``neighbors`` narrow-phase measurement found 10 rows
    # of 200k where they disagree at the boundary -- so a module that used both could accept a
    # point as a candidate and reject the same distance as an occupancy, and the ball radius is one
    # of the two distances BPA is entirely built out of.
    #
    # **The argument is consistency, not speed, and the speed was measured to say so.** Whole
    # ``ball_pivoting`` call, icosphere(4) cloud (2 562 points) at ``1.5 * mean_edge``, both
    # spellings alternated over six separate processes (``bpa-intermittent-illegal-access`` rules
    # out looping reconstructions in one process), 15 reps each: median-of-medians 21.5 ms squared
    # against 23.7 ms unsquared, but min-of-mins 19.7 against 17.0 -- the two orderings disagree, so
    # this is flat inside the run-to-run spread, matching the 0.997-1.003x the ``neighbors``
    # narrow-phase decline measured for the same swap. Face count 5 120 either way.
    #
    # **``wp.bvh_query_sphere`` (Warp 1.17) was built here and reverted: it is a 1.75x loss.** It
    # looks like the ideal fit -- the point BVH's bounds are degenerate, so an exact sphere-AABB
    # test at the shrunk radius *is* this predicate, with no narrow phase and no second radius,
    # where the grid must query at ``radius`` and filter at ``threshold`` because a cell walk
    # cannot express either radius exactly. It was also *correct*: the flat face buffer came back
    # equal element for element on the gate below. It is simply slower. Measured on the same
    # icosphere(4) cloud, one reconstruction per process, three alternating pairs -- **16.20 /
    # 15.74 / 15.54 ms** (min) on the grid against **27.61 / 27.99 / 28.12** on the sphere query.
    #
    # The reason is that this is a *small-radius, well-centred* query, which is the hash grid's
    # best case and not the BVH's: the grid was built with cell width ``radius``, so a probe here
    # reaches 27 cells by address arithmetic and stops, while the BVH pays a ~11-level root descent
    # per call and there are millions of calls. **Do not read the 2.4-8.9x that
    # ``pivot_front_edges`` gets from ``wp.tile_bvh_query_aabb`` as transferring to here** (see the
    # ``warp-hashgrid-no-per-cell-entry`` note): that win is a *block-cooperative* walk over a
    # ``2 * radius`` reach with ~88 candidates to split across 32 lanes, and every part of that
    # description is load-bearing. The ~1.91x candidate saving a ball enumeration does deliver is
    # not worth a tree descent at this radius.
    threshold = radius - BALL_EPS * radius
    threshold_sq = threshold * threshold
    query = wp.hash_grid_query(grid_id, center, radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            if wp.length_sq(center - points[j]) < threshold_sq:
                return False
    return True


@wp.kernel(enable_backward=False)
def begin_wave(counters: wp.array[wp.int32]) -> None:
    # Reset the per-wave counters and snapshot the face count the progress test compares against.
    # The vertex claim is *not* cleared here: ``propose_triangle`` clears the three vertices it is
    # about to contend for, which is the only part of an n-sized array a wave ever reads.
    if counters[CNT_CONTINUE] == 0:
        return
    counters[CNT_NEXT_FRONT] = 0
    counters[CNT_LIVE] = 0
    counters[CNT_PROPOSAL] = 0
    counters[CNT_PREV_FACE] = counters[CNT_FACE]


@wp.func
def push_front_edge(
    slot: wp.int32, capacity: wp.int32, counters: wp.array[wp.int32], out_front: wp.array[wp.int32]
) -> None:
    # Append an edge to the outgoing front list, bounded by the list's allocated capacity.
    #
    # ``front_out`` is sized ``3 * max_faces + n``, exactly the worst case the commit budget
    # admits, so there is no slack: any accounting slip would scatter past the end, and because
    # the buffer comes from the memory pool such a write lands in *another live allocation*
    # and corrupts it silently rather than faulting.
    #
    # Overflow is *recoverable*, which is why this drops rather than clamps: the front list is a
    # cache of the edge hash table, not the source of truth, and ``_BpaState.grow`` rebuilds it from
    # that table with ``collect_front_from_table``. So a dropped entry is restored by the same host
    # round-trip that doubles the budget, and raising ``CNT_GROW`` is what asks for it.
    #
    # This is hardening, **not** a fix for the intermittent ``CUDA error 700`` that
    # ``ball_pivoting`` still shows on a multi-cloud run: instrumenting ``CNT_OVERFLOW`` measured it
    # at **0**, with the front peaking near 0.1 % of capacity, so these scatters were cleared as the
    # cause. That fault's cause is still unidentified -- it needs the CUDA memory pool, and is
    # invisible to both ``compute-sanitizer`` and Warp's debug mode.
    position = wp.atomic_add(counters, CNT_NEXT_FRONT, 1)
    if position < capacity:
        out_front[position] = slot
        return
    counters[CNT_OVERFLOW] = 1
    counters[CNT_GROW] = 1


@wp.func
def propose_triangle(
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    # Append a candidate triangle to this wave's proposal list. Both the seed and the pivot path
    # write here, so the claim and commit kernels have a single dense list to walk.
    #
    # Releasing the three vertices for this wave's claim happens here rather than in a sweep over
    # all n: only a proposed vertex is ever read back, and two proposals sharing a vertex write the
    # same sentinel. That removes an n-wide launch from every wave, which on a small cloud was most
    # of the wave. The sentinel is the largest ``uint64`` because the claim is a ``wp.atomic_min``
    # over ``proposal_key``, which is unsigned and bounded by 2^63.
    out_owner[a] = UINT64_MAX_CONSTANT
    out_owner[b] = UINT64_MAX_CONSTANT
    out_owner[c] = UINT64_MAX_CONSTANT
    slot = wp.atomic_add(counters, CNT_PROPOSAL, 1)
    if slot >= capacity:
        # Dropping a proposal is safe and self-healing: the front edge that raised it stays live and
        # is searched again next wave. Only the *write* would be unsafe.
        counters[CNT_OVERFLOW] = 1
        return
    out_a[slot] = a
    out_b[slot] = b
    out_c[slot] = c


@wp.kernel(enable_backward=False)
def seed_triangles(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    point_used: wp.array[wp.bool],
    grid_id: wp.uint64,
    radius: wp.float32,
    clustering: wp.float32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    if counters[CNT_CONTINUE] == 0 or counters[CNT_SEEDING] == 0:
        return
    p = wp.int32(wp.tid())
    if point_used[p]:
        return

    # Gather the local orphan neighbourhood into scratch so it can be scanned as a double loop.
    # Only the lowest-index orphan in each neighbourhood seeds, so seed fronts start well separated
    # and do not collide into overlapping sheets before they can glue.
    #
    # ``wp.zeros``, not the ``wp.types.vector(length=K)`` register row that ``neighbors.py``'s k-NN
    # kernels hold their candidates in: measured, both spellings of this kernel end to end on bunny
    # at radius 2x the mean edge, the register row is 0.994x (min) / 1.000x (median) — no gain. The
    # k-NN row wins because every access to it is an unrolled compile-time slot; every access here
    # is a runtime index (``nbr[count]``, ``nbr[i0]``, ``nbr[i1]``), and a runtime index into a
    # vector spills it to local memory, which is where ``wp.zeros`` already puts it.
    #
    # And ``wp.fixedarray`` is not a third option: its own docstring says it is "only used during
    # codegen, and for type hints" -- it *is* the codegen type of a kernel-scope ``wp.zeros``, not a
    # separate storage class. So the choice here is registers or the stack, and both are measured.
    nbr = wp.zeros(shape=MAX_SEED_NEIGHBORS, dtype=wp.int32)
    count = wp.int32(0)
    query = wp.hash_grid_query(grid_id, points[p], 2.0 * radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if count >= MAX_SEED_NEIGHBORS:
            break
        if j != p and not point_used[j]:
            if j < p:
                return  # a lower-index orphan neighbour will seed this neighbourhood instead
            nbr[count] = j
            count += 1

    # Squared, matching the same vcglib clustering rule as tested in ``candidate_is_viable``; see
    # ``ball_is_empty`` for why one module must not spell one predicate two ways.
    min_cluster_sq = clustering * radius * clustering * radius
    for i0 in range(count):
        a = nbr[i0]
        for i1 in range(i0 + 1, count):
            b = nbr[i1]
            if wp.length_sq(points[a] - points[b]) < min_cluster_sq:
                continue
            center = compute_ball_center(
                points[p], points[a], points[b], normals[p] + normals[a] + normals[b], radius
            )
            if center[0] == wp.inf:
                continue
            if not is_compatible(
                points[p], points[a], points[b], normals[p], normals[a], normals[b]
            ):
                continue
            if ball_is_empty(grid_id, points, center, radius, p, a, b):
                propose_triangle(p, a, b, front_capacity, counters, out_owner, out_a, out_b, out_c)
                return


@wp.func
def point_is_available(
    p: wp.int32, point_used: wp.array[wp.bool], boundary_degree: wp.array[wp.int32]
) -> wp.bool:
    # Orphan, or still on the advancing front. A used point with no incident boundary edge is
    # fully interior and can never be pivoted onto again — which is what makes retirement sound.
    return (not point_used[p]) or boundary_degree[p] > 0


@wp.func
def edge_is_interior(u: wp.int32, v: wp.int32, edges: BpaEdgeTable) -> wp.bool:
    # Manifold guard: an edge already shared by two triangles may not gain a third. One hash probe,
    # where the previous design needed a binary search into a freshly sorted key array.
    slot = hash_find(pack_edge_key(u, v, edges.key_base), edges.key, edges.mask)
    return slot >= 0 and edges.count[slot] >= 2


@wp.func
def candidate_prefilter(
    points: wp.array[wp.vec3],
    p_src: wp.vec3,
    p_tgt: wp.vec3,
    src: wp.int32,
    tgt: wp.int32,
    opp: wp.int32,
    c: wp.int32,
    min_cluster_sq: wp.float32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
) -> wp.bool:
    # The cheap half of the candidate test: identity, availability and vcglib clustering. Compared
    # squared, since this runs once per point the grid hands back.
    if c == src or c == tgt or c == opp:
        return False
    if not point_is_available(c, point_used, boundary_degree):
        return False
    p_c = points[c]
    if wp.length_sq(p_c - p_src) < min_cluster_sq:
        return False
    return wp.length_sq(p_c - p_tgt) >= min_cluster_sq


@wp.func
def candidate_accepted(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    p_src: wp.vec3,
    p_tgt: wp.vec3,
    tri_norm: wp.vec3,
    src: wp.int32,
    tgt: wp.int32,
    c: wp.int32,
    center: wp.vec3,
    grid_id: wp.uint64,
    radius: wp.float32,
    crease_cos: wp.float32,
    edges: BpaEdgeTable,
) -> wp.bool:
    # The expensive half: crease, manifoldness, normal compatibility and the empty-ball test, in
    # increasing order of cost. Shared verbatim between the full search and the O(1) re-validation
    # of a cached candidate, so the two can never disagree about what a valid candidate is.
    #
    # ``tri_norm`` (the pivoting triangle's normal) is loop-invariant and passed in.
    if (
        crease_cos > -1.0
        and wp.abs(wp.dot(tri_norm, triangle_normal(p_src, p_tgt, points[c]))) < crease_cos
    ):
        return False
    if edge_is_interior(src, c, edges):
        return False
    if edge_is_interior(tgt, c, edges):
        return False
    if not is_compatible(p_src, p_tgt, points[c], normals[src], normals[tgt], normals[c]):
        return False
    return ball_is_empty(grid_id, points, center, radius, src, tgt, c)


@wp.kernel(enable_backward=False)
def pivot_front_edges(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    grid_id: wp.uint64,
    bvh_id: wp.uint64,
    radius: wp.float32,
    clustering: wp.float32,
    crease_cos: wp.float32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_in: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    out_owner: wp.array[wp.uint64],
    front_out: wp.array[wp.int32],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    # **One block per front edge, one warp per block.** The pivot search is a neighbourhood walk per
    # edge, and a wave has only a few hundred live edges (median 312 on ``bunny_decimated``), so
    # thread-per-edge left the device idle: the wave cost was nearly flat in the front size,
    # 0.425 ms at a front under 64 against 1.636 at 1024-4096. A warp an edge is what fills it.
    #
    # The candidate walk therefore runs on the **BVH**, not the hash grid: Warp's hash grid exposes
    # only a sequential per-thread iterator with no per-cell entry point, so its walk cannot be
    # split across lanes — and the walk is 70-73 % of a query's cost. ``tile_bvh_query_aabb``
    # hands one candidate per lane per step instead. Measured on this query shape, 2.4-8.9x over
    # the hash grid at the front sizes a wave actually has. (A *serial* BVH walk is 1.8x
    # **slower** than the hash grid, so the win is the cooperation, not the tree.)
    #
    # The box the BVH returns is a different superset of the true candidate set than the grid's
    # cells were — both are supersets, and ``candidate_prefilter`` plus ``compute_ball_center``'s
    # ``inf`` do the actual rejecting, so the accepted set is unchanged.
    #
    # ``ball_is_empty`` inside ``candidate_accepted`` stays a per-lane serial hash-grid query: at
    # that point every lane is testing a *different* candidate ball, so there is nothing for the
    # block to cooperate on. It runs 32-way concurrently instead, which is where its speedup comes
    # from.
    if counters[CNT_CONTINUE] == 0:
        return
    block, lane = wp.tid()
    seeding = counters[CNT_SEEDING] != 0
    min_cluster_sq = (clustering * radius) * (clustering * radius)
    # Grid-stride over the front so the launch dimension is a fixed constant — a hard requirement
    # for capturing the wave loop as a CUDA graph, and it also keeps the cost proportional to the
    # front rather than to its high-water mark.
    for i in range(block, counters[CNT_FRONT], grid_stride):
        slot = front_in[i]
        # An edge leaves the front for good when a second face closes it or its search failed.
        # Every lane reads the same state, so the whole block takes this branch together and the
        # tile reductions below are always reached uniformly.
        if edges.count[slot] != 1 or edges.state[slot] != EDGE_LIVE:
            continue
        if lane == 0:
            push_front_edge(slot, front_capacity, counters, front_out)
            wp.atomic_add(counters, CNT_LIVE, 1)
        if seeding:
            continue  # a seeding wave only carries the front forward

        src = edges.src[slot]
        tgt = edges.tgt[slot]
        opp = edges.opp[slot]
        p_src = points[src]
        p_tgt = points[tgt]
        tri_norm = triangle_normal(p_src, p_tgt, points[opp])

        # Re-validate the cached argmin first. It stays the argmin while it stays valid (the
        # candidate set only shrinks), and ~75-80% of front edges lose the vertex claim each wave
        # and come back here unchanged, so this is the difference from a full neighbourhood search.
        # Every lane evaluates it on identical data — the loads broadcast, so the redundancy is
        # free — and only lane 0 acts on the result.
        cached = edges.cand[slot]
        if cached >= 0 and candidate_prefilter(
            points, p_src, p_tgt, src, tgt, opp, cached, min_cluster_sq, point_used, boundary_degree
        ):
            cached_center = compute_ball_center(
                p_src, p_tgt, points[cached], normals[src] + normals[tgt] + normals[cached], radius
            )
            if cached_center[0] != wp.inf and candidate_accepted(
                points, normals, p_src, p_tgt, tri_norm, src, tgt, cached, cached_center,
                grid_id, radius, crease_cos, edges,
            ):  # fmt: skip
                if lane == 0:
                    propose_triangle(
                        src, tgt, cached, front_capacity, counters, out_owner, out_a, out_b, out_c
                    )
                continue

        center = compute_ball_center(
            p_src, p_tgt, points[opp], normals[src] + normals[tgt] + normals[opp], radius
        )
        if center[0] == wp.inf:
            edges.state[slot] = EDGE_RETIRED  # every lane stores the same value
            continue

        mp = wp.lerp(p_src, p_tgt, 0.5)
        axis = wp.normalize(p_tgt - p_src)
        a_dir = wp.normalize(center - mp)

        # Each lane keeps its own running best over the candidates it is handed. That weakens the
        # "reject a non-improving candidate before the expensive tests" pruning — a lane cannot see
        # the other lanes' minima — so more candidates reach ``candidate_accepted``. It is still the
        # same answer: the global minimum over accepted candidates is the minimum of the per-lane
        # minima, and the extra acceptance tests are paid for by their own 32-way concurrency.
        best_angle = TWO_PI
        best = wp.int32(-1)
        reach = 2.0 * radius
        # ``c < n_points`` is load-bearing, not defensive. ``wp.tile_bvh_query_aabb`` returns
        # **out-of-range** primitive indices whenever one traversal round finds more primitives
        # than its shared result buffer holds: ``warp/native/tile_bvh.h`` counts them with an
        # unconditional ``atomicAdd`` and guards only the *write* against
        # ``result_buffer_capacity = WP_TILE_BLOCK_DIM * 5`` (160 lanes-wide here), then reads
        # ``buffer[counter - block_size + lane]`` -- past the written region once the counter has
        # overrun it, so a lane gets uninitialised shared memory as an index. Diagnosed on
        # ``proximity.face_to_mesh_distance_tiled``, the tree's only other consumer of this
        # builtin, where ``compute-sanitizer`` named it as a read **12.26 GB past the nearest
        # allocation**; a garbage index is positive far more often than not, so the ``>= 0`` test
        # alone lets it through to ``points[c]``.
        #
        # **It is not the long-open intermittent ``CUDA error 700`` recorded against this function**
        # (CLAUDE.md section 16.3), which was the obvious guess -- same builtin, same block width --
        # and was tested rather than assumed: that section's own repro (three *different* clouds in
        # one process) faults **6 of 6** with this guard in place and 6 of 6 without, on the same
        # third cloud. So the guard rules the tile-BVH garbage-index path *out* as that defect's
        # cause, which is the useful half of the result; section 16.3's remaining leads stand.
        #
        # Guarding is still right here, and cheap: the overflow is a property of the builtin and the
        # query, not of the caller, so this walk is exposed to the same corruption whenever a round
        # overruns. What it cannot do is recover the primitives the overflow *dropped* -- that
        # defect is upstream's -- so a guarded round can still return a slightly incomplete
        # candidate set.
        #
        # It is output-neutral **by construction**, which is the only argument available: an index
        # ``>= n_points`` is never a primitive of this BVH, so the test can reject nothing the walk
        # was entitled to return. A measured claim is not available and should not be attempted --
        # this function's output is still run-to-run nondeterministic (``make_winding_consistent``
        # seeds each component from an arbitrary face, section 16.3), and the control proves the
        # comparison is blind: two *baseline* runs of one 40 000-point cloud differ from each other
        # in the face count, 69 346 against 69 347, before any change is applied.
        n_points = points.shape[0]
        query = wp.tile_bvh_query_aabb(bvh_id, mp - wp.vec3(reach), mp + wp.vec3(reach))
        while wp.tile_query_valid(query):
            c = wp.untile(wp.tile_bvh_query_next(query))
            if (
                c >= 0
                and c < n_points
                and candidate_prefilter(
                    points,
                    p_src,
                    p_tgt,
                    src,
                    tgt,
                    opp,
                    c,
                    min_cluster_sq,
                    point_used,
                    boundary_degree,
                )
            ):
                new_center = compute_ball_center(
                    p_src, p_tgt, points[c], normals[src] + normals[tgt] + normals[c], radius
                )
                if new_center[0] != wp.inf:
                    b_dir = wp.normalize(new_center - mp)
                    # Rotation about the edge from the current ball centre to this candidate's,
                    # signed by ``axis`` and folded into [0, 2pi).
                    #
                    # ``dihedral_angle`` is atan2(|a x b| along the axis, a . b), not
                    # ``acos(a . b)``, and here that is a correctness matter rather than a tidy-up:
                    # the candidate that *wins* is the one nearest coplanar with the current
                    # triangle, which is exactly where acos has an infinite derivative. Measured
                    # against a float64 reference on float32 directions, error at a true angle of
                    # 1e-3 / 1e-5 / 1e-7 rad: acos 2.3e-05 / 1.0e-05 / 1.0e-07, atan2 4.2e-11 /
                    # 2.5e-13 / 1.2e-15. At 1e-5 and below the acos error *equals the angle* -- it
                    # returns zero, so every candidate closer than ~1e-5 rad ranked identically and
                    # the winner fell out of float32 rounding. The two forms agree to 7.6e-12 over
                    # 200k random pairs otherwise, and this one is also cheaper: the cross product
                    # is the same one the sign test needed.
                    angle = dihedral_angle(a_dir, b_dir, axis)
                    if angle < 0.0:
                        angle += TWO_PI
                    if angle < best_angle and candidate_accepted(
                        points, normals, p_src, p_tgt, tri_norm, src, tgt, c, new_center,
                        grid_id, radius, crease_cos, edges,
                    ):  # fmt: skip
                        best_angle = angle
                        best = c

        # The winner must not depend on which lane happened to see it, which is what
        # ``tile_argmin``'s second stage is for. When no lane found anything every lane still holds
        # ``(TWO_PI, -1)``, so it returns -1.
        _block_angle, block_best = tile_argmin(best_angle, best)

        edges.cand[slot] = block_best
        if block_best < 0:
            edges.state[slot] = EDGE_RETIRED  # provably impossible; see the module docstring
        elif lane == 0:
            propose_triangle(
                src, tgt, block_best, front_capacity, counters, out_owner, out_a, out_b, out_c
            )


@wp.func
def wave_salt(wave: wp.int32) -> wp.int32:
    # Per-wave permutation seed for ``proposal_key``. Knuth's multiplicative hash, so consecutive
    # wave numbers give unrelated salts rather than salts differing in one bit, masked to 31 bits so
    # xoring it into a non-negative point index cannot set the sign bit -- which is what keeps the
    # packed key under 2^63 and clear of the unclaimed sentinel.
    return wp.int32((wp.uint32(wave) * wp.uint32(2654435761)) & wp.uint32(0x7FFFFFFF))


@wp.func
def proposal_key(a: wp.int32, b: wp.int32, salt: wp.int32) -> wp.uint64:
    # Priority of a proposed triangle, and the whole reason a run is reproducible: it is derived
    # from the proposal's own vertices, not from the order it reached ``propose_triangle``. The
    # slot that call hands out comes from a ``wp.atomic_add``, so it is the arrival order of a
    # thousand-block launch; making it the claim priority made *which* triangle won a contested
    # vertex a function of GPU scheduling, and the loser's front edge is retried a wave later
    # against mutated state, so the difference compounded into a different mesh (measured 44 179 /
    # 44 243 / 44 208 faces over three runs of one build) rather than a permuted one.
    #
    # ``a`` and ``b`` are the proposal's *source edge* -- the pivoting front edge ``(src, tgt)``,
    # or a seeding point and its first partner -- and that pair is unique among one wave's
    # proposals, which is what makes this a total order on them:
    #
    # * a pivot wave proposes at most once per live front edge, and a front edge *is* one slot of
    #   a table keyed by the undirected pair, so no two share ``{src, tgt}``;
    # * a seed wave proposes at most once per seeding point ``p``, and ``seed_triangles`` seeds
    #   only a point that is the lowest-indexed unused point of its own neighbourhood, so ``p`` is
    #   the smaller of its pair and distinct seeds give distinct pairs;
    # * the two never share a wave -- ``pivot_front_edges`` proposes nothing while ``CNT_SEEDING``.
    #
    # ``salt`` rotates that order **per wave**, and it is what keeps the ordering from being global.
    # An order fixed for the whole run starves a high-key front edge: it loses every contested
    # vertex it ever enters, is retried, and loses again, so a wave commits a smaller independent
    # set than a per-wave order gives. Measured over three trees in one session, worktree A/B on
    # ``bunny_decimated`` / ``bunny``: the arrival-order predecessor ran 80 / 152 waves, a globally
    # fixed key 208 / 304, and this salted key 96 / 176 -- 1.70x / 1.35x of wall clock recovered
    # against the fixed key, at an unchanged face count. Salting gives none of that back to the
    # scheduler: the salt is a function of ``CNT_WAVE``, device state the wave loop advances
    # deterministically, so a run remains reproducible call to call.
    #
    # Xoring both halves preserves the injectivity the total order needs: ``{a, b}`` is recoverable
    # from ``(min ^ salt, max ^ salt)``, so distinct source edges still give distinct keys within a
    # wave. The salt is masked to 31 bits precisely so this cannot break the bound below.
    #
    # A bit pack rather than ``pack_edge_key``'s ``lo + hi * base``, so that neither kernel
    # computing it has to carry the point count as an argument and so it cannot overflow: point
    # indices are ``int32`` and the salt cannot set their sign bit, so the key stays under 2^63 and
    # well below the unclaimed sentinel.
    # Spelled as a multiply because that is the same operation on ``uint64`` as a 32-bit shift and
    # reads as the pack it is.
    lo = wp.uint64(wp.uint32(wp.min(a, b) ^ salt))
    hi = wp.uint64(wp.uint32(wp.max(a, b) ^ salt))
    return hi * wp.uint64(4294967296) + lo


@wp.kernel(enable_backward=False)
def claim_triangle_vertices(
    tri_a: wp.array[wp.int32],
    tri_b: wp.array[wp.int32],
    tri_c: wp.array[wp.int32],
    grid_stride: wp.int32,
    proposal_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.uint64],
) -> None:
    # Priority claim (lowest ``proposal_key`` wins) on all three vertices of each proposed triangle.
    if counters[CNT_CONTINUE] == 0:
        return
    # ``propose_triangle`` increments the counter past the capacity before dropping, so the count
    # is not a safe bound on the list it indexes.
    proposals = wp.min(counters[CNT_PROPOSAL], proposal_capacity)
    # Same wave index, and so the same salt, as ``commit_triangles`` reads: ``end_wave`` is what
    # advances ``CNT_WAVE`` and it runs after both.
    salt = wave_salt(counters[CNT_WAVE])
    for t in range(wp.int32(wp.tid()), proposals, grid_stride):
        key = proposal_key(tri_a[t], tri_b[t], salt)
        wp.atomic_min(out_owner, tri_a[t], key)
        wp.atomic_min(out_owner, tri_b[t], key)
        wp.atomic_min(out_owner, tri_c[t], key)


@wp.func
def register_face_edge(
    u: wp.int32,
    v: wp.int32,
    opp: wp.int32,
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
) -> None:
    # Fold one edge of a just-committed triangle into the persistent state.
    #
    # A wave's committed triangles are vertex-disjoint (that is exactly what the ``owner`` claim
    # buys), so they share no edge and no vertex: within a wave each edge slot is touched by one
    # thread. The atomics still make it safe, but the count transitions are unambiguous — 0 -> 1
    # means a new boundary edge, 1 -> 2 means one just closed.
    slot = hash_find_or_insert(pack_edge_key(u, v, edges.key_base), edges.key, edges.mask)
    previous = wp.atomic_add(edges.count, slot, 1)
    if previous == 0:
        edges.src[slot] = u
        edges.tgt[slot] = v
        edges.opp[slot] = opp
        edges.cand[slot] = -1
        push_front_edge(slot, front_capacity, counters, front_out)
        wp.atomic_add(boundary_degree, u, 1)
        wp.atomic_add(boundary_degree, v, 1)
    elif previous == 1:
        wp.atomic_add(boundary_degree, u, -1)
        wp.atomic_add(boundary_degree, v, -1)


@wp.kernel(enable_backward=False)
def commit_triangles(
    tri_a: wp.array[wp.int32],
    tri_b: wp.array[wp.int32],
    tri_c: wp.array[wp.int32],
    owner: wp.array[wp.uint64],
    max_faces: wp.int32,
    grid_stride: wp.int32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The single mutator of the persistent state: a proposal that owns all three of its vertices
    # this wave becomes a triangle, and the same thread folds it into the face buffer, the used
    # mask, the edge table, the boundary degrees and the outgoing front.
    if counters[CNT_CONTINUE] == 0:
        return
    # See ``claim_triangle_vertices``: the proposal counter overshoots its list on overflow.
    proposals = wp.min(counters[CNT_PROPOSAL], front_capacity)
    # The triangle budget is tested for the *whole* wave rather than per triangle, because a
    # partial commit would hand the last slots out in ``wp.atomic_add`` arrival order — the one
    # thing this design does not let the answer depend on. Declining every proposal is also the
    # cheaper recovery: nothing is mutated, so each front edge keeps its cached candidate and the
    # same wave is re-proposed unchanged once the host has doubled the budget.
    #
    # ``CNT_PREV_FACE``, not ``CNT_FACE``: ``begin_wave`` snapshots it and no thread writes it, so
    # every thread reads the same value. Reading ``CNT_FACE`` here would race the ``atomic_add``
    # below and let a late thread decline a wave the early ones already committed to.
    #
    # It is conservative — most proposals lose the claim and never needed a slot — but the budget
    # starts at ``4 n + 16`` against a run that commits about ``2 n``, so this is the growth tail
    # and not the common path. Bounding the wave this way is also what lets the scatter below drop
    # its own range check: committed triangles are at most ``proposals``.
    if counters[CNT_PREV_FACE] + proposals > max_faces:
        counters[CNT_GROW] = 1
        return
    salt = wave_salt(counters[CNT_WAVE])
    for t in range(wp.int32(wp.tid()), proposals, grid_stride):
        a = tri_a[t]
        b = tri_b[t]
        c = tri_c[t]
        key = proposal_key(a, b, salt)
        if owner[a] != key or owner[b] != key or owner[c] != key:
            continue
        slot = wp.atomic_add(counters, CNT_FACE, 1)

        out_faces[slot * 3 + 0] = a
        out_faces[slot * 3 + 1] = b
        out_faces[slot * 3 + 2] = c
        point_used[a] = True
        point_used[b] = True
        point_used[c] = True
        register_face_edge(a, b, c, boundary_degree, front_capacity, counters, edges, front_out)
        register_face_edge(b, c, a, boundary_degree, front_capacity, counters, edges, front_out)
        register_face_edge(c, a, b, boundary_degree, front_capacity, counters, edges, front_out)


@wp.kernel(enable_backward=False)
def end_wave(max_waves: wp.int32, counters: wp.array[wp.int32]) -> None:
    # Advance the device-resident wave state and decide whether the loop keeps going.
    #
    # A wave either pivots the current front or seeds orphans; seeding runs whenever the previous
    # wave committed nothing, which is also the termination test — a stalled pivot followed by a
    # stalled seed means there is nothing left to do.
    if counters[CNT_CONTINUE] == 0:
        return
    counters[CNT_FRONT] = counters[CNT_NEXT_FRONT]
    counters[CNT_WAVE] = counters[CNT_WAVE] + 1
    if counters[CNT_GROW] != 0:
        # A wave that asked the host for room says nothing about progress, and must not be read as
        # a stall: ``commit_triangles`` declines the whole wave when the triangle budget is short,
        # and a front-list overflow is recovered by rebuilding the front from the edge table. Both
        # re-run this wave's work. Testing progress here would flip a pivot wave to seeding, or —
        # if this one was already seeding — raise ``CNT_DONE`` and end the run with a live front,
        # since ``_bpa_run`` checks done before it checks grow.
        counters[CNT_CONTINUE] = 0
        return
    if counters[CNT_FACE] == counters[CNT_PREV_FACE]:
        if counters[CNT_SEEDING] != 0:
            counters[CNT_DONE] = 1
            counters[CNT_CONTINUE] = 0
        else:
            counters[CNT_SEEDING] = 1
    else:
        counters[CNT_SEEDING] = 0
    # Hand control back to the host to compact a front that has accumulated too many retired
    # entries, or because the wave cap was hit. (Growing the budget is the early return above.)
    if counters[CNT_WAVE] >= max_waves:
        counters[CNT_CONTINUE] = 0
    elif counters[CNT_NEXT_FRONT] > 4 * counters[CNT_LIVE] + 1024:
        counters[CNT_CONTINUE] = 0


@wp.kernel(enable_backward=False)
def compact_front(
    front_in: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    front_out: wp.array[wp.int32],
) -> None:
    # Drop closed and retired edges from the front. ``pivot_front_edges`` already does this as a
    # side effect, but a long run of seeding waves (which carry the front forward untouched) or a
    # burst of commits can still leave it sparse; the caller runs this when it does.
    for i in range(wp.int32(wp.tid()), counters[CNT_FRONT], grid_stride):
        slot = front_in[i]
        if edges.count[slot] == 1 and edges.state[slot] == EDGE_LIVE:
            push_front_edge(slot, front_capacity, counters, front_out)


@wp.kernel(enable_backward=False)
def rehash_edges(
    old_key: wp.array[wp.uint64],
    old_count: wp.array[wp.int32],
    old_src: wp.array[wp.int32],
    old_tgt: wp.array[wp.int32],
    old_opp: wp.array[wp.int32],
    old_state: wp.array[wp.int32],
    old_cand: wp.array[wp.int32],
    new_mask: wp.int32,
    new_key: wp.array[wp.uint64],
    new_count: wp.array[wp.int32],
    new_src: wp.array[wp.int32],
    new_tgt: wp.array[wp.int32],
    new_opp: wp.array[wp.int32],
    new_state: wp.array[wp.int32],
    new_cand: wp.array[wp.int32],
) -> None:
    # Re-insert every occupied slot into a larger table when the triangle budget grows. Slot
    # indices change, so the caller rebuilds the front list from the new table afterwards.
    h = wp.int32(wp.tid())
    stored = old_key[h]
    if stored == wp.uint64(0):
        return
    key = stored - wp.uint64(1)  # undo ``encode_key``: 0 is the empty sentinel
    slot = hash_find_or_insert(key, new_key, new_mask)
    new_count[slot] = old_count[h]
    new_src[slot] = old_src[h]
    new_tgt[slot] = old_tgt[h]
    new_opp[slot] = old_opp[h]
    new_state[slot] = old_state[h]
    new_cand[slot] = old_cand[h]


@wp.kernel(enable_backward=False)
def collect_front_from_table(
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    edges: BpaEdgeTable,
    out_front: wp.array[wp.int32],
) -> None:
    # Rebuild the front list by scanning the edge table, after a rehash has moved every slot.
    h = wp.int32(wp.tid())
    if edges.key[h] == wp.uint64(0):
        return
    if edges.count[h] == 1 and edges.state[h] == EDGE_LIVE:
        push_front_edge(h, front_capacity, counters, out_front)
