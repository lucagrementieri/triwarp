"""
Kernels for wave-parallel ball-pivoting surface reconstruction.

Ports the geometry of Open3D's ``SurfaceReconstructionBallPivoting.cpp`` (``ComputeBallCenter``,
``IsCompatible``, ``FindCandidateVertex``, ``TryTriangleSeed``) to Warp. Instead of the serial
advancing front, every wave pivots a conflict-free independent set of front edges at once and
commits them through a two-phase vertex claim (``wp.atomic_min`` priority on all three vertices),
so at least the globally-lowest-priority triangle always commits and the loop cannot livelock.

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

from triwarp.constants import INT32_MAX_CONSTANT
from triwarp.kernels.grouping import hash_find, hash_find_or_insert, pack_edge_key
from triwarp.kernels.predicates import triangle_normal as face_normal

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
    tr_norm = face_normal(v1, v2, v3)
    pt_norm = wp.normalize(normal_sum)
    if wp.dot(tr_norm, pt_norm) < 0.0:
        tr_norm = -tr_norm
    return circ_center + wp.sqrt(height) * tr_norm


@wp.func
def is_compatible(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, na: wp.vec3, nb: wp.vec3, nc: wp.vec3
) -> bool:
    # The triangle normal must agree (within tolerance) with all three oriented point normals.
    normal = face_normal(a, b, c)
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
) -> bool:
    # True when no point other than the three defining ones lies strictly inside the ball.
    threshold = radius - BALL_EPS * radius
    query = wp.hash_grid_query(grid_id, center, radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            if wp.length(center - points[j]) < threshold:
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
    # cause. See ``plans/benchmark-improve.md`` C0 for what is still open.
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
    out_owner: wp.array[wp.int32],
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
    # of the wave.
    out_owner[a] = INT32_MAX_CONSTANT
    out_owner[b] = INT32_MAX_CONSTANT
    out_owner[c] = INT32_MAX_CONSTANT
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
    out_owner: wp.array[wp.int32],
    out_a: wp.array[wp.int32],
    out_b: wp.array[wp.int32],
    out_c: wp.array[wp.int32],
) -> None:
    if counters[CNT_CONTINUE] == 0 or counters[CNT_SEEDING] == 0:
        return
    p = int(wp.tid())
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

    min_cluster = clustering * radius
    for i0 in range(count):
        a = nbr[i0]
        for i1 in range(i0 + 1, count):
            b = nbr[i1]
            if wp.length(points[a] - points[b]) < min_cluster:
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
) -> bool:
    # Orphan, or still on the advancing front. A used point with no incident boundary edge is
    # fully interior and can never be pivoted onto again — which is what makes retirement sound.
    return (not point_used[p]) or boundary_degree[p] > 0


@wp.func
def edge_is_interior(
    u: wp.int32,
    v: wp.int32,
    key_base: wp.uint64,
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_mask: wp.int32,
) -> bool:
    # Manifold guard: an edge already shared by two triangles may not gain a third. One hash probe,
    # where the previous design needed a binary search into a freshly sorted key array.
    slot = hash_find(pack_edge_key(u, v, key_base), edge_key, edge_mask)
    return slot >= 0 and edge_count[slot] >= 2


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
) -> bool:
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
    key_base: wp.uint64,
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_mask: wp.int32,
) -> bool:
    # The expensive half: crease, manifoldness, normal compatibility and the empty-ball test, in
    # increasing order of cost. Shared verbatim between the full search and the O(1) re-validation
    # of a cached candidate, so the two can never disagree about what a valid candidate is.
    #
    # ``tri_norm`` (the pivoting triangle's normal) is loop-invariant and passed in.
    if (
        crease_cos > -1.0
        and wp.abs(wp.dot(tri_norm, face_normal(p_src, p_tgt, points[c]))) < crease_cos
    ):
        return False
    if edge_is_interior(src, c, key_base, edge_key, edge_count, edge_mask):
        return False
    if edge_is_interior(tgt, c, key_base, edge_key, edge_count, edge_mask):
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
    key_base: wp.uint64,
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_src: wp.array[wp.int32],
    edge_tgt: wp.array[wp.int32],
    edge_opp: wp.array[wp.int32],
    edge_state: wp.array[wp.int32],
    edge_cand: wp.array[wp.int32],
    edge_mask: wp.int32,
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_in: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.int32],
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
        if edge_count[slot] != 1 or edge_state[slot] != EDGE_LIVE:
            continue
        if lane == 0:
            push_front_edge(slot, front_capacity, counters, front_out)
            wp.atomic_add(counters, CNT_LIVE, 1)
        if seeding:
            continue  # a seeding wave only carries the front forward

        src = edge_src[slot]
        tgt = edge_tgt[slot]
        opp = edge_opp[slot]
        p_src = points[src]
        p_tgt = points[tgt]
        tri_norm = face_normal(p_src, p_tgt, points[opp])

        # Re-validate the cached argmin first. It stays the argmin while it stays valid (the
        # candidate set only shrinks), and ~75-80% of front edges lose the vertex claim each wave
        # and come back here unchanged, so this is the difference from a full neighbourhood search.
        # Every lane evaluates it on identical data — the loads broadcast, so the redundancy is
        # free — and only lane 0 acts on the result.
        cached = edge_cand[slot]
        if cached >= 0 and candidate_prefilter(
            points, p_src, p_tgt, src, tgt, opp, cached, min_cluster_sq, point_used, boundary_degree
        ):
            cached_center = compute_ball_center(
                p_src, p_tgt, points[cached], normals[src] + normals[tgt] + normals[cached], radius
            )
            if cached_center[0] != wp.inf and candidate_accepted(
                points, normals, p_src, p_tgt, tri_norm, src, tgt, cached, cached_center,
                grid_id, radius, crease_cos, key_base, edge_key, edge_count, edge_mask,
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
            edge_state[slot] = EDGE_RETIRED  # every lane stores the same value
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
        query = wp.tile_bvh_query_aabb(bvh_id, mp - wp.vec3(reach), mp + wp.vec3(reach))
        while wp.tile_query_valid(query):
            c = wp.untile(wp.tile_bvh_query_next(query))
            if c >= 0 and candidate_prefilter(
                points, p_src, p_tgt, src, tgt, opp, c, min_cluster_sq, point_used, boundary_degree
            ):
                new_center = compute_ball_center(
                    p_src, p_tgt, points[c], normals[src] + normals[tgt] + normals[c], radius
                )
                if new_center[0] != wp.inf:
                    b_dir = wp.normalize(new_center - mp)
                    angle = wp.acos(wp.dot(a_dir, b_dir))  # wp.acos auto-clamps to [-1, 1]
                    if wp.dot(wp.cross(a_dir, b_dir), axis) < 0.0:
                        angle = TWO_PI - angle
                    if angle < best_angle and candidate_accepted(
                        points, normals, p_src, p_tgt, tri_norm, src, tgt, c, new_center,
                        grid_id, radius, crease_cos, key_base, edge_key, edge_count, edge_mask,
                    ):  # fmt: skip
                        best_angle = angle
                        best = c

        # Two-stage reduction, so the winner does not depend on which lane happened to see it:
        # smallest angle, then smallest point index among the lanes attaining it. When no lane found
        # anything every lane still holds ``(TWO_PI, -1)``, so the second stage returns -1.
        block_angle = wp.tile_min(wp.tile(best_angle))[0]
        mine = best
        if best_angle != block_angle:
            mine = INT32_MAX_CONSTANT
        block_best = wp.tile_min(wp.tile(mine))[0]

        edge_cand[slot] = block_best
        if block_best < 0:
            edge_state[slot] = EDGE_RETIRED  # provably impossible; see the module docstring
        elif lane == 0:
            propose_triangle(
                src, tgt, block_best, front_capacity, counters, out_owner, out_a, out_b, out_c
            )


@wp.kernel(enable_backward=False)
def claim_triangle_vertices(
    tri_a: wp.array[wp.int32],
    tri_b: wp.array[wp.int32],
    tri_c: wp.array[wp.int32],
    grid_stride: wp.int32,
    counters: wp.array[wp.int32],
    out_owner: wp.array[wp.int32],
) -> None:
    # Priority claim (lowest index wins) on all three vertices of each proposed triangle.
    if counters[CNT_CONTINUE] == 0:
        return
    for t in range(int(wp.tid()), counters[CNT_PROPOSAL], grid_stride):
        wp.atomic_min(out_owner, tri_a[t], t)
        wp.atomic_min(out_owner, tri_b[t], t)
        wp.atomic_min(out_owner, tri_c[t], t)


@wp.func
def register_face_edge(
    u: wp.int32,
    v: wp.int32,
    opp: wp.int32,
    key_base: wp.uint64,
    edge_mask: wp.int32,
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_src: wp.array[wp.int32],
    edge_tgt: wp.array[wp.int32],
    edge_opp: wp.array[wp.int32],
    edge_cand: wp.array[wp.int32],
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    front_out: wp.array[wp.int32],
) -> None:
    # Fold one edge of a just-committed triangle into the persistent state.
    #
    # A wave's committed triangles are vertex-disjoint (that is exactly what the ``owner`` claim
    # buys), so they share no edge and no vertex: within a wave each edge slot is touched by one
    # thread. The atomics still make it safe, but the count transitions are unambiguous — 0 -> 1
    # means a new boundary edge, 1 -> 2 means one just closed.
    slot = hash_find_or_insert(pack_edge_key(u, v, key_base), edge_key, edge_mask)
    previous = wp.atomic_add(edge_count, slot, 1)
    if previous == 0:
        edge_src[slot] = u
        edge_tgt[slot] = v
        edge_opp[slot] = opp
        edge_cand[slot] = -1
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
    owner: wp.array[wp.int32],
    max_faces: wp.int32,
    key_base: wp.uint64,
    edge_mask: wp.int32,
    grid_stride: wp.int32,
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_src: wp.array[wp.int32],
    edge_tgt: wp.array[wp.int32],
    edge_opp: wp.array[wp.int32],
    edge_cand: wp.array[wp.int32],
    point_used: wp.array[wp.bool],
    boundary_degree: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    front_out: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The single mutator of the persistent state: a proposal that owns all three of its vertices
    # this wave becomes a triangle, and the same thread folds it into the face buffer, the used
    # mask, the edge table, the boundary degrees and the outgoing front.
    if counters[CNT_CONTINUE] == 0:
        return
    for t in range(int(wp.tid()), counters[CNT_PROPOSAL], grid_stride):
        a = tri_a[t]
        b = tri_b[t]
        c = tri_c[t]
        if owner[a] != t or owner[b] != t or owner[c] != t:
            continue
        if counters[CNT_FACE] >= max_faces:
            # Out of budget. Nothing is mutated, so the front edge keeps its cached candidate and
            # is simply re-proposed after the caller grows the buffer — no triangle is lost.
            counters[CNT_GROW] = 1
            continue
        slot = wp.atomic_add(counters, CNT_FACE, 1)
        if slot >= max_faces:
            counters[CNT_GROW] = 1
            continue

        out_faces[slot * 3 + 0] = a
        out_faces[slot * 3 + 1] = b
        out_faces[slot * 3 + 2] = c
        point_used[a] = True
        point_used[b] = True
        point_used[c] = True
        register_face_edge(
            a, b, c, key_base, edge_mask, edge_key, edge_count, edge_src, edge_tgt, edge_opp,
            edge_cand, boundary_degree, front_capacity, counters, front_out,
        )  # fmt: skip
        register_face_edge(
            b, c, a, key_base, edge_mask, edge_key, edge_count, edge_src, edge_tgt, edge_opp,
            edge_cand, boundary_degree, front_capacity, counters, front_out,
        )  # fmt: skip
        register_face_edge(
            c, a, b, key_base, edge_mask, edge_key, edge_count, edge_src, edge_tgt, edge_opp,
            edge_cand, boundary_degree, front_capacity, counters, front_out,
        )  # fmt: skip


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
    if counters[CNT_FACE] == counters[CNT_PREV_FACE]:
        if counters[CNT_SEEDING] != 0:
            counters[CNT_DONE] = 1
            counters[CNT_CONTINUE] = 0
        else:
            counters[CNT_SEEDING] = 1
    else:
        counters[CNT_SEEDING] = 0
    # Hand control back to the host to grow the triangle budget, to compact a front that has
    # accumulated too many retired entries, or because the wave cap was hit.
    if counters[CNT_GROW] != 0 or counters[CNT_WAVE] >= max_waves:
        counters[CNT_CONTINUE] = 0
    elif counters[CNT_NEXT_FRONT] > 4 * counters[CNT_LIVE] + 1024:
        counters[CNT_CONTINUE] = 0


@wp.kernel(enable_backward=False)
def compact_front(
    front_in: wp.array[wp.int32],
    edge_count: wp.array[wp.int32],
    edge_state: wp.array[wp.int32],
    grid_stride: wp.int32,
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    front_out: wp.array[wp.int32],
) -> None:
    # Drop closed and retired edges from the front. ``pivot_front_edges`` already does this as a
    # side effect, but a long run of seeding waves (which carry the front forward untouched) or a
    # burst of commits can still leave it sparse; the caller runs this when it does.
    for i in range(int(wp.tid()), counters[CNT_FRONT], grid_stride):
        slot = front_in[i]
        if edge_count[slot] == 1 and edge_state[slot] == EDGE_LIVE:
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
    h = int(wp.tid())
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
    edge_key: wp.array[wp.uint64],
    edge_count: wp.array[wp.int32],
    edge_state: wp.array[wp.int32],
    front_capacity: wp.int32,
    counters: wp.array[wp.int32],
    out_front: wp.array[wp.int32],
) -> None:
    # Rebuild the front list by scanning the edge table, after a rehash has moved every slot.
    h = int(wp.tid())
    if edge_key[h] == wp.uint64(0):
        return
    if edge_count[h] == 1 and edge_state[h] == EDGE_LIVE:
        push_front_edge(h, front_capacity, counters, out_front)
