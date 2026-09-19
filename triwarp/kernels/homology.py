"""
Kernels for [`triwarp.homology`][triwarp.homology]: one fused tree-cotree pass.

The module builds, in order, a rooted spanning tree of the vertex graph, a spanning **forest** of
the dual graph over the edges that tree left alone, and then traces one loop per leftover edge.
Every stage here is shaped by the same two facts about this call: it is **launch-bound**, and its
only consumer reads ``parents`` and ``distances`` and nothing else.

**The primal tree is a two-kernel level loop, not an order-exact traversal.** A breadth-first
search that has to reproduce ``scipy.sparse.csgraph.breadth_first_order`` spends most of a level
compacting its claims into a FIFO in ``(rank, ascending node id)`` order -- a tiled block scan plus
a serial advance, measured at 9.9 us of a 26.1 us level against 16.2 for the claim/commit pair that
actually builds the tree. Nothing downstream of a spanning tree reads a discovery order, so the
scan is gone and the level is ``bfs_push_level`` plus a one-thread ``bfs_advance``.

**What replaces the order as the tie-break is ``wp.atomic_min`` on the parent**, and it buys two
things beyond the launch. The tree stays deterministic -- ``parents[w]`` is the lowest-indexed
frontier vertex adjacent to ``w``, whatever order the threads ran in -- and the *adjacency* no
longer has to be column-sorted for that to hold, which is what lets the wrapper build the CSR with
a counting pass and a cursor scatter instead of routing 2 * n_edges triplets through
``warp.sparse.bsr_from_triplets``' radix sort.

``bfs_push_level`` races on ``out_distances`` deliberately: two frontier vertices claiming the same
neighbour both store the same ``level``, so the write is idempotent, and the ``seen == level`` arm
of the visit test is what keeps the second claimer from skipping the ``wp.atomic_min`` it still
has to take part in. Dropping that arm loses the minimum and makes the tree thread-order dependent.

**The dual side is a Boruvka forest, and a forest is all that is needed**: ``homology_generators``
reads the cotree as a *set* -- the generators are the edges in neither tree -- and the loop tracing
walks the **primal** parents. So the dual side has no root and no parent pointers, and its shape is
free to choose. Breadth-first is the one shape that is expensive here: the dual graph restricted to
non-primal-tree edges is *already nearly a tree*, so its diameter is enormous. Measured on the
benchmark meshes, a dual traversal ran **765-891 levels** against the primal's 116-192. Boruvka
needs ``O(log F)`` rounds instead. Each round gives every component its minimum-index incident
candidate edge, accepts those edges and unions the components; the union-find core is
[`find_representative`][triwarp.kernels.algorithms.connected_components.find_representative],
imported rather than re-derived, so the forest and ``graph.connected_component_labels`` share one
implementation of the pointer-jumping find.

**Why the accepted set is a forest.** Edge *indices* are distinct, so "minimum incident index" is a
strict total order on the candidate edges, and the classical Boruvka argument applies: a cycle among
the accepted edges would need a strictly decreasing cycle of minima. The round reads its components
from an immutable ``roots`` snapshot taken before any union, which is what makes that argument exact
here -- deciding against a ``label`` array that other threads are mutating would let one component
accept two edges in a round and could close a cycle.

**Both loops carry their own condition so the wrapper can capture them.** A per-round host readback
drains the pipeline the round's launches just filled, and there were about thirty of them across the
two loops; each loop now writes a device-side condition word that ``wp.capture_while`` reads, and
the wrapper synchronizes three times in the whole call. The Boruvka loop additionally counts its
rounds down in ``FOREST_ROUNDS``: the merge count alone is a sound termination argument, but a
captured loop that fails it hangs the device rather than returning a wrong answer, so the cap is
cheap insurance rather than a schedule.

**The tracing is two kernels and a scan where it used to be a Python loop per generator.** Each
generator edge ``(a, b)`` closes into a loop through the tree as ``a -> lca(a, b) -> b``, and its
length is ``depth(a) + depth(b) - 2 * depth(lca) + 1`` -- so ``generator_loop_lengths`` finds the
apex, the wrapper scans the lengths into offsets, and ``write_generator_loops`` fills each loop's
slice from both ends at once. ``distances`` is the depth array the level loop already wrote, which
is why dropping the *order* from the traversal and keeping the *distances* was the right half to
cut: the tracer needs the depths to size a loop without walking it.
"""

import warp as wp

from triwarp.constants import INT32_MAX
from triwarp.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from triwarp.kernels.reduce import ITEMS_PER_BLOCK_1D, tile_chunk

# One past the largest edge index any proposal can hold, so ``wp.atomic_min`` starts empty. The
# candidate count is bounded by the unique-edge count, which is well inside int32.
FOREST_NO_PROPOSAL = wp.constant(wp.int32(INT32_MAX))

# ``parents`` is seeded with this and narrowed by ``wp.atomic_min``, so "no parent yet" has to be
# larger than every vertex index rather than the ``-1`` a sequential traversal would write. It
# never collides with a real index, which is what lets ``dual_candidate_mask`` test parenthood
# without a special case for the root.
NO_PARENT = wp.constant(wp.int32(INT32_MAX))

# Slots of the primal level loop's state word. ``BFS_CHANGED`` is set by any thread that claimed a
# vertex this level and cleared by the next ``bfs_advance``, which copies it into ``BFS_CONDITION``
# -- the word ``wp.capture_while`` polls. Two slots rather than one because the condition has to
# survive the clear.
BFS_LEVEL = 0
BFS_CHANGED = 1
BFS_CONDITION = 2
BFS_STATE_SIZE = 3

# Slots of the Boruvka round's state word, same convention.
FOREST_MERGES = 0
FOREST_ROUNDS = 1
FOREST_CONDITION = 2
FOREST_STATE_SIZE = 3

# Slots of the one buffer that carries every host-visible count out of the decomposition, read back
# once. Slot 0 is filled over the edges and slots 1-2 over the vertices, by two different kernels,
# which is why they share a buffer rather than each owning one: the readback is the cost, not the
# four bytes.
COUNT_INTERIOR_EDGES = 0
COUNT_REACHED = 1
COUNT_REFERENCED = 2
COUNT_SIZE = 3


@wp.kernel
def vertex_degrees_and_interior_count(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    out_degree: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Two unrelated answers off one pass over the unique edges, because the pass is the cost: the
    # per-vertex degree that sizes the adjacency CSR, and the number of interior edges that the
    # closed-surface guard compares against ``3 * n_faces``.
    #
    # The interior count is a block fold rather than a conditional ``wp.atomic_add`` per edge
    # because on a closed mesh *every* edge is interior, so the conditional atomic is the
    # unconditional one and it serializes the whole launch on one address.
    chunk, lane = wp.tid()
    offset, remaining = tile_chunk(unique_edges.shape[0], chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    # ``tile_chunk`` reports what is left to the end of the array, not this block's share of it.
    n_rows = wp.min(remaining, ITEMS_PER_BLOCK_1D)

    interior = wp.int32(0)
    for k in range(lane, n_rows, wp.block_dim()):
        e = offset + k
        wp.atomic_add(out_degree, unique_edges[e, 0], 1)
        wp.atomic_add(out_degree, unique_edges[e, 1], 1)
        if edge_face_count[e] == 2:
            interior = interior + 1

    # Block-collective, so it runs outside the ``lane == 0`` guard.
    interior_total = wp.tile_sum(wp.tile(interior))[0]
    if lane == 0:
        wp.atomic_add(out_counts, COUNT_INTERIOR_EDGES, interior_total)


@wp.kernel
def scatter_adjacency(
    unique_edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
) -> None:
    # Both directed entries of each undirected edge, into the CSR row the counting pass sized.
    # ``cursor`` is zeroed per-vertex scratch, not an output: each vertex's rows are handed out by
    # ``wp.atomic_add`` on it, so the column order within a row is thread order and **not sorted**.
    # That is deliberate and is what this build costs less than ``bsr_from_triplets`` for -- the
    # level loop's ``wp.atomic_min`` tie-break does not read the column order, so nothing downstream
    # needs it sorted. A future caller that does need sorted rows must sort them, not assume them.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    out_columns[offsets[a] + wp.atomic_add(cursor, a, 1)] = b
    out_columns[offsets[b] + wp.atomic_add(cursor, b, 1)] = a


@wp.kernel
def bfs_seed(unique_edges: wp.array2d[wp.int32], out_distances: wp.array[wp.int32]) -> None:
    # Root the primal tree at the lowest-indexed *referenced* vertex, read on the device rather
    # than handed in, which is one host readback the call no longer takes.
    #
    # Rooting at vertex 0 unconditionally is the bug this replaces: on a mesh whose vertex 0 carries
    # no edges the "tree" is a single isolated node, every primal edge becomes a generator, and the
    # count comes back enormous rather than wrong-looking. ``edges_unique`` returns its rows
    # lexicographically sorted, so the first endpoint of the first row is the lowest-indexed
    # referenced vertex -- referenced by construction, and deterministic.
    out_distances[unique_edges[0, 0]] = 0


@wp.kernel
def bfs_push_level(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    state: wp.array[wp.int32],
    out_parents: wp.array[wp.int32],
    out_distances: wp.array[wp.int32],
) -> None:
    # One breadth-first level, pushed from the frontier. Launched at ``dim=n_vertices`` and
    # early-exiting off ``out_distances`` rather than reading a compacted frontier: the compaction
    # is a tiled scan per level and the early exit is one coalesced load, and the launch is the same
    # width either way.
    #
    # The visit test admits ``seen == level`` as well as an unvisited vertex, and that arm is
    # load-bearing rather than defensive. Two frontier vertices may reach the same neighbour in one
    # level; whichever gets there first stores ``level``, and without this arm the other would read
    # a visited vertex and skip the ``wp.atomic_min`` -- leaving ``parents`` dependent on which
    # thread ran first. Both stores write the same value, so the race on ``out_distances`` itself is
    # idempotent.
    v = wp.int32(wp.tid())
    level = state[BFS_LEVEL]
    if out_distances[v] != level - 1:
        return
    for k in range(offsets[v], offsets[v + 1]):
        w = columns[k]
        seen = out_distances[w]
        if seen < 0 or seen == level:
            out_distances[w] = level
            wp.atomic_min(out_parents, w, v)
            state[BFS_CHANGED] = 1


@wp.kernel
def bfs_advance(out_state: wp.array[wp.int32]) -> None:
    # Publish this level's claim flag as the loop condition, clear it, and step the level. One
    # thread, and the only reason the level loop is two launches rather than one: the flag has to be
    # cleared between levels and a thread of the pushing kernel cannot do it without racing the
    # threads still setting it.
    out_state[BFS_CONDITION] = out_state[BFS_CHANGED]
    out_state[BFS_CHANGED] = 0
    out_state[BFS_LEVEL] = out_state[BFS_LEVEL] + 1


@wp.kernel
def count_reached_and_referenced(
    offsets: wp.array[wp.int32], distances: wp.array[wp.int32], out_counts: wp.array[wp.int32]
) -> None:
    # The connectivity guard's two numbers, folded per block into the same buffer the interior-edge
    # count went to. A vertex is referenced exactly when its adjacency row is non-empty and reached
    # exactly when the level loop gave it a depth, so the two agree if and only if the referenced
    # vertices form one component. Unreferenced vertices are deliberately not counted: they carry no
    # edges, so they cannot change the generator count.
    chunk, lane = wp.tid()
    offset, remaining = tile_chunk(distances.shape[0], chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    n_rows = wp.min(remaining, ITEMS_PER_BLOCK_1D)

    reached = wp.int32(0)
    referenced = wp.int32(0)
    for k in range(lane, n_rows, wp.block_dim()):
        v = offset + k
        if distances[v] >= 0:
            reached = reached + 1
        if offsets[v + 1] > offsets[v]:
            referenced = referenced + 1

    # Block-collective, so both run outside the ``lane == 0`` guard.
    reached_total = wp.tile_sum(wp.tile(reached))[0]
    referenced_total = wp.tile_sum(wp.tile(referenced))[0]
    if lane == 0:
        wp.atomic_add(out_counts, COUNT_REACHED, reached_total)
        wp.atomic_add(out_counts, COUNT_REFERENCED, referenced_total)


@wp.kernel
def dual_candidate_mask(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    out_candidate: wp.array[wp.bool],
) -> None:
    # Whether the cotree may cross this dual edge: an interior edge (exactly two incident faces)
    # that the primal tree did not already claim. Non-manifold edges count higher than 2 and are
    # excluded, which is what the exactly-2 row grouping behind ``adjacency.face_adjacency`` does
    # with them too -- and it is also what keeps the Boruvka round away from ``edge_faces``' second
    # column on a boundary edge, which ``scatter_edge_incidence`` leaves unwritten.
    #
    # The primal-tree test and the candidate test were two kernels and an intermediate mask. An
    # undirected edge belongs to a rooted spanning tree exactly when one endpoint is the other's
    # parent, and ``NO_PARENT`` matches neither endpoint, so the root needs no special case.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    in_primal_tree = parents[b] == a or parents[a] == b
    out_candidate[e] = edge_face_count[e] == 2 and not in_primal_tree


@wp.kernel
def forest_round_setup(
    labels: wp.array[wp.int32],
    out_roots: wp.array[wp.int32],
    out_proposal: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Open a Boruvka round: freeze this round's component of every node, empty the proposal slots
    # and reset the merge counter. ``find_representative`` path-compresses ``labels`` as it goes, so
    # the snapshot doubles as the round's flatten pass.
    #
    # The proposal fill and the counter reset used to be two host-side ``fill_`` / ``zero_`` calls
    # per round. They are stores here because this kernel already runs at ``dim=n_faces`` and
    # touches ``out_proposal``'s every element anyway.
    f = wp.int32(wp.tid())
    out_roots[f] = find_representative(labels, f)
    out_proposal[f] = FOREST_NO_PROPOSAL
    if f == 0:
        out_state[FOREST_MERGES] = 0
        out_state[FOREST_ROUNDS] = out_state[FOREST_ROUNDS] - 1


@wp.func
def boruvka_cross_edge(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    e: wp.int32,
) -> tuple[wp.int32, wp.int32]:
    # The two components edge ``e`` joins this Boruvka round, or ``(-1, -1)`` when it joins none --
    # it is not a candidate, or both its faces already sit in one component. The sentinel form
    # rather than an early return, because a ``@wp.func`` cannot return for its caller; this is the
    # convention ``remesh._resolve_flip_quad_guarded`` already uses in this tree.
    #
    # Reading ``roots`` rather than ``labels`` is the whole point and is why the two round kernels
    # must open identically: ``roots`` is the round's frozen snapshot, so both the proposal and the
    # acceptance decide against the same partition and the round's outcome does not depend on
    # thread order. See this module's docstring.
    if not candidate[e]:
        return wp.int32(-1), wp.int32(-1)
    root_a = roots[edge_faces[e, 0]]
    root_b = roots[edge_faces[e, 1]]
    if root_a == root_b:
        return wp.int32(-1), wp.int32(-1)
    return root_a, root_b


@wp.kernel
def forest_propose(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    out_proposal: wp.array[wp.int32],
) -> None:
    # Offer each candidate edge to the components on both sides; the lowest edge index wins.
    e = wp.int32(wp.tid())
    root_a, root_b = boruvka_cross_edge(candidate, edge_faces, roots, e)
    if root_a < 0:
        return
    wp.atomic_min(out_proposal, root_a, e)
    wp.atomic_min(out_proposal, root_b, e)


@wp.kernel
def forest_link(
    candidate: wp.array[wp.bool],
    edge_faces: wp.array2d[wp.int32],
    roots: wp.array[wp.int32],
    proposal: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_in_forest: wp.array[wp.bool],
    out_state: wp.array[wp.int32],
) -> None:
    # Accept the edge either side chose and union the two components. Both endpoints may name the
    # same edge, and one edge may be the choice of two components at once; either way the union runs
    # once per accepted edge and the second attempt finds the roots already equal.
    #
    # The union is ``ecl_hook_edge`` verbatim rather than a second copy of its CAS retry: it takes a
    # representative and a node, so handing it the two roots this round snapshotted costs one
    # already-compressed find. Its termination argument (parents monotonically non-increasing, so
    # ``max(root_a, root_b)`` strictly decreases) therefore applies unchanged.
    e = wp.int32(wp.tid())
    root_a, root_b = boruvka_cross_edge(candidate, edge_faces, roots, e)
    if root_a < 0:
        return
    if proposal[root_a] != e and proposal[root_b] != e:
        return
    out_in_forest[e] = True
    wp.atomic_add(out_state, FOREST_MERGES, 1)
    ecl_hook_edge(labels, root_a, root_b)


@wp.kernel
def forest_advance(out_state: wp.array[wp.int32]) -> None:
    # Publish the round's merge count as the loop condition, gated by the round cap. A round that
    # merged nothing means every component is spanned, which is the real exit; the cap only matters
    # if the halving argument is ever violated, where it turns a hung device into a wrong answer the
    # partition identity in ``tests/test_homology.py`` catches.
    keep_going = wp.where(out_state[FOREST_ROUNDS] > 0, out_state[FOREST_MERGES], wp.int32(0))
    out_state[FOREST_CONDITION] = keep_going


@wp.func
def tree_apex(
    parents: wp.array[wp.int32], distances: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.int32:
    # Lowest common ancestor of two vertices in the rooted primal tree, by lifting the deeper one to
    # the other's depth and then climbing in step. Both walks are ``O(depth)`` dependent loads on
    # one thread, which is affordable here only because there are ``2 * g`` of them and not ``V``.
    x = a
    y = b
    while distances[x] > distances[y]:
        x = parents[x]
    while distances[y] > distances[x]:
        y = parents[y]
    while x != y:
        x = parents[x]
        y = parents[y]
    return x


@wp.kernel
def generator_loop_lengths(
    generator_edge_ids: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
    distances: wp.array[wp.int32],
    out_apex: wp.array[wp.int32],
    out_lengths: wp.array[wp.int32],
) -> None:
    # Size each generator's loop without writing it, so the wrapper can scan the lengths into
    # offsets and allocate once. The length is pure depth arithmetic given the apex -- the two legs
    # are ``depth(a) - depth(apex)`` and ``depth(b) - depth(apex)`` edges long and the apex is
    # counted once -- which is the whole reason the level loop keeps ``distances`` after dropping
    # its discovery order.
    g = wp.int32(wp.tid())
    e = generator_edge_ids[g]
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    apex = tree_apex(parents, distances, a, b)
    out_apex[g] = apex
    out_lengths[g] = distances[a] + distances[b] - 2 * distances[apex] + 1


@wp.kernel
def write_generator_loops(
    generator_edge_ids: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    parents: wp.array[wp.int32],
    apex: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_loops: wp.array[wp.int32],
) -> None:
    # Fill one generator's slice from both ends: the ``a`` leg forward from the slice's start up to
    # and including the apex, the ``b`` leg backward from its last slot down to the apex's child.
    # The two cursors meet exactly because ``generator_loop_lengths`` sized the slice from the same
    # two depths, so neither bound needs re-deriving and the apex is written once.
    #
    # The resulting order is the closed walk ``a -> apex -> b``, with the generator edge ``(b, a)``
    # closing it implicitly -- the start vertex is not repeated, matching ``boundary_loops``.
    g = wp.int32(wp.tid())
    e = generator_edge_ids[g]
    top = apex[g]

    slot = offsets[g]
    node = unique_edges[e, 0]
    while node != top:
        out_loops[slot] = node
        slot = slot + 1
        node = parents[node]
    out_loops[slot] = top

    back = offsets[g + 1] - 1
    node = unique_edges[e, 1]
    while node != top:
        out_loops[back] = node
        back = back - 1
        node = parents[node]
