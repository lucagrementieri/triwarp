"""
Kernels for [`triwarp.homology`][triwarp.homology]: the tree-cotree decomposition.

Two pieces. ``primal_tree_edge_mask`` flags the undirected edges a rooted spanning tree uses, from
the tree's parent array alone. The rest is a **Boruvka spanning forest** over the dual graph, which
replaces the breadth-first traversal that used to build the cotree.

The reason a forest and not a traversal: nothing reads the dual tree's *shape*. ``tree_cotree`` uses
it only as a set -- the generators are the edges in neither tree -- and ``_loop_through_tree`` walks
the **primal** parents. So the dual side needs no root and no parent pointers, and a breadth-first
tree is the one shape that is expensive to get here: the dual graph restricted to non-primal-tree
edges is *already nearly a tree*, so its diameter is enormous. Measured on the benchmark meshes, the
dual traversal ran **765-891 levels** against the primal's 116-192, and ``graph.bfs`` costs
``4 kernels x levels`` independently of the node count -- 32-42 ms of a 51-61 ms call.

Boruvka is the standard parallel answer and needs `O(log F)` rounds instead. Each round gives every
component its minimum-index incident candidate edge, accepts those edges into the forest and unions
the components; the union-find core is
[`find_representative`][triwarp.kernels.algorithms.connected_components.find_representative],
imported rather than re-derived, so the forest and ``graph.connected_component_labels`` share one
implementation of the pointer-jumping find.

**Why the accepted set is a forest.** Edge *indices* are distinct, so "minimum incident index" is a
strict total order on the candidate edges, and the classical Boruvka argument applies: a cycle among
the accepted edges would need a strictly decreasing cycle of minima. The round reads its components
from an immutable ``roots`` snapshot taken before any union, which is what makes that argument exact
here -- deciding against a ``label`` array that other threads are mutating would let one component
accept two edges in a round and could close a cycle.
"""

import warp as wp

from triwarp.kernels.algorithms.connected_components import ecl_hook_edge, find_representative

# One past the largest edge index any proposal can hold, so ``wp.atomic_min`` starts empty. The
# candidate count is bounded by the unique-edge count, which is well inside int32.
FOREST_NO_PROPOSAL = wp.constant(wp.int32(0x7FFFFFFF))


@wp.kernel
def primal_tree_edge_mask(
    unique_edges: wp.array2d[wp.int32], parents: wp.array[wp.int32], out_in_tree: wp.array[wp.bool]
) -> None:
    # An undirected edge belongs to a rooted spanning tree exactly when one endpoint is the other's
    # parent. ``parents`` holds ``-1`` at a root, which matches neither endpoint, so the root needs
    # no special case.
    e = wp.int32(wp.tid())
    a = unique_edges[e, 0]
    b = unique_edges[e, 1]
    out_in_tree[e] = parents[b] == a or parents[a] == b


@wp.func
def is_dual_candidate(edge_face_count: wp.int32, in_primal_tree: wp.bool) -> wp.bool:
    """Whether the cotree may cross this dual edge."""
    # An interior edge (exactly two incident faces) that the primal tree did not already claim.
    # Non-manifold edges count higher than 2 and are excluded, which is what the exactly-2 row
    # grouping behind ``adjacency.face_adjacency`` did with them.
    return edge_face_count == 2 and not in_primal_tree


@wp.kernel
def forest_snapshot_roots(labels: wp.array[wp.int32], out_roots: wp.array[wp.int32]) -> None:
    # Freeze this round's component of every node. ``find_representative`` path-compresses
    # ``labels`` as it goes, so this doubles as the round's flatten pass.
    f = wp.int32(wp.tid())
    out_roots[f] = find_representative(labels, f)


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
    out_merges: wp.array[wp.int32],
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
    wp.atomic_add(out_merges, 0, 1)
    ecl_hook_edge(labels, root_a, root_b)


@wp.func
def is_leftover_edge(candidate: wp.bool, in_forest: wp.bool) -> wp.bool:
    """Whether this edge is a homology generator: a candidate the cotree left out."""
    # Edges the primal tree took are already excluded from ``candidate``, so this is "in neither
    # tree" on a closed surface.
    return candidate and not in_forest
