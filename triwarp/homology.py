"""
Homology generators: the loops that make a surface's topology, found by tree-cotree.

On a closed genus-``g`` surface there are ``2 * g`` independent loops that cannot be contracted to a
point — two per handle, going *around* it and *through* it. They are what a cutting tool needs (cut
along them and the surface opens into a disk), and what tells a parametrizer where its seams have to
go.

The construction is Erickson & Whittlesey's tree-cotree decomposition, and it is a counting
argument as much as an algorithm: take a spanning tree of the vertex graph, then a spanning tree of
the *dual* graph that avoids the primal tree's edges, and every remaining edge closes exactly one
independent loop. Euler's formula fixes how many are left: ``E - (V - 1) - (F - 1) = 2 - chi``,
which is ``2 * g``.

Unlike the rest of this package there is no reference implementation to compare against —
potpourri3d does not expose geometry-central's homology code — so the tests check structural
invariants instead: the loop count against
[`euler_characteristic`][triwarp.measures.euler_characteristic], and that every loop is a closed
walk along real mesh edges that visits no vertex twice.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.kernels import array as kernel_array
from triwarp.kernels import homology as kernel_homology
from triwarp.kernels import scatter as kernel_scatter


def homology_generators(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, copy: bool = False
) -> list[wp.array[wp.int32]]:
    """
    Non-contractible loops of the surface, as ordered vertex-index cycles.

    Returns ``2 * g`` loops for a closed genus-``g`` surface and none for a sphere. Each loop is a
    closed walk along mesh edges: consecutive entries share an edge, and so do the last and first
    (the start vertex is not repeated, matching
    [`boundary_loops`][triwarp.boundary.boundary_loops]).

    The loops are a *basis*, not canonical: any generating set is as valid as any other, and this
    one falls out of the spanning trees the construction happens to build. They are also as long
    and as jagged as those trees, which is what
    [`shorten_loop`][triwarp.geodesic_walk.shorten_loop] is for.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count is used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be a closed, connected,
        edge-manifold surface: the counting argument this rests on assumes it.
    copy
        Return independent buffers instead of views into one packed allocation.

    Returns
    -------
    list[wp.array[wp.int32]]
        One ``wp.int32`` array of vertex indices per generator, on ``faces.device``. Empty for a
        sphere.

    Raises
    ------
    ValueError
        If the mesh has a boundary, since a surface with boundary has a different homology basis
        (every boundary loop contributes one, and the tree-cotree count no longer applies), or if
        its referenced vertices are not all in one connected component, since the count is then a
        sum over components that a single pair of spanning trees does not produce.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`shorten_loop`][triwarp.geodesic_walk.shorten_loop]
        Shortens these loops within their homotopy class, keeping them on mesh edges.
    [`tree_cotree`][triwarp.homology.tree_cotree]
    [`euler_characteristic`][triwarp.measures.euler_characteristic]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return []

    unique_edges, generator_edges, parents = tree_cotree(vertices, faces)
    del unique_edges
    generators = generator_edges.numpy()
    if len(generators) == 0:
        return []

    parents_np = parents.numpy()
    loops = [_loop_through_tree(int(a), int(b), parents_np) for a, b in generators]
    packed = wp.array(np.concatenate(loops), dtype=wp.int32, device=device)
    starts_np = np.cumsum([0, *(len(loop) for loop in loops[:-1])], dtype=np.int32)
    offsets = wp.array(starts_np, dtype=wp.int32, device=device)
    return tw.array.split(packed, offsets, copy=copy)


def _loop_through_tree(start: int, end: int, parents: np.ndarray) -> np.ndarray:
    """
    Close an edge into a loop through the spanning tree: ``start -> root``, ``root -> end``, edge.

    The two root paths share a suffix above their lowest common ancestor, and that shared part is
    dropped: otherwise the "loop" would walk up it and straight back down, a contractible spur that
    says nothing about the surface's topology.
    """
    path_start = [start]
    while parents[path_start[-1]] >= 0:
        path_start.append(int(parents[path_start[-1]]))
    path_end = [end]
    while parents[path_end[-1]] >= 0:
        path_end.append(int(parents[path_end[-1]]))

    shared = 0
    while (
        shared + 1 <= len(path_start)
        and shared + 1 <= len(path_end)
        and path_start[-1 - shared] == path_end[-1 - shared]
    ):
        shared += 1
    # Keep the lowest common ancestor once: it is a real corner of the loop.
    trimmed_start = path_start[: len(path_start) - shared + 1]
    trimmed_end = path_end[: len(path_end) - shared]
    return np.array(trimmed_start + trimmed_end[::-1], dtype=np.int32)


def tree_cotree(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[twt.Array2dInt32, twt.Array2dInt32, wp.array[wp.int32]]:
    """
    Tree-cotree decomposition: the edges a primal and a dual spanning tree both leave alone.

    Three pieces of structure, and the building block behind
    [`homology_generators`][triwarp.homology.homology_generators]:

    1. a breadth-first spanning tree of the **vertex** graph, whose ``parents`` the loop tracing
       walks,
    2. a spanning **forest** of the **dual** (face-adjacency) graph, restricted to dual edges whose
       primal edge is not already in the vertex tree,
    3. whatever edges belong to neither — exactly ``2 * g`` of them on a closed genus-``g`` surface.

    The dual side is a forest rather than a traversal because only its edge *set* is read, never its
    shape. That matters for cost as well as tidiness: the dual graph restricted to non-primal-tree
    edges is already nearly a tree, so a breadth-first traversal of it would run its diameter, which
    can be very large, and ``graph.bfs`` costs one pass of kernels per level whatever the node
    count. A Boruvka forest instead needs only ``O(log n_faces)`` rounds.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count is used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    unique_edges : twt.Array2dInt32
        ``(n_edges, 2)`` undirected edges, as [`edges_unique`][triwarp.edges.edges_unique] returns.
    generator_edges : twt.Array2dInt32
        ``(2 * g, 2)`` the leftover edges, one per homology generator.
    parents : wp.array[wp.int32]
        Length ``n_vertices`` primal spanning-tree parent of each vertex; ``-1`` at the root and
        at every unreferenced vertex, which the tree never reaches. All ``-1`` for a mesh with no
        edges at all, which has nothing to span and no generators.

    Raises
    ------
    ValueError
        If the mesh has a boundary, or is not connected (see
        [`homology_generators`][triwarp.homology.homology_generators]).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`homology_generators`][triwarp.homology.homology_generators]
    [`bfs`][triwarp.graph.bfs]
    [`edges_unique`][triwarp.edges.edges_unique]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    # One grouping of the edge rows answers everything: ``inverse`` maps each face corner to its
    # unique edge, and one scatter over it fills both the per-edge face count and the two incident
    # faces. That is the whole dual graph, indexed by unique edge -- which is why nothing here needs
    # to locate a shared edge's row afterwards. The host ``argsort`` + ``searchsorted`` pair this
    # replaces was doing exactly that lookup, over an ``inverse`` the same ``edges_unique`` call had
    # already returned and thrown away.
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    n_edges = int(unique_edges.shape[0])
    edge_face_count = wp.zeros(n_edges, dtype=wp.int32, device=device)
    edge_faces = twt.empty_2d((n_edges, 2), wp.int32, device=device)
    if n_edges > 0:
        wp.launch(
            kernel_scatter.scatter_edge_incidence,
            dim=int(inverse.shape[0]),
            inputs=[inverse, edge_face_count, edge_faces],
            device=device,
        )
    interior = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.map(kernel_array.equal, edge_face_count, wp.int32(2), out=interior)
    # The one readback: the closed-surface guard, whose message quotes the boundary-edge count.
    # ``face_count == 2`` is the same exactly-two-corners test the row grouping behind
    # ``face_adjacency`` applied, so a non-manifold edge fails this guard exactly as it did before.
    # ``reduce.sum`` raises on a zero-length array, so an edgeless mesh answers without it rather
    # than surfacing that as the boundary ``ValueError`` this function documents.
    n_interior = tw.reduce.sum(interior) if n_edges > 0 else 0
    if n_interior * 2 != 3 * n_faces:
        raise ValueError(
            "homology_generators requires a closed surface: this mesh has "
            f"{3 * n_faces - 2 * n_interior} boundary edge(s)."
        )
    if n_edges == 0:
        # No edges means no faces (a face carries three), so there is nothing to span and nothing
        # to leave over. Every vertex is its own primal-tree root.
        return (
            twt.as_array2d(unique_edges, wp.int32),
            twt.empty_2d((0, 2), wp.int32, device=device),
            wp.full(n_vertices, -1, dtype=wp.int32, device=device),
        )

    # Primal spanning tree over the vertex graph. This one stays a breadth-first traversal: the mesh
    # graph's diameter is small, and ``parents`` is the rooted tree the loop tracing walks.
    #
    # **The root has to be a referenced vertex, and the graph has to be connected**, because the
    # generator count is ``n_edges`` minus the two trees' edge counts and a tree that spans less
    # than its graph hands the difference over as generators. Rooting at vertex 0 unconditionally
    # is what breaks on an unreferenced one: the "tree" is then a single isolated node with no
    # edges, and every primal edge becomes a generator (measured: a 128-vertex genus-1 torus with
    # one unreferenced vertex prepended reported 129 generators instead of 2). ``edges_unique``
    # returns its rows lexicographically sorted, so the first endpoint of the first row is the
    # lowest-indexed referenced vertex -- referenced by construction, and deterministic.
    adjacency = tw.graph.edges_to_csr(n_vertices, unique_edges)
    root = int(read_scalar(unique_edges.flatten(), 0))
    order, parents, _ = tw.graph.bfs(adjacency, root)
    # The connectivity half of the same requirement, which ``homology_generators``' docstring has
    # always stated as a precondition and nothing checked. A vertex is referenced exactly when its
    # adjacency row is non-empty, and ``order`` holds what the traversal reached, so the two counts
    # agree exactly when the referenced vertices form one component. Unreferenced vertices are
    # deliberately not counted: they carry no edges, so they cannot change the generator count.
    referenced = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(kernel_array.less, adjacency.offsets[:-1], adjacency.offsets[1:], out=referenced)
    n_referenced = int(tw.reduce.sum(referenced))
    if int(order.shape[0]) != n_referenced:
        raise ValueError(
            "homology_generators requires a connected surface: the traversal reached "
            f"{int(order.shape[0])} of {n_referenced} referenced vertices."
        )

    # The edgeless case returned above, so these three no longer need a ``n_edges > 0`` guard.
    in_primal_tree = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.launch(
        kernel_homology.primal_tree_edge_mask,
        dim=n_edges,
        inputs=[unique_edges, parents, in_primal_tree],
        device=device,
    )
    candidate = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.map(kernel_homology.is_dual_candidate, edge_face_count, in_primal_tree, out=candidate)

    in_dual_tree = _dual_spanning_forest(candidate, edge_faces, n_faces)

    # A homology generator is a candidate the cotree left out. Edges the primal tree took are
    # already excluded from ``candidate``, so this is "in neither tree" on a closed surface --
    # which is ``array.mask_and_not`` and needs no predicate of its own.
    leftover_mask = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_and_not, candidate, in_dual_tree, out=leftover_mask)
    generator_edges = tw.array.gather(unique_edges, tw.array.flatnonzero(leftover_mask))
    return (
        twt.as_array2d(unique_edges, wp.int32),
        twt.as_array2d(generator_edges, wp.int32),
        parents,
    )


def _dual_spanning_forest(
    candidate: wp.array[wp.bool], edge_faces: twt.Array2dInt32, n_faces: int
) -> wp.array[wp.bool]:
    """
    Build a spanning forest of the dual graph over the ``candidate`` edges, as a per-edge mask.

    Boruvka: each round hands every component its lowest-indexed incident candidate edge, accepts
    those edges and unions the components, so the component count at least halves per round and the
    loop finishes in ``O(log n_faces)`` of them.

    A **forest** is all [`tree_cotree`][triwarp.homology.tree_cotree] needs — it reads the dual tree
    as a set, and the loop tracing walks the *primal* parents — so the dual side has no root and no
    parent pointers, and its shape is free to choose. The breadth-first shape is the expensive one
    here: the dual graph restricted to non-primal-tree edges is already nearly a tree, so its
    diameter is enormous, and ``graph.bfs`` costs one pass of kernels per level whatever the node
    count.
    """
    device = candidate.device
    n_candidates = int(candidate.shape[0])
    in_forest = wp.zeros(n_candidates, dtype=wp.bool, device=device)
    if n_candidates == 0 or n_faces == 0:
        return in_forest
    labels = tw.array.arange(n_faces, device=device)
    roots = wp.empty(n_faces, dtype=wp.int32, device=device)
    proposal = wp.empty(n_faces, dtype=wp.int32, device=device)
    merges = wp.zeros(1, dtype=wp.int32, device=device)
    # The round count is a cap, not a schedule: the readback below exits as soon as a round merges
    # nothing, which happens once every component is spanned. ``bit_length`` is ``ceil(log2)`` plus
    # one, so the cap can only be reached by a logic error.
    for _ in range(max(1, n_faces.bit_length()) + 1):
        merges.zero_()
        wp.launch(
            kernel_homology.forest_snapshot_roots,
            dim=n_faces,
            inputs=[labels, roots],
            device=device,
        )
        proposal.fill_(int(kernel_homology.FOREST_NO_PROPOSAL))
        wp.launch(
            kernel_homology.forest_propose,
            dim=n_candidates,
            inputs=[candidate, edge_faces, roots, proposal],
            device=device,
        )
        wp.launch(
            kernel_homology.forest_link,
            dim=n_candidates,
            inputs=[candidate, edge_faces, roots, proposal, labels, in_forest, merges],
            device=device,
        )
        # One 4-byte readback per round, and there is no bound on the rounds without it.
        if int(read_scalar(merges, 0)) == 0:
            break
    return in_forest
