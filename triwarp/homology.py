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
[`euler_characteristic`][triwarp.totals.euler_characteristic], and that every loop is a closed
walk along real mesh edges that visits no vertex twice.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt


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
    one falls out of the spanning trees the traversals happen to build. They are also not geodesic;
    shortening them is a separate problem (geometry-central's edge-flip machinery, not ported here).

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
        (every boundary loop contributes one, and the tree-cotree count no longer applies).

    See Also
    --------
    [`tree_cotree`][triwarp.homology.tree_cotree]
    [`euler_characteristic`][triwarp.totals.euler_characteristic]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
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


def tree_cotree(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[twt.Array2dInt32, twt.Array2dInt32, wp.array[wp.int32]]:
    """
    Tree-cotree decomposition: the edges a primal and a dual spanning tree both leave alone.

    Three traversals' worth of structure, and the building block behind
    [`homology_generators`][triwarp.homology.homology_generators]:

    1. a breadth-first spanning tree of the **vertex** graph,
    2. a breadth-first spanning tree of the **dual** (face-adjacency) graph, restricted to dual
       edges whose primal edge is not already in the vertex tree,
    3. whatever edges belong to neither — exactly ``2 * g`` of them on a closed genus-``g`` surface.

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
        Length ``n_vertices`` primal spanning-tree parent of each vertex; ``-1`` at the root.

    Raises
    ------
    ValueError
        If the mesh has a boundary (see
        [`homology_generators`][triwarp.homology.homology_generators]).

    See Also
    --------
    [`homology_generators`][triwarp.homology.homology_generators]
    [`bfs`][triwarp.graph.bfs]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    """
    device = faces.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_edges = int(unique_edges.shape[0])
    face_pairs, shared_edges = tw.adjacency.face_adjacency(
        faces, return_edges=True, n_vertices=n_vertices
    )
    if int(face_pairs.shape[0]) * 2 != 3 * n_faces:
        raise ValueError(
            "homology_generators requires a closed surface: this mesh has "
            f"{3 * n_faces - 2 * int(face_pairs.shape[0])} boundary edge(s)."
        )

    # Primal spanning tree over the vertex graph.
    adjacency = tw.graph.edges_to_csr(n_vertices, unique_edges)
    _, parents, _ = tw.graph.bfs(adjacency, 0)

    # Which undirected edges the primal tree uses. Done on the host: the filter it feeds is a
    # gather-and-compare over one int32 per edge, and the dual traversal needs it as a mask anyway.
    edges_np = unique_edges.numpy()
    parents_np = parents.numpy()
    in_primal_tree = (parents_np[edges_np[:, 1]] == edges_np[:, 0]) | (
        parents_np[edges_np[:, 0]] == edges_np[:, 1]
    )

    # Dual spanning tree over face adjacency, crossing only edges the primal tree left alone.
    # Locating each shared edge's row in ``unique_edges`` is a sorted-key lookup, vectorised: the
    # dict-of-tuples this replaces ran two Python loops with two ``int()`` calls per row, ~600k
    # dict operations on a 100k-vertex genus-2 surface. Both row sets are already on the host, so
    # the key is built here rather than through ``grouping.hash_indices_rows``, which would add two
    # launches and two readbacks to reach the same integers.
    shared_np = shared_edges.numpy()
    edge_keys = edges_np[:, 0].astype(np.int64) * n_vertices + edges_np[:, 1]
    shared_keys = shared_np[:, 0].astype(np.int64) * n_vertices + shared_np[:, 1]
    order = np.argsort(edge_keys)
    dual_edge_index = order[np.searchsorted(edge_keys[order], shared_keys)]
    crossable = ~in_primal_tree[dual_edge_index]
    dual_pairs = face_pairs.numpy()[crossable]
    dual_index = dual_edge_index[crossable]

    in_dual_tree = np.zeros(n_edges, dtype=bool)
    if len(dual_pairs) > 0:
        dual_adjacency = tw.graph.edges_to_csr(
            n_faces,
            twt.as_array2d(
                wp.array(np.ascontiguousarray(dual_pairs), dtype=wp.int32, device=device), wp.int32
            ),
        )
        _, dual_parents, _ = tw.graph.bfs(dual_adjacency, 0)
        dual_parents_np = dual_parents.numpy()
        used = (dual_parents_np[dual_pairs[:, 1]] == dual_pairs[:, 0]) | (
            dual_parents_np[dual_pairs[:, 0]] == dual_pairs[:, 1]
        )
        in_dual_tree[dual_index[used]] = True

    leftover = np.flatnonzero(~in_primal_tree & ~in_dual_tree)
    generator_edges = wp.array(
        np.ascontiguousarray(edges_np[leftover].reshape(-1, 2)), dtype=wp.int32, device=device
    )
    return (
        twt.as_array2d(unique_edges, wp.int32),
        twt.as_array2d(generator_edges, wp.int32),
        parents,
    )


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
