"""
Mesh boundary edges and vertices (Warp).

A mesh edge lies on the boundary when it appears exactly once among all triangle edges.
Boundary detection reuses :func:`triwarp.grouping.group_int_rows` (the analog of
``trimesh.grouping.group_rows(require_count=1)``), which hashes each sorted edge row and
returns the original row indices of edges occurring exactly once.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import boundary as kernel_boundary


def boundary_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> twt.Array2dInt32:
    """
    Undirected boundary edges (each row min-first), without preserving orientation.

    An edge belongs to the boundary when it appears exactly once among the sorted edges of
    all triangles.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first), as from
        :func:`triwarp.edges.faces_to_edges` with ``sorted=True``. Built from ``faces`` when
        ``None``.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_boundary, 2)`` sorted boundary edges on ``faces.device``. Empty ``(0, 2)``
        when the mesh has no boundary.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_int32_2d((0, 2), device=faces.device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)

    indices = tw.grouping.group_int_rows(edges_sorted, 1, int(vertices.shape[0])).flatten()
    return twt.as_array2d_int32(tw.array.gather(edges_sorted, indices))


def oriented_boundary_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> twt.Array2dInt32:
    """
    Directed boundary edges, preserving the orientation from the face winding.

    Boundary edges are detected on the sorted edges (appearing exactly once), but the
    directed ``(i, j)`` pairs are returned.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first). Built
        from ``faces`` when ``None``.
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges, as from
        :func:`triwarp.edges.faces_to_edges`. Built from ``faces`` when ``None``.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_boundary, 2)`` directed boundary edges on ``faces.device``. Empty
        ``(0, 2)`` when the mesh has no boundary.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_int32_2d((0, 2), device=faces.device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    if edges is None:
        edges = tw.edges.faces_to_edges(faces)

    indices = tw.grouping.group_int_rows(edges_sorted, 1, int(vertices.shape[0])).flatten()
    return twt.as_array2d_int32(tw.array.gather(edges, indices))


def boundary_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> list[wp.array[wp.int32]]:
    """
    Ordered vertex-index loops along each mesh boundary.

    Each boundary loop is a simple cycle in the directed-boundary-edge successor graph (on a
    manifold boundary every boundary vertex has exactly one outgoing boundary edge). Loops are
    grouped via connected-component labeling, then each vertex's ordinal position within its
    loop is ranked by following the successor chain from itself to its loop's canonical start
    (the smallest vertex index in the loop) — entirely GPU-parallel, mirroring
    ``igl::boundary_loop`` (`reference/libigl/include/igl/boundary_loop.cpp`, first overload).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base and
        the successor-array size).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges, forwarded to
        :func:`boundary_edges` and :func:`oriented_boundary_edges`.
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges, forwarded to
        :func:`oriented_boundary_edges`.

    Returns
    -------
    list[wp.array[wp.int32]]
        One array per boundary loop, each holding the ordered vertex indices around that loop
        (not repeating the start vertex), on ``faces.device``. Empty list when the mesh has no
        boundary.

    Notes
    -----
    Assumes a manifold boundary: each boundary vertex has exactly one outgoing and one incoming
    boundary edge. A vertex shared by more than one loop (non-manifold pinch point) keeps only
    one outgoing successor edge (last write wins).

    See Also
    --------
    :func:`boundary_loop`
    :func:`oriented_boundary_edges`
    :func:`igl.boundary_loop_all`
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return []

    undirected = boundary_edges(vertices, faces, edges_sorted)
    n_boundary_edges = int(undirected.shape[0])
    if n_boundary_edges == 0:
        return []

    directed = oriented_boundary_edges(vertices, faces, edges_sorted, edges)

    n_vertices = int(vertices.shape[0])
    next_vertex = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.scatter_successor,
        dim=n_boundary_edges,
        inputs=[directed, next_vertex],
        device=device,
    )

    labels = tw.graph.connected_component_labels_from_edges(undirected, node_count=n_vertices)

    boundary_vertices = boundary_vertex_indices(vertices, faces, edges_sorted)
    n_boundary_vertices = int(boundary_vertices.shape[0])

    label_min = wp.full(n_vertices, n_vertices, dtype=wp.int32, device=device)
    label_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.scatter_loop_min_and_count,
        dim=n_boundary_vertices,
        inputs=[boundary_vertices, labels, label_min, label_count],
        device=device,
    )

    position = wp.empty(n_boundary_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.rank_loop_positions,
        dim=n_boundary_vertices,
        inputs=[boundary_vertices, next_vertex, labels, label_min, label_count, position],
        device=device,
    )

    vertex_labels = tw.array.gather(labels, boundary_vertices)
    unique_labels, loop_index = tw.unique.unique_1d(vertex_labels, return_inverse=True)
    n_loops = int(unique_labels.shape[0])

    loop_sizes = tw.array.gather(label_count, unique_labels)
    offsets = wp.empty(n_loops, dtype=wp.int32, device=device)
    wp.utils.array_scan(loop_sizes, out_array=offsets, inclusive=False)

    flat_loops = wp.empty(n_boundary_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.scatter_loop_slot,
        dim=n_boundary_vertices,
        inputs=[boundary_vertices, loop_index, position, offsets, flat_loops],
        device=device,
    )

    offsets_np = offsets.numpy()
    loop_sizes_np = loop_sizes.numpy()
    return [
        wp.clone(flat_loops[int(offsets_np[i]) : int(offsets_np[i]) + int(loop_sizes_np[i])])
        for i in range(n_loops)
    ]


def boundary_loop(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.int32]:
    """
    Ordered vertex-index loop along the longest mesh boundary.

    Parameters
    ----------
    vertices, faces, edges_sorted, edges
        Forwarded to :func:`boundary_loops`.

    Returns
    -------
    wp.array[wp.int32]
        Ordered vertex indices around the longest boundary loop, on ``faces.device``. Empty
        when the mesh has no boundary.

    See Also
    --------
    :func:`boundary_loops`
    :func:`igl.boundary_loop`
    """
    loops = boundary_loops(vertices, faces, edges_sorted, edges)
    if not loops:
        return wp.empty(0, dtype=wp.int32, device=faces.device)
    return max(loops, key=lambda loop: int(loop.shape[0]))


def boundary_vertex_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.int32]:
    """
    Sorted unique vertex indices lying on the mesh boundary.

    A vertex belongs to the boundary when it belongs to at least one boundary edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges, forwarded to :func:`boundary_edges`.

    Returns
    -------
    wp.array[wp.int32]
        Sorted unique boundary vertex indices on ``faces.device``. Empty when the mesh has
        no boundary.
    """
    edges = boundary_edges(vertices, faces, edges_sorted)
    return tw.unique.unique_1d(edges.flatten())


def boundary_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.vec3]:
    """
    Coordinates of the vertices lying on the mesh boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges, forwarded to :func:`boundary_vertex_indices`.

    Returns
    -------
    wp.array[wp.vec3]
        Shape ``(n_boundary_vertices,)`` boundary vertex positions on ``vertices.device``,
        ordered by ascending vertex index. Empty when the mesh has no boundary.
    """
    indices = boundary_vertex_indices(vertices, faces, edges_sorted)
    return tw.array.gather(vertices, indices)
