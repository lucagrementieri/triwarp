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
