"""
Move a scalar or vector field between the mesh's elements: faces, vertices and edges.

Averaging is the only honest transfer between element types on a mesh, and the direction determines
what it costs: face-to-vertex
([`average_onto_vertices`][triwarp.interpolation.average_onto_vertices]) and edge-to-vertex
([`average_from_edges_onto_vertices`][triwarp.interpolation.average_from_edges_onto_vertices]) are
gathers over the incident elements, while vertex-to-face
([`average_onto_faces`][triwarp.interpolation.average_onto_faces]) is the mean of exactly three
corners and needs no adjacency at all.

These are unweighted means. For an area- or angle-weighted transfer of *normals* specifically, see
[`triwarp.vertices`][triwarp.vertices]; for a UV-space resampling, see
[`triwarp.texture`][triwarp.texture].
"""

from __future__ import annotations

from typing import cast

import warp as wp

import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import interpolation as kernel_interpolation
from triwarp.kernels import scatter as kernel_scatter


def average_onto_faces(
    faces: wp.array[wp.int32], vertex_values: wp.array[wp.float32]
) -> twt.Array1dFloat32:
    """
    Move a scalar field defined on vertices to faces by averaging (``igl::average_onto_faces``).

    Each face value is the mean of ``vertex_values`` at its three corners.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    vertex_values
        Length-``n_vertices`` scalar field defined on vertices.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_faces`` scalar field defined on faces. Empty when ``n_faces == 0``.
    """
    n_faces = int(faces.shape[0]) // 3
    device = vertex_values.device
    if n_faces == 0:
        return cast(twt.Array1dFloat32, wp.empty(0, dtype=wp.float32, device=device))

    out_face_values = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_interpolation.average_onto_faces,
        dim=n_faces,
        inputs=[faces, vertex_values, out_face_values],
        device=device,
    )
    return cast(twt.Array1dFloat32, out_face_values)


def average_onto_vertices(
    n_vertices: int, faces: wp.array[wp.int32], face_values: wp.array[wp.float32]
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on faces to vertices by averaging (``igl::average_onto_vertices``).

    Each vertex value is the mean of ``face_values`` over incident triangle corners.
    Vertices referenced by no face divide by a zero valence and are ``nan``, matching
    the unguarded division in ``igl::average_onto_vertices``.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    face_values
        Length-``n_faces`` scalar field defined on faces.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_vertices`` scalar field defined on vertices.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3

    out_sum = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    out_valence = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_face_values_sum_and_valence,
        dim=n_faces,
        inputs=[faces, face_values, out_sum, out_valence],
        device=device,
    )
    wp.map(wp.div, out_sum, out_valence, out=out_sum)
    return out_sum


def average_from_edges_onto_vertices(
    n_vertices: int,
    faces: wp.array[wp.int32],
    edges: twt.Array2dInt32,
    edges_orientation: twt.Array2dInt32,
    edge_values: wp.array[wp.float32],
) -> wp.array[wp.float32]:
    """
    Move a scalar field defined on edges to vertices by averaging.

    (``igl::average_from_edges_onto_vertices``).

    For each face half-edge with non-negative orientation, ``edge_values`` at the
    corresponding unique edge is accumulated onto both of its endpoint vertices; each
    vertex value is then the mean over its incident (positively oriented) half-edges.
    Vertices with zero valence are left at ``0``, matching the guarded division in
    ``igl::average_from_edges_onto_vertices``.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges
        Shape ``(n_faces, 3)`` mapping from each face half-edge to a unique edge index,
        as produced by ``igl::orient_halfedges``.
    edges_orientation
        Shape ``(n_faces, 3)`` half-edge orientation relative to its unique edge
        (``igl::orient_halfedges``); half-edges with a negative value are skipped.
    edge_values
        Length-``n_unique_edges`` scalar field defined on unique edges.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_vertices`` scalar field defined on vertices.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3

    out_sum = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    out_valence = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_edges_sum_and_valence,
        dim=n_faces,
        inputs=[faces, edges, edges_orientation, edge_values, out_sum, out_valence],
        device=device,
    )
    wp.map(kernel_array.divide_if_positive, out_sum, out_valence, out=out_sum)
    return out_sum
