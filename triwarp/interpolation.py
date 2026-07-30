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

[`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices] moves a field between two
*different* meshes instead of between one mesh's element types, by closest-point projection onto the
source surface — the operation you need after a remesh or a decimation has changed the vertex set
under a field.
"""

from __future__ import annotations

from typing import TypeVar, cast

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import interpolation as kernel_interpolation
from triwarp.kernels import scatter as kernel_scatter

DType = TypeVar("DType")
"""Element type of a transferred field: any Warp dtype closed under scaling and addition."""


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


def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[DType],
    target_vertices: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
) -> tuple[wp.array[DType], wp.array[wp.float32]]:
    """
    Resample a source mesh's per-vertex field onto another mesh's vertices.

    For each target vertex, the closest point on the *source surface* is found and the source field
    is interpolated there barycentrically. This is the transfer to use when two meshes describe the
    same shape at different resolutions — after a remesh, a decimation or a reconstruction — and a
    field computed on one of them has to follow. It is MeshLab's ``transfer_attributes_per_vertex``
    with ``vertexsampling=False``.

    Because the closest point is taken on the surface rather than at the nearest *vertex*, the
    result is continuous in the target positions and exact for a field that is already linear on the
    source triangles — which is what makes it safe to chain (a transfer onto a refinement of the
    same mesh reproduces the field, not a blur of it).

    Parameters
    ----------
    source_vertices
        ``(n_source,)`` source mesh vertex positions.
    source_faces
        Length-``3 * n_source_faces`` ``wp.int32`` source triangle index buffer.
    source_values
        Length-``n_source`` field on the source vertices. Any Warp dtype closed under scaling and
        addition works: ``wp.float32`` for a scalar, ``wp.vec3`` for a normal or a colour.
    target_vertices
        ``(n_target,)`` positions to sample at — usually another mesh's vertices, but any point set
        will do.
    max_dist
        Maximum search radius per target vertex. Targets with no source face within it keep the
        zero-filled output and report ``inf`` in the returned distance, which is how a caller
        detects them. When ``None``, derived from the box enclosing both meshes, so every target
        hits.

    Returns
    -------
    values : wp.array[DType]
        Length-``n_target`` transferred field on ``target_vertices.device``. Zero-filled wherever
        the closest-point query missed.
    distance : wp.array[wp.float32]
        Length-``n_target`` distance from each target vertex to the source surface — the transfer's
        own confidence measure, and ``inf`` for a miss.

    See Also
    --------
    [`average_onto_vertices`][triwarp.interpolation.average_onto_vertices]
    [`triwarp.proximity.closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    [`triwarp.triangles.barycentric_to_points`][triwarp.triangles.barycentric_to_points]
    """
    device = target_vertices.device
    n_target = int(target_vertices.shape[0])
    n_source = int(source_vertices.shape[0])
    if int(source_values.shape[0]) != n_source:
        raise ValueError(
            f"source_values must have one entry per source vertex, got "
            f"{source_values.shape[0]} for {n_source} vertices"
        )

    out_values = wp.zeros(n_target, dtype=source_values.dtype, device=device)
    if n_target == 0 or int(source_faces.shape[0]) == 0:
        return out_values, wp.full(n_target, float("inf"), dtype=wp.float32, device=device)

    closest, distance, face_id = tw.proximity.closest_point_on_mesh(
        source_vertices, source_faces, target_vertices, max_dist=max_dist
    )
    wp.launch(
        kernel_interpolation.transfer_onto_vertices,
        dim=n_target,
        inputs=[
            source_vertices,
            source_faces,
            source_values,
            closest,
            face_id,
            out_values,
            distance,
        ],
        device=device,
    )
    return out_values, distance
