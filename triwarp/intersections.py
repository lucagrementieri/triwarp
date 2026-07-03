"""Mesh-plane intersection utilities on NVIDIA Warp."""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import intersections as kernel_intersections


def segments_with_plane(
    start_points: wp.array[wp.vec3],
    end_points: wp.array[wp.vec3],
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    *,
    line_segments: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.bool]]:
    """
    Calculate plane-line intersections for batched segment endpoints.

    Each row pair ``(start_points[i], end_points[i])`` defines one line to test.
    Matches [`trimesh.intersections.plane_lines`][] with Trimesh's ``(2, n, 3)``
    layout expressed as two length-``n`` ``wp.vec3`` arrays.

    Parameters
    ----------
    plane_origin
        Point on the plane.
    plane_normal
        Plane normal vector.
    start_points
        ``(n,)`` first endpoint of each segment.
    end_points
        ``(n,)`` second endpoint of each segment.
    line_segments
        When ``True``, only mark intersections valid if endpoints lie on
        different sides of the plane.

    Returns
    -------
    intersections
        ``(n,)`` intersection points (undefined where ``valid`` is ``False``).
    valid
        ``(n,)`` mask indicating a valid intersection per segment.
    """
    if start_points.shape != end_points.shape:
        raise ValueError("start_points and end_points must have the same shape")
    n = int(start_points.shape[0])
    device = start_points.device
    intersections = wp.empty(n, dtype=wp.vec3, device=device)
    valid = wp.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return intersections, valid

    wp.launch(
        kernel_intersections.segments_with_plane,
        dim=n,
        inputs=[
            start_points,
            end_points,
            plane_origin,
            plane_normal,
            line_segments,
            intersections,
            valid,
        ],
        device=device,
    )
    return intersections, valid


def mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    return_faces: bool = False,
) -> wp.array[wp.vec3] | tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Intersect a mesh with a plane, returning line segments on the plane.

    Matches [`trimesh.intersections.mesh_plane`][] for indexed triangle meshes.
    To section a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.
    return_faces
        If ``True``, also return the source face index for each segment.

    Returns
    -------
    lines
        ``(m, 2)`` ``wp.vec3`` array of segment endpoints (logical shape ``(m, 2, 3)``).
    face_index
        Returned only when ``return_faces=True``; ``(m,)`` ``wp.int32`` source
        face indices into the mesh.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_segments = wp.empty((0, 2), dtype=wp.vec3, device=device)
        if return_faces:
            return empty_segments, wp.empty(0, dtype=wp.int32, device=device)
        return empty_segments

    vertex_dots = wp.empty(int(vertices.shape[0]), dtype=wp.float32, device=device)
    wp.launch(
        kernel_intersections.vertex_plane_dots,
        dim=int(vertices.shape[0]),
        inputs=[vertices, plane_origin, plane_normal, vertex_dots],
        device=device,
    )

    valid = wp.empty(n_faces, dtype=wp.bool, device=device)
    segments = wp.empty((n_faces, 2), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.mesh_with_plane_segments,
        dim=n_faces,
        inputs=[vertices, faces, vertex_dots, plane_origin, plane_normal, valid, segments],
        device=device,
    )

    hit_faces = tw.array.flatnonzero(valid)
    n_hit = int(hit_faces.shape[0])
    if n_hit == 0:
        empty_segments = wp.empty((0, 2), dtype=wp.vec3, device=device)
        if return_faces:
            return empty_segments, wp.empty(0, dtype=wp.int32, device=device)
        return empty_segments

    lines = wp.empty((n_hit, 2), dtype=wp.vec3, device=device)
    wp.copy(lines, segments[hit_faces])

    if not return_faces:
        return lines

    face_index = wp.empty(n_hit, dtype=wp.int32, device=device)
    wp.copy(face_index, hit_faces)
    return lines, face_index


def mesh_with_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
) -> wp.array[wp.vec3]:
    """
    Intersect two meshes, returning line segments along the intersection curve(s).

    Broad phase builds a ``wp.Mesh`` over the mesh with fewer faces and queries
    triangle AABBs via ``wp.mesh_query_aabb``; each triangle of the other mesh
    supplies the query box. Narrow phase
    applies separating-axis triangle tests and clips the intersection line to both
    triangles. Coplanar overlapping faces produce no segments.

    Parameters
    ----------
    vertices_a, faces_a
        First indexed triangle mesh.
    vertices_b, faces_b
        Second indexed triangle mesh.
    max_triangle_collisions
        Maximum broad-phase candidate pairs recorded per query triangle.

    Returns
    -------
    lines
        ``(m, 2)`` ``wp.vec3`` segment endpoints (logical shape ``(m, 2, 3)``).
    """
    device = vertices_a.device
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    if n_faces_a == 0 or n_faces_b == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")

    if n_faces_a <= n_faces_b:
        target_vertices, target_faces = vertices_a, faces_a
        query_vertices, query_faces = vertices_b, faces_b
    else:
        target_vertices, target_faces = vertices_b, faces_b
        query_vertices, query_faces = vertices_a, faces_a

    n_query = int(query_faces.shape[0]) // 3

    target_mesh = wp.Mesh(points=target_vertices, indices=target_faces)

    query_lower = wp.empty(n_query, dtype=wp.vec3, device=device)
    query_upper = wp.empty(n_query, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.face_aabb_bounds,
        dim=n_query,
        inputs=[query_vertices, query_faces, query_lower, query_upper],
        device=device,
    )

    target_indices, offsets, hit_counts = tw.proximity.query_mesh_aabb_bounds_with_offsets(
        target_mesh, query_lower, query_upper, max_hits=max_triangle_collisions
    )
    n_pairs = int(target_indices.shape[0])
    if n_pairs == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)

    pairs = twt.empty_int32_2d((n_pairs, 2), device=device)
    wp.launch(
        kernel_intersections.expand_query_target_pairs,
        dim=n_query,
        inputs=[offsets, hit_counts, target_indices, pairs],
        device=device,
    )

    valid = wp.empty(n_pairs, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.filter_intersecting_pairs,
        dim=n_pairs,
        inputs=[query_vertices, query_faces, target_vertices, target_faces, pairs, valid],
        device=device,
    )

    hit_pair_indices = tw.array.flatnonzero(valid)
    n_hit = int(hit_pair_indices.shape[0])
    if n_hit == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)

    hit_pairs = twt.empty_int32_2d((n_hit, 2), device=device)
    wp.copy(hit_pairs, pairs.reshape((-1, 2))[hit_pair_indices])

    segments = wp.empty((n_hit, 2), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.triangle_pair_segments,
        dim=n_hit,
        inputs=[query_vertices, query_faces, target_vertices, target_faces, hit_pairs, segments],
        device=device,
    )

    seg_valid = wp.empty(n_hit, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.segment_nondegenerate,
        dim=n_hit,
        inputs=[segments, seg_valid],
        device=device,
    )

    keep = tw.array.flatnonzero(seg_valid)
    n_keep = int(keep.shape[0])
    if n_keep == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)

    lines = wp.empty((n_keep, 2), dtype=wp.vec3, device=device)
    wp.copy(lines, segments[keep])
    return lines


def _compact_referenced_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Keep only vertices referenced by ``faces`` and reindex face indices from zero."""
    unique_idx, inverse = tw.unique.unique_1d(faces, return_inverse=True)
    n_unique = int(unique_idx.shape[0])
    compact_vertices = wp.empty(n_unique, dtype=wp.vec3, device=vertices.device)
    wp.copy(compact_vertices, vertices[unique_idx])
    compact_faces = inverse.reshape((-1,))
    return compact_vertices, compact_faces


def slice_mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Slice a mesh with a plane, returning the portion on the positive normal side.

    Matches [`trimesh.intersections.slice_faces_plane`][] for indexed triangle meshes
    (without UV handling). To slice a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.

    Returns
    -------
    new_vertices
        Vertices of the sliced mesh.
    new_faces
        Length-``3 * m`` flat triangle index buffer for the sliced mesh.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_vertices == 0:
        return vertices, faces

    vertex_dots = wp.empty(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_intersections.vertex_plane_dots,
        dim=n_vertices,
        inputs=[vertices, plane_origin, plane_normal, vertex_dots],
        device=device,
    )

    inside = wp.empty(n_faces, dtype=wp.bool, device=device)
    cut_quad = wp.empty(n_faces, dtype=wp.bool, device=device)
    cut_tri = wp.empty(n_faces, dtype=wp.bool, device=device)
    on_plane = wp.empty(n_faces, dtype=wp.bool, device=device)
    face_signs = twt.empty_int32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_intersections.classify_faces_for_slice,
        dim=n_faces,
        inputs=[faces, vertex_dots, inside, cut_quad, cut_tri, on_plane, face_signs],
        device=device,
    )
    wp.launch(
        kernel_intersections.resolve_on_plane_faces,
        dim=n_faces,
        inputs=[vertices, faces, plane_normal, on_plane, inside],
        device=device,
    )

    inside_idx = tw.array.flatnonzero(inside)
    quad_idx = tw.array.flatnonzero(cut_quad)
    tri_idx = tw.array.flatnonzero(cut_tri)
    n_in = int(inside_idx.shape[0])
    n_quad = int(quad_idx.shape[0])
    n_tri = int(tri_idx.shape[0])

    if n_quad + n_tri == 0:
        if n_in == 0:
            return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(
                0, dtype=wp.int32, device=device
            )
        return tw.selection.submesh_from_face_indices(
            vertices, faces, inside_idx, unique_indices=True
        )

    if n_in > 0:
        inside_faces_2d = twt.empty_int32_2d((n_in, 3), device=device)
        wp.copy(inside_faces_2d, faces.reshape((-1, 3))[inside_idx])
        inside_faces = inside_faces_2d.reshape((-1,))
    else:
        inside_faces = wp.empty(0, dtype=wp.int32, device=device)

    cut_vert_segments: list[wp.array[wp.vec3]] = []
    cut_face_segments: list[wp.array[wp.int32]] = []

    if n_quad > 0:
        quad_edge_points = wp.empty((n_quad, 3), dtype=wp.vec3, device=device)
        wp.launch(
            kernel_intersections.edge_plane_intersections,
            dim=n_quad,
            inputs=[vertices, faces, quad_idx, plane_origin, plane_normal, quad_edge_points],
            device=device,
        )
        quad_new_verts = wp.empty(2 * n_quad, dtype=wp.vec3, device=device)
        quad_new_faces = twt.empty_int32_2d((2 * n_quad, 3), device=device)
        wp.launch(
            kernel_intersections.emit_quad_cut,
            dim=n_quad,
            inputs=[
                faces,
                quad_idx,
                face_signs,
                quad_edge_points,
                wp.int32(n_vertices),
                quad_new_verts,
                quad_new_faces,
            ],
            device=device,
        )
        cut_vert_segments.append(quad_new_verts)
        cut_face_segments.append(quad_new_faces.reshape((-1,)))

    if n_tri > 0:
        tri_edge_points = wp.empty((n_tri, 3), dtype=wp.vec3, device=device)
        wp.launch(
            kernel_intersections.edge_plane_intersections,
            dim=n_tri,
            inputs=[vertices, faces, tri_idx, plane_origin, plane_normal, tri_edge_points],
            device=device,
        )
        tri_new_verts = wp.empty(2 * n_tri, dtype=wp.vec3, device=device)
        tri_new_faces = twt.empty_int32_2d((n_tri, 3), device=device)
        tri_vertex_base = wp.int32(n_vertices + 2 * n_quad)
        wp.launch(
            kernel_intersections.emit_tri_cut,
            dim=n_tri,
            inputs=[
                faces,
                tri_idx,
                face_signs,
                tri_edge_points,
                tri_vertex_base,
                tri_new_verts,
                tri_new_faces,
            ],
            device=device,
        )
        cut_vert_segments.append(tri_new_verts)
        cut_face_segments.append(tri_new_faces.reshape((-1,)))

    vert_segments = [vertices, *cut_vert_segments]
    face_segments = [inside_faces, *cut_face_segments]
    all_vertices, _ = tw.array.pack_1d_arrays(vert_segments)
    all_faces, _ = tw.array.pack_1d_arrays(face_segments)
    return _compact_referenced_vertices(all_vertices, all_faces)
