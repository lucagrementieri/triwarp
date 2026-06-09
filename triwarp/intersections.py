"""Mesh-plane intersection utilities on NVIDIA Warp."""

from __future__ import annotations

import warp as wp

from triwarp.kernels import intersections as kernel_intersections
import triwarp as tw
import triwarp.typing as twt


def segments_with_plane(
    start_points: wp.array[wp.vec3],
    end_points: wp.array[wp.vec3],
    plane_origin: wp.vec3,
    plane_normal: wp.vec3,
    *,
    line_segments: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.bool]]:
    """Calculate plane-line intersections for batched segment endpoints.

    Each row pair ``(start_points[i], end_points[i])`` defines one line to test.
    Matches :func:`trimesh.intersections.plane_lines` with Trimesh's ``(2, n, 3)``
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
    if end_points.device != device:
        raise ValueError("start_points and end_points must live on the same device")

    intersections = wp.empty(n, dtype=wp.vec3, device=device)
    valid = wp.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return intersections, valid

    wp.launch(
        kernel_intersections.segments_with_plane,
        dim=n,
        inputs=[start_points, end_points, plane_origin, plane_normal, line_segments, intersections, valid],
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
    """Intersect a mesh with a plane, returning line segments on the plane.

    Matches :func:`trimesh.intersections.mesh_plane` for indexed triangle meshes.
    To section a face subset, extract a submesh first (e.g.
    :func:`triwarp.selection.submesh_from_face_indices`).

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
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")

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
    """Intersect two meshes, returning line segments along the intersection curve(s).

    Broad phase uses a BVH over the mesh with fewer faces; each triangle of the
    other mesh queries that BVH with its axis-aligned bounding box. Narrow phase
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
    for name, arr in (("faces_a", faces_a), ("vertices_b", vertices_b), ("faces_b", faces_b)):
        if arr.device != device:
            raise ValueError(f"vertices_a and {name} must live on the same device, got {device} and {arr.device}")

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

    n_target = int(target_faces.shape[0]) // 3
    n_query = int(query_faces.shape[0]) // 3

    target_lower = wp.empty(n_target, dtype=wp.vec3, device=device)
    target_upper = wp.empty(n_target, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.face_aabb_bounds,
        dim=n_target,
        inputs=[target_vertices, target_faces, target_lower, target_upper],
        device=device,
    )

    bvh = tw.points.bvh_from_bounds(target_lower, target_upper)

    query_lower = wp.empty(n_query, dtype=wp.vec3, device=device)
    query_upper = wp.empty(n_query, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.face_aabb_bounds,
        dim=n_query,
        inputs=[query_vertices, query_faces, query_lower, query_upper],
        device=device,
    )

    target_indices, offsets, hit_counts = tw.points.query_bvh_aabb_bounds_with_offsets(
        bvh, query_lower, query_upper, max_hits=max_triangle_collisions
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
    wp.launch(kernel_intersections.segment_nondegenerate, dim=n_hit, inputs=[segments, seg_valid], device=device)

    keep = tw.array.flatnonzero(seg_valid)
    n_keep = int(keep.shape[0])
    if n_keep == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)

    lines = wp.empty((n_keep, 2), dtype=wp.vec3, device=device)
    wp.copy(lines, segments[keep])
    return lines
