"""Mesh-plane intersection utilities on NVIDIA Warp."""

from __future__ import annotations

import warp as wp

from triwarp.kernels import intersections as kernel_intersections
import triwarp as tw


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
