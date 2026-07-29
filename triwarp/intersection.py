"""
Cross-sections and level sets: where a mesh meets a plane, another mesh, or an isovalue.

A plane section and an isocontour are the same operation seen twice: intersecting a mesh with a
plane is exactly extracting the zero level set of that plane's signed distance. So
[`mesh_with_plane`][triwarp.intersection.mesh_with_plane] and
[`marching_triangles`][triwarp.intersection.marching_triangles] sit side by side here — the first
takes the field implicitly as a plane, the second takes any per-vertex scalar field — and both emit
the [`triwarp.polyline`][triwarp.polyline] convention: one array of points per curve, closed curves
not repeating their first point, and a parallel list of closed flags.

[`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh] is the genuine intersection in the set
sense, and [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane] keeps the cut
geometry rather than the curve.
"""

from __future__ import annotations

import itertools

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.kernels import intersection as kernel_intersections


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

    wp.map(
        kernel_intersections.plane_with_line,
        plane_origin,
        plane_normal,
        start_points,
        end_points,
        wp.bool(line_segments),
        out=[intersections, valid],
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
    wp.map(
        kernel_intersections.point_plane_dot, vertices, plane_origin, plane_normal, out=vertex_dots
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


def marching_triangles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    n_vertices: int | None = None,
) -> tuple[list[wp.array[wp.vec3]], list[bool]]:
    """
    Extract the ``values == isovalue`` level set as a list of polylines.

    Each returned curve is a connected component of the level set, oriented so that the region where
    ``values > isovalue`` lies to its left (with the vertex normals as up). Curves close up unless
    they run into a mesh boundary, so an open curve begins and ends on a boundary edge.

    A value equal to the isovalue counts as positive, which keeps every cut face at exactly one
    segment; a contour running exactly through a vertex therefore yields zero-length segments rather
    than an ambiguous junction.

    Crossings are matched by [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse], then
    linked on the host: the segment list is compacted on device first, so the readback is one
    ``int32`` pair per segment and the walk is over the compacted arrays only (the same successor-
    graph shape [`boundary_loops`][triwarp.boundary.boundary_loops] solves on device).

    !!! note "The returned arrays are views"
        Every curve slices one packed buffer, so holding a single curve keeps them all alive and
        writing into one writes into the shared allocation. ``wp.clone`` a curve for an independent
        buffer.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field (what
        [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] returns) is interpolated in
        ``float64``.
    isovalue
        Level to extract.
    n_vertices
        Total vertex count, forwarded to
        [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse] as the hash base. When
        ``None`` it is inferred there with a host readback.

    Returns
    -------
    curves : list[wp.array[wp.vec3]]
        One array of points per level-set component, in order along the curve, on
        ``vertices.device``. Closed curves do not repeat their first point.
    closed : list[bool]
        Whether each curve is a closed loop.

    Raises
    ------
    ValueError
        If two segments start on the same mesh edge, which means the faces are not consistently
        oriented (the level set cannot then be linked into oriented curves). Repair the winding with
        [`make_winding_consistent`][triwarp.repair.make_winding_consistent] first.

    See Also
    --------
    [`mesh_with_plane`][triwarp.intersection.mesh_with_plane]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    [`polyline_length`][triwarp.polyline.polyline_length]
    ``potpourri3d.MarchingTrianglesSolver``
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return [], []

    # Subtract the isovalue once so the kernel only has to test signs; ``wp.map`` keeps the
    # element-wise work out of the kernel and preserves the field's dtype.
    shifted = wp.empty(int(values.shape[0]), dtype=values.dtype, device=device)
    wp.map(wp.sub, values, values.dtype(isovalue), out=shifted)

    edge_ids = tw.edges.edges_unique_inverse(faces, n_vertices=n_vertices)
    valid = wp.empty(n_faces, dtype=wp.bool, device=device)
    segments = wp.empty((n_faces, 2), dtype=wp.vec3, device=device)
    segment_edges = wp.empty((n_faces, 2), dtype=wp.int32, device=device)
    wp.launch(
        kernel_intersections.marching_triangles_segments,
        dim=n_faces,
        inputs=[vertices, faces, shifted, edge_ids, valid, segments, segment_edges],
        device=device,
    )

    cut_faces = tw.array.flatnonzero(valid)
    n_segments = int(cut_faces.shape[0])
    if n_segments == 0:
        return [], []

    hit_segments = wp.empty((n_segments, 2), dtype=wp.vec3, device=device)
    wp.copy(hit_segments, segments[cut_faces])
    hit_edges = wp.empty((n_segments, 2), dtype=wp.int32, device=device)
    wp.copy(hit_edges, segment_edges[cut_faces])

    chains, closed = _link_segments(hit_edges.numpy())
    if not chains:
        return [], []

    # One gather assembles every curve: the chains index the flattened endpoint buffer, so the
    # packed result can be sliced per curve without a launch each.
    endpoints = hit_segments.reshape((2 * n_segments,))
    slots = wp.array(np.concatenate(chains), dtype=wp.int32, device=device)
    packed = tw.array.gather(endpoints, slots)
    bounds = np.cumsum([0] + [len(chain) for chain in chains])
    curves = [packed[int(begin) : int(end)] for begin, end in itertools.pairwise(bounds)]
    return curves, closed


def _link_segments(segment_edges: np.ndarray) -> tuple[list[np.ndarray], list[bool]]:
    """
    Chain oriented segments into curves, returning endpoint slots and closed flags.

    ``segment_edges[i]`` holds the unique-edge ids the two endpoints of segment ``i`` lie on. Since
    the segments are consistently oriented, an interior crossing edge appears once as some segment's
    outgoing endpoint and once as another's incoming endpoint, so the successor relation is a
    permutation on all but the boundary-terminated chains — the same successor-graph structure
    [`boundary_loops`][triwarp.boundary.boundary_loops] walks.

    Slots index the flattened endpoint buffer: endpoint ``e`` of segment ``i`` is ``2 * i + e``.
    """
    n_segments = int(segment_edges.shape[0])
    start_edge = segment_edges[:, 0]
    end_edge = segment_edges[:, 1]

    order = np.argsort(start_edge, kind="stable")
    sorted_starts = start_edge[order]
    if n_segments > 1 and (np.diff(sorted_starts) == 0).any():
        raise ValueError(
            "marching_triangles cannot link the level set: two segments start on the same edge, "
            "which means the faces are not consistently oriented."
        )
    position = np.searchsorted(sorted_starts, end_edge)
    found = (position < n_segments) & (
        sorted_starts[np.minimum(position, n_segments - 1)] == end_edge
    )
    successor = np.full(n_segments, -1, dtype=np.int64)
    successor[found] = order[position[found]]

    has_predecessor = np.zeros(n_segments, dtype=bool)
    has_predecessor[successor[successor >= 0]] = True

    chains: list[np.ndarray] = []
    closed: list[bool] = []
    visited = np.zeros(n_segments, dtype=bool)
    # Open curves first, from their unique starting segment; whatever is left is a cycle, entered at
    # its lowest-indexed segment so the result does not depend on face order.
    for start in np.concatenate([np.flatnonzero(~has_predecessor), np.arange(n_segments)]):
        if visited[start]:
            continue
        chain = []
        current = int(start)
        while current >= 0 and not visited[current]:
            visited[current] = True
            chain.append(current)
            current = int(successor[current])
        is_closed = current == int(start)
        slots = [2 * segment for segment in chain]
        if not is_closed:
            slots.append(2 * chain[-1] + 1)
        chains.append(np.array(slots, dtype=np.int32))
        closed.append(bool(is_closed))
    return chains, closed


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

    require_nonempty_mesh(target_faces, "mesh_with_mesh")
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
    wp.map(
        kernel_intersections.point_plane_dot, vertices, plane_origin, plane_normal, out=vertex_dots
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
    new_vertices, new_faces, _ = tw.repair.remove_unreferenced_vertices(all_vertices, all_faces)
    return new_vertices, new_faces
