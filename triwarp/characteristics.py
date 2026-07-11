from __future__ import annotations

from typing import cast

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import characteristics as kernel_characteristics
from triwarp.kernels import intersection as kernel_intersections
from triwarp.kernels import sample as kernel_sample


def euler_characteristic(faces: wp.array[wp.int32]) -> int:
    """
    Euler characteristic ``V - E + F`` of the mesh (topological invariant).

    Counts distinct referenced vertices, unique undirected edges, and faces, matching
    [`trimesh.Trimesh.euler_number`][] (which uses referenced vertices, ``edges_unique``, and
    faces). For a closed genus-``g`` surface this equals ``2 - 2 * g``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    int
        ``(#distinct referenced vertices) - (#unique edges) + (#faces)``. ``0`` for an empty mesh.

    See Also
    --------
    [`edges_unique`][triwarp.edges.edges_unique]
    [`trimesh.Trimesh.euler_number`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0

    n_referenced = int(tw.unique.unique_1d(faces).shape[0])
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=tw.vertices.n_vertices(faces))
    n_edges = int(unique_edges.shape[0])
    return n_referenced - n_edges + n_faces


def is_edge_manifold(faces: wp.array[wp.int32], allow_boundary_edges: bool = True) -> bool:
    """
    Whether every undirected mesh edge is shared by a manifold number of faces.

    Counts how many faces share each undirected edge. With ``allow_boundary_edges=True`` an edge is
    manifold when it belongs to one or two faces; with ``allow_boundary_edges=False`` it must belong
    to exactly two faces (no boundary or non-manifold edges), matching Open3D's ``IsEdgeManifold``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    allow_boundary_edges
        When ``True`` (default) boundary edges (used by a single face) are allowed. When ``False``
        every edge must be shared by exactly two faces.

    Returns
    -------
    bool
        ``True`` when all edges satisfy the manifold condition. Vacuously ``True`` for an empty
        mesh.

    See Also
    --------
    [`is_vertex_manifold`][triwarp.characteristics.is_vertex_manifold]
    [`is_watertight`][triwarp.characteristics.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_edge_manifold``; libigl ``is_edge_manifold``
    corresponds to the ``allow_boundary_edges=True`` case.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    keys = tw.unique.hash_indices_rows(edges_sorted, max_index=tw.vertices.n_vertices(faces))
    _, counts = tw.unique.unique_1d(keys, return_counts=True)

    min_count, max_count = tw.reduce.minmax(cast(twt.Array1dInt32, counts))
    if allow_boundary_edges:
        return max_count <= 2
    return min_count == 2 and max_count == 2


def edge_manifold_mask(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    allow_boundary_edges: bool = True,
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether all three of each face's undirected edges are edge-manifold.

    An edge is manifold when shared by one or two faces (``allow_boundary_edges=True``) or by
    exactly two faces (``allow_boundary_edges=False``); a face is flagged ``True`` only when all
    three of its edges qualify. This is a per-face collapse of libigl's ``BF`` (its per-corner
    ``is_edge_manifold`` matrix), and
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold] is ``True`` iff every entry of
    this mask is ``True``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges in
        [`faces_to_edges`][triwarp.edges.faces_to_edges] row order (each row min-first). When
        ``None``, built from ``faces``.
    allow_boundary_edges
        When ``True`` (default) boundary edges (used by a single face) count as manifold. When
        ``False`` every edge of a face must be shared by exactly two faces.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``faces.device``. Empty for an empty mesh.

    See Also
    --------
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]
    [`vertex_manifold_mask`][triwarp.characteristics.vertex_manifold_mask]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    keys = tw.unique.hash_indices_rows(edges_sorted, max_index=tw.vertices.n_vertices(faces))
    _, inverse, counts = tw.unique.unique_1d(keys, return_inverse=True, return_counts=True)

    n_unique = int(counts.shape[0])
    edge_ok = wp.empty(n_unique, dtype=wp.bool, device=device)
    wp.map(kernel_characteristics.edge_manifold, counts, wp.bool(allow_boundary_edges), out=edge_ok)

    out_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.face_edge_manifold_mask,
        dim=n_faces,
        inputs=[inverse, edge_ok, out_mask],
        device=device,
    )
    return out_mask


def is_vertex_manifold(faces: wp.array[wp.int32]) -> bool:
    """
    Whether every referenced vertex has a single edge-connected fan of faces.

    A vertex is manifold when the faces incident to it form one connected group under
    "shares an edge through this vertex" adjacency; two fans meeting only at the vertex (a bow-tie)
    are non-manifold. Unreferenced vertices in ``[0, max(faces)]`` are treated as non-manifold,
    matching libigl's ``is_vertex_manifold`` and Open3D's ``IsVertexManifold``.

    The check builds the corner graph (node ``3 * f + k`` per face corner), links the corresponding
    corners of edge-adjacent faces, and requires all corners of each vertex to fall in one connected
    component.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when every referenced vertex is manifold. Vacuously ``True`` for an empty mesh.

    See Also
    --------
    [`vertex_manifold_mask`][triwarp.characteristics.vertex_manifold_mask]
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]
    [`is_watertight`][triwarp.characteristics.is_watertight]

    Notes
    -----
    Corner adjacency is derived from
    [`face_adjacency`][triwarp.graph.face_adjacency], which pairs faces across edges shared by
    exactly two faces; on edge-non-manifold meshes (an edge shared by three or more faces) read the
    result together with [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]. Equivalent
    to libigl ``is_vertex_manifold``. The output is sized to ``max(faces) + 1`` so unreferenced
    vertices in that range count as non-manifold; use
    [`vertex_manifold_mask`][triwarp.characteristics.vertex_manifold_mask] for a per-vertex flag
    sized to a caller-provided vertex buffer.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    n_vertices = tw.vertices.n_vertices(faces)
    n_corners = n_faces * 3
    adjacency, adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    m = int(adjacency.shape[0])

    corner_edges = twt.empty_int32_2d((2 * m, 2), device=device)
    if m > 0:
        wp.launch(
            kernel_characteristics.build_corner_adjacency_edges,
            dim=m,
            inputs=[faces, adjacency, adjacency_edges, corner_edges],
            device=device,
        )

    labels = tw.graph.connected_component_labels_from_edges(corner_edges, node_count=n_corners)

    min_label = wp.full(
        n_vertices, tw.reduce.max_for_dtype(wp.int32), dtype=wp.int32, device=device
    )
    manifold = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.corner_vertex_reduce,
        dim=n_corners,
        inputs=[faces, labels, min_label, manifold],
        device=device,
    )
    wp.launch(
        kernel_characteristics.corner_vertex_check,
        dim=n_corners,
        inputs=[faces, labels, min_label, manifold],
        device=device,
    )
    return bool(tw.reduce.all(manifold))


def vertex_manifold_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """
    Per-vertex flag: whether each vertex has a single edge-connected fan of faces.

    A vertex is manifold when the faces incident to it form one connected group under
    "shares an edge through this vertex" adjacency; two fans meeting only at the vertex (a bow-tie)
    are non-manifold. Unreferenced vertices are flagged ``False``, matching libigl's
    ``is_vertex_manifold`` per-vertex output ``B`` and Open3D's ``GetNonManifoldVertices``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions. Only the length is used; it sets the output size so that
        trailing unreferenced vertices are reported (``False``).
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_vertices`` on ``faces.device``.

    See Also
    --------
    [`is_vertex_manifold`][triwarp.characteristics.is_vertex_manifold]
    [`edge_manifold_mask`][triwarp.characteristics.edge_manifold_mask]
    """
    device = faces.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.zeros(n_vertices, dtype=wp.bool, device=device)

    n_corners = n_faces * 3
    adjacency, adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    m = int(adjacency.shape[0])

    corner_edges = twt.empty_int32_2d((2 * m, 2), device=device)
    if m > 0:
        wp.launch(
            kernel_characteristics.build_corner_adjacency_edges,
            dim=m,
            inputs=[faces, adjacency, adjacency_edges, corner_edges],
            device=device,
        )

    labels = tw.graph.connected_component_labels_from_edges(corner_edges, node_count=n_corners)

    min_label = wp.full(
        n_vertices, tw.reduce.max_for_dtype(wp.int32), dtype=wp.int32, device=device
    )
    manifold = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.corner_vertex_reduce,
        dim=n_corners,
        inputs=[faces, labels, min_label, manifold],
        device=device,
    )
    wp.launch(
        kernel_characteristics.corner_vertex_check,
        dim=n_corners,
        inputs=[faces, labels, min_label, manifold],
        device=device,
    )
    return manifold


def is_self_intersecting(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, max_triangle_collisions: int = 32
) -> bool:
    """
    Whether any two non-adjacent triangles of the mesh intersect.

    Broad phase builds a ``warp.Mesh`` and queries each triangle's AABB for candidate
    overlaps; narrow phase runs a separating-axis triangle test on each candidate pair, skipping
    pairs that share a vertex. Mirrors Open3D's ``IsSelfIntersecting`` (AABB pre-test followed by a
    triangle-triangle test on non-neighbouring faces).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_triangle_collisions
        Broad-phase candidate cap per query triangle. Raise this for meshes with many triangles
        packed into overlapping bounding boxes.

    Returns
    -------
    bool
        ``True`` when at least one intersecting non-adjacent triangle pair exists. ``False`` for
        meshes with fewer than two faces.

    See Also
    --------
    [`self_intersecting_face_mask`][triwarp.characteristics.self_intersecting_face_mask]
    [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh]
    [`is_watertight`][triwarp.characteristics.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_self_intersecting``.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces < 2:
        return False
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")

    mesh = wp.Mesh(points=vertices, indices=faces)

    lower = wp.empty(n_faces, dtype=wp.vec3, device=device)
    upper = wp.empty(n_faces, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.face_aabb_bounds,
        dim=n_faces,
        inputs=[vertices, faces, lower, upper],
        device=device,
    )

    target_indices, offsets, hit_counts = tw.proximity.query_mesh_aabb_bounds_with_offsets(
        mesh, lower, upper, max_hits=max_triangle_collisions
    )
    n_pairs = int(target_indices.shape[0])
    if n_pairs == 0:
        return False

    pairs = twt.empty_int32_2d((n_pairs, 2), device=device)
    wp.launch(
        kernel_intersections.expand_query_target_pairs,
        dim=n_faces,
        inputs=[offsets, hit_counts, target_indices, pairs],
        device=device,
    )

    valid = wp.empty(n_pairs, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.filter_intersecting_pairs,
        dim=n_pairs,
        inputs=[vertices, faces, vertices, faces, pairs, valid],
        device=device,
    )
    return bool(tw.reduce.any(valid))


def self_intersecting_face_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, max_triangle_collisions: int = 32
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether each triangle intersects some non-adjacent triangle.

    Broad phase builds a ``warp.Mesh`` and queries each triangle's AABB for candidate overlaps;
    narrow phase runs a separating-axis triangle test on each candidate pair (skipping pairs that
    share a vertex), and both faces of every intersecting pair are flagged.
    [`is_self_intersecting`][triwarp.characteristics.is_self_intersecting] is ``True`` iff any entry
    of this mask is ``True``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_triangle_collisions
        Broad-phase candidate cap per query triangle. Raise this for meshes with many triangles
        packed into overlapping bounding boxes.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``faces.device``. All-``False`` for meshes with fewer than two faces.

    See Also
    --------
    [`is_self_intersecting`][triwarp.characteristics.is_self_intersecting]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    mask = wp.zeros(n_faces, dtype=wp.bool, device=device)
    if n_faces < 2:
        return mask
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")

    mesh = wp.Mesh(points=vertices, indices=faces)

    lower = wp.empty(n_faces, dtype=wp.vec3, device=device)
    upper = wp.empty(n_faces, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.face_aabb_bounds,
        dim=n_faces,
        inputs=[vertices, faces, lower, upper],
        device=device,
    )

    target_indices, offsets, hit_counts = tw.proximity.query_mesh_aabb_bounds_with_offsets(
        mesh, lower, upper, max_hits=max_triangle_collisions
    )
    n_pairs = int(target_indices.shape[0])
    if n_pairs == 0:
        return mask

    pairs = twt.empty_int32_2d((n_pairs, 2), device=device)
    wp.launch(
        kernel_intersections.expand_query_target_pairs,
        dim=n_faces,
        inputs=[offsets, hit_counts, target_indices, pairs],
        device=device,
    )

    valid = wp.empty(n_pairs, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.filter_intersecting_pairs,
        dim=n_pairs,
        inputs=[vertices, faces, vertices, faces, pairs, valid],
        device=device,
    )
    wp.launch(
        kernel_characteristics.mark_intersecting_faces,
        dim=n_pairs,
        inputs=[pairs, valid, mask],
        device=device,
    )
    return mask


def is_winding_consistent(faces: wp.array[wp.int32]) -> bool:
    """
    Whether every shared edge is traversed in opposite directions by its two faces.

    A mesh has consistent winding when, for each edge shared by two faces, the two faces list the
    edge's endpoints in opposite order (so their normals agree locally). This is a property of the
    current winding, unlike [`is_orientable`][triwarp.characteristics.is_orientable], which allows
    faces to be flipped. Matches [`trimesh.Trimesh.is_winding_consistent`][].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when winding is consistent across all shared edges. Vacuously ``True`` for an empty
        mesh or a mesh with no shared edges.

    See Also
    --------
    [`winding_consistent_mask`][triwarp.characteristics.winding_consistent_mask]
    [`is_orientable`][triwarp.characteristics.is_orientable]
    [`is_volume`][triwarp.characteristics.is_volume]
    [`trimesh.Trimesh.is_winding_consistent`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    mask = winding_consistent_mask(faces)
    if int(mask.shape[0]) == 0:
        return True
    return bool(tw.reduce.all(mask))


def winding_consistent_mask(
    faces: wp.array[wp.int32],
    edges: twt.Array2dInt32 | None = None,
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.bool]:
    """
    Per shared-edge flag: whether an undirected edge's two faces traverse it in opposite directions.

    Built over the undirected edges shared by exactly two faces (edge groups of length two). An
    entry is ``True`` when the two directed half-edges are reversed (locally consistent normals);
    boundary and non-manifold edges have no entry.
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent] is ``True`` iff every
    entry is ``True``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges in
        [`faces_to_edges`][triwarp.edges.faces_to_edges] row order. Built from ``faces`` when
        ``None``.
    edges_sorted
        Optional precomputed sorted (min-first) edges in the same row order. Built from ``faces``
        when ``None``.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_shared_edges`` on ``faces.device``. Empty when the mesh has no shared edges.

    See Also
    --------
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]
    [`face_orientation_mask`][triwarp.characteristics.face_orientation_mask]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    if edges is None:
        edges = tw.edges.faces_to_edges(faces)
    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)

    edge_groups = tw.grouping.group_int_rows(edges_sorted, length=2)
    n_groups = int(edge_groups.shape[0])
    if n_groups == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    consistent = wp.empty(n_groups, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.edge_pair_winding_mask,
        dim=n_groups,
        inputs=[edges, edge_groups, consistent],
        device=device,
    )
    return consistent


def _orientation_bits(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], twt.Array2dInt32, wp.array[wp.int32], int]:
    """
    Per-face Z2 orientation bits (relative to each component seed) plus signed face-adjacency edges.

    ``orient[f]`` is ``0`` for a face that agrees with its connected component's seed and ``1`` for
    a face that must be flipped to agree — the flip mask consumed by
    [`face_orientation_mask`][triwarp.characteristics.face_orientation_mask] and
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]. ``m`` is the number of
    face-adjacency rows; when ``m == 0`` the returned ``signed_edges`` / ``signs`` are empty.
    Assumes ``n_faces > 0`` (callers guard).
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    adjacency, adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    m = int(adjacency.shape[0])

    labels = tw.graph.connected_component_labels_from_edges(adjacency, node_count=n_faces)
    orient = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_characteristics.seed_orientation, dim=n_faces, inputs=[labels, orient], device=device
    )

    if m == 0:
        signed_edges = twt.empty_int32_2d((0, 2), device=device)
        signs = wp.empty(0, dtype=wp.int32, device=device)
        return orient, signed_edges, signs, m

    signed_edges = twt.empty_int32_2d((m, 2), device=device)
    signs = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_characteristics.build_signed_face_edges,
        dim=m,
        inputs=[faces, adjacency, adjacency_edges, signed_edges, signs],
        device=device,
    )

    changed = wp.zeros(1, dtype=wp.int32, device=device)
    for _ in range(n_faces):
        changed.zero_()
        wp.launch(
            kernel_characteristics.propagate_orientation,
            dim=m,
            inputs=[signed_edges, signs, orient, changed],
            device=device,
        )
        if changed.numpy().item() == 0:
            break

    return orient, signed_edges, signs, m


def is_orientable(faces: wp.array[wp.int32]) -> bool:
    """
    Whether the faces admit a consistent orientation (allowing per-face flips).

    A mesh is orientable when each face can be assigned a flip bit so that every pair of
    edge-adjacent faces traverses their shared edge in opposite directions. This is a topological
    property independent of the current winding, matching Open3D's ``IsOrientable``.

    Each face-adjacency edge carries a Z2 flip bit; the check seeds one bit per connected component,
    propagates bits across adjacencies, then verifies that no adjacency violates its constraint (a
    contradiction means non-orientable).

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when a consistent orientation exists. Vacuously ``True`` for an empty mesh.

    See Also
    --------
    [`face_orientation_mask`][triwarp.characteristics.face_orientation_mask]
    [`is_watertight`][triwarp.characteristics.is_watertight]
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_orientable``. Unlike
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent], orientability allows
    individual faces to be flipped, so a consistently-orientable mesh with inconsistent winding
    still returns ``True``.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    orient, signed_edges, signs, m = _orientation_bits(faces)
    if m == 0:
        return True

    device = faces.device
    conflict = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_characteristics.verify_orientation,
        dim=m,
        inputs=[signed_edges, signs, orient, conflict],
        device=device,
    )
    return conflict.numpy().item() == 0


def face_orientation_mask(faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a face must be flipped to make winding consistent within its patch.

    Runs the Z2 orientation propagation over the face-adjacency graph (seeding one arbitrary
    reference face per connected component) and returns ``True`` for every face whose winding
    disagrees with its component's seed. Applying these flips yields a consistently wound mesh, so
    this is the per-face flip mask consumed by
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]. A mesh that is already
    consistently wound yields an all-``False`` mask.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``faces.device``. Empty for an empty mesh.

    See Also
    --------
    [`is_orientable`][triwarp.characteristics.is_orientable]
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    The reference orientation is arbitrary per connected component, so on a non-orientable patch the
    mask is still a best-effort flood-fill (matching ``trimesh.repair.fix_winding``).
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    orient, _, _, _ = _orientation_bits(faces)
    mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.map(kernel_characteristics.orientation_bit, orient, out=mask)
    return mask


def is_watertight(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> bool:
    """
    Whether the mesh bounds a closed volume with no self-intersections.

    Follows Open3D's ``IsWatertight``: the mesh must be edge-manifold with no boundary edges,
    vertex-manifold, and free of self-intersections.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when the mesh is edge-manifold (no boundary), vertex-manifold, and not
        self-intersecting.

    See Also
    --------
    [`watertight_face_mask`][triwarp.characteristics.watertight_face_mask]
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]
    [`is_vertex_manifold`][triwarp.characteristics.is_vertex_manifold]
    [`is_self_intersecting`][triwarp.characteristics.is_self_intersecting]
    [`is_volume`][triwarp.characteristics.is_volume]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_watertight``. For the cheaper "every edge shared
    by exactly two faces" test (trimesh semantics) use
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold] with
    ``allow_boundary_edges=False``.
    """
    return (
        is_edge_manifold(faces, allow_boundary_edges=False)
        and is_vertex_manifold(faces)
        and not is_self_intersecting(vertices, faces)
    )


def watertight_face_mask(
    faces: wp.array[wp.int32], edges_sorted: twt.Array2dInt32 | None = None
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether all three of a face's undirected edges are shared by exactly two faces.

    This is [`edge_manifold_mask`][triwarp.characteristics.edge_manifold_mask] with
    ``allow_boundary_edges=False``: a face is ``True`` only when none of its edges is a boundary
    edge (used once) or a non-manifold edge (used three or more times). The faces that break
    watertightness (trimesh's ``broken_faces``) are ``flatnonzero(~watertight_face_mask(faces))``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges forwarded to
        [`edge_manifold_mask`][triwarp.characteristics.edge_manifold_mask].

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``faces.device``. Empty for an empty mesh.

    See Also
    --------
    [`is_watertight`][triwarp.characteristics.is_watertight]
    [`edge_manifold_mask`][triwarp.characteristics.edge_manifold_mask]
    """
    return edge_manifold_mask(faces, edges_sorted=edges_sorted, allow_boundary_edges=False)


def is_volume(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> bool:
    """
    Whether the mesh is a valid closed volume with outward-facing normals.

    Follows [`trimesh.Trimesh.is_volume`][]: the mesh must be watertight (every undirected edge
    shared by exactly two faces), winding-consistent, and enclose a positive signed volume (normals
    point outward). A mesh with inward-facing normals encloses a negative signed volume and is
    reported ``False``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    bool
        ``True`` when the mesh is watertight, winding-consistent, and has outward normals (positive
        signed volume). ``False`` for an empty mesh.

    See Also
    --------
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]
    [`is_watertight`][triwarp.characteristics.is_watertight]
    [`is_orientable`][triwarp.characteristics.is_orientable]
    [`trimesh.Trimesh.is_volume`][]

    Notes
    -----
    Watertightness here is trimesh's "every undirected edge shared by exactly two faces" (each
    sorted edge forms a group of two), and winding consistency requires the two directed copies of
    every shared edge to be reversed. The signed volume is the sum of per-face signed tetrahedron
    volumes ``dot(v0, cross(v1, v2)) / 6`` measured from the origin; for a closed surface this is
    independent of the reference point and its sign encodes the normal orientation.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return False

    device = vertices.device
    edges = tw.edges.faces_to_edges(faces)
    edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)

    edge_groups = tw.grouping.group_int_rows(edges_sorted, length=2)
    n_groups = int(edge_groups.shape[0])
    if n_groups * 2 != int(edges.shape[0]):
        return False  # not watertight: some undirected edge is not shared by exactly two faces

    consistent = wp.empty(n_groups, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.edge_pair_winding_mask,
        dim=n_groups,
        inputs=[edges, edge_groups, consistent],
        device=device,
    )
    if not bool(tw.reduce.all(consistent)):
        return False

    signed_volumes = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_sample.signed_tet_volumes,
        dim=n_faces,
        inputs=[vertices, faces, wp.vec3(0.0, 0.0, 0.0), signed_volumes],
        device=device,
    )
    return tw.reduce.sum(signed_volumes) > 0.0
