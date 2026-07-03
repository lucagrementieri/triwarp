from __future__ import annotations

from typing import cast

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import characteristics as kernel_characteristics
from triwarp.kernels import intersections as kernel_intersections


def _n_vertices(faces: wp.array[wp.int32]) -> int:
    """Vertex count inferred as ``max(faces) + 1`` (matches libigl ``F.maxCoeff() + 1``)."""
    if int(faces.shape[0]) == 0:
        return 0
    return int(faces.numpy().max()) + 1


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
    keys = tw.unique.hash_indices_rows(edges_sorted, max_index=_n_vertices(faces))
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
    keys = tw.unique.hash_indices_rows(edges_sorted, max_index=_n_vertices(faces))
    _, inverse, counts = tw.unique.unique_1d(keys, return_inverse=True, return_counts=True)

    n_unique = int(counts.shape[0])
    edge_ok = wp.empty(n_unique, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.edge_manifold_mask,
        dim=n_unique,
        inputs=[counts, allow_boundary_edges, edge_ok],
        device=device,
    )

    out_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_characteristics.face_edge_manifold_mask,
        dim=n_faces,
        inputs=[inverse, edge_ok, out_mask],
        device=device,
    )
    return out_mask


def _vertex_manifold_mask(faces: wp.array[wp.int32], n_vertices: int) -> wp.array[wp.bool]:
    """Per-vertex manifold flag over ``n_vertices`` vertices (shared by the public helpers)."""
    device = faces.device
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
    return _vertex_manifold_mask(faces, int(vertices.shape[0]))


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

    Notes
    -----
    Corner adjacency is derived from
    [`face_adjacency`][triwarp.graph.face_adjacency], which pairs faces across edges shared by
    exactly two faces; on edge-non-manifold meshes (an edge shared by three or more faces) read the
    result together with [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]. Equivalent
    to libigl ``is_vertex_manifold``.

    See Also
    --------
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]
    [`is_watertight`][triwarp.characteristics.is_watertight]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    manifold = _vertex_manifold_mask(faces, _n_vertices(faces))
    return bool(tw.reduce.all(manifold))


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
    [`mesh_with_mesh`][triwarp.intersections.mesh_with_mesh]
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
    [`is_edge_manifold`][triwarp.characteristics.is_edge_manifold]
    [`is_vertex_manifold`][triwarp.characteristics.is_vertex_manifold]
    [`is_self_intersecting`][triwarp.characteristics.is_self_intersecting]
    [`is_watertight`][triwarp.graph.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_watertight``. For the cheaper "every edge shared
    by exactly two faces" test (trimesh semantics) use
    [`is_watertight`][triwarp.graph.is_watertight].
    """
    return (
        is_edge_manifold(faces, allow_boundary_edges=False)
        and is_vertex_manifold(faces)
        and not is_self_intersecting(vertices, faces)
    )


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
    [`is_watertight`][triwarp.characteristics.is_watertight]
    [`is_watertight`][triwarp.graph.is_watertight]

    Notes
    -----
    Equivalent to ``open3d.geometry.TriangleMesh.is_orientable``. Unlike
    [`is_watertight`][triwarp.graph.is_watertight]'s winding flag, orientability allows individual
    faces to be flipped, so a consistently-orientable mesh with mixed winding still returns
    ``True``.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return True

    device = faces.device
    adjacency, adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    m = int(adjacency.shape[0])

    labels = tw.graph.connected_component_labels_from_edges(adjacency, node_count=n_faces)
    orient = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_characteristics.seed_orientation, dim=n_faces, inputs=[labels, orient], device=device
    )

    if m == 0:
        return True

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

    conflict = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_characteristics.verify_orientation,
        dim=m,
        inputs=[signed_edges, signs, orient, conflict],
        device=device,
    )
    return conflict.numpy().item() == 0


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
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=_n_vertices(faces))
    n_edges = int(unique_edges.shape[0])
    return n_referenced - n_edges + n_faces
