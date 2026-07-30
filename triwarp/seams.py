"""
Feature edges and the topological cut along them.

Two operations that are only useful together. [`crease_edges`][triwarp.seams.crease_edges] finds the
edges where the surface bends sharply — the ones a modeller would call hard edges — and
[`cut_along_edges`][triwarp.seams.cut_along_edges] *splits* the mesh along a given edge set,
duplicating vertices so that the two sides stop sharing them.

The cut is the operation that was missing. Marking edges is a threshold; separating the two sides of
a marked edge means rebuilding the vertex array, because the identity of a vertex is exactly what
has to change. Three things need it:

- **Hard normals.** A crease vertex shared by both sides averages a normal that belongs to neither.
- **UV seams.** A parametrization cannot be continuous across a closed surface, so a disk cut is a
  precondition rather than a nicety — [`lscm`][triwarp.parametrization.lscm] and
  [`harmonic`][triwarp.parametrization.harmonic] both require the caller to supply a mesh that
  already has a boundary.
- **Part separation.** Cutting every crease of an assembly and then splitting components
  ([`split`][triwarp.combine.split]) recovers the pieces.

Nothing here moves a vertex: [`cut_along_edges`][triwarp.seams.cut_along_edges] changes only which
vertex indices the faces name, so the surface is geometrically identical and topologically opened.
"""

from __future__ import annotations

import math

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import seams as kernel_seams


def crease_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    angle: float = 30.0,
    *,
    include_boundary: bool = False,
) -> twt.Array2dInt32:
    """
    Edges where the surface bends by more than ``angle``.

    MeshLab's ``compute_selection_crease_per_edge``, returned as an explicit edge list rather than a
    selection so it feeds straight into [`cut_along_edges`][triwarp.seams.cut_along_edges].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**. An edge is a crease when its two faces meet at *strictly*
        more than this, so ``0`` selects every interior edge whose faces are not exactly coplanar
        (which excludes the diagonals of a flat quad, and is usually what a caller wanting "all of
        them" means) and ``180`` selects none. Must be in ``[0, 180]``.
    include_boundary
        When ``True``, boundary edges are included in the result. A boundary edge has no dihedral
        angle at all, so it is neither a crease nor not one — but it *is* already a seam, and a
        caller building a cut set usually wants it. Defaults to ``False``, which is the pure
        dihedral answer.

    Returns
    -------
    twt.Array2dInt32
        ``(k, 2)`` vertex-index pairs on ``faces.device``, one row per selected edge. Row order
        follows [`face_adjacency`][triwarp.adjacency.face_adjacency] and is not sorted.

    Raises
    ------
    ValueError
        If ``angle`` is outside ``[0, 180]``.

    See Also
    --------
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
    [`triwarp.adjacency.face_adjacency_angles`][triwarp.adjacency.face_adjacency_angles]
    [`triwarp.boundary.boundary_edges`][triwarp.boundary.boundary_edges]
    """
    if not 0.0 <= angle <= 180.0:
        raise ValueError(f"angle must be in [0, 180] degrees, got {angle}")

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_int32_2d((0, 2), device=device)

    adjacency, adjacency_edges = tw.adjacency.face_adjacency(faces, return_edges=True)
    selected = twt.empty_int32_2d((0, 2), device=device)
    if int(adjacency.shape[0]) > 0:
        angles = tw.adjacency.face_adjacency_angles(vertices, faces, face_adjacency=adjacency)
        mask = wp.empty(int(angles.shape[0]), dtype=wp.bool, device=device)
        wp.launch(
            kernel_seams.crease_edge_mask,
            dim=int(angles.shape[0]),
            inputs=[angles, wp.float32(math.radians(angle)), mask],
            device=device,
        )
        selected = tw.array.gather(adjacency_edges, tw.array.flatnonzero(mask))

    if not include_boundary:
        return twt.as_array2d_int32(selected)
    boundary = tw.boundary.boundary_edges(vertices, faces)
    if int(boundary.shape[0]) == 0:
        return twt.as_array2d_int32(selected)
    if int(selected.shape[0]) == 0:
        return twt.as_array2d_int32(boundary)
    return twt.as_array2d_int32(tw.array.concatenate([selected, boundary]))


def cut_along_edges(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edges: twt.Array2dInt32
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Split the mesh along an edge set, duplicating vertices so the two sides no longer share them.

    Each *corner* (a face-vertex incidence) becomes a node, two corners at the same vertex are
    joined when the mesh edge between them is **not** in ``edges``, and each connected component of
    that graph becomes one output vertex. So a vertex whose whole fan is uncut survives as one
    vertex, a vertex crossed by a single marked edge on an open fan survives as one (the fan is
    still connected the long way round), and a vertex crossed by two survives as two. That is the
    correct local rule, and it is why the answer cannot be computed edge by edge.

    MeshLab's ``meshing_cut_along_crease_edges``. Geometrically a no-op — every output vertex sits
    exactly where its input did — and topologically the operation that turns a marked edge set into
    a boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer. Must be **edge-manifold**:
        the cut is defined through halfedge twins, and an edge with three faces has no well-defined
        "other side" (see [`halfedge_twins`][triwarp.halfedge.halfedge_twins]).
    edges
        ``(k, 2)`` vertex-index pairs to cut along, in either order per row. Rows that are not mesh
        edges are ignored, and boundary edges are already cuts so marking them changes nothing.
        Get a crease set from [`crease_edges`][triwarp.seams.crease_edges].

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Positions of the cut mesh, one per surviving corner component. Longer than the input's
        wherever a vertex was split, and **shorter** when the input had unreferenced vertices — only
        corners produce output vertices, so an unused vertex disappears.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer over the new vertices. Same length, same winding
        and same face order as the input; only the indices change.

    Raises
    ------
    ValueError
        If ``edges`` is not a rank-2 ``int32`` array with two columns, or ``faces`` is not
        edge-manifold.

    See Also
    --------
    [`crease_edges`][triwarp.seams.crease_edges]
    [`triwarp.combine.split`][triwarp.combine.split]
    [`triwarp.repair.remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]

    Notes
    -----
    The inverse operation is
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]: welding coincident
    positions back together closes every cut this makes, which is a useful round-trip check and a
    warning — a cut mesh must not be passed through a position-based weld if the seams are meant to
    survive.
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (k, 2), got {edges.shape}")

    device = faces.device
    n_halfedges = int(faces.shape[0]) // 3 * 3
    if n_halfedges == 0:
        return wp.clone(vertices), wp.clone(faces)

    n_vertices = int(vertices.shape[0])
    twins = tw.halfedge.halfedge_twins(faces, n_vertices=n_vertices)

    # Marked edges as a sorted key set, so the kernel tests membership with a binary search rather
    # than a per-halfedge scan. Keys match ``pack_edge_key``, which is what the kernel builds.
    marked_keys = wp.empty(0, dtype=wp.uint64, device=device)
    if int(edges.shape[0]) > 0:
        keys = tw.grouping.hash_indices_rows(edges, max_index=n_vertices)
        marked_keys, _order = tw.array.sort_pairs(keys)

    union_edges = twt.empty_int32_2d((n_halfedges, 2), device=device)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_seams.corner_union_edges,
        dim=n_halfedges,
        inputs=[faces, twins, marked_keys, wp.uint64(n_vertices), union_edges, count],
        device=device,
    )
    _n_union, (graph_edges,) = tw.array.trim_to_count(count, union_edges)

    # Components over the corners. ``validate=False``: both endpoints are halfedge indices this
    # kernel just produced, so they are in range by construction and the check would only add a
    # sync.
    labels = tw.graph.connected_component_labels_from_edges(
        twt.as_array2d_int32(graph_edges), node_count=n_halfedges, validate=False
    )
    unique_labels, corner_index = tw.grouping.unique_1d(labels, return_inverse=True)

    out_vertices = wp.empty(int(unique_labels.shape[0]), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_seams.scatter_corner_values,
        dim=n_halfedges,
        inputs=[faces, corner_index, vertices, out_vertices],
        device=device,
    )
    # ``corner_index`` *is* the new face buffer: corner ``h`` of the flat layout is entry ``h``.
    return out_vertices, wp.clone(corner_index)
