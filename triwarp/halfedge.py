"""
Implicit halfedge connectivity: edge twins and counter-clockwise vertex one-rings.

There is no halfedge *structure* here, only an indexing convention over the flat face buffer every
other module already uses. Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to
``faces[3f + (k + 1) % 3]``, so ``next`` and ``prev`` are index arithmetic and the halfedges of a
face are consecutive. Exactly two arrays are needed to navigate a mesh:
[`halfedge_twins`][triwarp.halfedge.halfedge_twins] to cross an edge, and
[`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] to rotate around a vertex.

This is the ordering the tangent-space machinery in [`triwarp.tangent_space`][triwarp.tangent_space]
is built on: a rotational order of the outgoing halfedges at a vertex is what turns per-corner
angles into a polar coordinate system on the tangent plane.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
from triwarp.constants import INT32_MAX
from triwarp.kernels import halfedge as kernel_halfedge


def halfedge_twins(faces: wp.array[wp.int32], n_vertices: int | None = None) -> wp.array[wp.int32]:
    """
    Opposite halfedge of every halfedge, or ``-1`` on a boundary.

    Halfedges ``h = 3 * f + k`` and ``twins[h]`` traverse the same undirected edge in opposite
    directions, so ``twins[twins[h]] == h`` wherever ``twins[h] != -1``. The halfedges left at
    ``-1`` are exactly the mesh boundary, in the orientation
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges] reports.

    Undirected edges are matched by packing each sorted endpoint pair into one key
    ([`hash_indices_rows`][triwarp.grouping.hash_indices_rows]), radix-sorting the keys with the
    halfedge index as payload, and pairing up the runs of equal keys — the same mechanism behind
    [`edges_unique`][triwarp.edges.edges_unique] and
    [`face_adjacency`][triwarp.adjacency.face_adjacency], without materializing the unique edge
    list.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    n_vertices
        Total vertex count, used as the key radix. When ``None`` it is inferred with
        [`n_vertices`][triwarp.vertices.n_vertices], which costs a host readback.

    Returns
    -------
    wp.array[wp.int32]
        Length ``3 * n_faces`` on ``faces.device``; ``-1`` for boundary halfedges.

    Raises
    ------
    ValueError
        If an undirected edge carries three or more halfedges (an edge-non-manifold mesh, where
        "the" opposite halfedge is not defined). Detecting this needs one 4-byte readback, so this
        function always synchronizes once.

    See Also
    --------
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]
    """
    device = faces.device
    n_halfedges = int(faces.shape[0]) // 3 * 3
    twins = wp.full(n_halfedges, -1, dtype=wp.int32, device=device)
    if n_halfedges == 0:
        return twins

    if n_vertices is None:
        n_vertices = tw.vertices.n_vertices(faces)

    # Edge rows are built from face indices, so they are non-negative and below the vertex count by
    # construction: the range check would only add a readback.
    edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
    sorted_keys, order = tw.array.sort_pairs(keys)

    nonmanifold = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_halfedge.pair_sorted_halfedges,
        dim=n_halfedges,
        inputs=[sorted_keys, order, twins, nonmanifold],
        device=device,
    )
    n_nonmanifold = int(nonmanifold.numpy()[0])
    if n_nonmanifold > 0:
        raise ValueError(
            f"halfedge_twins requires an edge-manifold mesh: {n_nonmanifold} edge(s) are shared by "
            f"three or more faces."
        )
    return twins


def vertex_one_rings(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32] | None = None,
    n_vertices: int | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Outgoing halfedges of every vertex in counter-clockwise order, as a CSR buffer.

    Vertex ``v`` owns ``ring_halfedges[offsets[v] : offsets[v + 1]]``, each entry a halfedge leaving
    ``v``; the ring size is therefore the number of incident *faces*, one less than the number of
    adjacent vertices at a boundary vertex. Consecutive entries are consecutive around ``v`` in the
    direction the face orientation calls counter-clockwise, because the rotation ``h ->
    twins[prev(h)]`` steps from edge ``v -> a`` to edge ``v -> b`` inside the CCW-oriented face
    ``(v, a, b)``.

    A boundary vertex starts its ring at its outgoing boundary halfedge (the clockwise-most edge of
    its fan) so the whole fan is enumerated; interior vertices start at their lowest-indexed
    outgoing halfedge, which makes the rotational *order* canonical but its starting point
    arbitrary. Isolated vertices get an empty row.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins]. When ``None`` it
        is computed here.
    n_vertices
        Total vertex count (the number of CSR rows). When ``None`` it is inferred with
        [`n_vertices`][triwarp.vertices.n_vertices], which costs a host readback.

    Returns
    -------
    offsets : wp.array[wp.int32]
        Length ``n_vertices + 1`` CSR row starts; ``offsets[-1] == 3 * n_faces``.
    ring_halfedges : wp.array[wp.int32]
        Length ``3 * n_faces`` outgoing halfedges, grouped and ordered per vertex.
    is_boundary : wp.array[wp.bool]
        Length ``n_vertices``; ``True`` where the vertex is incident to a boundary edge.

    Raises
    ------
    ValueError
        If a vertex's rotation closes before its whole fan is covered — a pinched,
        vertex-non-manifold vertex where two fans meet at a single index. Detecting this needs one
        4-byte readback, so this function always synchronizes once.

    See Also
    --------
    [`halfedge_twins`][triwarp.halfedge.halfedge_twins]
    [`halfedge_tangent_angles`][triwarp.tangent_space.halfedge_tangent_angles]
    """
    device = faces.device
    n_halfedges = int(faces.shape[0]) // 3 * 3

    if n_vertices is None:
        n_vertices = tw.vertices.n_vertices(faces)
    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)

    offsets = wp.zeros(n_vertices + 1, dtype=wp.int32, device=device)
    ring_halfedges = wp.full(n_halfedges, -1, dtype=wp.int32, device=device)
    is_boundary = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    if n_halfedges == 0 or n_vertices == 0:
        return offsets, ring_halfedges, is_boundary

    counts = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_halfedge.count_outgoing_halfedges,
        dim=n_halfedges,
        inputs=[faces, counts],
        device=device,
    )
    # Inclusive scan into offsets[1:] leaves the leading zero in place, giving the usual CSR bounds.
    wp.utils.array_scan(counts, out_array=offsets[1:], inclusive=True)

    interior_start = wp.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
    boundary_start = wp.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
    wp.launch(
        kernel_halfedge.ring_start_halfedges,
        dim=n_halfedges,
        inputs=[faces, twins, interior_start, boundary_start],
        device=device,
    )
    starts = wp.empty(n_vertices, dtype=wp.int32, device=device)
    wp.map(kernel_halfedge.select_ring_start, boundary_start, interior_start, out=starts)
    wp.map(kernel_halfedge.has_boundary_halfedge, boundary_start, out=is_boundary)

    incomplete = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_halfedge.write_one_rings,
        dim=n_vertices,
        inputs=[starts, twins, offsets, ring_halfedges, incomplete],
        device=device,
    )
    n_incomplete = int(incomplete.numpy()[0])
    if n_incomplete > 0:
        raise ValueError(
            f"vertex_one_rings requires a vertex-manifold mesh: {n_incomplete} vertex/vertices "
            f"have more than one fan of faces (a pinch point)."
        )
    return offsets, ring_halfedges, is_boundary
