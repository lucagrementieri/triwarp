"""
Isocontours of a scalar field on a triangle mesh (marching triangles).

The device work is one pass per face: a face whose vertex values straddle the isovalue contributes
exactly one segment, whose endpoints are linear crossings along the two edges incident to the vertex
that is alone in sign. Segments are then linked into curves through the unique-edge id each endpoint
sits on — every crossed interior edge is shared by exactly two faces, so the segments form chains
without any geometric search.

The output convention matches [`triwarp.polyline`][triwarp.polyline]: one array of points per curve,
closed curves not repeating their first point, and a parallel list of closed flags.
"""

from __future__ import annotations

import itertools

import numpy as np
import warp as wp

import triwarp as tw
from triwarp.kernels import contour as kernel_contour


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
        kernel_contour.marching_triangles_segments,
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
