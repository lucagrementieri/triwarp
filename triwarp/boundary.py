"""
Mesh boundary edges and vertices.

A mesh edge lies on the boundary when it appears exactly once among all triangle edges.
Boundary detection reuses [`group_int_rows`][triwarp.grouping.group_int_rows] (the analog of
``trimesh.grouping.group_rows(require_count=1)``), which hashes each sorted edge row and
returns the original row indices of edges occurring exactly once.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import boundary as kernel_boundary
from triwarp.kernels import scatter as kernel_scatter


def boundary_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> twt.Array2dInt32:
    """
    Undirected boundary edges (each row min-first), without preserving orientation.

    An edge belongs to the boundary when it appears exactly once among the sorted edges of
    all triangles.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first), as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges] with ``sorted=True``. Built from
        ``faces`` when ``None``.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_boundary, 2)`` sorted boundary edges on ``faces.device``. Empty ``(0, 2)``
        when the mesh has no boundary.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=faces.device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    return twt.as_array2d(
        tw.array.gather(edges_sorted, _boundary_rows(vertices, edges_sorted)), wp.int32
    )


def oriented_boundary_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> twt.Array2dInt32:
    """
    Directed boundary edges, preserving the orientation from the face winding.

    Boundary edges are detected on the sorted edges (appearing exactly once), but the
    directed ``(i, j)`` pairs are returned.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first). Built
        from ``faces`` when ``None``.
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges, as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges]. Built from ``faces`` when ``None``.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_boundary, 2)`` directed boundary edges on ``faces.device``. Empty
        ``(0, 2)`` when the mesh has no boundary.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=faces.device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    if edges is None:
        edges = tw.edges.faces_to_edges(faces)
    return twt.as_array2d(tw.array.gather(edges, _boundary_rows(vertices, edges_sorted)), wp.int32)


def boundary_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
    *,
    copy: bool = False,
) -> list[wp.array[wp.int32]]:
    """
    Ordered vertex-index loops along each mesh boundary.

    Each boundary loop is a simple cycle in the directed-boundary-edge successor graph (on a
    manifold boundary every boundary vertex has exactly one outgoing boundary edge). Loops are
    grouped via connected-component labeling, then each vertex's ordinal position within its
    loop is ranked by following the successor chain from itself to its loop's canonical start
    (the smallest vertex index in the loop) — entirely GPU-parallel, mirroring
    ``igl::boundary_loop`` (`reference/libigl/include/igl/boundary_loop.cpp`, first overload).

    All loops are found in one batched pass
    ([`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched]); this is
    [`split`][triwarp.array.split] over its packed result.

    !!! note "The returned arrays are views"
        Each loop slices the single packed buffer ``boundary_loops_batched`` produced, which costs
        no device memory and no launches. Two consequences: holding on to a single loop keeps the
        *whole* buffer alive, and writing into one loop writes into the shared allocation. Pass
        ``copy=True`` for independent buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base and
        the successor-array size).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges, forwarded to
        [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched].
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges, forwarded to
        [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched].
    copy
        Return independent buffers instead of views into the packed result.

    Returns
    -------
    list[wp.array[wp.int32]]
        One array per boundary loop, each holding the ordered vertex indices around that loop
        (not repeating the start vertex), on ``faces.device``. Empty list when the mesh has no
        boundary.

    Notes
    -----
    Assumes a **manifold** boundary: each boundary vertex lies on exactly two boundary edges. A
    vertex shared by more than one loop (a non-manifold pinch point) keeps only one outgoing
    successor edge, last write wins.

    Orientability is *not* assumed. On a non-orientable surface the winding cannot orient the
    boundary globally -- at the seam two boundary edges leave the same vertex -- so the loops are
    recovered from the undirected edges instead, and their direction is then an arbitrary but
    reproducible choice rather than the face winding's. Only the direction is arbitrary; the loops
    themselves are exact. Costs one extra host readback to detect the case.

    See Also
    --------
    [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched]
    [`boundary_loop`][triwarp.boundary.boundary_loop]
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]
    ``igl.boundary_loop_all``
    """
    flat_loops, offsets, _loop_sizes = boundary_loops_batched(vertices, faces, edges_sorted, edges)
    return tw.array.split(flat_loops, offsets, copy=copy)


def boundary_loops_batched(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every boundary loop at once, packed into one buffer plus per-loop offsets.

    Same loops, same order, same contents as
    [`boundary_loops`][triwarp.boundary.boundary_loops] — but with no per-loop Python and no
    per-loop allocation, which is the only form whose cost is independent of the loop *count*.
    Prefer it when a mesh has many small holes or when the loops feed straight into another
    batched kernel, as [`triwarp.holes`][triwarp.holes] does. The loop ordering itself is the
    general successor-graph machinery of [`successor_cycles`][triwarp.graph.successor_cycles];
    this function contributes the boundary-edge detection.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base and
        the successor-array size).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first), as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges] with ``sorted=True``. Built from
        ``faces`` when ``None``.
    edges
        Optional precomputed ``(n_faces * 3, 2)`` directed edges, as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges]. Built from ``faces`` when ``None``.

    Returns
    -------
    flat_loops
        Concatenated ordered vertex indices of every loop, on ``faces.device``.
    offsets
        Length-``n_loops`` exclusive prefix sum of the loop sizes: loop ``i`` occupies
        ``flat_loops[offsets[i] : offsets[i] + loop_sizes[i]]``. Not a total-terminated CSR
        array — the last loop ends at ``flat_loops.shape[0]``.
    loop_sizes
        Length-``n_loops`` vertex count per loop.

    See Also
    --------
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`boundary_loop`][triwarp.boundary.boundary_loop]
    [`successor_cycles`][triwarp.graph.successor_cycles]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        # Three *distinct* empty allocations, so callers may write into them independently.
        return tuple(wp.empty(0, dtype=wp.int32, device=device) for _ in range(3))

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    # One boundary detection for all three edge views below; ``boundary_edges`` /
    # ``oriented_boundary_edges`` / ``boundary_vertex_indices`` would each redo the row grouping.
    rows = _boundary_rows(vertices, edges_sorted)
    n_boundary_edges = int(rows.shape[0])
    if n_boundary_edges == 0:
        return tuple(wp.empty(0, dtype=wp.int32, device=device) for _ in range(3))

    if edges is None:
        edges = tw.edges.faces_to_edges(faces)
    directed = twt.as_array2d(tw.array.gather(edges, rows), wp.int32)

    n_vertices = int(vertices.shape[0])
    if _needs_unoriented_boundary_walk(directed, n_vertices):
        return _unoriented_boundary_cycles(
            twt.as_array2d(tw.array.gather(edges_sorted, rows), wp.int32), n_vertices
        )
    # ``validate=False``: ``directed`` holds vertex indices this function just gathered out of
    # ``faces``, so the range check would only re-derive a bound the caller already guarantees —
    # at the cost of a device synchronization.
    return tw.graph.successor_cycles(directed, n_vertices, validate=False)


def _needs_unoriented_boundary_walk(directed: twt.Array2dInt32, n_vertices: int) -> bool:
    """
    Whether the directed boundary edges fail to be a successor graph *and* an undirected walk fixes.

    They are one on every orientable surface, which is what lets
    [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched] hand them straight to
    [`successor_cycles`][triwarp.graph.successor_cycles]. Two different defects break that, and
    they need different answers, so this returns ``True`` for only one of them:

    - **A non-orientable seam.** The winding cannot be made consistent globally, so at the seam two
      boundary edges leave the same vertex and ``succ[tail] = head`` drops one silently. Measured
      on the Moebius fixture: exactly one vertex of 78 has out-degree 2, and the walk that follows
      returns 78 entries over only 40 distinct vertices. The boundary is still 2-regular, so
      [`_unoriented_boundary_cycles`][triwarp.boundary._unoriented_boundary_cycles] recovers it
      exactly -- this is the case worth taking.
    - **A pinch point**, where two loops meet at one vertex, which then has four boundary
      incidences. No 2-regular walk exists, so the undirected fallback has nothing better to offer
      -- it would have to drop neighbours too, just at a different place. The documented last-write-
      wins behaviour stands, and this returns ``False``.

    Conflating them is easy and was measured: an icosphere with every seventh face removed has
    pinch points and *no* orientability problem, and a gate reading only out-degree sends it down
    the fallback for 1.39x and no benefit.

    One 8-byte host readback. Both flags come from one pass over the boundary and one over the
    vertices, rather than two max-reductions, because at ~0.14 ms a reduction pair would be 5% of
    this function on an open mesh.
    """
    device = directed.device
    degrees = twt.empty_2d((n_vertices, 2), wp.int32, device=device)
    degrees.zero_()
    flags = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.count_boundary_degrees,
        dim=int(directed.shape[0]),
        inputs=[directed, degrees],
        device=device,
    )
    wp.launch(
        kernel_boundary.flag_boundary_degree_defects,
        dim=n_vertices,
        inputs=[degrees, flags],
        device=device,
    )
    has_seam, has_pinch = (int(flag) for flag in flags.numpy())
    return bool(has_seam and not has_pinch)


def _unoriented_boundary_cycles(
    boundary_edges: twt.Array2dInt32, n_vertices: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Boundary cycles of a mesh whose winding cannot orient them: walk the undirected edges instead.

    A manifold boundary is 2-regular whether or not the surface is orientable, so the cycles exist
    even where a *consistent direction* for them does not. The walk runs on **darts**: dart
    ``2 * v + s`` means "at vertex ``v``, arrived from neighbour slot ``s``", and its successor
    leaves by the other slot. That is a genuine successor graph by construction -- every dart has
    exactly one out-edge -- so [`successor_cycles`][triwarp.graph.successor_cycles] handles it
    unchanged, at twice the node count.

    Each undirected cycle therefore comes back **twice**, once per direction, over disjoint dart
    sets. The two mirrors share their lowest vertex ``v`` but start at darts ``2v`` and ``2v + 1``,
    so keeping the even-starting one picks exactly one per pair. With the neighbour slots sorted
    ascending, that direction is "leave the lowest vertex toward its larger neighbour" -- an
    arbitrary but *reproducible* choice, which is the honest answer when no winding defines one.

    On the Moebius fixture this returns the single 78-vertex cycle the surface actually has: all 78
    boundary vertices have boundary-degree 2, and the walk closes with every consecutive pair a
    real boundary edge. Two references get it wrong in different ways and neither is worth
    matching -- ``igl.boundary_loop_all`` cuts that cycle into ``1 + 39 + 38`` open chains (each
    has exactly one consecutive pair that is *not* a boundary edge) and ``boundary_loop`` reports
    the longest of them as 39, while a half-edge hole ring walks the band's *double* cover and reads
    156.

    The mirror filter and the re-pack run on the host, over a buffer bounded by the **boundary**
    rather than by the mesh, and only ever on a non-orientable surface.
    """
    device = boundary_edges.device
    neighbors = twt.empty_2d((n_vertices, 2), wp.int32, device=device)
    neighbors.fill_(-1)
    slot_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.scatter_boundary_neighbors,
        dim=int(boundary_edges.shape[0]),
        inputs=[boundary_edges, slot_count, neighbors],
        device=device,
    )
    wp.launch(
        kernel_boundary.sort_boundary_neighbor_slots,
        dim=n_vertices,
        inputs=[neighbors],
        device=device,
    )

    boundary_vertices = tw.grouping.unique_1d(boundary_edges.flatten())
    n_boundary = int(boundary_vertices.shape[0])
    dart_edges = twt.empty_2d((2 * n_boundary, 2), wp.int32, device=device)
    wp.launch(
        kernel_boundary.build_dart_successors,
        dim=(n_boundary, 2),
        inputs=[boundary_vertices, neighbors, dart_edges],
        device=device,
    )

    flat_darts, dart_offsets, _dart_sizes = tw.graph.successor_cycles(
        dart_edges, 2 * n_vertices, validate=False
    )
    darts_np = flat_darts.numpy()
    bounds_np = np.append(dart_offsets.numpy(), darts_np.shape[0])
    loops_np = [
        darts_np[start:stop] // 2
        for start, stop in itertools.pairwise(bounds_np)
        if darts_np[start] % 2 == 0
    ]
    if not loops_np:
        return tuple(wp.empty(0, dtype=wp.int32, device=device) for _ in range(3))

    sizes_np = np.array([loop.shape[0] for loop in loops_np], dtype=np.int32)
    return (
        wp.array(np.concatenate(loops_np), dtype=wp.int32, device=device),
        wp.array(np.concatenate([[0], np.cumsum(sizes_np)[:-1]]), dtype=wp.int32, device=device),
        wp.array(sizes_np, dtype=wp.int32, device=device),
    )


def loop_perimeters(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]]
) -> wp.array[wp.float32]:
    """
    Perimeter of every closed loop, in one launch.

    Takes what [`boundary_loops`][triwarp.boundary.boundary_loops] returns -- a list of vertex-index
    cycles whose last entry joins back to the first -- and measures them all together, so the cost
    is one launch and one packing pass rather than one call per loop. Equivalent to
    [`polyline_length`][triwarp.polyline.polyline_length] with ``closed=True`` on each loop's
    gathered positions, and the segmented form is why the fill's ``preserve_largest_hole`` can rank
    every rim for the price of one readback.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    loops
        Closed vertex-index cycles. An empty sequence gives an empty result; a loop of fewer than
        two entries measures ``0``.

    Returns
    -------
    wp.array[wp.float32]
        One perimeter per loop, in the order given, on ``vertices.device``.

    Raises
    ------
    ValueError
        If any loop is not a rank-1 ``wp.int32`` array.

    See Also
    --------
    [`loop_directed_areas`][triwarp.boundary.loop_directed_areas]
        The vector measure of the same loops: norm is the spanned area, direction is its normal.
    [`boundary_loops`][triwarp.boundary.boundary_loops]
        Produces the loops.
    [`polyline_length`][triwarp.polyline.polyline_length]
        The single-loop form, over positions rather than indices.
    """
    packed = _pack_loop_segments(vertices, loops, caller="loop_perimeters")
    if packed is None:
        return wp.empty(0, dtype=wp.float32, device=vertices.device)
    flat_loops, loop_id, starts, sizes, n_loops = packed
    perimeters = wp.zeros(n_loops, dtype=wp.float32, device=vertices.device)
    wp.launch(
        kernel_boundary.loop_perimeters,
        dim=int(flat_loops.shape[0]),
        inputs=[flat_loops, loop_id, starts, sizes, vertices, perimeters],
        device=vertices.device,
    )
    return perimeters


def loop_directed_areas(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]]
) -> wp.array[wp.vec3]:
    """
    Directed area vector of every loop, in one launch.

    Half the sum of ``p_i x p_{i+1}`` around each cycle. Its **norm** is the area of the planar
    polygon the loop spans and its **direction** is that polygon's normal, oriented by the loop's
    own winding -- so it answers "how big is this hole" and "which way does it face" at once, and
    the sign flips if the loop is reversed. A vector rather than a scalar for that reason: the area
    alone loses the orientation, which is what tells an outer boundary from an inner one.

    Origin-independent, because the cross products of a closed ring cancel any shift of the origin,
    so no centroid pass is needed. Exact for a planar loop; for a non-planar one it is the area of
    the loop's projection onto the plane normal to the result, which is the standard convention.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    loops
        Closed vertex-index cycles, as [`boundary_loops`][triwarp.boundary.boundary_loops] returns.

    Returns
    -------
    wp.array[wp.vec3]
        One directed area per loop, in the order given, on ``vertices.device``.

    Raises
    ------
    ValueError
        If any loop is not a rank-1 ``wp.int32`` array.

    See Also
    --------
    [`loop_perimeters`][triwarp.boundary.loop_perimeters]
        The scalar measure of the same loops.
    [`polyline_normal`][triwarp.polyline.polyline_normal]
        The single-loop direction, normalized and over positions.
    """
    packed = _pack_loop_segments(vertices, loops, caller="loop_directed_areas")
    if packed is None:
        return wp.empty(0, dtype=wp.vec3, device=vertices.device)
    flat_loops, loop_id, starts, sizes, n_loops = packed
    directed_areas = wp.zeros(n_loops, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_boundary.loop_directed_areas,
        dim=int(flat_loops.shape[0]),
        inputs=[flat_loops, loop_id, starts, sizes, vertices, directed_areas],
        device=vertices.device,
    )
    return directed_areas


def _pack_loop_segments(
    vertices: wp.array[wp.vec3], loops: Sequence[wp.array[wp.int32]], *, caller: str
) -> (
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32], int]
    | None
):
    """
    Pack a list of cycles into the four arrays a segmented per-loop kernel needs.

    ``loop_id`` inverts ``starts`` so a ``dim=total`` launch finds its own loop without a search,
    which is what lets both measures above run in one launch over every loop at once. ``None`` when
    there is nothing to measure.
    """
    device = vertices.device
    loops = list(loops)
    for loop in loops:
        if len(loop.shape) != 1 or loop.dtype is not wp.int32:
            raise ValueError(f"{caller}: every loop must be a rank-1 wp.int32 array")
    if not loops or all(int(loop.shape[0]) == 0 for loop in loops):
        return None
    flat_loops, starts = tw.array.pack_1d_arrays(loops)
    sizes_np = np.array([int(loop.shape[0]) for loop in loops], dtype=np.int32)
    loop_id = wp.array(
        np.repeat(np.arange(len(loops), dtype=np.int32), sizes_np), dtype=wp.int32, device=device
    )
    sizes = wp.array(sizes_np, dtype=wp.int32, device=device)
    return flat_loops, loop_id, starts, sizes, len(loops)


def boundary_loop(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.int32]:
    """
    Ordered vertex-index loop along the longest mesh boundary.

    Parameters
    ----------
    vertices, faces, edges_sorted, edges
        Forwarded to [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched].

    Returns
    -------
    wp.array[wp.int32]
        Ordered vertex indices around the longest boundary loop, on ``faces.device``. An
        independent buffer, not a view into the packed result. Empty when the mesh has no
        boundary.

    See Also
    --------
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched]
    ``igl.boundary_loop``
    """
    flat_loops, offsets, _loop_sizes = boundary_loops_batched(vertices, faces, edges_sorted, edges)
    loops = tw.array.split(flat_loops, offsets)
    if not loops:
        return wp.empty(0, dtype=wp.int32, device=faces.device)
    # Only the winner is materialized: the other loops stay views into the packed buffer.
    return wp.clone(max(loops, key=lambda loop: int(loop.shape[0])))


def boundary_vertex_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.int32]:
    """
    Sorted unique vertex indices lying on the mesh boundary.

    A vertex belongs to the boundary when it belongs to at least one boundary edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions; only the count is used (as the row-hash base).
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges, forwarded to
        [`boundary_edges`][triwarp.boundary.boundary_edges].

    Returns
    -------
    wp.array[wp.int32]
        Sorted unique boundary vertex indices on ``faces.device``. Empty when the mesh has
        no boundary.
    """
    edges = boundary_edges(vertices, faces, edges_sorted)
    return tw.grouping.unique_1d(edges.flatten())


def boundary_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.vec3]:
    """
    Coordinates of the vertices lying on the mesh boundary.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges, forwarded to
        [`boundary_vertex_indices`][triwarp.boundary.boundary_vertex_indices].

    Returns
    -------
    wp.array[wp.vec3]
        Shape ``(n_boundary_vertices,)`` boundary vertex positions on ``vertices.device``,
        ordered by ascending vertex index. Empty when the mesh has no boundary.
    """
    indices = boundary_vertex_indices(vertices, faces, edges_sorted)
    return tw.array.gather(vertices, indices)


def ears(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Find ear faces (triangles with exactly two boundary edges).

    For each ear face, ``ear_opp`` is the local index of the non-boundary edge, where local edge
    ``i`` is ``(faces[f, i], faces[f, (i + 1) % 3])``.

    !!! note "``igl::ears`` numbers the edge differently"

        The same quantity in ``igl.ears`` is indexed *opposite-vertex* style -- edge ``i`` is the
        one facing vertex ``i``, ``(faces[f, (i + 1) % 3], faces[f, (i + 2) % 3])`` -- because it
        reads its mask from ``igl::on_boundary``, whose columns are documented that way. The two
        agree on *which* faces are ears and differ on the index by a cyclic shift:
        ``triwarp_opp == (igl_opp + 1) % 3``. See
        ``tests/test_boundary.py::test_ears_match_igl``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first), as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges] with ``sorted=True``. Built from
        ``faces`` when ``None``.
    n_vertices
        Total number of vertices (used as the row-hash base). When ``None``, inferred from
        ``edges_sorted`` with a device-host sync.

    Returns
    -------
    ear : wp.array[wp.int32]
        Face indices of ear triangles on ``faces.device``. Empty when no ears exist.
    ear_opp : wp.array[wp.int32]
        Local edge index of the interior edge for each ear face, same length as ``ear``.

    See Also
    --------
    ``igl.ears``
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    empty = wp.empty(0, dtype=wp.int32, device=device)
    if n_faces == 0:
        return empty, empty

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)

    if n_vertices is None:
        n_vertices = tw.vertices.n_vertices(edges_sorted)

    boundary_rows = tw.grouping.group_int_rows(
        edges_sorted, 1, n_vertices, validate=False
    ).flatten()
    edge_boundary = wp.zeros(n_faces * 3, dtype=wp.bool, device=device)
    n_boundary_rows = int(boundary_rows.shape[0])
    if n_boundary_rows > 0:
        wp.launch(
            kernel_scatter.mark_membership_mask,
            dim=n_boundary_rows,
            inputs=[boundary_rows, wp.int32(n_faces * 3), edge_boundary],
            device=device,
        )

    out_ear = wp.empty(n_faces, dtype=wp.int32, device=device)
    out_ear_opp = wp.empty(n_faces, dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_boundary.find_ears,
        dim=n_faces,
        inputs=[edge_boundary, out_ear, out_ear_opp, counter],
        device=device,
    )

    n_ears, (ear, ear_opp) = tw.array.trim_to_count(counter, out_ear, out_ear_opp)
    if n_ears == 0:
        return empty, empty
    return ear, ear_opp


def _boundary_rows(
    vertices: wp.array[wp.vec3], edges_sorted: twt.Array2dInt32
) -> wp.array[wp.int32]:
    """Row indices of the triangle edges appearing exactly once — the boundary edges."""
    return tw.grouping.group_int_rows(
        edges_sorted, 1, int(vertices.shape[0]), validate=False
    ).flatten()
