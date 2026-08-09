"""
Mesh boundary edges and vertices.

A mesh edge lies on the boundary when it appears exactly once among all triangle edges.
Boundary detection reuses [`group_int_rows`][triwarp.grouping.group_int_rows] (the analog of
``trimesh.grouping.group_rows(require_count=1)``), which hashes each sorted edge row and
returns the original row indices of edges occurring exactly once.
"""

from __future__ import annotations

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
    Assumes a manifold boundary: each boundary vertex has exactly one outgoing and one incoming
    boundary edge. A vertex shared by more than one loop (non-manifold pinch point) keeps only
    one outgoing successor edge (last write wins).

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

    # ``validate=False``: ``directed`` holds vertex indices this function just gathered out of
    # ``faces``, so the range check would only re-derive a bound the caller already guarantees —
    # at the cost of a device synchronization.
    return tw.graph.successor_cycles(directed, int(vertices.shape[0]), validate=False)


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
