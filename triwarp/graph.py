from __future__ import annotations

import warp as wp
from typing import overload, Literal
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import array as kernel_array
import triwarp.typing as twt
import triwarp as tw


def faces_to_edges(faces: wp.array[wp.int32], sorted: bool = False) -> twt.Array2dInt32:
    """
    Directed triangle edges from a flat ``(i0, i1, i2)`` index buffer.

    For each face, emits the three directed edges ``(i0, i1)``, ``(i1, i2)``, and ``(i2, i0)``
    in row-major order, matching :func:`trimesh.geometry.faces_to_edges` on the same ``faces``
    layout. Runs on ``faces.device`` with one launched thread per face (``int32`` indices).

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer: consecutive triples
        ``(i0, i1, i2), (i0, i1, i2), ...`` of vertex indices, the same convention as
        :mod:`triwarp.triangles`.
    sorted
        If ``True``, sort the edges by the minimum vertex index first.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_faces * 3, 2)`` with rows ``edges[3*f + 0] = (i0, i1)``,
        ``edges[3*f + 1] = (i1, i2)``, ``edges[3*f + 2] = (i2, i0)`` for face ``f``.
        If ``n_faces == 0``, returns an empty ``(0, 2)`` array.

    Raises
    ------
    ValueError
        If ``faces.shape[0]`` is not divisible by ``3``.

    See Also
    --------
    :func:`trimesh.geometry.faces_to_edges`
    """
    n = int(faces.shape[0])
    if n % 3 != 0:
        raise ValueError(f"faces length must be divisible by 3, got {n}")
    n_faces = n // 3
    edges = twt.empty_int32_2d((n_faces * 3, 2), device=faces.device)
    wp.launch(
        kernel_graph.faces_to_edges_sorted if sorted else kernel_graph.faces_to_edges,
        dim=n_faces,
        inputs=[faces, edges],
        device=faces.device,
    )
    return twt.as_array2d_int32(edges)


@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: Literal[False] = False,
) -> twt.Array2dInt32: ...
@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: Literal[True],
) -> tuple[twt.Array2dInt32, twt.Array2dInt32]: ...
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: bool = False,
) -> twt.Array2dInt32 | tuple[twt.Array2dInt32, twt.Array2dInt32]:
    """
    Face index pairs that share an undirected mesh edge.

    Each output row lists two face indices whose triangles share an edge (vertex
    pair). On a closed manifold mesh every interior edge appears exactly twice in
    the edge list, so only edges with duplicate sorted rows are kept—boundary edges
    that appear once are omitted.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of triangle vertex indices, the
        same flat layout as :mod:`triwarp.triangles` and :func:`faces_to_edges`.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` edge rows with each row sorted
        so the smaller vertex index is first (as from :func:`faces_to_edges` with
        ``sorted=True``). When ``None``, edges are built from ``faces`` on
        ``faces.device``.
    return_edges
        If ``True``, also return the shared vertex indices for each adjacency row.

    Returns
    -------
    twt.Array2dInt32 or tuple of two such arrays
        **adjacency** — shape ``(m, 2)`` on ``faces.device``. Row ``k`` gives face
        indices ``(f0, f1)`` with ``f0 <= f1`` (rows sorted in-place). Faces
        ``faces[3*f0:3*f0+3]`` and ``faces[3*f1:3*f1+3]`` share an edge.

        When ``return_edges`` is ``True``, also returns **adjacency_edges** —
        shape ``(m, 2)`` with the sorted vertex pair for that shared edge (one row
        per adjacency pair, taken from the first matching edge row).

    Notes
    -----
    Duplicate-edge grouping uses :func:`triwarp.grouping.group_int_rows` with
    ``length=2``, equivalent to :func:`trimesh.grouping.group_rows` with
    ``require_count=2``. An empty mesh yields shape ``(0, 2)``.

    Examples
    --------
    Face-connected components (with NetworkX on CPU after ``.numpy()``):

    .. code-block:: python

        import networkx as nx

        adj = tw.graph.face_adjacency(faces_wp).numpy()
        graph = nx.Graph()
        graph.add_edges_from(adj)
        groups = nx.connected_components(graph)

    See Also
    --------
    :func:`faces_to_edges`
    :func:`trimesh.graph.face_adjacency`
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_array = twt.empty_int32_2d((0, 2), device=faces.device)
        if return_edges:
            return empty_array, twt.empty_int32_2d((0, 2), device=faces.device)
        return empty_array
    if edges_sorted is None:
        edges_sorted = faces_to_edges(faces, sorted=True)
    edges_face = wp.array([f for f in range(n_faces) for _ in range(3)], dtype=wp.int32, device=faces.device)
    edge_groups = tw.grouping.group_int_rows(edges_sorted, length=2, max_value=n_faces)
    adjacency = twt.empty_int32_2d((edge_groups.shape[0], 2), device=faces.device)
    wp.launch(
        kernel_array.gather_2d_from_1d,
        dim=edge_groups.shape,
        inputs=[edges_face, edge_groups, adjacency],
        device=faces.device,
    )
    tw.array.sort_rows(adjacency)
    if return_edges:
        adjacency_edges = twt.empty_int32_2d((edge_groups.shape[0], 2), device=faces.device)
        wp.launch(
            kernel_array.gather_rows,
            dim=edge_groups.shape[0],
            inputs=[edges_sorted, edge_groups[:, 0], adjacency_edges],
            device=faces.device,
        )
        return twt.as_array2d_int32(adjacency), twt.as_array2d_int32(adjacency_edges)
    return twt.as_array2d_int32(adjacency)


_compute_face_adjacency = face_adjacency


def face_adjacency_unshared(
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
) -> twt.Array2dInt32:
    """
    Vertex on each adjacent face that is not on their shared edge.

    For each row of ``face_adjacency``, column 0 is the unshared vertex index on
    the first face and column 1 on the second face. When a face does not have
    exactly one vertex off the shared edge (degenerate case), that entry is ``-1``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of triangle vertex indices, the
        same flat layout as :func:`face_adjacency`.
    face_adjacency
        Optional ``(m, 2)`` face index pairs from :func:`face_adjacency`. When
        ``None``, adjacency and shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        :func:`face_adjacency` with ``return_edges=True``). Must be supplied
        together with ``face_adjacency`` or omitted with it.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(m, 2)`` on ``faces.device``. Row ``k`` gives vertex indices into
        ``faces`` for the corners not on ``face_adjacency_edges[k]``, or ``-1``
        when degenerate.

    Raises
    ------
    ValueError
        If ``faces.shape[0]`` is not divisible by ``3``, or if only one of
        ``face_adjacency`` and ``face_adjacency_edges`` is provided, or if their
        row counts differ.

    See Also
    --------
    :func:`face_adjacency`
    :func:`trimesh.graph.face_adjacency_unshared`
    """
    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError("face_adjacency and face_adjacency_edges must both be provided or both omitted")
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = _compute_face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None and face_adjacency_edges is not None
    if face_adjacency.shape[0] != face_adjacency_edges.shape[0]:
        raise ValueError(
            (
                "face_adjacency and face_adjacency_edges row counts must match, "
                f"got {face_adjacency.shape[0]} and {face_adjacency_edges.shape[0]}"
            )
        )
    m = int(face_adjacency.shape[0])
    unshared = twt.empty_int32_2d((m, 2), device=faces.device)
    if m == 0:
        return unshared
    wp.launch(
        kernel_graph.face_adjacency_unshared,
        dim=m,
        inputs=[faces, face_adjacency, face_adjacency_edges, unshared],
        device=faces.device,
    )
    return twt.as_array2d_int32(unshared)
