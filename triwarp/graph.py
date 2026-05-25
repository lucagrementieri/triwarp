import warp as wp
from typing import overload, Literal, Optional, Union
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import array as kernel_array
import triwarp as tw


def faces_to_edges(faces: wp.array[wp.int32], sorted: bool = False) -> wp.array2d[wp.int32]:
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
    wp.array2d[wp.int32]
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
    edges = wp.empty((n_faces * 3, 2), dtype=wp.int32, device=faces.device)
    wp.launch(
        kernel_graph.faces_to_edges_sorted if sorted else kernel_graph.faces_to_edges,
        dim=n_faces,
        inputs=[faces, edges],
        device=faces.device,
    )
    return edges


@overload
def face_adjacency(
    faces: wp.array[wp.int32], edges_sorted: Optional[wp.array2d[wp.int32]], return_edges: Literal[False] = False
) -> wp.array2d[wp.int32]: ...
@overload
def face_adjacency(
    faces: wp.array[wp.int32], edges_sorted: Optional[wp.array2d[wp.int32]], return_edges: Literal[True]
) -> tuple[wp.array2d[wp.int32], wp.array2d[wp.int32]]: ...
def face_adjacency(
    faces: wp.array[wp.int32], edges_sorted: Optional[wp.array2d[wp.int32]] = None, return_edges: bool = False
) -> Union[wp.array2d[wp.int32], tuple[wp.array2d[wp.int32], wp.array2d[wp.int32]]]:
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
    wp.array2d[wp.int32] or tuple of two such arrays
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
    if edges_sorted is None:
        edges_sorted = faces_to_edges(faces, sorted=True)
    edges_face = wp.array([f for f in range(n_faces) for _ in range(3)], dtype=wp.int32, device=faces.device)
    edge_groups = tw.grouping.group_int_rows(edges_sorted, length=2, max_value=n_faces)
    adjacency = wp.empty((edge_groups.shape[0], 2), dtype=wp.int32, device=faces.device)
    wp.launch(
        kernel_array.gather_2d_from_1d,
        dim=edge_groups.shape,
        inputs=[edges_face, edge_groups, adjacency],
        device=faces.device,
    )
    tw.array.sort_rows(adjacency)
    if return_edges:
        adjacency_edges = wp.empty((edge_groups.shape[0], 2), dtype=wp.int32, device=faces.device)
        wp.launch(
            kernel_array.gather_rows,
            dim=edge_groups.shape[0],
            inputs=[edges_sorted, edge_groups[:, 0], adjacency_edges],
            device=faces.device,
        )
        return adjacency, adjacency_edges
    return adjacency
