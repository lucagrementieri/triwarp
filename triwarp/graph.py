from __future__ import annotations

from collections.abc import Sequence
from typing import overload, Literal

import warp as wp
import warp.sparse as wps
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import array as kernel_array
from triwarp.kernels import selection as kernel_selection
from triwarp.kernels.algorithms import connected_components as kernel_connected_components
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

    See Also
    --------
    :func:`trimesh.geometry.faces_to_edges`
    """
    n_faces = int(faces.shape[0]) // 3
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

    See Also
    --------
    :func:`faces_to_edges`
    :func:`connected_component_labels_from_edges`
    :func:`face_connected_component_labels`
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
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided,
        or if their row counts differ.

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


def concatenate(
    meshes_data: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Concatenate meshes, each given as ``(vertices, faces)`` on the same device.

    Face indices are renumbered with cumulative vertex offsets, matching
    :func:`trimesh.util.concatenate` (with triwarp's flat ``(3 * n_faces,)`` face
    layout instead of ``(n_faces, 3)``).

    Parameters
    ----------
    meshes
        Sequence of ``(vertices, faces)`` pairs using triwarp's flat face layout.
        An empty sequence yields empty arrays on ``cpu``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Combined vertices and reindexed faces on the shared device.

    Raises
    ------
    ValueError
        If any pair uses a different device.

    See Also
    --------
    :func:`split`
    :func:`trimesh.util.concatenate`
    """
    if len(meshes_data) == 0:
        return wp.empty(0, dtype=wp.vec3), wp.empty(0, dtype=wp.int32)

    device = meshes_data[0][0].device
    vertex_counts: list[int] = []
    total_indices = 0
    for i, (vertices, faces) in enumerate(meshes_data):
        if vertices.device != device or faces.device != device:
            raise ValueError(f"all arrays must live on the same device, got mismatch at index {i}")
        f = int(faces.shape[0])
        vertex_counts.append(int(vertices.shape[0]))
        total_indices += f

    if sum(vertex_counts) == 0:
        concatenated_vertices = wp.empty(0, dtype=wp.vec3, device=device)
    else:
        concatenated_vertices, _ = tw.array.pack_1d_arrays([vertices for vertices, _ in meshes_data])

    concatenated_faces = wp.empty(total_indices, dtype=wp.int32, device=device)

    vertex_offset = wp.int32(0)
    dest_offset = wp.int32(0)
    for count, (_, faces) in zip(vertex_counts, meshes_data, strict=True):
        f = int(faces.shape[0])
        if f > 0:
            wp.launch(
                kernel_selection.offset_copy_int32,
                dim=f,
                inputs=[faces, vertex_offset, dest_offset, concatenated_faces],
                device=device,
            )
            dest_offset += wp.int32(f)
        vertex_offset += wp.int32(count)

    return concatenated_vertices, concatenated_faces


def split(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
) -> list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]:
    """
    Split a mesh into connected components by face adjacency.

    Each returned pair is a compact ``(vertices, faces)`` submesh with vertices
    reindexed from zero, matching :func:`trimesh.graph.split` with
    ``only_watertight=False``. :func:`concatenate` on the result recovers the
    input mesh (up to vertex/face ordering within each body).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        :func:`face_adjacency`).

    Returns
    -------
    list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]
        One ``(vertices, faces)`` pair per face-connected component on
        ``vertices.device``. Empty when ``n_faces == 0``.

    Raises
    ------
    ValueError
        If ``vertices`` and ``faces`` live on different devices.

    See Also
    --------
    :func:`concatenate`
    :func:`face_connected_component_labels`
    :func:`triwarp.selection.submesh_from_face_indices`
    :func:`trimesh.graph.split`
    """
    device = vertices.device
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return []

    face_labels = face_connected_component_labels(faces)
    unique_labels = tw.unique.unique_1d(face_labels)

    meshes: list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = []
    for label in unique_labels.numpy():
        label_wp = wp.array([int(label)], dtype=wp.int32, device=device)
        face_indices = tw.array.flatnonzero(tw.array.isin(face_labels, label_wp))
        meshes.append(tw.selection.submesh_from_face_indices(vertices, faces, face_indices, unique_indices=True))
    return meshes


def edges_to_csr(
    node_count: int,
    edges: twt.Array2dInt32,
) -> wps.BsrMatrix[wp.float32]:
    """
    Undirected adjacency as a 1x1-block :class:`warp.sparse.BsrMatrix` (CSR form).

    Each undirected edge ``(a, b)`` contributes directed entries ``(a, b)`` and ``(b, a)``.

    Parameters
    ----------
    node_count
        Number of vertices ``0 .. node_count - 1``.
    edges
        ``(m, 2)`` ``wp.int32`` edge rows on the target device.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(node_count, node_count)`` adjacency with unit block values. Duplicate
        directed pairs from repeated input edges are merged (values summed).
    """
    device = edges.device
    m = int(edges.shape[0])

    n_entries = 2 * m
    rows = wp.empty(n_entries, dtype=wp.int32, device=device)
    cols = wp.empty(n_entries, dtype=wp.int32, device=device)
    if m > 0:
        wp.launch(
            kernel_graph.edges_to_adjacency,
            dim=m,
            inputs=[edges, rows, cols],
            device=device,
        )
    data = wp.ones(n_entries, dtype=wp.float32, device=device)
    return wps.bsr_from_triplets(node_count, node_count, rows, cols, data, prune_numerical_zeros=False)


def connected_component_labels(
    adjacency: wps.BsrMatrix[wp.Scalar],
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from a sparse adjacency matrix.

    Uses ECL-lite (ECL-CC style init, CAS hooking, intermediate pointer jumping)
    on the CSR structure of ``adjacency`` (1x1 BSR blocks). Hook passes run until
    a pass reports no merge attempts (``changed == 0``), no exhausted per-edge CAS
    retries (``incomplete == 0``), and a post-hook CSR edge check finds all
    endpoints sharing the same representative. At most ``node_count`` hook passes
    are attempted before raising. Labels identify nodes in the same component;
    values are not necessarily contiguous in ``0 .. k-1`` (compare partitions,
    not raw ids).

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block :class:`warp.sparse.BsrMatrix` form.
        Each nonzero ``(i, j)`` denotes an edge between nodes ``i`` and ``j``; for
        undirected graphs both ``(i, j)`` and ``(j, i)`` should be present.

    Returns
    -------
    wp.array[wp.int32]
        Length ``adjacency.nrow`` on ``adjacency.device``. Isolated nodes (empty rows)
        receive distinct labels. When ``nnz == 0``, ``labels[i] == i``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square or does not use 1x1 blocks.
    RuntimeError
        If hook passes reach ``node_count`` without converging, or edge
        verification still fails at that limit.

    See Also
    --------
    :func:`connected_component_labels_from_edges`
    :func:`edges_to_csr`
    :func:`face_connected_component_labels`
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")

    device = adjacency.device
    if node_count <= 1:
        return wp.zeros(node_count, dtype=wp.int32, device=device)
    if adjacency.nnz == 0:
        return wp.array(range(node_count), dtype=wp.int32, device=device)

    offsets = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    indices = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]

    labels = wp.empty(node_count, dtype=wp.int32, device=device)
    parents = wp.empty(node_count, dtype=wp.int32, device=device)

    wp.launch(
        kernel_connected_components.ecl_init_parent,
        dim=node_count,
        inputs=[offsets, indices, parents],
        device=device,
    )

    changed = wp.zeros(1, dtype=wp.int32, device=device)
    incomplete = wp.zeros(1, dtype=wp.int32, device=device)
    violations = wp.zeros(1, dtype=wp.int32, device=device)

    for _ in range(node_count):
        changed.zero_()
        incomplete.zero_()
        wp.launch(
            kernel_connected_components.ecl_hook,
            dim=node_count,
            inputs=[offsets, indices, parents, changed, incomplete],
            device=device,
        )
        if changed.numpy().item() != 0 or incomplete.numpy().item() != 0:
            continue

        violations.zero_()
        wp.launch(
            kernel_connected_components.ecl_finalize_and_verify,
            dim=node_count,
            inputs=[offsets, indices, parents, labels, violations],
            device=device,
        )
        if violations.numpy().item() == 0:
            return labels
    else:
        n_changed = changed.numpy().item()
        n_incomplete = incomplete.numpy().item()
        if n_changed != 0 or n_incomplete != 0:
            raise RuntimeError(
                f"connected_component_labels: hook passes did not converge after {node_count} iterations"
            )
        else:
            raise RuntimeError(f"connected_component_labels: edge verification failed after {node_count} iterations")


def connected_component_labels_from_edges(
    edges: twt.Array2dInt32,
    node_count: int | None = None,
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from an undirected edge list.

    Builds a CSR adjacency via :func:`edges_to_csr` and delegates to
    :func:`connected_component_labels`.

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` edge list. Each row ``(a, b)`` connects nodes ``a``
        and ``b`` (undirected; order does not matter).
    node_count
        Number of nodes ``0 .. node_count - 1``. When ``None``, inferred as
        ``max(edges) + 1`` if ``m > 0``, else ``0``.

    Returns
    -------
    wp.array[wp.int32]
        Length ``node_count`` on ``edges.device``.

    Raises
    ------
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``,
        or ``node_count`` is negative.

    See Also
    --------
    :func:`connected_component_labels`
    :func:`face_connected_component_labels`
    :func:`trimesh.graph.connected_component_labels`
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")

    device = edges.device
    m = int(edges.shape[0])

    if node_count is None:
        node_count = int(edges.numpy().max()) + 1 if m > 0 else 0
    elif node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    elif m == 0:
        return wp.array(range(node_count), dtype=wp.int32, device=device)
    else:
        edges_np = edges.numpy()
        if edges_np.min() < 0 or int(edges_np.max()) >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), got min={edges_np.min()} max={edges_np.max()}"
            )

    adjacency = edges_to_csr(node_count, edges)
    return connected_component_labels(adjacency)


def face_connected_component_labels(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Connected-component label per face (face-adjacency graph).

    Equivalent to :func:`connected_component_labels_from_edges` on :func:`face_adjacency`
    with ``node_count = n_faces``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same as :func:`face_adjacency`).

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces`` on ``faces.device``.

    See Also
    --------
    :func:`connected_component_labels`
    :func:`connected_component_labels_from_edges`
    :func:`face_adjacency`
    """
    n_faces = int(faces.shape[0]) // 3
    adjacency = face_adjacency(faces)
    return connected_component_labels_from_edges(adjacency, node_count=n_faces)
