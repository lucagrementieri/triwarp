from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, overload

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import init_range
from triwarp.kernels import array as kernel_array
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import selection as kernel_selection
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.algorithms import connected_components as kernel_connected_components


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
        same flat layout as [`triwarp.triangles`][triwarp.triangles] and
        [`faces_to_edges`][triwarp.edges.faces_to_edges].
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` edge rows with each row sorted
        so the smaller vertex index is first (as from
        [`faces_to_edges`][triwarp.edges.faces_to_edges] with
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
    Duplicate-edge grouping uses
    [`group_int_rows`][triwarp.grouping.group_int_rows] with
    ``length=2``, equivalent to [`trimesh.grouping.group_rows`][] with
    ``require_count=2``. An empty mesh yields shape ``(0, 2)``.

    See Also
    --------
    [`faces_to_edges`][triwarp.edges.faces_to_edges]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_connected_component_labels`][triwarp.graph.face_connected_component_labels]
    [`trimesh.graph.face_adjacency`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_array = twt.empty_int32_2d((0, 2), device=faces.device)
        if return_edges:
            return empty_array, twt.empty_int32_2d((0, 2), device=faces.device)
        return empty_array
    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    edges_face = tw.edges.edges_face(faces)
    # Hash edge rows over the vertex-index range (inferred max + 1); using ``n_faces`` as the base
    # is wrong whenever the largest vertex index is >= n_faces (e.g. small meshes with more
    # vertices than faces). The grouping partition is invariant to the (sufficiently large) base.
    edge_groups = tw.grouping.group_int_rows(edges_sorted, length=2)
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
        if edge_groups.shape[0] > 0:
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
        same flat layout as [`face_adjacency`][triwarp.graph.face_adjacency].
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.graph.face_adjacency]. When
        ``None``, adjacency and shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.graph.face_adjacency] with ``return_edges=True``).
        Must be supplied
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
    [`face_adjacency`][triwarp.graph.face_adjacency]
    [`trimesh.graph.face_adjacency_unshared`][]
    """
    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = _compute_face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None
    assert face_adjacency_edges is not None
    if face_adjacency.shape[0] != face_adjacency_edges.shape[0]:
        raise ValueError(
            "face_adjacency and face_adjacency_edges row counts must match, "
            f"got {face_adjacency.shape[0]} and {face_adjacency_edges.shape[0]}"
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


def face_adjacency_angles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between each pair of adjacent faces.

    For each row of ``face_adjacency``, the angle is computed from the two
    corresponding face normals (unit vectors). For a signed angle, combine with
    ``face_adjacency_convex`` once that attribute is available.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.graph.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.graph.face_adjacency]. When
        ``None``, adjacency is computed from ``faces``.
    face_normals
        Optional length-``n_faces`` unit face normals. When ``None``, normals
        are computed from ``vertices`` and ``faces`` via
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` unsigned angles in radians on ``faces.device``, one per
        ``face_adjacency`` row. Empty when there are no faces or no adjacency pairs.

    Raises
    ------
    ValueError
        If ``vertices`` and ``faces`` live on different devices.

    See Also
    --------
    [`face_adjacency`][triwarp.graph.face_adjacency]
    [`vector_angle`][triwarp.array.vector_angle]
    [`trimesh.Trimesh.face_adjacency_angles`][]
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if face_adjacency is None:
        face_adjacency = _compute_face_adjacency(faces)
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_angles = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_graph.face_adjacency_angles,
        dim=m,
        inputs=[face_normals, face_adjacency, out_angles],
        device=device,
    )
    return out_angles


def concatenate(
    meshes_data: Sequence[tuple[wp.array[wp.vec3], wp.array[wp.int32]]],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Concatenate meshes, each given as ``(vertices, faces)`` on the same device.

    Face indices are renumbered with cumulative vertex offsets, matching
    [`trimesh.util.concatenate`][] (with triwarp's flat ``(3 * n_faces,)`` face
    layout instead of ``(n_faces, 3)``).

    Parameters
    ----------
    meshes_data
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
    [`split`][triwarp.graph.split]
    [`trimesh.util.concatenate`][]
    """
    if len(meshes_data) == 0:
        return wp.empty(0, dtype=wp.vec3), wp.empty(0, dtype=wp.int32)

    device = meshes_data[0][0].device
    vertex_counts: list[int] = []
    total_indices = 0
    for _i, (vertices, faces) in enumerate(meshes_data):
        f = int(faces.shape[0])
        vertex_counts.append(int(vertices.shape[0]))
        total_indices += f

    if sum(vertex_counts) == 0:
        concatenated_vertices = wp.empty(0, dtype=wp.vec3, device=device)
    else:
        concatenated_vertices, _ = tw.array.pack_1d_arrays(
            [vertices for vertices, _ in meshes_data]
        )

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
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]:
    """
    Split a mesh into connected components by face adjacency.

    Each returned pair is a compact ``(vertices, faces)`` submesh with vertices
    reindexed from zero, matching [`trimesh.graph.split`][] with
    ``only_watertight=False``. [`concatenate`][triwarp.graph.concatenate] on the result recovers the
    input mesh (up to vertex/face ordering within each body).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.graph.face_adjacency]).

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
    [`concatenate`][triwarp.graph.concatenate]
    [`face_connected_component_labels`][triwarp.graph.face_connected_component_labels]
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`trimesh.graph.split`][]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return []

    face_labels = face_connected_component_labels(faces)
    unique_labels = tw.unique.unique_1d(face_labels)

    meshes: list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = []
    for label in unique_labels.numpy():
        label_wp = wp.array([int(label)], dtype=wp.int32, device=device)
        face_indices = tw.array.flatnonzero(tw.array.isin(face_labels, label_wp))
        meshes.append(
            tw.selection.submesh_from_face_indices(
                vertices, faces, face_indices, unique_indices=True
            )
        )
    return meshes


def edges_to_csr(node_count: int, edges: twt.Array2dInt32) -> wps.BsrMatrix[wp.float32]:
    """
    Undirected adjacency as a 1x1-block ``warp.sparse.BsrMatrix`` (CSR form).

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
        wp.launch(kernel_graph.edges_to_adjacency, dim=m, inputs=[edges, rows, cols], device=device)
    data = wp.ones(n_entries, dtype=wp.float32, device=device)
    return wps.bsr_from_triplets(
        node_count, node_count, rows, cols, data, prune_numerical_zeros=False
    )


def connected_component_labels(adjacency: wps.BsrMatrix[wp.Scalar]) -> wp.array[wp.int32]:
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
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form.
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
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`face_connected_component_labels`][triwarp.graph.face_connected_component_labels]
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
        return init_range(node_count, device)

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
                f"connected_component_labels: hook passes did not converge "
                f"after {node_count} iterations"
            )
        else:
            raise RuntimeError(
                f"connected_component_labels: edge verification failed "
                f"after {node_count} iterations"
            )


def connected_component_labels_from_edges(
    edges: twt.Array2dInt32, node_count: int | None = None
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from an undirected edge list.

    Builds a CSR adjacency via [`edges_to_csr`][triwarp.graph.edges_to_csr] and delegates to
    [`connected_component_labels`][triwarp.graph.connected_component_labels].

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
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`face_connected_component_labels`][triwarp.graph.face_connected_component_labels]
    [`trimesh.graph.connected_component_labels`][]
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")

    device = edges.device
    m = int(edges.shape[0])

    if node_count is None:
        node_count = tw.vertices.n_vertices(edges)
    elif node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    elif m == 0:
        return init_range(node_count, device)
    else:
        edges_np = edges.numpy()
        if edges_np.min() < 0 or int(edges_np.max()) >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), "
                f"got min={edges_np.min()} max={edges_np.max()}"
            )

    adjacency = edges_to_csr(node_count, edges)
    return connected_component_labels(adjacency)


def face_connected_component_labels(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Connected-component label per face (face-adjacency graph).

    Equivalent to
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    on [`face_adjacency`][triwarp.graph.face_adjacency]
    with ``node_count = n_faces``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same as
        [`face_adjacency`][triwarp.graph.face_adjacency]).

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces`` on ``faces.device``.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_adjacency`][triwarp.graph.face_adjacency]
    """
    n_faces = int(faces.shape[0]) // 3
    adjacency = face_adjacency(faces)
    return connected_component_labels_from_edges(adjacency, node_count=n_faces)


def bfs(
    adjacency: wps.BsrMatrix[wp.Scalar], source: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Single-source breadth-first search over a sparse CSR adjacency matrix.

    Runs a serial traversal from ``source`` (one device thread) so the discovery order, parent
    tree, and distances match [`scipy.sparse.csgraph.breadth_first_order`][] exactly when the
    adjacency columns are sorted ascending per row (as produced by
    [`edges_to_csr`][triwarp.graph.edges_to_csr]). This
    mirrors ``igl::bfs`` (`reference/libigl/include/igl/bfs.cpp`), additionally returning the BFS
    level of each node.

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form. Each nonzero
        ``(i, j)`` denotes an edge between nodes ``i`` and ``j``; for undirected graphs both
        ``(i, j)`` and ``(j, i)`` should be present (as from
        [`edges_to_csr`][triwarp.graph.edges_to_csr]).
    source
        Start node, in ``[0, node_count)``.

    Returns
    -------
    order
        ``wp.array[wp.int32]`` of the reachable nodes in BFS discovery order; length equals the
        number of nodes reachable from ``source`` (matches scipy's ``node_array``).
    parents
        Length ``node_count`` on ``adjacency.device``. ``parents[i]`` is the predecessor of ``i``
        in the BFS tree; ``-1`` for ``source`` and for unreachable nodes (scipy uses ``-9999``).
    distances
        Length ``node_count``. ``distances[i]`` is the BFS level (hop count) of ``i`` from
        ``source``; ``-1`` for unreachable nodes.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square, does not use 1x1 blocks, or ``source`` is out of range.

    See Also
    --------
    [`bfs_from_edges`][triwarp.graph.bfs_from_edges]
    [`bfs_multi_source`][triwarp.graph.bfs_multi_source]
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`scipy.sparse.csgraph.breadth_first_order`][]
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")
    if source < 0 or source >= node_count:
        raise ValueError(f"source must be in [0, {node_count}), got {source}")

    device = adjacency.device
    offsets = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    columns = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]

    parents = wp.full(node_count, -1, dtype=wp.int32, device=device)
    distances = wp.full(node_count, -1, dtype=wp.int32, device=device)
    order_buffer = wp.empty(node_count, dtype=wp.int32, device=device)
    reached = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(
        kernel_bfs.single_source_bfs_kernel,
        dim=1,
        inputs=[wp.int32(source), offsets, columns, order_buffer, parents, distances, reached],
        device=device,
    )
    n_reached = int(reached.numpy()[0])
    order = wp.clone(order_buffer[:n_reached])
    return order, parents, distances


def bfs_from_edges(
    edges: twt.Array2dInt32, source: int, node_count: int | None = None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Single-source BFS from an undirected edge list.

    Builds a CSR adjacency via [`edges_to_csr`][triwarp.graph.edges_to_csr] and delegates
    to [`bfs`][triwarp.graph.bfs].

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` edge list. Each row ``(a, b)`` connects nodes ``a`` and ``b``
        (undirected; order does not matter).
    source
        Start node, in ``[0, node_count)``.
    node_count
        Number of nodes ``0 .. node_count - 1``. When ``None``, inferred as ``max(edges) + 1`` if
        ``m > 0``, else ``0``.

    Returns
    -------
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
        ``(order, parents, distances)`` as in [`bfs`][triwarp.graph.bfs], on ``edges.device``.

    Raises
    ------
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``, ``node_count``
        is negative, or ``source`` is out of range.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")

    m = int(edges.shape[0])
    if node_count is None:
        node_count = tw.vertices.n_vertices(edges)
    else:
        if node_count < 0:
            raise ValueError(f"node_count must be non-negative, got {node_count}")
        if m > 0:
            edges_np = edges.numpy()
            if edges_np.min() < 0 or int(edges_np.max()) >= node_count:
                raise ValueError(
                    f"edge indices must lie in [0, {node_count}), "
                    f"got min={edges_np.min()} max={edges_np.max()}"
                )

    adjacency = edges_to_csr(node_count, edges)
    return bfs(adjacency, source)


def bfs_multi_source(
    adjacency: wps.BsrMatrix[wp.Scalar], sources: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Independent BFS reachable sets for many sources, packed as a CSR buffer.

    One device thread per source runs a topological BFS over ``adjacency`` (the geodesic-ball
    traversal with its geometric predicate disabled). Source ``sources[k]`` owns
    ``neighbors[offsets[k] : offsets[k + 1]]``, listed in BFS discovery order (the source itself
    first). Each thread uses fixed-capacity scratch of ``kernel_bfs._PER_SOURCE_MAX_NEIGHBORS``
    nodes; a source whose reachable set exceeds that has the surplus dropped and a warning emitted.
    For a single source with no capacity limit, use [`bfs`][triwarp.graph.bfs].

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form.
    sources
        Length-``k`` ``wp.int32`` start nodes, each in ``[0, node_count)``.

    Returns
    -------
    neighbors
        Flat ``wp.array[wp.int32]`` of reachable nodes for all sources, concatenated in source
        order (CSR column buffer).
    offsets
        Length-``k`` exclusive prefix sum of per-source counts (CSR starts); source ``k`` owns
        ``neighbors[offsets[k] : offsets[k + 1]]`` with ``offsets[k_total]`` implied as the total.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square, does not use 1x1 blocks, or a source is out of range.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`query_geodesic_ball`][triwarp.proximity.query_geodesic_ball]
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")

    device = adjacency.device
    k = int(sources.shape[0])
    if k == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, wp.empty(0, dtype=wp.int32, device=device)

    sources_np = sources.numpy()
    if sources_np.min() < 0 or int(sources_np.max()) >= node_count:
        raise ValueError(
            f"source indices must lie in [0, {node_count}), "
            f"got min={sources_np.min()} max={sources_np.max()}"
        )

    offsets_csr = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    columns = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]

    overflow = wp.zeros(1, dtype=wp.int32, device=device)
    counts = wp.empty(k, dtype=wp.int32, device=device)
    wp.launch(
        kernel_bfs.multi_source_bfs_count,
        dim=k,
        inputs=[offsets_csr, columns, sources, counts, overflow],
        device=device,
    )
    n_overflow = int(overflow.numpy()[0])
    if n_overflow > 0:
        warnings.warn(
            f"bfs_multi_source: {n_overflow} reachable-set capacity breaches "
            f"(fixed cap {kernel_bfs._PER_SOURCE_MAX_NEIGHBORS}); surplus nodes dropped.",
            stacklevel=2,
        )

    offsets = wp.empty(k, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=offsets, inclusive=False)
    total = int(offsets.numpy()[-1]) + int(counts.numpy()[-1])

    neighbors = wp.empty(total, dtype=wp.int32, device=device)
    overflow.zero_()
    wp.launch(
        kernel_bfs.multi_source_bfs_neighbors,
        dim=k,
        inputs=[offsets_csr, columns, sources, offsets, neighbors, overflow],
        device=device,
    )
    return neighbors, offsets
