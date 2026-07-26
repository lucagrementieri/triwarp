"""Mesh face-adjacency graph: which faces share an edge, and face-level connected components."""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import array as kernel_array


@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: Literal[False] = False,
    n_vertices: int | None = None,
) -> twt.Array2dInt32: ...
@overload
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: Literal[True],
    n_vertices: int | None = None,
) -> tuple[twt.Array2dInt32, twt.Array2dInt32]: ...
def face_adjacency(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    *,
    return_edges: bool = False,
    n_vertices: int | None = None,
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
    n_vertices
        Optional vertex count, used as the row-hashing radix. Supplying it skips the
        ``triwarp.reduce.minmax`` that would otherwise infer and validate the radix, and with it a
        host readback that serialises the device pipeline — worth passing from loops that call this
        once per pass. Must be greater than every index in ``faces``; see the warning on
        [`hash_indices_rows`][triwarp.grouping.hash_indices_rows].

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
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
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
    edge_groups = tw.grouping.group_int_rows(
        edges_sorted, length=2, max_value=n_vertices, validate=n_vertices is None
    )
    adjacency = twt.empty_int32_2d((edge_groups.shape[0], 2), device=faces.device)
    wp.launch(
        kernel_array.gather_2d_from_1d,
        dim=edge_groups.shape,
        inputs=[edges_face, edge_groups, adjacency],
        device=faces.device,
    )
    tw.array.sort_rows(adjacency)
    if return_edges:
        if edge_groups.shape[0] > 0:
            # ``edge_groups[:, 0]`` is a strided column view; Warp's fancy indexing reads the
            # underlying flat buffer and ignores the stride, so materialize a contiguous index
            # first. (Slicing an empty first axis also raises, hence the guard.)
            first_edge_index = wp.clone(edge_groups[:, 0])
            adjacency_edges = tw.array.gather(edges_sorted, first_edge_index)
        else:
            adjacency_edges = twt.empty_int32_2d((0, 2), device=faces.device)
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
        same flat layout as [`face_adjacency`][triwarp.adjacency.face_adjacency].
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When
        ``None``, adjacency and shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.adjacency.face_adjacency] with ``return_edges=True``).
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
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
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
        kernel_adjacency.face_adjacency_unshared,
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
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When
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
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    [`vector_angle`][triwarp.points.vector_angle]
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
        kernel_adjacency.face_adjacency_angles,
        dim=m,
        inputs=[face_normals, face_adjacency, out_angles],
        device=device,
    )
    return out_angles


def face_connected_component_labels(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Connected-component label per face (face-adjacency graph).

    Equivalent to
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    on [`face_adjacency`][triwarp.adjacency.face_adjacency]
    with ``node_count = n_faces``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces`` on ``faces.device``.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    """
    n_faces = int(faces.shape[0]) // 3
    adjacency = face_adjacency(faces)
    return tw.graph.connected_component_labels_from_edges(adjacency, node_count=n_faces)
