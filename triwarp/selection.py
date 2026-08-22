"""Face-subset extraction and vertex-selection morphology."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import arange
from triwarp.kernels import array as kernel_array
from triwarp.kernels import selection as kernel_selection


def region_boundary_edges(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], n_vertices: int | None = None
) -> twt.Array2dInt32:
    """
    Interior edges on the boundary of a face region.

    Returns the undirected edges that have exactly two incident faces, exactly one of which is in
    ``face_mask`` — i.e. the interior seam separating the region from the rest of the mesh (mesh
    boundary edges, with a single incident face, are excluded).

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` region mask.
    n_vertices
        Optional vertex count; inferred from ``faces`` when ``None``.

    Returns
    -------
    twt.Array2dInt32
        ``(k, 2)`` sorted vertex pairs on ``faces.device``.

    See Also
    --------
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]

    Notes
    -----
    Erosion here is vertex-based: a vertex survives when every 1-ring neighbour is also in the
    mask. MeshLab's Erode Selection is a *face*-based operation and gives a different answer; see
    ``benchmarks/test_selection.py``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=device)
    if n_vertices is None:
        n_vertices = tw.vertices.n_vertices(faces)
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    m = int(unique_edges.shape[0])
    count = wp.zeros(m, dtype=wp.int32, device=device)
    region_count = wp.zeros(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.edge_region_counts,
        dim=3 * n_faces,
        inputs=[inverse, face_mask, count, region_count],
        device=device,
    )
    flag = wp.empty(m, dtype=wp.bool, device=device)
    wp.map(kernel_selection.region_boundary_flag, count, region_count, out=flag)
    ids = tw.array.flatnonzero(flag)
    return twt.as_array2d(tw.array.gather(unique_edges, ids), wp.int32)


def exclude_fully_selected_components(
    faces: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    n_vertices: int,
    unique_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.bool]:
    """
    Drop selected vertices whose entire connected component is selected.

    A vertex-connected component that is wholly inside ``mask`` would make a region-smoothing
    Dirichlet system singular (no fixed anchor), so those components are removed from the
    selection; components with at least one unselected vertex are kept intact.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection.
    n_vertices
        Vertex count (component labels span ``0 .. n_vertices - 1``).
    unique_edges
        Optional precomputed ``(m, 2)`` unique edges; rebuilt when ``None``.

    Returns
    -------
    wp.array[wp.bool]
        Selection with fully-selected components removed, on ``mask.device``.
    """
    device = mask.device
    if n_vertices == 0:
        return wp.clone(mask)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    labels = tw.graph.connected_component_labels_from_edges(unique_edges, node_count=n_vertices)
    keep = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.keep_component_scatter,
        dim=n_vertices,
        inputs=[mask, labels, keep],
        device=device,
    )
    kept_per_vertex = keep[labels]  # Python-scope gather: component-keep flag per vertex
    out = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(kernel_selection.keep_selected, mask, kept_per_vertex, out=out)
    return out


@overload
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    unique_indices: bool = ...,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    unique_indices: bool = ...,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    unique_indices: bool = False,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Extract a face subset by index and reindex vertices from zero.

    Gathers the selected face triplets, compacts referenced vertices, and remaps
    face indices into the compact vertex buffer. Matches the core geometry step of
    [`trimesh.util.submesh`][] (without visuals, repair, watertight filtering, or
    append).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_indices
        1D ``wp.int32`` array of face indices into the source mesh
        (``0 .. n_faces - 1``), on the same device as ``vertices``.
    unique_indices
        If ``True``, ``face_indices`` is assumed to contain no duplicates and
        the deduplication pass is skipped.
    return_index
        If ``True``, also return the vertex map below -- which the extraction computes anyway, so it
        costs nothing.

    Returns
    -------
    sub_vertices : wp.array[wp.vec3]
        Compact positions on ``vertices.device``, length ``0`` when ``face_indices`` is empty.
    sub_faces : wp.array[wp.int32]
        Flat triangle index buffer into ``sub_vertices``.
    vertex_index : wp.array[wp.int32]
        Only when ``return_index`` is ``True``: length ``n_sub_vertices``, the **input** vertex each
        output vertex came from, ascending. That direction makes it a gather, so a per-vertex
        attribute follows the submesh with ``tw.array.gather(attribute, vertex_index)``.

    See Also
    --------
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
    [`concatenate`][triwarp.combine.concatenate]
    [`trimesh.util.submesh`][]
    """
    device = vertices.device
    k = int(face_indices.shape[0])
    if k == 0:
        empty_vertices = wp.empty(0, dtype=wp.vec3, device=device)
        empty_faces = wp.empty(0, dtype=wp.int32, device=device)
        if return_index:
            return empty_vertices, empty_faces, wp.empty(0, dtype=wp.int32, device=device)
        return empty_vertices, empty_faces

    if unique_indices:
        unique_face_indices = face_indices
        face_slots = arange(k, device)
    else:
        unique_face_indices, face_slots = tw.grouping.unique_1d(face_indices, return_inverse=True)

    unique_faces = tw.array.gather(faces.reshape((-1, 3)), unique_face_indices).reshape((-1,))

    unique_vertex_indices, remapped_faces = tw.grouping.unique_1d(unique_faces, return_inverse=True)
    sub_vertices = tw.array.gather(vertices, unique_vertex_indices)

    sub_faces = tw.array.gather(remapped_faces.reshape((-1, 3)), face_slots).reshape((-1,))

    if return_index:
        return sub_vertices, sub_faces, unique_vertex_indices
    return sub_vertices, sub_faces


def submeshes_from_face_groups(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    group_face_indices: wp.array[wp.int32],
    group_offsets: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Extract many face subsets at once: CSR groups in, CSR submeshes out.

    The batched form of
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]. Every launch count
    is ``O(1)`` in the number of groups, so extracting ten thousand components costs the same
    number of kernel launches and host synchronisations as extracting one — which is the whole
    point, since the per-group loop it replaces pays a sort, a scan and a readback *each*.

    How it stays batched: each corner of each selected face is packed into the single ``int64`` key
    ``group * n_vertices + vertex``, and one global
    [`unique_1d`][triwarp.grouping.unique_1d] deduplicates all groups together. Because the key is
    ``group * radix + vertex``, ascending key order is exactly ``(group, vertex)`` lexicographic
    order, so the unique slots come out already partitioned by group **and** ascending by original
    vertex index within each group — bit-identical to running ``unique_1d`` per group.

    A vertex shared by two groups is duplicated into both, matching
    [`trimesh.util.submesh`][] and the per-group loop.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    group_face_indices
        Concatenated face indices of every group, as ``wp.int32``. Indices must be unique within a
        group (duplicates would produce duplicate output faces).
    group_offsets
        Length-``k`` ``wp.int32`` start of each group in ``group_face_indices``, ascending, with
        ``group_offsets[0] == 0`` — the repo's no-terminator CSR convention (as in
        [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] and
        [`bfs_multi_source`][triwarp.graph.bfs_multi_source]). Groups must be non-empty.

    Returns
    -------
    vertices_all : wp.array[wp.vec3]
        Every group's compacted vertices, concatenated.
    vertex_offsets : wp.array[wp.int32]
        Length-``k`` start of each group in ``vertices_all``; group ``g`` owns
        ``vertices_all[vertex_offsets[g] : vertex_offsets[g + 1]]``, with the last group running to
        the end.
    faces_all : wp.array[wp.int32]
        Every group's reindexed flat faces, concatenated in the same group order. Group ``g`` owns
        ``faces_all[3 * group_offsets[g] : 3 * group_offsets[g + 1]]`` — the *input* offsets,
        scaled by three, because a group keeps exactly the faces it was given.

    Raises
    ------
    ValueError
        If ``k * n_vertices`` overflows the ``int64`` packing budget of ``2 ** 62``.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`split_batched`][triwarp.combine.split_batched]
    [`unique_1d`][triwarp.grouping.unique_1d]
    """
    device = vertices.device
    k = int(group_offsets.shape[0])
    n_selected = int(group_face_indices.shape[0])
    n_vertices = int(vertices.shape[0])
    if k == 0 or n_selected == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.zeros(k, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    radix = max(n_vertices, 1)
    if k * radix >= 1 << 62:
        raise ValueError(
            f"submeshes_from_face_groups cannot pack {k} groups x {radix} vertices into int64"
        )

    n_corners = 3 * n_selected
    keys = wp.empty(n_corners, dtype=wp.int64, device=device)
    wp.launch(
        kernel_selection.pack_group_vertex_keys,
        dim=n_corners,
        inputs=[faces, group_face_indices, group_offsets, wp.int64(radix), keys],
        device=device,
    )

    unique_keys, inverse = tw.grouping.unique_1d(keys, return_inverse=True)
    n_slots = int(unique_keys.shape[0])
    slot_groups = wp.empty(n_slots, dtype=wp.int32, device=device)
    vertex_ids = wp.empty(n_slots, dtype=wp.int32, device=device)
    wp.map(kernel_selection.group_of_key, unique_keys, wp.int64(radix), out=slot_groups)
    wp.map(kernel_selection.vertex_of_key, unique_keys, wp.int64(radix), out=vertex_ids)

    # Group starts from a histogram plus an exclusive scan rather than ``flatnonzero`` on the run
    # starts: no host synchronisation, and an empty group still gets a (zero-length) entry.
    group_counts = wp.zeros(k, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.count_group_slots,
        dim=n_slots,
        inputs=[slot_groups, group_counts],
        device=device,
    )
    vertex_offsets = wp.empty(k, dtype=wp.int32, device=device)
    wp.utils.array_scan(group_counts, out_array=vertex_offsets, inclusive=False)

    local_of_slot = wp.empty(n_slots, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.local_vertex_index,
        dim=n_slots,
        inputs=[slot_groups, vertex_offsets, local_of_slot],
        device=device,
    )

    return (
        tw.array.gather(vertices, vertex_ids),
        vertex_offsets,
        tw.array.gather(local_of_slot, inverse),
    )


@overload
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    *,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Extract a face subset selected by a per-face boolean mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array on the same device as ``vertices``.
    return_index
        If ``True``, also return the output-to-input vertex map, as
        [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices] documents.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[..., wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``, plus ``vertex_index`` when
        ``return_index`` is ``True``.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
        The complement: keep everything *outside* a region, and report the rims that opens.
    """
    face_indices = tw.array.flatnonzero(face_mask)
    if return_index:
        return submesh_from_face_indices(
            vertices, faces, face_indices, unique_indices=True, return_index=True
        )
    return submesh_from_face_indices(vertices, faces, face_indices, unique_indices=True)


def delete_region_keep_boundary(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], face_mask: wp.array[wp.bool]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], list[wp.array[wp.int32]]]:
    """
    Remove a face region and report the boundary loops the removal opened.

    The complement of
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]: it keeps everything the
    mask does *not* select, and it answers the question that makes the result usable -- **which rims
    are new**. That is the distinction a caller needs and cannot recover afterwards: the surviving
    mesh's boundary loops are the rims the deletion made *plus* whatever rims the input already had,
    and only the former should be filled. It is the natural input to
    [`triwarp.holes.fill_min_weight`][triwarp.holes.fill_min_weight], and the first half of
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array, ``True`` for each face to **delete**.

    Returns
    -------
    kept_vertices : wp.array[wp.vec3]
        Compact positions of the surviving submesh.
    kept_faces : wp.array[wp.int32]
        Flat triangle index buffer into ``kept_vertices``.
    new_loops : list[wp.array[wp.int32]]
        The boundary loops the deletion opened, as vertex-index cycles **into** ``kept_vertices`` --
        the same form [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] returns.
        A loop the input already had is excluded, so an open input's original rims do not appear.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.

    Examples
    --------
    ```python
    doomed = tw.validation.face_self_intersecting_mask(v, f)
    kept_v, kept_f, rims = tw.selection.delete_region_keep_boundary(v, f, doomed)
    ```

    Notes
    -----
    A loop counts as pre-existing when **every** one of its edges was already a boundary edge of the
    input. That is the right test rather than "any": deleting a face that touches an existing rim
    extends that rim rather than opening a new one, and the extended loop has to be reported --
    otherwise a caller filling only the new loops would leave the extension open.

    The loop classification is host-side, over the loops themselves rather than over the mesh: a
    boundary loop is short next to the surface it bounds, and the edge sets involved are already
    materialized by [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops]. Measured
    on an 82k-face sphere with a fifth of its faces deleted, that classification is **0.04 ms of a
    4.4 ms call**: the call is the loop *trace* (3.1 ms) plus the submesh extraction (0.9), so
    anything spent optimizing this function belongs in
    [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] rather than here.

    See Also
    --------
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        The same extraction with the mask's polarity reversed and no loop report.
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region]
        Delete and immediately fill, which is what this is usually the first half of.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )

    keep_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.map(kernel_selection.logical_not, face_mask, out=keep_mask)
    kept_vertices, kept_faces, vertex_index = submesh_from_face_mask(
        vertices, faces, keep_mask, return_index=True
    )
    if int(kept_faces.shape[0]) == 0:
        return kept_vertices, kept_faces, []

    kept_loops = tw.boundary.boundary_loops(kept_vertices, kept_faces)
    if not kept_loops:
        return kept_vertices, kept_faces, []

    # Undirected input boundary edges, as a host-side set in *input* indices -- the loops are mapped
    # into that space to be classified, since the submesh renumbered them. Computed *after* the loop
    # trace and only when there is something to classify: on a closed input this pass answers
    # nothing, and it is 0.46 ms of a 4.85 ms call on an 82k-face sphere.
    input_boundary = {
        (int(row[0]), int(row[1])) for row in tw.boundary.boundary_edges(vertices, faces).numpy()
    }
    to_input_np = vertex_index.numpy()

    new_loops: list[wp.array[wp.int32]] = []
    for loop in kept_loops:
        cycle_np = to_input_np[loop.numpy()]
        rolled_np = np.roll(cycle_np, -1)
        if all(
            (min(int(a), int(b)), max(int(a), int(b))) in input_boundary
            for a, b in zip(cycle_np, rolled_np, strict=True)
        ):
            continue  # every edge was already a rim: this loop is the input's, not the deletion's
        new_loops.append(loop)
    return kept_vertices, kept_faces, new_loops


def submesh_from_vertex_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_indices: wp.array[wp.int32],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract faces incident on the given vertices and reindex from zero.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_indices
        1D ``wp.int32`` array of vertex indices on the same device as ``vertices``.
    face_mode
        ``"all"`` selects faces whose three vertex indices all lie in ``vertex_indices``;
        ``"any"`` selects faces with at least one vertex index in ``vertex_indices``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    See Also
    --------
    [`face_indices_from_vertex_indices`][triwarp.selection.face_indices_from_vertex_indices]
    [`submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    """
    face_indices = face_indices_from_vertex_indices(faces, vertex_indices, face_mode=face_mode)
    return submesh_from_face_indices(vertices, faces, face_indices, unique_indices=True)


def submesh_from_vertex_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract faces incident on vertices selected by a boolean mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_mask
        Length-``n_vertices`` ``wp.bool`` array on the same device as ``vertices``.
    face_mode
        Passed to [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices].

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertex_mask`` length does not equal ``n_vertices``.

    See Also
    --------
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
    """
    n_vertices = int(vertices.shape[0])
    if int(vertex_mask.shape[0]) != n_vertices:
        raise ValueError(
            f"vertex_mask length must equal n_vertices={n_vertices}, got {vertex_mask.shape[0]}"
        )

    vertex_indices = tw.array.flatnonzero(vertex_mask)
    return submesh_from_vertex_indices(vertices, faces, vertex_indices, face_mode=face_mode)


def expand_vertex_mask(
    faces: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    hops: int,
    unique_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.bool]:
    """
    Grow a vertex selection by ``hops`` one-ring layers.

    Each round adds every vertex sharing a mesh edge with a currently-selected vertex, so after
    ``hops`` rounds the mask covers all vertices within graph distance ``hops`` of the input
    selection.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection to dilate.
    hops
        Number of one-ring dilation rounds (``0`` returns a copy).
    unique_edges
        Optional precomputed ``(m, 2)`` unique edges (from
        [`edges_unique`][triwarp.edges.edges_unique]); rebuilt when ``None``.

    Returns
    -------
    wp.array[wp.bool]
        Dilated mask on ``mask.device``.

    See Also
    --------
    [`shrink_vertex_mask`][triwarp.selection.shrink_vertex_mask]

    Notes
    -----
    Dilation is vertex-based: a vertex enters the mask when any 1-ring neighbour is in it.

    The per-round ``wp.clone`` is not worth removing. The kernel only *sets* bits, so each round
    must start from a copy of the previous mask; ping-ponging two preallocated buffers would keep
    the copy and drop only the allocation. Measured on an ``icosphere(5)`` selection: 0.073 ms at
    ``hops=1`` and 0.427 ms at ``hops=10``, i.e. **~0.043 ms per round** all-in, so the allocation
    is a fraction of a fraction and the ping-pong would buy less than the session-to-session drift.
    """
    device = mask.device
    n = int(mask.shape[0])
    current = wp.clone(mask)
    if hops <= 0 or n == 0:
        return current
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n)
    m = int(unique_edges.shape[0])
    for _ in range(hops):
        nxt = wp.clone(current)
        if m > 0:
            wp.launch(
                kernel_selection.dilate_vertex_mask,
                dim=m,
                inputs=[unique_edges, current, nxt],
                device=device,
            )
        current = nxt
    return current


def shrink_vertex_mask(
    faces: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    hops: int,
    unique_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.bool]:
    """
    Erode a vertex selection by ``hops`` one-ring layers.

    Implemented as the complement of an [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]
    of the complement: a vertex is removed if any vertex within ``hops`` hops is unselected.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    mask
        Length-``n_vertices`` ``wp.bool`` selection to erode.
    hops
        Number of one-ring erosion rounds.
    unique_edges
        Optional precomputed ``(m, 2)`` unique edges; rebuilt when ``None``.

    Returns
    -------
    wp.array[wp.bool]
        Eroded mask on ``mask.device``.

    See Also
    --------
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]

    Notes
    -----
    Erosion here is vertex-based: a vertex survives when every 1-ring neighbour is also in the
    mask. MeshLab's Erode Selection is a *face*-based operation and gives a different answer; see
    ``benchmarks/test_selection.py``.
    """
    device = mask.device
    n = int(mask.shape[0])
    if hops <= 0 or n == 0:
        return wp.clone(mask)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n)
    complement = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, mask, out=complement)
    dilated = expand_vertex_mask(faces, complement, hops, unique_edges)
    out = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, dilated, out=out)
    return out


def face_indices_from_vertex_indices(
    faces: wp.array[wp.int32],
    vertex_indices: wp.array[wp.int32],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> wp.array[wp.int32]:
    """
    Face indices whose vertex indices match a set under an all/any rule.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    vertex_indices
        1D ``wp.int32`` array of vertex indices on the same device as ``faces``.
    face_mode
        ``"all"`` keeps faces whose three vertex indices all lie in ``vertex_indices``;
        ``"any"`` keeps faces with at least one vertex index in ``vertex_indices``.

    Returns
    -------
    wp.array[wp.int32]
        Selected face indices on ``faces.device``. Empty when no faces match or
        ``vertex_indices`` is empty.

    Raises
    ------
    ValueError
        If ``face_mode`` is not ``"all"`` or ``"any"``.
    """
    if face_mode not in ("all", "any"):
        raise ValueError(f'face_mode must be "all" or "any", got {face_mode!r}')

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if int(vertex_indices.shape[0]) == 0 or n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    faces2d = faces.reshape((-1, 3))
    corner_hit = tw.array.isin(faces2d, vertex_indices)
    if face_mode == "all":
        face_hit = tw.reduce.all(corner_hit, axis=1)
    else:
        face_hit = tw.reduce.any(corner_hit, axis=1)
    return tw.array.flatnonzero(face_hit)
