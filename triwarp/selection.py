"""Face-subset extraction and vertex-selection morphology."""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_same_device
from triwarp.halfedge import halfedge_twins, require_matching_twins
from triwarp.kernels import array as kernel_array
from triwarp.kernels import grouping as kernel_grouping
from triwarp.kernels import selection as kernel_selection


def region_boundary_edges(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    n_vertices: int | None = None,
    *,
    oriented: bool = False,
) -> twt.Array2dInt32:
    """
    Interior edges on the boundary of a face region.

    Returns the edges that have exactly two incident faces, exactly one of which is in
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
    oriented
        Return each row as the **directed** pair belonging to its region face, instead of the
        ascending pair. That puts the region on the left of the contour, which is the orientation
        [`faces_left_of_contour`][triwarp.selection.faces_left_of_contour] reads — the two are
        inverse and round-trip exactly. The default ascending form carries no orientation at all,
        so feeding it to that function seeds *both* sides and returns the whole mesh.

    Returns
    -------
    twt.Array2dInt32
        ``(k, 2)`` vertex pairs on ``faces.device``, sorted ascending per row unless ``oriented``.

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``faces`` and ``face_mask`` are not all on one device.

    See Also
    --------
    [`faces_left_of_contour`][triwarp.selection.faces_left_of_contour]
        The inverse: turns this seam back into the region it bounds, given ``oriented=True``.
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]
    """
    require_same_device(faces=faces, face_mask=face_mask)
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )
    if n_faces == 0:
        return twt.empty_2d((0, 2), wp.int32, device=device)
    if n_vertices is None:
        # ``require_non_negative`` is free here and is the half of the packing's range check that a
        # bound derived from these same indices cannot supply.
        n_vertices = tw.array.index_bound(faces, require_non_negative=True)
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    m = int(unique_edges.shape[0])
    count = wp.zeros(m, dtype=wp.int32, device=device)
    region_count = wp.zeros(m, dtype=wp.int32, device=device)
    # `region_halfedge` is written only for edges the flag pass below keeps (region_count == 1,
    # exactly one write each), so an unwritten entry is never read and `wp.empty` is correct
    # (§3.3). It stays `None` on the non-oriented path: the kernel's write is itself guarded by
    # `oriented`, so this never indexes the null array.
    region_halfedge = wp.empty(m, dtype=wp.int32, device=device) if oriented else None
    wp.launch(
        kernel_selection.edge_region_counts,
        dim=3 * n_faces,
        inputs=[inverse, face_mask, oriented, count, region_count, region_halfedge],
        device=device,
    )
    flag = wp.empty(m, dtype=wp.bool, device=device)
    wp.map(kernel_selection.region_boundary_flag, count, region_count, out=flag)
    ids = tw.array.flatnonzero(flag)
    if not oriented:
        return twt.as_array2d(tw.array.gather(unique_edges, ids), wp.int32)

    oriented_edges = twt.empty_2d((int(ids.shape[0]), 2), wp.int32, device=device)
    wp.launch(
        kernel_selection.oriented_edges_from_halfedges,
        dim=int(ids.shape[0]),
        inputs=[faces, tw.array.gather(region_halfedge, ids), oriented_edges],
        device=device,
    )
    return twt.as_array2d(oriented_edges, wp.int32)


def faces_left_of_contour(
    faces: wp.array[wp.int32],
    contour_edges: twt.Array2dInt32,
    *,
    n_vertices: int | None = None,
    twins: wp.array[wp.int32] | None = None,
) -> wp.array[wp.bool]:
    """
    Flood-fill the faces on the left of a directed contour of mesh edges.

    The dual of [`region_boundary_edges`][triwarp.selection.region_boundary_edges]: that one turns a
    face region into the seam around it, this one turns a seam back into a region. Which of the two
    sides comes back is decided **only** by the contour's direction: the seeds are the faces that
    own the halfedges ``a -> b``, and a face's corners run counter-clockwise, so reversing the rows
    returns the complement. Nothing else about the contour matters, and a closed contour on a closed
    mesh therefore partitions it exactly.

    The fill is a connected-component labelling of the face-adjacency graph with the contour's dual
    edges removed, so a contour that does *not* separate the mesh returns everything reachable --
    which is the whole surface. That is the honest answer rather than a failure, but it means the
    result is worth checking against the input's face count when the contour is meant to close.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    contour_edges
        ``(k, 2)`` ``wp.int32`` **directed** vertex pairs, each an edge of the mesh. Consecutive
        rows need not be connected: the fill only cares about which edges are blocked and which
        halfedges seed it, so several disjoint contours can be passed at once. A row that is not a
        mesh edge blocks nothing and seeds nothing.
    n_vertices
        Total vertex count, used as the key radix. When ``None`` it is inferred with
        [`array.index_bound`][triwarp.array.index_bound], which costs a host readback.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins]. Building it is the
        single largest cost here, so pass it when several contours are filled on one mesh.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` mask, ``True`` for the faces on the left of the contour, on
        ``faces.device``. All-``False`` only when no contour row is a mesh edge whose left face
        exists.

    Raises
    ------
    TypeError
        If ``contour_edges`` is not a rank-2 ``wp.int32`` array.
    ValueError
        If ``contour_edges`` does not have two columns, or if ``twins`` is given and is not a
        twin table for ``faces``
        ([`require_matching_twins`][triwarp.halfedge.require_matching_twins] states what that
        means, and checking it costs one launch and one readback).
    RuntimeError
        If ``faces``, ``contour_edges`` and ``twins`` are not all on one device.

    Examples
    --------
    ```python
    seam = tw.selection.region_boundary_edges(f, face_mask, oriented=True)
    left = tw.selection.faces_left_of_contour(f, seam)
    assert np.array_equal(left.numpy(), face_mask.numpy())
    ```

    See Also
    --------
    [`region_boundary_edges`][triwarp.selection.region_boundary_edges]
        The dual: the seam around a region, which this turns back into a region.
    [`cut_along_edges`][triwarp.seams.cut_along_edges]
        Splits the mesh along the same kind of edge set instead of labelling its sides.
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        Turns the returned mask into a mesh.
    """
    require_same_device(faces=faces, contour_edges=contour_edges, twins=twins)
    require_matching_twins(faces, twins)
    twt.ensure_edge_pairs(contour_edges, "contour_edges")
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    left = wp.zeros(n_faces, dtype=wp.bool, device=device)
    n_contour = int(contour_edges.shape[0])
    if n_faces == 0 or n_contour == 0:
        return left
    if n_vertices is None:
        n_vertices = tw.array.index_bound(faces)
    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    base = wp.uint64(n_vertices)

    # Directed contour keys, sorted, and small: the kernel probes this once or twice per halfedge,
    # so a `k`-entry binary search replaces sorting a key per halfedge to look 320 of them up.
    contour_keys = wp.empty(n_contour, dtype=wp.uint64, device=device)
    wp.launch(
        kernel_grouping.pack_directed_index_keys,
        dim=n_contour,
        inputs=[contour_edges, base, contour_keys],
        device=device,
    )
    contour_keys = tw.array.sort_and_argsort(contour_keys)[0]

    seeds = wp.zeros(n_faces, dtype=wp.bool, device=device)
    cursor = wp.zeros(1, dtype=wp.int32, device=device)
    # An interior edge emits its dual edge once, from whichever half has the lower index, so this
    # bound is exact rather than generous.
    dual_edges = twt.empty_2d((3 * n_faces // 2 + 1, 2), wp.int32, device=device)
    wp.launch(
        kernel_selection.open_dual_edges_and_seeds,
        dim=3 * n_faces,
        inputs=[faces, twins, contour_keys, base, cursor, dual_edges, seeds],
        device=device,
    )
    _n_open, (open_edges,) = tw.array.trim_to_count(cursor, dual_edges)

    labels = tw.graph.connected_component_labels_from_edges(
        twt.as_array2d(open_edges, wp.int32), node_count=n_faces, validate=False
    )
    label_seeded = wp.zeros(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_selection.mark_labels_of_seeds,
        dim=n_faces,
        inputs=[labels, seeds, label_seeded],
        device=device,
    )
    # A label names a representative face, so the per-face answer is a gather of the per-label flag.
    wp.copy(left, label_seeded[labels])
    return left


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

    Raises
    ------
    ValueError
        If ``mask`` does not have one entry per vertex.
    RuntimeError
        If ``faces``, ``mask`` and ``unique_edges`` are not all on one device.
    """
    require_same_device(faces=faces, mask=mask, unique_edges=unique_edges)
    if int(mask.shape[0]) != n_vertices:
        raise ValueError(
            f"mask must have one entry per vertex, got {mask.shape[0]} for n_vertices={n_vertices}"
        )
    device = mask.device
    if n_vertices == 0:
        return wp.clone(mask)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices, validate=False)
    # ``validate=False``: ``edges_unique`` was given ``n_vertices`` as its packing radix, so every
    # endpoint it returns is already below it.
    labels = tw.graph.connected_component_labels_from_edges(
        unique_edges, node_count=n_vertices, validate=False
    )
    keep = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_selection.keep_component_scatter,
        dim=n_vertices,
        inputs=[mask, labels, keep],
        device=device,
    )
    out = wp.empty(n_vertices, dtype=wp.bool, device=device)
    # One launch reads the component-keep flag through the vertex's own label; a Python-scope
    # gather would allocate and fill a per-vertex copy of it for a second launch to consume.
    wp.launch(
        kernel_selection.keep_selected_by_component,
        dim=n_vertices,
        inputs=[mask, labels, keep, out],
        device=device,
    )
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

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``face_indices`` are not all on one device.

    See Also
    --------
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
    [`concatenate`][triwarp.combine.concatenate]
    [`trimesh.util.submesh`][]
    """
    require_same_device(vertices=vertices, faces=faces, face_indices=face_indices)
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
        face_slots = None
    else:
        unique_face_indices, face_slots = tw.grouping.unique_1d(face_indices, return_inverse=True)

    unique_faces = tw.array.gather(faces.reshape((-1, 3)), unique_face_indices).reshape((-1,))

    unique_vertex_indices, remapped_faces = tw.grouping.unique_1d(unique_faces, return_inverse=True)
    sub_vertices = tw.array.gather(vertices, unique_vertex_indices)

    # ``face_slots`` is ``arange(k)`` exactly when ``unique_face_indices == face_indices`` (the
    # ``unique_indices=True`` branch above), which makes the gather below the identity -- skip it
    # rather than pay an allocation and a launch to reproduce ``remapped_faces`` unchanged.
    if face_slots is None:
        sub_faces = remapped_faces
    else:
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
        [`successor_cycles`][triwarp.graph.successor_cycles]). Groups must be non-empty.

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
    RuntimeError
        If ``vertices``, ``faces``, ``group_face_indices`` and ``group_offsets`` are not all on one
        device.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`split_batched`][triwarp.combine.split_batched]
    [`unique_1d`][triwarp.grouping.unique_1d]
    """
    require_same_device(
        vertices=vertices,
        faces=faces,
        group_face_indices=group_face_indices,
        group_offsets=group_offsets,
    )
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
    wp.map(
        kernel_selection.group_and_vertex_of_key,
        unique_keys,
        wp.int64(radix),
        out=[slot_groups, vertex_ids],
    )

    # Group starts from a histogram plus an exclusive scan rather than ``flatnonzero`` on the run
    # starts: no host synchronisation, and an empty group still gets a (zero-length) entry.
    # Fusing the map above into the count below is declined: the count reads the map's output at
    # its own slot, so it would fuse, but it is one launch on a path that already runs a sort and
    # a scan, and both buffers the map writes are read again further down.
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

    Raises
    ------
    ValueError
        If ``face_mask`` does not have one entry per face.
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

    See Also
    --------
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
        The complement: keep everything *outside* a region, and report the rims that opens.
    """
    require_same_device(vertices=vertices, faces=faces, face_mask=face_mask)
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )
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
    RuntimeError
        If ``vertices``, ``faces`` and ``face_mask`` are not all on one device.

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
    materialized by [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops]. The call's
    cost is dominated by the loop trace and the submesh extraction, so anything spent optimizing
    this function further belongs in
    [`triwarp.boundary.boundary_loops`][triwarp.boundary.boundary_loops] rather than here.

    See Also
    --------
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        The same extraction with the mask's polarity reversed and no loop report.
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region]
        Delete and immediately fill, which is what this is usually the first half of.
    """
    require_same_device(vertices=vertices, faces=faces, face_mask=face_mask)
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(face_mask.shape[0]) != n_faces:
        raise ValueError(
            f"face_mask must have one entry per face, got {face_mask.shape[0]} for {n_faces}"
        )

    keep_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, face_mask, out=keep_mask)
    kept_vertices, kept_faces, vertex_index = submesh_from_face_mask(
        vertices, faces, keep_mask, return_index=True
    )
    if int(kept_faces.shape[0]) == 0:
        return kept_vertices, kept_faces, []

    flat_loops, loop_offsets, loop_sizes = tw.boundary.boundary_loops_batched(
        kept_vertices, kept_faces
    )
    n_loops = int(loop_offsets.shape[0])
    if n_loops == 0:
        return kept_vertices, kept_faces, []

    # Classify every loop at once on the device. The input's own boundary edges become a sorted
    # key table, each loop's edges are packed the same way and searched in it, and a loop whose
    # every edge is present is the input's rim rather than one the deletion opened. Computed only
    # when there is something to classify: on a closed input this pass answers nothing.
    #
    # Done on the host this was a readback per loop plus a Python membership test per rim edge, so
    # its cost grew with the *loop count* as much as with the mesh -- and both tables it needed
    # (the boundary edges and the submesh-to-input vertex map) crossed the bus whole to build a
    # ``set`` and an index array the device could search in place.
    input_boundary = tw.boundary.boundary_edges(vertices, faces)
    base = wp.uint64(int(vertices.shape[0]))
    boundary_keys = wp.empty(int(input_boundary.shape[0]), dtype=wp.uint64, device=device)
    wp.launch(
        kernel_grouping.pack_directed_index_keys,
        dim=int(input_boundary.shape[0]),
        inputs=[input_boundary, base, boundary_keys],
        device=device,
    )
    # ``boundary_edges`` rows are min-first, so the directed packing above is the undirected key
    # ``pack_edge_key`` rebuilds for each rim edge.
    boundary_keys = tw.array.sort_and_argsort(boundary_keys)[0]

    is_input_rim = wp.empty(n_loops, dtype=wp.bool, device=device)
    wp.launch(
        kernel_selection.loops_are_input_rims,
        dim=n_loops,
        inputs=[
            flat_loops,
            loop_offsets,
            loop_sizes,
            vertex_index,
            boundary_keys,
            base,
            is_input_rim,
        ],
        device=device,
    )
    # One readback, of one flag per loop, where the host form read every loop back separately.
    keep_loop = is_input_rim.numpy()
    loops = tw.array.split(flat_loops, loop_offsets)
    kept = [loop for loop, rim in zip(loops, keep_loop, strict=True) if not rim]
    return kept_vertices, kept_faces, kept


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

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_indices`` are not all on one device.

    See Also
    --------
    [`face_indices_from_vertex_indices`][triwarp.selection.face_indices_from_vertex_indices]
    [`submesh_from_vertex_mask`][triwarp.selection.submesh_from_vertex_mask]
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    """
    require_same_device(vertices=vertices, faces=faces, vertex_indices=vertex_indices)
    face_indices = face_indices_from_vertex_indices(
        faces, vertex_indices, face_mode=face_mode, n_vertices=int(vertices.shape[0])
    )
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
        ``"all"`` selects faces whose three corners are all selected; ``"any"`` selects faces with
        at least one corner selected. Same rule as
        [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]'s.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertex_mask`` length does not equal ``n_vertices``, or ``face_mode`` is not
        ``"all"`` or ``"any"``.
    RuntimeError
        If ``vertices``, ``faces`` and ``vertex_mask`` are not all on one device.

    Notes
    -----
    A mask is a *cheaper* input than the equivalent index list, not merely a more convenient one:
    the face reduction reads it directly, where the index form has to rebuild membership through
    [`isin`][triwarp.array.isin] first. The index form still pays that cost, because it starts from
    indices and has nothing else to go on.

    See Also
    --------
    [`submesh_from_vertex_indices`][triwarp.selection.submesh_from_vertex_indices]
        The index form, for a caller that holds a list rather than a mask.
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
        The per-face selection this reduces onto.
    """
    require_same_device(vertices=vertices, faces=faces, vertex_mask=vertex_mask)
    n_vertices = int(vertices.shape[0])
    if int(vertex_mask.shape[0]) != n_vertices:
        raise ValueError(
            f"vertex_mask length must equal n_vertices={n_vertices}, got {vertex_mask.shape[0]}"
        )

    face_mask = _face_mask_from_vertex_mask(faces, vertex_mask, face_mode=face_mode)
    return submesh_from_face_mask(vertices, faces, face_mask)


def _face_mask_from_vertex_mask(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    *,
    face_mode: Literal["all", "any"] = "all",
) -> wp.array[wp.bool]:
    """
    Reduce a per-vertex mask onto a per-face mask under the all/any rule.

    The mask counterpart of
    [`face_indices_from_vertex_indices`][triwarp.selection.face_indices_from_vertex_indices], and
    the reason a mask input is cheaper: one kernel with three O(1) mask lookups per face, against
    that function's ``isin`` (two min/max reductions with a host readback each, then either a
    radix sort or a span-sized lookup table) plus a row reduction plus a ``flatnonzero``.
    """
    if face_mode not in ("all", "any"):
        raise ValueError(f'face_mode must be "all" or "any", got {face_mode!r}')

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    out_face_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    if n_faces == 0:
        return out_face_mask

    wp.launch(
        kernel_selection.face_mask_from_vertex_mask,
        dim=n_faces,
        inputs=[faces, vertex_mask, wp.bool(face_mode == "all"), out_face_mask],
        device=device,
    )
    return out_face_mask


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

    Raises
    ------
    RuntimeError
        If ``faces``, ``mask`` and ``unique_edges`` are not all on one device.

    See Also
    --------
    [`shrink_vertex_mask`][triwarp.selection.shrink_vertex_mask]

    Notes
    -----
    Dilation is vertex-based: a vertex enters the mask when any 1-ring neighbour is in it.

    The per-round ``wp.clone`` is not worth removing. The kernel only *sets* bits, so each round
    must start from a copy of the previous mask; ping-ponging two preallocated buffers would keep
    the copy and drop only the allocation, which is a small fraction of the round's cost.
    """
    require_same_device(faces=faces, mask=mask, unique_edges=unique_edges)
    device = mask.device
    n = int(mask.shape[0])
    if hops <= 0 or n == 0:
        return wp.clone(mask)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n, validate=False)
    m = int(unique_edges.shape[0])
    current = mask
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

    Raises
    ------
    RuntimeError
        If ``faces``, ``mask`` and ``unique_edges`` are not all on one device.

    See Also
    --------
    [`expand_vertex_mask`][triwarp.selection.expand_vertex_mask]

    Notes
    -----
    Erosion here is vertex-based: a vertex survives when every 1-ring neighbour is also in the
    mask. MeshLab's Erode Selection is a *face*-based operation and gives a different answer.
    """
    require_same_device(faces=faces, mask=mask, unique_edges=unique_edges)
    device = mask.device
    n = int(mask.shape[0])
    if hops <= 0 or n == 0:
        return wp.clone(mask)
    if unique_edges is None:
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n, validate=False)
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
    n_vertices: int | None = None,
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
    n_vertices
        Optional vertex count, forwarded to [`isin`][triwarp.array.isin] as its ``max_index``.
        Supplying it skips the two min/max reductions and the two host readbacks that would
        otherwise infer the value span -- worth roughly half of the membership test. Must be
        greater than every index in ``faces`` and in ``vertex_indices``.

    Returns
    -------
    wp.array[wp.int32]
        Selected face indices on ``faces.device``. Empty when no faces match or
        ``vertex_indices`` is empty.

    Raises
    ------
    ValueError
        If ``face_mode`` is not ``"all"`` or ``"any"``.
    RuntimeError
        If ``faces`` and ``vertex_indices`` are not all on one device.
    """
    require_same_device(faces=faces, vertex_indices=vertex_indices)
    if face_mode not in ("all", "any"):
        raise ValueError(f'face_mode must be "all" or "any", got {face_mode!r}')

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if int(vertex_indices.shape[0]) == 0 or n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    faces2d = faces.reshape((-1, 3))
    corner_hit = tw.array.isin(faces2d, vertex_indices, max_index=n_vertices)
    if face_mode == "all":
        face_hit = tw.reduce.all(corner_hit, axis=1)
    else:
        face_hit = tw.reduce.any(corner_hit, axis=1)
    return tw.array.flatnonzero(face_hit)
