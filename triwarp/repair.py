"""
Mesh repair utilities (libigl unreferenced/duplicated vertex and duplicated face cleanup).

See [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices],
[`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices],
[`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces],
[`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces], and
[`collapse_small_triangles`][triwarp.repair.collapse_small_triangles].

Two further defects need geometry rather than topology to detect, so they have their own detector:
[`bad_face_mask`][triwarp.repair.bad_face_mask] flags faces that are too thin, misoriented against
their neighbourhood, or folded back over it.
[`remove_folded_faces`][triwarp.repair.remove_folded_faces] deletes the folded ones and
[`remove_t_vertices`][triwarp.repair.remove_t_vertices] flips away the slivers a T-junction leaves
behind.
"""

from __future__ import annotations

import math

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.grouping import hash_vector_rows, unique_1d, unique_faces, unique_rows
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import repair as kernel_repair
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import triangles as kernel_triangles


def remove_unreferenced_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_inverse: bool = False
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Remove vertices not referenced by any face and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Flat triangle index buffer.
    return_inverse
        If ``True``, also return ``inverse`` with ``new_vertices[inverse]`` sourcing
        ``vertices``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Referenced vertices only, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Face buffer with indices remapped into ``new_vertices``.
    remap : wp.array[wp.int32]
        Length ``n_vertices`` old-to-new map (``-1`` when unreferenced).
    inverse : wp.array[wp.int32], optional
        Present when ``return_inverse=True``.
    """
    device = faces.device
    n_vertices = int(vertices.shape[0])

    referenced = tw.array.indices_to_mask(faces, n_vertices, device=device)

    inverse = tw.array.flatnonzero(referenced)
    remap = wp.full(n_vertices, wp.int32(-1), dtype=wp.int32, device=device)
    n_referenced = int(tw.reduce.sum(referenced))
    if n_referenced > 0:
        wp.launch(
            kernel_scatter.scatter_index, dim=n_referenced, inputs=[inverse, remap], device=device
        )

    new_vertices = (
        tw.array.gather(vertices, inverse)
        if int(inverse.shape[0]) > 0
        else wp.empty(0, dtype=wp.vec3, device=vertices.device)
    )
    new_faces = tw.array.remap_indices(faces, remap)

    if return_inverse:
        return new_vertices, new_faces, remap, inverse
    return new_vertices, new_faces, remap


def remove_duplicated_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 0.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Merge duplicate vertex positions up to a coordinate tolerance and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Flat triangle index buffer.
    epsilon
        Uniqueness tolerance. Positive values snap coordinates to ``round(v / epsilon)``, so the
        tolerance is absolute and is the one you chose. ``0`` instead groups by a *relative*
        bucket about ``2.4e-4`` wide (see Notes) — pass an explicit ``epsilon`` unless that is
        what you want.

    Returns
    -------
    unique_vertices : wp.array[wp.vec3]
        Deduplicated vertex positions (first occurrence per equivalence class).
    unique_indices : wp.array[wp.int32]
        Length ``n_unique``. Original indices into ``vertices`` for each output row.
    inverse : wp.array[wp.int32]
        Length ``n_vertices``. Maps each input vertex to its slot in ``unique_vertices``.
    unique_faces : wp.array[wp.int32]
        Face buffer with indices remapped into ``unique_vertices``.

    Notes
    -----
    Both tolerance modes quantize positions and group by cell, so both separate a pair straddling a
    cell boundary however close it is. That is worth knowing about ``epsilon=0`` in particular,
    which is **not** an exact-equality test despite requiring no tolerance: it buckets by the high
    bits of each coordinate's ``float32`` representation, giving a *relative* cell about ``2.4e-4``
    wide. So ``1.0`` and ``1.000244`` merge, while two adjacent ``float32`` values on either side of
    a bucket edge do not. Prefer an explicit ``epsilon`` whenever the tolerance matters; use ``0``
    only to collapse positions that are already bitwise equal, which it does reliably (including
    across ``+0.0`` / ``-0.0``).

    Duplicates that are known from construction rather than measured are better collapsed directly:
    see [`revolve`][triwarp.creation.revolve], which derives them from its profile instead of
    hashing positions.

    See Also
    --------
    [`duplicate_vertex_inverse`][triwarp.repair.duplicate_vertex_inverse]
    [`hash_vector_rows`][triwarp.grouping.hash_vector_rows]
    """
    inverse = duplicate_vertex_inverse(vertices, epsilon)
    n = int(inverse.shape[0])
    n_unique = int(tw.reduce.max(inverse)) + 1
    unique_indices = wp.full(n_unique, wp.int32(n), dtype=wp.int32, device=inverse.device)
    wp.launch(
        kernel_edges.scatter_first_occurrence,
        dim=n,
        inputs=[inverse, unique_indices],
        device=inverse.device,
    )
    unique_vertices = tw.array.gather(vertices, unique_indices)
    unique_faces = tw.array.remap_indices(faces, inverse)
    return unique_vertices, unique_indices, inverse, unique_faces


def duplicate_vertex_inverse(vertices: wp.array[wp.vec3], epsilon: float) -> wp.array[wp.int32]:
    """
    Map each vertex to the slot of its coincident-vertex equivalence class.

    The inverse map produced by welding vertices at ``epsilon`` tolerance, without also
    computing the deduplicated vertex/face buffers — useful for remapping per-vertex
    attributes (colors, UVs, ...) to match a [`remove_duplicated_vertices`]
    [triwarp.repair.remove_duplicated_vertices] call made with the same ``epsilon``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    epsilon
        Uniqueness tolerance, with the same meaning as in
        [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]: positive values
        snap coordinates to ``round(v / epsilon)``, while ``0`` groups by a *relative* bucket about
        ``2.4e-4`` wide rather than testing for equality.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_vertices``. Maps each input vertex to its slot in the deduplicated set.

    See Also
    --------
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]
    [`hash_vector_rows`][triwarp.grouping.hash_vector_rows]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    if epsilon > 0.0:
        row_keys = hash_vector_rows(vertices, epsilon=epsilon)
        _, inverse = unique_1d(row_keys, return_inverse=True)
    else:
        rows = twt.empty_float32_2d((n, 3), device=device)
        wp.utils.array_cast(vertices, rows)
        _, inverse = unique_rows(rows, return_inverse=True)
    return inverse


def resolve_duplicated_faces(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Resolve duplicated triangles by orientation-aware cancellation rules (libigl).

    For each geometric duplicate:
    - equal positive and negative counts: remove all copies;
    - one extra positive copy: keep one positively oriented face;
    - one extra negative copy: keep one negatively oriented face;
    - otherwise raise ``ValueError`` when counts are not orientable.

    Parameters
    ----------
    faces
        Flat triangle index buffer.

    Returns
    -------
    resolved_faces : wp.array[wp.int32]
        Flat buffer of kept faces.
    kept_indices : wp.array[wp.int32]
        Original face indices into the input ``faces`` buffer.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    faces2d = faces.reshape((-1, 3))
    unique_faces_wp, inverse = unique_faces(faces, return_inverse=True)
    num_unique = int(unique_faces_wp.shape[0]) // 3

    # Per-group orientation stats scattered on device: member/signed counts plus the smallest
    # member index of each sign class (seeded with the ``n_faces`` sentinel).
    member_count = wp.zeros(num_unique, dtype=wp.int32, device=device)
    signed_count = wp.zeros(num_unique, dtype=wp.int32, device=device)
    first_member = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_positive = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_negative = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.scatter_duplicate_face_stats,
        dim=n_faces,
        inputs=[
            faces,
            unique_faces_wp,
            inverse,
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
        ],
        device=device,
    )

    keep = wp.empty(num_unique, dtype=wp.int32, device=device)
    error_group = wp.full(1, num_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.resolve_duplicate_groups,
        dim=num_unique,
        inputs=[
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
            keep,
            error_group,
        ],
        device=device,
    )
    first_error = int(error_group.numpy()[0])
    if first_error < num_unique:
        count = int(signed_count[first_error : first_error + 1].numpy()[0])
        raise ValueError(
            f"resolve_duplicated_faces: non-orientable duplicate face group {first_error} "
            f"with signed count {count}"
        )

    # Compact kept decisions in ascending group order (matches the reference emission order).
    keep_mask = wp.empty(num_unique, dtype=wp.bool, device=device)
    wp.map(kernel_array.greater_equal, keep, wp.int32(0), out=keep_mask)
    kept_slots = tw.array.flatnonzero(keep_mask)
    if int(kept_slots.shape[0]) == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    kept_wp = tw.array.gather(keep, kept_slots)
    resolved = tw.array.gather(faces2d, kept_wp).reshape((-1,))
    return resolved, kept_wp


def remove_degenerate_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop degenerate (zero-area) triangles and reindex, keeping vertex positions unchanged.

    Mirrors ``trimesh.Trimesh.nondegenerate_faces`` + ``update_faces``: a face is degenerate when
    two of its vertices coincide or its three vertices are collinear, detected by
    [`nondegenerate`][triwarp.triangles.nondegenerate] (both triangle altitudes exceed the merge
    tolerance). Surviving faces are unchanged; vertices left unreferenced after the drop are
    removed by the reindexing in
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask].

    Unlike [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles], no vertices are
    merged and no edges are collapsed: this only removes faces already degenerate in the input.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced by a non-degenerate face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the non-degenerate faces, remapped into ``new_vertices``.

    See Also
    --------
    [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles]
    [`nondegenerate`][triwarp.triangles.nondegenerate]
    [`trimesh.triangles.nondegenerate`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)

    keep_mask = tw.triangles.nondegenerate(vertices, faces)
    return tw.selection.submesh_from_face_mask(vertices, faces, keep_mask)


def remove_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove faces touching a non-manifold (>2-incident) edge, iterating until edge-manifold.

    Each pass keeps only faces whose three edges are each used by at most two faces
    ([`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]); dropping a face can make a
    neighbour manifold, so it repeats up to ``max_iter`` times (matching MeshLib's bounded
    hole-complicating-face removal loop).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_iter
        Maximum number of removal passes.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced after non-manifold faces are dropped, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the surviving (edge-manifold, up to ``max_iter`` passes) faces.

    See Also
    --------
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]
    """
    for _ in range(max_iter):
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break
        keep = tw.validation.edge_manifold_mask(faces, allow_boundary_edges=True)
        kept = tw.array.flatnonzero(keep)
        if int(kept.shape[0]) == n_faces:
            break  # already edge-manifold
        vertices, faces = tw.selection.submesh_from_face_mask(vertices, faces, keep)
    return vertices, faces


def collapse_small_triangles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 1e-6
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Collapse triangles smaller than a bounding-box-relative area threshold (libigl).

    Mirrors ``igl::collapse_small_triangles``. A triangle is *small* when its doubled area is below
    ``2 * epsilon * bbd ** 2``, where ``bbd`` is the diagonal of the axis-aligned bounding box of
    ``vertices``. Each small triangle has its **shortest edge** collapsed by merging that edge's two
    endpoints; the merged face (now carrying a repeated vertex) is discarded. The process repeats to
    a fixpoint, so triangles that only become small after a neighbouring collapse are also removed.

    This subsumes degenerate-triangle removal: an exactly degenerate face (zero area) is always
    below the threshold, so passing a small ``epsilon`` removes it. To drop only degenerate faces
    without any bounding-box-relative collapsing, use
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    epsilon
        Relative area tolerance. The doubled-area threshold is ``2 * epsilon * bbd ** 2``; larger
        values collapse more (and larger) triangles.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices surviving the collapse, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the surviving faces, remapped into ``new_vertices``.

    See Also
    --------
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`nondegenerate`][triwarp.triangles.nondegenerate]

    Notes
    -----
    Where ``igl::collapse_small_triangles`` merges vertices by a sequentially updated index map and
    recurses until no edge collapses, this resolves all shortest-edge merges of one pass at once via
    the connected-components closure of
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges],
    then loops over the shrinking mesh. Both converge to a mesh with no sub-threshold triangle. The
    surviving vertex of a collapsed edge keeps the position of the component representative (the
    lowest original index in its class) rather than libigl's longest-edge-preserving endpoint; for
    sub-threshold triangles the two endpoints are close enough that the difference is negligible.
    The bounding-box diagonal is measured once on the input ``vertices`` so the threshold is fixed
    across iterations.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0 or int(vertices.shape[0]) == 0:
        return wp.clone(vertices), wp.clone(faces)

    bbd = tw.proximity._default_mesh_query_max_dist(vertices)
    min_dbl_area = wp.float32(2.0 * epsilon * bbd * bbd)

    current_vertices = vertices
    current_faces = faces
    max_iterations = int(faces.shape[0])  # bounded: each collapsing pass drops at least one face
    for _ in range(max_iterations):
        n_current = int(current_faces.shape[0]) // 3
        if n_current == 0:
            break

        pairs = twt.empty_int32_2d((n_current, 2), device=device)
        flag = wp.empty(n_current, dtype=wp.int32, device=device)
        wp.launch(
            kernel_repair.small_triangle_collapse_edges,
            dim=n_current,
            inputs=[current_vertices, current_faces, min_dbl_area, pairs, flag],
            device=device,
        )

        if int(tw.reduce.sum(flag)) == 0:
            break

        # Non-flagged faces emit a self-pair (i0, i0); these are self-loops that leave the
        # connected-components closure unchanged, so all rows can be passed without filtering.
        n_vertices = int(current_vertices.shape[0])
        labels = tw.graph.connected_component_labels_from_edges(pairs, node_count=n_vertices)

        unique_labels, inverse = unique_1d(labels, return_inverse=True)
        n_unique = int(unique_labels.shape[0])
        unique_indices = wp.full(n_unique, wp.int32(n_vertices), dtype=wp.int32, device=device)
        wp.launch(
            kernel_edges.scatter_first_occurrence,
            dim=n_vertices,
            inputs=[inverse, unique_indices],
            device=device,
        )
        class_vertices = tw.array.gather(current_vertices, unique_indices)
        remapped_faces = tw.array.remap_indices(current_faces, inverse)

        keep_mask = tw.triangles.nondegenerate(class_vertices, remapped_faces)
        current_vertices, current_faces = tw.selection.submesh_from_face_mask(
            class_vertices, remapped_faces, keep_mask
        )

    return current_vertices, current_faces


def make_winding_consistent(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Flip faces so every shared edge is traversed in opposite directions by its two faces.

    Reuses the orientation flood-fill of
    [`face_orientation_mask`][triwarp.validation.face_orientation_mask] (one arbitrary seed
    face per connected component) and reverses the winding of every face whose orientation bit is
    set. The result satisfies
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]; an already-consistent
    mesh is returned unchanged. Mirrors ``trimesh.repair.fix_winding``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with corrected winding, on ``faces.device``. Vertices are unchanged.

    See Also
    --------
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]
    [`face_orientation_mask`][triwarp.validation.face_orientation_mask]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]

    Notes
    -----
    The reference winding within each connected component is arbitrary (the seed face keeps its
    orientation), matching ``trimesh.repair.fix_winding``'s BFS. Use
    [`make_volume`][triwarp.repair.make_volume] afterwards to also orient normals outward.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    orient, _, _, _ = tw.validation.face_orientation_bits(faces)
    out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.flip_faces_masked,
        dim=n_faces,
        inputs=[faces, orient, out_faces],
        device=device,
    )
    return out_faces


def make_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Orient faces so the mesh encloses a positive signed volume (normals point outward).

    Mirrors ``trimesh.repair.fix_inversion``. With ``multibody=False`` (default) the mesh is only
    corrected when it is watertight (every undirected edge shared by exactly two faces) and its
    total signed volume is negative, in which case every face is reversed. With ``multibody=True``
    each connected component is corrected independently by the sign of its own signed volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    multibody
        When ``True`` correct each connected component independently rather than the mesh as a
        whole.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with outward-oriented normals, on ``faces.device``. Vertices are
        unchanged.

    See Also
    --------
    [`is_volume`][triwarp.validation.is_volume]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]

    Notes
    -----
    The signed volume is ``sum(dot(v0, cross(v1, v2)) / 6)`` measured from the origin, as in
    [`is_volume`][triwarp.validation.is_volume]. Unlike ``trimesh.repair.fix_inversion``'s
    multibody path, this does not skip components that are not watertight/consistently wound: an
    open component's signed volume is ill-defined and may be flipped spuriously. Run
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] first (see
    [`make_normals_outward`][triwarp.repair.make_normals_outward]) and reserve ``multibody``
    for meshes whose bodies are individually closed.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    signed_volumes = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_triangles.signed_tet_volumes,
        dim=n_faces,
        inputs=[vertices, faces, wp.vec3(0.0, 0.0, 0.0), signed_volumes],
        device=device,
    )

    if multibody:
        labels = tw.adjacency.face_connected_component_labels(faces)
        accum = wp.zeros(n_faces, dtype=wp.float32, device=device)
        wp.launch(
            kernel_scatter.scatter_add_scalar,
            dim=n_faces,
            inputs=[signed_volumes, labels, accum],
            device=device,
        )
        flip = wp.empty(n_faces, dtype=wp.int32, device=device)
        # ``accum[labels]`` gathers each face's component volume (Python-scope gather).
        wp.map(kernel_repair.negative_volume_flag, accum[labels], out=flip)
        wp.launch(
            kernel_repair.flip_faces_masked,
            dim=n_faces,
            inputs=[faces, flip, out_faces],
            device=device,
        )
        return out_faces

    watertight = bool(tw.reduce.all(tw.validation.face_watertight_mask(faces)))
    if watertight and tw.reduce.sum(signed_volumes) < 0.0:
        flip = wp.full(n_faces, wp.int32(1), dtype=wp.int32, device=device)
        wp.launch(
            kernel_repair.flip_faces_masked,
            dim=n_faces,
            inputs=[faces, flip, out_faces],
            device=device,
        )
        return out_faces

    wp.copy(out_faces, faces)
    return out_faces


def make_normals_outward(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Make winding consistent and orient normals outward (winding fix followed by inversion fix).

    Equivalent to ``trimesh.repair.fix_normals``: first
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] gives every connected
    component a coherent winding, then [`make_volume`][triwarp.repair.make_volume] flips it (or each
    body, with ``multibody=True``) so normals point outward. On a watertight, orientable mesh the
    result satisfies [`is_volume`][triwarp.validation.is_volume].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    multibody
        Forwarded to [`make_volume`][triwarp.repair.make_volume]: correct each connected component
        independently.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with consistent winding and outward normals, on ``faces.device``.
        Vertices are unchanged.

    See Also
    --------
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_volume`][triwarp.repair.make_volume]
    [`is_volume`][triwarp.validation.is_volume]
    """
    wound = make_winding_consistent(faces)
    return make_volume(vertices, wound, multibody=multibody)


def bad_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_quality: float | None = 0.02,
    max_normal_angle: float | None = None,
    max_fold_angle: float | None = None,
) -> wp.array[wp.bool]:
    """
    Flag faces that are thin, misoriented relative to their neighbourhood, or folded over it.

    MeshLab's ``compute_selection_bad_faces``, and the detector behind
    [`remove_folded_faces`][triwarp.repair.remove_folded_faces]. The three criteria are independent
    and a face is bad if *any* enabled one fires; each is disabled by passing ``None``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    min_quality
        Flag a face whose ``radius_ratio``
        ([`face_quality`][triwarp.triangles.face_quality]) is below this. ``0`` is fully degenerate
        and ``1`` is equilateral, so this is a *thinness* gate; MeshLab's ``aratio``, whose default
        of ``0.02`` is this one. ``None`` disables it.
    max_normal_angle
        Flag a face whose normal is more than this many **degrees** from the direction of the sum of
        its edge-neighbours' normals — the local consensus. This catches a single face inserted the
        wrong way round in an otherwise consistent patch. MeshLab's ``nfratio`` (default ``60``,
        off by default). ``None`` disables it.
    max_fold_angle
        Flag a face that meets *some* neighbour at more than this many **degrees** — a fold, where
        the two triangles lie almost on top of each other with opposing normals. Of the two faces at
        such an edge only the one facing *against* its own wider neighbourhood is flagged, since
        only one of them is the mistake; a face whose neighbours give it no consensus (an isolated
        face, or a strip of exactly three) is therefore never flagged on this criterion alone.
        MeshLab's ``folded_faces_angle_threshold`` (default ``160``, off by default). Must be in
        ``(0, 180]``. ``None`` disables it.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` mask on ``faces.device``; ``True`` marks a bad face. All-``False`` when
        every criterion is disabled.

    Raises
    ------
    ValueError
        If ``max_normal_angle`` or ``max_fold_angle`` is outside ``(0, 180]``.

    See Also
    --------
    [`remove_folded_faces`][triwarp.repair.remove_folded_faces]
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`triwarp.triangles.face_quality`][triwarp.triangles.face_quality]
    """
    for name, angle in (("max_normal_angle", max_normal_angle), ("max_fold_angle", max_fold_angle)):
        if angle is not None and not 0.0 < angle <= 180.0:
            raise ValueError(f"{name} must be in (0, 180] degrees, got {angle}")

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    out_bad = wp.zeros(n_faces, dtype=wp.bool, device=device)
    if n_faces == 0:
        return out_bad

    quality = (
        tw.triangles.face_quality(vertices, faces, metric="radius_ratio")
        if min_quality is not None
        else wp.full(n_faces, 1.0, dtype=wp.float32, device=device)
    )
    face_normals, _areas = tw.triangles.face_normals_and_areas(vertices, faces)
    neighbor_sum = wp.zeros(n_faces, dtype=wp.vec3, device=device)
    max_angle = wp.zeros(n_faces, dtype=wp.float32, device=device)
    if max_normal_angle is not None or max_fold_angle is not None:
        adjacency = tw.adjacency.face_adjacency(faces)
        if int(adjacency.shape[0]) > 0:
            angles = tw.adjacency.face_adjacency_angles(
                vertices, faces, face_adjacency=adjacency, face_normals=face_normals
            )
            wp.launch(
                kernel_repair.accumulate_neighbor_normals,
                dim=int(adjacency.shape[0]),
                inputs=[face_normals, adjacency, angles, neighbor_sum, max_angle],
                device=device,
            )

    # -2 is unreachable for a cosine and -1 for the normalized quality, so a disabled criterion
    # simply never fires and the kernel needs no per-criterion flag.
    wp.launch(
        kernel_repair.bad_face_mask,
        dim=n_faces,
        inputs=[
            quality,
            face_normals,
            neighbor_sum,
            max_angle,
            wp.float32(min_quality if min_quality is not None else -1.0),
            wp.float32(
                math.cos(math.radians(max_normal_angle)) if max_normal_angle is not None else -2.0
            ),
            wp.float32(
                math.cos(math.radians(max_fold_angle)) if max_fold_angle is not None else -2.0
            ),
            out_bad,
        ],
        device=device,
    )
    return out_bad


def remove_folded_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, angle: float = 160.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop faces that fold back over their own ring, and reindex.

    A folded face is one whose dihedral angle to a neighbour is near ``pi``: the two triangles lie
    almost on top of each other with opposite normals, which is what a badly reconstructed or
    self-intersecting patch looks like locally. Such a face contributes no surface and breaks every
    normal-based computation downstream, so removing it is a repair rather than a simplification.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**; a face with a neighbour above it is dropped. MeshLab's
        ``folded_faces_angle_threshold``, whose default of ``160`` is this one. Must be in
        ``(0, 180]``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced by a kept face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the kept faces, remapped into ``new_vertices``.

    See Also
    --------
    [`bad_face_mask`][triwarp.repair.bad_face_mask]
    [`remove_t_vertices`][triwarp.repair.remove_t_vertices]
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]

    Notes
    -----
    MeshLab's ``meshing_remove_folded_faces`` *flips* the offending edge instead of deleting the
    face, which preserves the face count but can only help when the fold is a triangulation mistake
    rather than genuinely folded geometry. Deletion is the choice the rest of this module makes (see
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces] and
    [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces]), and it leaves a hole
    that [`triwarp.hole_filling`][triwarp.hole_filling] can retriangulate properly.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)
    folded = bad_face_mask(vertices, faces, min_quality=None, max_fold_angle=angle)
    keep = wp.empty(n_faces, dtype=wp.bool, device=faces.device)
    wp.map(kernel_array.mask_not, folded, out=keep)
    return tw.selection.submesh_from_face_mask(vertices, faces, keep)


def remove_t_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    threshold: float = 40.0,
    max_iter: int = 10,
) -> wp.array[wp.int32]:
    """
    Repair T-vertices by flipping the long edge of each sliver they create.

    A **T-vertex** is a vertex that sits in the interior of a neighbouring triangle's edge rather
    than at one of its corners — the classic symptom of two patches stitched at different
    resolutions. The vertex is topologically fine, but the triangle opposite it is a sliver: its
    apex lies (nearly) on the far edge, which sends its circumradius-to-inradius ratio to infinity
    and makes every cotangent weight, normal and curvature estimate around it unusable.

    The repair is a flip, not a deletion: flipping the sliver's long edge moves the diagonal off the
    T and leaves two well-shaped triangles, with the same vertices and the same face count.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Never modified.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    threshold
        Aspect ratio above which a triangle counts as a T-vertex sliver, in the
        ``aspect_ratio`` sense of [`face_quality`][triwarp.triangles.face_quality] (``1`` is
        equilateral, unbounded above). MeshLab's ``meshing_remove_t_vertices`` threshold, whose
        default of ``40`` is this one. Must be positive.
    max_iter
        Maximum number of parallel flip passes. Each pass commits a conflict-free independent set of
        flips; MeshLab's ``repeat=True`` is the same idea serially.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer with the slivers re-triangulated, on ``faces.device`` (a copy; the input is
        not modified).

    See Also
    --------
    [`triwarp.remesh.flip_by_objective`][triwarp.remesh.flip_by_objective]
    [`remove_folded_faces`][triwarp.repair.remove_folded_faces]
    [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles]

    Notes
    -----
    A flip cannot fix a T-vertex on the mesh **boundary** or on a non-manifold edge, because there
    is no second triangle to flip against. MeshLab offers an edge *collapse* method for that case;
    here the equivalent is [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles],
    which removes the sliver by merging its short edge instead.
    """
    return tw.remesh.flip_by_objective(
        vertices, faces, objective="t_vertex", aspect_threshold=threshold, max_iter=max_iter
    )
