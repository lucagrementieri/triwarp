"""
Mesh repair utilities (libigl unreferenced/duplicated vertex and duplicated face cleanup).

See [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices],
[`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices],
[`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces],
[`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces], and
[`collapse_small_triangles`][triwarp.repair.collapse_small_triangles].
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import repair as kernel_repair
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import triangles as kernel_triangles
from triwarp.unique import hash_vector_rows, unique_1d, unique_faces, unique_rows


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
    n_indices = int(faces.shape[0])

    referenced = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    if n_indices > 0:
        wp.launch(
            kernel_scatter.mark_membership_mask,
            dim=n_indices,
            inputs=[faces, wp.int32(n_vertices), referenced],
            device=device,
        )

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
    new_faces = _remap_flat_indices(faces, remap)

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
        Uniqueness tolerance. ``0`` requires exact match (via row hashing). Positive values
        round coordinates to ``round(v / epsilon)`` before deduplication.

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
    """
    inverse = _duplicate_vertex_inverse(vertices, epsilon)
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
    unique_faces = _remap_flat_indices(faces, inverse)
    return unique_vertices, unique_indices, inverse, unique_faces


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

    bbd = tw.proximity.default_mesh_query_max_dist(vertices)
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
        remapped_faces = _remap_flat_indices(current_faces, inverse)

        keep_mask = tw.triangles.nondegenerate(class_vertices, remapped_faces)
        current_vertices, current_faces = tw.selection.submesh_from_face_mask(
            class_vertices, remapped_faces, keep_mask
        )

    return current_vertices, current_faces


def make_winding_consistent(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Flip faces so every shared edge is traversed in opposite directions by its two faces.

    Reuses the orientation flood-fill of
    [`face_orientation_mask`][triwarp.characteristics.face_orientation_mask] (one arbitrary seed
    face per connected component) and reverses the winding of every face whose orientation bit is
    set. The result satisfies
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]; an already-consistent
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
    [`is_winding_consistent`][triwarp.characteristics.is_winding_consistent]
    [`face_orientation_mask`][triwarp.characteristics.face_orientation_mask]
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

    orient, _, _, _ = tw.characteristics._orientation_bits(faces)
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
    [`is_volume`][triwarp.characteristics.is_volume]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]

    Notes
    -----
    The signed volume is ``sum(dot(v0, cross(v1, v2)) / 6)`` measured from the origin, as in
    [`is_volume`][triwarp.characteristics.is_volume]. Unlike ``trimesh.repair.fix_inversion``'s
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
        labels = tw.graph.face_connected_component_labels(faces)
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

    watertight = bool(tw.reduce.all(tw.characteristics.watertight_face_mask(faces)))
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
    result satisfies [`is_volume`][triwarp.characteristics.is_volume].

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
    [`is_volume`][triwarp.characteristics.is_volume]
    """
    wound = make_winding_consistent(faces)
    return make_volume(vertices, wound, multibody=multibody)


def _duplicate_vertex_inverse(vertices: wp.array[wp.vec3], epsilon: float) -> wp.array[wp.int32]:
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


def _remap_flat_indices(
    indices: wp.array[wp.int32], remap: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    n = int(indices.shape[0])
    device = indices.device
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)
    out = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.gather_1d_skip_negative, dim=n, inputs=[indices, remap, out], device=device
    )
    return out
