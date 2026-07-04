"""
Mesh repair utilities (libigl unreferenced/duplicated vertex and duplicated face cleanup).

See [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices],
[`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices], and
[`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces].
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import repair as kernel_repair
from triwarp.kernels import sample as kernel_sample
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
            kernel_array.mark_membership_mask_bounded,
            dim=n_indices,
            inputs=[faces, wp.int32(n_vertices), referenced],
            device=device,
        )

    inverse = tw.array.flatnonzero(referenced)
    remap = wp.full(n_vertices, wp.int32(-1), dtype=wp.int32, device=device)
    n_referenced = int(tw.reduce.sum(referenced))
    if n_referenced > 0:
        wp.launch(
            kernel_array.scatter_index, dim=n_referenced, inputs=[inverse, remap], device=device
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
    if n_faces == 0:
        empty = wp.empty(0, dtype=wp.int32, device=faces.device)
        return empty, empty

    faces2d = faces.reshape((-1, 3))
    unique_faces_wp, inverse = unique_faces(faces, return_inverse=True)

    faces_np = faces2d.numpy()
    unique_np = unique_faces_wp.numpy().reshape(-1, 3)
    inverse_np = inverse.numpy()

    num_unique = int(unique_np.shape[0])
    kept: list[int] = []
    for ui in range(num_unique):
        member = np.flatnonzero(inverse_np == ui)
        urow = unique_np[ui]
        signed_ids: list[int] = []
        count = 0
        for fi in member:
            row = faces_np[fi]
            consistent = (
                (row[0] == urow[0] and row[1] == urow[1] and row[2] == urow[2])
                or (row[0] == urow[1] and row[1] == urow[2] and row[2] == urow[0])
                or (row[0] == urow[2] and row[1] == urow[0] and row[2] == urow[1])
            )
            signed = int(fi + 1) if consistent else -int(fi + 1)
            signed_ids.append(signed)
            count += 1 if consistent else -1

        if member.size == 1:
            kept.append(int(member[0]))
            continue
        if count == 1:
            for fid in signed_ids:
                if fid > 0:
                    kept.append(fid - 1)
                    break
        elif count == -1:
            for fid in signed_ids:
                if fid < 0:
                    kept.append(-fid - 1)
                    break
        elif count == 0:
            continue
        else:
            raise ValueError(
                f"resolve_duplicated_faces: non-orientable duplicate face group {ui} "
                f"with signed count {count}"
            )

    device = faces.device
    if len(kept) == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    kept_wp = wp.array(np.asarray(kept, dtype=np.int32), dtype=wp.int32, device=device)
    resolved = tw.array.gather(faces2d, kept_wp).reshape((-1,))
    return resolved, kept_wp


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
    [`make_normals_consistent`][triwarp.repair.make_normals_consistent]

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
    [`make_normals_consistent`][triwarp.repair.make_normals_consistent]

    Notes
    -----
    The signed volume is ``sum(dot(v0, cross(v1, v2)) / 6)`` measured from the origin, as in
    [`is_volume`][triwarp.characteristics.is_volume]. Unlike ``trimesh.repair.fix_inversion``'s
    multibody path, this does not skip components that are not watertight/consistently wound: an
    open component's signed volume is ill-defined and may be flipped spuriously. Run
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] first (see
    [`make_normals_consistent`][triwarp.repair.make_normals_consistent]) and reserve ``multibody``
    for meshes whose bodies are individually closed.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    signed_volumes = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_sample.signed_tet_volumes,
        dim=n_faces,
        inputs=[vertices, faces, wp.vec3(0.0, 0.0, 0.0), signed_volumes],
        device=device,
    )

    if multibody:
        labels = tw.graph.face_connected_component_labels(faces)
        accum = wp.zeros(n_faces, dtype=wp.float32, device=device)
        wp.launch(
            kernel_repair.accumulate_component_volume,
            dim=n_faces,
            inputs=[labels, signed_volumes, accum],
            device=device,
        )
        flip = wp.empty(n_faces, dtype=wp.int32, device=device)
        wp.launch(
            kernel_repair.mark_negative_component,
            dim=n_faces,
            inputs=[labels, accum, flip],
            device=device,
        )
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


def make_normals_consistent(
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
