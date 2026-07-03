"""
Mesh repair utilities (libigl ``remove_unreferenced_vertices``, ``remove_duplicated_vertices``,
``resolve_duplicated_faces``).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
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
            kernel_array.scatter_index,
            dim=n_referenced,
            inputs=[inverse, remap],
            device=device,
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
    wp.launch(kernel_array.gather_1d_skip_negative, dim=n, inputs=[indices, remap, out], device=device)
    return out
