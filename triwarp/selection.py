"""Mesh concatenation and face-subset extraction (Warp)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import warp as wp
import triwarp as tw
from triwarp.kernels import selection as kernel_selection


def _gather_faces(faces: wp.array[wp.int32], face_indices: wp.array[wp.int32]) -> wp.array[wp.int32]:
    k = int(face_indices.shape[0])
    if k == 0:
        return wp.empty(0, dtype=wp.int32, device=faces.device)
    gathered = faces.reshape((-1, 3))[face_indices]
    out = wp.empty((k, 3), dtype=wp.int32, device=faces.device)
    wp.copy(out, gathered)
    return out.reshape((-1,))


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


def submesh_from_face_indices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_indices: wp.array[wp.int32],
    *,
    unique_indices: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract a face subset by index and reindex vertices from zero.

    Gathers the selected face triplets, compacts referenced vertices, and remaps
    face indices into the compact vertex buffer. Matches the core geometry step of
    :func:`trimesh.util.submesh` (without visuals, repair, watertight filtering, or
    append).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        :func:`triwarp.graph.face_adjacency`).
    face_indices
        1D ``wp.int32`` array of face indices into the source mesh
        (``0 .. n_faces - 1``), on the same device as ``vertices``.
    unique_indices
        If ``True``, ``face_indices`` is assumed to contain no duplicates and
        the deduplication pass is skipped.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``. When
        ``face_indices`` is empty, both arrays have length ``0``.

    Raises
    ------
    ValueError
        If ``vertices``, ``faces``, and ``face_indices`` live on different devices.

    See Also
    --------
    :func:`submesh_from_face_mask`
    :func:`submesh_from_vertex_indices`
    :func:`concatenate`
    :func:`trimesh.util.submesh`
    """
    device = vertices.device
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")
    if face_indices.device != device:
        raise ValueError(
            f"face_indices must live on the same device as vertices, got {face_indices.device} and {device}"
        )

    k = int(face_indices.shape[0])
    if k == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)

    if unique_indices:
        unique_face_indices = face_indices
        face_slots = wp.array(range(k), dtype=wp.int32, device=device)
    else:
        unique_face_indices, face_slots = tw.unique.unique_1d(face_indices, return_inverse=True)

    unique_faces = _gather_faces(faces, unique_face_indices)

    unique_vertex_indices, remapped_faces = tw.unique.unique_1d(unique_faces, return_inverse=True)
    n_unique = int(unique_vertex_indices.shape[0])
    sub_vertices = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.copy(sub_vertices, vertices[unique_vertex_indices])

    sub_faces = _gather_faces(remapped_faces, face_slots)

    return sub_vertices, sub_faces


def submesh_from_face_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
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

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    ValueError
        If array devices differ.

    See Also
    --------
    :func:`submesh_from_face_indices`
    """
    device = vertices.device
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")
    if face_mask.device != device:
        raise ValueError(f"face_mask must live on the same device as vertices, got {face_mask.device} and {device}")

    face_indices = tw.array.flatnonzero(face_mask)
    return submesh_from_face_indices(vertices, faces, face_indices, unique_indices=True)


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
    :func:`face_indices_for_vertices`
    :func:`submesh_from_vertex_mask`
    :func:`submesh_from_face_indices`
    """
    device = vertices.device
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")
    if vertex_indices.device != device:
        raise ValueError(
            f"vertex_indices must live on the same device as vertices, got {vertex_indices.device} and {device}"
        )

    face_indices = face_indices_for_vertices(faces, vertex_indices, face_mode=face_mode)
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
        Passed to :func:`submesh_from_vertex_indices`.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Compact ``(sub_vertices, sub_faces)`` on ``vertices.device``.

    Raises
    ------
    ValueError
        If array devices differ or ``vertex_mask`` length does not equal ``n_vertices``.

    See Also
    --------
    :func:`submesh_from_vertex_indices`
    """
    n_vertices = int(vertices.shape[0])
    device = vertices.device
    if faces.device != device:
        raise ValueError(f"vertices and faces must live on the same device, got {device} and {faces.device}")
    if vertex_mask.device != device:
        raise ValueError(f"vertex_mask must live on the same device as vertices, got {vertex_mask.device} and {device}")
    if int(vertex_mask.shape[0]) != n_vertices:
        raise ValueError(f"vertex_mask length must equal n_vertices={n_vertices}, got {vertex_mask.shape[0]}")

    vertex_indices = tw.array.flatnonzero(vertex_mask)
    return submesh_from_vertex_indices(vertices, faces, vertex_indices, face_mode=face_mode)


def face_indices_for_vertices(
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
    if vertex_indices.device != device:
        raise ValueError(
            f"vertex_indices must live on the same device as faces, got {vertex_indices.device} and {device}"
        )

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
