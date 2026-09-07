"""
Assembling meshes from parts and splitting them back apart.

[`concatenate`][triwarp.combine.concatenate] joins whole meshes into one buffer pair without
touching their geometry, and [`split`][triwarp.combine.split] /
[`split_batched`][triwarp.combine.split_batched] decompose one back into its connected components.
Mirrors ``trimesh.util.concatenate`` and ``trimesh.Trimesh.split``.

Nothing here welds anything: ``concatenate`` packs parts side by side and leaves any seam open.
Joining two open meshes *across* one boundary loop each is [`stitch`][triwarp.holes.stitch] and its
variants, in [`triwarp.holes`][triwarp.holes], where the minimum-weight triangulation machinery
they share with the hole fillers lives.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import warp as wp

import triwarp as tw
from triwarp._device import require_same_device
from triwarp.kernels import array as kernel_array
from triwarp.kernels import combine as kernel_combine


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

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Combined vertices and reindexed faces on the shared device. An empty sequence carries no
        device to share, so the empty result is allocated on Warp's **current** device.

    See Also
    --------
    [`split`][triwarp.combine.split]
        The inverse, by connected component.
    [`concatenate`][triwarp.array.concatenate]
        The buffer-level operation of the same name, which joins plain 1-D arrays and does no
        reindexing. Both names are required: this one mirrors
        [`trimesh.util.concatenate`][], that one [`numpy.concatenate`][].
    [`trimesh.util.concatenate`][]
    """
    if len(meshes_data) == 0:
        return wp.empty(0, dtype=wp.vec3), wp.empty(0, dtype=wp.int32)

    device = meshes_data[0][0].device
    vertex_counts = [int(vertices.shape[0]) for vertices, _ in meshes_data]

    if sum(vertex_counts) == 0:
        concatenated_vertices = wp.empty(0, dtype=wp.vec3, device=device)
        vertex_offsets = wp.zeros(len(meshes_data), dtype=wp.int32, device=device)
    else:
        concatenated_vertices, vertex_offsets = tw.array.pack_1d_arrays(
            [vertices for vertices, _ in meshes_data]
        )

    concatenated_faces, piece_starts = tw.array.pack_1d_arrays([faces for _, faces in meshes_data])
    total_indices = int(concatenated_faces.shape[0])
    if total_indices > 0:
        wp.launch(
            kernel_combine.offset_packed_faces,
            dim=total_indices,
            inputs=[piece_starts, vertex_offsets, concatenated_faces],
            device=device,
        )

    return concatenated_vertices, concatenated_faces


def split(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, copy: bool = False
) -> list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]:
    """
    Split a mesh into connected components by face adjacency.

    Each returned pair is a compact ``(vertices, faces)`` submesh with vertices
    reindexed from zero, matching [`trimesh.graph.split`][] with
    ``only_watertight=False``. [`concatenate`][triwarp.combine.concatenate] on the
    result recovers the input mesh (up to vertex/face ordering within each body).

    All components are extracted in one batched pass
    ([`split_batched`][triwarp.combine.split_batched]); this is the slicing wrapper over it.

    !!! note "The returned arrays are views"
        Each pair slices the two shared buffers ``split_batched`` produced, which costs no device
        memory and no launches. Two consequences: holding on to a single component keeps *both*
        whole buffers alive, and writing into one component writes into the shared allocation. Pass
        ``copy=True`` for independent buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    copy
        Return independent buffers instead of views into the batched result. One clone per
        component, so the returned list no longer pins the batched buffers.

    Returns
    -------
    list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]]
        One ``(vertices, faces)`` pair per face-connected component on
        ``vertices.device``. Empty when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`split_batched`][triwarp.combine.split_batched]
    [`concatenate`][triwarp.combine.concatenate]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`repair.remove_small_components`][triwarp.repair.remove_small_components]
        Keep a *subset* of the components in one mesh, rather than taking them all apart.
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`trimesh.graph.split`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    vertices_all, vertex_offsets, faces_all, face_offsets = split_batched(vertices, faces)
    k = int(vertex_offsets.shape[0])
    if k == 0:
        return []

    vertex_bounds = [*vertex_offsets.list(), int(vertices_all.shape[0])]
    face_bounds = [*face_offsets.list(), int(faces_all.shape[0]) // 3]

    meshes: list[tuple[wp.array[wp.vec3], wp.array[wp.int32]]] = []
    for (v_begin, v_end), (f_begin, f_end) in zip(
        itertools.pairwise(vertex_bounds), itertools.pairwise(face_bounds), strict=True
    ):
        # A ``slice`` index builds a zero-copy ``wp.array`` view: pure Python, no device work.
        component = (vertices_all[v_begin:v_end], faces_all[3 * f_begin : 3 * f_end])
        meshes.append((wp.clone(component[0]), wp.clone(component[1])) if copy else component)
    return meshes


def split_batched(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Split a mesh into connected components, returned as two CSR buffers.

    Same decomposition as [`split`][triwarp.combine.split] — same components, same vertex and face
    order — but with no per-component Python at all, which is the only truly ``O(1)``-in-``k``
    form. Prefer it when the component count is large or when the components feed straight back
    into another batched kernel.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.

    Returns
    -------
    vertices_all : wp.array[wp.vec3]
        Every component's compacted vertices, concatenated.
    vertex_offsets : wp.array[wp.int32]
        Length-``k`` start of each component in ``vertices_all`` (no terminator; the last
        component runs to the end).
    faces_all : wp.array[wp.int32]
        Every component's reindexed flat faces, concatenated.
    face_offsets : wp.array[wp.int32]
        Length-``k`` start of each component in ``faces_all`` **in faces, not indices**: component
        ``g`` owns ``faces_all[3 * face_offsets[g] : 3 * face_offsets[g + 1]]``.

    All four arrays are empty (and ``k == 0``) when ``n_faces == 0``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`split`][triwarp.combine.split]
    [`submeshes_from_face_groups`][triwarp.selection.submeshes_from_face_groups]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_int32 = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_int32, empty_int32, empty_int32

    face_labels = tw.adjacency.face_connected_component_labels(faces)

    sorted_labels, sorted_face_ids = tw.array.sort_and_argsort(face_labels)

    # Segment boundaries of the label-sorted array: position 0, plus every label change. The
    # change test is the adjacent-element map over two shifted views of the same buffer; the
    # single-face guard is required because Warp rejects a zero-length slice outright.
    is_start = wp.empty(n_faces, dtype=wp.bool, device=device)
    is_start[:1].fill_(True)
    if n_faces > 1:
        wp.map(kernel_array.not_equal, sorted_labels[1:], sorted_labels[:-1], out=is_start[1:])
    face_offsets = tw.array.flatnonzero(is_start)

    if int(face_offsets.shape[0]) == 1:
        component_vertices, component_faces = tw.selection.submesh_from_face_indices(
            vertices, faces, sorted_face_ids, unique_indices=True
        )
        zero = wp.zeros(1, dtype=wp.int32, device=device)
        return component_vertices, zero, component_faces, zero

    vertices_all, vertex_offsets, faces_all = tw.selection.submeshes_from_face_groups(
        vertices, faces, sorted_face_ids, face_offsets
    )
    return vertices_all, vertex_offsets, faces_all, face_offsets
