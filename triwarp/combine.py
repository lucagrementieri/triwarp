"""
Assembling meshes from parts and splitting them back apart.

[`concatenate`][triwarp.combine.concatenate] and [`split`][triwarp.combine.split] combine and
decompose whole meshes by connected component. [`stitch`][triwarp.combine.stitch] (and its
``_min_weight`` / ``_nicely`` variants) instead **joins two open meshes** across one boundary
loop each into a single watertight seam — see [`triwarp.hole_filling`][triwarp.hole_filling] for
the lower-level triangulation engines these build on, and for closing holes within a single mesh.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import warp as wp

import triwarp as tw
from triwarp.kernels import array as kernel_array


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
    [`split`][triwarp.combine.split]
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

    vertex_offset = 0
    dest_offset = 0
    for count, (_, faces) in zip(vertex_counts, meshes_data, strict=True):
        f = int(faces.shape[0])
        if f > 0:
            wp.map(
                wp.add,
                faces,
                wp.int32(vertex_offset),
                out=concatenated_faces[dest_offset : dest_offset + f],
            )
            dest_offset += f
        vertex_offset += count

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
    ValueError
        If ``vertices`` and ``faces`` live on different devices.

    See Also
    --------
    [`split_batched`][triwarp.combine.split_batched]
    [`concatenate`][triwarp.combine.concatenate]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]
    [`trimesh.graph.split`][]
    """
    vertices_all, vertex_offsets, faces_all, face_offsets = split_batched(vertices, faces)
    k = int(vertex_offsets.shape[0])
    if k == 0:
        return []

    vertex_bounds = [*vertex_offsets.numpy().tolist(), int(vertices_all.shape[0])]
    face_bounds = [*face_offsets.numpy().tolist(), int(faces_all.shape[0]) // 3]

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

    See Also
    --------
    [`split`][triwarp.combine.split]
    [`submeshes_from_face_groups`][triwarp.selection.submeshes_from_face_groups]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_int32 = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_int32, empty_int32, empty_int32

    face_labels = tw.adjacency.face_connected_component_labels(faces)

    # One stable label sort replaces the per-component isin/flatnonzero full-array passes: the
    # sort is stable, so faces stay ascending within each component and components ascend by
    # label — the exact emission order of the previous per-label loop.
    labels_buffer = wp.empty(2 * n_faces, dtype=wp.int32, device=device)
    wp.copy(labels_buffer, face_labels, count=n_faces)
    face_ids = tw.array.init_sort_pair_indices(n_faces, -1, device)
    wp.utils.radix_sort_pairs(labels_buffer, face_ids, count=n_faces)
    sorted_labels = wp.clone(labels_buffer[:n_faces])
    sorted_face_ids = wp.clone(face_ids[:n_faces])

    # Segment boundaries of the label-sorted array: position 0, plus every label change. The
    # change test is the adjacent-element map over two shifted views of the same buffer; the
    # single-face guard is required because Warp rejects a zero-length slice outright.
    is_start = wp.empty(n_faces, dtype=wp.bool, device=device)
    is_start[:1].fill_(True)
    if n_faces > 1:
        wp.map(kernel_array.not_equal, sorted_labels[1:], sorted_labels[:-1], out=is_start[1:])
    face_offsets = tw.array.flatnonzero(is_start)

    if int(face_offsets.shape[0]) == 1:
        # Overwhelmingly the common call. Packing one group through the int64 key would be pure
        # overhead, and on a big single-component mesh it is a materially larger sort.
        component_vertices, component_faces = tw.selection.submesh_from_face_indices(
            vertices, faces, sorted_face_ids, unique_indices=True
        )
        zero = wp.zeros(1, dtype=wp.int32, device=device)
        return component_vertices, zero, component_faces, zero

    vertices_all, vertex_offsets, faces_all = tw.selection.submeshes_from_face_groups(
        vertices, faces, sorted_face_ids, face_offsets
    )
    return vertices_all, vertex_offsets, faces_all, face_offsets


def stitch(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes into one watertight mesh.

    Extracts the single boundary loop of each mesh
    ([`boundary_loops`][triwarp.boundary.boundary_loops]) and joins them with
    [`triangulate_boundaries`][triwarp.hole_filling.triangulate_boundaries]. Each mesh must have
    exactly one boundary loop (the promesh assumption); use ``triangulate_boundaries`` directly to
    join specific loops of multi-boundary meshes. **No smoothing** is applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices (larger-boundary mesh first), on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the bridge triangles, on ``faces_a.device``.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices.

    See Also
    --------
    [`triangulate_boundaries`][triwarp.hole_filling.triangulate_boundaries]
    [`stitch_min_weight`][triwarp.combine.stitch_min_weight]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return tw.hole_filling.triangulate_boundaries(
        vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0]
    )


def stitch_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes with a minimum-weight band (MeshLib ``stitchHoles``).

    Like [`stitch`][triwarp.combine.stitch] but joins the two rims with the metric-minimizing band
    of [`triangulate_boundaries_min_weight`][triwarp.hole_filling.triangulate_boundaries_min_weight]
    instead of the greedy correspondence. Each mesh must have exactly one boundary loop.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`triangulate_boundaries_min_weight`][triwarp.hole_filling.triangulate_boundaries_min_weight].
    up_dir
        Up direction for the ``"vertical"`` metric.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the band triangles.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices, or ``metric``
        is unknown.

    See Also
    --------
    [`triangulate_boundaries_min_weight`][triwarp.hole_filling.triangulate_boundaries_min_weight]
    [`stitch`][triwarp.combine.stitch]
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch_min_weight requires each mesh to have exactly one boundary loop (>=3 verts); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return tw.hole_filling.triangulate_boundaries_min_weight(
        vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0], metric, up_dir
    )


def stitch_nicely(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Stitch two open meshes with a smooth, refined band (MeshLib ``stitchHolesNicely``).

    Like [`stitch_min_weight`][triwarp.combine.stitch_min_weight] but the connecting band is then
    subdivided and smoothed by the same finisher as
    [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely]. Each mesh must have exactly one
    boundary loop. The cross-boundary smooth solve is always applied (MeshLib forces ``smoothBd``).

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`triangulate_boundaries_min_weight`][triwarp.hole_filling.triangulate_boundaries_min_weight].
    up_dir
        Up direction for the ``"vertical"`` stitch metric.
    triangulate_only
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    max_edge
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    max_edge_splits
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    max_angle_change_after_flip
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    smooth_curvature
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    natural_smooth
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    edge_weights
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].
    return_patch
        See [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely].

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices followed by any inserted band vertices, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the band faces.
    patch_mask : wp.array[wp.bool]
        Only when ``return_patch`` is ``True``: mask of the band faces.

    Raises
    ------
    ValueError
        If either mesh lacks exactly one boundary loop, or ``metric`` / ``edge_weights`` is unknown.

    See Also
    --------
    [`stitch_min_weight`][triwarp.combine.stitch_min_weight]
    [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely]
    """
    if metric not in tw.hole_filling._STITCH_METRIC_IDS:
        raise ValueError(
            f"metric must be one of {sorted(tw.hole_filling._STITCH_METRIC_IDS)}, got {metric!r}"
        )
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")

    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch_nicely requires each mesh to have exactly one boundary loop (>=3 verts); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )

    device = faces_a.device
    n_faces_before = (int(faces_a.shape[0]) + int(faces_b.shape[0])) // 3
    n_vertices_before = int(vertices_a.shape[0]) + int(vertices_b.shape[0])
    combined_vertices, combined_faces = tw.hole_filling.triangulate_boundaries_min_weight(
        vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0], metric, up_dir
    )
    n_faces_after = int(combined_faces.shape[0]) // 3
    patch_mask = tw.hole_filling._patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (combined_vertices, combined_faces)
        return (*result, patch_mask) if return_patch else result

    rim_loops = [
        wp.array(loops_a[0].numpy(), dtype=wp.int32, device=device),
        wp.array(loops_b[0].numpy() + int(vertices_a.shape[0]), dtype=wp.int32, device=device),
    ]
    target_edge = (
        max_edge
        if max_edge is not None
        else tw.hole_filling._mean_rim_edge_length(combined_vertices, rim_loops)
    )
    new_vertices, new_faces, out_patch = tw.hole_filling._finish_nicely(
        combined_vertices,
        combined_faces,
        n_vertices_before,
        patch_mask,
        target_edge,
        max_edge_splits,
        max_angle_change_after_flip,
        smooth_curvature,
        True,  # stitchHolesNicely forces smoothBd
        natural_smooth,
        edge_weights,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)
