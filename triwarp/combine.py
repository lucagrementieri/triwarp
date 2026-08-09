"""
Assembling meshes from parts and splitting them back apart.

[`concatenate`][triwarp.combine.concatenate] and [`split`][triwarp.combine.split] combine and
decompose whole meshes by connected component. [`stitch`][triwarp.combine.stitch] (and its
``_min_weight`` / ``_smooth`` variants) instead **joins two open meshes** across one boundary
loop each into a single watertight seam — see [`triwarp.holes`][triwarp.holes] for
the lower-level triangulation engines these build on, and for closing holes within a single mesh.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.holes import _BAD_TRIANGULATION_METRIC, _EdgeTable, _PackedLoops
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
        An empty sequence yields empty arrays on ``cpu``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        Combined vertices and reindexed faces on the shared device.

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
        # ``pack_1d_arrays`` already returns the exclusive scan of the segment sizes, on the
        # device and of exactly this length -- the renumbering offsets, for free. Recomputing them
        # with ``itertools.accumulate`` and uploading the result was a second copy of the same
        # numbers.
        concatenated_vertices, vertex_offsets = tw.array.pack_1d_arrays(
            [vertices for vertices, _ in meshes_data]
        )

    # Renumbering used to be one ``wp.map`` per input mesh, which put ~32 us of host-side launch
    # marshalling on every piece — the dominant cost of the call once there were more than a
    # handful. The packing copy is unavoidable (Warp has no gather across separate allocations),
    # but the renumbering collapses into a single launch over the packed buffer.
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
    # sort is stable (see tw.array.sort_and_argsort's Notes), so faces stay ascending within each
    # component and components ascend by label — the exact emission order of the previous
    # per-label loop.
    # Views into ``sort_and_argsort``'s scratch, deliberately not cloned: neither escapes this
    # frame. ``sorted_labels`` feeds only the adjacent-element map below, and ``sorted_face_ids``
    # only the submesh builders, which read it into fresh buffers.
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
    [`stitch_loops`][triwarp.combine.stitch_loops]. Each mesh must have
    exactly one boundary loop (the promesh assumption); use ``stitch_loops`` directly to
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
    [`stitch_loops`][triwarp.combine.stitch_loops]
    [`stitch_min_weight`][triwarp.combine.stitch_min_weight]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch"
    )
    return stitch_loops(vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b)


def stitch_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes with a minimum-weight band.

    Like [`stitch`][triwarp.combine.stitch] but joins the two rims with the metric-minimizing band
    of [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight]
    instead of the greedy correspondence. Each mesh must have exactly one boundary loop.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight].
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
    [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight]
    [`stitch`][triwarp.combine.stitch]

    Notes
    -----
    The band is MeshLib's ``stitchHoles``, and its metrics are that function's.
    """
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_min_weight"
    )
    return stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )


def stitch_smooth(
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
    Stitch two open meshes with a smooth, refined band.

    Like [`stitch_min_weight`][triwarp.combine.stitch_min_weight] but the connecting band is then
    subdivided and smoothed by the same finisher as
    [`fill_smooth`][triwarp.holes.fill_smooth]. Each mesh must have exactly one
    boundary loop. The cross-boundary smooth solve is always applied (MeshLib forces ``smoothBd``).

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight].
    up_dir
        Up direction for the ``"vertical"`` stitch metric.
    triangulate_only
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_edge
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_edge_splits
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_angle_change_after_flip
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    smooth_curvature
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    natural_smooth
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    edge_weights
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    return_patch
        See [`fill_smooth`][triwarp.holes.fill_smooth].

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
    [`fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    The three-stage pipeline is MeshLib's ``stitchHolesNicely``.
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")

    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_smooth"
    )

    device = faces_a.device
    n_faces_before = (int(faces_a.shape[0]) + int(faces_b.shape[0])) // 3
    n_vertices_before = int(vertices_a.shape[0]) + int(vertices_b.shape[0])
    combined_vertices, combined_faces = stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )
    n_faces_after = int(combined_faces.shape[0]) // 3
    patch_mask = tw.holes._patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (combined_vertices, combined_faces)
        return (*result, patch_mask) if return_patch else result

    # Independent copies, not views: ``boundary_loops`` returns slices of one shared packed buffer
    # and ``_mean_rim_edge_length`` must not alias it. ``wp.clone`` says exactly that; the round
    # trip through ``.numpy()`` these used to take said nothing and crossed the bus twice. The
    # second loop's shift into the combined numbering is elementwise, so it maps on the device.
    shifted_loop_b = wp.empty(int(loop_b.shape[0]), dtype=wp.int32, device=device)
    wp.map(wp.add, loop_b, wp.int32(int(vertices_a.shape[0])), out=shifted_loop_b)
    rim_loops = [wp.clone(loop_a), shifted_loop_b]
    target_edge = (
        max_edge
        if max_edge is not None
        else tw.holes._mean_rim_edge_length(combined_vertices, rim_loops)
    )
    new_vertices, new_faces, out_patch = tw.smoothing.refine_and_smooth_region(
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


def _single_boundary_loops(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    caller: str,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return the one boundary loop of each mesh, or raise naming ``caller``.

    The shared precondition of the whole ``stitch*`` family: each mesh must have exactly one hole
    to zip to the other's. Loops shorter than three vertices are degenerate rather than holes and
    are dropped before counting, so a mesh carrying one is rejected on its real loop count.
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            f"{caller} requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return loops_a[0], loops_b[0]


def stitch_loops(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes by triangulating the band between one boundary loop on each.

    The two rims are zippered into a watertight seam by a greedy, order-preserving correspondence
    (the Warp port of promesh's ``triangulate_boundaries``): every A-edge is matched to the B
    vertex minimizing the added-triangle perimeter, the association is rotated so its shortest
    pair comes first, and a longest-increasing-subsequence correction forces the matching to be
    monotone (non-self-intersecting). **No smoothing or refinement** is applied — the seam reuses
    only the two loops' existing vertices, adding ``len(loop_a) + len(loop_b)`` bridge triangles.

    The larger loop is treated as A (the meshes are swapped internally when
    ``len(loop_a) < len(loop_b)``), so the output is invariant to argument order. Both loops must
    wind following the face orientation, as returned by
    [`boundary_loops`][triwarp.boundary.boundary_loops]; the two boundaries are assumed to wind in
    opposite directions (as two open meshes facing each other do), so loop A is reversed to align
    them.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each
        mesh, indexing ``vertices_a`` / ``vertices_b`` respectively.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenation of ``vertices_a`` then ``vertices_b`` (larger loop first), on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the bridge triangles, on
        ``faces_a.device``.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices.

    See Also
    --------
    [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight]
        The more robust choice, and the one to reach for when seam quality matters: it minimizes a
        triangulation metric over the whole rim pair instead of correcting a greedy correspondence.
    [`stitch`][triwarp.combine.stitch]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`concatenate`][triwarp.combine.concatenate]

    Notes
    -----
    The correspondence and its monotonicity correction are inherently sequential and run on the
    host over the length-``len(loop_a)`` association array; the O(N·M) perimeter matrix, the
    reductions, and the triangle emission run in Warp kernels, and the perimeter matrix itself is
    never copied off the device.

    **This is the fast one, and that is the reason to keep it.** It is metric-free and makes one
    O(N·M) pass, where
    [`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight] fills the same size table
    with ``N + M`` sequential launches along the anti-diagonals. Measured on two facing cylinder
    rims (RTX 5090, min of 5), the DP costs **5.0x** at a 100-vertex rim, **7.7x** at 1 000 and
    **8.9x** at 4 000; on CPU, 5.2x / 7.5x / 7.2x. The gap widens with rim size, so the DP is never
    the cheaper option -- prefer it for the seam it produces, not for speed.
    """
    device = faces_a.device
    n = int(loop_a.shape[0])
    m = int(loop_b.shape[0])
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n
    if m < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n} and {m}")
    n_vertices_a = int(vertices_a.shape[0])

    # Reverse loop A so both rims wind the same way, then take rim positions.
    flipped_loop_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.cyclic_gather,
        dim=n,
        inputs=[loop_a, wp.int32(n), wp.int32(0), wp.bool(True), wp.int32(0), flipped_loop_a],
        device=device,
    )
    a_pos = tw.array.gather(vertices_a, flipped_loop_a)
    b_pos = tw.array.gather(vertices_b, loop_b)

    # perimeters[i, j] = |a_i - b_j| + |a_{i+1} - b_j| for A-edge i and B-vertex j.
    perimeters = twt.empty_2d((n, m), wp.float32, device=device)
    wp.launch(
        kernel_combine.boundary_perimeters,
        dim=(n, m),
        inputs=[a_pos, b_pos, wp.int32(n), perimeters],
        device=device,
    )

    col_min = wp.empty(n, dtype=wp.int32, device=device)
    val_min = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_combine.row_argmin,
        dim=n,
        inputs=[perimeters, wp.int32(m), col_min, val_min],
        device=device,
    )

    shift = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    shift_np = shift.numpy()
    shift_a = int(shift_np[0])
    shift_b = int(shift_np[1])

    edge_dev = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.rolled_edge_map,
        dim=n,
        inputs=[col_min, wp.int32(shift_a), wp.int32(shift_b), wp.int32(n), wp.int32(m), edge_dev],
        device=device,
    )
    edge = edge_dev.numpy()

    row_roll = shift_a
    col_roll = shift_b

    # Wraparound-group fix: if the lowest-index group is split across the ends of ``edge``, roll
    # A so the group is contiguous (promesh's correction before the monotonicity pass).
    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        if trailing > 0:
            edge = np.roll(edge, trailing)
            row_roll = (shift_a - trailing) % n

    # Monotonicity correction: re-pick the B vertex of every edge outside the longest
    # non-decreasing subsequence within the bracket of its stable neighbours, so the matching is
    # order-preserving. The sentinel value ``m`` closes the last bracket.
    edge_ext = np.append(edge, np.int32(m))
    out_edge = wp.array(edge_ext.astype(np.int32), dtype=wp.int32, device=device)
    if not np.all(np.diff(edge) >= 0):
        unsorted_indices = _non_increasing_indices(edge_ext)
        stable_indices = np.delete(np.arange(edge_ext.size), unsorted_indices)
        next_indices = stable_indices[np.searchsorted(stable_indices, unsorted_indices)]
        wp.launch(
            kernel_combine.resolve_corrections,
            dim=1,
            inputs=[
                perimeters,
                wp.array(unsorted_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(next_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.int32(int(unsorted_indices.size)),
                wp.int32(row_roll),
                wp.int32(col_roll),
                wp.int32(n),
                wp.int32(m),
                out_edge,
            ],
            device=device,
        )

    # Rolled loops referencing the concatenated vertex buffer (A first, then B offset).
    roll_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.cyclic_gather,
        dim=n,
        inputs=[
            flipped_loop_a,
            wp.int32(n),
            wp.int32(row_roll),
            wp.bool(False),
            wp.int32(0),
            roll_a,
        ],
        device=device,
    )
    roll_b = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.cyclic_gather,
        dim=m,
        inputs=[
            loop_b,
            wp.int32(m),
            wp.int32(col_roll),
            wp.bool(False),
            wp.int32(n_vertices_a),
            roll_b,
        ],
        device=device,
    )

    bridge_a = wp.empty(3 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.bridge_a_faces,
        dim=n,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), bridge_a],
        device=device,
    )
    bridge_b = wp.empty(3 * m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.bridge_b_faces,
        dim=m,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), wp.int32(m), bridge_b],
        device=device,
    )

    combined_vertices, combined_faces = tw.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, tw.array.concatenate([combined_faces, bridge_a, bridge_b])


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][triwarp.combine._non_increasing_indices]); the algorithm does
    not handle ties. Mirrors promesh's private helper of the same name.
    """
    p = np.zeros_like(numbers, dtype=np.int64)
    m = -np.ones(numbers.size + 1, dtype=np.int64)
    size = 0
    for i, x in enumerate(numbers):
        subseq_size = int(np.searchsorted(numbers[m[1 : size + 1]], x))
        p[i] = m[subseq_size]
        m[subseq_size + 1] = i
        size = max(subseq_size + 1, size)
    subseq = np.empty(size, numbers.dtype)
    k = m[size]
    for i in range(size - 1, -1, -1):
        subseq[i] = numbers[k]
        k = p[k]
    return subseq


def _non_increasing_indices(numbers: np.ndarray) -> np.ndarray:
    """
    Return the entry indices **not** in the longest non-decreasing subsequence of ``numbers``.

    Repeated integers are perturbed by adding ``linspace(0, 1, count, endpoint=False)`` within
    each equal-value group, so a run of equal values is treated as (weakly) increasing and kept.
    Mirrors promesh's private helper (with a NumPy grouping in place of ``trimesh.grouping.group``
    to avoid a runtime ``trimesh`` dependency).
    """
    different = numbers.astype(np.float64)
    order = np.argsort(numbers, kind="stable")
    sorted_values = numbers[order]
    cut = np.flatnonzero(np.diff(sorted_values)) + 1
    for group in np.split(order, cut):
        different[group] += np.linspace(0.0, 1.0, num=len(group), endpoint=False)
    subsequence = _longest_increasing_subsequence(different)
    absence_mask = np.isin(different, subsequence, assume_unique=True, invert=True)
    return np.flatnonzero(absence_mask)


# Stitch-metric name -> kernel selector (must match the METRIC_*_STITCH constants in
# kernels/holes.py).
_STITCH_METRIC_IDS = {"complex_stitch": 0, "edge_length_stitch": 1, "vertical": 2}


def _closest_loop_pair(a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3]) -> tuple[int, int]:
    """
    Return the closest vertex pair ``(i, j)`` between the two rims (MeshLib's start pair).

    The full ``(n_a, n_b)`` squared-distance matrix and its argmin run in Warp kernels (the same
    ``row_argmin`` / ``global_argmin`` reduction the greedy zippering uses); only the two winning
    indices come back to the host. Ties resolve to the smallest ``i`` then smallest ``j``, matching
    ``numpy.argmin`` on the flattened matrix.
    """
    device = a_pos.device
    n_a = int(a_pos.shape[0])
    n_b = int(b_pos.shape[0])
    dist_sq = twt.empty_2d((n_a, n_b), wp.float32, device=device)
    wp.launch(
        kernel_combine.pair_sq_distances,
        dim=(n_a, n_b),
        inputs=[a_pos, b_pos, dist_sq],
        device=device,
    )
    col_min = wp.empty(n_a, dtype=wp.int32, device=device)
    val_min = wp.empty(n_a, dtype=wp.float32, device=device)
    wp.launch(
        kernel_combine.row_argmin,
        dim=n_a,
        inputs=[dist_sq, wp.int32(n_b), col_min, val_min],
        device=device,
    )
    pair = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_combine.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n_a), pair],
        device=device,
    )
    pair_np = pair.numpy()
    return int(pair_np[0]), int(pair_np[1])


def stitch_loops_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes with a **minimum-weight** cylindrical band between one boundary loop on each.

    Ports MeshLib's ``stitchHoles``: the two rims are aligned at their closest vertex pair and
    zippered by the band of ``len(loop_a) + len(loop_b)`` triangles that minimizes a stitch metric,
    found by a grid dynamic program over the two loops (``dp[i, j]`` = best band consuming ``i``
    A-edges and ``j`` B-edges; each anti-diagonal is one parallel kernel launch). Reuses only the
    loops' existing vertices. The metric-free greedy
    [`stitch_loops`][triwarp.combine.stitch_loops] remains available.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each.
    metric
        Stitch metric to minimize (MeshLib ``MRMeshMetrics.cpp``):

        - ``"complex_stitch"`` (default) — triangle aspect ratio plus a dihedral-smoothness edge
          term between adjacent band triangles and the surface (``getComplexStitchMetric``).
        - ``"edge_length_stitch"`` — summed connection-edge length (``getEdgeLengthStitchMetric``).
        - ``"vertical"`` — penalizes band area and normal deviation from ``up_dir``
          (``getVerticalStitchMetric``); pass ``up_dir``.
    up_dir
        Up direction for the ``"vertical"`` metric (defaults to ``(0, 0, 1)``); ignored otherwise.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenation of ``vertices_a`` then ``vertices_b``, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the band triangles.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices, or ``metric`` is unknown.

    See Also
    --------
    [`stitch_loops`][triwarp.combine.stitch_loops]
    [`stitch_min_weight`][triwarp.combine.stitch_min_weight]
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")
    n_a = int(loop_a.shape[0])
    n_b = int(loop_b.shape[0])
    if n_a < 3 or n_b < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n_a} and {n_b}")

    device = faces_a.device
    metric_id = _STITCH_METRIC_IDS[metric]
    offset = int(vertices_a.shape[0])

    # Reverse loop A so the two rims wind oppositely (facing), then align both at the closest pair.
    # ``la`` / ``lb`` stay on the host for the sequential band traceback, but the closest-pair
    # search, rim gathers and rim-opposite lookups all run on device.
    # Measured, and deliberately left alone: this preamble (two readbacks, the closest-pair gather,
    # the host roll and two more uploads) is 5.0% of the call at a 100-vertex rim on CUDA and falls
    # to 0.7% at 1 000 and 0.4% at 4 000 -- the anti-diagonal DP below dominates and grows faster.
    # Folding the roll into ``cyclic_gather`` would buy a shrinking fraction of a noise floor.
    la = loop_a.numpy()[::-1].copy()
    lb = loop_b.numpy().copy()
    a_rim = tw.array.gather(vertices_a, wp.array(la, dtype=wp.int32, device=device))
    b_rim = tw.array.gather(vertices_b, wp.array(lb, dtype=wp.int32, device=device))
    start_a, start_b = _closest_loop_pair(a_rim, b_rim)
    la = np.roll(la, -start_a)
    lb = np.roll(lb, -start_b)

    la_wp = wp.array(la, dtype=wp.int32, device=device)
    lb_wp = wp.array(lb, dtype=wp.int32, device=device)
    a_pos = tw.array.gather(vertices_a, la_wp)
    b_pos = tw.array.gather(vertices_b, lb_wp)
    table_a = _EdgeTable(vertices_a, faces_a, tw.edges.faces_to_edges(faces_a, sorted=True))
    table_b = _EdgeTable(vertices_b, faces_b, tw.edges.faces_to_edges(faces_b, sorted=True))
    # The stitch DP works on exactly one rim per side, so each rim is its own one-loop batch.
    a_opp, a_opp_valid = table_a.rim_opposite(_PackedLoops(la_wp, np.array([n_a], dtype=np.int64)))
    b_opp, b_opp_valid = table_b.rim_opposite(_PackedLoops(lb_wp, np.array([n_b], dtype=np.int64)))
    up = wp.vec3(*(up_dir if up_dir is not None else (0.0, 0.0, 1.0)))

    dp = twt.as_array2d(
        wp.full((n_a + 1, n_b + 1), _BAD_TRIANGULATION_METRIC, dtype=wp.float32, device=device),
        wp.float32,
    )
    wp.launch(kernel_combine.set_dp_origin, dim=1, inputs=[dp], device=device)
    came = twt.as_array2d(wp.full((n_a + 1, n_b + 1), -1, dtype=wp.int32, device=device), wp.int32)
    for diag in range(1, n_a + n_b + 1):
        wp.launch(
            kernel_combine.stitch_dp_diag,
            dim=min(diag, n_a) - max(0, diag - n_b) + 1,
            inputs=[
                a_pos,
                b_pos,
                a_opp,
                a_opp_valid,
                b_opp,
                b_opp_valid,
                up,
                wp.int32(metric_id),
                wp.int32(n_a),
                wp.int32(n_b),
                wp.int32(diag),
                dp,
                came,
            ],
            device=device,
        )
    came_np = came.numpy()

    band = _stitch_band_triangles(came_np, la, lb, n_a, n_b, offset)
    combined_vertices, combined_faces = tw.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    band_faces = wp.array(band.reshape(-1), dtype=wp.int32, device=device)
    return combined_vertices, tw.array.concatenate([combined_faces, band_faces])


def _stitch_band_triangles(
    came_np: np.ndarray, la: np.ndarray, lb: np.ndarray, n_a: int, n_b: int, offset: int
) -> np.ndarray:
    """Trace the grid-DP came-from table from ``(n_a, n_b)`` to ``(0, 0)`` into band triangles."""
    triangles: list[tuple[int, int, int]] = []
    i, j = n_a, n_b
    while i > 0 or j > 0:
        if came_np[i, j] == 0:  # advanced A: triangle (a[i-1], a[i], b[j])
            triangles.append((int(la[(i - 1) % n_a]), int(la[i % n_a]), int(lb[j % n_b]) + offset))
            i -= 1
        elif came_np[i, j] == 1:  # advanced B: triangle (a[i], b[j], b[j-1])
            triangles.append(
                (int(la[i % n_a]), int(lb[j % n_b]) + offset, int(lb[(j - 1) % n_b]) + offset)
            )
            j -= 1
        else:  # unreachable cell (should not happen for valid rims)
            break
    return np.asarray(triangles, dtype=np.int32)


# ---------------------------------------------------------------------------
# "Nicely" pipeline: min-weight fill / stitch -> region subdivision -> region smoothing
# (MeshLib fillHoleNicely / stitchHolesNicely, MRFillHoleNicely.cpp).
# ---------------------------------------------------------------------------
