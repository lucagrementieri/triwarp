"""Mesh subdivision on NVIDIA Warp."""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import remesh as kernel_remesh


def subdivide(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Subdivide a mesh by splitting every face into four triangles.

    Each triangle is split by placing a new vertex at the midpoint of each
    edge. The four child triangles share these midpoints and preserve the
    original winding order, matching [`trimesh.remesh.subdivide`][] exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(new_vertices, new_faces)`` on ``vertices.device``.

    See Also
    --------
    [`trimesh.remesh.subdivide`][]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return vertices, faces

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_unique = int(unique_edges.shape[0])

    # Compute midpoint vertex for each unique edge
    out_midpoints = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_remesh.compute_midpoints,
        dim=n_unique,
        inputs=[vertices, unique_edges, out_midpoints],
        device=device,
    )

    # Build (n_faces, 3) array of midpoint vertex indices
    mid_idx = twt.empty_int32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_remesh.build_mid_idx,
        dim=n_faces,
        inputs=[inverse, wp.int32(n_vertices), mid_idx],
        device=device,
    )

    # Emit 4 new triangles per face, shape (n_faces*12,)
    out_new_faces = wp.empty(n_faces * 12, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.subdivide_faces,
        dim=n_faces,
        inputs=[faces, mid_idx, out_new_faces],
        device=device,
    )

    new_vertices, _ = tw.array.pack_1d_arrays([vertices, out_midpoints])
    return new_vertices, out_new_faces


@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Subdivide a mesh until every edge is at most ``max_edge`` long.

    Every edge longer than ``max_edge`` is bisected at a single shared midpoint,
    so the two faces on either side stay in sync and a watertight input stays
    watertight — no T-junctions (cracks) are introduced. Faces already small
    enough are left untouched. Each pass splits every over-long edge once and
    re-triangulates the incident faces with per-face templates (1, 2, or 3 split
    edges; the 2-split quad is cut along its shorter diagonal), iterating until
    no edge exceeds the threshold, matching
    [`trimesh.remesh.subdivide_to_size`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    max_edge
        Maximum length of any edge in the result.
    max_iter
        Maximum number of subdivision passes. A ``ValueError`` is raised if the
        mesh still has an over-long edge after this many passes.
    return_index
        If ``True``, also return the source face index of each output face.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Refined vertex positions on ``vertices.device`` (original vertices first,
        then the inserted edge midpoints).
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    index : wp.array[wp.int32]
        Only returned when ``return_index`` is ``True``: length ``n_out_faces``,
        the index of the original face each output face was refined from.

    Raises
    ------
    ValueError
        If any edge is still longer than ``max_edge`` after ``max_iter`` passes.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.remesh.subdivide_to_size`][]
    """
    device = vertices.device
    max_edge_f = wp.float32(max_edge)

    current_vertices = vertices
    current_faces = faces
    n_faces = int(faces.shape[0]) // 3
    index = tw.array.init_range(n_faces, device)

    if n_faces == 0:
        if return_index:
            return current_vertices, current_faces, index
        return current_vertices, current_faces

    for i in range(max_iter + 1):
        n_faces = int(current_faces.shape[0]) // 3
        n_vertices = int(current_vertices.shape[0])

        unique_edges, inverse = tw.edges.edges_unique(current_faces, n_vertices=n_vertices)
        m = int(unique_edges.shape[0])
        lengths = tw.edges.edges_unique_length(
            current_vertices, current_faces, unique_edges=unique_edges
        )

        # Flag the edges that are longer than the target length.
        long_mask = wp.empty(m, dtype=wp.bool, device=device)
        wp.map(kernel_remesh.is_long_edge, lengths, max_edge_f, out=long_mask)

        # Exclusive scan of the flags gives each long edge its new-vertex slot;
        # the inclusive total is the number of midpoints to add this pass.
        flags = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_cast(long_mask, flags)
        offsets = wp.empty(m, dtype=wp.int32, device=device)
        inclusive = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_scan(flags, out_array=offsets, inclusive=False)
        wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
        n_long = int(inclusive.numpy()[-1])

        # Every edge is short enough: we are done.
        if n_long == 0:
            break
        # Ran out of passes with over-long edges still present.
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")

        midpoint_idx = wp.empty(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.build_midpoint_index,
            dim=m,
            inputs=[long_mask, offsets, wp.int32(n_vertices), midpoint_idx],
            device=device,
        )

        new_mid = wp.empty(n_long, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.fill_edge_midpoints,
            dim=m,
            inputs=[current_vertices, unique_edges, long_mask, offsets, new_mid],
            device=device,
        )
        # Append midpoints so the new indices resolve during face emission.
        current_vertices, _ = tw.array.pack_1d_arrays([current_vertices, new_mid])

        face_mid = twt.empty_int32_2d((n_faces, 3), device=device)
        wp.launch(
            kernel_remesh.build_face_mid,
            dim=n_faces,
            inputs=[inverse, midpoint_idx, face_mid],
            device=device,
        )

        # Emit up to four triangles per face into fixed slots, then compact.
        out_faces = twt.empty_int32_2d((n_faces * 4, 3), device=device)
        out_valid = wp.empty(n_faces * 4, dtype=wp.bool, device=device)
        out_slot_index = wp.empty(n_faces * 4, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.emit_size_faces,
            dim=n_faces,
            inputs=[
                current_faces,
                face_mid,
                current_vertices,
                index,
                out_faces,
                out_valid,
                out_slot_index,
            ],
            device=device,
        )

        kept = tw.array.flatnonzero(out_valid)
        current_faces = tw.array.gather(out_faces, kept).reshape(-1)
        index = tw.array.gather(out_slot_index, kept)

    if return_index:
        return current_vertices, current_faces, index
    return current_vertices, current_faces
