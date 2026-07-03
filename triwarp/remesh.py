"""Mesh subdivision on NVIDIA Warp."""

from __future__ import annotations

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
