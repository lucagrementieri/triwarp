"""Surface sampling for triangular meshes (Warp)."""

from __future__ import annotations

import secrets

import warp as wp

from triwarp.kernels import sample as kernel_sample
from triwarp.triangles import face_normals_and_areas


def get_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbelow(2**31)
    return int(seed)


def sample_surface(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    count: int,
    face_weight: wp.array[wp.float32] | None = None,
    seed: int | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Sample points uniformly on a triangle mesh surface (area-weighted faces).

    Uses ``face_normals_and_areas`` for default triangle weights. Builds a CDF and
    draws triangle indices with ``wp.sample_cdf``, then uniform points with
    ``wp.sample_triangle`` (same scheme as :func:`trimesh.sample.sample_surface`).

    Parameters
    ----------
    vertices
        Vertex positions.
    faces
        Flat triangle indices ``(i0, i1, i2)`` per face.
    count
        Number of samples.
    face_weight
        Optional per-face weights (length = number of triangles). If ``None``,
        triangle areas from ``face_normals_and_areas`` are used.
    seed
        RNG seed for ``wp.rand_init``. If ``None``, a random seed is chosen.

    Returns
    -------
    samples
        ``(count,)`` sampled positions on the mesh surface.
    face_index
        ``(count,)`` triangle index for each sample.
    """
    n_faces = faces.shape[0] // 3
    if count == 0:
        return (wp.empty(0, dtype=wp.vec3, device=vertices.device), wp.empty(0, dtype=wp.int32, device=vertices.device))
    if face_weight is not None and face_weight.shape[0] != n_faces:
        raise ValueError(
            f"face_weight length must match number of triangles (expected {n_faces}, got {face_weight.shape[0]})"
        )

    if face_weight is None:
        _, weights = face_normals_and_areas(vertices, faces)
    else:
        weights = face_weight

    total = float(wp.utils.array_sum(weights))
    if total <= 0.0:
        raise ValueError("total face weight must be positive")
    cdf = wp.empty(n_faces, dtype=wp.float32, device=vertices.device)
    wp.utils.array_scan(weights, out_array=cdf)
    cdf = cdf / total

    out_points = wp.empty(count, dtype=wp.vec3, device=vertices.device)
    out_face_indices = wp.empty(count, dtype=wp.int32, device=vertices.device)
    wp.launch(
        kernel_sample.sample_surface,
        dim=count,
        inputs=[vertices, faces, cdf, get_seed(seed), out_points, out_face_indices],
        device=vertices.device,
    )
    return out_points, out_face_indices
