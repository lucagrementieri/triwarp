"""Surface and volume sampling for triangular meshes (Warp)."""

from __future__ import annotations

import secrets

import warp as wp

from triwarp.graph import faces_to_edges, is_watertight
from triwarp.kernels import sample as kernel_sample
from triwarp.triangles import centroid, face_normals_and_areas


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
    """
    Sample points uniformly on a triangle mesh surface (area-weighted faces).

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
        return (
            wp.empty(0, dtype=wp.vec3, device=vertices.device),
            wp.empty(0, dtype=wp.int32, device=vertices.device),
        )
    if face_weight is not None and face_weight.shape[0] != n_faces:
        raise ValueError(
            f"face_weight length must match number of triangles (expected {n_faces}, "
            f"got {face_weight.shape[0]})"
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


def volume_mesh(mesh: wp.Mesh, count: int, seed: int | None = None) -> wp.array[wp.vec3]:
    """
    Sample points uniformly inside a watertight triangle mesh (signed tet decomposition).

    Fans tetrahedra from the mesh's area-weighted surface centroid. Each sample is
    drawn proportional to the tet's signed volume contribution, then placed uniformly
    inside the selected tet via the order-statistics barycentric method (zero rejection
    for meshes that are star-shaped with respect to their centroid).

    Parameters
    ----------
    mesh
        Warp mesh object. ``mesh.points`` and ``mesh.indices`` supply the
        geometry; no BVH query is performed.
    count
        Number of samples to return.
    seed
        RNG seed. If ``None``, a random seed is chosen.

    Returns
    -------
    samples
        ``(count,)`` positions inside the mesh volume.

    Raises
    ------
    ValueError
        If the mesh is not watertight (open boundary edges detected).
    ValueError
        If some signed tet volumes are negative after fanning from the centroid
        (the mesh is not star-shaped with respect to its own centroid, e.g. a torus).
    """
    vertices = mesh.points
    faces = mesh.indices
    n_faces = faces.shape[0] // 3

    if count == 0:
        return wp.empty(0, dtype=wp.vec3, device=vertices.device)
    if n_faces == 0:
        raise ValueError("mesh has no faces")

    edges = faces_to_edges(faces)
    watertight, _ = is_watertight(edges)
    if not watertight:
        raise ValueError(
            "mesh is not watertight; tetrahedral decomposition requires a closed surface"
        )

    center = centroid(vertices, faces)

    signed_vols = wp.empty(n_faces, dtype=wp.float32, device=vertices.device)
    wp.launch(
        kernel_sample.signed_tet_volumes,
        dim=n_faces,
        inputs=[vertices, faces, center, signed_vols],
        device=vertices.device,
    )

    vols_np = signed_vols.numpy()
    total_vol = float(vols_np.sum())
    if total_vol == 0.0:
        raise ValueError("mesh has zero volume")

    if total_vol < 0.0:
        vols_np = -vols_np
        total_vol = -total_vol

    if float(vols_np.min()) < 0.0:
        raise ValueError(
            "mesh is not star-shaped with respect to its centroid (e.g. a torus); "
            "tetrahedral decomposition cannot sample it without rejection"
        )

    weights = wp.array(vols_np, dtype=wp.float32, device=vertices.device)
    cdf = wp.empty(n_faces, dtype=wp.float32, device=vertices.device)
    wp.utils.array_scan(weights, out_array=cdf)
    cdf = cdf / total_vol

    out_points = wp.empty(count, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_sample.sample_volume_tet,
        dim=count,
        inputs=[vertices, faces, center, cdf, get_seed(seed), out_points],
        device=vertices.device,
    )
    return out_points
