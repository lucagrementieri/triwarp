"""Surface and volume sampling for triangular meshes (Warp)."""

from __future__ import annotations

import math
import secrets

import numpy as np
import warp as wp

import triwarp.reduce as tw_reduce
from triwarp.array import append, flatnonzero
from triwarp.graph import faces_to_edges, is_watertight
from triwarp.kernels import sample as kernel_sample
from triwarp.proximity import query_hashgrid_ball_with_offsets
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


def sample_surface_poisson_disk(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    count: int,
    init_factor: float = 5.0,
    face_weight: wp.array[wp.float32] | None = None,
    seed: int | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample points on a triangle mesh surface with Poisson disk distribution.

    Uses Weighted Sample Elimination (Öztireli & Gross 2012): generates
    ``init_factor * count`` uniform surface samples, then iteratively removes
    the most "crowded" points in parallel rounds until ``count`` remain.
    Each round deletes all alive local weight-maxima simultaneously — points
    whose spatial separation exceeds ``r_max`` are independent and eliminated
    in the same round.

    Parameters
    ----------
    vertices
        Vertex positions.
    faces
        Flat triangle indices ``(i0, i1, i2)`` per face.
    count
        Number of output samples.
    init_factor
        Over-sampling factor; initial pool has ``init_factor * count`` points.
        Must satisfy ``init_factor >= 1``.
    face_weight
        Optional per-face weights passed to the initial uniform sampling.
    seed
        RNG seed for the initial uniform sampling. If ``None``, a random seed
        is chosen.

    Returns
    -------
    samples
        ``(count,)`` sampled positions on the mesh surface.
    face_index
        ``(count,)`` triangle index for each sample.

    Raises
    ------
    ValueError
        If ``init_factor < 1`` or if the mesh has no faces.
    """
    device = vertices.device

    if count == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    if init_factor < 1.0:
        raise ValueError(f"init_factor must be >= 1, got {init_factor}")

    init_count = max(math.ceil(init_factor * count), count)

    # 1. Initial uniform surface samples
    init_points, init_face_indices = sample_surface(
        vertices, faces, init_count, face_weight=face_weight, seed=seed
    )

    # 2. Surface area (for radius computation)
    _, areas = face_normals_and_areas(vertices, faces)
    surface_area = float(wp.utils.array_sum(areas))

    # 3. Poisson disk radii (Öztireli & Gross 2012 constants)
    alpha = wp.float32(8.0)
    beta = wp.float32(0.65)
    gamma = wp.float32(1.5)
    ratio = float(count) / float(init_count)
    r_max = wp.float32(2.0 * math.sqrt((surface_area / count) / (2.0 * math.sqrt(3.0))))
    r_min = wp.float32(r_max * beta * (1.0 - ratio**gamma))

    # 4. Neighbor lists (GPU, computed once for the full initial pool)
    nbr_idx, nbr_dists, offsets = query_hashgrid_ball_with_offsets(init_points, init_points, r_max)
    offsets = append(offsets, int(nbr_idx.shape[0]))

    # 5. Initial per-point weights (parallel)
    alive = wp.ones(init_count, dtype=wp.int32, device=device)
    weights = wp.zeros(init_count, dtype=wp.float32, device=device)
    wp.launch(
        kernel_sample.compute_poisson_weights,
        dim=init_count,
        inputs=[nbr_idx, nbr_dists, offsets, alive, r_max, r_min, alpha, weights],
        device=device,
    )

    # 6. Parallel round-based elimination
    alive_count = init_count
    is_max = wp.zeros(init_count, dtype=wp.int32, device=device)

    while alive_count > count:
        wp.launch(
            kernel_sample.find_local_maxima,
            dim=init_count,
            inputs=[weights, alive, nbr_idx, offsets, is_max],
            device=device,
        )

        n_max = tw_reduce.sum(is_max)
        if n_max == 0:
            break

        excess = alive_count - count
        if n_max <= excess:
            deleted_mask = is_max
        else:
            # Pick the top-excess local maxima by weight on CPU
            is_max_np = is_max.numpy()
            weights_np = weights.numpy()
            max_indices = np.where(is_max_np)[0]
            top_k = np.argsort(-weights_np[max_indices])[:excess]
            deleted_np = np.zeros(init_count, dtype=np.int32)
            deleted_np[max_indices[top_k]] = 1
            deleted_mask = wp.array(deleted_np, dtype=wp.int32, device=device)
            n_max = excess

        wp.launch(
            kernel_sample.apply_deletions,
            dim=init_count,
            inputs=[deleted_mask, alive],
            device=device,
        )
        wp.launch(
            kernel_sample.subtract_deleted_contributions,
            dim=init_count,
            inputs=[deleted_mask, nbr_idx, nbr_dists, offsets, alive, r_max, r_min, alpha, weights],
            device=device,
        )
        alive_count -= n_max

    # 7. GPU gather: convert alive mask to bool, get indices, copy selected rows
    alive_bool = wp.empty(init_count, dtype=wp.bool, device=device)
    wp.utils.array_cast(alive, alive_bool)
    indices = flatnonzero(alive_bool)

    selected_points = wp.empty(count, dtype=wp.vec3, device=device)
    selected_face_indices = wp.empty(count, dtype=wp.int32, device=device)
    wp.copy(selected_points, init_points[indices])
    wp.copy(selected_face_indices, init_face_indices[indices])
    return selected_points, selected_face_indices


def sample_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], count: int, seed: int | None = None
) -> wp.array[wp.vec3]:
    """
    Sample points uniformly inside a watertight triangle mesh volume.

    Fans tetrahedra from the mesh's area-weighted surface centroid, builds a CDF
    from signed tet volumes, and draws uniform points inside each selected tet via
    the order-statistics barycentric method (zero rejection for meshes that are
    star-shaped with respect to their centroid).

    Parameters
    ----------
    vertices
        Vertex positions.
    faces
        Flat triangle indices ``(i0, i1, i2)`` per face.
    count
        Number of samples.
    seed
        RNG seed for ``wp.rand_init``. If ``None``, a random seed is chosen.

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
    n_faces = faces.shape[0] // 3

    if count == 0:
        return wp.empty(0, dtype=wp.vec3, device=vertices.device)

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
