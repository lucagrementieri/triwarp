"""Surface and volume sampling for triangular meshes (Warp)."""

from __future__ import annotations

import math
import secrets

import numpy as np
import warp as wp

import triwarp as tw
from triwarp.array import append, concatenate, flatnonzero, gather, init_sort_pair_indices
from triwarp.kernels import sample as kernel_sample
from triwarp.kernels.algorithms import blue_noise as kernel_blue_noise
from triwarp.proximity import query_hashgrid_ball_with_offsets
from triwarp.triangles import centroid, face_normals_and_areas


def get_seed(seed: int | None) -> int:
    """
    Resolve an optional RNG seed to a concrete non-negative ``int32``-range seed.

    Parameters
    ----------
    seed
        User-provided seed, or ``None`` to draw a cryptographically random seed.

    Returns
    -------
    int
        ``seed`` unchanged when provided, otherwise a random value in ``[0, 2**31)``.
    """
    if seed is None:
        return secrets.randbelow(2**31)
    return int(seed)


def sample_fibonacci_sphere(count: int, device: wp.DeviceLike = None) -> wp.array[wp.vec3]:
    """
    Generate near-uniform unit vectors on the sphere via the Fibonacci spiral.

    Successive points are placed at multiples of the golden angle while their
    height ``z`` descends uniformly through ``(-1, 1)``, producing the Fibonacci
    lattice — a deterministic, low-discrepancy covering of the sphere that is far
    more even than independent random sampling for the same ``count``.

    Parameters
    ----------
    count
        Number of directions to generate.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors on the sphere. Empty when ``count`` is 0.

    See Also
    --------
    [`sample_fibonacci_hemisphere`][triwarp.sample.sample_fibonacci_hemisphere]
    """
    if count <= 0:
        return wp.empty(0, dtype=wp.vec3, device=device)
    out_directions = wp.empty(count, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_sample.fibonacci_lattice,
        dim=count,
        inputs=[count, wp.float32(2.0), out_directions],
        device=device,
    )
    return out_directions


def sample_fibonacci_hemisphere(count: int, device: wp.DeviceLike = None) -> wp.array[wp.vec3]:
    """
    Generate near-uniform unit vectors on the positive-``z`` hemisphere.

    Same Fibonacci-spiral construction as
    [`sample_fibonacci_sphere`][triwarp.sample.sample_fibonacci_sphere] but with
    ``z`` descending uniformly through ``(0, 1)``, so every direction has a
    positive ``z`` component. Because the hemisphere and its reflection tile the
    full sphere, pairing each direction ``n`` with its antipode ``-n`` (for
    example via a min/max reduction) covers all orientations with half the
    directions.

    Parameters
    ----------
    count
        Number of directions to generate.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    wp.array[wp.vec3]
        ``(count,)`` unit vectors on the positive-``z`` hemisphere. Empty when
        ``count`` is 0.

    See Also
    --------
    [`sample_fibonacci_sphere`][triwarp.sample.sample_fibonacci_sphere]
    """
    if count <= 0:
        return wp.empty(0, dtype=wp.vec3, device=device)
    out_directions = wp.empty(count, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_sample.fibonacci_lattice,
        dim=count,
        inputs=[count, wp.float32(1.0), out_directions],
        device=device,
    )
    return out_directions


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
    ``wp.sample_triangle`` (same scheme as [`trimesh.sample.sample_surface`][]).

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

        n_max = tw.reduce.sum(is_max)
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


def _bridson_blue_noise(
    pool_points: wp.array[wp.vec3], pool_faces: wp.array[wp.int32], radius: float, seed: int
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """GPU parallel Bridson dart-throwing on a uniform surface candidate pool."""
    device = pool_points.device
    nx = int(pool_points.shape[0])
    if nx == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    cell_size = radius / math.sqrt(3.0)
    inv_cell_size = wp.float32(1.0 / cell_size)
    rr = wp.float32(radius * radius)
    four_rr = wp.float32(4.0 * radius * radius)

    pts_np = pool_points.numpy()
    bbox_min = wp.vec3(*pts_np.min(axis=0).astype(np.float32))

    grid_coords = wp.empty(nx, dtype=wp.vec3i, device=device)
    wp.map(kernel_blue_noise.grid_coord, pool_points, bbox_min, inv_cell_size, out=grid_coords)

    comp = wp.empty(nx, dtype=wp.int32, device=device)
    max_coord = 0
    for axis in (0, 1, 2):
        wp.map(kernel_blue_noise.grid_component, grid_coords, wp.int32(axis), out=comp)
        max_coord = max(max_coord, tw.reduce.max(comp))
    grid_w = int(max_coord) + 1

    cell_keys = wp.empty(nx, dtype=wp.int64, device=device)
    wp.map(kernel_blue_noise.grid_cell_key, grid_coords, wp.int32(grid_w), out=cell_keys)

    keys_buf = wp.empty(2 * nx, dtype=wp.int64, device=device)
    wp.copy(keys_buf, cell_keys, count=nx)
    perm = init_sort_pair_indices(nx, nx, device)
    wp.utils.radix_sort_pairs(keys_buf, perm, count=nx)

    sorted_pool_idx = wp.empty(nx, dtype=wp.int32, device=device)
    wp.copy(sorted_pool_idx, perm, count=nx)

    sorted_keys = wp.empty(nx, dtype=wp.int64, device=device)
    wp.copy(sorted_keys, keys_buf, count=nx)

    unique_keys, counts = tw.unique.unique_1d(sorted_keys, return_counts=True)
    n_cells = int(unique_keys.shape[0])
    cell_offsets_inner = wp.empty(n_cells, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=cell_offsets_inner, inclusive=False)
    cell_offsets = append(cell_offsets_inner, nx)

    selected = wp.full(n_cells, -1, dtype=wp.int32, device=device)
    cand_alive = wp.ones(nx, dtype=wp.bool, device=device)

    has_candidates = wp.empty(n_cells, dtype=wp.bool, device=device)
    seed_cursor = 0
    active = wp.empty(0, dtype=wp.int32, device=device)
    collected_chunks: list[wp.array[wp.int32]] = []

    def _try_seed() -> bool:
        nonlocal seed_cursor, active
        wp.launch(
            kernel_blue_noise.mark_empty_candidate_cells,
            dim=n_cells,
            inputs=[cell_offsets, selected, has_candidates],
            device=device,
        )
        has_np = has_candidates.numpy()
        while seed_cursor < n_cells:
            if not bool(has_np[seed_cursor]):
                seed_cursor += 1
                continue
            out_spawned = wp.empty(1, dtype=wp.int32, device=device)
            out_success = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(
                kernel_blue_noise.bridson_seed_cell,
                dim=1,
                inputs=[
                    pool_points,
                    grid_coords,
                    sorted_pool_idx,
                    cell_offsets,
                    selected,
                    unique_keys,
                    cand_alive,
                    wp.int32(grid_w),
                    rr,
                    four_rr,
                    wp.int32(seed_cursor),
                    out_spawned,
                    out_success,
                ],
                device=device,
            )
            if int(out_success.numpy()[0]) != 0:
                active = wp.array([int(out_spawned.numpy()[0])], dtype=wp.int32, device=device)
                seed_cursor += 1
                return True
            seed_cursor += 1
        return False

    if not _try_seed():
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    round_idx = 0
    max_rounds = nx * 4
    while round_idx < max_rounds:
        active_count = int(active.shape[0])
        if active_count == 0:
            if not _try_seed():
                break
            round_idx += 1
            continue

        spawned = wp.empty(active_count, dtype=wp.int32, device=device)
        retire = wp.zeros(active_count, dtype=wp.int32, device=device)
        wp.launch(
            kernel_blue_noise.bridson_step,
            dim=active_count,
            inputs=[
                pool_points,
                grid_coords,
                sorted_pool_idx,
                cell_offsets,
                selected,
                unique_keys,
                cand_alive,
                active,
                wp.int32(grid_w),
                rr,
                four_rr,
                wp.int32(seed),
                wp.int32(round_idx),
                spawned,
                retire,
            ],
            device=device,
        )

        staying_mask = wp.empty(active_count, dtype=wp.bool, device=device)
        wp.map(kernel_blue_noise.int_is_zero, retire, out=staying_mask)
        spawned_mask = wp.empty(active_count, dtype=wp.bool, device=device)
        wp.map(kernel_blue_noise.spawned_is_valid, spawned, out=spawned_mask)

        retire_bool = wp.empty(active_count, dtype=wp.bool, device=device)
        wp.utils.array_cast(retire, retire_bool)

        staying_active = gather(active, flatnonzero(staying_mask))
        new_spawned = gather(spawned, flatnonzero(spawned_mask))
        retired = gather(active, flatnonzero(retire_bool))
        if int(retired.shape[0]) > 0:
            collected_chunks.append(retired)
        active = concatenate([staying_active, new_spawned])
        round_idx += 1

    if len(collected_chunks) == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    collected_indices = concatenate(collected_chunks)
    return gather(pool_points, collected_indices), gather(pool_faces, collected_indices)


def sample_surface_blue_noise(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], radius: float, seed: int | None = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample points on a triangle mesh surface with Bridson blue-noise distribution.

    Uses "Fast Poisson Disk Sampling in Arbitrary Dimensions" (Bridson 2007),
    following ``igl::blue_noise``. Generates a large uniform surface pool on GPU,
    then runs a parallel round-based active-list dart-throwing loop on GPU (all
    active points attempt to claim neighboring cells each round; hard minimum
    radius ``radius`` is enforced).

    Parameters
    ----------
    vertices
        Vertex positions.
    faces
        Flat triangle indices ``(i0, i1, i2)`` per face.
    radius
        Minimum Poisson disk radius (Euclidean distance in 3D).
    seed
        RNG seed for the initial uniform sampling and Bridson loop. If ``None``,
        a random seed is chosen.

    Returns
    -------
    samples
        ``(m,)`` sampled positions on the mesh surface. Count ``m`` is determined
        implicitly by ``radius`` and mesh area.
    face_index
        ``(m,)`` triangle index for each sample.

    Raises
    ------
    ValueError
        If ``radius <= 0`` or if the mesh has no faces.
    """
    device = vertices.device
    n_faces = faces.shape[0] // 3

    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")

    if n_faces == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    _, areas = face_normals_and_areas(vertices, faces)
    surface_area = float(wp.utils.array_sum(areas))
    expected = surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    nx = max(1, int(30.0 * expected))

    bridson_seed = get_seed(seed)
    init_points, init_face_indices = sample_surface(vertices, faces, nx, seed=seed)
    return _bridson_blue_noise(init_points, init_face_indices, radius, bridson_seed)


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
        If the mesh has zero total volume, or some signed tet volumes are negative after fanning
        from the centroid (the mesh is not star-shaped with respect to its own centroid, e.g. a
        torus).
    """
    n_faces = faces.shape[0] // 3

    if count == 0:
        return wp.empty(0, dtype=wp.vec3, device=vertices.device)

    if not tw.characteristics.is_edge_manifold(faces, allow_boundary_edges=False):
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
