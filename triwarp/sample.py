"""Surface and volume sampling for triangular meshes (Warp)."""

from __future__ import annotations

import math
import secrets
from typing import cast

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import append, flatnonzero, gather, init_sort_pair_indices
from triwarp.kernels import array as kernel_array
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

    # Bounding-box min and grid extent via device reductions (no full-pool host copies).
    bbox_min, _ = tw.proximity.aabb_bounds(pool_points)

    grid_coords = wp.empty(nx, dtype=wp.vec3i, device=device)
    wp.map(kernel_blue_noise.grid_coord, pool_points, bbox_min, inv_cell_size, out=grid_coords)

    coord_components = cast(twt.Array2dInt32, grid_coords.view(wp.int32))
    grid_w = int(tw.reduce.max(coord_components)) + 1

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

    seed_cursor = 0
    winner = wp.empty(1, dtype=wp.int32, device=device)
    seed_state = wp.empty(2, dtype=wp.int32, device=device)

    def _try_seed() -> wp.array[wp.int32] | None:
        # Parallel scan for the lowest seedable cell at/after the cursor (a read-only dry run
        # of the activation predicate), then a single-thread commit of the winner: one packed
        # 8-byte host read per drain event instead of a dim=1 launch + sync per scanned cell.
        nonlocal seed_cursor
        while seed_cursor < n_cells:
            winner.fill_(n_cells)
            wp.launch(
                kernel_blue_noise.bridson_seed_scan,
                dim=n_cells - seed_cursor,
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
                    wp.int32(seed_cursor),
                    winner,
                ],
                device=device,
            )
            wp.launch(
                kernel_blue_noise.bridson_seed_commit,
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
                    wp.int32(n_cells),
                    winner,
                    seed_state,
                ],
                device=device,
            )
            state_np = seed_state.numpy()
            if int(state_np[0]) >= n_cells:
                seed_cursor = n_cells
                return None
            seed_cursor = int(state_np[0]) + 1
            if int(state_np[1]) >= 0:
                return wp.clone(seed_state[1:2])
        return None

    active = _try_seed()
    if active is None:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    round_idx = 0
    max_rounds = nx * 4
    while round_idx < max_rounds:
        active_count = int(active.shape[0])
        if active_count == 0:
            active = _try_seed()
            if active is None:
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

        # Survivors (retire == 0) and fresh spawns compact into the next active list with one
        # scan + one scatter: a single small host read per round instead of three
        # flatnonzero/gather passes plus a concatenate.
        flags = wp.empty(2 * active_count, dtype=wp.int32, device=device)
        wp.map(kernel_blue_noise.is_zero_int32, retire, out=flags[:active_count])
        wp.map(kernel_blue_noise.is_nonnegative_int32, spawned, out=flags[active_count:])
        positions = wp.empty(2 * active_count, dtype=wp.int32, device=device)
        wp.utils.array_scan(flags, out_array=positions, inclusive=False)
        tail = 2 * active_count - 1
        total = int(positions[tail:].numpy()[0]) + int(flags[tail:].numpy()[0])
        if total == 0:
            active = wp.empty(0, dtype=wp.int32, device=device)
            round_idx += 1
            continue
        next_active = wp.empty(total, dtype=wp.int32, device=device)
        wp.launch(
            kernel_blue_noise.compact_active_and_spawned,
            dim=2 * active_count,
            inputs=[active, spawned, flags, positions, next_active],
            device=device,
        )
        active = next_active
        round_idx += 1

    # Every activated point was committed into ``selected`` at claim time (atomic CAS), so the
    # result is exactly the non-empty cells — including points still active at the round cap,
    # which the retirement-order bookkeeping used to drop.
    selected_mask = wp.empty(n_cells, dtype=wp.bool, device=device)
    wp.map(kernel_array.greater_equal, selected, wp.int32(0), out=selected_mask)
    kept_cells = flatnonzero(selected_mask)
    if int(kept_cells.shape[0]) == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    kept = gather(selected, kept_cells)
    return gather(pool_points, kept), gather(pool_faces, kept)


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
