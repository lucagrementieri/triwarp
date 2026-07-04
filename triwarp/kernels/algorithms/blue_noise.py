"""GPU parallel Bridson 2007 blue-noise (igl::blue_noise variant)."""

import warp as wp

from triwarp.kernels import array as kernel_array

G_FAR = wp.constant(wp.int32(2))
G_STEP = wp.constant(wp.int32(4))
INVALID = wp.constant(wp.int32(-1))
# Compile-time stack array size for neighbor shell (g=4 -> 9^3-1 cells).
_MAX_STEP_NEIGHBORS = 728


@wp.func
def cell_key(w: wp.int64, x: wp.int64, y: wp.int64, z: wp.int64) -> wp.int64:
    return x + w * (y + w * z)


@wp.func
def lookup_cell(sorted_unique_keys: wp.array[wp.int64], nk: wp.int64) -> wp.int32:
    if sorted_unique_keys.shape[0] == 0:
        return INVALID
    idx = kernel_array.binary_search_index(sorted_unique_keys, nk)
    if idx > wp.int32(0) and sorted_unique_keys[idx - wp.int32(1)] == nk:
        return idx - wp.int32(1)
    return INVALID


@wp.func
def far_enough(
    pool_points: wp.array[wp.vec3],
    grid_coords: wp.array[wp.vec3i],
    selected: wp.array[wp.int32],
    unique_keys: wp.array[wp.int64],
    grid_w: wp.int32,
    rr: wp.float32,
    mi: wp.int32,
) -> bool:
    xi = grid_coords[mi].x
    yi = grid_coords[mi].y
    zi = grid_coords[mi].z
    g = G_FAR
    w64 = wp.int64(grid_w)

    for dx in range(-g, g + 1):
        cx = xi + dx
        if cx < wp.int32(0) or cx >= grid_w:
            continue
        for dy in range(-g, g + 1):
            cy = yi + dy
            if cy < wp.int32(0) or cy >= grid_w:
                continue
            for dz in range(-g, g + 1):
                cz = zi + dz
                if cz < wp.int32(0) or cz >= grid_w:
                    continue
                if cx == xi and cy == yi and cz == zi:
                    continue
                nk = cell_key(w64, wp.int64(cx), wp.int64(cy), wp.int64(cz))
                cell_idx = lookup_cell(unique_keys, nk)
                if cell_idx < wp.int32(0):
                    continue
                ni = selected[cell_idx]
                if ni >= wp.int32(0):
                    diff = pool_points[mi] - pool_points[ni]
                    if wp.dot(diff, diff) < rr:
                        return False
    return True


@wp.func
def try_activate_cell(
    pool_points: wp.array[wp.vec3],
    grid_coords: wp.array[wp.vec3i],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    unique_keys: wp.array[wp.int64],
    cand_alive: wp.array[wp.bool],
    grid_w: wp.int32,
    rr: wp.float32,
    four_rr: wp.float32,
    parent: wp.int32,
    cell_idx: wp.int32,
) -> wp.int32:
    if cell_idx < wp.int32(0):
        return INVALID
    if selected[cell_idx] >= wp.int32(0):
        return INVALID

    start = int(cell_offsets[cell_idx])
    end = int(cell_offsets[cell_idx + 1])
    k = start
    while k < end:
        mi = int(sorted_pool_idx[k])
        if not cand_alive[mi]:
            k += 1
            continue
        if parent >= wp.int32(0):
            diff_parent = pool_points[parent] - pool_points[mi]
            if wp.dot(diff_parent, diff_parent) > four_rr:
                k += 1
                continue
        if far_enough(pool_points, grid_coords, selected, unique_keys, grid_w, rr, mi):
            old = wp.atomic_cas(selected, cell_idx, INVALID, wp.int32(mi))
            if old == INVALID:
                return wp.int32(mi)
            return INVALID
        cand_alive[mi] = False
        last = end - 1
        if k < last:
            sorted_pool_idx[k] = sorted_pool_idx[last]
        end = end - 1
    return INVALID


@wp.kernel
def compute_grid_coords(
    pool_points: wp.array[wp.vec3],
    bbox_min: wp.vec3,
    inv_cell_size: wp.float32,
    out_grid_coords: wp.array[wp.vec3i],
) -> None:
    i = int(wp.tid())
    p = pool_points[i] - bbox_min
    gx = wp.int32(wp.float32(p.x) * inv_cell_size)
    gy = wp.int32(wp.float32(p.y) * inv_cell_size)
    gz = wp.int32(wp.float32(p.z) * inv_cell_size)
    out_grid_coords[i] = wp.vec3i(gx, gy, gz)


@wp.kernel
def compute_cell_keys(
    grid_coords: wp.array[wp.vec3i], grid_w: wp.int32, out_cell_keys: wp.array[wp.int64]
) -> None:
    i = int(wp.tid())
    c = grid_coords[i]
    w64 = wp.int64(grid_w)
    out_cell_keys[i] = cell_key(w64, wp.int64(c.x), wp.int64(c.y), wp.int64(c.z))


@wp.kernel
def extract_grid_component(
    grid_coords: wp.array[wp.vec3i], axis: wp.int32, out_component: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    if axis == wp.int32(0):
        out_component[i] = grid_coords[i].x
    elif axis == wp.int32(1):
        out_component[i] = grid_coords[i].y
    else:
        out_component[i] = grid_coords[i].z


@wp.func
def shuffle_neighbors(neighbors: wp.array[wp.int32], count: wp.int32, state: wp.uint32) -> None:
    i = count - wp.int32(1)
    while i > wp.int32(0):
        j = wp.int32(wp.randu(state) % wp.uint32(i + wp.int32(1)))
        tmp = neighbors[i]
        neighbors[i] = neighbors[j]
        neighbors[j] = tmp
        i = i - wp.int32(1)


@wp.kernel
def bridson_step(
    pool_points: wp.array[wp.vec3],
    grid_coords: wp.array[wp.vec3i],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    unique_keys: wp.array[wp.int64],
    cand_alive: wp.array[wp.bool],
    active: wp.array[wp.int32],
    grid_w: wp.int32,
    rr: wp.float32,
    four_rr: wp.float32,
    seed: wp.int32,
    round_idx: wp.int32,
    out_spawned: wp.array[wp.int32],
    out_retire: wp.array[wp.int32],
) -> None:
    tid = int(wp.tid())
    parent = int(active[tid])
    out_spawned[tid] = INVALID
    out_retire[tid] = wp.int32(0)

    xi = grid_coords[parent].x
    yi = grid_coords[parent].y
    zi = grid_coords[parent].z
    g = G_STEP
    w = grid_w

    neighbors = wp.zeros(shape=_MAX_STEP_NEIGHBORS, dtype=wp.int32)
    n_neighbors = wp.int32(0)

    for dx in range(-g, g + 1):
        cx = xi + dx
        if cx < wp.int32(0) or cx >= w:
            continue
        for dy in range(-g, g + 1):
            cy = yi + dy
            if cy < wp.int32(0) or cy >= w:
                continue
            for dz in range(-g, g + 1):
                cz = zi + dz
                if cz < wp.int32(0) or cz >= w:
                    continue
                if cx == xi and cy == yi and cz == zi:
                    continue
                nk = cell_key(wp.int64(w), wp.int64(cx), wp.int64(cy), wp.int64(cz))
                cell_idx = lookup_cell(unique_keys, nk)
                if cell_idx < wp.int32(0):
                    continue
                if selected[cell_idx] >= wp.int32(0):
                    continue
                if int(cell_offsets[cell_idx + 1]) <= int(cell_offsets[cell_idx]):
                    continue
                neighbors[n_neighbors] = cell_idx
                n_neighbors = n_neighbors + wp.int32(1)

    if n_neighbors == wp.int32(0):
        out_retire[tid] = wp.int32(1)
        return

    mix = parent + int(round_idx) * 104729
    state = wp.rand_init(seed, mix)
    shuffle_neighbors(neighbors, n_neighbors, state)

    spawned = INVALID
    i = wp.int32(0)
    while i < n_neighbors:
        cell_idx = neighbors[i]
        mi = try_activate_cell(
            pool_points,
            grid_coords,
            sorted_pool_idx,
            cell_offsets,
            selected,
            unique_keys,
            cand_alive,
            grid_w,
            rr,
            four_rr,
            wp.int32(parent),
            cell_idx,
        )
        if mi >= wp.int32(0):
            spawned = mi
            break
        i = i + wp.int32(1)

    if spawned >= wp.int32(0):
        out_spawned[tid] = spawned
    else:
        out_retire[tid] = wp.int32(1)


@wp.kernel
def bridson_seed_cell(
    pool_points: wp.array[wp.vec3],
    grid_coords: wp.array[wp.vec3i],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    unique_keys: wp.array[wp.int64],
    cand_alive: wp.array[wp.bool],
    grid_w: wp.int32,
    rr: wp.float32,
    four_rr: wp.float32,
    cell_idx: wp.int32,
    out_spawned: wp.array[wp.int32],
    out_success: wp.array[wp.int32],
) -> None:
    if int(wp.tid()) != 0:
        return
    mi = try_activate_cell(
        pool_points,
        grid_coords,
        sorted_pool_idx,
        cell_offsets,
        selected,
        unique_keys,
        cand_alive,
        grid_w,
        rr,
        four_rr,
        INVALID,
        cell_idx,
    )
    if mi >= wp.int32(0):
        out_spawned[0] = mi
        out_success[0] = wp.int32(1)
    else:
        out_spawned[0] = INVALID
        out_success[0] = wp.int32(0)


@wp.kernel
def mark_empty_candidate_cells(
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    out_has_candidates: wp.array[wp.bool],
) -> None:
    c = int(wp.tid())
    out_has_candidates[c] = selected[c] < wp.int32(0) and cell_offsets[c + 1] > cell_offsets[c]


@wp.kernel
def int_is_zero(flags: wp.array[wp.int32], out_mask: wp.array[wp.bool]) -> None:
    i = int(wp.tid())
    out_mask[i] = flags[i] == wp.int32(0)


@wp.kernel
def spawned_is_valid(spawned: wp.array[wp.int32], out_mask: wp.array[wp.bool]) -> None:
    i = int(wp.tid())
    out_mask[i] = spawned[i] >= wp.int32(0)
