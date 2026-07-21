"""GPU parallel Bridson 2007 blue-noise (igl::blue_noise variant)."""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT
from triwarp.kernels import array as kernel_array

G_FAR = wp.constant(wp.int32(2))
G_STEP = wp.constant(wp.int32(4))
INVALID = wp.constant(wp.int32(-1))
# Candidate shell of ``bridson_step``: a (2 * G_STEP + 1)^3 cube of cells. The wrapper builds a
# per-cell neighbor table over this shell once (the grid never changes), so the round kernels
# replace every binary search over the sorted cell keys with a single table load; the G_FAR
# (5x5x5) neighborhood used by ``far_enough`` is indexed as a sub-block of the same table.
SHELL_W = 9  # 2 * G_STEP + 1 (Python scope: sizes the neighbor table)
SHELL_CELLS = SHELL_W * SHELL_W * SHELL_W
_SHELL_W = wp.constant(wp.int32(SHELL_W))
# Compile-time stack array size for the candidate list (shell minus the center cell).
#
# NOTE: rounds run propose -> resolve -> commit. The parallel propose phase is strictly
# read-only against the frozen committed state (no CAS, no candidate pruning), same-cell ties
# are resolved by active-list rank and cross-cell min-distance conflicts by cell index, and
# only then are winners committed. This makes each round a deterministic function of its input
# state — the earlier optimistic-CAS design relied on thread-timing stagger to avoid
# far_enough -> commit races across cells and produced real min-distance violations once the
# per-probe cost shrank. Do not reintroduce mutation into the propose phase.
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
def shell_slot(dx: wp.int32, dy: wp.int32, dz: wp.int32) -> wp.int32:
    # Flat index of offset (dx, dy, dz) in [-G_STEP, G_STEP]^3 within a neighbor-table row.
    return dx + G_STEP + _SHELL_W * (dy + G_STEP + _SHELL_W * (dz + G_STEP))


@wp.kernel
def init_point_cells(
    grid_coords: wp.array[wp.vec3i],
    grid_w: wp.int32,
    unique_keys: wp.array[wp.int64],
    out_point_cell: wp.array[wp.int32],
) -> None:
    # Compacted cell index of every pool point (its cell is occupied by construction).
    i = int(wp.tid())
    out_point_cell[i] = lookup_cell(unique_keys, grid_cell_key(grid_coords[i], grid_w))


@wp.kernel
def build_cell_neighbors(
    unique_keys: wp.array[wp.int64], grid_w: wp.int32, out_cell_neighbors: wp.array2d[wp.int32]
) -> None:
    # One-time table build: out_cell_neighbors[c, shell_slot(dx, dy, dz)] is the compacted index
    # of cell c's neighbor at that offset, or INVALID for out-of-bounds, unoccupied, and the
    # center slot itself — the round kernels then skip all of those with one branch.
    c, j = wp.tid()
    w64 = wp.int64(grid_w)
    key = unique_keys[c]
    x = wp.int32(key % w64)
    y = wp.int32((key / w64) % w64)
    z = wp.int32(key / (w64 * w64))
    dx = wp.int32(j) % _SHELL_W - G_STEP
    dy = (wp.int32(j) / _SHELL_W) % _SHELL_W - G_STEP
    dz = wp.int32(j) / (_SHELL_W * _SHELL_W) - G_STEP
    out_cell_neighbors[c, j] = INVALID
    if dx == wp.int32(0) and dy == wp.int32(0) and dz == wp.int32(0):
        return
    cx = x + dx
    cy = y + dy
    cz = z + dz
    if cx < wp.int32(0) or cx >= grid_w:
        return
    if cy < wp.int32(0) or cy >= grid_w:
        return
    if cz < wp.int32(0) or cz >= grid_w:
        return
    nk = cell_key(w64, wp.int64(cx), wp.int64(cy), wp.int64(cz))
    out_cell_neighbors[c, j] = lookup_cell(unique_keys, nk)


@wp.func
def far_enough(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    selected: wp.array[wp.int32],
    rr: wp.float32,
    mi: wp.int32,
) -> bool:
    # Same 5x5x5 sweep in the same (dx, dy, dz) order as the pre-table implementation, with the
    # per-cell binary search replaced by one table load (the table already encodes bounds, the
    # center-cell exclusion, and unoccupied cells as INVALID).
    row = point_cell[mi]
    g = G_FAR
    for dx in range(-g, g + 1):
        for dy in range(-g, g + 1):
            for dz in range(-g, g + 1):
                cell_idx = cell_neighbors[row, shell_slot(dx, dy, dz)]
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
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
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
        if far_enough(pool_points, point_cell, cell_neighbors, selected, rr, mi):
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


@wp.func
def find_far_candidate(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
    rr: wp.float32,
    four_rr: wp.float32,
    parent: wp.int32,
    cell_idx: wp.int32,
) -> wp.int32:
    # Read-only counterpart of ``try_activate_cell`` for the parallel propose phase: return the
    # first alive candidate in the bucket within 2r of the parent and far enough from every
    # committed point, without committing or pruning (a candidate that fails ``far_enough`` here
    # fails forever — committed points never disappear — so skipping equals the old kill).
    if cell_idx < wp.int32(0):
        return INVALID
    if selected[cell_idx] >= wp.int32(0):
        return INVALID
    start = int(cell_offsets[cell_idx])
    end = int(cell_offsets[cell_idx + 1])
    for k in range(start, end):
        mi = int(sorted_pool_idx[k])
        if not cand_alive[mi]:
            continue
        if parent >= wp.int32(0):
            diff_parent = pool_points[parent] - pool_points[mi]
            if wp.dot(diff_parent, diff_parent) > four_rr:
                continue
        if far_enough(pool_points, point_cell, cell_neighbors, selected, rr, mi):
            return wp.int32(mi)
    return INVALID


@wp.func
def grid_coord(point: wp.vec3, bbox_min: wp.vec3, inv_cell_size: wp.float32) -> wp.vec3i:
    p = point - bbox_min
    gx = wp.int32(wp.float32(p.x) * inv_cell_size)
    gy = wp.int32(wp.float32(p.y) * inv_cell_size)
    gz = wp.int32(wp.float32(p.z) * inv_cell_size)
    return wp.vec3i(gx, gy, gz)


@wp.func
def grid_cell_key(coord: wp.vec3i, grid_w: wp.int32) -> wp.int64:
    w64 = wp.int64(grid_w)
    return cell_key(w64, wp.int64(coord.x), wp.int64(coord.y), wp.int64(coord.z))


@wp.func
def grid_component(coord: wp.vec3i, axis: wp.int32) -> wp.int32:
    if axis == wp.int32(0):
        return coord.x
    if axis == wp.int32(1):
        return coord.y
    return coord.z


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
def bridson_propose(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
    active: wp.array[wp.int32],
    rr: wp.float32,
    four_rr: wp.float32,
    seed: wp.int32,
    round_idx: wp.int32,
    out_proposal_cell: wp.array[wp.int32],
    out_proposal_mi: wp.array[wp.int32],
    out_retire: wp.array[wp.int32],
) -> None:
    # Propose phase (read-only against frozen state): each active parent picks the first
    # candidate cell, in shuffled shell order, holding a candidate that passes the parent and
    # min-distance tests versus the *committed* set. Nothing is written besides this thread's
    # own outputs, so the proposal set is a deterministic function of the round's input state.
    tid = int(wp.tid())
    parent = int(active[tid])
    out_proposal_cell[tid] = INVALID
    out_proposal_mi[tid] = INVALID
    out_retire[tid] = wp.int32(0)

    row = point_cell[parent]
    g = G_STEP

    neighbors = wp.zeros(shape=_MAX_STEP_NEIGHBORS, dtype=wp.int32)
    n_neighbors = wp.int32(0)

    # Same (dx, dy, dz) candidate order as the pre-table implementation, so the shuffled try
    # order is unchanged; the table load replaces the per-cell binary search and already
    # encodes bounds/center/unoccupied as INVALID.
    for dx in range(-g, g + 1):
        for dy in range(-g, g + 1):
            for dz in range(-g, g + 1):
                cell_idx = cell_neighbors[row, shell_slot(dx, dy, dz)]
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

    i = wp.int32(0)
    while i < n_neighbors:
        cell_idx = neighbors[i]
        mi = find_far_candidate(
            pool_points,
            point_cell,
            cell_neighbors,
            sorted_pool_idx,
            cell_offsets,
            selected,
            cand_alive,
            rr,
            four_rr,
            wp.int32(parent),
            cell_idx,
        )
        if mi >= wp.int32(0):
            out_proposal_cell[tid] = cell_idx
            out_proposal_mi[tid] = mi
            return
        i = i + wp.int32(1)

    out_retire[tid] = wp.int32(1)


@wp.kernel
def resolve_same_cell(
    proposal_cell: wp.array[wp.int32], out_cell_owner: wp.array[wp.int32]
) -> None:
    # Same-cell ties resolve to the lowest active-list rank (out_cell_owner is pre-filled with
    # INT32_MAX by the wrapper each round): deterministic given the proposal set.
    t = int(wp.tid())
    c = proposal_cell[t]
    if c >= wp.int32(0):
        wp.atomic_min(out_cell_owner, c, wp.int32(t))


@wp.kernel
def resolve_cross_cell(
    pool_points: wp.array[wp.vec3],
    cell_neighbors: wp.array2d[wp.int32],
    proposal_cell: wp.array[wp.int32],
    proposal_mi: wp.array[wp.int32],
    cell_owner: wp.array[wp.int32],
    rr: wp.float32,
    out_conflict: wp.array[wp.int32],
) -> None:
    # Cross-cell min-distance conflicts between same-round cell winners resolve by cell index:
    # the winner in the higher-indexed cell yields. One read-only pass over frozen proposals —
    # chains may over-kill, which is conservative (losing parents simply retry next round).
    # Two proposals closer than r sit within G_FAR cells of each other (r = sqrt(3) * cell), so
    # scanning the G_FAR sub-shell of the neighbor table finds every conflicting pair.
    t = int(wp.tid())
    out_conflict[t] = wp.int32(0)
    c = proposal_cell[t]
    if c < wp.int32(0):
        return
    if cell_owner[c] != wp.int32(t):
        return
    mi = proposal_mi[t]
    g = G_FAR
    for dx in range(-g, g + 1):
        for dy in range(-g, g + 1):
            for dz in range(-g, g + 1):
                d = cell_neighbors[c, shell_slot(dx, dy, dz)]
                # Only smaller-indexed cells out-rank this proposal.
                if d < wp.int32(0) or d >= c:
                    continue
                t2 = cell_owner[d]
                if t2 == INT32_MAX_CONSTANT:
                    continue
                other = proposal_mi[t2]
                diff = pool_points[mi] - pool_points[other]
                if wp.dot(diff, diff) < rr:
                    out_conflict[t] = wp.int32(1)
                    return


@wp.kernel
def commit_proposals(
    proposal_cell: wp.array[wp.int32],
    proposal_mi: wp.array[wp.int32],
    cell_owner: wp.array[wp.int32],
    conflict: wp.array[wp.int32],
    out_selected: wp.array[wp.int32],
    out_spawned: wp.array[wp.int32],
) -> None:
    # Commit the surviving winners. Losers (same-cell or cross-cell) spawn nothing but keep
    # retire == 0, so their parents stay active and retry with a fresh shuffle next round.
    t = int(wp.tid())
    out_spawned[t] = INVALID
    c = proposal_cell[t]
    if c < wp.int32(0):
        return
    if cell_owner[c] != wp.int32(t):
        return
    if conflict[t] != wp.int32(0):
        return
    mi = proposal_mi[t]
    out_selected[c] = mi
    out_spawned[t] = mi


@wp.kernel
def prune_spawn_neighborhoods(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    spawned: wp.array[wp.int32],
    rr: wp.float32,
    out_cand_alive: wp.array[wp.bool],
) -> None:
    # Eager, deterministic candidate pruning: every candidate within r of a just-committed
    # spawn can never pass ``far_enough`` again (committed points never disappear), so kill it
    # now instead of re-evaluating the full min-distance test every later round. Writes are
    # idempotent False stores driven by the frozen spawn set — safe under concurrency and the
    # kill set equals the old lazy prune's. Bucket order is never mutated (the old swap-remove
    # compaction was a racy side effect); dead entries just short-circuit on the alive flag.
    t = int(wp.tid())
    s = spawned[t]
    if s < wp.int32(0):
        return
    row = point_cell[s]
    g = G_FAR
    for dx in range(-g, g + 1):
        for dy in range(-g, g + 1):
            for dz in range(-g, g + 1):
                d = cell_neighbors[row, shell_slot(dx, dy, dz)]
                if d < wp.int32(0):
                    continue
                start = int(cell_offsets[d])
                end = int(cell_offsets[d + 1])
                for k in range(start, end):
                    q = int(sorted_pool_idx[k])
                    if not out_cand_alive[q]:
                        continue
                    diff = pool_points[q] - pool_points[s]
                    if wp.dot(diff, diff) < rr:
                        out_cand_alive[q] = False


@wp.func
def cell_has_far_candidate(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
    rr: wp.float32,
    cell_idx: wp.int32,
) -> wp.bool:
    # Read-only dry run of ``try_activate_cell``'s success condition: an empty cell with at
    # least one alive candidate that is far enough from every committed point. No pruning and
    # no CAS, so it is safe to evaluate for every cell in parallel against frozen state.
    if selected[cell_idx] >= wp.int32(0):
        return False
    start = int(cell_offsets[cell_idx])
    end = int(cell_offsets[cell_idx + 1])
    for k in range(start, end):
        mi = int(sorted_pool_idx[k])
        if not cand_alive[mi]:
            continue
        if far_enough(pool_points, point_cell, cell_neighbors, selected, rr, mi):
            return True
    return False


@wp.kernel
def bridson_seed_scan(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
    rr: wp.float32,
    cursor: wp.int32,
    out_winner: wp.array[wp.int32],
) -> None:
    # One thread per remaining cell: the lowest-indexed seedable cell wins, matching the
    # sequential cursor scan exactly (failed cells' candidate pruning was a side effect only).
    c = cursor + wp.int32(wp.tid())
    # Exact pruning: a smaller confirmed-seedable index already holds the atomic_min, so this
    # cell can never be the final winner and the expensive predicate is skipped.
    if out_winner[0] < c:
        return
    if cell_has_far_candidate(
        pool_points,
        point_cell,
        cell_neighbors,
        sorted_pool_idx,
        cell_offsets,
        selected,
        cand_alive,
        rr,
        c,
    ):
        wp.atomic_min(out_winner, 0, c)


@wp.kernel
def bridson_seed_commit(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    sorted_pool_idx: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    selected: wp.array[wp.int32],
    cand_alive: wp.array[wp.bool],
    rr: wp.float32,
    four_rr: wp.float32,
    n_cells: wp.int32,
    winner: wp.array[wp.int32],
    out_seed: wp.array[wp.int32],
) -> None:
    # Single-thread commit of the scan winner (``out_seed`` = [winning cell, spawned pool
    # index or -1]); state is frozen between scan and commit, so the activation succeeds.
    if int(wp.tid()) != 0:
        return
    c = winner[0]
    out_seed[0] = c
    out_seed[1] = INVALID
    if c < n_cells:
        out_seed[1] = try_activate_cell(
            pool_points,
            point_cell,
            cell_neighbors,
            sorted_pool_idx,
            cell_offsets,
            selected,
            cand_alive,
            rr,
            four_rr,
            INVALID,
            c,
        )


@wp.func
def is_zero_int32(value: wp.int32) -> wp.int32:
    return wp.where(value == wp.int32(0), wp.int32(1), wp.int32(0))


@wp.func
def is_nonnegative_int32(value: wp.int32) -> wp.int32:
    return wp.where(value >= wp.int32(0), wp.int32(1), wp.int32(0))


@wp.kernel
def compact_active_and_spawned(
    active: wp.array[wp.int32],
    spawned: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    out_next_active: wp.array[wp.int32],
) -> None:
    # Scatter the surviving actives (first half of ``flags``) and fresh spawns (second half)
    # into their scanned positions: one kernel replaces the per-round flatnonzero/gather/
    # concatenate cascade (order matches the old staying-then-spawned concatenation).
    t = int(wp.tid())
    if flags[t] == 0:
        return
    k = active.shape[0]
    if t < k:
        out_next_active[positions[t]] = active[t]
    else:
        out_next_active[positions[t]] = spawned[t - k]
