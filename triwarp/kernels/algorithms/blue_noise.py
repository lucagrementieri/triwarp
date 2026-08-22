"""
Blue-noise surface sampling: maximal Poisson-disk selection from a dense pool.

**Randomized-priority parallel dart throwing.** Every pool point draws a priority from the seed; a
point is accepted when no *smaller-priority* point still in play sits within ``r`` of it; everything
within ``r`` of an accepted point is then discarded. Iterate until the pool is empty.

Three properties fall out of that, and they are why this replaced the Bridson active-list walk this
module used to hold:

- **The minimum distance is exact, not approximate.** Two accepted points cannot be within ``r`` of
  each other: within one round the larger-priority one of the pair would have seen the smaller and
  declined, and across rounds the later one would already have been discarded by the earlier one's
  ``r``-ball. Measured closest pair on three fixtures: ``1.000 r``, on the nose.
- **The result is maximal**, so the uncovered gap is bounded: a survivor is by definition a point no
  accepted point covers, so the loop cannot stop while an ``r``-gap remains. Measured worst gap
  1.08-1.10 ``r``, *tighter* than MeshLab's hierarchical dart throwing (1.11-1.19) and Open3D's
  sample elimination (1.20-1.22) at the same parameter.
- **It is the serial algorithm's distribution.** Accepting local priority minima and discarding
  their balls, repeatedly, visits the pool in exactly priority order, so the output is what
  sequential dart throwing over a uniformly random order gives — parallel, not approximate.
  And because each pass writes only its own thread's slot and reads only states it cannot itself
  change, a round is a deterministic function of its input state: same seed, same samples.

The round count is what makes it fast. The smallest-priority point still in play is always a local
minimum, so every round accepts something, and each acceptance discards a whole ``r``-ball — ~90
points on the 30x-oversampled pool ``igl::blue_noise`` sizes — which empties the pool in a handful
of rounds where an active-list walk needed ~100. Against the Bridson implementation on
``bunny_decimated``: **260 -> 44 ms at the 2k-sample radius (6.0x) and 568 -> 54 ms at half that
radius (10.6x)**, at 1-3 % more samples.

Background grid cells are ``r`` on a side, so a ``3x3x3`` neighbourhood contains every point within
``r`` and both sweeps are 27 cells. Bridson's, by contrast, had to enumerate a ``9x9x9`` shell per
active parent per round to find its ``[r, 2r]`` annulus — 729 cells against 27, which together with
the round count is the whole difference.

**Most of those 27 cells cannot affect the answer, and one word per cell says which.** Each sweep
vetoes a candidate on a single property of the points in a neighbouring cell — an acceptance for the
covering sweep, a smaller priority for the selection sweep — so a per-cell summary of that property
decides the whole cell without loading a point from it. Two summaries are rebuilt per round, at a
cost of two fills and one atomic pass over the work list; they remove work whose outcome was already
determined, so the accepted set is identical by construction rather than by tolerance. Measured at
the radii ``benchmarks/test_sample.py`` scores: the covering summary alone is **1.51x / 2.06x**, the
selection summary alone 1.06-1.08x, and the two together **1.70x / 1.72x / 2.33x**, with a
byte-identical ``state`` array in all four combinations.
"""

import warp as wp

from triwarp.kernels import array as kernel_array

INVALID = wp.constant(wp.int32(-1))

# Cells are ``r`` wide, so the ball of radius ``r`` around any point in a cell is contained in that
# cell's 3x3x3 neighbourhood. Unlike the Bridson table, the centre slot points at the cell *itself*:
# both sweeps below have to see the candidate's own bucket.
DART_SHELL_W = 3
DART_SHELL_CELLS = DART_SHELL_W * DART_SHELL_W * DART_SHELL_W
_DART_SHELL_W = wp.constant(wp.int32(DART_SHELL_W))

# Pool-point states. ``ALIVE`` is still in play, ``ACCEPTED`` is in the output, ``COVERED`` is
# within ``r`` of an accepted point and can never become either.
DART_ALIVE = wp.constant(wp.int32(0))
DART_ACCEPTED = wp.constant(wp.int32(1))
DART_COVERED = wp.constant(wp.int32(2))

# Empty-cell sentinel for the per-cell priority summary below. A drawn priority can legitimately
# *equal* it, which costs nothing: a thread only skips a cell whose summary is *strictly* greater
# than its own key, so a cell holding a real ``0xffffffff`` is never wrongly skipped -- the only
# thread that could skip it holds a smaller key, and a larger-priority point can never veto it.
DART_NO_PRIORITY = wp.constant(wp.uint32(0xFFFFFFFF))


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


@wp.kernel
def init_point_cells(
    grid_coords: wp.array[wp.vec3i],
    grid_w: wp.int32,
    unique_keys: wp.array[wp.int64],
    out_point_cell: wp.array[wp.int32],
) -> None:
    # Compacted cell index of every pool point (its cell is occupied by construction).
    i = wp.int32(wp.tid())
    out_point_cell[i] = lookup_cell(unique_keys, grid_cell_key(grid_coords[i], grid_w))


@wp.kernel
def dart_cell_neighbors(
    unique_keys: wp.array[wp.int64], grid_w: wp.int32, out_cell_neighbors: wp.array2d[wp.int32]
) -> None:
    # One-time table: ``out_cell_neighbors[c, s]`` is the compacted index of cell ``c``'s neighbour
    # at shell slot ``s``, or -1 when that neighbour is out of bounds or unoccupied. Built once (the
    # grid never changes), so the round kernels replace a binary search per cell with a table load.
    c, s = wp.tid()
    w64 = wp.int64(grid_w)
    key = unique_keys[c]
    x = wp.int32(key % w64)
    y = wp.int32((key / w64) % w64)
    z = wp.int32(key / (w64 * w64))
    dx = s % _DART_SHELL_W - wp.int32(1)
    dy = (s / _DART_SHELL_W) % _DART_SHELL_W - wp.int32(1)
    dz = s / (_DART_SHELL_W * _DART_SHELL_W) - wp.int32(1)
    out_cell_neighbors[c, s] = INVALID
    cx = x + dx
    cy = y + dy
    cz = z + dz
    if cx < wp.int32(0) or cx >= grid_w:
        return
    if cy < wp.int32(0) or cy >= grid_w:
        return
    if cz < wp.int32(0) or cz >= grid_w:
        return
    out_cell_neighbors[c, s] = lookup_cell(
        unique_keys, cell_key(w64, wp.int64(cx), wp.int64(cy), wp.int64(cz))
    )


@wp.kernel
def dart_priorities(seed: wp.int32, out_priority: wp.array[wp.uint32]) -> None:
    # The sampling order. Drawing it up front rather than per round is what makes the whole loop a
    # deterministic function of ``seed``: every round reads the same total order on the pool.
    i = wp.int32(wp.tid())
    out_priority[i] = wp.randu(wp.rand_init(seed, i))


@wp.kernel
def dart_cell_min_priority(
    priority: wp.array[wp.uint32],
    point_cell: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    out_cell_min_priority: wp.array[wp.uint32],
) -> None:
    # Per-cell summary for the selection sweep: the smallest priority any *alive* point in the cell
    # holds. ``dart_select_minima`` vetoes a candidate only from a strictly smaller priority, so a
    # cell whose minimum already loses to the candidate's key cannot contribute and is skipped
    # whole -- which is most of the 27, most rounds.
    #
    # Summarising the alive list rather than every not-COVERED point leaves the points ACCEPTED in
    # an *earlier* round out, and that is safe: such a point covered its own ``r``-ball in the round
    # it was accepted, so no point still alive now is within ``r`` of it and none of them could have
    # been vetoed by it anyway.
    t = wp.int32(wp.tid())
    i = alive[t]
    wp.atomic_min(out_cell_min_priority, point_cell[i], priority[i])


@wp.kernel
def dart_select_minima(
    pool_points: wp.array[wp.vec3],
    priority: wp.array[wp.uint32],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    bucket: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    cell_min_priority: wp.array[wp.uint32],
    rr: wp.float32,
    out_state: wp.array[wp.int32],
    out_cell_accepted: wp.array[wp.bool],
) -> None:
    # Accept every alive point that no *smaller-priority* point still in play sits within ``r`` of.
    #
    # Ties in the drawn priority fall back to the pool index, so the order is total and the "smaller
    # of the two" argument that keeps two acceptances apart has no exception. The pass writes only
    # its own slot, and the only state it reads of others is "not yet covered" — which this pass
    # never sets — so it is race-free and its output depends only on the round's input state.
    t = wp.int32(wp.tid())
    i = alive[t]
    my_key = priority[i]
    p = pool_points[i]
    row = point_cell[i]
    for s in range(DART_SHELL_CELLS):
        c = cell_neighbors[row, s]
        if c < wp.int32(0):
            continue
        # Nothing in this cell can veto the candidate. The candidate's own cell never trips this,
        # since its own key is one of the minimands.
        if cell_min_priority[c] > my_key:
            continue
        for k in range(cell_offsets[c], cell_offsets[c + 1]):
            j = bucket[k]
            if j == i or out_state[j] == DART_COVERED:
                continue
            other = priority[j]
            if other > my_key or (other == my_key and j > i):
                continue
            if wp.length_sq(pool_points[j] - p) < rr:
                return
    out_state[i] = DART_ACCEPTED
    # Summary for the covering sweep that follows: this cell now holds a point accepted *this*
    # round. Every writer stores the same value, so the unsynchronized store is benign.
    out_cell_accepted[row] = True


@wp.kernel
def dart_cover_neighbors(
    pool_points: wp.array[wp.vec3],
    point_cell: wp.array[wp.int32],
    cell_neighbors: wp.array2d[wp.int32],
    bucket: wp.array[wp.int32],
    cell_offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    cell_accepted: wp.array[wp.bool],
    rr: wp.float32,
    out_state: wp.array[wp.int32],
) -> None:
    # Retire every still-alive point within ``r`` of a point this round accepted. This is what makes
    # the result maximal — a survivor is a point no accepted point covers, so the loop cannot stop
    # while an uncovered gap of radius ``r`` remains.
    #
    # As above, only the thread's own slot is written, and a slot going ``ALIVE -> COVERED`` cannot
    # change another thread's ``== ACCEPTED`` test.
    t = wp.int32(wp.tid())
    i = alive[t]
    if out_state[i] != DART_ALIVE:
        return
    p = pool_points[i]
    row = point_cell[i]
    for s in range(DART_SHELL_CELLS):
        c = cell_neighbors[row, s]
        if c < wp.int32(0):
            continue
        # Cells with nothing accepted this round are skipped whole. Restricting the sweep to *this*
        # round's acceptances loses nothing: a point accepted earlier covered its ``r``-ball in that
        # same round, and this thread was already alive then, so it would not still be alive now.
        if not cell_accepted[c]:
            continue
        for k in range(cell_offsets[c], cell_offsets[c + 1]):
            j = bucket[k]
            if out_state[j] != DART_ACCEPTED:
                continue
            if wp.length_sq(pool_points[j] - p) < rr:
                out_state[i] = DART_COVERED
                return


@wp.kernel
def dart_alive_flags(
    alive: wp.array[wp.int32], state: wp.array[wp.int32], out_flag: wp.array[wp.int32]
) -> None:
    # 0/1 survivor flags over the *work list*, in the dtype ``wp.utils.array_scan`` wants, so the
    # exclusive scan of them is directly the compaction's write positions.
    t = wp.int32(wp.tid())
    out_flag[t] = wp.where(state[alive[t]] == DART_ALIVE, wp.int32(1), wp.int32(0))


@wp.kernel
def dart_compact_alive(
    alive: wp.array[wp.int32],
    state: wp.array[wp.int32],
    positions: wp.array[wp.int32],
    out_next: wp.array[wp.int32],
) -> None:
    # Next round's work list, from this round's: survivors keep their relative order, so the loop
    # walks a shrinking prefix instead of the whole pool. ``positions`` is the exclusive scan of the
    # survivor flags.
    t = wp.int32(wp.tid())
    i = alive[t]
    if state[i] == DART_ALIVE:
        out_next[positions[t]] = i
