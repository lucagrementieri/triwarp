"""
Kernels for ``triwarp.voxels``: cell indexing, tri-box voxelization, morphology, dense conversion.

Every kernel here that reads a grid takes the volume's ``uint64`` id and probes it with
``wp.volume_lookup_index``, which is ``O(1)`` and returns the voxel's linear index (``-1`` when the
cell is empty). That index is the row index of ``Volume.get_voxels()``, so a per-voxel payload is a
plain ``wp.array`` and no side table is ever built.

The half-voxel convention is the wrapper's: a volume whose translation is ``origin + 0.5 * s`` has
NanoVDB voxel ``i`` covering world ``[origin + i * s, origin + (i + 1) * s)``, so
``floor(world_to_index(p) + 0.5)`` is the cell containing ``p``.

**Two halves of this module convert between cells and world positions differently, and the split is
deliberate.** Every kernel *downstream* of a built grid -- ``point_cell``,
``cell_center_positions``, ``corner_positions`` -- goes through ``wp.volume_world_to_index`` /
``wp.volume_index_to_world``, so the convention above is read off the object that defines it
instead of re-implemented. The *voxelization* kernels -- ``voxel_cell``, ``voxel_cell_indices``,
``triangle_voxel_window`` -- run before any volume exists: they produce the cells
``Volume.allocate_by_voxels`` then builds a grid from, so there is no volume id to pass and the
``(origin, voxel_size)`` scalar form is required rather than preferred. Measured perf-neutral either
way, so this is a single-source-of-truth split and not a speed one.
"""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT
from triwarp.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from triwarp.kernels.array import binary_search_index, lattice_position, ravel_index
from triwarp.kernels.predicates import triangle_aabb, triangle_aabb_overlap
from triwarp.kernels.triangles import face_vertices, write_row_triple

# ---------------------------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------------------------


@wp.func
def voxel_cell(position: wp.vec3, origin: wp.vec3, inverse_size: wp.float32) -> wp.vec3i:
    # Integer voxel a position falls in, for the grid anchored at ``origin`` with cell width
    # ``1 / inverse_size``. ``wp.floor`` rather than a cast, so negative coordinates round the same
    # way positive ones do (a C-style truncation would fold the two cells either side of the origin
    # into one).
    local = (position - origin) * inverse_size
    return wp.vec3i(
        wp.int32(wp.floor(local[0])), wp.int32(wp.floor(local[1])), wp.int32(wp.floor(local[2]))
    )


@wp.func
def voxel_cell_center(cell: wp.vec3i, origin: wp.vec3, voxel_size: wp.float32) -> wp.vec3:
    # World position of a cell's centre: the inverse of ``voxel_cell`` above, up to the half-voxel
    # that names the centre rather than the lower corner.
    #
    # The hand-rolled form rather than ``wp.volume_index_to_world`` (which
    # ``cell_center_positions`` below uses, and which the module docstring names as the convention)
    # because every caller here is *pre-volume*: it holds an origin and a cell width and no
    # ``wp.Volume`` handle to ask. The two agree to 3.58e-07 over 3 929 voxels -- float32 rounding,
    # not a convention difference -- and the measurement is recorded on ``cell_center_positions``.
    return wp.vec3(
        origin[0] + (wp.float32(cell[0]) + 0.5) * voxel_size,
        origin[1] + (wp.float32(cell[1]) + 0.5) * voxel_size,
        origin[2] + (wp.float32(cell[2]) + 0.5) * voxel_size,
    )


@wp.func
def squared_distance_to_own_cell_center(
    position: wp.vec3, origin: wp.vec3, voxel_size: wp.float32
) -> wp.float32:
    # How far a point sits from the centre of the voxel it falls in, squared. The quantity a
    # "closest to the cell centre" cluster representative is chosen by.
    #
    # Named because that choice is a *two-pass* argmin -- one kernel reduces the winning distance
    # per cluster and a second re-tests it to break the tie by lowest index -- and the two passes
    # agree only while both compute this expression identically. Two copies of it is a silent
    # correctness hazard rather than a duplication nit: a change to one that rounds differently
    # leaves clusters with no representative at all.
    cell = voxel_cell(position, origin, 1.0 / voxel_size)
    return wp.length_sq(position - voxel_cell_center(cell, origin, voxel_size))


@wp.kernel
def voxel_cell_indices(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    cell = voxel_cell(points[v], origin, inverse_size)
    write_row_triple(out_cells, v, cell[0], cell[1], cell[2])


@wp.func
def triangle_voxel_window_from_vertices(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, origin: wp.vec3, inverse_size: wp.float32
) -> tuple[wp.vec3i, wp.vec3i]:
    # Inclusive lower/upper cell of the exact AABB window of a triangle already in hand. Shared by
    # the count and test passes below, which MUST enumerate the same window: the second writes into
    # the slots the first reserved, so a window that disagreed by one cell would write out of range.
    #
    # Open3D walks ``round((max - min) / vs) + 2`` cells, a strict superset of this one; a cell
    # outside the triangle's own AABB cannot overlap the triangle, so the *accepted* sets are
    # identical and only the number of rejected candidates differs.
    lower, upper = triangle_aabb(v0, v1, v2)
    return voxel_cell(lower, origin, inverse_size), voxel_cell(upper, origin, inverse_size)


@wp.func
def triangle_voxel_window(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_index: wp.int32,
    origin: wp.vec3,
    inverse_size: wp.float32,
) -> tuple[wp.vec3i, wp.vec3i]:
    # Gathers the face's own vertices and defers to the vertex-taking form above. Kept separate from
    # it (rather than folded into one signature) because ``test_triangle_candidates`` below needs
    # the gathered v0/v1/v2 for its own overlap test right after computing the window, and calling
    # through this wrapper would gather them a second time to get at them.
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    return triangle_voxel_window_from_vertices(v0, v1, v2, origin, inverse_size)


@wp.kernel
def count_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_counts: wp.array[wp.int32],
    out_counts_f32: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    lo, hi = triangle_voxel_window(vertices, faces, f, origin, inverse_size)
    # int64 so a wildly under-sized voxel does not wrap the product into a plausible small count --
    # each axis span widens to int64 *before* the subtraction, not after, so the subtraction itself
    # cannot already overflow int32 for a triangle whose window is that wide (the convention
    # ``kernels/graph.py``'s packed labels key uses: widen the operands, not their difference).
    span_x = wp.int64(hi[0]) - wp.int64(lo[0]) + wp.int64(1)
    span_y = wp.int64(hi[1]) - wp.int64(lo[1]) + wp.int64(1)
    span_z = wp.int64(hi[2]) - wp.int64(lo[2]) + wp.int64(1)
    span = span_x * span_y * span_z
    out_counts[f] = wp.int32(wp.min(span, wp.int64(INT32_MAX_CONSTANT)))
    out_counts_f32[f] = wp.float32(span)


@wp.kernel
def test_triangle_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    origin: wp.vec3,
    voxel_size: wp.float32,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    # One thread per (triangle, candidate cell) pair: per-triangle window sizes span orders of
    # magnitude on any real mesh, so a thread-per-triangle launch is load-imbalanced by that same
    # factor. ``binary_search_index`` recovers the owning triangle from the flat work-item id
    # (``wp.lower_bound`` clamps to ``n - 1`` and would misattribute the last window).
    #
    # The per-item decode below (``span_y``/``span_z``/``plane``/``i``/``j``/``k``) stays int32,
    # unlike ``count_triangle_candidates``'s widened axis spans above -- a single axis wide enough
    # to overflow int32 on its own would still misdecode here, but reaching that needs
    # ``max_candidates`` raised past its ``2**31`` default (already impractical: the candidate
    # buffers alone would be tens of GB at that width). Left as a documented residual rather than
    # widening this hot loop's arithmetic to int64 for a practically unreachable input; revisit only
    # if a caller legitimately needs a wider ``max_candidates``.
    item = wp.int32(wp.tid())
    f = binary_search_index(offsets, item) - 1
    # Gather the face's own vertices once and feed them to both the window (for this item's cell)
    # and the overlap test below, rather than calling ``triangle_voxel_window`` (which would gather
    # the same three rows again internally) -- this is the dominant work unit in the kernel, one
    # gather per (triangle, candidate-cell) work item rather than two.
    v0, v1, v2 = face_vertices(vertices, faces, f)
    lo, hi = triangle_voxel_window_from_vertices(v0, v1, v2, origin, inverse_size)
    span_y = hi[1] - lo[1] + 1
    span_z = hi[2] - lo[2] + 1

    local = item - offsets[f]
    plane = span_y * span_z
    i = local // plane
    rest = local % plane
    j = rest // span_z
    k = rest % span_z
    cell = wp.vec3i(lo[0] + i, lo[1] + j, lo[2] + k)

    half = wp.vec3(0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size)
    center = voxel_cell_center(cell, origin, voxel_size)
    write_row_triple(out_cells, item, cell[0], cell[1], cell[2])
    out_mask[item] = wp.where(triangle_aabb_overlap(center, half, v0, v1, v2), 1, 0)


# ---------------------------------------------------------------------------------------------
# Grid <-> world
# ---------------------------------------------------------------------------------------------


@wp.kernel
def cell_center_positions(
    volume: wp.uint64, voxels: wp.array2d[wp.int32], out_centers: wp.array[wp.vec3]
) -> None:
    # NanoVDB centres voxel ``i`` *on* index-space coordinate ``i``, so the integer cell coordinate
    # maps straight to the cell centre and the volume's own transform supplies the half-voxel shift
    # the module docstring describes. It agrees with the hand-rolled
    # ``origin + (cell + 0.5) * voxel_size`` to float32 rounding, i.e. no convention disagreement.
    v = wp.int32(wp.tid())
    out_centers[v] = wp.volume_index_to_world(
        volume,
        wp.vec3(wp.float32(voxels[v, 0]), wp.float32(voxels[v, 1]), wp.float32(voxels[v, 2])),
    )


@wp.func
def cell_slot(volume: wp.uint64, cells: wp.array2d[wp.int32], v: wp.int32) -> wp.int32:
    # The grid and ``get_voxels()`` share one numbering, so a cell's slot is also its payload row;
    # ``-1`` means the cell is not in the grid. Shared by the slot kernel and the occupancy one
    # below, which differ only in whether the caller wants the row or just its existence.
    return wp.volume_lookup_index(volume, cells[v, 0], cells[v, 1], cells[v, 2])


@wp.kernel
def lookup_cell_slots(
    volume: wp.uint64, cells: wp.array2d[wp.int32], out_slots: wp.array[wp.int32]
) -> None:
    v = wp.int32(wp.tid())
    out_slots[v] = cell_slot(volume, cells, v)


@wp.kernel
def cell_occupancy(
    volume: wp.uint64, cells: wp.array2d[wp.int32], present: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    # The probe and the comparison in one pass: the slot is a register here, where a separate
    # lookup kernel would write every one of them to global memory for a second launch to read
    # back and test. ``present`` is warp-uniform and selects membership or its complement, which
    # is what lets ``intersection`` and ``difference`` share this kernel instead of the second
    # paying a third launch to invert the first's answer.
    #
    # Measured against the lookup-kernel-plus-map form it replaced, output byte-identical: 2.0x on
    # ``occupancy_at_cells``, 2.1x on ``occupancy_at_points`` and 1.25x on ``difference``, which
    # carried the extra inversion launch.
    v = wp.int32(wp.tid())
    out_mask[v] = (cell_slot(volume, cells, v) >= 0) == present


@wp.func
def point_cell(volume: wp.uint64, position: wp.vec3) -> wp.vec3i:
    # NanoVDB centres voxel ``i`` on index-space coordinate ``i``, so the cell containing a point
    # is ``floor(uvw + 0.5)`` -- not ``round``, which sends ``-0.5`` to ``-1`` instead of ``0``.
    uvw = wp.volume_world_to_index(volume, position)
    return wp.vec3i(
        wp.int32(wp.floor(uvw[0] + 0.5)),
        wp.int32(wp.floor(uvw[1] + 0.5)),
        wp.int32(wp.floor(uvw[2] + 0.5)),
    )


@wp.func
def point_slot(volume: wp.uint64, position: wp.vec3) -> wp.int32:
    # The point twin of ``cell_slot``, and shared for the same reason: a query's voxel row, or
    # ``-1`` outside the grid.
    cell = point_cell(volume, position)
    return wp.volume_lookup_index(volume, cell[0], cell[1], cell[2])


@wp.kernel
def lookup_point_slots(
    volume: wp.uint64, points: wp.array[wp.vec3], out_slots: wp.array[wp.int32]
) -> None:
    p = wp.int32(wp.tid())
    out_slots[p] = point_slot(volume, points[p])


@wp.kernel
def point_occupancy(
    volume: wp.uint64, points: wp.array[wp.vec3], present: wp.bool, out_mask: wp.array[wp.bool]
) -> None:
    # The point form of ``cell_occupancy``; see it for why the probe and the test share a kernel.
    p = wp.int32(wp.tid())
    out_mask[p] = (point_slot(volume, points[p]) >= 0) == present


@wp.kernel
def pack_cell_keys(
    cells: wp.array2d[wp.int32],
    lower: wp.array[wp.int32],
    upper: wp.array[wp.int32],
    out_keys: wp.array[wp.uint64],
) -> None:
    # Column 0 is the least significant digit, matching ``kernels/grouping.pack_indices``: sorting
    # these keys reproduces ``grouping.unique_rows``'s row order exactly. The shift is
    # ``min(lower[c], 0)`` per axis -- order-preserving because it is a per-axis constant, and only
    # non-zero where a column actually goes negative, so a non-negative cell set keeps exactly the
    # keys ``grouping.hash_indices_rows`` would produce.
    #
    # ``lower`` / ``upper`` are the two three-element buffers ``reduce.minmax(cells, axis=0)``
    # returns, read here rather than passed in as ``wp.vec3i`` / ``wp.uint64`` scalars: every lane
    # wants the same six values, so they are broadcast loads out of L2, and taking them by value
    # would instead cost the wrapper two host readbacks. Byte-identical values, and worth about a
    # quarter of ``voxels.cells`` at every size -- the call is host-bound throughout, so removing
    # host work is the whole win.
    v = wp.int32(wp.tid())
    lo = wp.int32(0)
    hi = upper[0]
    for c in range(3):
        lo = wp.min(lo, lower[c])
        hi = wp.max(hi, upper[c])
    radix = wp.uint64(hi - lo + 1)
    key = wp.uint64(0)
    power = wp.uint64(1)
    for c in range(3):
        key = key + wp.uint64(wp.uint32(cells[v, c] - wp.min(lower[c], 0))) * power
        power = power * radix
    out_keys[v] = key


@wp.kernel
def lattice_points(lower: wp.vec3, step: wp.vec3, out_points: wp.array3d[wp.vec3]) -> None:
    # A dense node lattice, rank-2 destination. The position is ``array.lattice_position``, shared
    # with ``reconstruction.lattice_points``, which writes the same quantity into a flat row-major
    # buffer instead. This is one of the two sites this module's docstring names as running before
    # a ``wp.Volume`` exists, so ``wp.volume_index_to_world`` is not available to it.
    i, j, k = wp.tid()
    out_points[i, j, k] = lattice_position(lower, step, i, j, k)


# ---------------------------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------------------------


@wp.kernel
def bucket_point_slots(
    slots: wp.array[wp.int32],
    n_voxels: wp.int32,
    write_buckets: wp.bool,
    out_buckets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Points that fall outside the grid go into a sentinel bucket past the last voxel, so they sort
    # to the end and every real voxel's segment stays contiguous.
    #
    # ``write_buckets`` is warp-uniform: only the mean/sum pooling branch sorts by bucket, and the
    # min/max branch wants nothing from this launch but ``out_counts``. Writing the per-point
    # buckets for it anyway is one ``int32`` per point stored and an allocation to hold them, so
    # the selector lets that caller pass a length-zero buffer instead of a cloud-sized one.
    p = wp.int32(wp.tid())
    bucket = slots[p]
    if bucket < 0:
        bucket = n_voxels
    if write_buckets:
        out_buckets[p] = bucket
    wp.atomic_add(out_counts, bucket, 1)


@wp.kernel
def segment_reduce_vec3(
    order: wp.array[wp.int32],
    values: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    average: wp.bool,
    out_values: wp.array[wp.vec3],
) -> None:
    # One thread per voxel walking its segment in index order: the sum is bitwise reproducible,
    # which a float ``wp.atomic_add`` over the points would not be.
    v = wp.int32(wp.tid())
    start = offsets[v]
    count = counts[v]
    total = wp.vec3(0.0, 0.0, 0.0)
    for j in range(start, start + count):
        total = total + values[order[j]]
    if average and count > 0:
        total = total / wp.float32(count)
    out_values[v] = total


@wp.kernel
def pool_extremum_vec3(
    slots: wp.array[wp.int32],
    values: wp.array[wp.vec3],
    largest: wp.bool,
    out_values: wp.array[wp.vec3],
) -> None:
    # Component-wise atomic min / max: order-independent for floats, so no sort is needed here.
    p = wp.int32(wp.tid())
    slot = slots[p]
    if slot < 0:
        return
    if largest:
        wp.atomic_max(out_values, slot, values[p])
    else:
        wp.atomic_min(out_values, slot, values[p])


@wp.kernel
def zero_empty_voxels(counts: wp.array[wp.int32], out_values: wp.array[wp.vec3]) -> None:
    v = wp.int32(wp.tid())
    if counts[v] == 0:
        out_values[v] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------


@wp.kernel
def neighborhood_candidates(
    voxels: wp.array2d[wp.int32], neighbors: wp.array2d[wp.int32], out_cells: wp.array2d[wp.int32]
) -> None:
    v, m = wp.tid()
    row = v * neighbors.shape[0] + m
    write_row_triple(
        out_cells,
        row,
        voxels[v, 0] + neighbors[m, 0],
        voxels[v, 1] + neighbors[m, 1],
        voxels[v, 2] + neighbors[m, 2],
    )


@wp.kernel
def neighborhood_complete(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    interior: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 when every neighbour of the voxel is occupied (an interior voxel), 0 otherwise. Neighbours
    # of a voxel live in the same 8-cubed leaf most of the time, so the probes are cache-local.
    #
    # ``interior`` is warp-uniform and selects which of the two complementary answers to write:
    # ``erode`` wants the interior set and ``surface_voxels`` its complement. One selector rather
    # than a second kernel, because the complement is this kernel's own result negated -- writing
    # it here costs nothing, where a separate pass costs a launch and a full round trip of the
    # flags through global memory.
    #
    # Measured 1.2-1.3x on ``surface_voxels``, byte-identical.
    v = wp.int32(wp.tid())
    complete = wp.int32(1)
    for m in range(neighbors.shape[0]):
        i = voxels[v, 0] + neighbors[m, 0]
        j = voxels[v, 1] + neighbors[m, 1]
        k = voxels[v, 2] + neighbors[m, 2]
        if wp.volume_lookup_index(volume, i, j, k) < 0:
            complete = wp.int32(0)
    out_flags[v] = wp.where(interior, complete, 1 - complete)


@wp.func
def span_cell(axis: wp.int32, a: wp.int32, b: wp.int32, t: wp.int32) -> wp.vec3i:
    # ``(a, b)`` index the two axes other than ``axis``, ``t`` runs along ``axis``.
    if axis == 0:
        return wp.vec3i(t, a, b)
    if axis == 1:
        return wp.vec3i(a, t, b)
    return wp.vec3i(a, b, t)


@wp.kernel
def fill_axis_span(
    occupancy: wp.array3d[wp.bool],
    axis: wp.int32,
    length: wp.int32,
    out_filled: wp.array3d[wp.bool],
) -> None:
    # One thread per line along ``axis``: mark every cell between the first and the last occupied
    # one. trimesh's ``ops.fill_orthographic`` intersects the three axes' results.
    a, b = wp.tid()
    first = wp.int32(-1)
    last = wp.int32(-1)
    for t in range(length):
        cell = span_cell(axis, a, b, t)
        if occupancy[cell[0], cell[1], cell[2]]:
            if first < 0:
                first = t
            last = t
    for t in range(length):
        cell = span_cell(axis, a, b, t)
        out_filled[cell[0], cell[1], cell[2]] = first >= 0 and t >= first and t <= last


@wp.kernel
def intersect_occupancy(
    a: wp.array3d[wp.bool], b: wp.array3d[wp.bool], out_and: wp.array3d[wp.bool]
) -> None:
    i, j, k = wp.tid()
    out_and[i, j, k] = a[i, j, k] and b[i, j, k]


@wp.func
def flat_cell_index(i: wp.int32, j: wp.int32, k: wp.int32, ny: wp.int32, nz: wp.int32) -> wp.int32:
    return ravel_index(i, j, k, ny, nz)


@wp.kernel
def flood_init_parent(occupancy: wp.array3d[wp.bool], out_parents: wp.array[wp.int32]) -> None:
    # ECL-CC initialisation over the *empty* complement, with the 6-neighbour stencil implicit:
    # three backward probes, no edge list. An occupied cell is its own singleton and never unions.
    i, j, k = wp.tid()
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    v = flat_cell_index(i, j, k, ny, nz)
    out_parents[v] = v
    if occupancy[i, j, k]:
        return
    if i > 0 and not occupancy[i - 1, j, k]:
        out_parents[v] = flat_cell_index(i - 1, j, k, ny, nz)
        return
    if j > 0 and not occupancy[i, j - 1, k]:
        out_parents[v] = flat_cell_index(i, j - 1, k, ny, nz)
        return
    if k > 0 and not occupancy[i, j, k - 1]:
        out_parents[v] = flat_cell_index(i, j, k - 1, ny, nz)


@wp.kernel
def flood_hook(occupancy: wp.array3d[wp.bool], parents: wp.array[wp.int32]) -> None:
    # The three backward neighbours own each undirected edge exactly once, so every 6-connection
    # between two empty cells is hooked once. ``rep_v`` is carried across the three, ECL-CC's
    # ``vstat``.
    i, j, k = wp.tid()
    if occupancy[i, j, k]:
        return
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    v = flat_cell_index(i, j, k, ny, nz)
    rep_v = find_representative(parents, v)
    if i > 0 and not occupancy[i - 1, j, k]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i - 1, j, k, ny, nz))
    if j > 0 and not occupancy[i, j - 1, k]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i, j - 1, k, ny, nz))
    if k > 0 and not occupancy[i, j, k - 1]:
        rep_v = ecl_hook_edge(parents, rep_v, flat_cell_index(i, j, k - 1, ny, nz))


@wp.kernel
def mark_outside_roots(
    occupancy: wp.array3d[wp.bool], labels: wp.array[wp.int32], out_outside: wp.array[wp.bool]
) -> None:
    # The padded shell is empty by construction, so its components are exactly the "outside".
    i, j, k = wp.tid()
    nx = occupancy.shape[0]
    ny = occupancy.shape[1]
    nz = occupancy.shape[2]
    on_shell = i == 0 or j == 0 or k == 0 or i == nx - 1 or j == ny - 1 or k == nz - 1
    if not on_shell or occupancy[i, j, k]:
        return
    out_outside[labels[flat_cell_index(i, j, k, ny, nz)]] = True


@wp.kernel
def fill_enclosed_cells(
    occupancy: wp.array3d[wp.bool],
    labels: wp.array[wp.int32],
    outside: wp.array[wp.bool],
    out_filled: wp.array3d[wp.bool],
) -> None:
    i, j, k = wp.tid()
    if occupancy[i, j, k]:
        out_filled[i, j, k] = True
        return
    out_filled[i, j, k] = not outside[
        labels[flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])]
    ]


# ---------------------------------------------------------------------------------------------
# Dense conversion and meshing
# ---------------------------------------------------------------------------------------------


@wp.func
def cell_is_occupied(
    volume: wp.uint64, base: wp.vec3i, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.bool:
    return wp.volume_lookup_index(volume, base[0] + i, base[1] + j, base[2] + k) >= 0


@wp.kernel
def dense_occupancy(volume: wp.uint64, base: wp.vec3i, out_occupancy: wp.array3d[wp.bool]) -> None:
    i, j, k = wp.tid()
    out_occupancy[i, j, k] = cell_is_occupied(volume, base, i, j, k)


@wp.kernel
def dense_field(volume: wp.uint64, base: wp.vec3i, out_field: wp.array3d[wp.float32]) -> None:
    i, j, k = wp.tid()
    out_field[i, j, k] = wp.where(cell_is_occupied(volume, base, i, j, k), 1.0, 0.0)


@wp.kernel
def occupied_cells(
    occupancy: wp.array3d[wp.bool],
    base: wp.vec3i,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])
    write_row_triple(out_cells, row, base[0] + i, base[1] + j, base[2] + k)
    out_mask[row] = wp.where(occupancy[i, j, k], 1, 0)


@wp.kernel
def cell_corner_indices(
    corner_volume: wp.uint64, voxels: wp.array2d[wp.int32], out_corners: wp.array2d[wp.int32]
) -> None:
    # Corner ``c`` of a cell is ``cell + (c >> 2, (c >> 1) & 1, c & 1)`` -- x-major binary counting.
    v = wp.int32(wp.tid())
    for c in range(8):
        i = voxels[v, 0] + (c >> 2)
        j = voxels[v, 1] + ((c >> 1) & 1)
        k = voxels[v, 2] + (c & 1)
        out_corners[v, c] = wp.volume_lookup_index(corner_volume, i, j, k)


@wp.kernel
def corner_positions(
    volume: wp.uint64, corners: wp.array2d[wp.int32], out_positions: wp.array[wp.vec3]
) -> None:
    # Corner ``(i, j, k)`` is the *lower* corner of cell ``(i, j, k)``. A cell is centred on its
    # integer index-space coordinate, so its lower corner sits half a voxel below on every axis --
    # which is a shift in *index* space, where it is the NanoVDB convention itself rather than a
    # constant re-derived from the grid's translation.
    c = wp.int32(wp.tid())
    out_positions[c] = wp.volume_index_to_world(
        volume,
        wp.vec3(
            wp.float32(corners[c, 0]) - 0.5,
            wp.float32(corners[c, 1]) - 0.5,
            wp.float32(corners[c, 2]) - 0.5,
        ),
    )


@wp.kernel
def count_box_faces(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    cull_internal: wp.bool,
    out_counts: wp.array[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    if not cull_internal:
        out_counts[v] = 6
        return
    exposed = wp.int32(0)
    for d in range(6):
        i = voxels[v, 0] + neighbors[d, 0]
        j = voxels[v, 1] + neighbors[d, 1]
        k = voxels[v, 2] + neighbors[d, 2]
        if wp.volume_lookup_index(volume, i, j, k) < 0:
            exposed += 1
    out_counts[v] = exposed


@wp.kernel
def emit_box_faces(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    corners: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    face_corners: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    cull_internal: wp.bool,
    out_faces: wp.array[wp.int32],
) -> None:
    # Two triangles per exposed cube face, wound so the normal points away from the voxel.
    v = wp.int32(wp.tid())
    quad = offsets[v]
    for d in range(6):
        i = voxels[v, 0] + neighbors[d, 0]
        j = voxels[v, 1] + neighbors[d, 1]
        k = voxels[v, 2] + neighbors[d, 2]
        if cull_internal and wp.volume_lookup_index(volume, i, j, k) >= 0:
            continue
        a = corners[v, face_corners[d, 0]]
        b = corners[v, face_corners[d, 1]]
        c = corners[v, face_corners[d, 2]]
        e = corners[v, face_corners[d, 3]]
        base = quad * 6
        out_faces[base + 0] = a
        out_faces[base + 1] = b
        out_faces[base + 2] = c
        out_faces[base + 3] = a
        out_faces[base + 4] = c
        out_faces[base + 5] = e
        quad += 1
