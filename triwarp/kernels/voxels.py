"""
Kernels for ``triwarp.voxels``: cell indexing, tri-box voxelization, morphology, dense conversion.

Every kernel here that reads a grid takes the volume's ``uint64`` id and probes it with
``wp.volume_lookup_index``, which is ``O(1)`` and returns the voxel's linear index (``-1`` when the
cell is empty). That index is the row index of ``Volume.get_voxels()``, so a per-voxel payload is a
plain ``wp.array`` and no side table is ever built.

The half-voxel convention is the wrapper's: a volume whose translation is ``origin + 0.5 * s`` has
NanoVDB voxel ``i`` covering world ``[origin + i * s, origin + (i + 1) * s)``, so
``floor(world_to_index(p) + 0.5)`` is the cell containing ``p``.
"""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT
from triwarp.kernels.algorithms.connected_components import ecl_hook_edge, find_representative
from triwarp.kernels.array import binary_search_index
from triwarp.kernels.intersection import triangle_aabb_overlap
from triwarp.kernels.predicates import triangle_aabb
from triwarp.kernels.triangles import face_vertices

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


@wp.kernel
def voxel_cell_indices(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    inverse_size: wp.float32,
    out_cells: wp.array2d[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    cell = voxel_cell(points[v], origin, inverse_size)
    out_cells[v, 0] = cell[0]
    out_cells[v, 1] = cell[1]
    out_cells[v, 2] = cell[2]


@wp.func
def triangle_voxel_window(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_index: wp.int32,
    origin: wp.vec3,
    inverse_size: wp.float32,
) -> tuple[wp.vec3i, wp.vec3i]:
    # Inclusive lower/upper cell of the exact AABB window of face ``face_index``. Shared by the
    # count and test passes below, which MUST enumerate the same window: the second writes into the
    # slots the first reserved, so a window that disagreed by one cell would write out of range.
    #
    # Open3D walks ``round((max - min) / vs) + 2`` cells, a strict superset of this one; a cell
    # outside the triangle's own AABB cannot overlap the triangle, so the *accepted* sets are
    # identical and only the number of rejected candidates differs.
    v0, v1, v2 = face_vertices(vertices, faces, face_index)
    lower, upper = triangle_aabb(v0, v1, v2)
    return voxel_cell(lower, origin, inverse_size), voxel_cell(upper, origin, inverse_size)


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
    # int64 so a wildly under-sized voxel does not wrap the product into a plausible small count.
    span = wp.int64(hi[0] - lo[0] + 1) * wp.int64(hi[1] - lo[1] + 1) * wp.int64(hi[2] - lo[2] + 1)
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
    item = wp.int32(wp.tid())
    f = binary_search_index(offsets, item) - 1
    lo, hi = triangle_voxel_window(vertices, faces, f, origin, inverse_size)
    span_y = hi[1] - lo[1] + 1
    span_z = hi[2] - lo[2] + 1

    local = item - offsets[f]
    plane = span_y * span_z
    i = local // plane
    rest = local - i * plane
    j = rest // span_z
    k = rest - j * span_z
    cell = wp.vec3i(lo[0] + i, lo[1] + j, lo[2] + k)

    half = wp.vec3(0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size)
    center = wp.vec3(
        origin[0] + (wp.float32(cell[0]) + 0.5) * voxel_size,
        origin[1] + (wp.float32(cell[1]) + 0.5) * voxel_size,
        origin[2] + (wp.float32(cell[2]) + 0.5) * voxel_size,
    )
    out_cells[item, 0] = cell[0]
    out_cells[item, 1] = cell[1]
    out_cells[item, 2] = cell[2]
    v0, v1, v2 = face_vertices(vertices, faces, f)
    if triangle_aabb_overlap(center, half, v0, v1, v2):
        out_mask[item] = 1
    else:
        out_mask[item] = 0


# ---------------------------------------------------------------------------------------------
# Grid <-> world
# ---------------------------------------------------------------------------------------------


@wp.kernel
def cell_center_positions(
    voxels: wp.array2d[wp.int32],
    origin: wp.vec3,
    voxel_size: wp.float32,
    out_centers: wp.array[wp.vec3],
) -> None:
    v = wp.int32(wp.tid())
    out_centers[v] = wp.vec3(
        origin[0] + (wp.float32(voxels[v, 0]) + 0.5) * voxel_size,
        origin[1] + (wp.float32(voxels[v, 1]) + 0.5) * voxel_size,
        origin[2] + (wp.float32(voxels[v, 2]) + 0.5) * voxel_size,
    )


@wp.kernel
def lookup_cell_slots(
    volume: wp.uint64, cells: wp.array2d[wp.int32], out_slots: wp.array[wp.int32]
) -> None:
    v = wp.int32(wp.tid())
    out_slots[v] = wp.volume_lookup_index(volume, cells[v, 0], cells[v, 1], cells[v, 2])


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


@wp.kernel
def lookup_point_slots(
    volume: wp.uint64, points: wp.array[wp.vec3], out_slots: wp.array[wp.int32]
) -> None:
    p = wp.int32(wp.tid())
    cell = point_cell(volume, points[p])
    out_slots[p] = wp.volume_lookup_index(volume, cell[0], cell[1], cell[2])


@wp.func
def is_present(slot: wp.int32) -> wp.bool:
    return slot >= 0


@wp.kernel
def pack_cell_keys(
    cells: wp.array2d[wp.int32], base: wp.vec3i, radix: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # Column 0 is the least significant digit, matching ``kernels/grouping.pack_indices``: sorting
    # these keys reproduces ``grouping.unique_rows``'s row order exactly. ``base`` shifts negative
    # cells non-negative, which is order-preserving because it is a per-axis constant.
    v = wp.int32(wp.tid())
    key = wp.uint64(0)
    power = wp.uint64(1)
    for c in range(3):
        key = key + wp.uint64(wp.uint32(cells[v, c] - base[c])) * power
        power = power * radix
    out_keys[v] = key


@wp.kernel
def lattice_points(lower: wp.vec3, step: wp.vec3, out_points: wp.array3d[wp.vec3]) -> None:
    i, j, k = wp.tid()
    out_points[i, j, k] = wp.vec3(
        lower[0] + wp.float32(i) * step[0],
        lower[1] + wp.float32(j) * step[1],
        lower[2] + wp.float32(k) * step[2],
    )


# ---------------------------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------------------------


@wp.kernel
def bucket_point_slots(
    slots: wp.array[wp.int32],
    n_voxels: wp.int32,
    out_buckets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # Points that fall outside the grid go into a sentinel bucket past the last voxel, so they sort
    # to the end and every real voxel's segment stays contiguous.
    p = wp.int32(wp.tid())
    bucket = slots[p]
    if bucket < 0:
        bucket = n_voxels
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
    out_cells[row, 0] = voxels[v, 0] + neighbors[m, 0]
    out_cells[row, 1] = voxels[v, 1] + neighbors[m, 1]
    out_cells[row, 2] = voxels[v, 2] + neighbors[m, 2]


@wp.kernel
def neighborhood_complete(
    volume: wp.uint64,
    voxels: wp.array2d[wp.int32],
    neighbors: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 when every neighbour of the voxel is occupied (an interior voxel), 0 otherwise. Neighbours
    # of a voxel live in the same 8-cubed leaf most of the time, so the probes are cache-local.
    v = wp.int32(wp.tid())
    complete = wp.int32(1)
    for m in range(neighbors.shape[0]):
        i = voxels[v, 0] + neighbors[m, 0]
        j = voxels[v, 1] + neighbors[m, 1]
        k = voxels[v, 2] + neighbors[m, 2]
        if wp.volume_lookup_index(volume, i, j, k) < 0:
            complete = wp.int32(0)
    out_flags[v] = complete


@wp.func
def flip_flag(flag: wp.int32) -> wp.int32:
    return 1 - flag


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
    return (i * ny + j) * nz + k


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


@wp.kernel
def dense_occupancy(volume: wp.uint64, base: wp.vec3i, out_occupancy: wp.array3d[wp.bool]) -> None:
    i, j, k = wp.tid()
    out_occupancy[i, j, k] = (
        wp.volume_lookup_index(volume, base[0] + i, base[1] + j, base[2] + k) >= 0
    )


@wp.kernel
def dense_field(volume: wp.uint64, base: wp.vec3i, out_field: wp.array3d[wp.float32]) -> None:
    i, j, k = wp.tid()
    if wp.volume_lookup_index(volume, base[0] + i, base[1] + j, base[2] + k) >= 0:
        out_field[i, j, k] = 1.0
    else:
        out_field[i, j, k] = 0.0


@wp.kernel
def occupied_cells(
    occupancy: wp.array3d[wp.bool],
    base: wp.vec3i,
    out_cells: wp.array2d[wp.int32],
    out_mask: wp.array[wp.int32],
) -> None:
    i, j, k = wp.tid()
    row = flat_cell_index(i, j, k, occupancy.shape[1], occupancy.shape[2])
    out_cells[row, 0] = base[0] + i
    out_cells[row, 1] = base[1] + j
    out_cells[row, 2] = base[2] + k
    if occupancy[i, j, k]:
        out_mask[row] = 1
    else:
        out_mask[row] = 0


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
    corners: wp.array2d[wp.int32],
    origin: wp.vec3,
    voxel_size: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Corner ``(i, j, k)`` is the *lower* corner of cell ``(i, j, k)``, so it sits at
    # ``origin + (i, j, k) * voxel_size`` with no half-voxel shift.
    c = wp.int32(wp.tid())
    out_positions[c] = wp.vec3(
        origin[0] + wp.float32(corners[c, 0]) * voxel_size,
        origin[1] + wp.float32(corners[c, 1]) * voxel_size,
        origin[2] + wp.float32(corners[c, 2]) * voxel_size,
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
