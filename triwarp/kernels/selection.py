import warp as wp

from triwarp.kernels.array import binary_search_index, binary_search_sorted_contains, masked_at
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.triangles import corner_triple


@wp.kernel
def pack_group_vertex_keys(
    faces: wp.array[wp.int32],
    group_face_indices: wp.array[wp.int32],
    group_offsets: wp.array[wp.int32],
    radix: wp.int64,
    out_keys: wp.array[wp.int64],
) -> None:
    # ``group * radix + vertex`` for every corner of every selected face, with ``radix`` the source
    # vertex count. Ascending key order is exactly ``(group, vertex)`` lexicographic order, which is
    # what lets a single global ``unique_1d`` do the work of one dedup per group and still emit each
    # group's vertices ascending by original index.
    #
    # ``group_offsets`` is the CSR start of each group, so the group owning position ``p`` is the
    # last one starting at or before it — a binary search rather than a materialized per-position
    # label array. ``out_keys`` is ``wp.int64`` concretely, not ``wp.Int``: a generic packer cannot
    # construct the promoted value portably.
    corner = wp.int32(wp.tid())
    position = corner // 3
    group = binary_search_index(group_offsets, position) - 1
    face = group_face_indices[position]
    out_keys[corner] = wp.int64(group) * radix + wp.int64(faces[face * 3 + corner % 3])


@wp.func
def group_of_key(key: wp.int64, radix: wp.int64) -> wp.int32:
    """Group index packed into ``key`` by [`pack_group_vertex_keys`][]."""
    return wp.int32(key // radix)


@wp.func
def vertex_of_key(key: wp.int64, radix: wp.int64) -> wp.int32:
    """Source vertex index packed into ``key`` by [`pack_group_vertex_keys`][]."""
    return wp.int32(key % radix)


@wp.kernel
def count_group_slots(slot_groups: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Vertices per group, as a histogram over the sorted unique slots.
    wp.atomic_add(out_counts, slot_groups[wp.int32(wp.tid())], wp.int32(1))


@wp.kernel
def local_vertex_index(
    slot_groups: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    out_local: wp.array[wp.int32],
) -> None:
    # Rank of each global slot within its own group: the compacted, zero-based vertex index the
    # group's faces must refer to. A real kernel because the thread index *is* the datum.
    slot = wp.int32(wp.tid())
    out_local[slot] = slot - vertex_offsets[slot_groups[slot]]


@wp.kernel
def face_mask_from_vertex_mask(
    faces: wp.array[wp.int32],
    vertex_mask: wp.array[wp.bool],
    require_all: wp.bool,
    out_face_mask: wp.array[wp.bool],
) -> None:
    # Reduce a per-vertex selection onto its faces: keep a face when all three of its corners are
    # selected (``require_all``), or when at least one is. One warp-uniform branch rather than two
    # kernels, since the two differ by a parameter and not by an algorithm.
    #
    # This replaces the route the wrapper used to take -- ``flatnonzero`` the mask, ``array.isin``
    # the face buffer against the resulting index list, then reduce along the rows -- which
    # rebuilt as a lookup table exactly the membership the mask already *is*, at the cost of two
    # min/max reductions with a host readback each, a sort or a span-sized table, and a second
    # ``flatnonzero``. Reading the mask directly is one O(1) lookup per corner; see
    # ``masked_at`` for why the read is guarded.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    hit_a = masked_at(vertex_mask, a)
    hit_b = masked_at(vertex_mask, b)
    hit_c = masked_at(vertex_mask, c)
    if require_all:
        out_face_mask[f] = hit_a and hit_b and hit_c
    else:
        out_face_mask[f] = hit_a or hit_b or hit_c


@wp.kernel
def dilate_vertex_mask(
    unique_edges: wp.array2d[wp.int32], in_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # One edge dilation round: a vertex joins the mask if either endpoint of an incident edge is
    # already in it. ``out_mask`` must be pre-seeded with ``in_mask`` (this only adds neighbors).
    i = wp.int32(wp.tid())
    a = unique_edges[i, 0]
    b = unique_edges[i, 1]
    if in_mask[a]:
        out_mask[b] = wp.bool(True)
    if in_mask[b]:
        out_mask[a] = wp.bool(True)


@wp.kernel
def mark_region_halfedges(
    inverse: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    out_region_halfedge: wp.array[wp.int32],
) -> None:
    # One halfedge per unique edge that belongs to a *region* face. A seam edge has exactly one such
    # face by definition, so no race decides anything there; edges with two or none are not seam
    # edges and whatever lands is discarded.
    h = wp.int32(wp.tid())
    if face_mask[h // 3]:
        out_region_halfedge[inverse[h]] = h


@wp.kernel
def oriented_edges_from_halfedges(
    faces: wp.array[wp.int32], halfedges: wp.array[wp.int32], out_edges: wp.array2d[wp.int32]
) -> None:
    # A halfedge read as a directed vertex pair, which puts its own face on the left.
    i = wp.int32(wp.tid())
    out_edges[i, 0] = faces[halfedges[i]]
    out_edges[i, 1] = halfedge_destination(faces, halfedges[i])


@wp.kernel
def edge_region_counts(
    inverse: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
    out_region_count: wp.array[wp.int32],
) -> None:
    # Per unique edge: total incident-face count and how many of those faces are in the region.
    i = wp.int32(wp.tid())
    e = inverse[i]
    wp.atomic_add(out_count, e, wp.int32(1))
    if face_mask[i // 3]:
        wp.atomic_add(out_region_count, e, wp.int32(1))


@wp.func
def region_boundary_flag(count: wp.int32, region_count: wp.int32) -> wp.bool:
    # Interior region-boundary edge: exactly two incident faces, exactly one in the region.
    return count == 2 and region_count == 1


@wp.func
def keep_selected(selected: wp.bool, keep_flag: wp.int32) -> wp.bool:
    # Stay selected only if the vertex was selected and its component is not fully selected.
    return selected and keep_flag != 0


@wp.kernel
def keep_component_scatter(
    mask: wp.array[wp.bool], labels: wp.array[wp.int32], out_keep: wp.array[wp.int32]
) -> None:
    # A component is "kept" (not fully selected) if it has at least one unselected vertex.
    v = wp.int32(wp.tid())
    if not mask[v]:
        out_keep[labels[v]] = wp.int32(1)


@wp.kernel
def open_dual_edges_and_seeds(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    contour_keys_sorted: wp.array[wp.uint64],
    base: wp.uint64,
    cursor: wp.array[wp.int32],
    out_dual_edges: wp.array2d[wp.int32],
    out_seeds: wp.array[wp.bool],
) -> None:
    # One pass over halfedges answers both halves of the fill at once. A halfedge running *along*
    # the contour in its own direction puts its face on the left, since a face's corners run
    # counter-clockwise -- so the contour's orientation is the only thing deciding which side is
    # "left". And an edge on the contour in *either* direction is blocked, so it emits no dual edge.
    #
    # Doing it this way rather than through `face_adjacency` is deliberate: the dual graph, the
    # blocking test and the seeding all come from `twins`, which is one hash-and-group pass instead
    # of that plus a second key sort over every halfedge and a mask-compact over every dual edge.
    h = wp.int32(wp.tid())
    tail = faces[h]
    tip = halfedge_destination(faces, h)
    forward = wp.uint64(wp.uint32(tail)) + wp.uint64(wp.uint32(tip)) * base
    along_contour = binary_search_sorted_contains(contour_keys_sorted, forward)
    if along_contour:
        out_seeds[h // 3] = True

    twin = twins[h]
    if twin <= h:
        return  # boundary halfedge, or the far half of an edge the lower half already emitted
    backward = wp.uint64(wp.uint32(tip)) + wp.uint64(wp.uint32(tail)) * base
    if along_contour or binary_search_sorted_contains(contour_keys_sorted, backward):
        return
    slot = wp.atomic_add(cursor, 0, 1)
    out_dual_edges[slot, 0] = h // 3
    out_dual_edges[slot, 1] = twin // 3


@wp.kernel
def mark_labels_of_seeds(
    labels: wp.array[wp.int32], seeds: wp.array[wp.bool], out_label_seeded: wp.array[wp.bool]
) -> None:
    # One flag per component label, so the per-face answer becomes a gather. Labels name a
    # representative element, so the flag array is face-indexed rather than 0..k-1.
    f = wp.int32(wp.tid())
    if seeds[f]:
        out_label_seeded[labels[f]] = True
