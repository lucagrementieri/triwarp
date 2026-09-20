import warp as wp

from triwarp.kernels.array import (
    binary_search_index,
    binary_search_sorted_contains,
    masked_at,
    pack_directed_key,
    pack_edge_key,
)
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
def group_and_vertex_of_key(key: wp.int64, radix: wp.int64) -> tuple[wp.int32, wp.int32]:
    """``(group, vertex)`` packed into ``key`` by [`pack_group_vertex_keys`][], one divmod."""
    return wp.int32(key // radix), wp.int32(key % radix)


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
    write_region_halfedge: wp.bool,
    out_count: wp.array[wp.int32],
    out_region_count: wp.array[wp.int32],
    out_region_halfedge: wp.array[wp.int32],
) -> None:
    # Per unique edge: total incident-face count, how many of those faces are in the region, and
    # (only when the caller asked for the oriented form) which halfedge belongs to the region face.
    # A seam edge has exactly one region-incident face by definition, so no race decides that write;
    # edges with two or none region-incident faces are not seams and whatever lands there is
    # discarded downstream. `out_region_halfedge` is `None` and `write_region_halfedge` is `False`
    # together on the non-oriented path, so this thread never indexes the null array.
    i = wp.int32(wp.tid())
    e = inverse[i]
    wp.atomic_add(out_count, e, wp.int32(1))
    if face_mask[i // 3]:
        wp.atomic_add(out_region_count, e, wp.int32(1))
        if write_region_halfedge:
            out_region_halfedge[e] = i


@wp.func
def region_boundary_flag(count: wp.int32, region_count: wp.int32) -> wp.bool:
    # Interior region-boundary edge: exactly two incident faces, exactly one in the region.
    return count == 2 and region_count == 1


@wp.func
def keep_selected(selected: wp.bool, keep_flag: wp.int32) -> wp.bool:
    # Stay selected only if the vertex was selected and its component is not fully selected.
    return selected and keep_flag != 0


@wp.kernel
def keep_selected_by_component(
    mask: wp.array[wp.bool],
    labels: wp.array[wp.int32],
    keep: wp.array[wp.int32],
    out_mask: wp.array[wp.bool],
) -> None:
    # The per-vertex answer read straight through the vertex's own component label. Done at
    # Python scope this was a gather -- an ``indexedarray`` view, a cloud-sized allocation and a
    # launch to materialise it -- handed to a second launch that then compared it against
    # ``mask``. Both reads are at this thread's own index, so the intermediate never needed to
    # exist.
    v = wp.int32(wp.tid())
    out_mask[v] = keep_selected(mask[v], keep[labels[v]])


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
    forward = pack_directed_key(tail, tip, base)
    along_contour = binary_search_sorted_contains(contour_keys_sorted, forward)
    if along_contour:
        out_seeds[h // 3] = True

    twin = twins[h]
    if twin <= h:
        return  # boundary halfedge, or the far half of an edge the lower half already emitted
    backward = pack_directed_key(tip, tail, base)
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


@wp.kernel
def loops_are_input_rims(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    to_input: wp.array[wp.int32],
    input_boundary_keys: wp.array[wp.uint64],
    base: wp.uint64,
    out_is_input_rim: wp.array[wp.bool],
) -> None:
    # Per loop: was *every* one of its edges already a boundary edge of the input mesh? A loop for
    # which that holds is the input's own rim surfacing in the submesh rather than a rim the
    # deletion opened, which is what ``delete_region_keep_boundary`` filters on.
    #
    # ``to_input`` maps a submesh vertex back to its input index, because the submesh renumbered
    # them and ``input_boundary_keys`` is a sorted table in input indices.
    #
    # One thread per *loop*, walking its own rim, rather than one per rim vertex with a segment
    # label: boundary loops are few and short, which is the regime
    # ``kernels/array.segment_owner_labels`` names as the one where the per-segment shape wins --
    # and it needs no owner array, no total-terminated offsets and no scan to build them. The
    # early exit is what makes it cheap in the common case, since a loop the deletion opened
    # usually fails on one of its first edges.
    #
    # Measured against the host form, same loops: 1.58x on the whole public call at 46 rims and
    # 3.08x at 217 -- the win grows with the rim count, because what it removes is a readback per
    # rim. Flat on the CPU device, where ``wp.array.numpy()`` is a zero-copy view and the host
    # form was never paying for the transfers.
    ell = wp.int32(wp.tid())
    start = loop_starts[ell]
    size = loop_sizes[ell]
    is_rim = wp.int32(1)
    for k in range(size):
        a = to_input[flat_loops[start + k]]
        b = to_input[flat_loops[start + (k + 1) % size]]
        if not binary_search_sorted_contains(input_boundary_keys, pack_edge_key(a, b, base)):
            is_rim = wp.int32(0)
            break
    out_is_input_rim[ell] = is_rim != 0
