import warp as wp

from triwarp.kernels.adjacency import write_face_edge_keys
from triwarp.kernels.array import (
    binary_search_index,
    binary_search_sorted_contains,
    masked_at,
    pack_directed_key,
    pack_edge_key,
)
from triwarp.kernels.halfedge import halfedge_endpoints
from triwarp.kernels.scatter import mark_corners
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


@wp.kernel
def decode_group_vertex_keys(
    unique_keys: wp.array[wp.int64],
    radix: wp.int64,
    vertices: wp.array[wp.vec3],
    out_slot_groups: wp.array[wp.int32],
    out_group_counts: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
) -> None:
    # One thread per sorted unique ``(group, vertex)`` slot: record the slot's group, count it
    # into that group's histogram (the group's vertex count), and gather the source position it
    # names -- the three things the slot's key is needed for, read off one decode. The vertex id
    # itself is never stored, since the position gather is its only consumer.
    slot = wp.int32(wp.tid())
    key = unique_keys[slot]
    # The inverse of ``pack_group_vertex_keys``: one divmod.
    group = wp.int32(key // radix)
    vertex = wp.int32(key % radix)
    out_slot_groups[slot] = group
    wp.atomic_add(out_group_counts, group, wp.int32(1))
    out_vertices[slot] = vertices[vertex]


@wp.kernel
def local_corner_indices(
    inverse: wp.array[wp.int32],
    slot_groups: wp.array[wp.int32],
    vertex_offsets: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    # The compacted, zero-based vertex index each selected corner refers to: its slot's rank within
    # its own group. Resolved per corner through ``inverse`` rather than tabulated per slot and
    # gathered, since the corner is the only reader of a slot's rank.
    corner = wp.int32(wp.tid())
    slot = inverse[corner]
    out_faces[corner] = slot - vertex_offsets[slot_groups[slot]]


# The mask-driven submesh extraction shares one ``(n_faces + n_vertices,)`` rank buffer: the kept
# faces' 0/1 flags followed by their vertices' referenced flags, scanned inclusively as one array.
# A kept face's compact slot is then ``ranks[f] - 1`` and a kept vertex's is
# ``ranks[n_faces + v] - ranks[n_faces - 1] - 1``, so one scan orders both halves and the two
# output sizes come out of one small readback. Both orders are ascending by input index, which is
# what the index-driven path's sorted ``unique_1d`` produces.


@wp.kernel
def mark_submesh_faces_and_vertices(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    keep_masked: wp.bool,
    out_flags: wp.array[wp.int32],
) -> None:
    # ``out_flags`` arrives zeroed. A face is kept when its mask entry equals ``keep_masked``, so
    # one kernel serves a selection and its complement.
    f = wp.int32(wp.tid())
    if face_mask[f] != keep_masked:
        return
    n_faces = face_mask.shape[0]
    a, b, c = corner_triple(faces, f)
    out_flags[f] = 1
    mark_corners(out_flags, n_faces, a, b, c, wp.int32(1))


@wp.kernel
def submesh_counts(
    ranks: wp.array[wp.int32], n_faces: wp.int32, out_counts: wp.array[wp.int32]
) -> None:
    # The kept face count and the kept vertex count, side by side for a single readback.
    face_total = ranks[n_faces - 1]
    out_counts[0] = face_total
    out_counts[1] = ranks[ranks.shape[0] - 1] - face_total


@wp.kernel
def compact_submesh_faces(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    keep_masked: wp.bool,
    ranks: wp.array[wp.int32],
    out_faces: wp.array[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    if face_mask[f] != keep_masked:
        return
    n_faces = face_mask.shape[0]
    offset = ranks[n_faces - 1] + 1
    a, b, c = corner_triple(faces, f)
    slot = 3 * (ranks[f] - 1)
    out_faces[slot] = ranks[n_faces + a] - offset
    out_faces[slot + 1] = ranks[n_faces + b] - offset
    out_faces[slot + 2] = ranks[n_faces + c] - offset


@wp.kernel
def compact_submesh_vertices(
    vertices: wp.array[wp.vec3],
    ranks: wp.array[wp.int32],
    n_faces: wp.int32,
    out_vertices: wp.array[wp.vec3],
    out_vertex_index: wp.array[wp.int32],
) -> None:
    # A vertex is kept when its inclusive rank steps up; for vertex 0 the step is taken from the
    # face half's total, which is exactly the vertex half's baseline.
    v = wp.int32(wp.tid())
    rank = ranks[n_faces + v]
    if rank == ranks[n_faces + v - 1]:
        return
    slot = rank - ranks[n_faces - 1] - 1
    out_vertices[slot] = vertices[v]
    out_vertex_index[slot] = v


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
    faces: wp.array[wp.int32], in_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # One one-ring dilation round, reached through the faces: on a triangle mesh two vertices share
    # an edge exactly when they share a face, so marking every corner of a face with a selected
    # corner adds precisely the edge neighbours, and no unique-edge table has to exist. Measured
    # against the same round over a caller's unique edges, it is level at a hundred thousand faces
    # and 1.2x at tens of millions, so the edge form was retired rather than kept beside it.
    # ``out_mask`` must be pre-seeded with ``in_mask``; re-marking the selected corner is a no-op.
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    if in_mask[a] or in_mask[b] or in_mask[c]:
        mark_corners(out_mask, 0, a, b, c, wp.bool(True))


@wp.kernel
def mark_incident_vertices(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # Mark every corner of every selected face: the inverse direction of
    # ``face_mask_from_vertex_mask``. Concurrent writes all store ``True``, so the race is benign
    # and no atomic is needed. ``dilate_vertex_mask`` above is this composed with an any-corner
    # ``face_mask_from_vertex_mask`` in one pass, for a caller that does not keep the face mask.
    f = wp.int32(wp.tid())
    if not face_mask[f]:
        return
    a, b, c = corner_triple(faces, f)
    mark_corners(out_mask, 0, a, b, c, wp.bool(True))


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
    tail, tip = halfedge_endpoints(faces, h)
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
def deleted_face_edge_keys(
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    kept_ranks: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # The undirected edge keys of the *masked* faces only, three per face at the face's compact
    # rank among the masked faces, so the deleted region's edge table costs three keys per deleted
    # face rather than per mesh face. ``kept_ranks`` is the inclusive scan of the *unmasked* faces
    # (the submesh extraction's face half), so a masked face has ``f - kept_ranks[f]`` masked
    # faces before it.
    f = wp.int32(wp.tid())
    if not face_mask[f]:
        return
    write_face_edge_keys(faces, f, 3 * (f - kept_ranks[f]), base, out_keys)


@wp.kernel
def loops_are_input_rims(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    to_input: wp.array[wp.int32],
    deleted_edge_keys: wp.array[wp.uint64],
    base: wp.uint64,
    out_starts_and_rims: wp.array2d[wp.int32],
) -> None:
    # Per loop: was *every* one of its edges already a boundary edge of the input mesh? A loop for
    # which that holds is the input's own rim surfacing in the submesh rather than a rim the
    # deletion opened, which is what ``delete_region_keep_boundary`` filters on.
    #
    # Answered from the *deletion*, not from the input's boundary, so nothing here is mesh-sized:
    # a loop edge the submesh holds in exactly one kept face had ``1 + (deleted faces containing
    # it)`` faces in the input, so it was an input boundary edge exactly when no deleted face
    # contains it -- one search in ``deleted_edge_keys``, the sorted edge table of the deleted
    # faces in input indices (``to_input`` maps a submesh vertex back).
    #
    # That premise -- every consecutive loop pair is a boundary edge of the submesh -- is what
    # ``boundary_loops_batched`` guarantees on *any* input, which is what makes this safe without a
    # check. Its vertex and seam walks run only on a 2-regular rim. Its pinch walk follows
    # ``next_boundary_halfedge``, whose every successor starts where its predecessor ends (twins
    # are true opposites or ``-1``, even unvalidated), and no rotation can enter another boundary
    # halfedge's face (a face is entered through the twin of its incoming halfedge, which a
    # boundary halfedge lacks), so two rotations never merge and ``successor_cycles`` never sees
    # the colliding input that would leave slots at ``0``. What a mesh that is not edge-manifold
    # *can* do is make a loop go missing -- a rotation stopping at a three-faced edge dead-ends
    # -- and a missing loop defeats any classifier equally. ``tests/test_boundary.py`` pins the
    # guarantee on pinched and non-manifold rims. A kept-face count per loop edge would make it
    # checked here too, and measured as costly as the mesh-sized table this replaces.
    #
    # One thread per *loop*, walking its own rim, rather than one per rim vertex with a segment
    # label: boundary loops are few and short, which is the regime
    # ``kernels/array.segment_owner_labels`` names as the one where the per-segment shape wins --
    # and it needs no owner array, no total-terminated offsets and no scan to build them. The
    # early exit is what makes it cheap in the common case, since a loop the deletion opened
    # usually fails on one of its first edges.
    ell = wp.int32(wp.tid())
    start = loop_starts[ell]
    size = loop_sizes[ell]
    is_rim = wp.int32(1)
    for k in range(size):
        a = to_input[flat_loops[start + k]]
        b = to_input[flat_loops[start + (k + 1) % size]]
        if binary_search_sorted_contains(deleted_edge_keys, pack_edge_key(a, b, base)):
            is_rim = wp.int32(0)
            break
    # Row 0 carries the loop's start and row 1 the verdict, so the caller reads back the one
    # buffer it needs to cut the surviving loops out of ``flat_loops``, in one transfer.
    out_starts_and_rims[0, ell] = start
    out_starts_and_rims[1, ell] = is_rim
