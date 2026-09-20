import warp as wp

from triwarp.kernels.array import (
    declare_map_signatures,
    map_probe,
    map_probe_single,
    pack_ranked_key,
    update_argmin_pair,
)
from triwarp.kernels.halfedge import halfedge_destination, halfedge_next
from triwarp.kernels.predicates import triangle_aspect_ratio, triangle_normal
from triwarp.kernels.triangles import corner_triple, triangle_cross, write_corner_triple


@wp.func
def cyclic_match(
    a0: wp.int32, a1: wp.int32, a2: wp.int32, b0: wp.int32, b1: wp.int32, b2: wp.int32
) -> wp.bool:
    """Whether ``(a0, a1, a2)`` is a cyclic rotation of ``(b0, b1, b2)`` (same orientation)."""
    return (
        (a0 == b0 and a1 == b1 and a2 == b2)
        or (a0 == b1 and a1 == b2 and a2 == b0)
        or (a0 == b2 and a1 == b0 and a2 == b1)
    )


@wp.kernel
def scatter_duplicate_face_stats(
    faces: wp.array[wp.int32],
    unique_faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_member_count: wp.array[wp.int32],
    out_signed_count: wp.array[wp.int32],
    out_first_member: wp.array[wp.int32],
    out_first_positive: wp.array[wp.int32],
    out_first_negative: wp.array[wp.int32],
) -> None:
    # Per input face: orientation sign against its group representative (the first-occurring
    # face, gathered via ``inverse``) scattered into per-group stats. ``first_*`` slots hold the
    # smallest face index of each sign class (seeded with the n_faces sentinel).
    f = wp.int32(wp.tid())
    ui = inverse[f]
    consistent = cyclic_match(
        faces[f * 3],
        faces[f * 3 + 1],
        faces[f * 3 + 2],
        unique_faces[ui * 3],
        unique_faces[ui * 3 + 1],
        unique_faces[ui * 3 + 2],
    )
    wp.atomic_add(out_member_count, ui, wp.int32(1))
    wp.atomic_min(out_first_member, ui, f)
    if consistent:
        wp.atomic_add(out_signed_count, ui, wp.int32(1))
        wp.atomic_min(out_first_positive, ui, f)
    else:
        wp.atomic_add(out_signed_count, ui, wp.int32(-1))
        wp.atomic_min(out_first_negative, ui, f)


@wp.kernel
def resolve_duplicate_groups(
    member_count: wp.array[wp.int32],
    signed_count: wp.array[wp.int32],
    first_member: wp.array[wp.int32],
    first_positive: wp.array[wp.int32],
    first_negative: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_error_group: wp.array[wp.int32],
) -> None:
    # Keep-decision per duplicate group (igl::resolve_duplicated_faces): singletons stay; a net
    # +1/-1 orientation keeps the first member of the majority sign; a cancelling group drops;
    # anything else is non-orientable (the smallest offending group index is reported).
    ui = wp.int32(wp.tid())
    count = signed_count[ui]
    if member_count[ui] == 1:
        out_keep[ui] = first_member[ui]
    elif count == 1:
        out_keep[ui] = first_positive[ui]
    elif count == -1:
        out_keep[ui] = first_negative[ui]
    else:
        out_keep[ui] = wp.int32(-1)
        if count != 0:
            wp.atomic_min(out_error_group, 0, ui)


@wp.kernel
def reduce_largest_group(counts: wp.array[wp.int32], out_best: wp.array[wp.int64]) -> None:
    # The group with the most members, ties going to the lowest group index, reduced into one
    # ``int64`` by ``pack_ranked_key`` so the whole answer is a single ``wp.atomic_max``. Launch
    # over the group domain; ``out_best`` is one element seeded to ``-1``, which every real key
    # exceeds.
    #
    # An empty group contributes nothing, which matters because the group domain here is the *face*
    # domain -- a connected-component label is a representative face index, so most slots are empty.
    group = wp.int32(wp.tid())
    if counts[group] > 0:
        wp.atomic_max(out_best, 0, pack_ranked_key(counts[group], group))


@wp.kernel
def mark_largest_group_mask(
    groups: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    best: wp.array[wp.int64],
    out_mask: wp.array[wp.bool],
) -> None:
    # Flag every member of the group ``reduce_largest_group`` chose. Recomputing the key and testing
    # it against the reduced maximum is what keeps this readback-free: the winning group index is
    # never brought to the host, and the ``-1`` seed marks nothing when there are no groups at all.
    #
    # Not convertible to a ``wp.map`` over a gather, unlike the *threshold* criteria beside it in
    # ``repair.remove_small_components``: those are one table lookup compared against a scalar, so
    # ``wp.map(greater_equal, statistic[labels], threshold)`` is the whole kernel, whereas this
    # needs two gathers and a call to rebuild the key before it can compare.
    f = wp.int32(wp.tid())
    group = groups[f]
    out_mask[f] = pack_ranked_key(counts[group], group) == best[0]


@wp.kernel(enable_backward=False)
def small_triangle_collapse_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_dbl_area: wp.float32,
    out_pairs: wp.array2d[wp.int32],
    out_flag: wp.array[wp.int32],
) -> None:
    """Flag faces with double-area below ``min_dbl_area`` and emit their shortest edge (libigl)."""
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    dbl_area = wp.length(triangle_cross(vertices, faces, f))
    if dbl_area < min_dbl_area:
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        # Shortest of the three edges (0,1), (1,2), (2,0); collapse its endpoints together.
        best = wp.length_sq(v1 - v0)
        a = i0
        b = i1
        update_argmin_pair(best, a, b, wp.length_sq(v2 - v1), i1, i2)
        update_argmin_pair(best, a, b, wp.length_sq(v0 - v2), i2, i0)
        out_pairs[f, 0] = a
        out_pairs[f, 1] = b
        out_flag[f] = wp.int32(1)
    else:
        out_pairs[f, 0] = i0
        out_pairs[f, 1] = i0
        out_flag[f] = wp.int32(0)


@wp.kernel
def reverse_face_winding(faces: wp.array[wp.int32], out_faces: wp.array[wp.int32]) -> None:
    # np.fliplr on an (n, 3) face block. All three indices are read before any is written, so
    # this is safe to run in place (out_faces is faces).
    f = wp.int32(wp.tid())
    a, b, c = corner_triple(faces, f)
    write_corner_triple(out_faces, f, c, b, a)


@wp.func
def write_face_winding(
    faces: wp.array[wp.int32], f: wp.int32, reversed_winding: wp.bool, out_faces: wp.array[wp.int32]
) -> None:
    """Copy face ``f``'s corners, swapping corners 1 and 2 when ``reversed_winding``."""
    base = f * wp.int32(3)
    i0 = faces[base]
    i1 = faces[base + wp.int32(1)]
    i2 = faces[base + wp.int32(2)]
    out_faces[base] = i0
    if reversed_winding:
        out_faces[base + wp.int32(1)] = i2
        out_faces[base + wp.int32(2)] = i1
    else:
        out_faces[base + wp.int32(1)] = i1
        out_faces[base + wp.int32(2)] = i2


@wp.kernel
def flip_faces_masked(
    faces: wp.array[wp.int32], flip: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    """Copy ``faces`` to ``out_faces``, reversing winding (swap corners 1,2) where ``flip > 0``."""
    f = wp.int32(wp.tid())
    write_face_winding(faces, f, flip[f] > wp.int32(0), out_faces)


@wp.kernel
def flip_faces_by_component_volume(
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    component_volume: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
) -> None:
    # ``make_volume(multibody=True)``'s orientation pass: a face is reversed when the component it
    # belongs to encloses a negative signed volume, i.e. is wound inward.
    #
    # The component's volume is read through this face's own label rather than gathered into a
    # per-face buffer first: the Python-scope form was an ``indexedarray`` view, a ``wp.map`` to
    # materialise a face-sized flag array, and then this pass to consume it -- two launches and a
    # buffer for a value one indirection away. Measured 1.15x on ``make_volume(multibody=True)``.
    f = wp.int32(wp.tid())
    write_face_winding(faces, f, component_volume[labels[f]] < wp.float32(0.0), out_faces)


@wp.kernel
def halfedge_orientation_slots(
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    out_forward_count: wp.array[wp.int32],
    out_backward_count: wp.array[wp.int32],
    out_forward_corner: wp.array[wp.int32],
    out_backward_corner: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many half-edges traverse it each way, and which corner they belong to.
    #
    # Corner ``c = 3f + j`` is the half-edge ``(fv[j], fv[j + 1])``. ``unique_edges`` rows are
    # sorted min-first, so a half-edge is "forward" when it runs low index to high. A
    # *consistently oriented manifold* edge has exactly one of each; anything else -- two forward
    # (a flipped neighbour), three or more of either (a non-manifold edge), or one alone (a
    # boundary) -- is not mergeable, so its corner slot is never read and the ``atomic_max`` only
    # keeps the write deterministic.
    c = wp.int32(wp.tid())
    e = edge_of_corner[c]
    f = c // 3
    j = c % 3
    if faces[c] < faces[f * 3 + (j + 1) % 3]:
        wp.atomic_add(out_forward_count, e, 1)
        wp.atomic_max(out_forward_corner, e, c)
    else:
        wp.atomic_add(out_backward_count, e, 1)
        wp.atomic_max(out_backward_corner, e, c)


@wp.kernel
def corner_merge_links(
    forward_count: wp.array[wp.int32],
    backward_count: wp.array[wp.int32],
    forward_corner: wp.array[wp.int32],
    backward_corner: wp.array[wp.int32],
    out_links: wp.array2d[wp.int32],
) -> None:
    # Two corner-graph links per mergeable edge: one joining the corners at each endpoint.
    #
    # Nodes are ``(face, vertex slot)`` pairs under the same ``3f + k`` numbering, so node
    # ``3f + k`` *is* the corner whose vertex is ``faces[3f + k]``, and a link only ever joins two
    # copies of one original vertex. Connected components of this graph are the vertex copies.
    #
    # A non-mergeable edge emits two self-loops on node 0 rather than nothing, which keeps the
    # output a fixed ``(2 * n_edges, 2)`` buffer with no compaction pass; a self-loop merges
    # nothing.
    e = wp.int32(wp.tid())
    out_links[e * 2 + 0, 0] = 0
    out_links[e * 2 + 0, 1] = 0
    out_links[e * 2 + 1, 0] = 0
    out_links[e * 2 + 1, 1] = 0
    if forward_count[e] != 1 or backward_count[e] != 1:
        return

    forward = forward_corner[e]
    backward = backward_corner[e]
    # The forward half-edge runs (low, high) from its own corner; the backward one runs (high, low),
    # so its *next* slot holds the low endpoint.
    forward_next = halfedge_next(forward)
    backward_next = halfedge_next(backward)
    out_links[e * 2 + 0, 0] = forward  # low endpoint, forward face
    out_links[e * 2 + 0, 1] = backward_next  # low endpoint, backward face
    out_links[e * 2 + 1, 0] = forward_next  # high endpoint, forward face
    out_links[e * 2 + 1, 1] = backward  # high endpoint, backward face


@wp.func
def is_interior_degree3(ring_start: wp.int32, ring_end: wp.int32, on_boundary: wp.bool) -> wp.bool:
    # An interior vertex with exactly three incident faces. Its ring size *is* the face count, so a
    # boundary vertex with three faces has four neighbours and is excluded by the flag rather than
    # by the count.
    return not on_boundary and ring_end - ring_start == 3


@wp.func
def wins_degree3_conflict(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    v: wp.int32,
) -> wp.bool:
    # Two adjacent candidates share faces, so only one of them can be removed in a pass. The lowest
    # index wins, which makes the choice deterministic and independent of launch order -- the same
    # rule the Delaunay flip pass uses to resolve competing edges.
    #
    # Shared by the two kernels below rather than written twice: ``remove_degree3_vertices`` wants
    # the selection on its own, ``flatten_degree3_vertices`` wants it fused with the move it
    # implies, and this is the rule both are deciding.
    if not candidate[v]:
        return False
    for slot in range(ring_offsets[v], ring_offsets[v + 1]):
        neighbour = halfedge_destination(faces, ring_halfedges[slot])
        if candidate[neighbour] and neighbour < v:
            return False
    return True


@wp.kernel
def select_independent_degree3(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    out_selected: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
) -> None:
    # ``out_count`` is the size of the selection, which the caller needs to size the replacement
    # face buffer and to decide whether the pass did anything. Counting it here is one *conditional*
    # atomic per selected vertex -- contention scales with the (rare) selections, not with the
    # launch -- against the whole-array cast, reduction and readback the caller would otherwise run
    # to recover a number this kernel already knows.
    v = wp.int32(wp.tid())
    if not wins_degree3_conflict(faces, ring_offsets, ring_halfedges, candidate, v):
        return
    out_selected[v] = True
    wp.atomic_add(out_count, 0, 1)


@wp.kernel
def emit_degree3_replacement(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    selected: wp.array[wp.bool],
    cursor: wp.array[wp.int32],
    out_dropped: wp.array[wp.bool],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    # The three faces around a selected vertex become one: its link is a triangle already, since the
    # ring is counter-clockwise and has exactly three entries. Winding follows the ring, so the
    # replacement points the same way the fan did.
    v = wp.int32(wp.tid())
    if not selected[v]:
        return
    begin = ring_offsets[v]
    slot = wp.atomic_add(cursor, 0, 1)
    for k in range(3):
        out_dropped[ring_halfedges[begin + k] // 3] = True
        out_new_faces[slot, k] = halfedge_destination(faces, ring_halfedges[begin + k])


@wp.func
def next_boundary_halfedge(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], h: wp.int32
) -> wp.int32:
    # The boundary halfedge following ``h`` around the rim, or -1 if the fan does not close.
    #
    # Rotate around ``h``'s tip through the faces on ``h``'s own side until an outgoing halfedge
    # with no twin turns up. That sector is what makes this well defined where a per-*vertex* table
    # is not: a bowtie rim vertex -- two rim loops pinched at one point, edge-manifold and
    # consistently wound, so nothing upstream rejects it -- has two outgoing boundary halfedges,
    # and this picks the one bounding the same sector as ``h`` rather than whichever won a race.
    # The map is a bijection on boundary halfedges for the same reason, which is what lets the
    # caller invert it by scatter without an atomic.
    #
    # The bound is the face count because a fan cannot be longer than the mesh; it is never
    # approached, and it is here so a malformed twin table cannot spin forever.
    g = halfedge_next(h)
    for _ in range(faces.shape[0] // 3):
        if twins[g] < 0:
            return g
        g = halfedge_next(twins[g])
    return -1


@wp.kernel
def collect_rim_links_and_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    min_normal_dot: wp.float32,
    max_aspect_ratio: wp.float32,
    out_rim_next: wp.array[wp.int32],
    out_rim_prev: wp.array[wp.int32],
    out_candidate: wp.array[wp.bool],
) -> None:
    # The rim as a linked list over *halfedges*, from those that have no twin.
    #
    # Keyed by halfedge rather than by vertex deliberately. Edge-manifoldness -- the caller's only
    # validation -- does not make the rim a set of simple loops through the vertices: at a bowtie
    # vertex two rim loops pass through one point, so a per-vertex ``next`` / ``prev`` / ``face``
    # triple has two writers and no answer, and the three slots could be won by different halfedges
    # so that the fold-over normal gate tested a face not bordering the candidate triangle at all
    # (measured: the two devices disagreed on one five-vertex bowtie). Per halfedge each slot has
    # exactly one writer, both rim loops keep their own links, and the bordering faces are read off
    # the halfedges themselves rather than from a third table.
    #
    # The notch test below rides in this same pass. It reads ``rim_next`` only at its own
    # halfedge, which this thread has just computed and still holds in ``following`` -- so run
    # apart it cost a launch and a full round trip of the link table through global memory to
    # recover a successor that never had to leave a register. ``out_rim_prev`` is still scattered
    # for the emit pass that follows, which genuinely does need every link written.
    #
    # Output byte-identical, and the wall clock is **flat** within run-to-run noise: a pass is a
    # halfedge-twin build plus a handful of launches around one readback, so the host was already
    # waiting on the device. What this buys is a launch and a round trip of the link table, not a
    # measurable speed-up -- worth having, not worth quoting.
    h = wp.int32(wp.tid())
    if twins[h] >= 0:
        return
    following = next_boundary_halfedge(faces, twins, h)
    out_rim_next[h] = following
    if following >= 0:
        out_rim_prev[following] = h
    # A notch that can be closed by one triangle, indexed by the boundary halfedge *entering* it.
    # That halfedge runs ``previous -> v`` and its successor runs ``v -> following``, with the
    # surface on their left, so a face attached outside the rim must carry the halfedges
    # ``v -> previous`` and ``following -> v`` -- which is the triangle ``(following, v,
    # previous)``, and that winding is what makes the result consistently oriented rather than
    # merely watertight.
    #
    # Two gates, both from the caller: the new triangle's normal must agree with the two rim faces
    # it will border -- which are the two halfedges' own faces, so no lookup can pair it with the
    # wrong one -- and its aspect ratio must be finite enough to be worth adding.
    outgoing = following
    if outgoing < 0:
        return
    previous = faces[h]
    v = halfedge_destination(faces, h)
    following = halfedge_destination(faces, outgoing)
    if previous == following:
        return  # a two-edge rim loop spans no notch
    normal = triangle_normal(vertices[following], vertices[v], vertices[previous])
    if wp.length(normal) == 0.0:
        return
    if wp.dot(normal, face_normals[h // 3]) < min_normal_dot:
        return
    if wp.dot(normal, face_normals[outgoing // 3]) < min_normal_dot:
        return
    ratio = triangle_aspect_ratio(vertices[following], vertices[v], vertices[previous])
    if ratio > max_aspect_ratio:
        return
    out_candidate[h] = True


@wp.kernel
def emit_straighten_faces(
    faces: wp.array[wp.int32],
    rim_next: wp.array[wp.int32],
    rim_prev: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    cursor: wp.array[wp.int32],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    # One new face per accepted notch, and only where neither rim neighbour with a lower index also
    # qualifies -- two adjacent notches share a rim edge, so filling both in one pass would attach
    # two faces to it. Lowest halfedge index wins, which makes the choice deterministic.
    #
    # ``rim_next[h]`` is read unguarded because ``candidate[h]`` is set only where it is a real
    # halfedge; ``rim_prev[h]`` is guarded because a fan that failed to close leaves it unwritten.
    h = wp.int32(wp.tid())
    if not candidate[h]:
        return
    outgoing = rim_next[h]
    incoming = rim_prev[h]
    if candidate[outgoing] and outgoing < h:
        return
    if incoming >= 0 and candidate[incoming] and incoming < h:
        return
    slot = wp.atomic_add(cursor, 0, 1)
    out_new_faces[slot, 0] = halfedge_destination(faces, outgoing)
    out_new_faces[slot, 1] = halfedge_destination(faces, h)
    out_new_faces[slot, 2] = faces[h]


@wp.kernel
def select_and_flatten_degree3(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    out_selected: wp.array[wp.bool],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Move each selected vertex to the centroid of its three neighbours -- which lies in their
    # plane, so the little tetrahedral bump flattens into it.
    #
    # ``selected`` is an *independent* set of interior valence-3 vertices, and the independence is
    # what makes the centroid the answer rather than an approximation of it: two such vertices can
    # be neighbours (a tetrahedron is four of them, each adjacent to the other three), and moving
    # both at once puts neither in the other's *new* plane. Doing it anyway maps the regular
    # tetrahedron to a mirrored fraction of itself -- every normal flipped and the signed volume
    # negated -- which is why the caller runs ``select_independent_degree3`` first rather than
    # launching this over every candidate.
    #
    # The divisor is a literal 3 because ``selected`` implies interior valence 3; the ring walk is
    # over ``ring_offsets`` all the same, so a stale mask cannot make it read past the ring.
    # The selection and the move it implies, in one pass. They are the same thread's decision about
    # the same vertex, so running them apart cost a launch and a full round trip of the selection
    # mask through global memory to tell this kernel what the previous one had just decided. No
    # ``out_count`` here, unlike ``select_independent_degree3``: this caller's loop reads the
    # remaining candidate count instead, and the positions are written for *every* vertex because
    # the caller ping-pongs two buffers and an unwritten slot would hold an iteration-old value.
    #
    # Measured 1.12x on ``flatten_degree3_vertices`` at 3 413 flattened vertices, byte-identical.
    # The ring walk is done twice in the taken branch -- once to decide, once to average -- and
    # that is still cheaper than writing the mask out and reading it back.
    vertex = wp.int32(wp.tid())
    if not wins_degree3_conflict(faces, ring_offsets, ring_halfedges, candidate, vertex):
        out_positions[vertex] = positions[vertex]
        return
    out_selected[vertex] = True
    total = wp.vec3()
    for slot in range(ring_offsets[vertex], ring_offsets[vertex + 1]):
        total += positions[halfedge_destination(faces, ring_halfedges[slot])]
    out_positions[vertex] = total / wp.float32(3.0)


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not two.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was derived
    and what forks a ``wp.map`` module; only this module's *own* forking ops belong here (the shared
    builtins are declared there).

    One op: ``is_interior_degree3``, mapped over ``(ring_offsets[:-1], ring_offsets[1:],
    is_boundary)`` by both ``repair.remove_degree3_vertices`` and
    ``repair.flatten_degree3_vertices``. The length-1 row is not speculative -- a ring CSR over a
    one-vertex mesh makes both offset slices length 1, and the broadcast mask is part of the cache
    key, so that call would fork the module on first use.
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (is_interior_degree3, (dense(wp.int32), dense(wp.int32), dense(wp.bool)), wp.bool),
            (is_interior_degree3, (single(wp.int32), single(wp.int32), single(wp.bool)), wp.bool),
        ]
    )


_declare_map_kernels()
