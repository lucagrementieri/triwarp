import warp as wp

from triwarp.kernels.array import update_argmin_pair
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.predicates import triangle_aspect_ratio, triangle_normal
from triwarp.kernels.triangles import corner_triple, triangle_cross


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
def flip_faces_masked(
    faces: wp.array[wp.int32], flip: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    """Copy ``faces`` to ``out_faces``, reversing winding (swap corners 1,2) where ``flip > 0``."""
    f = wp.int32(wp.tid())
    base = f * wp.int32(3)
    i0 = faces[base]
    i1 = faces[base + wp.int32(1)]
    i2 = faces[base + wp.int32(2)]
    out_faces[base] = i0
    if flip[f] > wp.int32(0):
        out_faces[base + wp.int32(1)] = i2
        out_faces[base + wp.int32(2)] = i1
    else:
        out_faces[base + wp.int32(1)] = i1
        out_faces[base + wp.int32(2)] = i2


@wp.func
def negative_volume_flag(volume: wp.float32) -> wp.int32:
    """Flag a face for flipping when its component's signed volume is negative (inward)."""
    return wp.where(volume < wp.float32(0.0), wp.int32(1), wp.int32(0))


@wp.kernel
def accumulate_neighbor_normals(
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    adjacency_angles: wp.array[wp.float32],
    out_neighbor_sum: wp.array[wp.vec3],
    out_max_angle: wp.array[wp.float32],
) -> None:
    # One pass over the adjacency pairs, scattering both per-face quantities the bad-face criteria
    # need: the sum of the neighbouring face normals (whose direction is the local consensus) and
    # the sharpest dihedral angle to any neighbour (which is what a fold looks like).
    k = wp.int32(wp.tid())
    f0 = face_adjacency[k, 0]
    f1 = face_adjacency[k, 1]
    angle = adjacency_angles[k]
    wp.atomic_add(out_neighbor_sum, f0, face_normals[f1])
    wp.atomic_add(out_neighbor_sum, f1, face_normals[f0])
    wp.atomic_max(out_max_angle, f0, angle)
    wp.atomic_max(out_max_angle, f1, angle)


@wp.kernel
def bad_face_mask(
    quality: wp.array[wp.float32],
    face_normals: wp.array[wp.vec3],
    neighbor_sum: wp.array[wp.vec3],
    max_angle: wp.array[wp.float32],
    min_quality: wp.float32,
    max_normal_cos: wp.float32,
    max_fold_cos: wp.float32,
    out_bad: wp.array[wp.bool],
) -> None:
    # A face is bad if it is too thin, too far from its neighbourhood's consensus normal, or folded
    # back onto its own ring. Each criterion is disabled by passing a cosine of -2 / a quality of
    # -1, which no real value can reach, so the three gates compose without a separate flag
    # argument.
    f = wp.int32(wp.tid())
    if quality[f] < min_quality:
        out_bad[f] = wp.bool(True)
        return

    # Direction of the sum of the neighbouring normals: the local consensus. ``wp.normalize`` of a
    # zero vector is zero in Warp, which is how an isolated face -- and a face whose neighbours
    # cancel each other exactly -- ends up with no consensus to disagree with.
    consensus = wp.normalize(neighbor_sum[f])
    agreement = wp.dot(face_normals[f], consensus)
    if wp.length_sq(consensus) > 0.0 and agreement < max_normal_cos:
        out_bad[f] = wp.bool(True)
        return

    # Fold: some neighbour meets this face at nearly pi (an unsigned dihedral, so the gate is on its
    # cosine) *and* this face is the one facing against its own neighbourhood. Both halves matter --
    # a fold has two faces and only one of them is the mistake, so flagging both would delete a good
    # triangle along with it.
    out_bad[f] = wp.cos(max_angle[f]) < max_fold_cos and agreement < 0.0


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
    forward_next = (forward // 3) * 3 + (forward + 1) % 3
    backward_next = (backward // 3) * 3 + (backward + 1) % 3
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


@wp.kernel
def select_independent_degree3(
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    out_selected: wp.array[wp.bool],
) -> None:
    # Two adjacent candidates share faces, so only one of them can be removed in a pass. The lowest
    # index wins, which makes the choice deterministic and independent of launch order -- the same
    # rule the Delaunay flip pass uses to resolve competing edges.
    v = wp.int32(wp.tid())
    if not candidate[v]:
        return
    for slot in range(ring_offsets[v], ring_offsets[v + 1]):
        neighbour = halfedge_destination(faces, ring_halfedges[slot])
        if candidate[neighbour] and neighbour < v:
            return
    out_selected[v] = True


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


@wp.kernel
def collect_rim_links(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    out_rim_next: wp.array[wp.int32],
    out_rim_prev: wp.array[wp.int32],
    out_rim_face: wp.array[wp.int32],
) -> None:
    # The rim as a per-vertex linked list, from the halfedges that have no twin. A boundary vertex
    # of an edge-manifold mesh has exactly one outgoing boundary halfedge, so nothing races: each
    # slot is written once. ``rim_face`` is the face owning the *outgoing* one, which is the face
    # whose normal a new triangle at that vertex has to agree with.
    h = wp.int32(wp.tid())
    if twins[h] >= 0:
        return
    tail = faces[h]
    tip = halfedge_destination(faces, h)
    out_rim_next[tail] = tip
    out_rim_prev[tip] = tail
    out_rim_face[tail] = h // 3


@wp.kernel
def straighten_candidate_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    rim_next: wp.array[wp.int32],
    rim_prev: wp.array[wp.int32],
    rim_face: wp.array[wp.int32],
    min_normal_dot: wp.float32,
    max_aspect_ratio: wp.float32,
    out_candidate: wp.array[wp.bool],
) -> None:
    # A rim vertex whose notch can be closed by one triangle. The rim runs ``prev -> v -> next``
    # with the surface on its left, so a face attached outside it must carry the halfedges
    # ``v -> prev`` and ``next -> v`` -- which is the triangle ``(next, v, prev)``, and that winding
    # is what makes the result consistently oriented rather than merely watertight.
    #
    # Two gates, both from the caller: the new triangle's normal must agree with the two rim faces
    # it will border (which is what stops a *convex* corner being folded over), and its aspect ratio
    # must be finite enough to be worth adding.
    v = wp.int32(wp.tid())
    previous = rim_prev[v]
    following = rim_next[v]
    if previous < 0 or following < 0 or previous == following:
        return
    normal = triangle_normal(vertices[following], vertices[v], vertices[previous])
    if wp.length(normal) == 0.0:
        return
    if wp.dot(normal, face_normals[rim_face[previous]]) < min_normal_dot:
        return
    if wp.dot(normal, face_normals[rim_face[v]]) < min_normal_dot:
        return
    ratio = triangle_aspect_ratio(vertices[following], vertices[v], vertices[previous])
    if ratio > max_aspect_ratio:
        return
    out_candidate[v] = True


@wp.kernel
def emit_straighten_faces(
    faces: wp.array[wp.int32],
    rim_next: wp.array[wp.int32],
    rim_prev: wp.array[wp.int32],
    candidate: wp.array[wp.bool],
    cursor: wp.array[wp.int32],
    out_new_faces: wp.array2d[wp.int32],
) -> None:
    # One new face per accepted notch, and only where no rim neighbour with a lower index also
    # qualifies -- two adjacent notches share a rim edge, so filling both in one pass would attach
    # two faces to it.
    v = wp.int32(wp.tid())
    if not candidate[v]:
        return
    previous = rim_prev[v]
    following = rim_next[v]
    if (candidate[previous] and previous < v) or (candidate[following] and following < v):
        return
    slot = wp.atomic_add(cursor, 0, 1)
    out_new_faces[slot, 0] = following
    out_new_faces[slot, 1] = v
    out_new_faces[slot, 2] = previous


@wp.kernel
def flatten_degree3_positions(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    region: wp.array[wp.bool],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Move each interior valence-3 vertex to the centroid of its three neighbours -- which lies in
    # their plane, so the little tetrahedral bump flattens into it. Two such vertices cannot be
    # neighbours on an edge-manifold mesh (each would need the other's whole ring), so every thread
    # reads positions no other thread is writing and one pass is enough.
    vertex = wp.int32(wp.tid())
    begin = ring_offsets[vertex]
    end = ring_offsets[vertex + 1]
    if not region[vertex] or not is_interior_degree3(begin, end, is_boundary[vertex]):
        out_positions[vertex] = positions[vertex]
        return
    total = wp.vec3()
    for slot in range(begin, end):
        total += positions[halfedge_destination(faces, ring_halfedges[slot])]
    out_positions[vertex] = total / wp.float32(3.0)
