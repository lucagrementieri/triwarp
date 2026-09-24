import warp as wp

from triwarp.kernels.predicates import vector_angle


@wp.kernel
def edge_pair_winding_mask(
    edges: wp.array2d[wp.int32],
    edge_groups: wp.array2d[wp.int32],
    out_consistent: wp.array[wp.bool],
) -> None:
    tid = wp.int32(wp.tid())
    i0 = edge_groups[tid, 0]
    i1 = edge_groups[tid, 1]
    out_consistent[tid] = edges[i0, 1] == edges[i1, 0]


@wp.func
def local_index(faces: wp.array[wp.int32], f: wp.int32, v: wp.int32) -> wp.int32:
    """Position (0, 1, 2) of vertex ``v`` within face ``f``."""
    base = f * wp.int32(3)
    if faces[base] == v:
        return wp.int32(0)
    if faces[base + wp.int32(1)] == v:
        return wp.int32(1)
    return wp.int32(2)


@wp.func
def edge_forward_in_face(
    faces: wp.array[wp.int32], f: wp.int32, a: wp.int32, b: wp.int32
) -> wp.int32:
    """1 if face ``f`` traverses the shared edge as ``a -> b``, else 0 (``b -> a``)."""
    la = local_index(faces, f, a)
    nxt = faces[f * wp.int32(3) + (la + wp.int32(1)) % wp.int32(3)]
    if nxt == b:
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def edge_manifold(count: wp.int32, allow_boundary: wp.bool) -> wp.bool:
    """Per-unique-edge manifold flag from its face-share count."""
    if allow_boundary:
        return count <= wp.int32(2)
    return count == wp.int32(2)


@wp.kernel
def face_edge_manifold_mask(
    inverse: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    allow_boundary: wp.bool,
    out_mask: wp.array[wp.bool],
) -> None:
    """
    Per-face flag: True when all three of a face's undirected edges are edge-manifold.

    ``inverse`` maps each directed edge ``3 * f + k`` (row-major face order from
    ``faces_to_edges``) to its unique-edge index; ``counts`` is each unique edge's face-share count.

    The per-edge flag is [`edge_manifold`][triwarp.kernels.validation.edge_manifold] of the gathered
    count, evaluated here rather than mapped into a ``(n_unique,)`` bool table first: every entry of
    that table was read only through this gather, so the map was a launch and an allocation that
    moved no work.
    """
    f = wp.int32(wp.tid())
    u0 = inverse[3 * f]
    u1 = inverse[3 * f + 1]
    u2 = inverse[3 * f + 2]
    out_mask[f] = (
        edge_manifold(counts[u0], allow_boundary)
        and edge_manifold(counts[u1], allow_boundary)
        and edge_manifold(counts[u2], allow_boundary)
    )


@wp.kernel
def build_corner_adjacency_edges(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    """
    Two corner-graph edges per face-adjacency row (one per shared-edge endpoint).

    Corner node ``3 * f + k`` is the corner of face ``f`` at its local vertex ``k``.
    Two faces sharing edge ``(a, b)`` fan-connect their ``a`` corners and their ``b``
    corners, so the connected components of the corner graph are exactly the
    edge-connected fans around each vertex.
    """
    r = wp.int32(wp.tid())
    f0 = adjacency[r, 0]
    f1 = adjacency[r, 1]
    a = adjacency_edges[r, 0]
    b = adjacency_edges[r, 1]
    la0 = local_index(faces, f0, a)
    la1 = local_index(faces, f1, a)
    lb0 = local_index(faces, f0, b)
    lb1 = local_index(faces, f1, b)
    out_edges[2 * r, 0] = wp.int32(3) * f0 + la0
    out_edges[2 * r, 1] = wp.int32(3) * f1 + la1
    out_edges[2 * r + 1, 0] = wp.int32(3) * f0 + lb0
    out_edges[2 * r + 1, 1] = wp.int32(3) * f1 + lb1


@wp.kernel
def corner_vertex_reduce(
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_min_label: wp.array[wp.int32],
    out_referenced: wp.array[wp.bool],
) -> None:
    """Per-vertex minimum corner-component label and referenced flag."""
    c = wp.int32(wp.tid())
    v = faces[c]
    wp.atomic_min(out_min_label, v, labels[c])
    out_referenced[v] = True


@wp.kernel
def corner_vertex_check(
    faces: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    min_label: wp.array[wp.int32],
    out_mask: wp.array[wp.bool],
) -> None:
    """Clear a vertex flag when one of its corners is in a different fan."""
    c = wp.int32(wp.tid())
    v = faces[c]
    if labels[c] != min_label[v]:
        out_mask[v] = False


@wp.kernel
def build_signed_face_edges(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
    out_sign: wp.array[wp.int32],
) -> None:
    """
    Face-pair edges with a Z2 flip bit for orientability propagation.

    ``sign == 0`` when the two faces already traverse the shared edge in opposite
    directions (compatible orientations), ``sign == 1`` when a flip is required.
    """
    # The five opening statements match ``build_corner_adjacency_edges`` above, and that is
    # *not* an unfactored duplicate: they are this kernel's own argument list unpacked -- the
    # adjacency row's two faces and the two endpoints of the edge they share -- so a shared
    # ``@wp.func`` would name what the two signatures already say and cost a longer call than the
    # four reads it replaced. The two kernels diverge at the line after, which is the whole point
    # of each.
    r = wp.int32(wp.tid())
    f0 = adjacency[r, 0]
    f1 = adjacency[r, 1]
    a = adjacency_edges[r, 0]
    b = adjacency_edges[r, 1]
    d0 = edge_forward_in_face(faces, f0, a, b)
    d1 = edge_forward_in_face(faces, f1, a, b)
    out_edges[r, 0] = f0
    out_edges[r, 1] = f1
    out_sign[r] = wp.where(d0 == d1, wp.int32(1), wp.int32(0))


@wp.kernel
def verify_orientation(
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    out_conflict: wp.array[wp.int32],
) -> None:
    """Flag any face-adjacency edge whose endpoints violate the flip constraint."""
    r = wp.int32(wp.tid())
    f0 = edges[r, 0]
    f1 = edges[r, 1]
    if ((orient[f0] + orient[f1]) & wp.int32(1)) != signs[r]:
        out_conflict[0] = wp.int32(1)


@wp.kernel
def accumulate_neighbor_normals(
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    out_neighbor_sum: wp.array[wp.vec3],
    out_max_angle: wp.array[wp.float32],
) -> None:
    # One pass over the adjacency pairs, scattering both per-face quantities the bad-face criteria
    # need: the sum of the neighbouring face normals (whose direction is the local consensus) and
    # the sharpest dihedral angle to any neighbour (which is what a fold looks like).
    #
    # The dihedral angle is ``adjacency.face_adjacency_angles``' -- ``vector_angle`` of the pair's
    # two normals -- taken from the two normals this thread loads anyway, rather than read from a
    # ``(m,)`` table a launch of its own would have written for this pass alone.
    k = wp.int32(wp.tid())
    f0 = face_adjacency[k, 0]
    f1 = face_adjacency[k, 1]
    normal_0 = face_normals[f0]
    normal_1 = face_normals[f1]
    angle = vector_angle(normal_0, normal_1)
    wp.atomic_add(out_neighbor_sum, f0, normal_1)
    wp.atomic_add(out_neighbor_sum, f1, normal_0)
    wp.atomic_max(out_max_angle, f0, angle)
    wp.atomic_max(out_max_angle, f1, angle)


@wp.kernel
def face_defective_mask(
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
    #
    # The same sentinels also say which tables exist: a disabled criterion's inputs are never read,
    # so the wrapper passes ``None`` for them rather than allocating a table of constants (the two
    # branches are warp-uniform). A disabled pair of angle criteria reaches ``False`` below exactly
    # as their zeroed tables did: the zero consensus fails the normal test and ``cos(0)`` the fold.
    f = wp.int32(wp.tid())
    if min_quality > -1.0:
        if quality[f] < min_quality:
            out_bad[f] = wp.bool(True)
            return
    if max_normal_cos <= -2.0 and max_fold_cos <= -2.0:
        out_bad[f] = wp.bool(False)
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
