import warp as wp


@wp.kernel
def edge_pair_winding_mask(
    edges: wp.array2d[wp.int32],
    edge_groups: wp.array2d[wp.int32],
    out_consistent: wp.array[wp.bool],
) -> None:
    tid = int(wp.tid())
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


@wp.kernel
def mark_intersecting_faces(
    pairs: wp.array2d[wp.int32], valid: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    """Flag both faces of each intersecting candidate pair (idempotent ``True`` writes)."""
    p = int(wp.tid())
    if valid[p]:
        out_mask[pairs[p, 0]] = True
        out_mask[pairs[p, 1]] = True


@wp.func
def edge_manifold(count: wp.int32, allow_boundary: wp.bool) -> wp.bool:
    """Per-unique-edge manifold flag from its face-share count."""
    if allow_boundary:
        return count <= wp.int32(2)
    return count == wp.int32(2)


@wp.kernel
def face_edge_manifold_mask(
    inverse: wp.array[wp.int32], edge_manifold: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    """
    Per-face flag: True when all three of a face's undirected edges are edge-manifold.

    ``inverse`` maps each directed edge ``3 * f + k`` (row-major face order from
    ``faces_to_edges``) to its unique-edge index; ``edge_manifold`` is the per-unique-edge flag.
    """
    f = int(wp.tid())
    u0 = inverse[3 * f]
    u1 = inverse[3 * f + 1]
    u2 = inverse[3 * f + 2]
    out_mask[f] = edge_manifold[u0] and edge_manifold[u1] and edge_manifold[u2]


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
    r = int(wp.tid())
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
    c = int(wp.tid())
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
    c = int(wp.tid())
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
    r = int(wp.tid())
    f0 = adjacency[r, 0]
    f1 = adjacency[r, 1]
    a = adjacency_edges[r, 0]
    b = adjacency_edges[r, 1]
    d0 = edge_forward_in_face(faces, f0, a, b)
    d1 = edge_forward_in_face(faces, f1, a, b)
    out_edges[r, 0] = f0
    out_edges[r, 1] = f1
    if d0 == d1:
        out_sign[r] = wp.int32(1)
    else:
        out_sign[r] = wp.int32(0)


@wp.kernel
def seed_orientation(labels: wp.array[wp.int32], out_orient: wp.array[wp.int32]) -> None:
    """Seed one orientation bit (0) per component representative, -1 elsewhere."""
    f = int(wp.tid())
    if labels[f] == f:
        out_orient[f] = wp.int32(0)
    else:
        out_orient[f] = wp.int32(-1)


@wp.kernel
def propagate_orientation(
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    changed: wp.array[wp.int32],
) -> None:
    """Push assigned orientation bits across face-adjacency edges via the flip bit."""
    r = int(wp.tid())
    f0 = edges[r, 0]
    f1 = edges[r, 1]
    s = signs[r]
    o0 = orient[f0]
    o1 = orient[f1]
    if o0 >= wp.int32(0) and o1 < wp.int32(0):
        orient[f1] = (o0 + s) & wp.int32(1)
        changed[0] = wp.int32(1)
    elif o1 >= wp.int32(0) and o0 < wp.int32(0):
        orient[f0] = (o1 + s) & wp.int32(1)
        changed[0] = wp.int32(1)


@wp.kernel
def verify_orientation(
    edges: wp.array2d[wp.int32],
    signs: wp.array[wp.int32],
    orient: wp.array[wp.int32],
    conflict: wp.array[wp.int32],
) -> None:
    """Flag any face-adjacency edge whose endpoints violate the flip constraint."""
    r = int(wp.tid())
    f0 = edges[r, 0]
    f1 = edges[r, 1]
    if ((orient[f0] + orient[f1]) & wp.int32(1)) != signs[r]:
        conflict[0] = wp.int32(1)
