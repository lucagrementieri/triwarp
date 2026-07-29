import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT

# The intrinsic flip reuses the parallel flip core from ``kernels.remesh``: quad resolution with its
# guards (missing apex, inconsistent winding, a flip that would duplicate an existing edge) and the
# claim/commit pair that picks a conflict-free independent set. Only the *predicate* and the
# edge-length bookkeeping are new here, which is exactly what makes a flip intrinsic rather than
# extrinsic.
from triwarp.kernels.grouping import hash_slot, pack_edge_key
from triwarp.kernels.remesh import _resolve_flip_quad_guarded


@wp.kernel
def face_edge_lengths(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lengths: wp.array2d[wp.float32]
) -> None:
    # Column ``e`` is the edge *opposite* corner ``e``, the igl intrinsic convention that
    # ``laplacian.cotmatrix_entries_intrinsic`` reads.
    f = int(wp.tid())
    v0 = vertices[faces[f * 3 + 0]]
    v1 = vertices[faces[f * 3 + 1]]
    v2 = vertices[faces[f * 3 + 2]]
    out_lengths[f, 0] = wp.length(v2 - v1)
    out_lengths[f, 1] = wp.length(v0 - v2)
    out_lengths[f, 2] = wp.length(v1 - v0)


@wp.kernel
def triangle_inequality_slack(
    edge_lengths: wp.array2d[wp.float32], epsilon: wp.float32, out_slack: wp.array[wp.float32]
) -> None:
    # How far this triangle is from satisfying the strict triangle inequality with margin
    # ``epsilon``, expressed as the constant that would have to be added to all three of its edges.
    # Adding a constant lengthens the two short sides by ``2 * delta`` against the long side's
    # ``delta``, so half the shortfall is enough.
    f = int(wp.tid())
    a = edge_lengths[f, 0]
    b = edge_lengths[f, 1]
    c = edge_lengths[f, 2]
    worst = wp.max(wp.max(epsilon - (a + b - c), epsilon - (b + c - a)), epsilon - (c + a - b))
    out_slack[f] = wp.max(worst, 0.0) * 0.5


@wp.func
def add_constant(length: wp.float32, delta: wp.float32) -> wp.float32:
    return length + delta


@wp.func
def local_corner(faces: wp.array[wp.int32], f: wp.int32, vertex: wp.int32) -> wp.int32:
    # Which corner of face ``f`` holds ``vertex``, or -1. Needed because an edge-length table is
    # indexed by *corner*, while the flip machinery speaks in vertex indices.
    for k in range(3):
        if faces[f * 3 + k] == vertex:
            return k
    return wp.int32(-1)


@wp.func
def law_of_cosines_angle(adjacent_a: wp.float32, adjacent_b: wp.float32, opposite: wp.float32):
    # Angle between the two adjacent sides of a triangle, from its three side lengths alone. Every
    # geometric quantity the intrinsic flip needs comes through here -- no vertex position does.
    denominator = 2.0 * adjacent_a * adjacent_b
    if denominator <= TOLERANCE_ZERO_CONSTANT:
        return wp.float32(0.0)
    cosine = (adjacent_a * adjacent_a + adjacent_b * adjacent_b - opposite * opposite) / denominator
    return wp.acos(wp.clamp(cosine, -1.0, 1.0))


@wp.func
def edge_lengths_at(
    edge_lengths: wp.array2d[wp.float32],
    faces: wp.array[wp.int32],
    f: wp.int32,
    opposite_vertex: wp.int32,
) -> wp.float32:
    # The length of the edge of ``f`` that faces ``opposite_vertex``.
    corner = local_corner(faces, f, opposite_vertex)
    if corner < 0:
        return wp.float32(0.0)
    return edge_lengths[f, corner]


@wp.kernel
def intrinsic_delaunay_candidates(
    faces: wp.array[wp.int32],
    edge_lengths: wp.array2d[wp.float32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
    out_new_length: wp.array[wp.float32],
) -> None:
    # Mark the interior edges that violate the local Delaunay condition, and measure what the
    # flipped edge would be -- both from edge lengths only, which is what makes the retriangulation
    # intrinsic: no vertex moves, so the *surface* is unchanged and only its triangulation improves.
    k = int(wp.tid())
    out_flip[k] = False
    out_new_length[k] = 0.0
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    quad = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if quad[0] < wp.int32(0):
        return
    first = quad[0]
    second = quad[2]
    apex0 = quad[3]

    corner_f0_first = local_corner(faces, f0, first)
    corner_f0_second = local_corner(faces, f0, second)
    corner_f0_apex = local_corner(faces, f0, apex0)
    corner_f1_first = local_corner(faces, f1, first)
    corner_f1_second = local_corner(faces, f1, second)
    if corner_f0_first < 0 or corner_f0_second < 0 or corner_f0_apex < 0:
        return
    if corner_f1_first < 0 or corner_f1_second < 0:
        return

    shared = edge_lengths[f0, corner_f0_apex]
    first_apex0 = edge_lengths[f0, corner_f0_second]
    second_apex0 = edge_lengths[f0, corner_f0_first]
    first_apex1 = edge_lengths[f1, corner_f1_second]
    second_apex1 = edge_lengths[f1, corner_f1_first]

    # The Delaunay test: the two angles facing the shared edge sum past a straight angle exactly
    # when the edge's cotangent weight would go negative.
    angle0 = law_of_cosines_angle(first_apex0, second_apex0, shared)
    angle1 = law_of_cosines_angle(first_apex1, second_apex1, shared)
    if angle0 + angle1 <= wp.PI:
        return

    # Unfold both triangles about the shared edge and measure the other diagonal. The wedge angles
    # at ``first`` add because the two triangles lie on opposite sides of the shared edge.
    wedge0 = law_of_cosines_angle(shared, first_apex0, second_apex0)
    wedge1 = law_of_cosines_angle(shared, first_apex1, second_apex1)
    total = wedge0 + wedge1
    flipped = (
        first_apex0 * first_apex0
        + first_apex1 * first_apex1
        - (2.0 * first_apex0 * first_apex1 * wp.cos(total))
    )
    if flipped <= TOLERANCE_ZERO_CONSTANT:
        return
    out_new_length[k] = wp.sqrt(flipped)
    out_flip[k] = True


@wp.kernel
def update_flipped_lengths(
    faces: wp.array[wp.int32],
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    new_length: wp.array[wp.float32],
    face_claim: wp.array[wp.int32],
    edge_claim: wp.array[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_edge_lengths: wp.array2d[wp.float32],
) -> None:
    # Rewrite the two faces' edge-length rows for the flips that won their claims, reading the old
    # rows first. This must run *before* the connectivity rewrite, which is what still knows which
    # corner holds which vertex; a committed flip owns both its faces exclusively, so reading and
    # writing the same rows here is race-free.
    k = int(wp.tid())
    if not flip[k]:
        return
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if face_claim[f0] != k or face_claim[f1] != k:
        return
    if edge_claim[hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)] != k:
        return

    first = quad[k, 0]
    second = quad[k, 2]
    diagonal = new_length[k]

    first_apex0 = edge_lengths_at(out_edge_lengths, faces, f0, second)
    second_apex0 = edge_lengths_at(out_edge_lengths, faces, f0, first)
    first_apex1 = edge_lengths_at(out_edge_lengths, faces, f1, second)
    second_apex1 = edge_lengths_at(out_edge_lengths, faces, f1, first)

    # ``commit_flips`` rewrites f0 as (first, apex1, apex0) and f1 as (second, apex0, apex1); each
    # column holds the edge opposite that corner.
    out_edge_lengths[f0, 0] = diagonal
    out_edge_lengths[f0, 1] = first_apex0
    out_edge_lengths[f0, 2] = first_apex1
    out_edge_lengths[f1, 0] = diagonal
    out_edge_lengths[f1, 1] = second_apex1
    out_edge_lengths[f1, 2] = second_apex0
