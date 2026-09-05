from typing import Any

import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.adjacency import edge_endpoints, edge_pair_topology, write_face_edge_keys
from triwarp.kernels.array import (
    LOOP_CONDITION,
    LOOP_ROUND,
    binary_search_sorted_contains,
    lowbias32,
    pack_edge_key,
    to_vec2d,
    to_vec3,
    to_vec3d,
)
from triwarp.kernels.grouping import hash_slot, sorted_run_start
from triwarp.kernels.predicates import (
    delone_metrics,
    dihedral_angle,
    is_unfold_quadrangle_convex,
    law_of_cosines_angle,
    mincircle_diameter_sq,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    triangle_normal,
    vector_angle,
)
from triwarp.kernels.scatter import add_corner_triple, lock_two_rings
from triwarp.kernels.triangles import (
    corner_triple,
    face_normal,
    face_normals_and_area,
    face_vertices_vec3d,
    triangle_quality,
)
from triwarp.kernels.voxels import squared_distance_to_own_cell_center

# The collapse round loop's third state slot, **appended** after ``array.LOOP_ROUND`` and
# ``LOOP_CONDITION`` so the shared two keep their numbers: the total commits as of the end of
# the previous round, which is how ``end_collapse_round`` decides whether a round progressed.
COLLAPSE_COMMITS = wp.constant(wp.int32(2))
COLLAPSE_STATE_SIZE = 3

# Delaunay / Delone edge-flip constants. The flip predicate runs in float64 deliberately:
# circumcircle diameters of near-degenerate triangles have too large a rounding error in float32,
# which sends the flip loop non-terminating.
DELONE_CRITICAL_DOT = wp.constant(wp.float64(-0.9))
DELONE_EPS = wp.constant(wp.float64(1e-7))
NO_ANGLE_CHANGE_LIMIT = wp.constant(wp.float64(6.283185307179586))  # 2*pi (NoAngleChangeLimit)
F32_LARGE = wp.constant(wp.float32(3.0e38))  # "disabled gate" sentinel (~FLT_MAX)

# Loop subdivision stencil weights. The even-vertex relaxation uses Warren's beta rather than Loop's
# original trigonometric weight, which is the choice ``igl::loop`` makes; see `loop_even_positions`.
LOOP_ODD_ENDPOINT = wp.constant(wp.float32(3.0 / 8.0))
LOOP_ODD_OPPOSITE = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BOUNDARY_SELF = wp.constant(wp.float32(3.0 / 4.0))
LOOP_BOUNDARY_NEIGHBOR = wp.constant(wp.float32(1.0 / 8.0))
LOOP_BETA_VALENCE_3 = wp.constant(wp.float32(3.0 / 16.0))
LOOP_BETA_NUMERATOR = wp.constant(wp.float32(3.0 / 8.0))

# Which of Loop's three even-vertex rules applies, as returned by ``loop_even_weights``. The mode is
# what a caller needs beyond the two weights, because the neighbour weight lands on a *different*
# neighbour set in each case: none, the boundary neighbours only, or the whole 1-ring.
LOOP_EVEN_KEEP = wp.constant(wp.int32(0))
LOOP_EVEN_BOUNDARY = wp.constant(wp.int32(1))
LOOP_EVEN_INTERIOR = wp.constant(wp.int32(2))


@wp.func
def edge_midpoint(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.vec3:
    v0 = vertices[unique_edges[e, 0]]
    v1 = vertices[unique_edges[e, 1]]
    return wp.lerp(v0, v1, wp.float32(0.5))


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = wp.int32(wp.tid())
    out_midpoints[k] = edge_midpoint(vertices, unique_edges, k)


@wp.func
def split_face_four(fv: wp.vec3i, mv: wp.vec3i) -> tuple[wp.vec3i, wp.vec3i, wp.vec3i, wp.vec3i]:
    # 1 -> 4 loop-subdivision template: three corner triangles, then the central triangle.
    t0 = wp.vec3i(fv[0], mv[0], mv[2])
    t1 = wp.vec3i(mv[0], fv[1], mv[1])
    t2 = wp.vec3i(mv[2], mv[1], fv[2])
    t3 = wp.vec3i(mv[0], mv[1], mv[2])
    return t0, t1, t2, t3


@wp.kernel
def subdivide_faces(
    faces: wp.array[wp.int32], mid_idx: wp.array2d[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = wp.int32(wp.tid())
    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(mid_idx[f, 0], mid_idx[f, 1], mid_idx[f, 2])
    t0, t1, t2, t3 = split_face_four(fv, mv)
    base = f * 12
    out_faces[base + 0] = t0[0]
    out_faces[base + 1] = t0[1]
    out_faces[base + 2] = t0[2]
    out_faces[base + 3] = t1[0]
    out_faces[base + 4] = t1[1]
    out_faces[base + 5] = t1[2]
    out_faces[base + 6] = t2[0]
    out_faces[base + 7] = t2[1]
    out_faces[base + 8] = t2[2]
    out_faces[base + 9] = t3[0]
    out_faces[base + 10] = t3[1]
    out_faces[base + 11] = t3[2]


@wp.kernel
def loop_edge_opposites(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    out_opposite_sum: wp.array[wp.vec3],
    out_face_count: wp.array[wp.int32],
) -> None:
    # Per unique edge: how many faces use it, and the sum of the vertices opposite it in each.
    # Corner ``j`` of face ``f`` spans ``(fv[j], fv[j + 1])`` and its opposite vertex is
    # ``fv[j + 2]``, so one pass over the faces gathers both halves of the Loop odd-vertex stencil.
    f = wp.int32(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        wp.atomic_add(out_opposite_sum, e, vertices[faces[f * 3 + (j + 2) % 3]])
        wp.atomic_add(out_face_count, e, 1)


@wp.kernel
def loop_vertex_rings(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    out_valence: wp.array[wp.int32],
    out_ring_sum: wp.array[wp.vec3],
    out_boundary_count: wp.array[wp.int32],
    out_boundary_sum: wp.array[wp.vec3],
) -> None:
    # Per vertex: its valence and 1-ring position sum, plus the same two restricted to boundary
    # edges. Driven by the *unique* edge list rather than by the faces, so the valence is the number
    # of distinct neighbours on any input -- the count a per-face pass would have to deduplicate
    # (each neighbour appears twice around an interior vertex but once at a boundary).
    e = wp.int32(wp.tid())
    v0 = unique_edges[e, 0]
    v1 = unique_edges[e, 1]
    p0 = vertices[v0]
    p1 = vertices[v1]
    wp.atomic_add(out_valence, v0, 1)
    wp.atomic_add(out_valence, v1, 1)
    wp.atomic_add(out_ring_sum, v0, p1)
    wp.atomic_add(out_ring_sum, v1, p0)
    if edge_face_count[e] == 1:
        wp.atomic_add(out_boundary_count, v0, 1)
        wp.atomic_add(out_boundary_count, v1, 1)
        wp.atomic_add(out_boundary_sum, v0, p1)
        wp.atomic_add(out_boundary_sum, v1, p0)


@wp.func
def loop_odd_weights(edge_face_count: wp.int32) -> tuple[wp.float32, wp.float32]:
    # Loop's odd (edge) stencil as weights: 3/8 on each endpoint and 1/8 on each opposite vertex for
    # an interior edge, the midpoint rule otherwise. A boundary edge (one face) or a non-manifold
    # one (three or more) has no well-defined pair of opposite vertices, so both fall back together.
    #
    # Shared by the position kernel and the interpolation-operator triplet kernels: the rule is one
    # decision and lives in one place, since a copy that drifted would put the operator and the
    # positions ``subdivide_loop`` returns onto different surfaces.
    if edge_face_count == 2:
        return LOOP_ODD_ENDPOINT, LOOP_ODD_OPPOSITE
    return wp.float32(0.5), wp.float32(0.0)


@wp.func
def loop_even_weights(
    valence: wp.int32, boundary_count: wp.int32
) -> tuple[wp.float32, wp.float32, wp.int32]:
    # Loop's even (original) stencil as (self weight, neighbour weight, mode); see the LOOP_EVEN_*
    # constants for the mode. Warren's beta -- 3/16 at valence 3 and 3/(8n) above it -- which is the
    # variant ``igl::loop`` uses, not Loop's original trigonometric weight. Shared for the same
    # reason as ``loop_odd_weights``.
    if boundary_count == 2:
        # Boundary vertex: 3/4 of itself, 1/8 of each neighbour *along the boundary*. Its interior
        # neighbours do not enter, which is what keeps a shared boundary curve identical on both
        # sides of a seam.
        return LOOP_BOUNDARY_SELF, LOOP_BOUNDARY_NEIGHBOR, LOOP_EVEN_BOUNDARY
    if boundary_count == 0 and valence > 0:
        beta = LOOP_BETA_VALENCE_3
        if valence != 3:
            beta = LOOP_BETA_NUMERATOR / wp.float32(valence)
        return wp.float32(1.0) - wp.float32(valence) * beta, beta, LOOP_EVEN_INTERIOR
    # Anything else keeps its position: an isolated vertex with no edges, or a non-manifold boundary
    # vertex where one or three-plus boundary edges meet and neither stencil is defined.
    return wp.float32(1.0), wp.float32(0.0), LOOP_EVEN_KEEP


@wp.kernel
def loop_odd_positions(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    edge_opposite_sum: wp.array[wp.vec3],
    edge_face_count: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Loop's odd (edge) vertices: 3/8 on each endpoint and 1/8 on each of the two opposite vertices.
    e = wp.int32(wp.tid())
    endpoints = vertices[unique_edges[e, 0]] + vertices[unique_edges[e, 1]]
    endpoint_weight, opposite_weight = loop_odd_weights(edge_face_count[e])
    out_positions[e] = endpoint_weight * endpoints + opposite_weight * edge_opposite_sum[e]


@wp.kernel
def loop_even_positions(
    vertices: wp.array[wp.vec3],
    valence: wp.array[wp.int32],
    ring_sum: wp.array[wp.vec3],
    boundary_count: wp.array[wp.int32],
    boundary_sum: wp.array[wp.vec3],
    out_positions: wp.array[wp.vec3],
) -> None:
    # Loop's even (original) vertices, relaxed towards their 1-ring; the rule is
    # ``loop_even_weights``, and the mode says which neighbour sum the weight multiplies.
    v = wp.int32(wp.tid())
    self_weight, neighbor_weight, mode = loop_even_weights(valence[v], boundary_count[v])
    neighbor = ring_sum[v]
    if mode == LOOP_EVEN_BOUNDARY:
        neighbor = boundary_sum[v]
    out_positions[v] = self_weight * vertices[v] + neighbor_weight * neighbor


# The interpolation operator ``subdivide_loop(return_operator=True)`` assembles, emitted as triplets
# from the same three grids the positions come from and through the same two weight functions. Row
# ``v`` is the relocated original vertex ``v``; row ``n_vertices + e`` is the odd vertex on unique
# edge ``e``. Every slot is written -- a zero weight where a rule does not apply -- because
# ``triplet_buffers`` hands back uninitialized memory (CLAUDE.md section 4).
@wp.kernel
def loop_even_self_triplets(
    valence: wp.array[wp.int32],
    boundary_count: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    v = wp.int32(wp.tid())
    self_weight, _neighbor_weight, _mode = loop_even_weights(valence[v], boundary_count[v])
    out_rows[v] = v
    out_cols[v] = v
    out_values[v] = self_weight


@wp.kernel
def loop_edge_triplets(
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    valence: wp.array[wp.int32],
    boundary_count: wp.array[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # Four triplets per unique edge: each endpoint's contribution to the odd row, and each
    # endpoint's contribution to the *other* endpoint's even row. Whether that second pair carries
    # any weight depends on the receiving vertex's mode -- the whole 1-ring for an interior vertex,
    # only the boundary neighbours for a boundary one -- which is the one place the operator has to
    # know what the accumulating ``loop_vertex_rings`` pass knows.
    e = wp.int32(wp.tid())
    v0 = unique_edges[e, 0]
    v1 = unique_edges[e, 1]
    is_boundary_edge = edge_face_count[e] == 1
    slot = base + 4 * e

    endpoint_weight, _opposite_weight = loop_odd_weights(edge_face_count[e])
    odd_row = n_vertices + e
    out_rows[slot] = odd_row
    out_cols[slot] = v0
    out_values[slot] = endpoint_weight
    out_rows[slot + 1] = odd_row
    out_cols[slot + 1] = v1
    out_values[slot + 1] = endpoint_weight

    for side in range(2):
        receiver = v0
        donor = v1
        if side == 1:
            receiver = v1
            donor = v0
        _self_weight, neighbor_weight, mode = loop_even_weights(
            valence[receiver], boundary_count[receiver]
        )
        weight = wp.float32(0.0)
        if mode == LOOP_EVEN_INTERIOR or (mode == LOOP_EVEN_BOUNDARY and is_boundary_edge):
            weight = neighbor_weight
        out_rows[slot + 2 + side] = receiver
        out_cols[slot + 2 + side] = donor
        out_values[slot + 2 + side] = weight


@wp.kernel
def loop_opposite_triplets(
    faces: wp.array[wp.int32],
    edge_of_corner: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    n_vertices: wp.int32,
    base: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # The 1/8 wings of the odd stencil, over the same (face, corner) grid ``loop_edge_opposites``
    # sums them on: corner ``j`` spans ``(fv[j], fv[j+1])`` and its opposite vertex is ``fv[j+2]``.
    f = wp.int32(wp.tid())
    for j in range(3):
        e = edge_of_corner[f * 3 + j]
        _endpoint_weight, opposite_weight = loop_odd_weights(edge_face_count[e])
        slot = base + 3 * f + j
        out_rows[slot] = n_vertices + e
        out_cols[slot] = faces[f * 3 + (j + 2) % 3]
        out_values[slot] = opposite_weight


@wp.kernel
def build_midpoint_index(
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_offset: wp.int32,
    out_midpoint_idx: wp.array[wp.int32],
) -> None:
    e = wp.int32(wp.tid())
    if long_mask[e]:
        out_midpoint_idx[e] = vertex_offset + offsets[e]
    else:
        out_midpoint_idx[e] = wp.int32(-1)


@wp.kernel
def fill_edge_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_mid: wp.array[wp.vec3],
) -> None:
    e = wp.int32(wp.tid())
    if long_mask[e]:
        out_mid[offsets[e]] = edge_midpoint(vertices, unique_edges, e)


@wp.kernel
def fill_edge_mean_sizing(
    sizing: wp.array[wp.float32],
    unique_edges: wp.array2d[wp.int32],
    split_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_sizing: wp.array[wp.float32],
) -> None:
    # ``fill_edge_midpoints`` for the sizing field rather than the position: the value carried to a
    # new midpoint is the endpoint mean ``mark_edges_over_sizing_field`` tested the edge with.
    e = wp.int32(wp.tid())
    if split_mask[e]:
        out_sizing[offsets[e]] = wp.float32(0.5) * (
            sizing[unique_edges[e, 0]] + sizing[unique_edges[e, 1]]
        )


@wp.kernel
def mark_edges_over_sizing_field(
    unique_edges: wp.array2d[wp.int32],
    lengths: wp.array[wp.float32],
    sizing: wp.array[wp.float32],
    out_long: wp.array[wp.bool],
) -> None:
    # The scalar ``length > max_edge`` test against a per-vertex sizing field. An edge's own target
    # is the mean of its endpoints', which is the standard reading of a vertex-sampled sizing
    # function and keeps the test symmetric in the edge's orientation.
    e = wp.int32(wp.tid())
    target = wp.float32(0.5) * (sizing[unique_edges[e, 0]] + sizing[unique_edges[e, 1]])
    out_long[e] = lengths[e] > target


@wp.func
def _write_tri(
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
    slot: wp.int32,
    tri: wp.vec3i,
    valid: wp.bool,
    src: wp.int32,
) -> None:
    out_faces[slot, 0] = tri[0]
    out_faces[slot, 1] = tri[1]
    out_faces[slot, 2] = tri[2]
    out_valid[slot] = valid
    out_slot_index[slot] = src


@wp.kernel
def emit_size_faces(
    faces: wp.array[wp.int32],
    face_mid: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    index_in: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    src = index_in[f]

    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(face_mid[f, 0], face_mid[f, 1], face_mid[f, 2])

    s0 = wp.where(mv[0] >= 0, wp.int32(1), wp.int32(0))
    s1 = wp.where(mv[1] >= 0, wp.int32(1), wp.int32(0))
    s2 = wp.where(mv[2] >= 0, wp.int32(1), wp.int32(0))
    count = s0 + s1 + s2

    # Four output triangle slots; unused slots are marked invalid.
    t0 = wp.vec3i(0, 0, 0)
    t1 = wp.vec3i(0, 0, 0)
    t2 = wp.vec3i(0, 0, 0)
    t3 = wp.vec3i(0, 0, 0)
    n0 = False
    n1 = False
    n2 = False
    n3 = False

    if count == 0:
        # No split edges: the face passes through unchanged.
        t0 = fv
        n0 = True
    elif count == 1:
        # Rotate so the split edge is (a, b); fan its midpoint p to the
        # opposite corner c as [a, p, c], [p, b, c].
        j = wp.int32(0)
        if s1 == 1:
            j = 1
        if s2 == 1:
            j = 2
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        t0 = wp.vec3i(a, p, c)
        n0 = True
        t1 = wp.vec3i(p, b, c)
        n1 = True
    elif count == 2:
        # Rotate so the unsplit edge is (c, a); emit corner triangle [p, b, q]
        # plus the quad (a, p, q, c) cut along its shorter diagonal.
        u = wp.int32(0)
        if s1 == 0:
            u = 1
        if s2 == 0:
            u = 2
        j = (u + 1) % 3
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        q = mv[(j + 1) % 3]
        t0 = wp.vec3i(p, b, q)
        n0 = True
        d_aq = wp.length_sq(vertices[a] - vertices[q])
        d_pc = wp.length_sq(vertices[p] - vertices[c])
        if d_aq <= d_pc:
            t1 = wp.vec3i(a, p, q)
            t2 = wp.vec3i(a, q, c)
        else:
            t1 = wp.vec3i(a, p, c)
            t2 = wp.vec3i(p, q, c)
        n1 = True
        n2 = True
    else:
        # Three split edges: the regular 1 -> 4 split (matches subdivide).
        t0, t1, t2, t3 = split_face_four(fv, mv)
        n0 = True
        n1 = True
        n2 = True
        n3 = True

    base = f * 4
    _write_tri(out_faces, out_valid, out_slot_index, base + 0, t0, n0, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 1, t1, n1, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 2, t2, n2, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 3, t3, n3, src)


# ---------------------------------------------------------------------------
# Region-restricted subdivision helpers
# ---------------------------------------------------------------------------


@wp.kernel
def mark_region_edges(
    region_flags: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_edge_in_region: wp.array[wp.bool],
) -> None:
    # A unique edge is in/on the region boundary if at least one of its incident faces is in
    # the region -- region-border edges included. Benign write race: every
    # thread writing the same slot writes True.
    i = wp.int32(wp.tid())
    if region_flags[i // 3] != 0:
        out_edge_in_region[inverse[i]] = wp.bool(True)


@wp.func
def long_region_edge(length: wp.float32, max_edge: wp.float32, in_region: wp.bool) -> wp.bool:
    return in_region and length > max_edge


# ---------------------------------------------------------------------------
# Region-restricted density refinement (Liepa 2003, section 3)
# ---------------------------------------------------------------------------


@wp.func
def density_split_wanted(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    scale_a: wp.float32,
    scale_b: wp.float32,
    scale_c: wp.float32,
    alpha: wp.float32,
) -> wp.bool:
    # Liepa's density criterion for splitting a patch triangle at its centroid. Each vertex carries
    # a *scale attribute* -- the average length of the edges incident to it in the surrounding mesh
    # -- and the centroid inherits the mean of its three. The triangle is split when, for **every**
    # corner ``m``, the centroid is far from ``m`` relative to the centroid's own scale *and* the
    # centroid's scale is coarse relative to ``m``'s:
    #
    #     alpha * |centroid - v_m| > scale(centroid)   and   alpha * scale(centroid) > scale(v_m)
    #
    # The first clause is what refines; the second is what stops the recursion at the surrounding
    # sampling instead of running to the tolerance. Both must hold at all three corners, so a
    # triangle already matching its neighbourhood's density is left alone and the pass converges.
    #
    # ``alpha`` is the paper's ``sqrt(2)``, exposed because it is the one real knob: raising it
    # refines further, lowering it stops sooner.
    centroid = (a + b + c) / wp.float32(3.0)
    scale = (scale_a + scale_b + scale_c) / wp.float32(3.0)
    if alpha * scale <= wp.max(scale_a, wp.max(scale_b, scale_c)):
        return False
    return (
        alpha * wp.length(centroid - a) > scale
        and alpha * wp.length(centroid - b) > scale
        and alpha * wp.length(centroid - c) > scale
    )


@wp.kernel
def mark_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    alpha: wp.float32,
    out_split: wp.array[wp.int32],
) -> None:
    # Which region faces want a centroid split this pass, as 0/1 so the result scans directly into
    # the two offset tables ``emit_density_splits`` needs. A face outside the region never splits,
    # which is what keeps the refinement inside the patch.
    f = wp.int32(wp.tid())
    if not region[f]:
        out_split[f] = wp.int32(0)
        return
    i, j, k = corner_triple(faces, f)
    wanted = density_split_wanted(
        vertices[i], vertices[j], vertices[k], scale[i], scale[j], scale[k], alpha
    )
    out_split[f] = wp.where(wanted, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_density_splits(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    split: wp.array[wp.int32],
    split_offsets: wp.array[wp.int32],
    face_offsets: wp.array[wp.int32],
    n_vertices: wp.int32,
    out_positions: wp.array[wp.vec3],
    out_scale: wp.array[wp.float32],
    out_faces: wp.array[wp.int32],
    out_region: wp.array[wp.bool],
) -> None:
    # One pass of the 1 -> 3 centroid split, faces and new vertices in the same launch.
    #
    # A centroid split is **per-triangle independent** -- the new vertex is interior to the triangle
    # and no edge is divided -- so unlike edge bisection it needs none of ``subdivide_to_size``'s
    # crack-free 1/2/3 templates and no agreement with the neighbours. That is the whole reason this
    # criterion suits a parallel refinement: one scan for the new-vertex slots, one for the face
    # slots, and one kernel.
    #
    # Child faces inherit their parent's region membership, matching
    # ``subdivide_region_to_size``, so a caller's patch mask survives the pass.
    f = wp.int32(wp.tid())
    i, j, k = corner_triple(faces, f)
    base = face_offsets[f] * 3
    if split[f] == wp.int32(0):
        out_faces[base + 0] = i
        out_faces[base + 1] = j
        out_faces[base + 2] = k
        out_region[face_offsets[f]] = region[f]
        return

    slot = split_offsets[f]
    center = n_vertices + slot
    out_positions[slot] = (vertices[i] + vertices[j] + vertices[k]) / wp.float32(3.0)
    out_scale[slot] = (scale[i] + scale[j] + scale[k]) / wp.float32(3.0)
    out_faces[base + 0] = i
    out_faces[base + 1] = j
    out_faces[base + 2] = center
    out_faces[base + 3] = j
    out_faces[base + 4] = k
    out_faces[base + 5] = center
    out_faces[base + 6] = k
    out_faces[base + 7] = i
    out_faces[base + 8] = center
    for child in range(3):
        out_region[face_offsets[f] + child] = region[f]


@wp.func
def face_split_count(split: wp.int32) -> wp.int32:
    """How many faces this one becomes: three when split, one when not."""
    # The counts whose exclusive scan gives ``emit_density_splits`` its output face slots.
    return wp.int32(1) + wp.int32(2) * split


# ---------------------------------------------------------------------------
# Float64 Delone edge-flip predicate. Computed in double precision for the reason the constants
# above give.
# ---------------------------------------------------------------------------


@wp.func
def _segments_dist_sq_d(p1: wp.vec3d, q1: wp.vec3d, p2: wp.vec3d, q2: wp.vec3d) -> wp.float64:
    # Squared distance between segments [p1,q1] and [p2,q2] (Ericson, clamped closest points).
    eps = wp.float64(1e-30)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    aa = wp.length_sq(d1)
    ee = wp.length_sq(d2)
    f = wp.dot(d2, r)
    s = wp.float64(0.0)
    t = wp.float64(0.0)
    if aa <= eps and ee <= eps:
        return wp.length_sq(p1 - p2)
    if aa <= eps:
        t = wp.clamp(f / ee, wp.float64(0.0), wp.float64(1.0))
    else:
        cc = wp.dot(d1, r)
        if ee <= eps:
            s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
        else:
            bb = wp.dot(d1, d2)
            denom = aa * ee - bb * bb
            if denom != wp.float64(0.0):
                s = wp.clamp((bb * f - cc * ee) / denom, wp.float64(0.0), wp.float64(1.0))
            t = (bb * s + f) / ee
            if t < wp.float64(0.0):
                t = wp.float64(0.0)
                s = wp.clamp(-cc / aa, wp.float64(0.0), wp.float64(1.0))
            elif t > wp.float64(1.0):
                t = wp.float64(1.0)
                s = wp.clamp((bb - cc) / aa, wp.float64(0.0), wp.float64(1.0))
    cp1 = p1 + s * d1
    cp2 = p2 + t * d2
    return wp.length_sq(cp1 - cp2)


@wp.func
def _check_delone_quadrangle_d(
    a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, max_angle_change: wp.float64
) -> wp.bool:
    # Returns True to KEEP the current diagonal (a-c), False to flip to (b-d). Exact port of
    # checkDeloneQuadrangle(Vector3d, ...).
    n_abc = triangle_normal(a, b, c)
    n_acd = triangle_normal(a, c, d)
    old_pocket = wp.dot(n_abc, n_acd) < DELONE_CRITICAL_DOT

    n_abd = triangle_normal(a, b, d)
    n_dbc = triangle_normal(d, b, c)
    new_pocket = wp.dot(n_abd, n_dbc) < DELONE_CRITICAL_DOT

    if old_pocket != new_pocket:
        return new_pocket

    if old_pocket:
        metric_ac = wp.max(mincircle_diameter_sq(a, c, d), mincircle_diameter_sq(c, a, b))
        metric_bd = wp.max(mincircle_diameter_sq(b, d, a), mincircle_diameter_sq(d, b, c))
        return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)

    if max_angle_change < NO_ANGLE_CHANGE_LIMIT:
        old_angle = dihedral_angle(n_abd, n_dbc, d - b)
        new_angle = dihedral_angle(n_abc, n_acd, a - c)
        if wp.abs(old_angle - new_angle) > max_angle_change:
            return True

    metric_ac, metric_bd = delone_metrics(a, b, c, d)

    if wp.isinf(metric_ac):
        if wp.isinf(metric_bd):
            return wp.length_sq(a - c) <= wp.length_sq(b - d)
        return False
    return metric_ac <= metric_bd + DELONE_EPS * (metric_ac + metric_bd)


# ---------------------------------------------------------------------------
# 2D orientation / incircle predicate (for delaunay_triangulation)
# ---------------------------------------------------------------------------


@wp.func
def _incircle_d(a: wp.vec2d, b: wp.vec2d, c: wp.vec2d, d: wp.vec2d) -> wp.float64:
    # Positive iff d is inside the circumcircle of CCW triangle (a, b, c). The 3x3 determinant is
    # kept expanded in components (as in ``predicates.orient2d``) because the term order is what
    # makes the sign reliable near cocircularity; only the squared radii go through ``length_sq``.
    ad = a - d
    bd = b - d
    cd = c - d
    a2 = wp.length_sq(ad)
    b2 = wp.length_sq(bd)
    c2 = wp.length_sq(cd)
    return (
        ad[0] * (bd[1] * c2 - b2 * cd[1])
        - ad[1] * (bd[0] * c2 - b2 * cd[0])
        + a2 * (bd[0] * cd[1] - bd[1] * cd[0])
    )


# ---------------------------------------------------------------------------
# Shared parallel edge-flip core
# ---------------------------------------------------------------------------


@wp.kernel
def mark_edge_pair_starts(
    sorted_keys: wp.array[wp.uint64], n: wp.int32, out_starts: wp.array[wp.int32]
) -> None:
    # ``grouping.mark_group_starts`` specialized to ``length=2`` and emitting ``int32`` rather than
    # ``wp.bool``: the flip loop feeds this straight to ``warp.utils.array_scan``, which has no
    # bool overload, so a bool flag would only buy an ``array_cast``. Flags the position that
    # starts a run of *exactly* two equal keys, i.e. an edge shared by exactly two face corners.
    #
    # Differs from ``mark_unique_edge_starts`` below only in requiring the run to be exactly two:
    # that one takes every run whatever its length, because the decimation pass wants all unique
    # edges where a flip pass wants only the manifold-interior ones.
    i = wp.int32(wp.tid())
    start = wp.int32(0)
    if i + 2 <= n and sorted_run_start(sorted_keys, i):
        if sorted_keys[i] == sorted_keys[i + 1]:
            start = wp.int32(1)
            if i + 2 < n:
                if sorted_keys[i] == sorted_keys[i + 2]:
                    start = wp.int32(0)  # run longer than two
    out_starts[i] = start


@wp.kernel
def emit_flip_topology(
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    # The whole face-adjacency table a flip pass needs, in one launch over the ``3 * n_faces``
    # sorted edge slots: ``adjacency.face_adjacency``'s pair table, its shared-edge endpoints and
    # ``face_adjacency_unshared``'s opposite apexes. Everything comes out of the two grouped *edge*
    # indices, so no edge table is materialized and nothing is gathered through one.
    #
    # ``ranks`` is the inclusive scan of ``starts``, so ``ranks[i] - 1`` is the row a start writes
    # -- the same ascending-key row order the ``flatnonzero`` compaction inside
    # ``grouping.group`` produces, which is what keeps this byte-identical to the composed path.
    i = wp.int32(wp.tid())
    if starts[i] == 0:
        return
    slot = ranks[i] - 1
    shared_a, shared_b, face_0, face_1, unshared_0, unshared_1 = edge_pair_topology(
        faces, order[i], order[i + 1]
    )
    out_adjacency_edges[slot, 0] = shared_a
    out_adjacency_edges[slot, 1] = shared_b
    if face_0 <= face_1:
        out_adjacency[slot, 0] = face_0
        out_adjacency[slot, 1] = face_1
        out_unshared[slot, 0] = unshared_0
        out_unshared[slot, 1] = unshared_1
    else:
        out_adjacency[slot, 0] = face_1
        out_adjacency[slot, 1] = face_0
        out_unshared[slot, 0] = unshared_1
        out_unshared[slot, 1] = unshared_0


@wp.func
def _resolve_flip_quad(
    faces: wp.array[wp.int32], f0: wp.int32, u: wp.int32, v: wp.int32, d0: wp.int32, d1: wp.int32
) -> wp.vec4i:
    # Orient the flip quad so f0 traverses a->c (its apex d0 is the left apex "d"); the other
    # face's apex d1 is the right apex "b". Returns (a, b, c, d); a<0 marks inconsistent winding.
    a = wp.int32(-1)
    c = wp.int32(-1)
    for k in range(3):
        va = faces[f0 * 3 + k]
        vb = faces[f0 * 3 + (k + 1) % 3]
        if va == u and vb == v:
            a = u
            c = v
        if va == v and vb == u:
            a = v
            c = u
    return wp.vec4i(a, d1, c, d0)


@wp.func
def _resolve_flip_quad_guarded(
    faces: wp.array[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    k: wp.int32,
    f0: wp.int32,
    out_quad: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # Shared flip-candidate preamble: reject missing apexes, inconsistent winding, b == d, and
    # flips that would duplicate an existing edge. Returns the unpacked ``(a, b, c, d)`` with
    # a < 0 when not flippable, so every candidate kernel opens with the same two lines; out_quad[k]
    # is written only for valid quads (claim/commit read quad[k] only when the caller has set
    # out_flip[k], which stays False for rejected/non-flipped edges).
    a = wp.int32(-1)
    b = wp.int32(-1)
    c = wp.int32(-1)
    d = wp.int32(-1)
    d0 = unshared[k, 0]
    d1 = unshared[k, 1]
    if d0 >= 0 and d1 >= 0:
        quad = _resolve_flip_quad(faces, f0, adjacency_edges[k, 0], adjacency_edges[k, 1], d0, d1)
        if (
            quad[0] >= 0
            and quad[1] != quad[3]
            and not binary_search_sorted_contains(
                sorted_edge_keys, pack_edge_key(quad[1], quad[3], key_base)
            )
        ):
            a = quad[0]
            b = quad[1]
            c = quad[2]
            d = quad[3]
            out_quad[k, 0] = a
            out_quad[k, 1] = b
            out_quad[k, 2] = c
            out_quad[k, 3] = d
    return a, b, c, d


@wp.func
def _resolve_flip_quad_in_region(
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    k: wp.int32,
    out_quad: wp.array2d[wp.int32],
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32]:
    # ``_resolve_flip_quad_guarded`` plus the region test, folded into its ``a < 0`` contract: an
    # edge with either incident face outside the region is not flippable, for the same reason a
    # missing apex is not.
    #
    # Region-restricted rather than universal, because two of the four candidate kernels genuinely
    # have no region to restrict to. ``valence_flip_candidates`` is whole-mesh *by construction* --
    # its only caller, ``remesh._valence_flip_pass``, is reached from ``isotropic_remesh`` and takes
    # no ``region`` parameter, where ``delone`` and ``objective`` both go through ``_flip_setup`` --
    # and ``incircle_flip_candidates`` triangulates a planar point set. That asymmetry reads as an
    # oversight until someone opens the wrapper, which is why it is written down here.
    f0 = adjacency[k, 0]
    if region_flags[f0] == 0 or region_flags[adjacency[k, 1]] == 0:
        return wp.int32(-1), wp.int32(-1), wp.int32(-1), wp.int32(-1)
    return _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )


@wp.func
def flip_quad_positions_d(
    vertices: wp.array[Any], a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32
) -> tuple[wp.vec3d, wp.vec3d, wp.vec3d, wp.vec3d]:
    # The four corners of a flip quad, promoted to ``float64``. Every flip predicate in this module
    # -- convexity, the Delone empty-circumcircle test, the segment distance -- runs in float64 on a
    # float32 vertex buffer, because they are *branches*: a lost digit changes a flip decision
    # rather than a printed number.
    return (
        to_vec3d(vertices[a]),
        to_vec3d(vertices[b]),
        to_vec3d(vertices[c]),
        to_vec3d(vertices[d]),
    )


@wp.kernel
def delone_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    max_angle_change: wp.float32,
    max_deviation_sq: wp.float32,
    critical_aspect: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    a, b, c, d = _resolve_flip_quad_in_region(
        faces,
        adjacency,
        adjacency_edges,
        unshared,
        region_flags,
        sorted_edge_keys,
        key_base,
        k,
        out_quad,
    )
    if a < 0:
        return
    ap, bp, cp, dp = flip_quad_positions_d(vertices, a, b, c, d)
    if max_deviation_sq < F32_LARGE:
        if _segments_dist_sq_d(ap, cp, bp, dp) > wp.float64(max_deviation_sq):
            return
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    angle = wp.float64(max_angle_change)
    if critical_aspect < F32_LARGE and angle < NO_ANGLE_CHANGE_LIMIT:
        max_aspect = wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp))
        if max_aspect > wp.float64(critical_aspect):
            angle = NO_ANGLE_CHANGE_LIMIT
    out_flip[k] = not _check_delone_quadrangle_d(ap, bp, cp, dp, angle)


@wp.kernel
def incircle_flip_candidates(
    points: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if a < 0:
        return
    ap = to_vec2d(points[a])
    bp = to_vec2d(points[b])
    cp = to_vec2d(points[c])
    dp = to_vec2d(points[d])
    # Post-flip triangles (a, b, d) and (d, b, c) must both be positively oriented (convex quad).
    if orient2d(ap, bp, dp) <= wp.float64(0.0) or orient2d(dp, bp, cp) <= wp.float64(0.0):
        return
    # f0 = (a, c, d) is CCW; flip iff the opposite apex b lies inside its circumcircle.
    out_flip[k] = _incircle_d(ap, cp, dp, bp) > wp.float64(0.0)


@wp.func
def flip_claim_won(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    face_claim: wp.array[wp.int32],
    edge_claim: wp.array[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    k: wp.int32,
) -> tuple[wp.int32, wp.int32, wp.bool]:
    # Did candidate ``k`` win every claim ``claim_flips`` above wrote -- both its faces and the
    # hashed slot of the new diagonal it would create? Returns the two face indices alongside the
    # verdict because every caller needs them straight afterwards, and both are already loaded here.
    #
    # Shared by ``commit_flips`` and ``update_flipped_lengths``, which must agree *exactly* on which
    # candidates commit: the second rewrites the edge-length rows of the faces the first rewrites
    # the connectivity of, so a divergence between two copies of this guard would leave the two
    # tables describing different meshes. That is why it is one function rather than two identical
    # seven-statement runs.
    if not flip[k]:
        return wp.int32(-1), wp.int32(-1), False
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    if face_claim[f0] != k or face_claim[f1] != k:
        return f0, f1, False
    slot = hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
    if edge_claim[slot] != k:
        return f0, f1, False
    return f0, f1, True


@wp.kernel
def claim_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_face_claim: wp.array[wp.int32],
    out_edge_claim: wp.array[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    if not flip[k]:
        return
    wp.atomic_min(out_face_claim, adjacency[k, 0], k)
    wp.atomic_min(out_face_claim, adjacency[k, 1], k)
    slot = hash_slot(pack_edge_key(quad[k, 1], quad[k, 3], key_base), edge_claim_mask)
    wp.atomic_min(out_edge_claim, slot, k)


@wp.kernel
def commit_flips(
    flip: wp.array[wp.bool],
    quad: wp.array2d[wp.int32],
    adjacency: wp.array2d[wp.int32],
    face_claim: wp.array[wp.int32],
    edge_claim: wp.array[wp.int32],
    edge_claim_mask: wp.int32,
    key_base: wp.uint64,
    out_faces: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    f0, f1, won = flip_claim_won(
        flip, quad, adjacency, face_claim, edge_claim, edge_claim_mask, key_base, k
    )
    if not won:
        return
    a = quad[k, 0]
    b = quad[k, 1]
    c = quad[k, 2]
    d = quad[k, 3]
    # New diagonal b-d: faces become (a, b, d) and (c, d, b), preserving winding.
    out_faces[f0 * 3 + 0] = a
    out_faces[f0 * 3 + 1] = b
    out_faces[f0 * 3 + 2] = d
    out_faces[f1 * 3 + 0] = c
    out_faces[f1 * 3 + 1] = d
    out_faces[f1 * 3 + 2] = b
    wp.atomic_add(out_count, 0, 1)


# ===========================================================================
# Isotropic explicit remeshing (Botsch-Kobbelt split/collapse/flip/smooth/reproject)
#
# Split reuses subdivide_to_size. The kernels below add: feature/boundary classification
# (per-vertex FREE/CREASE/CORNER codes), a parallel edge-collapse primitive with full 1-ring
# locking + link-condition guard, valence-driven edge flips, tangential Laplacian smoothing, and
# reprojection of free vertices onto the original surface.
# ===========================================================================
FREE_VERTEX = wp.constant(wp.int32(0))
CREASE_VERTEX = wp.constant(wp.int32(1))
CORNER_VERTEX = wp.constant(wp.int32(2))

# What ``collapse_survivor`` decided about where the merged vertex may go.
COLLAPSE_REJECTED = wp.constant(wp.int32(0))  # the edge must not collapse at all
COLLAPSE_PINNED = wp.constant(wp.int32(1))  # the survivor keeps its own position
COLLAPSE_FREE = wp.constant(wp.int32(2))  # the caller places it -- midpoint, or a quadric optimum


@wp.func
def collapse_survivor(
    codes: wp.array[wp.int32], u: wp.int32, v: wp.int32, is_boundary: wp.bool
) -> tuple[wp.int32, wp.int32, wp.int32]:
    # Which endpoint of edge ``(u, v)`` survives the collapse, which one is removed, and whether
    # the survivor's position is pinned or free -- the feature rule alone, with no geometry in it.
    #
    # One rule, two decimators. ``collapse_candidates`` and ``quadric_collapse_candidates`` were
    # each carrying their own copy, expressed through a ``reject`` flag in one and early returns in
    # the other, and the second's comment claimed it "mirrors ``collapse_candidates``" -- a claim
    # only a shared function can keep true. They were in fact equivalent; nothing but this stopped
    # the next edit to either from silently diverging.
    #
    # The genuine difference between the two is what they do with ``COLLAPSE_FREE``: one takes the
    # midpoint, the other minimizes the summed quadric. That is the only thing their comments
    # should now claim to share.
    cu = codes[u]
    cv = codes[v]
    if cu == CORNER_VERTEX and cv == CORNER_VERTEX:
        # Two corners: the edge between two fixed points cannot shorten.
        return u, v, COLLAPSE_REJECTED
    if cu >= CREASE_VERTEX and cv >= CREASE_VERTEX:
        # Two feature vertices: collapse only along a boundary edge, and only when both are plain
        # creases. When that holds neither endpoint is preferred, so the placement stays free.
        if not (is_boundary and cu == CREASE_VERTEX and cv == CREASE_VERTEX):
            return u, v, COLLAPSE_REJECTED
        return u, v, COLLAPSE_FREE
    if cu >= CREASE_VERTEX:
        return u, v, COLLAPSE_PINNED
    if cv >= CREASE_VERTEX:
        return v, u, COLLAPSE_PINNED
    return u, v, COLLAPSE_FREE


@wp.kernel
def scatter_feature_edge_counts(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_face_count: wp.array[wp.int32],
    edge_faces: wp.array2d[wp.int32],
    feature_angle: wp.float32,
    out_feature_count: wp.array[wp.int32],
    out_boundary_vertex: wp.array[wp.bool],
) -> None:
    # Add 1 to both endpoints of every feature edge: a boundary edge, or an interior edge whose two
    # faces meet at more than ``feature_angle``. One launch over the unique edges answers both
    # questions from the incidence table, where ``_classify`` used to re-group the same 3 * n_faces
    # rows twice (once as boundary edges, once as face adjacency) to ask them separately.
    e = wp.int32(wp.tid())
    count = edge_face_count[e]
    boundary = count == 1
    feature = boundary
    if count == 2:
        f0 = edge_faces[e, 0]
        f1 = edge_faces[e, 1]
        normal_a, _area_a = face_normals_and_area(vertices, faces, f0)
        normal_b, _area_b = face_normals_and_area(vertices, faces, f1)
        feature = vector_angle(normal_a, normal_b) > feature_angle
    if feature:
        v0 = unique_edges[e, 0]
        v1 = unique_edges[e, 1]
        wp.atomic_add(out_feature_count, v0, 1)
        wp.atomic_add(out_feature_count, v1, 1)
        if boundary:
            # Every writer stores the same value, so the mask needs no atomic.
            out_boundary_vertex[v0] = True
            out_boundary_vertex[v1] = True


@wp.func
def finalize_vertex_codes(feature_count: wp.int32) -> wp.int32:
    # 0 feature edges -> FREE; exactly 2 -> CREASE (on a smooth feature/boundary line);
    # anything else (1 = feature endpoint, >=3 = junction) -> CORNER (frozen).
    code = CORNER_VERTEX
    if feature_count == 0:
        code = FREE_VERTEX
    elif feature_count == 2:
        code = CREASE_VERTEX
    return code


@wp.func
def csr_common_neighbor_count(
    offsets: wp.array[wp.int32], columns: wp.array[wp.int32], a: wp.int32, b: wp.int32
) -> wp.int32:
    # Number of vertices adjacent to both a and b (two nested scans; degrees are tiny).
    count = wp.int32(0)
    for i in range(offsets[a], offsets[a + 1]):
        w = columns[i]
        for j in range(offsets[b], offsets[b + 1]):
            if columns[j] == w:
                count += 1
    return count


@wp.func
def satisfies_link_condition(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    u: wp.int32,
    v: wp.int32,
    is_boundary: wp.bool,
) -> wp.bool:
    # May edge (u, v) collapse without changing the surface's topology? Exactly two vertices
    # adjacent to both endpoints for an interior edge -- the two apexes of its own faces -- and one
    # for a boundary edge. A third shared neighbour means the edge closes a tetrahedral loop the
    # collapse would pinch shut.
    #
    # Both collapse-candidate kernels test this, and it is a *decision rule* rather than an
    # arithmetic run: two copies can diverge into accepting an edge in one decimator and rejecting
    # it in the other, which is a correctness hazard the duplicate scans do not rank as one.
    required = 2
    if is_boundary:
        required = 1
    return csr_common_neighbor_count(offsets, columns, u, v) == required


@wp.func
def collapse_flips_normal(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    moved: wp.int32,
    partner: wp.int32,
    target: wp.vec3,
) -> wp.bool:
    # Would moving ``moved`` to ``target`` (and welding it onto ``partner``) invert any face it
    # still belongs to? The two faces containing *both* endpoints vanish in the collapse and are
    # skipped; every other incident face keeps its other two corners and must keep its orientation.
    #
    # This is the guard that separates a usable decimator from one that produces self-intersecting
    # geometry, and it is why the vertex-face CSR is built at all.
    for slot in range(vertex_face_offsets[moved], vertex_face_offsets[moved + 1]):
        f = vertex_faces[slot]
        i0, i1, i2 = corner_triple(faces, f)
        if i0 == partner or i1 == partner or i2 == partner:
            continue
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        before = wp.cross(p1 - p0, p2 - p0)
        if i0 == moved:
            p0 = target
        elif i1 == moved:
            p1 = target
        else:
            p2 = target
        after = wp.cross(p1 - p0, p2 - p0)
        before_length = wp.length(before)
        after_length = wp.length(after)
        if before_length <= 0.0:
            continue  # already degenerate: nothing to invert
        if after_length <= 0.0:
            return True  # the collapse would flatten it outright
        if wp.dot(before / before_length, after / after_length) < COLLAPSE_MIN_NORMAL_DOT:
            return True
    return False


# ---------------------------------------------------------------------------
# Readback-free decimation pass (see ``remesh._DecimationBuffers``)
#
# Every kernel below works on **fixed-capacity** buffers whose live prefix length lives in a device
# array, so a whole pass can be issued once and replayed as a CUDA graph. Two conventions carry the
# padding, and between them almost every kernel the pass reuses needs no guard of its own:
#
# - a **dummy vertex** at index ``n_vertices_capacity``, which every padded face corner and padded
#   edge endpoint points at. It is referenced by no real edge, so per-vertex kernels may run over it
#   freely; ``quadric_collapse_candidates`` is stopped on padded edges by freezing its code to
#   ``CORNER_VERTEX``, which is that kernel's first rejection test.
# - a **dummy edge slot** at index ``n_edges_capacity``, which every padded face corner's entry in
#   ``inverse`` points at, so ``scatter_edge_incidence`` can run over the whole corner buffer.
# ---------------------------------------------------------------------------

DECIMATION_FACES = wp.constant(wp.int32(0))
DECIMATION_VERTICES = wp.constant(wp.int32(1))
DECIMATION_EDGES = wp.constant(wp.int32(2))

EDGE_KEY_PAD = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))


@wp.kernel(enable_backward=False)
def collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    lengths: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    codes: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
) -> None:
    # ``low`` and ``high`` are per *vertex* rather than scalars so that one code path serves both
    # the uniform target and an adaptive sizing field; the uniform case fills them with a constant.
    # An edge's own band is the mean of its endpoints', matching ``mark_edges_over_sizing_field``.
    k = wp.int32(wp.tid())
    out_survivor[k] = -1
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    if lengths[k] >= wp.float32(0.5) * (low[u] + low[v]):
        return
    is_boundary = edge_face_count[k] == 1

    # Choose the surviving vertex and its target position; this decimator places a free collapse at
    # the edge midpoint, where ``quadric_collapse_candidates`` minimizes the summed quadric.
    s, r, placement = collapse_survivor(codes, u, v, is_boundary)
    if placement == COLLAPSE_REJECTED:
        return
    p = wp.lerp(vertices[u], vertices[v], 0.5)
    if placement == COLLAPSE_PINNED:
        p = vertices[s]

    if not satisfies_link_condition(offsets, columns, u, v, is_boundary):
        return

    # Anti-oscillation: reject if the collapse would create an edge longer than the high band. The
    # band is read at the far endpoint ``w``, so a collapse reaching into a finely-sized region is
    # judged by that region's target rather than by the survivor's.
    #
    # **Both** rings under a free placement, not just the removed vertex's. The reattached edges
    # from ``r``'s neighbours are the obvious new ones, but a ``COLLAPSE_FREE`` placement moves the
    # *survivor* to the midpoint as well, so every edge from ``s``'s own neighbours to ``p`` is
    # equally new and equally able to overshoot the band. Under ``COLLAPSE_PINNED`` the survivor
    # keeps its position and its ring is unchanged, which is the case one walk covers.
    for i in range(offsets[r], offsets[r + 1]):
        w = columns[i]
        if w != s and wp.length(p - vertices[w]) > high[w]:
            return
    if placement == COLLAPSE_FREE:
        for i in range(offsets[s], offsets[s + 1]):
            w = columns[i]
            if w != r and wp.length(p - vertices[w]) > high[w]:
                return

    # The fold veto, last because every test above rejects more cheaply. Spelled exactly as
    # ``quadric_collapse_candidates`` spells it -- both directions, unconditionally -- because this
    # is a *decision rule* shared by two decimators, and two spellings of one rule is the hazard
    # section 2.4 names rather than the duplicated arithmetic.
    #
    # It was absent here for a long time while the quadric decimator had it, which was a gap and
    # not a variant: the link condition is topological and the band walks above bound *lengths*, so
    # nothing else here notices a collapse that inverts an incident face. Measured against a
    # baseline worktree on a 133x133 graded saddle patch, three reps per arm, both arms
    # deterministic on this fixture:
    #
    #   ``_collapse_pass`` (5 passes): zero-area faces **1 -> 0**, minimum face area
    #   **0.0 -> 3.9e-12**, aspect p99 6260.79 -> 5882.00, time 11.09-11.58 ms -> 9.77-12.34 ms.
    #   ``isotropic_remesh(iterations=3)``: aspect p99 **3100-3116 -> 2007.16**, a 1.55x
    #   improvement against a baseline that itself drifts only ~0.5% run to run.
    #
    # So the guard is free: the two timing ranges overlap, and rejecting a collapse early removes
    # work downstream. The vertex-face CSR it needs costs 0.124 ms a pass against an 11 ms stage.
    # ``isotropic_remesh`` still leaves one degenerate face on that input, so this closes part of
    # what section 16.4 attributed to the smooth pass, not all of it.
    if collapse_flips_normal(
        vertices, faces, vertex_face_offsets, vertex_faces, r, s, p
    ) or collapse_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, s, r, p):
        return

    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = p


@wp.func
def wins_key_everywhere(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locks: wp.array[wp.int64],
    s: wp.int32,
    r: wp.int32,
    key: wp.int64,
) -> wp.bool:
    # Does ``key`` win at every vertex of the two closed 1-rings? The read half of
    # ``scatter.lock_two_rings``, and the same table: ``locks`` is a minimum over candidates
    # *including this one*, so the test is equality rather than ``<=``.
    #
    # Equality is only a sound win test because ``scramble_index`` is injective; it was not, and
    # two candidates sharing a key both passed this. This rule was also written out inline three
    # times before it was one function, and the copies had already diverged: the improvement that
    # hashed the key reached one of them and not the other.
    if locks[s] != key or locks[r] != key:
        return False
    for i in range(offsets[s], offsets[s + 1]):
        if locks[columns[i]] != key:
            return False
    for i in range(offsets[r], offsets[r + 1]):
        if locks[columns[i]] != key:
            return False
    return True


@wp.func
def scramble_index(index: wp.int32) -> wp.int64:
    # Spatially incoherent *and injective* lock key for the independent-set pass, from the
    # candidate's own index. Two separate properties, and the selection needs both.
    #
    # **Incoherent**, which is the load-bearing detail of the whole parallel selection.
    # ``edges_unique`` orders edges lexicographically by endpoint index, which on any structured
    # mesh is *spatially monotone* -- and a monotone key field has essentially one local minimum, so
    # a min-key lock commits a single collapse per pass however many candidates there are. Measured
    # on ``saddle_graded``: locking by raw edge index yields exactly **1** winner out of 51 546
    # candidates, and locking by quadric cost yields 23 (the cost field is smoothly graded there, so
    # it is monotone too). Hashing the index breaks the correlation and restores the expected
    # ~candidates/valence winners. That is what the high half carries: ``array.lowbias32`` with the
    # top bit cleared, so the key stays non-negative and ``INT64_MAX`` remains usable as the
    # unclaimed sentinel. The measurement behind the hash is recorded there, on the shared function.
    #
    # **Injective**, which is why the key is 64 bits and not the natural 32. ``wins_key_everywhere``
    # tests equality against a neighbourhood minimum, so two candidates sharing a key both win and
    # both commit -- overlapping 1-rings, which is a corrupted mesh rather than a worse one. A
    # masked ``lowbias32`` is exactly 2-to-1, and at scan-mesh candidate counts the collision rate
    # is ~m^2 / 2^32, i.e. not negligible. Appending the index in the low half restores injectivity
    # without disturbing the ordering the high half provides, so every candidate that used to win
    # uniquely still does and only a tie changes -- from "both commit" to "the lower index takes
    # it". This is what lets both paths run **one** lock pass: the quadric path used to follow this
    # with a second, raw-index lock (``claim_collapse_index``) purely to break such a tie, and the
    # isotropic path never did, which was the asymmetry that made the collision reachable at all.
    hashed = wp.int64(lowbias32(wp.uint32(index)) & wp.uint32(0x7FFFFFFF))
    return (hashed << wp.int64(32)) | wp.int64(index)


@wp.kernel(enable_backward=False)
def claim_collapse_key(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_min_key: wp.array[wp.int64],
) -> None:
    # The winning (smallest scrambled) key over the closed 1-rings of both endpoints.
    #
    # **Both collapse paths launch this same kernel**, exactly once each, and read the answer the
    # same way -- ``scramble_index`` being injective is what removed the quadric path's second,
    # raw-index lock pass, which existed only to break a key collision the isotropic path never
    # guarded against at all. It was two kernels -- ``claim_collapses`` and this -- whose bodies
    # became byte-identical once ``scatter.lock_two_rings`` was extracted and the isotropic path's
    # raw-index key was fixed; the duplicate scan found them the same pass that produced them,
    # which is section 3's point about a fusion not being done until the shared code has a name.
    # The key is ``scramble_index(k)`` and not ``k`` for the reason that function records at
    # length: a min-key lock over a *spatially monotone* key field has essentially one local
    # minimum, so it commits a single collapse per pass however many candidates there are. This
    # kernel locked by the raw edge index until it was measured -- on ``saddle`` at a 2x band,
    # 40 934 candidates yielded exactly **1** winner against 608 hashed, and five passes of
    # ``_collapse_pass`` removed **5** vertices of 17 689 against 2 761, at the same wall clock
    # (10.6 against 10.7 ms) because a pass is dominated by its rebuild rather than by its commits.
    # A flat ``creation.grid`` shows it without the lift and is what
    # ``test_collapse_pass_commits_a_useful_fraction_on_a_structured_patch`` asserts against.
    #
    # Two things measured with the change, interleaved over five alternating pairs at
    # ``iterations=3``, that the next reader will want. **Where it helps:** on ``saddle`` at
    # ``target = mean_edge`` the achieved-over-requested edge length goes 0.93 -> 0.98, the face
    # count 39 100 -> 35 488 (the input is 34 848) and the worst aspect ratio 136 -> 2.6, for 2.9 %
    # more wall clock. On the icospheres at half the mean edge -- the whole of
    # ``tests/test_remesh.py`` -- the output is *identical* either way, because there the split
    # stage does the work and collapse commits nothing under either key. That is why the suite
    # passed against the broken version. **Where it does not:** on ``saddle_graded`` the target
    # tracking improves the same way (0.73 -> 0.85, 64 867 -> 40 893 faces) but the worst triangles
    # get worse (99th-pct aspect 177 -> 7 440, three float32-degenerate faces against none). That
    # is not this key's defect -- at the tests' target the *raw* key leaves 192 degenerate faces
    # against the hashed key's 40 -- it is the unweighted ``_smooth_pass`` that
    # ``benchmarks/test_remesh.py::test_isotropic_remesh`` already records as open, unmasked here by
    # a collapse stage that finally commits. It needs a fold guard, not a different lock key.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    lock_two_rings(offsets, columns, s, removed[k], scramble_index(k), out_min_key)


@wp.kernel(enable_backward=False)
def commit_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    claim: wp.array[wp.int64],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    if not wins_key_everywhere(offsets, columns, claim, s, r, scramble_index(k)):
        return
    out_remap[r] = s
    out_positions[s] = pos[k]
    wp.atomic_add(out_count, 0, 1)


@wp.kernel
def faces_with_distinct_indices(faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]) -> None:
    # A face survives a vertex remap only if its three corners are still three distinct vertices.
    # Every decimation here ends in one: an edge collapse merges two of them, vertex clustering
    # sends two into the same cell.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    out_mask[f] = i0 != i1 and i1 != i2 and i0 != i2


@wp.kernel
def valence_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    valence: wp.array[wp.int32],
    boundary_vertex: wp.array[wp.bool],
    feature_angle: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    # Never flip a feature edge (sharp dihedral between the two incident faces).
    n0 = face_normal(vertices, faces, f0)
    n1 = face_normal(vertices, faces, f1)
    # ``vector_angle`` is the atan2 form; two coplanar faces across an edge is the common
    # case here and is exactly where ``acos(dot)`` loses its digits, and this is a *branch*,
    # so the lost digits change a flip decision rather than a printed number. The sibling
    # feature test in ``classify_vertices`` above already reads this way.
    if vector_angle(n0, n1) > feature_angle:
        return
    a, b, c, d = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if a < 0:
        return
    ap, bp, cp, dp = flip_quad_positions_d(vertices, a, b, c, d)
    if not is_unfold_quadrangle_convex(ap, bp, cp, dp):
        return
    # Shape guard. Convexity makes the flip *legal* but says nothing about the shape of what it
    # produces, and the valence objective below is blind to geometry: on a graded mesh it will
    # happily turn two slivers into two worse ones, which in float32 lands on exactly-zero area.
    # (Measured on ``saddle_graded``: the swap stage alone produced 3 992 zero-area faces out of
    # 92 100, and none survive this guard. ``delone_flip_candidates`` has its own deviation and
    # aspect gates; this is the equivalent for the valence objective.)
    #
    # ``triangle_aspect_ratio`` is circumradius / 2 * inradius and returns +inf for a degenerate
    # triangle, so the two tests below read as "never create a degenerate triangle" and "never make
    # the worse of the pair worse". Post-flip faces are (a, b, d) and (c, d, b) -- see commit_flips.
    aspect_after = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(cp, dp, bp))
    if not wp.isfinite(aspect_after):
        return
    if aspect_after > wp.max(triangle_aspect_ratio(ap, cp, dp), triangle_aspect_ratio(cp, ap, bp)):
        return
    # Target valence: 4 on the boundary, 6 in the interior.
    ta = wp.where(boundary_vertex[a], 4, 6)
    tb = wp.where(boundary_vertex[b], 4, 6)
    tc = wp.where(boundary_vertex[c], 4, 6)
    td = wp.where(boundary_vertex[d], 4, 6)
    va = valence[a]
    vb = valence[b]
    vc = valence[c]
    vd = valence[d]
    before = (
        (va - ta) * (va - ta)
        + (vb - tb) * (vb - tb)
        + (vc - tc) * (vc - tc)
        + (vd - td) * (vd - td)
    )
    after = (
        (va - 1 - ta) * (va - 1 - ta)
        + (vb + 1 - tb) * (vb + 1 - tb)
        + (vc - 1 - tc) * (vc - 1 - tc)
        + (vd + 1 - td) * (vd + 1 - td)
    )
    out_flip[k] = after < before


@wp.kernel
def accumulate_one_ring(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    out_sum: wp.array[wp.vec3],
    out_degree: wp.array[wp.int32],
) -> None:
    # Unweighted one-ring centroid. Note this is *not* the area-equalizing relaxation that
    # Botsch-Kobbelt specify: on a regular graded grid every vertex already sits at the plain
    # average of its neighbours, so this smoother is at a fixed point and cannot equalize the
    # sampling. Area-weighting it takes the 99th-percentile aspect ratio on such a patch from 352
    # to 20, but also makes ``is_watertight`` fail on ``cave_cube`` through a self-intersection at
    # *every* step size down to lam=0.1, so it needs a fold guard first. See
    # ``tests/test_remesh.py::test_remesh_emits_no_degenerate_faces``.
    e = wp.int32(wp.tid())
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    wp.atomic_add(out_sum, u, vertices[v])
    wp.atomic_add(out_degree, u, 1)
    wp.atomic_add(out_sum, v, vertices[u])
    wp.atomic_add(out_degree, v, 1)


@wp.func
def tangential_smooth_step(
    vertex: wp.vec3,
    code: wp.int32,
    normal: wp.vec3,
    ring_sum: wp.vec3,
    degree: wp.int32,
    lam: wp.float32,
) -> wp.vec3:
    # Move a free vertex toward its one-ring centroid, but only within the tangent plane, so the
    # surface is smoothed without being shrunk. Pinned vertices and isolated ones stay put.
    p = vertex
    if code != FREE_VERTEX or degree == 0:
        return p
    centroid = ring_sum / wp.float32(degree)
    delta = centroid - p
    tangential = project_out_normal(delta, normal)
    return p + lam * tangential


@wp.func
def reproject_vertices(
    vertex: wp.vec3, code: wp.int32, mesh_id: wp.uint64, max_dist: wp.float32
) -> wp.vec3:
    # Snap a free vertex back onto the closest point of the original surface, undoing the drift the
    # smoothing pass introduces. Pinned vertices and failed queries keep their position.
    if code != FREE_VERTEX:
        return vertex
    query = wp.mesh_query_point_no_sign(mesh_id, vertex, max_dist)
    if query.result:
        return wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    return vertex


@wp.func
def clamp_to_surface_band(
    vertex: wp.vec3, mesh_id: wp.uint64, max_deviation: wp.float32, max_dist: wp.float32
) -> wp.vec3:
    # Pull a vertex back until it is within ``max_deviation`` of the original surface, along the
    # line to its own closest point. Unlike ``reproject_vertices`` this applies to *every* vertex --
    # a crease or corner is exactly the kind that drifts and that reprojection refuses to touch --
    # and it moves the vertex only as far as the bound requires, so a vertex already inside the band
    # is untouched and detail is not flattened onto the input surface.
    query = wp.mesh_query_point_no_sign(mesh_id, vertex, max_dist)
    if not query.result:
        return vertex
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    offset = vertex - closest
    distance = wp.length(offset)
    if distance <= max_deviation:
        return vertex
    return closest + offset * (max_deviation / distance)


@wp.func
def local_corner(faces: wp.array[wp.int32], f: wp.int32, vertex: wp.int32) -> wp.int32:
    # Which corner of face ``f`` holds ``vertex``, or -1. Needed because an edge-length table is
    # indexed by *corner*, while the flip machinery speaks in vertex indices.
    for k in range(3):
        if faces[f * 3 + k] == vertex:
            return k
    return wp.int32(-1)


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
    k = wp.int32(wp.tid())
    out_flip[k] = False
    out_new_length[k] = 0.0
    f0 = adjacency[k, 0]
    f1 = adjacency[k, 1]
    # The shared guard's duplicate-edge test is load-bearing here and is *not* free: intrinsically,
    # the flipped edge is a different geodesic between the same two endpoints and is a legitimate
    # new edge, but a simplicial face buffer cannot hold two of them and the topology this pass
    # rebuilds is keyed on the vertex pair. So the guard is the reason a violating edge can be
    # unflippable, and therefore the reason a round can end with the triangulation still not
    # Delaunay -- which reads as convergence at the wrapper. ``remesh.intrinsic_delaunay``'s Notes
    # state that limitation for callers; a Delta-complex representation is what lifts it.
    first, _apex1, second, apex0 = _resolve_flip_quad_guarded(
        faces, adjacency_edges, unshared, sorted_edge_keys, key_base, k, f0, out_quad
    )
    if first < wp.int32(0):
        return

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
    k = wp.int32(wp.tid())
    f0, f1, won = flip_claim_won(
        flip, quad, adjacency, face_claim, edge_claim, edge_claim_mask, key_base, k
    )
    if not won:
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


@wp.kernel
def cluster_accumulate(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_sum: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    v = wp.int32(wp.tid())
    wp.atomic_add(out_sum, labels[v], vertices[v])
    wp.atomic_add(out_count, labels[v], 1)


@wp.kernel
def cluster_min_center_distance(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    voxel_size: wp.float32,
    out_min_distance: wp.array[wp.float32],
) -> None:
    # Pass 1 of the "closest to the cell centre" representative: the winning *distance* per cluster.
    # Split from the index pick so both passes use 32-bit atomics only; the two together are
    # deterministic because pass 2 breaks ties by lowest vertex index.
    v = wp.int32(wp.tid())
    wp.atomic_min(
        out_min_distance,
        labels[v],
        squared_distance_to_own_cell_center(vertices[v], origin, voxel_size),
    )


@wp.kernel
def cluster_pick_closest(
    labels: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    origin: wp.vec3,
    voxel_size: wp.float32,
    min_distance: wp.array[wp.float32],
    out_representative: wp.array[wp.int32],
) -> None:
    # Pass 2: whichever vertices tie for their cluster's winning distance, the lowest index wins.
    v = wp.int32(wp.tid())
    if (
        squared_distance_to_own_cell_center(vertices[v], origin, voxel_size)
        <= min_distance[labels[v]]
    ):
        wp.atomic_min(out_representative, labels[v], v)


# Objective for ``objective_flip_candidates``. A warp-uniform kernel argument rather than a
# ``wp.Function``, so both predicates share one compiled module (AGENTS.md section 4).
OBJECTIVE_PLANARITY = wp.constant(wp.int32(0))  # improve triangle shape on a near-planar quad
OBJECTIVE_CURVATURE = wp.constant(wp.int32(1))  # pick whichever diagonal bends the surface less
OBJECTIVE_T_VERTEX = wp.constant(wp.int32(2))  # break up a sliver whose apex sits on the far edge

# Relative margin a flip must beat the current diagonal by. Without it a quad whose two diagonals
# score equally (every quad of a regular grid) flips back and forth forever, one pass each way.
OBJECTIVE_EPS = wp.constant(wp.float32(1e-6))


@wp.kernel
def objective_flip_candidates(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    adjacency: wp.array2d[wp.int32],
    adjacency_edges: wp.array2d[wp.int32],
    unshared: wp.array2d[wp.int32],
    region_flags: wp.array[wp.int32],
    sorted_edge_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    objective: wp.int32,
    metric: wp.int32,
    planar_cos: wp.float32,
    aspect_threshold: wp.float32,
    out_flip: wp.array[wp.bool],
    out_quad: wp.array2d[wp.int32],
) -> None:
    # Quad convention (shared with ``delone_flip_candidates``): the current diagonal is a-c, with
    # faces (a, b, c) and (a, c, d); the flip replaces it with b-d, giving (a, b, d) and (d, b, c).
    #
    # Fourteen arguments and **deliberately not bundled into a ``@wp.struct``**, unlike
    # ``holes.stitch_dp_diag`` and ``holes.fill_dp_span``, which have the same width. Those launch
    # once per DP cell-diagonal or span -- hundreds of times, with nothing else in the loop. This
    # one launches once per *flip round*, and a round rebuilds the whole face adjacency around it:
    # measured on a noisy icosphere(4), ``flip_to_delaunay`` converges in **2** rounds at 514 us
    # each, so the nine bundleable arguments bound the saving at 18 us, **1.75 %** of the call. The
    # width alone is not the criterion; the launch count around it is.
    k = wp.int32(wp.tid())
    out_flip[k] = wp.bool(False)
    a, b, c, d = _resolve_flip_quad_in_region(
        faces,
        adjacency,
        adjacency_edges,
        unshared,
        region_flags,
        sorted_edge_keys,
        key_base,
        k,
        out_quad,
    )
    if a < 0:
        return
    ap = vertices[a]
    bp = vertices[b]
    cp = vertices[c]
    dp = vertices[d]

    # A non-convex quad has no valid flip: the new diagonal would fall outside it. The rest of this
    # kernel stays in float32 -- only the convexity branch needs the promoted corners, which is what
    # ``flip_quad_positions_d`` names.
    apd, bpd, cpd, dpd = flip_quad_positions_d(vertices, a, b, c, d)
    if not is_unfold_quadrangle_convex(apd, bpd, cpd, dpd):
        return

    if objective == OBJECTIVE_T_VERTEX:
        # A T-vertex shows up as a sliver: one apex sits (nearly) on the opposite edge, which drives
        # the circumradius-to-inradius ratio through the roof. Flip only when the sliver is *that*
        # bad and the flip actually improves it, so a merely thin triangle is left alone.
        old_worst = wp.max(triangle_aspect_ratio(ap, bp, cp), triangle_aspect_ratio(ap, cp, dp))
        if not (old_worst > aspect_threshold):  # also excludes a NaN ratio
            return
        new_worst = wp.max(triangle_aspect_ratio(ap, bp, dp), triangle_aspect_ratio(dp, bp, cp))
        out_flip[k] = new_worst < old_worst
        return

    normal_abc = triangle_normal(ap, bp, cp)
    normal_acd = triangle_normal(ap, cp, dp)
    normal_abd = triangle_normal(ap, bp, dp)
    normal_dbc = triangle_normal(dp, bp, cp)

    if objective == OBJECTIVE_PLANARITY:
        # Only rewrite a quad that is flat enough for the rewrite not to change the surface. The
        # gate is on the *cosine* of the dihedral so the kernel needs no inverse trigonometry.
        if wp.dot(normal_abc, normal_acd) < planar_cos:
            return
        old_score = wp.min(
            triangle_quality(ap, bp, cp, metric), triangle_quality(ap, cp, dp, metric)
        )
        new_score = wp.min(
            triangle_quality(ap, bp, dp, metric), triangle_quality(dp, bp, cp, metric)
        )
        out_flip[k] = new_score > old_score * (1.0 + OBJECTIVE_EPS)
        return

    # Curvature: keep whichever diagonal leaves the two triangles closer to coplanar. Unlike the
    # planarity objective this deliberately *does* change the surface -- that is the point.
    old_bend = wp.abs(dihedral_angle(normal_abc, normal_acd, cp - ap))
    new_bend = wp.abs(dihedral_angle(normal_abd, normal_dbc, dp - bp))
    out_flip[k] = new_bend < old_bend * (1.0 - OBJECTIVE_EPS)


# ---------------------------------------------------------------------------
# Quadric error metric (Garland-Heckbert) decimation
# ---------------------------------------------------------------------------

# Quadrics are accumulated in float64. That is not caution: the entries are sums of ``area * d^2``
# with ``d`` an absolute plane offset, so on a mesh whose coordinates are far from the origin they
# span many orders of magnitude and a float32 accumulation loses the small ones -- which are exactly
# the terms that distinguish two candidate collapses. libigl and MeshLab both use double here.

# Below this determinant (relative to the quadric's own scale) the 3x3 system is treated as singular
# and the optimum falls back to the edge midpoint: a planar neighbourhood has a whole plane of
# equally good positions and picking one by inversion amplifies noise.
QUADRIC_SINGULAR_EPS = wp.constant(wp.float64(1e-12))

# A collapse is rejected when it would turn an incident face's normal by more than this. Zero would
# allow a face to become exactly degenerate; 0.2 (~78 degrees) still permits real simplification of
# a curved region while refusing an outright fold.
COLLAPSE_MIN_NORMAL_DOT = wp.constant(wp.float32(0.2))


@wp.func
def plane_quadric(normal: wp.vec3d, offset: wp.float64, weight: wp.float64) -> wp.mat44d:
    # Garland-Heckbert fundamental quadric of the plane ``dot(normal, x) + offset = 0``, scaled by
    # ``weight``. Laid out so that ``[p, 1]^T Q [p, 1]`` is the weighted squared distance to the
    # plane: the leading 3x3 block is ``n n^T``, the last row and column are ``offset * n``, and the
    # corner is ``offset^2``.
    a = weight * normal[0]
    b = weight * normal[1]
    c = weight * normal[2]
    d = weight * offset
    return wp.mat44d(
        a * normal[0],
        a * normal[1],
        a * normal[2],
        a * offset,
        b * normal[0],
        b * normal[1],
        b * normal[2],
        b * offset,
        c * normal[0],
        c * normal[1],
        c * normal[2],
        c * offset,
        d * normal[0],
        d * normal[1],
        d * normal[2],
        d * offset,
    )


@wp.func
def quadric_error(quadric: wp.mat44d, p: wp.vec3d) -> wp.float64:
    # ``[p, 1]^T Q [p, 1]``: the accumulated squared distance from ``p`` to every plane folded into
    # ``Q``. Clamped at zero, since a float64 sum of positive-semidefinite terms can still land a
    # hair below it and a negative "error" would sort ahead of every real candidate.
    homogeneous = wp.vec4d(p[0], p[1], p[2], wp.float64(1.0))
    return wp.max(wp.float64(0.0), wp.dot(homogeneous, quadric * homogeneous))


@wp.func
def quadric_optimum(quadric: wp.mat44d, fallback: wp.vec3d) -> wp.vec3d:
    # Position minimizing the quadric: solve ``A p = -b`` for the leading 3x3 block ``A`` and the
    # last column ``b``. ``fallback`` (the edge midpoint) is returned when ``A`` is singular
    # relative to its own scale, which is the planar case -- there the minimum is a whole plane and
    # inverting a near-singular matrix would place the vertex arbitrarily far away.
    a = wp.mat33d(
        quadric[0, 0],
        quadric[0, 1],
        quadric[0, 2],
        quadric[1, 0],
        quadric[1, 1],
        quadric[1, 2],
        quadric[2, 0],
        quadric[2, 1],
        quadric[2, 2],
    )
    scale = wp.abs(quadric[0, 0]) + wp.abs(quadric[1, 1]) + wp.abs(quadric[2, 2])
    if scale <= wp.float64(0.0):
        return fallback
    if wp.abs(wp.determinant(a)) <= QUADRIC_SINGULAR_EPS * scale * scale * scale:
        return fallback
    b = wp.vec3d(quadric[0, 3], quadric[1, 3], quadric[2, 3])
    return -(wp.inverse(a) * b)


@wp.kernel
def accumulate_face_quadrics(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_quadrics: wp.array[wp.mat44d]
) -> None:
    # Area-weighted plane quadric of each face, scattered onto its three corners. Area weighting is
    # Garland-Heckbert's: a large triangle constrains its vertices more than a sliver does.
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
    cross = wp.cross(v1 - v0, v2 - v0)
    double_area = wp.length(cross)
    if double_area <= wp.float64(0.0):
        return
    normal = cross / double_area
    quadric = plane_quadric(normal, -wp.dot(normal, v0), double_area * wp.float64(0.5))
    add_corner_triple(out_quadrics, faces, f, quadric, quadric, quadric)


@wp.kernel
def pass_edge_keys(
    faces: wp.array[wp.int32],
    state: wp.array[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
) -> None:
    # ``adjacency.face_edge_keys`` over a fixed-capacity buffer: the same three keys per live face,
    # and a maximal sentinel for each padded one. Sentinels sort to the very end, so bounding the
    # grouping below by ``3 * n_faces`` excludes them exactly. That padding is the whole difference
    # between the two kernels; the keys themselves come from the shared ``write_face_edge_keys``.
    f = wp.int32(wp.tid())
    if f >= state[DECIMATION_FACES]:
        c = f * 3
        out_keys[c + 0] = EDGE_KEY_PAD
        out_keys[c + 1] = EDGE_KEY_PAD
        out_keys[c + 2] = EDGE_KEY_PAD
        return
    write_face_edge_keys(faces, f, base, out_keys)


@wp.kernel
def mark_unique_edge_starts(
    sorted_keys: wp.array[wp.uint64], state: wp.array[wp.int32], out_starts: wp.array[wp.int32]
) -> None:
    # Flag the first position of every run of equal keys among the live corners -- what
    # ``grouping.unique_1d`` answers with a hash table, over sorted keys instead, and emitting
    # ``int32`` so the scan that follows needs no cast. See ``mark_edge_pair_starts`` above for the
    # one condition the two differ by.
    i = wp.int32(wp.tid())
    start = wp.int32(0)
    if i < state[DECIMATION_FACES] * 3 and sorted_run_start(sorted_keys, i):
        start = wp.int32(1)
    out_starts[i] = start


@wp.kernel
def emit_unique_edges(
    faces: wp.array[wp.int32],
    order: wp.array[wp.int32],
    starts: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    state: wp.array[wp.int32],
    edge_capacity: wp.int32,
    out_unique_edges: wp.array2d[wp.int32],
    out_inverse: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # One launch for everything ``edges.edges_unique`` returns: the ascending unique edge rows and
    # the corner -> unique-edge map. ``ranks`` is the inclusive scan of ``starts``, so its entry
    # minus one
    # is the unique index of the key at sorted position ``i`` -- the same ascending-key order
    # ``unique_1d`` produces by sorting its compacted keys, which is what keeps the edge numbering
    # (and therefore ``scramble_index``'s lock keys) identical to the composed path.
    #
    # A padded corner is sent to the dummy edge slot so ``scatter_edge_incidence`` can run over the
    # whole corner buffer, and the last live position publishes the live edge count -- the tail read
    # that sizes ``flatnonzero``'s output, left on the device.
    i = wp.int32(wp.tid())
    live = state[DECIMATION_FACES] * 3
    corner = order[i]
    if i >= live:
        out_inverse[corner] = edge_capacity
        return
    e = ranks[i] - 1
    # The capacity is a bound the *previous* pass measured, and a collapse removes at least three
    # undirected edges and adds none, so an overflow cannot fire. Handled anyway -- and handled in
    # **both** writes, which is the whole point: clamping only the row write below would leave
    # ``out_inverse`` pointing a live corner at an edge slot past the end of the buffer, and
    # ``scatter_edge_incidence`` dereferences exactly that index, so a clamp meant to prevent an
    # out-of-bounds write would have left an out-of-bounds read. An overflowing corner goes to the
    # dummy slot, which is where a padded one already goes.
    out_inverse[corner] = wp.min(e, edge_capacity)
    if starts[i] != 0 and e < edge_capacity:
        a, b = edge_endpoints(faces, corner)
        out_unique_edges[e, 0] = a
        out_unique_edges[e, 1] = b
    if i + 1 == live:
        out_state[DECIMATION_EDGES] = wp.min(ranks[i], edge_capacity)


@wp.kernel
def pad_unique_edge_tail(
    state: wp.array[wp.int32], dummy_vertex: wp.int32, out_unique_edges: wp.array2d[wp.int32]
) -> None:
    # Point every padded edge row at the dummy vertex. Its code is frozen to CORNER_VERTEX, so
    # ``quadric_collapse_candidates`` rejects the row on its first test and never reads further.
    e = wp.int32(wp.tid())
    if e >= state[DECIMATION_EDGES]:
        out_unique_edges[e, 0] = dummy_vertex
        out_unique_edges[e, 1] = dummy_vertex


@wp.kernel
def reset_collapse_rounds(out_state: wp.array[wp.int32]) -> None:
    # dim=1. The round-loop state array at the start of a pass: ``array.LOOP_ROUND`` /
    # ``LOOP_CONDITION`` in the shared first two slots, with this loop's own third appended (see
    # ``COLLAPSE_COMMITS``). The condition starts at 1 because ``wp.capture_while`` reads it before
    # the first round.
    #
    # A kernel rather than ``array.assign``, because that is a host-to-device copy and the pass this
    # runs inside is captured as a graph.
    _ = wp.int32(wp.tid())
    out_state[LOOP_ROUND] = 0
    out_state[LOOP_CONDITION] = 1
    out_state[COLLAPSE_COMMITS] = 0


@wp.kernel
def edge_csr_triplets(
    unique_edges: wp.array2d[wp.int32],
    state: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_columns: wp.array[wp.int32],
) -> None:
    # ``graph.edges_to_csr``'s symmetric triplet expansion, with the padded rows sent out of range
    # instead of to the dummy vertex. ``warp.sparse.bsr_from_triplets`` drops an out-of-range index
    # silently, which is what is wanted here -- pointing them all at the dummy instead makes tens of
    # thousands of triplets collide on **one** entry, and its accumulation atomic then serializes:
    # measured 4.25 ms of a 4.82 ms pass on ``saddle``, 88 % of it, against 0.03 ms once dropped.
    e = wp.int32(wp.tid())
    a = e * 2
    if e >= state[DECIMATION_EDGES]:
        out_rows[a + 0] = -1
        out_columns[a + 0] = -1
        out_rows[a + 1] = -1
        out_columns[a + 1] = -1
        return
    u = unique_edges[e, 0]
    v = unique_edges[e, 1]
    out_rows[a + 0] = u
    out_columns[a + 0] = v
    out_rows[a + 1] = v
    out_columns[a + 1] = u


@wp.kernel
def freeze_dummy_vertex(dummy_vertex: wp.int32, out_codes: wp.array[wp.int32]) -> None:
    # dim=1. See ``pad_unique_edge_tail``.
    _ = wp.int32(wp.tid())
    out_codes[dummy_vertex] = CORNER_VERTEX


@wp.kernel
def collapse_pass_budgets(
    target_faces: wp.int32,
    state: wp.array[wp.int32],
    out_half: wp.array[wp.int32],
    out_surplus: wp.array[wp.int32],
) -> None:
    # dim=1. The two per-pass budgets the host used to compute: the cheapest-half cut over the
    # live candidates, and the pass's face surplus (an interior collapse removes two faces).
    _ = wp.int32(wp.tid())
    out_half[0] = wp.max(wp.int32(1), state[DECIMATION_EDGES] // 2)
    out_surplus[0] = (state[DECIMATION_FACES] - target_faces) // 2


@wp.kernel
def compact_faces(
    remapped: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    dummy_vertex: wp.int32,
    out_faces: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # Move the surviving faces to the front of the face buffer and pad the rest with the dummy
    # triangle, publishing the new face count. Survivors move strictly left and are read from a
    # separate buffer, so the compaction and the padding cannot race.
    f = wp.int32(wp.tid())
    kept = ranks[ranks.shape[0] - 1]
    if f == 0:
        out_state[DECIMATION_FACES] = kept
    if flags[f] != 0:
        slot = (ranks[f] - 1) * 3
        out_faces[slot + 0] = remapped[f * 3 + 0]
        out_faces[slot + 1] = remapped[f * 3 + 1]
        out_faces[slot + 2] = remapped[f * 3 + 2]
    if f >= kept:
        out_faces[f * 3 + 0] = dummy_vertex
        out_faces[f * 3 + 1] = dummy_vertex
        out_faces[f * 3 + 2] = dummy_vertex


@wp.kernel
def compact_vertices(
    positions: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_vertices: wp.array[wp.vec3],
    out_remap: wp.array[wp.int32],
    out_state: wp.array[wp.int32],
) -> None:
    # ``repair.remove_unreferenced_vertices`` with its host readback replaced by the scan it was
    # reading: ``ranks`` is the inclusive scan of the referenced mask, so ``ranks[v]-1`` is the new
    # index of vertex ``v`` and its last element is the surviving count. Reads and writes use
    # different buffers, so this may run over the whole capacity.
    v = wp.int32(wp.tid())
    kept = ranks[ranks.shape[0] - 1]
    if v == 0:
        out_state[DECIMATION_VERTICES] = kept
    if flags[v] != 0:
        slot = ranks[v] - 1
        out_remap[v] = slot
        out_vertices[slot] = positions[v]
    else:
        out_remap[v] = -1


@wp.kernel
def compose_vertex_index(
    collapse_remap: wp.array[wp.int32],
    compaction_remap: wp.array[wp.int32],
    index: wp.array[wp.int32],
) -> None:
    # Carry ``quadric_decimate``'s per-input-vertex provenance across one pass, in place: an input
    # vertex sits at some live slot, the pass's collapse sends that slot to its survivor, and the
    # compaction renumbers the survivor. Composing the two here rather than returning either one is
    # what keeps the map a single array of the *input* length -- fixed width, so the pass stays
    # capturable -- instead of a chain of per-pass maps the caller would have to fold itself.
    #
    # ``index`` is genuinely in place: it is both the pass's input and its result, so an ``out_``
    # prefix would read as write-only (CLAUDE.md section 3's first exemption class).
    i = wp.int32(wp.tid())
    current = index[i]
    if current >= 0:
        index[i] = compaction_remap[collapse_remap[current]]


@wp.kernel
def compact_face_provenance(
    source: wp.array[wp.int32],
    flags: wp.array[wp.int32],
    ranks: wp.array[wp.int32],
    out_source: wp.array[wp.int32],
) -> None:
    # The companion of ``compact_faces`` for its provenance column: a face that survives carries its
    # source-face id to the same slot the face itself moved to. Reads and writes are separate
    # buffers for the same reason ``compact_faces`` reads ``remapped``.
    f = wp.int32(wp.tid())
    if flags[f] != 0:
        out_source[ranks[f] - 1] = source[f]


@wp.kernel
def apply_vertex_remap(
    remap: wp.array[wp.int32], dummy_vertex: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # In place, one element per corner: a padded corner keeps pointing at the dummy vertex, whose
    # remap entry is -1 because no face references it.
    c = wp.int32(wp.tid())
    v = out_faces[c]
    if v != dummy_vertex:
        out_faces[c] = remap[v]


@wp.kernel
def quadric_collapse_candidates(
    unique_edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    quadrics: wp.array[wp.mat44d],
    codes: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    vertex_face_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
    out_removed: wp.array[wp.int32],
    out_pos: wp.array[wp.vec3],
    out_cost: wp.array[wp.float32],
) -> None:
    # Garland-Heckbert candidate: the cost of collapsing this edge and where its survivor lands.
    # ``out_cost`` is left at +inf for a rejected edge, so the caller's cost sort puts every
    # rejection past every candidate and the budget cut never picks one up.
    k = wp.int32(wp.tid())
    out_survivor[k] = -1
    out_cost[k] = wp.inf
    u = unique_edges[k, 0]
    v = unique_edges[k, 1]
    is_boundary = edge_face_count[k] == 1

    # The feature rule is ``collapse_survivor``, shared with ``collapse_candidates``. What differs
    # is only the free placement: that one takes the midpoint, this one the quadric's minimizer.
    s, r, placement = collapse_survivor(codes, u, v, is_boundary)
    if placement == COLLAPSE_REJECTED:
        return
    free_position = placement == COLLAPSE_FREE

    if not satisfies_link_condition(offsets, columns, u, v, is_boundary):
        return

    quadric = quadrics[u] + quadrics[v]
    midpoint = (to_vec3d(vertices[u]) + to_vec3d(vertices[v])) * wp.float64(0.5)
    optimum = midpoint
    if free_position:
        optimum = quadric_optimum(quadric, midpoint)
    else:
        optimum = to_vec3d(vertices[s])
    target = to_vec3(optimum)

    if collapse_flips_normal(
        vertices, faces, vertex_face_offsets, vertex_faces, r, s, target
    ) or collapse_flips_normal(vertices, faces, vertex_face_offsets, vertex_faces, s, r, target):
        return

    out_survivor[k] = s
    out_removed[k] = r
    out_pos[k] = target
    out_cost[k] = wp.float32(quadric_error(quadric, optimum))


@wp.kernel
def drop_collapses_past_budget(
    order: wp.array[wp.int32], budget: wp.array[wp.int32], out_survivor: wp.array[wp.int32]
) -> None:
    # Retire every candidate ranked past the budget, ``order`` being the ascending cost ranking.
    # Ranking by cost rather than by edge index is the whole difference between a quadric decimation
    # and a shortest-edge one: the cheapest collapse must win a contested ring.
    #
    # ``budget`` is a 1-element *device* array rather than a launch argument so the whole round loop
    # can run inside one ``wp.capture_while`` graph; ``begin_collapse_round`` writes it. A budget of
    # zero retires everything, which is how a pass that has exhausted its surplus stops committing
    # without the host being told.
    i = wp.int32(wp.tid())
    if i >= budget[0]:
        out_survivor[order[i]] = -1


@wp.kernel
def begin_collapse_round(
    surplus: wp.array[wp.int32], count: wp.array[wp.int32], out_budget: wp.array[wp.int32]
) -> None:
    # dim=1, first op of a round: how many collapses this round may still commit.
    #
    # ``surplus`` is ``(n_faces - target) // 2`` for the pass -- an interior collapse removes two
    # faces -- and ``count`` accumulates the commits of every round so far, so the rounds share one
    # budget. The floor of one while nothing has been committed yet is what lets a pass with a
    # surplus of a single face still finish the job; it cannot manufacture a commit, because the
    # budget only ever *trims* an independent set that is already chosen.
    #
    # A **device** array rather than a launch argument, because the pass that computes it is itself
    # replayed as a graph and the face count it comes from never reaches the host.
    _ = wp.int32(wp.tid())
    budget = surplus[0] - count[0]
    if count[0] == 0:
        budget = wp.max(budget, wp.int32(1))
    out_budget[0] = wp.max(budget, wp.int32(0))


@wp.kernel
def end_collapse_round(
    max_rounds: wp.int32, count: wp.array[wp.int32], out_state: wp.array[wp.int32]
) -> None:
    # dim=1, last op of a round: decide whether another round against this same scoring is worth
    # running. See ``reset_collapse_rounds`` for the slot table.
    #
    # It stops when the round committed nothing -- a further round cannot, since the state it
    # reads is then unchanged -- or at the round cap. A budget-exhausted pass stops through that
    # same test: ``begin_collapse_round`` writes a zero budget, so nothing commits.
    _ = wp.int32(wp.tid())
    out_state[LOOP_ROUND] = out_state[LOOP_ROUND] + wp.int32(1)
    progressed = count[0] > out_state[COLLAPSE_COMMITS]
    out_state[COLLAPSE_COMMITS] = count[0]
    keep_going = progressed and out_state[LOOP_ROUND] < max_rounds
    out_state[LOOP_CONDITION] = wp.where(keep_going, wp.int32(1), wp.int32(0))


@wp.kernel(enable_backward=False)
def mark_collapse_winners(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    min_key: wp.array[wp.int64],
    cost: wp.array[wp.float32],
    out_survivor: wp.array[wp.int32],
    out_cost: wp.array[wp.float32],
) -> None:
    # Pass 2 of 2: the win test, kept separate from the commit so the caller can apply its per-pass
    # budget *after* the independent set is known. Trimming members from an independent set keeps it
    # independent; trimming the candidate list beforehand would change which set is found.
    k = wp.int32(wp.tid())
    out_cost[k] = wp.inf
    s = survivor[k]
    if s < 0:
        out_survivor[k] = -1
        return
    r = removed[k]
    if not wins_key_everywhere(offsets, columns, min_key, s, r, scramble_index(k)):
        out_survivor[k] = -1
        return
    out_survivor[k] = s
    out_cost[k] = cost[k]


@wp.kernel(enable_backward=False)
def commit_selected_collapses(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    target_pos: wp.array[wp.vec3],
    out_remap: wp.array[wp.int32],
    out_positions: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
) -> None:
    # Apply an already-independent set: no claim test, because ``mark_collapse_winners`` established
    # independence and the budget cut only ever *removes* members from it.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    out_remap[removed[k]] = s
    out_positions[s] = target_pos[k]
    wp.atomic_add(out_count, 0, 1)


@wp.kernel(enable_backward=False)
def lock_collapse_neighborhoods(
    survivor: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    out_locked: wp.array[wp.int32],
) -> None:
    # Mark the closed 1-rings of both endpoints of every committed collapse, so a later
    # independent-set round in the *same* pass can be told which candidates the commit invalidated
    # (see ``drop_locked_candidates``). Plain stores rather than atomics: every write is the same
    # value.
    k = wp.int32(wp.tid())
    s = survivor[k]
    if s < 0:
        return
    r = removed[k]
    out_locked[s] = 1
    out_locked[r] = 1
    for i in range(offsets[s], offsets[s + 1]):
        out_locked[columns[i]] = 1
    for i in range(offsets[r], offsets[r + 1]):
        out_locked[columns[i]] = 1


@wp.kernel(enable_backward=False)
def drop_locked_candidates(
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    out_survivor: wp.array[wp.int32],
) -> None:
    # Restore the pass's candidate list for another independent-set round, retiring every candidate
    # whose two closed 1-rings touch an already-collapsed neighbourhood.
    #
    # That disjointness is exactly what makes reusing the pass's scoring legal. A candidate whose
    # closed 1-rings miss every locked vertex has *no incident face* holding a collapsed endpoint,
    # so its endpoints' quadrics, its cost, its target position, its link condition and its
    # normal-flip veto are all still the ones the scoring pass computed. Fail that test and the
    # candidate must wait for the next geometry rebuild.
    k = wp.int32(wp.tid())
    out_survivor[k] = -1
    s = candidates[k]
    if s < 0:
        return
    r = removed[k]
    if locked[s] != 0 or locked[r] != 0:
        return
    for i in range(offsets[s], offsets[s + 1]):
        if locked[columns[i]] != 0:
            return
    for i in range(offsets[r], offsets[r + 1]):
        if locked[columns[i]] != 0:
            return
    out_survivor[k] = s
