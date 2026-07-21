"""
Kernels for wave-parallel ball-pivoting surface reconstruction.

Ports the geometry of Open3D's ``SurfaceReconstructionBallPivoting.cpp`` (``ComputeBallCenter``,
``IsCompatible``, ``FindCandidateVertex``, ``TryTriangleSeed``) to Warp. Instead of the serial
advancing front with a persistent edge structure, the front is recomputed from the current triangle
soup each wave: front edges are the boundary edges (used by exactly one triangle), and manifoldness
is protected by a sorted interior-edge-key guard (edges already used by two triangles may not gain a
third). Every wave commits a conflict-free independent set of triangles through a two-phase
vertex-claim / commit (``wp.atomic_min`` priority on all three vertices), so at least the
globally-lowest-priority triangle always commits and the loop cannot livelock.
"""

import warp as wp

from triwarp.kernels.array import binary_search_sorted_contains
from triwarp.kernels.grouping import pack_edge_key

# Per-thread neighbour scratch for the seed search (Open3D re-scans the KNN result twice).
MAX_SEED_NEIGHBORS = 64
# Relative slack on the empty-ball test: absorbs float32 round-off so the three defining points
# (exactly on the ball in exact arithmetic) do not spuriously read as "inside".
BALL_EPS = wp.constant(wp.float32(1e-4))
TWO_PI = wp.constant(2.0 * wp.PI)


@wp.func
def face_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    n = wp.cross(b - a, c - a)
    length = wp.length(n)
    if length > 0.0:
        n = n / length
    return n


@wp.func
def compute_ball_center(
    v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, normal_sum: wp.vec3, radius: wp.float32
) -> wp.vec3:
    # Center of the radius-``radius`` ball touching all three points, on the ``normal_sum`` side.
    # Returns a sentinel of (inf, inf, inf) when the ball does not exist (points too far apart or
    # nearly collinear), which the caller treats as failure.
    fail = wp.vec3(wp.inf, wp.inf, wp.inf)
    c = wp.dot(v2 - v1, v2 - v1)
    b = wp.dot(v1 - v3, v1 - v3)
    a = wp.dot(v3 - v2, v3 - v2)
    alpha = a * (b + c - a)
    beta = b * (a + c - b)
    gamma = c * (a + b - c)
    abg = alpha + beta + gamma
    if abg < 1e-16:
        return fail
    circ_center = (alpha * v1 + beta * v2 + gamma * v3) / abg
    circ_radius2 = a * b * c
    sa = wp.sqrt(a)
    sb = wp.sqrt(b)
    sc = wp.sqrt(c)
    denom = (sa + sb + sc) * (sb + sc - sa) * (sc + sa - sb) * (sa + sb - sc)
    if denom <= 0.0:
        return fail
    circ_radius2 = circ_radius2 / denom
    height = radius * radius - circ_radius2
    if height < 0.0:
        return fail
    tr_norm = face_normal(v1, v2, v3)
    pt_norm = wp.normalize(normal_sum)
    if wp.dot(tr_norm, pt_norm) < 0.0:
        tr_norm = -tr_norm
    return circ_center + wp.sqrt(height) * tr_norm


@wp.func
def is_compatible(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, na: wp.vec3, nb: wp.vec3, nc: wp.vec3
) -> bool:
    # The triangle normal must agree (within tolerance) with all three oriented point normals.
    normal = face_normal(a, b, c)
    if wp.dot(normal, na) < -1e-16:
        normal = -normal
    return (
        wp.dot(normal, na) > -1e-16 and wp.dot(normal, nb) > -1e-16 and wp.dot(normal, nc) > -1e-16
    )


@wp.func
def ball_is_empty(
    grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    radius: wp.float32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
) -> bool:
    # True when no point other than the three defining ones lies strictly inside the ball.
    threshold = radius - BALL_EPS * radius
    query = wp.hash_grid_query(grid_id, center, radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if j != a and j != b and j != c:
            if wp.length(center - points[j]) < threshold:
                return False
    return True


@wp.kernel(enable_backward=False)
def seed_triangles(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    point_used: wp.array(dtype=wp.bool),
    grid_id: wp.uint64,
    radius: wp.float32,
    clustering: wp.float32,
    out_a: wp.array(dtype=wp.int32),
    out_b: wp.array(dtype=wp.int32),
    out_c: wp.array(dtype=wp.int32),
) -> None:
    p = int(wp.tid())
    out_a[p] = -1
    out_b[p] = -1
    out_c[p] = -1
    if point_used[p]:
        return

    # Gather the local orphan neighbourhood into scratch so it can be scanned as a double loop.
    # Only the lowest-index orphan in each neighbourhood seeds, so seed fronts start well separated
    # and do not collide into overlapping sheets before they can glue.
    nbr = wp.zeros(shape=MAX_SEED_NEIGHBORS, dtype=wp.int32)
    count = int(0)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
    query = wp.hash_grid_query(grid_id, points[p], 2.0 * radius)
    j = wp.int32(-1)
    while wp.hash_grid_query_next(query, j):
        if count >= MAX_SEED_NEIGHBORS:
            break
        if j != p and not point_used[j]:
            if j < p:
                return  # a lower-index orphan neighbour will seed this neighbourhood instead
            nbr[count] = j
            count += 1

    min_cluster = clustering * radius
    for i0 in range(count):
        a = nbr[i0]
        for i1 in range(i0 + 1, count):
            b = nbr[i1]
            if wp.length(points[a] - points[b]) < min_cluster:
                continue
            center = compute_ball_center(
                points[p], points[a], points[b], normals[p] + normals[a] + normals[b], radius
            )
            if center[0] == wp.inf:
                continue
            if not is_compatible(
                points[p], points[a], points[b], normals[p], normals[a], normals[b]
            ):
                continue
            if ball_is_empty(grid_id, points, center, radius, p, a, b):
                out_a[p] = p
                out_b[p] = a
                out_c[p] = b
                return


@wp.kernel(enable_backward=False)
def pivot_front_edges(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    front_src: wp.array(dtype=wp.int32),
    front_tgt: wp.array(dtype=wp.int32),
    front_opp: wp.array(dtype=wp.int32),
    point_available: wp.array(dtype=wp.bool),
    grid_id: wp.uint64,
    radius: wp.float32,
    clustering: wp.float32,
    crease_cos: wp.float32,
    interior_keys: wp.array(dtype=wp.uint64),
    key_base: wp.uint64,
    out_candidate: wp.array(dtype=wp.int32),
) -> None:
    e = int(wp.tid())
    out_candidate[e] = -1
    src = front_src[e]
    tgt = front_tgt[e]
    opp = front_opp[e]
    p_src = points[src]
    p_tgt = points[tgt]

    center = compute_ball_center(
        p_src, p_tgt, points[opp], normals[src] + normals[tgt] + normals[opp], radius
    )
    if center[0] == wp.inf:
        return

    mp = 0.5 * (p_src + p_tgt)
    axis = wp.normalize(p_tgt - p_src)
    a_dir = wp.normalize(center - mp)
    tri_norm = face_normal(p_src, p_tgt, points[opp])
    min_cluster = clustering * radius

    best_angle = TWO_PI
    best = int(-1)  # noqa: UP018, RUF046 — mutable Warp dynamic variable
    query = wp.hash_grid_query(grid_id, mp, 2.0 * radius)
    c = wp.int32(-1)
    while wp.hash_grid_query_next(query, c):
        if c == src or c == tgt or c == opp:
            continue
        if not point_available[c]:
            continue  # skip fully-interior (Inner) vertices: prevents overlapping sheets
        if wp.length(points[c] - p_src) < min_cluster or wp.length(points[c] - p_tgt) < min_cluster:
            continue
        new_center = compute_ball_center(
            p_src, p_tgt, points[c], normals[src] + normals[tgt] + normals[c], radius
        )
        if new_center[0] == wp.inf:
            continue
        b_dir = wp.normalize(new_center - mp)
        angle = wp.acos(wp.clamp(wp.dot(a_dir, b_dir), -1.0, 1.0))
        if wp.dot(wp.cross(a_dir, b_dir), axis) < 0.0:
            angle = TWO_PI - angle
        if angle >= best_angle:
            continue
        # Crease guard: reject if the new triangle folds too sharply against the current one.
        if crease_cos > -1.0:
            cand_norm = face_normal(p_src, p_tgt, points[c])
            if wp.abs(wp.dot(tri_norm, cand_norm)) < crease_cos:
                continue
        # Manifold guard: neither new edge may already be an interior (2-face) edge.
        if binary_search_sorted_contains(interior_keys, pack_edge_key(src, c, key_base)):
            continue
        if binary_search_sorted_contains(interior_keys, pack_edge_key(tgt, c, key_base)):
            continue
        if not is_compatible(p_src, p_tgt, points[c], normals[src], normals[tgt], normals[c]):
            continue
        if not ball_is_empty(grid_id, points, new_center, radius, src, tgt, c):
            continue
        best_angle = angle
        best = c
    out_candidate[e] = best


@wp.kernel(enable_backward=False)
def claim_triangle_vertices(
    tri_a: wp.array(dtype=wp.int32),
    tri_b: wp.array(dtype=wp.int32),
    tri_c: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.bool),
    out_owner: wp.array(dtype=wp.int32),
) -> None:
    # Priority claim (lowest index wins) on all three vertices of each proposed triangle.
    t = int(wp.tid())
    if not active[t]:
        return
    wp.atomic_min(out_owner, tri_a[t], t)
    wp.atomic_min(out_owner, tri_b[t], t)
    wp.atomic_min(out_owner, tri_c[t], t)


@wp.kernel(enable_backward=False)
def commit_triangles(
    tri_a: wp.array(dtype=wp.int32),
    tri_b: wp.array(dtype=wp.int32),
    tri_c: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.bool),
    owner: wp.array(dtype=wp.int32),
    max_faces: wp.int32,
    out_faces: wp.array(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
) -> None:
    # A proposed triangle commits only if it owns all three of its vertices this wave.
    t = int(wp.tid())
    if not active[t]:
        return
    a = tri_a[t]
    b = tri_b[t]
    c = tri_c[t]
    if owner[a] != t or owner[b] != t or owner[c] != t:
        return
    slot = wp.atomic_add(out_count, 0, 1)
    if slot >= max_faces:
        return
    out_faces[slot * 3 + 0] = a
    out_faces[slot * 3 + 1] = b
    out_faces[slot * 3 + 2] = c


@wp.kernel
def init_available_from_used(
    used: wp.array(dtype=wp.bool), out_available: wp.array(dtype=wp.bool)
) -> None:
    # Orphan (unused) points start available; front membership is added afterwards.
    i = int(wp.tid())
    out_available[i] = not used[i]


@wp.kernel
def mark_front_endpoints(
    front_src: wp.array(dtype=wp.int32),
    front_tgt: wp.array(dtype=wp.int32),
    out_available: wp.array(dtype=wp.bool),
) -> None:
    e = int(wp.tid())
    out_available[front_src[e]] = True
    out_available[front_tgt[e]] = True


@wp.kernel
def count_edge_faces(
    inverse: wp.array(dtype=wp.int32), out_count: wp.array(dtype=wp.int32)
) -> None:
    c = int(wp.tid())
    wp.atomic_add(out_count, inverse[c], 1)


@wp.kernel
def emit_front_edges(
    faces: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    edge_face_count: wp.array(dtype=wp.int32),
    out_src: wp.array(dtype=wp.int32),
    out_tgt: wp.array(dtype=wp.int32),
    out_opp: wp.array(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
) -> None:
    # One directed front edge per boundary (single-use) undirected edge, with its opposite vertex.
    corner = int(wp.tid())
    if edge_face_count[inverse[corner]] != 1:
        return
    f = corner / 3
    i = corner % 3
    a = faces[f * 3 + i]
    b = faces[f * 3 + (i + 1) % 3]
    opp = faces[f * 3 + (i + 2) % 3]
    slot = wp.atomic_add(out_count, 0, 1)
    out_src[slot] = a
    out_tgt[slot] = b
    out_opp[slot] = opp


@wp.kernel
def mark_interior_edge_keys(
    unique_edges: wp.array2d(dtype=wp.int32),
    edge_face_count: wp.array(dtype=wp.int32),
    key_base: wp.uint64,
    out_flag: wp.array(dtype=wp.int32),
    out_key: wp.array(dtype=wp.uint64),
) -> None:
    # Flag (1) and key each undirected edge already used by two faces (an interior edge).
    e = int(wp.tid())
    if edge_face_count[e] >= 2:
        out_flag[e] = 1
        out_key[e] = pack_edge_key(unique_edges[e, 0], unique_edges[e, 1], key_base)
    else:
        out_flag[e] = 0
        out_key[e] = wp.uint64(0)
