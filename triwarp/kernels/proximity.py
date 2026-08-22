import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, TOLERANCE_MERGE_CONSTANT, TWO_PI
from triwarp.kernels import triangles as kernel_triangles
from triwarp.kernels.neighbors import MAX_SEARCH_ATTEMPTS, complete_radius, deepen_radius
from triwarp.kernels.polyline import closest_point_on_segment
from triwarp.kernels.predicates import barycentric_2d


@wp.func
def mesh_aabb_collect(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    max_hits: wp.int32,
    write: wp.bool,
    base: wp.int32,
    out_indices: wp.array[wp.int32],
) -> wp.int32:
    # Count (``write=False``) or emit at ``base`` (``write=True``) up to ``max_hits`` face hits.
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face_idx = wp.int32(0)
    c = wp.int32(0)
    max_hits_i = max_hits
    while wp.mesh_query_aabb_next(query, face_idx) and c < max_hits_i:
        if write:
            out_indices[base + c] = face_idx
        c = c + 1
    return c


@wp.kernel
def query_mesh_aabb_count(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    out_counts: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    out_counts[tid] = mesh_aabb_collect(
        mesh_id,
        query_lower[tid],
        query_upper[tid],
        max_hits,
        wp.bool(False),
        wp.int32(0),
        out_counts,
    )


@wp.kernel
def query_mesh_aabb_neighbors(
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    max_hits: wp.int32,
    offsets: wp.array[wp.int32],
    out_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    mesh_aabb_collect(
        mesh_id,
        query_lower[tid],
        query_upper[tid],
        max_hits,
        wp.bool(True),
        offsets[tid],
        out_indices,
    )


@wp.kernel
def closest_point_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_face: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, p, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        out_closest[tid] = closest
        out_distance[tid] = wp.length(p - closest)
        out_face[tid] = query.face
    else:
        out_closest[tid] = p
        out_distance[tid] = max_dist
        out_face[tid] = wp.int32(-1)


@wp.kernel
def closest_point_on_edges(
    vertices: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    queries: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    max_dist: wp.float32,
    initial_radius: wp.float32,
    min_bound: wp.vec3,
    max_bound: wp.vec3,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_edge: wp.array[wp.int32],
) -> None:
    # The wireframe counterpart of ``closest_point_on_mesh``, and the reason it is a hand-written
    # traversal rather than a ``wp.mesh_query_point_no_sign`` over degenerate triangles: Warp's mesh
    # BVH **rejects** a zero-area triangle outright. Measured on Warp 1.16, 200 segments as
    # ``(a, b, b)`` triangles and 64 queries: ``result`` is false for 64 of 64 on both devices, so
    # that shortcut answers nothing at all rather than answering approximately.
    #
    # Iterative deepening, sharing ``complete_radius`` / ``deepen_radius`` with the k-NN kernels
    # next door: a scan of the cube ``[q +/- r]`` enumerates every edge whose *closest point* is
    # within ``r`` (that point is then inside the cube, so the edge's AABB overlaps it), which is
    # what makes ``best <= r`` a proof of exactness rather than a heuristic. One
    # ``wp.bvh_query_aabb`` call site, for the shared-stack reason the k-NN kernel records.
    tid = wp.tid()
    q = queries[tid]

    r_hard = wp.min(max_dist, complete_radius(q, min_bound, max_bound))
    r = wp.min(initial_radius, r_hard)
    best_distance = FLOAT32_INF_CONSTANT
    best_edge = wp.int32(-1)
    best_point = q
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        if attempt == MAX_SEARCH_ATTEMPTS - 1:
            r = r_hard  # forced-complete final attempt: exact whatever the growth did
        query = wp.bvh_query_aabb(bvh_id, q - wp.vec3(r), q + wp.vec3(r), root=-1)
        edge_index = wp.int32(0)
        while wp.bvh_query_next(query, edge_index):
            candidate = closest_point_on_segment(
                vertices[edges[edge_index, 0]], vertices[edges[edge_index, 1]], q
            )
            d = wp.length(candidate - q)
            # Acceptance is ``d <= max_dist``; ``r`` bounds only the enumeration.
            if d < best_distance and d <= max_dist:
                best_distance = d
                best_edge = edge_index
                best_point = candidate
        if best_distance <= r:
            break  # every edge outside the cube is farther than the best: certified exact
        if r >= r_hard:
            break  # the scan was already complete, so the answer is final
        r = deepen_radius(best_distance, r, r_hard)

    if best_edge < 0:
        # Miss convention copied from ``closest_point_on_mesh`` above, so the two agree.
        out_closest[tid] = q
        out_distance[tid] = max_dist
        out_edge[tid] = wp.int32(-1)
    else:
        out_closest[tid] = best_point
        out_distance[tid] = best_distance
        out_edge[tid] = best_edge


@wp.kernel
def edge_bounds(
    vertices: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    # Per-edge AABB, the input a segment BVH is built from.
    e = wp.int32(wp.tid())
    a = vertices[edges[e, 0]]
    b = vertices[edges[e, 1]]
    out_lower[e] = wp.min(a, b)
    out_upper[e] = wp.max(a, b)


@wp.func
def solid_angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.float32:
    """Signed solid angle subtended by triangle (a, b, c) at point p (``igl::solid_angle``)."""
    v0 = a - p
    v1 = b - p
    v2 = c - p
    vl0 = wp.length(v0)
    vl1 = wp.length(v1)
    vl2 = wp.length(v2)
    # det([v0; v1; v2]) as the scalar triple product — cheaper than materializing the matrix.
    detf = wp.dot(v0, wp.cross(v1, v2))
    dp0 = wp.dot(v1, v2)
    dp1 = wp.dot(v2, v0)
    dp2 = wp.dot(v0, v1)
    denom = vl0 * vl1 * vl2 + dp0 * vl0 + dp1 * vl1 + dp2 * vl2
    return wp.atan2(detf, denom) / TWO_PI


@wp.func
def solid_angle_at_face(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: int, p: wp.vec3
) -> wp.float32:
    v0, v1, v2 = kernel_triangles.face_vertices(vertices, faces, wp.int32(f))
    return solid_angle(v0, v1, v2, p)


@wp.func
def point_strictly_inside_aabb(p: wp.vec3, mesh_min: wp.vec3, mesh_max: wp.vec3) -> bool:
    return (
        p[0] > mesh_min[0]
        and p[1] > mesh_min[1]
        and p[2] > mesh_min[2]
        and p[0] < mesh_max[0]
        and p[1] < mesh_max[1]
        and p[2] < mesh_max[2]
    )


@wp.kernel
def contains_points_sign_parity(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    mesh_min: wp.vec3,
    mesh_max: wp.vec3,
    out_contains: wp.array[wp.bool],
) -> None:
    tid = wp.tid()
    p = points[tid]

    if not point_strictly_inside_aabb(p, mesh_min, mesh_max):
        out_contains[tid] = False
        return

    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_contains[tid] = query.result and query.sign < wp.float32(0.0)


@wp.func
def signed_distance_from_query(
    mesh_id: wp.uint64,
    p: wp.vec3,
    result: wp.bool,
    face: wp.int32,
    u: wp.float32,
    v: wp.float32,
    sign: wp.float32,
    max_dist: wp.float32,
) -> wp.float32:
    # Closest-point evaluation and signing shared by both signed-distance kernels: a miss reports
    # the cutoff, a point inside the merge tolerance of the surface stays positive (the sign is not
    # meaningful there), and everything else takes the sign the query returned.
    #
    # The query struct itself cannot be the parameter -- ``mesh_query_point_sign_parity`` and
    # ``mesh_query_point_sign_winding_number`` return different types -- so the fields the tail
    # reads are passed individually. The two *heads* stay separate kernels on purpose: their
    # builtins take different parameters and the winding one carries a precondition its caller
    # chose, which a runtime selector would hide.
    if not result:
        return max_dist
    closest = wp.mesh_eval_position(mesh_id, face, u, v)
    dist = wp.length(p - closest)
    if dist <= TOLERANCE_MERGE_CONSTANT:
        return dist
    return sign * dist


@wp.kernel
def signed_distance_on_mesh(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    n_sample: wp.int32,
    perturbation_scale: wp.float32,
    out_distance: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_parity(mesh_id, p, max_dist, n_sample, perturbation_scale)
    out_distance[tid] = signed_distance_from_query(
        mesh_id, p, query.result, query.face, query.u, query.v, query.sign, max_dist
    )


@wp.kernel
def signed_distance_on_mesh_winding(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    max_dist: wp.float32,
    accuracy: wp.float32,
    winding_threshold: wp.float32,
    out_distance: wp.array[wp.float32],
) -> None:
    # Only the sign differs from ``signed_distance_on_mesh``; the closest-point and tolerance-band
    # handling is the shared ``signed_distance_from_query``. ``mesh_id`` MUST come from a
    # ``wp.Mesh`` built with ``support_winding_number=True`` -- otherwise this builtin silently
    # falls back to ray parity (warp/native/mesh.h:1348).
    tid = wp.tid()
    p = points[tid]
    query = wp.mesh_query_point_sign_winding_number(
        mesh_id, p, max_dist, accuracy, winding_threshold
    )
    out_distance[tid] = signed_distance_from_query(
        mesh_id, p, query.result, query.face, query.u, query.v, query.sign, max_dist
    )


@wp.kernel
def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q = wp.int32(wp.tid())
    p = query_points[q]
    w = wp.float32(0.0)
    n_f = n_faces
    for f in range(n_f):
        v0, v1, v2 = kernel_triangles.face_vertices(vertices, faces, f)
        w = w + solid_angle(v0, v1, v2, p)
    out_winding[q] = w


@wp.kernel
def winding_number_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    n_slices: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    # One thread per (query, face slice): each walks a strided slice of the face list and commits
    # one atomic. Deliberately lane-free -- a block-wide `wp.tile_sum(wp.tile(...))` would be the
    # natural reduction, but `wp.launch_tiled` runs exactly one lane per block on the CPU
    # backend through Warp 1.16, so a per-lane tile holds one face and under-counts there.
    q, j = wp.tid()
    p = query_points[q]
    total = wp.float32(0.0)
    for face_idx in range(j, n_faces, n_slices):
        total = total + solid_angle_at_face(vertices, faces, face_idx, p)
    wp.atomic_add(out_winding, q, total)


@wp.func
def lift_vec2(p: wp.vec2) -> wp.vec3:
    """Embed a 2D point in the ``z = 0`` plane."""
    return wp.vec3(p[0], p[1], wp.float32(0.0))


@wp.kernel
def face_containing_point_2d(
    mesh_id: wp.uint64,
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec2],
    search_radius: wp.float32,
    barycentric_epsilon: wp.float32,
    out_face: wp.array[wp.int32],
) -> None:
    # Point location in a 2D triangulation: a closest-point query against the same triangulation
    # lifted to ``z = 0`` picks the *candidate* face, and a barycentric sign test decides.
    #
    # The candidate is sufficient rather than merely plausible: a point inside some triangle is at
    # distance zero from it, so the closest triangle is a containing one whenever any exists.
    #
    # The two-stage form is not redundant. Accepting on the query radius alone was measured to
    # misclassify ~0.2% of random queries on a 3 979-triangle Delaunay mesh -- the closest-point
    # distance for an in-plane point is not exactly zero in float32, so a radius tight enough to
    # reject points just outside the triangulation also rejects points just inside it, and no radius
    # separates the two (73 / 28 / 6 interior points missed at 1e-7 / 1e-6 / 1e-5 of the bounding
    # diagonal, against 0 / 14 / 59 exterior points falsely accepted at 1e-5 / 1e-4 / 1e-3). The
    # barycentric test is a sign test on the query's own coordinates, ~1000x sharper, so the radius
    # only has to be loose enough to find the candidate.
    tid = wp.int32(wp.tid())
    p = points[tid]
    out_face[tid] = wp.int32(-1)
    query = wp.mesh_query_point_no_sign(mesh_id, lift_vec2(p), search_radius)
    if not query.result:
        return

    face = wp.int32(query.face)
    barycentric = barycentric_2d(
        vertices[faces[face * 3 + 0]],
        vertices[faces[face * 3 + 1]],
        vertices[faces[face * 3 + 2]],
        p,
    )
    if wp.min(barycentric[0], wp.min(barycentric[1], barycentric[2])) >= -barycentric_epsilon:
        out_face[tid] = face
