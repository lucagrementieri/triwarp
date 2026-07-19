import warp as wp

from triwarp.constants import TILE_1D, TOLERANCE_MERGE_CONSTANT, TOLERANCE_PLANAR_CONSTANT, TWO_PI
from triwarp.kernels import triangles as kernel_triangles


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
    max_hits_i = int(max_hits)
    while wp.mesh_query_aabb_next(query, face_idx) and c < max_hits_i:
        if write:
            out_indices[base + c] = face_idx
        c = c + 1
    return c


@wp.kernel
def query_mesh_aabb_bounds_count(
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
def query_mesh_aabb_bounds_neighbors(
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


@wp.func
def solid_angle(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.float32:
    """Signed solid angle subtended by triangle (a, b, c) at point p (``igl::solid_angle``)."""
    v0 = a - p
    v1 = b - p
    v2 = c - p
    vl0 = wp.length(v0)
    vl1 = wp.length(v1)
    vl2 = wp.length(v2)
    detf = (
        v0[0] * v1[1] * v2[2]
        + v1[0] * v2[1] * v0[2]
        + v2[0] * v0[1] * v1[2]
        - v2[0] * v1[1] * v0[2]
        - v1[0] * v0[1] * v2[2]
        - v0[0] * v2[1] * v1[2]
    )
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
    if not query.result:
        out_distance[tid] = max_dist
        return
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    dist = wp.length(p - closest)
    if dist <= TOLERANCE_MERGE_CONSTANT:
        out_distance[tid] = dist
    else:
        out_distance[tid] = query.sign * dist


@wp.kernel
def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q = int(wp.tid())
    p = query_points[q]
    w = wp.float32(0.0)
    n_f = int(n_faces)
    for f in range(n_f):
        face_indices = faces[f * 3 : (f + 1) * 3]
        i0 = int(face_indices[0])
        i1 = int(face_indices[1])
        i2 = int(face_indices[2])
        w = w + solid_angle(vertices[i0], vertices[i1], vertices[i2], p)
    out_winding[q] = w


@wp.kernel
def winding_number_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    query_points: wp.array[wp.vec3],
    out_winding: wp.array[wp.float32],
) -> None:
    q, tile_i, t = wp.tid()
    n_f = int(n_faces)
    face_offset = int(tile_i) * TILE_1D
    if face_offset >= n_f:
        return

    remaining = n_f - face_offset
    count = remaining
    if count > TILE_1D:
        count = TILE_1D

    p = query_points[int(q)]
    face_idx = face_offset + int(t)
    contrib = wp.float32(0.0)
    if int(t) < count:
        contrib = solid_angle_at_face(vertices, faces, face_idx, p)

    tile = wp.tile(contrib)
    tile_sum = wp.tile_sum(tile)
    if t == 0:
        wp.tile_atomic_add(out_winding, tile_sum, (int(q),))


@wp.kernel
def init_sphere_radii_finite(
    distances: wp.array[wp.float32],
    out_radii: wp.array[wp.float32],
    out_not_converged: wp.array[wp.bool],
    out_needs_support: wp.array[wp.bool],
) -> None:
    # Finite longest-ray hits initialise directly; escaped rays (inf distance) are deferred to
    # the tiled support-point passes below. Their slots default to the "no valid support"
    # outcome so an empty support subset needs no fix-up.
    tid = wp.tid()
    d = distances[tid]
    if not wp.isinf(d):
        out_radii[tid] = d * wp.float32(0.5)
        out_not_converged[tid] = True
        out_needs_support[tid] = False
    else:
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        out_needs_support[tid] = True


@wp.func
def pack_support_candidate(projection: wp.float32, index: wp.int32) -> wp.uint64:
    # Order-preserving float32 -> uint32 mapping (sign bit set for non-negatives, all bits
    # inverted for negatives) packed above the bit-inverted index, so a single atomic_max
    # selects the greatest projection with the LOWEST index as the tie-break. Zero never
    # occurs as a real packed value, so it doubles as the "no candidate" sentinel.
    bits = wp.cast(projection, wp.uint32)
    if bits & wp.uint32(0x80000000) != wp.uint32(0):
        key = ~bits
    else:
        key = bits | wp.uint32(0x80000000)
    return (wp.uint64(key) << wp.uint64(32)) | wp.uint64(~wp.uint32(index))


@wp.kernel
def support_argmax_tiled(
    mesh_vertices: wp.array[wp.vec3],
    n_vertices: wp.int32,
    stride_blocks: wp.int32,
    normals: wp.array[wp.vec3],
    support_indices: wp.array[wp.int32],
    out_packed: wp.array[wp.uint64],
) -> None:
    # Support point of the vertex cloud per deferred query: argmax of dot(v, n). Each lane
    # strides over the vertices (grid-stride keeps the block count bounded), reduces its own
    # running best into a packed (projection, index) key, and the block commits one atomic.
    q, block_j, t = wp.tid()
    normal = normals[support_indices[int(q)]]
    stride = int(stride_blocks) * TILE_1D
    idx = int(block_j) * TILE_1D + int(t)
    best = wp.float32(-wp.inf)
    best_index = wp.int32(0)
    while idx < int(n_vertices):
        projection = wp.dot(mesh_vertices[idx], normal)
        if projection > best or (projection == best and idx < best_index):
            best = projection
            best_index = idx
        idx += stride
    packed = wp.uint64(0)
    if not wp.isinf(best):
        packed = pack_support_candidate(best, best_index)
    tile_best = wp.tile_max(wp.tile(packed))
    if t == 0:
        wp.atomic_max(out_packed, int(q), tile_best[0])


@wp.kernel
def init_sphere_radii_support(
    mesh_vertices: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    support_indices: wp.array[wp.int32],
    packed_support: wp.array[wp.uint64],
    out_radii: wp.array[wp.float32],
    out_not_converged: wp.array[wp.bool],
) -> None:
    # Tail pass over the deferred subset: decode the support point and derive the
    # tangent-sphere radius, scattering it back into the full arrays.
    q = int(wp.tid())
    tid = support_indices[q]
    packed = packed_support[q]
    if packed == wp.uint64(0):
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        return

    p = points[tid]
    n = normals[tid]
    best = wp.int32(~wp.uint32(packed & wp.uint64(0xFFFFFFFF)))
    max_proj = wp.dot(mesh_vertices[best], n) - wp.dot(p, n)

    if max_proj < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        return

    diff = mesh_vertices[best] - p
    denom = wp.float32(2.0) * wp.dot(diff, n)
    if wp.abs(denom) < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = wp.inf
        out_not_converged[tid] = False
        return

    out_radii[tid] = wp.dot(diff, diff) / denom
    out_not_converged[tid] = True


@wp.func
def sphere_center(point: wp.vec3, normal: wp.vec3, radius: wp.float32) -> wp.vec3:
    if wp.isinf(radius) or wp.isnan(radius):
        return wp.vec3(wp.nan, wp.nan, wp.nan)
    return point + normal * radius


@wp.kernel
def step_sphere_shrink(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    n_points: wp.array[wp.vec3],
    n_dists: wp.array[wp.float32],
    centers: wp.array[wp.vec3],
    old_radii: wp.array[wp.float32],
    convergence_threshold: wp.float32,
    not_converged: wp.array[wp.bool],
    out_radii: wp.array[wp.float32],
    out_centers: wp.array[wp.vec3],
    out_not_converged: wp.array[wp.bool],
) -> None:
    # Every lane writes all three outputs (converged lanes pass their state through), so the
    # wrapper can ping-pong two preallocated buffer sets instead of cloning per iteration, and
    # extra launches on a fully converged state are harmless no-ops.
    tid = wp.tid()
    p = points[tid]
    center = centers[tid]
    if not not_converged[tid]:
        out_radii[tid] = old_radii[tid]
        out_centers[tid] = center
        out_not_converged[tid] = False
        return

    dist_to_start = wp.length(center - p)

    if wp.abs(n_dists[tid] - dist_to_start) < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = old_radii[tid]
        out_centers[tid] = center
        out_not_converged[tid] = False
        return

    diff = n_points[tid] - p
    denom = wp.float32(2.0) * wp.dot(diff, normals[tid])
    if wp.abs(denom) < TOLERANCE_PLANAR_CONSTANT:
        out_radii[tid] = old_radii[tid]
        out_centers[tid] = center
        out_not_converged[tid] = False
        return

    new_r = wp.dot(diff, diff) / denom
    out_radii[tid] = new_r
    out_centers[tid] = p + normals[tid] * new_r
    out_not_converged[tid] = old_radii[tid] - new_r >= convergence_threshold
