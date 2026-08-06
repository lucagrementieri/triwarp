import warp as wp

from triwarp.constants import TOLERANCE_PLANAR_CONSTANT
from triwarp.kernels.tangent_space import any_perpendicular

# Weighting of a ray inside the bundle. Passed as a warp-uniform kernel argument so both schemes
# share one compiled module (see AGENTS.md section 4 on runtime selection).
WEIGHT_COSINE = wp.constant(wp.int32(0))  # Lambert's cosine law: the physical ambient integral
WEIGHT_UNIFORM = wp.constant(wp.int32(1))  # every direction counts once (libigl's convention)


@wp.func
def bundle_direction(
    local: wp.vec3, normal: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3
) -> wp.vec3:
    # Lift a direction given in the ``+z``-axis frame of a Fibonacci lattice into the frame whose
    # ``z`` is ``normal``. Keeping the lattice in one canonical frame and rotating it per point is
    # what lets every point share a single direction array while staying low-discrepancy -- folding
    # a whole-sphere lattice onto the normal's side (libigl's approach) would not.
    return basis_x * local[0] + basis_y * local[1] + normal * local[2]


@wp.kernel
def obscurance(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    tau: wp.float32,
    weight_mode: wp.int32,
    max_t: wp.float32,
    offset: wp.float32,
    out_occlusion: wp.array[wp.float32],
) -> None:
    # Weighted fraction of an outward hemisphere bundle that is blocked.
    #
    # ``tau <= 0`` gives binary ambient occlusion (a hit at any distance blocks fully); a positive
    # ``tau`` gives Iones et al.'s volumetric obscurance, where an occluder at distance ``t``
    # contributes ``exp(-tau t)`` so a distant wall barely darkens the point. Binary occlusion is
    # the ``tau -> 0`` limit of that, since a ray that escapes contributes nothing either way.
    i = int(wp.tid())
    normal = wp.normalize(normals[i])
    basis_x = any_perpendicular(normal)
    basis_y = wp.cross(normal, basis_x)
    origin = points[i] + normal * offset

    n_rays = directions.shape[0]
    total_weight = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    total_blocked = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    for r in range(n_rays):
        local = directions[r]
        # A hemisphere lattice has ``local[2] == dot(direction, normal)`` by construction, so the
        # cosine weight is already there and needs no dot product.
        weight = float(1.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
        if weight_mode == WEIGHT_COSINE:
            weight = local[2]
        total_weight += weight
        query = wp.mesh_query_ray(
            mesh_id, origin, bundle_direction(local, normal, basis_x, basis_y), max_t
        )
        if query.result:
            if tau > 0.0:
                total_blocked += weight * wp.exp(-tau * query.t)
            else:
                total_blocked += weight

    if total_weight <= 0.0:
        out_occlusion[i] = 0.0
        return
    out_occlusion[i] = total_blocked / total_weight


@wp.kernel
def shape_diameter(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    max_t: wp.float32,
    offset: wp.float32,
    trim: wp.float32,
    scratch: wp.array2d[wp.float32],
    out_diameter: wp.array[wp.float32],
) -> None:
    # Shapira et al.'s shape diameter function: a cone of rays *into* the volume, and the
    # outlier-trimmed cosine-weighted mean of the distances they travel before leaving it.
    #
    # The trimming is what makes this robust rather than just a mean: near a concavity a handful of
    # rays escape through the opening or cross the whole model, and those few dominate an untrimmed
    # average. The pass structure is dictated by that -- distances go into ``scratch`` first,
    # because the second pass must revisit them against a mean and deviation the first pass had not
    # finished computing yet, and re-casting the rays instead would double the only expensive part.
    i = int(wp.tid())
    normal = wp.normalize(normals[i])
    basis_x = any_perpendicular(normal)
    basis_y = wp.cross(normal, basis_x)
    # Inward, so the bundle's axis is ``-normal`` and the origin steps *below* the surface.
    axis = -normal
    origin = points[i] + axis * offset

    n_rays = directions.shape[0]
    total = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    total_sq = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    hits = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    for r in range(n_rays):
        local = directions[r]
        query = wp.mesh_query_ray(
            mesh_id, origin, bundle_direction(local, axis, basis_x, basis_y), max_t
        )
        distance = wp.inf
        if query.result:
            distance = offset + query.t  # the ray started ``offset`` inside the surface
            total += distance
            total_sq += distance * distance
            hits += 1.0
        scratch[i, r] = distance

    if hits == 0.0:
        out_diameter[i] = wp.inf  # an open surface with nothing on the other side
        return
    mean = total / hits
    deviation = wp.sqrt(wp.max(0.0, total_sq / hits - mean * mean))

    kept = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    weighted = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    for r in range(n_rays):
        distance = scratch[i, r]
        if not wp.isinf(distance) and wp.abs(distance - mean) <= trim * deviation:
            weight = directions[r][2]  # cosine of the angle from the cone axis
            kept += weight
            weighted += weight * distance
    if kept <= 0.0:
        out_diameter[i] = mean  # every ray trimmed away (only possible at trim = 0)
        return
    out_diameter[i] = weighted / kept


@wp.func
def init_sphere_radii_finite(distance: wp.float32) -> tuple[wp.float32, wp.bool, wp.bool]:
    # Finite longest-ray hits initialise directly; escaped rays (inf distance) are deferred to
    # the tiled support-point passes below. Their slots default to the "no valid support"
    # outcome so an empty support subset needs no fix-up.
    if not wp.isinf(distance):
        return distance * wp.float32(0.5), True, False
    return wp.inf, False, True


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
    n_slices: wp.int32,
    normals: wp.array[wp.vec3],
    support_indices: wp.array[wp.int32],
    out_packed: wp.array[wp.uint64],
) -> None:
    # Support point of the vertex cloud per deferred query: argmax of dot(v, n). One thread per
    # (query, vertex slice) strides over the vertices, reduces its own running best into a packed
    # (projection, index) key, and commits one atomic; the packed key's ordering makes atomic_max
    # the global argmax with the lowest index as tie-break. Lane-free on purpose -- the block-wide
    # `wp.tile_max` this replaces reduced a single lane on Warp 1.15's CPU backend, where
    # `wp.launch_tiled` runs one lane per block.
    q, j = wp.tid()
    normal = normals[support_indices[int(q)]]
    best = wp.float32(-wp.inf)
    best_index = wp.int32(0)
    for idx in range(int(j), int(n_vertices), int(n_slices)):
        projection = wp.dot(mesh_vertices[idx], normal)
        if projection > best or (projection == best and idx < best_index):
            best = projection
            best_index = idx
    if not wp.isinf(best):
        wp.atomic_max(out_packed, int(q), pack_support_candidate(best, best_index))


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

    out_radii[tid] = wp.length_sq(diff) / denom
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

    new_r = wp.length_sq(diff) / denom
    out_radii[tid] = new_r
    out_centers[tid] = p + normals[tid] * new_r
    out_not_converged[tid] = old_radii[tid] - new_r >= convergence_threshold
