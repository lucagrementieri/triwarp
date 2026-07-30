import warp as wp

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
