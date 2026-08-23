import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.array import pack_farthest_key, pack_nearest_key
from triwarp.kernels.intersection import point_plane_dot
from triwarp.kernels.reduce import outer_sum_chunk, tile_chunk


@wp.func
def point_plane_distance(
    point: wp.vec3, plane_normal: wp.vec3, plane_origin: wp.vec3
) -> wp.float32:
    # Signed perpendicular distance: the shared unnormalized plane dot divided by the normal
    # length, so a non-unit ``plane_normal`` behaves like trimesh's reference. This is the only
    # caller that needs the division, so the dot stays the primitive.
    return point_plane_dot(point, plane_normal, plane_origin) / wp.length(plane_normal)


@wp.func
def is_in_half_space(point: wp.vec3, plane_normal: wp.vec3, plane_origin: wp.vec3) -> wp.bool:
    # Strictly on the normal's side, so a point exactly on the plane is excluded. Only the sign of
    # the dot matters, which is why this reads the unnormalized primitive rather than
    # ``point_plane_distance``: a non-unit normal cannot change the answer and the division cannot
    # change the sign, but it can turn a large dot into an infinity.
    return point_plane_dot(point, plane_normal, plane_origin) > 0.0


@wp.func
def radial_sort_key(point: wp.vec3, origin: wp.vec3, axis0: wp.vec3, axis1: wp.vec3) -> wp.float32:
    v = point - origin
    # Negated angle: an ascending radix sort of these keys reproduces trimesh's
    # descending-angle order (`angles.argsort()[::-1]`).
    return -wp.atan2(wp.dot(v, axis0), wp.dot(v, axis1))


@wp.kernel
def centered_covariance(
    points: wp.array[wp.vec3], center: wp.array[wp.vec3], out_cov: wp.array[wp.mat33]
) -> None:
    # Scatter matrix C = sum_k outer(x_k - center, x_k - center). With a zero center this is
    # the uncentred Gram matrix G = sum_k outer(x_k, x_k).
    #
    # Stays in ``points`` rather than moving to ``reduce`` with the rest of the chunked-accumulate
    # family: the reusable half -- the tile skeleton and ``outer_sum_chunk`` -- is already in
    # ``reduce`` and imported from there, and what is left is a point-cloud statistic whose only
    # callers are ``points``' own ``gram_matrix`` / ``fit_line`` / ``fit_plane``. ``triwarp.reduce``
    # is axis-parametrized array reductions in NumPy's vocabulary; a mat33 of second moments is not
    # one of those (§11, the machinery half outranks the subject half).
    i, t = wp.tid()
    offset, remaining = tile_chunk(points.shape[0], i, TILE_1D)
    if remaining <= 0:
        return

    m = outer_sum_chunk(points, center[0], offset, remaining)

    if t == 0:
        wp.atomic_add(out_cov, 0, m)


@wp.kernel
def finalize_fit_line(m: wp.array[wp.mat33], out_axis: wp.array[wp.vec3]) -> None:
    # gram matrix of the (uncentred) points: M = sum_j outer(x_j, x_j)
    # the singular values / right singular vectors of M match the squared
    # singular values / right singular vectors of the (n, 3) point matrix.
    u, sigma, _v = wp.svd3(m[0])
    # axis = sum_i S_i * rsv_i, where S_i = sqrt(sigma_i) are the point singular values and the
    # right singular vectors are the columns of u -- i.e. exactly the matrix-vector product
    # u * (S_0, S_1, S_2). ``wp.sqrt`` is scalar-only, so the weight vector is built explicitly.
    axis = u * wp.vec3(wp.sqrt(sigma[0]), wp.sqrt(sigma[1]), wp.sqrt(sigma[2]))
    out_axis[0] = wp.normalize(axis)


@wp.kernel
def finalize_fit_plane(
    center: wp.array[wp.vec3],
    m: wp.array[wp.mat33],
    out_centroid: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
) -> None:
    # plane origin is the centroid of the point set; the covariance matrix of
    # the centred points was accumulated into m.
    u, _sigma, _v = wp.svd3(m[0])
    # normal is the singular vector with the smallest singular value
    # (svd3 returns singular values in descending order: last column of u).
    out_centroid[0] = center[0]
    out_normal[0] = wp.normalize(wp.vec3(u[0, 2], u[1, 2], u[2, 2]))


# Orientation modes for estimate_point_normals (mirror Open3D's orient methods).
ORIENT_CENTROID = wp.constant(wp.int32(0))  # outward from the cloud centroid (the default)
ORIENT_DIRECTION = wp.constant(wp.int32(1))  # align with a fixed direction
ORIENT_CAMERA = wp.constant(wp.int32(2))  # point toward a camera location


@wp.kernel
def finalize_principal_axes(
    center: wp.array[wp.vec3],
    m: wp.array[wp.mat33],
    out_rotation: wp.array[wp.mat33],
    out_eigenvalues: wp.array[wp.vec3],
    out_centroid: wp.array[wp.vec3],
) -> None:
    # ``wp.svd3`` returns singular values in descending order, so the columns of ``u`` are the
    # principal axes from widest to narrowest. The rows of the result are those axes, which is the
    # convention that makes ``rotation * p`` the coordinates of ``p`` in the principal frame.
    u, sigma, _v = wp.svd3(m[0])
    axis0 = wp.normalize(wp.vec3(u[0, 0], u[1, 0], u[2, 0]))
    axis1 = wp.normalize(wp.vec3(u[0, 1], u[1, 1], u[2, 1]))
    # Take the third axis from the cross product rather than from ``u``: that forces a proper
    # rotation (determinant +1) whatever sign convention the SVD chose, so the frame is always
    # right-handed and only the first two signs are free.
    axis2 = wp.cross(axis0, axis1)
    out_rotation[0] = wp.mat33(
        axis0[0], axis0[1], axis0[2], axis1[0], axis1[1], axis1[2], axis2[0], axis2[1], axis2[2]
    )
    out_eigenvalues[0] = sigma
    out_centroid[0] = center[0]


@wp.kernel
def estimate_point_normals(
    points: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    centroid: wp.array[wp.vec3],
    orient_mode: wp.int32,
    orient_reference: wp.vec3,
    out_normals: wp.array[wp.vec3],
) -> None:
    # Per-point normal = eigenvector of the smallest eigenvalue of the neighbourhood
    # covariance (the same choice Open3D's FastEigen3x3 makes).
    v = wp.int32(wp.tid())
    k = neighbor_idx.shape[1]

    # Local neighbourhood mean over the valid entries of the table. A self-query table
    # contains the point itself once, so it is naturally included (matching Open3D KNN).
    mean = wp.vec3(0.0, 0.0, 0.0)
    count = wp.float32(0.0)
    for i in range(k):
        nb = neighbor_idx[v, i]
        if nb >= 0:
            mean += points[nb]
            count += 1.0

    if count < 2.0:
        out_normals[v] = wp.vec3(0.0, 0.0, 1.0)  # too few neighbours (Open3D fallback)
        return
    mean = mean / count

    cov = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(k):
        nb = neighbor_idx[v, i]
        if nb >= 0:
            e = points[nb] - mean
            cov += wp.outer(e, e)

    if wp.ddot(cov, cov) <= 0.0:
        out_normals[v] = wp.vec3(0.0, 0.0, 1.0)  # coincident neighbours (zero covariance)
        return

    u, _sigma, _vt = wp.svd3(cov)
    normal = wp.normalize(wp.vec3(u[0, 2], u[1, 2], u[2, 2]))

    # Orientation: flip so the normal points along a per-point reference vector.
    ref = wp.vec3(0.0, 0.0, 0.0)
    if orient_mode == ORIENT_CENTROID:
        ref = points[v] - centroid[0]  # outward from the cloud centroid (star-shaped assumption)
    elif orient_mode == ORIENT_DIRECTION:
        ref = orient_reference
    else:
        ref = orient_reference - points[v]  # toward the camera location
    if wp.dot(normal, ref) < 0.0:
        normal = -normal
    out_normals[v] = normal


@wp.kernel
def neighbor_distance_moments(
    neighbor_distance: wp.array2d[wp.float32],
    out_mean: wp.array[wp.float32],
    out_rms: wp.array[wp.float32],
    out_count: wp.array[wp.int32],
) -> None:
    # First and second moments of each point's neighbour distances, over the *filled* slots only:
    # ``query_bvh_nearest`` leaves unused slots at ``inf`` (index -1), and a row can be short when
    # ``max_radius`` bites or the cloud is smaller than ``k``. An empty row reports zeros with a
    # zero count, which is how both callers detect it.
    #
    # The mean feeds Open3D's statistical criterion and the RMS is the LoOP "standard distance".
    i = wp.int32(wp.tid())
    k = neighbor_distance.shape[1]
    total = wp.float32(0.0)
    total_sq = wp.float32(0.0)
    count = wp.int32(0)
    for s in range(k):
        d = neighbor_distance[i, s]
        if not wp.isinf(d):
            total += d
            total_sq += d * d
            count += 1
    out_count[i] = count
    if count == 0:
        out_mean[i] = 0.0
        out_rms[i] = 0.0
        return
    inverse = 1.0 / wp.float32(count)
    out_mean[i] = total * inverse
    out_rms[i] = wp.sqrt(total_sq * inverse)


@wp.kernel
def local_outlier_factor(
    standard_distance: wp.array[wp.float32],
    neighbor_idx: wp.array2d[wp.int32],
    out_plof: wp.array[wp.float32],
) -> None:
    # LoOP's probabilistic local outlier factor: how far this point's standard distance sits above
    # the mean standard distance of its own neighbourhood. Zero when the neighbourhood is empty or
    # collapsed, so such a point never reads as an outlier on this term alone.
    #
    # The LoOP normalization factor lambda cancels here (it scales numerator and denominator
    # alike); it only enters through the cloud-wide nplof the caller divides by.
    i = wp.int32(wp.tid())
    k = neighbor_idx.shape[1]
    total = wp.float32(0.0)
    count = wp.int32(0)
    for s in range(k):
        j = neighbor_idx[i, s]
        if j >= 0:
            total += standard_distance[j]
            count += 1
    if count == 0 or total <= 0.0:
        out_plof[i] = 0.0
        return
    out_plof[i] = standard_distance[i] * wp.float32(count) / total - 1.0


@wp.func
def outlier_probability(plof: wp.float32, inverse_normalizer: wp.float32) -> wp.float32:
    # LoOP score: the error function of the normalized factor, clamped at zero so an
    # inlier-or-better point reads exactly 0 rather than a negative "probability".
    return wp.max(wp.float32(0.0), wp.erf(plof * inverse_normalizer))


@wp.func
def centered_square_if_counted(
    value: wp.float32, count: wp.int32, center: wp.float32
) -> wp.float32:
    # ``(value - center)^2`` for a counted row, 0 for an empty one: the masked variance term behind
    # the statistical threshold, so empty rows neither shift the mean nor inflate the deviation.
    if count == 0:
        return 0.0
    d = value - center
    return d * d


@wp.func
def is_statistical_outlier(
    mean_distance: wp.float32, count: wp.int32, threshold: wp.float32
) -> wp.bool:
    # Open3D's ``remove_statistical_outlier`` keeps a point when its mean neighbour distance is
    # strictly positive and strictly below the cloud threshold; everything else -- an empty
    # neighbourhood, a coincident one, or a far one -- is an outlier.
    return count == 0 or mean_distance <= 0.0 or mean_distance >= threshold


@wp.func
def is_finite_point(point: wp.vec3) -> wp.bool:
    # All three coordinates finite -- the row predicate behind ``point_finite_mask``. Any one NaN
    # or infinity condemns the point, which is what a downstream tree build or covariance fit
    # needs: a single non-finite coordinate poisons every reduction the point enters.
    return wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])


@wp.func
def zero_normalized_bits(value: wp.float32) -> wp.int32:
    # The coordinate's ``float32`` bit pattern, with ``-0.0`` folded onto ``+0.0``. ``wp.cast`` is a
    # bit reinterpretation, so it separates the two zeros -- and IEEE-754 equality does not, which
    # is the rule an exact-equality dedup has to reproduce. Every other value is its own bits.
    if value == 0.0:
        return wp.int32(0)
    return wp.cast(value, wp.int32)


@wp.func
def pack_xy_bits(point: wp.vec3) -> wp.int64:
    # Round one of the exact position key: the x and y bit patterns side by side in one int64.
    # Injective, because each half is exactly 32 bits wide -- which is the whole point, and the
    # difference from ``kernels.grouping.pack_vec3``, whose key *buckets* all three coordinates
    # into 21 bits apiece and so merges positions that merely agree to ~2.4e-4 relative.
    x_bits = wp.uint64(wp.uint32(zero_normalized_bits(point[0])))
    y_bits = wp.uint64(wp.uint32(zero_normalized_bits(point[1])))
    return wp.int64((x_bits << wp.uint64(32)) | y_bits)


@wp.func
def pack_class_z_bits(class_id: wp.int32, point: wp.vec3) -> wp.int64:
    # Round two: the (x, y) equivalence class from round one against the z bits. Injective for the
    # same reason -- ``class_id`` is an index into the round-one unique array, so it is below the
    # point count and fits the high 32 bits with room to spare.
    class_bits = wp.uint64(wp.uint32(class_id))
    z_bits = wp.uint64(wp.uint32(zero_normalized_bits(point[2])))
    return wp.int64((class_bits << wp.uint64(32)) | z_bits)


@wp.func
def unpack_farthest_index(key: wp.int64) -> wp.int32:
    return wp.int32(2147483647) - wp.int32(wp.uint32(wp.uint64(key) & wp.uint64(4294967295)))


@wp.kernel
def nearest_pair_keys(
    nearest_distances: wp.array2d[wp.float32], out_keys: wp.array[wp.int64]
) -> None:
    # One key per point, over column 1 of a ``k=2`` self-query table: column 0 is the point itself.
    # The thread index is the payload, which is what keeps this a kernel rather than a ``wp.map``.
    i = wp.int32(wp.tid())
    out_keys[i] = pack_nearest_key(nearest_distances[i, 1], i)


@wp.kernel
def seed_farthest_point(
    start: wp.int32, out_selected: wp.array[wp.int32], out_cursor: wp.array[wp.int32]
) -> None:
    # ``out_cursor`` is the loop's step counter, kept on the device so the iteration's two launches
    # take the same arguments every time and the body can be captured once and replayed.
    out_selected[0] = start
    out_cursor[0] = 0


@wp.kernel
def advance_farthest_point(
    points: wp.array[wp.vec3],
    selected: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    min_distance_sq: wp.array[wp.float32],
    best: wp.array[wp.int64],
) -> None:
    # One greedy iteration, fused: fold the point just selected into each point's running distance
    # to the chosen set, then have the same thread contribute its own updated value to the global
    # argmax. No cross-thread dependency to synchronize -- thread ``i`` reads and writes only
    # ``min_distance_sq[i]`` -- which is what lets the update and the reduction share a launch.
    #
    # The step comes from ``cursor`` rather than from a kernel argument, so every iteration issues
    # the identical launch and the wrapper can capture one iteration and replay it.
    i = wp.int32(wp.tid())
    chosen = points[selected[cursor[0]]]
    distance_sq = wp.length_sq(points[i] - chosen)
    if distance_sq < min_distance_sq[i]:
        min_distance_sq[i] = distance_sq
    wp.atomic_max(best, 0, pack_farthest_key(min_distance_sq[i], i))


@wp.kernel
def commit_farthest_point(
    best: wp.array[wp.int64], out_selected: wp.array[wp.int32], out_cursor: wp.array[wp.int32]
) -> None:
    # Decode the winning key into the next sample, advance the step and re-arm the accumulator, so
    # the loop needs no separate reset launch, no host readback and no per-iteration argument.
    step = out_cursor[0] + 1
    out_selected[step] = unpack_farthest_index(best[0])
    out_cursor[0] = step
    best[0] = wp.int64(-1)


@wp.func
def plane_basis(normal: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    # Kernel-scope mirror of ``triwarp.points.plane_basis``, for callers that hold the normal in
    # device memory and must not read it back to build the frame.
    unit_normal = wp.normalize(normal)
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(unit_normal[0]) > 0.9:
        axis = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(wp.cross(axis, unit_normal))
    v = wp.cross(unit_normal, u)
    return u, v
