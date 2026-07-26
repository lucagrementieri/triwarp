import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.reduce import cross_outer_sum_tile, sum1d_tile, weighted_sum_vec3_tile


@wp.func
def make_affine44(rotation: wp.mat33, translation: wp.vec3) -> wp.mat44:
    """Pack a 3x3 linear part and a translation into a homogeneous transform."""
    return wp.mat44(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        translation[0],
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        translation[1],
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        translation[2],
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(1.0),
    )


@wp.func
def weighted_centered_dot_tile(
    values: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    center: wp.vec3,
    w_sum: wp.float32,
    offset: int,
    remaining: int,
) -> wp.float32:
    count = wp.min(remaining, TILE_1D)
    result = wp.float32(0.0)
    for k in range(count):
        v = values[offset + k] - center
        result += (weights[offset + k] / w_sum) * wp.length_sq(v)
    return result


@wp.kernel
def accumulate_weighted_sums(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    out_w_sum: wp.array[wp.float32],
    out_a_sum: wp.array[wp.vec3],
    out_b_sum: wp.array[wp.vec3],
) -> None:
    i, t = wp.tid()
    n = weights.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    tile_w = sum1d_tile(weights, offset, remaining)
    tile_a = weighted_sum_vec3_tile(a, weights, offset, remaining)
    tile_b = weighted_sum_vec3_tile(b, weights, offset, remaining)

    if t == 0:
        wp.atomic_add(out_w_sum, 0, tile_w)
        wp.atomic_add(out_a_sum, 0, tile_a)
        wp.atomic_add(out_b_sum, 0, tile_b)


@wp.kernel
def accumulate_scale_and_cov(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    w_sum: wp.array[wp.float32],
    a_center_raw: wp.array[wp.vec3],
    b_center_raw: wp.array[wp.vec3],
    use_translation: bool,
    out_a_scale_sq: wp.array[wp.float32],
    out_b_scale_sq: wp.array[wp.float32],
    out_cov: wp.array[wp.mat33],
) -> None:
    i, t = wp.tid()
    n = weights.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    ws = w_sum[0]
    acenter = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    bcenter = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        acenter = a_center_raw[0] / ws
        bcenter = b_center_raw[0] / ws

    tile_a_sq = weighted_centered_dot_tile(a, weights, acenter, ws, offset, remaining)
    tile_b_sq = weighted_centered_dot_tile(b, weights, bcenter, ws, offset, remaining)
    # Cross-covariance: H[row, col] = sum_k mask_k * bc[k, row] * ac[k, col]
    tile_cov = cross_outer_sum_tile(a, b, weights, acenter, bcenter, offset, remaining)

    if t == 0:
        wp.atomic_add(out_a_scale_sq, 0, tile_a_sq)
        wp.atomic_add(out_b_scale_sq, 0, tile_b_sq)
        wp.atomic_add(out_cov, 0, tile_cov)


@wp.kernel
def build_procrustes_matrix(
    w_sum: wp.array[wp.float32],
    a_center_raw: wp.array[wp.vec3],
    b_center_raw: wp.array[wp.vec3],
    a_scale_sq: wp.array[wp.float32],
    b_scale_sq: wp.array[wp.float32],
    cov: wp.array[wp.mat33],
    use_reflection: bool,
    use_translation: bool,
    use_scale: bool,
    out_matrix: wp.array[wp.mat44],
) -> None:
    ws = w_sum[0]

    acenter = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    bcenter = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        acenter = a_center_raw[0] / ws
        bcenter = b_center_raw[0] / ws

    ascale = wp.float32(1.0)
    bscale = wp.float32(1.0)
    if use_scale:
        ascale = wp.sqrt(a_scale_sq[0])
        bscale = wp.sqrt(b_scale_sq[0])

    # Normalise cross-covariance by scale product
    inv_scales = wp.float32(1.0) / (bscale * ascale)
    target = cov[0] * inv_scales

    U = wp.mat33(wp.float32(0.0))  # noqa: N806
    sigma = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    V = wp.mat33(wp.float32(0.0))  # noqa: N806
    wp.svd3(target, U, sigma, V)

    Vt = wp.transpose(V)  # noqa: N806

    # wp.svd3 may return negative singular values; absorb their signs into a
    # diagonal correction matrix so R = U @ D @ V^T matches the numpy convention
    # (all-positive sigma) and correctly handles reflective optimal solutions.
    # wp.sign is -1 for negative components and +1 otherwise (including at exactly 0).
    d = wp.sign(sigma)

    if not use_reflection:
        # Ensure det(R) = 1 by flipping the last correction factor when needed
        R_test = U * wp.diag(d) * Vt  # noqa: N806
        if wp.determinant(R_test) < wp.float32(0.0):
            d = wp.vec3(d[0], d[1], -d[2])

    D = wp.diag(d)  # noqa: N806
    R = U * D * Vt  # noqa: N806

    s = wp.float32(1.0)
    if use_scale:
        s = bscale / ascale

    sR = s * R  # noqa: N806

    t = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        t = bcenter - sR * acenter

    out_matrix[0] = make_affine44(sR, t)


@wp.kernel
def apply_transform_mat44(
    points: wp.array[wp.vec3], matrix: wp.array[wp.mat44], out_points: wp.array[wp.vec3]
) -> None:
    i = int(wp.tid())
    # wp.transform_point(mat44, vec3) is exactly ``(M * vec4(p, 1)).xyz``.
    out_points[i] = wp.transform_point(matrix[0], points[i])


@wp.kernel
def accumulate_cost(
    transformed: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    w_sum: wp.array[wp.float32],
    out_cost: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    w_norm = weights[i] / w_sum[0]
    diff = b[i] - transformed[i]
    wp.atomic_add(out_cost, 0, w_norm * wp.length_sq(diff))


# --- Iterative closest point (ICP) -----------------------------------------


@wp.func
def distance_threshold_weight(
    distance: wp.float32, triangle_id: wp.int32, max_distance: wp.float32
) -> wp.float32:
    """Binary correspondence mask: 1 for a valid, in-range hit, 0 otherwise."""
    return wp.where(
        triangle_id >= wp.int32(0) and distance <= max_distance, wp.float32(1.0), wp.float32(0.0)
    )


@wp.func
def point_to_plane_residual(current: wp.vec3, closest: wp.vec3, normal: wp.vec3) -> wp.float32:
    """Signed point-to-plane residual ``dot(current - closest, normal)``."""
    return wp.dot(current - closest, normal)


@wp.func
def residual_valid(
    triangle_id: wp.int32, distance: wp.float32, max_distance: wp.float32
) -> wp.bool:
    """Whether a correspondence is a valid, in-range hit."""
    return triangle_id >= wp.int32(0) and distance <= max_distance


@wp.func
def abs_deviation(value: wp.float32, center: wp.float32) -> wp.float32:
    """Absolute deviation ``|value - center|`` (median-absolute-deviation building block)."""
    return wp.abs(value - center)


@wp.func
def robust_weight(residual: wp.float32, scale: wp.float32, kind: wp.int32) -> wp.float32:
    """
    IRLS weight for a residual under an M-estimator loss.

    ``kind``: 0 = none (unit weight), 1 = Huber (``k = scale``),
    2 = Tukey biweight (``c = scale``). A non-positive ``scale`` yields unit weight.
    """
    if kind == wp.int32(0) or scale <= wp.float32(0.0):
        return wp.float32(1.0)
    r = wp.abs(residual)
    if kind == wp.int32(1):
        # Huber
        if r <= scale:
            return wp.float32(1.0)
        return scale / r
    # Tukey biweight
    if r >= scale:
        return wp.float32(0.0)
    u = r / scale
    t = wp.float32(1.0) - u * u
    return t * t


@wp.func
def point_to_plane_tile(
    source: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: wp.float32,
    robust_kind: wp.int32,
    robust_scale: wp.float32,
    offset: int,
    remaining: int,
) -> tuple[wp.spatial_matrix, wp.spatial_vector, wp.float32]:
    count = wp.min(remaining, TILE_1D)
    jtj = wp.spatial_matrix(wp.float32(0.0))
    jtr = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    cost = wp.float32(0.0)
    for k in range(count):
        idx = offset + k
        if triangle_id[idx] < 0 or distance[idx] > max_distance:
            continue
        nrm = wp.normalize(normals[idx])
        x = source[idx]
        d = x - target[idx]
        r = wp.dot(d, nrm)
        w = robust_weight(r, robust_scale, robust_kind)
        # Jacobian of the point-to-plane residual: [x x n ; n]
        j = wp.spatial_vector(wp.cross(x, nrm), nrm)
        jtj += w * wp.outer(j, j)
        jtr += (w * r) * j
        cost += w * r * r
    return jtj, jtr, cost


@wp.kernel
def accumulate_point_to_plane(
    source: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: wp.float32,
    robust_kind: wp.int32,
    robust_scale: wp.float32,
    out_jtj: wp.array[wp.spatial_matrix],
    out_jtr: wp.array[wp.spatial_vector],
    out_cost: wp.array[wp.float32],
) -> None:
    i, t = wp.tid()
    n = source.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    tile_jtj, tile_jtr, tile_cost = point_to_plane_tile(
        source,
        target,
        normals,
        distance,
        triangle_id,
        max_distance,
        robust_kind,
        robust_scale,
        offset,
        remaining,
    )

    if t == 0:
        wp.atomic_add(out_jtj, 0, tile_jtj)
        wp.atomic_add(out_jtr, 0, tile_jtr)
        wp.atomic_add(out_cost, 0, tile_cost)


@wp.func
def solve_spd6(a: wp.spatial_matrix, b: wp.spatial_vector) -> wp.spatial_vector:
    """Solve the SPD 6x6 system ``a x = b`` via Cholesky (``a = L L^T``)."""
    lower = wp.spatial_matrix(wp.float32(0.0))
    for j in range(6):
        s = a[j, j]
        for k in range(j):
            s -= lower[j, k] * lower[j, k]
        if s < wp.float32(1e-20):
            s = wp.float32(1e-20)
        ljj = wp.sqrt(s)
        lower[j, j] = ljj
        for i in range(j + 1, 6):
            v = a[i, j]
            for k in range(j):
                v -= lower[i, k] * lower[j, k]
            lower[i, j] = v / ljj
    # Forward substitution: L y = b
    y = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for i in range(6):
        v = b[i]
        for k in range(i):
            v -= lower[i, k] * y[k]
        y[i] = v / lower[i, i]
    # Back substitution: L^T x = y
    x = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for ii in range(6):
        i = 5 - ii
        v = y[i]
        for k in range(i + 1, 6):
            v -= lower[k, i] * x[k]
        x[i] = v / lower[i, i]
    return x


@wp.kernel
def solve_point_to_plane(
    jtj: wp.array[wp.spatial_matrix],
    jtr: wp.array[wp.spatial_vector],
    damping: wp.float32,
    out_matrix: wp.array[wp.mat44],
) -> None:
    """Solve the linearized point-to-plane system and build the incremental transform."""
    a = jtj[0]
    b = jtr[0]

    # Levenberg-style diagonal damping, scaled by the mean diagonal magnitude,
    # keeps the system positive-definite for planar / rank-deficient targets.
    reg = damping * wp.trace(a) / wp.float32(6.0) + wp.float32(1e-12)
    for i in range(6):
        a[i, i] += reg

    delta = -solve_spd6(a, b)
    omega = wp.spatial_top(delta)
    tvec = wp.spatial_bottom(delta)

    angle = wp.length(omega)
    rot = wp.identity(n=3, dtype=wp.float32)
    if angle > wp.float32(1e-12):
        rot = wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(omega), angle))

    out_matrix[0] = make_affine44(rot, tvec)
