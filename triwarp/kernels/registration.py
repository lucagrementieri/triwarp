import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.reduce import cross_outer_sum_tile, sum1d_tile, weighted_sum_vec3_tile


@wp.func
def weighted_centered_dot_tile(
    values: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    center: wp.vec3,
    w_sum: wp.float32,
    offset: int,
    remaining: int,
) -> wp.float32:
    count = remaining
    if count > TILE_1D:
        count = TILE_1D
    result = wp.float32(0.0)
    for k in range(count):
        v = values[offset + k] - center
        result += (weights[offset + k] / w_sum) * wp.dot(v, v)
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

    U = wp.mat33(wp.float32(0.0))
    sigma = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    V = wp.mat33(wp.float32(0.0))
    wp.svd3(target, U, sigma, V)

    Vt = wp.transpose(V)

    # wp.svd3 may return negative singular values; absorb their signs into a
    # diagonal correction matrix so R = U @ D @ V^T matches the numpy convention
    # (all-positive sigma) and correctly handles reflective optimal solutions.
    d0 = wp.float32(1.0)
    d1 = wp.float32(1.0)
    d2 = wp.float32(1.0)
    if sigma[0] < wp.float32(0.0):
        d0 = wp.float32(-1.0)
    if sigma[1] < wp.float32(0.0):
        d1 = wp.float32(-1.0)
    if sigma[2] < wp.float32(0.0):
        d2 = wp.float32(-1.0)

    if not use_reflection:
        # Ensure det(R) = 1 by flipping the last correction factor when needed
        R_test = U * wp.diag(wp.vec3(d0, d1, d2)) * Vt
        if wp.determinant(R_test) < wp.float32(0.0):
            d2 = -d2

    D = wp.diag(wp.vec3(d0, d1, d2))
    R = U * D * Vt

    s = wp.float32(1.0)
    if use_scale:
        s = bscale / ascale

    sR = s * R

    t = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        t = bcenter - sR * acenter

    out_matrix[0] = wp.mat44(
        sR[0, 0],
        sR[0, 1],
        sR[0, 2],
        t[0],
        sR[1, 0],
        sR[1, 1],
        sR[1, 2],
        t[1],
        sR[2, 0],
        sR[2, 1],
        sR[2, 2],
        t[2],
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(1.0),
    )


@wp.kernel
def apply_transform_mat44(
    points: wp.array[wp.vec3], matrix: wp.array[wp.mat44], out_points: wp.array[wp.vec3]
) -> None:
    i = int(wp.tid())
    M = matrix[0]
    p = points[i]
    r = M * wp.vec4(p[0], p[1], p[2], wp.float32(1.0))
    out_points[i] = wp.vec3(r[0], r[1], r[2])


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
    wp.atomic_add(out_cost, 0, w_norm * wp.dot(diff, diff))
