import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.reduce import tile_chunk

# ---------------------------------------------------------------------------
# Packed Procrustes accumulator
#
# Every moment the fit needs lives in one ``float32`` buffer, so the whole thing costs a single
# allocation and a single memset instead of six of each — which is most of the host-side latency
# at the sizes this is called at (``icp`` calls it once per iteration, and the cost was flat in
# ``n`` from 8k points upward).
#
# Slots 1..24 are the *shifted* moments: sums of ``a - p`` and ``b - q`` where ``p = a[0]`` and
# ``q = b[0]``. Shifting by a point *of the cloud* is what lets one pass do the work of two. The
# textbook single-pass identity shifts by the origin, which makes the cancellation in
# ``E[x^2] - E[x]^2`` scale as ``(|centroid| / spread)^2`` — unbounded for a cloud far from the
# origin. Shifting by a sample point bounds it at ``(diameter / spread)^2``, a handful of bits.
# ---------------------------------------------------------------------------
ACC_W_SUM = wp.constant(0)  # sum w
ACC_A_SUM = wp.constant(1)  # vec3, slots 1..3:   sum w (a - p)
ACC_B_SUM = wp.constant(4)  # vec3, slots 4..6:   sum w (b - q)
ACC_A_SQ = wp.constant(7)  # sum w |a - p|^2
ACC_B_SQ = wp.constant(8)  # sum w |b - q|^2
ACC_COV = wp.constant(9)  # mat33, slots 9..17: sum_{w>0} outer(b - q, a - p), row-major
ACC_MASK_A = wp.constant(18)  # vec3, slots 18..20: sum_{w>0} (a - p)
ACC_MASK_B = wp.constant(21)  # vec3, slots 21..23: sum_{w>0} (b - q)
ACC_MASK_N = wp.constant(24)  # count of w > 0
ACC_COST = wp.constant(25)  # weighted mean squared residual
PROCRUSTES_ACC_SIZE = 26


@wp.func
def procrustes_shifts(
    a: wp.array[wp.vec3], b: wp.array[wp.vec3], use_translation: wp.bool
) -> tuple[wp.vec3, wp.vec3]:
    # Read the shift origins straight off the device (a broadcast load, no host round trip).
    # ``use_translation=False`` must shift by nothing at all, so the arithmetic stays bit-for-bit
    # the uncentered form the caller asked for.
    shift_a = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    shift_b = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation and a.shape[0] > 0:
        shift_a = a[0]
        shift_b = b[0]
    return shift_a, shift_b


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


@wp.kernel
def accumulate_procrustes_moments(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    use_translation: wp.bool,
    out_acc: wp.array[wp.float32],
) -> None:
    # Every moment the fit needs, in one pass over ``a`` and ``b``. The two-pass form this replaces
    # could not be fused because the covariance needs the centroid the first pass computes; the
    # shifted-moment identity in ``build_procrustes_matrix`` removes that dependency.
    #
    # Launched tiled, with *every* lane walking the whole chunk and lane 0 publishing the result.
    # That looks wasteful but is the faster shape here: all lanes read the same ``a[offset + k]``
    # on each step, so the loads broadcast out of one cache line, whereas one-thread-per-chunk
    # gives each lane its own 64-element run and the reads stop coalescing (measured ~10% slower
    # end-to-end on a 20k-point ICP). The redundant arithmetic is free — this is memory-bound.
    #
    # A zero-length ``weights`` means uniform weights, which is what ``icp`` passes: the
    # alternative is a ``wp.full(n, 1.0)`` allocation *and* fill on every iteration, for a value
    # the kernel can just assume.
    chunk, lane = wp.tid()
    offset, remaining = tile_chunk(a.shape[0], chunk, TILE_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, TILE_1D)
    uniform = weights.shape[0] == 0
    shift_a, shift_b = procrustes_shifts(a, b, use_translation)

    w_sum = wp.float32(0.0)
    a_sum = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    b_sum = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    a_sq = wp.float32(0.0)
    b_sq = wp.float32(0.0)
    cov = wp.mat33(wp.float32(0.0))
    mask_a = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    mask_b = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    mask_n = wp.float32(0.0)

    for k in range(count):
        index = offset + k
        w = wp.float32(1.0)
        if not uniform:
            w = weights[index]
        av = a[index] - shift_a
        bv = b[index] - shift_b
        w_sum += w
        a_sum += w * av
        b_sum += w * bv
        a_sq += w * wp.length_sq(av)
        b_sq += w * wp.length_sq(bv)
        # The covariance is weighted by *membership*, not magnitude — trimesh's convention, and
        # what ``test_procrustes_binary_weights`` pins.
        if w > wp.float32(0.0):
            cov += wp.outer(bv, av)
            mask_a += av
            mask_b += bv
            mask_n += wp.float32(1.0)

    if lane == 0:
        wp.atomic_add(out_acc, ACC_W_SUM, w_sum)
        wp.atomic_add(out_acc, ACC_A_SQ, a_sq)
        wp.atomic_add(out_acc, ACC_B_SQ, b_sq)
        wp.atomic_add(out_acc, ACC_MASK_N, mask_n)
        for c in range(3):
            wp.atomic_add(out_acc, ACC_A_SUM + c, a_sum[c])
            wp.atomic_add(out_acc, ACC_B_SUM + c, b_sum[c])
            wp.atomic_add(out_acc, ACC_MASK_A + c, mask_a[c])
            wp.atomic_add(out_acc, ACC_MASK_B + c, mask_b[c])
            for r in range(3):
                wp.atomic_add(out_acc, ACC_COV + c * 3 + r, cov[c, r])


@wp.func
def acc_vec3(acc: wp.array[wp.float32], base: wp.int32) -> wp.vec3:
    """Read a three-slot vector out of the packed accumulator."""
    return wp.vec3(acc[base], acc[base + 1], acc[base + 2])


@wp.kernel
def build_procrustes_matrix(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    acc: wp.array[wp.float32],
    use_reflection: wp.bool,
    use_translation: wp.bool,
    use_scale: wp.bool,
    out_matrix: wp.array[wp.mat44],
) -> None:
    ws = acc[ACC_W_SUM]
    shift_a, shift_b = procrustes_shifts(a, b, use_translation)

    # Centroids relative to the shift origins, then un-shifted back into world coordinates.
    a_rel = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    b_rel = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        a_rel = acc_vec3(acc, ACC_A_SUM) / ws
        b_rel = acc_vec3(acc, ACC_B_SUM) / ws
    acenter = shift_a + a_rel
    bcenter = shift_b + b_rel

    ascale = wp.float32(1.0)
    bscale = wp.float32(1.0)
    if use_scale:
        # Shifted second-moment identity: sum w |a - centroid|^2 / S_w = sum w |a - p|^2 / S_w
        # minus |centroid - p|^2.
        ascale = wp.sqrt(acc[ACC_A_SQ] / ws - wp.length_sq(a_rel))
        bscale = wp.sqrt(acc[ACC_B_SQ] / ws - wp.length_sq(b_rel))

    # Shifted cross-moment identity, over the membership-masked subset:
    # H = sum_m outer(b - bc, a - ac)
    #   = M - outer(Sm_b, ac') - outer(bc', Sm_a) + N_m outer(bc', ac')
    cov = wp.matrix_from_rows(
        acc_vec3(acc, ACC_COV + 0), acc_vec3(acc, ACC_COV + 3), acc_vec3(acc, ACC_COV + 6)
    )
    cov = (
        cov
        - wp.outer(acc_vec3(acc, ACC_MASK_B), a_rel)
        - wp.outer(b_rel, acc_vec3(acc, ACC_MASK_A))
        + acc[ACC_MASK_N] * wp.outer(b_rel, a_rel)
    )

    # Normalise cross-covariance by scale product
    inv_scales = wp.float32(1.0) / (bscale * ascale)
    target = cov * inv_scales

    U, sigma, V = wp.svd3(target)  # noqa: N806

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
    i = wp.int32(wp.tid())
    # wp.transform_point(mat44, vec3) is exactly ``(M * vec4(p, 1)).xyz``.
    out_points[i] = wp.transform_point(matrix[0], points[i])


@wp.kernel
def accumulate_cost(
    transformed: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    acc: wp.array[wp.float32],
) -> None:
    # Weighted mean squared residual, into the packed accumulator's cost slot. A zero-length
    # ``weights`` means uniform.
    i = wp.int32(wp.tid())
    w = wp.float32(1.0)
    if weights.shape[0] > 0:
        w = weights[i]
    wp.atomic_add(acc, ACC_COST, (w / acc[ACC_W_SUM]) * wp.length_sq(b[i] - transformed[i]))


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
    offset: wp.int32,
    remaining: wp.int32,
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
    offset, remaining = tile_chunk(source.shape[0], i, TILE_1D)
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
    """
    Solve the SPD 6x6 system ``a x = b`` via Cholesky (``a = L L^T``).

    Hand-written rather than routed through ``wp.dense_chol`` / ``wp.dense_subs`` /
    ``wp.dense_solve``, which look like exactly this function and are not usable here. Three
    independent reasons, any one sufficient: they are ``hidden=True`` with ``doc="WIP"``, so they
    are undocumented and unannounced in the installed Warp; they take ``wp.array[wp.float32]``
    rather than a register value, so adopting them means per-thread global scratch for ``A``,
    ``L``, ``b`` and ``x``, which is the shape measured at a 2x loss against registers; and they
    are ``float32``-only, so ``linalg``'s ``float64`` systems could not use them either.

    The two zero ``wp.spatial_vector``s below are written out longhand although the
    ``wp.spatial_matrix`` one line up is the broadcast ``wp.spatial_matrix(wp.float32(0.0))``. That
    asymmetry is Warp's, not a style slip: a one-argument ``wp.spatial_vector(x)`` binds ``x`` to
    the *templated* ``vec_t``'s ``dtype`` parameter and fails to parse -- *"Remove the extraneous
    ``dtype`` parameter when calling the templated version of ``wp.vec_t()``"*, still true in Warp
    1.16. There is no broadcast-fill spelling for it, so do not collapse these.
    """
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
