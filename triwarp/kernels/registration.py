import warp as wp

from triwarp.kernels.array import LOOP_CONDITION, LOOP_ROUND, LOOP_STATE_SIZE
from triwarp.kernels.predicates import normalize_or_zero
from triwarp.kernels.proximity import write_closest_point_query
from triwarp.kernels.reduce import ITEMS_PER_BLOCK_1D, block_chunk_1d, block_sum, commit_block_sum
from triwarp.kernels.transform import transform_point_mat44

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
# The moment slots, ``ACC_W_SUM`` through ``ACC_MASK_N``: what one block of
# ``accumulate_procrustes_moments`` folds. A plain constant, since a vector length must be one.
PROCRUSTES_MOMENT_SLOTS = wp.constant(25)
ACC_COST = wp.constant(25)  # weighted mean squared residual
PROCRUSTES_ACC_SIZE = 26

# Point-to-plane accumulator's two scalars, in one buffer for the same reason the moments above
# share one: ``icp_point_to_plane`` reads both back every iteration, ``accumulate_point_to_plane``
# writes both, and one buffer is one host sync instead of two.
ICP_COST = wp.constant(0)  # sum robust_loss(r); see ``robust_weight_and_loss``
ICP_WEIGHT_SUM = wp.constant(1)  # sum w
ICP_SCALAR_ACC_SIZE = 2
# ``icp_point_to_plane``'s device-loop state: the shared round and condition slots of
# ``kernels/array.py``, and a third recording that the loop stopped because nothing carried weight.
ICP_LOOP_WEIGHTLESS = wp.constant(wp.int32(LOOP_STATE_SIZE))
ICP_LOOP_STATE_SIZE = LOOP_STATE_SIZE + 1


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
def sample_weight(weights: wp.array[wp.float32], i: wp.int32) -> wp.float32:
    # Sample ``i``'s weight, a zero-length ``weights`` meaning uniform -- which is what ``icp``
    # passes: the alternative is a ``wp.full(n, 1.0)`` allocation *and* fill on every iteration,
    # for a value the kernel can just assume.
    w = wp.float32(1.0)
    if weights.shape[0] > 0:
        w = weights[i]
    return w


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
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` points, lanes striding that block's own chunk, and one packed
    # 25-slot tree reduction committing **one** atomic set per block.
    #
    # **The 64-fold redundant lane arithmetic is not the cost, and never was**: all lanes read the
    # same ``a[offset + k]`` and the loads broadcast out of one cache line, so giving each lane
    # exactly one element instead is worth a few percent at every size. What costs is the *atomic
    # contention* -- 25 hot addresses taking one add per block -- so the lever is the fold width.
    # Folding ``TILES_PER_BLOCK_1D`` tiles into each block cuts the block count sixteenfold: flat at
    # small ``n``, an order of magnitude at a million points.
    #
    # The tree is also *more* accurate than the serialized form, as in ``accumulate_cost``.
    #
    # No ``prefers_tiled_reduction`` branch: the lanes partition a chunk the block already owns and
    # stride by ``wp.block_dim()``, which reads 1 on CPU, so lane 0 walks the whole chunk and the
    # one-element tiles hold its true totals. The CPU clock is flat and the accuracy win holds
    # there too.
    #
    # A zero-length ``weights`` means uniform weights (``sample_weight``).
    chunk, lane = wp.tid()
    offset, count = block_chunk_1d(a.shape[0], chunk)
    if count <= 0:
        return
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

    for k in range(lane, count, wp.block_dim()):
        index = offset + k
        w = sample_weight(weights, index)
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

    # Packed into the accumulator's own slot layout and folded by **one** block reduction, which
    # is block-collective, so it runs outside the ``lane == 0`` guard and only the commit is
    # guarded. The default constructor, not a zero-fill: every slot is written just below.
    packed = wp.vector(length=PROCRUSTES_MOMENT_SLOTS, dtype=wp.float32)
    packed[ACC_W_SUM] = w_sum
    packed[ACC_A_SQ] = a_sq
    packed[ACC_B_SQ] = b_sq
    packed[ACC_MASK_N] = mask_n
    for c in range(3):
        packed[ACC_A_SUM + c] = a_sum[c]
        packed[ACC_B_SUM + c] = b_sum[c]
        packed[ACC_MASK_A + c] = mask_a[c]
        packed[ACC_MASK_B + c] = mask_b[c]
        for r in range(3):
            packed[ACC_COV + c * 3 + r] = cov[c, r]
    commit_block_sum(lane, packed, out_acc, 0)


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
        # minus |centroid - p|^2. Mathematically non-negative, but computed as a difference of two
        # independently tile-reduced sums (CLAUDE.md section 12.4, non-associative float32
        # accumulation), so a near-degenerate cloud (tightly clustered, or exactly duplicated
        # points) can land it at a tiny negative value from cancellation alone -- floored the same
        # way ``solve_spd6`` below floors its own Cholesky pivot for the identical reason, rather
        # than feeding ``wp.sqrt`` a negative argument and propagating NaN through the SVD.
        ascale = wp.sqrt(wp.max(acc[ACC_A_SQ] / ws - wp.length_sq(a_rel), wp.float32(1e-20)))
        bscale = wp.sqrt(wp.max(acc[ACC_B_SQ] / ws - wp.length_sq(b_rel), wp.float32(1e-20)))

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
    R = U * wp.diag(d) * Vt  # noqa: N806

    if not use_reflection:
        # Ensure det(R) = 1 by flipping the last correction factor when needed. Rebuilding ``R``
        # only on the branch that actually flips ``d`` -- the common, no-flip case reuses the
        # matrix just computed above instead of recomputing the identical product from the same,
        # unchanged ``d``.
        if wp.determinant(R) < wp.float32(0.0):
            d = wp.vec3(d[0], d[1], -d[2])
            R = U * wp.diag(d) * Vt  # noqa: N806

    s = wp.float32(1.0)
    if use_scale:
        s = bscale / ascale

    sR = s * R  # noqa: N806

    t = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if use_translation:
        t = bcenter - sR * acenter

    out_matrix[0] = make_affine44(sR, t)


@wp.kernel
def transform_and_accumulate_cost(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    matrix: wp.array[wp.mat44],
    acc: wp.array[wp.float32],
    out_transformed: wp.array[wp.vec3],
) -> None:
    # Apply the fitted transform to ``a``, publish the moved points, and reduce the weighted mean
    # squared residual against ``b`` into the packed accumulator's cost slot -- one pass, one
    # launch. A zero-length ``weights`` means uniform (``sample_weight``).
    #
    # The reduction is the lane-strided single-slot form of CLAUDE.md section 13.2, striding by
    # ``wp.block_dim()`` so it needs no ``prefers_tiled_reduction`` branch.
    #
    # **The transform only became worth fusing in once the reduction was flat.** As one
    # ``wp.atomic_add`` per thread to a constant slot the reduction dominated the pair and the
    # fusion was declined on that; at a launch floor the pair is two floors and removing one is most
    # of it (CLAUDE.md section 13.2's "re-read the declines either side"). Output is
    # **bit-identical**: the fusion only removes a round trip of ``out_transformed`` through global
    # memory, each lane keeping the point it just transformed in a register.
    # ``apply_transform_mat44`` keeps its three other callers; the shared arithmetic is its own
    # ``transform_point_mat44`` helper, called here rather than copied.
    chunk, lane = wp.tid()
    offset, count = block_chunk_1d(a.shape[0], chunk)
    if count <= 0:
        return
    transform = matrix[0]
    local = wp.float32(0.0)
    for k in range(lane, count, wp.block_dim()):
        i = offset + k
        moved = transform_point_mat44(a[i], transform)
        out_transformed[i] = moved
        w = sample_weight(weights, i)
        local += w * wp.length_sq(b[i] - moved)
    total = block_sum(local)
    if lane == 0:
        wp.atomic_add(acc, ACC_COST, total / acc[ACC_W_SUM])


# --- Iterative closest point (ICP) -----------------------------------------


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
def correspondence_normal(source: wp.array[wp.vec3], index: wp.int32) -> wp.vec3:
    # The target normal a correspondence points at, or the zero vector where there is none.
    # ``source`` is the face-normal table for a mesh target and the per-vertex one for a cloud,
    # which is why the two ICP passes below differ only in what they hand it.
    if index >= 0:
        return source[index]
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def distance_threshold_weight(
    distance: wp.float32, triangle_id: wp.int32, max_distance: wp.float32
) -> wp.float32:
    """Binary correspondence mask: 1 for a valid, in-range hit, 0 otherwise."""
    # Same predicate as ``residual_valid`` -- called rather than re-derived, so the "valid
    # correspondence" rule has one definition instead of two that a future edit could desync.
    # ``distance``/``triangle_id`` are transposed relative to ``residual_valid``'s own parameter
    # order because this one's ``wp.map`` call sites already pass them that way.
    return wp.where(
        residual_valid(triangle_id, distance, max_distance), wp.float32(1.0), wp.float32(0.0)
    )


@wp.kernel
def mesh_correspondence_pass(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    step: wp.array[wp.mat44],
    query_max: wp.float32,
    normal_source: wp.array[wp.vec3],
    out_points: wp.array[wp.vec3],
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_face: wp.array[wp.int32],
    out_normals: wp.array[wp.vec3],
) -> None:
    # One ICP iteration's whole correspondence step against a mesh target: the previous
    # iteration's rigid step applied to each source point, the closest-point query at the moved
    # point, and the target normal it points at.
    #
    # The step is ``apply_transform_mat44``'s own ``transform_point_mat44`` on the same operands,
    # so ``out_points`` holds the bits that kernel wrote when it ran as its own launch at the tail
    # of the previous iteration; the loop now applies the last step once after it exits instead.
    # The query re-reads ``out_points[tid]``, which this thread has just written, so
    # ``write_closest_point_query``'s publication protocol stays the one shared definition.
    #
    # The distance gate is not reported: the only reader of a per-point valid mask was an "is
    # anything left?" test, and the accumulation kernel's weight sum -- which it gates on the same
    # ``residual_valid`` -- already answers that.
    #
    # The three ran as three launches at the same width, and the second and third read nothing
    # but what the first had just written at their own index -- so each paid a launch and a full
    # round trip through global memory to re-read a face index this pass holds in a register.
    # Inside a loop that runs up to ``max_iterations`` times, that is the launch count of the
    # iteration rather than a one-off: measured, it takes a 13-iteration point-to-plane run from
    # 82 launches to 56, output byte-identical on the CPU device (the CUDA plateau is not
    # bit-reproducible on its own -- see ``accumulate_point_to_plane``).
    #
    # **The wall clock is flat all the same, and that is the honest reading**: this loop reads a
    # scalar back every iteration for its convergence test, so the host is already waiting on the
    # device rather than the other way round, and the launches it issues were overlapping with
    # work. What the fusion buys here is the two ``(n,)`` round trips through global memory and a
    # third of the launch count -- worth having, and not a speedup to quote. Do not re-measure
    # this against a run with the convergence break live: the two arms then stop at different
    # iterations and the ratio is fiction.
    tid = wp.int32(wp.tid())
    out_points[tid] = transform_point_mat44(points[tid], step[0])
    _distance, face = write_closest_point_query(
        mesh_id, out_points, query_max, tid, out_closest, out_distance, out_face
    )
    out_normals[tid] = correspondence_normal(normal_source, face)


@wp.kernel
def mesh_correspondence_weight_pass(
    mesh_id: wp.uint64,
    points: wp.array[wp.vec3],
    query_max: wp.float32,
    max_distance: wp.float32,
    out_closest: wp.array[wp.vec3],
    out_distance: wp.array[wp.float32],
    out_face: wp.array[wp.int32],
    out_weights: wp.array[wp.float32],
) -> None:
    # ``registration.icp``'s mesh-target correspondence step: the closest-point query and the
    # binary distance gate its Procrustes fit weights by. The sibling of
    # ``mesh_correspondence_pass``, which serves the point-to-plane loop and additionally gathers
    # a target normal -- the two differ by that gather and by whether the gate is reported as a
    # weight or as a mask, so they stay two kernels over the one shared
    # ``write_closest_point_query``.
    tid = wp.int32(wp.tid())
    distance, face = write_closest_point_query(
        mesh_id, points, query_max, tid, out_closest, out_distance, out_face
    )
    out_weights[tid] = distance_threshold_weight(distance, face, max_distance)


@wp.kernel
def cloud_correspondence_pass(
    target_vertices: wp.array[wp.vec3],
    normal_source: wp.array[wp.vec3],
    index: wp.array[wp.int32],
    out_closest: wp.array[wp.vec3],
    out_normals: wp.array[wp.vec3],
) -> None:
    # The cloud-target tail of ``mesh_correspondence_pass``: the nearest-neighbour search is a
    # ``neighbors.query_nearest`` call rather than a kernel here, so only what consumes its answer
    # fuses -- the matched point and its normal, both read at one correspondence's own index. The
    # point was a Python-scope gather and a ``wp.copy`` of its own before.
    #
    # ``index`` is never ``-1`` here (the search has no radius cap and the cloud is non-empty); the
    # clamp only keeps the read in range should that ever change.
    tid = wp.int32(wp.tid())
    i = index[tid]
    out_closest[tid] = target_vertices[wp.max(i, wp.int32(0))]
    out_normals[tid] = correspondence_normal(normal_source, i)


@wp.func
def abs_deviation(value: wp.float32, center: wp.float32) -> wp.float32:
    """Absolute deviation ``|value - center|`` (median-absolute-deviation building block)."""
    return wp.abs(value - center)


@wp.func
def robust_weight_and_loss(
    residual: wp.float32, scale: wp.float32, kind: wp.int32
) -> tuple[wp.float32, wp.float32]:
    """
    IRLS weight of a residual under an M-estimator, and its term of the objective.

    ``kind``: 0 = none (unit weight), 1 = Huber (``k = scale``),
    2 = Tukey biweight (``c = scale``). A non-positive ``scale`` yields unit weight. One dispatch
    for both halves, so the weight a residual is fitted with and the loss the loop stops on
    cannot disagree about which estimator is in force.

    The loss is ``weight * residual^2`` for no kernel and for Huber, where it grows with
    ``|residual|`` like the loss itself, and twice the Tukey biweight loss ``rho`` for Tukey:
    ``c^2 / 3 * (1 - (1 - u^2)^3)`` with ``u = |r| / c``, saturating at ``c^2 / 3`` once
    ``|r| >= c``. The factor two makes every kind read ``r^2`` for a small residual.

    **Tukey cannot use ``weight * residual^2``**, because the biweight redescends: a residual beyond
    ``c`` carries zero weight, so its term is zero, and as the fit pulls it inside the kernel the
    term *rises*. Registering from a start whose residuals mostly exceed ``c`` -- an explicit
    ``robust_scale`` below the initial misalignment, which is the normal way to pass one -- makes
    that sum climb for several iterations while the pose improves, and a decrease test reads the
    first climb as convergence. ``rho`` is nondecreasing in ``|r|`` and saturates rather than
    vanishing, so it falls as the fit improves. Measured on a 5-degree start with ``c`` at 1 % of
    the bounding-box diagonal: the weighted sum rose for five iterations and the loop stopped after
    two, 4.8 degrees from the answer; the loss falls monotonically and the loop runs to it.
    """
    if kind == wp.int32(0) or scale <= wp.float32(0.0):
        return wp.float32(1.0), residual * residual
    r = wp.abs(residual)
    if kind == wp.int32(1):
        # Huber
        w = wp.float32(1.0)
        if r > scale:
            w = scale / r
        return w, w * residual * residual
    # Tukey biweight: ``t * t * t`` is ``w * t``, the same product in the same order.
    if r >= scale:
        return wp.float32(0.0), scale * scale / wp.float32(3.0)
    u = r / scale
    t = wp.float32(1.0) - u * u
    w = t * t
    return w, scale * scale / wp.float32(3.0) * (wp.float32(1.0) - w * t)


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
    gather: wp.int32,
    offset: wp.int32,
    remaining: wp.int32,
    lane: wp.int32,
    stride: wp.int32,
) -> tuple[wp.spatial_matrix, wp.spatial_vector, wp.float32, wp.float32]:
    # ``lane`` / ``stride`` rather than ``wp.block_dim()`` because this is a ``@wp.func``: the
    # caller is the kernel that knows its own launch shape, and passing the stride in keeps this
    # usable from a serial caller too.
    #
    # ``gather`` selects where a correspondence's target point and normal are read. Unset, they
    # are ``target[idx]`` / ``normals[idx]``, one per correspondence, which is what a mesh target's
    # correspondence pass writes. Set, ``target`` and ``normals`` are a cloud target's own vertices
    # and per-vertex normals and are read at the nearest-neighbour index ``triangle_id[idx]`` --
    # what ``cloud_correspondence_pass`` would have gathered into the per-correspondence buffers,
    # read here directly so the gather is not a launch of its own. A valid correspondence has a
    # non-negative index, so this is that pass's read exactly.
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
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
    # Sum of the robust weight itself, not the "in-range hit" count ``residual_valid`` already
    # gates on: a Tukey kernel can drive *every* in-range correspondence's weight to exactly zero
    # (``r >= scale``, e.g. a ``robust_scale`` too tight for the residual distribution, or a
    # previous iteration's step overshooting far past what a first-iteration-derived scale
    # anticipated), which leaves ``jtj``/``jtr``/``cost`` all zero without a single rejection ever
    # firing. The caller uses this to tell that case apart from genuine convergence.
    weight_sum = wp.float32(0.0)
    for k in range(lane, count, stride):
        idx = offset + k
        if not residual_valid(triangle_id[idx], distance[idx], max_distance):
            continue
        # A mesh target's ``normals`` come from ``face_normals_and_areas``, which writes an exact
        # zero vector for a degenerate face (CLAUDE.md section 12.4) rather than raising -- a plain
        # ``wp.normalize`` on that entry is ``0/0``, and one poisoned lane's NaN spreads to the
        # whole block through the ``wp.tile_sum`` commit below. Same guard, same zero tolerance,
        # as ``transform.transform_normal_mat33``'s identical hazard.
        row = idx
        if gather != 0:
            row = triangle_id[idx]
        nrm = normalize_or_zero(normals[row], wp.float32(0.0))
        x = source[idx]
        r = point_to_plane_residual(x, target[row], nrm)
        w, loss = robust_weight_and_loss(r, robust_scale, robust_kind)
        # Jacobian of the point-to-plane residual: [x x n ; n]
        j = wp.spatial_vector(wp.cross(x, nrm), nrm)
        jtj += w * wp.outer(j, j)
        jtr += (w * r) * j
        cost += loss
        weight_sum += w
    return jtj, jtr, cost, weight_sum


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
    gather: wp.int32,
    out_jtj: wp.array[wp.spatial_matrix],
    out_jtr: wp.array[wp.spatial_vector],
    out_scalars: wp.array[wp.float32],
) -> None:
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` correspondences, lanes striding that block's own chunk, and three
    # block reductions (the normal matrix, the right-hand side, the two scalars) committing one
    # atomic set per block.
    #
    # It was every lane walking a ``TILE_1D`` chunk with lane 0 publishing, which put one add per
    # block on each of 43 hot addresses (36 for the normal matrix, 6 for the right-hand side, 1 for
    # the cost) at ``n / TILE_1D`` blocks. Same finding as ``accumulate_procrustes_moments``: the
    # redundant lanes were nearly free and the atomic contention was the cost, worth an order of
    # magnitude at a million correspondences. 43 reductions is a much larger fixed cost per block
    # than the moments kernel's 25, which is why this trails it at small ``n`` and catches up once
    # the fold has enough to amortize.
    #
    # The tree is the *more* accurate arm on every component, which matters here because ``out_jtj``
    # is the matrix ``point_to_plane_round`` factorizes.
    i, lane = wp.tid()
    offset, remaining = block_chunk_1d(source.shape[0], i)
    if remaining <= 0:
        return

    # ``gather``: see ``point_to_plane_tile``.
    tile_jtj, tile_jtr, tile_cost, tile_weight_sum = point_to_plane_tile(
        source,
        target,
        normals,
        distance,
        triangle_id,
        max_distance,
        robust_kind,
        robust_scale,
        gather,
        offset,
        remaining,
        lane,
        wp.block_dim(),
    )

    # Block-collective, so every lane runs all three and only the commit is guarded.
    total_jtj = block_sum(tile_jtj)
    total_jtr = block_sum(tile_jtr)
    scalars = block_sum(wp.vec2(tile_cost, tile_weight_sum))
    total_cost = scalars[0]
    total_weight_sum = scalars[1]

    if lane == 0:
        wp.atomic_add(out_jtj, 0, total_jtj)
        wp.atomic_add(out_jtr, 0, total_jtr)
        # One length-2 buffer, not two length-1 ones: ``icp_point_to_plane`` reads both of these
        # scalars back per iteration and they are written by this one launch, so sharing a buffer
        # lets it take one host sync rather than two -- a second read placed further downstream
        # drains a pipeline the first had already drained.
        # ``ICP_COST`` = 0, ``ICP_WEIGHT_SUM`` = 1.
        wp.atomic_add(out_scalars, ICP_COST, total_cost)
        wp.atomic_add(out_scalars, ICP_WEIGHT_SUM, total_weight_sum)


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
    1.17. There is no broadcast-fill spelling for it, so do not collapse these.
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
def point_to_plane_round(
    damping: wp.float32,
    threshold: wp.float64,
    max_iterations: wp.int32,
    out_jtj: wp.array[wp.spatial_matrix],
    out_jtr: wp.array[wp.spatial_vector],
    out_scalars: wp.array[wp.float32],
    out_total: wp.array[wp.mat44],
    out_old_cost: wp.array[wp.float64],
    out_step: wp.array[wp.mat44],
    out_state: wp.array[wp.int32],
) -> None:
    """
    Close one ``icp_point_to_plane`` iteration on the device and decide whether another runs.

    ``dim=1``. The host loop's whole per-iteration tail -- the weightless test, the 6x6 solve, the
    composition into the running transform and the convergence test -- so that the loop body is a
    fixed sequence ``_device.run_device_loop`` can record once and replay, with no readback:

    - **weightless**: no correspondence carried weight, so nothing is solved; ``out_state``
      records it (``ICP_LOOP_WEIGHTLESS``) and stops the loop, and the host reports ``inf``;
    - otherwise the damped system is solved, ``out_step`` written, ``out_total`` composed in place
      (``step * total``, the product and order the host ping-pong formed) and the accumulators
      zeroed for the next iteration -- after they are read, by this one thread, so no memset;
    - the loop continues while fewer than ``max_iterations`` have run and, past the first, the
      cost fell by at least ``threshold``. The test is the host's ``old_cost - cost < threshold``
      in ``float64``, where the host held both costs as Python floats of the ``float32`` values.

    Every ``out_`` argument but ``out_step`` is loop state, read *and* rewritten every round; they
    wear the prefix as ``kernels/array.loop_advance``'s ``out_state`` does.
    """
    if out_scalars[ICP_WEIGHT_SUM] <= wp.float32(0.0):
        out_state[ICP_LOOP_WEIGHTLESS] = 1
        out_state[LOOP_CONDITION] = 0
        return
    cost = wp.float64(out_scalars[ICP_COST])
    a = out_jtj[0]
    b = out_jtr[0]

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

    step = make_affine44(rot, tvec)
    out_step[0] = step
    out_total[0] = wp.mul(step, out_total[0])

    out_jtj[0] = wp.spatial_matrix(wp.float32(0.0))
    # Longhand for the reason ``solve_spd6`` gives: ``wp.spatial_vector`` has no broadcast fill.
    out_jtr[0] = wp.spatial_vector(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )
    for slot in range(ICP_SCALAR_ACC_SIZE):
        out_scalars[slot] = wp.float32(0.0)

    iteration = out_state[LOOP_ROUND]
    converged = iteration > 0 and out_old_cost[0] - cost < threshold
    out_old_cost[0] = cost
    out_state[LOOP_ROUND] = iteration + 1
    keep_going = not converged and iteration + 1 < max_iterations
    out_state[LOOP_CONDITION] = wp.where(keep_going, wp.int32(1), wp.int32(0))
