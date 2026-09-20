import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels.array import (
    declare_map_signatures,
    map_probe,
    map_probe_single,
    mat33_column,
    pack_farthest_key,
    pack_nearest_key,
    unpack_ranked_index,
)
from triwarp.kernels.predicates import point_plane_dot, triangle_normal
from triwarp.kernels.reduce import ITEMS_PER_BLOCK_1D, outer_sum_chunk, tile_chunk


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


@wp.kernel
def accumulate_counted_mean(
    count: wp.array[wp.int32], mean_distance: wp.array[wp.float32], out_totals: wp.array[wp.float64]
) -> None:
    # ``(rows with at least one neighbour, sum of their mean distances)`` into one ``(2,)``
    # accumulator, for ``points.statistical_outlier_mask``'s cloud mean.
    #
    # One kernel rather than a ``wp.map`` building a ``count > 0`` mask and two ``reduce.sum``
    # calls over it and over ``mean_distance``. The mask existed only to be counted, so it went
    # with them: a map, an ``n``-length allocation and two reductions -- each of which is a launch,
    # an allocation and a host readback -- against one launch and one sixteen-byte readback here,
    # which is a large fraction of the whole call.
    #
    # ``float64`` slots, not ``float32``: slot 0 is a *count*, and a float32 stops representing
    # consecutive integers at 2 ** 24, which a large cloud reaches. Summing the distances at the
    # wider precision is free alongside it and strictly better than the float32 tree it replaces.
    #
    # An empty row contributes zero to both sums, which is what lets the cloud mean be a plain
    # reduction: ``neighbor_distance_moments`` writes a zero mean for a row it counted nothing in.
    chunk, lane = wp.tid()
    offset, remaining = tile_chunk(count.shape[0], chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return
    # ``tile_chunk`` reports what is left to the end of the array, not this block's share of it.
    n_rows = wp.min(remaining, ITEMS_PER_BLOCK_1D)

    counted = wp.float64(0.0)
    total = wp.float64(0.0)
    for k in range(lane, n_rows, wp.block_dim()):
        if count[offset + k] > 0:
            counted = counted + wp.float64(1.0)
            total = total + wp.float64(mean_distance[offset + k])

    # Block-collective, so both run outside the ``lane == 0`` guard.
    counted_sum = wp.tile_sum(wp.tile(counted))[0]
    total_sum = wp.tile_sum(wp.tile(total))[0]
    if lane == 0:
        wp.atomic_add(out_totals, 0, counted_sum)
        wp.atomic_add(out_totals, 1, total_sum)


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
    # one of those (section 11, the machinery half outranks the subject half).
    #
    # Launched ``wp.launch_tiled(dim=blocks_1d(n), block_dim=TILE_1D)``: one block per
    # ``ITEMS_PER_BLOCK_1D`` points, lanes striding that block's own chunk, nine ``wp.tile_sum``
    # tree reductions, one ``wp.mat33`` atomic per block.
    #
    # **The redundant lane arithmetic is not the cost -- the block count is**, the same finding as
    # ``registration.accumulate_procrustes_moments``: what costs is one add per block on each of
    # nine hot addresses, so the lever is the fold width. Flat at small ``n``, an order of magnitude
    # at a million points. Nine reductions is the cheapest accumulator in this family to amortize,
    # which is why it turns over sooner than the 25- and 43-slot ``registration`` pair. The tree is
    # also the more accurate of the two, by about an order of magnitude at large ``n``.
    #
    # No ``prefers_tiled_reduction`` branch, for the reason in ``.claude/CLAUDE.md`` section 2.2:
    # the lanes partition a chunk the block already owns and stride by ``wp.block_dim()``, which
    # reads 1 on CPU. ``measures.centroid_tiled`` needs a device pair because *its* lanes partition
    # the outer work at a constant stride; this is the other form.
    chunk, lane = wp.tid()
    offset, remaining = tile_chunk(points.shape[0], chunk, ITEMS_PER_BLOCK_1D)
    if remaining <= 0:
        return

    m = outer_sum_chunk(points, center[0], offset, remaining, lane, wp.block_dim())

    # Block-collective, so all nine run outside the ``lane == 0`` guard. The default constructor,
    # not an explicit zero-fill: every one of the nine entries is unconditionally overwritten by
    # the loop below before ``total`` is ever read.
    total = wp.mat33()
    for r in range(3):
        for c in range(3):
            total[r, c] = wp.tile_sum(wp.tile(m[r, c]))[0]

    if lane == 0:
        wp.atomic_add(out_cov, 0, total)


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
    center: wp.array[wp.vec3], m: wp.array[wp.mat33], out_plane: wp.array[wp.vec3]
) -> None:
    # One output buffer rather than two, because both entries cross to the host together and a
    # readback costs far more than the row of it that is read: slot 0 is the normal, slot 1 the
    # centroid, in the order ``fit_plane`` returns them.
    #
    # plane origin is the centroid of the point set; the covariance matrix of
    # the centred points was accumulated into m.
    u, _sigma, _v = wp.svd3(m[0])
    # normal is the singular vector with the smallest singular value
    # (svd3 returns singular values in descending order: last column of u).
    out_plane[0] = wp.normalize(mat33_column(u, 2))
    out_plane[1] = center[0]


# Orientation modes for estimate_point_normals (mirror Open3D's orient methods).
ORIENT_CENTROID = wp.constant(wp.int32(0))  # outward from the cloud centroid (the default)
ORIENT_DIRECTION = wp.constant(wp.int32(1))  # align with a fixed direction
ORIENT_CAMERA = wp.constant(wp.int32(2))  # point toward a camera location


@wp.kernel
def finalize_principal_axes(
    center: wp.array[wp.vec3], m: wp.array[wp.mat33], out_frame: wp.array[wp.vec3]
) -> None:
    # One ``(5,)`` output buffer rather than three, because all three results cross to the host
    # together: rows 0-2 are the rotation's rows, row 3 the eigenvalues, row 4 the centroid --
    # the order ``principal_axes`` returns them in. One readback of five rows is several times
    # cheaper than three readbacks of one.
    # ``wp.svd3`` returns singular values in descending order, so the columns of ``u`` are the
    # principal axes from widest to narrowest. The rows of the result are those axes, which is the
    # convention that makes ``rotation * p`` the coordinates of ``p`` in the principal frame.
    u, sigma, _v = wp.svd3(m[0])
    axis0 = wp.normalize(mat33_column(u, 0))
    axis1 = wp.normalize(mat33_column(u, 1))
    # Take the third axis from the cross product rather than from ``u``: that forces a proper
    # rotation (determinant +1) whatever sign convention the SVD chose, so the frame is always
    # right-handed and only the first two signs are free.
    axis2 = wp.cross(axis0, axis1)
    out_frame[0] = axis0
    out_frame[1] = axis1
    out_frame[2] = axis2
    out_frame[3] = sigma
    out_frame[4] = center[0]


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
    #
    # ``neighbor_idx``'s shape is validated by the caller, but a value ``>= points.shape[0]`` is
    # not -- indexing ``points[nb]`` below then reads out of bounds with no exception on CUDA and
    # corrupts the host heap on CPU (CLAUDE.md section 12.1). Every in-repo caller
    # (``points.estimate_normals``) builds this table from a *self*-query
    # (``query_nearest(points, points, k)``), which can only ever produce an in-range index or the
    # documented ``-1`` sentinel already handled below, so the gap is latent rather than
    # demonstrated. A guard would cost an unconditional device-wide min/max reduction over the
    # whole table on every call, to protect against an input shape nothing currently constructs --
    # the speculative-generality case CLAUDE.md section 4.2 asks to leave unbuilt. Revisit if a
    # caller ever builds an asymmetric ``neighbor_idx`` (e.g. from
    # ``query_nearest(other_cloud, points, k)``) for this kernel.
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
    normal = wp.normalize(mat33_column(u, 2))

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
    # ``query_nearest`` leaves unused slots at ``inf`` (index -1), and a row can be short when
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
    #
    # Same latent gap as ``estimate_point_normals`` above: ``neighbor_idx[i, s]`` is trusted to be
    # either ``-1`` or a valid row of ``standard_distance``, and nothing here bounds-checks it. The
    # one in-repo caller (``points.outlier_probability``) only ever builds this table from a
    # self-query, so this stays a documented, unguarded assumption rather than a demonstrated bug —
    # see the longer note there for why a guard is not added speculatively.
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


@wp.kernel
def nearest_pair_keys(
    nearest_distances: wp.array2d[wp.float32], out_keys: wp.array[wp.int64]
) -> None:
    # One key per point, over column 1 of a ``k=2`` self-query table: column 0 is the point itself.
    # The thread index is the payload, which is what keeps this a kernel rather than a ``wp.map``.
    i = wp.int32(wp.tid())
    out_keys[i] = pack_nearest_key(nearest_distances[i, 1], i)


# Lanes per block for ``farthest_point_sample_block``, keyed on the cloud size. The kernel is one
# persistent block, so this is the whole launch's width and there is nothing else to tune.
#
# **Three brackets, and the middle one is what a Warp 1.17 re-probe added.** The original reading
# was taken on Warp 1.16 against the capture-and-replay loop this kernel replaced and sampled only
# two widths, which is why it read as two brackets. A finer sweep -- interleaved, with the selected
# index buffer **identical at every width in every row**, so this is a pure cost choice -- shows a
# third bracket in between that the coarse one straddled: a small cloud wants a narrow block, a
# large one wants the widest, and there is a real middle band that wants 512.
#
# The brackets are **not** an artefact of ``count``: re-run at a quarter of the sample count, the
# best width is the same at every size. Against the two-bracket dispatch this replaces the win is
# largest exactly at the old crossover, where a cloud that wants 512 lanes was handed 1 024. 2 048
# is where 256 and 512 measure a tie, which is what makes it a safe boundary in either direction.
#
# A constant whose sweep sampled only two values has not been shown to be a two-bracket problem.
FARTHEST_BLOCK_SMALL = 256
FARTHEST_BLOCK_MID = 512
FARTHEST_BLOCK_MID_FROM = 2048
FARTHEST_BLOCK_LARGE = 1024
FARTHEST_BLOCK_LARGE_FROM = 6144


@wp.kernel
def farthest_point_sample_block(
    points: wp.array[wp.vec3],
    start: wp.int32,
    count: wp.int32,
    min_distance_sq: wp.array[wp.float32],
    out_selected: wp.array[wp.int32],
) -> None:
    # The whole greedy sweep in one persistent block: ``count - 1`` rounds, each folding the point
    # just selected into every point's running distance to the chosen set and taking the global
    # arg-max, with ``wp.tile_max`` as both the reduction and the round barrier. Lane ``t`` owns
    # the points ``t, t + block_dim, ...`` throughout, so no other synchronization is needed --
    # including none between the initialization below and the first round.
    #
    # Why a single block wins here where it loses elsewhere (``kernels/holes.py::
    # fill_dp_span_tiled`` measured the opposite): a round is ``n`` distance updates, which one SM
    # finishes in about a microsecond, and the alternative -- one launch per round, even replayed
    # from a captured graph -- paid more than that in launch cost with the device idle. So the
    # per-round cost is what moved. The tie-break is the shipped one: ``pack_farthest_key`` orders
    # equal distances towards the lower index, and the block max over those keys is the same maximum
    # whatever lane holds it, so the selection is reproducible and matches the reference's strict
    # ``>`` arg-max.
    #
    # ``min_distance_sq`` is caller-allocated scratch, initialized here rather than by ``wp.full``
    # so the wrapper is one launch. The stride is ``wp.block_dim()`` rather than the constant so the
    # kernel is also correct on the CPU device, where ``wp.launch_tiled`` runs one lane per block
    # and ``wp.block_dim()`` reads 1: that lane walks the whole cloud each round and the tile max
    # returns its own key.
    _block, t = wp.tid()
    n = points.shape[0]
    stride = wp.block_dim()
    for i in range(t, n, stride):
        min_distance_sq[i] = wp.inf
    chosen_index = start
    if t == 0:
        out_selected[0] = start
    for step in range(1, count):
        chosen = points[chosen_index]
        best = wp.int64(-1)
        for i in range(t, n, stride):
            distance_sq = wp.min(min_distance_sq[i], wp.length_sq(points[i] - chosen))
            min_distance_sq[i] = distance_sq
            best = wp.max(best, pack_farthest_key(distance_sq, i))
        chosen_index = unpack_ranked_index(wp.tile_max(wp.tile(best))[0])
        if t == 0:
            out_selected[step] = chosen_index


@wp.kernel
def hull_support_extremes(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    n_slices: wp.int32,
    out_best_max: wp.array[wp.float32],
    out_best_min: wp.array[wp.float32],
) -> None:
    k, j = wp.tid()
    n_p = points.shape[0]
    direction = directions[k]
    # Strided slice, NOT a contiguous chunk: consecutive threads read consecutive points, so the
    # loads coalesce, and each thread contributes one atomic instead of one per point.
    #
    # Lane-free because the threads partition the **outer** work -- the cloud this reduction is
    # over -- rather than a sequence one block owns, so there is no `wp.block_dim()` for them to
    # stride by and a `wp.tile_max(wp.tile(...))` cannot be reached from here without changing the
    # launch. `wp.launch_tiled` runs one lane per block on the CPU device through Warp 1.17, and
    # that lane would then cover `1/block_dim` of the slice. See `.claude/CLAUDE.md` section 2.2;
    # `farthest_point_sample_block` below is the other side of the rule, and reduces with
    # `wp.tile_max` on both devices because its stride *is* `wp.block_dim()`.
    #
    # **Converting this to one block per direction was measured and refuted**, which is worth
    # recording because the analogous rewrite of `kernels/visibility.py::obscurance` was a large win
    # and the shapes look alike. They are not: `obscurance` launched `dim = n_points` with no second
    # dimension, so the outer dimension alone was starving the device, while this kernel's *slice*
    # dimension is what fills it. Collapsing that into `block_dim` lanes leaves one block per
    # direction: measured with answers bit-identical, that is a win on a small cloud -- where the
    # call is already microseconds -- and a several-fold **loss** on a large one, which is section
    # 13's decline shape exactly. Do not re-propose it from the comment above.
    local_max = wp.float32(-FLOAT32_INF_CONSTANT)
    local_min = wp.float32(FLOAT32_INF_CONSTANT)
    for i in range(j, n_p, n_slices):
        distance = wp.dot(direction, points[i])
        local_max = wp.max(local_max, distance)
        local_min = wp.min(local_min, distance)

    # A slice past the end of the cloud contributes nothing.
    if local_max > -FLOAT32_INF_CONSTANT:
        wp.atomic_max(out_best_max, k, local_max)
        wp.atomic_min(out_best_min, k, local_min)


@wp.func
def support_slack(best_max: wp.float32, best_min: wp.float32, tolerance: wp.float32) -> wp.float32:
    # Slack scales with the per-direction support extent so the test is scale-invariant and stays
    # above the float32 dot-product noise floor. Shared by `mark_hull_support` and
    # `support_indices`, which both tie-break against it: a future change to the formula (e.g. a
    # different scale-invariance fix) has one definition to change rather than two that can drift.
    return tolerance * (best_max - best_min)


@wp.kernel
def mark_hull_support(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    out_mask: wp.array[wp.bool],
) -> None:
    k, i = wp.tid()
    # A hemisphere direction n covers both +n (max, supports the vertex farthest
    # along n) and -n (min, supports the vertex farthest along -n).
    distance = wp.dot(directions[k], points[i])
    slack = support_slack(best_max[k], best_min[k], tolerance)
    if distance >= best_max[k] - slack or distance <= best_min[k] + slack:
        out_mask[i] = True


@wp.kernel
def support_indices(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    out_support: wp.array[wp.int32],
) -> None:
    k, i = wp.tid()
    slack = support_slack(best_max[k], best_min[k], tolerance)
    if wp.dot(directions[k], points[i]) >= best_max[k] - slack:
        # Lowest attaining index wins, so the shell is identical across launches even when
        # several points tie for the support along a direction.
        wp.atomic_min(out_support, k, i)


@wp.kernel
def shell_bounds(
    shell_vertices: wp.array[wp.vec3],
    out_centroid: wp.array[wp.vec3],
    out_radius: wp.array[wp.float32],
) -> None:
    # One thread: the shell has a few hundred vertices at most, and reducing on device keeps the
    # support sweep and the tetrahedron build in one launch chain with no host readback between.
    n = shell_vertices.shape[0]
    total = wp.vec3(0.0, 0.0, 0.0)
    for i in range(n):
        total = total + shell_vertices[i]
    center = total / wp.float32(n)

    # The radius is the length scale the interior margin is measured against, so that the margin is
    # a fraction of the construction's own size rather than of a tetrahedron's aspect ratio.
    radius = wp.float32(0.0)
    for i in range(n):
        radius = wp.max(radius, wp.length(shell_vertices[i] - center))

    out_centroid[0] = center
    out_radius[0] = radius


@wp.func
def outward_plane(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, interior: wp.vec3) -> wp.vec4:
    """Unit-normal plane through the triangle, oriented so ``interior`` has negative offset."""
    # `predicates.triangle_normal` is exactly this triangle's normalize(cross(...)), including the
    # degenerate-triangle convention (the zero vector) this function's own early return needs.
    normal = triangle_normal(p0, p1, p2)
    if wp.length_sq(normal) <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    offset = wp.dot(normal, p0)
    if wp.dot(normal, interior) > offset:
        return wp.vec4(-normal[0], -normal[1], -normal[2], -offset)
    return wp.vec4(normal[0], normal[1], normal[2], offset)


@wp.kernel
def tetrahedron_planes(
    shell_vertices: wp.array[wp.vec3],
    shell_faces: wp.array[wp.int32],
    centroid: wp.array[wp.vec3],
    flatness: wp.float32,
    out_planes: wp.array2d[wp.vec4],
    out_valid: wp.array[wp.bool],
) -> None:
    t = wp.int32(wp.tid())
    apex = centroid[0]
    a = shell_vertices[shell_faces[t * 3 + 0]]
    b = shell_vertices[shell_faces[t * 3 + 1]]
    c = shell_vertices[shell_faces[t * 3 + 2]]

    # Scale-free flatness test: the determinant of the three apex edges against their length
    # product. A sliver's face normals are ill-conditioned cross products of nearly parallel edges,
    # and that error is the only one that can cost the superset guarantee, so reject generously --
    # neighbouring well-shaped tetrahedra cover the same region. A wholly degenerate shell (a
    # coplanar or collinear cloud) rejects every tetrahedron, and the filter then keeps all points.
    ea = a - apex
    eb = b - apex
    ec = c - apex
    m = wp.matrix_from_cols(ea, eb, ec)
    if wp.abs(wp.determinant(m)) <= flatness * wp.length(ea) * wp.length(eb) * wp.length(ec):
        out_valid[t] = False
        return

    # Half-space form with *unit* normals, so the interior test below is a true distance and its
    # margin can be a length. Barycentric coordinates would be the cheaper test but their margin is
    # meaningless as a distance: these tetrahedra run from the centroid out to the shell, so a fixed
    # barycentric slack cuts a thick layer off the base and nothing off the sides.
    inner = (apex + a + b + c) / 4.0
    out_planes[t, 0] = outward_plane(a, b, c, inner)
    out_planes[t, 1] = outward_plane(apex, a, b, inner)
    out_planes[t, 2] = outward_plane(apex, b, c, inner)
    out_planes[t, 3] = outward_plane(apex, c, a, inner)
    out_valid[t] = True


@wp.kernel
def mark_hull_superset(
    points: wp.array[wp.vec3],
    planes: wp.array2d[wp.vec4],
    valid: wp.array[wp.bool],
    radius: wp.array[wp.float32],
    margin: wp.float32,
    out_mask: wp.array[wp.bool],
) -> None:
    i = wp.int32(wp.tid())
    point = points[i]
    # Strict interior only, by a real distance. A point on a tetrahedron's boundary can still be a
    # hull vertex, and requiring it to clear every face by `slack` means float32 error in the plane
    # evaluation can only keep a point that could have been dropped -- never drop a hull vertex.
    slack = margin * radius[0]
    n_tetra = valid.shape[0]
    keep = wp.int32(1)
    for t in range(n_tetra):
        if valid[t]:
            inside = wp.int32(1)
            for f in range(4):
                plane = planes[t, f]
                normal = wp.vec3(plane[0], plane[1], plane[2])
                if wp.dot(normal, point) - plane[3] >= -slack:
                    inside = wp.int32(0)
                    break
            if inside != 0:
                keep = wp.int32(0)
                break
    out_mask[i] = keep != 0


@wp.func
def radial_sort_key(point: wp.vec3, origin: wp.vec3, axis0: wp.vec3, axis1: wp.vec3) -> wp.float32:
    v = point - origin
    # Negated angle: an ascending radix sort of these keys reproduces trimesh's
    # descending-angle order (`angles.argsort()[::-1]`).
    return -wp.atan2(wp.dot(v, axis0), wp.dot(v, axis1))


def _declare_map_kernels() -> None:
    """
    Pre-declare this module's forking ``wp.map`` signatures so each builds one module, not three.

    See ``kernels/array.py::declare_map_signatures`` for why this exists, how the table was
    derived and what forks a ``wp.map`` module; only this module's *own* forking ops belong
    here (the shared builtins are declared there).

    Five ops are ``wp.map``'d from ``points.py``; all five are declared. ``is_finite_point``
    (``point_finite_mask``), ``pack_xy_bits``/``pack_class_z_bits`` (``point_duplicate_mask``'s two
    packing rounds) and ``radial_sort_key`` (``radial_sort``) are reachable from a length-1 point
    cloud exactly as ``is_in_half_space`` is, and each forks its own module the first time a call at
    the other length is seen.
    """
    dense, single = map_probe, map_probe_single
    declare_map_signatures(
        [
            (is_in_half_space, (dense(wp.vec3), wp.vec3(), wp.vec3()), wp.bool),
            (is_in_half_space, (single(wp.vec3), wp.vec3(), wp.vec3()), wp.bool),
            (is_finite_point, (dense(wp.vec3),), wp.bool),
            (is_finite_point, (single(wp.vec3),), wp.bool),
            (pack_xy_bits, (dense(wp.vec3),), wp.int64),
            (pack_xy_bits, (single(wp.vec3),), wp.int64),
            (pack_class_z_bits, (dense(wp.int32), dense(wp.vec3)), wp.int64),
            (pack_class_z_bits, (single(wp.int32), single(wp.vec3)), wp.int64),
            (radial_sort_key, (dense(wp.vec3), wp.vec3(), wp.vec3(), wp.vec3()), wp.float32),
            (radial_sort_key, (single(wp.vec3), wp.vec3(), wp.vec3(), wp.vec3()), wp.float32),
        ]
    )


_declare_map_kernels()
