import warp as wp

from triwarp.kernels.array import to_vec3d

# ---------------------------------------------------------------------------
# Region Dirichlet / least-squares smoothing (positionVertsSmoothly, MRLaplacian.cpp)
# ---------------------------------------------------------------------------


@wp.func
def _corner_cotan(p: wp.vec3, q: wp.vec3, o: wp.vec3) -> wp.float32:
    # Cotangent of the angle at corner ``o`` in triangle ``(o, p, q)`` (MeshLib leftCotan).
    a = p - o
    b = q - o
    cr = wp.length(wp.cross(a, b))
    if cr <= wp.float32(0.0):
        return wp.float32(0.0)
    return wp.dot(a, b) / cr


@wp.kernel
def edge_cotan_add(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_w: wp.array[wp.float32],
) -> None:
    # Accumulate each face corner's cotangent into its opposite unique edge; the two incident faces
    # sum to the cotangent edge weight cot(alpha) + cot(beta).
    f = int(wp.tid())
    v0 = faces[f * 3 + 0]
    v1 = faces[f * 3 + 1]
    v2 = faces[f * 3 + 2]
    p0 = vertices[v0]
    p1 = vertices[v1]
    p2 = vertices[v2]
    wp.atomic_add(out_w, inverse[f * 3 + 0], _corner_cotan(p0, p1, p2))
    wp.atomic_add(out_w, inverse[f * 3 + 1], _corner_cotan(p1, p2, p0))
    wp.atomic_add(out_w, inverse[f * 3 + 2], _corner_cotan(p2, p0, p1))


@wp.func
def clamp_cotan(w: wp.float32) -> wp.float32:
    # MeshLib clamps the summed cotangent edge weight (degenerate edges give arbitrarily high cot).
    return wp.clamp(w, wp.float32(-1.0), wp.float32(10.0))


@wp.kernel
def symmetric_weight_triplets(
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    i = int(wp.tid())
    a = unique_edges[i, 0]
    b = unique_edges[i, 1]
    w = wp.float64(weights[i])
    out_rows[2 * i] = a
    out_cols[2 * i] = b
    out_vals[2 * i] = w
    out_rows[2 * i + 1] = b
    out_cols[2 * i + 1] = a
    out_vals[2 * i + 1] = w


@wp.kernel
def dirichlet_system_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    stabilizer: wp.float64,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
    out_rhs_z: wp.array[wp.float64],
) -> None:
    # positionVertsSmoothlySharpBd: SPD umbrella system A = D - W over free verts (weights in the
    # CSR ``W``), fixed 1-ring neighbors folded into the right-hand side, plus optional stabilizer.
    v = int(wp.tid())
    if not free_mask[v]:
        return
    ri = free_map[v]
    start = offsets[v]
    end = offsets[v + 1]
    sum_w = stabilizer
    rhs = stabilizer * to_vec3d(points[v])
    base = start + v  # one reserved diagonal slot per vertex; off-diagonals follow
    for k in range(start, end):
        j = columns[k]
        w = values[k]
        sum_w += w
        if free_mask[j]:
            slot = base + 1 + (k - start)
            out_rows[slot] = ri
            out_cols[slot] = free_map[j]
            out_vals[slot] = -w
        else:
            rhs += w * to_vec3d(points[j])
    out_rows[base] = ri
    out_cols[base] = ri
    out_vals[base] = sum_w
    out_rhs_x[ri] = rhs[0]
    out_rhs_y[ri] = rhs[1]
    out_rhs_z[ri] = rhs[2]


@wp.kernel
def laplacian_ls_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    free_mask: wp.array[wp.bool],
    row_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    row_map: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
    out_rhs_z: wp.array[wp.float64],
) -> None:
    # positionVertsSmoothly: least-squares umbrella rows over R = free plus first-fixed-ring.
    # Row is
    # ``p_v = sum_d (w_vd/sumW) p_d``; free neighbors stay in M, fixed ones move to the RHS.
    # The
    # normal equations (M^T M) x = M^T b are assembled by the caller.
    v = int(wp.tid())
    if not row_mask[v]:
        return
    start = offsets[v]
    end = offsets[v + 1]
    sum_w = wp.float64(0.0)
    for k in range(start, end):
        sum_w += values[k]
    if sum_w == wp.float64(0.0):
        return
    r = row_map[v]
    base = start + v
    is_free = free_mask[v]
    rhs = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if not is_free:
        rhs = -to_vec3d(points[v])
    for k in range(start, end):
        j = columns[k]
        coeff = -values[k] / sum_w
        if free_mask[j]:
            slot = base + 1 + (k - start)
            out_rows[slot] = r
            out_cols[slot] = free_map[j]
            out_vals[slot] = coeff
        else:
            rhs -= coeff * to_vec3d(points[j])
    if is_free:
        out_rows[base] = r
        out_cols[base] = free_map[v]
        out_vals[base] = wp.float64(1.0)
    out_rhs_x[r] = rhs[0]
    out_rhs_y[r] = rhs[1]
    out_rhs_z[r] = rhs[2]


@wp.kernel
def scatter_free_solution(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_points: wp.array[wp.vec3],
) -> None:
    v = int(wp.tid())
    if free_mask[v]:
        i = free_map[v]
        out_points[v] = wp.vec3(wp.float32(sol_x[i]), wp.float32(sol_y[i]), wp.float32(sol_z[i]))


@wp.func
def laplacian_step(v_prev: wp.vec3d, lv: wp.vec3d, coeff: wp.float64) -> wp.vec3d:
    # Explicit diffusion step v' = v + coeff * (L·v - v); coeff = +lambda (shrink) or -nu (inflate).
    # ``wp.lerp`` extrapolates for coeff outside [0, 1], which the inflating step relies on.
    return wp.lerp(v_prev, lv, coeff)


@wp.func
def neighborhood_average(
    v_prev: wp.vec3d, lv: wp.vec3d, start: wp.int32, end: wp.int32
) -> wp.vec3d:
    # Closed 1-ring average: new_v = (v + deg * L·v) / (deg + 1), where L is the neighbors-only
    # averaging operator and deg = CSR row length (vertex degree). deg=0 -> new_v = v.
    deg = wp.float64(end - start)
    return (v_prev + deg * lv) / (deg + wp.float64(1.0))


@wp.func
def humphrey_residual(lv: wp.vec3d, original: wp.vec3d, q: wp.vec3d, alpha: wp.float64) -> wp.vec3d:
    # b = L·v - (alpha * original + (1 - alpha) * q), the Humphrey correction term.
    return lv - wp.lerp(q, original, alpha)


@wp.func
def humphrey_update(lv: wp.vec3d, b: wp.vec3d, lb: wp.vec3d, beta: wp.float64) -> wp.vec3d:
    # v' = L·v - (beta * b + (1 - beta) * L·b).
    return lv - wp.lerp(lb, b, beta)


@wp.func
def mut_dif_adil(normal: wp.vec3, v: wp.vec3d, lv: wp.vec3d) -> wp.float64:
    # adil = 1 / max(1e-12, |N . (V - L.V)|), the reciprocal normal-residual magnitude per vertex.
    d = wp.abs(wp.dot(to_vec3d(normal), v - lv))
    return wp.float64(1.0) / wp.max(wp.float64(1e-12), d)


@wp.func
def mut_dif_step(
    v_prev: wp.vec3d, lv: wp.vec3d, adil: wp.float64, mean_adil: wp.float64, lamb: wp.float64
) -> wp.vec3d:
    # v' = v + lamber * (L.v - v), lamber = max(0.2 * lamb, min(1.0, lamb * adil / mean_adil)).
    # Not ``wp.clamp``: the two differ once ``0.2 * lamb > 1``, and this nesting order is the one
    # trimesh's ``filter_mut_dif_laplacian`` uses (``np.maximum(..., np.minimum(...))``).
    lamber = wp.max(wp.float64(0.2) * lamb, wp.min(wp.float64(1.0), lamb * adil / mean_adil))
    return wp.lerp(v_prev, lv, lamber)


@wp.kernel
def mut_dif_step_scaled(
    positions: wp.array[wp.vec3d],
    lv: wp.array[wp.vec3d],
    adil: wp.array[wp.float64],
    adil_sum: wp.array[wp.float64],
    inv_n: wp.float64,
    lamb: wp.float64,
    out_next: wp.array[wp.vec3d],
) -> None:
    # ``mut_dif_step`` with the mean coefficient read from a device scalar (adil_sum[0] * inv_n),
    # so the smoothing loop never synchronises with the host. A real kernel rather than wp.map:
    # the length-1 ``adil_sum`` is a uniform argument, which wp.map cannot broadcast.
    i = int(wp.tid())
    mean_adil = adil_sum[0] * inv_n
    out_next[i] = mut_dif_step(positions[i], lv[i], adil[i], mean_adil, lamb)


@wp.func
def add_scaled_normal(v_prev: wp.vec3d, normal: wp.vec3, scale: wp.float64) -> wp.vec3d:
    # v' = v + scale * N; reused for the eps finite-difference probe and the volume correction.
    return v_prev + scale * to_vec3d(normal)


@wp.func
def extract_components(v: wp.vec3d) -> tuple[wp.float64, wp.float64, wp.float64]:
    return v[0], v[1], v[2]


@wp.func
def combine_components(x: wp.float64, y: wp.float64, z: wp.float64) -> wp.vec3d:
    return wp.vec3d(x, y, z)


@wp.kernel
def implicit_laplacian_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    lamb: wp.float64,
    nnz: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # Triplets for AA = (1 + lambda) * I - lambda * L (backward-Euler system, Article 2), where
    # L is the row-stochastic averaging operator. Off-diagonals reuse L's CSR positions; one
    # diagonal triplet per row is appended after the nnz off-diagonals.
    i = int(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    for k in range(start, end):
        out_rows[k] = i
        out_cols[k] = columns[k]
        out_vals[k] = -lamb * wp.float64(values[k])
    diag = nnz + i
    out_rows[diag] = i
    out_cols[diag] = i
    out_vals[diag] = wp.float64(1.0) + lamb
