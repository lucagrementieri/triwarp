import warp as wp

from triwarp.kernels.array import to_vec3d
from triwarp.kernels.triangles import face_vertices


@wp.func
def laplacian_step(v_prev: wp.vec3d, lv: wp.vec3d, coeff: wp.float64) -> wp.vec3d:
    # Explicit diffusion step v' = v + coeff * (L·v - v); coeff = +lambda (shrink) or -nu (inflate).
    return v_prev + coeff * (lv - v_prev)


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
    return lv - (alpha * original + (wp.float64(1.0) - alpha) * q)


@wp.func
def humphrey_update(lv: wp.vec3d, b: wp.vec3d, lb: wp.vec3d, beta: wp.float64) -> wp.vec3d:
    # v' = L·v - (beta * b + (1 - beta) * L·b).
    return lv - (beta * b + (wp.float64(1.0) - beta) * lb)


@wp.func
def mut_dif_adil(normal: wp.vec3, v: wp.vec3d, lv: wp.vec3d) -> wp.float64:
    # adil = 1 / max(1e-12, |N . (V - L.V)|), the reciprocal normal-residual magnitude per vertex.
    d = wp.abs(wp.dot(to_vec3d(normal), v - lv))
    return wp.float64(1.0) / wp.max(wp.float64(1e-12), d)


@wp.func
def mut_dif_step(
    v_prev: wp.vec3d, lv: wp.vec3d, adil: wp.float64, mean_adil: wp.float64, lamb: wp.float64
) -> wp.vec3d:
    # v' = v + lamber * (L.v - v), lamber = clamp(lamb * adil / mean_adil, 0.2 * lamb, 1.0).
    lamber = wp.max(wp.float64(0.2) * lamb, wp.min(wp.float64(1.0), lamb * adil / mean_adil))
    return v_prev + lamber * (lv - v_prev)


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
def signed_tet_volumes(
    vertices: wp.array[wp.vec3d], faces: wp.array[wp.int32], out_volumes: wp.array[wp.float64]
) -> None:
    # Signed volume of the tetrahedron (origin, v0, v1, v2); the sum over faces is the mesh volume.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(f))
    out_volumes[f] = wp.dot(v0, wp.cross(v1, v2)) / wp.float64(6.0)


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
