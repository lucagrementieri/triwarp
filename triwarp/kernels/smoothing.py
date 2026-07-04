import warp as wp


@wp.kernel
def to_vec3d(v_in: wp.array[wp.vec3], out_v: wp.array[wp.vec3d]) -> None:
    i = int(wp.tid())
    p = v_in[i]
    out_v[i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))


@wp.kernel
def to_vec3(v_in: wp.array[wp.vec3d], out_v: wp.array[wp.vec3]) -> None:
    i = int(wp.tid())
    p = v_in[i]
    out_v[i] = wp.vec3(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))


@wp.kernel
def laplacian_step(
    v_prev: wp.array[wp.vec3d], lv: wp.array[wp.vec3d], coeff: wp.float64, out_v: wp.array[wp.vec3d]
) -> None:
    # Explicit diffusion step v' = v + coeff * (L·v - v); coeff = +lambda (shrink) or -nu (inflate).
    i = int(wp.tid())
    out_v[i] = v_prev[i] + coeff * (lv[i] - v_prev[i])


@wp.kernel
def neighborhood_average_step(
    v_prev: wp.array[wp.vec3d],
    lv: wp.array[wp.vec3d],
    offsets: wp.array[wp.int32],
    out_v: wp.array[wp.vec3d],
) -> None:
    # Closed 1-ring average: new_v = (v + deg * L·v) / (deg + 1), where L is the neighbors-only
    # averaging operator and deg = CSR row length (vertex degree). deg=0 -> new_v = v.
    i = int(wp.tid())
    deg = wp.float64(offsets[i + 1] - offsets[i])
    out_v[i] = (v_prev[i] + deg * lv[i]) / (deg + wp.float64(1.0))


@wp.kernel
def humphrey_residual(
    lv: wp.array[wp.vec3d],
    original: wp.array[wp.vec3d],
    q: wp.array[wp.vec3d],
    alpha: wp.float64,
    out_b: wp.array[wp.vec3d],
) -> None:
    # b = L·v - (alpha * original + (1 - alpha) * q), the Humphrey correction term.
    i = int(wp.tid())
    out_b[i] = lv[i] - (alpha * original[i] + (wp.float64(1.0) - alpha) * q[i])


@wp.kernel
def humphrey_update(
    lv: wp.array[wp.vec3d],
    b: wp.array[wp.vec3d],
    lb: wp.array[wp.vec3d],
    beta: wp.float64,
    out_v: wp.array[wp.vec3d],
) -> None:
    # v' = L·v - (beta * b + (1 - beta) * L·b).
    i = int(wp.tid())
    out_v[i] = lv[i] - (beta * b[i] + (wp.float64(1.0) - beta) * lb[i])


@wp.kernel
def scale_vertices(factor: wp.float64, out_v: wp.array[wp.vec3d]) -> None:
    i = int(wp.tid())
    out_v[i] = factor * out_v[i]


@wp.kernel
def mut_dif_adil(
    normals: wp.array[wp.vec3],
    v: wp.array[wp.vec3d],
    lv: wp.array[wp.vec3d],
    out_adil: wp.array[wp.float64],
) -> None:
    # adil = 1 / max(1e-12, |N . (V - L.V)|), the reciprocal normal-residual magnitude per vertex.
    i = int(wp.tid())
    p = normals[i]
    nrm = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    d = wp.abs(wp.dot(nrm, v[i] - lv[i]))
    out_adil[i] = wp.float64(1.0) / wp.max(wp.float64(1e-12), d)


@wp.kernel
def mut_dif_step(
    v_prev: wp.array[wp.vec3d],
    lv: wp.array[wp.vec3d],
    adil: wp.array[wp.float64],
    mean_adil: wp.float64,
    lamb: wp.float64,
    out_v: wp.array[wp.vec3d],
) -> None:
    # v' = v + lamber * (L.v - v), lamber = clamp(lamb * adil / mean_adil, 0.2 * lamb, 1.0).
    i = int(wp.tid())
    lamber = wp.max(wp.float64(0.2) * lamb, wp.min(wp.float64(1.0), lamb * adil[i] / mean_adil))
    out_v[i] = v_prev[i] + lamber * (lv[i] - v_prev[i])


@wp.kernel
def add_scaled_normal(
    v_prev: wp.array[wp.vec3d],
    normals: wp.array[wp.vec3],
    scale: wp.float64,
    out_v: wp.array[wp.vec3d],
) -> None:
    # v' = v + scale * N; reused for the eps finite-difference probe and the volume correction.
    i = int(wp.tid())
    p = normals[i]
    nrm = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    out_v[i] = v_prev[i] + scale * nrm


@wp.kernel
def signed_tet_volumes(
    vertices: wp.array[wp.vec3d], faces: wp.array[wp.int32], out_volumes: wp.array[wp.float64]
) -> None:
    # Signed volume of the tetrahedron (origin, v0, v1, v2); the sum over faces is the mesh volume.
    f = int(wp.tid())
    v0 = vertices[faces[f * 3 + 0]]
    v1 = vertices[faces[f * 3 + 1]]
    v2 = vertices[faces[f * 3 + 2]]
    out_volumes[f] = wp.dot(v0, wp.cross(v1, v2)) / wp.float64(6.0)


@wp.kernel
def extract_components(
    v: wp.array[wp.vec3d],
    out_x: wp.array[wp.float64],
    out_y: wp.array[wp.float64],
    out_z: wp.array[wp.float64],
) -> None:
    i = int(wp.tid())
    p = v[i]
    out_x[i] = p[0]
    out_y[i] = p[1]
    out_z[i] = p[2]


@wp.kernel
def insert_components(
    x: wp.array[wp.float64],
    y: wp.array[wp.float64],
    z: wp.array[wp.float64],
    out_v: wp.array[wp.vec3d],
) -> None:
    i = int(wp.tid())
    out_v[i] = wp.vec3d(x[i], y[i], z[i])


@wp.kernel
def scale_by_diagonal(
    diag: wp.array[wp.float64], comp: wp.array[wp.float64], out_rhs: wp.array[wp.float64]
) -> None:
    # Right-hand side b = M·U for a diagonal (lumped) mass matrix M.
    i = int(wp.tid())
    out_rhs[i] = diag[i] * comp[i]


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
