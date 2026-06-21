import warp as wp

from triwarp.kernels import array as kernel_array

# Custom fixed-size float64 types for the 5x5 quadric-fit normal equations: the rest of the
# kernel runs in float32, but the least-squares solve is done in float64 for conditioning.
vec5d = wp.types.vector(length=5, dtype=wp.float64)
mat55d = wp.types.matrix(shape=(5, 5), dtype=wp.float64)

# ---------------------------------------------------------------------------
# principal_curvature helpers
# ---------------------------------------------------------------------------


@wp.func
def _build_reference_frame(
    vertex: wp.vec3, normal: wp.vec3, first_neighbor: wp.vec3
) -> tuple[wp.vec3, wp.vec3]:
    """Return (t1, t2) orthonormal tangent frame with t1 pointing toward first_neighbor."""
    diff = first_neighbor - vertex
    t1 = diff - normal * wp.dot(diff, normal)
    if wp.length(t1) < wp.float32(1e-6):
        # fallback: arbitrary perpendicular to normal
        if wp.abs(normal[0]) < wp.float32(0.9):
            t1 = wp.vec3(1.0, 0.0, 0.0) - normal * normal[0]
        else:
            t1 = wp.vec3(0.0, 1.0, 0.0) - normal * normal[1]
    t1 = wp.normalize(t1)
    t2 = wp.cross(normal, t1)
    t2 = wp.normalize(t2)
    return t1, t2


@wp.func
def _solve_normal_equations(ata: mat55d, atb: vec5d) -> tuple[vec5d, wp.bool]:
    """
    Solve the 5x5 system ``AtA x = Atb`` by Gaussian elimination with partial pivoting.
    Returns (solution, ok); ok is False if singular to tolerance 1e-14.
    """
    m = ata
    b = atb

    # forward elimination
    for col in range(5):
        # find pivot row
        pivot_row = col
        pivot_val = wp.abs(m[col, col])
        for row in range(col + 1, 5):
            v = wp.abs(m[row, col])
            if v > pivot_val:
                pivot_val = v
                pivot_row = row
        if pivot_val < wp.float64(1e-14):
            return b, False
        # swap rows col and pivot_row
        if pivot_row != col:
            for k in range(5):
                tmp = m[col, k]
                m[col, k] = m[pivot_row, k]
                m[pivot_row, k] = tmp
            tmp_b = b[col]
            b[col] = b[pivot_row]
            b[pivot_row] = tmp_b
        # eliminate rows below
        inv = wp.float64(1.0) / m[col, col]
        for row in range(col + 1, 5):
            factor = m[row, col] * inv
            for k in range(col, 5):
                m[row, k] = m[row, k] - factor * m[col, k]
            b[row] = b[row] - factor * b[col]

    # back-substitution: iterate col = 4, 3, 2, 1, 0
    x = vec5d()
    for back_idx in range(5):
        col = 4 - back_idx
        val = b[col]
        for k in range(col + 1, 5):
            val = val - m[col, k] * x[k]
        x[col] = val / m[col, col]
    return x, True


@wp.func
def _eigvec_sym2(m00: wp.float32, m01: wp.float32, lam: wp.float32, fallback: wp.vec2) -> wp.vec2:
    """
    Return the unit eigenvector of a 2x2 matrix ``[[m00, m01], [m10, m11]]`` for eigenvalue ``lam``,
    expressed in the (t1, t2) tangent-frame basis.

    Uses the first-row form ``v ~ [m01, lam - m00]``, which is valid for any 2x2 (symmetric or the
    non-symmetric Weingarten map) since it depends only on the top row. ``fallback`` is returned when
    that row is degenerate (near-diagonal matrix).
    """
    v = wp.vec2(m01, lam - m00)
    if wp.length(v) < wp.float32(1e-14):
        return fallback
    return wp.normalize(v)


@wp.func
def _principal_curvatures_from_monge(
    first_form: wp.vec3, second_form: wp.vec3, frame_independent: wp.bool
) -> tuple[wp.float32, wp.float32, wp.vec2, wp.vec2]:
    """
    Extract principal curvatures and directions from the fundamental forms (Monge patch).

    Takes the first fundamental form ``first_form = (E, F, G)`` and second fundamental form
    ``second_form = (L, M, N)`` and builds the shape operator (Weingarten map) as a 2x2 matrix whose
    (real) eigenvalues are the principal curvatures. The two formulations share ``m00``, ``m10`` and
    ``m11`` and differ only in the upper-right entry:

        m = [[L*G - M*F,  m01      ],
             [M*E - L*F,  N*E - M*F]] / (E*G - F*F)

    * ``frame_independent=True`` (textbook): ``m01 = (M*G - N*F) / (E*G - F*F)`` — the true
      generalized eigenvalue problem ``II*v = lam*I*v``. The eigenvalues are surface invariants and
      do not depend on the chosen tangent frame.
    * ``frame_independent=False``: ``m01 = M*E - L*F`` (reuses the lower-left term), reproducing
      ``igl::principal_curvature``'s ``finalEigenStuff`` verbatim. libigl forces this symmetry, which
      keeps the trace (mean curvature) exact but alters the determinant (eigenvalue spread) and makes
      the result depend on the reference frame.

    Returns (lam0, lam1, ev0, ev1) where lam0 <= lam1 are eigenvalues of ``m``. The caller negates
    them (libigl's ``c_val = -c_val``). The eigenvectors ev0, ev1 are unit ``wp.vec2`` in the
    (t1, t2) tangent-frame 2D basis.
    """
    e_ff = first_form[0]
    f_ff = first_form[1]
    g_ff = first_form[2]
    l_ff = second_form[0]
    m_ff = second_form[1]
    n_ff = second_form[2]

    inv_denom = wp.float32(1.0) / (e_ff * g_ff - f_ff * f_ff)
    m00 = (l_ff * g_ff - m_ff * f_ff) * inv_denom
    m10 = (m_ff * e_ff - l_ff * f_ff) * inv_denom
    m11 = (n_ff * e_ff - m_ff * f_ff) * inv_denom
    # Upper-right entry: the textbook Weingarten map uses (M*G - N*F); libigl forces symmetry by
    # reusing the lower-left (M*E - L*F), which is what makes its eigenvalues frame-dependent.
    if frame_independent:
        m01 = (m_ff * g_ff - n_ff * f_ff) * inv_denom
    else:
        m01 = m10

    # Eigenvalues of [[m00, m01], [m10, m11]] (ascending). The discriminant argument is
    # mathematically >= 0 for the Weingarten map; wp.max guards numerical dips near umbilics.
    half_trace = (m00 + m11) * wp.float32(0.5)
    half_diff = (m00 - m11) * wp.float32(0.5)
    disc = wp.sqrt(wp.max(half_diff * half_diff + m01 * m10, wp.float32(0.0)))
    lam0 = half_trace - disc
    lam1 = half_trace + disc

    # Eigenvectors of (m - lam*I)v = 0 → v ~ [m01, lam - m00], with axis fallbacks when degenerate.
    ev0 = _eigvec_sym2(m00, m01, lam0, wp.vec2(1.0, 0.0))
    ev1 = _eigvec_sym2(m00, m01, lam1, wp.vec2(0.0, 1.0))

    return lam0, lam1, ev0, ev1


# ---------------------------------------------------------------------------
# principal curvature kernel
# ---------------------------------------------------------------------------


@wp.kernel
def fit_principal_curvature(
    vertices: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    reference_neighbors: wp.array[wp.int32],
    frame_independent: wp.bool,
    out_pd1: wp.array[wp.vec3],
    out_pd2: wp.array[wp.vec3],
    out_pv1: wp.array[wp.float32],
    out_pv2: wp.array[wp.float32],
    out_valid: wp.array[wp.bool],
) -> None:
    """
    Fit a quadric surface in a local tangent frame per vertex and extract principal
    curvature directions and magnitudes. Matches igl::principal_curvature.

    """
    i = int(wp.tid())
    zero3 = wp.vec3(0.0, 0.0, 0.0)

    start = int(offsets[i])
    # offsets has length n_vertices (no sentinel); last vertex ends at neighbor_indices end
    if i + 1 < offsets.shape[0]:
        end = int(offsets[i + 1])
    else:
        end = int(neighbor_indices.shape[0])
    n_nbr = end - start

    if (
        n_nbr < 5
    ):  # need at least 5 non-self neighbors for quadric fit; degenerate cases caught by gauss_elim
        out_pd1[i] = zero3
        out_pd2[i] = zero3
        out_pv1[i] = wp.float32(0.0)
        out_pv2[i] = wp.float32(0.0)
        out_valid[i] = False
        return

    vertex = vertices[i]
    normal = wp.normalize(vertex_normals[i])

    # Build the tangent frame from the lowest-indexed mesh-adjacency neighbor, matching libigl's
    # computeReferenceFrame (adjacency_list[i][0]). When frame_independent is False, libigl extracts
    # curvature from a *symmetrized* shape operator whose eigenvalues depend on the chosen frame, so
    # the principal values only agree with igl::principal_curvature when this exact reference
    # direction is used. When frame_independent is True the eigenvalues are surface invariants, so
    # the exact frame is irrelevant (any orthonormal tangent basis yields the same result).
    ref = int(reference_neighbors[i])
    t1, t2 = _build_reference_frame(vertex, normal, vertices[ref])

    # Count neighbors passing projection-plane filter, including self (self always passes with dot=1).
    # Matches libigl's applyProjOnPlane which includes vv[self] because dot(n_i, n_i) = 1 > 0.
    n_valid = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    for k in range(n_nbr):
        j = int(neighbor_indices[start + k])
        if j == i:
            n_valid = n_valid + 1  # self always passes
            continue
        nj = wp.normalize(vertex_normals[j])
        if wp.dot(nj, normal) > wp.float32(0.0):
            n_valid = n_valid + 1

    # Mirror libigl: only apply filter if it leaves >= 6 AND fewer than the full set
    use_filter = n_valid >= 6

    # Least-squares quadric fit: accumulate the normal equations AᵀA x = Aᵀb.
    ata = mat55d()
    atb = vec5d()

    for k in range(n_nbr):
        j = int(neighbor_indices[start + k])
        if j == i:
            continue  # self contributes (0,0,0) — skip to avoid frame degeneration
        nj = wp.normalize(vertex_normals[j])
        if use_filter and wp.dot(nj, normal) <= wp.float32(0.0):
            continue

        diff = vertices[j] - vertex
        u = wp.float64(wp.dot(diff, t1))
        v_c = wp.float64(wp.dot(diff, t2))
        w = wp.float64(wp.dot(diff, normal))

        # row of A: [u², uv, v², u, v]
        r = vec5d(u * u, u * v_c, v_c * v_c, u, v_c)
        ata += wp.outer(r, r)
        atb += r * w

    solution, ok = _solve_normal_equations(ata, atb)
    if not ok:
        out_pd1[i] = zero3
        out_pd2[i] = zero3
        out_pv1[i] = wp.float32(0.0)
        out_pv2[i] = wp.float32(0.0)
        out_valid[i] = False
        return

    # Cast the float64 solution back to float32 for the rest of the kernel.
    a = wp.float32(solution[0])
    b = wp.float32(solution[1])
    c = wp.float32(solution[2])
    d = wp.float32(solution[3])
    e = wp.float32(solution[4])

    # First fundamental form coefficients
    E_ff = wp.float32(1.0) + d * d
    F_ff = d * e
    G_ff = wp.float32(1.0) + e * e
    denom = E_ff * G_ff - F_ff * F_ff

    if wp.abs(denom) < wp.float32(1e-14):
        out_pd1[i] = zero3
        out_pd2[i] = zero3
        out_pv1[i] = wp.float32(0.0)
        out_pv2[i] = wp.float32(0.0)
        out_valid[i] = False
        return

    # Normal z-component in local frame
    nz = wp.float32(1.0) / wp.sqrt(d * d + e * e + wp.float32(1.0))

    # Second fundamental form
    L_ff = wp.float32(2.0) * a * nz
    M_ff = b * nz
    N_ff = wp.float32(2.0) * c * nz

    first_form = wp.vec3(E_ff, F_ff, G_ff)
    second_form = wp.vec3(L_ff, M_ff, N_ff)
    lam0, lam1, ev0, ev1 = _principal_curvatures_from_monge(
        first_form, second_form, frame_independent
    )

    # Negate: the Monge patch height function curves downward for convex surfaces,
    # giving negative eigenvalues; convention is positive curvature for convex.
    k0 = -lam0
    k1 = -lam1

    # Reconstruct global directions from local eigenvectors
    dir0 = wp.normalize(t1 * ev0[0] + t2 * ev0[1])
    dir1 = wp.normalize(t1 * ev1[0] + t2 * ev1[1])

    # Assign so that PV1 >= PV2
    if k0 >= k1:
        out_pd1[i] = dir0
        out_pd2[i] = dir1
        out_pv1[i] = k0
        out_pv2[i] = k1
    else:
        out_pd1[i] = dir1
        out_pd2[i] = dir0
        out_pv1[i] = k1
        out_pv2[i] = k0
    out_valid[i] = True


@wp.func
def line_ball_intersection_segment(
    start_point: wp.vec3, end_point: wp.vec3, center: wp.vec3, radius: wp.float32
) -> wp.float32:
    segment = end_point - start_point
    oc = start_point - center
    r = radius
    ldotl = wp.dot(segment, segment)
    ldotoc = wp.dot(segment, oc)
    ocdotoc = wp.dot(oc, oc)
    discrim = ldotoc * ldotoc - ldotl * (ocdotoc - r * r)

    if discrim <= wp.float32(0.0):
        return wp.float32(0.0)

    sqrt_discrim = wp.sqrt(discrim)
    d1 = (-ldotoc - sqrt_discrim) / ldotl
    d2 = (-ldotoc + sqrt_discrim) / ldotl

    d1 = wp.clamp(d1, wp.float32(0.0), wp.float32(1.0))
    d2 = wp.clamp(d2, wp.float32(0.0), wp.float32(1.0))

    return (d2 - d1) * wp.sqrt(ldotl)


@wp.kernel
def edge_aabb_from_endpoints(
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    tid = int(wp.tid())
    v0 = vertices[face_adjacency_edges[tid, 0]]
    v1 = vertices[face_adjacency_edges[tid, 1]]
    out_lower[tid] = wp.vec3(wp.min(v0[0], v1[0]), wp.min(v0[1], v1[1]), wp.min(v0[2], v1[2]))
    out_upper[tid] = wp.vec3(wp.max(v0[0], v1[0]), wp.max(v0[1], v1[1]), wp.max(v0[2], v1[2]))


@wp.kernel
def accumulate_mean_curvature(
    queries: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    angles: wp.array[wp.float32],
    convex: wp.array[wp.bool],
    candidate_edge_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    radius: wp.float32,
    out_mean_curvature: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    edge_idx = candidate_edge_indices[tid]
    query_idx = kernel_array.binary_search_index(offsets, wp.int32(tid)) - wp.int32(1)

    e0 = face_adjacency_edges[edge_idx, 0]
    e1 = face_adjacency_edges[edge_idx, 1]
    start_point = vertices[e0]
    end_point = vertices[e1]
    center = queries[query_idx]

    length = line_ball_intersection_segment(start_point, end_point, center, radius)
    angle = angles[edge_idx]
    sign = wp.float32(1.0)
    if not convex[edge_idx]:
        sign = wp.float32(-1.0)

    wp.atomic_add(out_mean_curvature, query_idx, length * angle * sign * wp.float32(0.5))
