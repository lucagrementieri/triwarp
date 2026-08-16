import warp as wp

# The 5x5 quadric solve below uses Warp's own Householder QR instead of a hand-rolled elimination.
# Unlike ``warp.fem``'s solvers (which ``triwarp.reconstruction`` imports lazily to dodge a
# tens-of-seconds first-call codegen penalty), these are plain ``@wp.func``s that inline into this
# module — no fem codegen is triggered. The import itself is eager and costs ~0.15 s of
# ``import triwarp`` (measured), almost all of it ``warp/fem/__init__.py`` rather than ``linalg``.
from warp.fem.linalg import householder_qr_decomposition, solve_triangular

from triwarp.kernels import array as kernel_array
from triwarp.kernels.predicates import project_out_normal

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
    t1 = project_out_normal(diff, normal)
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
    Solve the 5x5 system ``AtA x = Atb`` by Householder QR.

    Returns (solution, ok); ok is False if singular to tolerance 1e-14.
    """
    q, r = householder_qr_decomposition(ata)
    # ``|R[k, k]|`` is the norm of column k after the preceding reflections — the QR analogue of
    # the partial-pivot magnitude the former Gaussian elimination tested, to within a sqrt(5)
    # factor, so the 1e-14 singularity threshold carries over unchanged.
    for k in range(5):
        if wp.abs(r[k, k]) < wp.float64(1e-14):
            return atb, False
    # ``Q R x = AtA x = Atb`` with ``Q`` orthonormal, so back-substitute against ``Q^T Atb``.
    return solve_triangular(r, wp.transpose(q) * atb), True


@wp.func
def _eigvec_sym2(m00: wp.float32, m01: wp.float32, lam: wp.float32, fallback: wp.vec2) -> wp.vec2:
    """
    Return the unit eigenvector of a 2x2 matrix for the eigenvalue ``lam``.

    The matrix is ``[[m00, m01], [m10, m11]]`` and the result is expressed in the (t1, t2)
    tangent-frame basis. Uses the first-row form ``v ~ [m01, lam - m00]``, which is valid for
    any 2x2 (symmetric or the non-symmetric Weingarten map) since it depends only on the top
    row. ``fallback`` is returned when that row is degenerate (near-diagonal matrix).
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
      ``igl::principal_curvature``'s ``finalEigenStuff`` verbatim. libigl forces this symmetry,
      which keeps the trace (mean curvature) exact but alters the determinant (eigenvalue spread)
      and makes the result depend on the reference frame.

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
) -> None:
    """
    Fit a quadric surface in a local tangent frame per vertex.

    Extract principal curvature directions and magnitudes. Matches igl::principal_curvature.
    """
    i = wp.int32(wp.tid())
    zero3 = wp.vec3(0.0, 0.0, 0.0)

    start = offsets[i]
    # offsets has length n_vertices (no sentinel); last vertex ends at neighbor_indices end
    if i + 1 < offsets.shape[0]:
        end = offsets[i + 1]
    else:
        end = neighbor_indices.shape[0]
    n_nbr = end - start

    if (
        n_nbr < 5
    ):  # need at least 5 non-self neighbors for quadric fit; degenerate cases caught by gauss_elim
        out_pd1[i] = zero3
        out_pd2[i] = zero3
        out_pv1[i] = wp.float32(0.0)
        out_pv2[i] = wp.float32(0.0)
        return

    vertex = vertices[i]
    normal = wp.normalize(vertex_normals[i])

    # Build the tangent frame from the lowest-indexed mesh-adjacency neighbor, matching libigl's
    # computeReferenceFrame (adjacency_list[i][0]). When frame_independent is False, libigl extracts
    # curvature from a *symmetrized* shape operator whose eigenvalues depend on the chosen frame, so
    # the principal values only agree with igl::principal_curvature when this exact reference
    # direction is used. When frame_independent is True the eigenvalues are surface invariants, so
    # the exact frame is irrelevant (any orthonormal tangent basis yields the same result).
    ref = reference_neighbors[i]
    t1, t2 = _build_reference_frame(vertex, normal, vertices[ref])

    # Count neighbors passing projection-plane filter, including self (self always passes,
    # dot=1).
    # Matches libigl's applyProjOnPlane which includes vv[self] because dot(n_i, n_i) = 1 > 0.
    n_valid = wp.int32(0)
    for k in range(n_nbr):
        j = neighbor_indices[start + k]
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
        j = neighbor_indices[start + k]
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
        return

    # Cast the float64 solution back to float32 for the rest of the kernel.
    a = wp.float32(solution[0])
    b = wp.float32(solution[1])
    c = wp.float32(solution[2])
    d = wp.float32(solution[3])
    e = wp.float32(solution[4])

    # First fundamental form coefficients
    E_ff = wp.float32(1.0) + d * d  # noqa: N806
    F_ff = d * e  # noqa: N806
    G_ff = wp.float32(1.0) + e * e  # noqa: N806
    denom = E_ff * G_ff - F_ff * F_ff

    if wp.abs(denom) < wp.float32(1e-14):
        out_pd1[i] = zero3
        out_pd2[i] = zero3
        out_pv1[i] = wp.float32(0.0)
        out_pv2[i] = wp.float32(0.0)
        return

    # Normal z-component in local frame
    nz = wp.float32(1.0) / wp.sqrt(d * d + e * e + wp.float32(1.0))

    # Second fundamental form
    L_ff = wp.float32(2.0) * a * nz  # noqa: N806
    M_ff = b * nz  # noqa: N806
    N_ff = wp.float32(2.0) * c * nz  # noqa: N806

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


@wp.func
def line_ball_intersection_segment(
    start_point: wp.vec3, end_point: wp.vec3, center: wp.vec3, radius: wp.float32
) -> wp.float32:
    segment = end_point - start_point
    oc = start_point - center
    r = radius
    ldotl = wp.length_sq(segment)
    ldotoc = wp.dot(segment, oc)
    ocdotoc = wp.length_sq(oc)
    discrim = ldotoc * ldotoc - ldotl * (ocdotoc - r * r)

    if discrim <= wp.float32(0.0):
        return wp.float32(0.0)

    sqrt_discrim = wp.sqrt(discrim)
    d1 = (-ldotoc - sqrt_discrim) / ldotl
    d2 = (-ldotoc + sqrt_discrim) / ldotl

    d1 = wp.clamp(d1, wp.float32(0.0), wp.float32(1.0))
    d2 = wp.clamp(d2, wp.float32(0.0), wp.float32(1.0))

    return (d2 - d1) * wp.length(segment)


@wp.kernel
def edge_aabb_from_endpoints(
    vertices: wp.array[wp.vec3],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_lower: wp.array[wp.vec3],
    out_upper: wp.array[wp.vec3],
) -> None:
    tid = wp.int32(wp.tid())
    v0 = vertices[face_adjacency_edges[tid, 0]]
    v1 = vertices[face_adjacency_edges[tid, 1]]
    out_lower[tid] = wp.min(v0, v1)  # wp.min / wp.max on vectors are element-wise
    out_upper[tid] = wp.max(v0, v1)


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
    tid = wp.int32(wp.tid())
    edge_idx = candidate_edge_indices[tid]
    query_idx = kernel_array.binary_search_index(offsets, tid) - wp.int32(1)

    e0 = face_adjacency_edges[edge_idx, 0]
    e1 = face_adjacency_edges[edge_idx, 1]
    start_point = vertices[e0]
    end_point = vertices[e1]
    center = queries[query_idx]

    length = line_ball_intersection_segment(start_point, end_point, center, radius)
    angle = angles[edge_idx]
    sign = wp.where(convex[edge_idx], wp.float32(1.0), wp.float32(-1.0))

    wp.atomic_add(out_mean_curvature, query_idx, length * angle * sign * wp.float32(0.5))
