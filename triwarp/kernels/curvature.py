import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels.linalg import solve_normal_equations
from triwarp.kernels.predicates import unit_tangent
from triwarp.kernels.tangent_space import any_perpendicular

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
    # Same construction as ``tangent_space.vertex_tangent_frames``: the reference direction is the
    # neighbour projected into the tangent plane, falling back to an arbitrary perpendicular when
    # that projection vanishes (the neighbour sits on the normal line through the vertex).
    t1 = any_perpendicular(normal)
    tangential, length = unit_tangent(first_neighbor - vertex, normal, TOLERANCE_ZERO_CONSTANT)
    if length > TOLERANCE_ZERO_CONSTANT:
        t1 = tangential
    # ``normal`` is unit and ``t1`` is a unit vector orthogonal to it, so the cross product is
    # already unit and needs no normalization.
    return t1, wp.cross(normal, t1)


@wp.func
def _eigvec_2x2(
    m00: wp.float32, m01: wp.float32, m10: wp.float32, m11: wp.float32, lam: wp.float32
) -> wp.vec2:
    """
    Return the unit eigenvector of a 2x2 matrix for the eigenvalue ``lam``.

    The matrix is ``[[m00, m01], [m10, m11]]`` and the result is expressed in the (t1, t2)
    tangent-frame basis. ``m - lam*I`` is singular, so its two rows are proportional and each
    gives the eigenvector as its own perpendicular: row 0 gives ``[m01, lam - m00]`` and row 1
    gives ``[lam - m11, m10]``. Both forms are valid for any 2x2 (symmetric or the non-symmetric
    Weingarten map), but either row can vanish on its own, so the longer of the two is taken.

    Picking by *length* rather than against an absolute threshold is what makes a diagonal matrix
    come out right. There, ``m01 = 0`` and the eigenvalue equals one of the diagonal entries, so
    for that eigenvalue row 0 is the zero row up to rounding -- a few ULP of ``m00``, far above any
    fixed epsilon -- and normalizing it returns an arbitrarily-signed ``(0, +-1)`` instead of the
    correct ``(1, 0)``. Reading row 1 instead recovers it.

    ``(1, 0)`` is returned when both rows vanish, i.e. ``m == lam*I`` (an umbilic point, where
    every direction is a principal direction and there is nothing to find). That test is taken
    *relative* to the matrix's own magnitude, because these entries are curvatures and so scale as
    one over the mesh's: an absolute epsilon would call a large mesh's whole shape operator
    umbilic. Which arbitrary direction comes back does not matter, because the caller derives the
    second principal direction from this one rather than solving for it again.
    """
    row0 = wp.vec2(m01, lam - m00)
    row1 = wp.vec2(lam - m11, m10)
    v = wp.where(wp.length_sq(row0) >= wp.length_sq(row1), row0, row1)
    magnitude = wp.max(wp.abs(m00) + wp.abs(m01), wp.abs(m10) + wp.abs(m11))
    floor = TOLERANCE_ZERO_CONSTANT * magnitude
    if wp.length_sq(v) <= floor * floor:
        return wp.vec2(1.0, 0.0)
    return wp.normalize(v)


@wp.func
def _principal_curvatures_from_monge(
    first_form: wp.vec3, second_form: wp.vec3, frame_independent: wp.bool
) -> tuple[wp.float32, wp.float32, wp.vec2]:
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

    Returns (lam0, lam1, ev0) where lam0 <= lam1 are eigenvalues of ``m``. The caller negates them
    (libigl's ``c_val = -c_val``). ``ev0`` is ``lam0``'s unit eigenvector as a ``wp.vec2`` in the
    (t1, t2) tangent-frame 2D basis.

    **Only the first eigenvector is returned, deliberately.** Principal directions are orthogonal
    -- the shape operator is self-adjoint with respect to the first fundamental form -- so the
    second is the first rotated a quarter turn in the tangent plane, and the caller gets it from a
    cross product with the vertex normal. Solving for it separately instead costs the guarantee:
    the two solves see the same numerically degenerate matrix at an umbilic point and can land on
    the *same* direction, which is the one answer that is wrong however arbitrary the choice is
    allowed to be. Measured on ``icosphere(3)``, which is umbilic everywhere: 4 of 642 vertices
    returned ``PD1`` and ``PD2`` exactly parallel (their eigenvalue gap is exactly 0.0 in float32)
    and 4 more came back 6 degrees from parallel (gap one ULP, 2.4e-07). The non-symmetric
    ``frame_independent=True`` branch is worse, because its eigenvectors are orthogonal under the
    first fundamental form rather than in the frame's own coordinates: on the suite's ``half_torus``
    it put 266 of 544 pairs off perpendicular, up to ``|PD1 . PD2| = 0.57`` -- 35 degrees -- at
    eigenvalue gaps of order 1, where nothing is degenerate at all.
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

    # Eigenvector of (m - lam0*I)v = 0. lam1's is the caller's cross product -- see the docstring.
    return lam0, lam1, _eigvec_2x2(m00, m01, m10, m11, lam0)


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
    end = offsets[i + 1]  # ``geodesic_ball`` returns the terminated (n + 1) CSR row bounds
    n_nbr = end - start

    # The ball includes the centre vertex itself, which the fit loop below skips, so a determined
    # 5-parameter quadric fit needs 6 entries here. Anything short of that leaves the normal
    # equations rank-deficient and ``solve_normal_equations`` would report it -- this only saves
    # running a solve whose answer is already known. Remaining degeneracies are caught there.
    if n_nbr < 6:
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
    # The neighbour normal is not normalized here or in the fit loop below: only the *sign* of the
    # dot product is read, and scaling by a positive length cannot change it.
    n_valid = wp.int32(0)
    for k in range(n_nbr):
        j = neighbor_indices[start + k]
        if j == i:
            n_valid = n_valid + 1  # self always passes
            continue
        if wp.dot(vertex_normals[j], normal) > wp.float32(0.0):
            n_valid = n_valid + 1

    # Mirror libigl: only apply the filter if it leaves at least 6 neighbours. libigl also requires
    # the filtered set to be strictly smaller than the full one, which is not repeated here because
    # it cannot change the answer -- when every neighbour passes, filtering removes nothing.
    use_filter = n_valid >= 6

    # Least-squares quadric fit: accumulate the normal equations AᵀA x = Aᵀb.
    ata = mat55d()
    atb = vec5d()

    for k in range(n_nbr):
        j = neighbor_indices[start + k]
        if j == i:
            continue  # self contributes (0,0,0) — skip to avoid frame degeneration
        if use_filter and wp.dot(vertex_normals[j], normal) <= wp.float32(0.0):
            continue

        diff = vertices[j] - vertex
        u = wp.float64(wp.dot(diff, t1))
        v_c = wp.float64(wp.dot(diff, t2))
        w = wp.float64(wp.dot(diff, normal))

        # row of A: [u², uv, v², u, v]
        r = vec5d(u * u, u * v_c, v_c * v_c, u, v_c)
        ata += wp.outer(r, r)
        atb += r * w

    solution, ok = solve_normal_equations(ata, atb)
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

    # First fundamental form coefficients. Its determinant needs no degeneracy guard: it expands
    # to (1 + d*d)(1 + e*e) - (d*e)^2 = 1 + d*d + e*e, which is >= 1 for every finite fit, so
    # ``_principal_curvatures_from_monge`` can divide by it unconditionally.
    E_ff = wp.float32(1.0) + d * d  # noqa: N806
    F_ff = d * e  # noqa: N806
    G_ff = wp.float32(1.0) + e * e  # noqa: N806

    # Normal z-component in local frame
    nz = wp.float32(1.0) / wp.sqrt(d * d + e * e + wp.float32(1.0))

    # Second fundamental form
    L_ff = wp.float32(2.0) * a * nz  # noqa: N806
    M_ff = b * nz  # noqa: N806
    N_ff = wp.float32(2.0) * c * nz  # noqa: N806

    first_form = wp.vec3(E_ff, F_ff, G_ff)
    second_form = wp.vec3(L_ff, M_ff, N_ff)
    lam0, lam1, ev0 = _principal_curvatures_from_monge(first_form, second_form, frame_independent)

    # Negate: the Monge patch height function curves downward for convex surfaces,
    # giving negative eigenvalues; convention is positive curvature for convex.
    k0 = -lam0
    k1 = -lam1

    # Reconstruct the global direction from the local eigenvector, and take the second as its
    # quarter turn in the tangent plane. (t1, t2, normal) is orthonormal, so the cross product is
    # already unit; it is also what guarantees the pair is a *frame* rather than two independently
    # solved vectors that can coincide -- see ``_principal_curvatures_from_monge``.
    dir0 = wp.normalize(t1 * ev0[0] + t2 * ev0[1])
    dir1 = wp.cross(normal, dir0)

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
