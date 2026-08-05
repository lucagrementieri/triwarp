import math

import warp as wp

from triwarp.kernels.array import sort3, to_vec3d
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.triangles import face_vertices, face_vertices_vec3d

TWO_PI_F64 = wp.constant(wp.float64(2.0 * math.pi))


@wp.func
def squared_edge_lengths(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3
) -> tuple[wp.float32, wp.float32, wp.float32]:
    l2_0 = wp.length_sq(v1 - v2)
    l2_1 = wp.length_sq(v2 - v0)
    l2_2 = wp.length_sq(v0 - v1)
    return l2_0, l2_1, l2_2


@wp.func
def doublearea_from_lengths(l0: wp.float32, l1: wp.float32, l2: wp.float32) -> wp.float32:
    # Kahan's numerically stable Heron form needs the sides sorted ascending.
    l0, l1, l2 = sort3(l0, l1, l2)
    arg = (l0 + (l1 + l2)) * (l2 - (l0 - l1)) * (l2 + (l0 - l1)) * (l0 + (l1 - l2))
    dbl_area = wp.float32(0.5) * wp.sqrt(wp.max(arg, wp.float32(0.0)))
    if wp.isnan(dbl_area):
        return wp.float32(0.0)
    return dbl_area


@wp.func
def cot_entries_from_l2(
    l2_0: wp.float32, l2_1: wp.float32, l2_2: wp.float32, dbl_area: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    # A zero-area triangle contributes nothing rather than an infinity. Its angles are 0 or pi, so
    # it has no finite cotangent, and ``doublearea_from_lengths`` deliberately reports 0.0 for one:
    # without this guard that 0 divides straight through to +-inf, and a *single* collapsed face
    # poisons the whole assembled operator -- and every solve against it -- with NaN.
    #
    # The test is against exact zero, not a tolerance. ``dbl_area`` is already clamped
    # non-negative, so this changes results only where they used to be non-finite; a merely
    # sliver triangle still yields its (huge, finite) weight, because that is ill-conditioning
    # rather than a division by zero and the fix for it is mollification -- see
    # ``laplacian.robust_laplacian``.
    denominator = wp.float32(4.0) * dbl_area
    if denominator <= wp.float32(0.0):
        return wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)
    c0 = (l2_1 + l2_2 - l2_0) / denominator
    c1 = (l2_2 + l2_0 - l2_1) / denominator
    c2 = (l2_0 + l2_1 - l2_2) / denominator
    return c0, c1, c2


@wp.func
def cot_entries_from_edge_lengths(
    l0: wp.float32, l1: wp.float32, l2: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    l2_0 = l0 * l0
    l2_1 = l1 * l1
    l2_2 = l2 * l2
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    return cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)


@wp.kernel
def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cot: wp.array2d[wp.Float]
) -> None:
    # ``out_cot`` is generic: the half-cotangent weights are computed in float32 (the vertex
    # precision) and cast to the requested output dtype (float32 or float64) at store time.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(f))
    l2_0, l2_1, l2_2 = squared_edge_lengths(v0, v1, v2)
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    c0, c1, c2 = cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)
    out_cot[f, 0] = type(out_cot[f, 0])(c0)
    out_cot[f, 1] = type(out_cot[f, 1])(c1)
    out_cot[f, 2] = type(out_cot[f, 2])(c2)


@wp.kernel
def cotmatrix_entries_intrinsic(
    edge_lengths: wp.array2d[wp.float32], out_cot: wp.array2d[wp.Float]
) -> None:
    f = int(wp.tid())
    l0 = edge_lengths[f, 0]
    l1 = edge_lengths[f, 1]
    l2 = edge_lengths[f, 2]
    c0, c1, c2 = cot_entries_from_edge_lengths(l0, l1, l2)
    out_cot[f, 0] = type(out_cot[f, 0])(c0)
    out_cot[f, 1] = type(out_cot[f, 1])(c1)
    out_cot[f, 2] = type(out_cot[f, 2])(c2)


@wp.func
def edge_weight(
    a: wp.int32, b: wp.int32, vertices: wp.array[wp.vec3], equal_weight: wp.int32
) -> wp.float32:
    if equal_weight != 0:
        return wp.float32(1.0)
    return wp.float32(1.0) / (wp.length(vertices[a] - vertices[b]) + wp.float32(1.0e-12))


@wp.kernel
def laplacian_triplets_directed(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # One triplet per directed triangle edge, matching trimesh's ``mesh.edges`` adjacency.
    # ``out_vals`` is generic: the float32 edge weight is cast to the requested output dtype.
    e = int(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    out_rows[e] = a
    out_cols[e] = b
    out_vals[e] = type(out_vals[e])(edge_weight(a, b, vertices, equal_weight))


@wp.kernel
def laplacian_triplets_symmetric(
    edges: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    equal_weight: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # Emits both directed pairs (a, b) and (b, a) from each unique undirected edge so the
    # adjacency is symmetric, matching trimesh's ``vertex_neighbors``. Duplicate multiplicity
    # cancels under row-normalization. ``out_vals`` is generic (float32 or float64).
    e = int(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    w = type(out_vals[e * 2])(edge_weight(a, b, vertices, equal_weight))
    base = e * 2
    out_rows[base + 0] = a
    out_cols[base + 0] = b
    out_vals[base + 0] = w
    out_rows[base + 1] = b
    out_cols[base + 1] = a
    out_vals[base + 1] = w


@wp.kernel
def row_normalize(offsets: wp.array[wp.int32], out_values: wp.array[wp.Float]) -> None:
    i = int(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    # ``out_values[0]`` is always valid: the launcher only runs this kernel when nnz > 0. It is
    # read solely to source the generic scalar type for the accumulator / zero literals.
    total = type(out_values[0])(0.0)
    for k in range(start, end):
        total += out_values[k]
    if total > type(out_values[0])(0.0):
        for k in range(start, end):
            out_values[k] = out_values[k] / total


@wp.kernel
def apply_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    v_in: wp.array[wp.vec3d],
    out_lv: wp.array[wp.vec3d],
) -> None:
    i = int(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        # Isolated vertex (empty row): the averaging operator acts as the identity so the
        # vertex does not drift toward the origin.
        out_lv[i] = v_in[i]
        return
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(start, end):
        w = wp.float64(values[k])
        acc += w * v_in[columns[k]]
    out_lv[i] = acc


@wp.kernel
def cotmatrix_triplets(
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # Emits the 12 cotangent-Laplacian COO triplets per triangle. ``cot_entries`` and ``out_vals``
    # are independent generic float types: the half-cotangent weights (typically float32, the
    # vertex precision) are cast to the requested matrix dtype, so a single ``bsr_from_triplets``
    # builds a float32 or float64 matrix natively. Building float64 values here (rather than
    # recasting a float32 matrix) dodges a Warp ``bsr_mm`` bug (still present in 1.15.0) triggered
    # by a second ``bsr_from_triplets`` rebuild — see issue_report.md.
    f = int(wp.tid())
    for e in range(3):
        c0 = (e + 1) % 3
        c1 = (e + 2) % 3
        source = faces[f * 3 + c0]
        dest = faces[f * 3 + c1]
        base = f * 12 + e * 4
        w = type(out_vals[base])(cot_entries[f, e])
        out_rows[base + 0] = source
        out_cols[base + 0] = dest
        out_vals[base + 0] = w
        out_rows[base + 1] = dest
        out_cols[base + 1] = source
        out_vals[base + 1] = w
        out_rows[base + 2] = source
        out_cols[base + 2] = source
        out_vals[base + 2] = -w
        out_rows[base + 3] = dest
        out_cols[base + 3] = dest
        out_vals[base + 3] = -w


@wp.func
def rotation22(angle: wp.float32) -> wp.mat22d:
    # Real 2x2 form of the unit complex number ``exp(i * angle)``: the rotation that re-expresses a
    # tangent vector in a neighbour's frame.
    c = wp.float64(wp.cos(angle))
    s = wp.float64(wp.sin(angle))
    return wp.mat22d(c, -s, s, c)


@wp.kernel
def connection_laplacian_triplets(
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    transport_angles: wp.array[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.mat22d],
) -> None:
    # The vector Laplacian's 12 block triplets per triangle, laid out exactly like
    # ``cotmatrix_triplets`` but with each off-diagonal weight turned into a rotation: the two
    # endpoints of an edge measure tangent directions from different reference directions, so a
    # difference between them is only meaningful after transporting one into the other's frame.
    #
    # Built positive semi-definite (positive diagonal), unlike ``cotmatrix``'s igl sign convention,
    # because every consumer here feeds it straight to a conjugate-gradient solve.
    f = int(wp.tid())
    identity = wp.mat22d(1.0, 0.0, 0.0, 1.0)
    for e in range(3):
        # Corner ``e``'s half-cotangent weights the opposite edge, which is halfedge ``e + 1``.
        h = f * 3 + (e + 1) % 3
        i = faces[h]
        j = halfedge_destination(faces, h)
        w = wp.float64(cot_entries[f, e])
        rho = transport_angles[h]
        base = f * 12 + e * 4

        out_rows[base + 0] = i
        out_cols[base + 0] = j
        out_vals[base + 0] = -w * rotation22(-rho)
        out_rows[base + 1] = j
        out_cols[base + 1] = i
        out_vals[base + 1] = -w * rotation22(rho)
        out_rows[base + 2] = i
        out_cols[base + 2] = i
        out_vals[base + 2] = w * identity
        out_rows[base + 3] = j
        out_cols[base + 3] = j
        out_vals[base + 3] = w * identity


@wp.kernel
def triangle_inequality_slack(
    edge_lengths: wp.array2d[wp.float32], epsilon: wp.float32, out_slack: wp.array[wp.float32]
) -> None:
    # How far this triangle is from satisfying the strict triangle inequality with margin
    # ``epsilon``, expressed as the constant that would have to be added to all three of its edges.
    # Adding a constant lengthens the two short sides by ``2 * delta`` against the long side's
    # ``delta``, so half the shortfall is enough.
    f = int(wp.tid())
    a = edge_lengths[f, 0]
    b = edge_lengths[f, 1]
    c = edge_lengths[f, 2]
    worst = wp.max(wp.max(epsilon - (a + b - c), epsilon - (b + c - a)), epsilon - (c + a - b))
    out_slack[f] = wp.max(worst, 0.0) * 0.5


@wp.func
def add_constant(length: wp.float32, delta: wp.float32) -> wp.float32:
    return length + delta


@wp.func
def reciprocal_or_zero(value: wp.Float) -> wp.Float:
    # ``igl::invert_diag`` semantics: zero entries stay zero instead of becoming infinities, so a
    # killed degree of freedom (boundary vertex, empty edge row) simply contributes nothing.
    if value > type(value)(0.0):
        return type(value)(1.0) / value
    return type(value)(0.0)


@wp.kernel
def sandwich_row_counts(
    a_offsets: wp.array[wp.int32],
    b_offsets: wp.array[wp.int32],
    inv_mass: wp.array[wp.Float],
    out_counts: wp.array[wp.int32],
) -> None:
    # Triplets emitted by row ``t`` of the product ``A diag(inv_mass) B``: the full outer product
    # of the two CSR rows, or nothing when the diagonal weight is zero.
    t = int(wp.tid())
    if inv_mass[t] > type(inv_mass[0])(0.0):
        out_counts[t] = (a_offsets[t + 1] - a_offsets[t]) * (b_offsets[t + 1] - b_offsets[t])
    else:
        out_counts[t] = 0


@wp.kernel
def sandwich_row_triplets(
    a_offsets: wp.array[wp.int32],
    a_columns: wp.array[wp.int32],
    a_values: wp.array[wp.Float],
    b_offsets: wp.array[wp.int32],
    b_columns: wp.array[wp.int32],
    b_values: wp.array[wp.Float],
    inv_mass: wp.array[wp.Float],
    segment_offsets: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``(A diag(d) B)_ij = sum_t A_ti d_t B_tj``; with both operands symmetric ``A_ti`` is row
    # ``t``'s entry at column ``i``, so each row of the diagonal sandwich is the scaled outer
    # product of the two matching CSR rows. This assembles the product without ``bsr_mm``, whose
    # chained form is nondeterministic on CUDA — see issue_report.md.
    t = int(wp.tid())
    weight = inv_mass[t]
    if weight <= type(inv_mass[0])(0.0):
        return
    cursor = int(segment_offsets[t])
    for a in range(a_offsets[t], a_offsets[t + 1]):
        row = a_columns[a]
        left = a_values[a] * weight
        for b in range(b_offsets[t], b_offsets[t + 1]):
            out_rows[cursor] = row
            out_cols[cursor] = b_columns[b]
            out_vals[cursor] = left * b_values[b]
            cursor += 1


@wp.kernel
def hessian_corner_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_gradients: wp.array[wp.vec3d],
    out_areas: wp.array[wp.float64],
) -> None:
    # Gradient of each corner's hat function inside its face, ``n x e_c / (2 A)`` with ``e_c`` the
    # CCW edge opposite corner ``c``. A degenerate face gets zero gradients (it contributes
    # nothing to the energy) rather than a division by its zero area.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    normal = wp.cross(v1 - v0, v2 - v0)
    dbl_area = wp.length(normal)
    out_areas[f] = wp.float64(0.5) * dbl_area
    zero = wp.float64(0.0)
    g0 = wp.vec3d(zero, zero, zero)
    g1 = wp.vec3d(zero, zero, zero)
    g2 = wp.vec3d(zero, zero, zero)
    if dbl_area > zero:
        unit = normal / dbl_area
        g0 = wp.cross(unit, v2 - v1) / dbl_area
        g1 = wp.cross(unit, v0 - v2) / dbl_area
        g2 = wp.cross(unit, v1 - v0) / dbl_area
    out_gradients[f * 3 + 0] = g0
    out_gradients[f * 3 + 1] = g1
    out_gradients[f * 3 + 2] = g2


@wp.kernel
def voronoi_mass(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_mass: wp.array[wp.float64]
) -> None:
    # The ``igl::massmatrix`` MASSMATRIX_TYPE_VORONOI lumping: true Voronoi quad areas on
    # non-obtuse triangles, the 1/2 : 1/4 : 1/4 split on obtuse ones (the obtuse corner gets the
    # half). A degenerate face contributes nothing (igl would emit NaN).
    f = int(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    l2_0 = wp.length_sq(v1 - v2)
    l2_1 = wp.length_sq(v2 - v0)
    l2_2 = wp.length_sq(v0 - v1)
    dbl_area = wp.length(wp.cross(v1 - v0, v2 - v0))
    if dbl_area <= wp.float64(0.0):
        return
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    two = wp.float64(2.0)
    cos0 = (l2_2 + l2_1 - l2_0) / (two * l1 * l2)
    cos1 = (l2_0 + l2_2 - l2_1) / (two * l2 * l0)
    cos2 = (l2_1 + l2_0 - l2_2) / (two * l0 * l1)
    bary0 = cos0 * l0
    bary1 = cos1 * l1
    bary2 = cos2 * l2
    total = bary0 + bary1 + bary2
    half_dbl = wp.float64(0.5) * dbl_area
    partial0 = bary0 / total * half_dbl
    partial1 = bary1 / total * half_dbl
    partial2 = bary2 / total * half_dbl
    quad0 = wp.float64(0.5) * (partial1 + partial2)
    quad1 = wp.float64(0.5) * (partial2 + partial0)
    quad2 = wp.float64(0.5) * (partial0 + partial1)
    # At most one angle can be obtuse, so the three rewrites cannot both fire.
    if cos0 < wp.float64(0.0):
        quad0 = wp.float64(0.25) * dbl_area
        quad1 = wp.float64(0.125) * dbl_area
        quad2 = wp.float64(0.125) * dbl_area
    if cos1 < wp.float64(0.0):
        quad0 = wp.float64(0.125) * dbl_area
        quad1 = wp.float64(0.25) * dbl_area
        quad2 = wp.float64(0.125) * dbl_area
    if cos2 < wp.float64(0.0):
        quad0 = wp.float64(0.125) * dbl_area
        quad1 = wp.float64(0.125) * dbl_area
        quad2 = wp.float64(0.25) * dbl_area
    wp.atomic_add(out_mass, faces[f * 3 + 0], quad0)
    wp.atomic_add(out_mass, faces[f * 3 + 1], quad1)
    wp.atomic_add(out_mass, faces[f * 3 + 2], quad2)


@wp.kernel
def zero_at_indices(indices: wp.array[wp.int32], out_values: wp.array[wp.Float]) -> None:
    i = int(wp.tid())
    out_values[indices[i]] = type(out_values[0])(0.0)


@wp.kernel
def hessian_energy_counts(
    vf_offsets: wp.array[wp.int32], inv_mass: wp.array[wp.float64], out_counts: wp.array[wp.int32]
) -> None:
    # Vertex ``k`` couples every ordered pair of its incident faces, 3 x 3 corners each; a killed
    # degree of freedom (boundary vertex) emits nothing.
    k = int(wp.tid())
    if inv_mass[k] > wp.float64(0.0):
        degree = vf_offsets[k + 1] - vf_offsets[k]
        out_counts[k] = 9 * degree * degree
    else:
        out_counts[k] = 0


@wp.func
def face_corner_of_vertex(faces: wp.array[wp.int32], face: wp.int32, vertex: wp.int32) -> wp.int32:
    if faces[face * 3 + 0] == vertex:
        return wp.int32(0)
    if faces[face * 3 + 1] == vertex:
        return wp.int32(1)
    return wp.int32(2)


@wp.kernel
def hessian_energy_triplets(
    faces: wp.array[wp.int32],
    vf_offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    gradients: wp.array[wp.vec3d],
    areas: wp.array[wp.float64],
    inv_mass: wp.array[wp.float64],
    segment_offsets: wp.array[wp.int32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # ``Q = H^T Minv H`` contracted analytically over the 9 component pairs of the stacked
    # Hessian: ``Q_ij = sum_k Minv_k sum_{f,g ni k} A_f A_g (g_fk . g_gk)(g_fi . g_gj)`` with
    # ``g_fc`` the hat-function gradient of corner ``c`` in face ``f``. One thread per vertex
    # ``k``; the per-thread work is quadratic in the vertex's valence.
    k = int(wp.tid())
    weight = inv_mass[k]
    if weight <= wp.float64(0.0):
        return
    start = vf_offsets[k]
    end = vf_offsets[k + 1]
    cursor = int(segment_offsets[k])
    for a in range(start, end):
        f = vertex_faces[a]
        corner_f = face_corner_of_vertex(faces, f, wp.int32(k))
        gradient_fk = gradients[f * 3 + corner_f]
        left = weight * areas[f]
        for b in range(start, end):
            g = vertex_faces[b]
            corner_g = face_corner_of_vertex(faces, g, wp.int32(k))
            pair = left * areas[g] * wp.dot(gradient_fk, gradients[g * 3 + corner_g])
            for ci in range(3):
                row = faces[f * 3 + ci]
                gradient_fi = gradients[f * 3 + ci]
                for cj in range(3):
                    out_rows[cursor] = row
                    out_cols[cursor] = faces[g * 3 + cj]
                    out_vals[cursor] = type(out_vals[0])(
                        pair * wp.dot(gradient_fi, gradients[g * 3 + cj])
                    )
                    cursor += 1


@wp.kernel
def internal_angles_and_sums(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_angles: wp.array2d[wp.float64],
    out_angle_sums: wp.array[wp.float64],
) -> None:
    # Interior angle at each corner (law of cosines, float64) plus the per-vertex angle sum the
    # curvature correction normalizes by. ``wp.acos`` clamps its argument, so a sliver face yields
    # 0 / pi rather than NaN.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    l2_0 = wp.length_sq(v1 - v2)
    l2_1 = wp.length_sq(v2 - v0)
    l2_2 = wp.length_sq(v0 - v1)
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    two = wp.float64(2.0)
    theta0 = wp.acos((l2_1 + l2_2 - l2_0) / (two * l1 * l2))
    theta1 = wp.acos((l2_2 + l2_0 - l2_1) / (two * l2 * l0))
    theta2 = wp.acos((l2_0 + l2_1 - l2_2) / (two * l0 * l1))
    out_angles[f, 0] = theta0
    out_angles[f, 1] = theta1
    out_angles[f, 2] = theta2
    wp.atomic_add(out_angle_sums, faces[f * 3 + 0], theta0)
    wp.atomic_add(out_angle_sums, faces[f * 3 + 1], theta1)
    wp.atomic_add(out_angle_sums, faces[f * 3 + 2], theta2)


@wp.func
def angle_defect_from_sum(angle_sum: wp.float64) -> wp.float64:
    return TWO_PI_F64 - angle_sum


@wp.func
def divide_or_zero(numerator: wp.float64, denominator: wp.float64) -> wp.float64:
    if denominator > wp.float64(0.0):
        return numerator / denominator
    return wp.float64(0.0)


@wp.kernel
def scatter_edge_halfedges(
    inverse: wp.array[wp.int32], cursor: wp.array[wp.int32], out_halfedges: wp.array2d[wp.int32]
) -> None:
    # Up to two halfedges per unique edge, in arbitrary order. On a non-edge-manifold edge the
    # third and later halfedges are dropped; the Crouzeix-Raviart discretization (like
    # ``igl::crouzeix_raviart_*``, which asserts edge-manifoldness) is undefined there.
    h = int(wp.tid())
    slot = wp.atomic_add(cursor, inverse[h], 1)
    if slot < 2:
        out_halfedges[inverse[h], slot] = h


@wp.kernel
def cr_gradient_rows(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: wp.array2d[wp.int32],
    edge_halfedges: wp.array2d[wp.int32],
    out_vertex_slots: wp.array2d[wp.int32],
    out_par: wp.array2d[wp.float64],
    out_perp: wp.array2d[wp.float64],
) -> None:
    # One row pair of ``igl::scalar_to_cr_vector_gradient`` per unique edge: the parallel and
    # perpendicular components of the CR gradient operator, supported on at most 4 vertices (the
    # edge's endpoints in slots 0/1, each incident face's apex in slot 2/3, ``-1`` when absent).
    # The edge's positive orientation is min-vertex-first, matching ``unique_edges``' sorted rows;
    # the assembled energy is invariant to that gauge choice.
    u = int(wp.tid())
    a = unique_edges[u, 0]
    out_vertex_slots[u, 0] = a
    out_vertex_slots[u, 1] = unique_edges[u, 1]
    for t in range(2):
        h = edge_halfedges[u, t]
        if h < 0:
            continue
        f = h // 3
        s = h % 3
        i = faces[f * 3 + s]
        j = faces[f * 3 + (s + 1) % 3]
        k = faces[f * 3 + (s + 2) % 3]
        vi = to_vec3d(vertices[i])
        vj = to_vec3d(vertices[j])
        vk = to_vec3d(vertices[k])
        eij = wp.length_sq(vi - vj)
        ejk = wp.length_sq(vj - vk)
        eki = wp.length_sq(vk - vi)
        dbl_area = wp.length(wp.cross(vj - vi, vk - vi))
        sqrt_eij = wp.sqrt(eij)
        if dbl_area <= wp.float64(0.0) or sqrt_eij <= wp.float64(0.0):
            continue
        orientation = wp.float64(-1.0)
        slot_i = 1
        slot_j = 0
        if i == a:
            orientation = wp.float64(1.0)
            slot_i = 0
            slot_j = 1
        six_sqrt = wp.float64(6.0) * sqrt_eij
        twelve_sqrt = wp.float64(12.0) * sqrt_eij
        out_par[u, slot_i] = out_par[u, slot_i] - orientation * dbl_area / six_sqrt
        out_perp[u, slot_i] = out_perp[u, slot_i] - orientation * (eij + ejk - eki) / twelve_sqrt
        out_par[u, slot_j] = out_par[u, slot_j] + orientation * dbl_area / six_sqrt
        out_perp[u, slot_j] = out_perp[u, slot_j] - orientation * (eij - ejk + eki) / twelve_sqrt
        out_vertex_slots[u, 2 + t] = k
        out_perp[u, 2 + t] = orientation * sqrt_eij / wp.float64(6.0)


@wp.func
def select3_int(value0: wp.int32, value1: wp.int32, value2: wp.int32, index: wp.int32) -> wp.int32:
    if index == 0:
        return value0
    if index == 1:
        return value1
    return value2


@wp.func
def select3_f64(
    value0: wp.float64, value1: wp.float64, value2: wp.float64, index: wp.int32
) -> wp.float64:
    if index == 0:
        return value0
    if index == 1:
        return value1
    return value2


@wp.func
def curved_pair_terms(
    eij: wp.float64,
    ejk: wp.float64,
    eki: wp.float64,
    dbl_area: wp.float64,
    o2: wp.float64,
    ki: wp.float64,
    kj: wp.float64,
    kk: wp.float64,
) -> tuple[wp.float64, wp.float64, wp.float64]:
    # Per edge-slot terms of ``igl::cr_vector_laplacian + cr_vector_curvature_correction``: the
    # diagonal weight, the parallel-parallel coupling to the previous slot's edge, and the
    # parallel-perpendicular one. ``eij``/``ejk``/``eki`` are squared lengths.
    lens = wp.sqrt(eij * eki)
    base = eij - ejk + eki
    two = wp.float64(2.0)
    diag = two * eij / dbl_area + (ki + kj + kk)
    cos_div = base / (two * lens)
    sin_div = wp.sqrt(
        wp.max(wp.float64(0.0), wp.float64(1.0) - base * base / (wp.float64(4.0) * eij * eki))
    )
    kcomb = ki - kj - kk
    off_pp = o2 * base * base / (two * lens * dbl_area) - o2 * kcomb * cos_div
    off_pq = -o2 * base / lens + o2 * kcomb * sin_div
    return diag, off_pp, off_pq


@wp.func
def halfedge_orientation(faces: wp.array[wp.int32], halfedge: wp.int32) -> wp.float64:
    # +1 when the halfedge runs from the smaller vertex index to the larger, i.e. agrees with the
    # min-first row ``edges_unique`` stores for its edge.
    if faces[halfedge] < halfedge_destination(faces, halfedge):
        return wp.float64(1.0)
    return wp.float64(-1.0)


@wp.kernel
def curved_hessian_triplets(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    angles: wp.array2d[wp.float64],
    scaled_kappa: wp.array[wp.float64],
    inv_mass: wp.array[wp.float64],
    edge_vertex_slots: wp.array2d[wp.int32],
    par: wp.array2d[wp.float64],
    perp: wp.array2d[wp.float64],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # Per face, the sandwich ``D^T Mi (L + K) Mi D`` restricted to this face's 6x6 block of
    # ``L + K`` (both operators couple edges within a face only): 9 ordered edge-slot pairs, each a
    # 2x2 block over the parallel/perpendicular components, contracted against the two edges' CR
    # gradient rows (up to 4 vertices each) -> a fixed 144 triplets per face, zero-padded.
    f = int(wp.tid())
    zero = wp.float64(0.0)
    base_out = f * 144
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    dbl_area = wp.length(wp.cross(v1 - v0, v2 - v0))
    if dbl_area <= zero:
        for empty in range(144):
            out_rows[base_out + empty] = 0
            out_cols[base_out + empty] = 0
            out_vals[base_out + empty] = type(out_vals[0])(0.0)
        return
    # Squared edge lengths, column e opposite corner e (the igl intrinsic convention).
    l2_0 = wp.length_sq(v1 - v2)
    l2_1 = wp.length_sq(v2 - v0)
    l2_2 = wp.length_sq(v0 - v1)
    # Curvature ingredients per corner c: scaledKappa(F(f,c)) * theta(f,c).
    kv0 = scaled_kappa[faces[f * 3 + 0]] * angles[f, 0]
    kv1 = scaled_kappa[faces[f * 3 + 1]] * angles[f, 1]
    kv2 = scaled_kappa[faces[f * 3 + 2]] * angles[f, 2]
    # igl edge slot e is triwarp halfedge (e + 1) % 3; o2[e] = oE(f,e) * oE(f,(e+2)%3).
    oh0 = halfedge_orientation(faces, wp.int32(f * 3 + 0))
    oh1 = halfedge_orientation(faces, wp.int32(f * 3 + 1))
    oh2 = halfedge_orientation(faces, wp.int32(f * 3 + 2))
    eid0 = inverse[f * 3 + 1]
    eid1 = inverse[f * 3 + 2]
    eid2 = inverse[f * 3 + 0]
    diag0, pp0, pq0 = curved_pair_terms(l2_0, l2_1, l2_2, dbl_area, oh0 * oh1, kv1, kv2, kv0)
    diag1, pp1, pq1 = curved_pair_terms(l2_1, l2_2, l2_0, dbl_area, oh1 * oh2, kv2, kv0, kv1)
    diag2, pp2, pq2 = curved_pair_terms(l2_2, l2_0, l2_1, dbl_area, oh2 * oh0, kv0, kv1, kv2)
    for alpha in range(3):
        edge_a = select3_int(eid0, eid1, eid2, wp.int32(alpha))
        mi_a = inv_mass[edge_a]
        for beta in range(3):
            edge_b = select3_int(eid0, eid1, eid2, wp.int32(beta))
            mimi = mi_a * inv_mass[edge_b]
            b_pp = zero
            b_pq = zero
            b_qp = zero
            if alpha == beta:
                b_pp = select3_f64(diag0, diag1, diag2, wp.int32(alpha))
            elif beta == (alpha + 2) % 3:
                b_pp = select3_f64(pp0, pp1, pp2, wp.int32(alpha))
                b_pq = select3_f64(pq0, pq1, pq2, wp.int32(alpha))
                b_qp = -b_pq
            else:
                b_pp = select3_f64(pp0, pp1, pp2, wp.int32(beta))
                b_qp = select3_f64(pq0, pq1, pq2, wp.int32(beta))
                b_pq = -b_qp
            pair_out = base_out + (alpha * 3 + beta) * 16
            for u_slot in range(4):
                row = edge_vertex_slots[edge_a, u_slot]
                par_u = par[edge_a, u_slot]
                perp_u = perp[edge_a, u_slot]
                for v_slot in range(4):
                    col = edge_vertex_slots[edge_b, v_slot]
                    position = pair_out + u_slot * 4 + v_slot
                    value = zero
                    out_row = 0
                    out_col = 0
                    if row >= 0 and col >= 0:
                        par_v = par[edge_b, v_slot]
                        perp_v = perp[edge_b, v_slot]
                        value = mimi * (
                            b_pp * (par_u * par_v + perp_u * perp_v)
                            + b_pq * par_u * perp_v
                            + b_qp * perp_u * par_v
                        )
                        out_row = row
                        out_col = col
                    out_rows[position] = out_row
                    out_cols[position] = out_col
                    out_vals[position] = type(out_vals[0])(value)


@wp.kernel
def crouzeix_raviart_cotmatrix_triplets(
    inverse: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.Float],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.Float],
) -> None:
    # 12 triplets per face: for each corner, the two edges meeting there couple with minus four
    # times the corner's half-cotangent, and each gains the same amount on its diagonal
    # (``igl::crouzeix_raviart_cotmatrix``'s triangle table). Edge ids come from
    # ``edges_unique``'s inverse; triwarp halfedge slot ``s`` spans corners ``s -> s+1``, so the
    # edge opposite corner ``c`` is halfedge ``(c + 1) % 3``.
    f = int(wp.tid())
    for c in range(3):
        edge_1 = inverse[f * 3 + (c + 1) % 3]
        edge_2 = inverse[f * 3 + (c + 2) % 3]
        weight = type(out_vals[0])(4.0) * type(out_vals[0])(cot_entries[f, (c + 2) % 3])
        base = f * 12 + c * 4
        out_rows[base + 0] = edge_1
        out_cols[base + 0] = edge_2
        out_vals[base + 0] = -weight
        out_rows[base + 1] = edge_2
        out_cols[base + 1] = edge_1
        out_vals[base + 1] = -weight
        out_rows[base + 2] = edge_1
        out_cols[base + 2] = edge_1
        out_vals[base + 2] = weight
        out_rows[base + 3] = edge_2
        out_cols[base + 3] = edge_2
        out_vals[base + 3] = weight


@wp.kernel
def crouzeix_raviart_mass_diag(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_mass: wp.array[wp.Float],
) -> None:
    # Each face donates a third of its area to each of its three edges — both
    # ``igl::crouzeix_raviart_massmatrix``'s diagonal and (per parallel/perpendicular component)
    # ``igl::cr_vector_mass``'s.
    f = int(wp.tid())
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    third = type(out_mass[0])(wp.length(wp.cross(v1 - v0, v2 - v0)) / wp.float64(6.0))
    wp.atomic_add(out_mass, inverse[f * 3 + 0], third)
    wp.atomic_add(out_mass, inverse[f * 3 + 1], third)
    wp.atomic_add(out_mass, inverse[f * 3 + 2], third)
