import warp as wp

from triwarp.kernels.array import sort3
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.triangles import face_vertices


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
