import warp as wp

from triwarp.kernels.array import OverloadTable
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.predicates import doublearea_from_lengths, squared_edge_lengths
from triwarp.kernels.triangles import face_vertices, row_triple

# ``cot_entries_from_l2`` and ``cot_entries_from_edge_lengths`` stay here rather than joining
# ``squared_edge_lengths`` / ``doublearea_from_lengths`` in ``kernels/predicates.py``: a cotangent
# weight is not a general triangle quantity, it is this operator's own entry, and the zero-area
# reasoning below is numerical defence *of the Laplacian* that belongs beside the thing it defends.


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
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
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
    f = wp.int32(wp.tid())
    l0, l1, l2 = row_triple(edge_lengths, f)
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
    e = wp.int32(wp.tid())
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
    e = wp.int32(wp.tid())
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
    i = wp.int32(wp.tid())
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


# One row of the row-stochastic averaging operator, as a ``@wp.func`` so a consumer can apply it
# and use the result in the same thread instead of round-tripping an intermediate buffer through
# global memory and a second launch. Every smoothing filter that iterates ``L`` does exactly that
# (see ``triwarp/smoothing.py``), so the row apply is the shared run rather than the kernel.
#
# Note the precision: the float32 weight is promoted to float64 because the accumulator is a
# ``wp.vec3d``. ``kernels/smoothing.diffuse_scalar_pass`` walks the same row on a float32 scalar
# field and accumulates in float32, so it cannot call this -- ``float64 * float32`` does not parse,
# and the float64-field form that would let one generic serve both was measured at 0.63-0.86x.
# That comment carries the numbers.


@wp.func
def operator_row(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[wp.vec3d],
    i: wp.int32,
) -> wp.vec3d:
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        # Isolated vertex (empty row): the averaging operator acts as the identity so the
        # vertex does not drift toward the origin.
        return field[i]
    acc = wp.vec3d(0.0, 0.0, 0.0)
    for k in range(start, end):
        w = wp.float64(values[k])
        acc += w * field[columns[k]]
    return acc


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
    # recasting a float32 matrix) avoids a second ``bsr_from_triplets`` rebuild, which re-sorts an
    # already-sorted CSR. This was previously described as dodging a Warp ``bsr_mm`` bug; that was
    # wrong. The nondeterminism came from sizing a rebuild's triplet buffers by ``BsrMatrix.nnz``
    # (the capacity the matrix was built with) instead of ``nnz_sync()`` (its entry count), leaving
    # an uninitialized tail for ``bsr_from_triplets`` to read back as triplets.
    f = wp.int32(wp.tid())
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
    f = wp.int32(wp.tid())
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
    # Adding ``delta`` to all three grows each slack ``a + b - c`` by exactly ``delta`` -- the two
    # short sides gain ``2 * delta`` and the long side gives ``delta`` of it back -- so the
    # shortfall is the constant, undivided. Sharp & Crane (2020) eq. 3 states the same rule, and
    # an independent port of it (kentechx/HoleFillingPy, MIT) computes the identical quantity a
    # different way -- ``max(2 * max(L) - sum(L)) + delta``, which is ``epsilon - min_f slack_f``
    # rearranged -- also undivided.
    #
    # ``TOLERANCE_MOLLIFY = 1e-5`` was swept here against both of its arms, on a float32 sliver
    # whose plain operator drops a coupling and on an icosphere(3) with one face collapsed onto
    # its own opposite edge. Reading ``delta`` in units of one ULP of the mean edge length, and
    # the perturbation as the relative change to the operator's rows that *no* degenerate face
    # touches:
    #
    #   epsilon   delta/ulp   sliver's coupling   max|cot|   clean-row deviation
    #     1e-8          0.1   dropped (no-op)         1.0                     0
    #     3e-8          0.4   dropped (no-op)         1.0                     0
    #     1e-7          1.3   dropped                 1.0              1.2e-07
    #     3e-7          4.0   restored              7.2e2              2.4e-07
    #     1e-6         13.5   restored              4.2e2              3.6e-07
    #     1e-5        134.9   restored              1.2e2              8.2e-06
    #     1e-4       1348.9   restored              3.9e1              8.0e-05
    #     1e-2     134890.1   restored              4.0e0              8.0e-03
    #
    # So the left arm is a float32 storage floor, not a numerical-quality one: under ~4 ULP the
    # added constant does not survive the store and the mollification silently does nothing, which
    # is the failure the whole function exists to prevent. The right arm is the deviation column,
    # which grows linearly with epsilon and reaches the ``rtol=1e-4`` the igl
    # ``intrinsic_delaunay_cotmatrix`` comparison runs at by epsilon = 1e-4. ``1e-5`` is ~1.5
    # decades clear of both, and is a decade *more* conservative than that port's own 1e-4, which
    # sits on the right arm. Its neighbours are each worse in one direction: 1e-6 leaves only
    # 13.5 ULP of headroom over a floor that moves with the mesh's length distribution, and 1e-4
    # starts eating the parity test's tolerance. A clean mesh yields ``delta == 0`` at every
    # epsilon probed, so none of this is paid where nothing is degenerate.
    f = wp.int32(wp.tid())
    a, b, c = row_triple(edge_lengths, f)
    worst = wp.max(wp.max(epsilon - (a + b - c), epsilon - (b + c - a)), epsilon - (c + a - b))
    out_slack[f] = wp.max(worst, 0.0)


@wp.func
def add_constant(length: wp.float32, delta: wp.float32) -> wp.float32:
    return length + delta


# Concrete overloads, registered at import -- see the long-form rationale in
# ``triwarp/kernels/reduce.py`` and the rule in CLAUDE.md section 4. In short: these kernels are
# generic, Warp instantiates an overload on the first launch at each new dtype, and a module's hash
# covers the instantiated set -- so a lazily-created overload rebuilds the whole module. Measured
# over the suite: 14 overloads created across **16** distinct module loads.
#
# ``triwarp.laplacian`` exposes the precision as a public ``dtype`` keyword documented as "may be
# float32 or float64", so both are reachable for every kernel here.
_MATRIX_DTYPES = (wp.float32, wp.float64)


# The concrete handles keyed by the caller's dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable]. Measured interleaved, min of 12, on an
# icosphere(6): ``laplacian.cotmatrix`` 372.7 -> 339.6 us (**1.10x**) and ``laplacian.laplacian``
# 380.6 -> 348.5 (1.09x), from removing one generic launch each. ``COTMATRIX_TRIPLETS`` keys on the
# pair ``(entry dtype, matrix dtype)`` because those two templates are independent, exactly as the
# registration already was.
COTMATRIX_ENTRIES: OverloadTable
COTMATRIX_ENTRIES_INTRINSIC: OverloadTable
ROW_NORMALIZE: OverloadTable
LAPLACIAN_TRIPLETS_SYMMETRIC: OverloadTable
LAPLACIAN_TRIPLETS_DIRECTED: OverloadTable
COTMATRIX_TRIPLETS: OverloadTable
CONNECTION_LAPLACIAN_TRIPLETS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global COTMATRIX_ENTRIES, COTMATRIX_ENTRIES_INTRINSIC, ROW_NORMALIZE
    global LAPLACIAN_TRIPLETS_SYMMETRIC, LAPLACIAN_TRIPLETS_DIRECTED
    global COTMATRIX_TRIPLETS, CONNECTION_LAPLACIAN_TRIPLETS
    COTMATRIX_ENTRIES = OverloadTable(
        cotmatrix_entries,
        {d: [wp.array[wp.vec3], wp.array[wp.int32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    COTMATRIX_ENTRIES_INTRINSIC = OverloadTable(
        cotmatrix_entries_intrinsic,
        {d: [wp.array2d[wp.float32], wp.array2d[d]] for d in _MATRIX_DTYPES},
    )
    ROW_NORMALIZE = OverloadTable(
        row_normalize, {d: [wp.array[wp.int32], wp.array[d]] for d in _MATRIX_DTYPES}
    )
    triplet_signature = {
        d: [
            wp.array2d[wp.int32],
            wp.array[wp.vec3],
            wp.int32,
            wp.array[wp.int32],
            wp.array[wp.int32],
            wp.array[d],
        ]
        for d in _MATRIX_DTYPES
    }
    LAPLACIAN_TRIPLETS_SYMMETRIC = OverloadTable(laplacian_triplets_symmetric, triplet_signature)
    LAPLACIAN_TRIPLETS_DIRECTED = OverloadTable(laplacian_triplets_directed, triplet_signature)
    # ``cot_entries`` and the matrix precision are *independent* templates: cotmatrix's
    # docstring says the entries "may be float32 or float64 regardless of dtype: the assembly
    # kernel casts them to the matrix precision", so this is a genuine 2x2, not a diagonal.
    COTMATRIX_TRIPLETS = OverloadTable(
        cotmatrix_triplets,
        {
            (entry_dtype, dtype): [
                wp.array[wp.int32],
                wp.array2d[entry_dtype],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[dtype],
            ]
            for dtype in _MATRIX_DTYPES
            for entry_dtype in _MATRIX_DTYPES
        },
    )
    # The connection Laplacian's values are always ``wp.mat22d``; only its cotangent entries
    # follow the caller, who may pass their own in place of the float64 default.
    CONNECTION_LAPLACIAN_TRIPLETS = OverloadTable(
        connection_laplacian_triplets,
        {
            d: [
                wp.array[wp.int32],
                wp.array2d[d],
                wp.array[wp.float32],
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.mat22d],
            ]
            for d in _MATRIX_DTYPES
        },
    )


_register_overloads()
