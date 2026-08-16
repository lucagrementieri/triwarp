import warp as wp

from triwarp import constants as twc
from triwarp.kernels.array import cross2, to_vec2d
from triwarp.kernels.laplacian import squared_edge_lengths
from triwarp.kernels.triangles import face_vertices

# Below this (float64) squared edge length the isometric rest-triangle flattening is treated as
# degenerate: its rest edges are zeroed so the ARAP local step contributes nothing for that face.
EPSILON_ARAP_EDGE_SQ = wp.constant(wp.float64(1.0e-20))


@wp.kernel
def flipped_faces_mask(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    fi = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, fi)
    e0 = v1 - v0
    e1 = v2 - v0
    # 2D signed area * 2 == det of libigl's homogeneous 3x3 matrix
    out_mask[fi] = cross2(e0, e1) < 0.0


@wp.kernel
def scatter_boundary_mask(
    boundary_indices: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # Mark every fixed (boundary) vertex; interior vertices keep the pre-set ``False``.
    b = wp.int32(wp.tid())
    out_mask[boundary_indices[b]] = True


@wp.kernel
def scatter_fixed_uv(
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Scatter the prescribed boundary positions into a ``(2, n_vertices)`` buffer (row 0 = u,
    # row 1 = v) so the system-assembly kernel can look up ``bc[c, j]`` by right-hand-side column
    # ``c`` and original vertex index ``j``. float64 to match the float64 conjugate-gradient path.
    b = wp.int32(wp.tid())
    i = boundary_indices[b]
    uv = boundary_uv[b]
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[1, i] = wp.float64(uv[1])


@wp.kernel
def scatter_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array2d[wp.float64],
    fixed_values: wp.array2d[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Reassemble the full ``(n_vertices,)`` UV field: fixed vertices keep their prescribed position
    # (``fixed_values`` is ``(2, n_vertices)``), free vertices read the solved value at their
    # compact index (``sol`` is ``(2, n_free)``). Handles the all-fixed case (``n_free == 0``): the
    # free branch is then never taken, so the empty ``sol`` is never indexed.
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        out_uv[i] = wp.vec2(wp.float32(fixed_values[0, i]), wp.float32(fixed_values[1, i]))
    else:
        ri = free_map[i]
        out_uv[i] = wp.vec2(wp.float32(sol[0, ri]), wp.float32(sol[1, ri]))


@wp.kernel
def boundary_edge_lengths(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], out_len: wp.array[wp.float32]
) -> None:
    # Segment length between consecutive boundary vertices; ``out_len[0] = 0`` seeds the arc-length
    # prefix sum (matches ``igl::map_vertices_to_circle``).
    i = wp.int32(wp.tid())
    if i == 0:
        out_len[0] = wp.float32(0.0)
    else:
        out_len[i] = wp.length(vertices[boundary[i]] - vertices[boundary[i - 1]])


@wp.kernel
def circle_positions(
    boundary: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    cumulative_length: wp.array[wp.float32],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Arc-length parametrization onto the unit circle: ``frac = len[i] * 2pi / total`` with the
    # total perimeter closing over the wrap edge ``bnd[0] -> bnd[n-1]``.
    i = wp.int32(wp.tid())
    n = cumulative_length.shape[0]
    wrap = wp.length(vertices[boundary[0]] - vertices[boundary[n - 1]])
    total = cumulative_length[n - 1] + wrap
    frac = cumulative_length[i] * twc.TWO_PI / total
    out_uv[i] = wp.vec2(wp.cos(frac), wp.sin(frac))


@wp.kernel
def scatter_pinned_stacked(
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
    n_vertices: wp.int32,
    out_fixed_mask: wp.array[wp.bool],
    out_fixed_values: wp.array2d[wp.float64],
) -> None:
    # Fused mask + value scatter for LSCM's stacked ``[u; v]`` DOFs: pin ``i`` fixes DOF ``i`` (u)
    # and ``i + n`` (v). ``out_fixed_values`` is ``(1, 2n)`` (single right-hand-side column).
    b = wp.int32(wp.tid())
    i = pinned_indices[b]
    uv = pinned_uv[b]
    out_fixed_mask[i] = True
    out_fixed_mask[i + n_vertices] = True
    out_fixed_values[0, i] = wp.float64(uv[0])
    out_fixed_values[0, i + n_vertices] = wp.float64(uv[1])


@wp.kernel
def scatter_solution_stacked(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol: wp.array[wp.float64],
    fixed_values: wp.array[wp.float64],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Unstack the solved ``[u; v]`` DOF vector into ``(n, 2)`` UV. DOF ``i`` holds u, DOF ``i + n``
    # holds v; each is either a pinned value (``fixed_values``) or a solved free value (``sol`` at
    # the compact free index). Locals are initialized before the branch per Warp's branch-scope
    # rule.
    i = wp.int32(wp.tid())
    n = out_uv.shape[0]
    u = wp.float64(0.0)
    v = wp.float64(0.0)
    if fixed_mask[i]:
        u = fixed_values[i]
    else:
        u = sol[free_map[i]]
    if fixed_mask[i + n]:
        v = fixed_values[i + n]
    else:
        v = sol[free_map[i + n]]
    out_uv[i] = wp.vec2(wp.float32(u), wp.float32(v))


@wp.kernel
def arap_rest_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float64],
    out_rest_edges: wp.array2d[wp.vec2d],
) -> None:
    # Isometrically flatten each triangle into the plane (igl project_isometrically_to_plane) and
    # store its three weight-folded rest edges ``c_e * p_e`` in igl edge order e0:(1,2), e1:(2,0),
    # e2:(0,1). Run once (dim = n_faces): folding the half-cotangent weight ``c_e`` in here removes
    # the cotangent lookup from the per-iteration local step. Computed in float64 (squared edge
    # lengths promoted from the float32 vertex precision) for a deterministic operator.
    f = wp.int32(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, f)
    l2_0f, l2_1f, l2_2f = squared_edge_lengths(v0, v1, v2)
    # l0 = |v1 - v2|, l1 = |v2 - v0|, l2 = |v0 - v1| (igl edge_lengths column order).
    l2_0 = wp.float64(l2_0f)
    l2_1 = wp.float64(l2_1f)
    l2_2 = wp.float64(l2_2f)
    c0 = cot_entries[f, 0]
    c1 = cot_entries[f, 1]
    c2 = cot_entries[f, 2]
    if l2_2 < EPSILON_ARAP_EDGE_SQ:
        # Degenerate base edge |v0 - v1|: emit zero rest edges (local step contributes nothing).
        out_rest_edges[f, 0] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        out_rest_edges[f, 1] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        out_rest_edges[f, 2] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        return
    # Flattened rest positions: P0 = (0, 0), P1 = (|v0 - v1|, 0), P2 from the law of cosines.
    base = wp.sqrt(l2_2)
    p2x = (-l2_0 + l2_1 + l2_2) / (wp.float64(2.0) * base)
    p2y_sq = wp.max(l2_1 - p2x * p2x, wp.float64(0.0))
    p1 = wp.vec2d(base, wp.float64(0.0))
    p2 = wp.vec2d(p2x, wp.sqrt(p2y_sq))
    # p_e = P_source - P_dest per igl edge, folded with c_e: e0 -> P1 - P2, e1 -> P2 (P0 = 0),
    # e2 -> -P1 (P0 - P1).
    out_rest_edges[f, 0] = c0 * (p1 - p2)
    out_rest_edges[f, 1] = c1 * p2
    out_rest_edges[f, 2] = -c2 * p1


@wp.func
def scatter_arap_edge(
    rotation: wp.mat22d,
    w: wp.vec2d,
    source: wp.int32,
    dest: wp.int32,
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
) -> None:
    # Rotate the weight-folded rest edge and scatter the ARAP right-hand side ``+r`` to the edge
    # source and ``-r`` to the edge dest. The two components go to separate 1D buffers so the atomic
    # adds stay 1D (row-view outputs of the (2, n_vertices) rotation RHS).
    r = rotation * w
    wp.atomic_add(out_rhs_x, source, r[0])
    wp.atomic_add(out_rhs_y, source, r[1])
    wp.atomic_add(out_rhs_x, dest, -r[0])
    wp.atomic_add(out_rhs_y, dest, -r[1])


@wp.kernel
def arap_local_step(
    faces: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    rest_edges: wp.array2d[wp.vec2d],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
) -> None:
    # Per face (dim = n_faces, once per iteration): fit the closest 2D rotation to the current UV
    # and scatter the rotation right-hand side, fused so no per-face rotation buffer is needed.
    # ``out_rhs_x`` / ``out_rhs_y`` are the two rows of the (2, n_vertices) rotation RHS and must be
    # zeroed before launch. Degenerate faces have zero rest edges, so S = 0, atan2(0, 0) = 0, and
    # the identity rotation scatters nothing.
    f = wp.int32(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    uv0 = uv[i0]
    uv1 = uv[i1]
    uv2 = uv[i2]
    # UV edges in igl order e0:(1,2), e1:(2,0), e2:(0,1), promoted to float64.
    u0 = to_vec2d(uv1) - to_vec2d(uv2)
    u1 = to_vec2d(uv2) - to_vec2d(uv0)
    u2 = to_vec2d(uv0) - to_vec2d(uv1)
    w0 = rest_edges[f, 0]
    w1 = rest_edges[f, 1]
    w2 = rest_edges[f, 2]
    # Covariance S = sum_e u_e (outer) w_e (rest edges already weight-folded with c_e).
    s = wp.outer(u0, w0) + wp.outer(u1, w1) + wp.outer(u2, w2)
    # Closest proper rotation (reflections forbidden), fit_rotations_planar closed form.
    theta = wp.atan2(s[1, 0] - s[0, 1], s[0, 0] + s[1, 1])
    cos_t = wp.cos(theta)
    sin_t = wp.sin(theta)
    rotation = wp.mat22d(cos_t, -sin_t, sin_t, cos_t)
    scatter_arap_edge(rotation, w0, i1, i2, out_rhs_x, out_rhs_y)
    scatter_arap_edge(rotation, w1, i2, i0, out_rhs_x, out_rhs_y)
    scatter_arap_edge(rotation, w2, i0, i1, out_rhs_x, out_rhs_y)


@wp.kernel
def arap_interior_rhs(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    rhs_const: wp.array2d[wp.float64],
    rhs_rot_x: wp.array[wp.float64],
    rhs_rot_y: wp.array[wp.float64],
    out_b: wp.array2d[wp.float64],
) -> None:
    # Assemble the interior right-hand side of the global step (dim = n_vertices). Interior rows get
    # the constant term ``rhs_const`` (the ``-(-L)_ub bc`` boundary contribution, already at the
    # compact free index) plus the scattered rotation RHS at the original vertex index; boundary
    # rows carry no unknown and are skipped. ``rhs_const`` / ``out_b`` are (2, n_interior) (rows
    # u, v); ``rhs_rot_*`` are (n_vertices,).
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        return
    ri = free_map[i]
    out_b[0, ri] = rhs_const[0, ri] + rhs_rot_x[i]
    out_b[1, ri] = rhs_const[1, ri] + rhs_rot_y[i]


@wp.kernel
def gather_interior_uv(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    uv: wp.array[wp.vec2],
    out_sol: wp.array2d[wp.float64],
) -> None:
    # Seed the conjugate-gradient warm start with the current interior UV (dim = n_vertices):
    # interior vertex ``i`` writes its UV into the two rows (u, v) of ``out_sol`` at the compact
    # free index; boundary vertices are skipped. ``out_sol`` is (2, n_interior).
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        return
    ri = free_map[i]
    p = uv[i]
    out_sol[0, ri] = wp.float64(p[0])
    out_sol[1, ri] = wp.float64(p[1])
