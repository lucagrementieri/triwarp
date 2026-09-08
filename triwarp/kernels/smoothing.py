import warp as wp

from triwarp.kernels.array import to_vec3d
from triwarp.kernels.laplacian import cot_entries_from_l2
from triwarp.kernels.linalg import free_row, selected_row, solve_normal_equations
from triwarp.kernels.predicates import (
    closest_point_on_segment,
    doublearea_from_lengths,
    plane_basis,
    squared_edge_lengths,
)
from triwarp.kernels.scatter import add_corner_triple
from triwarp.kernels.triangles import corner_triple

# Fixed-size float64 types for the 6-coefficient quadric fit in ``relax_approx``. The rest of the
# kernel runs in float32; the least-squares solve is float64 for conditioning, and it runs through
# ``linalg.solve_normal_equations``, which is rank-generic -- ``kernels/curvature.py``'s 5x5 quadric
# fit is the same call at a different width.
# DBL_EPSILON, the relative accuracy of a float64. The area-equalizing solve compares its system's
# determinant against this times the trace's power, which is the scale-free way to ask whether the
# 1-ring is degenerate enough that the solution cannot be trusted.
DOUBLE_EPSILON = wp.constant(wp.float64(2.220446049250313e-16))

vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)


# ---------------------------------------------------------------------------
# Region Dirichlet / least-squares smoothing
# ---------------------------------------------------------------------------


@wp.kernel
def edge_cotan_add(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_w: wp.array[wp.float32],
) -> None:
    # Accumulate each face corner's cotangent into its opposite unique edge; the two incident faces
    # sum to the cotangent edge weight cot(alpha) + cot(beta).
    #
    # The per-corner cotangent is ``laplacian.py``'s generic half-cotangent formula (its own
    # denominator/degenerate-triangle guard, rather than a second independent one derived from the
    # raw cross product) doubled back to a full cotangent, since this accumulator -- unlike
    # ``laplacian.cotmatrix`` -- wants ``cot(alpha) + cot(beta)`` rather than the half-cotangent
    # convention `laplacian.py`'s own docstring explains. ``inverse[f*3+k]`` is the unique edge id
    # of edge ``(v_k, v_{k+1})`` (``edges.faces_to_edges``'s convention), whose cotangent
    # contribution from this triangle is the angle *opposite* that edge -- i.e. the angle at the
    # corner not on it -- which is why the three half-cotangents land rotated by one slot below.
    f = wp.int32(wp.tid())
    v0, v1, v2 = corner_triple(faces, f)
    p0 = vertices[v0]
    p1 = vertices[v1]
    p2 = vertices[v2]
    l2_0, l2_1, l2_2 = squared_edge_lengths(p0, p1, p2)
    dbl_area = doublearea_from_lengths(wp.sqrt(l2_0), wp.sqrt(l2_1), wp.sqrt(l2_2))
    half_cotan0, half_cotan1, half_cotan2 = cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)
    two = wp.float32(2.0)
    add_corner_triple(out_w, inverse, f, two * half_cotan2, two * half_cotan0, two * half_cotan1)


@wp.func
def clamp_cotan(w: wp.float32) -> wp.float32:
    # The summed cotangent edge weight is clamped: a degenerate edge gives arbitrarily high cot.
    return wp.clamp(w, wp.float32(-1.0), wp.float32(10.0))


@wp.kernel
def symmetric_weight_triplets(
    unique_edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    i = wp.int32(wp.tid())
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
    # SPD umbrella system A = D - W over free verts, sharp boundary (weights in the
    # CSR ``W``), fixed 1-ring neighbors folded into the right-hand side, plus optional stabilizer.
    v = wp.int32(wp.tid())
    ri = selected_row(free_mask, free_map, v)
    if ri < 0:
        return
    start = offsets[v]
    end = offsets[v + 1]
    sum_w = stabilizer
    rhs = stabilizer * to_vec3d(points[v])
    base = start + v  # one reserved diagonal slot per vertex; off-diagonals follow
    for k in range(start, end):
        j = columns[k]
        w = values[k]
        sum_w += w
        cj = selected_row(free_mask, free_map, j)
        if cj >= 0:
            slot = base + 1 + (k - start)
            out_rows[slot] = ri
            out_cols[slot] = cj
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
    # Least-squares umbrella rows over R = free plus the first fixed ring. The row is
    # ``p_v = sum_d (w_vd / sumW) p_d``; free neighbours stay in M, fixed ones move to the
    # right-hand side, and the normal equations ``(M^T M) x = M^T b`` are assembled by the caller.
    #
    # Two partitions at once, both read through ``selected_row``: ``row_mask`` / ``row_map`` says
    # which vertices carry a row, ``free_mask`` / ``free_map`` which carry an unknown.
    v = wp.int32(wp.tid())
    r = selected_row(row_mask, row_map, v)
    if r < 0:
        return
    start = offsets[v]
    end = offsets[v + 1]
    sum_w = wp.float64(0.0)
    for k in range(start, end):
        sum_w += values[k]
    if sum_w == wp.float64(0.0):
        return
    base = start + v
    free_column = selected_row(free_mask, free_map, v)
    rhs = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if free_column < 0:
        rhs = -to_vec3d(points[v])
    for k in range(start, end):
        j = columns[k]
        coeff = -values[k] / sum_w
        cj = selected_row(free_mask, free_map, j)
        if cj >= 0:
            slot = base + 1 + (k - start)
            out_rows[slot] = r
            out_cols[slot] = cj
            out_vals[slot] = coeff
        else:
            rhs -= coeff * to_vec3d(points[j])
    if free_column >= 0:
        out_rows[base] = r
        out_cols[base] = free_column
        out_vals[base] = wp.float64(1.0)
    out_rhs_x[r] = rhs[0]
    out_rhs_y[r] = rhs[1]
    out_rhs_z[r] = rhs[2]


@wp.kernel
def gather_free_positions(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    out_sol_x: wp.array[wp.float64],
    out_sol_y: wp.array[wp.float64],
    out_sol_z: wp.array[wp.float64],
) -> None:
    # The inverse of ``scatter_free_solution``. Both region solves ask CG for the free vertices'
    # *new* positions, whose best available initial guess is their current ones -- and for a vertex
    # no face refers to it is the only one, because such a vertex contributes no row and CG never
    # writes its entry. Seeding from zeros leaves it at the origin; ``smoothing._free_positions``
    # carries the counts. The speed is the smaller half of why this exists.
    v = wp.int32(wp.tid())
    i = selected_row(free_mask, free_map, v)
    if i >= 0:
        p = points[v]
        out_sol_x[i] = wp.float64(p[0])
        out_sol_y[i] = wp.float64(p[1])
        out_sol_z[i] = wp.float64(p[2])


@wp.kernel
def scatter_free_solution(
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_points: wp.array[wp.vec3],
) -> None:
    # Write the reduced solve's answer back over the free vertices; a pinned one keeps the position
    # it arrived with. The inverse of ``gather_free_positions``, which seeds the same solve.
    v = wp.int32(wp.tid())
    i = selected_row(free_mask, free_map, v)
    if i >= 0:
        out_points[v] = wp.vec3(wp.float32(sol_x[i]), wp.float32(sol_y[i]), wp.float32(sol_z[i]))


@wp.kernel
def add_interior_mass_rhs(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    mass: wp.array[wp.float64],
    positions: wp.array[wp.vec3d],
    rhs: wp.array2d[wp.float64],
) -> None:
    # Add the linear term ``b_u = (M V)_u`` into the reduced right-hand side, which arrives holding
    # only ``-A_ub x_b`` from ``linalg.assemble_interior_system`` (that helper eliminates the pinned
    # columns of a quadratic form, which has no linear term of its own). ``rhs`` is genuinely
    # in-place -- an accumulator carrying that prior term in, not a fresh answer -- which is why it
    # does not carry the ``out_`` prefix reserved for write-only outputs (CLAUDE.md section 2.1).
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    if i < 0:
        return
    m = mass[v]
    p = positions[v]
    rhs[0, i] = rhs[0, i] + m * p[0]
    rhs[1, i] = rhs[1, i] + m * p[1]
    rhs[2, i] = rhs[2, i] + m * p[2]


@wp.kernel
def gather_free_positions_2d(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    out_solution: wp.array2d[wp.float64],
) -> None:
    # Seed the reduced solve with the free vertices' *current* positions: the exact inverse of
    # ``scatter_free_positions`` below, and the counterpart of ``gather_free_positions`` above for
    # the ``fixed_mask`` partition and the float64 storage the implicit-fairing flow carries.
    # Seeding from the right-hand side instead would leave a vertex no face refers to at the
    # origin -- it has an all-zero row and so a zero right-hand side, and CG never writes its
    # entry -- which is the failure ``gather_free_positions`` exists to avoid on the region solves.
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    if i < 0:
        return
    p = positions[v]
    out_solution[0, i] = p[0]
    out_solution[1, i] = p[1]
    out_solution[2, i] = p[2]


@wp.kernel
def scatter_free_positions(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    sol_z: wp.array[wp.float64],
    out_positions: wp.array[wp.vec3d],
) -> None:
    # Write the reduced solution back to the unpinned vertices only. Pinned ones are left holding
    # whatever they already have, which is their original position -- they never move.
    v = wp.int32(wp.tid())
    i = free_row(fixed_mask, free_map, v)
    if i < 0:
        return
    out_positions[v] = wp.vec3d(sol_x[i], sol_y[i], sol_z[i])


@wp.func
def rescale_about_center(position: wp.vec3d, center: wp.vec3d, scale: wp.float64) -> wp.vec3d:
    # (position - center) * scale + center: a uniform rescale about an arbitrary fixed point
    # rather than the origin. ``smoothing._apply_volume_constraint`` needs this, not a bare
    # multiply, because the mesh being smoothed is rarely centred at the origin and
    # ``trimesh.smoothing.filter_laplacian`` rescales about the mesh's own (fixed, initial) centre
    # of mass -- multiplying by ``scale`` alone silently translates the whole mesh on every pass
    # whenever the two points differ, and the error compounds with the iteration count.
    return (position - center) * scale + center


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
    i = wp.int32(wp.tid())
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
    i = wp.int32(wp.tid())
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


@wp.func
def scalar_laplacian_step(value: wp.float32, average: wp.float32, lamb: wp.float32) -> wp.float32:
    # Explicit diffusion step on a scalar field: move it a fraction ``lamb`` of the way to the
    # 1-ring average. ``lamb = 1`` replaces the value outright, which is MeshLab's single pass.
    return wp.lerp(value, average, lamb)


@wp.kernel
def apply_operator_scalar(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float32],
    field: wp.array[wp.float32],
    out_average: wp.array[wp.float32],
) -> None:
    # Scalar counterpart of ``kernels/laplacian.apply_operator``: one row of the row-stochastic
    # averaging operator against a per-vertex scalar. An isolated vertex (empty row) keeps its own
    # value, so it neither drifts to zero nor contaminates its (nonexistent) neighbours.
    i = wp.int32(wp.tid())
    start = offsets[i]
    end = offsets[i + 1]
    if end == start:
        out_average[i] = field[i]
        return
    total = wp.float32(0.0)
    for k in range(start, end):
        total += values[k] * field[columns[k]]
    out_average[i] = total


@wp.kernel
def accumulate_smoothed_normals(
    face_normals: wp.array[wp.vec3],
    face_areas: wp.array[wp.float32],
    face_adjacency: wp.array2d[wp.int32],
    threshold_cos: wp.float32,
    out_accumulated: wp.array[wp.vec3],
) -> None:
    # Area-weighted average of a face's normal with those of its edge-neighbours -- but only the
    # neighbours pointing *within* ``threshold_cos`` of it. That gate is the whole point: across a
    # crease the two normals disagree by more than the threshold and simply do not average, so a
    # sharp edge survives an arbitrary number of passes while noise on a flat region diffuses away.
    k = wp.int32(wp.tid())
    f0 = face_adjacency[k, 0]
    f1 = face_adjacency[k, 1]
    if wp.dot(face_normals[f0], face_normals[f1]) <= threshold_cos:
        return
    wp.atomic_add(out_accumulated, f0, face_areas[f1] * face_normals[f1])
    wp.atomic_add(out_accumulated, f1, face_areas[f0] * face_normals[f0])


@wp.func
def seed_weighted_normal(normal: wp.vec3, area: wp.float32) -> wp.vec3:
    # A face's own area-weighted normal: the seed of the accumulator above, so the face always
    # contributes to its own average even when every neighbour is across a crease.
    return area * normal


@wp.kernel
def fit_vertices_to_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    out_delta: wp.array[wp.vec3],
    out_count: wp.array[wp.float32],
) -> None:
    # One gradient step of the vertex-fitting half of two-step smoothing (Ohtake et al.): each
    # incident face wants its corner to lie in the plane through the face centroid with the
    # *filtered* normal, and the correction is the component of that offset along the normal.
    #
    # Summed per vertex and divided by the incident-face count by the caller, which is the step size
    # that makes the iteration a contraction without a tuning constant.
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    normal = face_normals[f]
    centroid = (vertices[i0] + vertices[i1] + vertices[i2]) / 3.0
    for k in range(3):
        v = faces[f * 3 + k]
        wp.atomic_add(out_delta, v, normal * wp.dot(normal, centroid - vertices[v]))
        wp.atomic_add(out_count, v, 1.0)


@wp.func
def apply_fit_step(position: wp.vec3, delta: wp.vec3, count: wp.float32) -> wp.vec3:
    if count <= 0.0:
        return position
    return position + delta / count


@wp.func
def unsharp_step(
    position: wp.vec3, smoothed: wp.vec3, weight: wp.float32, weight_original: wp.float32
) -> wp.vec3:
    # MeshLab's ``apply_coord_unsharp_mask``: add back a multiple of the high-frequency detail the
    # smoothing pass removed. ``weight_original = 1`` keeps the surface in place and only sharpens.
    return weight_original * position + weight * (position - smoothed)


@wp.func
def step_along_normal(position: wp.vec3, normal: wp.vec3, distance: wp.float32) -> wp.vec3:
    """Move a vertex along its own normal, which is one half of an inflation step."""
    return position + normal * distance


@wp.func
def select_position(smoothed: wp.vec3, original: wp.vec3, replace: wp.bool) -> wp.vec3:
    """Take the smoothed position only where the mask says to, leaving the rest untouched."""
    if replace:
        return smoothed
    return original


# ---------------------------------------------------------------------------
# Relaxation family: area equalization, volume-preserving relax, surface-fit relax
# ---------------------------------------------------------------------------


@wp.func
def limit_near_initial(target: wp.vec3, initial: wp.vec3, max_distance: wp.float32) -> wp.vec3:
    # Clamp a proposed position into a ball around where the vertex started. A negative radius means
    # no limit, which is how the wrapper spells ``max_displacement=None`` without a second kernel.
    if max_distance < wp.float32(0.0):
        return target
    offset = target - initial
    distance = wp.length(offset)
    if distance <= max_distance:
        return target
    return initial + offset * (max_distance / distance)


@wp.func
def _rotate_corner_to_front(
    first: wp.int32, second: wp.int32, third: wp.int32, vertex: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # The face's other two corners, in winding order starting after ``vertex``. Winding order is
    # what makes the pair an oriented opposite *edge* rather than an unordered pair.
    if first == vertex:
        return second, third
    if second == vertex:
        return third, first
    return first, second


@wp.func
def equal_area_position(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    vertex: wp.int32,
    normal: wp.vec3,
    no_shrinkage: wp.bool,
) -> wp.vec3:
    """
    Solve for the position minimizing the summed squared areas of the incident triangles.

    Twice the area of the triangle on the opposite edge ``(p, q)`` is ``|(x - p) x (q - p)|``, so
    the objective is a sum of quadratic forms in the free position ``x`` and its minimum is one
    linear solve. Accumulated in ``float64``: the matrix is a sum of rank-deficient terms and a
    near-degenerate 1-ring loses the answer entirely in ``float32``.

    With ``no_shrinkage`` the solve is restricted to the tangent plane through the current position,
    so the vertex slides across the surface instead of sinking into it -- an unconstrained minimum
    of *squared* area pulls the whole 1-ring inward.
    """
    current = positions[vertex]
    matrix = wp.mat33d()
    rhs = wp.vec3d()
    for slot in range(offsets[vertex], offsets[vertex + 1]):
        first, second, third = corner_triple(faces, vertex_faces[slot])
        opposite_start, opposite_end = _rotate_corner_to_front(first, second, third, vertex)
        first_position = to_vec3d(positions[opposite_start])
        edge = to_vec3d(positions[opposite_end]) - first_position
        # ``d d^T - |d|^2 I`` maps x to d x (d x x): the quadratic form whose value at x - p is
        # minus the squared area term.
        term = wp.outer(edge, edge) - wp.identity(n=3, dtype=wp.float64) * wp.dot(edge, edge)
        matrix += term
        rhs += term * first_position

    if no_shrinkage:
        # ``plane_basis`` renormalizes ``normal`` internally to build the tangent frame; the anchor
        # projection below must use that same unit vector, since it is only a projection onto the
        # normal axis when its argument has unit length.
        unit_normal = wp.normalize(normal)
        axis_x, axis_y = plane_basis(normal)
        basis_x = to_vec3d(axis_x)
        basis_y = to_vec3d(axis_y)
        mapped_x = matrix * basis_x
        mapped_y = matrix * basis_y
        off_diagonal = wp.dot(mapped_x, basis_y)
        planar = wp.mat22d(
            wp.dot(mapped_x, basis_x), off_diagonal, off_diagonal, wp.dot(mapped_y, basis_y)
        )
        determinant = wp.determinant(planar)
        trace = planar[0, 0] + planar[1, 1]
        if DOUBLE_EPSILON * wp.abs(trace * trace) >= wp.abs(determinant):
            return current
        anchor = to_vec3d(unit_normal) * wp.dot(to_vec3d(unit_normal), to_vec3d(current))
        reduced = rhs - matrix * anchor
        solution = wp.inverse(planar) * wp.vec2d(wp.dot(reduced, basis_x), wp.dot(reduced, basis_y))
        target = anchor + basis_x * solution[0] + basis_y * solution[1]
        return wp.vec3(wp.float32(target[0]), wp.float32(target[1]), wp.float32(target[2]))

    determinant = wp.determinant(matrix)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if DOUBLE_EPSILON * wp.abs(trace * trace * trace) >= wp.abs(determinant):
        return current
    target = wp.inverse(matrix) * rhs
    return wp.vec3(wp.float32(target[0]), wp.float32(target[1]), wp.float32(target[2]))


@wp.kernel
def equalize_area_step(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    no_shrinkage: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # One pass of area equalization: step each in-region vertex a fraction ``force`` of the way to
    # its own equal-area minimum, then clamp it back near where it started.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    if not region[vertex] or offsets[vertex] == offsets[vertex + 1]:
        out_positions[vertex] = current
        return
    target = equal_area_position(
        positions, faces, offsets, vertex_faces, vertex, normals[vertex], no_shrinkage
    )
    moved = current + (target - current) * force
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


@wp.kernel
def ring_push_forces(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    force: wp.float32,
    out_push: wp.array[wp.vec3],
) -> None:
    # The plain uniform-relax displacement each in-region vertex would take on its own. Kept as a
    # field rather than applied, because the volume correction below is its ring average.
    vertex = wp.int32(wp.tid())
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not region[vertex] or begin == end:
        out_push[vertex] = wp.vec3(0.0, 0.0, 0.0)
        return
    total = wp.vec3d()
    for slot in range(begin, end):
        total += to_vec3d(positions[columns[slot]])
    mean = total / wp.float64(end - begin)
    average = wp.vec3(wp.float32(mean[0]), wp.float32(mean[1]), wp.float32(mean[2]))
    out_push[vertex] = (average - positions[vertex]) * force


@wp.kernel
def apply_push_keeping_volume(
    positions: wp.array[wp.vec3],
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    region: wp.array[wp.bool],
    push: wp.array[wp.vec3],
    initial: wp.array[wp.vec3],
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Subtract the ring average of the displacement field from each vertex's own displacement. A
    # translation shared by a whole neighbourhood cancels, so the surface stops drifting inward
    # while the high-frequency part of the relax survives -- which is what preserves the volume.
    # The divisor is the full degree while the sum runs over in-region neighbours only, so a vertex
    # on the region's edge is corrected by less than one that is surrounded.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    begin = offsets[vertex]
    end = offsets[vertex + 1]
    if not region[vertex] or begin == end:
        out_positions[vertex] = current
        return
    total = wp.vec3()
    for slot in range(begin, end):
        neighbor = columns[slot]
        if region[neighbor]:
            total += push[neighbor]
    moved = current + push[vertex] - total / wp.float32(end - begin)
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


@wp.func
def _neighborhood_frame(
    positions: wp.array[wp.vec3], neighbors: wp.array[wp.int32], begin: wp.int32, end: wp.int32
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    # Principal frame of the neighbourhood point set: its centroid, then the two directions of
    # greatest spread and the one of least. The least-spread direction is the fitted plane's normal,
    # so the same decomposition serves both the planar and the quadric fit.
    count = wp.float32(end - begin)
    centroid = wp.vec3()
    for slot in range(begin, end):
        centroid += positions[neighbors[slot]]
    centroid /= count
    covariance = wp.mat33()
    for slot in range(begin, end):
        offset = positions[neighbors[slot]] - centroid
        covariance += wp.outer(offset, offset)
    _left, _singular, basis = wp.svd3(covariance / count)
    # ``wp.svd3`` orders the singular values descending, so the last column spans the least. Its
    # sign is arbitrary and irrelevant: every use below is a projection along the axis, not a side.
    axis_u = wp.vec3(basis[0, 0], basis[1, 0], basis[2, 0])
    axis_v = wp.vec3(basis[0, 1], basis[1, 1], basis[2, 1])
    axis_w = wp.vec3(basis[0, 2], basis[1, 2], basis[2, 2])
    return centroid, axis_u, axis_v, axis_w


@wp.kernel
def relax_approx_step(
    positions: wp.array[wp.vec3],
    neighbor_indices: wp.array[wp.int32],
    neighbor_offsets: wp.array[wp.int32],
    region: wp.array[wp.bool],
    initial: wp.array[wp.vec3],
    force: wp.float32,
    quadric: wp.bool,
    max_displacement: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Fit a local surface to the vertex's neighbourhood and step toward the point of that surface
    # above the vertex. The floor below is a uniform 6 for both the planar and the quadric fit --
    # not 3, even though a plane alone needs only that many -- so an under-populated neighbourhood
    # is left alone rather than fitted to whatever it has; ``smoothing.relax_approx``'s docstring
    # states this uniform floor as the intended behavior.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    begin = neighbor_offsets[vertex]
    # ``geodesic_ball`` returns starts without a sentinel, so the last row runs to the buffer's end.
    end = neighbor_indices.shape[0]
    if vertex + 1 < neighbor_offsets.shape[0]:
        end = neighbor_offsets[vertex + 1]
    if not region[vertex] or end - begin < 6:
        out_positions[vertex] = current
        return

    centroid, axis_u, axis_v, axis_w = _neighborhood_frame(positions, neighbor_indices, begin, end)
    offset = current - centroid
    # Initialized before the branch, per the kernel-scope scoping rule; the planar fit's answer is
    # exactly this, since the plane passes through the neighbourhood centroid.
    height = wp.float32(0.0)
    if quadric:
        # Least squares over ``w = a u^2 + b u v + c v^2 + d u + e v + f`` in the neighbourhood's
        # own frame: the fit is a graph over the plane the neighbourhood already lies closest to.
        normal_matrix = mat66d()
        normal_rhs = vec6d()
        for slot in range(begin, end):
            local = positions[neighbor_indices[slot]] - centroid
            u = wp.float64(wp.dot(local, axis_u))
            v = wp.float64(wp.dot(local, axis_v))
            row = vec6d(u * u, u * v, v * v, u, v, wp.float64(1.0))
            normal_matrix += wp.outer(row, row)
            normal_rhs += row * wp.float64(wp.dot(local, axis_w))
        coefficients, ok = solve_normal_equations(normal_matrix, normal_rhs)
        if ok:
            u = wp.float64(wp.dot(offset, axis_u))
            v = wp.float64(wp.dot(offset, axis_v))
            height = wp.float32(
                coefficients[0] * u * u
                + coefficients[1] * u * v
                + coefficients[2] * v * v
                + coefficients[3] * u
                + coefficients[4] * v
                + coefficients[5]
            )

    target = current + axis_w * (height - wp.dot(offset, axis_w))
    moved = current + (target - current) * force
    out_positions[vertex] = limit_near_initial(moved, initial[vertex], max_displacement)


@wp.kernel
def project_to_zero_isoline(
    positions: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    vertex_faces: wp.array[wp.int32],
    field: wp.array[wp.float64],
    free: wp.array[wp.bool],
    damping: wp.float32,
    out_positions: wp.array[wp.vec3],
) -> None:
    # Pull each free vertex onto the field's zero level set, which is where the region's rim curve
    # wants to be. Inside one triangle the level set is a straight segment between the crossings on
    # the two edges out of the *apex* -- the corner whose sign differs from the other two -- so the
    # nearest point of the whole curve to this vertex is the nearest over its incident triangles'
    # segments. Damped rather than snapped, because the field is recomputed from the moved positions
    # on the next pass and a full step oscillates.
    vertex = wp.int32(wp.tid())
    current = positions[vertex]
    if not free[vertex]:
        out_positions[vertex] = current
        return

    best = current
    best_distance = wp.float32(3.4028235e38)
    for slot in range(offsets[vertex], offsets[vertex + 1]):
        first, second, third = corner_triple(faces, vertex_faces[slot])
        value_first = field[first]
        value_second = field[second]
        value_third = field[third]
        # The apex is the corner alone on its side of zero. When every corner shares a sign the
        # level set misses the triangle entirely.
        apex = first
        left = second
        right = third
        if value_second * value_third > wp.float64(0.0):
            if value_first * value_second > wp.float64(0.0):
                continue
        elif value_first * value_third > wp.float64(0.0):
            apex = second
            left = third
            right = first
        else:
            apex = third
            left = first
            right = second

        value_apex = field[apex]
        gap_left = value_apex - field[left]
        gap_right = value_apex - field[right]
        apex_position = positions[apex]
        # A gap of exactly zero means ``apex`` and that neighbour already share the same (zero)
        # field value, so by linearity the whole edge between them -- not one interior point on
        # it -- lies on the level set; falling through to the crossing formula below would divide
        # by that zero. Using the edge itself as the candidate segment covers the doubly-degenerate
        # case too (every corner on the level set) -- some incident edge is still a valid witness.
        if gap_left == wp.float64(0.0):
            candidate = closest_point_on_segment(apex_position, positions[left], current)
        elif gap_right == wp.float64(0.0):
            candidate = closest_point_on_segment(apex_position, positions[right], current)
        else:
            crossing_left = apex_position + (positions[left] - apex_position) * wp.float32(
                value_apex / gap_left
            )
            crossing_right = apex_position + (positions[right] - apex_position) * wp.float32(
                value_apex / gap_right
            )
            candidate = closest_point_on_segment(crossing_left, crossing_right, current)
        distance = wp.length_sq(candidate - current)
        if distance < best_distance:
            best_distance = distance
            best = candidate

    out_positions[vertex] = current + (best - current) * damping


@wp.kernel
def scatter_free_scalar(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array[wp.float64],
    out_field: wp.array[wp.float64],
) -> None:
    # Write the reduced solve's answer back over the free entries, leaving the pinned ones as the
    # boundary values they were set to. The scalar sibling of ``scatter_free_solution``.
    vertex = wp.int32(wp.tid())
    row = free_row(fixed_mask, free_map, vertex)
    if row >= 0:
        out_field[vertex] = solution[row]


@wp.func
def free_in_mixed_component(
    free: wp.bool, component_size: wp.int32, free_in_component: wp.int32
) -> wp.bool:
    # Drop a vertex from the free set when its whole connected component is free: the harmonic
    # system over such a component has no boundary values to interpolate and is singular.
    return free and component_size != free_in_component


@wp.kernel
def mark_incident_vertices(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # Mark every corner of every selected face. Concurrent writes all store ``True``, so the race is
    # benign and no atomic is needed.
    face = wp.int32(wp.tid())
    if not face_mask[face]:
        return
    first, second, third = corner_triple(faces, face)
    out_mask[first] = True
    out_mask[second] = True
    out_mask[third] = True


@wp.func
def region_side_value(inside: wp.bool) -> wp.float64:
    # The field the rim curve is the zero set of: -1 on the region, +1 outside it. Any two values of
    # opposite sign would do; +-1 keeps the harmonic interpolant's scale comparable to nothing else,
    # which is fine because only its zero set is read.
    if inside:
        return wp.float64(-1.0)
    return wp.float64(1.0)
