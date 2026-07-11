import warp as wp

from triwarp import constants as twc
from triwarp.kernels.array import cross2
from triwarp.kernels.triangles import face_vertices


@wp.kernel
def flipped_faces_mask(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    fi = int(wp.tid())
    v0, v1, v2 = face_vertices(vertices, faces, wp.int32(fi))
    e0 = v1 - v0
    e1 = v2 - v0
    # 2D signed area * 2 == det of libigl's homogeneous 3x3 matrix
    out_mask[fi] = cross2(e0, e1) < 0.0


@wp.kernel
def scatter_boundary_mask(
    boundary_indices: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    # Mark every fixed (boundary) vertex; interior vertices keep the pre-set ``False``.
    b = int(wp.tid())
    out_mask[boundary_indices[b]] = True


@wp.func
def interior_flag(boundary: wp.bool) -> wp.int32:
    # ``1`` for interior (free) vertices, ``0`` for fixed ones; scanned into the interior remap.
    return wp.where(boundary, wp.int32(0), wp.int32(1))


@wp.kernel
def scatter_fixed_uv(
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    out_fixed_uv: wp.array[wp.vec2],
) -> None:
    # Scatter the prescribed boundary positions into a full-length ``(n_vertices,)`` buffer so the
    # system-assembly kernel can look up ``bc[j]`` by original vertex index.
    b = int(wp.tid())
    out_fixed_uv[boundary_indices[b]] = boundary_uv[b]


@wp.kernel
def interior_system_triplets(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.float64],
    boundary_mask: wp.array[wp.bool],
    interior_map: wp.array[wp.int32],
    fixed_uv: wp.array[wp.vec2],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
    out_rhs_x: wp.array[wp.float64],
    out_rhs_y: wp.array[wp.float64],
) -> None:
    # One thread per row ``i`` of the operator ``Q`` (positive semi-definite). Emit the
    # interior-interior block into COO triplets (remapped to the compact interior index) and move
    # fixed-column contributions to the right-hand side: ``Q_uu x_u = -Q_ub bc``. Boundary rows
    # leave their pre-zeroed output slots untouched. Assembled in float64: the biharmonic (k > 1)
    # operator squares the Laplacian condition number, beyond float32 conjugate gradient's reach.
    i = int(wp.tid())
    if boundary_mask[i]:
        return
    ri = interior_map[i]
    start = offsets[i]
    end = offsets[i + 1]
    rhs_x = wp.float64(0.0)
    rhs_y = wp.float64(0.0)
    for e in range(start, end):
        j = columns[e]
        q = values[e]
        if boundary_mask[j]:
            bc = fixed_uv[j]
            rhs_x -= q * wp.float64(bc[0])
            rhs_y -= q * wp.float64(bc[1])
        else:
            out_rows[e] = ri
            out_cols[e] = interior_map[j]
            out_vals[e] = q
    out_rhs_x[ri] = rhs_x
    out_rhs_y[ri] = rhs_y


@wp.func
def reciprocal64(value: wp.Float) -> wp.float64:
    # Diagonal inverse (``igl::invert_diag``) of the lumped mass into float64, for the ``k > 1``
    # operator ``Q = (-L) (M^-1 (-L))^(k-1)``. ``value`` is generic (float32 or float64). A zero
    # entry maps to zero, not infinity.
    v = wp.float64(value)
    if v != wp.float64(0.0):
        return wp.float64(1.0) / v
    return wp.float64(0.0)


@wp.kernel
def scatter_solution(
    boundary_mask: wp.array[wp.bool],
    interior_map: wp.array[wp.int32],
    sol_x: wp.array[wp.float64],
    sol_y: wp.array[wp.float64],
    fixed_uv: wp.array[wp.vec2],
    out_uv: wp.array[wp.vec2],
) -> None:
    # Reassemble the full ``(n_vertices,)`` UV field: fixed vertices keep their prescribed position,
    # interior vertices read the solved value at their compact index.
    i = int(wp.tid())
    if boundary_mask[i]:
        out_uv[i] = fixed_uv[i]
    else:
        ri = interior_map[i]
        out_uv[i] = wp.vec2(wp.float32(sol_x[ri]), wp.float32(sol_y[ri]))


@wp.kernel
def boundary_edge_lengths(
    boundary: wp.array[wp.int32], vertices: wp.array[wp.vec3], out_len: wp.array[wp.float32]
) -> None:
    # Segment length between consecutive boundary vertices; ``out_len[0] = 0`` seeds the arc-length
    # prefix sum (matches ``igl::map_vertices_to_circle``).
    i = int(wp.tid())
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
    i = int(wp.tid())
    n = cumulative_length.shape[0]
    wrap = wp.length(vertices[boundary[0]] - vertices[boundary[n - 1]])
    total = cumulative_length[n - 1] + wrap
    frac = cumulative_length[i] * twc.TWO_PI / total
    out_uv[i] = wp.vec2(wp.cos(frac), wp.sin(frac))
