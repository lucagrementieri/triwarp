"""
Kernels for the heat-method geodesic distance solver (Crane et al. 2013).

Everything runs in ``float64``: far from the source the diffused heat decays exponentially and
would underflow ``float32``, destroying the gradient direction and collapsing the far field. The
per-face half-cotangent weights are reused from :mod:`triwarp.laplacian` (they are ``O(1)`` and
numerically safe in ``float32``); only the assembled operators, the diffused field, and the two
linear solves need double precision.
"""

import warp as wp


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.kernel
def cotmatrix_triplets_f64(
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float64],
) -> None:
    # float64 rebuild of ``kernel_laplacian.cotmatrix_triplets`` (igl convention: negative
    # diagonal). Reuses the shared float32 half-cotangent weights.
    f = int(wp.tid())
    for e in range(3):
        c0 = (e + 1) % 3
        c1 = (e + 2) % 3
        source = faces[f * 3 + c0]
        dest = faces[f * 3 + c1]
        w = wp.float64(cot_entries[f, e])
        base = f * 12 + e * 4
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


@wp.kernel
def vertex_lumped_mass(
    faces: wp.array[wp.int32], areas: wp.array[wp.float32], out_mass: wp.array[wp.float64]
) -> None:
    # Barycentric (lumped) mass: each face donates a third of its area to each incident vertex.
    f = int(wp.tid())
    third = wp.float64(areas[f]) / wp.float64(3.0)
    wp.atomic_add(out_mass, faces[f * 3 + 0], third)
    wp.atomic_add(out_mass, faces[f * 3 + 1], third)
    wp.atomic_add(out_mass, faces[f * 3 + 2], third)


@wp.kernel
def seed_source_indicator(sources: wp.array[wp.int32], out_u0: wp.array[wp.float64]) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller).
    t = int(wp.tid())
    out_u0[sources[t]] = wp.float64(1.0)


@wp.kernel
def face_gradient_normalized(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    u: wp.array[wp.float64],
    out_x: wp.array[wp.vec3d],
) -> None:
    # Per-face gradient of the scalar field u, then X = -grad(u)/|grad(u)| (unit, points away
    # from the source).  grad(u) = 1/(2A) * sum_i u_i (n x e_i^opp), e_i^opp the CCW edge opposite
    # vertex i. Geometry is read in float32 (input precision) and promoted; u is float64.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = to_vec3d(vertices[i0])
    v1 = to_vec3d(vertices[i1])
    v2 = to_vec3d(vertices[i2])
    n = to_vec3d(normals[f])
    area = wp.float64(areas[f])

    grad = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if area > wp.float64(0.0):
        e0 = v2 - v1  # opposite vertex i0
        e1 = v0 - v2  # opposite vertex i1
        e2 = v1 - v0  # opposite vertex i2
        grad = (u[i0] * wp.cross(n, e0) + u[i1] * wp.cross(n, e1) + u[i2] * wp.cross(n, e2)) / (
            wp.float64(2.0) * area
        )

    length = wp.length(grad)
    unit = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if length > wp.float64(0.0):
        unit = -grad / length
    out_x[f] = unit


@wp.kernel
def integrated_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    field: wp.array[wp.vec3d],
    out_div: wp.array[wp.float64],
) -> None:
    # Cotangent integrated divergence of the per-face vector field, accumulated per vertex.
    # cot_entries[f, k] = 1/2 cot(angle at corner k); each vertex gets contributions from the two
    # edges of the triangle incident to it, weighted by the cotangent opposite those edges.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = to_vec3d(vertices[i0])
    v1 = to_vec3d(vertices[i1])
    v2 = to_vec3d(vertices[i2])
    x = field[f]
    c0 = wp.float64(cot_entries[f, 0])
    c1 = wp.float64(cot_entries[f, 1])
    c2 = wp.float64(cot_entries[f, 2])

    d0 = c2 * wp.dot(v1 - v0, x) + c1 * wp.dot(v2 - v0, x)
    d1 = c0 * wp.dot(v2 - v1, x) + c2 * wp.dot(v0 - v1, x)
    d2 = c1 * wp.dot(v0 - v2, x) + c0 * wp.dot(v1 - v2, x)

    wp.atomic_add(out_div, i0, d0)
    wp.atomic_add(out_div, i1, d1)
    wp.atomic_add(out_div, i2, d2)


@wp.kernel
def negate_field(field: wp.array[wp.float64], out_field: wp.array[wp.float64]) -> None:
    # Flip sign so the Poisson right-hand side matches the positive semi-definite operator ``-L``.
    t = int(wp.tid())
    out_field[t] = -field[t]


@wp.kernel
def shift_field(offset: wp.float64, out_field: wp.array[wp.float64]) -> None:
    # Subtract a constant so the distance is zero at the source set (in place).
    t = int(wp.tid())
    out_field[t] = out_field[t] - offset
