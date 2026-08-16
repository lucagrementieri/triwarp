"""
Kernels for the heat-method geodesic distance solver (Crane et al. 2013).

Everything runs in ``float64``: far from the source the diffused heat decays exponentially and
would underflow ``float32``, destroying the gradient direction and collapsing the far field. The
per-face half-cotangent weights are reused from ``triwarp.laplacian`` (they are ``O(1)`` and
numerically safe in ``float32``); only the assembled operators, the diffused field, and the two
linear solves need double precision.
"""

import warp as wp

from triwarp.kernels.triangles import corner_triple, face_unit_gradient, face_vertices_vec3d


@wp.kernel
def seed_source_indicator(sources: wp.array[wp.int32], out_u0: wp.array[wp.float64]) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller).
    t = wp.int32(wp.tid())
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
    # X = -grad(u)/|grad(u)|: the unit field pointing *away* from the source, which is the direction
    # the Poisson stage integrates back into a distance.
    f = wp.int32(wp.tid())
    out_x[f] = -face_unit_gradient(vertices, faces, normals, areas, u, f)


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
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
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
