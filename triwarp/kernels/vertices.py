import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.triangles import triangle_cross


@wp.kernel
def face_crosses(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cross: wp.array[wp.vec3]
) -> None:
    f = wp.int32(wp.tid())
    out_cross[f] = triangle_cross(vertices, faces, f)


@wp.kernel
def max_vertex_normal_weights(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unit_face_normals: wp.bool,
    out_weights: wp.array2d[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    cross_len = wp.length(triangle_cross(vertices, faces, f))
    base = f * wp.int32(3)
    for c in range(3):
        i0 = faces[base + c]
        i1 = faces[base + (c + 1) % 3]
        i2 = faces[base + (c + 2) % 3]
        inv_denom = max_corner_inverse_edge_length_sq(vertices, i0, i1, i2)
        if unit_face_normals:
            out_weights[f, c] = cross_len * inv_denom
        else:
            out_weights[f, c] = inv_denom


@wp.func
def max_corner_inverse_edge_length_sq(
    vertices: wp.array[wp.vec3], i0: wp.int32, i1: wp.int32, i2: wp.int32
) -> wp.float32:
    """Per-corner MWSELR factor ``1 / (||e1||^2 * ||e2||^2)``."""
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    denom = wp.length_sq(e1) * wp.length_sq(e2)
    if denom > TOLERANCE_ZERO_CONSTANT:
        return wp.float32(1.0) / denom
    return wp.float32(0.0)
