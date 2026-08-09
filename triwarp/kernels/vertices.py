import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT, TWO_PI


@wp.kernel
def face_crosses(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cross: wp.array[wp.vec3]
) -> None:
    f = wp.tid()
    face = faces[f * 3 : (f + 1) * 3]
    v0 = vertices[face[0]]
    e1 = vertices[face[1]] - v0
    e2 = vertices[face[2]] - v0
    out_cross[f] = wp.cross(e1, e2)


@wp.kernel
def max_vertex_normal_weights(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unit_face_normals: wp.bool,
    out_weights: wp.array2d[wp.float32],
) -> None:
    f = wp.tid()
    face = faces[f * 3 : (f + 1) * 3]
    v0 = vertices[face[0]]
    e1 = vertices[face[1]] - v0
    e2 = vertices[face[2]] - v0
    cross_len = wp.length(wp.cross(e1, e2))
    for c in range(3):
        i0 = face[c]
        i1 = face[(c + 1) % 3]
        i2 = face[(c + 2) % 3]
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


@wp.func
def angle_defect(angle_sum: wp.float32) -> wp.float32:
    """Angle defect at a vertex: a full turn minus the incident corner angles."""
    return TWO_PI - angle_sum
