import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.predicates import unit_tangent


@wp.kernel
def splat_curve_normals(
    vertices: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    normals: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_field: wp.array[wp.vec2d],
) -> None:
    # The signed heat method's source term: each curve segment contributes its own *normal* -- the
    # tangent direction perpendicular to it -- to the two vertices it connects, weighted by half the
    # segment's length. Diffusing normals rather than an indicator is what makes the result signed:
    # the field arrives at a point already knowing which side of the curve it is on.
    s = int(wp.tid())
    a = segments[s, 0]
    b = segments[s, 1]
    edge = vertices[b] - vertices[a]
    length = wp.length(edge)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return
    direction = edge / length
    weight = wp.float64(0.5 * length)

    for k in range(2):
        v = a
        if k == 1:
            v = b
        normal = normals[v]
        # The segment direction as this vertex sees it, then rotated a quarter turn in the tangent
        # plane. ``cross(normal, direction)`` is the left normal, which is the orientation
        # geometry-central signs with: the region a counter-clockwise curve encloses comes out
        # positive.
        tangential, tangential_length = unit_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
        if tangential_length <= TOLERANCE_ZERO_CONSTANT:
            continue
        curve_normal = wp.cross(normal, tangential)
        wp.atomic_add(
            out_field,
            v,
            weight
            * wp.vec2d(
                wp.float64(wp.dot(curve_normal, basis_x[v])),
                wp.float64(wp.dot(curve_normal, basis_y[v])),
            ),
        )


@wp.func
def normalize_or_zero(vector: wp.vec2d) -> wp.vec2d:
    # Away from every source the diffused field decays; where it has decayed to nothing there is no
    # direction left to normalize and zero is the honest answer.
    length = wp.length(vector)
    if length <= wp.float64(TOLERANCE_ZERO_CONSTANT):
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    return vector / length


@wp.kernel
def vertex_field_to_face_field(
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    field: wp.array[wp.vec2d],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_face_field: wp.array[wp.vec3d],
) -> None:
    # Average the three corners' tangent vectors into one per-face vector, in world space, so the
    # existing cotangent divergence can integrate it. Each corner's 2D components mean nothing
    # outside its own frame, so they have to be expanded to 3D *before* averaging.
    f = int(wp.tid())
    normal = normals[f]
    total = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        v = faces[f * 3 + k]
        value = field[v]
        total += wp.float32(value[0]) * basis_x[v] + wp.float32(value[1]) * basis_y[v]
    tangential, _length = unit_tangent(total, normal, TOLERANCE_ZERO_CONSTANT)
    out_face_field[f] = wp.vec3d(
        wp.float64(tangential[0]), wp.float64(tangential[1]), wp.float64(tangential[2])
    )


@wp.kernel
def scatter_free_rhs(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Compact a full-length right-hand side down to the unpinned degrees of freedom, in the layout
    # ``linalg.solve_spd_columns`` expects (one row per right-hand side).
    i = int(wp.tid())
    if fixed_mask[i]:
        return
    out_rhs[0, free_map[i]] = values[i]


@wp.kernel
def gather_free_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    out_field: wp.array[wp.float64],
) -> None:
    # Expand the reduced solution back over every vertex; the pinned ones keep the value they were
    # pinned to, which for a zero level set is zero.
    i = int(wp.tid())
    if fixed_mask[i]:
        out_field[i] = wp.float64(0.0)
        return
    out_field[i] = solution[0, free_map[i]]
