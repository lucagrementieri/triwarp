import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.triangles import face_unit_gradient


@wp.func
def block_mass(mass: wp.float64) -> wp.mat22d:
    # The scalar lumped mass, as one 2x2 block per vertex: the vector problem carries two unknowns
    # per vertex and the same area weight applies to both.
    return wp.mat22d(mass, wp.float64(0.0), wp.float64(0.0), mass)


@wp.kernel
def seed_source_scalars(
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_indicator: wp.array[wp.float64],
    out_weighted: wp.array[wp.float64],
) -> None:
    # Scalar extension needs two right-hand sides: where the sources are, and what they carry.
    s = int(wp.tid())
    v = sources[s]
    wp.atomic_add(out_indicator, v, wp.float64(1.0))
    wp.atomic_add(out_weighted, v, values[s])


@wp.kernel
def seed_source_vectors(
    sources: wp.array[wp.int32], vectors: wp.array[wp.vec2d], out_field: wp.array[wp.vec2d]
) -> None:
    s = int(wp.tid())
    wp.atomic_add(out_field, sources[s], vectors[s])


@wp.func
def divide_positive(
    numerator: wp.float64, denominator: wp.float64, floor: wp.float64
) -> wp.float64:
    # Away from every source the indicator decays to ~0; guard the ratio rather than emit inf.
    # ``floor`` is a fraction of the indicator field's own maximum, never an absolute value -- see
    # ``scale_to_magnitude`` below for why an absolute one is a silent wrong answer on a large mesh.
    if denominator <= floor:
        return wp.float64(0.0)
    return numerator / denominator


@wp.func
def scale_to_magnitude(direction: wp.vec2d, magnitude: wp.float64, floor: wp.float64) -> wp.vec2d:
    # The vector heat method splits a transported vector into a direction (from the vector
    # diffusion) and a magnitude (from a scalar extension): short-time vector diffusion smears
    # magnitudes but preserves directions well.
    #
    # ``floor`` is a fraction of the direction field's own maximum, never an absolute length. The
    # diffused field carries the mesh's scale as ~1/scale^2 -- measured max |direction| of 6.24e-01
    # at unit scale against 6.24e-13 at 1e6 -- so an absolute cutoff turns into "return zero
    # everywhere" on a large mesh: 91 of 162 vertices came back at zero magnitude on an
    # ``icosphere(2)`` scaled by 1e5. Relative, the same field is identical to every digit printed
    # at both scales.
    #
    # The floor separates a vanished direction from a represented one. It does *not* separate signal
    # from round-off and must not be raised in an attempt to: the smallest genuinely diffused value
    # on ``half_torus`` is 8.0e-10 of the maximum, *below* the 8.7e-09 of round-off left at a point
    # where the transported copies cancel exactly. That round-off is the float32 frames' noise
    # rather than the solve's, so it does not move when the conjugate-gradient tolerance is
    # tightened from 1e-8 to 1e-14 (measured: 8.7e-09, then 1.0e-08 at every tighter tolerance).
    # The two populations overlap and no magnitude cut tells them apart.
    length = wp.length(direction)
    if length <= floor:
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    return (magnitude / length) * direction


@wp.func
def to_vec2(v: wp.vec2d) -> wp.vec2:
    return wp.vec2(wp.float32(v[0]), wp.float32(v[1]))


@wp.func
def to_vec2d(v: wp.vec2) -> wp.vec2d:
    return wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))


@wp.func
def tangent_to_world(tangent: wp.vec2, basis_x: wp.vec3, basis_y: wp.vec3) -> wp.vec3:
    return tangent[0] * basis_x + tangent[1] * basis_y


@wp.kernel
def scatter_face_field_to_vertices(
    faces: wp.array[wp.int32],
    face_areas: wp.array[wp.float32],
    field: wp.array[wp.vec3d],
    out_vertex_field: wp.array[wp.vec3],
) -> None:
    # Area-weighted average of a per-face vector field onto vertices, as a plain scatter-add: the
    # weights are the same for all three corners so no normalization is needed before projecting.
    f = int(wp.tid())
    area = face_areas[f]
    value = wp.vec3(
        wp.float32(field[f][0]) * area,
        wp.float32(field[f][1]) * area,
        wp.float32(field[f][2]) * area,
    )
    for k in range(3):
        wp.atomic_add(out_vertex_field, faces[f * 3 + k], value)


@wp.kernel
def face_gradient_unit(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    out_gradient: wp.array[wp.vec3d],
) -> None:
    # For a distance field this points away from the source: the radial direction the log map's
    # angle is measured against.
    f = int(wp.tid())
    out_gradient[f] = face_unit_gradient(vertices, faces, normals, areas, values, wp.int32(f))


@wp.func
def world_to_tangent_unit(value: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3) -> wp.vec2:
    # Express a 3D vertex field in each vertex's tangent basis, normalized. Only the direction
    # survives, which is all the log map's angle needs.
    tangent = wp.vec2(wp.dot(value, basis_x), wp.dot(value, basis_y))
    length = wp.length(tangent)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return wp.vec2(0.0, 0.0)
    return tangent / length


@wp.kernel
def log_map_from_angles(
    radial: wp.array[wp.vec2],
    transported: wp.array[wp.vec2],
    distance: wp.array[wp.float64],
    out_log: wp.array[wp.vec2],
) -> None:
    # Polar coordinates of each vertex as seen from the source.
    #
    # ``transported`` is the source's reference direction parallel-transported to this vertex, and
    # ``radial`` points away from the source here. The angle between them is preserved by transport
    # along the connecting geodesic, so it *is* the angle at which that geodesic leaves the
    # source -- which with the distance gives the vertex's position in the source's tangent plane.
    v = int(wp.tid())
    reference = transported[v]
    outward = radial[v]
    r = wp.float32(distance[v])
    if wp.length(reference) <= TOLERANCE_ZERO_CONSTANT or wp.length(outward) <= (
        TOLERANCE_ZERO_CONSTANT
    ):
        # On the cut locus the transported directions arriving from either side cancel and there is
        # no angle to report -- the log map genuinely has none there. Keep the radius and use angle
        # zero, so the magnitude still means what it should.
        out_log[v] = wp.vec2(r, 0.0)
        return
    angle = wp.atan2(
        reference[0] * outward[1] - reference[1] * outward[0],
        reference[0] * outward[0] + reference[1] * outward[1],
    )
    out_log[v] = wp.vec2(r * wp.cos(angle), r * wp.sin(angle))
