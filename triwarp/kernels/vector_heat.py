import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.halfedge import halfedge_destination


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
    # ``laplacian.cotmatrix_triplets`` but with each off-diagonal weight turned into a rotation: the
    # two endpoints of an edge measure tangent directions from different reference directions, so a
    # difference between them is only meaningful after transporting one into the other's frame.
    #
    # Built positive semi-definite (positive diagonal), unlike ``cotmatrix``'s igl sign convention,
    # because every consumer here feeds it straight to a conjugate-gradient solve.
    f = int(wp.tid())
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
def block_mass(mass: wp.array[wp.float64], out_blocks: wp.array[wp.mat22d]) -> None:
    # The scalar lumped mass, as one 2x2 block per vertex: the vector problem carries two unknowns
    # per vertex and the same area weight applies to both.
    i = int(wp.tid())
    m = mass[i]
    out_blocks[i] = wp.mat22d(m, 0.0, 0.0, m)


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
def divide_positive(numerator: wp.float64, denominator: wp.float64) -> wp.float64:
    # Away from every source the indicator decays to ~0; guard the ratio rather than emit inf.
    if denominator <= wp.float64(TOLERANCE_ZERO_CONSTANT):
        return wp.float64(0.0)
    return numerator / denominator


@wp.func
def scale_to_magnitude(direction: wp.vec2d, magnitude: wp.float64) -> wp.vec2d:
    # The vector heat method splits a transported vector into a direction (from the vector
    # diffusion) and a magnitude (from a scalar extension): short-time vector diffusion smears
    # magnitudes but preserves directions well.
    length = wp.length(direction)
    if length <= wp.float64(TOLERANCE_ZERO_CONSTANT):
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
    # Unit gradient of a per-vertex scalar within each triangle. For a distance field this points
    # away from the source, which is the radial direction the log map is measured against.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    normal = normals[f]
    area = areas[f]
    if area <= TOLERANCE_ZERO_CONSTANT:
        out_gradient[f] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        return
    # grad = sum_k value_k * (normal x opposite_edge_k) / (2 * area)
    gradient = (
        wp.float32(values[i0]) * wp.cross(normal, vertices[i2] - vertices[i1])
        + wp.float32(values[i1]) * wp.cross(normal, vertices[i0] - vertices[i2])
        + wp.float32(values[i2]) * wp.cross(normal, vertices[i1] - vertices[i0])
    ) / (2.0 * area)
    length = wp.length(gradient)
    if length > TOLERANCE_ZERO_CONSTANT:
        gradient = gradient / length
    out_gradient[f] = wp.vec3d(
        wp.float64(gradient[0]), wp.float64(gradient[1]), wp.float64(gradient[2])
    )


@wp.kernel
def world_to_tangent_unit(
    field: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_tangent: wp.array[wp.vec2],
) -> None:
    # Express a 3D vertex field in each vertex's tangent basis, normalized. Only the direction
    # survives, which is all the log map's angle needs.
    v = int(wp.tid())
    value = field[v]
    tangent = wp.vec2(wp.dot(value, basis_x[v]), wp.dot(value, basis_y[v]))
    length = wp.length(tangent)
    if length <= TOLERANCE_ZERO_CONSTANT:
        out_tangent[v] = wp.vec2(0.0, 0.0)
        return
    out_tangent[v] = tangent / length


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
