import warp as wp

from triwarp.constants import PI, TOLERANCE_ZERO_CONSTANT, TWO_PI
from triwarp.kernels.halfedge import halfedge_destination
from triwarp.kernels.tangent_space import corner_angle


@wp.func
def face_normal_of(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], f: wp.int32) -> wp.vec3:
    v0 = vertices[faces[f * 3 + 0]]
    v1 = vertices[faces[f * 3 + 1]]
    v2 = vertices[faces[f * 3 + 2]]
    return wp.normalize(wp.cross(v1 - v0, v2 - v0))


@wp.func
def exit_edge(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    f: wp.int32,
    normal: wp.vec3,
    point: wp.vec3,
    direction: wp.vec3,
    entry_edge: wp.int32,
    length_epsilon: wp.float32,
) -> tuple[wp.int32, wp.float32]:
    # Nearest edge the ray ``point + t * direction`` leaves this triangle through, solving
    # ``point + t d = A + s (B - A)`` in the face's plane for each edge. Two things stop the walk
    # exiting through an edge it already stands on: the edge it entered through is skipped outright,
    # and a crossing nearer than ``length_epsilon`` is rejected -- a walk starting at a *vertex*
    # stands on two edges at once, and without this it spins around the fan making no progress.
    best_edge = wp.int32(-1)
    best_t = wp.float32(0.0)
    for k in range(3):
        if k == entry_edge:
            continue
        a = vertices[faces[f * 3 + k]]
        b = vertices[faces[f * 3 + (k + 1) % 3]]
        edge = b - a
        denom = wp.dot(normal, wp.cross(direction, edge))
        if wp.abs(denom) <= TOLERANCE_ZERO_CONSTANT:
            continue
        t = -wp.dot(normal, wp.cross(point - a, edge)) / denom
        s = wp.dot(normal, wp.cross(point - a, direction)) / -denom
        if t <= length_epsilon or s < wp.float32(0.0) or s > wp.float32(1.0):
            continue
        if best_edge == wp.int32(-1) or t < best_t:
            best_edge = k
            best_t = t
    return best_edge, best_t


@wp.func
def unfold_direction(
    direction: wp.vec3, axis: wp.vec3, normal_from: wp.vec3, normal_to: wp.vec3
) -> wp.vec3:
    # Rotate the direction about the shared edge by the dihedral angle: unfold the two triangles
    # into a common plane and keep walking straight. That is what makes the path a *straightest*
    # geodesic rather than merely a shortest one.
    unit_axis = wp.normalize(axis)
    angle = wp.atan2(
        wp.dot(wp.cross(normal_from, normal_to), unit_axis), wp.dot(normal_from, normal_to)
    )
    rotated = wp.quat_rotate(wp.quat_from_axis_angle(unit_axis, angle), direction)
    # Re-project: the rotation is exact in theory but drifts, and a direction with a component along
    # the new normal would walk off the surface.
    tangential = rotated - wp.dot(rotated, normal_to) * normal_to
    length = wp.length(tangential)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return rotated
    return tangential / length


@wp.func
def trace_walk(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    start_face: wp.int32,
    start_point: wp.vec3,
    start_direction: wp.vec3,
    arc_length: wp.float32,
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    write_begin: wp.int32,
    out_points: wp.array[wp.vec3],
) -> wp.int32:
    # Straightest-geodesic walk from one point, for the arc length ``|start_direction|``. Returns
    # the number of polyline points; writes them from ``write_begin`` when that is non-negative, so
    # the counting and writing passes share this one implementation.
    face = start_face
    normal = face_normal_of(vertices, faces, face)
    point = start_point
    tangential = start_direction - wp.dot(start_direction, normal) * normal
    tangential_length = wp.length(tangential)
    remaining = arc_length

    count = wp.int32(0)
    if write_begin >= wp.int32(0):
        out_points[write_begin] = point
    count += wp.int32(1)
    if remaining <= TOLERANCE_ZERO_CONSTANT or tangential_length <= TOLERANCE_ZERO_CONSTANT:
        return count

    direction = tangential / tangential_length
    entry_edge = wp.int32(-1)
    for _step in range(max_steps):
        edge, distance = exit_edge(
            vertices, faces, face, normal, point, direction, entry_edge, length_epsilon
        )
        if edge == wp.int32(-1):
            # No exit found: a degenerate triangle, or a direction grazing a corner. Stop here
            # rather than leave the surface.
            break
        if distance >= remaining:
            point = point + remaining * direction
            if write_begin >= wp.int32(0):
                out_points[write_begin + count] = point
            count += wp.int32(1)
            remaining = wp.float32(0.0)
            break

        point = point + distance * direction
        remaining -= distance
        if write_begin >= wp.int32(0):
            out_points[write_begin + count] = point
        count += wp.int32(1)

        twin = twins[face * 3 + edge]
        if twin == wp.int32(-1):
            break  # the path ran into the mesh boundary
        a = vertices[faces[face * 3 + edge]]
        b = vertices[faces[face * 3 + (edge + 1) % 3]]
        next_face = twin // wp.int32(3)
        next_normal = face_normal_of(vertices, faces, next_face)
        direction = unfold_direction(direction, b - a, normal, next_normal)
        face = next_face
        normal = next_normal
        entry_edge = twin % wp.int32(3)
    return count


@wp.func
def start_direction_at_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    vertex: wp.int32,
    direction: wp.vec3,
) -> tuple[wp.int32, wp.vec3]:
    # Which incident face a direction leaves the vertex through, and the 3D direction to use.
    #
    # The naive test -- project into each face's plane and ask which wedge contains the result --
    # has no answer for a direction pointing away from the surface, and picks the wrong face near
    # the normal. The intrinsic flattening does have one: rescaling the incident corner angles to a
    # full turn makes the fan a disk, so *every* tangent direction lands in exactly one wedge. Same
    # construction as ``tangent.halfedge_tangent_angles``, same convention geometry-central uses.
    begin = ring_offsets[vertex]
    end = ring_offsets[vertex + 1]
    if end <= begin:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)

    # Polar angle of the direction in the vertex's tangent frame, in [0, 2*pi).
    angle = wp.atan2(wp.dot(direction, basis_y[vertex]), wp.dot(direction, basis_x[vertex]))
    if angle < wp.float32(0.0):
        angle += TWO_PI

    total = float(0.0)  # noqa: UP018 — mutable Warp dynamic variable
    for j in range(begin, end):
        total += corner_angle(face_angles, ring_halfedges[j])
    if total <= TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
    full_turn = TWO_PI
    if is_boundary[vertex]:
        full_turn = PI
    scale = full_turn / total
    if angle > full_turn:
        # A boundary vertex's fan spans half a disk; a direction outside it points off the surface.
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)

    # Walk the ring until the accumulated (rescaled) angle passes the target.
    accumulated = float(0.0)  # noqa: UP018 — mutable Warp dynamic variable
    chosen = ring_halfedges[end - 1]
    offset_in_wedge = float(0.0)  # noqa: UP018 — mutable Warp dynamic variable
    for j in range(begin, end):
        h = ring_halfedges[j]
        wedge = scale * corner_angle(face_angles, h)
        if angle <= accumulated + wedge or j == end - 1:
            chosen = h
            offset_in_wedge = (angle - accumulated) / scale
            break
        accumulated += wedge

    # Undo the rescale: rotate the chosen halfedge's direction by the true in-face angle.
    f = chosen // wp.int32(3)
    normal = face_normal_of(vertices, faces, f)
    edge = vertices[halfedge_destination(faces, chosen)] - vertices[vertex]
    tangential = edge - wp.dot(edge, normal) * normal
    length = wp.length(tangential)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return wp.int32(-1), wp.vec3(0.0, 0.0, 0.0)
    rotation = wp.quat_from_axis_angle(normal, offset_in_wedge)
    return f, wp.quat_rotate(rotation, tangential / length)


@wp.kernel
def trace_from_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    start_faces: wp.array[wp.int32],
    start_bary: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    # One ray per thread. ``offsets`` is empty on the counting pass, which is how the two passes
    # share ``trace_walk``.
    r = int(wp.tid())
    f = start_faces[r]
    bary = start_bary[r]
    point = (
        bary[0] * vertices[faces[f * 3 + 0]]
        + bary[1] * vertices[faces[f * 3 + 1]]
        + bary[2] * vertices[faces[f * 3 + 2]]
    )
    write_begin = wp.int32(-1)
    if offsets.shape[0] > 0:
        write_begin = offsets[r]
    # The trace length is the direction's component in the *face's* plane: a direction leaving the
    # surface traces only what is tangential to it.
    direction = directions[r]
    normal = face_normal_of(vertices, faces, f)
    arc_length = wp.length(direction - wp.dot(direction, normal) * normal)
    out_counts[r] = trace_walk(
        vertices,
        faces,
        twins,
        f,
        point,
        direction,
        arc_length,
        max_steps,
        length_epsilon,
        write_begin,
        out_points,
    )


@wp.kernel
def trace_from_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_angles: wp.array2d[wp.float32],
    ring_offsets: wp.array[wp.int32],
    ring_halfedges: wp.array[wp.int32],
    is_boundary: wp.array[wp.bool],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    vertex_normals: wp.array[wp.vec3],
    start_vertices: wp.array[wp.int32],
    directions: wp.array[wp.vec3],
    max_steps: wp.int32,
    length_epsilon: wp.float32,
    offsets: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
    out_points: wp.array[wp.vec3],
) -> None:
    r = int(wp.tid())
    v = start_vertices[r]
    direction = directions[r]
    f, in_face = start_direction_at_vertex(
        vertices,
        faces,
        face_angles,
        ring_offsets,
        ring_halfedges,
        is_boundary,
        basis_x,
        basis_y,
        v,
        direction,
    )
    write_begin = wp.int32(-1)
    if offsets.shape[0] > 0:
        write_begin = offsets[r]
    if f == wp.int32(-1):
        # An isolated vertex, or a direction pointing out of a boundary vertex's fan: nowhere to go.
        if write_begin >= wp.int32(0):
            out_points[write_begin] = vertices[v]
        out_counts[r] = wp.int32(1)
        return
    # The trace length is measured in the *vertex's* tangent plane, not in the plane of whichever
    # incident face the walk starts in -- the vertex has one tangent space and the fan's faces each
    # tilt differently out of it.
    normal = vertex_normals[v]
    arc_length = wp.length(direction - wp.dot(direction, normal) * normal)
    out_counts[r] = trace_walk(
        vertices,
        faces,
        twins,
        f,
        vertices[v],
        in_face,
        arc_length,
        max_steps,
        length_epsilon,
        write_begin,
        out_points,
    )
