import warp as wp

from triwarp.kernels.predicates import orient2d

# Below this magnitude a normal component is snapped to zero before the spherical conversion in
# ``sweep_transforms``, matching trimesh's ``vector_to_spherical``. Without it a Z+ normal whose
# x/y are float noise gets a meaningless azimuth.
SPHERICAL_SNAP = wp.constant(wp.float32(1e-8))


@wp.func
def project_to_radius(v: wp.vec3, radius: wp.float32) -> wp.vec3:
    # trimesh's icosphere projection, kept in its additive form (v + unit * (r - |v|)) rather
    # than the algebraically equal unit * r, so the float32 result matches the reference bit
    # for bit.
    return v + wp.normalize(v) * (radius - wp.length(v))


@wp.func
def lift_vec2(p: wp.vec2, z: wp.float32) -> wp.vec3:
    # 2D point -> 3D at a fixed height (trimesh's util.stack_3D plus a z offset).
    return wp.vec3(p[0], p[1], z)


@wp.kernel
def reverse_face_winding(faces: wp.array[wp.int32], out_faces: wp.array[wp.int32]) -> None:
    # np.fliplr on an (n, 3) face block. All three indices are read before any is written, so
    # this is safe to run in place (out_faces is faces).
    f = int(wp.tid())
    a = faces[f * 3 + 0]
    b = faces[f * 3 + 1]
    c = faces[f * 3 + 2]
    out_faces[f * 3 + 0] = c
    out_faces[f * 3 + 1] = b
    out_faces[f * 3 + 2] = a


@wp.func
def revolve_template_triangle(t: wp.int32, per: wp.int32) -> wp.vec3i:
    # trimesh's quad template [0, per, 1, 1, per, per + 1] tiled over profile segment i = t // 2
    # and offset by i: two triangles per segment, `per` being the vertex stride between slices.
    i = t / 2
    if t % 2 == 0:
        return wp.vec3i(i, i + per, i + 1)
    return wp.vec3i(i + 1, i + per, i + per + 1)


@wp.func
def revolve_vertex_slot(
    s: wp.int32,
    i: wp.int32,
    n_slices: wp.int32,
    column: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    on_axis: wp.array[wp.bool],
) -> wp.int32:
    # Final index of the vertex at slice `s`, profile point `i`. The three per-profile-point tables
    # are built on the host and encode every coincidence a revolution produces, so the buffer is
    # written in its final, already-merged layout: a profile point on the revolution axis owns one
    # vertex for the whole revolution, and a profile whose last point repeats its first shares that
    # column. The modulus closes a full revolution by folding the last slice onto slice 0.
    j = column[i]
    if on_axis[j]:
        return offsets[j]
    return offsets[j] + s % n_slices


@wp.kernel
def revolve_vertices(
    linestring: wp.array[wp.vec2],
    angle: wp.float32,
    n_points: wp.int32,
    n_slices: wp.int32,
    column: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    on_axis: wp.array[wp.bool],
    out_vertices: wp.array[wp.vec3],
) -> None:
    # The 2D profile X becomes the revolution radius and the 2D profile Y the height along Z. Only
    # the thread that *owns* a slot writes it, so shared slots have a single deterministic writer
    # (slice 0 for an axis point, the representative column for a closed profile) — matching
    # trimesh's merge, which also keeps the first occurrence.
    s, i = wp.tid()
    if column[i] == i and (s == 0 or not on_axis[i]):
        # theta = np.linspace(0, angle, n_points)[s] -- written as a fraction of the span so the
        # final slice lands exactly on `angle` instead of accumulating a step.
        theta = angle * wp.float32(s) / wp.float32(n_points - 1)
        p = linestring[i]
        slot = revolve_vertex_slot(s, i, n_slices, column, offsets, on_axis)
        out_vertices[slot] = wp.vec3(wp.cos(theta) * p[0], wp.sin(theta) * p[0], p[1])


@wp.kernel
def revolve_faces(
    keep: wp.array[wp.int32],
    per: wp.int32,
    n_keep: wp.int32,
    n_slices: wp.int32,
    column: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    on_axis: wp.array[wp.bool],
    out_faces: wp.array[wp.int32],
) -> None:
    # `keep` lists the template triangles that survived the host-side degenerate-area filter. Each
    # template index addresses slice 0 or slice 1 of the profile grid, so it splits into a slice
    # offset and a profile point before being mapped to its final vertex slot.
    s, r = wp.tid()
    tri = revolve_template_triangle(keep[r], per)
    base = (wp.int32(s) * n_keep + wp.int32(r)) * 3
    for k in range(3):
        g = tri[k]
        out_faces[base + k] = revolve_vertex_slot(
            wp.int32(s) + g / per, g % per, n_slices, column, offsets, on_axis
        )


@wp.kernel
def revolve_cap_faces(
    cap_faces: wp.array[wp.int32],
    slice_index: wp.int32,
    reverse: wp.bool,
    n_slices: wp.int32,
    column: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    on_axis: wp.array[wp.bool],
    out_faces: wp.array[wp.int32],
) -> None:
    # Place a profile triangulation on one end slice of a partial revolution. `reverse` reverses the
    # winding (trimesh's np.fliplr) so the far cap faces outward too.
    t = int(wp.tid())
    a = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 0], n_slices, column, offsets, on_axis)
    b = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 1], n_slices, column, offsets, on_axis)
    c = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 2], n_slices, column, offsets, on_axis)
    if reverse:
        out_faces[t * 3 + 0] = c
        out_faces[t * 3 + 1] = b
        out_faces[t * 3 + 2] = a
    else:
        out_faces[t * 3 + 0] = a
        out_faces[t * 3 + 1] = b
        out_faces[t * 3 + 2] = c


@wp.kernel
def offset_cap_faces(
    cap_faces: wp.array[wp.int32], offset: wp.int32, reverse: wp.bool, out_faces: wp.array[wp.int32]
) -> None:
    # Shift a cap triangulation onto one end of a revolved / swept / extruded mesh. `reverse`
    # reverses the winding (trimesh's np.fliplr) so that cap's normals point outward too.
    t = int(wp.tid())
    a = cap_faces[t * 3 + 0] + offset
    b = cap_faces[t * 3 + 1] + offset
    c = cap_faces[t * 3 + 2] + offset
    if reverse:
        out_faces[t * 3 + 0] = c
        out_faces[t * 3 + 1] = b
        out_faces[t * 3 + 2] = a
    else:
        out_faces[t * 3 + 0] = a
        out_faces[t * 3 + 1] = b
        out_faces[t * 3 + 2] = c


@wp.kernel
def triangulation_signed_areas(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_areas: wp.array[wp.float32]
) -> None:
    # Twice the signed area of each 2D triangle: positive for counter-clockwise winding. The mean
    # sign decides whether `extrude_triangulation` has to flip the triangulation to agree with the
    # sign of the extrusion height.
    f = int(wp.tid())
    out_areas[f] = orient2d(
        vertices[faces[f * 3 + 0]], vertices[faces[f * 3 + 1]], vertices[faces[f * 3 + 2]]
    )


@wp.kernel
def extrude_wall_faces(
    boundary: wp.array2d[wp.int32], stride: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # Two triangles bridging boundary edge (a, b) between the bottom cap (indices a, b) and the
    # top cap (a + stride, b + stride). trimesh builds these from a 4-vertex soup per edge and
    # relies on its vertex merge to fuse them onto the caps; indexing the caps directly makes the
    # result watertight by construction, with no merge pass.
    e = int(wp.tid())
    a = boundary[e, 0]
    b = boundary[e, 1]
    out_faces[e * 6 + 0] = b + stride
    out_faces[e * 6 + 1] = a + stride
    out_faces[e * 6 + 2] = b
    out_faces[e * 6 + 3] = b
    out_faces[e * 6 + 4] = a + stride
    out_faces[e * 6 + 5] = a


@wp.func
def path_tangent(path: wp.array[wp.vec3], i: wp.int32) -> wp.vec3:
    # Unit vector of path segment i -> i + 1.
    return wp.normalize(path[i + 1] - path[i])


@wp.kernel
def sweep_plane_normals(
    path: wp.array[wp.vec3], connect_closed: wp.bool, out_normals: wp.array[wp.vec3]
) -> None:
    # One plane normal per path vertex: the end planes lie along their single adjacent segment,
    # interior planes bisect the two. trimesh unitizes the sum rather than halving it because
    # opposing segments can cancel.
    i = int(wp.tid())
    last = path.shape[0] - 1
    normal = wp.vec3(0.0, 0.0, 0.0)
    if i == 0:
        normal = path_tangent(path, 0)
        if connect_closed:
            # A closed path averages the first and last planes so the seam has one frame.
            normal = wp.normalize(normal + path_tangent(path, last - 1))
    elif i == last:
        normal = path_tangent(path, last - 1)
    else:
        normal = wp.normalize(path_tangent(path, i) + path_tangent(path, i - 1))
    out_normals[i] = normal


@wp.func
def snap_spherical(value: wp.float32) -> wp.float32:
    if wp.abs(value) < SPHERICAL_SNAP:
        return 0.0
    return value


@wp.kernel
def sweep_transforms(
    path: wp.array[wp.vec3],
    angles: wp.array[wp.float32],
    normals: wp.array[wp.vec3],
    out_transforms: wp.array[wp.mat44],
) -> None:
    # The rotation taking Z+ onto normals[i], pre-rolled by angles[i], with path[i] as origin.
    # Unrolled by trimesh from inv(Rz(roll) @ Rx(phi) @ Rz(pi/2 - theta)), so it is the identity
    # for a Z+ normal and needs no matrix inverse at runtime.
    i = int(wp.tid())
    normal = normals[i]
    theta = wp.atan2(snap_spherical(normal[1]), snap_spherical(normal[0]))
    phi = wp.acos(snap_spherical(normal[2]))
    cos_theta, sin_theta = wp.cos(theta), wp.sin(theta)
    cos_phi, sin_phi = wp.cos(phi), wp.sin(phi)
    cos_roll, sin_roll = wp.cos(angles[i]), wp.sin(angles[i])
    origin = path[i]
    out_transforms[i] = wp.mat44(
        -sin_roll * cos_phi * cos_theta + sin_theta * cos_roll,
        sin_roll * sin_theta + cos_phi * cos_roll * cos_theta,
        sin_phi * cos_theta,
        origin[0],
        -sin_roll * sin_theta * cos_phi - cos_roll * cos_theta,
        -sin_roll * cos_theta + sin_theta * cos_phi * cos_roll,
        sin_phi * sin_theta,
        origin[1],
        sin_phi * sin_roll,
        -sin_phi * cos_roll,
        cos_phi,
        origin[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def sweep_slice_vertices(
    ring: wp.array[wp.vec2],
    transforms: wp.array[wp.mat44],
    stride: wp.int32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    s, i = wp.tid()
    out_vertices[s * stride + i] = wp.transform_point(transforms[s], lift_vec2(ring[i], 0.0))


@wp.kernel
def sweep_wall_faces(
    boundary: wp.array2d[wp.int32],
    stride: wp.int32,
    n_vertices: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    # Two triangles per boundary edge per slice, bridging slice s to slice s + 1. The modulus
    # wraps the final slice back onto slice 0 when the path is closed and connected; otherwise no
    # index reaches n_vertices and it is a no-op.
    s, e = wp.tid()
    n_boundary = boundary.shape[0]
    offset = wp.int32(s) * stride
    a = boundary[e, 0] + offset
    b = boundary[e, 1] + offset
    base = (wp.int32(s) * n_boundary + wp.int32(e)) * 6
    out_faces[base + 0] = a % n_vertices
    out_faces[base + 1] = b % n_vertices
    out_faces[base + 2] = (a + stride) % n_vertices
    out_faces[base + 3] = (b + stride) % n_vertices
    out_faces[base + 4] = (a + stride) % n_vertices
    out_faces[base + 5] = b % n_vertices


@wp.func
def write_prism_face(
    slot: wp.int32,
    offset: wp.int32,
    a: wp.int32,
    b: wp.int32,
    c: wp.int32,
    flip: wp.bool,
    out_faces: wp.array[wp.int32],
) -> None:
    # One triangle of a prism's 8-face template, offset into its own 6-vertex block. `flip`
    # reverses the winding for prisms whose source triangle faces the plane (trimesh's
    # ``f_seq[cross > 0] = np.fliplr(f)``).
    if flip:
        out_faces[slot * 3 + 0] = offset + c
        out_faces[slot * 3 + 1] = offset + b
        out_faces[slot * 3 + 2] = offset + a
    else:
        out_faces[slot * 3 + 0] = offset + a
        out_faces[slot * 3 + 1] = offset + b
        out_faces[slot * 3 + 2] = offset + c


@wp.kernel
def truncated_prism_geometry(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    transform: wp.mat44,
    inverse: wp.mat44,
    out_vertices: wp.array[wp.vec3],
    out_faces: wp.array[wp.int32],
) -> None:
    # One independent watertight prism per input triangle: the triangle itself, its projection onto
    # the truncation plane, and the six side triangles bridging them. Vertices 0-2 are the source
    # triangle and 3-5 its projection.
    f = int(wp.tid())
    v0 = vertices[faces[f * 3 + 0]]
    v1 = vertices[faces[f * 3 + 1]]
    v2 = vertices[faces[f * 3 + 2]]
    t0 = wp.transform_point(transform, v0)
    t1 = wp.transform_point(transform, v1)
    t2 = wp.transform_point(transform, v2)

    base = f * 6
    out_vertices[base + 0] = v0
    out_vertices[base + 1] = v1
    out_vertices[base + 2] = v2
    out_vertices[base + 3] = wp.transform_point(inverse, wp.vec3(t0[0], t0[1], 0.0))
    out_vertices[base + 4] = wp.transform_point(inverse, wp.vec3(t1[0], t1[1], 0.0))
    out_vertices[base + 5] = wp.transform_point(inverse, wp.vec3(t2[0], t2[1], 0.0))

    flip = wp.cross(t1 - t0, t2 - t0)[2] > 0.0
    slot = f * 8
    write_prism_face(slot + 0, base, 2, 1, 0, flip, out_faces)
    write_prism_face(slot + 1, base, 3, 4, 5, flip, out_faces)
    write_prism_face(slot + 2, base, 0, 1, 4, flip, out_faces)
    write_prism_face(slot + 3, base, 1, 2, 5, flip, out_faces)
    write_prism_face(slot + 4, base, 2, 0, 3, flip, out_faces)
    write_prism_face(slot + 5, base, 4, 3, 0, flip, out_faces)
    write_prism_face(slot + 6, base, 5, 4, 1, flip, out_faces)
    write_prism_face(slot + 7, base, 3, 5, 2, flip, out_faces)


@wp.kernel
def random_soup_vertices(seed: wp.int32, out_vertices: wp.array[wp.vec3]) -> None:
    i = int(wp.tid())
    state = wp.rand_init(seed, i)
    out_vertices[i] = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)
