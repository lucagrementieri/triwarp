import math

import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import lift_vec2
from triwarp.kernels.polyline import segment_displacement
from triwarp.kernels.predicates import orient2d
from triwarp.kernels.triangles import write_corner_triple_reversible

SQRT3 = wp.constant(wp.float32(math.sqrt(3.0)))
PI_F = wp.constant(wp.float32(math.pi))

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


@wp.kernel
def grid_vertices(
    nx: wp.int32,
    ny: wp.int32,
    width: wp.float64,
    height: wp.float64,
    origin_x: wp.float64,
    origin_y: wp.float64,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # The (nx, ny) lattice of a flat patch in the z = 0 plane, row-major with X the slow axis, one
    # thread per vertex.
    #
    # Written as ``extent * (k / (n - 1))`` rather than ``k * (extent / (n - 1))``: the fraction is
    # exactly 1 at the last sample, so the far edge lands on the extent without the endpoint
    # special case ``numpy.linspace`` needs. The arithmetic is float64 for the same reason the host
    # build was -- the float32 store then rounds off an exact value rather than an accumulated one.
    i, j = wp.tid()
    x = origin_x + width * (wp.float64(i) / wp.float64(nx - 1))
    y = origin_y + height * (wp.float64(j) / wp.float64(ny - 1))
    # ``wp.float32(...)``, not ``wp.cast``: the latter is a same-size bit reinterpretation and
    # fails to compile on a float64 source ("source and destination must have the same size").
    out_vertices[i * ny + j] = wp.vec3(wp.float32(x), wp.float32(y), wp.float32(0.0))


@wp.kernel
def grid_faces(ny: wp.int32, out_faces: wp.array[wp.int32]) -> None:
    # The two triangles of one quad cell, one thread per cell, wound counter-clockwise seen from
    # +Z. With X the slow axis a cell's corners are ``corner``, ``corner + ny`` (next X) and
    # ``+ 1`` (next Y), and cell ``(i, j)`` owns face slots ``2 * (i * (ny - 1) + j)`` and the next.
    i, j = wp.tid()
    corner = i * ny + j
    slot = (i * (ny - 1) + j) * 6
    out_faces[slot + 0] = corner
    out_faces[slot + 1] = corner + ny
    out_faces[slot + 2] = corner + ny + 1
    out_faces[slot + 3] = corner
    out_faces[slot + 4] = corner + ny + 1
    out_faces[slot + 5] = corner + 1


# Vertex and edge counts of the base icosahedron, which set the two block boundaries of the
# icosphere's closed-form vertex numbering: the 12 corners, then 30 blocks of ``n - 1`` base-edge
# points, then 20 blocks of face-interior points.
ICOSAHEDRON_VERTICES = wp.constant(wp.int32(12))
ICOSAHEDRON_EDGES = wp.constant(wp.int32(30))


@wp.func
def icosphere_vertex_index(
    table: wp.array2d[wp.int32], n: wp.int32, f: wp.int32, i: wp.int32, j: wp.int32
) -> wp.int32:
    # Global index of the icosphere vertex at barycentric ``(i, j, n - i - j)`` of base face ``f``,
    # under the numbering described in the ``table`` docstring of ``_icosphere_face_table``: the 12
    # base corners, then the interior points of each base edge from its lower-numbered endpoint,
    # then the interior points of each base face in row-major barycentric order.
    #
    # The map is many-to-one on purpose: a point on a shared base edge is addressed by both of the
    # faces holding it and lands on the same index from either side, which is what makes the result
    # watertight with no merge pass.
    k = n - i - j
    index = 0
    if i == n:
        index = table[f, 0]
    elif j == n:
        index = table[f, 1]
    elif k == n:
        index = table[f, 2]
    elif k == 0 or i == 0 or j == 0:
        # On a base edge. ``side`` is 0 for a->b (parameter j), 1 for b->c (k), 2 for c->a (i),
        # and the stored flag says whether that side runs against the edge's own direction.
        side = 0
        t = j
        if i == 0:
            side = 1
            t = k
        elif j == 0:
            side = 2
            t = i
        if table[f, 6 + side] != 0:
            t = n - t
        index = ICOSAHEDRON_VERTICES + table[f, 3 + side] * (n - 1) + t - 1
    else:
        interior = (n - 1) * (n - 2) // 2
        local = (i - 1) * (n - 1) - (i - 1) * i // 2 + (j - 1)
        index = ICOSAHEDRON_VERTICES + ICOSAHEDRON_EDGES * (n - 1) + f * interior + local
    return index


@wp.kernel
def icosphere_generation(
    table: wp.array2d[wp.int32],
    n: wp.int32,
    level: wp.int32,
    radius: wp.float32,
    out_vertices: wp.array[wp.vec3],
) -> None:
    # One refinement generation of the icosphere, in place: every vertex introduced at this level
    # is the projected midpoint of an edge of the previous level, and both of that edge's endpoints
    # are already written. The grid is one thread per (base face, level-local barycentric pair), so
    # a vertex on a shared base edge is computed twice from the same two parents -- identical
    # arithmetic, identical result.
    f, bi, bj = wp.tid()
    bk = level - bi - bj
    if bk < 0:
        return
    # A vertex is new at this level exactly when its level-local coordinates have two odd entries;
    # all-even means it already existed one level up.
    if (bi & 1) + (bj & 1) + (bk & 1) != 2:
        return

    step = n // level
    i = bi * step
    j = bj * step
    # The two parents are the ends of the previous level's edge this vertex bisects, which is the
    # pair of coordinates that came out odd.
    i0 = i
    j0 = j
    i1 = i
    j1 = j
    if (bi & 1) != 0 and (bj & 1) != 0:
        i0 = i + step
        j0 = j - step
        i1 = i - step
        j1 = j + step
    elif (bj & 1) != 0:
        j0 = j + step
        j1 = j - step
    else:
        i0 = i + step
        i1 = i - step

    a = icosphere_vertex_index(table, n, f, i0, j0)
    b = icosphere_vertex_index(table, n, f, i1, j1)
    # Lerp from the lower index, matching ``remesh.subdivide``'s sorted unique edges, so the
    # float32 midpoint is the same one the iterated-subdivide implementation produced.
    lo = wp.min(a, b)
    hi = wp.max(a, b)
    midpoint = wp.lerp(out_vertices[lo], out_vertices[hi], wp.float32(0.5))
    out_vertices[icosphere_vertex_index(table, n, f, i, j)] = project_to_radius(midpoint, radius)


@wp.kernel
def icosphere_faces(
    table: wp.array2d[wp.int32], n: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # The ``n ** 2`` sub-triangles of one base face, one per thread over the full ``n x n`` block.
    # The ``n (n + 1) / 2`` upward triangles are the threads with ``i + j < n``; the rest of the
    # block is remapped by ``(i, j) -> (n - 1 - i, n - 1 - j)`` onto the ``n (n - 1) / 2``
    # downward ones, which is a bijection -- so every thread writes exactly one triangle and no
    # prefix-sum over rows is needed.
    f, i, j = wp.tid()
    slot = (f * n * n + i * n + j) * 3
    if i + j < n:
        out_faces[slot + 0] = icosphere_vertex_index(table, n, f, i + 1, j)
        out_faces[slot + 1] = icosphere_vertex_index(table, n, f, i, j + 1)
        out_faces[slot + 2] = icosphere_vertex_index(table, n, f, i, j)
    else:
        di = n - 1 - i
        dj = n - 1 - j
        out_faces[slot + 0] = icosphere_vertex_index(table, n, f, di, dj + 1)
        out_faces[slot + 1] = icosphere_vertex_index(table, n, f, di + 1, dj)
        out_faces[slot + 2] = icosphere_vertex_index(table, n, f, di + 1, dj + 1)


@wp.func
def revolve_template_triangle(t: wp.int32, per: wp.int32) -> wp.vec3i:
    # trimesh's quad template [0, per, 1, 1, per, per + 1] tiled over profile segment i = t // 2
    # and offset by i: two triangles per segment, `per` being the vertex stride between slices.
    i = t // 2
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
    base = (s * n_keep + r) * 3
    for k in range(3):
        g = tri[k]
        out_faces[base + k] = revolve_vertex_slot(
            s + g // per, g % per, n_slices, column, offsets, on_axis
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
    t = wp.int32(wp.tid())
    a = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 0], n_slices, column, offsets, on_axis)
    b = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 1], n_slices, column, offsets, on_axis)
    c = revolve_vertex_slot(slice_index, cap_faces[t * 3 + 2], n_slices, column, offsets, on_axis)
    write_corner_triple_reversible(out_faces, t, a, b, c, reverse)


@wp.kernel
def offset_cap_faces_both(
    cap_faces: wp.array[wp.int32], far_offset: wp.int32, out_faces: wp.array[wp.int32]
) -> None:
    # Both caps of an extruded or swept solid in one launch. Row 0 is the near cap -- no vertex
    # offset, winding reversed so its normals point outward -- and row 1 is the far cap, shifted by
    # ``far_offset`` and keeping the input winding. Every caller writes the two into adjacent
    # blocks of one buffer, which is what lets a single launch cover them.
    end, t = wp.tid()
    n_cap = cap_faces.shape[0] // 3
    offset = wp.where(end == 0, wp.int32(0), far_offset)
    a = cap_faces[t * 3 + 0] + offset
    b = cap_faces[t * 3 + 1] + offset
    c = cap_faces[t * 3 + 2] + offset
    write_corner_triple_reversible(out_faces, end * n_cap + t, a, b, c, end == 0)


@wp.kernel
def triangulation_signed_areas(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], out_areas: wp.array[wp.float32]
) -> None:
    # Twice the signed area of each 2D triangle: positive for counter-clockwise winding. The mean
    # sign decides whether `extrude_triangulation` has to flip the triangulation to agree with the
    # sign of the extrusion height.
    f = wp.int32(wp.tid())
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
    e = wp.int32(wp.tid())
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
    return wp.normalize(segment_displacement(path, i))


@wp.func
def snap_spherical(value: wp.float32) -> wp.float32:
    if wp.abs(value) < SPHERICAL_SNAP:
        return 0.0
    return value


@wp.kernel
def sweep_transforms(
    path: wp.array[wp.vec3],
    angles: wp.array[wp.float32],
    connect_closed: wp.bool,
    out_transforms: wp.array[wp.mat44],
) -> None:
    # The rotation taking Z+ onto normals[i], pre-rolled by angles[i], with path[i] as origin.
    # Unrolled by trimesh from inv(Rz(roll) @ Rx(phi) @ Rz(pi/2 - theta)), so it is the identity
    # for a Z+ normal and needs no matrix inverse at runtime.
    i = wp.int32(wp.tid())
    # One plane normal per path vertex, formed here rather than in a buffer of its own: the end
    # planes lie along their single adjacent segment, interior planes bisect the two. trimesh
    # unitizes the sum rather than halving it because opposing segments can cancel.
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
    # A degenerate (near-zero) normal has no direction to convert -- two consecutive path tangents
    # cancelling at a sharp path reversal, which the branch above can produce -- so leave both
    # angles at the Z+-identity zero rather than computing one, matching trimesh's
    # ``vector_to_spherical`` (``unitize(..., check_valid=True)`` marks such a row invalid and its
    # spherical angles stay at their zeroed default). Without this, ``wp.acos(0.0) == pi/2`` here
    # would instead rotate local +Z onto +X -- an arbitrary ~90 degree twist, not the identity a
    # degenerate plane should fall back to. Initialized before the branch, per the kernel-scope
    # conditional-scoping rule.
    theta = wp.float32(0.0)
    phi = wp.float32(0.0)
    if wp.length(normal) > TOLERANCE_ZERO_CONSTANT:
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
    out_vertices[s * stride + i] = wp.transform_point(
        transforms[s], lift_vec2(ring[i], wp.float32(0.0))
    )


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
    offset = s * stride
    a = boundary[e, 0] + offset
    b = boundary[e, 1] + offset
    base = (s * n_boundary + e) * 6
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
    write_corner_triple_reversible(out_faces, slot, offset + a, offset + b, offset + c, flip)


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
    f = wp.int32(wp.tid())
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
    i = wp.int32(wp.tid())
    state = wp.rand_init(seed, i)
    out_vertices[i] = wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5, wp.randf(state) - 0.5)


# --- parametric surfaces -----------------------------------------------------------------
#
# One @wp.func per analytic surface, each the map VTK's ``vtkParametric*::Evaluate`` computes and in
# VTK's own frame (several are an x/y swap or a z flip away from the textbook form, which is folded
# into the expressions here). Verified against ``Evaluate`` over the whole sampled lattice: the
# largest disagreement is 2.5e-06 on a surface of scale 13.5, i.e. float32 rounding. The dispatch
# below is a warp-uniform branch over an int kind, so the whole family compiles into one module --
# ``wp.launch`` cannot take a ``wp.Function`` argument, and a kernel factory per surface would build
# eighteen kernels for one ``Literal``.

SURFACE_BOHEMIAN_DOME = wp.constant(wp.int32(0))
SURFACE_BOUR = wp.constant(wp.int32(1))
SURFACE_BOY = wp.constant(wp.int32(2))
SURFACE_CATALAN_MINIMAL = wp.constant(wp.int32(3))
SURFACE_CONIC_SPIRAL = wp.constant(wp.int32(4))
SURFACE_CROSS_CAP = wp.constant(wp.int32(5))
SURFACE_DINI = wp.constant(wp.int32(6))
SURFACE_ENNEPER = wp.constant(wp.int32(7))
SURFACE_FIGURE8_KLEIN = wp.constant(wp.int32(8))
SURFACE_HENNEBERG = wp.constant(wp.int32(9))
SURFACE_KLEIN = wp.constant(wp.int32(10))
SURFACE_KUEN = wp.constant(wp.int32(11))
SURFACE_MOBIUS = wp.constant(wp.int32(12))
SURFACE_PLUCKER_CONOID = wp.constant(wp.int32(13))
SURFACE_PSEUDOSPHERE = wp.constant(wp.int32(14))
SURFACE_ROMAN = wp.constant(wp.int32(15))
SURFACE_SUPER_ELLIPSOID = wp.constant(wp.int32(16))
SURFACE_SUPER_TOROID = wp.constant(wp.int32(17))

# ``vtkParametricKuen::DeltaV0``: the v it substitutes for the v = 0 row, where ``log(tan(v / 2))``
# is -inf. Its default, and the value the surface "has the best appearance with".
KUEN_DELTA_V0 = wp.constant(wp.float32(0.05))


@wp.func
def signed_power(value: wp.float32, exponent: wp.float32) -> wp.float32:
    """``sign(value) * |value| ** exponent``, the superquadric shape function."""
    return wp.sign(value) * wp.pow(wp.abs(value), exponent)


@wp.func
def surface_bohemian_dome(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(0.5 * wp.cos(u), 1.5 * wp.cos(v) + 0.5 * wp.sin(u), wp.sin(v))


@wp.func
def surface_bour(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        u * wp.cos(v) - 0.5 * u * u * wp.cos(2.0 * v),
        -u * wp.sin(v) - 0.5 * u * u * wp.sin(2.0 * v),
        4.0 / 3.0 * wp.pow(u, 1.5) * wp.cos(1.5 * v),
    )


@wp.func
def surface_boy(u: wp.float32, v: wp.float32) -> wp.vec3:
    # VTK evaluates a *polynomial* Steiner-type immersion of the unit-sphere point, not the rational
    # Apery form -- there is no denominator, and ``ZScale`` (0.125) scales only the third component.
    a = wp.cos(u) * wp.sin(v)
    b = wp.sin(u) * wp.sin(v)
    c = wp.cos(v)
    s = a + b + c
    return wp.vec3(
        0.5
        * (
            2.0 * a * a
            - b * b
            - c * c
            + 2.0 * b * c * (b * b - c * c)
            + c * a * (a * a - c * c)
            + a * b * (b * b - a * a)
        ),
        0.5 * SQRT3 * (b * b - c * c + c * a * (c * c - a * a) + a * b * (b * b - a * a)),
        0.125 * s * (s * s * s + 4.0 * (b - a) * (c - b) * (a - c)),
    )


@wp.func
def surface_catalan_minimal(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        u - wp.sin(u) * wp.cosh(v),
        1.0 - wp.cos(u) * wp.cosh(v),
        4.0 * wp.sin(0.5 * u) * wp.sinh(0.5 * v),
    )


@wp.func
def surface_conic_spiral(u: wp.float32, v: wp.float32) -> wp.vec3:
    taper = 1.0 - v / (2.0 * PI_F)
    return wp.vec3(
        0.2 * taper * wp.cos(2.0 * v) * (1.0 + wp.cos(u)) + 0.1 * wp.cos(2.0 * v),
        0.2 * taper * wp.sin(2.0 * v) * (1.0 + wp.cos(u)) + 0.1 * wp.sin(2.0 * v),
        v / (2.0 * PI_F) + 0.2 * taper * wp.sin(u),
    )


@wp.func
def surface_cross_cap(u: wp.float32, v: wp.float32) -> wp.vec3:
    cu, su = wp.cos(u), wp.sin(u)
    cv, sv = wp.cos(v), wp.sin(v)
    return wp.vec3(cu * wp.sin(2.0 * v), su * wp.sin(2.0 * v), cv * cv - cu * cu * sv * sv)


@wp.func
def surface_dini(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        wp.cos(u) * wp.sin(v), wp.sin(u) * wp.sin(v), wp.cos(v) + wp.log(wp.tan(0.5 * v)) + 0.2 * u
    )


@wp.func
def surface_enneper(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(u - u * u * u / 3.0 + u * v * v, v - v * v * v / 3.0 + v * u * u, u * u - v * v)


@wp.func
def surface_figure8_klein(u: wp.float32, v: wp.float32) -> wp.vec3:
    half_u = 0.5 * u
    radial = 1.0 + wp.cos(half_u) * wp.sin(v) - 0.5 * wp.sin(half_u) * wp.sin(2.0 * v)
    return wp.vec3(
        radial * wp.cos(u),
        radial * wp.sin(u),
        wp.sin(half_u) * wp.sin(v) + 0.5 * wp.cos(half_u) * wp.sin(2.0 * v),
    )


@wp.func
def surface_henneberg(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(
        2.0 * wp.sinh(u) * wp.cos(v) - 2.0 / 3.0 * wp.sinh(3.0 * u) * wp.cos(3.0 * v),
        2.0 * wp.sinh(u) * wp.sin(v) + 2.0 / 3.0 * wp.sinh(3.0 * u) * wp.sin(3.0 * v),
        2.0 * wp.cosh(2.0 * u) * wp.cos(2.0 * v),
    )


@wp.func
def surface_klein(u: wp.float32, v: wp.float32) -> wp.vec3:
    cu, su = wp.cos(u), wp.sin(u)
    cv, sv = wp.cos(v), wp.sin(v)
    cu2 = cu * cu
    cu3 = cu2 * cu
    cu4 = cu2 * cu2
    cu5 = cu4 * cu
    cu6 = cu4 * cu2
    cu7 = cu6 * cu
    return wp.vec3(
        -2.0
        / 15.0
        * cu
        * (3.0 * cv - 30.0 * su + 90.0 * cu4 * su - 60.0 * cu6 * su + 5.0 * cu * cv * su),
        -1.0
        / 15.0
        * su
        * (
            3.0 * cv
            - 3.0 * cu2 * cv
            - 48.0 * cu4 * cv
            + 48.0 * cu6 * cv
            - 60.0 * su
            + 5.0 * cu * cv * su
            - 5.0 * cu3 * cv * su
            - 80.0 * cu5 * cv * su
            + 80.0 * cu7 * cv * su
        ),
        2.0 / 15.0 * (3.0 + 5.0 * cu * su) * sv,
    )


@wp.func
def surface_kuen(u: wp.float32, v: wp.float32) -> wp.vec3:
    # Both ends of the v domain are singular and VTK names a value at each: it substitutes
    # ``DeltaV0`` for v = 0, and reports v = pi as the pole (0, 0, -1) rather than the +inf the
    # limit of ``log(tan(v / 2))`` actually goes to. Both rows are sampled, so both are replicated
    # here -- and the second guard doubles as the float32 one, since ``tan(v / 2)`` turns negative
    # a single ulp past pi and ``log`` of it is NaN.
    v_safe = v
    if v_safe <= 0.0:
        v_safe = KUEN_DELTA_V0
    sv = wp.sin(v_safe)
    position = wp.vec3(0.0, 0.0, -1.0)
    if sv > 0.0:
        denominator = 1.0 + u * u * sv * sv
        # VTK's frame swaps the first two components relative to the textbook form.
        position = wp.vec3(
            2.0 * (wp.sin(u) - u * wp.cos(u)) * sv / denominator,
            2.0 * (wp.cos(u) + u * wp.sin(u)) * sv / denominator,
            wp.log(wp.tan(0.5 * v_safe)) + 2.0 * wp.cos(v_safe) / denominator,
        )
    return position


@wp.func
def surface_mobius(u: wp.float32, v: wp.float32) -> wp.vec3:
    # VTK does not halve v, and the half-angle roles are the opposite of the usual writing:
    # the radius carries sin(u/2) and the height cos(u/2). First two components swapped.
    radial = 1.0 - v * wp.sin(0.5 * u)
    return wp.vec3(radial * wp.sin(u), radial * wp.cos(u), v * wp.cos(0.5 * u))


@wp.func
def surface_plucker_conoid(u: wp.float32, v: wp.float32) -> wp.vec3:
    return wp.vec3(u * wp.sin(v), u * wp.cos(v), wp.sin(2.0 * v))


@wp.func
def surface_pseudosphere(u: wp.float32, v: wp.float32) -> wp.vec3:
    sech = 1.0 / wp.cosh(u)
    return wp.vec3(sech * wp.cos(v), sech * wp.sin(v), u - wp.tanh(u))


@wp.func
def surface_roman(u: wp.float32, v: wp.float32) -> wp.vec3:
    cv = wp.cos(v)
    s2v = wp.sin(2.0 * v)
    return wp.vec3(0.5 * cv * cv * wp.sin(2.0 * u), 0.5 * wp.sin(u) * s2v, 0.5 * wp.cos(u) * s2v)


@wp.func
def surface_super_ellipsoid(
    u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32
) -> wp.vec3:
    cv = signed_power(wp.cos(v), n1)
    sv = signed_power(wp.sin(v), n1)
    cu = signed_power(wp.cos(u), n2)
    su = signed_power(wp.sin(u), n2)
    # First two components swapped, matching VTK's frame.
    return wp.vec3(cv * su, cv * cu, sv)


@wp.func
def surface_super_toroid(u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32) -> wp.vec3:
    cu = signed_power(wp.cos(u), n2)
    su = signed_power(wp.sin(u), n2)
    cv = signed_power(wp.cos(v), n1)
    sv = signed_power(wp.sin(v), n1)
    return wp.vec3(su * (1.0 + 0.5 * cv), cu * (1.0 + 0.5 * cv), 0.5 * sv)


@wp.func
def parametric_position(
    kind: wp.int32, u: wp.float32, v: wp.float32, n1: wp.float32, n2: wp.float32
) -> wp.vec3:
    """Evaluate surface ``kind`` at ``(u, v)``; ``n1`` / ``n2`` are for the superquadrics only."""
    position = wp.vec3()
    if kind == SURFACE_BOHEMIAN_DOME:
        position = surface_bohemian_dome(u, v)
    elif kind == SURFACE_BOUR:
        position = surface_bour(u, v)
    elif kind == SURFACE_BOY:
        position = surface_boy(u, v)
    elif kind == SURFACE_CATALAN_MINIMAL:
        position = surface_catalan_minimal(u, v)
    elif kind == SURFACE_CONIC_SPIRAL:
        position = surface_conic_spiral(u, v)
    elif kind == SURFACE_CROSS_CAP:
        position = surface_cross_cap(u, v)
    elif kind == SURFACE_DINI:
        position = surface_dini(u, v)
    elif kind == SURFACE_ENNEPER:
        position = surface_enneper(u, v)
    elif kind == SURFACE_FIGURE8_KLEIN:
        position = surface_figure8_klein(u, v)
    elif kind == SURFACE_HENNEBERG:
        position = surface_henneberg(u, v)
    elif kind == SURFACE_KLEIN:
        position = surface_klein(u, v)
    elif kind == SURFACE_KUEN:
        position = surface_kuen(u, v)
    elif kind == SURFACE_MOBIUS:
        position = surface_mobius(u, v)
    elif kind == SURFACE_PLUCKER_CONOID:
        position = surface_plucker_conoid(u, v)
    elif kind == SURFACE_PSEUDOSPHERE:
        position = surface_pseudosphere(u, v)
    elif kind == SURFACE_ROMAN:
        position = surface_roman(u, v)
    elif kind == SURFACE_SUPER_ELLIPSOID:
        position = surface_super_ellipsoid(u, v, n1, n2)
    elif kind == SURFACE_SUPER_TOROID:
        position = surface_super_toroid(u, v, n1, n2)
    return position


@wp.kernel
def random_hills_vertices(
    amplitude: wp.float32,
    x_variance: wp.float32,
    y_variance: wp.float32,
    hill_centers: wp.array[wp.vec2],
    sample_u: wp.array[wp.float32],
    sample_v: wp.array[wp.float32],
    out_vertices: wp.array[wp.vec3],
) -> None:
    t = wp.int32(wp.tid())
    x = sample_u[t]
    y = sample_v[t]
    height = wp.float32(0.0)
    for h in range(hill_centers.shape[0]):
        offset = wp.vec2(x, y) - hill_centers[h]
        height += wp.exp(
            -0.5 * (offset[0] * offset[0] / x_variance + offset[1] * offset[1] / y_variance)
        )
    out_vertices[t] = wp.vec3(x, y, amplitude * height)
