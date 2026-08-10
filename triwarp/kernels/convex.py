import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT


@wp.kernel
def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    out_projections: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    normal = face_normals[face_adjacency[tid, 0]]
    origin = vertices[face_adjacency_edges[tid, 0]]
    vid_other = face_adjacency_unshared[tid, 1]
    vector_other = vertices[vid_other] - origin
    out_projections[tid] = wp.dot(vector_other, normal)


@wp.kernel
def hull_support_extremes(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    n_slices: wp.int32,
    out_best_max: wp.array[wp.float32],
    out_best_min: wp.array[wp.float32],
) -> None:
    k, j = wp.tid()
    n_p = int(points.shape[0])
    direction = directions[int(k)]
    # Strided slice, NOT a contiguous chunk: consecutive threads read consecutive points, so the
    # loads coalesce, and each thread contributes one atomic instead of one per point.
    #
    # A block-wide `wp.tile_max(wp.tile(...))` reduction would be the natural fit here and was what
    # this kernel used, but `wp.launch_tiled` runs exactly ONE lane per block on the Warp CPU
    # backend through Warp 1.16 (`wp.tid()`'s lane index is always 0), so a tile of per-lane
    # values holds one element there and returns a wrong extreme. This form is lane-free.
    local_max = float(-FLOAT32_INF_CONSTANT)
    local_min = float(FLOAT32_INF_CONSTANT)
    for i in range(int(j), n_p, int(n_slices)):
        distance = wp.dot(direction, points[i])
        local_max = wp.max(local_max, distance)
        local_min = wp.min(local_min, distance)

    # A slice past the end of the cloud contributes nothing.
    if local_max > -FLOAT32_INF_CONSTANT:
        wp.atomic_max(out_best_max, int(k), local_max)
        wp.atomic_min(out_best_min, int(k), local_min)


@wp.kernel
def mark_hull_support(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    out_mask: wp.array[wp.bool],
) -> None:
    k, i = wp.tid()
    # A hemisphere direction n covers both +n (max, supports the vertex farthest
    # along n) and -n (min, supports the vertex farthest along -n).
    distance = wp.dot(directions[int(k)], points[int(i)])
    # Slack scales with the per-direction support extent so the test is
    # scale-invariant and stays above the float32 dot-product noise floor.
    slack = tolerance * (best_max[int(k)] - best_min[int(k)])
    if distance >= best_max[int(k)] - slack or distance <= best_min[int(k)] + slack:
        out_mask[int(i)] = True


@wp.kernel
def support_indices(
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    best_max: wp.array[wp.float32],
    best_min: wp.array[wp.float32],
    tolerance: wp.float32,
    out_support: wp.array[wp.int32],
) -> None:
    k, i = wp.tid()
    slack = tolerance * (best_max[int(k)] - best_min[int(k)])
    if wp.dot(directions[int(k)], points[int(i)]) >= best_max[int(k)] - slack:
        # Lowest attaining index wins, so the shell is identical across launches even when
        # several points tie for the support along a direction.
        wp.atomic_min(out_support, int(k), int(i))


@wp.kernel
def shell_bounds(
    shell_vertices: wp.array[wp.vec3],
    out_centroid: wp.array[wp.vec3],
    out_radius: wp.array[wp.float32],
) -> None:
    # One thread: the shell has a few hundred vertices at most, and reducing on device keeps the
    # support sweep and the tetrahedron build in one launch chain with no host readback between.
    n = int(shell_vertices.shape[0])
    total = wp.vec3(0.0, 0.0, 0.0)
    for i in range(n):
        total = total + shell_vertices[i]
    center = total / float(n)

    # The radius is the length scale the interior margin is measured against, so that the margin is
    # a fraction of the construction's own size rather than of a tetrahedron's aspect ratio.
    radius = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    for i in range(n):
        radius = wp.max(radius, wp.length(shell_vertices[i] - center))

    out_centroid[0] = center
    out_radius[0] = radius


@wp.func
def outward_plane(p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, interior: wp.vec3) -> wp.vec4:
    """Unit-normal plane through the triangle, oriented so ``interior`` has negative offset."""
    normal = wp.cross(p1 - p0, p2 - p0)
    length = wp.length(normal)
    if length <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    normal = normal / length
    offset = wp.dot(normal, p0)
    if wp.dot(normal, interior) > offset:
        return wp.vec4(-normal[0], -normal[1], -normal[2], -offset)
    return wp.vec4(normal[0], normal[1], normal[2], offset)


@wp.kernel
def tetrahedron_planes(
    shell_vertices: wp.array[wp.vec3],
    shell_faces: wp.array[wp.int32],
    centroid: wp.array[wp.vec3],
    flatness: wp.float32,
    out_planes: wp.array2d[wp.vec4],
    out_valid: wp.array[wp.bool],
) -> None:
    t = int(wp.tid())
    apex = centroid[0]
    a = shell_vertices[shell_faces[t * 3 + 0]]
    b = shell_vertices[shell_faces[t * 3 + 1]]
    c = shell_vertices[shell_faces[t * 3 + 2]]

    # Scale-free flatness test: the determinant of the three apex edges against their length
    # product. A sliver's face normals are ill-conditioned cross products of nearly parallel edges,
    # and that error is the only one that can cost the superset guarantee, so reject generously --
    # neighbouring well-shaped tetrahedra cover the same region. A wholly degenerate shell (a
    # coplanar or collinear cloud) rejects every tetrahedron, and the filter then keeps all points.
    ea = a - apex
    eb = b - apex
    ec = c - apex
    m = wp.matrix_from_cols(ea, eb, ec)
    if wp.abs(wp.determinant(m)) <= flatness * wp.length(ea) * wp.length(eb) * wp.length(ec):
        out_valid[t] = False
        return

    # Half-space form with *unit* normals, so the interior test below is a true distance and its
    # margin can be a length. Barycentric coordinates would be the cheaper test but their margin is
    # meaningless as a distance: these tetrahedra run from the centroid out to the shell, so a fixed
    # barycentric slack cuts a thick layer off the base and nothing off the sides.
    inner = (apex + a + b + c) / 4.0
    out_planes[t, 0] = outward_plane(a, b, c, inner)
    out_planes[t, 1] = outward_plane(apex, a, b, inner)
    out_planes[t, 2] = outward_plane(apex, b, c, inner)
    out_planes[t, 3] = outward_plane(apex, c, a, inner)
    out_valid[t] = True


@wp.kernel
def mark_hull_superset(
    points: wp.array[wp.vec3],
    planes: wp.array2d[wp.vec4],
    valid: wp.array[wp.bool],
    radius: wp.array[wp.float32],
    margin: wp.float32,
    out_mask: wp.array[wp.bool],
) -> None:
    i = int(wp.tid())
    point = points[i]
    # Strict interior only, by a real distance. A point on a tetrahedron's boundary can still be a
    # hull vertex, and requiring it to clear every face by `slack` means float32 error in the plane
    # evaluation can only keep a point that could have been dropped -- never drop a hull vertex.
    slack = margin * radius[0]
    n_tetra = int(valid.shape[0])
    keep = wp.int32(1)
    for t in range(n_tetra):
        if valid[t]:
            inside = wp.int32(1)
            for f in range(4):
                plane = planes[t, f]
                normal = wp.vec3(plane[0], plane[1], plane[2])
                if wp.dot(normal, point) - plane[3] >= -slack:
                    inside = wp.int32(0)
                    break
            if inside != 0:
                keep = wp.int32(0)
                break
    out_mask[i] = keep != 0
