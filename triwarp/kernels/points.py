import warp as wp


@wp.kernel
def point_plane_distance(
    points: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    out_distances: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    w = points[tid] - plane_origin
    out_distances[tid] = wp.dot(plane_normal, w) / wp.length(plane_normal)


@wp.kernel
def radial_sort_key(
    points: wp.array[wp.vec3],
    origin: wp.vec3,
    axis0: wp.vec3,
    axis1: wp.vec3,
    out_keys: wp.array[wp.float32],
) -> None:
    tid = wp.tid()
    v = points[tid] - origin
    # Negated angle: an ascending radix sort of these keys reproduces trimesh's
    # descending-angle order (`angles.argsort()[::-1]`).
    out_keys[tid] = -wp.atan2(wp.dot(v, axis0), wp.dot(v, axis1))


@wp.kernel
def finalize_fit_line(
    m: wp.array[wp.mat33],
    out_axis: wp.array[wp.vec3],
) -> None:
    # gram matrix of the (uncentred) points: M = sum_j outer(x_j, x_j)
    # the singular values / right singular vectors of M match the squared
    # singular values / right singular vectors of the (n, 3) point matrix.
    u, sigma, _v = wp.svd3(m[0])
    # axis = sum_i S_i * rsv_i, where S_i = sqrt(sigma_i) are the point
    # singular values and the right singular vectors are columns of u.
    axis = wp.vec3(0.0, 0.0, 0.0)
    for i in range(3):
        s = wp.sqrt(sigma[i])
        axis += s * wp.vec3(u[0, i], u[1, i], u[2, i])
    out_axis[0] = wp.normalize(axis)


@wp.kernel
def finalize_fit_plane(
    center: wp.array[wp.vec3],
    m: wp.array[wp.mat33],
    out_centroid: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
) -> None:
    # plane origin is the centroid of the point set; the covariance matrix of
    # the centred points was accumulated into m.
    u, _sigma, _v = wp.svd3(m[0])
    # normal is the singular vector with the smallest singular value
    # (svd3 returns singular values in descending order: last column of u).
    out_centroid[0] = center[0]
    out_normal[0] = wp.normalize(wp.vec3(u[0, 2], u[1, 2], u[2, 2]))
