import warp as wp

import triwarp as tw
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import points as kernel_points
from triwarp.kernels import reduce as kernel_reduce


def point_plane_distance(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 = None
) -> wp.array[wp.float32]:
    """
    Minimum perpendicular distance of each point to a plane.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space as ``wp.vec3``.
    plane_normal
        Plane normal vector as ``wp.vec3``; need not be unit length, as the
        distance is normalized by its magnitude.
    plane_origin
        Point on the plane as ``wp.vec3``. When ``None``, the origin
        ``(0, 0, 0)`` is used.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` signed distances from each point to the plane on
        ``points.device``.
    """
    if plane_origin is None:
        plane_origin = wp.vec3(0.0, 0.0, 0.0)
    n = int(points.shape[0])
    out_distances = wp.empty(n, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_points.point_plane_distance,
        dim=n,
        inputs=[points, plane_normal, plane_origin, out_distances],
        device=points.device,
    )
    return out_distances


def centroid(points: wp.array[wp.vec3]) -> wp.array[wp.vec3]:
    """
    Mean position ``sum(points) / n`` as a ``(1,)`` device ``wp.vec3`` array.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.vec3]
        Shape ``(1,)`` device array holding the centroid on ``points.device``.
        All-zeros when ``points`` is empty.
    """
    device = points.device
    n = int(points.shape[0])
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    if n == 0:
        return out
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_reduce.sum_vec3_1d_tiled,
        dim=[n_tiles],
        inputs=[points, out],
        block_dim=TILE_1D,
        device=device,
    )
    wp.launch(kernel_array.divide, dim=1, inputs=[out, wp.float32(n)], device=device)
    return out


def fit_line(points: wp.array[wp.vec3]) -> wp.vec3:
    """
    Approximate major axis of a point set via SVD.

    The major axis is the dominant direction of the point distribution,
    recovered from the singular value decomposition of the (uncentered)
    point matrix.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    wp.vec3
        Unit vector along the approximate major axis. The result is
        direction-only: its sign is governed by the SVD convention and is
        not meaningful.
    """
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return wp.vec3(0.0, 0.0, 0.0)

    # Pass 1: accumulate the (uncentred) Gram matrix G = sum_j outer(x_j, x_j).
    gram = tw.array.gram_matrix(points)

    # Pass 2: SVD of the 3x3 matrix and axis extraction (single thread).
    out_axis = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_points.finalize_fit_line,
        dim=1,
        inputs=[gram, out_axis],
        device=device,
    )
    return wp.vec3(*out_axis.numpy()[0].tolist())


def fit_plane(points: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """
    Fit a plane to a point set using SVD.

    The plane origin is the centroid of the points and the normal is the
    singular vector with the smallest singular value of the centered
    covariance matrix.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(centroid, normal)`` where ``centroid`` is a point on the plane
        and ``normal`` is its unit normal. The normal is sign-ambiguous: its
        orientation is governed by the SVD convention.
    """
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)

    # Pass 1: centroid (plane origin) on-device.
    center = centroid(points)

    # Pass 2: covariance matrix of the centred points. Centring before the
    # outer products (rather than via the sum(x x^T) - n c c^T identity) avoids
    # float32 catastrophic cancellation.
    cov = tw.array.centered_covariance(points, center=center)

    # Pass 3: SVD of the 3x3 covariance and centroid/normal extraction (single thread).
    out_centroid = wp.empty(1, dtype=wp.vec3, device=device)
    out_normal = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_points.finalize_fit_plane,
        dim=1,
        inputs=[center, cov, out_centroid, out_normal],
        device=device,
    )
    return (
        wp.vec3(*out_centroid.numpy()[0].tolist()),
        wp.vec3(*out_normal.numpy()[0].tolist()),
    )
