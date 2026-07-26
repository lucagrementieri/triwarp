import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TILE_1D, TOLERANCE_ZERO
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
    wp.map(
        kernel_points.point_plane_distance, points, plane_normal, plane_origin, out=out_distances
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
    wp.map(wp.div, out, wp.float32(n), out=out)
    return out


def gram_matrix(points: wp.array[wp.vec3]) -> wp.array[wp.mat33]:
    """
    Uncentered Gram (scatter) matrix ``G = sum_k outer(x_k, x_k)``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the ``3x3`` Gram matrix on
        ``points.device``. All-zeros when ``points`` is empty.
    """
    device = points.device
    n = int(points.shape[0])
    out = wp.zeros(1, dtype=wp.mat33, device=device)
    if n == 0:
        return out
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    # The uncentred Gram matrix is the scatter matrix around a zero center.
    zero_center = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch_tiled(
        kernel_points.centered_covariance,
        dim=[n_tiles],
        inputs=[points, zero_center, out],
        block_dim=TILE_1D,
        device=device,
    )
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
    gram = gram_matrix(points)

    # Pass 2: SVD of the 3x3 matrix and axis extraction (single thread).
    out_axis = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(kernel_points.finalize_fit_line, dim=1, inputs=[gram, out_axis], device=device)
    return wp.vec3(*out_axis.numpy()[0].tolist())


def centered_covariance(
    points: wp.array[wp.vec3], center: wp.array[wp.vec3] | None = None
) -> wp.array[wp.mat33]:
    """
    Centered scatter matrix ``C = sum_k outer(x_k - mu, x_k - mu)`` (no ``1/n``).

    Centering happens inside the outer-product loop (rather than via the
    ``sum(x x^T) - n mu mu^T`` identity) to avoid float32 catastrophic
    cancellation.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    center
        Optional precomputed centroid as a ``(1,)`` ``wp.vec3`` device array.
        When ``None`` it is computed on-device as ``sum(points) / n``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the centered ``3x3`` scatter matrix
        on ``points.device``. All-zeros when ``points`` is empty.
    """
    device = points.device
    n = int(points.shape[0])
    out = wp.zeros(1, dtype=wp.mat33, device=device)
    if n == 0:
        return out
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    if center is None:
        center = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch_tiled(
            kernel_reduce.sum_vec3_1d_tiled,
            dim=[n_tiles],
            inputs=[points, center],
            block_dim=TILE_1D,
            device=device,
        )
        wp.map(wp.div, center, wp.float32(n), out=center)
    wp.launch_tiled(
        kernel_points.centered_covariance,
        dim=[n_tiles],
        inputs=[points, center, out],
        block_dim=TILE_1D,
        device=device,
    )
    return out


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
    cov = centered_covariance(points, center=center)

    # Pass 3: SVD of the 3x3 covariance and centroid/normal extraction (single thread).
    out_centroid = wp.empty(1, dtype=wp.vec3, device=device)
    out_normal = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_points.finalize_fit_plane,
        dim=1,
        inputs=[center, cov, out_centroid, out_normal],
        device=device,
    )
    return (wp.vec3(*out_centroid.numpy()[0].tolist()), wp.vec3(*out_normal.numpy()[0].tolist()))


def plane_basis(normal: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
    """
    Right-handed orthonormal basis ``(u, v)`` spanning the plane with the given ``normal``.

    Parameters
    ----------
    normal
        Plane normal as ``wp.vec3``; need not be unit length.

    Returns
    -------
    tuple[wp.vec3, wp.vec3]
        ``(u, v)`` unit vectors perpendicular to each other and to ``normal``, such that
        ``(u, v, normalize(normal))`` is right-handed.
    """
    unit_normal = wp.normalize(normal)
    axis = wp.vec3(1.0, 0.0, 0.0)
    if abs(unit_normal[0]) > 0.9:
        axis = wp.vec3(0.0, 1.0, 0.0)
    u = wp.normalize(wp.cross(axis, unit_normal))
    v = wp.cross(unit_normal, u)
    return u, v


def covariance(points: wp.array[wp.vec3], ddof: int = 1) -> wp.array[wp.mat33]:
    """
    Sample covariance matrix ``(1 / (n - ddof)) sum_k outer(x_k - mu, x_k - mu)``.

    Matches ``numpy.cov(points.T, ddof=ddof)`` for the default ``ddof=1``.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    ddof
        Delta degrees of freedom; the divisor is ``n - ddof``. Defaults to ``1``.

    Returns
    -------
    wp.array[wp.mat33]
        Shape ``(1,)`` device array holding the ``3x3`` covariance matrix on
        ``points.device``.

    Raises
    ------
    ValueError
        If ``n - ddof <= 0``.
    """
    n = int(points.shape[0])
    if n - ddof <= 0:
        raise ValueError(f"covariance requires n > ddof, got n={n}, ddof={ddof}")
    out = centered_covariance(points)
    wp.map(wp.div, out, wp.float32(n - ddof), out=out)
    return out


def estimate_normals(
    points: wp.array[wp.vec3],
    neighbor_idx: twt.Array2dInt32,
    *,
    orient_reference: wp.vec3 | None = None,
    camera_location: wp.vec3 | None = None,
) -> wp.array[wp.vec3]:
    """
    Estimate per-point normals by PCA over each point's neighbourhood.

    Each normal is the eigenvector of the smallest eigenvalue of the local
    covariance matrix accumulated over the point's neighbours — the same choice
    made by MeshLib (``PointAccumulator``) and Open3D (``FastEigen3x3``), so the
    result matches both references up to sign. The neighbourhood is supplied by
    the caller as ``neighbor_idx``: build it with
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] using a plain
    ``k`` for a k-nearest (KNN) neighbourhood, or with ``max_radius`` set for a
    radius-bounded (hybrid) neighbourhood — mirroring the two neighbour modes of
    Open3D's ``estimate_normals(max_nn, radius)``.

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device.
    neighbor_idx
        ``(n, k)`` int32 table of neighbour indices per point, as returned by
        [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] (unused slots
        marked ``-1``). A self-query table includes each point itself once, which
        is counted normally.
    orient_reference
        When given, normals are flipped to align with this fixed direction
        (``dot(normal, orient_reference) >= 0``), matching Open3D's
        ``orient_normals_to_align_with_direction``. Mutually exclusive with
        ``camera_location``.
    camera_location
        When given, normals are flipped to point toward this location
        (``dot(normal, camera_location - point) >= 0``), matching Open3D's
        ``orient_normals_towards_camera_location``. Mutually exclusive with
        ``orient_reference``.

    Returns
    -------
    wp.array[wp.vec3]
        Length ``n`` unit normals on ``points.device``. When neither orientation
        argument is given, normals are oriented outward from the whole-cloud
        centroid — a best-effort global orientation valid for star-shaped clouds
        (Open3D leaves the sign arbitrary instead). Points with fewer than two
        valid neighbours (or a degenerate neighbourhood) receive a fallback
        ``(0, 0, 1)`` normal.

    Raises
    ------
    ValueError
        If both ``orient_reference`` and ``camera_location`` are given.

    See Also
    --------
    [`triwarp.points.fit_plane`][triwarp.points.fit_plane]
    [`triwarp.reconstruction.triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]
    """
    if orient_reference is not None and camera_location is not None:
        raise ValueError("pass at most one of orient_reference and camera_location")

    twt.ensure_ndim(neighbor_idx, 2, dtype=wp.int32)

    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.vec3, device=device)

    if camera_location is not None:
        orient_mode = kernel_points.ORIENT_CAMERA
        reference = camera_location
    elif orient_reference is not None:
        orient_mode = kernel_points.ORIENT_DIRECTION
        reference = orient_reference
    else:
        orient_mode = kernel_points.ORIENT_CENTROID
        reference = wp.vec3(0.0, 0.0, 0.0)

    center = centroid(points)
    out_normals = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_points.estimate_point_normals,
        dim=n,
        inputs=[points, neighbor_idx, center, orient_mode, reference, out_normals],
        device=device,
    )
    return out_normals


def vector_angle(a: wp.array[wp.vec3], b: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between pairs of unit vectors.

    For each index ``i``, computes ``abs(arccos(clip(dot(a[i], b[i]), -1, 1)))``.
    Matches [`trimesh.geometry.vector_angle`][] on stacked pairs.

    Parameters
    ----------
    a
        Length-``n`` unit vectors on the target device.
    b
        Length-``n`` unit vectors on the same device as ``a``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` unsigned angles in radians on ``a.device``. Empty when ``n == 0``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` live on different devices or have different lengths.

    See Also
    --------
    [`trimesh.geometry.vector_angle`][]
    """
    device = a.device
    n = int(a.shape[0])
    if n != int(b.shape[0]):
        raise ValueError(f"a and b must have the same length, got {n} and {b.shape[0]}")

    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_angles = wp.empty(n, dtype=wp.float32, device=device)
    wp.map(kernel_array.vector_angle_vec, a, b, out=out_angles)
    return out_angles


def radial_sort(
    points: wp.array[wp.vec3], origin: wp.vec3, normal: wp.vec3, start: wp.vec3 | None = None
) -> wp.array[wp.vec3]:
    """
    Sort points radially (by angle) around an axis and return them reordered.

    Points are projected onto two axes perpendicular to ``normal`` and ordered by
    the angle ``atan2`` of the projection, in **descending** order (matching
    [`trimesh.points.radial_sort`][]).

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    origin
        Point to sort around as ``wp.vec3``.
    normal
        Axis to sort around as ``wp.vec3``.
    start
        Optional ``wp.vec3`` specifying the start position in counter-clockwise
        order when viewed along ``normal``. Must not be parallel with ``normal``.
        When ``None``, an arbitrary perpendicular axis is used.

    Returns
    -------
    wp.array[wp.vec3]
        Length ``n`` array of the input points reordered by descending angle, on
        ``points.device``. Empty when ``points`` is empty.

    Raises
    ------
    ValueError
        If ``start`` is provided and is (near-)parallel with ``normal``.
    """
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.vec3, device=device)

    # Build two axes perpendicular to each other and the normal, onto which the
    # points are projected to recover an angle. Done on the host since the axes
    # are a single O(1) setup shared by every point.
    if start is None:
        axis0 = wp.vec3(normal[0], normal[2], -normal[1])
        axis1 = wp.cross(normal, axis0)
    else:
        unit_normal = wp.normalize(normal)
        unit_start = wp.normalize(start)
        if abs(1.0 - abs(wp.dot(unit_normal, unit_start))) < TOLERANCE_ZERO:
            raise ValueError("start must not be parallel with normal")
        axis0 = wp.cross(unit_start, unit_normal)
        axis1 = wp.cross(axis0, unit_normal)

    out_keys = wp.empty(n, dtype=wp.float32, device=device)
    wp.map(kernel_points.radial_sort_key, points, origin, axis0, axis1, out=out_keys)

    # Ascending radix sort of the negated angles yields the descending-angle order.
    _sorted_keys, order = tw.array.sort_pairs(out_keys, fill_value=n)
    out = wp.empty(n, dtype=wp.vec3, device=device)
    wp.copy(out, points[order])
    return out
