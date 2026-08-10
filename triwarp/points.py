"""
Unstructured point-cloud geometry: fitting, second moments and normal estimation.

No connectivity here -- everything takes a bare ``(n,)`` array of positions. Two groups:

- **Whole-cloud fits.** [`centroid`][triwarp.points.centroid],
  [`covariance`][triwarp.points.covariance] and
  [`centered_covariance`][triwarp.points.centered_covariance] give the second moments;
  [`fit_line`][triwarp.points.fit_line] and [`fit_plane`][triwarp.points.fit_plane] read the
  dominant and weakest eigenvector off them. [`gram_matrix`][triwarp.points.gram_matrix] is the
  uncentered form, for callers that want to center differently.
- **Per-point.** [`estimate_normals`][triwarp.points.estimate_normals] fits a plane to each point's
  k-nearest neighbourhood, which is how an unoriented cloud acquires normals before
  reconstruction. [`outlier_probability`][triwarp.points.outlier_probability] and
  [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask] score the same
  neighbourhood for isolation, which is how a scanned cloud loses its stragglers *before* the
  normals are fitted. [`plane_basis`][triwarp.points.plane_basis] and
  [`radial_sort`][triwarp.points.radial_sort] then let a caller work in the tangent plane it
  defines.

Normals from [`estimate_normals`][triwarp.points.estimate_normals] are *unoriented* -- a plane fit
cannot pick a side. See [`triwarp.repair`][triwarp.repair] for orientation propagation.
"""

import math

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TILE_1D, TOLERANCE_ZERO
from triwarp.kernels import array as kernel_array
from triwarp.kernels import points as kernel_points
from triwarp.kernels import reduce as kernel_reduce


def point_plane_distance(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 | None = None
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
    # The uncentred Gram matrix is the scatter matrix around a zero center.
    zero_center = wp.zeros(1, dtype=wp.vec3, device=points.device)
    return centered_covariance(points, center=zero_center)


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
    return out_axis.list()[0]


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
        center = centroid(points)
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
    return (out_centroid.list()[0], out_normal.list()[0])


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


def outlier_probability(
    neighbor_idx: twt.Array2dInt32, neighbor_distance: twt.Array2dFloat32, *, scale: float = 3.0
) -> wp.array[wp.float32]:
    """
    Local Outlier Probability (LoOP) of each point, in ``[0, 1]``.

    A point is an outlier to the extent that its own neighbourhood is *stretched* relative to the
    neighbourhoods of the points in it. Following Kriegel et al., each point gets a "standard
    distance" ``sigma`` (the RMS distance to its neighbours), a local factor
    ``plof = sigma / mean(sigma over the neighbourhood) - 1``, and a probability
    ``max(0, erf(plof / (nplof * sqrt(2))))`` where ``nplof = scale * sqrt(mean(plof^2))`` over the
    whole cloud. Because the normalization is cloud-wide, the score is comparable across points but
    **not** across clouds — it is a rank, calibrated so that a threshold near ``0.8`` selects the
    tail.

    This is the measure behind MeshLab's ``compute_selection_point_cloud_outliers``, whose own
    ``propthreshold`` default is ``0.8``; threshold the result to reproduce a selection:

    ```python
    from triwarp.kernels import array as kernel_array

    probability = tw.points.outlier_probability(neighbor_idx, neighbor_distance)
    outlier_mask = wp.empty(probability.shape, dtype=wp.bool, device=probability.device)
    wp.map(kernel_array.greater, probability, wp.float32(0.8), out=outlier_mask)
    outliers = tw.array.flatnonzero(outlier_mask)
    ```

    Parameters
    ----------
    neighbor_idx
        ``(n, k)`` int32 neighbour table, as returned by
        [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] (unused slots marked ``-1``).
        A self-query table includes each point itself, which is counted normally — the same
        convention MeshLab's k-d tree query uses.
    neighbor_distance
        ``(n, k)`` float32 distances aligned with ``neighbor_idx``; unused slots are ``inf``.
    scale
        The LoOP normalization factor ``lambda``. Larger values make the score more conservative
        (fewer points near ``1``). MeshLab fixes it at ``3``, which is the default here.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n`` probabilities on ``neighbor_idx.device``. A point whose neighbour row is empty
        scores ``0``. All-zeros when the cloud has no spread at all.

    Raises
    ------
    ValueError
        If ``scale <= 0``, or the two tables disagree in shape.

    See Also
    --------
    [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask]
    [`estimate_normals`][triwarp.points.estimate_normals]
    [`triwarp.neighbors.query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
    """
    twt.ensure_ndim(neighbor_idx, 2, dtype=wp.int32)
    twt.ensure_ndim(neighbor_distance, 2, dtype=wp.float32)
    if neighbor_idx.shape != neighbor_distance.shape:
        raise ValueError(
            "neighbor_idx and neighbor_distance must have the same shape, got "
            f"{neighbor_idx.shape} and {neighbor_distance.shape}"
        )
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    device = neighbor_idx.device
    n = int(neighbor_idx.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    _mean, standard_distance, _count = _neighbor_distance_moments(neighbor_distance)
    plof = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_points.local_outlier_factor,
        dim=n,
        inputs=[standard_distance, neighbor_idx, plof],
        device=device,
    )

    # nplof = scale * sqrt(E[plof^2]) over the whole cloud: one fused sum-of-squares reduction and
    # one readback, because the normalizer is a scalar every point divides by and Warp cannot pass
    # a device scalar as a uniform argument.
    normalizer = scale * math.sqrt(wp.utils.array_inner(plof, plof) / n)
    out_probability = wp.zeros(n, dtype=wp.float32, device=device)
    if normalizer <= 0.0:
        return out_probability  # a cloud with no spread: every plof is zero
    wp.map(
        kernel_points.outlier_probability,
        plof,
        wp.float32(1.0 / (normalizer * math.sqrt(2.0))),
        out=out_probability,
    )
    return out_probability


def statistical_outlier_mask(
    neighbor_distance: twt.Array2dFloat32, *, std_ratio: float = 2.0
) -> wp.array[wp.bool]:
    """
    Flag points whose mean neighbour distance exceeds ``mean + std_ratio * std`` over the cloud.

    Open3D's ``remove_statistical_outlier`` criterion, and the cheaper cousin of
    [`outlier_probability`][triwarp.points.outlier_probability]: one global threshold on a single
    per-point statistic rather than a neighbourhood-relative one. It is the right choice for a
    cloud of roughly uniform density and the wrong one for a cloud that is dense in places and
    sparse in others, where the sparse region is entirely above a global threshold.

    Parameters
    ----------
    neighbor_distance
        ``(n, k)`` float32 distances from
        [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]; unused slots are ``inf`` and
        are excluded from the per-point mean. A self-query table contributes one zero distance per
        row, exactly as Open3D's ``SearchKNN`` does, so pass the same ``k`` Open3D gets as
        ``nb_neighbors``.
    std_ratio
        Multiplier on the cloud-wide standard deviation of the per-point means. Lower is stricter.
        Open3D's own examples use ``2.0``, the default here.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``neighbor_distance.device``; ``True`` marks an **outlier** (the
        complement of Open3D's *keep* mask). A point with an empty or fully coincident
        neighbourhood is marked as an outlier, matching Open3D's ``avg > 0`` guard.

    Raises
    ------
    ValueError
        If ``neighbor_distance`` is not a rank-2 float32 array.

    See Also
    --------
    [`outlier_probability`][triwarp.points.outlier_probability]
    [`triwarp.neighbors.query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
    """
    twt.ensure_ndim(neighbor_distance, 2, dtype=wp.float32)

    device = neighbor_distance.device
    n = int(neighbor_distance.shape[0])
    out_mask = wp.zeros(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    mean_distance, _rms, count = _neighbor_distance_moments(neighbor_distance)
    # Cloud mean and (ddof=1) deviation over the *counted* rows only, exactly as Open3D divides by
    # its ``valid_distances``. Empty rows contribute zero to both sums, so a plain reduction works.
    counted_mask = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(kernel_array.greater, count, wp.int32(0), out=counted_mask)
    counted = int(tw.reduce.sum(counted_mask))
    if counted < 2:
        return out_mask  # no deviation to threshold against
    cloud_mean = float(tw.reduce.sum(mean_distance)) / float(counted)
    deviations = wp.empty(n, dtype=wp.float32, device=device)
    wp.map(
        kernel_points.centered_square_if_counted,
        mean_distance,
        count,
        wp.float32(cloud_mean),
        out=deviations,
    )
    cloud_std = math.sqrt(float(tw.reduce.sum(deviations)) / float(counted - 1))

    wp.map(
        kernel_points.is_statistical_outlier,
        mean_distance,
        count,
        wp.float32(cloud_mean + std_ratio * cloud_std),
        out=out_mask,
    )
    return out_mask


def _neighbor_distance_moments(
    neighbor_distance: twt.Array2dFloat32,
) -> tuple[wp.array[wp.float32], wp.array[wp.float32], wp.array[wp.int32]]:
    """Per-row ``(mean, rms, count)`` of a neighbour-distance table, ignoring ``inf`` slots."""
    device = neighbor_distance.device
    n = int(neighbor_distance.shape[0])
    out_mean = wp.empty(n, dtype=wp.float32, device=device)
    out_rms = wp.empty(n, dtype=wp.float32, device=device)
    out_count = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_points.neighbor_distance_moments,
        dim=n,
        inputs=[neighbor_distance, out_mean, out_rms, out_count],
        device=device,
    )
    return out_mean, out_rms, out_count


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
        If ``a`` and ``b`` have different lengths.

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
    _sorted_keys, order = tw.array.sort_and_argsort(out_keys, fill_value=n)
    return tw.array.gather(points, order)
