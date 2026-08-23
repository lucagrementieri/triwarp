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
- **Cleanup and subsampling.** Four masks and one selector, all returning indices or
  ``wp.array[wp.bool]`` rather than a copied cloud, so a caller pays for the gather only if it
  wants one: [`point_finite_mask`][triwarp.points.point_finite_mask] and
  [`point_duplicate_mask`][triwarp.points.point_duplicate_mask] are the two exact predicates,
  [`radius_outlier_mask`][triwarp.points.radius_outlier_mask] and
  [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask] the two density ones (an
  absolute floor and a cloud-relative threshold), and
  [`farthest_point_sample`][triwarp.points.farthest_point_sample] picks an exact count spread over
  the cloud's support. Run the exact predicates first: they are cheap and they remove the inputs
  the others are ill-defined on.

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
from triwarp.kernels import predicates as kernel_predicates
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


def half_space_mask(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 | None = None
) -> wp.array[wp.bool]:
    """
    Flag the points strictly on the normal's side of a plane.

    The unbounded selection primitive: one plane cuts space in two and this is the half the normal
    points into. Composes into a convex region by intersecting several masks, which is what makes it
    worth having next to the box query — a slab, a wedge or a frustum is a handful of these, and
    none of them needs a spatial index because every point is tested independently.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.
    plane_normal
        Plane normal as ``wp.vec3``, pointing into the selected half. Need not be unit length: only
        the sign of the projection is read.
    plane_origin
        Point on the plane as ``wp.vec3``. When ``None``, the origin ``(0, 0, 0)`` is used.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**, the same sense
        as [`point_finite_mask`][triwarp.points.point_finite_mask].

    Examples
    --------
    ```python
    upper = tw.points.half_space_mask(v, wp.vec3(0.0, 0.0, 1.0))
    kept = tw.array.gather(v, tw.array.flatnonzero(upper))
    ```

    Notes
    -----
    The test is **strict**, so a point exactly on the plane is excluded and the two masks for
    opposite normals are disjoint rather than overlapping. That is the convention MeshLib's
    ``findHalfSpacePoints`` uses (measured: with the plane ``z = 1``, a point at ``z = 1`` is in
    neither half), and it makes the pair of masks a partition of the points off the plane.

    See Also
    --------
    [`point_plane_distance`][triwarp.points.point_plane_distance]
        The signed distance this thresholds, when the magnitude is wanted too.
    [`triwarp.neighbors.query_bvh_box`][triwarp.neighbors.query_bvh_box]
        The bounded counterpart: selection by a box, through a BVH.
    [`triwarp.array.flatnonzero`][triwarp.array.flatnonzero]
    """
    if plane_origin is None:
        plane_origin = wp.vec3(0.0, 0.0, 0.0)
    n = int(points.shape[0])
    out_mask = wp.empty(n, dtype=wp.bool, device=points.device)
    if n == 0:
        return out_mask

    wp.map(kernel_points.is_in_half_space, points, plane_normal, plane_origin, out=out_mask)
    return out_mask


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
    wp.launch_tiled(
        kernel_reduce.sum_vec3_1d_tiled,
        dim=[kernel_reduce.blocks_1d(n)],
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
    Major axis of a point set, weighted across all three singular directions.

    The result is ``normalize(S @ V)`` over the SVD of the **uncentered** point matrix -- a sum of
    all three right singular vectors weighted by their singular values. It is **not** the first
    principal axis, and the two
    part company as soon as the cloud is neither strongly elongated nor centred on the origin:
    measured against the leading eigenvector of the covariance, ``|dot|`` is 1.000 on a 1000:1
    needle but **0.939** on a 3:1:0.2 cloud and **0.812** once that cloud is offset from the origin.
    For the principal frame, use
    [`principal_axes`][triwarp.points.principal_axes].

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    wp.vec3
        Unit vector along the weighted major axis. The result is
        direction-only: its sign is governed by the SVD convention and is
        not meaningful.

    Notes
    -----
    This is the quantity [`trimesh.points.major_axis`][] computes. Because the weighted sum mixes
    all three singular vectors, its value depends on each one's sign, which no SVD fixes -- so the
    two implementations agree to ``1.000`` where one singular value dominates and to ``0.992`` on a
    moderate cloud.

    See Also
    --------
    [`principal_axes`][triwarp.points.principal_axes]
        The first principal axis, and the full frame, of the *centred* cloud.
    [`fit_plane`][triwarp.points.fit_plane]
        The complementary weakest direction.
    [`trimesh.points.major_axis`][]
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
        ``(normal, centroid)`` where ``normal`` is the plane's unit normal and ``centroid`` is a
        point on it. The normal is sign-ambiguous: its orientation is governed by the SVD
        convention. Orientation first, position last, matching
        [`principal_axes`][triwarp.points.principal_axes] and
        [`bounds.oriented_bounding_box`][triwarp.bounds.oriented_bounding_box] -- and so the pair
        splats directly into the ``(plane_normal, plane_origin)`` argument order every plane entry
        point takes. [`trimesh.points.plane_fit`][] returns the two the other way round.

    See Also
    --------
    [`principal_axes`][triwarp.points.principal_axes]
        All three axes at once; this normal is its third row.
    [`plane_basis`][triwarp.points.plane_basis]
        Two in-plane axes completing this normal into a frame.
    """
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        # Both zero, so the order is readability rather than behaviour: normal, then centroid.
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
    return (out_normal.list()[0], out_centroid.list()[0])


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


def principal_axes(points: wp.array[wp.vec3]) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
    """
    Principal frame of a point cloud: its three axes, spreads and centroid.

    The eigenvectors of the centred covariance, ordered widest first, as the **rows** of a proper
    rotation -- so ``rotation * (p - centroid)`` are the coordinates of ``p`` in the principal
    frame, and ``wp.transpose(rotation)`` maps back. This is the fit
    [`fit_line`][triwarp.points.fit_line] and [`fit_plane`][triwarp.points.fit_plane] each read one
    direction of: the first row is the true major axis and the third is the plane normal.

    Parameters
    ----------
    points
        ``(n,)`` positions in space as ``wp.vec3``.

    Returns
    -------
    rotation : wp.mat33
        Principal axes as rows, in descending order of spread. A proper rotation: orthonormal with
        determinant ``+1``, which fixes the third row's sign, so only the first two are ambiguous.
    eigenvalues : wp.vec3
        The three spreads, descending. These are the eigenvalues of the *scatter* matrix
        [`centered_covariance`][triwarp.points.centered_covariance] returns, so they carry no
        ``1 / n``; divide by ``n - 1`` for the sample variances along each axis.
    centroid : wp.vec3
        Mean position, the origin of the frame.

    Notes
    -----
    Matches ``pyvista.principal_axes`` up to the sign of the first two axes, which no
    eigendecomposition fixes.

    Examples
    --------
    ```python
    rotation, eigenvalues, centroid = tw.points.principal_axes(v)
    ```

    See Also
    --------
    [`fit_line`][triwarp.points.fit_line]
        A different estimator, and the reason this exists; see its docstring.
    [`fit_plane`][triwarp.points.fit_plane]
    [`covariance`][triwarp.points.covariance]
    [`triwarp.bounds.oriented_bounding_box`][triwarp.bounds.oriented_bounding_box]
        A searched box rather than a covariance fit, and tighter for it.
    """
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), wp.vec3(), wp.vec3()

    center = centroid(points)
    scatter = centered_covariance(points, center=center)
    out_rotation = wp.empty(1, dtype=wp.mat33, device=device)
    out_eigenvalues = wp.empty(1, dtype=wp.vec3, device=device)
    out_centroid = wp.empty(1, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_points.finalize_principal_axes,
        dim=1,
        inputs=[center, scatter, out_rotation, out_eigenvalues, out_centroid],
        device=device,
    )
    return (out_rotation.list()[0], out_eigenvalues.list()[0], out_centroid.list()[0])


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
    covariance matrix accumulated over the point's neighbours — the standard
    choice, which Open3D's ``FastEigen3x3`` also makes, so the result matches the
    references up to sign. The neighbourhood is supplied by
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


def radius_outlier_mask(
    points: wp.array[wp.vec3], radius: float, min_neighbors: int, *, grid: wp.HashGrid | None = None
) -> wp.array[wp.bool]:
    """
    Flag points with at most ``min_neighbors`` other points within ``radius``.

    Open3D's ``remove_radius_outlier`` criterion, and the local counterpart of
    [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask]: an *absolute* density
    floor rather than a cloud-wide threshold on a per-point statistic. That makes it the right
    choice where the density is known in advance (a scanner's nominal spacing) and the wrong one
    where it varies across the cloud, which is the trade the two run in opposite directions.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.
    radius
        Search radius, inclusive at exactly ``radius``. Must be ``> 0``.
    min_neighbors
        A point survives when it has **strictly more** than this many neighbours, counting
        *itself* — Open3D's ``nb_points`` rule exactly, so the same number gives the same answer.
        Must be ``>= 1``.
    grid
        Optional pre-built hash grid over ``points``, from
        [`hashgrid_from_points`][triwarp.neighbors.hashgrid_from_points]. Pass it to hoist the
        build out of a sweep over several radii.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``; ``True`` marks an **outlier** (the complement of
        Open3D's *keep* mask), matching
        [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask].

    Raises
    ------
    ValueError
        If ``radius <= 0`` or ``min_neighbors < 1``.

    Notes
    -----
    The counts are exact, and the rule is the reference's — but Open3D's own
    ``remove_radius_outlier`` shares one ``KDTreeFlann`` across an OpenMP loop and its radius search
    is not thread-safe under that sharing, so it returns a *different answer run to run*: measured
    three distinct keep sets (43 / 44 / 45 points) over eight repetitions of one 500-point cloud,
    differing by one or two points each time. Querying the same tree serially reproduces this
    function exactly on that cloud. Expect a comparison against the filter to disagree on a handful
    of borderline points, and do not read that as a difference in the criterion.

    Examples
    --------
    ```python
    mask = tw.points.radius_outlier_mask(v, 1.0, 3)
    outlier_indices = tw.array.flatnonzero(mask)
    ```

    See Also
    --------
    [`statistical_outlier_mask`][triwarp.points.statistical_outlier_mask]
    [`outlier_probability`][triwarp.points.outlier_probability]
    [`triwarp.neighbors.query_hashgrid_ball_count`][triwarp.neighbors.query_hashgrid_ball_count]
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")

    device = points.device
    n = int(points.shape[0])
    out_mask = wp.zeros(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    # A self-query counts the point itself once, at distance 0 -- which is what makes
    # ``min_neighbors`` comparable with Open3D's ``nb_points`` without an off-by-one correction.
    counts = tw.neighbors.query_hashgrid_ball_count(points, points, radius, grid=grid)
    wp.map(kernel_array.less_equal, counts, wp.int32(min_neighbors), out=out_mask)
    return out_mask


def point_finite_mask(points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
    """
    Flag points whose three coordinates are all finite.

    Open3D's ``remove_non_finite_points`` predicate. A single ``NaN`` or infinity poisons every
    reduction the point enters — a bounding box, a covariance, a BVH build — so this is the first
    pass over a cloud read off a scanner, before any of the fits in this module.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``. ``True`` marks a point to **keep**, which is the
        opposite sense from the outlier and duplicate masks in this module — the name is the tell:
        this one is named for what it selects.

    Examples
    --------
    ```python
    finite = tw.points.point_finite_mask(v)
    cleaned = tw.array.gather(v, tw.array.flatnonzero(finite))
    ```

    See Also
    --------
    [`point_duplicate_mask`][triwarp.points.point_duplicate_mask]
        The other exact-predicate cleanup pass; run this one first, since a ``NaN`` position is
        never equal to itself under Open3D's rule.
    [`triwarp.array.flatnonzero`][triwarp.array.flatnonzero]
    """
    device = points.device
    n = int(points.shape[0])
    out_mask = wp.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    wp.map(kernel_points.is_finite_point, points, out=out_mask)
    return out_mask


def point_duplicate_mask(points: wp.array[wp.vec3]) -> wp.array[wp.bool]:
    """
    Flag every point that repeats an **exactly** equal position seen earlier in the cloud.

    Open3D's ``remove_duplicated_points`` rule: bit-for-bit coordinate equality, keeping the
    *first* occurrence of each distinct position. This is deliberately not
    [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows], whose ``wp.vec3`` key buckets
    each coordinate to about ``2.4e-4`` relative and so merges positions that are merely close;
    reach for that one when a *tolerance* is what you want, and for this one when equality is.

    Two injective 64-bit key rounds do it: the ``x`` and ``y`` bit patterns pack side by side into
    one ``int64``, that key's equivalence class packs against the ``z`` bits, and the second class
    identifies the position exactly. Each half is exactly 32 bits, so neither round can collide.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask on ``points.device``; ``True`` marks a **duplicate**, i.e. every
        occurrence but the first of each distinct position. Gathering by the complement keeps one
        representative of each in first-occurrence order, as Open3D does.

    Notes
    -----
    ``-0.0`` and ``+0.0`` merge, because IEEE-754 equality holds between them and the reference
    compares with ``==``; the packing folds the two bit patterns together to reproduce that.
    ``NaN`` is the one case that does **not** match the reference: ``NaN != NaN`` makes every
    ``NaN`` row its own class for Open3D, where a bit pattern is a bit pattern here and identical
    ``NaN`` rows merge. Run [`point_finite_mask`][triwarp.points.point_finite_mask] first and the
    question does not arise. Positions are compared at ``float32``, so two points that differ only
    below ``float32`` resolution are one position here and two for a ``float64`` reference.

    Examples
    --------
    ```python
    duplicates = tw.points.point_duplicate_mask(v)
    n_distinct = int(v.shape[0]) - int(tw.reduce.sum(duplicates))
    ```

    See Also
    --------
    [`point_finite_mask`][triwarp.points.point_finite_mask]
    [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows]
        The tolerance-bucketed sibling, and the right choice when exact equality is too strict.
    [`triwarp.grouping.first_occurrence_indices`][triwarp.grouping.first_occurrence_indices]
    """
    device = points.device
    n = int(points.shape[0])
    out_mask = wp.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return out_mask

    key_xy = wp.empty(n, dtype=wp.int64, device=device)
    wp.map(kernel_points.pack_xy_bits, points, out=key_xy)
    _unique_xy, class_xy = tw.grouping.unique_1d(key_xy, return_inverse=True)

    key_xyz = wp.empty(n, dtype=wp.int64, device=device)
    wp.map(kernel_points.pack_class_z_bits, class_xy, points, out=key_xyz)
    unique_xyz, class_xyz = tw.grouping.unique_1d(key_xyz, return_inverse=True)

    # The class count is the length of the unique-key array in hand, so the representative pass
    # needs no reduction of its own.
    first = tw.grouping.first_occurrence_indices(class_xyz, int(unique_xyz.shape[0]))
    wp.map(kernel_array.mask_not, tw.array.indices_to_mask(first, n), out=out_mask)
    return out_mask


def farthest_point_sample(
    points: wp.array[wp.vec3], count: int, *, start: int = 0
) -> wp.array[wp.int32]:
    """
    Greedily pick ``count`` points, each as far as possible from those already picked.

    Open3D's ``farthest_point_down_sample``: the classic maximin subsample, which spreads its
    output over the cloud's *support* rather than its density — so unlike
    [`triwarp.voxels.voxel_down_sample`][triwarp.voxels.voxel_down_sample] it returns an exact
    count, and unlike a random subsample it cannot leave a hole. The greedy choice makes it
    inherently sequential in ``count``: each iteration is one pass over the cloud, fused so that
    the distance update and the global arg-max share a launch and nothing is read back to the host
    in between.

    Parameters
    ----------
    points
        ``(n,)`` point positions as ``wp.array[wp.vec3]``.
    count
        Number of points to select; must satisfy ``0 <= count <= n``. Zero returns an empty array,
        as Open3D's ``num_samples=0`` does.
    start
        Index of the first sample, Open3D's ``start_index``. Defaults to ``0``, which is its
        default too.

    Returns
    -------
    wp.array[wp.int32]
        Length-``count`` indices into ``points``, on ``points.device``, in **selection order**:
        entry 0 is ``start`` and each later entry is the point farthest from every earlier one.
        Open3D returns the selected *points* instead, and its ``SelectByIndex`` emits them in
        ascending index order, so the sequence here is strictly more information than the reference
        gives back — the sets are the same.

    Raises
    ------
    ValueError
        If ``count`` is not in ``[0, n]``, or ``start`` is not a valid index into ``points``.

    Notes
    -----
    Ties are broken towards the lower index, matching the reference's strict ``>`` arg-max, and that
    is what makes the answer reproducible: the per-iteration arg-max is a single ``wp.atomic_max``
    over a key that packs the squared distance against the complemented index, so the winner does
    not depend on the order the atomics land in. Distances are compared in ``float32`` where Open3D
    uses ``float64``, so a cloud with two points at nearly equal distance from the chosen set can
    diverge from that reference at the point where the tie is resolved — and then, being greedy, for
    the rest of the sequence.

    Examples
    --------
    ```python
    indices = tw.points.farthest_point_sample(v, 4)
    spread = tw.array.gather(v, indices)
    ```

    See Also
    --------
    [`triwarp.sample.sample_surface_blue_noise`][triwarp.sample.sample_surface_blue_noise]
        The same "well-spread points" goal from a *mesh*, by dart throwing rather than greedily.
    [`triwarp.voxels.voxel_down_sample`][triwarp.voxels.voxel_down_sample]
    [`triwarp.neighbors.nearest_neighbor_distance`][triwarp.neighbors.nearest_neighbor_distance]
    """
    device = points.device
    n = int(points.shape[0])
    if not 0 <= count <= n:
        raise ValueError(f"count must be in [0, {n}], got {count}")

    out_selected = wp.empty(count, dtype=wp.int32, device=device)
    if count == 0:
        return out_selected
    if not 0 <= start < n:
        raise ValueError(f"start must be in [0, {n}), got {start}")

    cursor = wp.empty(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_points.seed_farthest_point,
        dim=1,
        inputs=[wp.int32(start), out_selected, cursor],
        device=device,
    )
    if count == 1:
        return out_selected

    min_distance_sq = wp.full(n, wp.float32(math.inf), dtype=wp.float32, device=device)
    # One int64 accumulator, re-armed by ``commit_farthest_point`` so the loop is two launches per
    # iteration rather than three -- and so the selected index never crosses to the host, which is
    # what keeps a ``count``-long loop off the ~0.1 ms-per-readback budget.
    best = wp.array([wp.int64(-1)], dtype=wp.int64, device=device)

    def iteration() -> None:
        wp.launch(
            kernel_points.advance_farthest_point,
            dim=n,
            inputs=[points, out_selected, cursor, min_distance_sq, best],
            device=device,
        )
        wp.launch(
            kernel_points.commit_farthest_point,
            dim=1,
            inputs=[best, out_selected, cursor],
            device=device,
        )

    # The greedy sweep is inherently sequential -- ``count - 1`` rounds of two dependent launches --
    # so its host cost is the whole story on a small cloud: measured 2 049 launches and **79 %
    # host** for ``count=1024`` on 2 562 points, against 34 % on 40 962 points where the distance
    # update is real work. With the step counter on the device every round issues the identical
    # pair, so one round is captured and every round is a replay: ~1.17 us against ~13.4 us issued.
    # Capturing costs about what issuing costs, so capturing the *whole* loop would buy nothing --
    # the win is that one captured round is replayed ``count - 1`` times. Measured at
    # ``count=1024``: 28.75 -> 4.34 ms (6.6x) on 2 562 points, and 28.85 -> 21.12 ms (1.37x) on
    # 40 962, where the per-round device work is real and the host was never the limit.
    if not wp.get_device(device).is_cuda:
        for _ in range(count - 1):
            iteration()
        return out_selected
    with wp.ScopedCapture(device) as capture:
        iteration()
    # Capture *records* the round without executing it, so all ``count - 1`` rounds are replays.
    for _ in range(count - 1):
        wp.capture_launch(capture.graph)
    return out_selected


def vector_angle(a: wp.array[wp.vec3], b: wp.array[wp.vec3]) -> wp.array[wp.float32]:
    """
    Unsigned angle in radians between pairs of vectors.

    For each index ``i``, computes ``atan2(norm(cross(a[i], b[i])), dot(a[i], b[i]))``, which lies
    in ``[0, pi]``. Matches [`trimesh.geometry.vector_angle`][] on stacked pairs, and agrees with
    its ``arccos(dot)`` to round-off on unit input while staying accurate for nearly parallel or
    nearly antiparallel pairs, where ``arccos`` loses most of its digits. Being scale-free, it also
    accepts unnormalized vectors, and pairs where either vector is zero read ``0.0``.

    Parameters
    ----------
    a
        Length-``n`` vectors on the target device.
    b
        Length-``n`` vectors on the same device as ``a``.

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
    wp.map(kernel_predicates.vector_angle, a, b, out=out_angles)
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
