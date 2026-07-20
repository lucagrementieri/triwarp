"""Point cloud registration (Procrustes analysis, ICP) on NVIDIA Warp."""

from __future__ import annotations

import math
from typing import Literal, cast, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import registration as kernel_registration


@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: Literal[False] = False,
) -> wp.array[wp.mat44]: ...
@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: Literal[True] = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]: ...
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: bool = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float] | wp.array[wp.mat44]:
    """
    Find the optimal transform mapping point cloud *a* onto *b*.

    Warp port of [`trimesh.registration.procrustes`][]. Accepts and returns
    GPU-resident arrays; no NumPy in the hot path.

    Parameters
    ----------
    a, b:
        Corresponding point clouds, shape ``(n,)``, dtype ``wp.vec3``.
    weights:
        Per-point weights, shape ``(n,)``, dtype ``wp.float32``.
        Uniform weights are assumed when ``None``.
    reflection:
        Allow reflections in the rotation component.
    translation:
        Allow translation.
    scale:
        Allow uniform scaling.
    return_cost:
        When ``True`` (default) also return the transformed points and cost.

    Returns
    -------
    matrix:
        ``wp.array[wp.mat44]`` of shape ``(1,)`` encoding the transform.
    transformed:
        ``wp.array[wp.vec3]`` of shape ``(n,)`` — image of *a* under the transform.
        Only returned when ``return_cost=True``.
    cost:
        Weighted sum of squared distances between *transformed* and *b*.
        Only returned when ``return_cost=True``.
    """
    n = a.shape[0]
    device = a.device

    if weights is None:
        weights = wp.full(n, wp.float32(1.0), dtype=wp.float32, device=device)

    # --- Phase 1a: weighted sums for centroids ---
    w_sum = wp.zeros(1, dtype=wp.float32, device=device)
    a_sum = wp.zeros(1, dtype=wp.vec3, device=device)
    b_sum = wp.zeros(1, dtype=wp.vec3, device=device)
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_registration.accumulate_weighted_sums,
        dim=[n_tiles],
        inputs=[a, b, weights, w_sum, a_sum, b_sum],
        block_dim=TILE_1D,
        device=device,
    )

    # --- Phase 1b: weighted scale^2 and cross-covariance ---
    a_scale_sq = wp.zeros(1, dtype=wp.float32, device=device)
    b_scale_sq = wp.zeros(1, dtype=wp.float32, device=device)
    cov = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch_tiled(
        kernel_registration.accumulate_scale_and_cov,
        dim=[n_tiles],
        inputs=[a, b, weights, w_sum, a_sum, b_sum, translation, a_scale_sq, b_scale_sq, cov],
        block_dim=TILE_1D,
        device=device,
    )

    # --- Phase 2: SVD and 4x4 matrix construction (single thread) ---
    out_matrix = wp.zeros(1, dtype=wp.mat44, device=device)
    wp.launch(
        kernel_registration.build_procrustes_matrix,
        dim=1,
        inputs=[
            w_sum,
            a_sum,
            b_sum,
            a_scale_sq,
            b_scale_sq,
            cov,
            reflection,
            translation,
            scale,
            out_matrix,
        ],
        device=device,
    )

    if not return_cost:
        return out_matrix

    # --- Phase 3a: apply transform to all points ---
    out_transformed = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_registration.apply_transform_mat44,
        dim=n,
        inputs=[a, out_matrix, out_transformed],
        device=device,
    )

    # --- Phase 3b: weighted cost reduction ---
    out_cost = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_registration.accumulate_cost,
        dim=n,
        inputs=[out_transformed, b, weights, w_sum, out_cost],
        device=device,
    )

    cost = float(out_cost.numpy()[0])
    return out_matrix, out_transformed, cost


_ROBUST_KINDS: dict[str, int] = {"none": 0, "huber": 1, "tukey": 2}


def _identity_mat44(device: wp.DeviceLike) -> wp.array[wp.mat44]:
    """Return a ``(1,)`` array holding the 4x4 identity transform."""
    return wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)


def _resolve_initial(
    initial: wp.array[wp.mat44] | wp.mat44 | None, device: wp.DeviceLike
) -> wp.array[wp.mat44]:
    """Normalize ``initial`` to a ``(1,)`` ``wp.mat44`` device array."""
    if initial is None:
        return _identity_mat44(device)
    if isinstance(initial, wp.array):
        return wp.clone(initial)
    return wp.array([initial], dtype=wp.mat44, device=device)


def _is_mesh_target(target_faces: wp.array[wp.int32] | None) -> bool:
    return target_faces is not None and int(target_faces.shape[0]) // 3 > 0


def icp(
    a: wp.array[wp.vec3],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32] | None = None,
    *,
    initial: wp.array[wp.mat44] | wp.mat44 | None = None,
    max_iterations: int = 20,
    threshold: float = 1e-5,
    max_distance: float | None = None,
    reflection: bool = False,
    translation: bool = True,
    scale: bool = False,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]:
    """
    Point-to-point iterative closest point registration.

    Aligns the source cloud *a* onto a target mesh or point cloud by repeatedly
    matching each source point to its nearest target counterpart and solving the
    resulting correspondence with [`procrustes`][triwarp.registration.procrustes].
    Following the ``pytorch3d`` formulation, every iteration re-aligns the
    *original* *a* to the current correspondences, so the returned transform is a
    single map from *a* to the target (no incremental-composition drift). Only a
    rough initial alignment is required for a good result.

    Warp port combining [`trimesh.registration.icp`][] and
    ``pytorch3d.ops.iterative_closest_point``. Correspondence search runs on the
    GPU via [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    (mesh target) or [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest]
    (point-cloud target); the alignment is the GPU tiled SVD of
    [`procrustes`][triwarp.registration.procrustes].

    Parameters
    ----------
    a
        Source point cloud, shape ``(n,)``, dtype ``wp.vec3``.
    target_vertices
        Target vertex positions, shape ``(m,)``, dtype ``wp.vec3``. Used as the
        target point cloud when ``target_faces`` is ``None``.
    target_faces
        Flat ``(f * 3,)`` triangle index buffer. When given (and non-empty),
        correspondences are the closest points on the triangle surface; otherwise
        the target is the point cloud ``target_vertices``.
    initial
        Initial transform seeding the first correspondence search, as a ``(1,)``
        ``wp.mat44`` array or a scalar ``wp.mat44``. Identity when ``None``.
    max_iterations
        Maximum number of ICP iterations.
    threshold
        Stop when the cost decreases by less than this between iterations.
    max_distance
        Reject correspondences farther than this (weight ``0``, excluded from the
        fit). No rejection when ``None``.
    reflection, translation, scale
        Forwarded to [`procrustes`][triwarp.registration.procrustes]. Defaults are
        rigid (`reflection=False, scale=False`); set ``scale=True`` for a
        similarity transform.

    Returns
    -------
    matrix
        ``(1,)`` ``wp.mat44`` transform mapping *a* onto the target.
    transformed
        ``(n,)`` ``wp.vec3`` image of *a* under *matrix*.
    cost
        Weighted mean squared correspondence distance at the final iteration.

    See Also
    --------
    [`procrustes`][triwarp.registration.procrustes]
    [`icp_point_to_plane`][triwarp.registration.icp_point_to_plane]
    [`trimesh.registration.icp`][]
    """
    device = a.device
    n = int(a.shape[0])
    is_mesh = _is_mesh_target(target_faces)

    if n == 0 or int(target_vertices.shape[0]) == 0:
        return _identity_mat44(device), wp.clone(a), math.inf

    initial_matrix = _resolve_initial(initial, device)
    current = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_registration.apply_transform_mat44,
        dim=n,
        inputs=[a, initial_matrix, current],
        device=device,
    )
    total = initial_matrix
    transformed = wp.clone(current)
    cost = math.inf

    mesh: wp.Mesh | None = None
    query_max = wp.float32(0.0)
    if is_mesh:
        assert target_faces is not None
        require_nonempty_mesh(target_faces, "icp")
        mesh = wp.Mesh(points=wp.clone(target_vertices), indices=wp.clone(target_faces))
        query_max = tw.proximity._default_mesh_query_max_dist(mesh.points, current)
        if max_distance is not None:
            query_max = max(query_max, max_distance)

    # Correspondence and weight buffers are allocated once and refilled every iteration.
    closest = wp.empty(n, dtype=wp.vec3, device=device)
    distance_mesh = wp.empty(n, dtype=wp.float32, device=device)
    triangle_id_mesh = wp.empty(n, dtype=wp.int32, device=device)
    weights: wp.array[wp.float32] | None = (
        wp.empty(n, dtype=wp.float32, device=device) if max_distance is not None else None
    )

    old_cost = math.inf
    for _ in range(max_iterations):
        if is_mesh:
            assert mesh is not None
            distance = distance_mesh
            triangle_id = triangle_id_mesh
            wp.launch(
                kernel_proximity.closest_point_on_mesh,
                dim=n,
                inputs=[mesh.id, current, wp.float32(query_max), closest, distance, triangle_id],
                device=device,
            )
        else:
            index, distance = tw.neighbors.query_bvh_nearest(target_vertices, current, 1)
            wp.copy(closest, target_vertices[index])
            triangle_id = index

        if max_distance is not None and weights is not None:
            wp.map(
                kernel_registration.distance_threshold_weight,
                distance,
                triangle_id,
                wp.float32(max_distance),
                out=weights,
            )
            if float(tw.reduce.sum(weights)) == 0.0:
                break

        total, transformed, cost = procrustes(
            a, closest, weights=weights, reflection=reflection, translation=translation, scale=scale
        )
        current = transformed
        if old_cost - cost < threshold:
            break
        old_cost = cost

    return total, transformed, cost


def _robust_scale_from_residuals(
    current: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    distance: wp.array[wp.float32],
    triangle_id: wp.array[wp.int32],
    max_distance: float,
    kind: int,
) -> float:
    """Robust scale (Huber/Tukey) from the MAD of the current point-to-plane residuals."""
    device = current.device
    n = int(current.shape[0])
    residual = wp.empty(n, dtype=wp.float32, device=device)
    wp.map(kernel_registration.point_to_plane_residual, current, closest, normals, out=residual)
    valid = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(
        kernel_registration.residual_valid,
        triangle_id,
        distance,
        wp.float32(max_distance),
        out=valid,
    )
    valid_indices = tw.array.flatnonzero(valid)
    k = int(valid_indices.shape[0])
    if k == 0:
        return 0.0
    kept = tw.array.gather(residual, valid_indices)
    # Median and MAD on device (sort-based); only the scalar results cross to the host.
    median = tw.reduce.median(cast(twt.Array1dFloat32, kept))
    deviation = wp.empty(k, dtype=wp.float32, device=device)
    wp.map(kernel_registration.abs_deviation, kept, wp.float32(median), out=deviation)
    mad = tw.reduce.median(cast(twt.Array1dFloat32, deviation))
    sigma = float(1.4826 * mad)
    if sigma <= 0.0:
        # Standard-deviation fallback: sqrt(mean((r - mean)^2)).
        mean = float(tw.reduce.mean(cast(twt.Array1dFloat32, kept)))
        wp.map(kernel_registration.abs_deviation, kept, wp.float32(mean), out=deviation)
        wp.map(kernel_array.square_scalar, deviation, out=deviation)
        sigma = float(tw.reduce.mean(cast(twt.Array1dFloat32, deviation))) ** 0.5
    if sigma <= 0.0:
        return 0.0
    # 95% asymptotic efficiency tuning constants.
    return 1.345 * sigma if kind == 1 else 4.685 * sigma


def icp_point_to_plane(
    a: wp.array[wp.vec3],
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32] | None = None,
    *,
    target_normals: wp.array[wp.vec3] | None = None,
    initial: wp.array[wp.mat44] | wp.mat44 | None = None,
    max_iterations: int = 20,
    threshold: float = 1e-6,
    max_distance: float | None = None,
    robust_kernel: Literal["none", "huber", "tukey"] = "none",
    robust_scale: float | None = None,
    damping: float = 1e-6,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]:
    """
    Point-to-plane iterative closest point registration with optional robust loss.

    Minimizes the squared distance from each transformed source point to the
    *tangent plane* of its target correspondence, which converges faster than the
    point-to-point metric on smooth surfaces. Each iteration linearizes the
    residual, assembles the ``6x6`` Gauss-Newton system on the GPU, solves it, and
    composes the incremental rigid step into the running transform.

    Warp port of the ``libigl`` ``iterative_closest_point`` (rigid, point-to-plane)
    and Open3D's robust ``TransformationEstimationPointToPlane``. An optional
    Huber or Tukey M-estimator down-weights outlier correspondences.

    Parameters
    ----------
    a
        Source point cloud, shape ``(n,)``, dtype ``wp.vec3``.
    target_vertices
        Target vertex positions, shape ``(m,)``, dtype ``wp.vec3``.
    target_faces
        Flat ``(f * 3,)`` triangle index buffer. When given, target normals are
        the closest triangles' face normals; otherwise the target is the point
        cloud ``target_vertices`` and ``target_normals`` is required.
    target_normals
        Per-vertex unit normals for a point-cloud target, shape ``(m,)``. Required
        (and only used) when ``target_faces`` is ``None``. Estimate them with
        [`estimate_normals`][triwarp.points.estimate_normals] if absent.
    initial
        Initial transform, as a ``(1,)`` ``wp.mat44`` array or scalar ``wp.mat44``.
        Identity when ``None``.
    max_iterations
        Maximum number of ICP iterations.
    threshold
        Stop when the cost decreases by less than this between iterations.
    max_distance
        Reject correspondences farther than this. No rejection when ``None``.
    robust_kernel
        M-estimator applied to residuals: ``"none"``, ``"huber"``, or ``"tukey"``.
    robust_scale
        Tuning constant of the robust kernel (Huber ``k`` / Tukey ``c``). When
        ``None`` and a robust kernel is active, derived from the median absolute
        deviation of the first iteration's residuals.
    damping
        Relative Levenberg damping added to the normal-equations diagonal (scaled
        by its mean magnitude) for conditioning on planar/degenerate targets.

    Returns
    -------
    matrix
        ``(1,)`` ``wp.mat44`` rigid transform mapping *a* onto the target.
    transformed
        ``(n,)`` ``wp.vec3`` image of *a* under *matrix*.
    cost
        Robust-weighted sum of squared point-to-plane residuals at the final
        iteration.

    Raises
    ------
    ValueError
        If the target is a point cloud (``target_faces is None``) and
        ``target_normals`` is not provided.

    See Also
    --------
    [`icp`][triwarp.registration.icp]
    [`estimate_normals`][triwarp.points.estimate_normals]
    [`normals_at_closest_faces`][triwarp.proximity.normals_at_closest_faces]
    """
    device = a.device
    n = int(a.shape[0])
    is_mesh = _is_mesh_target(target_faces)
    kind = _ROBUST_KINDS[robust_kernel]

    if not is_mesh and target_normals is None:
        raise ValueError(
            "icp_point_to_plane requires target_normals for a point-cloud target "
            "(target_faces is None)."
        )

    if n == 0 or int(target_vertices.shape[0]) == 0:
        return _identity_mat44(device), wp.clone(a), math.inf

    initial_matrix = _resolve_initial(initial, device)
    current = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_registration.apply_transform_mat44,
        dim=n,
        inputs=[a, initial_matrix, current],
        device=device,
    )
    total = initial_matrix
    transformed = wp.clone(current)
    cost = math.inf

    face_normals: wp.array[wp.vec3] | None = None
    mesh: wp.Mesh | None = None
    query_max = wp.float32(0.0)
    if is_mesh:
        assert target_faces is not None
        require_nonempty_mesh(target_faces, "icp_point_to_plane")
        mesh = wp.Mesh(points=wp.clone(target_vertices), indices=wp.clone(target_faces))
        face_normals, _ = tw.triangles.face_normals_and_areas(target_vertices, target_faces)
        query_max = tw.proximity._default_mesh_query_max_dist(mesh.points, current)
        if max_distance is not None:
            query_max = max(query_max, max_distance)

    max_d = max_distance if max_distance is not None else math.inf
    scale_value = robust_scale
    old_cost = math.inf
    n_tiles = (n + TILE_1D - 1) // TILE_1D

    # All per-iteration buffers are allocated once; the accumulators are zeroed in place and
    # ``current``/``updated`` ping-pong. ``total`` is cloned so composing in place never mutates
    # a caller-provided initial transform. The per-iteration cost read stays: it is the
    # stopping criterion (a 4-byte transfer).
    closest = wp.empty(n, dtype=wp.vec3, device=device)
    distance = wp.empty(n, dtype=wp.float32, device=device)
    triangle_id_mesh = wp.empty(n, dtype=wp.int32, device=device)
    normals = wp.empty(n, dtype=wp.vec3, device=device)
    jtj = wp.zeros(1, dtype=wp.spatial_matrix, device=device)
    jtr = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    cost_acc = wp.zeros(1, dtype=wp.float32, device=device)
    step = wp.empty(1, dtype=wp.mat44, device=device)
    updated = wp.empty(n, dtype=wp.vec3, device=device)
    total = wp.clone(initial_matrix)

    for iteration in range(max_iterations):
        # --- correspondence + target normals ---
        if is_mesh:
            assert mesh is not None
            assert face_normals is not None
            triangle_id = triangle_id_mesh
            wp.launch(
                kernel_proximity.closest_point_on_mesh,
                dim=n,
                inputs=[mesh.id, current, wp.float32(query_max), closest, distance, triangle_id],
                device=device,
            )
            wp.launch(
                kernel_array.gather_vec_skip_negative,
                dim=n,
                inputs=[face_normals, triangle_id, normals],
                device=device,
            )
        else:
            assert target_normals is not None
            index, distance = tw.neighbors.query_bvh_nearest(target_vertices, current, 1)
            wp.copy(closest, target_vertices[index])
            wp.launch(
                kernel_array.gather_vec_skip_negative,
                dim=n,
                inputs=[target_normals, index, normals],
                device=device,
            )
            triangle_id = index

        # --- resolve robust scale on the first iteration ---
        if kind != 0 and scale_value is None:
            scale_value = _robust_scale_from_residuals(
                current, closest, normals, distance, triangle_id, max_d, kind
            )

        # --- assemble and solve the linearized point-to-plane system ---
        jtj.zero_()
        jtr.zero_()
        cost_acc.zero_()
        wp.launch_tiled(
            kernel_registration.accumulate_point_to_plane,
            dim=[n_tiles],
            inputs=[
                current,
                closest,
                normals,
                distance,
                triangle_id,
                wp.float32(max_d),
                wp.int32(kind),
                wp.float32(scale_value if scale_value is not None else 0.0),
                jtj,
                jtr,
                cost_acc,
            ],
            block_dim=TILE_1D,
            device=device,
        )
        wp.launch(
            kernel_registration.solve_point_to_plane,
            dim=1,
            inputs=[jtj, jtr, wp.float32(damping), step],
            device=device,
        )

        # --- apply the incremental step and compose into the running transform ---
        wp.launch(
            kernel_registration.apply_transform_mat44,
            dim=n,
            inputs=[current, step, updated],
            device=device,
        )
        current, updated = updated, current
        transformed = current
        wp.map(wp.mul, step, total, out=total)

        cost = float(cost_acc.numpy()[0])
        if iteration > 0 and old_cost - cost < threshold:
            break
        old_cost = cost

    return total, transformed, cost
