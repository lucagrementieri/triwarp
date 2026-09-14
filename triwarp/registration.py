"""Point cloud registration: Procrustes analysis and ICP."""

from __future__ import annotations

import math
from typing import Literal, TypedDict, cast, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_nonempty_mesh, require_same_device
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import registration as kernel_registration
from triwarp.kernels import transform as kernel_transform


# The ``return_cost=True`` overload comes first because it is the *default*: overload resolution
# picks the first match, so listing ``Literal[False]`` first would type the bare
# ``procrustes(a, b)`` call as returning the matrix alone, when it in fact returns the three-tuple.
# The default cannot simply be dropped from the ``False`` overload -- ``return_cost`` follows
# defaulted parameters, so a non-default there is a syntax error.
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
    return_cost: bool = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float] | wp.array[wp.mat44]: ...
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

    Raises
    ------
    RuntimeError
        If ``a``, ``b`` and ``weights`` are not all on one device.
    ValueError
        If ``b`` (or a non-empty ``weights``) does not have the same length as ``a``, or a
        non-empty ``weights`` sums to zero.

    Notes
    -----
    Latency-bound at every size that matters: the fit is two kernel launches (four with
    ``return_cost``), one allocation for the packed moment accumulator and one readback, so the
    cost is nearly flat in ``n``. Callers in a loop should use the workspace form — see
    [`icp`][triwarp.registration.icp], which allocates once outside its iteration.

    **The returned matrix is column-vector, which is the transpose of pytorch3d's.**
    ``pytorch3d.ops.corresponding_points_alignment`` solves the row-vector form ``s X R + T = Y``,
    so its ``R`` is this matrix's linear block divided by the scale and *transposed*.
    """
    require_same_device(a=a, b=b, weights=weights)
    n = int(a.shape[0])
    device = a.device
    if int(b.shape[0]) != n:
        raise ValueError(f"a and b must have the same length, got {n} and {b.shape[0]}")
    # A zero-length weights array is the kernels' own "uniform weights" sentinel (see
    # ``_uniform_weights``), not a caller mistake -- only a non-empty mismatch is a real error.
    if weights is not None and int(weights.shape[0]) not in (0, n):
        raise ValueError(f"weights must have the same length as a, got {weights.shape[0]} and {n}")

    if n == 0:
        matrix = _identity_mat44(device)
        if not return_cost:
            return matrix
        return matrix, wp.clone(a), 0.0

    workspace = _procrustes_workspace(n, device, return_cost=return_cost)
    weighted = weights is not None and int(weights.shape[0]) == n
    result = _procrustes_into(
        a, b, weights, reflection, translation, scale, return_cost, workspace, weighted
    )

    # An all-zero non-empty ``weights`` divides by zero inside the kernel (the accumulated weight
    # sum is the denominator of every centroid and, with ``scale=True``, of the singular values
    # too), silently returning a matrix of NaN with no exception. The fit *above* already
    # accumulates that sum, so the check reads it back rather than running a reduction over
    # ``weights`` first -- the fit it would have skipped is two launches, against a launch, an
    # allocation and a host sync for the reduction. Its NaN output is discarded by the raise.
    if not return_cost:
        matrix = cast(wp.array[wp.mat44], result)
        if weighted and float(read_scalar(workspace["acc"], kernel_registration.ACC_W_SUM)) == 0.0:
            raise ValueError("weights sum to zero: no point carries any weight")
        return matrix
    matrix, transformed, cost, weight_sum = cast(
        tuple[wp.array[wp.mat44], wp.array[wp.vec3], float, float], result
    )
    if weighted and weight_sum == 0.0:
        raise ValueError("weights sum to zero: no point carries any weight")
    return matrix, transformed, cost


_ROBUST_KINDS: dict[str, int] = {"none": 0, "huber": 1, "tukey": 2}


class _TargetIndex(TypedDict):
    """Everything [`query_nearest`][triwarp.neighbors.query_nearest] can be told once."""

    accelerator: wp.Bvh
    initial_radius: float
    bounds: tuple[wp.vec3, wp.vec3]


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
    (mesh target) or [`query_nearest`][triwarp.neighbors.query_nearest]
    (point-cloud target); the alignment is the GPU tiled SVD of
    [`procrustes`][triwarp.registration.procrustes].

    Its ``R`` is the **transpose** of the linear block returned here, because it solves the
    row-vector form ``s X R + T = Y``; the *converged transform* is what is comparable, not the
    iteration count, since ``relative_rmse_thr`` is its own stopping rule.

    Parameters
    ----------
    a
        Source point cloud, shape ``(n,)``, dtype ``wp.vec3``.
    target_vertices
        Target vertex positions, shape ``(m,)``, dtype ``wp.vec3``. Used as the
        target point cloud when ``target_faces`` is ``None``.
    target_faces
        Flat ``(f * 3,)`` triangle index buffer. When given, the internally built ``wp.Mesh``
        aliases ``target_vertices`` and ``target_faces`` rather than copying them; do not mutate
        them for the duration of the call. When given (and non-empty),
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

    Raises
    ------
    RuntimeError
        If ``a``, ``target_vertices``, ``target_faces`` and ``initial`` are not all on one device.

    See Also
    --------
    [`procrustes`][triwarp.registration.procrustes]
    [`icp_point_to_plane`][triwarp.registration.icp_point_to_plane]
    [`trimesh.registration.icp`][]
    """
    require_same_device(
        a=a, target_vertices=target_vertices, target_faces=target_faces, initial=initial
    )
    device = a.device
    n = int(a.shape[0])

    if n == 0 or int(target_vertices.shape[0]) == 0:
        return _identity_mat44(device), wp.clone(a), math.inf

    initial_matrix, current = _seed_transform(a, initial, device)
    total = initial_matrix
    transformed = wp.clone(current)
    cost = math.inf

    mesh, query_max, target_index = _resolve_icp_target(
        target_vertices, target_faces, current, max_distance, "icp"
    )

    # Correspondence and weight buffers are allocated once and refilled every iteration, and so is
    # the Procrustes workspace — the fit is latency-bound, so its ~10 per-call allocations would
    # otherwise dominate an iteration that is already down to a handful of launches.
    closest = wp.empty(n, dtype=wp.vec3, device=device)
    distance_mesh = wp.empty(n, dtype=wp.float32, device=device)
    triangle_id_mesh = wp.empty(n, dtype=wp.int32, device=device)
    weights: wp.array[wp.float32] | None = (
        wp.empty(n, dtype=wp.float32, device=device) if max_distance is not None else None
    )
    workspace = _procrustes_workspace(n, device, return_cost=True)
    # The second ping-pong slot; see ``_ProcrustesWorkspace`` and the fit call below.
    workspace["spare_matrix"] = wp.empty(1, dtype=wp.mat44, device=device)
    workspace["spare_transformed"] = wp.empty(n, dtype=wp.vec3, device=device)

    # Both ping-pong views built once: ``_slot`` assembles a dict, and doing that per iteration is
    # ~1.5 us of pure Python on a loop whose whole iteration is ~95.
    slots = (_slot(workspace, 0), _slot(workspace, 1))
    parity = 0
    old_cost = math.inf
    # Built on first use rather than here, for the reason in ``icp_point_to_plane``: a cached
    # ``wp.map`` call re-resolves its op and signature every iteration, and its inputs come back
    # from ``_correspondences``, whose mesh and cloud branches return different buffers. They are
    # the same objects on every iteration, so one construction serves the loop.
    threshold_weight: wp.Kernel | None = None
    for _ in range(max_iterations):
        distance, triangle_id = _correspondences(
            mesh,
            target_vertices,
            target_index,
            current,
            query_max,
            closest,
            distance_mesh,
            triangle_id_mesh,
        )

        if max_distance is not None and weights is not None:
            if threshold_weight is None:
                threshold_weight = wp.map(
                    kernel_registration.distance_threshold_weight,
                    distance,
                    triangle_id,
                    wp.float32(max_distance),
                    out=weights,
                    return_kernel=True,
                )
            wp.launch(
                threshold_weight,
                dim=n,
                inputs=[distance, triangle_id, wp.float32(max_distance)],
                outputs=[weights],
                device=device,
            )

        # The fit runs *before* the "every correspondence was rejected" test, not after, because
        # the test's own quantity is one of the moments the fit accumulates (``ACC_W_SUM``) and
        # riding on its readback is one host sync per iteration instead of two. The cost of
        # inverting the order is one wasted fit on the terminal iteration -- two launches against
        # a launch, an allocation and a sync every iteration.
        #
        # **The answer is unchanged, and the ping-pong is what makes that true.** A fit writes the
        # workspace's ``matrix`` and ``transformed`` in place, so running one more would otherwise
        # overwrite the last *good* result with the degenerate one (an all-zero weight sum is the
        # denominator of every centroid, so it fits a matrix of NaN). Alternating the slot leaves
        # the previous fit's buffers untouched, and ``total`` / ``transformed`` / ``cost`` are only
        # rebound once the fit is known to be sound -- exactly what breaking before the fit used to
        # guarantee.
        new_total, new_transformed, new_cost, weight_sum = cast(
            tuple[wp.array[wp.mat44], wp.array[wp.vec3], float, float],
            _procrustes_into(
                a,
                closest,
                weights,
                reflection,
                translation,
                scale,
                True,
                slots[parity],
                max_distance is not None,
            ),
        )
        if max_distance is not None and weight_sum == 0.0:
            break
        total, transformed, cost = new_total, new_transformed, new_cost
        current = transformed
        parity ^= 1
        if old_cost - cost < threshold:
            break
        old_cost = cost

    return total, transformed, cost


class _ProcrustesWorkspace(TypedDict):
    """Buffers a [`procrustes`][triwarp.registration.procrustes] fit writes into."""

    acc: wp.array[wp.float32]
    matrix: wp.array[wp.mat44]
    transformed: wp.array[wp.vec3] | None
    # The second half of a ping-pong, allocated only by ``icp``. A fit writes ``matrix`` and
    # ``transformed`` *in place*, so a caller that keeps the previous fit's answer while running
    # one more -- which is what lets ``icp`` decide "was that fit degenerate?" from the
    # accumulator the fit itself filled -- needs somewhere else for the new one to land.
    spare_matrix: wp.array[wp.mat44] | None
    spare_transformed: wp.array[wp.vec3] | None
    uniform_weights: wp.array[wp.float32]


def _procrustes_workspace(
    n: int, device: wp.DeviceLike, *, return_cost: bool
) -> _ProcrustesWorkspace:
    """Allocate the buffers one Procrustes fit needs; reuse across iterations of a loop."""
    return {
        "acc": wp.empty(kernel_registration.PROCRUSTES_ACC_SIZE, dtype=wp.float32, device=device),
        "matrix": wp.empty(1, dtype=wp.mat44, device=device),
        "transformed": wp.empty(n, dtype=wp.vec3, device=device) if return_cost else None,
        "spare_matrix": None,
        "spare_transformed": None,
        "uniform_weights": _uniform_weights(device),
    }


_UNIFORM_WEIGHTS: dict[str, wp.array[wp.float32]] = {}


def _uniform_weights(device: wp.DeviceLike) -> wp.array[wp.float32]:
    """
    Return the kernels' "uniform weights" sentinel: a zero-length array, shared per device.

    Length zero means "every weight is 1", so the common weightless call needs neither a
    ``wp.full(n, 1.0)`` allocation nor its fill — and since the buffer carries no data, one
    instance per device serves every caller.
    """
    key = str(device)
    if key not in _UNIFORM_WEIGHTS:
        _UNIFORM_WEIGHTS[key] = wp.empty(0, dtype=wp.float32, device=device)
    return _UNIFORM_WEIGHTS[key]


def _slot(workspace: _ProcrustesWorkspace, parity: int) -> _ProcrustesWorkspace:
    """
    One half of the ping-pong: the same workspace with its output buffers swapped on odd ``parity``.

    A Procrustes fit writes ``matrix`` and ``transformed`` **in place**, so a caller that wants to
    run one more fit while still holding the previous one's answer has to send the new one
    somewhere else. Only ``icp`` does; every other caller leaves the spare slots ``None`` and gets
    this workspace back unchanged. The accumulator is deliberately *shared* between the slots --
    it is rewritten from scratch by every fit and read back before the next one starts.
    """
    if parity == 0 or workspace["spare_transformed"] is None:
        return workspace
    return {
        **workspace,
        "matrix": cast(wp.array[wp.mat44], workspace["spare_matrix"]),
        "transformed": workspace["spare_transformed"],
        "spare_matrix": workspace["matrix"],
        "spare_transformed": workspace["transformed"],
    }


def _procrustes_into(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None,
    reflection: bool,
    translation: bool,
    scale: bool,
    return_cost: bool,
    workspace: _ProcrustesWorkspace,
    need_weight_sum: bool = False,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float, float] | wp.array[wp.mat44]:
    """Run one Procrustes fit into caller-owned buffers. See ``procrustes`` for the semantics."""
    n = int(a.shape[0])
    device = a.device
    acc = workspace["acc"]
    out_matrix = workspace["matrix"]
    if weights is None:
        weights = workspace["uniform_weights"]

    acc.zero_()
    wp.launch_tiled(
        kernel_registration.accumulate_procrustes_moments,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[a, b, weights, translation, acc],
        block_dim=TILE_1D,
        device=device,
    )
    wp.launch(
        kernel_registration.build_procrustes_matrix,
        dim=1,
        inputs=[a, b, acc, reflection, translation, scale, out_matrix],
        device=device,
    )

    if not return_cost:
        return out_matrix

    out_transformed = workspace["transformed"]
    assert out_transformed is not None
    # One launch, not two: this both writes ``out_transformed`` and reduces the residual against
    # it. See the kernel for why fusing became worth it only after the reduction was flattened.
    wp.launch_tiled(
        kernel_registration.transform_and_accumulate_cost,
        dim=kernel_reduce.blocks_1d(n),
        inputs=[a, b, weights, out_matrix, acc, out_transformed],
        block_dim=TILE_1D,
        device=device,
    )
    # One readback for the whole accumulator; ICP's convergence test needs the cost on the host,
    # and an extra device-side pass would cost more than the readback. ``ACC_W_SUM`` rides along in
    # the same transfer: ``icp``'s "every correspondence was rejected" guard needs it, and reading
    # it here rather than reducing the weight array separately is one host sync per iteration
    # instead of two -- but only when ``need_weight_sum`` says a caller wants it.
    cost_slot = int(kernel_registration.ACC_COST)
    if not need_weight_sum:
        return out_matrix, out_transformed, float(read_scalar(acc, cost_slot)), 0.0
    # Both scalars in *one* transfer when the caller wants both. Two ``read_scalar`` calls measured
    # worse than this (1.011 against 0.950 ms on a ten-iteration fit) even though the second read
    # rides on a drained pipeline, and one ``read_scalar`` is better than this when only the cost
    # is wanted, because ``.numpy()`` allocates a host array where ``read_scalar`` reuses a cached
    # scratch. Hence the flag rather than one spelling for both callers.
    acc_np = acc.numpy()
    return (
        out_matrix,
        out_transformed,
        float(acc_np[cost_slot]),
        float(acc_np[int(kernel_registration.ACC_W_SUM)]),
    )


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
        cloud ``target_vertices`` and ``target_normals`` is required. When given, the internally
        built ``wp.Mesh`` aliases ``target_vertices`` and ``target_faces`` rather than copying
        them; do not mutate them for the duration of the call.
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
        If ``robust_kernel`` is not one of the three names, or if the target is a point cloud
        (``target_faces is None``) and ``target_normals`` is not provided, or does not have the
        same length as ``target_vertices``.
    RuntimeError
        If ``a``, ``target_vertices``, ``target_faces``, ``target_normals`` and ``initial`` are not
        all on one device.

    See Also
    --------
    [`icp`][triwarp.registration.icp]
    [`estimate_normals`][triwarp.points.estimate_normals]
    [`normals_at_closest_faces`][triwarp.proximity.normals_at_closest_faces]
    """
    require_same_device(
        a=a,
        target_vertices=target_vertices,
        target_faces=target_faces,
        target_normals=target_normals,
        initial=initial,
    )
    device = a.device
    n = int(a.shape[0])
    is_mesh = _is_mesh_target(target_faces)
    if robust_kernel not in _ROBUST_KINDS:
        raise ValueError(
            f"robust_kernel must be one of {list(_ROBUST_KINDS)}, got {robust_kernel!r}"
        )
    kind = _ROBUST_KINDS[robust_kernel]

    if not is_mesh and target_normals is None:
        raise ValueError(
            "icp_point_to_plane requires target_normals for a point-cloud target "
            "(target_faces is None)."
        )
    if (
        not is_mesh
        and target_normals is not None
        and int(target_normals.shape[0]) != int(target_vertices.shape[0])
    ):
        # ``target_normals`` feeds ``gather_vec_skip_negative`` below, indexed by a nearest-vertex
        # id that ranges over ``target_vertices``; a shorter buffer is an out-of-bounds read on
        # both devices (CLAUDE.md §12.1), not merely a wrong answer.
        raise ValueError(
            "target_normals must have the same length as target_vertices, got "
            f"{target_normals.shape[0]} and {target_vertices.shape[0]}."
        )

    if n == 0 or int(target_vertices.shape[0]) == 0:
        return _identity_mat44(device), wp.clone(a), math.inf

    initial_matrix, current = _seed_transform(a, initial, device)
    transformed = wp.clone(current)
    cost = math.inf

    mesh, query_max, target_index = _resolve_icp_target(
        target_vertices, target_faces, current, max_distance, "icp_point_to_plane"
    )
    face_normals: wp.array[wp.vec3] | None = None
    if is_mesh:
        assert target_faces is not None
        face_normals, _ = tw.triangles.face_normals_and_areas(target_vertices, target_faces)
    # ``target_index`` is already the point-cloud one from ``_resolve_icp_target`` above when
    # ``not is_mesh`` -- rebuilding it here duplicated a ``wp.Bvh`` build and an ``aabb`` reduction
    # for no behavioral difference.

    max_d = max_distance if max_distance is not None else math.inf
    scale_value = robust_scale
    old_cost = math.inf

    # All per-iteration buffers are allocated once; the accumulators are zeroed in place and
    # ``current``/``updated`` ping-pong. ``total`` is cloned so composing in place never mutates
    # a caller-provided initial transform. The per-iteration cost read stays: it is the
    # stopping criterion (a 4-byte transfer).
    closest = wp.empty(n, dtype=wp.vec3, device=device)
    distance_mesh = wp.empty(n, dtype=wp.float32, device=device)
    triangle_id_mesh = wp.empty(n, dtype=wp.int32, device=device)
    normals = wp.empty(n, dtype=wp.vec3, device=device)
    # Only allocated when correspondences can actually be rejected -- see the all-rejected guard
    # below, which needs a buffer to reduce over.
    valid: wp.array[wp.bool] | None = (
        wp.empty(n, dtype=wp.bool, device=device) if max_distance is not None else None
    )
    jtj = wp.zeros(1, dtype=wp.spatial_matrix, device=device)
    jtr = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    # One buffer for both scalars the loop reads back, not two: they are written by the same
    # launch, so reading them together costs one host sync per iteration where reading them at
    # their two natural points cost two -- and the second of those sat three launches downstream,
    # so it drained a pipeline the first had already drained rather than riding on it.
    scalar_acc = wp.zeros(kernel_registration.ICP_SCALAR_ACC_SIZE, dtype=wp.float32, device=device)
    step = wp.empty(1, dtype=wp.mat44, device=device)
    updated = wp.empty(n, dtype=wp.vec3, device=device)
    total = wp.clone(initial_matrix)

    # The per-iteration maps are hoisted to their kernels. A cached ``wp.map`` call still resolves
    # its op and input signature in Python on every call -- measured 23.7-24.8 us against 12.6-13.2
    # for the launch it wraps -- so two of them is ~21 us of a ~200 us iteration. End to end that
    # is 1.02-1.06x at 3 000 source points and 3 or 10 iterations, which is small and free.
    #
    # Measure this kind of change with ``threshold=0.0``, or the reading is fiction: with the
    # convergence break live the two arms stop at *different* iterations and the ratio reported
    # 1.86x, none of which was the hoist. The plateau is also not bit-reproducible -- the
    # point-to-plane normal equations are accumulated with ``float32`` atomics, so two runs of the
    # identical build disagree in the last digits once the cost stops moving.
    #
    # ``residual_valid`` is built on first use rather than here because the buffers it maps over
    # come back from ``_correspondences``, whose mesh and cloud branches return different arrays;
    # they are the same objects every iteration, so one construction still serves the whole loop.
    accumulate_step = wp.map(wp.mul, step, total, out=total, return_kernel=True)
    residual_valid: wp.Kernel | None = None

    for iteration in range(max_iterations):
        # --- correspondence + target normals ---
        distance, triangle_id = _correspondences(
            mesh,
            target_vertices,
            target_index,
            current,
            query_max,
            closest,
            distance_mesh,
            triangle_id_mesh,
        )
        # One gather either way: for a mesh target ``triangle_id`` indexes the face normals, for a
        # cloud it indexes the target's own per-vertex normals.
        normal_source = face_normals if mesh is not None else target_normals
        assert normal_source is not None
        wp.launch(
            kernel_array.gather_vec_skip_negative,
            dim=n,
            inputs=[normal_source, triangle_id, normals],
            device=device,
        )

        # --- bail out once no correspondence survives the distance gate ---
        # Mirrors ``icp``'s ``sum(weights) == 0`` check: without it, every accumulator below stays
        # at its zeroed initial value, the damped 6x6 solve returns a zero step, and the loop
        # reports ``cost=0.0`` -- indistinguishable from a perfect fit -- instead of stopping with
        # the last real cost (or ``math.inf`` if nothing ever matched).
        if valid is not None:
            if residual_valid is None:
                residual_valid = wp.map(
                    kernel_registration.residual_valid,
                    triangle_id,
                    distance,
                    wp.float32(max_d),
                    out=valid,
                    return_kernel=True,
                )
            wp.launch(
                residual_valid,
                dim=n,
                inputs=[triangle_id, distance, wp.float32(max_d)],
                outputs=[valid],
                device=device,
            )
            if not tw.reduce.any(valid):
                break

        # --- resolve robust scale on the first iteration ---
        if kind != 0 and scale_value is None:
            scale_value = _robust_scale_from_residuals(
                current, closest, normals, distance, triangle_id, max_d, kind
            )

        # --- assemble and solve the linearized point-to-plane system ---
        jtj.zero_()
        jtr.zero_()
        scalar_acc.zero_()
        wp.launch_tiled(
            kernel_registration.accumulate_point_to_plane,
            dim=kernel_reduce.blocks_1d(n),
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
                scalar_acc,
            ],
            block_dim=TILE_1D,
            device=device,
        )

        # --- bail out once every in-range correspondence's own robust weight collapsed to zero ---
        # The ``valid`` guard above only catches a correspondence rejected by *distance*; a Tukey
        # kernel can drive every remaining correspondence's weight to exactly zero on its own
        # (``|residual| >= scale``, reachable through an explicit ``robust_scale`` too tight for the
        # residual distribution, or an earlier iteration's step overshooting far past what a
        # first-iteration-derived scale anticipated) while ``residual_valid`` -- and so ``valid`` --
        # still accepts every one of them. Without this, ``jtj``/``jtr`` stay exactly zero, the
        # damped solve returns a near-identity step, and ``cost`` reads ``0.0`` -- indistinguishable
        # from a perfect fit -- for the rest of the run. Reproduced with
        # ``robust_kernel="tukey", robust_scale=1e-9``: every residual exceeds so tight a scale, and
        # the loop returned the identity transform with ``cost=0.0`` on a cloud still offset by
        # (1.0, 0.5, -0.3) from its target.
        # Both scalars in one readback -- ``.numpy()`` on the two-element buffer rather than two
        # ``read_scalar`` calls, which is the one shape that helper does not cover (it returns a
        # single element). ``cost`` is the *post-accumulate* cost the convergence test below needs
        # and this launch is what wrote it, so reading it here rather than at the end of the
        # iteration reads the identical value. Measured 1.052-1.065x on a 10-iteration fit.
        scalars = scalar_acc.numpy()
        if float(scalars[kernel_registration.ICP_WEIGHT_SUM]) <= 0.0:
            break
        cost = float(scalars[kernel_registration.ICP_COST])

        wp.launch(
            kernel_registration.solve_point_to_plane,
            dim=1,
            inputs=[jtj, jtr, wp.float32(damping), step],
            device=device,
        )

        # --- apply the incremental step and compose into the running transform ---
        wp.launch(
            kernel_transform.apply_transform_mat44,
            dim=n,
            inputs=[current, step, updated],
            device=device,
        )
        current, updated = updated, current
        transformed = current
        wp.launch(accumulate_step, dim=1, inputs=[step, total], outputs=[total], device=device)

        if iteration > 0 and old_cost - cost < threshold:
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


def _identity_mat44(device: wp.DeviceLike) -> wp.array[wp.mat44]:
    """Return a ``(1,)`` array holding the 4x4 identity transform."""
    identity = wp.mat44(
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
    )
    return wp.array([identity], dtype=wp.mat44, device=device)


def _correspondences(
    mesh: wp.Mesh | None,
    target_vertices: wp.array[wp.vec3],
    target_index: _TargetIndex | None,
    current: wp.array[wp.vec3],
    query_max: wp.float32,
    closest: wp.array[wp.vec3],
    distance_mesh: wp.array[wp.float32],
    triangle_id_mesh: wp.array[wp.int32],
) -> tuple[wp.array[wp.float32], wp.array[wp.int32]]:
    """
    Match every source position against the target, writing the matched point into ``closest``.

    Writes the matched point of every source position into ``closest``. A mesh target runs the BVH
    closest-point kernel into the caller's preallocated buffers; a cloud target runs the k-NN query,
    which returns its own, so the returned pair is the caller's buffers in the first case and fresh
    views in the second -- both loops rebind rather than assuming.

    Parameters
    ----------
    mesh
        Mesh target, or ``None`` for a cloud target.
    target_vertices
        ``(m,)`` target positions.
    target_index
        Precomputed k-NN state; required when ``mesh`` is ``None``.
    current
        ``(n,)`` transformed source positions to match.
    query_max
        Search radius for the mesh closest-point query.
    closest
        ``(n,)`` output, written either way.
    distance_mesh, triangle_id_mesh
        Preallocated ``(n,)`` buffers the mesh branch writes into.

    Returns
    -------
    tuple[wp.array[wp.float32], wp.array[wp.int32]]
        ``(distance, correspondence_index)`` -- a face index for a mesh target, a vertex index for
        a cloud one.
    """
    if mesh is not None:
        wp.launch(
            kernel_proximity.closest_point_on_mesh,
            dim=int(current.shape[0]),
            inputs=[
                mesh.id,
                current,
                wp.float32(query_max),
                closest,
                distance_mesh,
                triangle_id_mesh,
            ],
            device=current.device,
        )
        return distance_mesh, triangle_id_mesh

    assert target_index is not None
    index, distance = tw.neighbors.query_nearest(target_vertices, current, 1, **target_index)
    # Not ``tw.array.gather``: ``closest`` is allocated once outside the ICP loop, and a gather
    # would add one allocation per iteration.
    wp.copy(closest, target_vertices[index])
    return distance, index


def _seed_transform(
    a: wp.array[wp.vec3], initial: wp.array[wp.mat44] | wp.mat44 | None, device: wp.DeviceLike
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3]]:
    """
    Normalize the initial transform and apply it, giving both ICP loops their starting state.

    Parameters
    ----------
    a
        ``(n,)`` source point cloud.
    initial
        Initial transform as a ``(1,)`` ``wp.mat44`` array, a scalar ``wp.mat44``, or ``None`` for
        the identity.
    device
        Device to allocate on.

    Returns
    -------
    tuple[wp.array[wp.mat44], wp.array[wp.vec3]]
        ``(initial_matrix, current)`` -- the resolved transform and the image of ``a`` under it.
    """
    initial_matrix = _resolve_initial(initial, device)
    current = wp.empty(int(a.shape[0]), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_transform.apply_transform_mat44,
        dim=int(a.shape[0]),
        inputs=[a, initial_matrix, current],
        device=device,
    )
    return initial_matrix, current


def _resolve_initial(
    initial: wp.array[wp.mat44] | wp.mat44 | None, device: wp.DeviceLike
) -> wp.array[wp.mat44]:
    """Normalize ``initial`` to a ``(1,)`` ``wp.mat44`` device array."""
    if initial is None:
        return _identity_mat44(device)
    if isinstance(initial, wp.array):
        return wp.clone(initial)
    return wp.array([initial], dtype=wp.mat44, device=device)


def _resolve_icp_target(
    target_vertices: wp.array[wp.vec3],
    target_faces: wp.array[wp.int32] | None,
    current: wp.array[wp.vec3],
    max_distance: float | None,
    caller: str,
) -> tuple[wp.Mesh | None, wp.float32, _TargetIndex | None]:
    """
    Build the loop-invariant search state for whichever target kind was supplied.

    A mesh target gets a ``wp.Mesh`` and a query radius; a point-cloud target gets the BVH,
    bounding box and density estimate of [`_target_index`][triwarp.registration._target_index].
    Exactly one of the two is non-``None``.

    Parameters
    ----------
    target_vertices
        ``(m,)`` target positions.
    target_faces
        Flat triangle index buffer, or ``None`` / empty for a point-cloud target.
    current
        The transformed source, used only to size the mesh query radius.
    max_distance
        Correspondence rejection distance; widens the query radius when larger than the box
        diagonal.
    caller
        Calling function's name, for ``require_nonempty_mesh``'s error message.

    Returns
    -------
    tuple[wp.Mesh | None, wp.float32, _TargetIndex | None]
        ``(mesh, query_max, target_index)``.
    """
    if not _is_mesh_target(target_faces):
        return None, wp.float32(0.0), _target_index(target_vertices)

    assert target_faces is not None
    require_nonempty_mesh(target_faces, caller)
    # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
    mesh = wp.Mesh(points=target_vertices, indices=target_faces)
    query_max = tw.bounds.enclosing_diagonal(mesh.points, current)
    if max_distance is not None:
        query_max = max(query_max, max_distance)
    return mesh, query_max, None


def _target_index(target_vertices: wp.array[wp.vec3]) -> _TargetIndex:
    """
    Precompute the k-NN search state for a point-cloud target.

    Only the *source* moves between ICP iterations, so the target's BVH, bounding box and density
    estimate are all loop-invariant. Hoisting them turns each iteration's correspondence step into
    a single launch with no host synchronisation at all.
    """
    bounds = tw.bounds.aabb(target_vertices)
    return {
        "accelerator": tw.neighbors.bvh_from_points(target_vertices),
        "initial_radius": tw.neighbors.knn_initial_radius(target_vertices, 1, bounds=bounds),
        "bounds": bounds,
    }


def _is_mesh_target(target_faces: wp.array[wp.int32] | None) -> bool:
    return target_faces is not None and int(target_faces.shape[0]) // 3 > 0
