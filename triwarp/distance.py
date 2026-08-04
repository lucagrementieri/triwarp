"""
Chamfer and Hausdorff distances between point clouds and meshes on NVIDIA Warp.

All metrics are fully GPU-resident: they compose the nearest-neighbor and
point-to-surface primitives in [`triwarp.proximity`][] with the tiled reductions
in [`triwarp.reduce`][], and never move per-element data to the host.

Chamfer distances follow the ``pytorch3d`` convention and are built on **squared**
Euclidean distances (via [`square`][triwarp.array.square]). Hausdorff distances follow
libigl's ``igl::hausdorff`` and reduce the (already Euclidean) per-element distances with
a maximum, so no squaring or final square root is needed.

Two families of geometry are supported and can be mixed:

- ``points_to_points`` -- both sides are point clouds
  (nearest neighbor via [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]).
- ``points_to_mesh`` -- a point cloud versus a triangle mesh
  (point-to-surface via [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
  one way, mesh-vertex nearest neighbor the other way).
- ``mesh_to_mesh`` -- both sides are triangle meshes (vertex-to-surface both directions).
"""

from __future__ import annotations

from typing import Any, Literal, cast, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import ITEMS_PER_SLICE
from triwarp.kernels import distance as kernel_distance

_PointReduction = Literal["mean", "sum", "max"]
# Differentiable Chamfer losses support only additive point reductions ("max" has no
# useful gradient here, and "None" is a per-point array rather than a scalar loss).
_DiffReduction = Literal["mean", "sum"]

# Per-point squared-distance result when ``point_reduction is None``.
_UnreducedChamfer = twt.Array1dFloat32 | tuple[twt.Array1dFloat32, twt.Array1dFloat32]


def _validate_point_reduction(point_reduction: _PointReduction | None) -> None:
    if point_reduction is not None and point_reduction not in ("mean", "sum", "max"):
        raise ValueError('point_reduction must be one of "mean", "sum", "max" or None')


def _square(distances: wp.array[wp.float32]) -> twt.Array1dFloat32:
    return cast(twt.Array1dFloat32, tw.array.square(cast(Any, distances)))


def _reduce(distances: twt.Array1dFloat32, point_reduction: _PointReduction) -> float:
    if point_reduction == "mean":
        return tw.reduce.mean(distances)
    if point_reduction == "sum":
        return tw.reduce.sum(distances)
    return tw.reduce.max(distances)


def _chamfer(
    d_forward: wp.array[wp.float32],
    d_backward: wp.array[wp.float32] | None,
    point_reduction: _PointReduction | None,
    single_directional: bool,
) -> float | _UnreducedChamfer:
    """
    Combine forward/backward Euclidean distances into a Chamfer value.

    Distances are squared element-wise (pytorch3d convention) before reduction.
    """
    sq_forward = _square(d_forward)
    if point_reduction is None:
        if single_directional:
            return sq_forward
        return sq_forward, _square(cast(wp.array[wp.float32], d_backward))

    reduced_forward = _reduce(sq_forward, point_reduction)
    if single_directional:
        return reduced_forward

    sq_backward = _square(cast(wp.array[wp.float32], d_backward))
    reduced_backward = _reduce(sq_backward, point_reduction)
    if point_reduction == "max":
        return max(reduced_forward, reduced_backward)
    return reduced_forward + reduced_backward


def _hausdorff(
    d_forward: wp.array[wp.float32],
    d_backward: wp.array[wp.float32] | None,
    single_directional: bool,
) -> float:
    """
    Directed (or symmetric) Hausdorff distance from Euclidean distances.

    ``max(d)`` over Euclidean per-element distances equals ``sqrt(max(d**2))``,
    so this matches libigl's ``sqrt(max(dba, dab))`` without squaring.
    """
    directed_forward = tw.reduce.max(cast(twt.Array1dFloat32, d_forward))
    if single_directional:
        return directed_forward
    directed_backward = tw.reduce.max(cast(twt.Array1dFloat32, d_backward))
    return max(directed_forward, directed_backward)


def _empty_chamfer(
    point_reduction: _PointReduction | None, single_directional: bool, device: wp.DeviceLike
) -> float | _UnreducedChamfer:
    if point_reduction is not None:
        return 0.0
    if single_directional:
        return cast(twt.Array1dFloat32, wp.empty(0, dtype=wp.float32, device=device))
    return (
        cast(twt.Array1dFloat32, wp.empty(0, dtype=wp.float32, device=device)),
        cast(twt.Array1dFloat32, wp.empty(0, dtype=wp.float32, device=device)),
    )


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------


@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
) -> float: ...
@overload
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: None,
    single_directional: bool = ...,
) -> _UnreducedChamfer: ...
def chamfer_points_to_points(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between two point clouds.

    For each point in ``x`` the squared Euclidean distance to its nearest neighbor
    in ``y`` is accumulated (and symmetrically for ``y`` into ``x`` unless
    ``single_directional``), following the ``pytorch3d`` convention.

    Parameters
    ----------
    x
        ``(n,)`` query point cloud as ``wp.vec3``.
    y
        ``(m,)`` target point cloud as ``wp.vec3``.
    point_reduction
        How to reduce the per-point squared distances of each direction:
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None`` to return the
        per-point squared distances unreduced. ``"max"`` yields a directed
        Hausdorff-like value.
    single_directional
        If ``True``, only the ``x -> y`` term is computed.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        The reduced Chamfer distance as a Python ``float`` when ``point_reduction``
        is not ``None``. Otherwise the per-point squared distances: a single
        ``(n,)`` array when ``single_directional``, else an ``((n,), (m,))`` tuple.
        Empty inputs yield ``0.0`` (reduced) or empty arrays.

    See Also
    --------
    [`chamfer_points_to_mesh`][triwarp.distance.chamfer_points_to_mesh]
    [`chamfer_mesh_to_mesh`][triwarp.distance.chamfer_mesh_to_mesh]
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]
    """
    _validate_point_reduction(point_reduction)
    device = x.device
    n = int(x.shape[0])
    m = int(y.shape[0])
    if n == 0 or m == 0:
        return _empty_chamfer(point_reduction, single_directional, device)

    d_forward = tw.neighbors.query_hashgrid_nearest(y, x, k=1)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.neighbors.query_hashgrid_nearest(x, y, k=1)[1]
    return _chamfer(d_forward, d_backward, point_reduction, single_directional)


@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
) -> float: ...
@overload
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: bool = ...,
) -> _UnreducedChamfer: ...
def chamfer_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between a point cloud and a triangle mesh.

    The forward term is the squared distance from each point to the mesh surface
    (exact, via [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]);
    the backward term is the squared distance from each mesh vertex to its nearest
    point in the cloud.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud as ``wp.vec3``.
    vertices
        ``(v,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    point_reduction
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None``. See
        [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``points -> mesh surface`` term is computed.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points].
        The forward array has length ``n`` and the backward array length ``v``.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points]
    [`chamfer_mesh_to_mesh`][triwarp.distance.chamfer_mesh_to_mesh]
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    """
    _validate_point_reduction(point_reduction)
    device = points.device
    n = int(points.shape[0])
    v = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n == 0 or v == 0 or n_faces == 0:
        return _empty_chamfer(point_reduction, single_directional, device)

    d_forward = tw.proximity.closest_point_on_mesh(vertices, faces, points)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.neighbors.query_hashgrid_nearest(points, vertices, k=1)[1]
    return _chamfer(d_forward, d_backward, point_reduction, single_directional)


@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction = ...,
    single_directional: bool = ...,
) -> float: ...
@overload
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: None,
    single_directional: bool = ...,
) -> _UnreducedChamfer: ...
def chamfer_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    point_reduction: _PointReduction | None = "mean",
    single_directional: bool = False,
) -> float | _UnreducedChamfer:
    """
    Chamfer distance between two triangle meshes (vertex-to-surface).

    The forward term is the squared distance from each vertex of mesh ``A`` to the
    surface of mesh ``B``; the backward term is the reverse. This mirrors the
    vertex-based approach of libigl while accumulating (mean/sum) instead of maximizing.

    Parameters
    ----------
    vertices_a, faces_a
        Vertices ``(va,)`` (``wp.vec3``) and flat faces ``(fa * 3,)`` (``wp.int32``) of mesh ``A``.
    vertices_b, faces_b
        Vertices ``(vb,)`` and flat faces ``(fb * 3,)`` of mesh ``B``.
    point_reduction
        ``"mean"`` (default), ``"sum"``, ``"max"``, or ``None``. See
        [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``A -> surface(B)`` term is computed.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points].
        The forward array has length ``va`` and the backward array length ``vb``.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points]
    [`chamfer_points_to_mesh`][triwarp.distance.chamfer_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][triwarp.distance.hausdorff_mesh_to_mesh]
    """
    _validate_point_reduction(point_reduction)
    device = vertices_a.device
    va = int(vertices_a.shape[0])
    vb = int(vertices_b.shape[0])
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    if va == 0 or vb == 0 or n_faces_a == 0 or n_faces_b == 0:
        return _empty_chamfer(point_reduction, single_directional, device)

    d_forward = tw.proximity.closest_point_on_mesh(vertices_b, faces_b, vertices_a)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.proximity.closest_point_on_mesh(vertices_a, faces_a, vertices_b)[1]
    return _chamfer(d_forward, d_backward, point_reduction, single_directional)


# ---------------------------------------------------------------------------
# Differentiable Chamfer losses
# ---------------------------------------------------------------------------
#
# The Chamfer functions above return a Python ``float`` (a host scalar) and are not
# differentiable. The ``*_loss`` variants below instead return a length-1 ``wp.float32``
# device array carrying the (squared, pytorch3d-convention) Chamfer loss, so gradients
# can be back-propagated with a caller-owned ``wp.Tape``.
#
# Autodiff strategy (mirrors pytorch3d): the nearest-neighbor / closest-face assignment
# is a non-differentiable ``argmin`` and is computed *outside* the tape; the assignment
# is then held constant while the per-element squared distance is recomputed by a
# differentiable kernel (see [`triwarp.kernels.distance`]). Gradients flow to the point
# positions (and, for the surface terms, the mesh ``vertices``).
#
# Usage::
#
#     x = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     y = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     tape = wp.Tape()
#     loss = tw.distance.chamfer_points_to_points_loss(x, y, tape=tape)
#     tape.backward(loss=loss)
#     grad_x = x.grad  # dloss/dx
#
# Passing ``tape=None`` still returns the loss value but records nothing (no gradient).


def _validate_diff_reduction(point_reduction: _DiffReduction) -> None:
    if point_reduction not in ("mean", "sum"):
        raise ValueError(
            'Differentiable Chamfer losses support point_reduction "mean" or "sum" only. '
            'Use the non-differentiable chamfer_* functions for "max"/None.'
        )


def _reduction_scale(point_reduction: _DiffReduction, count: int) -> float:
    return (1.0 / count) if point_reduction == "mean" else 1.0


def _loss_slices(count: int) -> int:
    """Thread count for the lane-free chamfer reductions: one thread per strided slice."""
    return max(1, (count + ITEMS_PER_SLICE - 1) // ITEMS_PER_SLICE)


def _zero_loss(device: wp.DeviceLike) -> twt.Array1dFloat32:
    zeros = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    return cast(twt.Array1dFloat32, zeros)


def chamfer_points_to_points_loss(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
) -> twt.Array1dFloat32:
    """
    Differentiable Chamfer loss between two point clouds.

    Squared-distance analogue of
    [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points] that returns
    a length-1 ``wp.float32`` device array instead of a host ``float``, so the loss can be
    back-propagated to ``x`` and ``y`` through a caller-owned ``wp.Tape``.

    The nearest-neighbor assignment (via
    [`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest]) is computed
    outside ``tape`` and held constant during the backward pass, matching pytorch3d's
    ``chamfer_distance`` gradient.

    Parameters
    ----------
    x
        ``(n,)`` query point cloud as ``wp.vec3``. Set ``requires_grad=True`` for gradients.
    y
        ``(m,)`` target point cloud as ``wp.vec3``. Set ``requires_grad=True`` for gradients.
    tape
        Caller-owned ``wp.Tape`` into which the differentiable kernels are recorded. When
        ``None`` the loss is still computed but no operations are taped (no gradient).
    point_reduction
        ``"mean"`` (default) or ``"sum"``; reduces the per-point squared distances of each
        direction. ``"max"`` and ``None`` are not supported (see
        [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points]).
    single_directional
        If ``True``, only the ``x -> y`` term is accumulated.

    Returns
    -------
    wp.array[wp.float32]
        Length-1 device array holding the Chamfer loss (``requires_grad=True``). Empty
        inputs yield a length-1 zero array.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.distance.chamfer_points_to_points]
    [`chamfer_points_to_mesh_loss`][triwarp.distance.chamfer_points_to_mesh_loss]
    [`chamfer_mesh_to_mesh_loss`][triwarp.distance.chamfer_mesh_to_mesh_loss]
    """
    _validate_diff_reduction(point_reduction)
    device = x.device
    n = int(x.shape[0])
    m = int(y.shape[0])
    loss = _zero_loss(device)
    if n == 0 or m == 0:
        return loss

    # Non-differentiable nearest-neighbor indices (computed outside the tape).
    nearest_xy = tw.neighbors.query_hashgrid_nearest(y, x, k=1)[0]
    nearest_yx = None
    if not single_directional:
        nearest_yx = tw.neighbors.query_hashgrid_nearest(x, y, k=1)[0]

    def _record() -> None:
        wp.launch(
            kernel_distance.chamfer_nn_term,
            dim=_loss_slices(n),
            inputs=[
                x,
                y,
                nearest_xy,
                wp.float32(_reduction_scale(point_reduction, n)),
                wp.int32(_loss_slices(n)),
                loss,
            ],
            device=device,
        )
        if not single_directional:
            wp.launch(
                kernel_distance.chamfer_nn_term,
                dim=_loss_slices(m),
                inputs=[
                    y,
                    x,
                    nearest_yx,
                    wp.float32(_reduction_scale(point_reduction, m)),
                    wp.int32(_loss_slices(m)),
                    loss,
                ],
                device=device,
            )

    if tape is not None:
        with tape:
            _record()
    else:
        _record()
    return loss


def chamfer_points_to_mesh_loss(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
) -> twt.Array1dFloat32:
    """
    Differentiable Chamfer loss between a point cloud and a triangle mesh.

    Squared-distance analogue of
    [`chamfer_points_to_mesh`][triwarp.distance.chamfer_points_to_mesh]. The forward term
    is the exact point-to-surface squared distance (closest triangle held constant during
    backprop); the backward term is the squared distance from each mesh vertex to its
    nearest point in the cloud. Gradients flow to ``points`` **and** ``vertices``.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud as ``wp.vec3``.
    vertices
        ``(v,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    tape
        Caller-owned ``wp.Tape`` for the backward pass, or ``None`` to skip taping.
    point_reduction
        ``"mean"`` (default) or ``"sum"``.
    single_directional
        If ``True``, only the ``points -> mesh surface`` term is accumulated.

    Returns
    -------
    wp.array[wp.float32]
        Length-1 device array holding the Chamfer loss (``requires_grad=True``).

    See Also
    --------
    [`chamfer_points_to_mesh`][triwarp.distance.chamfer_points_to_mesh]
    [`chamfer_points_to_points_loss`][triwarp.distance.chamfer_points_to_points_loss]
    [`chamfer_mesh_to_mesh_loss`][triwarp.distance.chamfer_mesh_to_mesh_loss]
    """
    _validate_diff_reduction(point_reduction)
    device = points.device
    n = int(points.shape[0])
    v = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    loss = _zero_loss(device)
    if n == 0 or v == 0 or n_faces == 0:
        return loss

    # Non-differentiable closest-face and nearest-neighbor assignment (outside the tape).
    face_id = tw.proximity.closest_point_on_mesh(vertices, faces, points)[2]
    nearest_vp = None
    if not single_directional:
        nearest_vp = tw.neighbors.query_hashgrid_nearest(points, vertices, k=1)[0]

    def _record() -> None:
        wp.launch(
            kernel_distance.chamfer_surface_term,
            dim=_loss_slices(n),
            inputs=[
                points,
                vertices,
                faces,
                face_id,
                wp.float32(_reduction_scale(point_reduction, n)),
                wp.int32(_loss_slices(n)),
                loss,
            ],
            device=device,
        )
        if not single_directional:
            wp.launch(
                kernel_distance.chamfer_nn_term,
                dim=_loss_slices(v),
                inputs=[
                    vertices,
                    points,
                    nearest_vp,
                    wp.float32(_reduction_scale(point_reduction, v)),
                    wp.int32(_loss_slices(v)),
                    loss,
                ],
                device=device,
            )

    if tape is not None:
        with tape:
            _record()
    else:
        _record()
    return loss


def chamfer_mesh_to_mesh_loss(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    tape: wp.Tape | None = None,
    point_reduction: _DiffReduction = "mean",
    single_directional: bool = False,
) -> twt.Array1dFloat32:
    """
    Differentiable Chamfer loss between two triangle meshes (vertex-to-surface).

    Squared-distance analogue of
    [`chamfer_mesh_to_mesh`][triwarp.distance.chamfer_mesh_to_mesh]. The forward term is
    the squared distance from each vertex of mesh ``A`` to the surface of mesh ``B`` (with
    the closest triangle held constant during backprop); the backward term is the reverse.
    Gradients flow to both meshes' vertices.

    Parameters
    ----------
    vertices_a, faces_a
        Vertices ``(va,)`` (``wp.vec3``) and flat faces ``(fa * 3,)`` (``wp.int32``) of mesh ``A``.
    vertices_b, faces_b
        Vertices ``(vb,)`` and flat faces ``(fb * 3,)`` of mesh ``B``.
    tape
        Caller-owned ``wp.Tape`` for the backward pass, or ``None`` to skip taping.
    point_reduction
        ``"mean"`` (default) or ``"sum"``.
    single_directional
        If ``True``, only the ``A -> surface(B)`` term is accumulated.

    Returns
    -------
    wp.array[wp.float32]
        Length-1 device array holding the Chamfer loss (``requires_grad=True``).

    See Also
    --------
    [`chamfer_mesh_to_mesh`][triwarp.distance.chamfer_mesh_to_mesh]
    [`chamfer_points_to_points_loss`][triwarp.distance.chamfer_points_to_points_loss]
    [`chamfer_points_to_mesh_loss`][triwarp.distance.chamfer_points_to_mesh_loss]
    """
    _validate_diff_reduction(point_reduction)
    device = vertices_a.device
    va = int(vertices_a.shape[0])
    vb = int(vertices_b.shape[0])
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    loss = _zero_loss(device)
    if va == 0 or vb == 0 or n_faces_a == 0 or n_faces_b == 0:
        return loss

    # Non-differentiable closest-face assignment for each direction (outside the tape).
    face_id_ab = tw.proximity.closest_point_on_mesh(vertices_b, faces_b, vertices_a)[2]
    face_id_ba = None
    if not single_directional:
        face_id_ba = tw.proximity.closest_point_on_mesh(vertices_a, faces_a, vertices_b)[2]

    def _record() -> None:
        wp.launch(
            kernel_distance.chamfer_surface_term,
            dim=_loss_slices(va),
            inputs=[
                vertices_a,
                vertices_b,
                faces_b,
                face_id_ab,
                wp.float32(_reduction_scale(point_reduction, va)),
                wp.int32(_loss_slices(va)),
                loss,
            ],
            device=device,
        )
        if not single_directional:
            wp.launch(
                kernel_distance.chamfer_surface_term,
                dim=_loss_slices(vb),
                inputs=[
                    vertices_b,
                    vertices_a,
                    faces_a,
                    face_id_ba,
                    wp.float32(_reduction_scale(point_reduction, vb)),
                    wp.int32(_loss_slices(vb)),
                    loss,
                ],
                device=device,
            )

    if tape is not None:
        with tape:
            _record()
    else:
        _record()
    return loss


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------


def hausdorff_points_to_points(
    x: wp.array[wp.vec3], y: wp.array[wp.vec3], *, single_directional: bool = False
) -> float:
    """
    Hausdorff distance between two point clouds.

    The directed distance ``d(x, y) = max_i min_j ||x_i - y_j||`` is computed (and
    symmetrized with ``d(y, x)`` unless ``single_directional``). Equivalent to
    symmetrizing [`scipy.spatial.distance.directed_hausdorff`][].

    Parameters
    ----------
    x
        ``(n,)`` query point cloud as ``wp.vec3``.
    y
        ``(m,)`` target point cloud as ``wp.vec3``.
    single_directional
        If ``True``, return only the directed distance ``d(x, y)``.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    See Also
    --------
    [`hausdorff_points_to_mesh`][triwarp.distance.hausdorff_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][triwarp.distance.hausdorff_mesh_to_mesh]
    [`scipy.spatial.distance.directed_hausdorff`][]
    """
    n = int(x.shape[0])
    m = int(y.shape[0])
    if n == 0 or m == 0:
        return 0.0

    d_forward = tw.neighbors.query_hashgrid_nearest(y, x, k=1)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.neighbors.query_hashgrid_nearest(x, y, k=1)[1]
    return _hausdorff(d_forward, d_backward, single_directional)


def hausdorff_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    single_directional: bool = False,
) -> float:
    """
    Hausdorff distance between a point cloud and a triangle mesh.

    The forward distance is the maximum over points of the distance to the mesh
    surface; the backward distance is the maximum over mesh vertices of the distance
    to the nearest point in the cloud.

    Parameters
    ----------
    points
        ``(n,)`` query point cloud as ``wp.vec3``.
    vertices
        ``(v,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    single_directional
        If ``True``, return only the directed ``points -> mesh surface`` distance.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    See Also
    --------
    [`hausdorff_points_to_points`][triwarp.distance.hausdorff_points_to_points]
    [`hausdorff_mesh_to_mesh`][triwarp.distance.hausdorff_mesh_to_mesh]
    """
    n = int(points.shape[0])
    v = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n == 0 or v == 0 or n_faces == 0:
        return 0.0

    d_forward = tw.proximity.closest_point_on_mesh(vertices, faces, points)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.neighbors.query_hashgrid_nearest(points, vertices, k=1)[1]
    return _hausdorff(d_forward, d_backward, single_directional)


def hausdorff_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    single_directional: bool = False,
) -> float:
    """
    Hausdorff distance between two triangle meshes.

    Direct port of libigl's ``igl::hausdorff`` (vertex-based overload): the distance
    from every vertex of one mesh to the surface of the other is computed in both
    directions and the overall maximum is returned. This equals
    ``sqrt(max(d_ba, d_ab))`` in libigl, where ``d_*`` are squared distances.

    Parameters
    ----------
    vertices_a, faces_a
        Vertices ``(va,)`` (``wp.vec3``) and flat faces ``(fa * 3,)`` (``wp.int32``) of mesh ``A``.
    vertices_b, faces_b
        Vertices ``(vb,)`` and flat faces ``(fb * 3,)`` of mesh ``B``.
    single_directional
        If ``True``, return only the directed ``A -> surface(B)`` distance.

    Returns
    -------
    float
        The (symmetric or directed) Hausdorff distance. ``0.0`` for empty inputs.

    See Also
    --------
    [`chamfer_mesh_to_mesh`][triwarp.distance.chamfer_mesh_to_mesh]
    [`hausdorff_points_to_mesh`][triwarp.distance.hausdorff_points_to_mesh]
    """
    va = int(vertices_a.shape[0])
    vb = int(vertices_b.shape[0])
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    if va == 0 or vb == 0 or n_faces_a == 0 or n_faces_b == 0:
        return 0.0

    d_forward = tw.proximity.closest_point_on_mesh(vertices_b, faces_b, vertices_a)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.proximity.closest_point_on_mesh(vertices_a, faces_a, vertices_b)[1]
    return _hausdorff(d_forward, d_backward, single_directional)
