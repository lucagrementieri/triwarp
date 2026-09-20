"""
Chamfer and Hausdorff distances between point clouds and meshes.

All metrics are fully GPU-resident: they compose the nearest-neighbor and
point-to-surface primitives in [`proximity`][triwarp.proximity] with the tiled reductions
in [`reduce`][triwarp.reduce], and never move per-element data to the host.

Chamfer distances follow the ``pytorch3d`` convention and are built on **squared**
Euclidean distances. Hausdorff distances follow libigl's ``igl::hausdorff`` and reduce the
(already Euclidean) per-element distances with a maximum, so no squaring or final square root is
needed.

One place the convention does *not* carry over, because the two libraries answer different
questions: ``pytorch3d.loss.point_mesh_face_distance`` sums point-to-triangle with
**face**-to-point, where
[`chamfer_points_to_mesh`][triwarp.metrics.chamfer_points_to_mesh]'s backward direction is
mesh-*vertex* to nearest query. The forward halves agree, but the whole scalars differ by
construction and are not comparable.

Two families of geometry are supported and can be mixed:

- ``points_to_points`` -- both sides are point clouds
  (nearest neighbor via [`query_nearest`][triwarp.neighbors.query_nearest]).
- ``points_to_mesh`` -- a point cloud versus a triangle mesh
  (point-to-surface via [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
  one way, mesh-vertex nearest neighbor the other way).
- ``mesh_to_mesh`` -- both sides are triangle meshes (vertex-to-surface both directions).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, cast, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import prefers_tiled_reduction, require_same_device, slice_count
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import metrics as kernel_metrics

_PointReduction = Literal["mean", "sum", "max"]
# Differentiable Chamfer losses support only additive point reductions ("max" has no
# useful gradient here, and "None" is a per-point array rather than a scalar loss).
_DiffReduction = Literal["mean", "sum"]

# Per-point squared-distance result when ``point_reduction is None``.
_UnreducedChamfer = twt.Array1dFloat32 | tuple[twt.Array1dFloat32, twt.Array1dFloat32]

# Forward/backward per-element distances from one of the three geometry dispatches; ``None`` in the
# second slot is the single-directional case.
_DistancePair = tuple[wp.array[wp.float32], wp.array[wp.float32] | None]


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
    ``single_directional``), following the ``pytorch3d`` convention, whose ``single_directional``
    keyword maps onto this one exactly.

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

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_mesh`][triwarp.metrics.chamfer_points_to_mesh]
    [`chamfer_mesh_to_mesh`][triwarp.metrics.chamfer_mesh_to_mesh]
    [`query_nearest`][triwarp.neighbors.query_nearest]
    """
    require_same_device(x=x, y=y)
    _validate_point_reduction(point_reduction)
    distances = _distances_points_to_points(x, y, single_directional)
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, x.device)
    return _chamfer(*distances, point_reduction, single_directional)


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
        [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``points -> mesh surface`` term is computed.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points].
        The forward array has length ``n`` and the backward array length ``v``, except when
        ``points``, ``vertices`` or ``faces`` is empty -- a mesh with no faces counts as empty here
        even when ``vertices`` is not, since there is no surface for the forward term to measure
        against -- in which case both arrays are length 0, matching
        [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points]'s own convention.

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices`` and ``faces`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points]
    [`chamfer_mesh_to_mesh`][triwarp.metrics.chamfer_mesh_to_mesh]
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    """
    require_same_device(points=points, vertices=vertices, faces=faces)
    _validate_point_reduction(point_reduction)
    distances = _distances_points_to_mesh(points, vertices, faces, single_directional)
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, points.device)
    return _chamfer(*distances, point_reduction, single_directional)


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
        [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points].
    single_directional
        If ``True``, only the ``A -> surface(B)`` term is computed.

    Returns
    -------
    float or wp.array[wp.float32] or tuple
        Same layout as [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points].
        The forward array has length ``va`` and the backward array length ``vb``, except when any
        of ``vertices_a``, ``faces_a``, ``vertices_b`` or ``faces_b`` is empty -- a mesh with no
        faces counts as empty here even when its vertices are not -- in which case both arrays are
        length 0.

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"``, ``"sum"``, ``"max"`` or ``None``.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points]
    [`chamfer_points_to_mesh`][triwarp.metrics.chamfer_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][triwarp.metrics.hausdorff_mesh_to_mesh]
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    _validate_point_reduction(point_reduction)
    distances = _distances_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional
    )
    if distances is None:
        return _empty_chamfer(point_reduction, single_directional, vertices_a.device)
    return _chamfer(*distances, point_reduction, single_directional)


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
# differentiable kernel (see [`triwarp.kernels.metrics`]). Gradients flow to the point
# positions (and, for the surface terms, the mesh ``vertices``).
#
# Usage::
#
#     x = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     y = wp.array(..., dtype=wp.vec3, requires_grad=True)
#     tape = wp.Tape()
#     loss = tw.metrics.chamfer_points_to_points_loss(x, y, tape=tape)
#     tape.backward(loss=loss)
#     grad_x = x.grad  # dloss/dx
#
# Passing ``tape=None`` still returns the loss value but records nothing (no gradient).


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
    [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points] that returns
    a length-1 ``wp.float32`` device array instead of a host ``float``, so the loss can be
    back-propagated to ``x`` and ``y`` through a caller-owned ``wp.Tape``.

    The nearest-neighbor assignment (via
    [`query_nearest`][triwarp.neighbors.query_nearest]) is computed
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
        [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points]).
    single_directional
        If ``True``, only the ``x -> y`` term is accumulated.

    Returns
    -------
    wp.array[wp.float32]
        Length-1 device array holding the Chamfer loss (``requires_grad=True``). Empty
        inputs yield a length-1 zero array.

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_points_to_points`][triwarp.metrics.chamfer_points_to_points]
    [`chamfer_points_to_mesh_loss`][triwarp.metrics.chamfer_points_to_mesh_loss]
    [`chamfer_mesh_to_mesh_loss`][triwarp.metrics.chamfer_mesh_to_mesh_loss]
    """
    require_same_device(x=x, y=y)
    _validate_diff_reduction(point_reduction)
    device = x.device
    n = int(x.shape[0])
    m = int(y.shape[0])
    loss = _zero_loss(device)
    if n == 0 or m == 0:
        return loss

    # Non-differentiable nearest-neighbor indices (computed outside the tape). The backward search
    # is seeded from the forward one's own distances -- see `_backward_radius`.
    nearest_xy, dist_xy = tw.neighbors.query_nearest(y, x, k=1)
    terms = [lambda: _launch_nn_term(x, y, nearest_xy, _reduction_scale(point_reduction, n), loss)]
    if not single_directional:
        nearest_yx = tw.neighbors.query_nearest(
            x, y, k=1, initial_radius=_backward_radius(dist_xy)
        )[0]
        terms.append(
            lambda: _launch_nn_term(y, x, nearest_yx, _reduction_scale(point_reduction, m), loss)
        )

    _accumulate_chamfer_terms(tape, terms)
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
    [`chamfer_points_to_mesh`][triwarp.metrics.chamfer_points_to_mesh]. The forward term
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

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices`` and ``faces`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_points_to_mesh`][triwarp.metrics.chamfer_points_to_mesh]
    [`chamfer_points_to_points_loss`][triwarp.metrics.chamfer_points_to_points_loss]
    [`chamfer_mesh_to_mesh_loss`][triwarp.metrics.chamfer_mesh_to_mesh_loss]
    """
    require_same_device(points=points, vertices=vertices, faces=faces)
    _validate_diff_reduction(point_reduction)
    device = points.device
    n = int(points.shape[0])
    v = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    loss = _zero_loss(device)
    if n == 0 or v == 0 or n_faces == 0:
        return loss

    # Non-differentiable closest-face and nearest-neighbor assignment (outside the tape). The
    # backward search is seeded from the forward one's own distances -- see `_backward_radius`.
    _, dist_forward, face_id = tw.proximity.closest_point_on_mesh(vertices, faces, points)
    terms = [
        lambda: _launch_surface_term(
            points, vertices, faces, face_id, _reduction_scale(point_reduction, n), loss
        )
    ]
    if not single_directional:
        nearest_vp = tw.neighbors.query_nearest(
            points, vertices, k=1, initial_radius=_backward_radius(dist_forward)
        )[0]
        terms.append(
            lambda: _launch_nn_term(
                vertices, points, nearest_vp, _reduction_scale(point_reduction, v), loss
            )
        )

    _accumulate_chamfer_terms(tape, terms)
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
    [`chamfer_mesh_to_mesh`][triwarp.metrics.chamfer_mesh_to_mesh]. The forward term is
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

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.
    ValueError
        If ``point_reduction`` is not ``"mean"`` or ``"sum"``.

    See Also
    --------
    [`chamfer_mesh_to_mesh`][triwarp.metrics.chamfer_mesh_to_mesh]
    [`chamfer_points_to_points_loss`][triwarp.metrics.chamfer_points_to_points_loss]
    [`chamfer_points_to_mesh_loss`][triwarp.metrics.chamfer_points_to_mesh_loss]
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
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
    terms = [
        lambda: _launch_surface_term(
            vertices_a, vertices_b, faces_b, face_id_ab, _reduction_scale(point_reduction, va), loss
        )
    ]
    if not single_directional:
        face_id_ba = tw.proximity.closest_point_on_mesh(vertices_a, faces_a, vertices_b)[2]
        terms.append(
            lambda: _launch_surface_term(
                vertices_b,
                vertices_a,
                faces_a,
                face_id_ba,
                _reduction_scale(point_reduction, vb),
                loss,
            )
        )

    _accumulate_chamfer_terms(tape, terms)
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

    Raises
    ------
    RuntimeError
        If ``x`` and ``y`` are not all on one device.

    See Also
    --------
    [`hausdorff_points_to_mesh`][triwarp.metrics.hausdorff_points_to_mesh]
    [`hausdorff_mesh_to_mesh`][triwarp.metrics.hausdorff_mesh_to_mesh]
    [`scipy.spatial.distance.directed_hausdorff`][]
    """
    require_same_device(x=x, y=y)
    distances = _distances_points_to_points(x, y, single_directional)
    if distances is None:
        return 0.0
    return _hausdorff(*distances, single_directional)


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

    Raises
    ------
    RuntimeError
        If ``points``, ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`hausdorff_points_to_points`][triwarp.metrics.hausdorff_points_to_points]
    [`hausdorff_mesh_to_mesh`][triwarp.metrics.hausdorff_mesh_to_mesh]
    """
    require_same_device(points=points, vertices=vertices, faces=faces)
    distances = _distances_points_to_mesh(points, vertices, faces, single_directional)
    if distances is None:
        return 0.0
    return _hausdorff(*distances, single_directional)


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

    Raises
    ------
    RuntimeError
        If ``vertices_a``, ``faces_a``, ``vertices_b`` and ``faces_b`` are not all on one device.

    See Also
    --------
    [`chamfer_mesh_to_mesh`][triwarp.metrics.chamfer_mesh_to_mesh]
    [`hausdorff_points_to_mesh`][triwarp.metrics.hausdorff_points_to_mesh]
    """
    require_same_device(
        vertices_a=vertices_a, faces_a=faces_a, vertices_b=vertices_b, faces_b=faces_b
    )
    distances = _distances_mesh_to_mesh(
        vertices_a, faces_a, vertices_b, faces_b, single_directional
    )
    if distances is None:
        return 0.0
    return _hausdorff(*distances, single_directional)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
#
# Every helper below is shared by two or three of the public entry points above -- the
# validation guards and reductions by all three ``chamfer_*`` functions, the three geometry
# dispatches by a ``chamfer_*`` and a ``hausdorff_*`` pair each, and the launch/tape machinery
# by the ``*_loss`` family -- so they are grouped in one trailing block rather than placed after
# any single caller.


def _validate_point_reduction(point_reduction: _PointReduction | None) -> None:
    if point_reduction is not None and point_reduction not in ("mean", "sum", "max"):
        raise ValueError('point_reduction must be one of "mean", "sum", "max" or None')


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


def _square(distances: wp.array[wp.float32]) -> twt.Array1dFloat32:
    n = int(distances.shape[0])
    squared = twt.empty_1d(n, wp.float32, device=distances.device)
    if n > 0:
        wp.map(kernel_array.square_scalar, distances, out=squared)
    return squared


def _reduce(distances: twt.Array1dFloat32, point_reduction: _PointReduction) -> float:
    if point_reduction == "mean":
        return tw.reduce.mean(distances)
    if point_reduction == "sum":
        return tw.reduce.sum(distances)
    return tw.reduce.max(distances)


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
        return twt.empty_1d(0, wp.float32, device=device)
    return (twt.empty_1d(0, wp.float32, device=device), twt.empty_1d(0, wp.float32, device=device))


# The three geometry dispatches below are each shared by a ``chamfer_*`` and a ``hausdorff_*``
# entry point, which differ only in how they reduce the pair and what they return for degenerate
# input. ``None`` means degenerate, leaving that choice to the caller.
#
# Two of the three run a *second* nearest-neighbour search back over the same pair, and that
# backward search is seeded from the forward one's own answer -- see ``_backward_radius``.


def _distances_points_to_points(
    x: wp.array[wp.vec3], y: wp.array[wp.vec3], single_directional: bool
) -> _DistancePair | None:
    """Nearest-neighbour distances between two clouds, both directions unless directed."""
    if int(x.shape[0]) == 0 or int(y.shape[0]) == 0:
        return None
    d_forward = tw.neighbors.query_nearest(y, x, k=1)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.neighbors.query_nearest(
            x, y, k=1, initial_radius=_backward_radius(d_forward)
        )[1]
    return cast(_DistancePair, (d_forward, d_backward))


def _distances_points_to_mesh(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    single_directional: bool,
) -> _DistancePair | None:
    """Point-to-surface distances forward, cloud nearest-neighbour distances back."""
    if int(points.shape[0]) == 0 or int(vertices.shape[0]) == 0 or int(faces.shape[0]) == 0:
        return None
    d_forward = tw.proximity.closest_point_on_mesh(vertices, faces, points)[1]
    d_backward = None
    if not single_directional:
        # The forward half is point-to-*surface*, so it is a lower bound on the point-to-vertex
        # answer this search wants -- still the right scale to seed it with.
        d_backward = tw.neighbors.query_nearest(
            points, vertices, k=1, initial_radius=_backward_radius(d_forward)
        )[1]
    return cast(_DistancePair, (d_forward, d_backward))


def _backward_radius(d_forward: twt.Array1dFloat32) -> float | None:
    """
    Seed the backward search's radius from the forward search's answer distances.

    [`query_nearest`][triwarp.neighbors.query_nearest] defaults its ``initial_radius`` to
    [`knn_initial_radius`][triwarp.neighbors.knn_initial_radius], which inverts the *target
    cloud's* density -- and under ``"hashgrid"`` that number also fixes the cell width. For two
    clouds sampled from the same surface the density is the right scale, which is why the default
    is what it is. For two clouds that are **displaced** it is not: the answer sits at the
    displacement rather than at the density scale, so the search can widen into an exact scan of
    the whole cloud.

    Both directions of a pair share one distance scale, so the forward half already holds the
    estimate the backward search needs -- the answer is unchanged either way, since this only sets
    where the search ladder *starts* and every row still certifies itself. ``max`` rather than a
    mean or a median because overshooting the radius only costs a coarser grid, while
    undershooting costs a full extra deepening round per row.

    Returns
    -------
    float | None
        The radius, or ``None`` to keep the default when the forward answer carries no finite
        distance to learn from (an empty pair, or every slot unfilled under a ``max_radius``).
    """
    if int(d_forward.shape[0]) == 0:
        return None
    radius = tw.reduce.max(d_forward)
    return radius if math.isfinite(radius) and radius > 0.0 else None


def _distances_mesh_to_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    single_directional: bool,
) -> _DistancePair | None:
    """
    Each mesh's vertices to the other's surface. Vertex-sampled, not a true surface metric.

    The two ``closest_point_on_mesh`` calls below have no data dependency between them, and were
    one of five such pairs across the tree measured against a two-``wp.Stream`` overlap (joined by
    ``wait_stream``) instead of the sequential default-stream form -- indistinguishable from noise
    on all five. See CLAUDE.md section 14.9 for the two structural reasons (host-launch-overhead
    dominance and CG's periodic host-readback check). Kept sequential; do not reintroduce stream
    overlap here without a call site whose own device share is much larger.
    """
    if (
        int(vertices_a.shape[0]) == 0
        or int(vertices_b.shape[0]) == 0
        or int(faces_a.shape[0]) == 0
        or int(faces_b.shape[0]) == 0
    ):
        return None
    d_forward = tw.proximity.closest_point_on_mesh(vertices_b, faces_b, vertices_a)[1]
    d_backward = None
    if not single_directional:
        d_backward = tw.proximity.closest_point_on_mesh(vertices_a, faces_a, vertices_b)[1]
    return d_forward, d_backward


def _validate_diff_reduction(point_reduction: _DiffReduction) -> None:
    if point_reduction not in ("mean", "sum"):
        raise ValueError(
            'Differentiable Chamfer losses support point_reduction "mean" or "sum" only. '
            'Use the non-differentiable chamfer_* functions for "max"/None.'
        )


def _reduction_scale(point_reduction: _DiffReduction, count: int) -> float:
    return (1.0 / count) if point_reduction == "mean" else 1.0


def _zero_loss(device: wp.DeviceLike) -> twt.Array1dFloat32:
    zeros = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    return cast(twt.Array1dFloat32, zeros)


def _accumulate_chamfer_terms(tape: wp.Tape | None, terms: list[Callable[[], None]]) -> None:
    """
    Run every term's launch closure into ``loss``, inside ``tape`` when there is one.

    Each of the three ``chamfer_*_loss`` entry points builds ``terms`` as one zero-argument closure
    per direction it accumulates (one when ``single_directional``, two otherwise) and hands them
    here instead of each writing its own ``_record`` closure. A closure already captures whichever
    assignment it was built from, so no caller needs an ``| None`` placeholder and a narrowing
    assertion for the direction it did not take. Recording is opt-in because a ``wp.Tape`` context
    is only wanted when the caller intends to backpropagate; passing ``tape=None`` still runs every
    term, just untaped.
    """

    def _record() -> None:
        for term in terms:
            term()

    if tape is not None:
        with tape:
            _record()
    else:
        _record()


def _launch_nn_term(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: float,
    loss: twt.Array1dFloat32,
) -> None:
    """Accumulate the point-to-point Chamfer term for ``x`` into ``loss``."""
    _launch_reduction_pair(
        kernel_metrics.chamfer_nn_term_tiled,
        kernel_metrics.chamfer_nn_term_sliced,
        int(x.shape[0]),
        x.device,
        [x, y, nearest, wp.float32(scale)],
        loss,
    )


def _launch_surface_term(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: float,
    loss: twt.Array1dFloat32,
) -> None:
    """Accumulate the point-to-surface Chamfer term for ``points`` into ``loss``."""
    _launch_reduction_pair(
        kernel_metrics.chamfer_surface_term_tiled,
        kernel_metrics.chamfer_surface_term_sliced,
        int(points.shape[0]),
        points.device,
        [points, vertices, faces, face_id, wp.float32(scale)],
        loss,
    )


def _launch_reduction_pair(
    kernel_tiled: wp.Kernel,
    kernel_sliced: wp.Kernel,
    n: int,
    device: wp.DeviceLike,
    common_inputs: list[object],
    loss: twt.Array1dFloat32,
) -> None:
    """
    Dispatch one of a matched ``*_tiled`` / ``*_sliced`` kernel pair over ``n`` items into ``loss``.

    The block-reducing ``*_tiled`` kernel runs on CUDA, the portable strided-slice one on CPU, where
    ``wp.launch_tiled`` runs a single lane per block (see
    [`prefers_tiled_reduction`][triwarp._device.prefers_tiled_reduction]). ``common_inputs`` is
    every argument both kernels share, in the order they declare it, before the sliced kernel's
    extra ``slices`` count and both kernels' trailing ``loss``.

    Shared by [`_launch_nn_term`][triwarp.metrics._launch_nn_term] and
    [`_launch_surface_term`][triwarp.metrics._launch_surface_term] (its last caller), whose only
    difference is which kernel pair and argument list they carry.
    """
    if prefers_tiled_reduction(device):
        wp.launch_tiled(
            kernel_tiled,
            dim=[(n + TILE_1D - 1) // TILE_1D],
            inputs=[*common_inputs, loss],
            block_dim=TILE_1D,
            device=device,
        )
    else:
        slices = slice_count(n, device)
        wp.launch(
            kernel_sliced,
            dim=slices,
            inputs=[*common_inputs, wp.int32(slices), loss],
            device=device,
        )
