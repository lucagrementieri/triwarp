"""
Differentiable kernels for Chamfer distance losses.

The nearest-neighbor / closest-face assignment is a non-differentiable ``argmin``
and is computed by the proximity primitives *outside* the autodiff tape (see
[`triwarp.distance`][]). These kernels consume that fixed assignment and compute
per-element squared distances as pure arithmetic of the input coordinates, so
Warp's reverse-mode autodiff (``wp.Tape``) flows gradients back to the point
positions (and, for the surface terms, the mesh vertices).

Each term accumulates a *scaled* contribution into a length-1 loss accumulator via
``wp.atomic_add`` (which has a well-defined adjoint), so ``"sum"`` and ``"mean"``
reductions differ only by the ``scale`` passed from Python scope. Following the
``pytorch3d`` convention the distances are **squared** Euclidean distances.
"""

import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.triangles import face_vertices

# Relative coplanarity tolerance for the triangle-interior test, matching
# warp.fem's ``project_on_tri_at_origin``.
_TRI_DET_TOLERANCE = wp.constant(wp.float32(1.0e-6))


@wp.func
def project_segment_sq_dist(q: wp.vec3, segment: wp.vec3, length_sq: wp.float32) -> wp.float32:
    """Squared distance from ``q`` to the segment ``[0, segment]`` (clamped projection)."""
    s = wp.clamp(wp.dot(q, segment) / length_sq, wp.float32(0.0), wp.float32(1.0))
    return wp.length_sq(q - s * segment)


@wp.func
def point_triangle_sq_dist(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    """
    Squared Euclidean distance from point ``p`` to triangle ``(a, b, c)``.

    Adapts warp.fem's ``project_on_tri_at_origin``: the closest point lies either in
    the triangle interior (barycentric ``s, t`` both non-negative with ``s + t <= 1``)
    or on one of the three edges. Every branch is a smooth function of ``p, a, b, c``,
    so Warp differentiates whichever branch is taken. Degenerate (near-zero-area)
    triangles fall through to the edge projections.
    """
    q = p - a
    e1 = b - a
    e2 = c - a
    e1e1 = wp.length_sq(e1)
    e1e2 = wp.dot(e1, e2)
    e2e2 = wp.length_sq(e2)
    det = e1e1 * e2e2 - e1e2 * e1e2

    # Initialize before branching (Warp leaves branch-local variables uninitialized
    # when the branch is not taken).
    dist_sq = wp.float32(0.0)
    inside = wp.int32(0)
    if det > e1e1 * e2e2 * _TRI_DET_TOLERANCE:
        e1p = wp.dot(e1, q)
        e2p = wp.dot(e2, q)
        s = (e2e2 * e1p - e1e2 * e2p) / det
        t = (e1e1 * e2p - e1e2 * e1p) / det
        if s >= wp.float32(0.0) and t >= wp.float32(0.0) and s + t <= wp.float32(1.0):
            dist_sq = wp.length_sq(q - s * e1 - t * e2)
            inside = wp.int32(1)

    if inside == wp.int32(0):
        d_e1 = project_segment_sq_dist(q, e1, e1e1)
        d_e2 = project_segment_sq_dist(q, e2, e2e2)
        d_e12 = project_segment_sq_dist(q - e1, e2 - e1, wp.length_sq(e2 - e1))
        dist_sq = wp.min(wp.vec3(d_e1, d_e2, d_e12))

    return dist_sq


@wp.kernel
def chamfer_nn_term_tiled(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||^2`` into ``out_loss[0]``.

    ``nearest[i]`` is the (fixed, non-differentiable) index in ``y`` closest to ``x[i]``.
    CUDA path, launched via ``wp.launch_tiled`` (block ``TILE_1D``): each block reduces its lanes
    cooperatively and commits one atomic; out-of-range lanes contribute zero. Tile ops carry
    adjoints, so the kernel stays differentiable under ``wp.Tape``.

    The CPU device MUST use
    [`chamfer_nn_term_sliced`][triwarp.kernels.distance.chamfer_nn_term_sliced] instead:
    ``wp.launch_tiled`` runs exactly one lane per block there, so this block reduction would see
    one point per tile and the loss would come out roughly 64x too small.
    """
    i, t = wp.tid()
    idx = i * TILE_1D + int(t)
    contrib = wp.float32(0.0)
    if idx < x.shape[0]:
        diff = x[idx] - y[nearest[idx]]
        contrib = scale * wp.length_sq(diff)
    total = wp.tile_sum(wp.tile(contrib))
    if t == 0:
        wp.tile_atomic_add(out_loss, total, (0,))


@wp.kernel
def chamfer_nn_term_sliced(
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    nearest: wp.array[wp.int32],
    scale: wp.float32,
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * ||x[i] - y[nearest[i]]||^2`` into ``out_loss[0]``.

    Portable path, correct on both devices: one thread per slice walks a strided slice of ``x``,
    accumulates locally and commits one ``wp.atomic_add``. Both the dynamic loop and the atomic
    carry adjoints, so the kernel stays differentiable under ``wp.Tape``. Lane-free, so it is
    correct on the CPU device where
    [`chamfer_nn_term_tiled`][triwarp.kernels.distance.chamfer_nn_term_tiled] is not; it gives up
    the block shuffle-reduce and measures 1.57x slower on CUDA at 500k points (19.0 -> 29.7 us),
    which is why both exist.
    """
    j = wp.tid()
    total = wp.float32(0.0)
    for idx in range(int(j), x.shape[0], int(n_slices)):
        diff = x[idx] - y[nearest[idx]]
        total = total + scale * wp.length_sq(diff)
    wp.atomic_add(out_loss, 0, total)


@wp.kernel
def chamfer_surface_term_tiled(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])^2`` into ``out_loss[0]``.

    ``face_id[i]`` is the (fixed, non-differentiable) index of the triangle of the mesh
    closest to ``points[i]``. Gradients flow to both ``points`` and ``vertices``. Points
    with ``face_id[i] < 0`` (no face within the search radius) contribute nothing. Same tiled
    CUDA reduction as [`chamfer_nn_term_tiled`][triwarp.kernels.distance.chamfer_nn_term_tiled],
    and CPU-unsafe for the same reason.
    """
    i, t = wp.tid()
    idx = i * TILE_1D + int(t)
    contrib = wp.float32(0.0)
    if idx < points.shape[0]:
        f = face_id[idx]
        if f >= 0:
            a, b, c = face_vertices(vertices, faces, f)
            contrib = scale * point_triangle_sq_dist(points[idx], a, b, c)
    total = wp.tile_sum(wp.tile(contrib))
    if t == 0:
        wp.tile_atomic_add(out_loss, total, (0,))


@wp.kernel
def chamfer_surface_term_sliced(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_id: wp.array[wp.int32],
    scale: wp.float32,
    n_slices: wp.int32,
    out_loss: wp.array[wp.float32],
) -> None:
    """
    Accumulate ``scale * d(points[i], triangle face_id[i])^2`` into ``out_loss[0]``.

    ``face_id[i]`` is the (fixed, non-differentiable) index of the triangle of the mesh
    closest to ``points[i]``. Gradients flow to both ``points`` and ``vertices``. Points
    with ``face_id[i] < 0`` (no face within the search radius) contribute nothing. Same lane-free
    sliced reduction as
    [`chamfer_nn_term_sliced`][triwarp.kernels.distance.chamfer_nn_term_sliced], for the same
    reason.
    """
    j = wp.tid()
    total = wp.float32(0.0)
    for idx in range(int(j), points.shape[0], int(n_slices)):
        f = face_id[idx]
        if f >= 0:
            a, b, c = face_vertices(vertices, faces, f)
            total = total + scale * point_triangle_sq_dist(points[idx], a, b, c)
    wp.atomic_add(out_loss, 0, total)
