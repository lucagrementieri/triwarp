"""
The signed heat method: signed distance to a set of oriented curves on a surface.

Diffusing an *indicator* of a curve tells you how far away it is; diffusing the curve's **normal**
tells you which side you are on as well (Feng & Crane 2024). That is the whole idea: seed a tangent
vector field with each curve segment's normal, diffuse it for a short time with the connection
Laplacian, normalize it into a unit field, and integrate that field back into a scalar — which comes
out negative on one side of the curve and positive on the other.

Compared to the alternative (compute unsigned distance, then work out the sign separately with
winding numbers or ray casts) the sign here is a by-product of the same solve, and it degrades
gracefully: the method does not need the curve to be closed, watertight, or even connected, because
nothing about it depends on an inside/outside test.

Curves are given as **vertex paths** — the form
[`homology_generators`][triwarp.homology.homology_generators],
[`boundary_loops`][triwarp.boundary.boundary_loops] and
[`boundary_loop`][triwarp.boundary.boundary_loop] all produce — packed into one flat buffer with CSR
offsets. Curves at arbitrary barycentric points are not accepted yet.

Like the rest of the heat-method family this is CUDA-only: every stage is a conjugate-gradient
solve.
"""

from __future__ import annotations

import itertools

import numpy as np
import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.typing as twt
from triwarp._device import require_cuda
from triwarp.kernels.heat import distance as kernel_heat_distance
from triwarp.kernels.heat import signed as kernel_heat_signed

_CG_TOLERANCE = 1e-8


def heat_signed_distance(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    closed: bool = True,
    level_set_constraint: str = "zero_set",
    operators: tw.heat.vector.VectorHeatOperators | None = None,
) -> wp.array[wp.float64]:
    """
    Signed distance from every vertex to a set of oriented curves.

    The result is positive inside the region a counter-clockwise curve encloses and negative outside
    it, with the curve itself at (or near) zero — geometry-central's convention. Orientation is what
    fixes the sign: reversing a curve's vertex order negates the whole field.

    Three stages, mirroring ``potpourri3d.MeshSignedHeatSolver.compute_distance``:

    1. each curve segment splats its normal onto its two endpoints, weighted by half its length;
    2. the connection Laplacian diffuses that tangent field for a short time ``t`` and it is
       normalized, giving a unit field that approximates the signed distance's gradient;
    3. a Poisson solve integrates the field back into a scalar.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    curve_vertices
        Vertex indices along the curves, packed one curve after another. Consecutive entries should
        be adjacent on the mesh; nothing breaks if they are not, but the source is then splatted
        along a chord rather than along the surface.
    curve_offsets
        Length ``n_curves + 1`` CSR bounds into ``curve_vertices``. When ``None`` the whole buffer
        is treated as a single curve.
    t
        Diffusion time; defaults to the squared mean edge length. Larger values smooth the field.
    closed
        Whether each curve closes back on its first vertex (adding one more segment). Signed
        distance is most meaningful for closed curves; an open curve gives a field whose sign flips
        across it but which has no consistent far-field meaning.
    level_set_constraint
        ``"zero_set"`` pins the curve vertices to exactly zero and solves the Poisson problem on the
        rest, through the machinery behind
        [`min_quad_with_fixed`][triwarp.linalg.min_quad_with_fixed].
        ``"none"`` solves unconstrained and then shifts the field so the curve's mean is zero,
        which leaves the level set slightly off the curve but is cheaper and pins nothing. Both are
        modes ``potpourri3d`` offers under the same names.
    operators
        Optional precomputed
        [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators] for this mesh — the
        vector heat system, the scalar operators and the frames. They depend on the mesh alone, so
        passing them back skips every assembly on calls after the first, which for this method is
        three matrices.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` signed distance field on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``level_set_constraint`` is not ``"zero_set"`` or ``"none"``.
    NotImplementedError
        On the CPU device: every stage is a conjugate-gradient solve, and ``warp.optim.linear.cg``
        returns NaN on the CPU in Warp 1.14-1.15.

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
    [`homology_generators`][triwarp.homology.homology_generators]
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    """
    if level_set_constraint not in ("zero_set", "none"):
        raise ValueError(
            f'level_set_constraint must be "zero_set" or "none", got {level_set_constraint!r}'
        )
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_vertices == 0 or n_faces == 0 or int(curve_vertices.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)
    require_cuda(device, "heat_signed_distance")

    if operators is None:
        operators = tw.heat.vector.vector_heat_operators(vertices, faces, t)
    vector_system, scalar, frames = operators
    basis_x, basis_y, vertex_normals = frames
    poisson_system, poisson_preconditioner = scalar[3], scalar[4]
    cot_entries, face_normals = scalar[5], scalar[6]

    # Stage 1: splat each segment's normal onto its endpoints.
    segments = _curve_segments(curve_vertices, curve_offsets, closed=closed)
    source = wp.zeros(n_vertices, dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_heat_signed.splat_curve_normals,
        dim=int(segments.shape[0]),
        inputs=[vertices, segments, vertex_normals, basis_x, basis_y, source],
        device=device,
    )

    # Stage 2: diffuse the tangent field, then keep only its direction.
    diffused = tw.heat.vector.diffuse_tangent_field(vector_system, source)
    unit_field = wp.empty(n_vertices, dtype=wp.vec2d, device=device)
    wp.map(kernel_heat_signed.normalize_or_zero, diffused, out=unit_field)

    # Stage 3: integrate the unit field back into a scalar with a Poisson solve. The cotangent
    # weights and face normals come from the same bundle, so the Poisson stage and the diffusion
    # cannot drift apart.
    face_field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat_signed.vertex_field_to_face_field,
        dim=n_faces,
        inputs=[faces, face_normals, unit_field, basis_x, basis_y, face_field],
        device=device,
    )
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat_distance.integrated_divergence,
        dim=n_faces,
        inputs=[vertices, faces, cot_entries, face_field, divergence],
        device=device,
    )

    if level_set_constraint == "zero_set":
        return _solve_poisson_zero_set(
            poisson_system, divergence, curve_vertices, n_vertices, device
        )
    return _solve_poisson_shifted(
        poisson_system, poisson_preconditioner, divergence, curve_vertices, n_vertices, device
    )


def _curve_segments(
    curve_vertices: wp.array[wp.int32], curve_offsets: wp.array[wp.int32] | None, *, closed: bool
) -> twt.Array2dInt32:
    """
    Expand CSR vertex paths into a flat ``(n_segments, 2)`` list of endpoint pairs.

    Built on the host: a curve on a surface is small (thousands of vertices at most, against the
    mesh's millions), the offsets have to be read to know where the curves end anyway, and doing it
    here keeps the splat kernel free of the wrap-around bookkeeping that ``closed`` implies.
    """
    indices = curve_vertices.numpy()
    bounds = (
        np.array([0, len(indices)], dtype=np.int64)
        if curve_offsets is None
        else curve_offsets.numpy().astype(np.int64)
    )

    pairs: list[np.ndarray] = []
    for begin, end in itertools.pairwise(bounds):
        curve = indices[begin:end]
        if len(curve) < 2:
            continue
        pairs.append(np.stack([curve[:-1], curve[1:]], axis=1))
        if closed:
            pairs.append(np.array([[curve[-1], curve[0]]], dtype=curve.dtype))
    if not pairs:
        return twt.empty_int32_2d((0, 2), device=curve_vertices.device)
    segments = np.ascontiguousarray(np.concatenate(pairs), dtype=np.int32)
    return twt.as_array2d_int32(wp.array(segments, dtype=wp.int32, device=curve_vertices.device))


def _solve_poisson_zero_set(
    operator: wps.BsrMatrix[wp.float64],
    divergence: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem with the curve pinned to zero.

    ``linalg.min_quad_with_fixed`` minimizes ``0.5 x' Q x`` and has no place for a linear term, so
    the pieces underneath it are used directly: the free-free block comes from
    [`assemble_interior_system`][triwarp.linalg.assemble_interior_system] (whose own right-hand side
    is zero here, the pinned values being zero) and the divergence is compacted into it.
    """
    fixed_mask = tw.array.indices_to_mask(curve_vertices, n_vertices)
    free_map, n_free = twl.free_partition(fixed_mask)
    if n_free == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    zeros = wp.zeros((1, n_vertices), dtype=wp.float64, device=device)
    operator_uu, rhs = twl.assemble_interior_system(
        operator, fixed_mask, free_map, twt.as_array2d_float(zeros, dtype=wp.float64), n_free
    )
    # Flip sign with the operator: the Poisson right-hand side is -div for the -L convention.
    negated = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.neg, divergence, out=negated)
    wp.launch(
        kernel_heat_signed.scatter_free_rhs,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, negated, rhs],
        device=device,
    )

    solution = wp.zeros((1, n_free), dtype=wp.float64, device=device)
    twl.solve_spd_columns(
        operator_uu, rhs, twt.as_array2d_float(solution, dtype=wp.float64), tol=_CG_TOLERANCE
    )
    field = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat_signed.gather_free_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, solution, field],
        device=device,
    )
    return field


def _solve_poisson_shifted(
    operator: wps.BsrMatrix[wp.float64],
    preconditioner: wpl.LinearOperator,
    divergence: wp.array[wp.float64],
    curve_vertices: wp.array[wp.int32],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """
    Solve the Poisson problem unconstrained, then shift so the curve's mean value is zero.

    The unconstrained system is singular up to a constant (a pure Neumann problem), which conjugate
    gradient handles while the right-hand side is consistent; the shift afterwards picks that
    constant, and putting the curve at zero is the choice that makes the result a distance.
    """
    negated = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.neg, divergence, out=negated)

    field = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(operator, negated, field, tol=_CG_TOLERANCE, preconditioner=preconditioner)
    offset = tw.reduce.mean(tw.array.gather(field, curve_vertices))
    wp.map(wp.sub, field, wp.float64(offset), out=field)
    return field
