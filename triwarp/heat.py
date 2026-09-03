"""
Heat-diffusion methods on triangle meshes.

Three solvers that share one idea: diffuse a quantity over the surface for a short time ``t``, then
recover the answer from the *direction* of the resulting field rather than its magnitude. Short-time
heat flow approximates the geodesic kernel, so a single sparse solve carries information that a
combinatorial shortest-path search would have to walk edge by edge. Each solver differs only in what
it diffuses and how it reads the result back:

- **Geodesic distance.** [`heat_geodesic`][triwarp.heat.heat_geodesic] diffuses a scalar indicator
  from source vertices, normalizes its gradient, and integrates that unit field back with a Poisson
  solve. This is the heat method of Crane et al. (``igl::heat_geodesics``,
  ``potpourri3d.MeshHeatMethodDistanceSolver``).
- **Signed distance to curves.** [`heat_signed_distance`][triwarp.heat.heat_signed_distance]
  diffuses the normals of a set of oriented curves, then solves a Poisson problem against that field
  to get a *signed* distance whose zero set is the curves
  (``potpourri3d.MeshSignedHeatSolver``).
- **Vector-valued transport.**
  [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors],
  [`extend_scalar`][triwarp.heat.extend_scalar] and [`log_map`][triwarp.heat.log_map] diffuse
  *tangent vectors* through the connection Laplacian, which transports each vector into its
  neighbour's frame before differencing (``potpourri3d.MeshVectorHeatSolver``).

The three are one module rather than three because they are one family and because two of them
cannot be separated: the vector solvers import [`heat_operators`][triwarp.heat.heat_operators]
and [`heat_geodesic`][triwarp.heat.heat_geodesic],
[`VectorHeatOperators`][triwarp.heat.VectorHeatOperators] embeds the scalar method's operator
tuple, and [`log_map`][triwarp.heat.log_map]'s radius *is* ``heat_geodesic``'s answer.

All three run in ``float64``, because the diffused field decays exponentially and underflows
``float32``. They run on either device.

The operators these solvers assemble are **not** here -- the cotangent and connection Laplacians
live in [`triwarp.laplacian`][triwarp.laplacian], tangent frames in
[`triwarp.tangent_space`][triwarp.tangent_space], and the batched conjugate-gradient machinery in
[`triwarp.linalg`][triwarp.linalg]. This module is the three algorithms only.
"""

from __future__ import annotations

import itertools

import numpy as np
import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.reduce as twr
import triwarp.typing as twt
from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.edges import mean_unique_edge_length
from triwarp.kernels import array as kernel_array
from triwarp.kernels import heat as kernel_heat
from triwarp.kernels import predicates as kernel_predicates
from triwarp.kernels import scatter as kernel_scatter
from triwarp.laplacian import (
    connection_laplacian,
    cotmatrix,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
    mollify_intrinsic,
)
from triwarp.tangent_space import vertex_tangent_frames
from triwarp.triangles import face_normals_and_areas

# Every solve here is a conjugate-gradient one and they all converge at the same tolerance; it was
# spelled three times when these were three modules.
_CG_TOLERANCE = 1e-8


HeatOperators = tuple[
    wps.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    wps.BsrMatrix[wp.float64],
    wps.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    wp.array[wp.float32],
    wp.array[wp.vec3],
    wp.array[wp.float32],
]
"""What [`heat_operators`][triwarp.heat.heat_operators] returns for the heat method's
solves."""


def heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    use_robust: bool = False,
    cot_entries: twt.Array2dFloat | None = None,
) -> HeatOperators:
    """
    Assemble the source-independent operators the heat method solves against.

    Every quantity here depends on the mesh alone, not on the source set, so a caller computing
    distance from many different sources on one mesh can build these once and pass them back through
    ``heat_geodesic(..., operators=...)``. That is the split
    ``potpourri3d.MeshHeatMethodDistanceSolver`` and ``igl::heat_geodesics`` expose as a stateful
    solver object; here it stays a plain tuple of buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default).
    use_robust
        Build the Laplacian from *mollified* edge lengths
        ([`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]) instead of straight from vertex
        positions. Costs one extra pass and two host readbacks, and is what lets the method run on a
        mesh with degenerate triangles at all. It leaves a clean mesh's operator unchanged.

        This is mollification **only**, not the intrinsic Delaunay retriangulation that
        [`robust_laplacian`][triwarp.laplacian.robust_laplacian] also does by default (and that
        ``potpourri3d``'s identically-named flag includes). The reason is structural rather than a
        shortcut: flipping changes which faces exist, and the gradient and divergence stages below
        integrate over faces. Swapping in an operator built on a different triangulation while those
        stages still use the original one is not a cheap approximation, it is inconsistent — so a
        fully intrinsic heat method needs intrinsic *mass*, *gradient* and *divergence* as well. Use
        [`robust_laplacian`][triwarp.laplacian.robust_laplacian] directly where only the operator
        matters (smoothing, parametrization, spectral work).
    cot_entries
        Optional precomputed [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries], shape
        ``(n_faces, 3)``. Depends on the mesh alone, so a caller assembling these operators at
        several diffusion times reuses one table -- and
        [`Trimesh.cotmatrix_entries`][triwarp.mesh.Trimesh.cotmatrix_entries] has it cached. Both
        precisions are accepted: the assembly casts to the matrix dtype in a single build.

    Returns
    -------
    heat_system : warp.sparse.BsrMatrix
        ``M - t * L`` in ``float64``, the heat-diffusion system.
    heat_preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi preconditioner for ``heat_system``.
    laplacian : warp.sparse.BsrMatrix
        The ``float64`` cotangent stiffness matrix ``L`` (igl sign convention, so ``-L`` is positive
        semi-definite).
    poisson_system : warp.sparse.BsrMatrix
        ``-L``, the positive-semi-definite Poisson operator.
    poisson_preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi preconditioner for ``poisson_system``.
    cot_entries : wp.array[wp.float32]
        Per-face half-cotangent weights, reused by the divergence.
    face_normals : wp.array[wp.vec3]
        One unit normal per face.
    face_areas : wp.array[wp.float32]
        One area per face.

    Notes
    -----
    The Poisson operator and both Jacobi preconditioners are here because they satisfy this
    function's own contract — they depend on the mesh alone — and
    [`heat_geodesic`][triwarp.heat.heat_geodesic] used to rebuild all three on every call.
    Measured on ``sphere_small`` (2 562 vertices), that was 0.33 ms for the ``bsr_axpy`` and 0.12 ms
    per preconditioner out of a 6.29 ms amortized call, and the same on ``sphere_med`` where the
    call is 12.57 ms: about 10 % and 5 % respectively. It is *only* those three — the solver
    **state** is deliberately not cached, because a ``warp.optim.linear`` state captures its
    right-hand-side and solution buffers at construction, which would make these operators
    stateful and unsafe to share between two concurrent solves.

    !!! note "Caching the solver state would not pay, and this is why"
        The obvious next step — a persistent single-rhs solver mirroring
        [`spd_column_solver`][triwarp.linalg.spd_column_solver], keeping the CG loop's captured
        graph alive across calls — was priced and **declined**: there is no host-side per-iteration
        cost for it to remove. Measured on the amortized path, both solves cold-started the way
        ``heat_geodesic`` actually starts them:

        | | ``sphere_small`` | ``sphere_med`` |
        |---|---|---|
        | amortized call | 5.65 ms | 11.66 ms |
        | heat solve alone | 1.98 | 2.02 |
        | Poisson solve alone | **3.30** | **9.28** |
        | the two together | **93.5 %** | **97.0 %** |

        and the whole call tracks the CG tolerance almost linearly (``sphere_med``: 12.65 / 10.21 /
        5.16 / 2.44 ms at ``tol`` = 1e-10 / 1e-6 / 1e-3 / 1e-1). So the call is **iteration-bound,
        and the Poisson half is the expensive one**. That the cost is *device*-side rather than host
        follows from ``check_every=0``, which takes the host out of the loop entirely and measures a
        1.9x **loss** warm. The only real lever is conditioning — a preconditioner stronger than
        Jacobi on a cotangent operator — which is the same conclusion the constrained-solve family
        reaches in [`harmonic`][triwarp.parametrization.harmonic].

        Two traps for anyone re-measuring this. ``solve_spd`` **warm-starts from whatever the
        solution buffer already holds**, so timing it in a loop over one buffer makes every rep
        after the first converge in ~0 iterations and reports 0.52 ms instead of 3.30. And this
        tuple's *third* field is the raw (singular) Laplacian, not the Poisson system — handing it
        a right-hand side runs CG to its 25 620-iteration cap.

    Raises
    ------
    ValueError
        If both ``cot_entries`` and ``use_robust`` are given: ``use_robust`` exists to build that
        very table from mollified edge lengths, so the two ask for different weights.

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    [`Trimesh.heat_operators`][triwarp.mesh.Trimesh.heat_operators]
    """
    if cot_entries is not None and use_robust:
        raise ValueError(
            "cot_entries and use_robust are mutually exclusive: use_robust rebuilds the "
            "half-cotangent table from mollified edge lengths."
        )
    if t is None:
        # The unique-edge average, which is what ``igl::heat_geodesics`` uses for its timestep.
        h = mean_unique_edge_length(vertices, faces)
        t = h * h

    # Per-face half-cotangent weights (float32, O(1) and safe) reused for both the Laplacian and
    # the divergence. The cotangent stiffness follows the igl convention (negative diagonal, so
    # ``-L`` is positive semi-definite) but is assembled here in float64.
    if use_robust:
        # Mollified lengths: one global constant added to every edge so no triangle is degenerate.
        # The gradient and divergence stages below still use the extrinsic positions, so this makes
        # the *solves* robust rather than turning the whole method intrinsic.
        lengths, _ = mollify_intrinsic(vertices, faces)
        cot_entries = cotmatrix_entries_intrinsic(lengths)
    elif cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces)
    # ``cotmatrix`` casts the shared float32 half-cotangent weights to float64 and assembles the
    # operator natively in a single build, avoiding a recast rebuild (see cotmatrix's kernel note).
    laplacian = cotmatrix(vertices, faces, cot_entries=cot_entries, dtype=wp.float64)

    # Face normals / areas (float32) for the gradient; the lumped mass is built natively in float64
    # by ``mass_matrix_entries``.
    normals, areas = face_normals_and_areas(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)

    # Heat system (M - t L). ``bsr_axpy`` overwrites the mass matrix in place (no longer needed).
    mass_diag = wps.bsr_diag(diag=mass)
    heat_system = wps.bsr_axpy(x=laplacian, y=mass_diag, alpha=-float(t), beta=1.0)
    # Poisson operator ``-L`` and the two Jacobi preconditioners: mesh-only, so they belong here
    # rather than in every solve. See Notes.
    poisson_system = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    return (
        heat_system,
        wpl.preconditioner(heat_system, "diag"),
        laplacian,
        poisson_system,
        wpl.preconditioner(poisson_system, "diag"),
        cot_entries,
        normals,
        areas,
    )


def heat_geodesic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    t: float | None = None,
    operators: HeatOperators | None = None,
    *,
    use_robust: bool = False,
) -> wp.array[wp.float64]:
    """
    Approximate geodesic distance to the nearest source vertex (Crane et al. heat method).

    Diffuses heat from the source vertices for a short time ``t``, normalizes the resulting
    gradient into a unit vector field pointing away from the sources, and integrates it back into a
    distance field by solving a Poisson problem. Both solves are sparse, symmetric positive
    (semi-)definite systems handled on-device by conjugate gradient. The result is an
    *approximation* of the true geodesic distance (typically a few percent error), matching
    ``igl::heat_geodesics``.

    The computation runs in ``float64``: the diffused heat decays exponentially away from the
    source and would underflow ``float32``, collapsing the far field.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices. The returned distance is measured to
        the nearest source and is zero at the source set.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default), which balances accuracy and smoothing. Ignored when
        ``operators`` is given, which already fixes it.
    operators
        Optional precomputed [`heat_operators`][triwarp.heat.heat_operators] for this mesh.
        They depend on the mesh only, so passing them back skips the assembly on every solve after
        the first — worth it when computing distance from many different source sets.
    use_robust
        Forwarded to [`heat_operators`][triwarp.heat.heat_operators]: build the Laplacian
        from mollified edge lengths, which is what makes the solves survive degenerate triangles.
        Ignored when ``operators`` is supplied. ``potpourri3d.MeshHeatMethodDistanceSolver`` has
        the same flag and defaults it to ``True``; this defaults to ``False`` so the plain call
        stays exactly ``igl::heat_geodesics``.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` geodesic distance field on ``vertices.device``.

    See Also
    --------
    [`heat_operators`][triwarp.heat.heat_operators]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length]
    [`marching_triangles`][triwarp.intersection.marching_triangles]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    if n_vertices == 0 or n_faces == 0 or int(sources.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    if operators is None:
        operators = heat_operators(vertices, faces, t, use_robust=use_robust)
    (
        heat_system,
        heat_preconditioner,
        _laplacian,
        poisson_system,
        poisson_preconditioner,
        cot_entries,
        normals,
        areas,
    ) = operators

    # Heat solve: (M - t L) u = u0, with u0 the source indicator.
    u0 = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.seed_source_indicator,
        dim=int(sources.shape[0]),
        inputs=[sources, u0],
        device=device,
    )

    heat = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(heat_system, u0, heat, tol=_CG_TOLERANCE, preconditioner=heat_preconditioner)

    # Unit vector field X = -grad(u)/|grad(u)|.
    field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat.face_unit_gradients,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, wp.float64(-1.0), field],
        device=device,
    )

    # Integrated divergence b = div(X), then Poisson solve L phi = b, i.e. (-L) phi = -b with the
    # positive semi-definite operator.
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.integrated_divergence,
        dim=n_faces,
        inputs=[vertices, faces, cot_entries, field, divergence],
        device=device,
    )
    # Flip sign so the Poisson right-hand side matches the positive semi-definite operator ``-L``.
    neg_divergence = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.neg, divergence, out=neg_divergence)

    phi = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(
        poisson_system,
        neg_divergence,
        phi,
        tol=_CG_TOLERANCE,
        preconditioner=poisson_preconditioner,
    )

    # Shift so the distance field is zero at the (nearest) source. For a correctly signed field
    # the global minimum sits at the source set, so subtracting it yields a nonnegative field.
    # Device reduction, not ``phi.numpy().min()``: ``phi`` is float64, so a readback moves 8 B per
    # vertex across the bus to produce one scalar. Measured interleaved (RTX 5090, min of 30):
    # CUDA 0.42x at 5k vertices, crossing over near 100k, 3.96x at 500k and 13.69x at 2M. CPU
    # regresses 6-13x throughout -- Warp's CPU reduction against vectorized NumPy -- and that is
    # the accepted price under CLAUDE.md section 13, which decides on the CUDA number.
    offset = float(twr.min(phi))
    wp.map(wp.sub, phi, wp.float64(offset), out=phi)
    return phi


# --------------------------------------------------------------------------------------
# The signed heat method: signed distance to a set of oriented curves
# --------------------------------------------------------------------------------------


def heat_signed_distance(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    curve_vertices: wp.array[wp.int32],
    curve_offsets: wp.array[wp.int32] | None = None,
    t: float | None = None,
    *,
    closed: bool = True,
    level_set_constraint: str = "zero_set",
    operators: tw.heat.VectorHeatOperators | None = None,
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
        [`vector_heat_operators`][triwarp.heat.vector_heat_operators] for this mesh — the
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

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
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

    if operators is None:
        operators = tw.heat.vector_heat_operators(vertices, faces, t)
    vector_system, scalar, frames = operators
    basis_x, basis_y, vertex_normals = frames
    poisson_system, poisson_preconditioner = scalar[3], scalar[4]
    cot_entries, face_normals = scalar[5], scalar[6]

    # Stage 1: splat each segment's normal onto its endpoints.
    segments = _curve_segments(curve_vertices, curve_offsets, closed=closed)
    source = wp.zeros(n_vertices, dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_heat.splat_curve_normals,
        dim=int(segments.shape[0]),
        inputs=[vertices, segments, vertex_normals, basis_x, basis_y, source],
        device=device,
    )

    # Stage 2: diffuse the tangent field, then keep only its direction.
    diffused = tw.heat.diffuse_tangent_field(vector_system, source)
    unit_field = wp.empty(n_vertices, dtype=wp.vec2d, device=device)
    wp.map(
        kernel_predicates.normalize_or_zero,
        diffused,
        wp.float64(TOLERANCE_ZERO_CONSTANT),
        out=unit_field,
    )

    # Stage 3: integrate the unit field back into a scalar with a Poisson solve. The cotangent
    # weights and face normals come from the same bundle, so the Poisson stage and the diffusion
    # cannot drift apart.
    face_field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat.vertex_field_to_face_field,
        dim=n_faces,
        inputs=[faces, face_normals, unit_field, basis_x, basis_y, face_field],
        device=device,
    )
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.integrated_divergence,
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
        return twt.empty_2d((0, 2), wp.int32, device=curve_vertices.device)
    segments = np.ascontiguousarray(np.concatenate(pairs), dtype=np.int32)
    return twt.as_array2d(
        wp.array(segments, dtype=wp.int32, device=curve_vertices.device), wp.int32
    )


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
        operator, fixed_mask, free_map, twt.as_array2d(zeros, wp.float64), n_free
    )
    # Flip sign with the operator: the Poisson right-hand side is -div for the -L convention.
    negated = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.neg, divergence, out=negated)
    wp.launch(
        kernel_heat.scatter_free_rhs,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, negated, rhs],
        device=device,
    )

    solution = wp.zeros((1, n_free), dtype=wp.float64, device=device)
    twl.solve_spd_columns(operator_uu, rhs, twt.as_array2d(solution, wp.float64), tol=_CG_TOLERANCE)
    field = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.gather_free_solution,
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


# --------------------------------------------------------------------------------------
# The vector heat method: transport, scalar extension and the logarithmic map
# --------------------------------------------------------------------------------------

# A diffused field is treated as having vanished below this fraction of its *own* maximum. Both
# fields these solvers divide by -- the direction field and the source indicator -- carry the mesh's
# scale as ~1/scale^2, so the cutoff has to be relative or the whole answer silently goes to zero on
# a mesh measured in millimetres. It is deliberately far below the round-off floor (~1e-08 of the
# maximum): see ``kernels/heat/vector.py`` for why nothing here can separate noise from signal.
_RELATIVE_ZERO = 1e-12

# Below this fraction of the direction field's maximum, a transported direction cannot be told from
# the round-off the solve leaves where the transported copies cancel, measured at 8.7e-09 of the
# maximum. A decade above that, and the *only* thing it drives is the mask
# ``transport_tangent_vectors`` returns alongside its vectors -- no value is zeroed by it, so
# flagging a marginal vertex costs the caller nothing.
_RESOLVED_FRACTION = 1e-7


VectorHeatOperators = tuple[
    wps.BsrMatrix[wp.float64],
    HeatOperators,
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]],
]
"""What [`vector_heat_operators`][triwarp.heat.vector_heat_operators] returns: the vector
heat system, the scalar [`heat_operators`][triwarp.heat.heat_operators], and the frames."""


def vector_heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    scalar_operators: HeatOperators | None = None,
    frames: tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None = None,
) -> VectorHeatOperators:
    """
    Assemble everything the vector-valued solvers need before their solves.

    Three pieces, none of which depends on a source:

    1. the **vector heat system** ``M + t * L_connection``, whose ``2 x 2`` blocks act on tangent
       vectors ([`connection_laplacian`][triwarp.laplacian.connection_laplacian]);
    2. the scalar [`heat_operators`][triwarp.heat.heat_operators], for the magnitude
       extension and the distance field the log map needs;
    3. the [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] every 2D component
       is measured in.

    Pass the result back through any solver's ``operators=`` argument to skip the assembly — most of
    the cost on a coarse mesh, and all of it when the solve converges quickly. That is the split
    ``potpourri3d.MeshVectorHeatSolver`` gets from being an object; here it stays a plain tuple.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    t
        Diffusion time for both the vector and the scalar systems. When ``None``, defaults to the
        squared mean edge length.
    scalar_operators
        Optional prebuilt [`heat_operators`][triwarp.heat.heat_operators] bundle to place
        in the second field instead of assembling one. **Must have been built at the same ``t``**,
        which nothing here can check: it is the caller's half of the shared-timestep contract the
        Notes below describe. [`Trimesh.heat_operators`][triwarp.mesh.Trimesh.heat_operators]
        caches one at this function's own default.
    frames
        Optional prebuilt [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] as
        ``(basis_x, basis_y, normal)`` -- the gauge, which depends on the mesh alone and not on
        ``t``. [`Trimesh.vertex_tangent_frames`][triwarp.mesh.Trimesh.vertex_tangent_frames] caches
        it.

    Returns
    -------
    vector_system : warp.sparse.BsrMatrix
        ``M + t * L_connection`` in ``float64`` with ``wp.mat22d`` blocks.
    scalar : tuple
        The [`heat_operators`][triwarp.heat.heat_operators] bundle for the same ``t``.
    frames : tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]
        ``(basis_x, basis_y, normal)`` per vertex.

    Notes
    -----
    The vector and scalar systems must share ``t``: [`log_map`][triwarp.heat.log_map]'s
    radius is asserted to *be* the [`heat_geodesic`][triwarp.heat.heat_geodesic] distance,
    so two diffusion times would split a quantity that is supposed to be one number. That is why
    ``scalar_operators`` is the one argument here that cannot be validated.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`log_map`][triwarp.heat.log_map]
    [`heat_signed_distance`][triwarp.heat.heat_signed_distance]
    [`Trimesh.vector_heat_operators`][triwarp.mesh.Trimesh.vector_heat_operators]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if t is None:
        # Shares the scalar solver's timestep convention -- the unique-edge mean, matching
        # ``igl::heat_geodesics``. The two solvers must agree: ``log_map``'s radius is asserted to
        # *be* the ``heat_geodesic`` distance, so giving them different diffusion times would split
        # a quantity that is supposed to be one number.
        h = tw.edges.mean_unique_edge_length(vertices, faces)
        t = h * h

    connection = connection_laplacian(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)
    mass_blocks = wp.empty(n_vertices, dtype=wp.mat22d, device=device)
    wp.map(kernel_heat.block_mass, mass, out=mass_blocks)
    vector_system = wps.bsr_axpy(
        x=connection, y=wps.bsr_diag(diag=mass_blocks), alpha=float(t), beta=1.0
    )
    if scalar_operators is None:
        scalar_operators = heat_operators(vertices, faces, t)
    if frames is None:
        frames = vertex_tangent_frames(vertices, faces)
    return vector_system, scalar_operators, frames


def extend_scalar(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    t: float | None = None,
    operators: tw.heat.HeatOperators | None = None,
) -> wp.array[wp.float64]:
    """
    Extend values from a few source vertices over the whole surface by nearest-source interpolation.

    Diffuses the values and an indicator of where they came from for the same short time, then
    divides one by the other. The ratio is what makes the result interpolate rather than decay: both
    numerator and denominator fall off away from the sources at the same rate, so their quotient
    stays close to the value of the nearest source, and blends smoothly where two sources compete.
    Matches ``potpourri3d.MeshVectorHeatSolver.extend_scalar``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices.
    values
        ``(n_sources,)`` ``wp.float64`` value carried by each source.
    t
        Diffusion time; defaults to the squared mean edge length.
    operators
        Optional precomputed [`heat_operators`][triwarp.heat.heat_operators] for this mesh.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` extended field on ``vertices.device``.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_sources = int(sources.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or n_sources == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    if operators is None:
        operators = heat_operators(vertices, faces, t)
    heat_system, heat_preconditioner = operators[0], operators[1]

    indicator = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    weighted = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values, indicator, weighted],
        device=device,
    )

    diffused_indicator = _solve_scalar(
        heat_system, indicator, n_vertices, device, heat_preconditioner
    )
    diffused_values = _solve_scalar(heat_system, weighted, n_vertices, device, heat_preconditioner)
    # The indicator decays away from the sources *and* carries the mesh's scale, so the "there is no
    # source anywhere near here" cutoff is a fraction of its own maximum. One host readback, as in
    # ``transport_tangent_vectors``.
    floor = wp.float64(_RELATIVE_ZERO * tw.reduce.max(diffused_indicator))
    extended = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_heat.divide_positive, diffused_values, diffused_indicator, floor, out=extended)
    return extended


def _solve_scalar(
    system: wps.BsrMatrix[wp.float64],
    right_hand_side: wp.array[wp.float64],
    n_vertices: int,
    device: wp.DeviceLike,
    preconditioner: wpl.LinearOperator,
) -> wp.array[wp.float64]:
    """Diffuse one scalar right-hand side through an already-assembled heat system."""
    solution = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(
        system, right_hand_side, solution, tol=_CG_TOLERANCE, preconditioner=preconditioner
    )
    return solution


def transport_tangent_vectors(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    t: float | None = None,
    operators: VectorHeatOperators | None = None,
) -> tuple[wp.array[wp.vec2], wp.array[wp.bool]]:
    """
    Parallel-transport tangent vectors from a few source vertices to every vertex.

    Three solves, following the vector heat method: the connection Laplacian diffuses the source
    vectors (which preserves their *directions* well but smears their magnitudes), while a scalar
    extension of the source magnitudes supplies the length. The result at each vertex is the source
    vector carried along the shortest path to it — the field a "drag this arrow across the surface"
    tool needs. Matches ``potpourri3d.MeshVectorHeatSolver.transport_tangent_vectors``, which
    returns the vectors alone.

    The second return exists because the vectors are not self-describing: a zero is ambiguous and a
    *non*-zero one is not always meaningful. A vertex is unresolved when the diffused direction that
    reached it is shorter than ``1e-07`` of the field's maximum — a decade above the round-off left
    where the transported copies cancel, measured at ``8.7e-09`` of the maximum. Below that line a
    direction cannot be told from noise, and the mask says so rather than the value being altered:
    no vector here is changed by it. Three situations it separates:

    * **Nothing reached the vertex** — another connected component, or short-time diffusion
      underflowing. Unresolved, and the vector is zero.
    * **The cut locus** — several shortest paths arrive and their copies cancel. Unresolved, but the
      vector may still have *full length*, pointing wherever the round-off landed. This is the case
      that differs between CPU and CUDA, and the one a caller cannot otherwise detect.
    * **An ordinary vertex.** Resolved.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices.
    vectors
        ``(n_sources,)`` tangent vectors, each in *its own source vertex's* frame.
    t
        Diffusion time; defaults to the squared mean edge length. Ignored when ``operators`` is
        given, which already fixes it.
    operators
        Optional precomputed
        [`vector_heat_operators`][triwarp.heat.vector_heat_operators] for this mesh: they
        depend on the mesh alone, so passing them back skips the assembly on every call after the
        first.

    Returns
    -------
    transported : wp.array[wp.vec2]
        ``(n_vertices,)`` transported vectors, each in that vertex's own frame. No frames need to be
        passed in: the components come out in the canonical frames of
        [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] by construction (see
        [`connection_laplacian`][triwarp.laplacian.connection_laplacian]). A **zero** vector means
        the field vanished there: another connected component, or short-time diffusion underflowing
        before it arrived. The second is the common case on a fine mesh, because the default ``t``
        shrinks with the edge length — 39 461 of 40 962 vertices on a subdivision-6 icosphere, which
        is the method behaving as designed rather than a failure. Pass a larger ``t`` to reach
        further.
    resolved : wp.array[wp.bool]
        ``(n_vertices,)`` — ``True`` where the transported direction carries information.

    See Also
    --------
    [`log_map`][triwarp.heat.log_map]
    [`connection_laplacian`][triwarp.laplacian.connection_laplacian]
    [`tangent_to_world`][triwarp.heat.tangent_to_world]

    Notes
    -----
    !!! note "The direction is undefined on the cut locus"

        Where several shortest paths of equal length arrive, the copies they carry cancel, and what
        is left is round-off rather than a direction. This is not a pathological case: the corner of
        a cube shell diagonally opposite the source receives three copies 120 degrees apart whose
        sum is *exactly* zero, for every source vector and every diffusion time. What comes back is
        then decided by the arithmetic — on CUDA enough round-off survives to be scaled up to full
        length in an arbitrary direction, while on CPU the same point can cancel to exactly zero and
        read as unreached. ``resolved`` is ``False`` on both, and is the only way to tell.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_sources = int(sources.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or n_sources == 0:
        return (
            wp.zeros(n_vertices, dtype=wp.vec2, device=device),
            wp.zeros(n_vertices, dtype=wp.bool, device=device),
        )

    if operators is None:
        operators = vector_heat_operators(vertices, faces, t)
    vector_system, scalar, _ = operators

    direction = _diffuse_from_sources(vector_system, sources, vectors, n_vertices, device)

    magnitudes = wp.empty(n_sources, dtype=wp.float64, device=device)
    wp.map(wp.length, _as_vec2d(vectors), out=magnitudes)
    extended = extend_scalar(vertices, faces, sources, magnitudes, operators=scalar)

    # Both questions below are asked relative to the field, because the field's length carries the
    # mesh's scale. One host readback: ``reduce.max`` returns a Python scalar, ~0.1 ms against the
    # three conjugate-gradient solves this function has already run.
    lengths = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.length, direction, out=lengths)
    maximum = tw.reduce.max(lengths)

    scaled = wp.empty(n_vertices, dtype=wp.vec2d, device=device)
    wp.map(
        kernel_heat.scale_to_magnitude,
        direction,
        extended,
        wp.float64(_RELATIVE_ZERO * maximum),
        out=scaled,
    )
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(kernel_array.to_vec2, scaled, out=transported)

    # Same field and the same relative comparison as ``scale_to_magnitude``' floor above, at a
    # *higher* floor: that one asks "did this vanish?" and must stay below every genuine value,
    # this one asks "can this be told from round-off?" and must stay above it. So a vertex can be
    # reported unresolved while still carrying a full-length vector, which is the cut-locus case.
    resolved = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(kernel_array.greater, lengths, wp.float64(_RESOLVED_FRACTION * maximum), out=resolved)
    return transported, resolved


def log_map(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    source: int,
    t: float | None = None,
    operators: VectorHeatOperators | None = None,
) -> wp.array[wp.vec2]:
    """
    Logarithmic map: every vertex's position in the source vertex's tangent plane.

    ``log_map(...)[v]`` is the 2D point in the *source's* frame whose length is the geodesic
    distance to ``v``, and whose direction is the initial direction of the geodesic that reaches
    ``v``. It is the inverse of the exponential map
    [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex]
    computes, and the standard way to lay out a local coordinate patch around a point.

    Assembled from two fields that are each cheap: the distance to the source
    ([`heat_geodesic`][triwarp.heat.heat_geodesic]) gives the radius, and the source's
    reference direction parallel-transported outwards gives the angle — at any vertex the angle
    between that transported direction and the outward radial direction is exactly the angle at
    which the connecting geodesic left the source, because transport along that geodesic preserves
    it. This is
    the ``VectorHeat`` strategy in ``potpourri3d.MeshVectorHeatSolver.compute_log_map``; its
    ``AffineLocal`` and ``AffineAdaptive`` strategies solve a small dense problem per vertex and are
    deliberately not ported.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    source
        Index of the vertex the map is centred on.
    t
        Diffusion time; defaults to the squared mean edge length. Ignored when ``operators`` is
        given.
    operators
        Optional precomputed
        [`vector_heat_operators`][triwarp.heat.vector_heat_operators]. Pass the same bundle
        used elsewhere when the frames matter: the *angles* this function returns are measured from
        the source's ``basis_x``.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` log-map coordinates in the source vertex's frame; ``(0, 0)`` at the
        source.
        On the cut locus — the antipode of a closed surface, where geodesics from the source arrive
        from every side — there is no direction to report, and the entry keeps the correct magnitude
        with an arbitrary angle.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.vec2, device=device)

    if operators is None:
        operators = vector_heat_operators(vertices, faces, t)
    vector_system, scalar, frames = operators
    basis_x, basis_y, _ = frames

    sources = wp.array([source], dtype=wp.int32, device=device)
    # The source's own reference direction, transported outwards: this is the "which way was x?"
    # field the angle is measured against.
    reference = wp.array([[1.0, 0.0]], dtype=wp.vec2, device=device)
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(
        kernel_array.to_vec2,
        _diffuse_from_sources(vector_system, sources, reference, n_vertices, device),
        out=transported,
    )

    # Radial direction: the unit gradient of the distance field, averaged onto vertices and
    # expressed in each vertex's frame.
    distance = heat_geodesic(vertices, faces, sources, operators=scalar)
    normals, areas = scalar[6], scalar[7]
    n_faces = int(faces.shape[0]) // 3
    face_gradient = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat.face_unit_gradients,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, distance, wp.float64(1.0), face_gradient],
        device=device,
    )
    vertex_gradient = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_heat.scatter_face_field_to_vertices,
        dim=n_faces,
        inputs=[faces, areas, face_gradient, vertex_gradient],
        device=device,
    )
    radial = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(kernel_heat.world_to_tangent_unit, vertex_gradient, basis_x, basis_y, out=radial)

    logarithm = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_heat.log_map_from_angles,
        dim=n_vertices,
        inputs=[radial, transported, distance, logarithm],
        device=device,
    )
    return logarithm


def tangent_to_world(
    tangent: wp.array[wp.vec2], basis_x: wp.array[wp.vec3], basis_y: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Expand per-vertex tangent vectors into 3D using their frames.

    The only way to compare a tangent field against another library's: each library measures 2D
    components from its own reference direction, but ``a * basis_x + b * basis_y`` is the same 3D
    vector either way.

    Parameters
    ----------
    tangent
        ``(n_vertices,)`` tangent vectors in each vertex's frame.
    basis_x, basis_y
        The frames those components refer to, from
        [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames].

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` world-space vectors on ``tangent.device``.

    See Also
    --------
    [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames]
    """
    world = wp.empty(int(tangent.shape[0]), dtype=wp.vec3, device=tangent.device)
    wp.map(kernel_heat.tangent_to_world, tangent, basis_x, basis_y, out=world)
    return world


def _diffuse_from_sources(
    system: wps.BsrMatrix[wp.float64],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.vec2d]:
    """Seed a tangent field at the source vertices, then diffuse it."""
    field = wp.zeros(n_vertices, dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_scatter.scatter_add,
        dim=int(sources.shape[0]),
        inputs=[_as_vec2d(vectors), sources, field],
        device=device,
    )
    return diffuse_tangent_field(system, field)


def diffuse_tangent_field(
    system: wps.BsrMatrix[wp.float64], source: wp.array[wp.vec2d]
) -> wp.array[wp.vec2d]:
    """
    Short-time diffusion of a tangent-vector field: solve ``(M + t L_connection) X = source``.

    Public because the source term is where the vector-valued methods differ from one another — a
    handful of vertices for parallel transport, a whole splatted curve for
    [`heat_signed_distance`][triwarp.heat.heat_signed_distance] — while the solve is the same
    for all of them.

    Only the *directions* of the result carry meaning: magnitudes decay away from the source, and
    every caller replaces them, either with a scalar extension or by normalizing outright.

    Parameters
    ----------
    system
        The vector heat system from
        [`vector_heat_operators`][triwarp.heat.vector_heat_operators].
    source
        ``(n_vertices,)`` ``wp.vec2d`` right-hand side, in each vertex's own tangent frame.

    Returns
    -------
    wp.array[wp.vec2d]
        ``(n_vertices,)`` diffused field on ``source.device``.

    See Also
    --------
    [`vector_heat_operators`][triwarp.heat.vector_heat_operators]
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    """
    n_vertices = int(source.shape[0])
    diffused = wp.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
    if n_vertices == 0:
        return diffused
    twl.solve_spd(system, source, diffused, tol=_CG_TOLERANCE)
    return diffused


def _as_vec2d(vectors: wp.array[wp.vec2]) -> wp.array[wp.vec2d]:
    """Widen a tangent field to float64, the precision the diffusion solves run in."""
    widened = wp.empty(int(vectors.shape[0]), dtype=wp.vec2d, device=vectors.device)
    wp.map(kernel_array.to_vec2d, vectors, out=widened)
    return widened
