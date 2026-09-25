"""
Heat-diffusion methods on triangle meshes.

Three solvers that share one idea: diffuse a quantity over the surface for a short time ``t``, then
recover the answer from the *direction* of the resulting field rather than its magnitude. Short-time
heat flow approximates the geodesic kernel, so a single sparse solve carries information a
combinatorial shortest-path search would have to walk edge by edge. Each solver differs only in what
it diffuses and how it reads the result back:

- **Geodesic distance.** [`heat_geodesic`][triwarp.heat.heat_geodesic] diffuses a scalar indicator
  from source vertices, normalizes its gradient, and integrates that unit field back with a Poisson
  solve. The heat method of Crane et al. (``igl::heat_geodesics``,
  ``potpourri3d.MeshHeatMethodDistanceSolver``).
- **Signed distance to curves.** [`heat_signed_distance`][triwarp.heat.heat_signed_distance]
  diffuses the normals of a set of oriented curves, then solves a Poisson problem against that field
  to get a *signed* distance whose zero set is the curves (``potpourri3d.MeshSignedHeatSolver``).
- **Vector-valued transport.**
  [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors],
  [`extend_scalar`][triwarp.heat.extend_scalar] and [`log_map`][triwarp.heat.log_map] diffuse
  *tangent vectors* through the connection Laplacian, which transports each vector into its
  neighbour's frame before differencing (``potpourri3d.MeshVectorHeatSolver``).

The three are one module because two of them cannot be separated: the vector solvers import
[`heat_operators`][triwarp.heat.heat_operators] and [`heat_geodesic`][triwarp.heat.heat_geodesic],
[`VectorHeatOperators`][triwarp.heat.VectorHeatOperators] embeds the scalar method's operator tuple,
and [`log_map`][triwarp.heat.log_map]'s radius *is* ``heat_geodesic``'s answer.

All three run in ``float64``, because the diffused field decays exponentially and underflows
``float32``. They run on either device.

The operators these solvers assemble are **not** here -- the cotangent and connection Laplacians
live in [`triwarp.laplacian`][triwarp.laplacian], tangent frames in
[`triwarp.tangent_space`][triwarp.tangent_space], and the batched conjugate-gradient machinery in
[`triwarp.linalg`][triwarp.linalg]. This module is the three algorithms only.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.reduce as twr
import triwarp.typing as twt
from triwarp._device import require_same_device
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import heat as kernel_heat
from triwarp.kernels import predicates as kernel_predicates
from triwarp.kernels import reduce as kernel_reduce
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

# Rounds per chunk of a heat solve that must reach every vertex, and the per-vertex relative change
# below which a chunk counts as converged (see ``_diffuse``). The change collapses by six or more
# orders of magnitude in the chunk after the field settles -- measured on spheres, a hemisphere and
# the two bunnies, from about 1 to 1e-7 or less -- so the threshold is not a tuning knob.
_HEAT_CHUNK = 64
_HEAT_CHANGE_TOLERANCE = 1e-6

# A bound for the entries that never settle relative to themselves (see ``_diffuse``): the solve
# stops this many chunks after it reached every vertex. The number of rounds a field needs past
# full reach does not grow with the mesh -- it is the heat system's own conditioning, fixed by
# ``t = h ** 2`` -- and measured at 80-150 on spheres from 2.5 k to 41 k vertices, a hemisphere and
# both bunnies, so three chunks (192 rounds) covers every one.
_HEAT_SETTLE_CHUNKS = 3


HeatOperators = tuple[
    wps.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    wps.BsrMatrix[wp.float64],
    wps.BsrMatrix[wp.float64],
    wpl.LinearOperator,
    twt.Array2dFloat32,
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
        Jacobi-Chebyshev polynomial preconditioner for ``poisson_system``
        ([`chebyshev_preconditioner`][triwarp.linalg.chebyshev_preconditioner]), built on its first
        apply.
    cot_entries : twt.Array2dFloat32
        ``(n_faces, 3)`` per-face half-cotangent weights, reused by the divergence. Always
        ``float32`` regardless of the ``cot_entries`` precision passed in or built internally.
    face_normals : wp.array[wp.vec3]
        One unit normal per face.
    face_areas : wp.array[wp.float32]
        One area per face.

    Notes
    -----
    The Poisson operator and both preconditioners are here because they satisfy this
    function's own contract — they depend on the mesh alone — so
    [`heat_geodesic`][triwarp.heat.heat_geodesic] need not rebuild all three on every call, which
    is unnecessary work whenever the mesh is unchanged across several solves. The solver
    **state** is not in the tuple: [`solve_spd`][triwarp.linalg.solve_spd] keeps one per operator
    itself, owning its own buffers, so passing these operators back also replays the solves'
    recorded loops rather than recording them again -- and, as with the polynomial's vectors below,
    two solves against one operator must run on one stream. The remaining lever for repeated calls
    is conditioning. The Poisson solve is the long one -- ``-L`` is the
    ill-conditioned operator here, where the heat system's mass term keeps it close to diagonal --
    so it takes the polynomial preconditioner and the heat system keeps Jacobi, which a
    well-conditioned solve of a few tens of iterations cannot beat. The polynomial holds working
    vectors of its own, so two solves against one ``poisson_preconditioner`` must run on one
    stream, as every caller here does.

    !!! note
        ``solve_spd`` **warm-starts from whatever the solution buffer already holds**, so reusing
        one buffer across unrelated right-hand sides carries over the previous solution as the
        initial guess. And this tuple's *third* field is the raw (singular) Laplacian, not the
        Poisson system — handing it a right-hand side runs CG to its iteration cap.

    Raises
    ------
    ValueError
        If both ``cot_entries`` and ``use_robust`` are given: ``use_robust`` exists to build that
        very table from mollified edge lengths, so the two ask for different weights.
    RuntimeError
        If ``vertices``, ``faces`` and ``cot_entries`` are not all on one device.

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    [`Trimesh.heat_operators`][triwarp.mesh.Trimesh.heat_operators]
    """
    require_same_device(vertices=vertices, faces=faces, cot_entries=cot_entries)
    if cot_entries is not None and use_robust:
        raise ValueError(
            "cot_entries and use_robust are mutually exclusive: use_robust rebuilds the "
            "half-cotangent table from mollified edge lengths."
        )
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
    if t is None:
        # The unique-edge average, which is what ``igl::heat_geodesics`` uses for its timestep --
        # read off the Laplacian's own sparsity, which already holds the unique edges.
        h = _mean_edge_length(vertices, laplacian)
        t = h * h

    # The divergence kernel this bundle feeds (``kernels/heat.py::unit_gradient_divergence``) is
    # hardcoded ``wp.array2d[wp.float32]`` -- ``cotmatrix`` above accepts either precision because
    # it casts internally, but this tuple's own ``cot_entries`` field is documented and used
    # downstream as float32 only, so a caller-supplied float64 table must be narrowed before it is
    # returned rather than passed through at whatever precision it arrived in.
    if cot_entries.dtype is not wp.float32:
        narrowed_cot_entries = twt.empty_2d(
            (int(cot_entries.shape[0]), int(cot_entries.shape[1])),
            wp.float32,
            device=vertices.device,
        )
        wp.utils.array_cast(cot_entries.flatten(), narrowed_cot_entries.flatten())
        cot_entries = narrowed_cot_entries

    # Face normals / areas (float32) for the gradient; the lumped mass is built natively in float64
    # by ``mass_matrix_entries``.
    normals, areas = face_normals_and_areas(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)

    # Heat system (M - t L). ``bsr_axpy`` overwrites the mass matrix in place (no longer needed).
    mass_diag = wps.bsr_diag(diag=mass)
    heat_system = cast(
        "wps.BsrMatrix[wp.float64]",
        wps.bsr_axpy(x=laplacian, y=mass_diag, alpha=-float(t), beta=1.0),
    )
    # Poisson operator ``-L`` and the two preconditioners: mesh-only, so they belong here rather
    # than in every solve. See Notes.
    poisson_system = cast("wps.BsrMatrix[wp.float64]", wps.bsr_axpy(x=laplacian, alpha=-1.0))
    return (
        heat_system,
        twl.jacobi_preconditioner(heat_system),
        laplacian,
        poisson_system,
        twl.chebyshev_preconditioner(poisson_system),
        # Narrowed to float32 by the block above, whichever precision it arrived in.
        cast(twt.Array2dFloat32, cot_entries),
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

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``sources`` are not all on one device.

    See Also
    --------
    [`heat_operators`][triwarp.heat.heat_operators]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length]
    [`marching_triangles`][triwarp.intersection.marching_triangles]
    """
    require_same_device(vertices=vertices, faces=faces, sources=sources, operators=operators)
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

    # Heat solve: (M - t L) u = u0, with u0 the source indicator, run until every vertex's heat has
    # converged relative to its own size (``_diffuse``) -- not to a residual tolerance, which the
    # far field sits hundreds of orders of magnitude below. Neumann on a boundary, as
    # geometry-central (``potpourri3d``) and MeshLab take it. ``igl::heat_geodesics_solve``
    # averages it with the solution pinned to zero on the boundary instead, and against exact
    # polyhedral geodesics (``igl.exact_geodesic``) that average is the less accurate of the two:
    # equal on a hemisphere, and 1.15 % against 0.93 % mean error (4.9 % against 3.1 % worst) of
    # the distance range on a half torus.
    u0 = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.seed_source_indicator,
        dim=int(sources.shape[0]),
        inputs=[sources, u0],
        device=device,
    )

    heat = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    _diffuse(heat_system, u0, heat, heat_preconditioner)

    # Integrated divergence b = div(X) of the unit field X = -grad(u)/|grad(u)|, then a Poisson
    # solve L phi = b, i.e. (-L) phi = -b with the positive semi-definite operator. One launch: the
    # field is formed and integrated per face, so it never occupies an (n_faces,) buffer. The kernel
    # integrates ``-X`` and so accumulates ``-b`` directly: the divergence is linear in the field
    # and negation is exact, so this is the negated sum without a pass to negate it.
    neg_divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.unit_gradient_divergence,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, cot_entries, neg_divergence],
        device=device,
    )

    phi = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(
        poisson_system,
        neg_divergence,
        phi,
        tol=_CG_TOLERANCE,
        preconditioner=poisson_preconditioner,
    )

    # Shift so the field's mean over the sources is zero, and orient it positive -- the
    # ``igl::heat_geodesics_solve`` convention, which makes a single source's distance exactly
    # zero. One gathered mean and one global mean, both reductions on the device.
    offset = float(twr.mean(tw.array.gather(phi, sources)))
    wp.map(wp.sub, phi, wp.float64(offset), out=phi)
    if float(twr.mean(cast(twt.ArrayNdFloat64, phi))) < 0.0:
        wp.map(wp.neg, phi, out=phi)
    return phi


def _diffuse(
    system: wps.BsrMatrix[Any],
    rhs: wp.array[Any],
    solution: wp.array[Any],
    preconditioner: wpl.LinearOperator | None,
) -> None:
    """
    Solve a heat system until every vertex's value has converged relative to its own size.

    The heat method's diffused sources fall by a near-constant factor per ring of vertices, so the
    far field is hundreds of orders of magnitude below the peak -- ``float64`` holds it, and igl's
    direct factorization resolves it -- and a conjugate gradient stopped on its residual leaves it
    as noise: an iterate after ``k`` rounds is a degree-``k`` polynomial in the operator applied to
    the sources, so it is exactly zero more than ``k`` rings away, and the residual converges long
    before ``k`` reaches the far side of the mesh. Measured on a unit ``icosphere(5)``, the old
    residual-stopped solve left 92 % of the vertices without heat and put ``heat_geodesic`` up to
    2.3 off the great-circle distance.

    So the solve runs in warm-restarted chunks of ``_HEAT_CHUNK`` rounds at a zero tolerance, and
    after each chunk ``kernels/heat.heat_chunk_change`` measures, over every entry of ``solution``,
    the largest relative change and the number of entries reached. It stops once no entry is newly
    reached and none moved by more than ``_HEAT_CHANGE_TOLERANCE`` -- or, for an entry that never
    settles because it cancels toward zero (a transported vector on the cut locus, whose value is
    round-off relative to itself), ``_HEAT_SETTLE_CHUNKS`` chunks after the last vertex was
    reached. One readback a chunk.

    Jacobi rather than the Jacobi-Chebyshev polynomial, although a polynomial round reaches a dozen
    rings where a Jacobi round reaches one: measured 1.3-1.8x faster on the sphere and **wrong** on
    ``bunny`` -- the distance 0.9 of its range off igl's and the scalar extension divergent -- where
    obtuse triangles make the heat system's off-diagonal entries positive.

    ``system`` is a scalar operator with ``float64`` right-hand sides, a ``wp.mat22d`` one with
    ``wp.vec2d`` ones, or a scalar one with ``(n_columns, n)`` ones.
    """
    device = rhs.device
    flat = _as_flat_float64(solution)
    n = int(flat.shape[0])
    previous = wp.zeros(n, dtype=wp.float64, device=device)
    stats = wp.zeros(2, dtype=wp.float64, device=device)
    rows = int(solution.shape[-1]) if solution.ndim == 2 else int(solution.shape[0])
    reached = -1.0
    reach_chunks = 0
    for chunk in range(1, max(2, (twl.CG_MAXITER_FACTOR * rows) // _HEAT_CHUNK) + 1):
        if solution.ndim == 2:
            twl.solve_spd_columns(
                system, rhs, solution, tol=0.0, maxiter=_HEAT_CHUNK, check_every=0
            )
        else:
            twl.solve_spd(
                system,
                rhs,
                solution,
                tol=0.0,
                maxiter=_HEAT_CHUNK,
                check_every=0,
                preconditioner=preconditioner,
            )
        stats.zero_()
        wp.launch_tiled(
            kernel_heat.heat_chunk_change,
            dim=[kernel_reduce.blocks_1d(n)],
            inputs=[flat, previous],
            outputs=[stats],
            block_dim=TILE_1D,
            device=device,
        )
        change, now_reached = (float(x) for x in stats.numpy())
        if now_reached != reached:
            reached, reach_chunks = now_reached, chunk
            continue
        settled = chunk - reach_chunks >= _HEAT_SETTLE_CHUNKS
        if change < _HEAT_CHANGE_TOLERANCE or settled:
            return


def _as_flat_float64(values: wp.array[Any]) -> wp.array[wp.float64]:
    """View a contiguous scalar, ``wp.vec2d`` or rank-2 ``float64`` array as flat ``float64``."""
    if values.dtype == wp.vec2d:
        return values.view(wp.float64).flatten()
    return values.flatten()


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
    RuntimeError
        If ``vertices``, ``faces``, ``curve_vertices`` and ``curve_offsets`` are not all on one
        device.

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`homology_generators`][triwarp.homology.homology_generators]
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    """
    require_same_device(
        vertices=vertices, faces=faces, curve_vertices=curve_vertices, curve_offsets=curve_offsets
    )
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
    vector_system, scalar, frames, vector_preconditioner = operators
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

    # Stage 2: diffuse the tangent field, then keep only its direction. The floor below which a
    # vector is treated as vanished has to be relative to the field's own maximum, not the fixed
    # ``TOLERANCE_ZERO_CONSTANT`` -- this field carries the mesh's scale exactly like the vector
    # heat method's direction field below, and an absolute floor zeroes most of it on a mesh not
    # near unit scale (confirmed: 111 of 162 vertices at a 1e-6 scale, all resolved relative to the
    # field's own maximum).
    diffused = tw.heat.diffuse_tangent_field(
        vector_system, source, preconditioner=vector_preconditioner
    )
    # Normalized without underflow and zero only where the field is exactly zero: it is converged
    # per vertex (``_diffuse``), so its far field is a direction however small.
    unit_field = wp.empty(n_vertices, dtype=wp.vec2d, device=device)
    wp.map(kernel_predicates.stable_normalize, diffused, out=unit_field)

    # Stage 3: integrate the unit field back into a scalar with a Poisson solve. The cotangent
    # weights and face normals come from the same bundle, so the Poisson stage and the diffusion
    # cannot drift apart.
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat.vertex_field_divergence,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            face_normals,
            unit_field,
            basis_x,
            basis_y,
            cot_entries,
            divergence,
        ],
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
    # Flip sign with the operator: the Poisson right-hand side is -div for the -L convention; the
    # compaction negates as it goes.
    wp.launch(
        kernel_heat.scatter_negated_free_rhs,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, divergence, rhs],
        device=device,
    )

    solution = wp.zeros((1, n_free), dtype=wp.float64, device=device)
    # The same Poisson operator as the unpinned solve, less the curve's rows, so the same
    # polynomial preconditioner; see ``heat_operators``' Notes.
    twl.solve_spd_columns(
        operator_uu,
        rhs,
        twt.as_array2d(solution, wp.float64),
        tol=_CG_TOLERANCE,
        preconditioner="chebyshev",
    )
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
    # In place: ``divergence`` is this call's own scratch, read by nothing after the solve.
    wp.map(wp.neg, divergence, out=divergence)

    field = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(operator, divergence, field, tol=_CG_TOLERANCE, preconditioner=preconditioner)
    offset = tw.reduce.mean(tw.array.gather(field, curve_vertices))
    wp.map(wp.sub, field, wp.float64(offset), out=field)
    return field


# --------------------------------------------------------------------------------------
# The vector heat method: transport, scalar extension and the logarithmic map
# --------------------------------------------------------------------------------------

# Below this fraction of the diffused *magnitudes at the same vertex*, a transported direction
# cannot be told from the round-off left where the transported copies cancel (the cut locus). The
# ratio is 1 where the copies agree -- a vector's length never exceeds the heat of its magnitude --
# so the test is local and scale-free, and a far vertex is resolved however small its field.
# Round-off at an exact cancellation measured 1.4e-7 of the local magnitude (``cave_cube``'s
# antipodal corner: the transport angles are ``float32``), genuine directions 0.36 and more, so the
# threshold sits
# three decades above the round-off. The *only* thing it drives is the mask
# ``transport_tangent_vectors`` returns alongside its vectors -- no value is zeroed by it, so
# flagging a marginal vertex costs the caller nothing.
_RESOLVED_FRACTION = 1e-4


VectorHeatOperators = tuple[
    wps.BsrMatrix[wp.mat22d],
    HeatOperators,
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]],
    wpl.LinearOperator,
]
"""What [`vector_heat_operators`][triwarp.heat.vector_heat_operators] returns: the vector
heat system, the scalar [`heat_operators`][triwarp.heat.heat_operators], the frames, and the
vector system's own Jacobi preconditioner."""


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

    Four pieces, none of which depends on a source:

    1. the **vector heat system** ``M + t * L_connection``, whose ``2 x 2`` blocks act on tangent
       vectors ([`connection_laplacian`][triwarp.laplacian.connection_laplacian]);
    2. the scalar [`heat_operators`][triwarp.heat.heat_operators], for the magnitude
       extension and the distance field the log map needs;
    3. the [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] every 2D component
       is measured in;
    4. the vector system's own Jacobi preconditioner.

    Pass the result back through any solver's ``operators=`` argument to skip the assembly of all
    four on every call after the first. That is the split ``potpourri3d.MeshVectorHeatSolver`` gets
    from being an object; here it stays a plain tuple.
    [`diffuse_tangent_field`][triwarp.heat.diffuse_tangent_field] threads the fourth piece into
    [`solve_spd`][triwarp.linalg.solve_spd]'s own ``preconditioner=``, so a caller running many
    transports or log maps against one mesh pays for it once rather than on
    every diffusion solve.

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
    preconditioner : ``warp.optim.linear.LinearOperator``
        Jacobi preconditioner for ``vector_system``.

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces`` and ``frames`` are not all on one device.

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
    require_same_device(vertices=vertices, faces=faces, frames=frames)
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    connection = connection_laplacian(vertices, faces)
    if t is None:
        # Shares the scalar solver's timestep convention -- the unique-edge mean, matching
        # ``igl::heat_geodesics``. The two solvers must agree: ``log_map``'s radius is asserted to
        # *be* the ``heat_geodesic`` distance, so giving them different diffusion times would split
        # a quantity that is supposed to be one number. They do by construction: both read the
        # mean off their operator's sparsity, and the connection Laplacian's is the cotangent
        # Laplacian's, twelve triplets per face and nothing pruned.
        h = _mean_edge_length(vertices, connection)
        t = h * h

    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)
    mass_blocks = wp.empty(n_vertices, dtype=wp.mat22d, device=device)
    wp.map(kernel_heat.block_mass, mass, out=mass_blocks)
    vector_system = cast(
        "wps.BsrMatrix[wp.mat22d]",
        wps.bsr_axpy(x=connection, y=wps.bsr_diag(diag=mass_blocks), alpha=float(t), beta=1.0),
    )
    if scalar_operators is None:
        scalar_operators = heat_operators(vertices, faces, t)
    if frames is None:
        frames = vertex_tangent_frames(vertices, faces)
    preconditioner = twl.jacobi_preconditioner(vector_system)
    return vector_system, scalar_operators, frames, preconditioner


def _mean_edge_length(vertices: wp.array[wp.vec3], operator: wps.BsrMatrix[Any]) -> float:
    """
    Mean unique-edge length, read off an operator with one entry per edge.

    The strict upper triangle of the heat method's Laplacians is the mesh's unique edge set, so the
    timestep costs one launch and one readback over an operator that already exists, where
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] would re-sort every edge of
    the mesh to recover the same set. ``0`` for a mesh with no edges, as that function returns.
    """
    device = vertices.device
    n_rows = int(operator.nrow)
    if n_rows == 0:
        return 0.0
    sum_and_count = wp.zeros(2, dtype=wp.float64, device=device)
    wp.launch_tiled(
        kernel_heat.upper_edge_length_sum_and_count,
        dim=[kernel_reduce.blocks_1d(n_rows)],
        inputs=[operator.offsets, operator.columns, vertices, sum_and_count],
        block_dim=TILE_1D,
        device=device,
    )
    total, count = (float(x) for x in sum_and_count.numpy())
    return total / count if count > 0.0 else 0.0


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

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``sources`` and ``values`` are not all on one device.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    """
    require_same_device(vertices=vertices, faces=faces, sources=sources, values=values)
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_sources = int(sources.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or n_sources == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    if operators is None:
        operators = heat_operators(vertices, faces, t)
    return _extend(operators[0], sources, values, n_vertices, device)[0]


def _extend(
    heat_system: wps.BsrMatrix[wp.float64],
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    n_vertices: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.float64], wp.array[wp.float64]]:
    """``extend_scalar``'s body, also returning the diffused ``values`` it divides."""
    n_sources = int(sources.shape[0])
    # The indicator and the weighted values diffuse through the same operator, so they are one
    # batched two-column solve (``linalg.solve_spd_columns``) rather than two independent ones,
    # which shares the launches and converges on the worse-behaved of the two columns.
    rhs = twt.as_array2d(wp.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64)
    wp.launch(
        kernel_heat.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values, rhs[0], rhs[1]],
        device=device,
    )
    diffused = twt.as_array2d(
        wp.zeros((2, n_vertices), dtype=wp.float64, device=device), wp.float64
    )
    _diffuse(heat_system, rhs, diffused, None)
    diffused_indicator, diffused_values = twt.as_dense(diffused[0]), twt.as_dense(diffused[1])

    # Converged per vertex, so only an exactly zero indicator -- a component no source reaches --
    # has no value to extend (``divide_nonzero``).
    extended = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_heat.divide_nonzero, diffused_values, diffused_indicator, out=extended)
    return extended, diffused_values


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
    vector carried along the shortest path to it. Matches
    ``potpourri3d.MeshVectorHeatSolver.transport_tangent_vectors``, which returns the vectors alone.

    The second return exists because the vectors are not self-describing: a zero is ambiguous and a
    *non*-zero one is not always meaningful. A vertex is unresolved when the diffused direction that
    reached it is shorter than ``1e-07`` of the field's maximum -- about a decade above the
    round-off left where the transported copies cancel. Below that line a direction cannot be told
    from noise, and the mask says so rather than the value being altered: no vector here is changed
    by it. Three situations it separates:

    * **Nothing reached the vertex** -- another connected component, or short-time diffusion
      underflowing. Unresolved, and the vector is zero.
    * **The cut locus** -- several shortest paths arrive and their copies cancel. Unresolved, but
      the
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

    Raises
    ------
    RuntimeError
        If ``vertices``, ``faces``, ``sources`` and ``vectors`` are not all on one device.

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
        then decided by the arithmetic -- on CUDA enough round-off survives to be scaled up to full
        length in an arbitrary direction, while on CPU the same point can cancel to exactly zero and
        read as unreached. ``resolved`` is ``False`` on both, and is the only way to tell.
    """
    require_same_device(vertices=vertices, faces=faces, sources=sources, vectors=vectors)
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
    vector_system, scalar, _, vector_preconditioner = operators

    # Widened once: the seed and the per-source magnitudes both read the float64 vectors.
    vectors_d = _as_vec2d(vectors)
    direction = _diffuse_from_sources(
        vector_system, sources, vectors_d, n_vertices, device, preconditioner=vector_preconditioner
    )

    magnitudes = wp.empty(n_sources, dtype=wp.float64, device=device)
    wp.map(wp.length, vectors_d, out=magnitudes)
    extended, diffused_magnitudes = _extend(scalar[0], sources, magnitudes, n_vertices, device)

    # One map for the whole tail: the rescale, the narrowing to the field's storage precision and
    # the resolution test all read one vertex's own data, so running them apart costs two extra
    # launches and a full round trip of the rescaled float64 field. Resolution is asked locally,
    # against the diffused magnitudes at the same vertex -- see the kernel func.
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    resolved = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(
        kernel_heat.transported_and_resolved,
        direction,
        extended,
        diffused_magnitudes,
        wp.float64(_RESOLVED_FRACTION),
        out=[transported, resolved],
    )
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

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors]
    [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex]
    """
    require_same_device(vertices=vertices, faces=faces)
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.vec2, device=device)

    if operators is None:
        operators = vector_heat_operators(vertices, faces, t)
    vector_system, scalar, frames, vector_preconditioner = operators
    basis_x, basis_y, _ = frames

    sources = wp.array([source], dtype=wp.int32, device=device)
    # The source's own reference direction, transported outwards: this is the "which way was x?"
    # field the angle is measured against. Raw (unnormalized) magnitude, exactly like
    # ``transport_tangent_vectors``' own ``direction`` -- so the cut-locus test below has to floor
    # it relative to its own maximum for the same reason that function does (§ its docstring).
    reference = wp.array([[1.0, 0.0]], dtype=wp.vec2d, device=device)
    transported_raw = _diffuse_from_sources(
        vector_system, sources, reference, n_vertices, device, preconditioner=vector_preconditioner
    )
    # Normalized in ``float64`` before it is narrowed, which the far field needs to survive.
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(kernel_heat.narrow_direction, transported_raw, out=transported)

    # Radial direction: the unit gradient of the distance field, averaged onto vertices and
    # expressed in each vertex's frame.
    distance = heat_geodesic(vertices, faces, sources, operators=scalar)
    normals, areas = scalar[6], scalar[7]
    n_faces = int(faces.shape[0]) // 3
    vertex_gradient = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_heat.scatter_unit_gradient_to_vertices,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, distance, vertex_gradient],
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

    Raises
    ------
    RuntimeError
        If ``tangent``, ``basis_x`` and ``basis_y`` are not all on one device.

    See Also
    --------
    [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames]
    """
    require_same_device(tangent=tangent, basis_x=basis_x, basis_y=basis_y)
    world = wp.empty(int(tangent.shape[0]), dtype=wp.vec3, device=tangent.device)
    wp.map(kernel_heat.tangent_to_world, tangent, basis_x, basis_y, out=world)
    return world


def _diffuse_from_sources(
    system: wps.BsrMatrix[wp.float64],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2d],
    n_vertices: int,
    device: wp.DeviceLike,
    *,
    preconditioner: wpl.LinearOperator | None = None,
) -> wp.array[wp.vec2d]:
    """Seed a tangent field at the source vertices, then diffuse it."""
    field = wp.zeros(n_vertices, dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_scatter.SCATTER_ADD[wp.vec2d],
        dim=int(sources.shape[0]),
        inputs=[vectors, sources, field],
        device=device,
    )
    return diffuse_tangent_field(system, field, preconditioner=preconditioner)


def diffuse_tangent_field(
    system: wps.BsrMatrix[wp.float64],
    source: wp.array[wp.vec2d],
    *,
    preconditioner: wpl.LinearOperator | None = None,
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
    preconditioner
        Optional Jacobi preconditioner for ``system``, the fourth field of
        [`vector_heat_operators`][triwarp.heat.vector_heat_operators]'s return. Built here when
        ``None``, which is the cost passing it back through ``operators=`` skips.

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
    _diffuse(system, source, diffused, preconditioner)
    return diffused


def _as_vec2d(vectors: wp.array[wp.vec2]) -> wp.array[wp.vec2d]:
    """Widen a tangent field to float64, the precision the diffusion solves run in."""
    widened = wp.empty(int(vectors.shape[0]), dtype=wp.vec2d, device=vectors.device)
    wp.map(kernel_array.to_vec2d, vectors, out=widened)
    return widened
