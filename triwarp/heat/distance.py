"""Geodesic distance on triangle meshes via the heat method."""

from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp.linalg as twl
import triwarp.reduce as twr
from triwarp._device import require_cuda
from triwarp.edges import mean_unique_edge_length
from triwarp.kernels.heat import distance as kernel_heat_distance
from triwarp.laplacian import (
    cotmatrix,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
    mollify_intrinsic,
)
from triwarp.triangles import face_normals_and_areas

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
"""What [`heat_operators`][triwarp.heat.distance.heat_operators] returns for the heat method's
solves."""


def heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    use_robust: bool = False,
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
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] used to rebuild all three on every call.
    Measured on ``sphere_small`` (2 562 vertices), that was 0.33 ms for the ``bsr_axpy`` and 0.12 ms
    per preconditioner out of a 6.29 ms amortized call, and the same on ``sphere_med`` where the
    call is 12.57 ms: about 10 % and 5 % respectively. It is *only* those three — the solver
    **state** is deliberately not cached, because a ``warp.optim.linear`` state captures its
    right-hand-side and solution buffers at construction, which would make these operators
    stateful and unsafe to share between two concurrent solves.

    See Also
    --------
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    """
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
    else:
        cot_entries = cotmatrix_entries(vertices, faces)
    # ``cotmatrix`` casts the shared float32 half-cotangent weights to float64 and assembles the
    # operator natively in a single build (see issue_report.md).
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
        Optional precomputed [`heat_operators`][triwarp.heat.distance.heat_operators] for this mesh.
        They depend on the mesh only, so passing them back skips the assembly on every solve after
        the first — worth it when computing distance from many different source sets.
    use_robust
        Forwarded to [`heat_operators`][triwarp.heat.distance.heat_operators]: build the Laplacian
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
    NotImplementedError
        If a non-trivial solve is required on the CPU device. The two conjugate-gradient solves use
        ``warp.optim.linear.cg``, which returns NaN on the CPU device in Warp 1.14-1.15; a CUDA
        device is required. (Empty meshes or empty source sets return a zero field without solving
        and are allowed on any device.)

    See Also
    --------
    [`heat_operators`][triwarp.heat.distance.heat_operators]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length]
    [`marching_triangles`][triwarp.intersection.marching_triangles]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    if n_vertices == 0 or n_faces == 0 or int(sources.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    # Both stages are conjugate-gradient solves, which Warp cannot run on the CPU.
    require_cuda(device, "heat_geodesic")

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
        kernel_heat_distance.seed_source_indicator,
        dim=int(sources.shape[0]),
        inputs=[sources, u0],
        device=device,
    )

    heat = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(heat_system, u0, heat, tol=_CG_TOLERANCE, preconditioner=heat_preconditioner)

    # Unit vector field X = -grad(u)/|grad(u)|.
    field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat_distance.face_gradient_normalized,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, field],
        device=device,
    )

    # Integrated divergence b = div(X), then Poisson solve L phi = b, i.e. (-L) phi = -b with the
    # positive semi-definite operator.
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat_distance.integrated_divergence,
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
