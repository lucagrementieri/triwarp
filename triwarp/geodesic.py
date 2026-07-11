"""Geodesic distance on triangle meshes (GPU heat method)."""

from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

from triwarp.edges import mean_edge_length
from triwarp.kernels import geodesic as kernel_geodesic
from triwarp.kernels import laplacian as kernel_laplacian
from triwarp.laplacian import cotmatrix_entries, mass_matrix_entries
from triwarp.triangles import face_normals_and_areas

_CG_TOLERANCE = 1e-8


def heat_geodesic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    t: float | None = None,
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
        ``igl::heat_geodesics`` default), which balances accuracy and smoothing.

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
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mean_edge_length`][triwarp.edges.mean_edge_length]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    if n_vertices == 0 or n_faces == 0 or int(sources.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    # The two linear solves rely on ``warp.optim.linear.cg``, which returns NaN on the CPU device
    # in Warp 1.14-1.15 (even for a trivial well-conditioned system). Require a CUDA device.
    if wp.get_device(device).is_cpu:
        raise NotImplementedError(
            "heat_geodesic requires a CUDA device: warp.optim.linear.cg produces NaN on the CPU "
            "device in Warp 1.14-1.15."
        )

    if t is None:
        h = mean_edge_length(vertices, faces)
        t = h * h

    # Per-face half-cotangent weights (float32, O(1) and safe) reused for both the Laplacian and
    # the divergence. The cotangent stiffness follows the igl convention (negative diagonal, so
    # ``-L`` is positive semi-definite) but is assembled here in float64.
    cot_entries = cotmatrix_entries(vertices, faces)
    n_triplets = 12 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=wp.float64, device=device)
    # The generic ``cotmatrix_triplets`` casts the shared float32 half-cotangent weights to
    # float64, assembling the operator natively in a single build (see issue_report.md).
    wp.launch(
        kernel_laplacian.cotmatrix_triplets,
        dim=n_faces,
        inputs=[faces, cot_entries, rows, cols, vals],
        device=device,
    )
    laplacian = wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )

    # Face normals / areas (float32) for the gradient below; the lumped mass is built natively in
    # float64 by ``mass_matrix_entries``.
    normals, areas = face_normals_and_areas(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)

    # Heat solve: (M - t L) u = u0, with u0 the source indicator. ``bsr_axpy`` overwrites the mass
    # matrix in place (no longer needed) to form the system.
    u0 = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_geodesic.seed_source_indicator,
        dim=int(sources.shape[0]),
        inputs=[sources, u0],
        device=device,
    )
    mass_diag = wps.bsr_diag(diag=mass)
    heat_system = wps.bsr_axpy(x=laplacian, y=mass_diag, alpha=-float(t), beta=1.0)

    heat = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wpl.cg(
        heat_system,
        u0,
        heat,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_vertices,
        M=wpl.preconditioner(heat_system, "diag"),
    )

    # Unit vector field X = -grad(u)/|grad(u)|.
    field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_geodesic.face_gradient_normalized,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, field],
        device=device,
    )

    # Integrated divergence b = div(X), then Poisson solve L phi = b, i.e. (-L) phi = -b with the
    # positive semi-definite operator.
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_geodesic.integrated_divergence,
        dim=n_faces,
        inputs=[vertices, faces, cot_entries, field, divergence],
        device=device,
    )
    poisson_system = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    neg_divergence = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_geodesic.negate_field,
        dim=n_vertices,
        inputs=[divergence, neg_divergence],
        device=device,
    )

    phi = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wpl.cg(
        poisson_system,
        neg_divergence,
        phi,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_vertices,
        M=wpl.preconditioner(poisson_system, "diag"),
    )

    # Shift so the distance field is zero at the (nearest) source. For a correctly signed field
    # the global minimum sits at the source set, so subtracting it yields a nonnegative field.
    offset = float(phi.numpy().min())
    wp.launch(kernel_geodesic.shift_field, dim=n_vertices, inputs=[offset, phi], device=device)
    return phi
