"""
Laplacian smoothing filters (Warp), for vertex positions and for per-vertex scalar fields.

Most of the module moves *geometry*: [`filter_laplacian`][triwarp.smoothing.filter_laplacian],
[`filter_taubin`][triwarp.smoothing.filter_taubin],
[`filter_humphrey`][triwarp.smoothing.filter_humphrey],
[`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average],
[`filter_mut_dif_laplacian`][triwarp.smoothing.filter_mut_dif_laplacian] and
[`filter_implicit_fairing`][triwarp.smoothing.filter_implicit_fairing] all diffuse vertex positions
through the same row-stochastic 1-ring operator, differing in the time integration and in what they
do to counteract shrinkage. [`position_verts_smoothly`][triwarp.smoothing.position_verts_smoothly]
and its sharp-boundary variant instead solve a Dirichlet problem over a *region*, holding the rest
of the mesh fixed.

Three functions break that pattern by working on the *normal* field instead of positions, which is
what lets them keep a crease sharp: [`filter_normals`][triwarp.smoothing.filter_normals] diffuses
face normals with a crease gate, [`filter_two_step`][triwarp.smoothing.filter_two_step] then refits
the vertices to them, and [`filter_unsharp_mask`][triwarp.smoothing.filter_unsharp_mask] runs the
whole idea backwards to *sharpen*.

The last two functions run the same operator over a per-vertex **scalar** field rather than
positions: [`filter_scalar_laplacian`][triwarp.smoothing.filter_scalar_laplacian] diffuses it, and
[`saturate_scalar_gradient`][triwarp.smoothing.saturate_scalar_gradient] caps how fast it may vary
along an edge. The second is not a smoothing filter at all — it is a one-sided Lipschitz projection,
which is what turns a raw scalar into a usable sizing or falloff field.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.typing as twt
from triwarp import laplacian
from triwarp._device import require_cuda
from triwarp.constants import TILE_1D
from triwarp.kernels import laplacian as kernel_laplacian
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import smoothing as kernel_smoothing
from triwarp.kernels import triangles as kernel_triangles
from triwarp.triangles import face_normals_and_areas
from triwarp.vertices import mean_vertex_normals


def _apply_operator(
    operator: wps.BsrMatrix[wp.float32], v_in: wp.array[wp.vec3d], out_lv: wp.array[wp.vec3d]
) -> None:
    wp.launch(
        kernel_laplacian.apply_operator,
        dim=int(v_in.shape[0]),
        inputs=[operator.offsets, operator.columns, operator.values, v_in, out_lv],
        device=v_in.device,
    )


def _mesh_volume(positions: wp.array[wp.vec3d], faces: wp.array[wp.int32]) -> float:
    n_faces = int(faces.shape[0]) // 3
    device = positions.device
    volumes = wp.empty(n_faces, dtype=wp.float64, device=device)
    origin = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    wp.launch(
        kernel_triangles.signed_tet_volumes,
        dim=n_faces,
        inputs=[positions, faces, origin, volumes],
        device=device,
    )
    # Device-side tiled sum: only the 8-byte total crosses to the host, not the whole array.
    total = wp.zeros(1, dtype=wp.float64, device=device)
    n_tiles = (n_faces + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_reduce.sum1d_tiled,
        dim=[n_tiles],
        inputs=[volumes, total],
        block_dim=TILE_1D,
        device=device,
    )
    return float(total.numpy()[0])


def _apply_volume_constraint(
    positions: wp.array[wp.vec3d], faces: wp.array[wp.int32], vol_ini: float
) -> None:
    vol_new = _mesh_volume(positions, faces)
    if vol_new != 0.0:
        factor = (vol_ini / vol_new) ** (1.0 / 3.0)
        wp.map(wp.mul, positions, wp.float64(factor), out=positions)


def filter_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.5,
    iterations: int = 10,
    implicit_time_integration: bool = False,
    volume_constraint: bool = True,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Laplacian mesh smoothing (Vollmer et al.; Desbrun et al. implicit fairing).

    Diffuses each vertex toward the mean of its 1-ring neighbors. With ``implicit_time_integration``
    the diffusion is solved implicitly (backward Euler, unconditionally stable); otherwise an
    explicit forward-Euler step is used (stable for ``lamb <= 1``). Mirrors
    [`trimesh.smoothing.filter_laplacian`][] but returns a new vertex array instead of mutating
    the mesh in place.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    lamb
        Diffusion speed. ``0`` leaves the mesh unchanged; larger values smooth faster.
    iterations
        Number of smoothing passes.
    implicit_time_integration
        If ``True`` solve ``((1 + lamb) I - lamb L) V' = V`` each pass via conjugate gradient
        (**CUDA only** — raises on CPU). If ``False`` apply the explicit step
        ``V' = V + lamb (L V - V)``.
    volume_constraint
        If ``True`` rescale the mesh after each pass to preserve its initial volume, counteracting
        Laplacian shrinkage.
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Autogenerated (uniform weights) when ``None``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_taubin`][triwarp.smoothing.filter_taubin]
    [`filter_humphrey`][triwarp.smoothing.filter_humphrey]
    [`filter_implicit_fairing`][triwarp.smoothing.filter_implicit_fairing]
    [`trimesh.smoothing.filter_laplacian`][]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces)
    )
    positions = tw.array._as_vec3d(vertices)
    vol_ini = _mesh_volume(positions, faces) if volume_constraint else 0.0

    if implicit_time_integration:
        require_cuda(device, "filter_laplacian(implicit_time_integration=True)")
        system = _build_implicit_system(operator, lamb, n, device)
        precond = wpl.preconditioner(system, "diag")
        components = _empty_components(n, device)
        solutions = _empty_components(n, device)
        for _ in range(iterations):
            wp.map(kernel_smoothing.extract_components, positions, out=list(components))
            for rhs, solution in zip(components, solutions, strict=True):
                wp.copy(solution, rhs)
                twl.solve_spd(
                    system,
                    rhs,
                    solution,
                    tol=twl.CG_TOLERANCE,
                    maxiter=10 * n,
                    preconditioner=precond,
                    name="filter_laplacian(implicit_time_integration=True)",
                )
            wp.map(kernel_smoothing.combine_components, *solutions, out=positions)
            if volume_constraint:
                _apply_volume_constraint(positions, faces, vol_ini)
    else:
        lv = wp.empty(n, dtype=wp.vec3d, device=device)
        nxt = wp.empty(n, dtype=wp.vec3d, device=device)
        coeff = wp.float64(lamb)
        step = wp.map(
            kernel_smoothing.laplacian_step, positions, lv, coeff, out=nxt, return_kernel=True
        )
        for _ in range(iterations):
            _apply_operator(operator, positions, lv)
            wp.launch(step, dim=n, inputs=[positions, lv, coeff], outputs=[nxt], device=device)
            positions, nxt = nxt, positions
            if volume_constraint:
                _apply_volume_constraint(positions, faces, vol_ini)

    return tw.array._as_vec3(positions)


def filter_humphrey(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    alpha: float = 0.1,
    beta: float = 0.5,
    iterations: int = 10,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Laplacian smoothing with Humphrey (HC) filtering (Vollmer et al.).

    Each pass performs a Laplacian smoothing step and then pushes the vertices partway back
    toward their original and pre-smoothing positions, strongly reducing the shrinkage of plain
    Laplacian smoothing. Mirrors [`trimesh.smoothing.filter_humphrey`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    alpha
        Shrinkage control in ``[0, 1]``: ``0`` ignores the original positions, ``1`` disables
        smoothing.
    beta
        Correction aggressiveness in ``[0, 1]``: ``0`` disables smoothing, ``1`` is most aggressive.
    iterations
        Number of smoothing passes.
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Autogenerated (uniform weights) when ``None``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`trimesh.smoothing.filter_humphrey`][]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces)
    )
    positions = tw.array._as_vec3d(vertices)
    original = wp.empty(n, dtype=wp.vec3d, device=device)
    wp.copy(original, positions)

    lv = wp.empty(n, dtype=wp.vec3d, device=device)
    b = wp.empty(n, dtype=wp.vec3d, device=device)
    lb = wp.empty(n, dtype=wp.vec3d, device=device)
    nxt = wp.empty(n, dtype=wp.vec3d, device=device)
    alpha64 = wp.float64(alpha)
    beta64 = wp.float64(beta)
    residual = wp.map(
        kernel_smoothing.humphrey_residual,
        lv,
        original,
        positions,
        alpha64,
        out=b,
        return_kernel=True,
    )
    update = wp.map(
        kernel_smoothing.humphrey_update, lv, b, lb, beta64, out=nxt, return_kernel=True
    )
    for _ in range(iterations):
        # ``positions`` doubles as the previous-iterate ``q`` (it is only read this pass).
        _apply_operator(operator, positions, lv)
        wp.launch(
            residual, dim=n, inputs=[lv, original, positions, alpha64], outputs=[b], device=device
        )
        _apply_operator(operator, b, lb)
        wp.launch(update, dim=n, inputs=[lv, b, lb, beta64], outputs=[nxt], device=device)
        positions, nxt = nxt, positions

    return tw.array._as_vec3(positions)


def filter_taubin(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.5,
    nu: float = 0.5,
    iterations: int = 10,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Taubin lambda/nu mesh smoothing (Vollmer et al.).

    Alternates a shrinking diffusion step (``+lamb``) on even passes with an inflating step
    (``-nu``) on odd passes, so the low-frequency shape is preserved while high-frequency noise is
    removed with little volume loss. Mirrors [`trimesh.smoothing.filter_taubin`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    lamb
        Shrinking step size in ``[0, 1]``.
    nu
        Inflating step size in ``[0, 1]``; typically chosen so ``0 < 1/lamb - 1/nu < 0.1``.
    iterations
        Number of smoothing passes (shrink and inflate alternate by pass index).
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Autogenerated (uniform weights) when ``None``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`trimesh.smoothing.filter_taubin`][]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces)
    )
    positions = tw.array._as_vec3d(vertices)
    lv = wp.empty(n, dtype=wp.vec3d, device=device)
    nxt = wp.empty(n, dtype=wp.vec3d, device=device)
    step = wp.map(
        kernel_smoothing.laplacian_step,
        positions,
        lv,
        wp.float64(lamb),
        out=nxt,
        return_kernel=True,
    )
    for index in range(iterations):
        _apply_operator(operator, positions, lv)
        coeff = lamb if index % 2 == 0 else -nu
        wp.launch(
            step, dim=n, inputs=[positions, lv, wp.float64(coeff)], outputs=[nxt], device=device
        )
        positions, nxt = nxt, positions

    return tw.array._as_vec3(positions)


def filter_neighborhood_average(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    iterations: int = 10,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Uniform neighborhood-average smoothing over the closed 1-ring.

    Each pass replaces every vertex by the equal-weight average of itself and its direct
    neighbors, ``v' = (v + sum(v_neighbors)) / (1 + degree)``. Unlike
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian] (which averages the open 1-ring,
    excluding the vertex, and scales the result by ``lamb``) this is the simplest possible
    smoother, with no diffusion coefficient. It is the CPU/GPU-portable analogue of Open3D's
    ``TriangleMesh.filter_smooth_simple``.

    Isolated vertices (degree 0) are left unchanged.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    iterations
        Number of smoothing passes.
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Autogenerated (uniform weights) when ``None``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`filter_taubin`][triwarp.smoothing.filter_taubin]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    # Symmetric adjacency (undirected 1-ring) so boundary vertices average over all their
    # neighbors, matching Open3D's ``adjacency_list``; the directed default is asymmetric there.
    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces, symmetric=True)
    )
    positions = tw.array._as_vec3d(vertices)
    lv = wp.empty(n, dtype=wp.vec3d, device=device)
    nxt = wp.empty(n, dtype=wp.vec3d, device=device)
    # CSR row bounds as aligned per-vertex inputs: degree(i) = offsets[i + 1] - offsets[i].
    starts = operator.offsets[:-1]
    ends = operator.offsets[1:]
    step = wp.map(
        kernel_smoothing.neighborhood_average,
        positions,
        lv,
        starts,
        ends,
        out=nxt,
        return_kernel=True,
    )
    for _ in range(iterations):
        _apply_operator(operator, positions, lv)
        wp.launch(step, dim=n, inputs=[positions, lv, starts, ends], outputs=[nxt], device=device)
        positions, nxt = nxt, positions

    return tw.array._as_vec3(positions)


def filter_mut_dif_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.5,
    iterations: int = 10,
    volume_constraint: bool = True,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Laplacian smoothing with a mutable diffusion coefficient (Barroqueiro et al.).

    Each pass runs a Laplacian diffusion step whose speed is adapted **per vertex**: the
    coefficient ``lamber`` is scaled by how much of the vertex's Laplacian residual lies along
    its normal, so flat regions diffuse quickly while feature regions diffuse slowly. When
    ``volume_constraint`` is set, the mesh is inflated along its vertex normals (rather than
    isotropically rescaled) to counteract shrinkage. Mirrors
    [`trimesh.smoothing.filter_mut_dif_laplacian`][] but returns a new vertex array instead of
    mutating the mesh in place.

    Following the reference, the vertex normals (and the finite-difference step ``eps``) are
    computed once from the input mesh and held constant across all passes.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    lamb
        Diffusion speed. ``0`` leaves the mesh unchanged; larger values smooth faster.
    iterations
        Number of smoothing passes.
    volume_constraint
        If ``True`` inflate the mesh along its vertex normals after each pass to preserve its
        initial volume (meaningful only for watertight meshes).
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Autogenerated (uniform weights) when ``None``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`filter_humphrey`][triwarp.smoothing.filter_humphrey]
    [`filter_taubin`][triwarp.smoothing.filter_taubin]
    [`trimesh.smoothing.filter_mut_dif_laplacian`][]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces)
    )
    positions = tw.array._as_vec3d(vertices)

    # Vertex normals and eps are computed once from the input mesh and reused every pass, matching
    # the trimesh reference (which reads normals off the un-mutated mesh inside its loop).
    face_normals, areas = face_normals_and_areas(vertices, faces)
    normals = mean_vertex_normals(n, faces, face_normals)
    vol_ini = _mesh_volume(positions, faces) if volume_constraint else 0.0
    eps = 0.01 * float(tw.reduce.max(areas)) ** 0.5 if volume_constraint else 0.0

    lv = wp.empty(n, dtype=wp.vec3d, device=device)
    adil = wp.empty(n, dtype=wp.float64, device=device)
    adil_sum = wp.zeros(1, dtype=wp.float64, device=device)
    nxt = wp.empty(n, dtype=wp.vec3d, device=device)
    probe = wp.empty(n, dtype=wp.vec3d, device=device) if volume_constraint else None
    slope = 0.0
    inv_n = wp.float64(1.0 / n)
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    adil_kernel = wp.map(
        kernel_smoothing.mut_dif_adil, normals, positions, lv, out=adil, return_kernel=True
    )
    for index in range(iterations):
        # The mean diffusion coefficient is reduced on device and consumed by the step kernel
        # directly, so the loop body issues no host synchronisation.
        _apply_operator(operator, positions, lv)
        wp.launch(
            adil_kernel, dim=n, inputs=[normals, positions, lv], outputs=[adil], device=device
        )
        adil_sum.zero_()
        wp.launch_tiled(
            kernel_reduce.sum1d_tiled,
            dim=[n_tiles],
            inputs=[adil, adil_sum],
            block_dim=TILE_1D,
            device=device,
        )
        wp.launch(
            kernel_smoothing.mut_dif_step_scaled,
            dim=n,
            inputs=[positions, lv, adil, adil_sum, inv_n, wp.float64(lamb)],
            outputs=[nxt],
            device=device,
        )
        positions, nxt = nxt, positions
        if volume_constraint:
            vol = _mesh_volume(positions, faces)
            if index == 0:
                wp.map(
                    kernel_smoothing.add_scaled_normal,
                    positions,
                    normals,
                    wp.float64(eps),
                    out=probe,
                )
                vol2 = _mesh_volume(probe, faces)
                slope = eps / (vol2 - vol) if vol2 != vol else 0.0
            wp.map(
                kernel_smoothing.add_scaled_normal,
                positions,
                normals,
                wp.float64(slope * (vol_ini - vol)),
                out=positions,
            )

    return tw.array._as_vec3(positions)


def filter_implicit_fairing(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.1,
    iterations: int = 10,
    *,
    pin_boundary: bool = True,
) -> wp.array[wp.vec3]:
    """
    Implicit fairing with the cotangent Laplace-Beltrami operator (Desbrun et al.).

    Solves the backward-Euler diffusion ``(M - lamb L) V' = M V`` each pass, where ``L`` is the
    cotangent stiffness matrix ([`cotmatrix`][triwarp.laplacian.cotmatrix]) and ``M`` the
    barycentric lumped mass matrix ([`mass_matrix`][triwarp.laplacian.mass_matrix]), both recomputed
    from the current geometry. This is the most accurate of the smoothing filters (curvature flow;
    libigl tutorial 205, MeshLib ``Laplacian`` in cotangent mode) and is **CUDA only** (raises on
    CPU). The solve uses ``float64`` throughout.

    On a mesh with an open boundary the flow needs a boundary condition, which is what
    ``pin_boundary`` supplies: without one the unconstrained boundary is pulled inward, collapsing
    the triangles there, and since the system is solved by conjugate gradient (the only Warp solver
    available) further passes diverge rather than degrade gracefully. Pinning turns the pass into a
    Dirichlet problem over the interior, which is the formulation Desbrun et al. give for a surface
    with boundary and is well posed for any number of passes.

    A closed mesh has no boundary, so ``pin_boundary`` costs nothing and changes nothing there --
    the solve is over every vertex either way, and curvature flow on a closed surface still shrinks
    it (a sphere contracts toward a point, which is the method working, not failing).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    lamb
        Diffusion time step. Larger values smooth faster; the implicit solve is unconditionally
        stable.
    iterations
        Number of smoothing passes.
    pin_boundary
        Hold boundary vertices at their input positions and solve only for the interior. A no-op on
        a closed mesh. Set ``False`` for the unconstrained flow, which is free to shrink the
        boundary — expect [`solve_spd`][triwarp.linalg.solve_spd] to warn once it stops converging.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix`][triwarp.laplacian.mass_matrix]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n == 0 or n_faces == 0 or iterations == 0:
        out = wp.empty(n, dtype=wp.vec3, device=device)
        wp.copy(out, vertices)
        return out

    require_cuda(device, "filter_implicit_fairing")
    positions = tw.array._as_vec3d(vertices)
    components = _empty_components(n, device)
    rhs = _empty_components(n, device)
    solutions = _empty_components(n, device)

    # Boundary topology is fixed for the whole flow, so the partition is built once. ``None`` means
    # "solve over every vertex" -- either the caller asked for the unconstrained flow, or the mesh
    # is closed and there is nothing to pin.
    dirichlet = _dirichlet_state(vertices, faces, positions, n, device) if pin_boundary else None

    for _ in range(iterations):
        current = tw.array._as_vec3(positions)
        # Rebuilt every iteration on purpose: the cotangent weights depend on ``current``, which
        # the fairing step moves, so implicit fairing must re-linearise on the moving surface.
        cot_entries = laplacian.cotmatrix_entries(current, faces)
        stiffness = laplacian.cotmatrix(current, faces, cot_entries=cot_entries, dtype=wp.float64)

        mass = laplacian.mass_matrix_entries(current, faces, dtype=wp.float64)

        # Right-hand side b = M V, formed before ``bsr_axpy`` mutates the mass matrix.
        wp.map(kernel_smoothing.extract_components, positions, out=list(components))
        for component, b in zip(components, rhs, strict=True):
            wp.map(wp.mul, mass, component, out=b)

        # A = M - lamb L (SPD: L has a negative diagonal, so subtracting it adds to the diagonal).
        system = wps.bsr_axpy(x=stiffness, y=wps.bsr_diag(diag=mass), alpha=-float(lamb), beta=1.0)

        if dirichlet is None:
            precond = wpl.preconditioner(system, "diag")
            for b, solution, component in zip(rhs, solutions, components, strict=True):
                wp.copy(solution, component)
                twl.solve_spd(
                    system,
                    b,
                    solution,
                    tol=twl.CG_TOLERANCE,
                    maxiter=10 * n,
                    preconditioner=precond,
                    name="filter_implicit_fairing",
                )
            wp.map(kernel_smoothing.combine_components, *solutions, out=positions)
            continue

        # Dirichlet pass: eliminate the pinned rows and columns, then solve over the interior.
        # ``assemble_interior_system`` supplies ``-A_ub x_b``; the linear term ``(M V)_u`` is added
        # on top, since that helper eliminates a quadratic form which has none of its own.
        interior_system, interior_rhs = twl.assemble_interior_system(
            system, dirichlet.fixed_mask, dirichlet.free_map, dirichlet.pinned, dirichlet.n_free
        )
        wp.launch(
            kernel_smoothing.add_interior_mass_rhs,
            dim=n,
            inputs=[dirichlet.fixed_mask, dirichlet.free_map, mass, positions, interior_rhs],
            device=device,
        )
        interior_precond = wpl.preconditioner(interior_system, "diag")
        solution_2d = dirichlet.solution
        for column in range(3):
            wp.copy(solution_2d[column], interior_rhs[column])
            twl.solve_spd(
                interior_system,
                interior_rhs[column],
                solution_2d[column],
                tol=twl.CG_TOLERANCE,
                maxiter=10 * dirichlet.n_free,
                preconditioner=interior_precond,
                name="filter_implicit_fairing",
            )
        wp.launch(
            kernel_smoothing.scatter_free_positions,
            dim=n,
            inputs=[
                dirichlet.fixed_mask,
                dirichlet.free_map,
                solution_2d[0],
                solution_2d[1],
                solution_2d[2],
                positions,
            ],
            device=device,
        )

    return tw.array._as_vec3(positions)


class _Dirichlet(NamedTuple):
    """Everything a boundary-pinned fairing pass needs, built once because the boundary is fixed."""

    fixed_mask: wp.array[wp.bool]
    free_map: wp.array[wp.int32]
    n_free: int
    pinned: twt.Array2dFloat
    """``(3, n_vertices)`` prescribed values; only the pinned rows are ever read."""
    solution: twt.Array2dFloat
    """``(3, n_free)`` scratch for the reduced solve."""


def _dirichlet_state(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    positions: wp.array[wp.vec3d],
    n: int,
    device: wp.DeviceLike,
) -> _Dirichlet | None:
    """
    Boundary-pinned solve state, or ``None`` when there is no boundary to pin.

    Returning ``None`` for a closed mesh is deliberate: it routes the caller down the unreduced
    path, so a watertight input runs exactly the same code as it did before pinning existed.
    """
    boundary_wp = tw.boundary.boundary_vertex_indices(vertices, faces)
    if int(boundary_wp.shape[0]) == 0:
        return None
    fixed_mask = tw.array.indices_to_mask(boundary_wp, n, device=device)
    free_map, n_free = twl.free_partition(fixed_mask)
    if n_free == 0:
        return None
    pinned = wp.zeros((3, n), dtype=wp.float64, device=device)
    wp.map(
        kernel_smoothing.extract_components, positions, out=[pinned[column] for column in range(3)]
    )
    return _Dirichlet(
        fixed_mask,
        free_map,
        n_free,
        twt.as_array2d_float(pinned, dtype=wp.float64),
        twt.as_array2d_float(
            wp.zeros((3, n_free), dtype=wp.float64, device=device), dtype=wp.float64
        ),
    )


def _empty_components(
    n: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.float64], wp.array[wp.float64], wp.array[wp.float64]]:
    return (
        wp.empty(n, dtype=wp.float64, device=device),
        wp.empty(n, dtype=wp.float64, device=device),
        wp.empty(n, dtype=wp.float64, device=device),
    )


def _build_implicit_system(
    operator: wps.BsrMatrix[wp.float32], lamb: float, n: int, device: wp.DeviceLike
) -> wps.BsrMatrix[wp.float64]:
    nnz = int(operator.nnz)
    n_triplets = nnz + n
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=wp.float64, device=device)
    wp.launch(
        kernel_smoothing.implicit_laplacian_triplets,
        dim=n,
        inputs=[
            operator.offsets,
            operator.columns,
            operator.values,
            wp.float64(lamb),
            wp.int32(nnz),
            rows,
            cols,
            vals,
        ],
        device=device,
    )
    return wps.bsr_from_triplets(n, n, rows, cols, vals, prune_numerical_zeros=False)


# ---------------------------------------------------------------------------
# Region smoothing solves (positionVertsSmoothly / positionVertsSmoothlySharpBd)
# ---------------------------------------------------------------------------


def _edge_weight_matrix(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edge_weights: str
) -> wps.BsrMatrix[wp.float64]:
    """Symmetric ``(n, n)`` float64 edge-weight matrix (zero diagonal); unit or clamped cotan."""
    device = vertices.device
    n = int(vertices.shape[0])
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n)
    m = int(unique_edges.shape[0])
    weights = wp.zeros(m, dtype=wp.float32, device=device)
    if edge_weights == "unit":
        weights.fill_(1.0)
    elif edge_weights == "cotan":
        wp.launch(
            kernel_smoothing.edge_cotan_add,
            dim=int(faces.shape[0]) // 3,
            inputs=[vertices, faces, inverse, weights],
            device=device,
        )
        wp.map(kernel_smoothing.clamp_cotan, weights, out=weights)
    else:
        raise ValueError(f"edge_weights must be 'unit' or 'cotan', got {edge_weights!r}")
    rows = wp.empty(2 * m, dtype=wp.int32, device=device)
    cols = wp.empty(2 * m, dtype=wp.int32, device=device)
    vals = wp.empty(2 * m, dtype=wp.float64, device=device)
    wp.launch(
        kernel_smoothing.symmetric_weight_triplets,
        dim=m,
        inputs=[unique_edges, weights, rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(n, n, rows, cols, vals, prune_numerical_zeros=False)


def position_verts_smoothly_sharp_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    free_mask: wp.array[wp.bool],
    stabilizer: float = 0.0,
) -> wp.array[wp.vec3]:
    """
    Reposition a free vertex region as the umbrella-Laplacian solution with a sharp fixed boundary.

    Ports MeshLib ``positionVertsSmoothlySharpBd``: the free vertices (``free_mask``) are moved to
    the solution of the graph-Laplacian Dirichlet system ``(D - W) x = b`` (unit edge weights),
    where the fixed one-ring neighbours are folded into ``b``, so the region rim stays sharp
    (a hard C⁰ constraint). Fixed vertices keep their positions.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    free_mask
        Length-``n_vertices`` ``wp.bool`` mask of the vertices to reposition.
    stabilizer
        Optional attraction to the original position (adds to the diagonal), which also makes the
        system solvable for a fully-free component. Defaults to ``0``.

    Returns
    -------
    wp.array[wp.vec3]
        New vertex positions on ``vertices.device`` (a copy; fixed vertices unchanged).

    Raises
    ------
    NotImplementedError
        On a CPU device (``warp.optim.linear.cg`` produces NaN on CPU in Warp 1.14-1.15).

    See Also
    --------
    [`position_verts_smoothly`][triwarp.smoothing.position_verts_smoothly]
    [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely]

    Notes
    -----
    The system is SPD only when no free connected component is entirely free (or ``stabilizer >
    0``); the hole-filling pipeline guarantees this because the patch rim is always fixed.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    out = wp.clone(vertices)
    if int(faces.shape[0]) == 0 or n == 0:
        return out
    free_map, n_free = tw.array.mask_to_index_map(free_mask)
    if n_free == 0:
        return out
    require_cuda(device, "position_verts_smoothly_sharp_boundary")

    weight_matrix = _edge_weight_matrix(vertices, faces, "unit")
    nnz = int(weight_matrix.nnz)
    size = nnz + n
    out_rows = wp.zeros(size, dtype=wp.int32, device=device)
    out_cols = wp.zeros(size, dtype=wp.int32, device=device)
    out_vals = wp.zeros(size, dtype=wp.float64, device=device)
    # One contiguous (3, n_free) right-hand side: its rows are contiguous 1-D views, so the
    # assembly kernel writes them directly and the three columns solve in one batched CG.
    rhs = wp.zeros((3, n_free), dtype=wp.float64, device=device)
    wp.launch(
        kernel_smoothing.dirichlet_system_triplets,
        dim=n,
        inputs=[
            weight_matrix.offsets,
            weight_matrix.columns,
            weight_matrix.values,
            free_mask,
            free_map,
            vertices,
            wp.float64(stabilizer),
            out_rows,
            out_cols,
            out_vals,
            rhs[0],
            rhs[1],
            rhs[2],
        ],
        device=device,
    )
    system = wps.bsr_from_triplets(
        n_free, n_free, out_rows, out_cols, out_vals, prune_numerical_zeros=False
    )
    sol = wp.zeros((3, n_free), dtype=wp.float64, device=device)
    twl.solve_spd_columns(
        system,
        twt.as_array2d_float(rhs, dtype=wp.float64),
        twt.as_array2d_float(sol, dtype=wp.float64),
        tol=twl.CG_TOLERANCE,
        maxiter=10 * n_free,
    )
    wp.launch(
        kernel_smoothing.scatter_free_solution,
        dim=n,
        inputs=[free_mask, free_map, sol[0], sol[1], sol[2], out],
        device=device,
    )
    return out


def position_verts_smoothly(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    free_mask: wp.array[wp.bool],
    edge_weights: str = "cotan",
) -> wp.array[wp.vec3]:
    """
    Reposition a free vertex region so the surface is smooth across the region boundary too.

    Ports MeshLib ``positionVertsSmoothly`` (the ``Laplacian`` least-squares solve with
    ``RememberShape::No``): every vertex in the region and its first fixed ring contributes the
    umbrella equation ``p_v = Σ_d (w_vd / ΣW) p_d``; free vertices are unknowns and fixed
    neighbours move to the right-hand side. The normal equations ``(MᵀM) x = Mᵀ b`` are solved per
    coordinate, giving a patch that is smooth (C¹) *across* the region rim, unlike the sharp-rim
    [`position_verts_smoothly_sharp_boundary`][triwarp.smoothing.position_verts_smoothly_sharp_boundary].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    free_mask
        Length-``n_vertices`` ``wp.bool`` mask of the vertices to reposition.
    edge_weights
        ``"cotan"`` (default, clamped cotangent weights) or ``"unit"`` (uniform).

    Returns
    -------
    wp.array[wp.vec3]
        New vertex positions on ``vertices.device`` (a copy; fixed vertices unchanged).

    Raises
    ------
    NotImplementedError
        On a CPU device.
    ValueError
        If ``edge_weights`` is not ``"cotan"`` or ``"unit"``.

    See Also
    --------
    [`position_verts_smoothly_sharp_boundary`][triwarp.smoothing.position_verts_smoothly_sharp_boundary]
    [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    out = wp.clone(vertices)
    if int(faces.shape[0]) == 0 or n == 0:
        return out
    free_map, n_free = tw.array.mask_to_index_map(free_mask)
    if n_free == 0:
        return out
    require_cuda(device, "position_verts_smoothly")

    row_mask = tw.selection.expand_vertex_mask(faces, free_mask, 1)
    row_map, n_rows = tw.array.mask_to_index_map(row_mask)
    weight_matrix = _edge_weight_matrix(vertices, faces, edge_weights)
    nnz = int(weight_matrix.nnz)
    size = nnz + n
    rows = wp.zeros(size, dtype=wp.int32, device=device)
    cols = wp.zeros(size, dtype=wp.int32, device=device)
    vals = wp.zeros(size, dtype=wp.float64, device=device)
    rhs_x = wp.zeros(n_rows, dtype=wp.float64, device=device)
    rhs_y = wp.zeros(n_rows, dtype=wp.float64, device=device)
    rhs_z = wp.zeros(n_rows, dtype=wp.float64, device=device)
    wp.launch(
        kernel_smoothing.laplacian_ls_triplets,
        dim=n,
        inputs=[
            weight_matrix.offsets,
            weight_matrix.columns,
            weight_matrix.values,
            free_mask,
            row_mask,
            free_map,
            row_map,
            vertices,
            rows,
            cols,
            vals,
            rhs_x,
            rhs_y,
            rhs_z,
        ],
        device=device,
    )
    # M is (n_rows x n_free); build M and M^T from the same (swapped) triplets in a single build
    # each (never rebuilt/recast — bsr_mm determinism), then A = M^T M is SPD.
    m_matrix = wps.bsr_from_triplets(n_rows, n_free, rows, cols, vals, prune_numerical_zeros=False)
    mt_matrix = wps.bsr_from_triplets(
        n_free, n_rows, wp.clone(cols), wp.clone(rows), wp.clone(vals), prune_numerical_zeros=False
    )
    system = wps.bsr_mm(mt_matrix, m_matrix)
    # A^T b straight into the rows of one contiguous buffer, so the three columns batch.
    atb = wp.zeros((3, n_free), dtype=wp.float64, device=device)
    for column, component in enumerate((rhs_x, rhs_y, rhs_z)):
        wps.bsr_mv(mt_matrix, component, atb[column], alpha=1.0, beta=0.0)
    sol = wp.zeros((3, n_free), dtype=wp.float64, device=device)
    twl.solve_spd_columns(
        system,
        twt.as_array2d_float(atb, dtype=wp.float64),
        twt.as_array2d_float(sol, dtype=wp.float64),
        tol=twl.CG_TOLERANCE,
        maxiter=10 * n_free,
    )
    wp.launch(
        kernel_smoothing.scatter_free_solution,
        dim=n,
        inputs=[free_mask, free_map, sol[0], sol[1], sol[2], out],
        device=device,
    )
    return out


def filter_scalar_laplacian(
    values: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.5,
    iterations: int = 10,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.float32]:
    """
    Diffuse a per-vertex scalar field through the 1-ring averaging operator.

    The scalar counterpart of [`filter_laplacian`][triwarp.smoothing.filter_laplacian], running the
    identical explicit step ``q' = q + lamb (L q - q)`` on a field rather than on positions. Reach
    for it whenever a computed per-vertex quantity is noisier than the thing it will drive —
    curvature before it selects features, a heat or occlusion field before it becomes a weight, a
    per-vertex sizing function before it drives remeshing.

    On a **closed** mesh, ``lamb=1.0, iterations=1`` is exactly MeshLab's
    ``apply_scalar_smoothing_per_vertex``: one unweighted 1-ring average with the vertex's own value
    excluded. The defaults instead match the rest of this module (a partial step, repeated), which
    is the gentler behaviour a caller usually wants.

    !!! note "It differs from MeshLab on a boundary"
        VCG's ``VertexQualityLaplacian`` smooths a boundary vertex along the **boundary curve
        only** — it averages just that vertex's two boundary neighbours and ignores the rest of its
        ring, so the boundary values evolve as an independent 1D field. That is a different
        operator, not this one with other constants, and it is why the pymeshlab oracle in
        ``tests/test_smoothing.py`` runs on closed fixtures. Here a boundary vertex averages its
        whole ring like any other; to hold the boundary fixed instead, restore its values after the
        call.

    Parameters
    ----------
    values
        Length-``n_vertices`` ``wp.float32`` field to smooth. Not modified.
    vertices
        ``(n_vertices,)`` mesh vertex positions — needed only to build the operator, and ignored
        when ``laplacian_operator`` is supplied.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    lamb
        Step size. ``0`` leaves the field unchanged, ``1`` replaces each value by its 1-ring
        average outright. Values above ``1`` overshoot and are unstable.
    iterations
        Number of passes. ``0`` returns a copy of the input.
    laplacian_operator
        Optional precomputed row-stochastic operator (see
        [`laplacian`][triwarp.laplacian.laplacian]). Pass it to hoist the build out of a loop, or
        to smooth with inverse-edge-length weights instead.

        When ``None`` it is built as ``laplacian(vertices, faces, symmetric=True)`` — the
        **symmetric** 1-ring, unlike [`filter_laplacian`][triwarp.smoothing.filter_laplacian],
        which defaults to trimesh's directed ``mesh.edges`` adjacency. The two agree on a closed
        mesh and differ on a boundary vertex, where the directed adjacency is missing some of its
        neighbours; for a scalar field the symmetric ring is both the defensible choice and the one
        MeshLab makes.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n_vertices`` smoothed field on ``values.device``.

    Raises
    ------
    ValueError
        If ``values`` is not length ``n_vertices``.

    See Also
    --------
    [`saturate_scalar_gradient`][triwarp.smoothing.saturate_scalar_gradient]
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`triwarp.laplacian.laplacian`][triwarp.laplacian.laplacian]
    """
    device = values.device
    n = int(vertices.shape[0])
    if int(values.shape[0]) != n:
        raise ValueError(
            f"values must have one entry per vertex, got {values.shape[0]} for {n} vertices"
        )

    out = wp.empty(n, dtype=wp.float32, device=device)
    wp.copy(out, values)
    if n == 0 or iterations == 0:
        return out

    operator = (
        laplacian_operator
        if laplacian_operator is not None
        else laplacian.laplacian(vertices, faces, symmetric=True)
    )
    average = wp.empty(n, dtype=wp.float32, device=device)
    nxt = wp.empty(n, dtype=wp.float32, device=device)
    coeff = wp.float32(lamb)
    step = wp.map(
        kernel_smoothing.scalar_laplacian_step, out, average, coeff, out=nxt, return_kernel=True
    )
    for _ in range(iterations):
        wp.launch(
            kernel_smoothing.apply_operator_scalar,
            dim=n,
            inputs=[operator.offsets, operator.columns, operator.values, out, average],
            device=device,
        )
        wp.launch(step, dim=n, inputs=[out, average, coeff], outputs=[nxt], device=device)
        out, nxt = nxt, out
    return out


def saturate_scalar_gradient(
    values: wp.array[wp.float32],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    threshold: float = 1.0,
    max_iterations: int = 0,
) -> wp.array[wp.float32]:
    """
    Cap how fast a per-vertex scalar may grow with distance, by lowering values only.

    Enforces the one-sided Lipschitz bound ``q_i <= q_j + |p_i - p_j| / threshold`` on every edge,
    which after convergence gives the *upper envelope*
    ``q(v) = min_u (q_0(u) + d(u, v) / threshold)`` over shortest paths ``d`` through the edge
    graph. Nothing is ever raised, so every local minimum of the input survives untouched and only
    peaks that rise too steeply out of them are shaved down.

    This is MeshLab's ``apply_scalar_saturation_per_vertex`` (VCG ``VertexSaturate``), and the
    standard way to make a raw scalar usable as a **sizing field**: an adaptive remesher fed an
    ungraded target-length field produces a band of bad triangles where the field jumps, and this is
    the projection that removes the jump while respecting the field's small values.

    !!! note "``threshold`` is a reciprocal slope"
        The admissible change per unit distance is ``1 / threshold``, matching MeshLab. So a
        *larger* ``threshold`` is a *stricter* cap — ``threshold=2`` allows half the variation
        ``threshold=1`` does.

    Parameters
    ----------
    values
        Length-``n_vertices`` ``wp.float32`` field. Not modified.
    vertices
        ``(n_vertices,)`` mesh vertex positions; edge lengths are measured from these.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    threshold
        Reciprocal of the maximum admissible slope; must be positive.
    max_iterations
        Cap on relaxation passes. Each pass propagates the bound one edge further, so the number
        needed is the graph diameter of the region that violates it. ``0`` (the default) means
        ``n_vertices``, which can never be exceeded.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n_vertices`` saturated field on ``values.device``.

    Raises
    ------
    ValueError
        If ``threshold <= 0``, ``max_iterations < 0``, or ``values`` is not length ``n_vertices``.

    See Also
    --------
    [`filter_scalar_laplacian`][triwarp.smoothing.filter_scalar_laplacian]
    [`triwarp.remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")

    device = values.device
    n = int(vertices.shape[0])
    if int(values.shape[0]) != n:
        raise ValueError(
            f"values must have one entry per vertex, got {values.shape[0]} for {n} vertices"
        )

    out = wp.empty(n, dtype=wp.float32, device=device)
    wp.copy(out, values)
    if n == 0 or int(faces.shape[0]) == 0:
        return out

    adjacency = tw.graph.edges_to_csr(n, tw.edges.faces_to_edges(faces))
    nxt = wp.empty(n, dtype=wp.float32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)
    inverse_threshold = wp.float32(1.0 / threshold)
    # The pass count is data-dependent (it is the diameter of the violating region), and the flag
    # readback is ~0.1 ms against ~1 ms of launches per pass, so checking every pass is the cheaper
    # side of that trade -- see the ``linalg`` note on ``check_every``.
    for _ in range(max_iterations or n):
        changed.zero_()
        wp.launch(
            kernel_smoothing.saturate_gradient_pass,
            dim=n,
            inputs=[
                adjacency.offsets,
                adjacency.columns,
                vertices,
                inverse_threshold,
                out,
                nxt,
                changed,
            ],
            device=device,
        )
        out, nxt = nxt, out
        if int(changed.numpy()[0]) == 0:
            break
    return out


def filter_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    iterations: int = 20,
    threshold: float = 60.0,
    face_adjacency: twt.Array2dInt32 | None = None,
) -> wp.array[wp.vec3]:
    """
    Smooth the *face normals* without moving a vertex, preserving creases.

    Each pass replaces a face's normal with the area-weighted average of its own and those of its
    edge-neighbours **whose normal is within ``threshold`` of it**. That gate is what makes this
    feature-preserving rather than isotropic: across a crease the two normals disagree by more than
    the threshold and never average, so a sharp edge survives any number of passes while noise on a
    flat region diffuses away in a few. MeshLab's ``apply_normal_smoothing_per_face``.

    The result is a normal field that is no longer the geometric normal of any triangle — it is the
    *target* for [`filter_two_step`][triwarp.smoothing.filter_two_step]'s second half, and useful on
    its own for shading.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Not modified.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    iterations
        Number of diffusion passes over the normal field. MeshLab's ``stepnormalnum``, whose default
        of ``20`` is this one.
    threshold
        Angle in **degrees** beyond which two neighbouring faces refuse to average. ``0`` averages
        nothing and ``180`` averages everything (isotropic). MeshLab's ``normalthr``, default
        ``60``. Must be in ``[0, 180]``.
    face_adjacency
        Optional ``(m, 2)`` adjacency from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]; recomputed when ``None``. Pass it to
        hoist the build out of a loop — the topology never changes here, so it is safe to reuse.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_faces`` unit normals on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 180]``.

    See Also
    --------
    [`filter_two_step`][triwarp.smoothing.filter_two_step]
    [`triwarp.triangles.face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
    """
    if not 0.0 <= threshold <= 180.0:
        raise ValueError(f"threshold must be in [0, 180] degrees, got {threshold}")

    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    normals, areas = face_normals_and_areas(vertices, faces)
    if n_faces == 0 or iterations <= 0:
        return normals

    if face_adjacency is None:
        face_adjacency = tw.adjacency.face_adjacency(faces)
    m = int(face_adjacency.shape[0])
    threshold_cos = wp.float32(math.cos(math.radians(threshold)))

    accumulated = wp.empty(n_faces, dtype=wp.vec3, device=device)
    for _ in range(iterations):
        wp.map(kernel_smoothing.seed_weighted_normal, normals, areas, out=accumulated)
        if m > 0:
            wp.launch(
                kernel_smoothing.accumulate_smoothed_normals,
                dim=m,
                inputs=[normals, areas, face_adjacency, threshold_cos, accumulated],
                device=device,
            )
        wp.map(wp.normalize, accumulated, out=normals)
    return normals


def filter_two_step(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    iterations: int = 3,
    threshold: float = 60.0,
    normal_iterations: int = 20,
    fit_iterations: int = 20,
) -> wp.array[wp.vec3]:
    """
    Feature-preserving smoothing: filter the face normals, then fit the vertices to them.

    The one filter in this module that does **not** blur a crease. Every other scheme here diffuses
    positions directly, which cannot distinguish noise from a sharp edge — both are high-frequency.
    Two-step smoothing (Ohtake et al., and MeshLab's ``apply_coord_two_steps_smoothing``) separates
    the two questions:

    1. **Where should the surface face?** Diffuse the *normal* field with
       [`filter_normals`][triwarp.smoothing.filter_normals], whose threshold refuses to average
       across a crease. Noise is removed; the crease is not.
    2. **Where should the vertices go?** Move each vertex so the planes through its incident faces'
       centroids with those filtered normals agree as well as possible — a gradient step repeated
       ``fit_iterations`` times, with no free step size (the average over incident faces is the
       step).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Not modified.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. The topology never changes.
    iterations
        Number of outer passes; each one re-derives the normals from the current positions and
        refits. MeshLab's ``stepsmoothnum``, default ``3``.
    threshold
        Crease threshold in **degrees** for the normal filter. See
        [`filter_normals`][triwarp.smoothing.filter_normals]. MeshLab's ``normalthr``, default
        ``60``.
    normal_iterations
        Normal-diffusion passes per outer pass. MeshLab's ``stepnormalnum``, default ``20``.
    fit_iterations
        Vertex-fitting gradient steps per outer pass. MeshLab's ``stepfitnum``, default ``20``.

    Returns
    -------
    wp.array[wp.vec3]
        Smoothed ``(n_vertices,)`` vertex positions on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 180]``.

    See Also
    --------
    [`filter_normals`][triwarp.smoothing.filter_normals]
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`filter_unsharp_mask`][triwarp.smoothing.filter_unsharp_mask]

    Notes
    -----
    There is no volume constraint and none is needed: the fitting step moves each vertex *along* the
    filtered normals rather than toward its neighbours' mean, so the systematic inward drift that
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian] has to correct for does not arise. A
    flat region is already a fixed point of both halves.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    out = wp.empty(n, dtype=wp.vec3, device=device)
    wp.copy(out, vertices)
    if n == 0 or n_faces == 0 or iterations <= 0:
        return out

    # The topology is fixed, so the adjacency is built once for every pass of both halves.
    adjacency = tw.adjacency.face_adjacency(faces)
    delta = wp.empty(n, dtype=wp.vec3, device=device)
    counts = wp.empty(n, dtype=wp.float32, device=device)
    for _ in range(iterations):
        normals = filter_normals(
            out, faces, iterations=normal_iterations, threshold=threshold, face_adjacency=adjacency
        )
        for _fit in range(fit_iterations):
            delta.zero_()
            counts.zero_()
            wp.launch(
                kernel_smoothing.fit_vertices_to_normals,
                dim=n_faces,
                inputs=[out, faces, normals, delta, counts],
                device=device,
            )
            wp.map(kernel_smoothing.apply_fit_step, out, delta, counts, out=out)
    return out


def filter_unsharp_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    weight: float = 0.3,
    weight_original: float = 1.0,
    iterations: int = 5,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Sharpen a surface by adding back the detail a smoothing pass removes.

    The inverse of a smoothing filter, built from one: the difference between the mesh and its
    Laplacian-smoothed self *is* its high-frequency content, so adding a multiple of that difference
    exaggerates every feature. MeshLab's ``apply_coord_unsharp_mask``, and the standard way to
    recover crispness lost to an earlier smoothing or a decimation.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Not modified.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    weight
        How much of the detail to add back. ``0`` leaves the mesh unchanged and large values amplify
        noise as readily as features. MeshLab's ``weight``, default ``0.3``.
    weight_original
        Multiplier on the original position, MeshLab's ``weightorig``. ``1`` (the default) keeps the
        surface where it is and only sharpens; anything else scales the whole mesh about the origin.
    iterations
        Laplacian passes used to define "smoothed", so this sets the *scale* of detail that gets
        amplified: more passes remove lower frequencies and therefore sharpen more coarsely.
        MeshLab's ``iterations``, default ``5``.
    laplacian_operator
        Optional precomputed operator, forwarded to
        [`filter_laplacian`][triwarp.smoothing.filter_laplacian].

    Returns
    -------
    wp.array[wp.vec3]
        Sharpened ``(n_vertices,)`` vertex positions on ``vertices.device``.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`filter_two_step`][triwarp.smoothing.filter_two_step]
    """
    device = vertices.device
    n = int(vertices.shape[0])
    out = wp.empty(n, dtype=wp.vec3, device=device)
    if n == 0:
        return out
    # No volume constraint on the inner smoothing: its rescaling would leak into the difference and
    # show up as a uniform scaling of the sharpened mesh rather than as detail.
    smoothed = filter_laplacian(
        vertices,
        faces,
        lamb=1.0,
        iterations=iterations,
        volume_constraint=False,
        laplacian_operator=laplacian_operator,
    )
    wp.map(
        kernel_smoothing.unsharp_step,
        vertices,
        smoothed,
        wp.float32(weight),
        wp.float32(weight_original),
        out=out,
    )
    return out
