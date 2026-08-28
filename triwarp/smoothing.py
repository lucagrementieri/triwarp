"""
Laplacian smoothing filters, for vertex positions and for per-vertex scalar fields.

Most of the module moves *geometry*: [`filter_laplacian`][triwarp.smoothing.filter_laplacian],
[`filter_taubin`][triwarp.smoothing.filter_taubin],
[`filter_humphrey`][triwarp.smoothing.filter_humphrey],
[`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average],
[`filter_mut_dif_laplacian`][triwarp.smoothing.filter_mut_dif_laplacian] and
[`filter_implicit_fairing`][triwarp.smoothing.filter_implicit_fairing] all diffuse vertex positions
through the same row-stochastic 1-ring operator, differing in the time integration and in what they
do to counteract shrinkage. [`smooth_region`][triwarp.smoothing.smooth_region]
and its sharp-boundary variant instead solve a Dirichlet problem over a *region*, holding the rest
of the mesh fixed.

Three functions break that pattern by working on the *normal* field instead of positions, which is
what lets them keep a crease sharp: [`filter_normals`][triwarp.smoothing.filter_normals] diffuses
face normals with a crease gate, [`filter_two_step`][triwarp.smoothing.filter_two_step] then refits
the vertices to them, and [`filter_sharpen`][triwarp.smoothing.filter_sharpen] runs the
whole idea backwards to *sharpen*.

A second group relaxes toward something *other* than a Laplacian residual, which is what lets each
member fix a failure the filters above cannot see:
[`equalize_triangle_areas`][triwarp.smoothing.equalize_triangle_areas] evens out triangle areas,
[`relax_keep_volume`][triwarp.smoothing.relax_keep_volume] removes the shrinkage locally instead of
rescaling it away at the end, [`relax_approx`][triwarp.smoothing.relax_approx] fits a plane or a
quadric to a whole geodesic neighbourhood rather than averaging a 1-ring, and
[`filter_spikes`][triwarp.smoothing.filter_spikes] moves only the vertices that fail an angle test.
The first three take a vertex ``region`` and a ``max_displacement`` bound, so a relaxation can be
confined to where it is wanted and kept within a tolerance of the surface it started from;
``filter_spikes`` needs neither, because the set it touches is the answer to its own test.
[`smooth_region_boundary`][triwarp.smoothing.smooth_region_boundary] completes the region trio by
smoothing the region's *rim curve*, where the two above it smooth across the rim or inside it.

[`filter_scalar_laplacian`][triwarp.smoothing.filter_scalar_laplacian] runs the same operator over a
per-vertex **scalar** field rather than positions. Capping how fast such a field may vary along an
edge — the other half of turning a raw scalar into a usable sizing field — is not a smoothing filter
at all but a one-sided Lipschitz projection, and lives in
[`shortest_path_envelope`][triwarp.graph.shortest_path_envelope].
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.typing as twt
from triwarp import laplacian
from triwarp.constants import TILE_1D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import laplacian as kernel_laplacian
from triwarp.kernels import reduce as kernel_reduce
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import smoothing as kernel_smoothing
from triwarp.triangles import face_normals_and_areas
from triwarp.vertices import mean_vertex_normals

# Fraction of the way to the level set each ``smooth_region_boundary`` pass moves. A full step
# overshoots, because the field is rebuilt from the moved positions and the level set moves too.
_ISOLINE_DAMPING = wp.float32(0.75)


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
    # This six-line prologue opens five of the filters (here, ``filter_humphrey``,
    # ``filter_taubin``, ``filter_neighborhood_average`` and ``filter_mut_dif_laplacian``) and a
    # ``_smoothing_setup`` helper for it has been proposed twice and declined twice. The reason is
    # not line count: the two halves have no common consumer. ``filter_implicit_fairing`` needs the
    # guard and ``_as_vec3d`` but builds no operator, ``filter_neighborhood_average`` needs the
    # operator built ``symmetric=True``, and a helper returning both would have to hand back an
    # optional operator that three of the six callers immediately unwrap -- which reads worse than
    # the six lines it replaces. Do not re-propose it without a shape that avoids the ``Optional``.
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0 or iterations == 0:
        return wp.clone(vertices)

    operator = _resolved_operator(vertices, faces, laplacian_operator)
    positions = _as_vec3d(vertices)
    vol_ini = tw.measures.volume(positions, faces) if volume_constraint else 0.0

    if implicit_time_integration:
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

    return _as_vec3(positions)


def _apply_volume_constraint(
    positions: wp.array[wp.vec3d], faces: wp.array[wp.int32], vol_ini: float
) -> None:
    vol_new = tw.measures.volume(positions, faces)
    if vol_new != 0.0:
        factor = (vol_ini / vol_new) ** (1.0 / 3.0)
        wp.map(wp.mul, positions, wp.float64(factor), out=positions)


def inflate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pressure: float,
    *,
    iterations: int = 3,
    pre_smooth: bool = True,
    gradual: bool = True,
    lamb: float = 0.5,
) -> wp.array[wp.vec3]:
    """
    Inflate a surface under uniform pressure: normal displacement alternating with smoothing.

    The "balloon" flow. Each pass moves every vertex along its own
    [`vertex_normals`][triwarp.vertices.vertex_normals] at ``weighting="area"`` and then relaxes
    with one Laplacian pass, which is what keeps the triangles from shearing as the surface grows.
    Uses of it: puffing a thin shell out to a printable thickness, opening a collapsed scan, and
    supplying a starting surface a fitting loop can shrink back onto data.

    ``pressure`` is an absolute distance per pass, in the mesh's own units, so scale it off
    something intrinsic -- a fraction of the mean edge length is the usual choice. With ``gradual``
    pass ``k`` of ``n`` uses ``pressure * (k + 1) / n`` rather than the full amount, which is
    gentler on a mesh with fine triangles: the early passes let the relaxation redistribute
    before the later ones push hard.

    !!! note "Not the implicit formulation"
        The reference inflations displace and then solve an *implicit* Laplacian system, where
        this runs one explicit pass -- so the two produce different surfaces from the same
        ``pressure``, and the tests compare the properties an inflation must have (volume grows,
        the displacement is normal-aligned) rather than positions. Pass the result through
        [`filter_implicit_fairing`][triwarp.smoothing.filter_implicit_fairing] if you want that
        formulation; it is CUDA-only, which is why it is not the default here.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    pressure
        Distance to move along the normal per pass. Negative deflates.
    iterations
        Number of displace-and-relax passes.
    pre_smooth
        Run one relaxation pass *before* the first displacement. Worth it on a noisy input, where
        the normals are what the noise corrupts and displacing along them amplifies it.
    gradual
        Ramp the pressure linearly across the passes instead of applying it in full each time.
    lamb
        Relaxation strength of each Laplacian pass, as
        [`filter_laplacian`][triwarp.smoothing.filter_laplacian] takes it.

    Returns
    -------
    wp.array[wp.vec3]
        Inflated positions on ``vertices.device``. The connectivity is untouched, so ``faces``
        remains valid.

    Raises
    ------
    ValueError
        If ``iterations`` is negative.

    See Also
    --------
    [`offset_mesh`][triwarp.levelset.offset_mesh]
        The *exact* outward offset, through a signed distance field -- it changes the connectivity
        and cannot self-intersect, where this keeps the mesh and can.
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
        The relaxation half of each pass.
    [`thicken_mesh`][triwarp.levelset.thicken_mesh]
        Turns a surface into a solid shell, which is the other way to give it thickness.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    device = faces.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or iterations == 0:
        return wp.clone(vertices)

    positions = wp.clone(vertices)
    # The relaxation operator is hoisted for the same reason the step kernel below is, and it is the
    # larger of the two: ``filter_laplacian`` builds the uniform operator per call, and at
    # ``equal_weight=True`` that operator is the mesh's topology, which no pass changes. Same
    # finding as ``filter_spikes``, which is where it was measured.
    operator = laplacian.laplacian(vertices, faces)
    if pre_smooth:
        positions = filter_laplacian(
            positions, faces, lamb, iterations=1, laplacian_operator=operator
        )
    # Hoisted once: the displacement is the same map every pass, and section 4 records that a
    # per-iteration wrapper loop should not re-derive it.
    step_kernel = wp.map(
        kernel_smoothing.step_along_normal,
        positions,
        positions,
        wp.float32(0.0),
        out=positions,
        return_kernel=True,
    )
    for step in range(iterations):
        amount = pressure * (step + 1) / iterations if gradual else pressure
        normals = tw.vertices.vertex_normals(positions, faces)
        displaced = wp.empty(n_vertices, dtype=wp.vec3, device=device)
        wp.launch(
            step_kernel,
            dim=n_vertices,
            inputs=[positions, normals, wp.float32(amount)],
            outputs=[displaced],
            device=device,
        )
        positions = filter_laplacian(
            displaced, faces, lamb, iterations=1, laplacian_operator=operator
        )
    return positions


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
        return wp.clone(vertices)

    operator = _resolved_operator(vertices, faces, laplacian_operator)
    positions = _as_vec3d(vertices)
    original = wp.clone(positions)

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

    return _as_vec3(positions)


def filter_spikes(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_angle_sum: float,
    *,
    max_iter: int = 10,
    return_count: bool = False,
) -> wp.array[wp.vec3] | tuple[wp.array[wp.vec3], int]:
    """
    Pull needle-like vertices back onto the surface, leaving every other vertex untouched.

    A *spike* is a vertex whose incident corner angles sum to less than ``min_angle_sum``. On a
    flat surface that sum is ``2 * pi``; the sharper the cone, the smaller it gets, and a vertex on
    a thin needle -- the classic scan and reconstruction artifact -- has a sum near zero. So the
    threshold is read in radians against a full turn, and a value like ``0.5 * pi`` selects
    genuinely degenerate cones while leaving ordinary sharp features alone.

    Each pass replaces the flagged vertices by the average of their closed 1-rings and **nothing
    else moves**: this is not a smoothing filter restricted to a region, it is a repair that touches
    only the vertices that fail the test. Flattening a spike changes its neighbours' angle sums, so
    the pass repeats -- a needle whose base is itself spiky needs more than one -- and stops as soon
    as a pass finds none.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    min_angle_sum
        Angle-sum threshold in radians. A vertex is a spike when its incident angles sum below it.
        Compare against ``2 * pi``, the flat value.
    max_iter
        Cap on the number of passes.
    return_count
        If ``True``, also return ``flattened``.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Repaired positions on ``vertices.device``. Connectivity is untouched, so ``faces`` stays
        valid.
    flattened : int, optional
        Present when ``return_count=True``. How many vertex moves were made, summed over the passes
        -- so a vertex fixed twice counts twice, which makes it a report rather than a count of
        anything in the output. Zero means nothing was flagged and the buffer is the input's. The
        pass loop stops itself as soon as a pass finds no spike, so the default return is a bare
        position buffer that drops into a filter chain like every other function here.

    Raises
    ------
    ValueError
        If ``max_iter`` is negative.

    See Also
    --------
    [`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average]
        The unrestricted version of the move this makes, over every vertex.
    [`vertex_defects`][triwarp.vertices.vertex_defects]
        ``2 * pi`` minus the same angle sum, which is what the threshold is read against.
    [`validation.face_defective_mask`][triwarp.validation.face_defective_mask]
        Flags the *faces* a spike produces, where this flags the vertex itself.
    """
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    device = faces.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or max_iter == 0:
        cloned = wp.clone(vertices)
        return (cloned, 0) if return_count else cloned

    positions = wp.clone(vertices)
    flattened = 0
    spikes = wp.empty(n_vertices, dtype=wp.bool, device=device)
    # Built once and handed to every pass. ``filter_neighborhood_average`` would build it per call,
    # and at ``equal_weight=True`` -- its default, and what ``_resolved_operator`` asks for -- the
    # operator reads no positions at all: every off-diagonal weight is ``1`` before the row
    # normalization, so it is the mesh's *topology*, which no pass changes. Hoisting it is therefore
    # exactly equivalent, and it is where this row's cost was: 10 builds at 1.27 ms is **13.0 of the
    # call's 19.6 ms** on ``bunny_decimated`` (RTX 5090), and a build is flat in the mesh, which is
    # what made the whole row flat in the mesh -- 15.65 ms at 16 301 faces against 15.16 at 69 630.
    operator = laplacian.laplacian(vertices, faces, symmetric=True)
    for _ in range(max_iter):
        defects = tw.vertices.vertex_defects(
            n_vertices, faces, tw.triangles.face_angles(positions, faces)
        )
        # The spike test is ``angle_sum < min_angle_sum``, applied to the angle *defect*
        # ``2 * pi - angle_sum`` instead: the condition becomes ``defect > 2 * pi - min_angle_sum``,
        # and the defect is the quantity ``vertices.vertex_defects`` already returns -- re-deriving
        # the sum would mean scattering the same corner angles a second time.
        wp.map(kernel_array.greater, defects, wp.float32(2.0 * math.pi - min_angle_sum), out=spikes)
        # One readback per pass, and it is the stopping test: whether any vertex is still a spike is
        # a device-side fact that a Python loop cannot branch on otherwise.
        n_spikes = int(tw.reduce.sum(tw.array.astype(spikes, wp.int32)))
        if n_spikes == 0:
            break
        smoothed = filter_neighborhood_average(
            positions, faces, iterations=1, laplacian_operator=operator
        )
        wp.map(kernel_smoothing.select_position, smoothed, positions, spikes, out=positions)
        flattened += n_spikes
    return (positions, flattened) if return_count else positions


def equalize_triangle_areas(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    iterations: int = 1,
    force: float = 0.5,
    no_shrinkage: bool = False,
    region: wp.array[wp.bool] | None = None,
    max_displacement: float | None = None,
    *,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None = None,
) -> wp.array[wp.vec3]:
    """
    Even out triangle *areas* by moving vertices, without touching the connectivity.

    Every other filter in this module drives the surface toward a Laplacian residual of zero, which
    equalizes *edge* directions and says nothing about area: a patch of long thin triangles beside a
    patch of fat ones is already Laplacian-smooth. This one minimizes the summed squared areas of
    each vertex's incident triangles instead, which is the objective that actually redistributes
    them -- a vertex sitting close to one of its opposite edges is pushed away from it.

    Twice a triangle's area is ``|(x - p) x (q - p)|`` for the free vertex ``x`` over its opposite
    edge ``(p, q)``, so the per-vertex objective is a sum of quadratic forms and its minimum is one
    3x3 solve, taken in ``float64``. Each pass steps a fraction ``force`` of the way there, from the
    *previous* pass's positions, so the whole mesh moves at once and the result does not depend on
    a vertex ordering.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    iterations
        Number of passes. ``0`` returns a copy.
    force
        Fraction of the way to the per-vertex minimum each pass moves. ``1.0`` jumps straight there
        and oscillates on an irregular mesh; the default is the usual under-relaxation.
    no_shrinkage
        Constrain each vertex to the tangent plane through its current position, so it slides
        across the surface instead of sinking into it. The unconstrained minimum of *squared* area
        pulls the whole 1-ring inward, so leave this on when the shape matters and off when only
        the triangle quality does.
    region
        ``(n_vertices,)`` boolean mask of the vertices allowed to move. ``None`` moves every one.
        Vertices outside it stay exactly where they are, and are still read as neighbours.
    max_displacement
        Clamp every vertex into a ball of this radius around its **input** position, applied after
        each pass. ``None`` leaves the relaxation unbounded. Use it when the surface has to stay
        within a tolerance of the scan it came from.
    vertex_faces
        Optional precomputed [`vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency] as
        ``(offsets, vertex_faces)``. Depends on the connectivity alone -- which this filter never
        changes -- so one CSR serves every call over the same mesh, and
        [`Trimesh.vertex_face_adjacency`][triwarp.mesh.Trimesh.vertex_face_adjacency] has it cached.

    Returns
    -------
    wp.array[wp.vec3]
        Relaxed ``(n_vertices,)`` positions on ``vertices.device``. Connectivity is untouched, so
        ``faces`` stays valid.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, if ``max_displacement`` is negative, or if ``region`` is not
        a length-``n_vertices`` ``wp.bool`` array.

    !!! note "It equalizes areas, not shapes"
        Nothing here bounds a triangle's aspect ratio, and a long thin triangle can have exactly the
        right area. For shape, flip the triangulation
        ([`flip_by_objective`][triwarp.remesh.flip_by_objective]) or remesh
        ([`isotropic_remesh`][triwarp.remesh.isotropic_remesh]); this moves vertices only.

    See Also
    --------
    [`relax_keep_volume`][triwarp.smoothing.relax_keep_volume]
        The other member of this group: a uniform relax that does not shrink the shape.
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
        The plain diffusion this is the area-driven alternative to.
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]
        Equalizes edge *lengths*, by changing the connectivity as well.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    flags, limit = _relaxation_state(vertices, iterations, region, max_displacement)
    if flags is None:
        return wp.clone(vertices)

    offsets, vf_indices = (
        vertex_faces
        if vertex_faces is not None
        else tw.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    )
    positions = wp.clone(vertices)
    normals = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    nxt = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    for _ in range(iterations):
        if no_shrinkage:
            # Recomputed per pass rather than hoisted: the tangent plane the solve is confined to is
            # the one through the vertex *now*, and after a pass that is a different plane.
            normals = tw.vertices.vertex_normals(positions, faces)
        wp.launch(
            kernel_smoothing.equalize_area_step,
            dim=n_vertices,
            inputs=[
                positions,
                faces,
                offsets,
                vf_indices,
                normals,
                flags,
                vertices,
                wp.float32(force),
                no_shrinkage,
                limit,
                nxt,
            ],
            device=device,
        )
        positions, nxt = nxt, positions
    return positions


def relax_keep_volume(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    iterations: int = 1,
    force: float = 0.5,
    region: wp.array[wp.bool] | None = None,
    max_displacement: float | None = None,
) -> wp.array[wp.vec3]:
    """
    Uniform relaxation with the shrinkage taken out **locally**, rather than rescaled away.

    A plain 1-ring average moves every vertex toward the inside of the surface it curves around,
    so a closed shape loses volume with every pass.
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]'s ``volume_constraint`` answers that
    by rescaling the whole mesh at the end, which restores the total and is wrong everywhere the
    shrinkage was not uniform. This answers it per vertex instead: the relax displacement is
    computed as a *field*, and each vertex then has its own neighbourhood's **average
    displacement** subtracted from its own.

    That difference is the whole method. A translation shared by a neighbourhood -- which is what
    shrinkage locally is -- cancels exactly, while the high-frequency part of the displacement,
    which is the noise the relax was for, survives untouched. No global rescale, and a mesh with a
    flat region and a curved one is treated correctly in both.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    iterations
        Number of passes. ``0`` returns a copy.
    force
        Fraction of the way to the 1-ring average each pass moves before the correction.
    region
        ``(n_vertices,)`` boolean mask of the vertices allowed to move. ``None`` moves every one.
        A neighbour outside the region contributes its position to the average but no displacement
        to the correction, so a vertex on the region's edge is corrected by less than an interior
        one -- which is what stops the region from tearing away from the rest.
    max_displacement
        Clamp every vertex into a ball of this radius around its **input** position, applied after
        each pass. ``None`` leaves the relaxation unbounded.

    Returns
    -------
    wp.array[wp.vec3]
        Relaxed ``(n_vertices,)`` positions on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, if ``max_displacement`` is negative, or if ``region`` is not
        a length-``n_vertices`` ``wp.bool`` array.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
        The plain diffusion, whose ``volume_constraint`` is the *global* rescale this replaces.
    [`filter_taubin`][triwarp.smoothing.filter_taubin]
        The other anti-shrinkage answer: alternate a shrinking pass with an inflating one.
    [`equalize_triangle_areas`][triwarp.smoothing.equalize_triangle_areas]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    flags, limit = _relaxation_state(vertices, iterations, region, max_displacement)
    if flags is None:
        return wp.clone(vertices)

    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    adjacency = tw.graph.edges_to_csr(n_vertices, unique_edges)
    offsets, columns = adjacency.offsets, adjacency.columns
    positions = wp.clone(vertices)
    push = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    nxt = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    for _ in range(iterations):
        wp.launch(
            kernel_smoothing.ring_push_forces,
            dim=n_vertices,
            inputs=[positions, offsets, columns, flags, wp.float32(force), push],
            device=device,
        )
        wp.launch(
            kernel_smoothing.apply_push_keeping_volume,
            dim=n_vertices,
            inputs=[positions, offsets, columns, flags, push, vertices, limit, nxt],
            device=device,
        )
        positions, nxt = nxt, positions
    return positions


def relax_approx(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dilate_radius: float,
    iterations: int = 1,
    force: float = 0.5,
    fit: Literal["planar", "quadric"] = "planar",
    region: wp.array[wp.bool] | None = None,
    max_displacement: float | None = None,
) -> wp.array[wp.vec3]:
    """
    Move each vertex onto a surface fitted to its **neighbourhood**, rather than toward its 1-ring.

    Every diffusion filter here is a 1-ring operator, so the largest feature it can see is one edge
    across and its idea of "where the surface is" is the average of six neighbours. This fits a
    local surface -- a least-squares plane, or a quadric graph over that plane -- to every vertex
    inside a geodesic ball of ``dilate_radius`` and moves the vertex onto it. The ball is what makes
    it different in kind: noise smaller than the radius is averaged out in one pass instead of being
    diffused away over many, and the fitted surface is not pulled off the shape by the vertex being
    fitted.

    ``fit="planar"`` flattens: it is the strongest of the two and will take the curvature out of a
    genuinely curved surface if the radius is large. ``fit="quadric"`` keeps curvature, because a
    curved surface *is* in its model space, and is the one to use on anything but a plate.

    The neighbourhood is a **geodesic** ball ([`geodesic_ball`][triwarp.neighbors.geodesic_ball]),
    not a Euclidean one, so the opposite wall of a thin tube is excluded even when it is close --
    which is exactly the case that corrupts a fit.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dilate_radius
        Radius of the neighbourhood ball, in world units. It has no scale-free default: below one
        edge length the ball is the vertex alone and the call is a no-op, and the useful range
        starts at a small multiple of the mean edge length
        ([`edges_unique_length`][triwarp.edges.edges_unique_length]).
    iterations
        Number of passes. ``0`` returns a copy. The neighbourhoods are built **once**, from the
        input positions, and reused -- the ball's membership is a topological choice and rebuilding
        it per pass would cost more than the fit.
    force
        Fraction of the way to the fitted surface each pass moves.
    fit
        ``"planar"`` for a least-squares plane, ``"quadric"`` for a 6-coefficient quadric graph over
        it.
    region
        ``(n_vertices,)`` boolean mask of the vertices allowed to move. ``None`` moves every one.
        Vertices outside it still populate their neighbours' balls.
    max_displacement
        Clamp every vertex into a ball of this radius around its **input** position, applied after
        each pass. ``None`` leaves the relaxation unbounded.

    Returns
    -------
    wp.array[wp.vec3]
        Relaxed ``(n_vertices,)`` positions on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, if ``dilate_radius`` is not positive, if
        ``max_displacement`` is negative, if ``fit`` is not one of the two names, or if ``region``
        is not a length-``n_vertices`` ``wp.bool`` array.

    !!! note "A vertex with too small a ball does not move"
        A plane fit needs three independent points and the quadric six, so a vertex whose ball holds
        fewer than six vertices is left exactly where it is rather than fitted to whatever it has.
        On a mesh whose edges are longer than ``dilate_radius`` that is *every* vertex and the call
        returns the input -- check the result moved before concluding the parameters were right.

    See Also
    --------
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
        The 1-ring diffusion this replaces with a neighbourhood fit.
    [`principal_curvature`][triwarp.curvature.principal_curvature]
        Fits a quadric over the same geodesic balls, to measure rather than to move.
    [`geodesic_ball`][triwarp.neighbors.geodesic_ball]
    """
    if fit not in ("planar", "quadric"):
        raise ValueError(f"fit must be 'planar' or 'quadric', got {fit!r}")
    if dilate_radius <= 0.0:
        raise ValueError(f"dilate_radius must be positive, got {dilate_radius}")
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    flags, limit = _relaxation_state(vertices, iterations, region, max_displacement)
    if flags is None:
        return wp.clone(vertices)

    # Built once from the input positions: which vertices are in the ball is a decision about the
    # surface's connectivity, and the relaxation moves everything by less than the radius anyway.
    # ``min_count=1`` disables the nearest-neighbour backfill: a ball too small to fit is a
    # documented no-op here, and backfilling would quietly fit something else instead.
    neighbor_indices, neighbor_offsets, _ = tw.neighbors.geodesic_ball(
        vertices, faces, dilate_radius, min_count=1
    )
    positions = wp.clone(vertices)
    nxt = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    for _ in range(iterations):
        wp.launch(
            kernel_smoothing.relax_approx_step,
            dim=n_vertices,
            inputs=[
                positions,
                neighbor_indices,
                neighbor_offsets,
                flags,
                vertices,
                wp.float32(force),
                fit == "quadric",
                limit,
                nxt,
            ],
            device=device,
        )
        positions, nxt = nxt, positions
    return positions


def _relaxation_state(
    vertices: wp.array[wp.vec3],
    iterations: int,
    region: wp.array[wp.bool] | None,
    max_displacement: float | None,
) -> tuple[wp.array[wp.bool] | None, wp.float32]:
    """
    Validate the axes the relaxation family shares, and materialize the region mask.

    Returns ``(None, ...)`` when there is nothing to do, so each caller's early return is one line.
    The displacement limit is carried as a ``float32`` with **negative meaning unbounded**, which is
    how ``None`` crosses into a kernel without a second launch path.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    if max_displacement is not None and max_displacement < 0.0:
        raise ValueError(f"max_displacement must be non-negative, got {max_displacement}")
    n_vertices = int(vertices.shape[0])
    limit = wp.float32(-1.0 if max_displacement is None else max_displacement)
    if n_vertices == 0 or iterations == 0:
        return None, limit
    if region is None:
        return wp.full(n_vertices, True, dtype=wp.bool, device=vertices.device), limit
    if len(region.shape) != 1 or region.shape[0] != n_vertices or region.dtype is not wp.bool:
        raise ValueError(
            f"region must be a length-{n_vertices} wp.bool array, got shape {tuple(region.shape)} "
            f"of {region.dtype}"
        )
    return region, limit


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
        return wp.clone(vertices)

    operator = _resolved_operator(vertices, faces, laplacian_operator)
    positions = _as_vec3d(vertices)
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

    return _as_vec3(positions)


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
        return wp.clone(vertices)

    # Symmetric adjacency (undirected 1-ring) so boundary vertices average over all their
    # neighbors, matching Open3D's ``adjacency_list``; the directed default is asymmetric there.
    operator = _resolved_operator(vertices, faces, laplacian_operator, symmetric=True)
    positions = _as_vec3d(vertices)
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

    return _as_vec3(positions)


def filter_mut_dif_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    lamb: float = 0.5,
    iterations: int = 10,
    volume_constraint: bool = True,
    laplacian_operator: wps.BsrMatrix[wp.float32] | None = None,
    *,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
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
    face_normals
        Optional length-``n_faces`` unit face normals and matching areas from
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][triwarp.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][triwarp.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        See ``face_normals``.

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
        return wp.clone(vertices)

    operator = _resolved_operator(vertices, faces, laplacian_operator)
    positions = _as_vec3d(vertices)

    # Vertex normals and eps are computed once from the input mesh and reused every pass, matching
    # the trimesh reference (which reads normals off the un-mutated mesh inside its loop).
    if face_normals is None or face_areas is None:
        face_normals, face_areas = face_normals_and_areas(vertices, faces)
    normals = mean_vertex_normals(n, faces, face_normals)
    vol_ini = tw.measures.volume(positions, faces) if volume_constraint else 0.0
    eps = 0.01 * float(tw.reduce.max(face_areas)) ** 0.5 if volume_constraint else 0.0

    lv = wp.empty(n, dtype=wp.vec3d, device=device)
    adil = wp.empty(n, dtype=wp.float64, device=device)
    adil_sum = wp.zeros(1, dtype=wp.float64, device=device)
    nxt = wp.empty(n, dtype=wp.vec3d, device=device)
    probe = wp.empty(n, dtype=wp.vec3d, device=device) if volume_constraint else None
    slope = 0.0
    inv_n = wp.float64(1.0 / n)
    n_blocks = kernel_reduce.blocks_1d(n)
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
            dim=[n_blocks],
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
            vol = tw.measures.volume(positions, faces)
            if index == 0:
                wp.map(
                    kernel_smoothing.add_scaled_normal,
                    positions,
                    normals,
                    wp.float64(eps),
                    out=probe,
                )
                vol2 = tw.measures.volume(probe, faces)
                slope = eps / (vol2 - vol) if vol2 != vol else 0.0
            wp.map(
                kernel_smoothing.add_scaled_normal,
                positions,
                normals,
                wp.float64(slope * (vol_ini - vol)),
                out=positions,
            )

    return _as_vec3(positions)


def _apply_operator(
    operator: wps.BsrMatrix[wp.float32], v_in: wp.array[wp.vec3d], out_lv: wp.array[wp.vec3d]
) -> None:
    wp.launch(
        kernel_laplacian.apply_operator,
        dim=int(v_in.shape[0]),
        inputs=[operator.offsets, operator.columns, operator.values, v_in, out_lv],
        device=v_in.device,
    )


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
    libigl tutorial 205) and is **CUDA only** (raises on
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
        return wp.clone(vertices)

    positions = _as_vec3d(vertices)
    components = _empty_components(n, device)
    rhs = _empty_components(n, device)
    solutions = _empty_components(n, device)

    # Boundary topology is fixed for the whole flow, so the partition is built once. ``None`` means
    # "solve over every vertex" -- either the caller asked for the unconstrained flow, or the mesh
    # is closed and there is nothing to pin.
    dirichlet = _dirichlet_state(vertices, faces, positions, n, device) if pin_boundary else None

    for _ in range(iterations):
        current = _as_vec3(positions)
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

    return _as_vec3(positions)


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
        twt.as_array2d(pinned, wp.float64),
        twt.as_array2d(wp.zeros((3, n_free), dtype=wp.float64, device=device), wp.float64),
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
    # ``nnz_sync()``, never ``operator.nnz``: after ``bsr_from_triplets`` the ``nnz`` field is a
    # stale cache holding the triplet *capacity* it was handed, duplicates included, and only a
    # ``nnz_sync()`` repairs it (no other operation does, so whether ``nnz`` reads correctly depends
    # on unrelated earlier code). The uniform default is duplicate-free so the two agree there, but
    # a caller-supplied ``cotmatrix`` (12 triplets per face) overshoots 3.4x on an ``icosphere(3)``,
    # and that gap in these ``wp.empty`` buffers would reach ``bsr_from_triplets`` uninitialized.
    # Out-of-range garbage indices are dropped silently, but any landing in ``[0, n)`` accumulate a
    # garbage value into a real entry: measured ``‖values‖ = 1.1e13`` against the correct 84.3 with
    # the pool holding plausible indices. One host readback per call, not per pass.
    nnz = operator.nnz_sync()
    n_triplets = nnz + n
    rows, cols, vals = tw.array.triplet_buffers(n_triplets, wp.float64, device)
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


def smooth_region_fixed_rim(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    free_mask: wp.array[wp.bool],
    stabilizer: float = 0.0,
) -> wp.array[wp.vec3]:
    """
    Reposition a free vertex region as the umbrella-Laplacian solution with a sharp fixed boundary.

    The free vertices (``free_mask``) are moved to the solution of the graph-Laplacian Dirichlet
    system ``(D - W) x = b`` (unit edge weights), where the fixed one-ring neighbours are folded
    into ``b``, so the region rim stays sharp
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

    See Also
    --------
    [`smooth_region`][triwarp.smoothing.smooth_region]
    [`smooth_region_boundary`][triwarp.smoothing.smooth_region_boundary]
        The third case: smooths the rim's own path rather than the surface on either side of it.
    [`fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    The system is SPD only when no free connected component is entirely free (or ``stabilizer >
    0``); the hole-filling pipeline guarantees this because the patch rim is always fixed.

    A free vertex that no face refers to contributes no row, so the solve does not constrain it
    and it is returned where it was. That is what the ``solution`` seed decides rather than the
    system: it is seeded with the current positions, not with zeros.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    out = wp.clone(vertices)
    if int(faces.shape[0]) == 0 or n == 0:
        return out
    free_map, n_free = tw.array.mask_to_index_map(free_mask)
    if n_free == 0:
        return out

    weight_matrix = _edge_weight_matrix(vertices, faces, "unit")
    nnz = int(weight_matrix.nnz)
    size = nnz + n
    # ``dirichlet_system_triplets`` emits conditionally, so most slots stay unwritten. Padding must
    # be a *hole*, not a value: a ``(0, 0, 0.0)`` triplet is a harmless structural zero, but every
    # one of them accumulates onto entry ``(0, 0)`` and ``bsr_from_triplets``' accumulation atomic
    # serializes them. An out-of-range index is dropped instead -- ``n_free`` is one past the last
    # row and column of the ``(n_free, n_free)`` system.
    out_rows = wp.empty(size, dtype=wp.int32, device=device)
    out_cols = wp.empty(size, dtype=wp.int32, device=device)
    out_rows.fill_(n_free)
    out_cols.fill_(n_free)
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
    sol = _free_positions(vertices, free_mask, free_map, n_free)
    twl.solve_spd_columns(
        system,
        twt.as_array2d(rhs, wp.float64),
        twt.as_array2d(sol, wp.float64),
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


def smooth_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    free_mask: wp.array[wp.bool],
    edge_weights: str = "cotan",
) -> wp.array[wp.vec3]:
    """
    Reposition a free vertex region so the surface is smooth across the region boundary too.

    Every vertex in the region and its first fixed ring contributes the umbrella equation
    ``p_v = Σ_d (w_vd / ΣW) p_d``; free vertices are unknowns and fixed neighbours move to the
    right-hand side. The normal equations ``(MᵀM) x = Mᵀ b`` are solved per
    coordinate, giving a patch that is smooth (C¹) *across* the region rim, unlike the sharp-rim
    [`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim].

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
    ValueError
        If ``edge_weights`` is not ``"cotan"`` or ``"unit"``.

    See Also
    --------
    [`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim]
    [`smooth_region_boundary`][triwarp.smoothing.smooth_region_boundary]
        The third case: smooths the rim's own path rather than the surface on either side of it.
    [`fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    A free vertex that no face refers to contributes no row, so the least-squares system does not
    constrain it and it is returned where it was. That is what the ``solution`` seed decides rather
    than the system: it is seeded with the current positions, not with zeros.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    out = wp.clone(vertices)
    if int(faces.shape[0]) == 0 or n == 0:
        return out
    free_map, n_free = tw.array.mask_to_index_map(free_mask)
    if n_free == 0:
        return out

    row_mask = tw.selection.expand_vertex_mask(faces, free_mask, 1)
    row_map, n_rows = tw.array.mask_to_index_map(row_mask)
    weight_matrix = _edge_weight_matrix(vertices, faces, edge_weights)
    nnz = int(weight_matrix.nnz)
    size = nnz + n
    # ``laplacian_ls_triplets`` emits conditionally, so most slots stay unwritten. Padding must be a
    # *hole*, not a value: a ``(0, 0, 0.0)`` triplet is a harmless structural zero, but every one of
    # them accumulates onto entry ``(0, 0)`` and ``bsr_from_triplets``' accumulation atomic
    # serializes them -- measured 14.3 ms against 0.45 ms (31.8x) for the ~185 000 unwritten slots
    # of a 35 947-vertex mesh. Both arrays are filled one past their own extent, so the padding is
    # out of range as the *row* index of ``M`` (``rows``) and of ``M^T`` (``cols``) alike.
    rows = wp.empty(size, dtype=wp.int32, device=device)
    cols = wp.empty(size, dtype=wp.int32, device=device)
    rows.fill_(n_rows)
    cols.fill_(n_free)
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
    # each (both single builds, so bsr_mm gets operands whose nnz is exact), then A = M^T M is SPD.
    m_matrix = wps.bsr_from_triplets(n_rows, n_free, rows, cols, vals, prune_numerical_zeros=False)
    mt_matrix = wps.bsr_from_triplets(
        n_free, n_rows, wp.clone(cols), wp.clone(rows), wp.clone(vals), prune_numerical_zeros=False
    )
    system = wps.bsr_mm(mt_matrix, m_matrix)
    # A^T b straight into the rows of one contiguous buffer, so the three columns batch.
    atb = wp.zeros((3, n_free), dtype=wp.float64, device=device)
    for column, component in enumerate((rhs_x, rhs_y, rhs_z)):
        wps.bsr_mv(mt_matrix, component, atb[column], alpha=1.0, beta=0.0)
    sol = _free_positions(vertices, free_mask, free_map, n_free)
    # The normal equations are the worst-conditioned system this package solves -- Jacobi-
    # preconditioned conjugate gradient takes 6 541 iterations on ``bunny``'s benchmarked free
    # region -- and the solve is 99 % of the call, so a multigrid hierarchy is worth its setup
    # there: the *solve* alone measures 2.46x, and 2.55x on the CPU device.
    #
    # ``"auto"`` rather than ``"multigrid"`` because *this same function* is also called on hole
    # patches by ``refine_and_smooth_region``, whose systems are small. Asking for the hierarchy
    # unconditionally took ``holes.fill_smooth`` to **0.21x** on ``holes_many``: every patch paid a
    # 15 ms setup to save nothing.
    #
    # What separates a patch from a region is *not* what three earlier attempts looked for. Size,
    # Jacobi iteration count and probe-rate extrapolation all interleave the two classes, and a
    # size threshold on its own is refuted outright by this function's own population: a patch of
    # 1 000 unknowns loses under a forced hierarchy while ``bunny_decimated``'s 2 043-unknown
    # region wins 2.23x. The axis that does separate them is the operator's off-diagonal dominance
    # crossed with a size floor, which is what ``"auto"`` now gates on -- ``linalg``'s
    # ``CG_MULTIGRID_DOMINANCE`` carries all 29 systems and the one 1.26x it forgoes. End to end
    # against forcing Jacobi: **2.02x** on ``bunny_decimated``'s benchmarked region, 1.80x on
    # ``bunny``'s and 1.58x on a 10 239-unknown icosphere, with all four hole-patch chains flat at
    # 0.98-1.04x.
    twl.solve_spd_columns(
        system,
        twt.as_array2d(atb, wp.float64),
        twt.as_array2d(sol, wp.float64),
        tol=twl.CG_TOLERANCE,
        maxiter=10 * n_free,
        preconditioner="auto",
    )
    wp.launch(
        kernel_smoothing.scatter_free_solution,
        dim=n,
        inputs=[free_mask, free_map, sol[0], sol[1], sol[2], out],
        device=device,
    )
    return out


# ---------------------------------------------------------------------------
# Region smoothing solves (positionVertsSmoothly / positionVertsSmoothlySharpBd)
# ---------------------------------------------------------------------------


def _free_positions(
    vertices: wp.array[wp.vec3],
    free_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    n_free: int,
) -> wp.array[wp.float64]:
    """
    Gather the free vertices' current positions as a ``(3, n_free)`` float64 initial guess.

    Both region solves ask conjugate gradient for the free vertices' *new* positions, and
    [`solve_spd_columns`][triwarp.linalg.solve_spd_columns] takes its ``solution`` argument as the
    initial guess, so this is what the solver starts from.

    It is a **correctness** requirement before it is a warm start. A vertex no face refers to
    contributes no row to either system, so CG never writes its entry and it keeps whatever the
    seed held: from ``wp.zeros`` such a vertex is silently moved to the origin -- 7 of
    ``bunny_decimated``'s free vertices and 297 of ``bunny``'s 1 113 unreferenced ones, up to 60 %
    of the bounding-box diagonal. Seeded from the current positions it stays put, which is the only
    defensible answer for an unknown the system does not constrain.

    The speed is the smaller half and is worth stating so nobody expects more of it: the stopping
    test is relative to the right-hand side's norm rather than to the initial residual, so the seed
    cannot change which answers count as converged, only how far CG has to travel. Measured
    interleaved, ``min`` of 5 -- iterations 1 784 -> 1 708 on ``bunny_decimated``'s benchmarked
    region, 6 541 -> 6 167 on ``bunny``'s under Jacobi and 310 -> 263 under the V-cycle, which is
    **1.04-1.06x** on the whole ``smooth_region`` / ``smooth_region_fixed_rim`` call, reproduced
    across two sessions. On the hole-filling chains it is **flat**: two sessions measured
    ``refill_region`` at 1.02x and 0.94x on the same row, which is that chain's own run-to-run
    spread rather than a gain or a regression, so do not attribute either to this.
    """
    device = vertices.device
    guess = wp.empty((3, n_free), dtype=wp.float64, device=device)
    wp.launch(
        kernel_smoothing.gather_free_positions,
        dim=int(vertices.shape[0]),
        inputs=[free_mask, free_map, vertices, guess[0], guess[1], guess[2]],
        device=device,
    )
    return guess


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
    rows, cols, vals = tw.array.triplet_buffers(2 * m, wp.float64, device)
    wp.launch(
        kernel_smoothing.symmetric_weight_triplets,
        dim=m,
        inputs=[unique_edges, weights, rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(n, n, rows, cols, vals, prune_numerical_zeros=False)


def refine_and_smooth_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_vertices_before: int,
    patch_face_mask: wp.array[wp.bool],
    max_edge: float,
    max_edge_splits: int,
    max_angle_change_after_flip: float,
    smooth_curvature: bool,
    smooth_boundary: bool,
    natural_smooth: bool,
    edge_weights: str,
    *,
    refine: Literal["max_edge", "density"] = "max_edge",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Subdivide the patch and smooth its new vertices.

    Shared finisher of [`fill_smooth`][triwarp.holes.fill_smooth] and
    [`stitch_smooth`][triwarp.holes.stitch_smooth].

    ``refine`` selects the subdivision criterion, and the two are genuinely different questions
    rather than two tunings of one: ``"max_edge"`` bisects region edges longer than ``max_edge``
    ([`remesh.subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]), and
    ``"density"`` splits region triangles at their centroid while their sampling is coarser than
    the surrounding mesh's -- Liepa's criterion, in
    [`remesh.refine_region_to_density`][triwarp.remesh.refine_region_to_density]. ``max_edge`` and
    ``max_edge_splits`` are ignored under ``"density"``, which takes its target from the mesh
    instead of from an argument.

    Raises
    ------
    ValueError
        If ``refine`` is neither ``"max_edge"`` nor ``"density"``.
    """
    device = faces.device
    if refine == "density":
        vertices, faces, patch_face_mask = tw.remesh.refine_region_to_density(
            vertices, faces, patch_face_mask, max_angle_change=max_angle_change_after_flip
        )
    elif refine == "max_edge":
        vertices, faces, patch_face_mask = tw.remesh.subdivide_region_to_size(
            vertices,
            faces,
            patch_face_mask,
            max_edge=max_edge,
            max_splits=max_edge_splits,
            max_angle_change=max_angle_change_after_flip,
        )
    else:
        raise ValueError(f"unknown refine {refine!r}, expected 'max_edge' or 'density'")
    if not smooth_curvature:
        return vertices, faces, patch_face_mask

    n = int(vertices.shape[0])
    # New (interior patch) vertices are the tail appended by subdivision, minus mesh-boundary verts.
    new_verts = wp.zeros(n, dtype=wp.bool, device=device)
    # Warp rejects a zero-length slice, and subdivision may have added no vertices at all.
    if n > n_vertices_before:
        new_verts[n_vertices_before:].fill_(True)
    bd_mask = _boundary_verts_mask(vertices, faces)
    free = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_and_not, new_verts, bd_mask, out=free)

    vertices = smooth_region_fixed_rim(vertices, faces, free)
    if smooth_boundary:
        vertices = smooth_region(vertices, faces, free, edge_weights)

    if natural_smooth:
        edges_bd = tw.selection.region_boundary_edges(faces, patch_face_mask, n_vertices=n)
        endpoints = wp.clone(twt.as_array2d(edges_bd, wp.int32).reshape(-1))
        incident = tw.array.indices_to_mask(endpoints, n, device=device)
        incident = tw.selection.expand_vertex_mask(faces, incident, 5)
        incident = tw.selection.shrink_vertex_mask(faces, incident, 2)
        incident = tw.selection.exclude_fully_selected_components(faces, incident, n)
        # A 1-byte-per-vertex copy is cheap, so this is the latest crossover of the reduction
        # family: measured on CUDA 0.65x at 400k vertices, 1.09x at 1M, 1.65x at 2M, 3.23x at 8M.
        # Taken on the CUDA number per the GPU-first rule; the losing sizes are sub-0.1 ms.
        if tw.reduce.any(incident):
            bd_mask = _boundary_verts_mask(vertices, faces)
            free2 = wp.empty(n, dtype=wp.bool, device=device)
            wp.map(kernel_array.mask_and_not, incident, bd_mask, out=free2)
            vertices = smooth_region_fixed_rim(vertices, faces, free2)
            vertices = smooth_region(vertices, faces, free2, edge_weights)

    return vertices, faces, patch_face_mask


def smooth_region_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    iterations: int = 4,
    *,
    vertex_faces: tuple[wp.array[wp.int32], wp.array[wp.int32]] | None = None,
) -> wp.array[wp.vec3]:
    """
    Straighten the *rim curve* of a face region, by sliding the vertices that lie on it.

    The third member of the region family, and the one whose subject is the boundary itself.
    [`smooth_region`][triwarp.smoothing.smooth_region] smooths the surface *across* the rim and
    [`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim] holds the rim still while
    smoothing inside it; both leave the rim's own path through the mesh exactly where the face
    selection put it. That path is usually ragged -- a selection by angle, by height or by a paint
    stroke follows triangle edges and zigzags -- and this is what makes it a smooth curve, without
    moving the surface off itself in any visible way.

    The rim is smoothed as a **level set** rather than as a polyline. A field is pinned to ``-1`` on
    the region's vertices and ``+1`` outside, its harmonic interpolation is solved over the band of
    vertices that touch both sides, and each of those vertices is moved onto the zero level set of
    the result. Harmonic interpolation is smooth, so its zero set is a smooth curve; the vertices
    slide along the surface to sit on it. Repeating recomputes the field from the moved positions,
    and the move is damped so the sequence settles rather than oscillates.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    region
        ``(n_faces,)`` boolean mask selecting the region whose rim is smoothed. The rim is derived
        from it; only vertices that touch both a selected and an unselected face may move.
    iterations
        Number of solve-and-project passes. ``0`` returns a copy.
    vertex_faces
        Optional precomputed [`vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency] as
        ``(offsets, vertex_faces)``. Depends on the connectivity alone -- which this filter never
        changes -- so one CSR serves every call over the same mesh, and
        [`Trimesh.vertex_face_adjacency`][triwarp.mesh.Trimesh.vertex_face_adjacency] has it cached.

    Returns
    -------
    wp.array[wp.vec3]
        Positions on ``vertices.device`` with the rim band slid along the surface. Connectivity and
        the face mask are untouched, so ``region`` stays valid against the result.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, or if ``region`` is not a length-``n_faces`` ``wp.bool``
        array.

    !!! note "It slides vertices; it does not re-cut the rim"
        The rim can only become as smooth as the *existing* triangulation lets it: a vertex slides
        to the level set but the rim still passes through the same vertices, so a curve that wants
        to cross a triangle diagonally cannot. Retriangulating the band first
        ([`flip_by_objective`][triwarp.remesh.flip_by_objective], or
        [`split_faces_along_field`][triwarp.intersection.split_faces_along_field] to cut along the
        level set outright) is the way to get past that, and is deliberately not folded in here --
        it changes the face buffer, which this promises not to.

    See Also
    --------
    [`smooth_region`][triwarp.smoothing.smooth_region]
        Smooths the surface across the rim, leaving the rim's path alone.
    [`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim]
        Smooths inside the rim, holding it fixed.
    [`region_boundary_edges`][triwarp.selection.region_boundary_edges]
        The rim as an edge list, which is what this leaves in a better place.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if len(region.shape) != 1 or region.shape[0] != n_faces or region.dtype is not wp.bool:
        raise ValueError(
            f"region must be a length-{n_faces} wp.bool array, got shape {tuple(region.shape)} "
            f"of {region.dtype}"
        )
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or n_faces == 0 or iterations == 0:
        return wp.clone(vertices)

    inside = _incident_vertex_mask(faces, region, n_vertices)
    free = _region_rim_vertices(faces, region, inside)
    fixed_mask = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, free, out=fixed_mask)
    field = wp.empty((1, n_vertices), dtype=wp.float64, device=device)
    wp.map(kernel_smoothing.region_side_value, inside, out=field[0])

    offsets, vf_indices = (
        vertex_faces
        if vertex_faces is not None
        else tw.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    )
    positions = wp.clone(vertices)
    nxt = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    solved = wp.empty(n_vertices, dtype=wp.float64, device=device)
    for _ in range(iterations):
        # The cotangent weights depend on the positions, so the field is re-solved every pass; the
        # pinned +-1 entries are the same each time and only the band's values change.
        operator = wps.bsr_scale(laplacian.cotmatrix(positions, faces, dtype=wp.float64), -1.0)
        solution, free_map, n_free = twl.min_quad_with_fixed(
            operator, fixed_mask, twt.as_array2d(field, wp.float64)
        )
        if n_free == 0:
            break
        wp.copy(solved, field[0])
        wp.launch(
            kernel_smoothing.scatter_free_scalar,
            dim=n_vertices,
            inputs=[fixed_mask, free_map, solution[0], solved],
            device=device,
        )
        wp.launch(
            kernel_smoothing.project_to_zero_isoline,
            dim=n_vertices,
            inputs=[positions, faces, offsets, vf_indices, solved, free, _ISOLINE_DAMPING, nxt],
            device=device,
        )
        positions, nxt = nxt, positions
    return positions


def _incident_vertex_mask(
    faces: wp.array[wp.int32], region: wp.array[wp.bool], n_vertices: int
) -> wp.array[wp.bool]:
    """Mark the vertices touched by at least one selected face."""
    mask = wp.zeros(n_vertices, dtype=wp.bool, device=faces.device)
    wp.launch(
        kernel_smoothing.mark_incident_vertices,
        dim=int(faces.shape[0]) // 3,
        inputs=[faces, region, mask],
        device=faces.device,
    )
    return mask


def _region_rim_vertices(
    faces: wp.array[wp.int32], region: wp.array[wp.bool], inside: wp.array[wp.bool]
) -> wp.array[wp.bool]:
    """
    Mark the band touching both a selected and an unselected face: the vertices free to move.

    A connected component lying entirely in the band is dropped from it: with no vertex pinned, the
    harmonic system over that component has nothing to interpolate and is singular.
    """
    device = faces.device
    n_vertices = int(inside.shape[0])
    outside_region = wp.empty(int(region.shape[0]), dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, region, out=outside_region)
    outside = _incident_vertex_mask(faces, outside_region, n_vertices)
    free = wp.empty(n_vertices, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_and, inside, outside, out=free)

    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    labels = tw.graph.connected_component_labels_from_edges(unique_edges, n_vertices)
    component_size = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_scatter.count_occurrences,
        dim=n_vertices,
        inputs=[labels, component_size],
        device=device,
    )
    free_labels = tw.array.gather(labels, tw.array.flatnonzero(free))
    free_in_component = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    if int(free_labels.shape[0]) > 0:
        wp.launch(
            kernel_scatter.count_occurrences,
            dim=int(free_labels.shape[0]),
            inputs=[free_labels, free_in_component],
            device=device,
        )
    wp.map(
        kernel_smoothing.free_in_mixed_component,
        free,
        tw.array.gather(component_size, labels),
        tw.array.gather(free_in_component, labels),
        out=free,
    )
    return free


def _boundary_verts_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """Length-``n_vertices`` mask of mesh-boundary vertices."""
    device = faces.device
    n = int(vertices.shape[0])
    boundary = tw.boundary.boundary_vertex_indices(vertices, faces)
    return tw.array.indices_to_mask(boundary, n, device=device)


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
    [`shortest_path_envelope`][triwarp.graph.shortest_path_envelope]
    [`filter_laplacian`][triwarp.smoothing.filter_laplacian]
    [`triwarp.laplacian.laplacian`][triwarp.laplacian.laplacian]
    """
    device = values.device
    n = int(vertices.shape[0])
    if int(values.shape[0]) != n:
        raise ValueError(
            f"values must have one entry per vertex, got {values.shape[0]} for {n} vertices"
        )

    out = wp.clone(values)
    if n == 0 or iterations == 0:
        return out

    operator = _resolved_operator(vertices, faces, laplacian_operator, symmetric=True)
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


def filter_normals(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    iterations: int = 20,
    threshold: float = 60.0,
    face_adjacency: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
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
    face_normals
        Optional length-``n_faces`` unit face normals and matching areas from
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][triwarp.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][triwarp.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        See ``face_normals``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_faces`` unit normals on ``vertices.device``. A fresh buffer even when
        ``face_normals`` is passed: the passes rewrite the field in place.

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
    if face_normals is None or face_areas is None:
        face_normals, face_areas = face_normals_and_areas(vertices, faces)
    # Each pass rewrites the normal field in place, so a caller-supplied buffer is copied rather
    # than clobbered -- ``Trimesh.face_normals`` is one array shared with every other reader.
    normals, areas = wp.clone(face_normals), face_areas
    if n_faces == 0 or iterations <= 0:
        return normals

    if face_adjacency is None:
        face_adjacency = tw.adjacency.face_adjacency(faces, n_vertices=int(vertices.shape[0]))
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
    [`filter_sharpen`][triwarp.smoothing.filter_sharpen]

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
    out = wp.clone(vertices)
    if n == 0 or n_faces == 0 or iterations <= 0:
        return out

    # The topology is fixed, so the adjacency is built once for every pass of both halves.
    adjacency = tw.adjacency.face_adjacency(faces, n_vertices=n)
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


def filter_sharpen(
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
    exaggerates every feature. This is **unsharp masking**, the photographic technique, applied to
    vertex positions instead of pixels — MeshLab spells it ``apply_coord_unsharp_mask`` — and it is
    the standard way to recover crispness lost to an earlier smoothing or a decimation.

    The name says what it returns, which is *positions* rather than a mask: the ``*_mask`` family
    in this package is boolean throughout.

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


# ---------------------------------------------------------------------------
# Private cross-cutting helpers
# ---------------------------------------------------------------------------


def _resolved_operator(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    laplacian_operator: wps.BsrMatrix[wp.float32] | None,
    *,
    symmetric: bool = False,
) -> wps.BsrMatrix[wp.float32]:
    """
    Return the caller's row-stochastic operator, or the uniform-weight default built here.

    Every position filter here takes an optional prebuilt operator so a caller running several
    passes pays for the assembly once; this is the one place that decides what ``None`` means.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    laplacian_operator
        Prebuilt operator, or ``None`` to build the uniform-weight one.
    symmetric
        Build the undirected (symmetric) 1-ring operator instead of the directed default. Only
        [`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average] wants this,
        to match Open3D's ``adjacency_list`` at boundary vertices.

    Returns
    -------
    warp.sparse.BsrMatrix
        The operator to diffuse through.
    """
    if laplacian_operator is not None:
        return laplacian_operator
    return laplacian.laplacian(vertices, faces, symmetric=symmetric)


def _as_vec3d(vertices: wp.array[wp.vec3]) -> wp.array[wp.vec3d]:
    """Widen a ``wp.vec3`` array to ``wp.vec3d``: the seam where the float64 solves start."""
    out = wp.empty(int(vertices.shape[0]), dtype=wp.vec3d, device=vertices.device)
    wp.utils.array_cast(vertices, out)
    return out


def _as_vec3(positions: wp.array[wp.vec3d]) -> wp.array[wp.vec3]:
    """Narrow a ``wp.vec3d`` array back to ``wp.vec3``: the seam where the float64 solves end."""
    out = wp.empty(int(positions.shape[0]), dtype=wp.vec3, device=positions.device)
    wp.utils.array_cast(positions, out)
    return out
