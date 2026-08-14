"""
Flatten a mesh into the plane: harmonic, Tutte, ARAP and LSCM parametrizations.

Four maps, differing in what they hold fixed and what they minimize. The first three pin the
boundary and solve for the interior:
[`harmonic`][triwarp.parametrization.harmonic] minimizes Dirichlet energy with cotangent weights,
[`tutte`][triwarp.parametrization.tutte] is the same solve with uniform weights (guaranteeing an
injective map for a convex boundary), and [`arap`][triwarp.parametrization.arap] alternates local
rotation fits with a global solve to trade conformality for low area distortion.
[`lscm`][triwarp.parametrization.lscm] instead pins only two vertices and lets the boundary find its
own shape, minimizing conformal rather than Dirichlet energy.

[`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle] supplies the boundary
condition the fixed-boundary three need, and
[`flipped_faces`][triwarp.parametrization.flipped_faces] is the diagnostic that says whether a
result is actually injective. Ports of the corresponding ``igl::`` routines; every solve is
conjugate-gradient.
"""

from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
import triwarp.typing as twt
from triwarp.kernels import parametrization as kernel_parametrization
from triwarp.laplacian import cotmatrix, cotmatrix_entries, mass_matrix_entries, uniform_laplacian

_CG_TOLERANCE = 1e-8


def flipped_faces_mask(vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a triangle is inverted (negative 2D signed area) in the parametrization.

    For each triangle the 2D signed area of its three UV vertices is computed; a face is flagged
    ``True`` when that area is strictly negative, i.e. the triangle has folded over (flipped
    orientation) in the 2D domain. Mirrors libigl's ``flipped_triangles`` per-triangle test
    (determinant of the homogeneous ``3 x 3`` vertex matrix ``< 0``). Degenerate (zero-area)
    triangles are **not** flagged, matching the strict ``< 0`` comparison.
    [`flipped_faces`][triwarp.parametrization.flipped_faces] is the index form of this mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates) as ``wp.vec2``.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``vertices.device``. Empty for an empty mesh.

    See Also
    --------
    [`flipped_faces`][triwarp.parametrization.flipped_faces]

    Notes
    -----
    Equivalent to the per-triangle predicate behind libigl ``flipped_triangles``: the 2D cross
    product ``(v1 - v0) x (v2 - v0)`` equals ``det([[x0, x1, x2], [y0, y1, y2], [1, 1, 1]])``, so a
    ``True`` entry corresponds exactly to a triangle libigl would list as flipped.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    out_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_parametrization.flipped_faces_mask,
        dim=n_faces,
        inputs=[vertices, faces, out_mask],
        device=device,
    )
    return out_mask


def flipped_faces(vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Return the indices of triangles inverted (negative 2D signed area) in the parametrization.

    Convenience wrapper returning ``flatnonzero`` of
    [`flipped_faces_mask`][triwarp.parametrization.flipped_faces_mask]: the indices into ``faces``
    of triangles whose 2D signed area is strictly negative (folded over in the UV domain). Matches
    libigl's ``flipped_triangles``, which returns the same list of flipped-triangle indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates) as ``wp.vec2``.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        Ascending face indices of the flipped triangles on ``vertices.device``. Empty when no
        triangle is flipped.

    See Also
    --------
    [`flipped_faces_mask`][triwarp.parametrization.flipped_faces_mask]
    [`flatnonzero`][triwarp.array.flatnonzero]
    """
    return tw.array.flatnonzero(flipped_faces_mask(vertices, faces))


def map_vertices_to_circle(
    vertices: wp.array[wp.vec3], boundary: wp.array[wp.int32]
) -> wp.array[wp.vec2]:
    """
    Map an ordered boundary loop onto the unit circle by arc length (``map_vertices_to_circle``).

    Places boundary vertex ``i`` at angle ``2*pi * len[i] / total``, where ``len[i]`` is the
    cumulative edge length from ``boundary[0]`` to ``boundary[i]`` along the loop and ``total`` is
    the full perimeter (closing over the wrap edge). The result is in loop order: row ``i`` is the
    position of ``boundary[i]``. Feed it as ``boundary_uv`` to
    [`tutte`][triwarp.parametrization.tutte] or [`harmonic`][triwarp.parametrization.harmonic] for a
    disk parametrization.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    boundary
        Ordered boundary-loop vertex indices, e.g. from
        [`boundary_loop`][triwarp.boundary.boundary_loop].

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_boundary,)`` unit-circle positions on ``vertices.device``, aligned with ``boundary``.

    See Also
    --------
    [`boundary_loop`][triwarp.boundary.boundary_loop]
    [`tutte`][triwarp.parametrization.tutte]
    [`harmonic`][triwarp.parametrization.harmonic]
    """
    device = vertices.device
    n_boundary = int(boundary.shape[0])
    out_uv = wp.empty(n_boundary, dtype=wp.vec2, device=device)
    if n_boundary == 0:
        return out_uv

    segment_lengths = wp.empty(n_boundary, dtype=wp.float32, device=device)
    wp.launch(
        kernel_parametrization.boundary_edge_lengths,
        dim=n_boundary,
        inputs=[boundary, vertices, segment_lengths],
        device=device,
    )
    cumulative = wp.empty(n_boundary, dtype=wp.float32, device=device)
    wp.utils.array_scan(segment_lengths, out_array=cumulative, inclusive=True)
    wp.launch(
        kernel_parametrization.circle_positions,
        dim=n_boundary,
        inputs=[boundary, vertices, cumulative, out_uv],
        device=device,
    )
    return out_uv


def harmonic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int = 1,
) -> wp.array[wp.vec2]:
    """
    Harmonic parametrization with fixed boundary.

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][triwarp.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is solved with conjugate gradient.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    boundary_indices
        ``wp.int32`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. from
        [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]).
    k
        Harmonic power (``>= 1``). ``k > 1`` additionally uses the barycentric lumped mass matrix
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]. The operator is assembled in
        float64 (built native, never recast) so the ill-conditioned ``k > 1`` solve is accurate.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, or if there are interior vertices but ``boundary_indices`` is empty (the
        Dirichlet system would be singular).

    See Also
    --------
    [`tutte`][triwarp.parametrization.tutte]
    [`k_harmonic`][triwarp.energies.k_harmonic]
        The *operator* this minimizes, as opposed to this *map* -- the two are the only two
        "harmonic" names in the package and they are not interchangeable.
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]

    Notes
    -----
    Matches ``igl::harmonic``, ``k`` included.

    !!! note "An iterative solve against a direct factorization; it stays 1.2-2.2x behind"
        Measured against ``igl`` on the benchmark's ``patch`` and ``quality`` axes: 254 ms against
        128 at ``saddle`` with ``k=2``, 59 against 28 at ``saddle_small`` with ``k=2``, 83 against
        38 for the conditioning row, and ``lscm`` 29 against 24. That gap is **CG iteration count**,
        and it is not assembly: the assembly was rebuilt to emit CSR directly and won 2.4-9.8x on
        the rows where assembly *was* the cost, while measuring **flat** on exactly these (254 vs
        257 ms at ``k=2``) — the ``k=2`` biharmonic operator has ~5x the ``nnz`` and squares the
        condition number, so the solve dominates and always did. Every factorizing reference is flat
        along the triangle-quality axis for the same reason, which is the other side of the same
        observation.

        So this is a **deliberate trade, not an open defect**: triwarp pays 1.2-2.2x on the solve
        and wins ~13x on setup, because it factors nothing. Closing it needs a preconditioner
        stronger
        than Jacobi (incomplete Cholesky, or an algebraic-multigrid V-cycle) — a substantial
        subsystem that no in-repo caller has asked for, and not obviously a win at these sizes.
        **Do not re-open this as an assembly problem; that has now measured flat three times.**
        [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] reaches the same conclusion from its
        own measurements.
    """
    if k < 1:
        raise ValueError(f"harmonic power k must be >= 1, got {k}.")
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0:
        return wp.empty(0, dtype=wp.vec2, device=device)
    laplacian = cotmatrix(vertices, faces, dtype=wp.float64)
    mass_diag = mass_matrix_entries(vertices, faces, dtype=wp.float64) if k > 1 else None
    return _solve_fixed_boundary(
        laplacian, mass_diag, k, n_vertices, boundary_indices, boundary_uv, device
    )


def tutte(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int = 1,
) -> wp.array[wp.vec2]:
    """
    Tutte embedding with fixed boundary (uniform-Laplacian parametrization).

    Identical to [`harmonic`][triwarp.parametrization.harmonic] except the operator is the
    combinatorial [`uniform_laplacian`][triwarp.laplacian.uniform_laplacian] instead of the
    cotangent one — for ``k == 1`` that Laplacian is the *only* difference. Because the uniform
    Laplacian's free-free block is a diagonally dominant M-matrix, the Tutte embedding of a mesh
    with a convex boundary is guaranteed bijective (fold-free), unlike the harmonic/conformal maps.
    For
    ``k > 1`` the mass matrix is the identity (matching libigl's ``speye`` graph-Laplacian variant).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions (used for the count/device; the uniform weights
        ignore geometry).
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    boundary_indices
        ``wp.int32`` indices of the fixed (constrained) vertices.
    boundary_uv
        ``(n_boundary,)`` target UV positions for ``boundary_indices``, in the same order (e.g. a
        convex loop from
        [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]).
    k
        Laplacian power (``>= 1``). ``k == 1`` is the classic Tutte embedding.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``k < 1``, or if there are interior vertices but ``boundary_indices`` is empty.

    See Also
    --------
    [`harmonic`][triwarp.parametrization.harmonic]
    [`uniform_laplacian`][triwarp.laplacian.uniform_laplacian]
    [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]
    """
    if k < 1:
        raise ValueError(f"tutte power k must be >= 1, got {k}.")
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0:
        return wp.empty(0, dtype=wp.vec2, device=device)
    laplacian = uniform_laplacian(vertices, faces, dtype=wp.float64)
    return _solve_fixed_boundary(
        laplacian, None, k, n_vertices, boundary_indices, boundary_uv, device
    )


def _solve_fixed_boundary(
    laplacian: wps.BsrMatrix[wp.float64],
    mass_diag: wp.array[wp.float64] | None,
    k: int,
    n_vertices: int,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    device: wp.DeviceLike,
) -> wp.array[wp.vec2]:
    """
    Solve the fixed-boundary quadratic minimization shared by ``harmonic`` and ``tutte``.

    Forms the positive-semi-definite operator ``Q = -L`` for ``k == 1`` and
    ``Q = (-L) (M^-1 (-L))^(k-1)`` for ``k > 1`` (``M`` the diagonal mass, identity when
    ``mass_diag is None``) via [`k_harmonic`][triwarp.energies.k_harmonic],
    then solves the interior Dirichlet system ``Q_uu x_u = -Q_ub bc`` per UV column with conjugate
    gradient, keeping the fixed vertices at ``boundary_uv``. ``laplacian`` must be float64:
    ``k > 1`` squares its condition number.
    """
    q = tw.energies.k_harmonic(laplacian, mass_diag, k=k)

    # A mesh with interior vertices and no fixed boundary is a singular Dirichlet system. Raised up
    # front: once every vertex is fixed (n_vertices > 0, n_boundary == 0 is impossible here because
    # n_vertices > 0 implies interior vertices exist) this cannot be satisfied.
    n_boundary = int(boundary_indices.shape[0])
    if n_boundary == 0:
        raise ValueError(
            "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
            "otherwise singular."
        )

    fixed_mask, fixed_values = _scatter_constraints(
        n_vertices, boundary_indices, boundary_uv, device
    )

    sol, free_map, _ = twl.min_quad_with_fixed(
        q, fixed_mask, twt.as_array2d(fixed_values, wp.float64), tol=_CG_TOLERANCE
    )

    out_uv = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, sol, fixed_values, out_uv],
        device=device,
    )
    return out_uv


def arap(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    fixed_indices: wp.array[wp.int32],
    fixed_uv: wp.array[wp.vec2],
    uv_init: wp.array[wp.vec2],
    max_iterations: int = 10,
    tolerance: float = 1e-7,
) -> wp.array[wp.vec2]:
    """
    As-rigid-as-possible (ARAP) parametrization with fixed vertices.

    Minimizes the ARAP energy of the 2D parametrization by local/global alternation, starting from
    ``uv_init`` and keeping ``fixed_indices`` pinned to ``fixed_uv`` at every iteration. The local
    step fits, per triangle, the closest rotation between the isometrically flattened rest triangle
    and its current UV image (closed-form 2D polar decomposition, reflections forbidden); the global
    step solves the cotangent-Laplacian Poisson system ``(-L)_uu U_u = (K R)_u - (-L)_ub bc`` for
    each UV column with conjugate gradient.

    A good ``uv_init`` matters: ARAP is non-convex, so feed a fold-free initial map such as
    [`harmonic`][triwarp.parametrization.harmonic] or [`tutte`][triwarp.parametrization.tutte], with
    the boundary placed by
    [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]. Inspect the result
    for inverted triangles with [`flipped_faces`][triwarp.parametrization.flipped_faces].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    fixed_indices
        ``wp.int32`` indices of the pinned (constrained) vertices — libigl's ``b``. At least one is
        required whenever the mesh has interior vertices (the ARAP global system is otherwise a
        singular, translation-invariant Poisson problem). Pinning the whole boundary loop reproduces
        the classic fixed-boundary ARAP disk parametrization.
    fixed_uv
        ``(n_fixed,)`` target UV positions for ``fixed_indices``, in the same order — libigl's
        ``bc``.
    uv_init
        ``(n_vertices,)`` initial UV coordinates (the warm start). Never mutated; the interior
        values seed the first local step and the conjugate-gradient warm start, and the pinned rows
        are
        overwritten with ``fixed_uv`` before the first iteration.
    max_iterations
        Number of local/global iterations (``>= 1``). libigl defaults to ``10``.
    tolerance
        Relative residual tolerance of the inner conjugate-gradient solve (``> 0``). See Notes for
        why the default is looser than the ``1e-8`` the other solvers in this module use.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh. Rows at
        ``fixed_indices`` equal ``fixed_uv`` exactly.

    Raises
    ------
    ValueError
        If ``max_iterations < 1``, if ``tolerance <= 0``, or if there are interior vertices but
        ``fixed_indices`` is empty.

    See Also
    --------
    [`harmonic`][triwarp.parametrization.harmonic]
    [`tutte`][triwarp.parametrization.tutte]
    [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]
    [`flipped_faces`][triwarp.parametrization.flipped_faces]
    [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]

    Notes
    -----
    Uses the *elements* ARAP energy (one rotation per triangle), libigl's default for the flat
    ``dim = 2`` parametrization case; the covariance scatter is built per corner of the flattened
    mesh, which is why the ``SPOKES`` / ``SPOKES_AND_RIMS`` (per-vertex) energies do not apply here.
    The half-cotangent weights ``c_e`` come from
    [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries] with no clamping (matching libigl),
    and the closest 2D rotation is the closed form ``theta = atan2(S10 - S01, S00 + S11)`` of
    ``igl::fit_rotations_planar`` (libigl's scale-invariant ``S /= max|S|`` normalization is
    unnecessary for ``atan2`` and is skipped). The cotangent operator and the two conjugate-gradient
    solves run in float64 for determinism while the UV field is stored ``float32`` between
    iterations (module convention); the ~1e-7/iteration drift is well under the pinned-boundary
    tolerance for the default iteration count.

    **Why ``tolerance`` defaults to ``1e-7`` and not ``1e-8``.** Unlike
    [`harmonic`][triwarp.parametrization.harmonic] or [`lscm`][triwarp.parametrization.lscm], whose
    single solve *is* the answer, ARAP's global solves are inner steps of a truncated outer
    iteration: solving one more accurately than the outer iteration's own truncation error is wasted
    work. Measured on an RTX 5090 at ``max_iterations=10``, against the same run at ``1e-8``:

    | mesh | vertices | speedup | max UV change | one more outer iteration changes |
    |---|---|---|---|---|
    | saddle patch | 4.6k | -13 % | 7.7e-07 | 6.3e-06 |
    | saddle patch | 17.7k | -18 % | 1.4e-06 | 5.8e-06 |
    | bunny (decimated) | 8.2k | -29 % | 1.1e-05 | 5.8e-05 |
    | bunny | 35.9k | -32 % | 1.3e-05 | 8.9e-06 |

    On every mesh the error the looser tolerance introduces is at or below the error the caller
    already accepts by stopping at ``max_iterations``, and agreement with ``igl.arap_solve`` stays
    at ``5e-07`` or better (the regression tests compare at ``1e-4``). Pass ``tolerance=1e-8`` to
    restore the previous behaviour. Going further to ``1e-6`` is roughly twice as fast again
    (-25 % to -55 %) but lets the inner error reach ``1.3e-04``, above the outer truncation error on
    the largest mesh, so it is not the default.
    """
    if max_iterations < 1:
        raise ValueError(f"arap max_iterations must be >= 1, got {max_iterations}.")
    if tolerance <= 0.0:
        raise ValueError(f"arap tolerance must be > 0, got {tolerance}.")
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0:
        return wp.empty(0, dtype=wp.vec2, device=device)
    n_faces = int(faces.shape[0]) // 3

    # Cotangents computed once and reused by both the Laplacian build and the rest-edge flattening;
    # single native-float64 operator build, so nothing is recast or rebuilt (see cotmatrix docs).
    cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
    laplacian = cotmatrix(vertices, faces, cot_entries=cot_entries, dtype=wp.float64)

    # Partition vertices into pinned (fixed) and interior (free). ``fixed_values`` is the
    # (2, n_vertices) prescribed-UV buffer (row 0 = u, row 1 = v) shared with the assembly / scatter
    # kernels; ``interior_map`` compacts free vertices into the reduced system.
    fixed_mask, fixed_values = _scatter_constraints(n_vertices, fixed_indices, fixed_uv, device)
    fixed_values_2d = twt.as_array2d(fixed_values, wp.float64)
    interior_map, n_interior = twl.free_partition(fixed_mask)

    out_uv = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    if n_interior == 0:
        # Every vertex pinned: the prescribed positions are the whole answer, no solve.
        empty_sol = wp.zeros((2, 0), dtype=wp.float64, device=device)
        wp.launch(
            kernel_parametrization.scatter_solution,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, empty_sol, fixed_values_2d, out_uv],
            device=device,
        )
        return out_uv

    # Interior vertices with nothing pinned leave the ARAP global system translation-invariant
    # (singular). Raised up front, mirroring harmonic / tutte.
    if int(fixed_indices.shape[0]) == 0:
        raise ValueError(
            "arap requires at least one fixed vertex when the mesh has interior vertices; the ARAP "
            "global system is otherwise singular (translation invariant)."
        )

    # Global-step operator: interior block of Q = -L and the constant boundary term -(-L)_ub bc.
    # ``future work``: libigl also supports rotation groups ``G`` (shared rotations across grouped
    # faces, replacing the per-face fit with a group-summed covariance) and ``with_dynamics`` (a
    # mass-matrix + timestep term added to Q and the right-hand side); both are out of scope here.
    neg_l = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    q_uu, rhs_const = twl.assemble_interior_system(
        neg_l, fixed_mask, interior_map, fixed_values_2d, n_interior
    )

    # Weight-folded rest edges of the isometrically flattened triangles (internal buffer, plain
    # wp.empty; kernels index it as wp.array2d per CLAUDE.md).
    rest_edges = wp.empty((n_faces, 3), dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_parametrization.arap_rest_edges,
        dim=n_faces,
        inputs=[vertices, faces, cot_entries, rest_edges],
        device=device,
    )

    # Pre-loop buffers (no allocation inside the loop). ``sol`` (2, n_interior) holds the
    # warm-started CG solution per column; seed it from ``uv_init`` interior values, then
    # reconstruct the working ``out_uv`` with the constraints enforced for iteration 1.
    sol = wp.zeros((2, n_interior), dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.gather_interior_uv,
        dim=n_vertices,
        inputs=[fixed_mask, interior_map, uv_init, sol],
        device=device,
    )
    wp.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, interior_map, sol, fixed_values_2d, out_uv],
        device=device,
    )
    rhs_rot_x = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    rhs_rot_y = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    b = wp.empty((2, n_interior), dtype=wp.float64, device=device)

    # One batched CG state for both UV columns, built once outside the loop: its temporaries and
    # batch layout are reused across iterations, and it reads ``b`` / writes ``sol`` in place, so
    # the warm start is simply whatever ``sol`` already holds.
    solver = twl.spd_column_solver(
        q_uu,
        twt.as_array2d(b, wp.float64),
        twt.as_array2d(sol, wp.float64),
        tol=tolerance,
        maxiter=10 * n_interior,
    )

    for _ in range(max_iterations):
        rhs_rot_x.zero_()
        rhs_rot_y.zero_()
        # Local step: fit per-face rotations from ``out_uv`` and scatter the rotation RHS.
        wp.launch(
            kernel_parametrization.arap_local_step,
            dim=n_faces,
            inputs=[faces, out_uv, rest_edges, rhs_rot_x, rhs_rot_y],
            device=device,
        )
        # Global step RHS: constant boundary term + rotation term, restricted to interior rows.
        wp.launch(
            kernel_parametrization.arap_interior_rhs,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, rhs_const, rhs_rot_x, rhs_rot_y, b],
            device=device,
        )
        # Both UV columns in one batched symmetric-PD solve, warm-started from the previous ``sol``.
        solver()
        # Reconstruct the full UV field, re-enforcing the pinned constraints for the next iteration.
        wp.launch(
            kernel_parametrization.scatter_solution,
            dim=n_vertices,
            inputs=[fixed_mask, interior_map, sol, fixed_values_2d, out_uv],
            device=device,
        )
    return out_uv


def _scatter_constraints(
    n_vertices: int, indices: wp.array[wp.int32], uv: wp.array[wp.vec2], device: wp.DeviceLike
) -> tuple[wp.array[wp.bool], wp.array[wp.float64]]:
    """
    Expand a list of pinned vertices into the dense mask and prescribed-UV buffers the solvers take.

    ``fixed_mask`` marks the constrained vertices; ``fixed_values`` is the ``(2, n_vertices)``
    prescribed-UV buffer (row 0 = u, row 1 = v) the assembly and scatter kernels read. Shared by
    [`_solve_fixed_boundary`][triwarp.parametrization._solve_fixed_boundary] and
    [`arap`][triwarp.parametrization.arap]; an empty ``indices`` yields an all-``False`` mask and
    an all-zero value buffer, which each caller rejects on its own terms.
    """
    fixed_mask = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    fixed_values = wp.zeros((2, n_vertices), dtype=wp.float64, device=device)
    n_fixed = int(indices.shape[0])
    if n_fixed > 0:
        wp.launch(
            kernel_parametrization.scatter_boundary_mask,
            dim=n_fixed,
            inputs=[indices, fixed_mask],
            device=device,
        )
        wp.launch(
            kernel_parametrization.scatter_fixed_uv,
            dim=n_fixed,
            inputs=[indices, uv, fixed_values],
            device=device,
        )
    return fixed_mask, fixed_values


def lscm(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
) -> wp.array[wp.vec2]:
    """
    Constrained least-squares conformal map.

    Computes the conformal (angle-preserving) parametrization that minimizes the LSCM (Levy)
    conformal energy subject to a set of pinned vertices, by solving a single quadratic program
    over the stacked ``[u; v]`` vector of ``2 * n_vertices`` unknowns with the LSCM Hessian
    [`lscm_hessian`][triwarp.energies.lscm_hessian] as the operator. Unlike
    [`harmonic`][triwarp.parametrization.harmonic] / [`tutte`][triwarp.parametrization.tutte] (which
    pin the whole boundary and solve two independent columns), LSCM couples ``u`` and ``v`` through
    the boundary vector-area term, so it needs only a few pins — typically **two** — to fix the
    remaining similarity-transform (rotation + scale + translation) degree of freedom.

    The interior system is solved with conjugate gradient. Closed meshes are valid input: the
    boundary vector-area matrix is then zero and the
    Hessian reduces to ``-repdiag(L, 2)``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    pinned_indices
        ``wp.int32`` indices of the pinned (constrained) vertices — igl's ``b``. At least two are
        required (unless the mesh has fewer than two vertices) to remove the conformal map's
        similarity-transform null space.
    pinned_uv
        ``(n_pinned,)`` target UV positions for ``pinned_indices``, in the same order (igl's
        ``bc``).

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` UV coordinates on ``vertices.device``. Empty for an empty mesh.

    Raises
    ------
    ValueError
        If fewer than two vertices are pinned (and the mesh has at least two vertices).

    See Also
    --------
    [`lscm_hessian`][triwarp.energies.lscm_hessian]
    [`vector_area_matrix`][triwarp.energies.vector_area_matrix]
    [`harmonic`][triwarp.parametrization.harmonic]
    [`flipped_faces`][triwarp.parametrization.flipped_faces]

    Notes
    -----
    The unknowns are stacked ``[u; v]`` (all ``u`` DOFs then all ``v`` DOFs), matching igl's
    ``lscm``: pin ``i`` fixes DOF ``i`` (``u``) and DOF ``i + n_vertices`` (``v``). The returned
    ``Q`` of ``igl.lscm`` equals ``-repdiag(L, 2) - 2 A`` exactly (see
    [`lscm_hessian`][triwarp.energies.lscm_hessian]).
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.vec2, device=device)

    n_pinned = int(pinned_indices.shape[0])
    if n_pinned < 2 and n_pinned < n:
        raise ValueError(
            "lscm requires at least two pinned vertices to remove the conformal map's "
            f"similarity-transform null space; got {n_pinned}."
        )

    q = tw.energies.lscm_hessian(vertices, faces)
    fixed_mask = wp.zeros(2 * n, dtype=wp.bool, device=device)
    fixed_values = wp.zeros((1, 2 * n), dtype=wp.float64, device=device)
    if n_pinned > 0:
        wp.launch(
            kernel_parametrization.scatter_pinned_stacked,
            dim=n_pinned,
            inputs=[pinned_indices, pinned_uv, wp.int32(n), fixed_mask, fixed_values],
            device=device,
        )

    sol, free_map, _ = twl.min_quad_with_fixed(
        q, fixed_mask, twt.as_array2d(fixed_values, wp.float64), tol=_CG_TOLERANCE
    )

    out_uv = wp.empty(n, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_parametrization.scatter_solution_stacked,
        dim=n,
        inputs=[fixed_mask, free_map, sol[0], fixed_values[0], out_uv],
        device=device,
    )
    return out_uv
