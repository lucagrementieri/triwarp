from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_cuda
from triwarp.kernels import parametrization as kernel_parametrization
from triwarp.laplacian import cotmatrix, mass_matrix_entries, uniform_laplacian

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
    ``mass_diag is None``), then solves the interior Dirichlet system ``Q_uu x_u = -Q_ub bc`` per
    UV column with conjugate gradient, keeping the fixed vertices at ``boundary_uv``. ``laplacian``
    must be float64: ``k > 1`` squares its condition number, and building the operator natively in a
    single ``bsr_from_triplets`` (never recast/rebuilt) is what keeps ``bsr_mm`` deterministic — see
    issue_report.md.
    """
    # Operator Q. neg_l aliases -L; bsr_mm returns fresh matrices so neg_l stays valid across the
    # accumulation. Chained products stay deterministic because neg_l is a single-build operator.
    neg_l = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    q = neg_l
    if k > 1:
        if mass_diag is None:
            for _ in range(k - 1):
                q = wps.bsr_mm(q, neg_l)
        else:
            inv_mass = wp.empty(n_vertices, dtype=wp.float64, device=device)
            wp.map(kernel_parametrization.reciprocal64, mass_diag, out=inv_mass)
            mass_inv = wps.bsr_diag(diag=inv_mass)
            for _ in range(k - 1):
                q = wps.bsr_mm(wps.bsr_mm(q, mass_inv), neg_l)

    # A mesh with interior vertices and no fixed boundary is a singular Dirichlet system. Raised up
    # front (CPU-safe): once every vertex is fixed (n_vertices > 0, n_boundary == 0 is impossible
    # here because n_vertices > 0 implies interior vertices exist) this cannot be satisfied.
    n_boundary = int(boundary_indices.shape[0])
    if n_boundary == 0:
        raise ValueError(
            "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
            "otherwise singular."
        )

    # Fixed mask + prescribed positions scattered to a (2, n_vertices) buffer (row 0 = u, 1 = v).
    fixed_mask = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    fixed_values = wp.zeros((2, n_vertices), dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.scatter_boundary_mask,
        dim=n_boundary,
        inputs=[boundary_indices, fixed_mask],
        device=device,
    )
    wp.launch(
        kernel_parametrization.scatter_fixed_uv,
        dim=n_boundary,
        inputs=[boundary_indices, boundary_uv, fixed_values],
        device=device,
    )

    sol, free_map, _ = _min_quad_with_fixed_columns(
        q, fixed_mask, twt.as_array2d_float(fixed_values, dtype=wp.float64), device
    )

    out_uv = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[fixed_mask, free_map, sol, fixed_values, out_uv],
        device=device,
    )
    return out_uv


def _min_quad_with_fixed_columns(
    q: wps.BsrMatrix[wp.float64],
    fixed_mask: wp.array[wp.bool],
    fixed_values: twt.Array2dFloat,
    device: wp.DeviceLike,
) -> tuple[twt.Array2dFloat, wp.array[wp.int32], int]:
    """
    Solve ``min_quad_with_fixed`` for a stacked operator with ``n_rhs`` right-hand-side columns.

    Generalizes the fixed-value quadratic minimization shared by ``harmonic`` / ``tutte`` (2
    independent UV columns over ``n_vertices`` unknowns) and ``lscm`` (1 coupled column over ``2n``
    unknowns): ``fixed_mask`` marks the fixed DOFs among ``n_dofs = fixed_mask.shape[0]``, and
    ``fixed_values`` is ``(n_rhs, n_dofs)``. Returns the solved free values ``(n_rhs, n_free)``, the
    compact free-index remap, and the free count. The all-fixed case (``n_free == 0``) returns an
    empty solution without a solve (and without requiring CUDA), leaving reconstruction to the
    caller's scatter kernel.
    """
    n_dofs = int(fixed_mask.shape[0])
    n_rhs = int(fixed_values.shape[0])

    # Compact free remap: exclusive scan of the free indicator gives each free DOF its index in the
    # reduced system; the inclusive scan's last entry is the free count.
    flags = wp.empty(n_dofs, dtype=wp.int32, device=device)
    wp.map(kernel_parametrization.interior_flag, fixed_mask, out=flags)
    free_map = wp.empty(n_dofs, dtype=wp.int32, device=device)
    inclusive = wp.empty(n_dofs, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=free_map, inclusive=False)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    n_free = int(inclusive.numpy()[-1])

    sol = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    if n_free == 0:
        # Every DOF is fixed: the prescribed values are the whole answer, no solve needed.
        return twt.as_array2d_float(sol, dtype=wp.float64), free_map, n_free

    require_cuda(device, "harmonic / tutte / lscm")

    # Assemble the free-free block Q_uu and the right-hand sides -Q_ub bc from Q's CSR.
    nnz = int(q.nnz)
    out_rows = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_cols = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_vals = wp.zeros(nnz, dtype=wp.float64, device=device)
    rhs = wp.zeros((n_rhs, n_free), dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.interior_system_triplets,
        dim=n_dofs,
        inputs=[
            q.offsets,
            q.columns,
            q.values,
            fixed_mask,
            free_map,
            fixed_values,
            out_rows,
            out_cols,
            out_vals,
            rhs,
        ],
        device=device,
    )
    q_uu = wps.bsr_from_triplets(
        n_free, n_free, out_rows, out_cols, out_vals, prune_numerical_zeros=False
    )

    # One diagonal preconditioner shared by every symmetric-PD column solve. Row views of the
    # row-major (n_rhs, n_free) buffers are contiguous, so they serve directly as CG vectors.
    preconditioner = wpl.preconditioner(q_uu, "diag")
    for c in range(n_rhs):
        wpl.cg(q_uu, rhs[c], sol[c], tol=_CG_TOLERANCE, maxiter=10 * n_free, M=preconditioner)
    return twt.as_array2d_float(sol, dtype=wp.float64), free_map, n_free


def harmonic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int = 1,
) -> wp.array[wp.vec2]:
    """
    Harmonic parametrization with fixed boundary (``igl::harmonic``).

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][triwarp.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is solved with conjugate gradient, so a **CUDA device is required** whenever
    there are interior vertices to solve for (``warp.optim.linear.cg`` returns NaN on the CPU in
    Warp 1.14-1.15).

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
    NotImplementedError
        On a CPU device when an interior solve is required.

    See Also
    --------
    [`tutte`][triwarp.parametrization.tutte]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`map_vertices_to_circle`][triwarp.parametrization.map_vertices_to_circle]
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
    Requires a **CUDA device** whenever there are interior vertices to solve for.

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
    NotImplementedError
        On a CPU device when an interior solve is required.

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


def lscm(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    pinned_indices: wp.array[wp.int32],
    pinned_uv: wp.array[wp.vec2],
) -> wp.array[wp.vec2]:
    """
    Constrained least-squares conformal map (``igl::lscm``).

    Computes the conformal (angle-preserving) parametrization that minimizes the LSCM (Levy)
    conformal energy subject to a set of pinned vertices, by solving a single quadratic program
    over the stacked ``[u; v]`` vector of ``2 * n_vertices`` unknowns with the LSCM Hessian
    [`lscm_hessian`][triwarp.parametrization.lscm_hessian] as the operator. Unlike
    [`harmonic`][triwarp.parametrization.harmonic] / [`tutte`][triwarp.parametrization.tutte] (which
    pin the whole boundary and solve two independent columns), LSCM couples ``u`` and ``v`` through
    the boundary vector-area term, so it needs only a few pins — typically **two** — to fix the
    remaining similarity-transform (rotation + scale + translation) degree of freedom.

    The interior system is solved with conjugate gradient, so a **CUDA device is required** whenever
    there are free vertices to solve for (``warp.optim.linear.cg`` returns NaN on the CPU in Warp
    1.14-1.15). Closed meshes are valid input: the boundary vector-area matrix is then zero and the
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
    NotImplementedError
        On a CPU device when a free-vertex solve is required.

    See Also
    --------
    [`lscm_hessian`][triwarp.parametrization.lscm_hessian]
    [`vector_area_matrix`][triwarp.parametrization.vector_area_matrix]
    [`harmonic`][triwarp.parametrization.harmonic]
    [`flipped_faces`][triwarp.parametrization.flipped_faces]

    Notes
    -----
    The unknowns are stacked ``[u; v]`` (all ``u`` DOFs then all ``v`` DOFs), matching igl's
    ``lscm``: pin ``i`` fixes DOF ``i`` (``u``) and DOF ``i + n_vertices`` (``v``). The returned
    ``Q`` of ``igl.lscm`` equals ``-repdiag(L, 2) - 2 A`` exactly (see
    [`lscm_hessian`][triwarp.parametrization.lscm_hessian]).
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

    q = lscm_hessian(vertices, faces)
    fixed_mask = wp.zeros(2 * n, dtype=wp.bool, device=device)
    fixed_values = wp.zeros((1, 2 * n), dtype=wp.float64, device=device)
    if n_pinned > 0:
        wp.launch(
            kernel_parametrization.scatter_pinned_stacked,
            dim=n_pinned,
            inputs=[pinned_indices, pinned_uv, wp.int32(n), fixed_mask, fixed_values],
            device=device,
        )

    sol, free_map, _ = _min_quad_with_fixed_columns(
        q, fixed_mask, twt.as_array2d_float(fixed_values, dtype=wp.float64), device
    )

    out_uv = wp.empty(n, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_parametrization.scatter_solution_stacked,
        dim=n,
        inputs=[fixed_mask, free_map, sol[0], fixed_values[0], out_uv],
        device=device,
    )
    return out_uv


def lscm_hessian(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float64]:
    """
    LSCM Hessian ``Q = -repdiag(L, 2) - 2 A`` (``igl::lscm_hessian``).

    Assembles the ``(2n, 2n)`` symmetric operator behind the least-squares conformal map, where
    ``L`` is the cotangent Laplacian [`cotmatrix`][triwarp.laplacian.cotmatrix] (negative-diagonal
    convention), ``repdiag(L, 2)`` is the block-diagonal ``[[L, 0], [0, L]]``, and ``A`` is the
    boundary [`vector_area_matrix`][triwarp.parametrization.vector_area_matrix]. Built natively in
    float64 in a single ``bsr_from_triplets`` (the within-quadrant repdiag triplets and the
    cross-quadrant ``-2 A`` triplets never collide), so it feeds the float64 conjugate-gradient
    solve directly. Matches the ``Q`` returned by ``igl.lscm`` exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(2n, 2n)`` float64 matrix in 1x1-block BSR form on ``vertices.device``.

    See Also
    --------
    [`lscm`][triwarp.parametrization.lscm]
    [`vector_area_matrix`][triwarp.parametrization.vector_area_matrix]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    n = int(vertices.shape[0])
    device = vertices.device
    laplacian = cotmatrix(vertices, faces, dtype=wp.float64)
    # The real compressed-CSR entry count is offsets[-1], not laplacian.nnz: bsr_from_triplets
    # reports nnz as the (over-allocated) triplet capacity, so sizing by nnz would leave an
    # uninitialized gap in the wp.empty buffers that bsr_from_triplets reads back as garbage.
    n_entries = int(laplacian.offsets.numpy()[-1])
    boundary = tw.boundary.oriented_boundary_edges(vertices, faces)
    n_be = int(boundary.shape[0])

    # Combined triplet buffers: 2 per Laplacian entry (the two diagonal blocks) plus 4 per oriented
    # boundary edge (the vector-area cross-quadrant terms). Every slot is written, so wp.empty.
    total = 2 * n_entries + 4 * n_be
    rows = wp.empty(total, dtype=wp.int32, device=device)
    cols = wp.empty(total, dtype=wp.int32, device=device)
    vals = wp.empty(total, dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.neg_repdiag2_triplets,
        dim=n,
        inputs=[
            laplacian.offsets,
            laplacian.columns,
            laplacian.values,
            wp.int32(n),
            rows,
            cols,
            vals,
        ],
        device=device,
    )
    if n_be > 0:
        wp.launch(
            kernel_parametrization.vector_area_triplets,
            dim=n_be,
            inputs=[
                boundary,
                wp.int32(n),
                wp.float64(-2.0),
                rows[2 * n_entries :],
                cols[2 * n_entries :],
                vals[2 * n_entries :],
            ],
            device=device,
        )
    return wps.bsr_from_triplets(2 * n, 2 * n, rows, cols, vals, prune_numerical_zeros=False)


def vector_area_matrix(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float64]:
    """
    Boundary vector-area matrix ``A`` (``igl::vector_area_matrix``).

    Assembles the ``(2n, 2n)`` matrix that turns the ``[u; v]`` quadratic form into the signed area
    enclosed by the boundary UV curve: for each **oriented** boundary edge ``(i, j)`` (from the face
    winding, via [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]) it adds the
    cross-quadrant entries ``(i+n, j, -1/4)``, ``(j, i+n, -1/4)``, ``(i, j+n, +1/4)``,
    ``(j+n, i, +1/4)``. On a closed mesh (no boundary) ``A`` is the zero matrix. Built natively in
    float64 in a single ``bsr_from_triplets``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions; only the count and device are used.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(2n, 2n)`` float64 matrix in 1x1-block BSR form on ``vertices.device``.

    See Also
    --------
    [`lscm_hessian`][triwarp.parametrization.lscm_hessian]
    [`lscm`][triwarp.parametrization.lscm]
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]
    """
    n = int(vertices.shape[0])
    device = vertices.device
    boundary = tw.boundary.oriented_boundary_edges(vertices, faces)
    n_be = int(boundary.shape[0])
    if n_be == 0:
        return wps.bsr_from_triplets(
            2 * n,
            2 * n,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float64, device=device),
            prune_numerical_zeros=False,
        )

    rows = wp.empty(4 * n_be, dtype=wp.int32, device=device)
    cols = wp.empty(4 * n_be, dtype=wp.int32, device=device)
    vals = wp.empty(4 * n_be, dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.vector_area_triplets,
        dim=n_be,
        inputs=[boundary, wp.int32(n), wp.float64(1.0), rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(2 * n, 2 * n, rows, cols, vals, prune_numerical_zeros=False)
