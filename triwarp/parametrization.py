from __future__ import annotations

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
from triwarp.kernels import parametrization as kernel_parametrization
from triwarp.laplacian import cotmatrix, mass_matrix_entries, uniform_laplacian

_CG_TOLERANCE = 1e-8


def _require_cuda(device: wp.DeviceLike) -> None:
    if wp.get_device(device).is_cpu:
        raise NotImplementedError(
            "harmonic / tutte require a CUDA device: warp.optim.linear.cg produces NaN on the CPU "
            "device in Warp 1.14.0."
        )


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


def _to_float64_bsr(
    matrix: wps.BsrMatrix[wp.float32], n: int, device: wp.DeviceLike
) -> wps.BsrMatrix[wp.float64]:
    """Recast a square 1x1-block float32 ``BsrMatrix`` to float64, preserving its sparsity."""
    nnz = int(matrix.nnz)
    rows = wp.empty(nnz, dtype=wp.int32, device=device)
    wp.launch(
        kernel_parametrization.expand_offsets_to_rows,
        dim=n,
        inputs=[matrix.offsets, rows],
        device=device,
    )
    values64 = wp.empty(nnz, dtype=wp.float64, device=device)
    wp.utils.array_cast(matrix.values, values64)
    return wps.bsr_from_triplets(
        n, n, rows, matrix.columns, values64, prune_numerical_zeros=False
    )


def _solve_fixed_boundary(
    laplacian: wps.BsrMatrix[wp.float32],
    mass_diag: wp.array[wp.float32] | None,
    n_vertices: int,
    boundary_indices: wp.array[wp.int32],
    boundary_uv: wp.array[wp.vec2],
    k: int,
    device: wp.DeviceLike,
) -> wp.array[wp.vec2]:
    """
    Solve the fixed-boundary quadratic minimization shared by ``harmonic`` and ``tutte``.

    Forms the operator ``Q = -L`` for ``k == 1`` and ``Q = (-L) (M^-1 (-L))^(k-1)`` for ``k > 1``
    (``M`` the diagonal mass, identity when ``mass_diag is None``), then solves the interior
    Dirichlet system ``Q_uu x_u = -Q_ub bc`` per UV column with conjugate gradient, keeping the
    fixed vertices at ``boundary_uv``.
    """
    # Operator Q (positive semi-definite), assembled in float64: for k > 1 the operator squares the
    # Laplacian condition number, which float32 conjugate gradient cannot resolve. q and neg_l alias
    # -L initially; bsr_mm returns fresh matrices, so neg_l stays valid across the accumulation.
    # ``wp.synchronize()`` guards each sparse matrix-matrix product: warp's bsr_mm reuses internal
    # work buffers that race when one product's result is consumed by the next before it completes,
    # producing nondeterministic (occasionally garbage) entries.
    laplacian64 = _to_float64_bsr(laplacian, n_vertices, device)
    neg_l = wps.bsr_axpy(x=laplacian64, alpha=-1.0)
    q = neg_l
    if k > 1:
        if mass_diag is None:
            for _ in range(k - 1):
                wp.synchronize()
                q = wps.bsr_mm(q, neg_l)
            wp.synchronize()
        else:
            mass64 = wp.empty(n_vertices, dtype=wp.float64, device=device)
            wp.utils.array_cast(mass_diag, mass64)
            inv_mass = wp.empty(n_vertices, dtype=wp.float64, device=device)
            wp.launch(
                kernel_parametrization.reciprocal,
                dim=n_vertices,
                inputs=[mass64, inv_mass],
                device=device,
            )
            mass_inv = wps.bsr_diag(diag=inv_mass)
            for _ in range(k - 1):
                wp.synchronize()
                half = wps.bsr_mm(q, mass_inv)
                wp.synchronize()
                q = wps.bsr_mm(half, neg_l)
            wp.synchronize()

    # Boundary mask + prescribed positions scattered to full length.
    n_boundary = int(boundary_indices.shape[0])
    boundary_mask = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    fixed_uv = wp.zeros(n_vertices, dtype=wp.vec2, device=device)
    if n_boundary > 0:
        wp.launch(
            kernel_parametrization.scatter_boundary_mask,
            dim=n_boundary,
            inputs=[boundary_indices, boundary_mask],
            device=device,
        )
        wp.launch(
            kernel_parametrization.scatter_fixed_uv,
            dim=n_boundary,
            inputs=[boundary_indices, boundary_uv, fixed_uv],
            device=device,
        )

    # Compact interior remap: exclusive scan of the interior indicator gives each free vertex its
    # index in the reduced system; the inclusive scan's last entry is the interior count.
    flags = wp.empty(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_parametrization.interior_flags,
        dim=n_vertices,
        inputs=[boundary_mask, flags],
        device=device,
    )
    interior_map = wp.empty(n_vertices, dtype=wp.int32, device=device)
    inclusive = wp.empty(n_vertices, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=interior_map, inclusive=False)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    n_interior = int(inclusive.numpy()[-1])

    if n_interior == 0:
        # Every vertex is fixed: the prescribed positions are the whole answer, no solve needed.
        return fixed_uv
    if n_boundary == 0:
        raise ValueError(
            "harmonic / tutte require at least one fixed boundary vertex; the Dirichlet system is "
            "otherwise singular."
        )

    _require_cuda(device)

    # Assemble the interior-interior block Q_uu and the right-hand sides -Q_ub bc from Q's CSR.
    nnz = int(q.nnz)
    out_rows = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_cols = wp.zeros(nnz, dtype=wp.int32, device=device)
    out_vals = wp.zeros(nnz, dtype=wp.float64, device=device)
    rhs_x = wp.zeros(n_interior, dtype=wp.float64, device=device)
    rhs_y = wp.zeros(n_interior, dtype=wp.float64, device=device)
    wp.launch(
        kernel_parametrization.interior_system_triplets,
        dim=n_vertices,
        inputs=[
            q.offsets,
            q.columns,
            q.values,
            boundary_mask,
            interior_map,
            fixed_uv,
            out_rows,
            out_cols,
            out_vals,
            rhs_x,
            rhs_y,
        ],
        device=device,
    )
    q_uu = wps.bsr_from_triplets(
        n_interior, n_interior, out_rows, out_cols, out_vals, prune_numerical_zeros=False
    )

    # One diagonal preconditioner shared by both symmetric-PD column solves.
    preconditioner = wpl.preconditioner(q_uu, "diag")
    sol_x = wp.zeros(n_interior, dtype=wp.float64, device=device)
    sol_y = wp.zeros(n_interior, dtype=wp.float64, device=device)
    wpl.cg(
        q_uu,
        rhs_x,
        sol_x,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_interior,
        M=preconditioner,
        use_cuda_graph=False,
    )
    wpl.cg(
        q_uu,
        rhs_y,
        sol_y,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_interior,
        M=preconditioner,
        use_cuda_graph=False,
    )

    out_uv = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_parametrization.scatter_solution,
        dim=n_vertices,
        inputs=[boundary_mask, interior_map, sol_x, sol_y, fixed_uv, out_uv],
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
    Harmonic parametrization with fixed boundary (``igl::harmonic``).

    Minimizes the ``k``-harmonic energy built from the cotangent Laplacian
    [`cotmatrix`][triwarp.laplacian.cotmatrix] subject to the boundary vertices being pinned to
    ``boundary_uv``. For ``k == 1`` this is the harmonic map (each interior UV is the
    cotangent-weighted average of its neighbors); ``k == 2`` is the biharmonic map, and so on. The
    interior system is
    solved with conjugate gradient, so a **CUDA device is required** whenever there are interior
    vertices to solve for (``warp.optim.linear.cg`` returns NaN on the CPU in Warp 1.14.0).

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
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries].

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
    laplacian = cotmatrix(vertices, faces)
    mass_diag = mass_matrix_entries(vertices, faces) if k > 1 else None
    return _solve_fixed_boundary(
        laplacian, mass_diag, n_vertices, boundary_indices, boundary_uv, k, device
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
    cotangent one — for ``k == 1`` this is the *only* difference. Because the uniform Laplacian's
    free-free block is a diagonally dominant M-matrix, the Tutte embedding of a mesh with a convex
    boundary is guaranteed bijective (fold-free), unlike the harmonic/conformal maps. For ``k > 1``
    the mass matrix is the identity (matching libigl's ``speye`` graph-Laplacian variant). Requires
    a **CUDA device** whenever there are interior vertices to solve for.

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
    laplacian = uniform_laplacian(vertices, faces)
    return _solve_fixed_boundary(
        laplacian, None, n_vertices, boundary_indices, boundary_uv, k, device
    )
