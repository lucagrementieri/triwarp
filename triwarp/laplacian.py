from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp.typing as twt
from triwarp.edges import edges_unique, faces_to_edges
from triwarp.kernels import laplacian as kernel_laplacian
from triwarp.kernels import scatter as kernel_scatter
from triwarp.triangles import face_normals_and_areas


def cotmatrix_entries(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> twt.Array2dFloat32:
    """
    Per-triangle half-cotangent weights (``igl::cotmatrix_entries``).

    For each triangle face, column ``e`` stores ``1/2 * cot(angle at vertex e)`` for the
    edge opposite that vertex. Columns follow igl edge order: opposite vertices 0, 1, 2.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    twt.Array2dFloat32
        Shape ``(n_faces, 3)`` on ``faces.device``. Empty ``(0, 3)`` when ``n_faces == 0``.

    See Also
    --------
    [`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return twt.empty_float32_2d((0, 3), device=device)

    out_cot = twt.empty_float32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_laplacian.cotmatrix_entries,
        dim=n_faces,
        inputs=[vertices, faces, out_cot],
        device=device,
    )
    return twt.as_array2d_float32(out_cot)


def cotmatrix_entries_intrinsic(edge_lengths: twt.Array2dFloat32) -> twt.Array2dFloat32:
    """
    Per-triangle half-cotangent weights from edge lengths.

    (``igl::cotmatrix_entries`` intrinsic overload).

    Each row gives the three edge lengths opposite vertices 0, 1, and 2 of the corresponding
    triangle.

    Parameters
    ----------
    edge_lengths
        ``(n_faces, 3)`` edge lengths on the target device.

    Returns
    -------
    twt.Array2dFloat32
        Shape ``(n_faces, 3)`` on ``edge_lengths.device``. Empty ``(0, 3)`` when ``n_faces == 0``.

    See Also
    --------
    [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    twt.ensure_ndim(edge_lengths, 2, dtype=wp.float32)
    n_faces = int(edge_lengths.shape[0])
    device = edge_lengths.device
    if n_faces == 0:
        return twt.empty_float32_2d((0, 3), device=device)

    out_cot = twt.empty_float32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_laplacian.cotmatrix_entries_intrinsic,
        dim=n_faces,
        inputs=[edge_lengths, out_cot],
        device=device,
    )
    return twt.as_array2d_float32(out_cot)


def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: twt.Array2dFloat32 | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Cotangent stiffness matrix / discrete Laplacian (``igl::cotmatrix``).

    Builds the sparse ``(n_vertices, n_vertices)`` matrix from triangle geometry. Diagonal
    entries are **negative** (each row sums to zero); ``-L`` is positive semi-definite on
    closed meshes.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    cot_entries
        Optional precomputed ``(n_faces, 3)`` weights from
        [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]. When ``None``, entries are
        computed from ``vertices`` and ``faces``.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` cotangent matrix in 1x1-block BSR form on
        ``vertices.device``.

    See Also
    --------
    [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    """
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device

    if n_faces == 0:
        return wps.bsr_from_triplets(
            n_vertices,
            n_vertices,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            prune_numerical_zeros=False,
        )

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces)

    n_triplets = 12 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=wp.float32, device=device)
    wp.launch(
        kernel_laplacian.cotmatrix_triplets,
        dim=n_faces,
        inputs=[faces, cot_entries, rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )


def laplacian_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    equal_weight: bool = True,
    symmetric: bool | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.float32]]:
    """
    Per-edge weight triplets for the 1-ring Laplacian, before assembly.

    The analogue of [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries] for the umbrella
    operator, assembled and row-normalized by [`laplacian`][triwarp.laplacian.laplacian].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    equal_weight
        If ``True`` every edge weight is ``1`` (uniform umbrella weights). If ``False`` the
        weight is inverse edge length ``1 / (‖vi - vj‖ + 1e-12)``.
    symmetric
        Adjacency shape. If ``False`` one triplet per directed triangle edge is emitted (trimesh's
        ``mesh.edges``); on meshes with a boundary this is asymmetric. If ``True`` each unique
        undirected edge emits both directed pairs (trimesh's ``vertex_neighbors``), giving a
        symmetric adjacency. When ``None`` (default) the trimesh convention is used:
        ``symmetric = not equal_weight``.

    Returns
    -------
    tuple of wp.array
        ``(rows, cols, vals)`` on ``faces.device``. Length ``3 * n_faces`` for the directed case,
        ``2 * n_unique_edges`` for the symmetric case.

    See Also
    --------
    [`laplacian`][triwarp.laplacian.laplacian]
    [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]
    """
    if symmetric is None:
        symmetric = not equal_weight
    device = faces.device
    equal_weight_flag = wp.int32(1 if equal_weight else 0)
    if not symmetric:
        # One directed triplet per triangle edge, matching trimesh's ``edges_to_coo(mesh.edges)``.
        edges = faces_to_edges(faces)
        m = int(edges.shape[0])
        rows = wp.empty(m, dtype=wp.int32, device=device)
        cols = wp.empty(m, dtype=wp.int32, device=device)
        vals = wp.empty(m, dtype=wp.float32, device=device)
        if m > 0:
            wp.launch(
                kernel_laplacian.laplacian_triplets_directed,
                dim=m,
                inputs=[edges, vertices, equal_weight_flag, rows, cols, vals],
                device=device,
            )
        return rows, cols, vals
    # Both directed pairs of each unique undirected edge, matching trimesh's ``vertex_neighbors``
    # (every neighbor counted once).
    unique_edges, _ = edges_unique(faces)
    m_unique = int(unique_edges.shape[0])
    rows = wp.empty(2 * m_unique, dtype=wp.int32, device=device)
    cols = wp.empty(2 * m_unique, dtype=wp.int32, device=device)
    vals = wp.empty(2 * m_unique, dtype=wp.float32, device=device)
    if m_unique > 0:
        wp.launch(
            kernel_laplacian.laplacian_triplets_symmetric,
            dim=m_unique,
            inputs=[unique_edges, vertices, equal_weight_flag, rows, cols, vals],
            device=device,
        )
    return rows, cols, vals


def laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    equal_weight: bool = True,
    symmetric: bool | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Row-normalized 1-ring averaging operator (uniform / umbrella Laplacian).

    Builds the sparse ``(n_vertices, n_vertices)`` matrix whose row ``i`` holds the weights of
    the neighbors of vertex ``i``, normalized so each row sums to ``1``. Applying it to vertex
    positions replaces each vertex by the weighted mean of its 1-ring, matching
    [`trimesh.smoothing.laplacian_calculation`][] with ``equal_weight``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    equal_weight
        If ``True`` all neighbors are weighted equally (``1 / degree``). If ``False`` neighbors
        are weighted by inverse edge length before normalization.
    symmetric
        Adjacency shape (see [`laplacian_entries`][triwarp.laplacian.laplacian_entries]). If
        ``False`` the directed ``mesh.edges`` adjacency is used (asymmetric on boundaries); if
        ``True`` the symmetric ``vertex_neighbors`` adjacency is used. When ``None`` (default)
        the trimesh convention ``symmetric = not equal_weight`` is used, so the default matches
        [`trimesh.smoothing.laplacian_calculation`][] for both weightings. The two choices differ
        only on meshes with an open boundary.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` row-stochastic matrix in 1x1-block BSR form on
        ``vertices.device``. Isolated vertices (empty rows) map to themselves under
        [`filter_laplacian`][triwarp.smoothing.filter_laplacian] et al.

    See Also
    --------
    [`laplacian_entries`][triwarp.laplacian.laplacian_entries]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`trimesh.smoothing.laplacian_calculation`][]
    """
    n_vertices = int(vertices.shape[0])
    device = vertices.device
    rows, cols, vals = laplacian_entries(
        vertices, faces, equal_weight=equal_weight, symmetric=symmetric
    )
    operator = wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )
    if n_vertices > 0 and operator.nnz > 0:
        wp.launch(
            kernel_laplacian.row_normalize,
            dim=n_vertices,
            inputs=[operator.offsets, operator.values],
            device=device,
        )
    return operator


def uniform_laplacian(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float32]:
    """
    Combinatorial (graph) Laplacian ``L = A - diag(deg)`` from mesh connectivity.

    Builds the sparse ``(n_vertices, n_vertices)`` matrix with unit off-diagonal weights on every
    undirected edge and the negated vertex degree on the diagonal, ignoring geometry. Mirrors
    libigl's uniform-weight ``igl::harmonic`` variant (``L = A - diag(rowsum(A))`` from
    ``igl::adjacency_matrix``). Diagonal entries are **negative** (each row sums to zero), so ``-L``
    is positive semi-definite — the same sign convention as
    [`cotmatrix`][triwarp.laplacian.cotmatrix]. This is the operator behind the Tutte embedding
    [`tutte`][triwarp.parametrization.tutte], whose interior block ``-L_uu = diag(deg) - A`` is a
    diagonally dominant M-matrix (hence positive definite for a well-posed Dirichlet problem).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only the count and device are used; positions do
        not affect the uniform weights.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` combinatorial Laplacian in 1x1-block BSR form on
        ``vertices.device``.

    See Also
    --------
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`laplacian`][triwarp.laplacian.laplacian]
    [`tutte`][triwarp.parametrization.tutte]
    """
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device

    if n_faces == 0:
        return wps.bsr_from_triplets(
            n_vertices,
            n_vertices,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            prune_numerical_zeros=False,
        )

    # Symmetric unit-weight adjacency: each undirected edge emits both directed (a, b) and (b, a)
    # triplets with weight 1, matching ``igl::adjacency_matrix`` (all non-zeros forced to one).
    rows, cols, vals = laplacian_entries(vertices, faces, equal_weight=True, symmetric=True)
    adjacency = wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )

    # Vertex degrees as the row sums ``A @ 1``, then ``L = A - diag(deg)``.
    degree = wp.empty(n_vertices, dtype=wp.float32, device=device)
    ones = wp.ones(n_vertices, dtype=wp.float32, device=device)
    wps.bsr_mv(adjacency, ones, degree, alpha=1.0, beta=0.0)
    return wps.bsr_axpy(x=adjacency, y=wps.bsr_diag(diag=degree), alpha=1.0, beta=-1.0)


def mass_matrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.float32]:
    """
    Per-vertex barycentric lumped mass (diagonal of ``igl::massmatrix``).

    Each triangle donates a third of its area to each of its three vertices, so entry ``i`` is
    the summed one-third incident-face area at vertex ``i``. This is the
    ``MASSMATRIX_TYPE_BARYCENTRIC`` lumping; it is the diagonal of
    [`mass_matrix`][triwarp.laplacian.mass_matrix].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n_vertices`` diagonal on ``vertices.device``.

    See Also
    --------
    [`mass_matrix`][triwarp.laplacian.mass_matrix]
    """
    n_vertices = int(vertices.shape[0])
    device = vertices.device
    mass = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    n_faces = int(faces.shape[0]) // 3
    if n_faces > 0:
        _, areas = face_normals_and_areas(vertices, faces)
        wp.launch(
            kernel_scatter.scatter_face_thirds,
            dim=n_faces,
            inputs=[faces, areas, wp.float32(3.0), mass],
            device=device,
        )
    return mass


def mass_matrix(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wps.BsrMatrix[wp.float32]:
    """
    Diagonal barycentric lumped mass matrix (``igl::massmatrix``, barycentric).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` diagonal matrix in 1x1-block BSR form on
        ``vertices.device``.

    See Also
    --------
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    return wps.bsr_diag(diag=mass_matrix_entries(vertices, faces))
