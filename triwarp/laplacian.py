"""
Discrete Laplacians on a triangle mesh, and the intrinsic repairs that keep them well-behaved.

Every Laplacian this package builds lives here: the cotangent (stiffness) operator and its
vector-valued sibling, the combinatorial and row-normalized 1-ring operators, and the lumped mass
matrix that pairs with them.

The cotangent operator needs only *edge lengths*, not vertex positions — which is what makes it
repairable without moving anything. A sliver triangle produces a huge cotangent weight and solves
that are ill-conditioned; a *degenerate* one, whose three edge lengths fail the triangle inequality
outright (common after ``float32`` rounding, decimation, or a boolean), has no finite weight at all.
The assembly refuses to divide by such a face's zero area, so it contributes nothing — the operator
stays finite, but that face's edge couplings are simply **missing** from it, which is a wrong
operator rather than an unusable one.

[`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic] (Sharp & Crane 2020) fixes both by adding
one global constant to every edge length — the smallest that restores the triangle inequality with a
margin. The perturbation is slight and uniform, which beats the alternatives: the operator stays
symmetric, no vertex moves, no connectivity changes, and a mesh that is already fine gets
``delta = 0`` and is untouched. Mollification makes the weights *finite*;
[`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay] (in
[`triwarp.remesh`][triwarp.remesh], beside the extrinsic flipper) makes them *non-negative*.
[`robust_laplacian`][triwarp.laplacian.robust_laplacian] combines them.

Beyond the Laplacians themselves, the module builds the higher-order operators assembled *from*
them: the integrated k-harmonic operator
[`harmonic_integrated`][triwarp.laplacian.harmonic_integrated] (the quadratic form behind
biharmonic interpolation), the Hessian smoothness energies
[`hessian_energy`][triwarp.laplacian.hessian_energy] /
[`curved_hessian_energy`][triwarp.laplacian.curved_hessian_energy] (its natural-boundary
alternatives), and the edge-based Crouzeix-Raviart pair
[`crouzeix_raviart_cotmatrix`][triwarp.laplacian.crouzeix_raviart_cotmatrix] /
[`crouzeix_raviart_massmatrix`][triwarp.laplacian.crouzeix_raviart_massmatrix].
"""

from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TOLERANCE_MOLLIFY
from triwarp.edges import edges_unique, face_edge_lengths, faces_to_edges
from triwarp.kernels import laplacian as kernel_laplacian
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import triangles as kernel_triangles
from triwarp.reduce import max as reduce_max
from triwarp.tangent_space import halfedge_transport_angles
from triwarp.triangles import face_normals_and_areas


def face_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float64],
    *,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.vec3d]:
    """
    Gradient of a per-vertex scalar field inside each face, as a vector in that face's plane.

    The piecewise-linear gradient is constant per triangle:
    ``grad = 1/(2A) * sum_k values[k] * (n x e_k)``, with ``e_k`` the counter-clockwise edge
    opposite corner ``k``. It satisfies ``dot(grad, e) == values[end] - values[start]`` for every
    edge of the face, which is the property that makes it the discrete gradient rather than a finite
    difference.

    Accumulated in ``float64`` and returned as ``wp.vec3d``: the fields this serves decay
    exponentially (diffused heat, geodesic distance), and a ``float32`` sum of the three cross
    products loses the far field. A degenerate face gets the zero vector rather than a division by
    its zero area.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    values
        ``(n_vertices,)`` ``wp.float64`` scalar field.
    face_normals, face_areas
        Optional precomputed per-face unit normals and areas, as returned by
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]. Recomputed when
        either is ``None``.

    Returns
    -------
    wp.array[wp.vec3d]
        ``(n_faces,)`` gradient vectors on ``vertices.device``.

    Notes
    -----
    ``igl.grad(V, F)`` is the same operator in *matrix* form, a sparse ``(3 * n_faces, n_vertices)``
    map whose product with the field stacks the gradients as ``[all x; all y; all z]``. triwarp
    returns the applied result instead of the matrix, because that is what every in-repo consumer
    wants -- the heat method takes the normalized gradient face by face and never needs the operator
    itself. ``tests/test_laplacian.py`` compares the two through that product.

    See Also
    --------
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
    ``igl.grad``
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    gradients = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    if n_faces == 0:
        return gradients

    if face_normals is None or face_areas is None:
        face_normals, face_areas = face_normals_and_areas(vertices, faces)
    wp.launch(
        kernel_triangles.face_gradients,
        dim=n_faces,
        inputs=[vertices, faces, face_normals, face_areas, values, gradients],
        device=device,
    )
    return gradients


def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float32
) -> twt.Array2dFloat:
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
    dtype
        Scalar type of the returned weights: ``wp.float32`` (default) or ``wp.float64``. The weights
        are computed in float32 (the vertex precision) and cast to ``dtype`` on write; request
        ``wp.float64`` to feed a native float64 [`cotmatrix`][triwarp.laplacian.cotmatrix] build.

    Returns
    -------
    twt.Array2dFloat
        Shape ``(n_faces, 3)`` on ``faces.device``. Empty ``(0, 3)`` when ``n_faces == 0``.

    See Also
    --------
    [`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return twt.empty_float_2d((0, 3), dtype=dtype, device=device)

    out_cot = twt.empty_float_2d((n_faces, 3), dtype=dtype, device=device)
    wp.launch(
        kernel_laplacian.cotmatrix_entries,
        dim=n_faces,
        inputs=[vertices, faces, out_cot],
        device=device,
    )
    return twt.as_array2d_float(out_cot, dtype=dtype)


def cotmatrix_entries_intrinsic(
    edge_lengths: twt.Array2dFloat32, dtype: type = wp.float32
) -> twt.Array2dFloat:
    """
    Per-triangle half-cotangent weights from edge lengths.

    (``igl::cotmatrix_entries`` intrinsic overload).

    Each row gives the three edge lengths opposite vertices 0, 1, and 2 of the corresponding
    triangle.

    Parameters
    ----------
    edge_lengths
        ``(n_faces, 3)`` ``float32`` edge lengths on the target device.
    dtype
        Scalar type of the returned weights: ``wp.float32`` (default) or ``wp.float64``.

    Returns
    -------
    twt.Array2dFloat
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
        return twt.empty_float_2d((0, 3), dtype=dtype, device=device)

    out_cot = twt.empty_float_2d((n_faces, 3), dtype=dtype, device=device)
    wp.launch(
        kernel_laplacian.cotmatrix_entries_intrinsic,
        dim=n_faces,
        inputs=[edge_lengths, out_cot],
        device=device,
    )
    return twt.as_array2d_float(out_cot, dtype=dtype)


def cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: twt.Array2dFloat | None = None,
    dtype: type = wp.float32,
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
        computed from ``vertices`` and ``faces`` in ``dtype``. May be ``float32`` or ``float64``
        regardless of ``dtype``: the assembly kernel casts them to the matrix precision.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the matrix feeds an ill-conditioned solve (e.g. the biharmonic operator
        in [`harmonic`][triwarp.parametrization.harmonic]); the entries are always built in a single
        ``bsr_from_triplets`` in the requested precision.

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
            wp.empty(0, dtype=dtype, device=device),
            prune_numerical_zeros=False,
        )

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=dtype)

    n_triplets = 12 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=dtype, device=device)
    # One generic kernel handles both precisions: it casts the (float32 or float64) half-cotangent
    # weights to the matrix dtype, assembling a native float32/float64 matrix in a single build.
    wp.launch(
        kernel_laplacian.cotmatrix_triplets,
        dim=n_faces,
        inputs=[faces, cot_entries, rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )


def robust_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    dtype: type = wp.float32,
    *,
    use_intrinsic_delaunay: bool = True,
) -> wps.BsrMatrix[wp.float32]:
    """
    Cotangent Laplacian that a bad triangulation cannot poison, via mollification and flips.

    Two independent repairs, both intrinsic — no vertex moves, so the surface is unchanged:

    * **mollification** adds one constant to every edge length so that no triangle is degenerate,
      which is what keeps the weights finite at all
      ([`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]);
    * **intrinsic Delaunay flips** retriangulate until no edge has a negative cotangent weight,
      which is what makes the operator satisfy a maximum principle
      ([`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay]).

    With both on this is ``igl::intrinsic_delaunay_cotmatrix``, and the operator
    ``potpourri3d``'s ``use_robust=True`` solvers build. Turn the flips off for a drop-in
    [`cotmatrix`][triwarp.laplacian.cotmatrix] that keeps every edge coupling: the plain operator
    drops the ones belonging to a degenerate face, because a zero-area triangle has no finite
    cotangent to contribute.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    epsilon
        Triangle-inequality margin, relative to the mean edge length. The default ``1e-5`` is
        Sharp & Crane's.
    dtype
        Scalar type of the matrix: ``wp.float32`` (default) or ``wp.float64``.
    use_intrinsic_delaunay
        Flip to the intrinsic Delaunay triangulation first (default), the name and the default
        ``potpourri3d``'s solvers use. The vertex set — and so the matrix's shape and meaning — is
        the same either way; only the edges it sums over change.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` cotangent stiffness matrix, in ``cotmatrix``'s sign convention
        (negative diagonal, so ``-L`` is positive semi-definite).

    See Also
    --------
    [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay]
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    """
    if use_intrinsic_delaunay:
        intrinsic_faces, lengths, _ = tw.remesh.intrinsic_delaunay(vertices, faces, epsilon=epsilon)
    else:
        intrinsic_faces = faces
        lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    entries = cotmatrix_entries_intrinsic(lengths, dtype=dtype)
    return cotmatrix(vertices, intrinsic_faces, cot_entries=entries, dtype=dtype)


def mollify_intrinsic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    edge_lengths: twt.Array2dFloat32 | None = None,
) -> tuple[twt.Array2dFloat32, float]:
    """
    Add the smallest constant to every edge length that makes every triangle non-degenerate.

    Returns the mollified ``(n_faces, 3)`` length table and the constant used. The constant is a
    single global number, which is the point: it keeps the perturbation uniform, so the operators
    built from these lengths stay symmetric and no triangle is treated as a special case.

    ``delta`` is zero, and the lengths unchanged, whenever every triangle already satisfies the
    triangle inequality with margin ``epsilon * mean_edge_length``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    epsilon
        Required margin, relative to the mean edge length.
    edge_lengths
        Optional precomputed ``(n_faces, 3)`` table from
        [`face_edge_lengths`][triwarp.edges.face_edge_lengths]; recomputed here when ``None``.

    Returns
    -------
    lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` mollified edge lengths, column ``e`` opposite corner ``e``.
    delta : float
        The constant added to every length. Reading it costs one host readback, and it is returned
        because it is the honest measure of how much the geometry had to be changed.

    See Also
    --------
    [`robust_laplacian`][triwarp.laplacian.robust_laplacian]
    [`face_edge_lengths`][triwarp.edges.face_edge_lengths]
    [`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.empty_float32_2d((0, 3), device=device), 0.0

    if edge_lengths is None:
        edge_lengths = face_edge_lengths(vertices, faces)

    scale = float(reduce_max(edge_lengths))
    slack = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_laplacian.triangle_inequality_slack,
        dim=n_faces,
        inputs=[edge_lengths, wp.float32(epsilon * scale), slack],
        device=device,
    )
    delta = float(reduce_max(slack))
    if delta <= 0.0:
        return twt.as_array2d_float32(edge_lengths), 0.0

    mollified = twt.empty_float32_2d((n_faces, 3), device=device)
    wp.map(kernel_laplacian.add_constant, edge_lengths, wp.float32(delta), out=mollified)
    return twt.as_array2d_float32(mollified), delta


def connection_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: twt.Array2dFloat | None = None,
    transport_angles: wp.array[wp.float32] | None = None,
) -> wps.BsrMatrix[wp.float64]:
    """
    Vector (connection) Laplacian: the cotangent Laplacian for *tangent vector* fields.

    Same cotangent weights and same sparsity as [`cotmatrix`][triwarp.laplacian.cotmatrix], but each
    scalar becomes a ``2 x 2`` block and each off-diagonal weight is multiplied by the rotation that
    re-expresses a tangent vector in the neighbouring vertex's frame
    ([`halfedge_transport_angles`][triwarp.tangent_space.halfedge_transport_angles]). Without those
    rotations a difference between vectors at two vertices would subtract components measured from
    two unrelated reference directions.

    Assembled ``float64`` and **positive semi-definite** (positive diagonal) — the opposite sign to
    ``cotmatrix``'s igl convention — because its consumers feed it to a conjugate-gradient solve. It
    is symmetric, since transporting from ``i`` to ``j`` and back are inverse rotations.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    cot_entries
        Optional precomputed ``(n_faces, 3)`` half-cotangent weights from
        [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries].
    transport_angles
        Optional precomputed per-halfedge
        [`halfedge_transport_angles`][triwarp.tangent_space.halfedge_transport_angles].

        There is deliberately no ``frames`` argument. The gauge is fixed by the one-ring
        flattening — angles are measured from each vertex's first outgoing halfedge, the same
        convention [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] uses to
        pick ``basis_x`` — so a caller-supplied frame cannot change these angles, and solutions are
        already consistent with the frames that convention produces.

    Returns
    -------
    warp.sparse.BsrMatrix
        ``(n_vertices, n_vertices)`` matrix of ``wp.mat22d`` blocks on ``vertices.device``.

    See Also
    --------
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`halfedge_transport_angles`][triwarp.tangent_space.halfedge_transport_angles]
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
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
            wp.empty(0, dtype=wp.mat22d, device=device),
            prune_numerical_zeros=False,
        )

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=wp.float64)
    if transport_angles is None:
        transport_angles = halfedge_transport_angles(vertices, faces)

    n_triplets = 12 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=wp.mat22d, device=device)
    wp.launch(
        kernel_laplacian.connection_laplacian_triplets,
        dim=n_faces,
        inputs=[faces, cot_entries, transport_angles, rows, cols, vals],
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
    dtype: type = wp.float32,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], twt.Array1dFloat]:
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
    dtype
        Scalar type of the returned ``vals``: ``wp.float32`` (default) or ``wp.float64``. The
        assembly kernel casts the float32 edge weights to ``dtype`` so the matrix built from these
        triplets is native float32/float64.

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
        vals = wp.empty(m, dtype=dtype, device=device)
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
    vals = wp.empty(2 * m_unique, dtype=dtype, device=device)
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
    dtype: type = wp.float32,
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
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the operator feeds a linear-system solve; the matrix is built and
        row-normalized natively in the requested precision (single ``bsr_from_triplets``).

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
        vertices, faces, equal_weight=equal_weight, symmetric=symmetric, dtype=dtype
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
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float32
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
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` for the higher-power Tutte operator in
        [`tutte`][triwarp.parametrization.tutte].

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
            wp.empty(0, dtype=dtype, device=device),
            prune_numerical_zeros=False,
        )

    # Symmetric unit-weight adjacency: each undirected edge emits both directed (a, b) and (b, a)
    # triplets with weight 1, matching ``igl::adjacency_matrix`` (all non-zeros forced to one).
    # The triplet values are emitted natively in ``dtype`` (single build, no recast).
    rows, cols, vals = laplacian_entries(
        vertices, faces, equal_weight=True, symmetric=True, dtype=dtype
    )
    adjacency = wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )

    # Vertex degrees as the row sums ``A @ 1``, then ``L = A - diag(deg)``.
    degree = wp.empty(n_vertices, dtype=dtype, device=device)
    ones = wp.ones(n_vertices, dtype=dtype, device=device)
    wps.bsr_mv(adjacency, ones, degree, alpha=1.0, beta=0.0)
    return wps.bsr_axpy(x=adjacency, y=wps.bsr_diag(diag=degree), alpha=1.0, beta=-1.0)


def mass_matrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float32
) -> twt.Array1dFloat:
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
    dtype
        Scalar type of the returned diagonal: ``wp.float32`` (default) or ``wp.float64``. Request
        ``wp.float64`` to feed a float64 solve (e.g. geodesic heat method, implicit fairing)
        without a downstream recast.

    Returns
    -------
    twt.Array1dFloat
        Length-``n_vertices`` diagonal on ``vertices.device``.

    See Also
    --------
    [`mass_matrix`][triwarp.laplacian.mass_matrix]
    """
    n_vertices = int(vertices.shape[0])
    device = vertices.device
    mass = wp.zeros(n_vertices, dtype=dtype, device=device)
    n_faces = int(faces.shape[0]) // 3
    if n_faces > 0:
        _, areas = face_normals_and_areas(vertices, faces)
        if dtype != wp.float32:
            # scatter_face_thirds shares one float dtype across areas/count/mass; promote the
            # float32 face areas so the scatter specializes to the requested precision.
            areas_typed = wp.empty(n_faces, dtype=dtype, device=device)
            wp.utils.array_cast(areas, areas_typed)
            areas = areas_typed
        wp.launch(
            kernel_scatter.scatter_face_thirds,
            dim=n_faces,
            inputs=[faces, areas, dtype(3.0), mass],
            device=device,
        )
    return mass


def mass_matrix(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float32
) -> wps.BsrMatrix[wp.float32]:
    """
    Diagonal barycentric lumped mass matrix (``igl::massmatrix``, barycentric).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``. Use
        ``wp.float64`` when the mass matrix feeds a float64 linear-system solve.

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
    return wps.bsr_diag(diag=mass_matrix_entries(vertices, faces, dtype=dtype))


def harmonic_integrated(
    laplacian: wps.BsrMatrix[wp.float32], mass: twt.Array1dFloat | None = None, k: int = 2
) -> wps.BsrMatrix[wp.float32]:
    """
    Integrated k-harmonic operator ``Q = (-L) (M^-1 (-L))^(k-1)`` from a Laplacian and a mass.

    The quadratic form whose minimizers are k-harmonic functions: ``k == 1`` gives the Dirichlet
    energy ``-L`` (positive semi-definite for a [`cotmatrix`][triwarp.laplacian.cotmatrix]-sign
    Laplacian), ``k == 2`` the biharmonic operator ``L M^-1 L`` behind
    [`harmonic`][triwarp.parametrization.harmonic]'s smooth interpolation, and so on
    (``igl::harmonic_integrated_from_laplacian_and_mass``). Like igl's, the composition is not
    numerically robust for ``k > 2`` — the entries grow as the k-th power of the inverse mesh
    size — so high powers want a float64 ``laplacian``.

    Each power is assembled by one triplet pass over matching CSR rows —
    ``(A M^-1 B)_ij = sum_t A_ti M_t^-1 B_tj`` with both operands symmetric — followed by a single
    ``bsr_from_triplets``. Deliberately **no** ``warp.sparse.bsr_mm``: the chained sparse triple
    product is exactly the shape that reproduces its nondeterministic-output bug (Warp 1.15,
    ``issue_report.md``).

    Parameters
    ----------
    laplacian
        Square 1x1-block BSR Laplacian in igl's sign convention (negative diagonal, each row
        summing to zero), e.g. from [`cotmatrix`][triwarp.laplacian.cotmatrix] or
        [`uniform_laplacian`][triwarp.laplacian.uniform_laplacian].
    mass
        Length-``n_vertices`` lumped mass diagonal, e.g. from
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]; cast to the Laplacian's
        scalar type if it differs. ``None`` (identity mass) composes plain powers of ``-L``, the
        [`tutte`][triwarp.parametrization.tutte] convention. Zero entries are treated as killed
        degrees of freedom (their rows contribute nothing), matching ``igl::invert_diag``.
    k
        Harmonic power (``>= 1``): 1 harmonic, 2 biharmonic, 3 triharmonic, ...

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite operator in the Laplacian's
        scalar type on its device.

    Raises
    ------
    ValueError
        If ``k < 1``.

    See Also
    --------
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    [`hessian_energy`][triwarp.laplacian.hessian_energy]
    [`harmonic`][triwarp.parametrization.harmonic]
    """
    if k < 1:
        raise ValueError(f"harmonic power k must be >= 1, got {k}.")
    negated = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    if k == 1:
        return negated

    dtype = laplacian.values.dtype
    n_rows = int(laplacian.nrow)
    device = laplacian.values.device
    if mass is None:
        inverse_mass = wp.ones(n_rows, dtype=dtype, device=device)
    else:
        if mass.dtype != dtype:
            mass_typed = wp.empty(n_rows, dtype=dtype, device=device)
            wp.utils.array_cast(mass, mass_typed)
            mass = mass_typed
        inverse_mass = wp.empty(n_rows, dtype=dtype, device=device)
        wp.map(kernel_laplacian.reciprocal_or_zero, mass, out=inverse_mass)

    operator = negated
    for _ in range(k - 1):
        operator = _diagonal_sandwich(operator, inverse_mass, negated)
    return operator


def _diagonal_sandwich(
    a: wps.BsrMatrix[wp.float32], inverse_mass: wp.array[wp.Float], b: wps.BsrMatrix[wp.float32]
) -> wps.BsrMatrix[wp.float32]:
    """
    Assemble ``A diag(inverse_mass) B`` for symmetric ``A``, ``B`` by one triplet pass.

    Row ``t`` of the product is the outer product of ``A``'s and ``B``'s rows ``t`` scaled by the
    diagonal weight, so the whole product is one count kernel, one scan, one emission kernel and a
    single ``bsr_from_triplets`` — never ``bsr_mm``.
    """
    n_rows = int(a.nrow)
    device = inverse_mass.device
    counts = wp.empty(n_rows, dtype=wp.int32, device=device)
    wp.launch(
        kernel_laplacian.sandwich_row_counts,
        dim=n_rows,
        inputs=[a.offsets, b.offsets, inverse_mass, counts],
        device=device,
    )
    segment_offsets = wp.zeros(n_rows + 1, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=segment_offsets[1:], inclusive=True)
    # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
    n_triplets = int(segment_offsets[n_rows : n_rows + 1].numpy()[0])

    dtype = a.values.dtype
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=dtype, device=device)
    if n_triplets > 0:
        wp.launch(
            kernel_laplacian.sandwich_row_triplets,
            dim=n_rows,
            inputs=[
                a.offsets,
                a.columns,
                a.values,
                b.offsets,
                b.columns,
                b.values,
                inverse_mass,
                segment_offsets,
                rows,
                cols,
                vals,
            ],
            device=device,
        )
    return wps.bsr_from_triplets(n_rows, n_rows, rows, cols, vals, prune_numerical_zeros=False)


def hessian_energy(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float64
) -> wps.BsrMatrix[wp.float64]:
    """
    Hessian smoothness energy with natural boundary conditions (``igl::hessian_energy``).

    The mixed-FEM quadratic form ``Q = H^T M^-1 H`` of Stein et al. 2018, *Natural Boundary
    Conditions for Smoothing in Geometry Processing*: ``x' Q x`` integrates the squared Hessian of
    the piecewise-linear field ``x``, so minimizing it smooths **without** the boundary distortion
    the clamped biharmonic operator
    ([`harmonic_integrated`][triwarp.laplacian.harmonic_integrated] at ``k == 2``) produces —
    linear functions are exactly in its null space, boundary or not. ``M`` is the Voronoi lumped
    mass with boundary degrees of freedom killed, per the reference.

    Rather than materializing the sparse ``(9 n_faces, n_vertices)`` stacked Hessian ``H``, the
    product is contracted analytically over its nine component pairs and assembled in one triplet
    pass per vertex: ``Q_ij = sum_k M_k^-1 sum_{f,g ni k} A_f A_g (g_fk . g_gk)(g_fi . g_gj)``
    with ``g_fc`` corner ``c``'s hat-function gradient in face ``f``. The per-vertex triplet count
    is ``9 * valence^2``, so cost is quadratic in valence. A degenerate face contributes nothing
    (igl emits NaN there).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``.
        Float64 is the default, unlike the first-order operators in this module, because the
        entries scale as the inverse fourth power of the mesh size and the operator exists to be
        solved against.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    See Also
    --------
    [`curved_hessian_energy`][triwarp.laplacian.curved_hessian_energy]
    [`harmonic_integrated`][triwarp.laplacian.harmonic_integrated]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    """
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    if n_faces == 0:
        return _empty_square_operator(n_vertices, dtype, device)

    gradients = wp.empty(3 * n_faces, dtype=wp.vec3d, device=device)
    areas = wp.empty(n_faces, dtype=wp.float64, device=device)
    wp.launch(
        kernel_laplacian.hessian_corner_gradients,
        dim=n_faces,
        inputs=[vertices, faces, gradients, areas],
        device=device,
    )

    mass = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_laplacian.voronoi_mass, dim=n_faces, inputs=[vertices, faces, mass], device=device
    )
    inverse_mass = _interior_inverse(vertices, faces, mass)

    vf_offsets, vertex_faces = tw.adjacency.vertex_face_adjacency(faces, n_vertices=n_vertices)
    counts = wp.empty(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_laplacian.hessian_energy_counts,
        dim=n_vertices,
        inputs=[vf_offsets, inverse_mass, counts],
        device=device,
    )
    segment_offsets = wp.zeros(n_vertices + 1, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=segment_offsets[1:], inclusive=True)
    # Host readback: only the device knows the scan total, and it sizes the triplet buffers.
    n_triplets = int(segment_offsets[n_vertices : n_vertices + 1].numpy()[0])

    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=dtype, device=device)
    if n_triplets > 0:
        wp.launch(
            kernel_laplacian.hessian_energy_triplets,
            dim=n_vertices,
            inputs=[
                faces,
                vf_offsets,
                vertex_faces,
                gradients,
                areas,
                inverse_mass,
                segment_offsets,
                rows,
                cols,
                vals,
            ],
            device=device,
        )
    return wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )


def _interior_inverse(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], mass: wp.array[wp.float64]
) -> wp.array[wp.float64]:
    """Invert a mass diagonal, first zeroing (in place) its boundary degrees of freedom."""
    device = mass.device
    boundary = tw.boundary.boundary_vertex_indices(vertices, faces)
    if int(boundary.shape[0]) > 0:
        wp.launch(
            kernel_laplacian.zero_at_indices,
            dim=int(boundary.shape[0]),
            inputs=[boundary, mass],
            device=device,
        )
    inverse_mass = wp.empty(int(mass.shape[0]), dtype=wp.float64, device=device)
    wp.map(kernel_laplacian.reciprocal_or_zero, mass, out=inverse_mass)
    return inverse_mass


def curved_hessian_energy(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], dtype: type = wp.float64
) -> wps.BsrMatrix[wp.float64]:
    """
    Curved Hessian smoothness energy on the Crouzeix-Raviart discretization.

    ``igl::curved_hessian_energy``, from Stein et al. 2020, *A Smoothness Energy without Boundary
    Distortion for Curved Surfaces*: where [`hessian_energy`][triwarp.laplacian.hessian_energy]
    treats the surface as locally flat, this one carries the Gaussian curvature into the operator
    through a per-vertex angle-defect correction, so the energy is intrinsic to the curved surface
    rather than to its triangles' planes. Constant functions are exactly in its null space.

    Assembled as ``Q = D^T M^-1 (L + K) M^-1 D`` over edge-based Crouzeix-Raviart vector elements
    — ``D`` the scalar-to-CR-vector gradient, ``M`` the CR vector mass, ``L`` the CR vector
    Laplacian and ``K`` the curvature correction — but contracted per face in one pass: ``L + K``
    couples edges within a face only, so each face emits its own 6x6 block sandwiched between its
    edges' gradient rows (a fixed 144 triplets per face), and no intermediate ``(2 n_edges, ...)``
    matrix or sparse product exists. Requires an edge-manifold mesh, like the igl original (which
    asserts it); a degenerate face contributes nothing.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float64`` (default) or ``wp.float32``,
        with float64 the default for the same conditioning reason as
        [`hessian_energy`][triwarp.laplacian.hessian_energy].

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_vertices, n_vertices)`` positive semi-definite energy matrix on
        ``vertices.device``.

    See Also
    --------
    [`hessian_energy`][triwarp.laplacian.hessian_energy]
    [`crouzeix_raviart_cotmatrix`][triwarp.laplacian.crouzeix_raviart_cotmatrix]
    [`vertex_defects`][triwarp.vertices.vertex_defects]
    """
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    if n_faces == 0:
        return _empty_square_operator(n_vertices, dtype, device)

    unique_edges, inverse = edges_unique(faces, n_vertices=n_vertices)
    n_edges = int(unique_edges.shape[0])

    angles = wp.empty((n_faces, 3), dtype=wp.float64, device=device)
    angle_sums = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_laplacian.internal_angles_and_sums,
        dim=n_faces,
        inputs=[vertices, faces, angles, angle_sums],
        device=device,
    )
    # Angle defect, zeroed on the boundary (curvature is only corrected at interior vertices),
    # weighted by the actual angle sum -- igl::cr_vector_curvature_correction's kappa scaling.
    kappa = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_laplacian.angle_defect_from_sum, angle_sums, out=kappa)
    boundary = tw.boundary.boundary_vertex_indices(vertices, faces)
    if int(boundary.shape[0]) > 0:
        wp.launch(
            kernel_laplacian.zero_at_indices,
            dim=int(boundary.shape[0]),
            inputs=[boundary, kappa],
            device=device,
        )
    scaled_kappa = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_laplacian.divide_or_zero, kappa, angle_sums, out=scaled_kappa)

    mass = wp.zeros(n_edges, dtype=wp.float64, device=device)
    wp.launch(
        kernel_laplacian.crouzeix_raviart_mass_diag,
        dim=n_faces,
        inputs=[vertices, faces, inverse, mass],
        device=device,
    )
    inverse_mass = wp.empty(n_edges, dtype=wp.float64, device=device)
    wp.map(kernel_laplacian.reciprocal_or_zero, mass, out=inverse_mass)

    edge_halfedges = wp.full((n_edges, 2), -1, dtype=wp.int32, device=device)
    cursor = wp.zeros(n_edges, dtype=wp.int32, device=device)
    wp.launch(
        kernel_laplacian.scatter_edge_halfedges,
        dim=3 * n_faces,
        inputs=[inverse, cursor, edge_halfedges],
        device=device,
    )
    vertex_slots = wp.full((n_edges, 4), -1, dtype=wp.int32, device=device)
    par = wp.zeros((n_edges, 4), dtype=wp.float64, device=device)
    perp = wp.zeros((n_edges, 4), dtype=wp.float64, device=device)
    wp.launch(
        kernel_laplacian.cr_gradient_rows,
        dim=n_edges,
        inputs=[vertices, faces, unique_edges, edge_halfedges, vertex_slots, par, perp],
        device=device,
    )

    n_triplets = 144 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=dtype, device=device)
    wp.launch(
        kernel_laplacian.curved_hessian_triplets,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            inverse,
            angles,
            scaled_kappa,
            inverse_mass,
            vertex_slots,
            par,
            perp,
            rows,
            cols,
            vals,
        ],
        device=device,
    )
    return wps.bsr_from_triplets(
        n_vertices, n_vertices, rows, cols, vals, prune_numerical_zeros=False
    )


def crouzeix_raviart_cotmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: twt.Array2dFloat | None = None,
    dtype: type = wp.float32,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart cotangent stiffness matrix (``igl::crouzeix_raviart_cotmatrix``).

    The nonconforming-FEM sibling of [`cotmatrix`][triwarp.laplacian.cotmatrix]: degrees of
    freedom live on edge midpoints, so the matrix is ``(n_edges, n_edges)`` and each face couples
    its three edges pairwise with minus four times the half-cotangent at their shared corner
    (positive diagonal — the igl sign convention for this operator, opposite to ``cotmatrix``'s).
    Rows follow [`edges_unique`][triwarp.edges.edges_unique]'s edge numbering. Requires an
    edge-manifold mesh, like the igl original (which asserts it).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    cot_entries
        Optional precomputed ``(n_faces, 3)`` weights from
        [`cotmatrix_entries`][triwarp.laplacian.cotmatrix_entries]; computed in ``dtype`` when
        ``None``.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        Optional precomputed edge numbering from
        [`edges_unique`][triwarp.edges.edges_unique] — pass both or neither. Sharing it with
        [`crouzeix_raviart_massmatrix`][triwarp.laplacian.crouzeix_raviart_massmatrix] keeps the
        two operators on identical rows without recomputing the sort.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_edges, n_edges)`` stiffness matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If exactly one of ``unique_edges`` / ``edge_map`` is provided.

    See Also
    --------
    [`crouzeix_raviart_massmatrix`][triwarp.laplacian.crouzeix_raviart_massmatrix]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`edges_unique`][triwarp.edges.edges_unique]
    """
    unique_edges, edge_map = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return _empty_square_operator(n_edges, dtype, device)

    if cot_entries is None:
        cot_entries = cotmatrix_entries(vertices, faces, dtype=dtype)

    n_triplets = 12 * n_faces
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    vals = wp.empty(n_triplets, dtype=dtype, device=device)
    wp.launch(
        kernel_laplacian.crouzeix_raviart_cotmatrix_triplets,
        dim=n_faces,
        inputs=[edge_map, cot_entries, rows, cols, vals],
        device=device,
    )
    return wps.bsr_from_triplets(n_edges, n_edges, rows, cols, vals, prune_numerical_zeros=False)


def crouzeix_raviart_massmatrix(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    dtype: type = wp.float32,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    edge_map: wp.array[wp.int32] | None = None,
) -> wps.BsrMatrix[wp.float32]:
    """
    Edge-based Crouzeix-Raviart mass matrix (``igl::crouzeix_raviart_massmatrix``).

    Diagonal ``(n_edges, n_edges)``: each face donates a third of its area to each of its three
    edges, so an interior edge's entry is a third of its two incident faces' summed area. Rows
    follow [`edges_unique`][triwarp.edges.edges_unique]'s edge numbering, the same numbering
    [`crouzeix_raviart_cotmatrix`][triwarp.laplacian.crouzeix_raviart_cotmatrix] uses.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    dtype
        Scalar block type of the assembled matrix: ``wp.float32`` (default) or ``wp.float64``.
    unique_edges, edge_map
        Optional precomputed edge numbering from
        [`edges_unique`][triwarp.edges.edges_unique] — pass both or neither.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(n_edges, n_edges)`` diagonal mass matrix in 1x1-block BSR form on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If exactly one of ``unique_edges`` / ``edge_map`` is provided.

    See Also
    --------
    [`crouzeix_raviart_cotmatrix`][triwarp.laplacian.crouzeix_raviart_cotmatrix]
    [`mass_matrix`][triwarp.laplacian.mass_matrix]
    [`edges_unique`][triwarp.edges.edges_unique]
    """
    unique_edges, edge_map = _edge_numbering(vertices, faces, unique_edges, edge_map)
    n_edges = int(unique_edges.shape[0])
    n_faces = int(faces.shape[0]) // 3
    device = faces.device

    mass = wp.zeros(n_edges, dtype=dtype, device=device)
    if n_faces > 0:
        wp.launch(
            kernel_laplacian.crouzeix_raviart_mass_diag,
            dim=n_faces,
            inputs=[vertices, faces, edge_map, mass],
            device=device,
        )
    return wps.bsr_diag(diag=mass)


def _edge_numbering(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32 | None,
    edge_map: wp.array[wp.int32] | None,
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
    """Validate or build the shared ``edges_unique`` numbering the edge-based operators index."""
    if (unique_edges is None) != (edge_map is None):
        raise ValueError("pass unique_edges and edge_map together, or neither.")
    if unique_edges is None or edge_map is None:
        unique_edges, edge_map = edges_unique(faces, n_vertices=int(vertices.shape[0]))
    return unique_edges, edge_map


def _empty_square_operator(
    n_rows: int, dtype: type, device: wp.DeviceLike
) -> wps.BsrMatrix[wp.float32]:
    """Zero-nnz square operator, the empty-mesh return shared by the assembly wrappers."""
    return wps.bsr_from_triplets(
        n_rows,
        n_rows,
        wp.empty(0, dtype=wp.int32, device=device),
        wp.empty(0, dtype=wp.int32, device=device),
        wp.empty(0, dtype=dtype, device=device),
        prune_numerical_zeros=False,
    )
