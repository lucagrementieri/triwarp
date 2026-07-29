"""
Intrinsic geometry: edge-length tables and the mollification that makes them usable.

The cotangent Laplacian only needs *edge lengths*, not vertex positions — which is what makes it
possible to repair a bad mesh without moving anything. A sliver triangle produces a huge cotangent
weight and a Laplacian whose solves are ill-conditioned or NaN; a *degenerate* one, where the three
edge lengths fail the triangle inequality outright (common after float32 rounding, decimation, or a
boolean), produces a negative area and no valid weight at all.

Intrinsic mollification (Sharp & Crane 2020) fixes both by adding one global constant to every edge
length — the smallest that makes every triangle satisfy the triangle inequality with a margin. It
changes the geometry slightly and uniformly, which is preferable to the alternatives: the operator
stays symmetric, no vertex moves, no connectivity changes, and the perturbation vanishes as the mesh
improves (a mesh that is already fine gets ``delta = 0`` and is untouched).

The second repair is [`intrinsic_delaunay`][triwarp.intrinsic.intrinsic_delaunay]: flipping edges,
again without moving anything, until no cotangent weight is negative. Mollification makes the
operator *finite*; the flips make it *well-behaved* — a Laplacian with non-negative weights obeys a
maximum principle, so its solves cannot invent extrema and are far better conditioned on a
badly-shaped mesh.

Use [`robust_laplacian`][triwarp.intrinsic.robust_laplacian] as a drop-in for
[`cotmatrix`][triwarp.laplacian.cotmatrix] whenever the input is not known to be clean; it does both
by default and is then ``igl::intrinsic_delaunay_cotmatrix``. Inside the heat method,
``heat_geodesic(..., use_robust=True)`` applies mollification only — see that function for why
retriangulating cannot be dropped into a pipeline whose later stages integrate over faces.
"""

from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import INT32_MAX
from triwarp.kernels import intrinsic as kernel_intrinsic
from triwarp.kernels import remesh as kernel_remesh
from triwarp.laplacian import cotmatrix, cotmatrix_entries_intrinsic
from triwarp.reduce import max as reduce_max

_MOLLIFY_EPSILON = 1e-5


def robust_laplacian(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = _MOLLIFY_EPSILON,
    dtype: type = wp.float32,
    *,
    use_intrinsic_delaunay: bool = True,
) -> wps.BsrMatrix[wp.float32]:
    """
    Cotangent Laplacian that a bad triangulation cannot poison, via mollification and flips.

    Two independent repairs, both intrinsic — no vertex moves, so the surface is unchanged:

    * **mollification** adds one constant to every edge length so that no triangle is degenerate,
      which is what keeps the weights finite at all
      ([`mollify_intrinsic`][triwarp.intrinsic.mollify_intrinsic]);
    * **intrinsic Delaunay flips** retriangulate until no edge has a negative cotangent weight,
      which is what makes the operator satisfy a maximum principle
      ([`intrinsic_delaunay`][triwarp.intrinsic.intrinsic_delaunay]).

    With both on this is ``igl::intrinsic_delaunay_cotmatrix``, and the operator
    ``potpourri3d``'s ``use_robust=True`` solvers build. Turn the flips off for a drop-in
    [`cotmatrix`][triwarp.laplacian.cotmatrix] that merely cannot produce NaN.

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
    [`intrinsic_delaunay`][triwarp.intrinsic.intrinsic_delaunay]
    [`mollify_intrinsic`][triwarp.intrinsic.mollify_intrinsic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    """
    if use_intrinsic_delaunay:
        intrinsic_faces, lengths, _ = intrinsic_delaunay(vertices, faces, epsilon=epsilon)
    else:
        intrinsic_faces = faces
        lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    entries = cotmatrix_entries_intrinsic(lengths, dtype=dtype)
    return cotmatrix(vertices, intrinsic_faces, cot_entries=entries, dtype=dtype)


def intrinsic_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = _MOLLIFY_EPSILON,
    max_iter: int = 100,
) -> tuple[wp.array[wp.int32], twt.Array2dFloat32, int]:
    """
    Retriangulate to the intrinsic Delaunay triangulation, without moving a vertex.

    Flips edges whose two opposite angles sum past ``pi`` — exactly the edges whose cotangent weight
    is negative — until none is left. The flip is *intrinsic*: the new edge is not a straight line
    in space but the geodesic across the two triangles, and its length comes from unfolding them
    into a plane and measuring the other diagonal. The surface, its vertices and its metric are all
    untouched; only which pairs of vertices count as connected changes, so every operator built from
    the result is a better-behaved operator for the *same* geometry.

    Its practical effect is that the cotangent weights all become non-negative, which is what a
    Laplacian needs to satisfy a maximum principle: no spurious extrema, no negative diffusion, far
    better conditioned solves on a badly-shaped mesh.

    Flips run in parallel rounds, each committing a conflict-free independent set (the same engine
    behind [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]) with the edge-length table carried
    alongside the connectivity.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Not modified.
    epsilon
        Mollification margin applied before flipping, relative to the mean edge length: a degenerate
        triangle has no well-defined angles to test.
    max_iter
        Cap on the number of parallel flip rounds.

    Returns
    -------
    intrinsic_faces : wp.array[wp.int32]
        Length-``3 * n_faces`` connectivity of the intrinsic triangulation, over the same vertices.
    edge_lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` intrinsic edge lengths for those faces, column ``e`` opposite corner ``e``.
    n_flips : int
        How many edges were flipped. Zero means the input was already intrinsically Delaunay.

    Notes
    -----
    A flip that would duplicate an existing edge is skipped rather than allowed to create a
    multi-edge, so a few non-Delaunay edges can survive on coarse meshes — geometry-central's
    signpost machinery represents those, this does not. The count is small in practice: on the
    fixtures used in the tests the result matches ``igl.intrinsic_delaunay_cotmatrix`` exactly.

    See Also
    --------
    [`robust_laplacian`][triwarp.intrinsic.robust_laplacian]
    [`mollify_intrinsic`][triwarp.intrinsic.mollify_intrinsic]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    intrinsic_faces = wp.clone(faces)
    if n_faces == 0:
        return intrinsic_faces, lengths, 0

    total = 0
    for _ in range(max_iter):
        edges_sorted = tw.edges.faces_to_edges(intrinsic_faces, sorted=True)
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            intrinsic_faces, edges_sorted, return_edges=True, n_vertices=n_vertices
        )
        n_interior = int(adjacency.shape[0])
        if n_interior == 0:
            break
        unshared = tw.adjacency.face_adjacency_unshared(intrinsic_faces, adjacency, adjacency_edges)
        keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
        sorted_keys, _order = tw.array.sort_pairs(keys)

        flip = wp.zeros(n_interior, dtype=wp.bool, device=device)
        quad = twt.empty_int32_2d((n_interior, 4), device=device)
        new_length = wp.empty(n_interior, dtype=wp.float32, device=device)
        wp.launch(
            kernel_intrinsic.intrinsic_delaunay_candidates,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                lengths,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                wp.uint64(n_vertices),
                flip,
                quad,
                new_length,
            ],
            device=device,
        )

        # Independent set: a flip commits only if it wins both incident faces and its new edge's
        # hashed slot, so no two committed flips share a face or invent the same edge.
        table = 1
        while table < 4 * n_interior + 1:
            table <<= 1
        face_claim = wp.full(n_faces, INT32_MAX, dtype=wp.int32, device=device)
        edge_claim = wp.full(table, INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                face_claim,
                edge_claim,
            ],
            device=device,
        )
        # Lengths first: this pass needs the *old* connectivity to know which corner holds which
        # vertex, and ``commit_flips`` is about to overwrite it.
        wp.launch(
            kernel_intrinsic.update_flipped_lengths,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                flip,
                quad,
                adjacency,
                new_length,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                lengths,
            ],
            device=device,
        )
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                intrinsic_faces,
                count,
            ],
            device=device,
        )
        committed = int(count.numpy()[0])
        total += committed
        if committed == 0:
            break
    return intrinsic_faces, lengths, total


def mollify_intrinsic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = _MOLLIFY_EPSILON,
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
        [`face_edge_lengths`][triwarp.intrinsic.face_edge_lengths]; recomputed here when ``None``.

    Returns
    -------
    lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` mollified edge lengths, column ``e`` opposite corner ``e``.
    delta : float
        The constant added to every length. Reading it costs one host readback, and it is returned
        because it is the honest measure of how much the geometry had to be changed.

    See Also
    --------
    [`robust_laplacian`][triwarp.intrinsic.robust_laplacian]
    [`face_edge_lengths`][triwarp.intrinsic.face_edge_lengths]
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
        kernel_intrinsic.triangle_inequality_slack,
        dim=n_faces,
        inputs=[edge_lengths, wp.float32(epsilon * scale), slack],
        device=device,
    )
    delta = float(reduce_max(slack))
    if delta <= 0.0:
        return twt.as_array2d_float32(edge_lengths), 0.0

    mollified = twt.empty_float32_2d((n_faces, 3), device=device)
    wp.map(kernel_intrinsic.add_constant, edge_lengths, wp.float32(delta), out=mollified)
    return twt.as_array2d_float32(mollified), delta


def face_edge_lengths(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> twt.Array2dFloat32:
    """
    Per-face edge lengths, in the intrinsic column order the cotangent formulas expect.

    Column ``e`` holds the length of the edge *opposite* corner ``e``, matching
    [`cotmatrix_entries_intrinsic`][triwarp.laplacian.cotmatrix_entries_intrinsic] and
    ``igl::cotmatrix_entries``' intrinsic overload. Unlike
    [`edges_unique_length`][triwarp.edges.edges_unique_length] this is a per-*corner* table: an
    interior edge appears twice, which is what lets the entries be perturbed per face.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    twt.Array2dFloat32
        ``(n_faces, 3)`` edge lengths on ``vertices.device``.

    See Also
    --------
    [`mollify_intrinsic`][triwarp.intrinsic.mollify_intrinsic]
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    lengths = twt.empty_float32_2d((n_faces, 3), device=device)
    if n_faces == 0:
        return twt.as_array2d_float32(lengths)
    wp.launch(
        kernel_intrinsic.face_edge_lengths,
        dim=n_faces,
        inputs=[vertices, faces, lengths],
        device=device,
    )
    return twt.as_array2d_float32(lengths)
