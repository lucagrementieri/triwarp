"""
Triangle edges: the directed corner edges, the deduplicated undirected set, and their lengths.

Three representations, and which one a caller wants depends on what an edge means to them:

- **directed corner edges** ([`faces_to_edges`][triwarp.edges.faces_to_edges]) — three per face, in
  corner order, so index ``3 * f + k`` is face ``f``'s edge ``k``. This is the halfedge indexing
  [`triwarp.halfedge`][triwarp.halfedge] builds on.
- **unique undirected edges** ([`edges_unique`][triwarp.edges.edges_unique]) — one per mesh edge,
  with [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse] mapping each corner back to it.
- **per-face length tables** ([`face_edge_lengths`][triwarp.edges.face_edge_lengths]) — an
  ``(n_faces, 3)`` table in the *opposite-corner* column order the cotangent formulas want, which is
  the intrinsic description [`triwarp.laplacian`][triwarp.laplacian] consumes.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import init_repeat_index
from triwarp.kernels import edges as kernel_edges


def faces_to_edges(
    faces: wp.array[wp.int32],
    sorted: bool = False,  # noqa: A002
) -> twt.Array2dInt32:
    """
    Directed triangle edges from a flat ``(i0, i1, i2)`` index buffer.

    For each face emits three directed edges ``(i0, i1)``, ``(i1, i2)``, ``(i2, i0)`` in
    row-major order, matching [`trimesh.geometry.faces_to_edges`][].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer of consecutive vertex-index triples.
    sorted
        If ``True``, each output row has its smaller vertex index first (undirected edges).
        When ``True`` and ``edges`` is already provided the rows are sorted in-place on a copy.

    Returns
    -------
    twt.Array2dInt32
        Shape ``(n_faces * 3, 2)``. Empty ``(0, 2)`` array when ``n_faces == 0``.

    See Also
    --------
    [`trimesh.geometry.faces_to_edges`][]
    """
    n_faces = int(faces.shape[0]) // 3
    edges = twt.empty_int32_2d((n_faces * 3, 2), device=faces.device)
    wp.launch(
        kernel_edges.faces_to_edges,
        dim=n_faces,
        inputs=[faces, wp.bool(sorted), edges],
        device=faces.device,
    )
    return twt.as_array2d_int32(edges)


def edges_face(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Face index for each directed edge produced by [`faces_to_edges`][triwarp.edges.faces_to_edges].

    Returns an array of length ``n_faces * 3`` where entry ``3*f + k`` equals ``f``,
    matching [`trimesh.Trimesh.edges_face`][].

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces * 3`` face indices on ``faces.device``.

    See Also
    --------
    [`trimesh.Trimesh.edges_face`][]
    """
    n_faces = int(faces.shape[0]) // 3
    return init_repeat_index(n_faces * 3, 3, faces.device)


def edges_unique(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]:
    """
    Return unique undirected edges and their inverse mapping into the sorted edge list.

    Equivalent to [`trimesh.Trimesh.edges_unique`][] and
    [`trimesh.Trimesh.edges_unique_inverse`][] computed together.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed ``(n_faces * 3, 2)`` sorted edges (each row min-first).
        When ``None``, built from ``faces``.
    n_vertices
        Total number of vertices (used as the hash base). When ``None``, inferred from
        ``edges_sorted`` with a device-host sync.

    Returns
    -------
    unique_edges : twt.Array2dInt32
        Shape ``(m, 2)`` unique undirected vertex pairs, ``m <= n_faces * 3``.
    inverse : wp.array[wp.int32]
        Length ``n_faces * 3``. ``unique_edges[inverse[i]] == edges_sorted[i]``.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique`][]
    [`trimesh.Trimesh.edges_unique_inverse`][]
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device

    if n_faces == 0:
        empty_edges = twt.empty_int32_2d((0, 2), device=device)
        empty_inv = wp.empty(0, dtype=wp.int32, device=device)
        return empty_edges, empty_inv

    if edges_sorted is None:
        edges_sorted = faces_to_edges(faces, sorted=True)

    n_edges = int(edges_sorted.shape[0])

    if n_vertices is None:
        n_vertices = tw.vertices.n_vertices(edges_sorted)

    keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices)
    unique_keys, inverse = tw.grouping.unique_1d(keys, return_inverse=True)
    n_unique = int(unique_keys.shape[0])

    first_occ = wp.full(n_unique, n_edges, dtype=wp.int32, device=device)
    wp.launch(
        kernel_edges.scatter_first_occurrence,
        dim=n_edges,
        inputs=[inverse, first_occ],
        device=device,
    )

    unique_edges_out = tw.array.gather(edges_sorted, first_occ)

    return twt.as_array2d_int32(unique_edges_out), inverse


def edges_unique_inverse(
    faces: wp.array[wp.int32],
    edges_sorted: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> wp.array[wp.int32]:
    """
    Inverse mapping from sorted edges into [`edges_unique`][triwarp.edges.edges_unique].

    Maps [`faces_to_edges`][triwarp.edges.faces_to_edges] (sorted) into `edges_unique`:
    ``edges_unique(faces)[edges_unique_inverse(faces)] == faces_to_edges(faces, sorted=True)``

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_sorted
        Optional precomputed sorted edges. When ``None``, built from ``faces``.
    n_vertices
        Total vertex count. When ``None``, inferred from ``edges_sorted`` with a device-host sync.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_faces * 3`` inverse indices.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique_inverse`][]
    """
    return edges_unique(faces, edges_sorted=edges_sorted, n_vertices=n_vertices)[1]


def edges_unique_length(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32 | None = None,
    n_vertices: int | None = None,
) -> wp.array[wp.float32]:
    """
    Euclidean length of each unique undirected edge.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    unique_edges
        Optional precomputed unique edges ``(m, 2)``. When ``None``, computed from ``faces``.
    n_vertices
        Total vertex count passed to [`edges_unique`][triwarp.edges.edges_unique]. Ignored when
        ``unique_edges`` is already provided.

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` edge lengths on ``faces.device``.

    See Also
    --------
    [`trimesh.Trimesh.edges_unique_length`][]
    """
    if unique_edges is None:
        unique_edges, _ = edges_unique(faces, n_vertices=n_vertices)

    m = int(unique_edges.shape[0])
    device = faces.device
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(kernel_edges.edge_lengths, dim=m, inputs=[vertices, unique_edges, out], device=device)
    return out


def edges_length(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edges_in: twt.Array2dInt32 | None = None
) -> wp.array[wp.float32]:
    """
    Euclidean length of every directed edge (one per face half-edge).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.
    edges_in
        Optional precomputed directed edges ``(n_faces * 3, 2)``. When ``None``, computed
        from ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n_faces * 3`` edge lengths on ``faces.device``.
    """
    if edges_in is None:
        edges_in = faces_to_edges(faces)

    n = int(edges_in.shape[0])
    device = faces.device
    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(kernel_edges.edge_lengths, dim=n, inputs=[vertices, edges_in, out], device=device)
    return out


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
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    lengths = twt.empty_float32_2d((n_faces, 3), device=device)
    if n_faces == 0:
        return twt.as_array2d_float32(lengths)
    wp.launch(
        kernel_edges.face_edge_lengths,
        dim=n_faces,
        inputs=[vertices, faces, lengths],
        device=device,
    )
    return twt.as_array2d_float32(lengths)


def mean_edge_length(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean length of the ``3 * n_faces`` **per-face** edges (``CurvatureCalculator::getAverageEdge``).

    Every face contributes all three of its edges, so an edge shared by two faces is counted twice
    and a boundary edge once. That weighting is deliberate, not an oversight: it is the average
    libigl's ``CurvatureCalculator::getAverageEdge`` computes, which is what
    ``igl::principal_curvature`` calls to set its sphere-search radius, and what
    [`principal_curvature`][triwarp.curvature.principal_curvature] therefore uses here.

    !!! note "There are two edge averages, and they are not interchangeable"
        [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] counts each edge once,
        matching ``igl::avg_edge_length`` and MeshLab's ``avg_edge_length``. The two agree exactly
        on a closed manifold mesh -- every edge has two incident faces there, so the doubling is
        uniform -- and diverge on anything with a boundary or a non-manifold edge: measured
        **0.452405 against 0.449910** on an open half-torus and **0.293087 against 0.291590** on a
        hemisphere.

        Reach for this one when reproducing libigl's curvature; reach for the unique-edge one when
        reproducing anything else, including the heat method's timestep.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    float
        Mean length over the ``3 * n_faces`` per-face edges. ``0.0`` when ``n_faces == 0``.

    See Also
    --------
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length]
        The same average taken over the unique edges instead.
    [`edges_length`][triwarp.edges.edges_length]
        The per-face lengths this averages.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0.0
    return tw.reduce.mean(edges_length(vertices, faces))


def mean_unique_edge_length(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean length of the **unique** undirected edges (``igl::avg_edge_length``).

    Each edge is counted once, however many faces share it. This is the definition
    ``igl::avg_edge_length`` and MeshLab's ``avg_edge_length`` both use, and the one
    ``igl::heat_geodesics`` picks its diffusion timestep from -- so it is what
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] and
    [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators] use here.

    See [`mean_edge_length`][triwarp.edges.mean_edge_length] for the per-face average, how far the
    two diverge on an open mesh, and why libigl carries both.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    float
        Mean length over the unique undirected edges. ``0.0`` when ``n_faces == 0``.

    See Also
    --------
    [`mean_edge_length`][triwarp.edges.mean_edge_length]
        The same average taken over the per-face edges instead.
    [`edges_unique_length`][triwarp.edges.edges_unique_length]
        The per-edge lengths this averages.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0.0
    return tw.reduce.mean(edges_unique_length(vertices, faces))
