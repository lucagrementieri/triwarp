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


def mean_edge_length(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Mean length of all per-face triangle edges (libigl ``getAverageEdge``).

    Averages the three edges of every face (``3 * n_faces`` directed edges from
    [`faces_to_edges`][triwarp.edges.faces_to_edges]), matching the per-face edge mean used to
    scale the sphere-search radius in
    [`principal_curvature`][triwarp.curvature.principal_curvature].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` face index buffer.

    Returns
    -------
    float
        Mean edge length over all ``3 * n_faces`` per-face edges. ``0.0`` when
        ``n_faces == 0``.
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0.0
    lengths = edges_length(vertices, faces)
    return tw.reduce.mean(lengths)
