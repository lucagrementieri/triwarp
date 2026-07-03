from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp.typing as twt
from triwarp.kernels import laplacian as kernel_laplacian


def cotmatrix_entries(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> twt.Array2dFloat32:
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
    Per-triangle half-cotangent weights from edge lengths (``igl::cotmatrix_entries`` intrinsic overload).

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
