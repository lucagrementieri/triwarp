import warp as wp

from triwarp.kernels import graph as kernel_graph


def faces_to_edges(faces: wp.array[wp.int32]) -> wp.array2d[wp.int32]:
    """
    Directed triangle edges from a flat ``(i0, i1, i2)`` index buffer.

    For each face, emits the three directed edges ``(i0, i1)``, ``(i1, i2)``, and ``(i2, i0)``
    in row-major order, matching :func:`trimesh.geometry.faces_to_edges` on the same ``faces``
    layout. Runs on ``faces.device`` with one launched thread per face (``int32`` indices).

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` buffer: consecutive triples
        ``(i0, i1, i2), (i0, i1, i2), ...`` of vertex indices, the same convention as
        :mod:`triwarp.triangles`.

    Returns
    -------
    wp.array2d[wp.int32]
        Shape ``(n_faces * 3, 2)`` with rows ``edges[3*f + 0] = (i0, i1)``,
        ``edges[3*f + 1] = (i1, i2)``, ``edges[3*f + 2] = (i2, i0)`` for face ``f``.
        If ``n_faces == 0``, returns an empty ``(0, 2)`` array.

    Raises
    ------
    ValueError
        If ``faces.shape[0]`` is not divisible by ``3``.

    See Also
    --------
    :func:`trimesh.geometry.faces_to_edges`
    """
    n = int(faces.shape[0])
    if n % 3 != 0:
        raise ValueError(f"faces length must be divisible by 3, got {n}")
    n_faces = n // 3
    edges = wp.empty((n_faces * 3, 2), dtype=wp.int32, device=faces.device)
    wp.launch(
        kernel_graph.faces_to_edges,
        dim=n_faces,
        inputs=[faces, edges],
        device=faces.device,
    )
    return edges
