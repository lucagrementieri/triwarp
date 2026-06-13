from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import convex as kernel_convex


def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float32]:
    """
    Project each adjacent face pair's non-shared vertex onto the first face plane.

    For each row of ``face_adjacency``, the dot product is taken between the
    normal of face ``face_adjacency[k, 0]`` and the vector from one endpoint of
    the shared edge to the unshared vertex on ``face_adjacency[k, 1]``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        :func:`triwarp.graph.face_adjacency`).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        :func:`triwarp.graph.face_adjacency`. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        :func:`triwarp.graph.face_adjacency` with ``return_edges=True``).
        Must be supplied together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair from
        :func:`triwarp.graph.face_adjacency_unshared`. When ``None``, computed
        from ``faces`` and the adjacency data.
    face_normals
        Optional length-``n_faces`` unit face normals. When ``None``, normals
        are computed from ``vertices`` and ``faces`` via
        :func:`triwarp.triangles.face_normals_and_areas`.

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` projections on ``faces.device``, one per ``face_adjacency``
        row. Empty when there are no faces or no adjacency pairs.

    Raises
    ------
    ValueError
        If ``vertices`` and ``faces`` live on different devices, or if only one
        of ``face_adjacency`` and ``face_adjacency_edges`` is provided.

    See Also
    --------
    :func:`face_adjacency_convex`
    :attr:`trimesh.Trimesh.face_adjacency_projections`
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None
    assert face_adjacency_edges is not None

    if face_adjacency_unshared is None:
        face_adjacency_unshared = tw.graph.face_adjacency_unshared(
            faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
        )
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_projections = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_convex.face_adjacency_projections,
        dim=m,
        inputs=[
            vertices,
            face_normals,
            face_adjacency,
            face_adjacency_edges,
            face_adjacency_unshared,
            out_projections,
        ],
        device=device,
    )
    return out_projections


def face_adjacency_convex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.bool]:
    """
    Return face pairs that are adjacent and locally convex.

    A pair is locally convex when the unshared vertex of the second face,
    projected onto the plane of the first face, has a projection less than
    :data:`triwarp.constants.TOLERANCE_MERGE`.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        :func:`triwarp.graph.face_adjacency`).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        :func:`triwarp.graph.face_adjacency`. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs. Must be supplied
        together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair.
    face_normals
        Optional length-``n_faces`` unit face normals.

    Returns
    -------
    wp.array[wp.bool]
        Length ``m`` boolean mask on ``faces.device``, one per
        ``face_adjacency`` row. Empty when there are no faces or no adjacency
        pairs.

    See Also
    --------
    :func:`face_adjacency_projections`
    :attr:`trimesh.Trimesh.face_adjacency_convex`
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    projections = face_adjacency_projections(
        vertices,
        faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
        face_adjacency_unshared=face_adjacency_unshared,
        face_normals=face_normals,
    )
    out_convex = wp.empty(m, dtype=wp.bool, device=device)
    wp.launch(
        kernel_convex.face_adjacency_convex, dim=m, inputs=[projections, out_convex], device=device
    )
    return out_convex
