"""Point-cloud surface reconstruction on NVIDIA Warp."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import reconstruction as kernel_reconstruction


def _repeated_oriented_triangles(
    candidates: twt.Array2dInt32, n_points: int, repetitions: int
) -> twt.Array2dInt32:
    """Keep one oriented representative per candidate triangle repeated exactly ``repetitions`` times.

    Candidate triangles are grouped by their sorted (unoriented) vertex key; groups of the
    requested size contribute their first oriented triangle. Mirrors MeshLib's
    ``findRepeatedOrientedTriangles`` for the trusted-normal case.
    """
    device = candidates.device
    n_candidates = int(candidates.shape[0])
    if n_candidates < repetitions:
        return twt.empty_int32_2d((0, 3), device=device)

    sorted_keys = twt.empty_int32_2d((n_candidates, 3), device=device)
    wp.launch(
        kernel_reconstruction.canonicalize_triangles,
        dim=n_candidates,
        inputs=[candidates, sorted_keys],
        device=device,
    )
    groups = tw.grouping.group_int_rows(sorted_keys, repetitions, max_value=n_points)
    n_groups = int(groups.shape[0])
    if n_groups == 0:
        return twt.empty_int32_2d((0, 3), device=device)

    reps = wp.empty(n_groups, dtype=wp.int32, device=device)
    wp.launch(
        kernel_reconstruction.copy_first_column,
        dim=n_groups,
        inputs=[groups, reps],
        device=device,
    )
    return twt.as_array2d_int32(tw.array.gather(candidates, reps))


def triangulate_point_cloud(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None = None,
    *,
    num_neighbours: int = 0,
    radius: float = 0.0,
    crit_angle: float = math.pi / 2.0,
    boundary_angle: float = 0.9 * math.pi,
    max_neighbours: int = 32,
    crit_hole_length: float = -1.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Reconstruct a triangle mesh from an (optionally oriented) point cloud.

    GPU port of MeshLib's ``triangulatePointCloud``
    (``reference/MeshLib/source/MRMesh/MRPointCloudTriangulation.cpp``). Each point independently
    builds a local triangle fan over its nearest neighbours in the tangent plane, greedily
    optimised toward a Delaunay-like fan (``build_local_triangulations``); triangles that recur in
    two or three of these local triangulations are kept and assembled into a triangle-soup mesh
    (vertices are the input points). Non-manifold and degenerate faces are dropped and small
    boundary holes are filled.

    Unlike MeshLib, orientation relies on **trusted normals** (the parallel
    ``findRepeatedOrientedTriangles`` path): supply ``normals`` for arbitrary geometry, or leave
    them ``None`` to estimate them by PCA oriented outward from the cloud centroid (valid for
    star-shaped clouds only — see [`estimate_normals`][triwarp.points.estimate_normals]).

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device.
    normals
        Optional ``(n,)`` oriented unit normals. When ``None``, normals are estimated.
    num_neighbours
        Number of nearest neighbours used per point when ``radius <= 0``. Defaults to
        ``max_neighbours`` when neither ``num_neighbours`` nor ``radius`` is set.
    radius
        When positive, only neighbours within this distance are used (still capped at
        ``max_neighbours`` nearest). Mutually exclusive with ``num_neighbours``.
    crit_angle
        Maximum dihedral angle (radians) tolerated in a fan triangle before an edge is flipped.
    boundary_angle
        A point is a fan boundary when its neighbour ring has an angular gap wider than this.
    max_neighbours
        Per-point neighbour cap. Must not exceed
        ``kernels.reconstruction.MAX_NEIGHBOURS`` (the compile-time scratch size).
    crit_hole_length
        Boundary loops with perimeter at most this value are filled. When negative, defaults to
        ``0.1 *`` the point-cloud bounding-box diagonal (matching MeshLib).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        The referenced input points, compacted from index zero (unreferenced points dropped),
        on ``points.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer into ``vertices``.

    Raises
    ------
    ValueError
        If both ``num_neighbours`` and ``radius`` are positive, or ``max_neighbours`` exceeds the
        compile-time cap.

    Notes
    -----
    The per-point fan is bounded by ``max_neighbours``; MeshLib's automatic radius increase is not
    reproduced, so very sparse or highly non-uniform clouds may leave extra boundary holes.

    See Also
    --------
    [`triwarp.stitching.fill_holes_min_weight`][]
    [`triwarp.repair.remove_degenerate_faces`][]
    """
    if num_neighbours > 0 and radius > 0.0:
        raise ValueError("pass at most one of num_neighbours and radius")
    if max_neighbours > kernel_reconstruction.MAX_NEIGHBOURS:
        raise ValueError(
            f"max_neighbours must be <= {kernel_reconstruction.MAX_NEIGHBOURS}, got {max_neighbours}"
        )

    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return points, wp.empty(0, dtype=wp.int32, device=device)

    k = num_neighbours if num_neighbours > 0 else max_neighbours
    k = min(k, max_neighbours)

    # Dense (n, k+1) nearest-neighbour table; slot 0 is the point itself and is skipped in-kernel.
    neighbor_idx, neighbor_dist = tw.proximity.query_bvh_nearest(points, points, k=k + 1)

    if normals is None:
        normals = tw.points.estimate_normals(points, neighbor_idx)

    # Per-point local fan triangulation.
    out_tris = wp.empty((n, k, 3), dtype=wp.int32, device=device)
    out_valid = wp.zeros((n, k), dtype=wp.bool, device=device)
    wp.launch(
        kernel_reconstruction.build_local_triangulations,
        dim=n,
        inputs=[
            points,
            normals,
            neighbor_idx,
            neighbor_dist,
            wp.float32(radius),
            wp.float32(crit_angle),
            wp.float32(boundary_angle),
            out_tris,
            out_valid,
        ],
        device=device,
    )

    # Compact the emitted candidate triangles.
    kept = tw.array.flatnonzero(out_valid.reshape(-1))
    if int(kept.shape[0]) == 0:
        return points, wp.empty(0, dtype=wp.int32, device=device)
    candidates = twt.as_array2d_int32(tw.array.gather(out_tris.reshape((n * k, 3)), kept))

    # Repeated oriented triangles: t3 (3 reps) preferred, then t2 (2 reps).
    t3 = _repeated_oriented_triangles(candidates, n, 3)
    t2 = _repeated_oriented_triangles(candidates, n, 2)

    return _assemble_faces(points, t3, t2, crit_hole_length)


def _assemble_faces(
    points: wp.array[wp.vec3],
    t3: twt.Array2dInt32,
    t2: twt.Array2dInt32,
    crit_hole_length: float,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Combine t3/t2 triangles into a clean mesh: dedup, drop degenerate/non-manifold, fill holes.

    Triangle soup is assembled from the repeated oriented triangles (Stage 4), then degenerate,
    duplicate, and non-manifold faces are removed and small boundary holes are filled (Stage 5).
    Vertices are the referenced input points, compacted from index zero.
    """
    device = points.device
    parts = [
        f for f in (t3.reshape(-1), t2.reshape(-1)) if int(f.shape[0]) > 0
    ]
    if not parts:
        return wp.clone(points), wp.empty(0, dtype=wp.int32, device=device)
    faces = parts[0] if len(parts) == 1 else tw.array.concatenate(parts)

    # Drop exact duplicate faces, then degenerate faces (which also compacts the vertex set).
    faces, _ = tw.repair.resolve_duplicated_faces(faces)
    vertices, faces = tw.repair.remove_degenerate_faces(points, faces)

    # Remove faces on non-manifold edges so the result is edge-manifold (MeshLib
    # findHoleComplicatingFaces loop). This is required before boundary extraction: hole filling and
    # boundary_loops assume a manifold boundary.
    vertices, faces = _drop_non_manifold_faces(vertices, faces)

    # Fill small boundary holes (MeshLib makeMesh_ tail).
    if int(faces.shape[0]) > 0:
        hole_length = crit_hole_length
        if hole_length < 0.0:
            lo, hi = tw.proximity.aabb_bounds(points)
            hole_length = 0.1 * float(wp.length(hi - lo))
        faces = _fill_small_holes(vertices, faces, hole_length)

    return vertices, faces


def _drop_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Remove faces touching a non-manifold (>2-incident) edge, iterating until edge-manifold.

    Each pass keeps only faces whose three edges are each used by at most two faces
    ([`edge_manifold_mask`][triwarp.characteristics.edge_manifold_mask]); dropping a face can make a
    neighbour manifold, so it repeats up to ``max_iter`` times (matching MeshLib's bounded
    hole-complicating-face removal loop).
    """
    for _ in range(max_iter):
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break
        keep = tw.characteristics.edge_manifold_mask(faces, allow_boundary_edges=True)
        kept = tw.array.flatnonzero(keep)
        if int(kept.shape[0]) == n_faces:
            break  # already edge-manifold
        vertices, faces = tw.selection.submesh_from_face_mask(vertices, faces, keep)
    return vertices, faces


def _fill_small_holes(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_perimeter: float
) -> wp.array[wp.int32]:
    """Fill only boundary loops whose perimeter is at most ``max_perimeter`` (Stage 5).

    Intended open boundaries (large loops) are left untouched; spurious small holes are sealed by
    the shared min-weight interval DP ([`_fill_loops`][triwarp.stitching._fill_loops]).
    """
    loops = tw.boundary.boundary_loops(vertices, faces)
    if not loops:
        return faces

    vertices_np = vertices.numpy()
    small_loops = []
    for loop in loops:
        loop_np = loop.numpy()
        if int(loop_np.shape[0]) < 3:
            continue
        ring = vertices_np[loop_np]
        perimeter = float(
            np.linalg.norm(np.diff(ring, axis=0, append=ring[:1]), axis=1).sum()
        )
        if perimeter <= max_perimeter:
            small_loops.append(loop)

    if not small_loops:
        return faces
    return tw.stitching._fill_loops(
        vertices, faces, small_loops, "plane_normalized", True
    )
