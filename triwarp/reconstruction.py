"""Point-cloud surface reconstruction on NVIDIA Warp."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import reconstruction as kernel_reconstruction
from triwarp.kernels import remesh as kernel_remesh


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _lexicographic_triangulation(points: np.ndarray) -> np.ndarray:
    """
    Sequential lexicographic incremental triangulation.

    NumPy port of ``igl::lexicographic_triangulation``.

    ``points`` is ``(n, 2)`` ``float64``. Returns an ``(n_faces, 3)`` ``int32`` array of CCW
    triangles over the input point indices, or an empty ``(0, 3)`` array when the points are
    collinear. The output is not yet Delaunay — the caller flips it to Delaunay on device.
    """
    n = points.shape[0]
    order = np.lexsort((points[:, 1], points[:, 0]))
    p0 = points[order[0]]
    p1 = points[order[1]]
    faces: list[tuple[int, int, int]] = []
    boundary: list[int] = []
    for i in range(2, n):
        curr = points[order[i]]
        ci = int(order[i])
        if len(faces) == 0:
            # Every point so far is collinear; the first off-line point fans the prefix.
            orientation = _orient2d(p0, p1, curr)
            if orientation != 0.0:
                if orientation > 0.0:
                    for j in range(i - 1):
                        faces.append((int(order[j]), int(order[j + 1]), ci))
                else:
                    for j in range(i - 1):
                        faces.append((int(order[j + 1]), int(order[j]), ci))
                boundary = [int(order[j]) for j in range(i + 1)]
                if orientation < 0.0:
                    boundary.reverse()
            continue

        nb = len(boundary)
        orientations = [
            _orient2d(points[boundary[j]], points[boundary[(j + 1) % nb]], curr) for j in range(nb)
        ]
        for j in range(nb):
            if orientations[j] < 0.0:
                faces.append((boundary[(j + 1) % nb], boundary[j], ci))

        # The visible edges form one contiguous arc; L starts it, R ends it (first kept vertex).
        left = right = -1
        for j in range(nb):
            prev = (j - 1) % nb
            if orientations[j] >= 0.0 and orientations[prev] < 0.0:
                right = j
            elif orientations[j] < 0.0 and orientations[prev] >= 0.0:
                left = j
        # Keep the non-visible arc R..L (forward), then insert curr between L and R.
        kept: list[int] = []
        k = right
        while True:
            kept.append(boundary[k])
            if k == left:
                break
            k = (k + 1) % nb
        kept.append(ci)
        boundary = kept

    if len(faces) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def delaunay_triangulation(points: wp.array[wp.vec2], max_iter: int = 1000) -> wp.array[wp.int32]:
    """
    Delaunay triangulation of a 2D point set.

    Ports ``igl::delaunay_triangulation``: a sequential lexicographic incremental triangulation
    seeds the mesh, then interior edges are flipped until every one satisfies the empty-circumcircle
    (Delaunay) criterion. The flip loop reuses the parallel edge-flip machinery of
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] with a float64 in-circle predicate; ties
    (cocircular points) are left unflipped, which guarantees termination.

    Parameters
    ----------
    points
        ``(n,)`` 2D point positions on the target device.
    max_iter
        Maximum number of parallel flip passes.

    Returns
    -------
    wp.array[wp.int32]
        Flat length-``3 * n_faces`` buffer of CCW triangles over the input point indices, on
        ``points.device``. Empty when the points are collinear.

    Raises
    ------
    ValueError
        If fewer than 3 points are given.

    See Also
    --------
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]

    Notes
    -----
    The initial triangulation runs on the host (an inherently sequential hull sweep); only the
    flips run on device. A float64 in-circle determinant is used rather than exact predicates, so
    near-cocircular inputs may resolve either ambiguous diagonal.
    """
    device = points.device
    n = int(points.shape[0])
    if n < 3:
        raise ValueError(f"delaunay_triangulation requires at least 3 points, got {n}")

    faces_np = _lexicographic_triangulation(points.numpy().astype(np.float64))
    if faces_np.shape[0] == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    faces = wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        wp.launch(
            kernel_remesh.incircle_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                points,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                key_base,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    tw.remesh._flip_interior_edges(faces, n, launch, max_iter)
    return faces


def _repeated_oriented_triangles(
    candidates: twt.Array2dInt32, n_points: int, repetitions: int
) -> twt.Array2dInt32:
    """
    Keep one oriented representative per candidate triangle repeated exactly ``repetitions`` times.

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

    return twt.as_array2d_int32(tw.array.gather(candidates, wp.clone(groups[:, 0])))


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
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight]
    [`triwarp.repair.remove_degenerate_faces`][]
    """
    if num_neighbours > 0 and radius > 0.0:
        raise ValueError("pass at most one of num_neighbours and radius")
    if max_neighbours > kernel_reconstruction.MAX_NEIGHBOURS:
        raise ValueError(
            "max_neighbours must be <= "
            f"{kernel_reconstruction.MAX_NEIGHBOURS}, got {max_neighbours}"
        )

    device = points.device
    n = int(points.shape[0])
    if n == 0:
        return points, wp.empty(0, dtype=wp.int32, device=device)

    k = num_neighbours if num_neighbours > 0 else max_neighbours
    k = min(k, max_neighbours)

    # Dense (n, k+1) nearest-neighbour table; slot 0 is the point itself and is skipped in-kernel.
    neighbor_idx, neighbor_dist = tw.neighbors.query_bvh_nearest(points, points, k=k + 1)

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
    points: wp.array[wp.vec3], t3: twt.Array2dInt32, t2: twt.Array2dInt32, crit_hole_length: float
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Combine t3/t2 triangles into a clean mesh: dedup, drop degenerate/non-manifold, fill holes.

    Triangle soup is assembled from the repeated oriented triangles (Stage 4), then degenerate,
    duplicate, and non-manifold faces are removed and small boundary holes are filled (Stage 5).
    Vertices are the referenced input points, compacted from index zero.
    """
    device = points.device
    parts = [f for f in (t3.reshape(-1), t2.reshape(-1)) if int(f.shape[0]) > 0]
    if not parts:
        return wp.clone(points), wp.empty(0, dtype=wp.int32, device=device)
    faces = parts[0] if len(parts) == 1 else tw.array.concatenate(parts)

    # Drop exact duplicate faces, then degenerate faces (which also compacts the vertex set).
    faces, _ = tw.repair.resolve_duplicated_faces(faces)
    vertices, faces = tw.repair.remove_degenerate_faces(points, faces)

    # Remove faces on non-manifold edges so the result is edge-manifold (MeshLib
    # findHoleComplicatingFaces loop). This is required before boundary extraction: hole filling and
    # boundary_loops assume a manifold boundary.
    vertices, faces = tw.repair.remove_non_manifold_faces(vertices, faces)

    # Fill small boundary holes (MeshLib makeMesh_ tail).
    if int(faces.shape[0]) > 0:
        hole_length = crit_hole_length
        if hole_length < 0.0:
            lo, hi = tw.bounds.aabb_bounds(points)
            hole_length = 0.1 * float(wp.length(hi - lo))
        faces = tw.hole_filling.fill_small_holes(vertices, faces, hole_length)

    return vertices, faces
