"""
Building a triangle mesh out of something that is not one: a point cloud, or a scalar field.

- **From an oriented point cloud.**
  [`screened_poisson`][triwarp.reconstruction.screened_poisson] fits an implicit indicator function
  and contours it, so it closes gaps and returns a watertight surface that need not pass through
  any input point; [`ball_pivoting`][triwarp.reconstruction.ball_pivoting] instead *interpolates*
  the points, rolling a ball over them, and leaves a hole wherever the ball falls through.
  [`triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud] picks between them.
- **From a scalar field.** [`marching_cubes`][triwarp.reconstruction.marching_cubes] extracts an
  iso-level set of a dense lattice.
- **From a mesh.** [`resample_uniform`][triwarp.reconstruction.resample_uniform] samples an
  existing mesh's signed distance field on a uniform grid and re-extracts it, which is the way to
  get a clean, uniformly-sampled surface out of a self-intersecting or badly-triangulated one.
- **In the plane.** [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation]
  triangulates a 2D point set.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp
import warp.optim.linear as wpl

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_cuda
from triwarp.kernels import reconstruction as kernel_reconstruction
from triwarp.kernels import remesh as kernel_remesh
from triwarp.kernels.algorithms import ball_pivoting as kernel_bpa

if TYPE_CHECKING:
    # Type-checking only: the adaptive-backend helpers import ``warp.fem`` lazily (inside the
    # functions) so ``import triwarp`` never pays its tens-of-seconds first-call codegen unused.
    import warp.fem as fem


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
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
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

    Triangle soup is assembled from the repeated oriented triangles (Stage 4), then handed to the
    shared cleanup tail [`_clean_reconstruction`][triwarp.reconstruction._clean_reconstruction]
    (Stage 5). Vertices are the referenced input points, compacted from index zero.

    Orientation is **not** re-derived here (``orient=False``): the local fans are already wound
    consistently from the trusted per-point normals, and re-deriving it would break that contract.
    Measured on an inward-normal icosphere cloud, ``make_normals_outward`` rewinds the whole mesh
    from its signed volume (volume ``-4.15`` -> ``+4.15``, agreement with the input normals
    ``100%`` -> ``0%``), so a caller passing inward normals would silently get an outward mesh.
    """
    parts = [f for f in (t3.reshape(-1), t2.reshape(-1)) if int(f.shape[0]) > 0]
    if not parts:
        return wp.clone(points), wp.empty(0, dtype=wp.int32, device=points.device)
    faces = parts[0] if len(parts) == 1 else tw.array.concatenate(parts)
    return _clean_reconstruction(points, faces, crit_hole_length, orient=False)


def screened_poisson(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    *,
    depth: int = 8,
    full_depth: int = 5,
    scale: float = 1.1,
    point_weight: float = 4.0,
    solver_iterations: int = 100,
    solver_tolerance: float = 1e-6,
    confidence: bool = False,
    method: Literal["dense", "adaptive"] = "dense",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Screened-Poisson surface reconstruction from an oriented point cloud (Kazhdan PoissonRecon).

    GPU-native reimplementation of the screened-Poisson filter (PyMeshLab's
    ``generate_surface_reconstruction_screened_poisson``, ``open3d``'s
    ``create_from_point_cloud_poisson``).
    A dense node-centered grid over a padded bounding cube is used instead of PoissonRecon's octree:
    the oriented normals are trilinearly splatted into a smoothed vector field ``V`` (with a density
    weight ``W``), the indicator function ``x`` is recovered by solving the screened-Poisson system
    ``(L_N + point_weight * W) x = -div V`` matrix-free with a Jacobi-preconditioned conjugate
    gradient, and the iso-surface is extracted with ``warp.MarchingCubes`` at the iso-value given by
    the average of ``x`` over the input points. A **cascadic** coarse-to-fine schedule solves the
    system from ``full_depth`` up to ``depth``, prolonging each level's solution as the next level's
    initial guess (mirroring PoissonRecon's multigrid hierarchy).

    ``normals`` are **required** and must be globally consistently oriented (all pointing outward or
    all inward): the reconstruction encodes the surface orientation in the sign of the vector field,
    and [`estimate_normals`][triwarp.points.estimate_normals] performs no global orientation. A
    **CUDA device is required** (``warp.optim.linear.cg`` returns NaN on the CPU device in Warp
    1.14-1.15).

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device (``n >= 3``).
    normals
        ``(n,)`` oriented unit normals (their magnitude is used as a confidence weight when
        ``confidence`` is set).
    depth
        Finest grid depth: the finest grid has ``2**depth + 1`` nodes per axis. Memory grows as the
        cube of this; ``depth=8`` (a ``257**3`` grid, ~0.7 GB peak) is a safe default and ``depth``
        is capped at ``10``. This is the single biggest speed lever here, and the *only* one whose
        quality cost is not negligible — see Notes for the measured curve before lowering it.
    full_depth
        Coarsest depth of the cascade (``3 <= full_depth <= depth``). The system is solved at every
        depth from ``full_depth`` to ``depth``.
    scale
        Ratio between the reconstruction cube's side and the point cloud's largest bounding-box
        extent (``> 0``); ``1.1`` pads the cloud by 10 %.
    point_weight
        Screening weight ``alpha`` tying the iso-surface to the input samples. ``0`` recovers the
        unscreened Poisson reconstruction (a tiny epsilon is still added to keep the operator SPD).
    solver_iterations
        Maximum conjugate-gradient iterations per cascade level.
    solver_tolerance
        Relative residual tolerance for the conjugate-gradient solve.
    confidence
        When ``True``, weight each sample's splat by ``|normals[i]|`` (treating the normal magnitude
        as a per-sample confidence), matching PoissonRecon's ``confidence`` flag.
    method
        Solver backend. ``"dense"`` (default) uses the dense node-centered grid described above.
        ``"adaptive"`` uses a ``warp.fem`` adaptive Nanogrid refined only near the samples with a
        variational (finite-element) assembly: fewer degrees of freedom for the same finest
        resolution, so it reaches higher ``depth`` on the same memory budget. The two backends agree
        up to discretization. A point-source weak form rings when the cells are much finer than the
        sampling, so the adaptive backend caps the finest near-surface cell at roughly the mean
        sample spacing (PoissonRecon-style); a ``depth`` above what the sampling supports then only
        refines the extraction lattice. It lazily imports ``warp.fem`` (a one-time codegen cost of
        tens of seconds on first use).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Iso-surface vertices on ``points.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer, oriented outward
        ([`make_normals_outward`][triwarp.repair.make_normals_outward]).

    Raises
    ------
    ValueError
        If ``points`` has fewer than 3 points, or the depth/scale parameters are out of range.
    NotImplementedError
        On a CPU device (the conjugate-gradient solve requires CUDA).

    See Also
    --------
    [`triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]
    [`ball_pivoting`][triwarp.reconstruction.ball_pivoting]
    [`estimate_normals`][triwarp.points.estimate_normals]

    Notes
    -----
    The whole system is solved in index space (grid spacing ``1``); the world scale is reapplied by
    marching cubes via the cube bounds, so the extracted level set is invariant to that scale. The
    per-node density normalization ``V /= max(W, eps)`` is cruder than PoissonRecon's kernel-density
    estimate and can bulge on strongly anisotropic clouds. The screening weight ``point_weight`` is
    not area-normalized as in PoissonRecon, so its numeric scale differs; ``4`` is a reasonable
    default for unit-scale clouds.

    **Choosing ``depth``.** The useful depth follows the cloud's *sampling density*, not its size,
    and past that point extra depth costs time without buying fidelity. Measured on an RTX 5090
    over the bunny vertex clouds (mean symmetric chamfer back to the source mesh, in units of the
    cloud's mean point spacing):

    | depth | dense, 16k points | dense, 69k points |
    |---|---|---|
    | 6 | 14 ms, 0.044 | 14 ms, 0.096 |
    | 7 | 26 ms, 0.042 | 27 ms, 0.043 |
    | 8 (default) | 133 ms, 0.048 | 136 ms, 0.057 |
    | 9 | 968 ms, 0.207 | 966 ms, 0.060 |

    Both clouds bottom out at ``depth=7``, where the reconstruction is **5x faster than the default
    and no less accurate**; ``depth=9`` on the sparser cloud is 5x *worse*, as the grid outruns the
    samples. The default stays at ``8`` because it matches PoissonRecon's own and because mean
    chamfer on two clouds from a single source is too narrow a basis on which to move a fidelity
    default — but for a known cloud density, measure and lower it. The ``adaptive`` backend is much
    flatter in depth (it already caps near-surface refinement at the sample spacing) and is the
    better choice when the depth wanted for the extraction lattice exceeds what the sampling
    supports.
    """
    if not (3 <= full_depth <= depth <= 10):
        raise ValueError(
            "screened_poisson requires 3 <= full_depth <= depth <= 10, got "
            f"full_depth={full_depth}, depth={depth}."
        )
    if scale <= 0.0:
        raise ValueError(f"screened_poisson requires scale > 0, got {scale}.")
    if method not in ("dense", "adaptive"):
        raise ValueError(f"screened_poisson method must be 'dense' or 'adaptive', got {method!r}.")

    device = points.device
    require_cuda(device, "screened_poisson")
    n = int(points.shape[0])
    if n < 3:
        raise ValueError(f"screened_poisson requires at least 3 points, got {n}.")

    cube_lower, cube_upper, cube_size = _poisson_cube(points, scale)
    # Effective screening weight: a floor keeps the operator SPD even at point_weight == 0.
    screen = max(float(point_weight), 1e-4)

    if method == "adaptive":
        vertices, faces = _screened_poisson_adaptive(
            points,
            normals,
            cube_lower,
            cube_upper,
            cube_size,
            depth=depth,
            full_depth=full_depth,
            screen=screen,
            solver_iterations=solver_iterations,
            solver_tolerance=solver_tolerance,
            confidence=confidence,
        )
    else:
        solution, res = _poisson_dense_solve(
            points,
            normals,
            cube_lower,
            cube_size,
            depth,
            full_depth,
            screen,
            confidence,
            solver_iterations,
            solver_tolerance,
            device,
        )
        inv_cell = float(res - 1) / cube_size
        sampled = wp.empty(n, dtype=wp.float32, device=device)
        wp.launch(
            kernel_reconstruction.sample_field_trilinear,
            dim=n,
            inputs=[solution, res, cube_lower, inv_cell, points, sampled],
            device=device,
        )
        iso = _poisson_iso_value(sampled, normals, confidence)
        vertices, faces = _extract_poisson_surface(solution, res, iso, cube_lower, cube_upper)

    if int(faces.shape[0]) > 0:
        faces = tw.repair.make_normals_outward(vertices, faces)
    return vertices, faces


def _poisson_cube(points: wp.array[wp.vec3], scale: float) -> tuple[wp.vec3, wp.vec3, float]:
    """Return the padded cubic reconstruction domain (lower, upper, side) around the cloud AABB."""
    lo, hi = tw.bounds.aabb_bounds(points)
    center = 0.5 * (lo + hi)
    max_extent = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
    if max_extent <= 0.0:
        raise ValueError("screened_poisson requires points with a non-degenerate bounding box.")
    cube_size = scale * max_extent
    half = wp.vec3(0.5 * cube_size, 0.5 * cube_size, 0.5 * cube_size)
    return center - half, center + half, cube_size


def _poisson_iso_value(
    sampled: wp.array[wp.float32], normals: wp.array[wp.vec3], confidence: bool
) -> float:
    """
    Iso-value for the extraction: the mean of the solution at the input samples.

    PoissonRecon's ``GetIsoValue``. With ``confidence`` set, the mean is weighted by
    ``|normals[i]|`` (the same per-sample confidence the splat/quadrature weights use) instead of
    uniform; the two weightings are deliberately kept distinct rather than unified. Shared by both
    backends, whose only difference is how ``sampled`` was produced.
    """
    sampled_np = sampled.numpy()
    if not confidence:
        return float(sampled_np.mean())
    weights_np = np.linalg.norm(normals.numpy(), axis=1)
    if float(weights_np.sum()) <= 0.0:
        return 0.0
    return float(np.average(sampled_np, weights=weights_np))


def _extract_poisson_surface(
    field: wp.array[wp.float32], res: int, iso: float, cube_lower: wp.vec3, cube_upper: wp.vec3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Marching-cubes the ``res**3`` scalar lattice ``field`` at ``iso`` over the cube.

    The shared extraction tail of both backends: the dense grid feeds its solution buffer straight
    in, the adaptive backend first samples its finite-element field onto a dense lattice
    ([`_extract_poisson_surface_fem`][triwarp.reconstruction._extract_poisson_surface_fem]). The
    result is un-oriented; [`screened_poisson`][triwarp.reconstruction.screened_poisson] orients it.
    """
    return wp.MarchingCubes.extract_surface_marching_cubes(
        field.reshape((res, res, res)), wp.float32(iso), cube_lower, cube_upper
    )


def _poisson_dense_solve(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_size: float,
    depth: int,
    full_depth: int,
    screen: float,
    confidence: bool,
    solver_iterations: int,
    solver_tolerance: float,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.float32], int]:
    """Run the cascadic dense solve, returning the finest solution buffer and its resolution."""
    prev_solution: wp.array[wp.float32] | None = None
    prev_res = 0
    for level in range(full_depth, depth + 1):
        res = (1 << level) + 1
        inv_cell = float(res - 1) / cube_size
        n_nodes = res * res * res
        if prev_solution is None:
            initial = wp.zeros(n_nodes, dtype=wp.float32, device=device)
        else:
            initial = wp.empty(n_nodes, dtype=wp.float32, device=device)
            wp.launch(
                kernel_reconstruction.prolong_grid,
                dim=(res, res, res),
                inputs=[prev_solution, prev_res, res, initial],
                device=device,
            )
        prev_solution = _poisson_solve_level(
            points,
            normals,
            cube_lower,
            inv_cell,
            res,
            screen,
            confidence,
            initial,
            solver_iterations,
            solver_tolerance,
            device,
        )
        prev_res = res

    assert prev_solution is not None
    return prev_solution, prev_res


def _poisson_solve_level(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    inv_cell: float,
    res: int,
    screen: float,
    confidence: bool,
    initial: wp.array[wp.float32],
    solver_iterations: int,
    solver_tolerance: float,
    device: wp.DeviceLike,
) -> wp.array[wp.float32]:
    """
    Solve one cascade level ``(L_N + screen * W) x = -div V`` matrix-free from ``initial``.

    Splats the oriented normals into the vector field ``V`` and density weight ``W`` on a
    ``res**3`` node grid, normalizes ``V``, builds the negative-divergence right-hand side, and runs
    a Jacobi-preconditioned conjugate gradient with the screened Laplacian applied as a
    ``warp.optim.linear.LinearOperator``. Returns the ``res**3`` solution buffer (``initial`` is the
    warm start and is returned in place).
    """
    n_nodes = res * res * res
    vx = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    vy = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    vz = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    weights = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    wp.launch(
        kernel_reconstruction.splat_normals,
        dim=int(points.shape[0]),
        inputs=[
            points,
            normals,
            cube_lower,
            wp.float32(inv_cell),
            res,
            wp.int32(1 if confidence else 0),
            vx,
            vy,
            vz,
            weights,
        ],
        device=device,
    )
    wp.launch(
        kernel_reconstruction.normalize_vector_field,
        dim=n_nodes,
        inputs=[weights, vx, vy, vz],
        device=device,
    )
    rhs = wp.empty(n_nodes, dtype=wp.float32, device=device)
    wp.launch(
        kernel_reconstruction.negative_divergence,
        dim=(res, res, res),
        inputs=[vx, vy, vz, res, rhs],
        device=device,
    )
    inv_diag = wp.empty(n_nodes, dtype=wp.float32, device=device)
    wp.launch(
        kernel_reconstruction.screened_inverse_diagonal,
        dim=(res, res, res),
        inputs=[weights, wp.float32(screen), res, inv_diag],
        device=device,
    )

    operator = _screened_operator(weights, screen, res, n_nodes, device)
    preconditioner = _diagonal_operator(inv_diag, n_nodes, device)
    wpl.cg(
        operator, rhs, initial, tol=solver_tolerance, maxiter=solver_iterations, M=preconditioner
    )
    return initial


def _screened_operator(
    weights: wp.array[wp.float32], screen: float, res: int, n_nodes: int, device: wp.DeviceLike
) -> wpl.LinearOperator:
    """Matrix-free screened Laplacian ``A = L_N + screen * diag(W)`` as a ``LinearOperator``."""

    def matvec(x, y, z, alpha, beta):
        wp.launch(
            kernel_reconstruction.screened_laplacian_matvec,
            dim=(res, res, res),
            inputs=[x, y, weights, wp.float32(screen), wp.float32(alpha), wp.float32(beta), res, z],
            device=device,
        )

    return wpl.LinearOperator((n_nodes, n_nodes), wp.float32, wp.get_device(device), matvec)


def _diagonal_operator(
    inv_diag: wp.array[wp.float32], n_nodes: int, device: wp.DeviceLike
) -> wpl.LinearOperator:
    """Jacobi (inverse-diagonal) preconditioner as a ``LinearOperator``."""
    # ``wpl.cg`` applies the preconditioner once per iteration, so the mapped kernel is derived
    # once here and only relaunched inside the loop (CLAUDE.md section 4). ``return_kernel=True``
    # returns before mapping, so passing ``inv_diag`` for the x / y / out slots writes nothing --
    # it only supplies the dtype and length that ``x``, ``y`` and ``z`` will have.
    precond = wp.map(
        kernel_reconstruction.diagonal_precond_axpby,
        inv_diag,
        inv_diag,
        inv_diag,
        wp.float32(0.0),
        wp.float32(0.0),
        out=inv_diag,
        return_kernel=True,
    )

    def matvec(x, y, z, alpha, beta):
        wp.launch(
            precond,
            dim=n_nodes,
            inputs=[x, y, inv_diag, wp.float32(alpha), wp.float32(beta)],
            outputs=[z],
            device=device,
        )

    return wpl.LinearOperator((n_nodes, n_nodes), wp.float32, wp.get_device(device), matvec)


def _screened_poisson_adaptive(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    cube_upper: wp.vec3,
    cube_size: float,
    *,
    depth: int,
    full_depth: int,
    screen: float,
    solver_iterations: int,
    solver_tolerance: float,
    confidence: bool,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Solve the screened-Poisson system on a ``warp.fem`` adaptive Nanogrid and extract its surface.

    Backs the ``method="adaptive"`` path of
    [`screened_poisson`][triwarp.reconstruction.screened_poisson]. ``warp.fem`` and its integrand
    module are imported lazily here (not at module scope) so ``import triwarp`` never pays the
    tens-of-seconds warp.fem first-call codegen unless the adaptive backend is used.

    The solve is variational instead of the dense backend's finite-difference stencil: a dense
    coarse grid at ``2**full_depth`` is refined toward the samples with an ``fem.ImplicitField``
    oracle that never carves voxels (full-cube coverage), then the weak Laplacian
    ``integral(grad u . grad v)``,
    an exact point-measure screening term, and a point-source right-hand side are assembled over the
    cells and a ``PicQuadrature`` at the samples and solved with a diagonal-preconditioned conjugate
    gradient. Everything runs in index space (finest spacing ``1``) so the screening-vs-gradient
    balance matches the dense backend's index-space calibration. The (un-oriented) iso-surface
    ``(vertices, faces)`` is returned; the caller orients it outward.
    """
    import warp.fem as fem

    from triwarp.kernels.algorithms import poisson_fem as kernel_poisson_fem

    device = points.device
    n = int(points.shape[0])

    res_fine = 1 << depth
    scale_to_index = float(res_fine) / cube_size

    # Index-space sample positions, unit normals, and per-sample quadrature measures.
    positions = wp.empty(n, dtype=wp.vec3, device=device)
    wp.map(
        kernel_poisson_fem.world_to_index,
        points,
        cube_lower,
        wp.float32(scale_to_index),
        out=positions,
    )
    unit_normals = wp.empty(n, dtype=wp.vec3, device=device)
    wp.map(wp.normalize, normals, out=unit_normals)
    if confidence:
        measures = wp.empty(n, dtype=wp.float32, device=device)
        wp.map(wp.length, normals, out=measures)
    else:
        measures = wp.full(n, wp.float32(1.0), device=device)

    # Match the finest near-surface cell size to the sample spacing (PoissonRecon-style): a
    # point-source weak form rings if cells are much finer than the sampling, so cap the effective
    # octree depth at ~two cells per mean nearest-neighbour distance (never below full_depth, never
    # above the requested depth). ``depth`` beyond this only refines the extraction lattice, which
    # merely samples the already-smooth field more densely.
    _idx, dist = tw.neighbors.query_bvh_nearest(points, points, k=2)
    nn = dist.numpy()[:, 1]
    finite = nn[np.isfinite(nn) & (nn > 0.0)]
    spacing = float(finite.mean()) if finite.size > 0 else cube_size / float(res_fine)
    grid_depth = int(np.floor(np.log2(max(2.0 * cube_size / spacing, 1.0))))
    grid_depth = max(full_depth, min(depth, grid_depth))

    res_coarse = 1 << full_depth
    level_count = grid_depth - full_depth + 1
    coarse_voxel = float(1 << (depth - full_depth))
    fine_voxel = float(1 << (depth - grid_depth))
    spacing_idx = spacing * scale_to_index
    band_r = max(2.0 * fine_voxel, 1.5 * spacing_idx)
    falloff = max(coarse_voxel, 2.0 * spacing_idx)

    # Coarse dense base grid covering the whole cube in index space, then refine toward the samples.
    ijk = np.stack(np.meshgrid(*(np.arange(res_coarse),) * 3, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    # Translate by half a voxel so the voxel-centered grid spans exactly ``[0, 2**depth]`` per axis.
    # Without it the domain is ``[-coarse_voxel/2, ...]`` and the outer lattice shell falls outside;
    # failed lookups there would leave zeros that marching cubes reads as a spurious surface.
    coarse_grid = wp.Volume.allocate_by_voxels(
        wp.array(ijk.astype(np.int32), dtype=wp.vec3i, device=device),
        voxel_size=coarse_voxel,
        translation=(0.5 * coarse_voxel, 0.5 * coarse_voxel, 0.5 * coarse_voxel),
        device=device,
    )
    hashgrid = tw.neighbors.hashgrid_from_points(positions, band_r + falloff)
    refinement = fem.ImplicitField(
        domain=fem.Cells(fem.Nanogrid(coarse_grid)),
        func=kernel_poisson_fem.refinement_oracle,
        values={
            "grid": hashgrid.id,
            "pts": positions,
            "r": wp.float32(band_r),
            "falloff": wp.float32(falloff),
        },
    )
    geometry = fem.adaptive_nanogrid_from_field(
        coarse_grid, level_count, refinement_field=refinement, grading="face"
    )

    # Weak-form assembly: stiffness + screening = source.
    space = fem.make_polynomial_space(geometry, degree=1, dtype=float)
    domain = fem.Cells(geometry)
    test = fem.make_test(space, domain=domain)
    trial = fem.make_trial(space, domain=domain)
    quadrature = fem.PicQuadrature(domain, positions, measures)

    matrix = fem.integrate(kernel_poisson_fem.diffusion_form, fields={"u": trial, "v": test})
    matrix += fem.integrate(
        kernel_poisson_fem.screening_form,
        quadrature=quadrature,
        fields={"u": trial, "v": test},
        values={"screen": wp.float32(screen)},
    )
    rhs = fem.integrate(
        kernel_poisson_fem.source_form,
        quadrature=quadrature,
        fields={"v": test},
        values={"normals": unit_normals},
        output_dtype=float,
    )

    solution = wp.zeros_like(rhs)
    wpl.cg(
        matrix,
        rhs,
        solution,
        tol=solver_tolerance,
        maxiter=solver_iterations,
        M=wpl.preconditioner(matrix, "diag"),
    )
    field = space.make_field()
    field.dof_values = solution

    sampled = wp.zeros(n, dtype=wp.float32, device=device)
    fem.interpolate(
        kernel_poisson_fem.sample_field,
        at=domain,
        dim=n,
        fields={"u": field},
        values={"positions": positions, "out_values": sampled},
    )
    iso = _poisson_iso_value(sampled, normals, confidence)

    return _extract_poisson_surface_fem(
        field, domain, iso, cube_lower, cube_upper, depth, res_fine, device
    )


def _extract_poisson_surface_fem(
    field: fem.DiscreteField,
    domain: fem.GeometryDomain,
    iso: float,
    cube_lower: wp.vec3,
    cube_upper: wp.vec3,
    depth: int,
    res_fine: int,
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sample the adaptive ``field`` onto a dense lattice and marching-cube the ``iso`` surface.

    The lattice resolution is capped at ``2**min(depth, 9) + 1``: ``warp.MarchingCubes`` needs about
    nine times the field bytes in scratch, so a full-cube call overflows above depth 9. A
    slab-chunked pass would lift that cap, but ``warp.MarchingCubes`` is crack-free only within a
    single grid -- its per-cell face triangulation is not consistent across independent invocations,
    so welding independent slabs leaves non-manifold seams -- and depth 9-10 already oversamples the
    spacing-capped solve, so the simple capped extraction is used.
    """
    import warp.fem as fem

    from triwarp.kernels.algorithms import poisson_fem as kernel_poisson_fem

    res = (1 << min(depth, 9)) + 1
    step_index = float(res_fine) / float(res - 1)
    positions = wp.empty(res * res * res, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_poisson_fem.lattice_positions,
        dim=(res, res, res),
        inputs=[wp.float32(step_index), res, wp.float32(float(res_fine) - 1e-3), positions],
        device=device,
    )
    values = wp.zeros(res * res * res, dtype=wp.float32, device=device)
    fem.interpolate(
        kernel_poisson_fem.sample_field,
        at=domain,
        dim=res * res * res,
        fields={"u": field},
        values={"positions": positions, "out_values": values},
    )
    return _extract_poisson_surface(values, res, iso, cube_lower, cube_upper)


def marching_cubes(
    field: twt.Array3dFloat32, iso: float = 0.0, *, bounds: tuple[wp.vec3, wp.vec3] | None = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extract the ``iso`` level set of a dense scalar lattice as a triangle mesh.

    The extraction tail of every implicit-surface pipeline, exposed on its own so a caller with a
    field of their own — an SDF, an occupancy volume, a simulation state — does not have to route it
    through [`screened_poisson`][triwarp.reconstruction.screened_poisson] to get a surface out. It
    is what [`resample_uniform`][triwarp.reconstruction.resample_uniform] is built from.

    Parameters
    ----------
    field
        ``(nx, ny, nz)`` ``wp.float32`` lattice of scalar values, with ``x`` the slowest axis. The
        surface is extracted where the field crosses ``iso``; the sign convention is the caller's,
        and the winding follows it (with triwarp's outside-positive
        [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] convention the
        normals come out pointing outward).
    iso
        Level to extract. Defaults to ``0``, which is the zero level set of a signed distance field.
    bounds
        ``(lower, upper)`` world-space corners the lattice spans, so ``field[0, 0, 0]`` sits at
        ``lower`` and ``field[nx - 1, ny - 1, nz - 1]`` at ``upper``. When ``None`` the result is in
        *index* space: vertex coordinates are lattice indices.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Level-set vertices on ``field.device``. Empty when the field does not cross ``iso``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer.

    Raises
    ------
    ValueError
        If ``field`` is not a rank-3 ``wp.float32`` array, or any of its dimensions is below 2.

    See Also
    --------
    [`resample_uniform`][triwarp.reconstruction.resample_uniform]
    [`screened_poisson`][triwarp.reconstruction.screened_poisson]
    [`triwarp.proximity.signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    [`triwarp.voxels.to_field`][triwarp.voxels.to_field]
    [`triwarp.voxels.grid_points`][triwarp.voxels.grid_points]

    Notes
    -----
    A thin wrapper over Warp's own ``warp.MarchingCubes``, so the triangulation, its vertex
    deduplication and its handling of the ambiguous cube cases are Warp's rather than triwarp's. The
    consequence worth knowing is that the result is **not guaranteed manifold** at an ambiguous
    cell, and can carry duplicate vertices where two cells agree on a crossing —
    [`resample_uniform`][triwarp.reconstruction.resample_uniform] runs
    [`triwarp.repair`][triwarp.repair] over it for exactly that reason.
    """
    field = twt.as_array3d_float32(field)
    shape = tuple(int(dim) for dim in field.shape)
    if min(shape) < 2:
        raise ValueError(f"field must be at least 2 wide along every axis, got {shape}")

    if bounds is None:
        lower = wp.vec3(0.0, 0.0, 0.0)
        upper = wp.vec3(float(shape[0] - 1), float(shape[1] - 1), float(shape[2] - 1))
    else:
        lower, upper = bounds
    return wp.MarchingCubes.extract_surface_marching_cubes(field, wp.float32(iso), lower, upper)


def resample_uniform(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    voxel_size: float | None = None,
    offset: float = 0.0,
    sign_mode: Literal["parity", "winding"] = "winding",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Rebuild a mesh by sampling its signed distance field on a uniform grid and re-extracting it.

    MeshLab's ``generate_resampled_uniform_mesh``. Three uses, all the same call:

    - **Regularize.** The output triangulation comes from the grid, not from the input, so every
      pathology of the input topology — self-intersections, non-manifold edges, duplicated or
      inverted faces, a soup — is simply not carried over. This is the bluntest repair there is and
      the one that always works.
    - **Offset.** A positive ``offset`` extracts the level set *outside* the surface (a dilation)
      and a negative one inside (an erosion): how shells, clearances and tool paths are built.
    - **Shrink-wrap.** A coarse ``voxel_size`` with ``offset=0`` gives a watertight envelope of a
      complicated input.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    voxel_size
        Grid spacing. Defaults to ``1 %`` of the bounding-box diagonal, which is a ~100-cell grid
        across the mesh. **Cost is cubic in the reciprocal**, so halving it is eight times the field
        evaluation; and no feature thinner than a voxel survives.
    offset
        Level set to extract, in world units. ``0`` (the default) reproduces the surface.
    sign_mode
        How the sign of the distance field is decided, forwarded to
        [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]. The default here is
        ``"winding"`` rather than that function's ``"parity"``: resampling is usually applied to a
        mesh that is *broken*, and ray parity has no principled answer through a hole, while the
        generalized winding number degrades gracefully.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Resampled vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive.

    See Also
    --------
    [`marching_cubes`][triwarp.reconstruction.marching_cubes]
    [`triwarp.proximity.signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    [`triwarp.remesh.cluster_decimate`][triwarp.remesh.cluster_decimate]
    [`triwarp.voxels.voxelize_mesh`][triwarp.voxels.voxelize_mesh]

    Notes
    -----
    The grid is padded by three voxels beyond the bounding box *plus* ``offset``, so a dilated level
    set is never clipped by the lattice boundary and the extracted surface is always closed. That
    padding is why the memory cost is a little above ``(extent / voxel_size) ** 3``.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)

    lower, upper = tw.bounds.aabb_bounds(vertices)
    diagonal = float(wp.length(upper - lower))
    if voxel_size is None:
        voxel_size = 0.01 * diagonal
    if voxel_size <= 0.0:
        raise ValueError(f"resample_uniform requires voxel_size > 0, got {voxel_size}")

    # Three voxels of slack beyond the box and beyond the offset, so a dilated level set closes
    # inside the lattice instead of being cut off by it.
    pad = 3.0 * voxel_size + max(offset, 0.0)
    grid_lower = wp.vec3(lower[0] - pad, lower[1] - pad, lower[2] - pad)
    grid_upper = wp.vec3(upper[0] + pad, upper[1] + pad, upper[2] + pad)
    resolution = wp.vec3i(
        *(
            max(2, math.ceil((grid_upper[axis] - grid_lower[axis]) / voxel_size) + 1)
            for axis in range(3)
        )
    )
    spacing = wp.vec3(
        *((grid_upper[axis] - grid_lower[axis]) / float(resolution[axis] - 1) for axis in range(3))
    )

    n_points = int(resolution[0]) * int(resolution[1]) * int(resolution[2])
    points = wp.empty(n_points, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_reconstruction.lattice_points,
        dim=(int(resolution[0]), int(resolution[1]), int(resolution[2])),
        inputs=[resolution, grid_lower, spacing, points],
        device=device,
    )
    field = tw.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode=sign_mode)
    out_vertices, out_faces = marching_cubes(
        twt.as_array3d_float32(
            field.reshape((int(resolution[0]), int(resolution[1]), int(resolution[2])))
        ),
        iso=offset,
        bounds=(grid_lower, grid_upper),
    )
    # Warp's extractor emits a vertex per crossing per cell, so coincident duplicates are normal
    # rather than exceptional; welding them is what makes the result a closed surface. Hole filling
    # is off (``crit_hole_length=0``): a level set of a signed field is already closed, so any hole
    # here would be a symptom worth surfacing rather than patching over.
    return _clean_reconstruction(out_vertices, out_faces, 0.0)


def ball_pivoting(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None = None,
    *,
    radius: float = 0.0,
    clustering: float = 0.2,
    crease_angle: float = math.pi / 2.0,
    max_waves: int = 0,
    crit_hole_length: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Ball-pivoting surface reconstruction from an oriented point cloud (interpolating).

    GPU-native, wave-parallel port of the ball-pivoting algorithm (PyMeshLab's
    ``generate_surface_reconstruction_ball_pivoting``, ``open3d``'s
    ``create_from_point_cloud_ball_pivoting``, vcglib BPA). A ball of the given ``radius`` is rolled
    over the oriented points; wherever it rests on three points without containing another point it
    forms a triangle, and it then pivots around each boundary edge of the growing front to add the
    next triangle. The output mesh **interpolates** the input points (its vertices are a subset of
    ``points``), unlike the implicit
    [`screened_poisson`][triwarp.reconstruction.screened_poisson].

    The advancing front is a **persistent** device-side structure — an edge hash table plus a
    compacted list of its boundary edges, mutated in place as triangles commit. Each wave pivots
    every live front edge independently through a hash-grid ball query and commits a conflict-free
    independent set via a two-phase vertex claim. Manifoldness is protected by an interior-edge
    guard (an edge already shared by two triangles never gains a third). The result is passed
    through the standard cleanup tail (deduplicate, drop degenerate / non-manifold faces, orient
    outward, optionally fill small holes).

    Parameters
    ----------
    points
        ``(n,)`` point positions on the target device (``n >= 3``).
    normals
        Optional ``(n,)`` oriented unit normals. When ``None``, they are estimated by PCA oriented
        outward from the centroid ([`estimate_normals`][triwarp.points.estimate_normals]; valid for
        star-shaped clouds only).
    radius
        Ball radius. When ``<= 0``, it is auto-guessed as ``1.5 x`` the mean nearest-neighbour
        spacing.
    clustering
        Candidate points closer than ``clustering * radius`` to an edge endpoint are rejected
        (vcglib clustering, as a fraction of ``radius``).
    crease_angle
        Maximum dihedral angle (radians) between a new triangle and the one it pivots from; larger
        folds are rejected. Use ``>= pi`` to disable the crease guard.
    max_waves
        Maximum number of parallel waves, where each wave either pivots the current front or seeds
        orphan triangles. ``0`` picks a generous safety cap from the point count.
    crit_hole_length
        Boundary loops with perimeter at most this value are filled at the end. ``0`` disables hole
        filling; a negative value defaults to ``0.1 x`` the bounding-box diagonal.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        The referenced input points, compacted from index zero, on ``points.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer, oriented outward.

    Raises
    ------
    ValueError
        If fewer than 3 points are given.

    See Also
    --------
    [`screened_poisson`][triwarp.reconstruction.screened_poisson]
    [`triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]
    [`estimate_normals`][triwarp.points.estimate_normals]

    Notes
    -----
    A single radius is used (the classic multi-radius schedule is a documented follow-up). Wave
    order is non-deterministic, so the exact triangulation varies run to run, and the triangle
    budget grows on demand rather than raising.

    The result is **interpolating and edge-manifold**, and on a densely, uniformly sampled closed
    surface it is watertight in practice: a subdivided icosphere reconstructs to exactly ``2 v - 4``
    faces with no boundary edge at all, and ``bunny`` (a real scan, so genuinely open in places)
    comes out at 1.97 faces per referenced vertex with 1.7% boundary edges. Watertightness is not
    *guaranteed* — the wave-parallel front has no per-vertex fan structure to align colliding
    sheets with, so pathological sampling can still leave a seam. Regions sampled too sparsely for
    the ball to rest are left as holes by construction (widen ``radius`` or set
    ``crit_hole_length``); for an implicit surface that is watertight by construction use
    [`screened_poisson`][triwarp.reconstruction.screened_poisson].
    """
    device = points.device
    n = int(points.shape[0])
    if n < 3:
        return wp.clone(points), wp.empty(0, dtype=wp.int32, device=device)

    # Nearest-neighbour table drives both the radius auto-guess and (if needed) normal estimation.
    neighbor_idx, neighbor_dist = tw.neighbors.query_bvh_nearest(points, points, k=7)
    if radius <= 0.0:
        distances = neighbor_dist.numpy()[:, 1:]
        finite = distances[np.isfinite(distances) & (distances > 0.0)]
        spacing = float(finite.mean()) if finite.size > 0 else 1.0
        radius = 1.5 * spacing
    if normals is None:
        normals = tw.points.estimate_normals(points, neighbor_idx)

    # Cell width equal to the ball radius, not to the ``2 * radius`` neighbourhood the pivot
    # searches: the inner empty-ball test is by far the most frequent query, and a cell twice its
    # radius made it enumerate ~8x the points it needed. Measured 12% end-to-end; going finer than
    # this loses more to cell-probe overhead than it saves in point tests.
    grid = tw.neighbors.hashgrid_from_points(points, radius)
    crease_cos = math.cos(crease_angle) if crease_angle < math.pi else -1.0

    state = _BpaState(points, normals, grid, radius, clustering, crease_cos, 4 * n + 16)
    _bpa_run(state, max_waves if max_waves > 0 else 16 * n)

    count = int(state.counters.numpy()[kernel_bpa.CNT_FACE])
    faces = wp.clone(state.all_faces[: count * 3])
    return _clean_reconstruction(points, faces, crit_hole_length)


class _BpaState:
    """
    Every buffer a ball-pivoting run touches, allocated once and mutated in place.

    The wave loop does no allocation and no host synchronisation, which is what makes it cheap
    enough to matter: the previous design rebuilt the advancing front from the whole triangle soup
    on every wave (a sort, a scan and three readbacks), and that host traffic — not the pivot
    search — was two thirds of the runtime.
    """

    def __init__(
        self,
        points: wp.array[wp.vec3],
        normals: wp.array[wp.vec3],
        grid: wp.HashGrid,
        radius: float,
        clustering: float,
        crease_cos: float,
        max_faces: int,
    ) -> None:
        """Allocate the persistent state for a cloud of ``points`` and a triangle budget."""
        self.points = points
        self.normals = normals
        self.grid = grid
        self.radius = wp.float32(radius)
        self.clustering = wp.float32(clustering)
        self.crease_cos = wp.float32(crease_cos)
        self.device = points.device
        self.n = int(points.shape[0])
        self.key_base = wp.uint64(self.n)

        self.counters = wp.zeros(kernel_bpa.BPA_COUNTERS, dtype=wp.int32, device=self.device)
        self.counters[kernel_bpa.CNT_SEEDING : kernel_bpa.CNT_SEEDING + 1].fill_(1)
        self.point_used = wp.zeros(self.n, dtype=wp.bool, device=self.device)
        self.boundary_degree = wp.zeros(self.n, dtype=wp.int32, device=self.device)
        self.owner = wp.empty(self.n, dtype=wp.int32, device=self.device)
        self._allocate_budget(max_faces)

    def _allocate_budget(self, max_faces: int) -> None:
        """(Re)size everything that scales with the triangle budget."""
        self.max_faces = max_faces
        self.all_faces = wp.empty(max_faces * 3, dtype=wp.int32, device=self.device)
        # A triangle soup of ``max_faces`` faces has at most ``3 * max_faces`` distinct edges, and
        # the insert probe does not terminate on a full table, so the hash is sized for a load
        # factor of at most 1/2. The front and the proposal list share the same bound.
        capacity = 1 << (6 * max_faces - 1).bit_length()
        self.edge_mask = wp.int32(capacity - 1)
        self.edge_key = wp.zeros(capacity, dtype=wp.uint64, device=self.device)
        self.edge_count = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.edge_src = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_tgt = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_opp = wp.empty(capacity, dtype=wp.int32, device=self.device)
        self.edge_state = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.edge_cand = wp.empty(capacity, dtype=wp.int32, device=self.device)

        front_capacity = 3 * max_faces + self.n
        self.front_in = wp.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.front_out = wp.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_a = wp.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_b = wp.empty(front_capacity, dtype=wp.int32, device=self.device)
        self.tri_c = wp.empty(front_capacity, dtype=wp.int32, device=self.device)
        # Grid-stride bounds: fixed launch dimensions, which a captured graph requires. Sized from
        # the cloud, not from the budget — the live front peaks well below the point count, and a
        # fixed 64k-wide launch spent most of a small mesh's wave scheduling no-op threads.
        self.front_grid = min(front_capacity, max(_BPA_MIN_GRID, self.n))
        self.claim_grid = self.front_grid

    def grow(self) -> None:
        """
        Double the triangle budget in place, preserving the run in progress.

        Slot indices move when the edge table is rehashed, so the front list is rebuilt from the
        new table rather than remapped.
        """
        old = (
            self.edge_key,
            self.edge_count,
            self.edge_src,
            self.edge_tgt,
            self.edge_opp,
            self.edge_state,
            self.edge_cand,
        )
        old_capacity = int(self.edge_key.shape[0])
        old_faces = self.all_faces
        old_count = min(int(self.counters.numpy()[kernel_bpa.CNT_FACE]), self.max_faces)

        self._allocate_budget(2 * self.max_faces)
        wp.copy(self.all_faces, old_faces, count=old_count * 3)
        wp.launch(
            kernel_bpa.rehash_edges,
            dim=old_capacity,
            inputs=[
                *old,
                self.edge_mask,
                self.edge_key,
                self.edge_count,
                self.edge_src,
                self.edge_tgt,
                self.edge_opp,
                self.edge_state,
                self.edge_cand,
            ],
            device=self.device,
        )
        # The over-increment from the losing threads of the overflowing wave never wrote a face.
        counters = self.counters.numpy()
        counters[kernel_bpa.CNT_FACE] = old_count
        counters[kernel_bpa.CNT_NEXT_FRONT] = 0
        counters[kernel_bpa.CNT_GROW] = 0
        self.counters.assign(counters)
        wp.launch(
            kernel_bpa.collect_front_from_table,
            dim=int(self.edge_key.shape[0]),
            inputs=[self.edge_key, self.edge_count, self.edge_state, self.counters],
            outputs=[self.front_in],
            device=self.device,
        )
        self._adopt_next_front()

    def compact(self) -> None:
        """Drop closed and retired edges from the front list."""
        self.counters[kernel_bpa.CNT_NEXT_FRONT : kernel_bpa.CNT_NEXT_FRONT + 1].zero_()
        wp.launch(
            kernel_bpa.compact_front,
            dim=self.front_grid,
            inputs=[
                self.front_in,
                self.edge_count,
                self.edge_state,
                self.front_grid,
                self.counters,
                self.front_out,
            ],
            device=self.device,
        )
        self.front_in, self.front_out = self.front_out, self.front_in
        self._adopt_next_front()

    def _adopt_next_front(self) -> None:
        wp.copy(
            self.counters[kernel_bpa.CNT_FRONT : kernel_bpa.CNT_FRONT + 1],
            self.counters[kernel_bpa.CNT_NEXT_FRONT : kernel_bpa.CNT_NEXT_FRONT + 1],
        )


# Floor on the launch width of the grid-strided wave kernels, for clouds too small to fill the
# device on their own.
_BPA_MIN_GRID = 1 << 12


def _bpa_wave(state: _BpaState, max_waves: int) -> None:
    """Queue one wave: seed or pivot, claim, commit, advance. No allocations, no readbacks."""
    device = state.device
    wp.launch(kernel_bpa.begin_wave, dim=1, inputs=[state.counters], device=device)
    wp.launch(
        kernel_bpa.seed_triangles,
        dim=state.n,
        inputs=[
            state.points,
            state.normals,
            state.point_used,
            state.grid.id,
            state.radius,
            state.clustering,
            state.counters,
            state.owner,
            state.tri_a,
            state.tri_b,
            state.tri_c,
        ],
        device=device,
    )
    wp.launch(
        kernel_bpa.pivot_front_edges,
        dim=state.front_grid,
        inputs=[
            state.points,
            state.normals,
            state.grid.id,
            state.radius,
            state.clustering,
            state.crease_cos,
            state.key_base,
            state.edge_key,
            state.edge_count,
            state.edge_src,
            state.edge_tgt,
            state.edge_opp,
            state.edge_state,
            state.edge_cand,
            state.edge_mask,
            state.point_used,
            state.boundary_degree,
            state.front_in,
            state.front_grid,
            state.counters,
            state.owner,
            state.front_out,
            state.tri_a,
            state.tri_b,
            state.tri_c,
        ],
        device=device,
    )
    wp.launch(
        kernel_bpa.claim_triangle_vertices,
        dim=state.claim_grid,
        inputs=[
            state.tri_a,
            state.tri_b,
            state.tri_c,
            state.claim_grid,
            state.counters,
            state.owner,
        ],
        device=device,
    )
    wp.launch(
        kernel_bpa.commit_triangles,
        dim=state.claim_grid,
        inputs=[
            state.tri_a,
            state.tri_b,
            state.tri_c,
            state.owner,
            wp.int32(state.max_faces),
            state.key_base,
            state.edge_mask,
            state.claim_grid,
            state.edge_key,
            state.edge_count,
            state.edge_src,
            state.edge_tgt,
            state.edge_opp,
            state.edge_cand,
            state.point_used,
            state.boundary_degree,
            state.counters,
            state.front_out,
            state.all_faces,
        ],
        device=device,
    )
    wp.launch(
        kernel_bpa.end_wave, dim=1, inputs=[wp.int32(max_waves), state.counters], device=device
    )


# Waves queued between host synchronisations. The wave loop is device-driven — ``end_wave`` keeps
# the seeding flag, the progress test and the continue flag in ``counters`` — so the host only ever
# needs to look in order to *stop*, and it can queue a batch and let the device run ahead. A wave
# that runs after the flag clears costs six no-op launches, which is far less than a sync.
#
# ``wp.capture_while`` would remove even that, and it was tried: on this workload the conditional
# graph's per-iteration overhead (~0.25 ms/wave on ``bunny_decimated``) is larger than the sync it
# replaces, because a batch already amortises the sync over eight waves.
_BPA_WAVES_PER_BATCH = 8

# Host round-trips beyond the batching are only taken to grow the budget (doubling, so
# logarithmically many) or to compact a front that ``pivot_front_edges`` left sparse (which needs
# the front to have grown 4x against its live count, so also rare).
_BPA_MAX_BATCHES = 1 << 16


def _bpa_run(state: _BpaState, max_waves: int) -> None:
    """
    Drive the wave loop until the front and the orphan set are both exhausted.

    The host is woken once per batch, and only acts when the device asks it to — to grow the
    triangle budget, to compact a sparse front, or to stop.
    """
    state.counters[kernel_bpa.CNT_CONTINUE : kernel_bpa.CNT_CONTINUE + 1].fill_(1)
    for _ in range(_BPA_MAX_BATCHES):
        for _ in range(_BPA_WAVES_PER_BATCH):
            _bpa_wave(state, max_waves)
            state.front_in, state.front_out = state.front_out, state.front_in
        counters = state.counters.numpy()
        if counters[kernel_bpa.CNT_CONTINUE]:
            continue
        if counters[kernel_bpa.CNT_DONE] or counters[kernel_bpa.CNT_WAVE] >= max_waves:
            return
        if counters[kernel_bpa.CNT_GROW]:
            state.grow()
        else:
            state.compact()
        state.counters[kernel_bpa.CNT_CONTINUE : kernel_bpa.CNT_CONTINUE + 1].fill_(1)


def _clean_reconstruction(
    points: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    crit_hole_length: float,
    *,
    orient: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Shared reconstruction cleanup: dedup, drop degenerate/non-manifold, orient, fill small holes.

    Duplicate faces go first, then degenerate ones (which also compacts the vertex set), then faces
    on non-manifold edges so the result is edge-manifold (MeshLib's ``findHoleComplicatingFaces``
    loop) — required before boundary extraction, since both hole filling and ``boundary_loops``
    assume a manifold boundary. ``crit_hole_length`` follows the public convention: ``0`` skips hole
    filling, a negative value means ``0.1 x`` the point-cloud bounding-box diagonal (MeshLib
    ``makeMesh_`` tail). ``orient=False`` keeps the incoming winding for callers that already have a
    trusted orientation.
    """
    if int(faces.shape[0]) == 0:
        return wp.clone(points), faces

    faces, _ = tw.repair.resolve_duplicated_faces(faces)
    vertices, faces = tw.repair.remove_degenerate_faces(points, faces)
    vertices, faces = tw.repair.remove_non_manifold_faces(vertices, faces)

    if int(faces.shape[0]) > 0:
        if orient:
            faces = tw.repair.make_normals_outward(vertices, faces)
        if crit_hole_length != 0.0:
            hole_length = crit_hole_length
            if hole_length < 0.0:
                lo, hi = tw.bounds.aabb_bounds(points)
                hole_length = 0.1 * float(wp.length(hi - lo))
            faces = tw.holes.fill_small(vertices, faces, hole_length)
    return vertices, faces
