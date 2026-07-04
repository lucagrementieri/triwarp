"""
Basic boundary hole filling (Warp).

Every boundary of a triangle mesh is an ordered vertex loop
([`boundary_loops`][triwarp.boundary.boundary_loops], the analog of
``trimesh.repair.fill_holes``'s ``nx.cycle_basis`` and ``igl::boundary_loop_all``). Each loop
of ``B`` vertices is sealed with a purely topological triangulation — **no smoothing or
refinement**:

- [`fill_holes_fan`][triwarp.stitching.fill_holes_fan] fans ``B - 2`` triangles from the loop's
  first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
- [`fill_holes_cone`][triwarp.stitching.fill_holes_cone] inserts one centroid vertex per hole
  and cones ``B`` triangles onto it (``igl::topological_hole_fill``,
  ``trimesh.repair.stitch(insert_vertices=True)``).
- [`fill_holes_min_weight`][triwarp.stitching.fill_holes_min_weight] instead computes the
  **minimum-weight triangulation** of each loop (the Liepa/Klincsek interval DP ported from
  MeshLib's ``fillHole``): the ``B - 2`` triangles over the existing loop vertices that minimize a
  geometric metric (plane-normalized circumcircle by default, with a min-area fallback), avoiding
  chords that would duplicate existing mesh edges. This is the robust, general-purpose filler for
  non-convex and non-planar holes; it adds no vertices.

Because ``boundary_loops`` orders each loop following the face-winding direction of the
boundary half-edges, a fill triangle sharing a rim edge is emitted **reversed** on that edge so
its winding is consistent with the adjacent original face. This assumes the input mesh is
consistently wound; run [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
first on meshes with mixed winding.

The complementary operation joins **two** open meshes across one boundary loop each:
[`stitch`][triwarp.stitching.stitch] (and the lower-level
[`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]) zippers the two rims with a
band of bridge triangles via a greedy correspondence, producing a single watertight seam.
[`stitch_min_weight`][triwarp.stitching.stitch_min_weight] (and
[`triangulate_boundaries_min_weight`][triwarp.stitching.triangulate_boundaries_min_weight]) instead
choose the band that minimizes a stitch metric (the MeshLib ``stitchHoles`` grid DP). No smoothing
or refinement is applied.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import stitching as kernel_stitching


def _fillable_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> list[wp.array[wp.int32]]:
    """
    Boundary loops (>= 3 vertices) eligible for hole filling, as per-loop vertex-index arrays.

    When ``preserve_largest_hole`` is ``True`` the single largest loop — the one with the greatest
    perimeter arc length ([`closed_polyline_length`][triwarp.polyline.closed_polyline_length]; the
    first one on a tie) — is excluded, leaving it open. This is the standard cut for disk-topology
    repair and UV parametrization, where exactly one boundary must survive.
    """
    loops = [
        loop for loop in tw.boundary.boundary_loops(vertices, faces) if int(loop.shape[0]) >= 3
    ]
    if preserve_largest_hole and len(loops) > 0:
        perimeters = [
            tw.polyline.closed_polyline_length(tw.array.gather(vertices, loop)) for loop in loops
        ]
        largest = max(range(len(loops)), key=lambda i: perimeters[i])
        del loops[largest]
    return loops


def _hole_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int, int] | None:
    """
    Pack fillable boundary loops (>= 3 vertices) for on-device triangulation.

    Returns ``(flat_loops, loop_starts, n_loops, total)`` where ``flat_loops`` concatenates the
    loop vertex indices, ``loop_starts`` is each loop's start offset in ``flat_loops`` (the
    exclusive scan of the loop sizes, on device), ``n_loops`` is the loop count, and ``total`` is
    ``flat_loops.shape[0]``. Returns ``None`` when there is no fillable boundary loop. Both counts
    are read from array shapes (host metadata), so no device buffer is copied to the host.

    See [`_fillable_loops`][triwarp.stitching._fillable_loops] for ``preserve_largest_hole``.
    """
    loops = _fillable_loops(vertices, faces, preserve_largest_hole)
    if len(loops) == 0:
        return None
    flat_loops, loop_starts = tw.array.pack_1d_arrays(loops)
    return flat_loops, loop_starts, len(loops), int(flat_loops.shape[0])


def fill_holes_fan(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> wp.array[wp.int32]:
    """
    Fill every boundary hole with a triangle fan from each loop's first vertex.

    A boundary loop of ``B`` vertices is sealed with ``B - 2`` triangles all sharing the loop's
    first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
    The fill is topological: a fan is only geometrically ideal for convex holes. The vertex
    buffer is unchanged.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (most vertices) open and fill only
        the rest. This turns a mesh with a known disk-like topology into a single-boundary disk,
        which is the input a robust UV parametrization expects.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of the original faces followed by the new fill triangles, on
        ``faces.device``. A watertight or empty mesh — or, with ``preserve_largest_hole``, a mesh
        whose only hole is the largest — is returned unchanged (a copy).

    See Also
    --------
    [`fill_holes_cone`][triwarp.stitching.fill_holes_cone]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return wp.clone(faces)

    flat_loops, loop_starts, n_loops, total = packed
    n_tri = total - 2 * n_loops
    fill_faces = wp.empty(3 * n_tri, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.fan_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), fill_faces],
        device=device,
    )
    return tw.array.concatenate([faces, fill_faces])


def fill_holes_cone(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Fill every boundary hole by coning it onto a new centroid vertex.

    A boundary loop of ``B`` vertices is sealed with ``B`` triangles fanning from one new vertex
    placed at the loop's centroid (``igl::topological_hole_fill``,
    ``trimesh.repair.stitch(insert_vertices=True)``). Unlike
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan] this appends one vertex per hole, so the
    vertex buffer grows.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (most vertices) open and fill only
        the rest. This turns a mesh with a known disk-like topology into a single-boundary disk,
        which is the input a robust UV parametrization expects.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by one centroid per filled hole, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Original faces followed by the new cone triangles, on ``faces.device``. A watertight or
        empty mesh — or, with ``preserve_largest_hole``, a mesh whose only hole is the largest —
        is returned unchanged (copies).

    See Also
    --------
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return wp.clone(vertices), wp.clone(faces)

    flat_loops, loop_starts, n_loops, total = packed
    n_vertices = int(vertices.shape[0])

    centroids = wp.empty(n_loops, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_stitching.loop_centroids,
        dim=n_loops,
        inputs=[vertices, flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), centroids],
        device=device,
    )

    fill_faces = wp.empty(3 * total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.cone_faces,
        dim=n_loops,
        inputs=[
            flat_loops,
            loop_starts,
            wp.int32(total),
            wp.int32(n_loops),
            wp.int32(n_vertices),
            fill_faces,
        ],
        device=device,
    )
    return (
        tw.array.concatenate([vertices, centroids]),
        tw.array.concatenate([faces, fill_faces]),
    )


# Fill-metric name -> kernel selector (must match the METRIC_* constants in kernels/stitching.py).
_METRIC_IDS = {
    "plane_normalized": 0,
    "min_area": 1,
    "circumscribed": 2,
    "plane": 3,
    "min_tri_angle": 4,
    "edge_length": 5,
    "universal": 6,
    "max_dihedral": 7,
    "complex_fill": 8,
}
# Metrics that accumulate with ``max`` instead of ``sum`` (kernel COMBINE_MAX == 1); default sum.
_METRIC_COMBINE = {"max_dihedral": 1}
_BAD_TRIANGULATION_METRIC = 1e10  # kernel ``BAD_METRIC``; a forced-bad triangulation reaches it.


def _edge_third_vertices(faces_np: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected mesh edge to the third vertex of every face using it."""
    edge_third: dict[tuple[int, int], list[int]] = {}
    for tri in faces_np:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v, w in ((a, b, c), (b, c, a), (a, c, b)):
            edge_third.setdefault((u, v) if u < v else (v, u), []).append(w)
    return edge_third


def _rim_opposite(
    loop_np: np.ndarray,
    vertices_np: np.ndarray,
    edge_third: dict[tuple[int, int], list[int]],
    device: wp.DeviceLike,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Per rim edge ``(loop[i], loop[(i+1) % B])``, the opposite vertex of its single adjacent face.

    Used for the ``smoothBd`` boundary term of the dihedral fill metrics: each hole rim edge borders
    exactly one existing triangle, whose third vertex blends the fill's dihedral into the surface.
    Marked invalid (``0``) when the edge has no unique adjacent face.
    """
    b = int(loop_np.shape[0])
    positions = np.zeros((b, 3), dtype=np.float32)
    valid = np.zeros(b, dtype=np.int32)
    for i in range(b):
        u = int(loop_np[i])
        v = int(loop_np[(i + 1) % b])
        thirds = edge_third.get((u, v) if u < v else (v, u), [])
        if len(thirds) == 1:
            positions[i] = vertices_np[thirds[0]]
            valid[i] = 1
    return (
        wp.array(positions, dtype=wp.vec3, device=device),
        wp.array(valid, dtype=wp.int32, device=device),
    )


def _forbidden_chords(
    loop_np: np.ndarray, edges_np: np.ndarray, device: wp.DeviceLike
) -> twt.Array2dInt32:
    """
    ``(B, B)`` mask of interior chords ``(i, j)`` that already exist as a mesh edge.

    Marks only **non-adjacent** loop positions (distance ``2 .. B - 2``); the rim edges ``(i, i+1)``
    and the closing edge ``(0, B-1)`` are the hole boundary itself and stay allowed. This is
    MeshLib's ``MultipleEdgesResolveMode::Simple`` guard against non-manifold fills.
    """
    b = int(loop_np.shape[0])
    position = {int(v): p for p, v in enumerate(loop_np)}
    mask = np.zeros((b, b), dtype=np.int32)
    for u, v in edges_np:
        pu = position.get(int(u))
        pv = position.get(int(v))
        if pu is None or pv is None:
            continue
        lo, hi = (pu, pv) if pu < pv else (pv, pu)
        if 2 <= hi - lo <= b - 2:
            mask[lo, hi] = 1
            mask[hi, lo] = 1
    return twt.as_array2d_int32(wp.array(mask, dtype=wp.int32, device=device))


def _run_hole_dp(
    loop_pos: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    forbidden: twt.Array2dInt32,
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_area: float,
    metric_id: int,
    combine_id: int,
    smooth_boundary: bool,
    device: wp.DeviceLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the interval DP for one hole; return the host ``(dp, prev)`` tables."""
    b = int(loop_pos.shape[0])
    dp = twt.empty_float32_2d((b, b), device=device)
    prev = twt.empty_int32_2d((b, b), device=device)
    wp.launch(
        kernel_stitching.init_dp_base, dim=(b, b), inputs=[dp, prev, wp.int32(b)], device=device
    )
    for span in range(2, b):
        wp.launch(
            kernel_stitching.fill_dp_span,
            dim=b - span,
            inputs=[
                loop_pos,
                plane_normal,
                forbidden,
                rim_opp_pos,
                rim_opp_valid,
                wp.float32(char_area),
                wp.int32(metric_id),
                wp.int32(combine_id),
                wp.int32(1 if smooth_boundary else 0),
                wp.int32(span),
                wp.int32(b),
                dp,
                prev,
            ],
            device=device,
        )
    return dp.numpy(), prev.numpy()


def _traceback_triangles(prev_np: np.ndarray, loop_np: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Extract the ``B - 2`` fill triangles from the DP predecessor table.

    Each interval ``(i, j)`` splits at apex ``k = prev[i, j]`` into triangle ``(i, j, k)`` (reversed
    rim winding, matching ``fan_faces``) and sub-intervals ``(i, k)``, ``(k, j)``. An apex of ``-1``
    (an unfillable interval, e.g. a fully pinched hole) is skipped.
    """
    triangles: list[tuple[int, int, int]] = []
    stack = [(0, int(loop_np.shape[0]) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        k = int(prev_np[i, j])
        if k < 0:
            continue
        triangles.append((int(loop_np[i]), int(loop_np[j]), int(loop_np[k])))
        stack.append((i, k))
        stack.append((k, j))
    return triangles


def fill_holes_min_weight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    resolve_multiple_edges: bool = True,
    preserve_largest_hole: bool = False,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Fill every boundary hole with a minimum-weight triangulation over its existing vertices.

    Ports MeshLib's ``fillHole`` (the classic Liepa/Klincsek interval dynamic program): each
    boundary loop of ``B`` vertices is sealed with the ``B - 2`` triangles that minimize a geometric
    metric, reusing only existing vertices (the vertex buffer is unchanged). This is far more robust
    than [`fill_holes_fan`][triwarp.stitching.fill_holes_fan] for non-convex or non-planar holes
    and, unlike [`fill_holes_cone`][triwarp.stitching.fill_holes_cone], adds no vertices.

    The ``O(B^3)`` DP runs on device as ``B`` parallel kernel launches (one per triangulation span);
    the small ``B x B`` predecessor table is traced back on the host to emit triangles.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    metric
        Which MeshLib fill metric to minimize (ported from ``MRMeshMetrics.cpp``):

        - ``"plane_normalized"`` (default) — circumcircle-diameter times aspect ratio, penalizing
          triangles flipped or tilted more than 60 degrees off the hole plane
          (``getPlaneNormalizedFillMetric``); falls back to ``"min_area"`` when the best
          triangulation is still bad (non-planar/degenerate).
        - ``"min_area"`` — summed triangle area; always yields a filling (``getMinAreaMetric``).
        - ``"circumscribed"`` — summed circumcircle diameter (``getCircumscribedMetric``; MeshLib's
          own ``fillHole`` default).
        - ``"plane"`` — circumcircle diameter with a flipped-normal penalty
          (``getPlaneFillMetric``).
        - ``"min_tri_angle"`` — maximizes the minimal triangle angle (``getMinTriAngleMetric``).
        - ``"edge_length"`` — summed new-edge length (``getEdgeLengthFillMetric``).
        - ``"universal"`` — circumcircle diameter plus a dihedral-smoothing edge term; the smooth,
          general-purpose choice (``getUniversalMetric``).
        - ``"max_dihedral"`` — minimizes the maximal dihedral angle (``getMaxDihedralAngleMetric``).
        - ``"complex_fill"`` — area/aspect triangle term plus a strong dihedral edge term
          (``getComplexFillMetric``).

        The dihedral (edge-based) metrics — ``universal``, ``max_dihedral``, ``complex_fill``,
        ``edge_length`` — blend into the surrounding surface via ``smooth_boundary``.
    resolve_multiple_edges
        When ``True`` (default), forbid the triangulation from creating a chord that duplicates an
        existing mesh edge, avoiding non-manifold results on pinched holes (MeshLib's ``Simple``
        multiple-edges mode).
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest — the single-boundary disk a robust UV parametrization expects.
    smooth_boundary
        When ``True`` (default, MeshLib ``smoothBd``), the dihedral edge metrics also score the hole
        rim edges against the existing adjacent faces, so the patch blends smoothly into the surface
        instead of turning sharply at the boundary. No effect on the triangle-only metrics.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of the original faces followed by the fill triangles, on ``faces.device``.
        A watertight or empty mesh — or, with ``preserve_largest_hole``, a mesh whose only hole is
        the largest — is returned unchanged (a copy).

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported metric names.

    See Also
    --------
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan]
    [`fill_holes_cone`][triwarp.stitching.fill_holes_cone]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    if metric not in _METRIC_IDS:
        raise ValueError(
            f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}"
        )
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(faces)
    loops = _fillable_loops(vertices, faces, preserve_largest_hole)
    return _fill_loops(vertices, faces, loops, metric, resolve_multiple_edges, smooth_boundary)


def _fill_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: list[wp.array[wp.int32]],
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate the given boundary ``loops`` and append the fill faces.

    Shared body of [`fill_holes_min_weight`][triwarp.stitching.fill_holes_min_weight]: each loop is
    sealed by the interval DP under ``metric`` (with a ``min_area`` fallback when the primary metric
    yields a bad triangulation), reusing only existing vertices.
    """
    device = faces.device
    if len(loops) == 0:
        return wp.clone(faces)

    vertices_np = vertices.numpy()
    edges_np = tw.edges.faces_to_edges(faces, sorted=True).numpy()
    edge_third = _edge_third_vertices(faces.numpy().reshape(-1, 3))
    primary_id = _METRIC_IDS[metric]
    combine_id = _METRIC_COMBINE.get(metric, 0)
    min_area_id = _METRIC_IDS["min_area"]

    triangles: list[tuple[int, int, int]] = []
    for loop in loops:
        loop_np = loop.numpy()
        b = int(loop.shape[0])
        loop_pos = tw.array.gather(vertices, loop)
        plane_normal = tw.polyline.closed_polyline_normal(loop_pos)
        forbidden = (
            _forbidden_chords(loop_np, edges_np, device)
            if resolve_multiple_edges
            else twt.as_array2d_int32(wp.zeros((b, b), dtype=wp.int32, device=device))
        )
        rim_opp_pos, rim_opp_valid = _rim_opposite(loop_np, vertices_np, edge_third, device)
        edge_sq = wp.empty(b, dtype=wp.float32, device=device)
        wp.launch(
            kernel_stitching.closed_edge_sq_lengths,
            dim=b,
            inputs=[loop_pos, wp.int32(b), edge_sq],
            device=device,
        )
        max_edge_sq = float(tw.reduce.max(edge_sq))
        char_area = 1.0 / max_edge_sq if max_edge_sq > 0.0 else 1.0

        dp_np, prev_np = _run_hole_dp(
            loop_pos, plane_normal, forbidden, rim_opp_pos, rim_opp_valid,
            char_area, primary_id, combine_id, smooth_boundary, device,
        )
        if primary_id != min_area_id and dp_np[0, b - 1] >= _BAD_TRIANGULATION_METRIC:
            _, prev_np = _run_hole_dp(
                loop_pos, plane_normal, forbidden, rim_opp_pos, rim_opp_valid,
                char_area, min_area_id, 0, smooth_boundary, device,
            )
        triangles.extend(_traceback_triangles(prev_np, loop_np))

    if len(triangles) == 0:
        return wp.clone(faces)
    fill_faces = wp.array(
        np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device
    )
    return tw.array.concatenate([faces, fill_faces])


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][triwarp.stitching._non_increasing_indices]); the algorithm does
    not handle ties. Mirrors promesh's private helper of the same name.
    """
    p = np.zeros_like(numbers, dtype=np.int64)
    m = -np.ones(numbers.size + 1, dtype=np.int64)
    size = 0
    for i, x in enumerate(numbers):
        subseq_size = int(np.searchsorted(numbers[m[1 : size + 1]], x))
        p[i] = m[subseq_size]
        m[subseq_size + 1] = i
        size = max(subseq_size + 1, size)
    subseq = np.empty(size, numbers.dtype)
    k = m[size]
    for i in range(size - 1, -1, -1):
        subseq[i] = numbers[k]
        k = p[k]
    return subseq


def _non_increasing_indices(numbers: np.ndarray) -> np.ndarray:
    """
    Return the entry indices **not** in the longest non-decreasing subsequence of ``numbers``.

    Repeated integers are perturbed by adding ``linspace(0, 1, count, endpoint=False)`` within
    each equal-value group, so a run of equal values is treated as (weakly) increasing and kept.
    Mirrors promesh's private helper (with a NumPy grouping in place of ``trimesh.grouping.group``
    to avoid a runtime ``trimesh`` dependency).
    """
    different = numbers.astype(np.float64)
    order = np.argsort(numbers, kind="stable")
    sorted_values = numbers[order]
    cut = np.flatnonzero(np.diff(sorted_values)) + 1
    for group in np.split(order, cut):
        different[group] += np.linspace(0.0, 1.0, num=len(group), endpoint=False)
    subsequence = _longest_increasing_subsequence(different)
    absence_mask = np.isin(different, subsequence, assume_unique=True, invert=True)
    return np.flatnonzero(absence_mask)


def triangulate_boundaries(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes by triangulating the band between one boundary loop on each.

    The two rims are zippered into a watertight seam by a greedy, order-preserving correspondence
    (the Warp port of promesh's ``triangulate_boundaries``): every A-edge is matched to the B
    vertex minimizing the added-triangle perimeter, the association is rotated so its shortest
    pair comes first, and a longest-increasing-subsequence correction forces the matching to be
    monotone (non-self-intersecting). **No smoothing or refinement** is applied — the seam reuses
    only the two loops' existing vertices, adding ``len(loop_a) + len(loop_b)`` bridge triangles.

    The larger loop is treated as A (the meshes are swapped internally when
    ``len(loop_a) < len(loop_b)``), so the output is invariant to argument order. Both loops must
    wind following the face orientation, as returned by
    [`boundary_loops`][triwarp.boundary.boundary_loops]; the two boundaries are assumed to wind in
    opposite directions (as two open meshes facing each other do), so loop A is reversed to align
    them.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each
        mesh, indexing ``vertices_a`` / ``vertices_b`` respectively.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenation of ``vertices_a`` then ``vertices_b`` (larger loop first), on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the bridge triangles, on
        ``faces_a.device``.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices.

    See Also
    --------
    [`stitch`][triwarp.stitching.stitch]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`concatenate`][triwarp.graph.concatenate]

    Notes
    -----
    The correspondence and its monotonicity correction are inherently sequential and run on the
    host over the length-``len(loop_a)`` association array; the O(N·M) perimeter matrix, the
    reductions, and the triangle emission run in Warp kernels, and the perimeter matrix itself is
    never copied off the device.
    """
    device = faces_a.device
    n = int(loop_a.shape[0])
    m = int(loop_b.shape[0])
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n
    if m < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n} and {m}")
    n_vertices_a = int(vertices_a.shape[0])

    # Reverse loop A so both rims wind the same way, then take rim positions.
    flipped_loop_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.flip_loop,
        dim=n,
        inputs=[loop_a, wp.int32(n), flipped_loop_a],
        device=device,
    )
    a_pos = tw.array.gather(vertices_a, flipped_loop_a)
    b_pos = tw.array.gather(vertices_b, loop_b)

    # perimeters[i, j] = |a_i - b_j| + |a_{i+1} - b_j| for A-edge i and B-vertex j.
    perimeters = twt.empty_float32_2d((n, m), device=device)
    wp.launch(
        kernel_stitching.boundary_perimeters,
        dim=(n, m),
        inputs=[a_pos, b_pos, wp.int32(n), perimeters],
        device=device,
    )

    col_min = wp.empty(n, dtype=wp.int32, device=device)
    val_min = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_stitching.row_argmin,
        dim=n,
        inputs=[perimeters, wp.int32(m), col_min, val_min],
        device=device,
    )

    shift = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    shift_np = shift.numpy()
    shift_a = int(shift_np[0])
    shift_b = int(shift_np[1])

    edge_dev = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_edge_map,
        dim=n,
        inputs=[col_min, wp.int32(shift_a), wp.int32(shift_b), wp.int32(n), wp.int32(m), edge_dev],
        device=device,
    )
    edge = edge_dev.numpy()

    row_roll = shift_a
    col_roll = shift_b

    # Wraparound-group fix: if the lowest-index group is split across the ends of ``edge``, roll
    # A so the group is contiguous (promesh's correction before the monotonicity pass).
    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        if trailing > 0:
            edge = np.roll(edge, trailing)
            row_roll = (shift_a - trailing) % n

    # Monotonicity correction: re-pick the B vertex of every edge outside the longest
    # non-decreasing subsequence within the bracket of its stable neighbours, so the matching is
    # order-preserving. The sentinel value ``m`` closes the last bracket.
    edge_ext = np.append(edge, np.int32(m))
    out_edge = wp.array(edge_ext.astype(np.int32), dtype=wp.int32, device=device)
    if not np.all(np.diff(edge) >= 0):
        unsorted_indices = _non_increasing_indices(edge_ext)
        stable_indices = np.delete(np.arange(edge_ext.size), unsorted_indices)
        next_indices = stable_indices[np.searchsorted(stable_indices, unsorted_indices)]
        wp.launch(
            kernel_stitching.resolve_corrections,
            dim=1,
            inputs=[
                perimeters,
                wp.array(unsorted_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(next_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.int32(int(unsorted_indices.size)),
                wp.int32(row_roll),
                wp.int32(col_roll),
                wp.int32(n),
                wp.int32(m),
                out_edge,
            ],
            device=device,
        )

    # Rolled loops referencing the concatenated vertex buffer (A first, then B offset).
    roll_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_loop_a,
        dim=n,
        inputs=[flipped_loop_a, wp.int32(row_roll), wp.int32(n), roll_a],
        device=device,
    )
    roll_b = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_loop_b,
        dim=m,
        inputs=[loop_b, wp.int32(col_roll), wp.int32(n_vertices_a), wp.int32(m), roll_b],
        device=device,
    )

    bridge_a = wp.empty(3 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.bridge_a_faces,
        dim=n,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), bridge_a],
        device=device,
    )
    bridge_b = wp.empty(3 * m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.bridge_b_faces,
        dim=m,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), wp.int32(m), bridge_b],
        device=device,
    )

    combined_vertices, combined_faces = tw.graph.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, tw.array.concatenate([combined_faces, bridge_a, bridge_b])


def stitch(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes into one watertight mesh.

    Extracts the single boundary loop of each mesh
    ([`boundary_loops`][triwarp.boundary.boundary_loops]) and joins them with
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]. Each mesh must have
    exactly one boundary loop (the promesh assumption); use ``triangulate_boundaries`` directly to
    join specific loops of multi-boundary meshes. **No smoothing** is applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices (larger-boundary mesh first), on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the bridge triangles, on ``faces_a.device``.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices.

    See Also
    --------
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return triangulate_boundaries(vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0])


# Stitch-metric name -> kernel selector (must match the METRIC_*_STITCH constants in
# kernels/stitching.py).
_STITCH_METRIC_IDS = {"complex_stitch": 0, "edge_length_stitch": 1, "vertical": 2}


def _closest_loop_pair(
    a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3]
) -> tuple[int, int]:
    """
    Return the closest vertex pair ``(i, j)`` between the two rims (MeshLib's start pair).

    The full ``(n_a, n_b)`` squared-distance matrix and its argmin run in Warp kernels (the same
    ``row_argmin`` / ``global_argmin`` reduction the greedy zippering uses); only the two winning
    indices come back to the host. Ties resolve to the smallest ``i`` then smallest ``j``, matching
    ``numpy.argmin`` on the flattened matrix.
    """
    device = a_pos.device
    n_a = int(a_pos.shape[0])
    n_b = int(b_pos.shape[0])
    dist_sq = twt.empty_float32_2d((n_a, n_b), device=device)
    wp.launch(
        kernel_stitching.pair_sq_distances,
        dim=(n_a, n_b),
        inputs=[a_pos, b_pos, dist_sq],
        device=device,
    )
    col_min = wp.empty(n_a, dtype=wp.int32, device=device)
    val_min = wp.empty(n_a, dtype=wp.float32, device=device)
    wp.launch(
        kernel_stitching.row_argmin,
        dim=n_a,
        inputs=[dist_sq, wp.int32(n_b), col_min, val_min],
        device=device,
    )
    pair = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n_a), pair],
        device=device,
    )
    pair_np = pair.numpy()
    return int(pair_np[0]), int(pair_np[1])


def triangulate_boundaries_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes with a **minimum-weight** cylindrical band between one boundary loop on each.

    Ports MeshLib's ``stitchHoles``: the two rims are aligned at their closest vertex pair and
    zippered by the band of ``len(loop_a) + len(loop_b)`` triangles that minimizes a stitch metric,
    found by a grid dynamic program over the two loops (``dp[i, j]`` = best band consuming ``i``
    A-edges and ``j`` B-edges; each anti-diagonal is one parallel kernel launch). Reuses only the
    loops' existing vertices. The metric-free greedy
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries] remains available.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each.
    metric
        Stitch metric to minimize (MeshLib ``MRMeshMetrics.cpp``):

        - ``"complex_stitch"`` (default) — triangle aspect ratio plus a dihedral-smoothness edge
          term between adjacent band triangles and the surface (``getComplexStitchMetric``).
        - ``"edge_length_stitch"`` — summed connection-edge length (``getEdgeLengthStitchMetric``).
        - ``"vertical"`` — penalizes band area and normal deviation from ``up_dir``
          (``getVerticalStitchMetric``); pass ``up_dir``.
    up_dir
        Up direction for the ``"vertical"`` metric (defaults to ``(0, 0, 1)``); ignored otherwise.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenation of ``vertices_a`` then ``vertices_b``, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the band triangles.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices, or ``metric`` is unknown.

    See Also
    --------
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]
    [`stitch_min_weight`][triwarp.stitching.stitch_min_weight]
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")
    n_a = int(loop_a.shape[0])
    n_b = int(loop_b.shape[0])
    if n_a < 3 or n_b < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n_a} and {n_b}")

    device = faces_a.device
    metric_id = _STITCH_METRIC_IDS[metric]
    va_np = vertices_a.numpy()
    vb_np = vertices_b.numpy()
    offset = int(vertices_a.shape[0])

    # Reverse loop A so the two rims wind oppositely (facing), then align both at the closest pair.
    # ``la`` / ``lb`` stay on the host for the sequential band traceback, but the closest-pair search
    # gathers rim positions on device and reduces there.
    la = loop_a.numpy()[::-1].copy()
    lb = loop_b.numpy().copy()
    a_rim = tw.array.gather(vertices_a, wp.array(la, dtype=wp.int32, device=device))
    b_rim = tw.array.gather(vertices_b, wp.array(lb, dtype=wp.int32, device=device))
    start_a, start_b = _closest_loop_pair(a_rim, b_rim)
    la = np.roll(la, -start_a)
    lb = np.roll(lb, -start_b)

    a_pos = wp.array(va_np[la], dtype=wp.vec3, device=device)
    b_pos = wp.array(vb_np[lb], dtype=wp.vec3, device=device)
    a_third = _edge_third_vertices(faces_a.numpy().reshape(-1, 3))
    b_third = _edge_third_vertices(faces_b.numpy().reshape(-1, 3))
    a_opp, a_opp_valid = _rim_opposite(la, va_np, a_third, device)
    b_opp, b_opp_valid = _rim_opposite(lb, vb_np, b_third, device)
    up = wp.vec3(*(up_dir if up_dir is not None else (0.0, 0.0, 1.0)))

    dp = twt.as_array2d_float32(
        wp.full((n_a + 1, n_b + 1), _BAD_TRIANGULATION_METRIC, dtype=wp.float32, device=device)
    )
    wp.launch(kernel_stitching.set_dp_origin, dim=1, inputs=[dp], device=device)
    came = twt.as_array2d_int32(wp.full((n_a + 1, n_b + 1), -1, dtype=wp.int32, device=device))
    for diag in range(1, n_a + n_b + 1):
        wp.launch(
            kernel_stitching.stitch_dp_diag,
            dim=min(diag, n_a) - max(0, diag - n_b) + 1,
            inputs=[
                a_pos, b_pos, a_opp, a_opp_valid, b_opp, b_opp_valid, up,
                wp.int32(metric_id), wp.int32(n_a), wp.int32(n_b), wp.int32(diag), dp, came,
            ],
            device=device,
        )
    came_np = came.numpy()

    band = _stitch_band_triangles(came_np, la, lb, n_a, n_b, offset)
    combined_vertices, combined_faces = tw.graph.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    band_faces = wp.array(band.reshape(-1), dtype=wp.int32, device=device)
    return combined_vertices, tw.array.concatenate([combined_faces, band_faces])


def _stitch_band_triangles(
    came_np: np.ndarray, la: np.ndarray, lb: np.ndarray, n_a: int, n_b: int, offset: int
) -> np.ndarray:
    """Trace the grid-DP came-from table from ``(n_a, n_b)`` to ``(0, 0)`` into band triangles."""
    triangles: list[tuple[int, int, int]] = []
    i, j = n_a, n_b
    while i > 0 or j > 0:
        if came_np[i, j] == 0:  # advanced A: triangle (a[i-1], a[i], b[j])
            triangles.append((int(la[(i - 1) % n_a]), int(la[i % n_a]), int(lb[j % n_b]) + offset))
            i -= 1
        elif came_np[i, j] == 1:  # advanced B: triangle (a[i], b[j], b[j-1])
            triangles.append(
                (int(la[i % n_a]), int(lb[j % n_b]) + offset, int(lb[(j - 1) % n_b]) + offset)
            )
            j -= 1
        else:  # unreachable cell (should not happen for valid rims)
            break
    return np.asarray(triangles, dtype=np.int32)


def stitch_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes with a minimum-weight band (MeshLib ``stitchHoles``).

    Like [`stitch`][triwarp.stitching.stitch] but joins the two rims with the metric-minimizing band
    of [`triangulate_boundaries_min_weight`][triwarp.stitching.triangulate_boundaries_min_weight]
    instead of the greedy correspondence. Each mesh must have exactly one boundary loop.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`triangulate_boundaries_min_weight`][triwarp.stitching.triangulate_boundaries_min_weight].
    up_dir
        Up direction for the ``"vertical"`` metric.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the band triangles.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices, or ``metric``
        is unknown.

    See Also
    --------
    [`triangulate_boundaries_min_weight`][triwarp.stitching.triangulate_boundaries_min_weight]
    [`stitch`][triwarp.stitching.stitch]
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch_min_weight requires each mesh to have exactly one boundary loop (>=3 verts); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return triangulate_boundaries_min_weight(
        vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0], metric, up_dir
    )
