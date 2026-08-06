"""
Closing mesh boundary holes.

Every boundary of a triangle mesh is an ordered vertex loop
([`boundary_loops`][triwarp.boundary.boundary_loops], the analog of
``trimesh.repair.fill_holes``'s ``nx.cycle_basis`` and ``igl::boundary_loop_all``). Each loop
of ``B`` vertices is sealed with a purely topological triangulation — **no smoothing or
refinement**:

- [`fill_fan`][triwarp.holes.fill_fan] fans ``B - 2`` triangles from the loop's
  first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
- [`fill_cone`][triwarp.holes.fill_cone] inserts one centroid vertex per hole
  and cones ``B`` triangles onto it (``igl::topological_hole_fill``,
  ``trimesh.repair.stitch(insert_vertices=True)``).
- [`fill_min_weight`][triwarp.holes.fill_min_weight] instead computes the
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

Joining **two** open meshes across one boundary loop each is a different operation and lives in
[`triwarp.combine`][triwarp.combine] -- [`stitch`][triwarp.combine.stitch] and its variants, over
the loop-level engines [`stitch_loops`][triwarp.combine.stitch_loops] and
[`stitch_loops_min_weight`][triwarp.combine.stitch_loops_min_weight]. What it shares with this
module is the minimum-weight machinery: the same interval DP, the same rim bookkeeping and the same
metric vocabulary, applied to a band between two rims rather than a cap over one.

For a smooth, well-graded patch, [`fill_smooth`][triwarp.holes.fill_smooth] and
[`combine.stitch_smooth`][triwarp.combine.stitch_smooth] run the full MeshLib ``fillHoleNicely`` /
``stitchHolesNicely`` pipeline on top of the min-weight fill/stitch: the patch is refined to a
target edge length with Delaunay edge flips
([`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]) and its new interior
vertices are smoothed into the surrounding surface — a sharp-boundary umbrella solve
([`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim])
followed by a cross-boundary least-squares solve
([`smooth_region`][triwarp.smoothing.smooth_region]), with an optional
``natural_smooth`` collar that blends the patch into the neighbouring surface.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import holes as kernel_holes


class _PackedLoops:
    """
    Every fillable loop of one mesh in one packed buffer, plus the host metadata to index it.

    ``flat_loops`` concatenates the loop vertex indices; loop ``ell`` occupies
    ``flat_loops[starts[ell] : starts[ell] + sizes[ell]]``. ``loop_id`` inverts that mapping so a
    ``dim=total`` kernel can find its own loop without a search, and ``dp_offsets`` is the exclusive
    scan of ``sizes ** 2`` — where each loop's ``B x B`` dynamic-programming block begins in the
    ragged tables. The four device arrays are uploaded once and shared by every stage of the fill.
    """

    def __init__(self, flat_loops: wp.array[wp.int32], sizes_np: np.ndarray) -> None:
        device = flat_loops.device
        self.device = device
        self.flat_loops = flat_loops
        self.sizes_np = sizes_np.astype(np.int64)
        self.starts_np = np.concatenate([[0], np.cumsum(self.sizes_np)[:-1]]).astype(np.int64)
        self.dp_offsets_np = np.concatenate(
            [[0], np.cumsum(self.sizes_np * self.sizes_np)[:-1]]
        ).astype(np.int64)

        self.n_loops = int(sizes_np.shape[0])
        self.total = int(self.sizes_np.sum())
        self.max_size = int(self.sizes_np.max())
        self.dp_total = int((self.sizes_np * self.sizes_np).sum())

        self.starts = wp.array(self.starts_np.astype(np.int32), dtype=wp.int32, device=device)
        self.sizes = wp.array(self.sizes_np.astype(np.int32), dtype=wp.int32, device=device)
        self.dp_offsets = wp.array(
            self.dp_offsets_np.astype(np.int32), dtype=wp.int32, device=device
        )
        self.loop_id = wp.array(
            np.repeat(np.arange(self.n_loops, dtype=np.int32), self.sizes_np),
            dtype=wp.int32,
            device=device,
        )

    def loop_slice(self, index: int) -> slice:
        """Host slice of ``flat_loops`` (and of any other length-``total`` buffer) for a loop."""
        start = int(self.starts_np[index])
        return slice(start, start + int(self.sizes_np[index]))

    def dp_block(self, table_np: np.ndarray, index: int) -> np.ndarray:
        """``(B, B)`` view of one loop's block inside a host copy of a ragged DP table."""
        start = int(self.dp_offsets_np[index])
        size = int(self.sizes_np[index])
        return table_np[start : start + size * size].reshape(size, size)


def _hole_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
    edges_sorted: twt.Array2dInt32 | None = None,
) -> _PackedLoops | None:
    """
    Pack the fillable boundary loops (>= 3 vertices) of a mesh for on-device triangulation.

    Returns ``None`` when there is no fillable boundary loop. Costs **one** host readback (the loop
    offsets, which the ragged indexing needs anyway), plus a second one only under
    ``preserve_largest_hole``.

    When ``preserve_largest_hole`` is ``True`` the single largest loop — the one with the greatest
    perimeter arc length; the first one on a tie — is excluded, leaving it open. This is the
    standard cut for disk-topology repair and UV parametrization, where exactly one boundary must
    survive. The perimeters come from one segmented-sum launch over the packed loops rather than a
    [`polyline_length`][triwarp.polyline.polyline_length] call per loop (two synchronizations
    each).
    """
    flat_loops, offsets, _sizes = tw.boundary.boundary_loops_batched(vertices, faces, edges_sorted)
    if int(offsets.shape[0]) == 0:
        return None

    starts_np = offsets.numpy().astype(np.int64)
    sizes_np = np.diff(np.append(starts_np, int(flat_loops.shape[0])))
    keep_np = sizes_np >= 3
    if preserve_largest_hole and bool(keep_np.any()):
        all_loops = _PackedLoops(flat_loops, sizes_np)
        perimeter_np = _loop_perimeters(vertices, all_loops)
        # Only a fillable loop can be the one preserved, matching the pre-filter order.
        candidates_np = np.flatnonzero(keep_np)
        keep_np[candidates_np[int(np.argmax(perimeter_np[candidates_np]))]] = False
    if not keep_np.any():
        return None
    if keep_np.all():
        return _PackedLoops(flat_loops, sizes_np)

    # Compaction is one gather over the dropped loops' slots, not one copy per surviving loop.
    keep_index_np = np.concatenate(
        [
            np.arange(start, start + size)
            for start, size in zip(starts_np[keep_np], sizes_np[keep_np], strict=True)
        ]
    )
    keep_index_wp = wp.array(
        keep_index_np.astype(np.int32), dtype=wp.int32, device=flat_loops.device
    )
    return _PackedLoops(tw.array.gather(flat_loops, keep_index_wp), sizes_np[keep_np])


def _loop_perimeters(vertices: wp.array[wp.vec3], loops: _PackedLoops) -> np.ndarray:
    """Measure the closed arc length of every packed loop (one launch, one readback)."""
    perimeter = wp.zeros(loops.n_loops, dtype=wp.float32, device=loops.device)
    wp.launch(
        kernel_holes.loop_perimeters,
        dim=loops.total,
        inputs=[loops.flat_loops, loops.loop_id, loops.starts, loops.sizes, vertices, perimeter],
        device=loops.device,
    )
    return perimeter.numpy()


def _unpack_loops(loops: _PackedLoops) -> list[wp.array[wp.int32]]:
    """Per-loop **views** into the packed buffer, for the callers that still want a list."""
    return [loops.flat_loops[loops.loop_slice(index)] for index in range(loops.n_loops)]


def fill_fan(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
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
    [`fill_cone`][triwarp.holes.fill_cone]
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

    flat_loops, loop_starts = packed.flat_loops, packed.starts
    n_loops, total = packed.n_loops, packed.total
    n_tri = total - 2 * n_loops
    fill_faces = wp.empty(3 * n_tri, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.fan_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), fill_faces],
        device=device,
    )
    return tw.array.concatenate([faces, fill_faces])


def fill_cone(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Fill every boundary hole by coning it onto a new centroid vertex.

    A boundary loop of ``B`` vertices is sealed with ``B`` triangles fanning from one new vertex
    placed at the loop's centroid (``igl::topological_hole_fill``,
    ``trimesh.repair.stitch(insert_vertices=True)``). Unlike
    [`fill_fan`][triwarp.holes.fill_fan] this appends one vertex per hole, so
    the vertex buffer grows.

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
    [`fill_fan`][triwarp.holes.fill_fan]
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

    flat_loops, loop_starts = packed.flat_loops, packed.starts
    n_loops, total = packed.n_loops, packed.total
    n_vertices = int(vertices.shape[0])

    centroids = wp.empty(n_loops, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_holes.loop_centroids,
        dim=n_loops,
        inputs=[vertices, flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), centroids],
        device=device,
    )

    fill_faces = wp.empty(3 * total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.cone_faces,
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
    return (tw.array.concatenate([vertices, centroids]), tw.array.concatenate([faces, fill_faces]))


# Fill-metric name -> kernel selector (must match the METRIC_* constants in
# kernels/holes.py).
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


class _EdgeTable:
    """
    Device edge->third-vertex table for the pre-DP hole-fill stage.

    Built once per fill call from the (unsorted) packed sorted-edge keys: probing a rim edge
    with a binary search over ``sorted_keys`` yields its occurrence count (adjacent-face count)
    and, via ``sorted_rows`` -> ``thirds``, the opposite vertex of the single adjacent face. A
    ``(n_vertices,)`` slot scratch supports the forbidden-chord mask of every loop at once.
    """

    def __init__(
        self, vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], edges_sorted: twt.Array2dInt32
    ) -> None:
        device = faces.device
        n_rows = int(edges_sorted.shape[0])
        n_vertices = tw.vertices.n_vertices(edges_sorted)
        self.vertices = vertices
        self.edges_sorted = edges_sorted
        self.max_index = wp.uint64(n_vertices)
        self.n_vertices = n_vertices
        self.device = device

        self.thirds = wp.empty(n_rows, dtype=wp.int32, device=device)
        wp.launch(
            kernel_holes.edge_third_vertex, dim=n_rows, inputs=[faces, self.thirds], device=device
        )

        keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices)
        sorted_keys, sorted_rows = tw.array.sort_and_argsort(keys)
        # Cloned: the views alias scratch that must not be shared with a later sort.
        self.sorted_keys = wp.clone(sorted_keys)
        self.sorted_rows = wp.clone(sorted_rows)

    def rim_opposite(self, loops: _PackedLoops) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
        """Opposite-vertex position + validity per rim edge, for every loop in one launch."""
        positions = wp.empty(loops.total, dtype=wp.vec3, device=self.device)
        valid = wp.empty(loops.total, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel_holes.rim_opposite_from_table,
            dim=loops.total,
            inputs=[
                loops.flat_loops,
                loops.loop_id,
                loops.starts,
                loops.sizes,
                self.vertices,
                self.sorted_keys,
                self.sorted_rows,
                self.thirds,
                self.max_index,
                positions,
                valid,
            ],
            device=self.device,
        )
        return positions, valid

    def forbidden_chords(self, loops: _PackedLoops) -> wp.array[wp.int32]:
        """
        Ragged mask of the chords that already exist as mesh edges, for every loop at once.

        The ``(n_vertices,)`` scratch holds each vertex's *flat* slot rather than its position
        within one loop, which is what lets every loop's mask be marked by a single pass over the
        mesh edges — the per-loop version ran one full ``dim = 3 * n_faces`` pass per hole, and
        cleared the scratch between them.
        """
        slot = wp.full(self.n_vertices, -1, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel_holes.scatter_loop_positions,
            dim=loops.total,
            inputs=[loops.flat_loops, slot],
            device=self.device,
        )
        mask = wp.zeros(loops.dp_total, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel_holes.mark_forbidden_chords,
            dim=int(self.edges_sorted.shape[0]),
            inputs=[
                self.edges_sorted,
                slot,
                loops.loop_id,
                loops.starts,
                loops.sizes,
                loops.dp_offsets,
                mask,
            ],
            device=self.device,
        )
        return mask


def _run_hole_dp(
    loops: _PackedLoops,
    loop_pos: wp.array[wp.vec3],
    plane_normals: wp.array[wp.vec3],
    forbidden: wp.array[wp.int32],
    rim_opp_pos: wp.array[wp.vec3],
    rim_opp_valid: wp.array[wp.int32],
    char_areas: wp.array[wp.float32],
    active: wp.array[wp.int32],
    metric_id: int,
    combine_id: int,
    smooth_boundary: bool,
    dp: wp.array[wp.float32],
    prev: wp.array[wp.int32],
    tiled: bool | None = None,
) -> None:
    """
    Fill the ragged ``dp`` / ``prev`` tables for every loop flagged in ``active``, in place.

    One launch per triangulation span **across all loops**, so the launch count is
    ``max(B) - 1`` for the whole mesh rather than ``B - 1`` per hole.

    ``tiled`` selects the per-span engine: a block per interval with its lanes striding the apex
    loop (CUDA default), or one thread per interval (CPU, and the tie-break reference). Both
    produce byte-identical ``dp`` / ``prev``; ``tiled`` exists so a test can force either. It must
    stay ``False`` on CPU, where ``wp.launch_tiled`` runs a single lane per block and the strided
    apex loop would silently cover only every ``HOLE_DP_BLOCK``-th apex.
    """
    device = loops.device
    if tiled is None:
        tiled = not wp.get_device(device).is_cpu
    wp.launch(
        kernel_holes.init_dp_base,
        dim=(loops.n_loops, loops.max_size),
        inputs=[loops.sizes, loops.dp_offsets, active, dp, prev],
        device=device,
    )
    for span in range(2, loops.max_size):
        inputs = [
            loop_pos,
            loops.starts,
            loops.sizes,
            loops.dp_offsets,
            active,
            plane_normals,
            forbidden,
            rim_opp_pos,
            rim_opp_valid,
            char_areas,
            wp.int32(metric_id),
            wp.int32(combine_id),
            wp.int32(1 if smooth_boundary else 0),
            wp.int32(span),
            dp,
            prev,
        ]
        dim = (loops.n_loops, loops.max_size - span)
        if tiled:
            wp.launch_tiled(
                kernel_holes.fill_dp_span_tiled,
                dim=dim,
                inputs=inputs,
                block_dim=kernel_holes.HOLE_DP_BLOCK,
                device=device,
            )
        else:
            wp.launch(kernel_holes.fill_dp_span, dim=dim, inputs=inputs, device=device)


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


def fill_min_weight(
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
    than [`fill_fan`][triwarp.holes.fill_fan] for non-convex or non-planar holes
    and, unlike [`fill_cone`][triwarp.holes.fill_cone], adds no vertices.

    The ``O(B^3)`` DP runs on device as one parallel kernel launch per triangulation span, and
    **every hole is solved in the same launches**: the per-loop ``B x B`` tables are packed into one
    ragged buffer, so the launch count is ``max(B) - 1`` for the whole mesh rather than ``B - 1``
    per hole, and the chord test, the plane normals and the min-area fallback decision are likewise
    one pass each. Only the ``O(B)`` traceback is host-side, over a single predecessor buffer. A
    mesh with many small holes therefore costs about what one hole costs; before this was batched,
    512 three-vertex holes ran to 376 ms, of which ~100 % was per-hole overhead.

    Each span launch puts a **block** on every interval rather than a thread, with the block's lanes
    striding the apex loop and a two-stage tile reduction picking the winner. That is where the
    parallelism is: the interval grid is only ``n_loops * (max(B) - span)`` wide, so a mesh with a
    couple of long rims had a few hundred threads carrying the whole cubic term. Worth **4.8-7.6x on
    two 512-vertex rims and 3.0x on 512 three-vertex ones**, at a byte-identical triangulation — the
    reduction reproduces the DP's smallest-apex tie-break exactly, which it has to, because the tie
    decides the triangles (see ``kernels/holes.py::fill_dp_span_tiled``).

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
    [`fill_fan`][triwarp.holes.fill_fan]
    [`fill_cone`][triwarp.holes.fill_cone]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    if metric not in _METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}")
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(faces)
    edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    loops = _hole_loops(vertices, faces, preserve_largest_hole, edges_sorted)
    if loops is None:
        return wp.clone(faces)
    return _fill_packed_loops(
        vertices, faces, loops, metric, resolve_multiple_edges, smooth_boundary, edges_sorted
    )


def fill_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: list[wp.array[wp.int32]],
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate the given boundary ``loops`` and append the fill faces.

    Lower-level engine shared by
    [`fill_min_weight`][triwarp.holes.fill_min_weight] and
    [`triwarp.reconstruction.triangulate_point_cloud`]
    [triwarp.reconstruction.triangulate_point_cloud] (to close only a caller-selected subset of
    boundary loops): every loop is sealed **together** by the interval DP under ``metric`` (with a
    ``min_area`` fallback where the primary metric yields a bad triangulation), reusing only
    existing vertices. Cost is set by the longest loop, not by the loop count; see
    [`fill_min_weight`][triwarp.holes.fill_min_weight].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    loops
        Boundary loops to fill, as ordered vertex-index arrays (``>= 3`` vertices each), e.g. from
        [`boundary_loops`][triwarp.boundary.boundary_loops].
    metric
        Fill metric name; see [`fill_min_weight`][triwarp.holes.fill_min_weight].
    resolve_multiple_edges
        When ``True``, forbid chords that duplicate an existing mesh edge.
    smooth_boundary
        See [`fill_min_weight`][triwarp.holes.fill_min_weight].

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of ``faces`` followed by the fill triangles for ``loops``, on
        ``faces.device``. Unchanged (a copy) when ``loops`` is empty.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    """
    if len(loops) == 0:
        return wp.clone(faces)
    return _fill_packed_loops(
        vertices, faces, _pack_loops(loops), metric, resolve_multiple_edges, smooth_boundary
    )


def _fill_packed_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: _PackedLoops,
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool,
    edges_sorted: twt.Array2dInt32 | None = None,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate every packed loop **together** and append the fill faces.

    The engine behind [`fill_loops`][triwarp.holes.fill_loops]. Everything before the
    traceback is batched across loops — one Newell-normal and longest-edge pass, one chord pass over
    the mesh, one ragged ``dp`` / ``prev`` pair, one launch per span rather than per (loop, span),
    and a device-side min-area retry mask instead of a host branch per loop. What is left on the
    host is the ``O(B)`` traceback, which reads *one* packed predecessor table.
    """
    device = faces.device
    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    edge_table = _EdgeTable(vertices, faces, edges_sorted)
    primary_id = _METRIC_IDS[metric]
    combine_id = _METRIC_COMBINE.get(metric, 0)
    min_area_id = _METRIC_IDS["min_area"]

    loop_pos = tw.array.gather(vertices, loops.flat_loops)
    plane_normals = wp.zeros(loops.n_loops, dtype=wp.vec3, device=device)
    max_edge_sq = wp.zeros(loops.n_loops, dtype=wp.float32, device=device)
    wp.launch(
        kernel_holes.loop_rim_metrics,
        dim=loops.total,
        inputs=[
            loops.flat_loops,
            loops.loop_id,
            loops.starts,
            loops.sizes,
            vertices,
            max_edge_sq,
            plane_normals,
        ],
        device=device,
    )
    wp.map(wp.normalize, plane_normals, out=plane_normals)
    char_areas = wp.empty(loops.n_loops, dtype=wp.float32, device=device)
    wp.map(kernel_holes.char_area_from_max, max_edge_sq, out=char_areas)

    forbidden = (
        edge_table.forbidden_chords(loops)
        if resolve_multiple_edges
        else wp.zeros(loops.dp_total, dtype=wp.int32, device=device)
    )
    rim_opp_pos, rim_opp_valid = edge_table.rim_opposite(loops)

    dp = wp.empty(loops.dp_total, dtype=wp.float32, device=device)
    prev = wp.empty(loops.dp_total, dtype=wp.int32, device=device)
    all_loops = wp.ones(loops.n_loops, dtype=wp.int32, device=device)
    _run_hole_dp(
        loops,
        loop_pos,
        plane_normals,
        forbidden,
        rim_opp_pos,
        rim_opp_valid,
        char_areas,
        all_loops,
        primary_id,
        combine_id,
        smooth_boundary,
        dp,
        prev,
    )

    if primary_id != min_area_id:
        # *Which* loops the primary metric failed on is decided on device and fed straight back in
        # as the re-run's active mask, so the fallback is one more batched pass rather than a branch
        # per loop. Loops the primary metric handled keep their ``prev`` rows.
        retry = wp.empty(loops.n_loops, dtype=wp.int32, device=device)
        wp.launch(
            kernel_holes.flag_bad_triangulations,
            dim=loops.n_loops,
            inputs=[loops.sizes, loops.dp_offsets, dp, retry],
            device=device,
        )
        # ...but *whether* any loop failed is worth one host read of ``n_loops`` int32s, because a
        # pass with an all-zero mask is still a full ``max(B) - 1`` launch sweep -- 510 launches on
        # ``rim_short`` -- in which every thread returns immediately. Measured back to back:
        # 155.6 -> 152.5 ms on ``rim_short`` and 4.16 -> 4.03 on ``holes_many``, so about 3 ms and
        # 3 %. That is the *marshalling* of an empty sweep and nothing else; do not expect more from
        # it. In particular the ~33 ms gap between the ``plane_normalized`` default and a single
        # ``min_area`` pass is **not** this pass -- it is ``plane_normalized``'s own per-span kernel
        # doing more work (plane normals, dihedral terms) than ``min_area``'s. ``prev`` is read back
        # a few lines below regardless, so this adds no synchronisation point that was not there.
        if int(retry.numpy().sum()) > 0:
            _run_hole_dp(
                loops,
                loop_pos,
                plane_normals,
                forbidden,
                rim_opp_pos,
                rim_opp_valid,
                char_areas,
                retry,
                min_area_id,
                0,
                smooth_boundary,
                dp,
                prev,
            )

    prev_np = prev.numpy()
    flat_np = loops.flat_loops.numpy()
    triangles: list[tuple[int, int, int]] = []
    for index in range(loops.n_loops):
        triangles.extend(
            _traceback_triangles(loops.dp_block(prev_np, index), flat_np[loops.loop_slice(index)])
        )

    if len(triangles) == 0:
        return wp.clone(faces)
    fill_faces = wp.array(
        np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=wp.int32, device=device
    )
    return tw.array.concatenate([faces, fill_faces])


def _pack_loops(loops: list[wp.array[wp.int32]]) -> _PackedLoops:
    """Concatenate a caller's per-loop arrays into the packed form the fill engine consumes."""
    flat_loops, _offsets = tw.array.pack_1d_arrays(loops)
    sizes_np = np.asarray([int(loop.shape[0]) for loop in loops], dtype=np.int64)
    return _PackedLoops(flat_loops, sizes_np)


def fill_small(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_perimeter: float
) -> wp.array[wp.int32]:
    """
    Fill only boundary loops whose perimeter is at most ``max_perimeter``.

    Intended open boundaries (large loops) are left untouched; spurious small holes are sealed by
    the shared min-weight interval DP ([`fill_loops`][triwarp.holes.fill_loops]). Used by
    [`triwarp.reconstruction.triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]
    to seal small gaps left by sparse or non-uniform point-cloud sampling (MeshLib ``makeMesh_``
    tail).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_perimeter
        Only boundary loops with perimeter at most this value are filled.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of ``faces`` followed by the fill triangles for the small loops, on
        ``faces.device``. Unchanged (a copy is not made; returns ``faces``) when there is no
        boundary loop at or under ``max_perimeter``.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    [`fill_loops`][triwarp.holes.fill_loops]
    """
    packed = _hole_loops(vertices, faces)
    if packed is None:
        return faces

    # One segmented-sum launch measures every rim, so the selection costs a single readback rather
    # than a copy of the whole vertex buffer plus one of each loop.
    small_np = _loop_perimeters(vertices, packed) <= max_perimeter
    if not small_np.any():
        return faces
    if not small_np.all():
        kept = zip(_unpack_loops(packed), small_np, strict=True)
        packed = _pack_loops([loop for loop, keep in kept if keep])
    return _fill_packed_loops(vertices, faces, packed, "plane_normalized", True, True)


def _mean_rim_edge_length(vertices: wp.array[wp.vec3], loops: list[wp.array[wp.int32]]) -> float:
    """Mean edge length over the rims of the given loops (the derived subdivision target)."""
    total = 0.0
    count = 0
    vertices_np = vertices.numpy()
    for loop in loops:
        loop_np = loop.numpy()
        pos = vertices_np[loop_np]
        seg = np.linalg.norm(pos - np.roll(pos, -1, axis=0), axis=1)
        total += float(seg.sum())
        count += int(seg.shape[0])
    return total / count if count > 0 else 0.0


def _patch_mask(
    n_faces_before: int, n_faces_after: int, device: wp.DeviceLike
) -> wp.array[wp.bool]:
    """Boolean face mask marking the trailing ``[n_faces_before, n_faces_after)`` fill faces."""
    mask = np.zeros(n_faces_after, dtype=bool)
    mask[n_faces_before:] = True
    return wp.array(mask, dtype=wp.bool, device=device)


def fill_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    preserve_largest_hole: bool = False,
    resolve_multiple_edges: bool = True,
    smooth_boundary: bool = True,
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Fill every boundary hole with a smooth, refined patch.

    Runs the full three-stage pipeline: a minimum-weight triangulation seals each hole over its
    existing rim vertices ([`fill_min_weight`][triwarp.holes.fill_min_weight]),
    the patch is then subdivided to ``max_edge`` with Delaunay edge flips
    ([`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]), and finally the new
    interior patch vertices are smoothed into the surrounding surface
    ([`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim]
    then [`smooth_region`][triwarp.smoothing.smooth_region]). Unlike the purely
    topological fillers, this produces a well-graded, curvature-continuous patch.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    metric
        Minimum-weight fill metric; see
        [`fill_min_weight`][triwarp.holes.fill_min_weight].
    triangulate_only
        When ``True``, only fill (no subdivision or smoothing) — equivalent to
        [`fill_min_weight`][triwarp.holes.fill_min_weight].
    max_edge
        Target maximum patch edge length. ``None`` (default) derives it from the mean rim edge
        length of the holes being filled (MeshLib's ``maxEdgeLen = 0`` budget target has no
        parallel analogue).
    max_edge_splits
        Soft cap on the number of edge splits during subdivision (``maxEdgeSplits``).
    max_angle_change_after_flip
        Dihedral-angle-change gate for the Delaunay flip pass (default 30°).
    smooth_curvature
        When ``True`` (default), smooth the new patch vertices after subdivision.
    natural_smooth
        When ``True``, additionally grow a collar around the patch and smooth it so the patch
        blends into the surrounding surface (MeshLib ``naturalSmooth``).
    edge_weights
        Laplacian edge weights for the cross-boundary smooth solve: ``"cotan"`` (default) or
        ``"unit"``.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop open.
    resolve_multiple_edges
        When ``True`` (default), forbid fill chords that duplicate existing mesh edges.
    smooth_boundary
        When ``True`` (default), also run the cross-boundary smooth solve so the patch is C¹ across
        its rim (MeshLib ``smoothBd``). Also tunes the fill metric's rim edge terms.
    return_patch
        When ``True``, also return a length-``n_out_faces`` ``wp.bool`` mask of the patch faces.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by the inserted patch vertices, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Original faces followed by the patch faces.
    patch_mask : wp.array[wp.bool]
        Only when ``return_patch`` is ``True``: mask of the patch faces in ``new_faces``.

    Raises
    ------
    ValueError
        If ``metric`` or ``edge_weights`` is unknown.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    [`stitch_smooth`][triwarp.combine.stitch_smooth]
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]

    Notes
    -----
    The three-stage pipeline is MeshLib's ``fillHoleNicely``.

    The subdivision and smoothing stages require a CUDA device (``warp.optim.linear.cg`` produces
    NaN on CPU in Warp 1.14-1.15); ``triangulate_only=True`` stays CPU-capable. Winding is
    consistent with the surrounding faces only for a consistently wound input.
    """
    if metric not in _METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}")
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        empty = _patch_mask(int(faces.shape[0]) // 3, int(faces.shape[0]) // 3, device)
        result = (wp.clone(vertices), wp.clone(faces))
        return (*result, empty) if return_patch else result

    n_faces_before = int(faces.shape[0]) // 3
    n_vertices_before = int(vertices.shape[0])
    faces_filled = _fill_packed_loops(
        vertices, faces, packed, metric, resolve_multiple_edges, smooth_boundary
    )
    n_faces_after = int(faces_filled.shape[0]) // 3
    patch_mask = _patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (wp.clone(vertices), faces_filled)
        return (*result, patch_mask) if return_patch else result

    target_edge = (
        max_edge if max_edge is not None else _mean_rim_edge_length(vertices, _unpack_loops(packed))
    )
    new_vertices, new_faces, out_patch = tw.smoothing.refine_and_smooth_region(
        vertices,
        faces_filled,
        n_vertices_before,
        patch_mask,
        target_edge,
        max_edge_splits,
        max_angle_change_after_flip,
        smooth_curvature,
        smooth_boundary,
        natural_smooth,
        edge_weights,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)
