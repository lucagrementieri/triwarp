"""
Closing mesh boundary holes (Warp).

Every boundary of a triangle mesh is an ordered vertex loop
([`boundary_loops`][triwarp.boundary.boundary_loops], the analog of
``trimesh.repair.fill_holes``'s ``nx.cycle_basis`` and ``igl::boundary_loop_all``). Each loop
of ``B`` vertices is sealed with a purely topological triangulation — **no smoothing or
refinement**:

- [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan] fans ``B - 2`` triangles from the loop's
  first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
- [`fill_holes_cone`][triwarp.hole_filling.fill_holes_cone] inserts one centroid vertex per hole
  and cones ``B`` triangles onto it (``igl::topological_hole_fill``,
  ``trimesh.repair.stitch(insert_vertices=True)``).
- [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight] instead computes the
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

The two lower-level engines behind [`combine.stitch`][triwarp.combine.stitch] and
[`combine.stitch_min_weight`][triwarp.combine.stitch_min_weight] — which join **two** open
meshes across one boundary loop each — also live here:
[`triangulate_boundaries`][triwarp.hole_filling.triangulate_boundaries] zippers the two rims
with a band of bridge triangles via a greedy correspondence, producing a single watertight seam.
[`triangulate_boundaries_min_weight`][triwarp.hole_filling.triangulate_boundaries_min_weight]
instead chooses the band that minimizes a stitch metric (the MeshLib ``stitchHoles`` grid DP).
No smoothing or refinement is applied by these lower-level fillers.

For a smooth, well-graded patch, [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely] and
[`combine.stitch_nicely`][triwarp.combine.stitch_nicely] run the full MeshLib ``fillHoleNicely`` /
``stitchHolesNicely`` pipeline on top of the min-weight fill/stitch: the patch is refined to a
target edge length with Delaunay edge flips
([`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]) and its new interior
vertices are smoothed into the surrounding surface — a sharp-boundary umbrella solve
([`position_verts_smoothly_sharp_boundary`][triwarp.smoothing.position_verts_smoothly_sharp_boundary])
followed by a cross-boundary least-squares solve
([`position_verts_smoothly`][triwarp.smoothing.position_verts_smoothly]), with an optional
``natural_smooth`` collar that blends the patch into the neighbouring surface.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import hole_filling as kernel_hole_filling


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
    [`closed_polyline_length`][triwarp.polyline.closed_polyline_length] call (two synchronizations)
    per loop.
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
        kernel_hole_filling.loop_perimeters,
        dim=loops.total,
        inputs=[loops.flat_loops, loops.loop_id, loops.starts, loops.sizes, vertices, perimeter],
        device=loops.device,
    )
    return perimeter.numpy()


def _unpack_loops(loops: _PackedLoops) -> list[wp.array[wp.int32]]:
    """Per-loop **views** into the packed buffer, for the callers that still want a list."""
    return [loops.flat_loops[loops.loop_slice(index)] for index in range(loops.n_loops)]


def fill_holes_fan(
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
    [`fill_holes_cone`][triwarp.hole_filling.fill_holes_cone]
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
        kernel_hole_filling.fan_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), fill_faces],
        device=device,
    )
    return tw.array.concatenate([faces, fill_faces])


def fill_holes_cone(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], preserve_largest_hole: bool = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Fill every boundary hole by coning it onto a new centroid vertex.

    A boundary loop of ``B`` vertices is sealed with ``B`` triangles fanning from one new vertex
    placed at the loop's centroid (``igl::topological_hole_fill``,
    ``trimesh.repair.stitch(insert_vertices=True)``). Unlike
    [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan] this appends one vertex per hole, so
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
    [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan]
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
        kernel_hole_filling.loop_centroids,
        dim=n_loops,
        inputs=[vertices, flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), centroids],
        device=device,
    )

    fill_faces = wp.empty(3 * total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.cone_faces,
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
# kernels/hole_filling.py).
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
            kernel_hole_filling.edge_third_vertex,
            dim=n_rows,
            inputs=[faces, self.thirds],
            device=device,
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
            kernel_hole_filling.rim_opposite_from_table,
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
            kernel_hole_filling.scatter_loop_positions,
            dim=loops.total,
            inputs=[loops.flat_loops, slot],
            device=self.device,
        )
        mask = wp.zeros(loops.dp_total, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel_hole_filling.mark_forbidden_chords,
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
        kernel_hole_filling.init_dp_base,
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
                kernel_hole_filling.fill_dp_span_tiled,
                dim=dim,
                inputs=inputs,
                block_dim=kernel_hole_filling.HOLE_DP_BLOCK,
                device=device,
            )
        else:
            wp.launch(kernel_hole_filling.fill_dp_span, dim=dim, inputs=inputs, device=device)


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
    than [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan] for non-convex or non-planar holes
    and, unlike [`fill_holes_cone`][triwarp.hole_filling.fill_holes_cone], adds no vertices.

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
    decides the triangles (see ``kernels/hole_filling.py::fill_dp_span_tiled``).

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
    [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan]
    [`fill_holes_cone`][triwarp.hole_filling.fill_holes_cone]
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
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight] and
    [`triwarp.reconstruction.triangulate_point_cloud`]
    [triwarp.reconstruction.triangulate_point_cloud] (to close only a caller-selected subset of
    boundary loops): every loop is sealed **together** by the interval DP under ``metric`` (with a
    ``min_area`` fallback where the primary metric yields a bad triangulation), reusing only
    existing vertices. Cost is set by the longest loop, not by the loop count; see
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight].

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
        Fill metric name; see [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight].
    resolve_multiple_edges
        When ``True``, forbid chords that duplicate an existing mesh edge.
    smooth_boundary
        See [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight].

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of ``faces`` followed by the fill triangles for ``loops``, on
        ``faces.device``. Unchanged (a copy) when ``loops`` is empty.

    See Also
    --------
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight]
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

    The engine behind [`fill_loops`][triwarp.hole_filling.fill_loops]. Everything before the
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
        kernel_hole_filling.loop_rim_metrics,
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
    wp.map(kernel_hole_filling.char_area_from_max, max_edge_sq, out=char_areas)

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
            kernel_hole_filling.flag_bad_triangulations,
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


def fill_small_holes(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_perimeter: float
) -> wp.array[wp.int32]:
    """
    Fill only boundary loops whose perimeter is at most ``max_perimeter``.

    Intended open boundaries (large loops) are left untouched; spurious small holes are sealed by
    the shared min-weight interval DP ([`fill_loops`][triwarp.hole_filling.fill_loops]). Used by
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
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight]
    [`fill_loops`][triwarp.hole_filling.fill_loops]
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


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][triwarp.hole_filling._non_increasing_indices]); the algorithm does
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
    [`stitch`][triwarp.combine.stitch]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`concatenate`][triwarp.combine.concatenate]

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
        kernel_hole_filling.cyclic_gather,
        dim=n,
        inputs=[loop_a, wp.int32(n), wp.int32(0), wp.bool(True), wp.int32(0), flipped_loop_a],
        device=device,
    )
    a_pos = tw.array.gather(vertices_a, flipped_loop_a)
    b_pos = tw.array.gather(vertices_b, loop_b)

    # perimeters[i, j] = |a_i - b_j| + |a_{i+1} - b_j| for A-edge i and B-vertex j.
    perimeters = twt.empty_float32_2d((n, m), device=device)
    wp.launch(
        kernel_hole_filling.boundary_perimeters,
        dim=(n, m),
        inputs=[a_pos, b_pos, wp.int32(n), perimeters],
        device=device,
    )

    col_min = wp.empty(n, dtype=wp.int32, device=device)
    val_min = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_hole_filling.row_argmin,
        dim=n,
        inputs=[perimeters, wp.int32(m), col_min, val_min],
        device=device,
    )

    shift = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    shift_np = shift.numpy()
    shift_a = int(shift_np[0])
    shift_b = int(shift_np[1])

    edge_dev = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.rolled_edge_map,
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
            kernel_hole_filling.resolve_corrections,
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
        kernel_hole_filling.cyclic_gather,
        dim=n,
        inputs=[
            flipped_loop_a,
            wp.int32(n),
            wp.int32(row_roll),
            wp.bool(False),
            wp.int32(0),
            roll_a,
        ],
        device=device,
    )
    roll_b = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.cyclic_gather,
        dim=m,
        inputs=[
            loop_b,
            wp.int32(m),
            wp.int32(col_roll),
            wp.bool(False),
            wp.int32(n_vertices_a),
            roll_b,
        ],
        device=device,
    )

    bridge_a = wp.empty(3 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.bridge_a_faces,
        dim=n,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), bridge_a],
        device=device,
    )
    bridge_b = wp.empty(3 * m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.bridge_b_faces,
        dim=m,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), wp.int32(m), bridge_b],
        device=device,
    )

    combined_vertices, combined_faces = tw.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, tw.array.concatenate([combined_faces, bridge_a, bridge_b])


# Stitch-metric name -> kernel selector (must match the METRIC_*_STITCH constants in
# kernels/hole_filling.py).
_STITCH_METRIC_IDS = {"complex_stitch": 0, "edge_length_stitch": 1, "vertical": 2}


def _closest_loop_pair(a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3]) -> tuple[int, int]:
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
        kernel_hole_filling.pair_sq_distances,
        dim=(n_a, n_b),
        inputs=[a_pos, b_pos, dist_sq],
        device=device,
    )
    col_min = wp.empty(n_a, dtype=wp.int32, device=device)
    val_min = wp.empty(n_a, dtype=wp.float32, device=device)
    wp.launch(
        kernel_hole_filling.row_argmin,
        dim=n_a,
        inputs=[dist_sq, wp.int32(n_b), col_min, val_min],
        device=device,
    )
    pair = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_hole_filling.global_argmin,
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
    [`triangulate_boundaries`][triwarp.hole_filling.triangulate_boundaries] remains available.

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
    [`triangulate_boundaries`][triwarp.hole_filling.triangulate_boundaries]
    [`stitch_min_weight`][triwarp.combine.stitch_min_weight]
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")
    n_a = int(loop_a.shape[0])
    n_b = int(loop_b.shape[0])
    if n_a < 3 or n_b < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n_a} and {n_b}")

    device = faces_a.device
    metric_id = _STITCH_METRIC_IDS[metric]
    offset = int(vertices_a.shape[0])

    # Reverse loop A so the two rims wind oppositely (facing), then align both at the closest pair.
    # ``la`` / ``lb`` stay on the host for the sequential band traceback, but the closest-pair
    # search, rim gathers and rim-opposite lookups all run on device.
    la = loop_a.numpy()[::-1].copy()
    lb = loop_b.numpy().copy()
    a_rim = tw.array.gather(vertices_a, wp.array(la, dtype=wp.int32, device=device))
    b_rim = tw.array.gather(vertices_b, wp.array(lb, dtype=wp.int32, device=device))
    start_a, start_b = _closest_loop_pair(a_rim, b_rim)
    la = np.roll(la, -start_a)
    lb = np.roll(lb, -start_b)

    la_wp = wp.array(la, dtype=wp.int32, device=device)
    lb_wp = wp.array(lb, dtype=wp.int32, device=device)
    a_pos = tw.array.gather(vertices_a, la_wp)
    b_pos = tw.array.gather(vertices_b, lb_wp)
    table_a = _EdgeTable(vertices_a, faces_a, tw.edges.faces_to_edges(faces_a, sorted=True))
    table_b = _EdgeTable(vertices_b, faces_b, tw.edges.faces_to_edges(faces_b, sorted=True))
    # The stitch DP works on exactly one rim per side, so each rim is its own one-loop batch.
    a_opp, a_opp_valid = table_a.rim_opposite(_PackedLoops(la_wp, np.array([n_a], dtype=np.int64)))
    b_opp, b_opp_valid = table_b.rim_opposite(_PackedLoops(lb_wp, np.array([n_b], dtype=np.int64)))
    up = wp.vec3(*(up_dir if up_dir is not None else (0.0, 0.0, 1.0)))

    dp = twt.as_array2d_float32(
        wp.full((n_a + 1, n_b + 1), _BAD_TRIANGULATION_METRIC, dtype=wp.float32, device=device)
    )
    wp.launch(kernel_hole_filling.set_dp_origin, dim=1, inputs=[dp], device=device)
    came = twt.as_array2d_int32(wp.full((n_a + 1, n_b + 1), -1, dtype=wp.int32, device=device))
    for diag in range(1, n_a + n_b + 1):
        wp.launch(
            kernel_hole_filling.stitch_dp_diag,
            dim=min(diag, n_a) - max(0, diag - n_b) + 1,
            inputs=[
                a_pos,
                b_pos,
                a_opp,
                a_opp_valid,
                b_opp,
                b_opp_valid,
                up,
                wp.int32(metric_id),
                wp.int32(n_a),
                wp.int32(n_b),
                wp.int32(diag),
                dp,
                came,
            ],
            device=device,
        )
    came_np = came.numpy()

    band = _stitch_band_triangles(came_np, la, lb, n_a, n_b, offset)
    combined_vertices, combined_faces = tw.combine.concatenate(
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


# ---------------------------------------------------------------------------
# "Nicely" pipeline: min-weight fill / stitch -> region subdivision -> region smoothing
# (MeshLib fillHoleNicely / stitchHolesNicely, MRFillHoleNicely.cpp).
# ---------------------------------------------------------------------------


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


def _boundary_verts_mask(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """Length-``n_vertices`` mask of mesh-boundary vertices (MeshLib ``findBdVerts``)."""
    device = faces.device
    n = int(vertices.shape[0])
    boundary = tw.boundary.boundary_vertex_indices(vertices, faces)
    return tw.array.indices_to_mask(boundary, n, device=device)


def _finish_nicely(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_vertices_before: int,
    patch_face_mask: wp.array[wp.bool],
    max_edge: float,
    max_edge_splits: int,
    max_angle_change_after_flip: float,
    smooth_curvature: bool,
    smooth_boundary: bool,
    natural_smooth: bool,
    edge_weights: str,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Subdivide the patch and smooth its new vertices.

    Ports MeshLib ``subdivideFillingNicely`` + ``smoothFillingNicely``.

    Shared finisher of [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely] and
    [`stitch_nicely`][triwarp.combine.stitch_nicely].
    """
    device = faces.device
    vertices, faces, patch_face_mask = tw.remesh.subdivide_region_to_size(
        vertices,
        faces,
        patch_face_mask,
        max_edge=max_edge,
        max_splits=max_edge_splits,
        max_angle_change=max_angle_change_after_flip,
    )
    if not smooth_curvature:
        return vertices, faces, patch_face_mask

    n = int(vertices.shape[0])
    # New (interior patch) vertices are the tail appended by subdivision, minus mesh-boundary verts.
    new_verts_np = np.zeros(n, dtype=bool)
    new_verts_np[n_vertices_before:] = True
    new_verts = wp.array(new_verts_np, dtype=wp.bool, device=device)
    bd_mask = _boundary_verts_mask(vertices, faces)
    free = wp.empty(n, dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_and_not, new_verts, bd_mask, out=free)

    vertices = tw.smoothing.position_verts_smoothly_sharp_boundary(vertices, faces, free)
    if smooth_boundary:
        vertices = tw.smoothing.position_verts_smoothly(vertices, faces, free, edge_weights)

    if natural_smooth:
        edges_bd = tw.selection.region_boundary_edges(faces, patch_face_mask, n_vertices=n)
        endpoints = wp.clone(twt.as_array2d_int32(edges_bd).reshape(-1))
        incident = tw.array.indices_to_mask(endpoints, n, device=device)
        incident = tw.selection.expand_vertex_mask(faces, incident, 5)
        incident = tw.selection.shrink_vertex_mask(faces, incident, 2)
        incident = tw.selection.exclude_fully_selected_components(faces, incident, n)
        if bool(incident.numpy().any()):
            bd_mask = _boundary_verts_mask(vertices, faces)
            free2 = wp.empty(n, dtype=wp.bool, device=device)
            wp.map(kernel_array.mask_and_not, incident, bd_mask, out=free2)
            vertices = tw.smoothing.position_verts_smoothly_sharp_boundary(vertices, faces, free2)
            vertices = tw.smoothing.position_verts_smoothly(vertices, faces, free2, edge_weights)

    return vertices, faces, patch_face_mask


def _patch_mask(
    n_faces_before: int, n_faces_after: int, device: wp.DeviceLike
) -> wp.array[wp.bool]:
    """Boolean face mask marking the trailing ``[n_faces_before, n_faces_after)`` fill faces."""
    mask = np.zeros(n_faces_after, dtype=bool)
    mask[n_faces_before:] = True
    return wp.array(mask, dtype=wp.bool, device=device)


def fill_holes_nicely(
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
    Fill every boundary hole with a smooth, refined patch (MeshLib ``fillHoleNicely``).

    Runs the full three-stage pipeline: a minimum-weight triangulation seals each hole over its
    existing rim vertices ([`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight]),
    the patch is then subdivided to ``max_edge`` with Delaunay edge flips
    ([`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]), and finally the new
    interior patch vertices are smoothed into the surrounding surface
    ([`position_verts_smoothly_sharp_boundary`][triwarp.smoothing.position_verts_smoothly_sharp_boundary]
    then [`position_verts_smoothly`][triwarp.smoothing.position_verts_smoothly]). Unlike the purely
    topological fillers, this produces a well-graded, curvature-continuous patch.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    metric
        Minimum-weight fill metric; see
        [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight].
    triangulate_only
        When ``True``, only fill (no subdivision or smoothing) — equivalent to
        [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight].
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
    [`fill_holes_min_weight`][triwarp.hole_filling.fill_holes_min_weight]
    [`stitch_nicely`][triwarp.combine.stitch_nicely]
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]

    Notes
    -----
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
    new_vertices, new_faces, out_patch = _finish_nicely(
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
