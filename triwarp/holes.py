"""
Closing mesh boundary holes, and stitching two open meshes along a rim.

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
  **minimum-weight triangulation** of each loop (the Liepa/Klincsek interval DP): the ``B - 2``
  triangles over the existing loop vertices that minimize a
  geometric metric (plane-normalized circumcircle by default, with a min-area fallback), avoiding
  chords that would duplicate existing mesh edges. This is the robust, general-purpose filler for
  non-convex and non-planar holes; it adds no vertices.

Because ``boundary_loops`` orders each loop following the face-winding direction of the
boundary half-edges, a fill triangle sharing a rim edge is emitted **reversed** on that edge so
its winding is consistent with the adjacent original face. This assumes the input mesh is
consistently wound; run [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
first on meshes with mixed winding.

Joining **two** open meshes across one boundary loop each is the same family and lives here too:
[`stitch`][triwarp.holes.stitch] and its variants, over the loop-level engines
[`stitch_loops`][triwarp.holes.stitch_loops] and
[`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight]. It is the *same* minimum-weight
machinery -- the same interval DP, the same rim bookkeeping and the same metric vocabulary --
applied to a band between two rims rather than a cap over one, which is why the two halves share a
module rather than a private import list.

For a smooth, well-graded patch, [`fill_smooth`][triwarp.holes.fill_smooth] and
[`stitch_smooth`][triwarp.holes.stitch_smooth] run a three-stage pipeline on top of the
min-weight fill/stitch: the patch is refined to a
target edge length with Delaunay edge flips
([`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]) and its new interior
vertices are smoothed into the surrounding surface — a sharp-boundary umbrella solve
([`smooth_region_fixed_rim`][triwarp.smoothing.smooth_region_fixed_rim])
followed by a cross-boundary least-squares solve
([`smooth_region`][triwarp.smoothing.smooth_region]), with an optional
``natural_smooth`` collar that blends the patch into the neighbouring surface.

Not every boundary wants closing. [`extend_hole`][triwarp.holes.extend_hole] and
[`build_bottom`][triwarp.holes.build_bottom] *extrude* a rim -- out to a plane you place, or down
to a base fitted under each rim's own lowest point -- which leaves the mesh open with a planar rim
that a min-weight fill then closes without folding; that two-step is how a scanned shell becomes a
printable solid. And [`bridge_edges`][triwarp.holes.bridge_edges] and
[`bridge_edges_smooth`][triwarp.holes.bridge_edges_smooth] are the *local* form of stitching: they
join one boundary edge to another with a small patch or a curved strip, leaving the rest of both
boundaries open, which is what joins two tubes at a chosen seam or adds a handle where the rim
family would consume the whole loop.
[`join_closest_components`][triwarp.holes.join_closest_components] is the driver over the first of
those: it settles *which* edges to bridge, welding several open shells into one connected surface
with one rim, for a filler to close afterwards.

The return shape follows from that: a filler that only triangulates existing rim vertices returns
``faces`` alone, while one that inserts a vertex -- a cone's apex, a refined patch's interior, an
extrusion's projected ring -- returns ``(vertices, faces)``, because the position buffer grew too.

Every filler returns a buffer **independent of** ``faces``, including on the no-op path where
there was no hole to fill, so a caller may write into the result without disturbing its input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar
from triwarp.kernels import array as kernel_array
from triwarp.kernels import holes as kernel_holes
from triwarp.kernels import scatter as kernel_scatter


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

    def perimeters(self, vertices: wp.array[wp.vec3]) -> np.ndarray:
        """
        Measure the closed arc length of every packed loop, returning it on the host.

        Delegates to [`boundary.loop_perimeters_batched`][triwarp.boundary.loop_perimeters_batched]
        rather than launching the segmented kernel again here: the packed layout this class holds
        *is* that function's argument list, and ``loop_id`` is passed rather than rebuilt, so the
        call costs exactly what the private copy this replaced did.
        """
        return tw.boundary.loop_perimeters_batched(
            vertices, self.flat_loops, self.starts, self.sizes, loop_id=self.loop_id
        ).numpy()

    def loop_slice(self, index: int) -> slice:
        """Host slice of ``flat_loops`` (and of any other length-``total`` buffer) for a loop."""
        start = int(self.starts_np[index])
        return slice(start, start + int(self.sizes_np[index]))

    def dp_block(self, table_np: np.ndarray, index: int) -> np.ndarray:
        """``(B, B)`` view of one loop's block inside a host copy of a ragged DP table."""
        start = int(self.dp_offsets_np[index])
        size = int(self.sizes_np[index])
        return table_np[start : start + size * size].reshape(size, size)


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
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest. This turns a mesh with a known disk-like topology into a single-boundary
        disk, which is the input a robust UV parametrization expects.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of the original faces followed by the new fill triangles, on
        ``faces.device``. A watertight or empty mesh — or, with ``preserve_largest_hole``, a mesh
        whose only hole is the largest — is returned unchanged (a copy).

    See Also
    --------
    [`fill_cone`][triwarp.holes.fill_cone]
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        Far more robust on a non-convex or non-planar hole, and it also adds no vertices.
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
        inputs=[flat_loops, loop_starts, packed.sizes, fill_faces],
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
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest. This turns a mesh with a known disk-like topology into a single-boundary
        disk, which is the input a robust UV parametrization expects.

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
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        Far more robust on a non-convex or non-planar hole, and unlike this it adds no vertices.
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
        inputs=[vertices, flat_loops, loop_starts, packed.sizes, centroids],
        device=device,
    )

    fill_faces = wp.empty(3 * total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.cone_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_starts, packed.sizes, wp.int32(n_vertices), fill_faces],
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
        n_vertices = tw.array.index_bound(edges_sorted)
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

    The classic Liepa/Klincsek interval dynamic program: each boundary loop of ``B`` vertices is
    sealed with the ``B - 2`` triangles that minimize a geometric metric, reusing only existing
    vertices (the vertex buffer is unchanged). This is far more robust
    than [`fill_fan`][triwarp.holes.fill_fan] for non-convex or non-planar holes
    and, unlike [`fill_cone`][triwarp.holes.fill_cone], adds no vertices.

    The ``O(B^3)`` DP runs on device as one parallel kernel launch per triangulation span, and
    **every hole is solved in the same launches**: the per-loop ``B x B`` tables are packed into one
    ragged buffer, so the launch count is ``max(B) - 1`` for the whole mesh rather than ``B - 1``
    per hole, and the chord test, the plane normals and the min-area fallback decision are likewise
    one pass each. Only the ``O(B)`` traceback is host-side, over a single predecessor buffer. A
    mesh with many small holes therefore costs about what one hole costs.

    Each span launch puts a **block** on every interval rather than a thread, with the block's lanes
    striding the apex loop and a two-stage tile reduction picking the winner: the interval grid is
    only ``n_loops * (max(B) - span)`` wide, so a mesh with a couple of long rims would otherwise
    have only a few hundred threads carrying the whole cubic term. The reduction reproduces the DP's
    smallest-apex tie-break exactly, which it has to, because the tie decides the triangles (see
    ``kernels/holes.py::fill_dp_span_tiled``).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    metric
        Which fill metric to minimize:

        - ``"plane_normalized"`` (default) — circumcircle-diameter times aspect ratio, penalizing
          triangles flipped or tilted more than 60 degrees off the hole plane; falls back to
          ``"min_area"`` when the best triangulation is still bad (non-planar/degenerate).
        - ``"min_area"`` — summed triangle area; never rejects a triangulation as bad, which is
          what makes it the fallback the other metrics fall back *to*. That is a statement about
          the metric only: see ``resolve_multiple_edges`` for the one thing that can still leave an
          interval unfilled under it.
        - ``"circumscribed"`` — summed circumcircle diameter.
        - ``"plane"`` — circumcircle diameter with a flipped-normal penalty.
        - ``"min_tri_angle"`` — maximizes the minimal triangle angle.
        - ``"edge_length"`` — summed new-edge length.
        - ``"universal"`` — circumcircle diameter plus a dihedral-smoothing edge term; the smooth,
          general-purpose choice.
        - ``"max_dihedral"`` — minimizes the maximal dihedral angle.
        - ``"complex_fill"`` — area/aspect triangle term plus a strong dihedral edge term.

        The dihedral (edge-based) metrics — ``universal``, ``max_dihedral``, ``complex_fill``,
        ``edge_length`` — blend into the surrounding surface via ``smooth_boundary``.
    resolve_multiple_edges
        When ``True`` (default), forbid the triangulation from creating a chord that duplicates an
        existing mesh edge, avoiding non-manifold results on pinched holes. Such a chord is
        forbidden outright rather than re-routed, and the constraint binds under **every** metric
        including the ``min_area`` fallback. An interval with no admissible apex left contributes
        no triangles and is skipped silently, so a hole whose every triangulation is blocked comes
        back still open rather than raising — pass ``False`` to trade that for a possibly
        non-manifold fill.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (greatest perimeter) open and fill
        only the rest — the single-boundary disk a robust UV parametrization expects.
    smooth_boundary
        When ``True`` (default), the dihedral edge metrics also score the hole
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


def fill_loops_min_weight(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: list[wp.array[wp.int32]],
    metric: str,
    resolve_multiple_edges: bool,
    smooth_boundary: bool = True,
) -> wp.array[wp.int32]:
    """
    Min-weight-triangulate the given boundary ``loops`` and append the fill faces.

    The loop-selecting form of [`fill_min_weight`][triwarp.holes.fill_min_weight]: same engine, but
    the caller names which boundary loops to close rather than getting every one of them. Both
    wrappers hand the packed loops to one shared private engine, so the triangulation is identical
    where the loop sets are. Every loop is sealed **together** by the interval DP under ``metric``
    (with a ``min_area`` fallback where the primary metric yields a bad triangulation), reusing
    only existing vertices. Cost is set by the longest loop, not by the loop count; see
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

    The engine behind [`fill_loops_min_weight`][triwarp.holes.fill_loops_min_weight]. Everything
    before the traceback is batched across loops — one Newell-normal and longest-edge pass, one
    chord pass over the mesh, one ragged ``dp`` / ``prev`` pair, one launch per span rather than
    per (loop, span),
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
        # ...but *whether* any loop failed is worth one host read, because a pass with an all-zero
        # mask is still a full ``max(B) - 1`` launch sweep in which every thread returns
        # immediately -- that is pure marshalling and buys nothing on its own. ``prev`` is read
        # back a few lines below regardless, so this adds no synchronisation point that was not
        # there. The test itself is a device reduction rather than a full readback of the
        # ``n_loops`` buffer, so it stays cheap even on a mesh with many loops.
        if tw.reduce.max(retry) > 0:
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
    ``max(B) - 1`` for the whole mesh rather than ``B - 1`` per hole. The launch count is the
    floor on this engine: a *blocked* interval DP that tiled the ``(i, j)`` plane could cut it
    further, but that is a new kernel with in-tile sequencing rather than a knob, and it is
    unbuilt. Folding the spans into one persistent block per loop -- lanes striding the
    ``(interval, apex)`` pairs, a tile reduction as the level barrier -- was tried and loses,
    because the cubic apex-evaluation work then runs on a single SM instead of being spread across
    one block per span by the whole device; beating both needs a grid-wide barrier, which Warp
    does not expose.

    ``tiled`` selects the per-span engine: a block per interval with its lanes striding the apex
    loop, or one thread per interval. Both produce byte-identical ``dp`` / ``prev`` on **both**
    devices, so this is a pure cost knob and ``None`` picks whichever is faster on the device at
    hand -- tiled on CUDA, where a block has lanes to spread the apex loop over; serial on CPU,
    where it has one and the two engines do the same work. ``tiled`` exists so a test can force
    either and compare them.

    That portability rests on ``fill_dp_span_tiled`` striding by ``wp.block_dim()`` rather than by
    the ``HOLE_DP_BLOCK`` it is launched with: the two agree on CUDA, and on CPU the former reads 1,
    so the single lane covers every apex and the tile reductions after it degenerate to one-element
    tiles holding that lane's own answer. With the constant it would silently minimize over every
    32nd apex instead.
    """
    device = loops.device
    if tiled is None:
        tiled = not wp.get_device(device).is_cpu
    # One of two lane counts, picked from the rim count *and* the longest rim -- see
    # ``kernels/holes.hole_dp_block``, whose table shows why both matter. Read once here rather
    # than per span so the whole sweep shares one module hash.
    block = kernel_holes.hole_dp_block(loops.max_size, loops.n_loops)
    wp.launch(
        kernel_holes.init_dp_base,
        dim=(loops.n_loops, loops.max_size),
        inputs=[loops.sizes, loops.dp_offsets, active, dp, prev],
        device=device,
    )
    # Built once, outside the loop: every field is invariant across spans, and a wp.launch argument
    # costs host time whatever it holds. Rebuilding it per span would give the saving straight back.
    tables = kernel_holes.HoleFillTables()
    tables.loop_pos = loop_pos
    tables.loop_starts = loops.starts
    tables.loop_sizes = loops.sizes
    tables.dp_offsets = loops.dp_offsets
    tables.active = active
    tables.plane_normals = plane_normals
    tables.forbidden = forbidden
    tables.rim_opp_pos = rim_opp_pos
    tables.rim_opp_valid = rim_opp_valid
    tables.char_areas = char_areas
    tables.metric_id = wp.int32(metric_id)
    tables.combine_id = wp.int32(combine_id)
    tables.smooth_bd = wp.int32(1 if smooth_boundary else 0)
    for span in range(2, loops.max_size):
        inputs = [tables, wp.int32(span), dp, prev]
        dim = (loops.n_loops, loops.max_size - span)
        if tiled:
            wp.launch_tiled(
                kernel_holes.fill_dp_span_tiled,
                dim=dim,
                inputs=inputs,
                block_dim=block,
                device=device,
            )
        else:
            wp.launch(kernel_holes.fill_dp_span, dim=dim, inputs=inputs, device=device)


def _pack_loops(loops: list[wp.array[wp.int32]]) -> _PackedLoops:
    """Concatenate a caller's per-loop arrays into the packed form the fill engine consumes."""
    # ``copy=False``: every kernel below takes ``flat_loops`` as an input and none writes it.
    flat_loops, _offsets = tw.array.pack_1d_arrays(loops, copy=False)
    sizes_np = np.asarray([int(loop.shape[0]) for loop in loops], dtype=np.int64)
    return _PackedLoops(flat_loops, sizes_np)


def fill_small(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_perimeter: float | None = None,
    *,
    max_edges: int | None = None,
) -> wp.array[wp.int32]:
    """
    Fill only the *small* boundary loops, sized either by perimeter or by boundary-edge count.

    Intended open boundaries (large loops) are left untouched; spurious small holes are sealed by
    the shared min-weight interval DP
    ([`fill_loops_min_weight`][triwarp.holes.fill_loops_min_weight]). Used by
    [`triwarp.reconstruction.triangulate_point_cloud`][triwarp.reconstruction.triangulate_point_cloud]
    to seal small gaps left by sparse or non-uniform point-cloud sampling.

    Exactly one of the two thresholds is given, and they are genuinely different questions rather
    than two spellings of one:

    - ``max_perimeter`` is a **length**, so it is scale-dependent and sampling-independent -- the
      right threshold when "small" means *small on the object*.
    - ``max_edges`` is a **count**, so it is scale-independent and sampling-dependent -- the right
      threshold when "small" means *few triangles to patch*, which is what bounds the ``O(B^3)``
      interval DP behind the fill.

    Neither converts into the other without knowing the rim's sampling, and the reference
    implementations split the same way: two of the three take a boundary-edge count and only one
    takes a length. On a mesh whose rims are sampled unevenly the two thresholds select **opposite**
    loops -- ``tests/test_holes.py`` builds exactly that case, a 16-vertex rim of perimeter 2.41
    beside a 5-vertex rim of perimeter 3.16.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_perimeter
        Only boundary loops with perimeter at most this value are filled. Positional for backward
        compatibility. Mutually exclusive with ``max_edges``.
    max_edges
        Only boundary loops with at most this many boundary edges are filled -- **inclusive**, so a
        24-edge rim is filled at ``max_edges=24`` and left open at 23. A closed loop has as many
        edges as vertices, so this is also its vertex count. Mutually exclusive with
        ``max_perimeter``.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of ``faces`` followed by the fill triangles for the small loops, on
        ``faces.device``. A copy of ``faces`` when no boundary loop meets the threshold.

    Raises
    ------
    ValueError
        If neither ``max_perimeter`` nor ``max_edges`` is given, or if both are.

    Notes
    -----
    ``max_edges`` costs nothing to evaluate: the loop sizes are already host-side metadata of the
    packed loops, so that branch skips the segmented perimeter launch and its readback entirely.

    The inclusive bound is stated because it is easy to get wrong from the outside -- one reference
    whose parameter this matches documents itself as "less than" and is measurably "at most".

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    [`fill_loops_min_weight`][triwarp.holes.fill_loops_min_weight]
    [`fillable_loop_mask`][triwarp.holes.fillable_loop_mask]
        Which loops the DP can fill at all, a separate question from which are small.
    """
    if (max_perimeter is None) == (max_edges is None):
        raise ValueError("pass exactly one of max_perimeter or max_edges")

    packed = _hole_loops(vertices, faces)
    if packed is None:
        return wp.clone(faces)

    if max_edges is not None:
        small_np = packed.sizes_np <= max_edges
    else:
        # One segmented-sum launch measures every rim, so the selection costs a single readback
        # rather than a copy of the whole vertex buffer plus one of each loop.
        small_np = packed.perimeters(vertices) <= max_perimeter
    if not small_np.any():
        return wp.clone(faces)
    if not small_np.all():
        kept = zip(_unpack_loops(packed), small_np, strict=True)
        packed = _pack_loops([loop for loop, keep in kept if keep])
    return _fill_packed_loops(vertices, faces, packed, "plane_normalized", True, True)


def _unpack_loops(loops: _PackedLoops) -> list[wp.array[wp.int32]]:
    """Per-loop **views** into the packed buffer, for the callers that still want a list."""
    return [loops.flat_loops[loops.loop_slice(index)] for index in range(loops.n_loops)]


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
    refine: Literal["max_edge", "density"] = "max_edge",
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
        length of the holes being filled; a sequential budget target has no parallel analogue.
    max_edge_splits
        Soft cap on the number of edge splits during subdivision.
    max_angle_change_after_flip
        Dihedral-angle-change gate for the Delaunay flip pass (default 30°).
    smooth_curvature
        When ``True`` (default), smooth the new patch vertices after subdivision.
    natural_smooth
        When ``True``, additionally grow a collar around the patch and smooth it so the patch
        blends into the surrounding surface.
    edge_weights
        Laplacian edge weights for the cross-boundary smooth solve: ``"cotan"`` (default) or
        ``"unit"``.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop open.
    resolve_multiple_edges
        When ``True`` (default), forbid fill chords that duplicate existing mesh edges.
    smooth_boundary
        When ``True`` (default), also run the cross-boundary smooth solve so the patch is C¹ across
        its rim. Also tunes the fill metric's rim edge terms.
    refine
        Which subdivision criterion the refinement stage uses. ``"max_edge"`` (default) bisects
        patch edges longer than ``max_edge``; ``"density"`` splits patch triangles at their centroid
        while their sampling is coarser than the surrounding mesh's, which is Liepa's criterion and
        what the reference hole fillers do. The two differ only on a **graded** neighbourhood, where
        a single target length cannot be right at both ends of the grading: ``"density"`` keeps the
        patch triangle size closer to the local surrounding scale there than a single ``max_edge``
        target can. ``max_edge`` and ``max_edge_splits`` are ignored under ``"density"``.
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
        If ``metric``, ``edge_weights`` or ``refine`` is unknown.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    [`stitch_smooth`][triwarp.holes.stitch_smooth]
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]

    Notes
    -----
    The three stages are: minimum-weight fill, refine the patch to the target edge length, then
    smooth its new interior vertices into the surrounding surface.

    Winding is consistent with the surrounding faces only for a consistently wound input.
    """
    if metric not in _METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}")
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")
    _check_refine(refine)

    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        empty = _patch_mask(int(faces.shape[0]) // 3, int(faces.shape[0]) // 3, device)
        result = (wp.clone(vertices), wp.clone(faces))
        return (*result, empty) if return_patch else result

    n_faces_before = n_faces
    n_vertices_before = int(vertices.shape[0])
    faces_filled = _fill_packed_loops(
        vertices, faces, packed, metric, resolve_multiple_edges, smooth_boundary
    )
    n_faces_after = int(faces_filled.shape[0]) // 3
    patch_mask = _patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (wp.clone(vertices), faces_filled)
        return (*result, patch_mask) if return_patch else result

    target_edge = max_edge if max_edge is not None else _mean_rim_edge_length(vertices, packed)
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
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


def refill_region(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    metric: str = "plane_normalized",
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    smooth_curvature: bool = True,
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Replace a face region with a fresh patch: delete it, then fill the hole nicely.

    [`fill_smooth`][triwarp.holes.fill_smooth] applied to a region rather than to the mesh's own
    holes -- the operation for *rebuilding* a piece of surface that is wrong rather than missing: a
    self-intersecting band, a noisy patch, a set of faces a user painted. The region is removed, the
    rims that opens are triangulated by the same minimum-weight DP, and the patch is refined and
    smoothed to match its surroundings.

    Only the rims the deletion **opened** are filled. An input that already had a boundary keeps it,
    which is what makes this usable on an open mesh at all --
    [`triwarp.selection.delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
    is where that distinction is drawn.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` ``wp.bool`` array, ``True`` for each face to **replace**.
    metric
        Minimum-weight fill metric, as in [`fill_min_weight`][triwarp.holes.fill_min_weight].
    triangulate_only
        Stop after the minimum-weight triangulation, skipping the refinement and smoothing. The
        patch is then ``n - 2`` triangles per rim of ``n`` vertices and adds no new vertices.
    max_edge
        Target edge length for the refinement. ``None`` uses the mean rim edge length, so the
        patch arrives at roughly the surrounding mesh's resolution.
    smooth_curvature
        Minimize curvature rather than area when smoothing the patch, as in
        [`fill_smooth`][triwarp.holes.fill_smooth].
    refine
        Subdivision criterion for the refinement stage; see
        [`fill_smooth`][triwarp.holes.fill_smooth].
    return_patch
        Also return the per-face mask of the new patch, for a caller that wants to keep working on
        it.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[..., wp.array[wp.bool]]
        ``(vertices, faces)`` of the rebuilt mesh, plus the patch mask when ``return_patch`` is
        ``True``. When the mask selects nothing, or selects faces whose removal opens no new rim,
        the surviving mesh is returned with an all-``False`` patch.

    Raises
    ------
    ValueError
        If ``metric`` or ``refine`` is not one of the supported names, or ``face_mask`` does not
        have one entry per face.

    Examples
    --------
    ```python
    bad = tw.validation.face_self_intersecting_mask(v, f)
    patched_v, patched_f = tw.holes.refill_region(v, f, bad)
    ```

    See Also
    --------
    [`fill_smooth`][triwarp.holes.fill_smooth]
        The same pipeline over the mesh's existing holes, when nothing needs deleting first.
    [`triwarp.selection.delete_region_keep_boundary`][triwarp.selection.delete_region_keep_boundary]
        The first half, when the rims are wanted rather than filled.
    [`triwarp.repair.remove_folded_faces`][triwarp.repair.remove_folded_faces]
        One of several ways to produce the mask this takes.
    """
    if metric not in _METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_METRIC_IDS)}, got {metric!r}")
    _check_refine(refine)

    kept_vertices, kept_faces, new_loops = tw.selection.delete_region_keep_boundary(
        vertices, faces, face_mask
    )
    device = faces.device
    n_kept_faces = int(kept_faces.shape[0]) // 3
    if not new_loops:
        empty = _patch_mask(n_kept_faces, n_kept_faces, device)
        result = (kept_vertices, kept_faces)
        return (*result, empty) if return_patch else result

    packed = _pack_loops(new_loops)
    faces_filled = _fill_packed_loops(kept_vertices, kept_faces, packed, metric, True, True)
    patch_mask = _patch_mask(n_kept_faces, int(faces_filled.shape[0]) // 3, device)
    if triangulate_only:
        result = (kept_vertices, faces_filled)
        return (*result, patch_mask) if return_patch else result

    target_edge = max_edge
    if target_edge is None:
        target_edge = _mean_rim_edge_length(kept_vertices, packed)
    new_vertices, new_faces, out_patch = tw.smoothing.refine_and_smooth_region(
        kept_vertices,
        faces_filled,
        int(kept_vertices.shape[0]),
        patch_mask,
        target_edge,
        1000,
        math.radians(30.0),
        smooth_curvature,
        True,
        False,
        "cotan",
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


def extend_hole(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extend every boundary rim out to a plane, adding the ruled surface between them.

    Each rim vertex is projected orthogonally onto the plane, and the rim is bridged to that ring of
    projections with two triangles per rim edge. The result is open again -- the new rim lies
    *in* the plane -- which is what makes this the step before a flat cap rather than a cap itself:
    the extended rim is planar, so
    [`fill_min_weight`][triwarp.holes.fill_min_weight] closes it with a triangulation that has no
    reason to fold.

    The typical use is turning a scanned shell into a solid: extend its rims to a base plane, then
    fill. Doing that in one step with a min-weight fill instead gives a curved cap over the original
    rim, which is a different shape and usually not the wanted one.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    plane_normal
        The plane's normal. Need not be unit length in principle, but **is assumed to be** -- the
        projection scales with it otherwise. Normalize it.
    plane_origin
        A point on the target plane.
    loops
        Rims to extend. ``None`` extends every one, via
        [`boundary_loops`][triwarp.boundary.boundary_loops]; pass a subset to extend only those.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        The input positions, unchanged and in order, with one projected vertex appended per rim
        vertex.
    faces : wp.array[wp.int32]
        The input faces with the bridge triangles appended -- two per rim edge.

    Raises
    ------
    ValueError
        If any loop is not a rank-1 ``wp.int32`` array.

    !!! note "The rim may cross the plane"
        Nothing here checks which side of the plane a rim vertex is on. A rim straddling it gets an
        extension that folds through the plane, which is geometrically what "project each vertex"
        means and is almost never wanted -- place the plane clear of the rim, and check with
        [`bounds.aabb`][triwarp.bounds.aabb] if the input is not yours.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        Closes the planar rim this leaves.
    [`build_bottom`][triwarp.holes.build_bottom]
        The same extension with the plane fitted to each rim, instead of placed by the caller.
    [`stitch_loops`][triwarp.holes.stitch_loops]
        Bridges two rims that both already exist, where this generates the second one.
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    packed = _packed_rims(vertices, faces, loops)
    if packed is None:
        return wp.clone(vertices), wp.clone(faces)
    origins = wp.full(packed.n_loops, plane_origin, dtype=wp.vec3, device=faces.device)
    return _extend_packed_rims(vertices, faces, packed, plane_normal, origins)


def build_bottom(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    direction: wp.vec3,
    hole_extension: float = 0.0,
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extend every boundary rim down to a flat base placed under its own lowest vertex.

    This is [`extend_hole`][triwarp.holes.extend_hole] with the plane chosen for you: for each rim
    separately, the plane with normal ``direction`` through that rim's most extreme vertex in the
    ``-direction`` sense, pushed a further ``hole_extension`` past it. So the base sits flush with
    the lowest point of the rim and nothing folds -- which is what makes this the safe form when the
    rim is not level and a hand-placed plane would cut through it.

    Each rim gets its **own** plane. A mesh with two rims at different heights therefore gets two
    bases, not one shared one; pass a single-element ``loops`` to bottom just one of them.

    The result is still open -- the base is a rim in the plane, not a cap. Follow with
    [`fill_min_weight`][triwarp.holes.fill_min_weight] to close it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    direction
        The "up" direction the base is placed against: the plane's normal, and the rim is extended
        towards ``-direction``. Assumed unit length -- ``hole_extension`` is measured in its units
        and the projection scales with it otherwise.
    hole_extension
        Extra distance past the extreme vertex, along ``-direction``. ``0.0`` puts the base exactly
        through the lowest rim vertex, which leaves that vertex unmoved.
    loops
        Rims to extend. ``None`` extends every one, via
        [`boundary_loops`][triwarp.boundary.boundary_loops].

    Returns
    -------
    vertices : wp.array[wp.vec3]
        The input positions, unchanged and in order, with one projected vertex appended per rim
        vertex.
    faces : wp.array[wp.int32]
        The input faces with the bridge triangles appended -- two per rim edge.

    Raises
    ------
    ValueError
        If any loop is not a rank-1 ``wp.int32`` array.

    See Also
    --------
    [`extend_hole`][triwarp.holes.extend_hole]
        The general form, where the plane is yours to place.
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        Closes the flat rim this leaves.
    """
    device = faces.device
    packed = _packed_rims(vertices, faces, loops)
    if packed is None:
        return wp.clone(vertices), wp.clone(faces)

    extremes = wp.full(packed.n_loops, wp.float32(math.inf), dtype=wp.float32, device=device)
    wp.launch(
        kernel_holes.loop_extreme_projection,
        dim=int(packed.indices.shape[0]),
        inputs=[vertices, packed.indices, packed.loop_id, direction, extremes],
        device=device,
    )
    origins = wp.empty(packed.n_loops, dtype=wp.vec3, device=device)
    wp.map(
        kernel_holes.plane_origin_from_extreme,
        extremes,
        direction,
        wp.float32(hole_extension),
        out=origins,
    )
    return _extend_packed_rims(vertices, faces, packed, direction, origins)


class _PackedRims(NamedTuple):
    """One flat buffer of rim vertices, with the per-loop bookkeeping the extension kernels read."""

    indices: wp.array[wp.int32]
    loop_id: wp.array[wp.int32]
    starts: wp.array[wp.int32]
    sizes: wp.array[wp.int32]
    n_loops: int


def _packed_rims(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]] | None,
) -> _PackedRims | None:
    """Pack the rims to extend, or ``None`` when there is nothing to extend."""
    device = faces.device
    if loops is None:
        loops = tw.boundary.boundary_loops(vertices, faces)
    loops = list(loops)
    for loop in loops:
        if len(loop.shape) != 1 or loop.dtype is not wp.int32:
            raise ValueError("every loop must be a rank-1 wp.int32 array of vertex indices")
    if not loops or all(int(loop.shape[0]) == 0 for loop in loops):
        return None

    # ``copy=False``: the extension kernels only read the rim indices, so loops that came from
    # ``boundary_loops`` are re-used in place rather than re-packed rim by rim.
    packed, starts = tw.array.pack_1d_arrays(loops, copy=False)
    sizes_np = np.array([int(loop.shape[0]) for loop in loops], dtype=np.int32)
    loop_id = wp.array(
        np.repeat(np.arange(len(loops), dtype=np.int32), sizes_np), dtype=wp.int32, device=device
    )
    sizes = wp.array(sizes_np, dtype=wp.int32, device=device)
    return _PackedRims(packed, loop_id, starts, sizes, len(loops))


def _extend_packed_rims(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    rims: _PackedRims,
    plane_normal: wp.vec3,
    plane_origins: wp.array[wp.vec3],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Project every packed rim vertex onto its loop's plane and bridge the two rings."""
    device = faces.device
    total = int(rims.indices.shape[0])
    n_vertices = int(vertices.shape[0])
    extended_vertices = wp.empty(n_vertices + total, dtype=wp.vec3, device=device)
    wp.copy(extended_vertices[:n_vertices], vertices)
    wp.launch(
        kernel_holes.project_loop_to_plane,
        dim=total,
        inputs=[
            vertices,
            rims.indices,
            rims.loop_id,
            plane_normal,
            plane_origins,
            extended_vertices[n_vertices:],
        ],
        device=device,
    )

    n_faces = int(faces.shape[0]) // 3
    extended_faces = wp.empty(3 * (n_faces + 2 * total), dtype=wp.int32, device=device)
    wp.copy(extended_faces[: 3 * n_faces], faces)
    wp.launch(
        kernel_holes.bridge_loop_to_ring,
        dim=total,
        inputs=[
            rims.indices,
            rims.loop_id,
            rims.starts,
            rims.sizes,
            wp.int32(n_vertices),
            extended_faces[3 * n_faces :].reshape((2 * total, 3)),
        ],
        device=device,
    )
    return extended_vertices, extended_faces


def fillable_loop_mask(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    loops: Sequence[wp.array[wp.int32]] | None = None,
) -> wp.array[wp.bool]:
    """
    Which boundary loops a min-weight fill can close without producing an invalid mesh.

    [`fill_min_weight`][triwarp.holes.fill_min_weight] triangulates a rim over its **own** vertices,
    which fails in two combinatorial ways that have nothing to do with the metric. This names them
    per loop, so the policy is the caller's rather than a flag's:

    * **A repeated vertex.** A rim that visits one vertex twice is pinched there, and any
      triangulation of it folds through that pinch. A vertex shared by two *different* rims is the
      same pinch spread across two loops, and disqualifies both.
    * **A chord.** Two loop vertices that are *not* neighbours along the rim but are already joined
      by a mesh edge. The fill may propose that pair as a fill edge, and the mesh already has one,
      so that edge ends up with three faces. This is what
      ``fill_min_weight(resolve_multiple_edges=True)`` repairs *after* the fact, and a ``True`` here
      is exactly the case where that repair has nothing to do.

    !!! note "Conservative on the chord, and not the reference predicate"
        The chord test is **sufficient, not necessary**: a chord-free loop cannot produce a
        duplicated edge whatever the dynamic program chooses, but a loop *with* a chord may still
        fill cleanly if the program happens to avoid it. A four-vertex rim with one diagonal already
        present is the smallest example -- there are two triangulations and only one of them
        collides. So read ``False`` as "check the result", not as "cannot be filled".

        It is also **not** a port of the reference predicate that flags faces *complicating* a
        hole, which measures something else: on that same square it flags **no** face. And the
        reference's third predicate, whether a loop is the "outer" one, is deliberately absent --
        which boundary of an open surface is outer is a property of an embedding, not of the mesh,
        so there is nothing intrinsic to compute. The practical form of that question, *which rim
        is the big one*, is already
        answered by ``fill_min_weight(preserve_largest_hole=True)`` and by
        [`loop_perimeters`][triwarp.boundary.loop_perimeters].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    loops
        The boundary loops to test. When ``None`` they are computed with
        [`boundary_loops`][triwarp.boundary.boundary_loops]; pass them when you already have them,
        since the returned mask is indexed by their order and a caller almost always needs both.

    Returns
    -------
    wp.array[wp.bool]
        One entry per loop, ``True`` where the loop is simple, shares no vertex with another loop,
        and is chord-free, on ``faces.device``. ``True`` is a guarantee; ``False`` is a warning,
        per the note above.

    Raises
    ------
    ValueError
        If any loop is not a rank-1 ``wp.int32`` array.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        The fill this predicts the safety of.
    [`boundary_loops`][triwarp.boundary.boundary_loops]
        Produces the loops, in the order this mask indexes.
    [`loop_perimeters`][triwarp.boundary.loop_perimeters]
        Ranks the same loops by size, which is the other question a caller asks about a rim.
    """
    device = faces.device
    if loops is None:
        loops = tw.boundary.boundary_loops(vertices, faces)
    loops = list(loops)
    for loop in loops:
        if len(loop.shape) != 1 or loop.dtype is not wp.int32:
            raise ValueError("every loop must be a rank-1 wp.int32 array of vertex indices")
    if not loops:
        return wp.empty(0, dtype=wp.bool, device=device)

    # ``vertices`` is the domain, so its length is the bound ``index_bound`` would go to the
    # device to re-derive. An unreferenced vertex only widens the two tables below, which are
    # indexed by vertex id.
    n_vertices = int(vertices.shape[0])
    # **One** readback for all the loops, not one each: they are concatenated on the device first,
    # and the sizes are already on the host. A scan mesh carries dozens of rims, so a per-loop
    # readback would cost a sync apiece.
    #
    # A readback at all because the position table cannot be built on the device: a pinched loop
    # wants two entries for one vertex, and the pinch is what disqualifies it. Its size is bounded
    # by the *boundary* rather than by the mesh.
    sizes = [int(loop.shape[0]) for loop in loops]
    flat_np = tw.array.concatenate(list(loops), copy=False).numpy()
    bounds = np.cumsum([0, *sizes])
    loops_np = [flat_np[bounds[index] : bounds[index + 1]] for index in range(len(sizes))]
    fillable_np = np.ones(len(loops_np), dtype=bool)
    for index, loop_np in enumerate(loops_np):
        if np.unique(loop_np).shape[0] != loop_np.shape[0]:
            fillable_np[index] = False  # pinched within itself, and would corrupt the table below

    # A vertex on two different rims is a pinch *between* loops: filling either one leaves that
    # vertex non-manifold. It also cannot be represented in the two vertex-indexed tables below,
    # which hold one owner apiece -- so writing the loops in order would let the later loop
    # silently overwrite the earlier one's ownership and hide the earlier one's chords, returning
    # ``True`` for a loop that has one. ``True`` is a guarantee, so every loop touching a shared
    # vertex answers ``False`` and is kept out of the tables. Counting owners up front rather
    # than resolving collisions in the write loop keeps the answer independent of loop order.
    owned = [loops_np[index] for index in range(len(loops_np)) if fillable_np[index]]
    if owned:
        owner_count = np.bincount(np.concatenate(owned), minlength=n_vertices)
        for index, loop_np in enumerate(loops_np):
            if fillable_np[index] and (owner_count[loop_np] > 1).any():
                fillable_np[index] = False

    loop_of_vertex_np = np.full(n_vertices, -1, dtype=np.int32)
    position_np = np.zeros(n_vertices, dtype=np.int32)
    for index, loop_np in enumerate(loops_np):
        if fillable_np[index]:
            loop_of_vertex_np[loop_np] = index
            position_np[loop_np] = np.arange(loop_np.shape[0], dtype=np.int32)

    if fillable_np.any():
        unique_edges, _inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
        has_chord = wp.zeros(len(loops_np), dtype=wp.bool, device=device)
        wp.launch(
            kernel_holes.mark_loops_with_chords,
            dim=int(unique_edges.shape[0]),
            inputs=[
                unique_edges,
                wp.array(loop_of_vertex_np, dtype=wp.int32, device=device),
                wp.array(position_np, dtype=wp.int32, device=device),
                wp.array(np.array(sizes, dtype=np.int32), dtype=wp.int32, device=device),
                has_chord,
            ],
            device=device,
        )
        fillable_np &= ~has_chord.numpy()
    return wp.array(fillable_np, dtype=wp.bool, device=device)


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
    [`stitch_loops`][triwarp.holes.stitch_loops]. Each mesh must have
    exactly one boundary loop (the promesh assumption); use ``stitch_loops`` directly to
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
    [`stitch_loops`][triwarp.holes.stitch_loops]
    [`stitch_min_weight`][triwarp.holes.stitch_min_weight]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch"
    )
    return stitch_loops(vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b)


def stitch_min_weight(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes with a minimum-weight band.

    Like [`stitch`][triwarp.holes.stitch] but joins the two rims with the metric-minimizing band
    of [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight]
    instead of the greedy correspondence. Each mesh must have exactly one boundary loop.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight].
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
    [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight]
    [`stitch`][triwarp.holes.stitch]

    Notes
    -----
    The band is the minimum-weight two-loop stitch; see
    [`stitch_min_weight`][triwarp.holes.stitch_min_weight] for the metrics.
    """
    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_min_weight"
    )
    return stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )


def stitch_smooth(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    metric: str = "complex_stitch",
    up_dir: tuple[float, float, float] | None = None,
    *,
    triangulate_only: bool = False,
    max_edge: float | None = None,
    max_edge_splits: int = 1000,
    max_angle_change_after_flip: float = math.radians(30.0),
    smooth_curvature: bool = True,
    natural_smooth: bool = False,
    edge_weights: str = "cotan",
    refine: Literal["max_edge", "density"] = "max_edge",
    return_patch: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]
):
    """
    Stitch two open meshes with a smooth, refined band.

    Like [`stitch_min_weight`][triwarp.holes.stitch_min_weight] but the connecting band is then
    subdivided and smoothed by the same finisher as
    [`fill_smooth`][triwarp.holes.fill_smooth]. Each mesh must have exactly one
    boundary loop. The cross-boundary smooth solve is always applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    metric
        Stitch metric; see
        [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight].
    up_dir
        Up direction for the ``"vertical"`` stitch metric.
    triangulate_only
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_edge
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_edge_splits
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    max_angle_change_after_flip
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    smooth_curvature
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    natural_smooth
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    edge_weights
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    refine
        See [`fill_smooth`][triwarp.holes.fill_smooth].
    return_patch
        See [`fill_smooth`][triwarp.holes.fill_smooth].

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices followed by any inserted band vertices, on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the band faces.
    patch_mask : wp.array[wp.bool]
        Only when ``return_patch`` is ``True``: mask of the band faces.

    Raises
    ------
    ValueError
        If either mesh lacks exactly one boundary loop, or ``metric`` / ``edge_weights`` /
        ``refine`` is unknown.

    See Also
    --------
    [`stitch_min_weight`][triwarp.holes.stitch_min_weight]
    [`fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    The three stages are: minimum-weight stitch, refine the band to the target edge length, then
    smooth its new interior vertices into both surrounding surfaces.
    """
    if metric not in _STITCH_METRIC_IDS:
        raise ValueError(f"metric must be one of {sorted(_STITCH_METRIC_IDS)}, got {metric!r}")
    if edge_weights not in ("cotan", "unit"):
        raise ValueError(f"edge_weights must be 'cotan' or 'unit', got {edge_weights!r}")
    _check_refine(refine)

    loop_a, loop_b = _single_boundary_loops(
        vertices_a, faces_a, vertices_b, faces_b, caller="stitch_smooth"
    )

    device = faces_a.device
    n_faces_before = (int(faces_a.shape[0]) + int(faces_b.shape[0])) // 3
    n_vertices_before = int(vertices_a.shape[0]) + int(vertices_b.shape[0])
    combined_vertices, combined_faces = stitch_loops_min_weight(
        vertices_a, faces_a, loop_a, vertices_b, faces_b, loop_b, metric, up_dir
    )
    n_faces_after = int(combined_faces.shape[0]) // 3
    patch_mask = _patch_mask(n_faces_before, n_faces_after, device)

    if triangulate_only:
        result = (combined_vertices, combined_faces)
        return (*result, patch_mask) if return_patch else result

    # Independent copies, not views: ``boundary_loops`` returns slices of one shared packed buffer
    # and ``_mean_rim_edge_length`` must not alias it. ``wp.clone`` says exactly that; the round
    # trip through ``.numpy()`` these used to take said nothing and crossed the bus twice. The
    # second loop's shift into the combined numbering is elementwise, so it maps on the device.
    shifted_loop_b = wp.empty(int(loop_b.shape[0]), dtype=wp.int32, device=device)
    wp.map(wp.add, loop_b, wp.int32(int(vertices_a.shape[0])), out=shifted_loop_b)
    rim_loops = [wp.clone(loop_a), shifted_loop_b]
    target_edge = (
        max_edge if max_edge is not None else _mean_rim_edge_length(combined_vertices, rim_loops)
    )
    new_vertices, new_faces, out_patch = tw.smoothing.refine_and_smooth_region(
        combined_vertices,
        combined_faces,
        n_vertices_before,
        patch_mask,
        target_edge,
        max_edge_splits,
        max_angle_change_after_flip,
        smooth_curvature,
        True,  # the reference pipeline forces the cross-boundary smooth here
        natural_smooth,
        edge_weights,
        refine=refine,
    )
    return (new_vertices, new_faces, out_patch) if return_patch else (new_vertices, new_faces)


def _check_refine(refine: str) -> None:
    """
    Reject an unknown subdivision criterion, for the three ``*_smooth`` / ``refill`` entry points.

    [`smoothing.refine_and_smooth_region`][triwarp.smoothing.refine_and_smooth_region] raises on
    the same names, but each of these functions has paths that never reach it -- an empty mesh,
    ``triangulate_only``, a face mask selecting nothing -- so without this the documented
    ``Raises`` would hold on one path and not another, and a typo'd criterion would be silently
    accepted on the cheap ones.

    Raises
    ------
    ValueError
        If ``refine`` is neither ``"max_edge"`` nor ``"density"``.
    """
    if refine not in ("max_edge", "density"):
        raise ValueError(f"unknown refine {refine!r}, expected 'max_edge' or 'density'")


def _single_boundary_loops(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    caller: str,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return the one boundary loop of each mesh, or raise naming ``caller``.

    The shared precondition of the whole ``stitch*`` family: each mesh must have exactly one hole
    to zip to the other's. Loops shorter than three vertices are degenerate rather than holes and
    are dropped before counting, so a mesh carrying one is rejected on its real loop count.
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            f"{caller} requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return loops_a[0], loops_b[0]


def stitch_loops(
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
    ``len(loop_a) < len(loop_b)``), so the output is invariant to argument order **except when the
    two loops are the same length**: the swap is on a strict inequality, so equal-length rims keep
    the caller's order, and A and B are not interchangeable — A's edges are matched to B's
    vertices, not the reverse — so two equal-length rims can produce a different seam either way
    round. Both loops must
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
        Concatenation of ``vertices_a`` then ``vertices_b`` (larger loop first), on the device of
        the **larger** loop's mesh — which is the caller's ``faces_a`` unless the swap above fired.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the bridge triangles, on
        that same device.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices.

    See Also
    --------
    [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight]
        The more robust choice, and the one to reach for when seam quality matters: it minimizes a
        triangulation metric over the whole rim pair instead of correcting a greedy correspondence.
    [`stitch`][triwarp.holes.stitch]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`concatenate`][triwarp.combine.concatenate]

    Notes
    -----
    The correspondence and its monotonicity correction are inherently sequential and run on the
    host over the length-``len(loop_a)`` association array; the O(N·M) perimeter matrix, the
    reductions, and the triangle emission run in Warp kernels, and the perimeter matrix itself is
    never copied off the device.

    **This is the fast one, and that is the reason to keep it.** It is metric-free and makes one
    O(N·M) pass, where
    [`stitch_loops_min_weight`][triwarp.holes.stitch_loops_min_weight] fills the same size table
    with ``N + M`` sequential launches along the anti-diagonals, and that gap widens with rim size.
    The DP is never the cheaper option -- prefer it for the seam it produces, not for speed.
    """
    n = int(loop_a.shape[0])
    m = int(loop_b.shape[0])
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n
    # Read *after* the swap: every allocation and launch below is on the A mesh's device, and the
    # swap is what decides which of the caller's two meshes that is.
    device = faces_a.device
    if m < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n} and {m}")
    n_vertices_a = int(vertices_a.shape[0])

    # Reverse loop A so both rims wind the same way, then take rim positions.
    flipped_loop_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.cyclic_gather,
        dim=n,
        inputs=[loop_a, wp.int32(n), wp.int32(0), wp.bool(True), wp.int32(0), flipped_loop_a],
        device=device,
    )
    a_pos = tw.array.gather(vertices_a, flipped_loop_a)
    b_pos = tw.array.gather(vertices_b, loop_b)

    # perimeters[i, j] = |a_i - b_j| + |a_{i+1} - b_j| for A-edge i and B-vertex j.
    perimeters = twt.empty_2d((n, m), wp.float32, device=device)
    wp.launch(
        kernel_holes.boundary_perimeters,
        dim=(n, m),
        inputs=[a_pos, b_pos, wp.int32(n), perimeters],
        device=device,
    )

    col_min = wp.empty(n, dtype=wp.int32, device=device)
    val_min = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_holes.row_argmin,
        dim=n,
        inputs=[perimeters, wp.int32(m), col_min, val_min],
        device=device,
    )

    shift = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    shift_np = shift.numpy()
    shift_a = int(shift_np[0])
    shift_b = int(shift_np[1])

    edge_dev = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.rolled_edge_map,
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
            kernel_holes.resolve_corrections,
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
        kernel_holes.cyclic_gather,
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
        kernel_holes.cyclic_gather,
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
        kernel_holes.bridge_a_faces,
        dim=n,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), bridge_a],
        device=device,
    )
    bridge_b = wp.empty(3 * m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.bridge_b_faces,
        dim=m,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), wp.int32(m), bridge_b],
        device=device,
    )

    combined_vertices, combined_faces = tw.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, tw.array.concatenate([combined_faces, bridge_a, bridge_b])


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


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][triwarp.holes._non_increasing_indices]); the algorithm does
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


# Stitch-metric name -> kernel selector (must match the METRIC_*_STITCH constants in
# kernels/holes.py).
_STITCH_METRIC_IDS = {"complex_stitch": 0, "edge_length_stitch": 1, "vertical": 2}


def stitch_loops_min_weight(
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

    The two rims are aligned at their closest vertex pair and zippered by the band of
    ``len(loop_a) + len(loop_b)`` triangles that minimizes a stitch metric, found by a grid dynamic
    program over the two loops (``dp[i, j]`` = best band consuming ``i`` A-edges and ``j`` B-edges;
    each anti-diagonal is one parallel kernel launch). Reuses only the
    loops' existing vertices. The metric-free greedy
    [`stitch_loops`][triwarp.holes.stitch_loops] remains available.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each.
    metric
        Stitch metric to minimize:

        - ``"complex_stitch"`` (default) — triangle aspect ratio plus a dihedral-smoothness edge
          term between adjacent band triangles and the surface.
        - ``"edge_length_stitch"`` — summed connection-edge length.
        - ``"vertical"`` — penalizes band area and normal deviation from ``up_dir``; pass
          ``up_dir``.
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
    [`stitch_loops`][triwarp.holes.stitch_loops]
    [`stitch_min_weight`][triwarp.holes.stitch_min_weight]
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
    # search, rim gathers and rim-opposite lookups all run on device. This preamble (two readbacks,
    # the closest-pair gather, the host roll and two more uploads) is a fixed cost that the
    # anti-diagonal DP below dominates and outgrows as the rims lengthen.
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

    dp = twt.as_array2d(
        wp.full((n_a + 1, n_b + 1), _BAD_TRIANGULATION_METRIC, dtype=wp.float32, device=device),
        wp.float32,
    )
    wp.launch(kernel_holes.set_dp_origin, dim=1, inputs=[dp], device=device)
    came = twt.as_array2d(wp.full((n_a + 1, n_b + 1), -1, dtype=wp.int32, device=device), wp.int32)
    # Built once, outside the loop: the ten values below are the same on every one of the
    # ``n_a + n_b`` launches, and a wp.launch argument costs host time linearly on both devices, so
    # bundling them once here rather than passing all ten on every launch is a real saving.
    tables = kernel_holes.StitchTables()
    tables.a_pos = a_pos
    tables.b_pos = b_pos
    tables.a_opp = a_opp
    tables.a_opp_valid = a_opp_valid
    tables.b_opp = b_opp
    tables.b_opp_valid = b_opp_valid
    tables.up = up
    tables.metric_id = wp.int32(metric_id)
    tables.n_a = wp.int32(n_a)
    tables.n_b = wp.int32(n_b)
    for diag in range(1, n_a + n_b + 1):
        wp.launch(
            kernel_holes.stitch_dp_diag,
            dim=min(diag, n_a) - max(0, diag - n_b) + 1,
            inputs=[tables, wp.int32(diag), dp, came],
            device=device,
        )
    came_np = came.numpy()

    band = _stitch_band_triangles(came_np, la, lb, n_a, n_b, offset)
    combined_vertices, combined_faces = tw.combine.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    band_faces = wp.array(band.reshape(-1), dtype=wp.int32, device=device)
    return combined_vertices, tw.array.concatenate([combined_faces, band_faces])


def _closest_loop_pair(a_pos: wp.array[wp.vec3], b_pos: wp.array[wp.vec3]) -> tuple[int, int]:
    """
    Return the closest vertex pair ``(i, j)`` between the two rims: the band's start pair.

    The full ``(n_a, n_b)`` squared-distance matrix and its argmin run in Warp kernels (the same
    ``row_argmin`` / ``global_argmin`` reduction the greedy zippering uses); only the two winning
    indices come back to the host. Ties resolve to the smallest ``i`` then smallest ``j``, matching
    ``numpy.argmin`` on the flattened matrix.

    ``kernels/holes.reduce_closest_cross_label_pair`` answers the same question in **one** kernel
    with no matrix at all, by reducing a ``pack_nearest_key`` atomic over labelled members, but the
    two are not one function wearing two hats: that kernel takes members and labels over a shared
    vertex buffer, this takes two separate position arrays, and the three launches here reuse
    ``row_argmin`` / ``global_argmin``, which the caller needs anyway for its *perimeter* objective.
    """
    device = a_pos.device
    n_a = int(a_pos.shape[0])
    n_b = int(b_pos.shape[0])
    dist_sq = twt.empty_2d((n_a, n_b), wp.float32, device=device)
    wp.launch(
        kernel_holes.pair_sq_distances,
        dim=(n_a, n_b),
        inputs=[a_pos, b_pos, dist_sq],
        device=device,
    )
    col_min = wp.empty(n_a, dtype=wp.int32, device=device)
    val_min = wp.empty(n_a, dtype=wp.float32, device=device)
    wp.launch(
        kernel_holes.row_argmin,
        dim=n_a,
        inputs=[dist_sq, wp.int32(n_b), col_min, val_min],
        device=device,
    )
    pair = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n_a), pair],
        device=device,
    )
    pair_np = pair.numpy()
    return int(pair_np[0]), int(pair_np[1])


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


def bridge_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    validate: bool = True,
) -> wp.array[wp.int32]:
    """
    Join two boundary edges with a two-triangle patch, leaving the rest of both rims open.

    Where [`stitch_loops`][triwarp.holes.stitch_loops] consumes two *complete* rims, this is the
    local operation underneath it: pick one edge on each side and close only that gap. Two open
    rims become one, so a bridge is how a tube is joined to another tube at a chosen seam, how a
    partially torn boundary is tacked back together, and -- when both edges lie on the *same* rim --
    how a handle is added, since bridging one loop to itself splits it into two.

    The patch is the quadrilateral ``(a1, a0, b1, b0)`` split along the ``a0 - b0`` diagonal, so it
    adds **two triangles and no vertices**. When the two edges already share a vertex -- they are
    consecutive along one rim -- that quadrilateral is a triangle and only **one** is added.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Only read to validate the two edges; the patch
        itself is purely topological, and nothing moves.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    edge_a
        A boundary edge as ``(v0, v1)``, **directed the way its face winds it** -- a row of
        [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges].
    edge_b
        The second boundary edge, in the same direction convention.
    validate
        Check that both edges are boundary edges of ``faces`` and that the patch would not
        duplicate an existing edge. Costs one pass over the mesh edges and one readback; pass
        ``False`` when the edges came from
        [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges] and the pairing is
        known good.

    Returns
    -------
    wp.array[wp.int32]
        The input faces with the patch appended, on ``faces.device``. The appended block is the
        tail -- ``3`` entries longer than the input for the shared-vertex case, ``6`` otherwise --
        so its size is what says which case was taken.

    Raises
    ------
    ValueError
        If the two edges are the same edge, if ``validate`` is set and either is not a boundary
        edge of ``faces``, or if ``validate`` is set and the patch would create a second edge
        between a pair of vertices that already share one (which would leave the mesh
        non-manifold).

    Examples
    --------
    ```python
    edges_np = tw.boundary.oriented_boundary_edges(open_v, open_f).numpy()
    bridged_f = tw.holes.bridge_edges(
        open_v, open_f, tuple(edges_np[0]), tuple(edges_np[len(edges_np) // 2])
    )
    ```

    See Also
    --------
    [`bridge_edges_smooth`][triwarp.holes.bridge_edges_smooth]
        The multi-segment form, which curves the patch and adds vertices.
    [`join_closest_components`][triwarp.holes.join_closest_components]
        The driver over this, when the pair to bridge is not the caller's to name.
    [`stitch_loops`][triwarp.holes.stitch_loops]
        Joins two rims completely, where this joins one edge of each.
    [`oriented_boundary_edges`][triwarp.boundary.oriented_boundary_edges]
        Produces the edges this takes, already in the right direction.
    """
    edge_a = (int(edge_a[0]), int(edge_a[1]))
    edge_b = (int(edge_b[0]), int(edge_b[1]))
    _check_bridge_edges(
        vertices, faces, edge_a, edge_b, _bridge_joined_pairs(edge_a, edge_b), validate
    )
    triangles = _bridge_triangles(edge_a, edge_b)
    patch = wp.array(
        np.asarray(triangles, dtype=np.int32).reshape(-1), dtype=wp.int32, device=faces.device
    )
    return tw.array.concatenate([faces, patch])


def _bridge_joined_pairs(edge_a: tuple[int, int], edge_b: tuple[int, int]) -> list[tuple[int, int]]:
    """List the pairs of *existing* vertices the flat patch joins: its two sides and diagonal."""
    a0, a1 = edge_a
    b0, b1 = edge_b
    incident = {(min(a0, a1), max(a0, a1)), (min(b0, b1), max(b0, b1))}
    pairs = []
    for triangle in _bridge_triangles(edge_a, edge_b):
        for k in range(3):
            u, v = triangle[k], triangle[(k + 1) % 3]
            key = (min(u, v), max(u, v))
            if key not in incident and key not in pairs:
                pairs.append(key)
    return pairs


def bridge_edges_smooth(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    sampling_step: float,
    validate: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two boundary edges with a curved strip that leaves both surfaces smoothly.

    [`bridge_edges`][triwarp.holes.bridge_edges] spans the gap with a flat patch, which creases
    against both surfaces as soon as the two edges are more than a triangle apart. This spans it
    with a strip of quadrilaterals instead, following the cubic through the two edge midpoints
    whose end tangents lie **in** the two incident triangles -- so the strip leaves each surface in
    the direction that surface was already going, and the crease is spread over the whole strip
    rather than concentrated at its two ends.

    The strip is subdivided until its segments are no longer than ``sampling_step``, and its width
    tapers linearly from the length of ``edge_a`` to that of ``edge_b``. Its two boundary chains
    start at the two ends of ``edge_a`` and finish at the two ends of ``edge_b``, so the existing
    four vertices are reused and only the interior ones are new. As in the flat form, two edges
    that already share a vertex give a fan from that vertex rather than a strip.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    edge_a
        A boundary edge as ``(v0, v1)``, directed the way its face winds it.
    edge_b
        The second boundary edge, in the same direction convention.
    sampling_step
        Target segment length along the strip. Smaller means more segments and a smoother patch;
        a step at or above the span gives the single segment
        [`bridge_edges`][triwarp.holes.bridge_edges] would have produced, on the curve rather than
        on the chord.
    validate
        Check that both edges are boundary edges of ``faces`` and that the patch would not
        duplicate an existing edge, as in [`bridge_edges`][triwarp.holes.bridge_edges].

    Returns
    -------
    vertices : wp.array[wp.vec3]
        The input positions, unchanged and in order, with the strip's interior vertices appended.
    faces : wp.array[wp.int32]
        The input faces with the strip's triangles appended.

    Raises
    ------
    ValueError
        If ``sampling_step`` is not positive, if either edge is not a directed edge of ``faces``
        (checked whatever ``validate`` says, since the strip needs both incident faces), or for any
        of the reasons [`bridge_edges`][triwarp.holes.bridge_edges] raises.

    !!! note "The curve is a cubic, not an optimum"
        The strip follows one cubic Hermite segment fitted to the two edge midpoints and the two
        incident-face tangents. Nothing minimizes its bending energy or checks it for
        self-intersection, so a step far smaller than the span across two nearly opposed edges can
        fold; run [`fix_self_intersections`][triwarp.repair.fix_self_intersections] if the input
        pairing is not yours to choose.

    See Also
    --------
    [`bridge_edges`][triwarp.holes.bridge_edges]
        The single-quadrilateral form, which adds no vertices.
    [`fill_smooth`][triwarp.holes.fill_smooth]
        The same idea for a whole rim: a patch refined and faired rather than merely spanned.
    """
    if sampling_step <= 0.0:
        raise ValueError(f"sampling_step must be positive, got {sampling_step}")
    device = faces.device
    a0, a1 = int(edge_a[0]), int(edge_a[1])
    b0, b1 = int(edge_b[0]), int(edge_b[1])
    # A strip with interior samples joins nothing but its own new vertices, so it has no pair to
    # check; the one-segment case falls through to the flat patch below, which checks its own.
    _check_bridge_edges(vertices, faces, (a0, a1), (b0, b1), (), validate)

    # One launch to find each edge's opposite corner, then one gather of the six positions the
    # spline needs. Both are here so the host never reads back a buffer that scales with the mesh.
    query = wp.array(np.array([[a0, a1], [b0, b1]], dtype=np.int32), dtype=wp.int32, device=device)
    opposites = wp.full(2, wp.int32(-1), dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.directed_edge_opposites,
        dim=(int(faces.shape[0]) // 3, 2),
        inputs=[faces, query, opposites],
        device=device,
    )
    opposites_np = opposites.numpy()
    if int(opposites_np[0]) < 0 or int(opposites_np[1]) < 0:
        raise ValueError("both edges must be directed edges of faces, wound as their face winds")
    corners = np.array([a0, a1, b0, b1, int(opposites_np[0]), int(opposites_np[1])], dtype=np.int32)
    gathered = wp.empty(6, dtype=wp.vec3, device=device)
    wp.copy(gathered, vertices[wp.array(corners, dtype=wp.int32, device=device)])
    positions_np = gathered.numpy().astype(np.float64)

    n_vertices = int(vertices.shape[0])
    interior_np, strip_np = _bridge_strip(positions_np, corners, n_vertices, sampling_step)
    if interior_np.shape[0] == 0:
        # One segment: the strip *is* the flat patch, so hand it over with the caller's own
        # ``validate`` -- that path adds edges between existing vertices and has its own check.
        return wp.clone(vertices), bridge_edges(vertices, faces, (a0, a1), (b0, b1), validate)

    bridged_vertices = wp.empty(n_vertices + interior_np.shape[0], dtype=wp.vec3, device=device)
    wp.copy(bridged_vertices[:n_vertices], vertices)
    wp.copy(
        bridged_vertices[n_vertices:],
        wp.array(interior_np.astype(np.float32), dtype=wp.vec3, device=device),
    )
    strip = wp.array(strip_np.reshape(-1), dtype=wp.int32, device=device)
    return bridged_vertices, tw.array.concatenate([faces, strip])


def _bridge_strip(
    positions_np: np.ndarray, corners: np.ndarray, n_vertices: int, sampling_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample the bridge cubic and build the strip: interior positions, and the triangle rows.

    Host-side NumPy over six positions and a handful of samples -- the strip never scales with the
    mesh, so a kernel here would buy a launch and no parallelism.
    """
    a0, a1, b0, b1 = (int(corners[k]) for k in range(4))
    pa0, pa1, pb0, pb1, pta, ptb = positions_np

    center_a, center_b = 0.5 * (pa0 + pa1), 0.5 * (pb0 + pb1)
    # The tangent lies in the incident triangle's plane, runs across its edge and points into it, so
    # the spline's end control points sit inside the two surfaces and the curve leaves them
    # tangentially rather than at an angle.
    tangent_a = _unit(np.cross(_unit(np.cross(pa1 - pa0, pta - pa0)), _unit(pa1 - pa0)))
    tangent_b = _unit(np.cross(_unit(np.cross(pb1 - pb0, ptb - pb0)), _unit(pb1 - pb0)))
    # The cubic leaves A along -tangent_a and arrives at B along +tangent_b, so it continues each
    # surface rather than turning off it, and both velocities are scaled by the span -- a cubic
    # whose end velocities do not grow with the gap it crosses is a chord with a kink at each end.
    span = float(np.linalg.norm(center_b - center_a))
    velocity_0 = -span * tangent_a
    velocity_1 = span * tangent_b

    def hermite(u: np.ndarray) -> np.ndarray:
        column = u[:, None]
        squared, cubed = column * column, column * column * column
        return (
            (2.0 * cubed - 3.0 * squared + 1.0) * center_a
            + (cubed - 2.0 * squared + column) * velocity_0
            + (-2.0 * cubed + 3.0 * squared) * center_b
            + (cubed - squared) * velocity_1
        )

    dense = hermite(np.linspace(0.0, 1.0, 64))
    arc_length = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
    n_segments = max(1, math.ceil(arc_length / sampling_step))
    if n_segments == 1:
        rows = [t for t in ((a1, a0, b0), (a0, b1, b0)) if len(set(t)) == 3]
        return np.zeros((0, 3), dtype=np.float64), np.asarray(rows, dtype=np.int32).reshape(-1, 3)

    # The strip's two chains run a1 -> b0 and a0 -> b1, which are the quadrilateral's own two sides.
    # Their width tapers from one edge's length to the other's, so both ends land on the existing
    # four vertices exactly and only the interior samples are new.
    width_dir_a, width_dir_b = _unit(pa0 - pa1), _unit(pb1 - pb0)
    length_a = float(np.linalg.norm(pa1 - pa0))
    length_b = float(np.linalg.norm(pb1 - pb0))
    parameters = np.linspace(0.0, 1.0, n_segments + 1)
    samples = hermite(parameters)
    directions = (1.0 - parameters)[:, None] * width_dir_a + parameters[:, None] * width_dir_b
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    half_width = (0.5 * ((1.0 - parameters) * length_a + parameters * length_b))[:, None]
    minus_np = samples - half_width * directions
    plus_np = samples + half_width * directions

    # A shared vertex collapses that side of the strip onto the vertex itself, turning the strip
    # into a fan -- the multi-segment form of the flat patch's one-triangle case.
    interior: list[np.ndarray] = []
    minus_ids, plus_ids = [a1], [a0]
    for i in range(1, n_segments):
        for shared, ring, ids in ((a1 == b0, minus_np, minus_ids), (a0 == b1, plus_np, plus_ids)):
            if shared:
                ids.append(ids[0])
            else:
                ids.append(n_vertices + len(interior))
                interior.append(ring[i])
    minus_ids.append(b0)
    plus_ids.append(b1)

    rows = []
    for i in range(n_segments):
        for triangle in (
            (minus_ids[i], plus_ids[i], minus_ids[i + 1]),
            (plus_ids[i], plus_ids[i + 1], minus_ids[i + 1]),
        ):
            if len(set(triangle)) == 3:
                rows.append(triangle)
    return (
        np.asarray(interior, dtype=np.float64).reshape(-1, 3),
        np.asarray(rows, dtype=np.int32).reshape(-1, 3),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize, leaving a zero vector alone."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def join_closest_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_distance: float | None = None,
    max_joins: int | None = None,
) -> wp.array[wp.int32]:
    """
    Bridge the closest pairs of *open* components until the mesh is connected.

    The driver over [`bridge_edges`][triwarp.holes.bridge_edges], and the step that turns a repair
    pipeline's output into a **single** solid rather than several. Where ``bridge_edges`` joins two
    edges the caller names and [`stitch_loops`][triwarp.holes.stitch_loops] consumes two complete
    rims, this answers *"these are several open shells; make them one"* -- which no other entry
    point here does, because the question it has to settle first is *which* pieces to join.

    The rule is greedy nearest-link agglomeration, i.e. Kruskal over the components with the
    distance between their boundary vertices as the edge weight: take the globally closest pair of
    boundary vertices belonging to different components, bridge it, and repeat while a cross-
    component pair remains. Each join adds exactly **two triangles and no vertices** -- the two rims
    it touches become one open rim, left for a filler to close -- so the result grows by
    ``2 * (k - 1)`` faces for ``k`` open components.

    Ties are broken by the lower boundary-vertex index, so the answer does not depend on thread
    order.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Nothing moves and nothing is added, so this is only
        read.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_distance
        Refuse a join whose two boundary vertices are further apart than this. ``None`` (default)
        joins unconditionally, which is what the reference implementations do -- and which on a mesh
        holding genuinely separate objects welds them together, so pass a bound when the input might
        not be one broken surface.
    max_joins
        Stop after this many bridges. ``None`` (default) continues until one component remains or no
        admissible pair is left.

    Returns
    -------
    wp.array[wp.int32]
        The input face buffer with the bridge triangles appended, on ``faces.device``. A copy of
        ``faces`` when there is nothing to join -- fewer than two components, or no component with a
        boundary.

    Raises
    ------
    ValueError
        If ``max_joins`` is negative, or if a chosen pair admits no valid bridge at either of its
        two incident boundary edges.

    See Also
    --------
    [`bridge_edges`][triwarp.holes.bridge_edges]
        The primitive this drives, and where the two-triangle patch is defined.
    [`stitch_loops`][triwarp.holes.stitch_loops]
        Close two rims *completely* rather than tacking them together at one seam.
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        What to run afterwards: the merged rim is left open on purpose.
    [`repair.remove_small_components`][triwarp.repair.remove_small_components]
        The other answer to a multi-component mesh -- throw the extra pieces away instead.

    Notes
    -----
    The loop is host-sequential over the ``k - 1`` joins and recomputes the component labelling and
    the boundary edges from the updated face buffer each round, so the cost is ``O(k * n_faces)``.
    That is deliberate: ``k`` is the number of *open* components, which does not grow with the mesh,
    and recomputing makes the merge and the rim update fall out rather than needing a union-find
    and an incremental rim edit whose correctness would be much harder to see. The pairing itself is
    on the device.

    A component with no boundary -- a closed shell -- has nothing to bridge to and is left alone, so
    an input of closed shells comes back unchanged rather than raising.
    """
    if max_joins is not None and max_joins < 0:
        raise ValueError(f"max_joins must be non-negative, got {max_joins}")
    max_distance_sq = wp.float32(float("inf") if max_distance is None else float(max_distance) ** 2)

    current = faces
    joins = 0
    while max_joins is None or joins < max_joins:
        pair = _closest_cross_component_edges(vertices, current, max_distance_sq)
        if pair is None:
            break
        current = bridge_edges(vertices, current, pair[0], pair[1])
        joins += 1

    return wp.clone(faces) if joins == 0 else current


# ``pack_nearest_key`` is non-negative for any real candidate, so the largest ``int64`` is a seed no
# pair can reach -- which is what makes "no admissible pair" a value rather than a second flag.
_NEAREST_KEY_SEED = (1 << 63) - 1


def _closest_cross_component_edges(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_distance_sq: wp.float32
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Pick the two oriented boundary edges to bridge next, or ``None`` when nothing is left to join.

    ``None`` covers every reason at once: one component, no boundary at all, or no cross-component
    boundary-vertex pair within ``max_distance_sq``.

    The candidate set is the **first column** of the oriented boundary edges rather than a separate
    boundary-vertex list, and that is what makes the answer an *edge* pair with no search: row ``i``
    of that table is already a boundary edge wound the way its face winds it, i.e. exactly what
    [`bridge_edges`][triwarp.holes.bridge_edges] takes, so the winning slots name their own edges.
    The column is materialized with ``wp.clone`` because a column view is strided and Warp's
    Python-scope gather silently ignores an index array's stride.

    Only three scalars come back: the packed winner key and the two rows it names. The boundary
    table itself never leaves the device.
    """
    device = faces.device
    boundary = tw.boundary.oriented_boundary_edges(vertices, faces)
    n_boundary = int(boundary.shape[0])
    if n_boundary == 0:
        return None

    n_faces = int(faces.shape[0]) // 3
    face_labels = tw.adjacency.face_connected_component_labels(faces)
    vertex_labels = wp.full(int(vertices.shape[0]), wp.int32(-1), dtype=wp.int32, device=device)
    wp.launch(
        kernel_scatter.scatter_face_labels_to_vertices,
        dim=3 * n_faces,
        inputs=[faces, face_labels, vertex_labels],
        device=device,
    )
    members = wp.clone(twt.as_array2d(boundary, wp.int32)[:, 0])
    labels = tw.array.gather(vertex_labels, members)

    best = wp.array([wp.int64(_NEAREST_KEY_SEED)], dtype=wp.int64, device=device)
    partner = wp.empty(n_boundary, dtype=wp.int32, device=device)
    wp.launch(
        kernel_holes.reduce_closest_cross_label_pair,
        dim=n_boundary,
        inputs=[vertices, members, labels, max_distance_sq, best, partner],
        device=device,
    )
    key = int(read_scalar(best, 0))
    if key == _NEAREST_KEY_SEED:
        return None

    # The packed key's low half is the winning slot; its partner is what that thread found.
    slot_a = key & 0xFFFFFFFF
    slot_b = int(read_scalar(partner, slot_a))
    rows_np = tw.array.gather(
        boundary,
        wp.array(np.array([slot_a, slot_b], dtype=np.int32), dtype=wp.int32, device=device),
    ).numpy()
    return (int(rows_np[0, 0]), int(rows_np[0, 1])), (int(rows_np[1, 0]), int(rows_np[1, 1]))


def _check_bridge_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
    joined: Sequence[tuple[int, int]],
    validate: bool,
) -> None:
    """
    Reject a bridge that is degenerate, not on the boundary, or would duplicate an edge.

    Both membership tests run on device against a handful of query rows, so the only readback is
    those few flags -- an earlier version built Python sets over every mesh edge, which is one
    interpreter pass per edge and made ``validate=True`` unusable on a scan mesh.
    """
    if tuple(edge_a) == tuple(edge_b):
        raise ValueError("edge_a and edge_b must be different edges")
    if not validate:
        return

    on_rim = _rows_present(
        tw.boundary.oriented_boundary_edges(vertices, faces), [edge_a, edge_b], faces.device
    )
    for name, edge, present in (("edge_a", edge_a, on_rim[0]), ("edge_b", edge_b, on_rim[1])):
        if not present:
            raise ValueError(
                f"{name}={edge} is not a boundary edge of faces, wound as its face winds it"
            )

    if not joined:
        return
    # ``joined`` is what the patch adds *between vertices that already exist*; an interior vertex it
    # invents cannot collide with anything. The lookup is over the mesh's own edges rather than the
    # rim's, since a chord across a thin neck is usually an interior edge.
    sorted_pairs = [(min(u, v), max(u, v)) for u, v in joined]
    collides = _rows_present(
        tw.edges.faces_to_edges(faces, sorted=True), sorted_pairs, faces.device
    )
    for (u, v), present in zip(joined, collides, strict=True):
        if present:
            raise ValueError(
                f"bridging {tuple(edge_a)} to {tuple(edge_b)} would add a second edge between "
                f"{u} and {v}, leaving the mesh non-manifold; pick a different pair"
            )


def _rows_present(
    rows: twt.Array2dInt32, queries: Sequence[tuple[int, int]], device: wp.DeviceLike
) -> list[bool]:
    """Test a handful of index rows for membership in a table of them, on device."""
    if int(rows.shape[0]) == 0:
        return [False] * len(queries)
    query_wp = wp.array(np.asarray(queries, dtype=np.int32), dtype=wp.int32, device=device)
    present = wp.zeros(len(queries), dtype=wp.bool, device=device)
    wp.launch(
        kernel_array.mark_rows_present,
        dim=(int(rows.shape[0]), len(queries)),
        inputs=[rows, query_wp, present],
        device=device,
    )
    return present.list()


def _bridge_triangles(
    edge_a: tuple[int, int], edge_b: tuple[int, int]
) -> list[tuple[int, int, int]]:
    """Build the one or two triangles of the flat patch, wound against both edges' own faces."""
    a0, a1 = edge_a
    b0, b1 = edge_b
    # The quadrilateral runs (a1, a0, b1, b0): each edge is traversed backwards from the way its own
    # face winds it, which is what makes the patch's outward side agree with the mesh's. It is split
    # along the a0-b0 diagonal, so a shared endpoint *on that diagonal* is the one case the split
    # cannot express: at a0 == b0 both halves fold onto a line and the patch comes out empty, and at
    # a1 == b1 the two halves are the same triangle wound opposite ways. Both are named here, and
    # the surviving triangle is the collapsed quad's own traversal. (Sharing both endpoints means
    # the two edges are one edge, which the caller rejects before reaching this.)
    if a0 == b0:
        return [(a1, a0, b1)]
    if a1 == b1:
        return [(a1, a0, b0)]
    # A *crossed* shared endpoint (a0 == b1 or a1 == b0) collapses exactly one half, which the
    # degeneracy filter removes correctly -- the survivor is already the right triangle.
    triangles = [(a1, a0, b0), (a0, b1, b0)]
    return [t for t in triangles if len(set(t)) == 3]


# ---------------------------------------------------------------------------
# Smooth-patch pipeline: min-weight fill / stitch -> region subdivision -> region smoothing.
# ---------------------------------------------------------------------------


def _patch_mask(
    n_faces_before: int, n_faces_after: int, device: wp.DeviceLike
) -> wp.array[wp.bool]:
    """Boolean face mask marking the trailing ``[n_faces_before, n_faces_after)`` fill faces."""
    mask = wp.zeros(n_faces_after, dtype=wp.bool, device=device)
    # Warp rejects a zero-length slice, and the "nothing was filled" caller passes an empty range.
    if n_faces_after > n_faces_before:
        mask[n_faces_before:].fill_(True)
    return mask


def _mean_rim_edge_length(
    vertices: wp.array[wp.vec3], loops: _PackedLoops | list[wp.array[wp.int32]]
) -> float:
    """
    Mean edge length over the rims of the given loops (the derived subdivision target).

    Every rim is closed, so its edge count is its vertex count and the mean is the total perimeter
    over ``sum(sizes)`` -- which makes this one
    [`_PackedLoops.perimeters`][triwarp.holes._PackedLoops.perimeters] launch plus its one
    readback, the same measurement [`fill_small`][triwarp.holes.fill_small] already pays. A list
    is accepted because
    [`triwarp.holes.stitch_smooth`][triwarp.holes.stitch_smooth] holds its two rims
    individually; it is packed here rather than at that call site so the private surface stays one
    name wide.
    """
    packed = loops if isinstance(loops, _PackedLoops) else _pack_loops(loops)
    count = int(packed.sizes_np.sum())
    if count == 0:
        return 0.0
    return float(packed.perimeters(vertices).sum()) / count


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

    That second readback stays on the host on purpose, and not because the argmax could not run on
    the device: what the host needs is the resulting *mask*, to build the ragged gather index that
    compacts the surviving loops. Reducing on the device would still have to bring the winner back,
    so it would add a launch and remove nothing.

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
        perimeter_np = all_loops.perimeters(vertices)
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
