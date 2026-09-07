"""
Removing what should not be in a mesh: redundant elements, bad triangles and inconsistent winding.

[`make_solid`][triwarp.repair.make_solid] is the composite most callers want -- a broken digitised
surface in, a single watertight solid out -- and everything below is a stage of it that is also
useful on its own.

Five defects are visible in the index buffer alone, and each has a remover:
[`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices],
[`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices],
[`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces],
[`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces], and
[`collapse_small_triangles`][triwarp.repair.collapse_small_triangles].

A sixth remover answers a different question -- not *which elements are wrong* but *which parts of
the mesh are not the mesh*: [`remove_small_components`][triwarp.repair.remove_small_components]
drops face-connected components that are too small, by face count, area or bounding-box diameter.
It is the first step of every repair pipeline, and the debris it removes is not defective in
itself.

Two further defects need geometry rather than topology to detect, so they are found by a threshold
rather than a rule: [`validation.face_defective_mask`][triwarp.validation.face_defective_mask] flags
faces that are too thin, misoriented against their neighbourhood, or folded back over it -- it lives
with the other per-element detectors. [`remove_folded_faces`][triwarp.repair.remove_folded_faces]
deletes the folded ones and [`flip_t_vertices`][triwarp.repair.flip_t_vertices] flips away the
slivers a T-junction leaves behind.

The verb predicts the return shape, and that is a rule rather than a coincidence:

- **``remove_*`` / ``collapse_*``** change the element count, so they return ``(vertices, faces)``
  or more -- there is a new position buffer because vertices went away or moved.
- **``make_*``** preserve positions and counts and rewrite only the index buffer, so they return
  ``faces`` alone -- *unless* the name is a whole-mesh **outcome** rather than a property of the
  index buffer, which is [`make_solid`][triwarp.repair.make_solid] alone. It is the composite of
  most of this module and returns ``(vertices, faces)`` like the removers it runs; the three
  property-fixers beside it ([`make_winding_consistent`][triwarp.repair.make_winding_consistent],
  [`make_volume`][triwarp.repair.make_volume],
  [`make_normals_outward`][triwarp.repair.make_normals_outward]) return ``faces``.
- **``reverse_winding``** is the one verb outside that scheme, and it obeys the same shape rule
  for the same reason: it rewrites only the index buffer, so it returns ``faces``. It is not a
  ``make_*`` because it establishes no property -- it flips orientation unconditionally, where the
  three ``make_*`` fixers decide face by face.
- **A verb that only *moves* vertices** returns the positions alone, since neither buffer of indices
  changes: [`flatten_degree3_vertices`][triwarp.repair.flatten_degree3_vertices] is the only one,
  and it is here rather than in [`triwarp.smoothing`][triwarp.smoothing] -- whose every member has
  that same signature -- because it is the gentler half of a pair with
  [`remove_degree3_vertices`][triwarp.repair.remove_degree3_vertices]: same defect, same test, one
  answering it by deleting the vertex and one by flattening the bump. Splitting the pair across two
  modules to satisfy the shape rule would cost more than the exception does.
- **``*_mask``** are detectors: they return a ``wp.array[wp.bool]`` and mutate nothing. The one this
  module used to hold now lives in [`triwarp.validation`][triwarp.validation].

Two names carry a verb of their own and follow one of the rules anyway, which is worth saying so the
list does not read as exhaustive. [`flip_t_vertices`][triwarp.repair.flip_t_vertices] is named for
the ``make_*`` pattern rather than the ``remove_*`` one because it removes nothing: it flips the
long edge of each sliver a T-junction leaves, so the face count is unchanged and only ``faces``
comes back. [`straighten_boundary`][triwarp.repair.straighten_boundary] is the same shape under a
geometric verb. And [`fix_self_intersections`][triwarp.repair.fix_self_intersections],
[`split_non_manifold_vertices`][triwarp.repair.split_non_manifold_vertices],
[`collapse_small_triangles`][triwarp.repair.collapse_small_triangles] and
[`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces] all change the element count
and all return what the first rule says.

There is no ``eliminate_*``. Two functions used to spell it that way and both opened their own
summary line with the word *"Remove"* -- one verb in the name and another in the sentence
mkdocstrings renders beside it -- so they are
[`remove_degree3_vertices`][triwarp.repair.remove_degree3_vertices] and
[`remove_tunnels`][triwarp.repair.remove_tunnels].
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device, require_valid_faces
from triwarp.grouping import hash_vector_rows, unique_1d, unique_faces, unique_rows
from triwarp.kernels import array as kernel_array
from triwarp.kernels import bounds as kernel_bounds
from triwarp.kernels import repair as kernel_repair
from triwarp.kernels import scatter as kernel_scatter

# Lattice resolution for ``fix_self_intersections(method="voxel")``, in samples across the mesh's
# bounding-box diagonal. 128 is the same order as ``offset.offset_mesh``'s automatic floor and costs
# a 128 ** 3 field (8 MB); a caller who needs the surface resolved finer passes ``voxel_size``.
_VOXEL_REBUILD_RESOLUTION = 128


def make_solid(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    keep_largest: bool = True,
    join_components: bool = False,
    max_iter: int = 10,
    inner_iter: int = 3,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Turn a broken digitised surface into a single watertight solid.

    The composite this module's pieces exist to make possible, and the operation a caller reaching
    for "repair" usually means: debris removed, holes closed, degeneracies collapsed and
    self-intersections cut out and refilled, alternating until nothing is left to fix. Every stage
    is a public function here or in [`triwarp.holes`][triwarp.holes]; what this adds is the order
    and the loop, which is where the difficulty actually is -- closing a hole can create a
    self-intersection, and cutting one out reopens a hole.

    The stages, in order:

    0. The connectivity repair the reference implementation performs inside its *loader*, and which
       therefore does not look like a stage at all until it is missing:
       [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices],
       [`make_winding_consistent`][triwarp.repair.make_winding_consistent] and
       [`split_non_manifold_vertices`][triwarp.repair.split_non_manifold_vertices]. Without it a
       mesh whose defect is a non-manifold *edge* comes back with the right Euler characteristic and
       one component and is still not watertight, because no later stage looks at edge manifoldness.
    1. ``keep_largest`` -> [`remove_small_components`][triwarp.repair.remove_small_components],
       so the scan debris goes before anything expensive runs on it.
    2. ``join_components`` ->
       [`holes.join_closest_components`][triwarp.holes.join_closest_components], for an input whose
       pieces are meant to be one surface rather than a largest piece plus rubbish. Mutually useful
       with ``keep_largest`` rather than exclusive: keep the big piece *and* weld what is left.
    3. [`holes.fill_min_weight`][triwarp.holes.fill_min_weight] if any boundary remains.
    4. Up to ``max_iter`` rounds of
       [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces] and
       [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles], then
       [`fix_self_intersections`][triwarp.repair.fix_self_intersections] at
       ``max_iter=inner_iter``. The loop exits as soon as a round changes nothing.
    5. A final fill if stage 4 reopened a boundary -- which it routinely does, since cutting an
       intersecting region out is what opens one. Nothing geometric runs after it, and that is not
       an omission: filling a 3-vertex rim produces one sliver, a degeneracy pass deletes the sliver
       and reopens the rim, and the two trade the same faces indefinitely if run again after the
       fill.

    Under ``keep_largest`` the component filter runs **inside** stage 4 as well as at the top, and
    that is not belt and braces: cutting an intersecting band out can disconnect the surface, so the
    extra piece does not exist yet when stage 1 looks. For example, on a torus whose inner wall
    crosses itself, the intersection repair alone can leave two closed shells where the input was
    one.

    !!! warning "It returns the best it managed, not a guarantee"
        There is no success flag, deliberately. Convergence is not guaranteed for any input -- a
        patch can intersect something, and its repair can open another hole -- so on a stubborn mesh
        this returns a *partly* repaired surface rather than looping harder or raising. Ask
        [`validation.is_watertight`][triwarp.validation.is_watertight] if the answer matters.

        ``join_components`` is where that bites in practice, and the failure mode is worth knowing
        because it is not a bug in any stage: welding several shells leaves **one** rim spanning all
        of them, and if that rim is badly non-planar the minimum-weight patch across it
        self-intersects, so stage 4 cuts the patch out and undoes the join. For example, three
        hemispherical bowls joined at coplanar rims converge to one solid, but the same bowls
        tilted 45 degrees relative to each other come back as **three separate shells**. Rims that
        are far from coplanar want
        [`holes.stitch_loops`][triwarp.holes.stitch_loops] or a per-pair
        [`holes.bridge_edges`][triwarp.holes.bridge_edges] followed by a targeted fill, not this.

        The reference implementation prints a diagnostic here and its own caller reports that
        diagnostic **inverted**, which is worth knowing only as a reason not to trust a boolean of
        this shape from anywhere: read the mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    keep_largest
        Drop every connected component but the one with the most faces, first.
    join_components
        Bridge the remaining open components together instead of leaving them separate.
    max_iter
        Cap on the alternating degeneracy / self-intersection rounds.
    inner_iter
        Cap on the cut-and-refill passes inside each self-intersection repair.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Positions of the repaired mesh, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Flat face buffer of the repaired mesh.

    Raises
    ------
    ValueError
        If ``max_iter`` or ``inner_iter`` is negative, or if a face references a vertex index
        ``vertices`` does not cover.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`validation.is_watertight`][triwarp.validation.is_watertight]
        The question this does not answer for you.
    [`remove_small_components`][triwarp.repair.remove_small_components]
    [`fix_self_intersections`][triwarp.repair.fix_self_intersections]
    [`holes.fill_min_weight`][triwarp.holes.fill_min_weight]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]
        What to run afterwards if the winding has to face outward as well as be consistent.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_iter < 0 or inner_iter < 0:
        raise ValueError(
            f"max_iter and inner_iter must be non-negative, got {max_iter}, {inner_iter}"
        )

    if int(faces.shape[0]) == 0:
        return vertices, faces

    # This is the trust boundary for "a broken digitised surface": every stage below indexes
    # ``vertices[faces]`` without a bound check, the same way every other per-face kernel wrapper in
    # this package does, so an out-of-range index arriving here would otherwise reach the first one
    # silently (§12.1's memory-safety class) rather than raising a Python exception.
    require_valid_faces(faces, int(vertices.shape[0]), "make_solid")

    # Stage 0. The reference does this inside its *loader*, which is why it is easy to leave out and
    # why leaving it out is dangerous: skipping it can leave a mesh at the right Euler
    # characteristic and one component while still **not watertight**, because nothing downstream
    # addresses a non-manifold edge.
    vertices, faces, _remap = remove_unreferenced_vertices(vertices, faces)
    faces = make_winding_consistent(faces)
    vertices, faces, _source = split_non_manifold_vertices(vertices, faces)

    if keep_largest:
        vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
    if join_components:
        faces = tw.holes.join_closest_components(vertices, faces)

    faces = _fill_any_boundary(vertices, faces)
    for _ in range(max_iter):
        n_faces_before = int(faces.shape[0])
        vertices, faces = remove_degenerate_faces(vertices, faces)
        vertices, faces = collapse_small_triangles(vertices, faces)
        vertices, faces = fix_self_intersections(vertices, faces, max_iter=inner_iter)
        # Cutting an intersecting band out can *disconnect* the surface, so the component filter has
        # to run again here and not only at the top: a single pass at the start cannot see a
        # component that did not exist yet.
        if keep_largest:
            vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
        if int(faces.shape[0]) == n_faces_before:
            break

    # Stage 4 opens a rim whenever it cuts an intersecting region out, so the last fill is not a
    # belt-and-braces repeat of stage 3 -- it is what makes the common case come back closed. And
    # nothing geometric may run *after* it: filling a small rim produces one sliver, degeneracy
    # removal deletes that sliver and reopens the rim, and the two then trade the same faces
    # forever -- so the fill goes last.
    faces = _fill_any_boundary(vertices, faces)
    if keep_largest:
        vertices, faces = remove_small_components(vertices, faces, keep_largest=True)
    return vertices, faces


def _fill_any_boundary(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """Close every boundary loop, or hand the buffer back untouched when there is none."""
    if int(tw.boundary.boundary_edges(vertices, faces).shape[0]) == 0:
        return faces
    return tw.holes.fill_min_weight(vertices, faces)


def remove_unreferenced_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_inverse: bool = False
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Remove vertices not referenced by any face and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Flat triangle index buffer.
    return_inverse
        If ``True``, also return ``inverse`` with ``new_vertices[inverse]`` sourcing
        ``vertices``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Referenced vertices only, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Face buffer with indices remapped into ``new_vertices``.
    remap : wp.array[wp.int32]
        Length ``n_vertices`` old-to-new map (``-1`` when unreferenced).
    inverse : wp.array[wp.int32], optional
        Present when ``return_inverse=True``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_vertices = int(vertices.shape[0])

    referenced = tw.array.indices_to_mask(faces, n_vertices, device=device)

    # ``flatnonzero`` already paid the readback that sizes its own output, and that size *is* the
    # referenced count -- a separate ``reduce.sum`` of the mask would be a second scan and a second
    # host sync for a number already in hand.
    inverse = tw.array.flatnonzero(referenced)
    n_referenced = int(inverse.shape[0])
    remap = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
    if n_referenced > 0:
        wp.launch(
            kernel_scatter.scatter_index, dim=n_referenced, inputs=[inverse, remap], device=device
        )

    new_vertices = (
        tw.array.gather(vertices, inverse)
        if n_referenced > 0
        else wp.empty(0, dtype=wp.vec3, device=vertices.device)
    )
    new_faces = tw.array.remap_indices(faces, remap)

    if return_inverse:
        return new_vertices, new_faces, remap, inverse
    return new_vertices, new_faces, remap


def remove_duplicated_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 0.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Merge duplicate vertex positions up to a coordinate tolerance and remap face indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Flat triangle index buffer.
    epsilon
        Uniqueness tolerance. Positive values snap coordinates to ``round(v / epsilon)``, so the
        tolerance is absolute and is the one you chose. ``0`` instead groups by a *relative*
        bucket about ``2.4e-4`` wide (see Notes) — pass an explicit ``epsilon`` unless that is
        what you want.

    Returns
    -------
    unique_vertices : wp.array[wp.vec3]
        Deduplicated vertex positions (first occurrence per equivalence class).
    unique_indices : wp.array[wp.int32]
        Length ``n_unique``. Original indices into ``vertices`` for each output row.
    inverse : wp.array[wp.int32]
        Length ``n_vertices``. Maps each input vertex to its slot in ``unique_vertices``.
    unique_faces : wp.array[wp.int32]
        Face buffer with indices remapped into ``unique_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Both tolerance modes quantize positions and group by cell, so both separate a pair straddling a
    cell boundary however close it is. That is worth knowing about ``epsilon=0`` in particular,
    which is **not** an exact-equality test despite requiring no tolerance: it buckets by the high
    bits of each coordinate's ``float32`` representation, giving a *relative* cell about ``2.4e-4``
    wide. So ``1.0`` and ``1.000244`` merge, while two adjacent ``float32`` values on either side of
    a bucket edge do not. Prefer an explicit ``epsilon`` whenever the tolerance matters; use ``0``
    only to collapse positions that are already bitwise equal, which it does reliably (including
    across ``+0.0`` / ``-0.0``).

    Duplicates that are known from construction rather than measured are better collapsed directly:
    see [`revolve`][triwarp.creation.revolve], which derives them from its profile instead of
    hashing positions.

    See Also
    --------
    [`duplicate_vertex_inverse`][triwarp.repair.duplicate_vertex_inverse]
    [`hash_vector_rows`][triwarp.grouping.hash_vector_rows]
    """
    require_same_device(vertices=vertices, faces=faces)
    inverse = duplicate_vertex_inverse(vertices, epsilon)
    # No ``n_unique`` to pass: ``duplicate_vertex_inverse`` discards the unique array internally,
    # so the class count genuinely is not available here and the reduction is the only way to it.
    unique_indices = tw.grouping.first_occurrence_indices(inverse)
    unique_vertices = tw.array.gather(vertices, unique_indices)
    unique_faces = tw.array.remap_indices(faces, inverse)
    return unique_vertices, unique_indices, inverse, unique_faces


def duplicate_vertex_inverse(vertices: wp.array[wp.vec3], epsilon: float) -> wp.array[wp.int32]:
    """
    Map each vertex to the slot of its coincident-vertex equivalence class.

    The inverse map produced by welding vertices at ``epsilon`` tolerance, without also
    computing the deduplicated vertex/face buffers — useful for remapping per-vertex
    attributes (colors, UVs, ...) to match a [`remove_duplicated_vertices`]
    [triwarp.repair.remove_duplicated_vertices] call made with the same ``epsilon``.

    This is the shared equivalence map the dedup remaps by, not a remover -- which is why it lives
    here rather than in [`triwarp.grouping`][triwarp.grouping] beside
    [`first_occurrence_indices`][triwarp.grouping.first_occurrence_indices], whose shape it has. Its
    key is a *tolerance-quantized position*, and choosing that tolerance is mesh-repair policy;
    ``grouping`` is deliberately dtype-generic and geometry-free, so an ``epsilon`` there would be
    the first crack in that contract.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    epsilon
        Uniqueness tolerance, with the same meaning as in
        [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]: positive values
        snap coordinates to ``round(v / epsilon)``, while ``0`` groups by a *relative* bucket about
        ``2.4e-4`` wide rather than testing for equality.

    Returns
    -------
    wp.array[wp.int32]
        Length ``n_vertices``. Maps each input vertex to its slot in the deduplicated set.

    See Also
    --------
    [`remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]
        The remover this map is the shared half of.
    [`hash_vector_rows`][triwarp.grouping.hash_vector_rows]
    [`grouping.first_occurrence_indices`][triwarp.grouping.first_occurrence_indices]
        The geometry-free counterpart: the same class-to-representative reduction over any key.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    if epsilon > 0.0:
        row_keys = hash_vector_rows(vertices, epsilon=epsilon)
        _, inverse = unique_1d(row_keys, return_inverse=True)
    else:
        rows = twt.empty_2d((n, 3), wp.float32, device=device)
        wp.utils.array_cast(vertices, rows)
        _, inverse = unique_rows(rows, return_inverse=True)
    return inverse


def resolve_duplicated_faces(
    faces: wp.array[wp.int32],
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Resolve duplicated triangles by orientation-aware cancellation rules.

    For each geometric duplicate:
    - equal positive and negative counts: remove all copies;
    - one extra positive copy: keep one positively oriented face;
    - one extra negative copy: keep one negatively oriented face;
    - otherwise raise ``ValueError`` when counts are not orientable.

    Parameters
    ----------
    faces
        Flat triangle index buffer.

    Returns
    -------
    resolved_faces : wp.array[wp.int32]
        Flat buffer of kept faces.
    kept_indices : wp.array[wp.int32]
        Original face indices into the input ``faces`` buffer.

    Raises
    ------
    ValueError
        If a duplicate group's signed count is not orientable, i.e. its positive and negative
        copies differ by more than one.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    faces2d = faces.reshape((-1, 3))
    unique_faces_wp, inverse = unique_faces(faces, return_inverse=True)
    num_unique = int(unique_faces_wp.shape[0]) // 3

    # Per-group orientation stats scattered on device: member/signed counts plus the smallest
    # member index of each sign class (seeded with the ``n_faces`` sentinel).
    member_count = wp.zeros(num_unique, dtype=wp.int32, device=device)
    signed_count = wp.zeros(num_unique, dtype=wp.int32, device=device)
    first_member = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_positive = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    first_negative = wp.full(num_unique, n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.scatter_duplicate_face_stats,
        dim=n_faces,
        inputs=[
            faces,
            unique_faces_wp,
            inverse,
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
        ],
        device=device,
    )

    keep = wp.empty(num_unique, dtype=wp.int32, device=device)
    error_group = wp.full(1, num_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.resolve_duplicate_groups,
        dim=num_unique,
        inputs=[
            member_count,
            signed_count,
            first_member,
            first_positive,
            first_negative,
            keep,
            error_group,
        ],
        device=device,
    )
    first_error = int(read_scalar(error_group, 0))
    if first_error < num_unique:
        count = int(read_scalar(signed_count, first_error))
        raise ValueError(
            f"resolve_duplicated_faces: non-orientable duplicate face group {first_error} "
            f"with signed count {count}"
        )

    # Compact kept decisions in ascending group order (matches the reference emission order).
    keep_mask = wp.empty(num_unique, dtype=wp.bool, device=device)
    wp.map(kernel_array.greater_equal, keep, wp.int32(0), out=keep_mask)
    kept_slots = tw.array.flatnonzero(keep_mask)
    if int(kept_slots.shape[0]) == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, empty

    kept_wp = tw.array.gather(keep, kept_slots)
    resolved = tw.array.gather(faces2d, kept_wp).reshape((-1,))
    return resolved, kept_wp


def remove_degenerate_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop degenerate (zero-area) triangles and reindex, keeping vertex positions unchanged.

    Mirrors ``trimesh.Trimesh.nondegenerate_faces`` + ``update_faces``: a face is degenerate when
    two of its vertices coincide or its three vertices are collinear, detected by
    [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask] (both triangle altitudes
    exceed the merge tolerance). Surviving faces are unchanged; vertices left unreferenced after the
    drop are removed by the reindexing in
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask].

    Unlike [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles], no vertices are
    merged and no edges are collapsed: this only removes faces already degenerate in the input.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced by a non-degenerate face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the non-degenerate faces, remapped into ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles]
    [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask]
    [`trimesh.triangles.nondegenerate`][]
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)

    keep_mask = tw.triangles.face_nondegenerate_mask(vertices, faces)
    return tw.selection.submesh_from_face_mask(vertices, faces, keep_mask)


def remove_non_manifold_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], max_iter: int = 3
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove faces touching a non-manifold (>2-incident) edge, iterating until edge-manifold.

    Each pass keeps only faces whose three edges are each used by at most two faces
    ([`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]); dropping a face can make a
    neighbour manifold, so it repeats up to ``max_iter`` times.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    max_iter
        Maximum number of removal passes.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced after non-manifold faces are dropped, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the surviving (edge-manifold, up to ``max_iter`` passes) faces.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`edge_manifold_mask`][triwarp.validation.edge_manifold_mask]
    """
    require_same_device(vertices=vertices, faces=faces)
    for _ in range(max_iter):
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break
        keep = tw.validation.edge_manifold_mask(faces, allow_boundary_edges=True)
        kept = tw.array.flatnonzero(keep)
        if int(kept.shape[0]) == n_faces:
            break  # already edge-manifold
        vertices, faces = tw.selection.submesh_from_face_mask(vertices, faces, keep)
    return vertices, faces


def remove_small_components(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    keep_largest: bool = False,
    min_faces: int | None = None,
    min_area: float | None = None,
    min_diameter: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop face-connected components that are too small, by one of four measures of "small".

    Debris -- a stray shell from a scan, a sliver left by a boolean, a component a decimation
    stranded -- is what every repair pipeline removes first, and until now
    [`combine.split`][triwarp.combine.split] handed back *every* component and left the caller to
    write the argmax. This is that step, and it stays on the device: the per-component statistic is
    accumulated by one scatter and thresholded by one kernel, so nothing is read back and no
    component is ever materialized as its own mesh.

    Exactly one criterion may be given, because they are four different questions rather than four
    spellings of one and combining them would hide which one rejected a component:

    - ``keep_largest`` keeps the single component with the **most faces** and drops every other, so
      the result always has exactly one component. Face count, not area or diameter: a small dense
      shell outranks a large coarse one. A tie goes to the component whose lowest face index is
      smallest, which makes the choice deterministic rather than dependent on thread order.
    - ``min_faces`` keeps components with at least that many faces -- scale-free, and the measure
      that tracks *how much data* a component carries.
    - ``min_area`` keeps components whose summed triangle area reaches it.
    - ``min_diameter`` keeps components whose axis-aligned bounding-box diagonal reaches it, which
      is the measure that survives a component being a thin sheet of many tiny triangles.

    All three ``min_*`` bounds are **inclusive**, matching the reference implementations: a
    component of exactly the threshold face count, or of exactly the threshold diagonal, survives.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    keep_largest
        Keep only the component with the most faces.
    min_faces
        Minimum face count for a component to be kept.
    min_area
        Minimum summed triangle area for a component to be kept.
    min_diameter
        Minimum bounding-box diagonal for a component to be kept.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices of the surviving components, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the surviving faces, in their input order.

    Raises
    ------
    ValueError
        If no criterion is given, or if more than one is.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`combine.split`][triwarp.combine.split]
        The whole decomposition, when every component is wanted rather than a subset.
    [`holes.join_closest_components`][triwarp.holes.join_closest_components]
        The other answer to a multi-component mesh -- weld the pieces together instead of
        discarding them.
    [`adjacency.face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
        The labelling this thresholds.
    [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces]

    Notes
    -----
    A component's label is a representative *face* index rather than a dense ``0..k-1`` id, so the
    per-component statistic is an ``n_faces``-long array of which only ``k`` slots are ever written.
    That is deliberate: densifying the labels first would cost a sort and a search per face to save
    an allocation, and it is the allocation that is cheap.
    """
    require_same_device(vertices=vertices, faces=faces)
    given = (keep_largest, min_faces is not None, min_area is not None, min_diameter is not None)
    if sum(given) != 1:
        raise ValueError(
            "pass exactly one of keep_largest, min_faces, min_area or min_diameter, "
            f"got {sum(given)}"
        )

    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return vertices, faces

    labels = tw.adjacency.face_connected_component_labels(faces)
    keep = wp.empty(n_faces, dtype=wp.bool, device=device)

    if min_area is not None:
        areas = tw.triangles.face_quality(vertices, faces, metric="area")
        statistic = wp.zeros(n_faces, dtype=wp.float32, device=device)
        wp.launch(
            kernel_scatter.SCATTER_ADD[areas.dtype],
            dim=n_faces,
            inputs=[areas, labels, statistic],
            device=device,
        )
        # Gather-then-compare at Python scope rather than a kernel: ``statistic[labels]`` is a
        # per-component table read through the per-face label, and ``wp.map`` over that
        # ``indexedarray`` is the standard elementwise-op idiom (and ``make_volume`` below already
        # uses it). ``labels`` is a dense array, so the strided-index hazard -- which applies to a
        # *column* of a rank-2 buffer -- does not arise. The bound is inclusive at every criterion,
        # matching both references.
        wp.map(kernel_array.greater_equal, statistic[labels], wp.float32(min_area), out=keep)
    elif min_diameter is not None:
        diagonals = _component_diagonals(vertices, faces, labels)
        wp.map(kernel_array.greater_equal, diagonals[labels], wp.float32(min_diameter), out=keep)
    else:
        counts = wp.zeros(n_faces, dtype=wp.int32, device=device)
        wp.launch(
            kernel_scatter.count_occurrences, dim=n_faces, inputs=[labels, counts], device=device
        )
        if min_faces is not None:
            wp.map(kernel_array.greater_equal, counts[labels], wp.int32(min_faces), out=keep)
        else:
            # ``-1`` is below every packed key, so the reduction needs no separate seeding pass and
            # the winning label never reaches the host -- the mask kernel recomputes its key.
            best = wp.array([wp.int64(-1)], dtype=wp.int64, device=device)
            wp.launch(
                kernel_repair.reduce_largest_group,
                dim=n_faces,
                inputs=[counts, best],
                device=device,
            )
            wp.launch(
                kernel_repair.mark_largest_group_mask,
                dim=n_faces,
                inputs=[labels, counts, best, keep],
                device=device,
            )

    return tw.selection.submesh_from_face_mask(vertices, faces, keep)


def _component_diagonals(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], labels: wp.array[wp.int32]
) -> wp.array[wp.float32]:
    """Bounding-box diagonal per component, indexed by the component's label."""
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    # ``+inf`` in all six slots seeds both ends at once: the packing stores the upper corner negated
    # so every update is a ``wp.atomic_min``, and a component no face names stays at the seed, which
    # ``packed_box_diagonals`` reports as a zero diagonal rather than as ``nan``.
    corners = wp.full(6 * n_faces, value=math.inf, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_group_bounds,
        dim=n_faces,
        inputs=[vertices, faces, labels, corners],
        device=device,
    )
    diagonals = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_bounds.packed_box_diagonals, dim=n_faces, inputs=[corners, diagonals], device=device
    )
    return diagonals


def split_non_manifold_vertices(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Make a mesh manifold and orientable by duplicating vertices, keeping every face.

    The non-lossy counterpart of
    [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces], which deletes geometry
    to reach the same property: this changes no position and drops no triangle, it only splits
    vertices apart, so the surface is unchanged and the face count is exactly preserved. Use it to
    feed a mesh to code that requires manifold input -- several reference libraries (e.g.
    potpourri3d) reject non-manifold meshes outright, and so do triwarp's own halfedge consumers.

    Two corners are kept together only across an edge that is **manifold and consistently
    oriented**: exactly one half-edge each way. Every other edge -- a boundary edge, one shared by
    three or more faces, or one whose two faces traverse it the same way -- separates the copies. So
    a bowtie vertex splits in two, an edge with three faces splits into three boundary edges, and a
    flipped face is cut free of its neighbours rather than reoriented (that is
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]'s job, and running it first
    leaves less to split).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        ``(n_new,)`` positions with ``n_new >= n_referenced``; each is a copy of the original vertex
        it came from, so the point set is unchanged as a *set*.
    new_faces : wp.array[wp.int32]
        Flat buffer of the same ``n_faces`` triangles in input order, indexing ``new_vertices``.
    source : wp.array[wp.int32]
        Length ``n_new`` map from each new vertex to the original it duplicates, so
        ``new_vertices == vertices[source]`` and a per-vertex attribute transfers with
        [`gather`][triwarp.array.gather]. This is ``igl.split_nonmanifold``'s ``SVI``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Notes
    -----
    Vertices unreferenced by any face are **dropped**, since a new vertex only exists as some face's
    corner -- the same convention as
    [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices], and the reason
    this function cannot simply return an ``n_vertices``-length remap.

    Remove degenerate faces first, with
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]: a triangle with a repeated
    index has two corners at one vertex that no edge can join, so it survives as two copies of
    that vertex and the face stays degenerate.

    The implementation is a connected-components pass over a graph of ``3 * n_faces`` corner nodes
    rather than a per-vertex star walk: one kernel counts each edge's half-edges by direction, a
    second emits two links per mergeable edge, and
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    does the merging. ``igl::split_nonmanifold`` instead explodes the mesh to ``3 * n_faces``
    singleton vertices and greedily re-merges pairs, re-testing manifoldness after each candidate --
    order-dependent and sequential by construction. The two agree exactly on a manifold mesh, a
    bowtie vertex, a consistently-wound fan of three faces on one edge (both split it into three),
    a flipped face and an open boundary.

    **They differ on one input class, by design.** Where an edge carries one half-edge in one
    direction and *several* in the other -- which is what a duplicated face produces -- igl keeps
    one arbitrarily chosen pair joined, while this splits every copy: 18 vertices against igl's 15
    on an icosahedron with one face duplicated, and 8 527 against 8 320 on ``bunny_decimated``,
    whose 87 duplicated faces are its only non-manifoldness. Both results are edge- and
    vertex-manifold with the input's face count; this one is order-independent and makes no
    arbitrary choice. Note that
    [`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces] is not a way around the
    difference: libigl's cancellation rules it implements cover a ``+1``/``-1`` imbalance, so a face
    duplicated in the *same* orientation makes it raise rather than dropping the copy.

    See Also
    --------
    [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`is_edge_manifold`][triwarp.validation.is_edge_manifold]
    ``igl.split_nonmanifold``
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_index = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=vertices.device), faces, empty_index

    n_corners = 3 * n_faces
    _unique_edges, edge_of_corner = tw.edges.edges_unique(faces, n_vertices=int(vertices.shape[0]))
    n_unique = int(_unique_edges.shape[0])

    forward_count = wp.zeros(n_unique, dtype=wp.int32, device=device)
    backward_count = wp.zeros(n_unique, dtype=wp.int32, device=device)
    forward_corner = wp.full(n_unique, -1, dtype=wp.int32, device=device)
    backward_corner = wp.full(n_unique, -1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.halfedge_orientation_slots,
        dim=n_corners,
        inputs=[
            faces,
            edge_of_corner,
            forward_count,
            backward_count,
            forward_corner,
            backward_corner,
        ],
        device=device,
    )

    links = twt.empty_2d((2 * n_unique, 2), wp.int32, device=device)
    wp.launch(
        kernel_repair.corner_merge_links,
        dim=n_unique,
        inputs=[forward_count, backward_count, forward_corner, backward_corner, links],
        device=device,
    )

    # ``validate=False``: the links are corner ids this function just built, so the range check
    # would buy nothing but a full readback of the link buffer.
    labels = tw.graph.connected_component_labels_from_edges(
        links, node_count=n_corners, validate=False
    )
    # ECL-CC labels each component by its smallest node id, and a node id *is* a corner, so the
    # representative's vertex is the original this copy came from.
    representatives, new_faces = tw.grouping.unique_1d(labels, return_inverse=True)
    source = tw.array.gather(faces, representatives)
    return tw.array.gather(vertices, source), new_faces, source


def collapse_small_triangles(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], epsilon: float = 1e-6
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Collapse triangles smaller than a bounding-box-relative area threshold.

    Mirrors ``igl::collapse_small_triangles``. A triangle is *small* when its doubled area is below
    ``2 * epsilon * bbd ** 2``, where ``bbd`` is the diagonal of the axis-aligned bounding box of
    ``vertices``. Each small triangle has its **shortest edge** collapsed by merging that edge's two
    endpoints; the merged face (now carrying a repeated vertex) is discarded. The process repeats to
    a fixpoint, so triangles that only become small after a neighbouring collapse are also removed.

    This subsumes degenerate-triangle removal: an exactly degenerate face (zero area) is always
    below the threshold, so passing a small ``epsilon`` removes it. To drop only degenerate faces
    without any bounding-box-relative collapsing, use
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    epsilon
        Relative area tolerance. The doubled-area threshold is ``2 * epsilon * bbd ** 2``; larger
        values collapse more (and larger) triangles.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices surviving the collapse, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the surviving faces, remapped into ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
    [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask]

    Notes
    -----
    Where ``igl::collapse_small_triangles`` merges vertices by a sequentially updated index map and
    recurses until no edge collapses, this resolves all shortest-edge merges of one pass at once via
    the connected-components closure of
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges],
    then loops over the shrinking mesh. Both converge to a mesh with no sub-threshold triangle. The
    surviving vertex of a collapsed edge keeps the position of the component representative (the
    lowest original index in its class) rather than libigl's longest-edge-preserving endpoint; for
    sub-threshold triangles the two endpoints are close enough that the difference is negligible.
    The bounding-box diagonal is measured once on the input ``vertices`` so the threshold is fixed
    across iterations.
    """
    require_same_device(vertices=vertices, faces=faces)
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0 or int(vertices.shape[0]) == 0:
        return wp.clone(vertices), wp.clone(faces)

    bbd = tw.bounds.enclosing_diagonal(vertices)
    min_dbl_area = wp.float32(2.0 * epsilon * bbd * bbd)

    current_vertices = vertices
    current_faces = faces
    max_iterations = int(faces.shape[0])  # bounded: each collapsing pass drops at least one face
    for _ in range(max_iterations):
        n_current = int(current_faces.shape[0]) // 3
        if n_current == 0:
            break

        pairs = twt.empty_2d((n_current, 2), wp.int32, device=device)
        flag = wp.empty(n_current, dtype=wp.int32, device=device)
        wp.launch(
            kernel_repair.small_triangle_collapse_edges,
            dim=n_current,
            inputs=[current_vertices, current_faces, min_dbl_area, pairs, flag],
            device=device,
        )

        if int(tw.reduce.sum(flag)) == 0:
            break

        # Non-flagged faces emit a self-pair (i0, i0); these are self-loops that leave the
        # connected-components closure unchanged, so all rows can be passed without filtering.
        n_vertices = int(current_vertices.shape[0])
        labels = tw.graph.connected_component_labels_from_edges(pairs, node_count=n_vertices)

        unique_labels, inverse = unique_1d(labels, return_inverse=True)
        unique_indices = tw.grouping.first_occurrence_indices(inverse, int(unique_labels.shape[0]))
        class_vertices = tw.array.gather(current_vertices, unique_indices)
        remapped_faces = tw.array.remap_indices(current_faces, inverse)

        keep_mask = tw.triangles.face_nondegenerate_mask(class_vertices, remapped_faces)
        current_vertices, current_faces = tw.selection.submesh_from_face_mask(
            class_vertices, remapped_faces, keep_mask
        )

    return current_vertices, current_faces


def straighten_boundary(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    min_normal_dot: float = 0.9,
    max_aspect_ratio: float = 10.0,
    iterations: int = 1,
    return_count: bool = False,
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], int]:
    """
    Close concave notches in the mesh's rim, one triangle at a time.

    A rim that came out of a clip, a decimation or a scan is *ragged*: it zig-zags by one triangle
    even where the surface is smooth. Each pass adds the single triangle that spans a rim vertex's
    two neighbours, wherever that triangle faces the same way as the surface it joins and is not a
    sliver. No vertex moves and none is added -- the mesh only gains faces, so the interior is
    untouched and every existing index stays valid.

    Two gates, and both matter. ``min_normal_dot`` is what distinguishes a notch from a corner: a
    *convex* rim corner would be closed by a triangle facing away from the surface, and filling it
    folds the mesh over. ``max_aspect_ratio`` stops the pass trading a ragged rim for a fan of
    slivers. Neither has a natural default, which is why both are keywords rather than constants.

    Adjacent notches share a rim edge, so one pass closes a maximal independent set of them --
    lowest index wins, deterministically -- and ``iterations`` passes go round again on what is
    left. A pass that closes nothing ends the loop early.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Read only; nothing moves.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be edge-manifold, since the
        rim is found through the halfedge twins.
    min_normal_dot
        Minimum cosine between the new triangle's normal and each of the two rim faces it will
        border. ``1.0`` accepts only a perfectly flat notch; ``0.0`` accepts any notch that is not
        folded back on the surface.
    max_aspect_ratio
        Largest circum-radius over twice the in-radius the new triangle may have, as
        [`face_quality`][triwarp.triangles.face_quality] measures it.
    iterations
        Number of independent-set passes.
    return_count
        If ``True``, also return ``added``.

    Returns
    -------
    faces : wp.array[wp.int32]
        Flat triangle index buffer with the new triangles appended. The vertex buffer is unchanged
        and is not returned.
    added : int, optional
        Present when ``return_count=True``. How many triangles were added, summed over the passes.
        Zero means no notch passed both gates, and the buffer is the input's. A diagnostic: the
        pass loop already stops itself when a pass closes nothing, so a caller needs this only to
        report what happened.

    Raises
    ------
    ValueError
        If ``iterations`` is negative, or propagated from
        [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when the mesh is not edge-manifold.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
        Closes a rim *completely*; this only tidies its shape and leaves it open.
    [`ears`][triwarp.boundary.ears]
        Finds the faces with two rim edges, which is the dual situation -- an ear sticks out where a
        notch cuts in.
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    require_same_device(vertices=vertices, faces=faces)
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    device = faces.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or iterations == 0:
        return (faces, 0) if return_count else faces

    added = 0
    for _ in range(iterations):
        n_faces = int(faces.shape[0]) // 3
        n_halfedges = 3 * n_faces
        twins = tw.halfedge.halfedge_twins(faces, n_vertices=n_vertices)
        rim_next = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
        rim_prev = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
        rim_face = wp.full(n_vertices, -1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_repair.collect_rim_links,
            dim=n_halfedges,
            inputs=[faces, twins, rim_next, rim_prev, rim_face],
            device=device,
        )
        candidate = wp.zeros(n_vertices, dtype=wp.bool, device=device)
        wp.launch(
            kernel_repair.straighten_candidate_mask,
            dim=n_vertices,
            inputs=[
                vertices,
                faces,
                tw.triangles.face_normals_and_areas(vertices, faces)[0],
                rim_next,
                rim_prev,
                rim_face,
                wp.float32(min_normal_dot),
                wp.float32(max_aspect_ratio),
                candidate,
            ],
            device=device,
        )
        cursor = wp.zeros(1, dtype=wp.int32, device=device)
        new_faces = twt.empty_2d((n_vertices, 3), wp.int32, device=device)
        wp.launch(
            kernel_repair.emit_straighten_faces,
            dim=n_vertices,
            inputs=[faces, rim_next, rim_prev, candidate, cursor, new_faces],
            device=device,
        )
        # One readback per pass, and it is the loop's own stopping test: how many notches the pass
        # actually closed is a device-side fact and a Python loop cannot branch on it otherwise.
        n_added, (accepted,) = tw.array.trim_to_count(cursor, new_faces)
        if n_added == 0:
            break
        faces = tw.array.concatenate([faces, accepted.reshape(3 * n_added)])
        added += n_added
    return (faces, added) if return_count else faces


def remove_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    max_iter: int = 8,
    return_count: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]] | tuple[wp.array[wp.vec3], wp.array[wp.int32], int]
):
    """
    Remove interior vertices with exactly three incident faces, collapsing each fan to one triangle.

    A valence-3 interior vertex carries no information the surface needs: its three faces tile the
    triangle formed by its three neighbours, so deleting it and keeping that triangle changes the
    connectivity and nothing else. They are a standard residue of subdivision, of decimation and of
    hole filling, and they make every downstream valence statistic worse.

    Adjacent candidates share faces, so a pass removes a maximal **independent** set -- the
    lowest-indexed of any two neighbouring candidates wins, deterministically -- and the loop
    repeats until none is left or ``max_iter`` passes have run. Removing one vertex can create
    another, which is why ``return_count`` reports what happened rather than promising it.

    !!! note "The other half of a double-face pass already exists"
        A pair of triangles on the same three vertices is
        [`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces]' job, and this
        function deliberately does not repeat it. The two together are what a "remove double faces"
        pass means elsewhere; they are kept apart because they have different orientation rules and
        different answers on a non-orientable mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be edge-manifold, since the
        fan around a vertex is what this reasons about.
    max_iter
        Cap on the number of passes. Each pass removes an independent set, so a chain of adjacent
        candidates needs one pass per link; the default covers any chain length likely in practice.
    return_count
        If ``True``, also return ``removed``.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Positions of the result, with the removed vertices compacted away. No position moves.
    faces : wp.array[wp.int32]
        Flat triangle index buffer, three faces shorter per removed vertex plus one longer.
    removed : int, optional
        Present when ``return_count=True``. How many vertices were removed. Zero means the input
        had none and the buffers are it. A diagnostic: the pass loop stops itself when a pass finds
        no candidate, so nothing about calling this correctly depends on reading the count.

    Raises
    ------
    ValueError
        If ``max_iter`` is negative, or propagated from
        [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when the mesh is not edge-manifold.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`flatten_degree3_vertices`][triwarp.repair.flatten_degree3_vertices]
        The geometric answer to the same defect: move the vertex instead of deleting it, which
        leaves the connectivity and every per-vertex attribute intact.
    [`resolve_duplicated_faces`][triwarp.repair.resolve_duplicated_faces]
        The other half: triangles repeated on the same three vertices.
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]
        Removes faces by *geometry*; this one removes a vertex by its connectivity alone.
    [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings]
        The fan this walks.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_iter < 0:
        raise ValueError(f"max_iter must be non-negative, got {max_iter}")
    device = faces.device
    removed = 0
    for _ in range(max_iter):
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break
        n_vertices = int(vertices.shape[0])
        ring_halfedges, ring_offsets, is_boundary = tw.halfedge.vertex_one_rings(
            faces, n_vertices=n_vertices
        )
        candidate = wp.empty(n_vertices, dtype=wp.bool, device=device)
        wp.map(
            kernel_repair.is_interior_degree3,
            ring_offsets[:-1],
            ring_offsets[1:],
            is_boundary,
            out=candidate,
        )
        selected = wp.zeros(n_vertices, dtype=wp.bool, device=device)
        wp.launch(
            kernel_repair.select_independent_degree3,
            dim=n_vertices,
            inputs=[faces, ring_offsets, ring_halfedges, candidate, selected],
            device=device,
        )
        # One readback per pass, and it is the loop's own termination test: the pass count is what
        # bounds it, and there is no device-side way to stop a Python loop.
        n_selected = int(tw.reduce.sum(tw.array.astype(selected, wp.int32)))
        if n_selected == 0:
            break

        cursor = wp.zeros(1, dtype=wp.int32, device=device)
        dropped = wp.zeros(n_faces, dtype=wp.bool, device=device)
        new_faces = twt.empty_2d((n_selected, 3), wp.int32, device=device)
        wp.launch(
            kernel_repair.emit_degree3_replacement,
            dim=n_vertices,
            inputs=[faces, ring_offsets, ring_halfedges, selected, cursor, dropped, new_faces],
            device=device,
        )
        keep = wp.empty(n_faces, dtype=wp.bool, device=device)
        wp.map(kernel_array.mask_not, dropped, out=keep)
        kept = tw.array.gather(faces.reshape((n_faces, 3)), tw.array.flatnonzero(keep))
        faces = tw.array.concatenate([kept.reshape(-1), new_faces.reshape(3 * n_selected)])
        removed += n_selected
    if removed == 0:
        return (vertices, faces, 0) if return_count else (vertices, faces)
    # Compacted **once**, after the loop rather than inside it. A dead vertex has an empty ring and
    # so is never a candidate, which is what makes deferring safe; doing it per pass added a full
    # vertex-and-face pass to every iteration for no change in the answer.
    vertices, faces, _index = remove_unreferenced_vertices(vertices, faces)
    return (vertices, faces, removed) if return_count else (vertices, faces)


def flatten_degree3_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool] | None = None,
    *,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
) -> wp.array[wp.vec3]:
    """
    Flatten each interior valence-3 vertex into the plane of its three neighbours.

    A valence-3 interior vertex sits on a little tetrahedral bump: three triangles meeting at a
    point over the triangle its neighbours form. Moving it to their centroid puts it **in** that
    triangle's plane, so the bump disappears and the three faces become coplanar -- which is what
    makes a subdivision, a decimation or a hole fill stop leaving visible pimples.

    The gentler half of a pair.
    [`remove_degree3_vertices`][triwarp.repair.remove_degree3_vertices]
    answers the same defect by deleting the vertex and keeping one triangle, which changes the
    connectivity; this keeps every vertex and every face and only moves positions, so a caller
    holding per-vertex attributes or a face selection can use it and the other one would invalidate
    both.

    One pass is enough and there is no iteration count: two interior valence-3 vertices cannot be
    neighbours on an edge-manifold mesh, so no move changes another's answer.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be edge-manifold, since the
        fan around a vertex is what this reasons about.
    region
        ``(n_vertices,)`` boolean mask restricting which vertices may be flattened. ``None``
        flattens every one that qualifies.
    rings
        Optional precomputed [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] as
        ``(ring_halfedges, offsets, is_boundary)``. Depends on the connectivity alone, so one CSR
        serves every fan walk over the same mesh --
        [`Trimesh.vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings] has it cached, and
        passing it skips the vertex-manifold check and the host readback that check costs.

    Returns
    -------
    wp.array[wp.vec3]
        Positions on ``vertices.device``, with the qualifying vertices at their neighbours'
        centroid. Connectivity is untouched, so ``faces`` stays valid.

    Raises
    ------
    ValueError
        If ``region`` is not a length-``n_vertices`` ``wp.bool`` array, or propagated from
        [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when the mesh is not edge-manifold.
    RuntimeError
        If ``vertices``, ``faces``, ``region`` and ``rings`` are not all on one device.

    See Also
    --------
    [`remove_degree3_vertices`][triwarp.repair.remove_degree3_vertices]
        The topological answer to the same defect: delete the vertex instead of moving it.
    [`equalize_triangle_areas`][triwarp.smoothing.equalize_triangle_areas]
        Relaxes every vertex toward an area objective, where this hard-sets only the valence-3 ones.
    """
    require_same_device(vertices=vertices, faces=faces, region=region, rings=rings)
    device = faces.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return wp.clone(vertices)
    if region is None:
        region = wp.full(n_vertices, True, dtype=wp.bool, device=device)
    elif len(region.shape) != 1 or region.shape[0] != n_vertices or region.dtype is not wp.bool:
        raise ValueError(
            f"region must be a length-{n_vertices} wp.bool array, got shape {tuple(region.shape)} "
            f"of {region.dtype}"
        )

    ring_halfedges, ring_offsets, is_boundary = (
        rings if rings is not None else tw.halfedge.vertex_one_rings(faces, n_vertices=n_vertices)
    )
    flattened = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_repair.flatten_degree3_positions,
        dim=n_vertices,
        inputs=[vertices, faces, ring_offsets, ring_halfedges, is_boundary, region, flattened],
        device=device,
    )
    return flattened


def reverse_winding(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Reverse every face's winding, flipping the surface's orientation.

    Rewrites each triangle ``(a, b, c)`` as ``(c, b, a)``, which negates every face normal and the
    enclosed signed volume. Unconditional: unlike
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] and
    [`make_normals_outward`][triwarp.repair.make_normals_outward], which decide per face, this
    flips all of them, so a consistently wound mesh stays consistent and an inconsistent one stays
    inconsistent.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        New face buffer of the same length. ``faces`` is not modified.

    Notes
    -----
    An involution: applying it twice returns the original buffer exactly.

    Each face's signed volume negates *exactly* -- reversing a triangle's corners swaps two
    arguments of a scalar triple product, which negates the same floating-point products rather
    than recomputing them -- so the total negates bit-for-bit wherever the reduction visits faces
    in a fixed order. The example below asserts only the sign, since that is what holds
    independently of the reduction.

    The corner that stays first is a convention, and this one matches ``np.fliplr`` -- which is
    what ``trimesh.Trimesh.invert`` applies, so the two agree elementwise rather than only up to a
    rotation of each row. ``kernels.repair.flip_faces_masked``, which the per-face flippers use,
    keeps corner 0 instead; both reverse orientation and they differ by a cyclic rotation.

    Examples
    --------
    ```python
    flipped = tw.repair.reverse_winding(f)
    assert tw.measures.volume(v, flipped) < 0.0 < tw.measures.volume(v, f)
    ```

    See Also
    --------
    [`triwarp.mesh.Trimesh.invert`][]
        The cached-mesh form, which carries what survives a flip.
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]
    [`triwarp.validation.is_winding_consistent`][]
    """
    reversed_faces = wp.empty(int(faces.shape[0]), dtype=wp.int32, device=faces.device)
    n_faces = int(faces.shape[0]) // 3
    if n_faces > 0:
        wp.launch(
            kernel_repair.reverse_face_winding,
            dim=n_faces,
            inputs=[faces, reversed_faces],
            device=faces.device,
        )
    return reversed_faces


def make_winding_consistent(faces: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Flip faces so every shared edge is traversed in opposite directions by its two faces.

    Reuses the orientation flood-fill of
    [`face_flip_mask`][triwarp.validation.face_flip_mask] (one arbitrary seed
    face per connected component) and reverses the winding of every face whose orientation bit is
    set. The result satisfies
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent] **whenever one exists**,
    which is to say whenever the mesh is
    [`is_orientable`][triwarp.validation.is_orientable]; an already-consistent mesh is returned
    unchanged. Mirrors ``trimesh.repair.fix_winding``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with corrected winding, on ``faces.device``. Vertices are unchanged.

    See Also
    --------
    [`is_winding_consistent`][triwarp.validation.is_winding_consistent]
    [`face_flip_mask`][triwarp.validation.face_flip_mask]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]

    Notes
    -----
    The reference winding within each connected component is arbitrary (the seed face keeps its
    orientation), matching ``trimesh.repair.fix_winding``'s BFS. Use
    [`make_volume`][triwarp.repair.make_volume] afterwards to also orient normals outward.

    On a **non-orientable** mesh no consistent winding exists, so this cannot succeed and does not
    fail either: the flood-fill orients everything it reaches and the contradiction is left on a
    seam. The seam can even end up with *more* inconsistent edges than before the pass ran, not
    merely the same ones — so treat the result as unrepaired rather than partly repaired, and test
    with [`is_orientable`][triwarp.validation.is_orientable] first if that matters.
    """
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    orient, _, _, _ = tw.validation.face_orientation_bits(faces)
    out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.flip_faces_masked,
        dim=n_faces,
        inputs=[faces, orient, out_faces],
        device=device,
    )
    return out_faces


def make_volume(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Orient faces so the mesh encloses a positive signed volume (normals point outward).

    Mirrors ``trimesh.repair.fix_inversion``. With ``multibody=False`` (default) the mesh is only
    corrected when it is watertight (every undirected edge shared by exactly two faces) and its
    total signed volume is negative, in which case every face is reversed. With ``multibody=True``
    each connected component is corrected independently by the sign of its own signed volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    multibody
        When ``True`` correct each connected component independently rather than the mesh as a
        whole.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with outward-oriented normals, on ``faces.device``. Vertices are
        unchanged.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`is_volume`][triwarp.validation.is_volume]
        Test whether this succeeded.
    [`volume`][triwarp.measures.volume]
        Measure the result.
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_normals_outward`][triwarp.repair.make_normals_outward]

    Notes
    -----
    The signed volume is ``sum(dot(v0, cross(v1, v2)) / 6)`` measured from the origin, as in
    [`is_volume`][triwarp.validation.is_volume]. Unlike ``trimesh.repair.fix_inversion``'s
    multibody path, this does not skip components that are not watertight/consistently wound: an
    open component's signed volume is ill-defined and may be flipped spuriously. Run
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] first (see
    [`make_normals_outward`][triwarp.repair.make_normals_outward]) and reserve ``multibody``
    for meshes whose bodies are individually closed.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = int(faces.shape[0]) // 3
    device = faces.device
    if n_faces == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    if multibody:
        out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
        signed_volumes = tw.triangles.face_signed_volumes(vertices, faces)
        labels = tw.adjacency.face_connected_component_labels(faces)
        accum = wp.zeros(n_faces, dtype=wp.float32, device=device)
        wp.launch(
            kernel_scatter.SCATTER_ADD[signed_volumes.dtype],
            dim=n_faces,
            inputs=[signed_volumes, labels, accum],
            device=device,
        )
        flip = wp.empty(n_faces, dtype=wp.int32, device=device)
        # ``accum[labels]`` gathers each face's component volume (Python-scope gather).
        wp.map(kernel_repair.negative_volume_flag, accum[labels], out=flip)
        wp.launch(
            kernel_repair.flip_faces_masked,
            dim=n_faces,
            inputs=[faces, flip, out_faces],
            device=device,
        )
        return out_faces

    # The predicate, not ``all(face_watertight_mask(faces))``: the mask additionally builds
    # ``unique_1d``'s inverse and runs a per-face gather pass, only to be reduced to one bool.
    # Both answer "is every undirected edge shared by exactly two faces", because every unique edge
    # in the table comes from a face.
    if not tw.validation.is_edge_manifold(
        faces, allow_boundary_edges=False, n_vertices=int(vertices.shape[0])
    ):
        return wp.clone(faces)

    signed_volumes = tw.triangles.face_signed_volumes(vertices, faces)
    if tw.reduce.sum(signed_volumes) >= 0.0:
        return wp.clone(faces)

    out_faces = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    flip = wp.full(n_faces, 1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_repair.flip_faces_masked, dim=n_faces, inputs=[faces, flip, out_faces], device=device
    )
    return out_faces


def make_normals_outward(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, multibody: bool = False
) -> wp.array[wp.int32]:
    """
    Make winding consistent and orient normals outward (winding fix followed by inversion fix).

    Equivalent to ``trimesh.repair.fix_normals``: first
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent] gives every connected
    component a coherent winding, then [`make_volume`][triwarp.repair.make_volume] flips it (or each
    body, with ``multibody=True``) so normals point outward. On a watertight, orientable mesh the
    result satisfies [`is_volume`][triwarp.validation.is_volume].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    multibody
        Forwarded to [`make_volume`][triwarp.repair.make_volume]: correct each connected component
        independently.

    Returns
    -------
    wp.array[wp.int32]
        New flat face buffer with consistent winding and outward normals, on ``faces.device``.
        Vertices are unchanged.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
    [`make_volume`][triwarp.repair.make_volume]
    [`is_volume`][triwarp.validation.is_volume]
    """
    require_same_device(vertices=vertices, faces=faces)
    wound = make_winding_consistent(faces)
    return make_volume(vertices, wound, multibody=multibody)


def remove_folded_faces(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, angle: float = 160.0
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Drop faces that fold back over their own ring, and reindex.

    A folded face is one whose dihedral angle to a neighbour is near ``pi``: the two triangles lie
    almost on top of each other with opposite normals, which is what a badly reconstructed or
    self-intersecting patch looks like locally. Such a face contributes no surface and breaks every
    normal-based computation downstream, so removing it is a repair rather than a simplification.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    angle
        Dihedral threshold in **degrees**; a face with a neighbour above it is dropped. MeshLab's
        ``folded_faces_angle_threshold``, whose default of ``160`` is this one. Must be in
        ``(0, 180]``.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Vertices still referenced by a kept face, compacted from index zero.
    new_faces : wp.array[wp.int32]
        Flat buffer of the kept faces, remapped into ``new_vertices``.

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`validation.face_defective_mask`][triwarp.validation.face_defective_mask]
    [`flip_t_vertices`][triwarp.repair.flip_t_vertices]
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces]

    Notes
    -----
    MeshLab's ``meshing_remove_folded_faces`` *flips* the offending edge instead of deleting the
    face, which preserves the face count but can only help when the fold is a triangulation mistake
    rather than genuinely folded geometry. Deletion is the choice the rest of this module makes (see
    [`remove_degenerate_faces`][triwarp.repair.remove_degenerate_faces] and
    [`remove_non_manifold_faces`][triwarp.repair.remove_non_manifold_faces]), and it leaves a hole
    that [`triwarp.holes`][triwarp.holes] can retriangulate properly.
    """
    require_same_device(vertices=vertices, faces=faces)
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)
    folded = tw.validation.face_defective_mask(
        vertices, faces, min_quality=None, max_fold_angle=angle
    )
    keep = wp.empty(n_faces, dtype=wp.bool, device=faces.device)
    wp.map(kernel_array.mask_not, folded, out=keep)
    return tw.selection.submesh_from_face_mask(vertices, faces, keep)


def fix_self_intersections(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    method: Literal["local", "voxel"] = "local",
    max_expand: int = 1,
    max_iter: int = 3,
    voxel_size: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Remove a mesh's self-intersections, either locally or by rebuilding it.

    triwarp could *detect* a self-intersection
    ([`triwarp.validation.face_self_intersecting_mask`][triwarp.validation.face_self_intersecting_mask])
    and not repair one. This is the repair, in the two forms that exist:

    - ``"local"`` cuts the trouble out and rebuilds it. The intersecting faces are dilated by
      ``max_expand`` rings, that region is deleted, and the rims it opens are refilled by the
      minimum-weight patch -- so the surface away from the intersection is **untouched**. Iterated,
      because a patch can intersect something itself.
    - ``"voxel"`` rebuilds the whole surface as the zero level set of its own signed distance field.
      A level set cannot self-intersect, so this always terminates and resamples everything --
      including the parts that were fine. Read the qualification in the Notes: the level set is
      clean, its *triangulation* can still carry an artifact at an ambiguous marching-cubes cell.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    method
        ``"local"`` (default) to cut and refill, ``"voxel"`` to rebuild through a distance field.
    max_expand
        Rings of faces added around each intersecting face before deleting, in the ``"local"``
        method. Larger takes more surface with it and is likelier to succeed in one pass.
    max_iter
        Cap on cut-and-refill passes. The loop also stops as soon as nothing intersects. Raising it
        does **not** help a deep interpenetration and inflates the mesh; see the Notes.
    voxel_size
        Lattice spacing for the ``"voxel"`` method. ``None`` uses 1/128 of the bounding-box
        diagonal.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``vertices.device``. A clean input is returned as a copy.

    Raises
    ------
    ValueError
        If ``method`` is not ``"local"`` or ``"voxel"``, ``max_expand`` is negative, or ``max_iter``
        is less than 1.
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    Examples
    --------
    ```python
    clean_v, clean_f = tw.repair.fix_self_intersections(v, f)
    ```

    Notes
    -----
    **Success is not guaranteed by either method, and neither is asserted.** For ``"local"``: a
    region whose rim cannot be triangulated without crossing something, or one that grows to swallow
    the mesh, leaves intersections behind; the loop stops at ``max_iter`` and returns what it has.
    For ``"voxel"``: the level set is clean, but Warp's ``MarchingCubes`` can emit a touching or
    non-manifold pair at an ambiguous cell, and that is resolution-dependent: a finer lattice can
    introduce a handful of such faces where a coarser one has none. So check with
    [`triwarp.validation.is_self_intersecting`][triwarp.validation.is_self_intersecting] when it
    matters. Stating this is better than a loop that cannot terminate, and better than a promise the
    extraction does not keep.

    **What the ``"local"`` method is for.** It clears a *shallow* self-intersection outright -- a
    torus whose tube passes through itself can be fully cleared at either dilation budget -- and
    only reduces a *deep* one, such as two icospheres overlapping by a third of their diameter. That
    is the method's shape rather than a tuning failure. Cutting out a lens-shaped overlap leaves a
    rim whose minimum-weight patch runs back through the other shell, so the pass converges only
    where the damage is a band. Reach for ``"voxel"`` when two closed pieces genuinely
    interpenetrate: a level set has no notion of two shells.

    **On that input class the result is nondeterministic and ``max_iter`` is not a quality knob**,
    which "only reduces" does not by itself tell you. Past a certain point, raising it does not
    reduce the residual intersections further while the face count climbs past the input's -- each
    pass is refilling a rim that the next one cuts out again. Repeated runs on the same input can
    also produce a slightly different result, from the refill chain's own atomic-ordering
    nondeterminism (as in
    [`triwarp.remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh]), so that variation is not
    a regression. Do not raise ``max_iter`` hoping for convergence on a deep interpenetration; the
    answer is ``"voxel"``.

    The two methods differ in what they preserve, not in quality. ``"local"`` keeps the input's
    triangulation everywhere it did not cut, so a per-vertex attribute survives outside the patch;
    ``"voxel"`` keeps nothing but the shape, and its accuracy is the lattice's.

    See Also
    --------
    [`triwarp.validation.face_self_intersecting_mask`][triwarp.validation.face_self_intersecting_mask]
        The detector, and what to check the result with.
    [`triwarp.holes.refill_region`][triwarp.holes.refill_region]
        The cut-and-refill step each ``"local"`` pass runs.
    [`triwarp.levelset.offset_mesh`][triwarp.levelset.offset_mesh]
        The same level-set machinery at a non-zero distance.
    """
    require_same_device(vertices=vertices, faces=faces)
    if method not in ("local", "voxel"):
        raise ValueError(f"method must be 'local' or 'voxel', got {method!r}")
    if max_expand < 0:
        raise ValueError("max_expand must be non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    if int(faces.shape[0]) == 0:
        return wp.clone(vertices), wp.clone(faces)

    if method == "voxel":
        spacing = voxel_size
        if spacing is None:
            spacing = float(tw.bounds.enclosing_diagonal(vertices)) / _VOXEL_REBUILD_RESOLUTION
        field, box = tw.proximity.signed_distance_grid(
            vertices, faces, spacing, pad=2, sign_mode="winding"
        )
        return tw.levelset.marching_cubes(field, 0.0, bounds=box)

    current_vertices, current_faces = wp.clone(vertices), wp.clone(faces)
    for _ in range(max_iter):
        bad_mask = tw.validation.face_self_intersecting_mask(current_vertices, current_faces)
        # Two readbacks per pass, and each decides the loop. Deliberately *not* ``tw.reduce.any`` /
        # ``tw.reduce.all``: a device reduction has a roughly fixed cost, while copying a ``bool``
        # array of length ``n_faces`` is cheaper below roughly a million faces, which covers
        # ordinary mesh sizes. Revisit if that stops being true for the meshes this runs on.
        if not bool(bad_mask.numpy().any()):
            break
        region = _dilate_face_mask(
            current_faces, bad_mask, max_expand, int(current_vertices.shape[0])
        )
        if bool(region.numpy().all()):
            break  # the region swallowed the mesh: refilling it would delete everything
        current_vertices, current_faces = tw.holes.refill_region(
            current_vertices, current_faces, region
        )
        if int(current_faces.shape[0]) == 0:
            break
    return current_vertices, current_faces


def _dilate_face_mask(
    faces: wp.array[wp.int32], face_mask: wp.array[wp.bool], hops: int, n_vertices: int
) -> wp.array[wp.bool]:
    """
    Grow a face selection by ``hops`` rings, through the vertices it touches.

    Face adjacency is not needed for this and is not built: a face ring is the faces incident on
    the selection's vertex ring, so the growth happens on the *vertex* mask -- where
    [`triwarp.selection.expand_vertex_mask`][triwarp.selection.expand_vertex_mask] already does
    it -- and is mapped back with ``face_mode="any"``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    face_mask
        Length-``n_faces`` selection to grow.
    hops
        Rings to add. Zero returns the selection's own faces, which is *not* the input mask: it is
        every face sharing a vertex with it, since a cut has to leave a rim rather than a slit.
    n_vertices
        Length of the vertex buffer ``faces`` indexes, supplied by the caller. Not inferred from
        ``faces.max()``: that is a whole-buffer readback, and ``fix_self_intersections`` calls this
        once per pass, to recover a number the caller is already holding -- which is what
        ``face_adjacency(n_vertices=...)`` exists to avoid.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` grown selection on ``faces.device``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3

    selected_corners = tw.array.gather(
        faces.reshape((-1, 3)), tw.array.flatnonzero(face_mask)
    ).reshape((-1,))
    vertex_mask = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    wp.launch(
        kernel_scatter.mark_membership_mask,
        dim=int(selected_corners.shape[0]),
        inputs=[selected_corners, wp.int32(n_vertices), vertex_mask],
        device=device,
    )
    if hops > 0:
        vertex_mask = tw.selection.expand_vertex_mask(faces, vertex_mask, hops)

    grown_faces = tw.selection.face_indices_from_vertex_indices(
        faces, tw.array.flatnonzero(vertex_mask), face_mode="any", n_vertices=n_vertices
    )
    return tw.array.indices_to_mask(grown_faces, n_faces, device=device)


def remove_tunnels(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_length: float,
    *,
    metric: str = "plane_normalized",
    max_iter: int = 100,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], int]:
    """
    Remove thin handles by cutting along their short tunnel loops and sealing the two rims.

    A handle is a genus the surface did not need: a scanning artifact where two sheets fused, or a
    reconstruction that bridged across a gap. Its signature is a **short non-contractible loop** --
    short being the whole test, since every genus of the intended shape has loops the size of the
    shape. So: take a homology basis, shorten each loop within its class
    ([`shorten_loop`][triwarp.geodesic_walk.shorten_loop]), keep the ones that come in under
    ``max_length``, cut along those and fill the boundary loops the cut opens. Cutting a surface
    along a non-separating cycle and sealing the two rims it creates drops the genus by exactly one,
    so ``2 * removed`` is the rise in
    [`euler_characteristic`][triwarp.measures.euler_characteristic] -- verified rather than
    reported, and the invariant to assert if you extend this.

    Nothing is removed when no loop is short enough, and the input is returned unchanged -- so this
    is safe to run on a mesh whose genus is intended, provided ``max_length`` is below the scale of
    its real handles.

    !!! note "One disjoint pass per call"
        The loops kept are pairwise **vertex-disjoint**, shortest first. Cutting along two loops
        that cross is not the same operation as cutting along each in turn -- the shared vertex is
        split by both cuts at once -- and without the restriction the genus can stop dropping one
        per loop, or the surface can shatter into extra pieces. The cost is that one call removes at
        most one tunnel per disjoint family, so a mesh whose basis loops all overlap needs to be run
        again. Call it in a loop until ``removed`` is ``0``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Must be a closed, connected,
        edge-manifold surface, which is what the homology basis needs.
    max_length
        Loops at or under this length are removed. It is an absolute length in the mesh's own
        units, so scale it off something intrinsic -- the mean edge length times the number of
        triangles a real handle would take to go round.
    metric
        Triangulation metric for the two rims, as
        [`fill_min_weight`][triwarp.holes.fill_min_weight] takes it.
    max_iter
        Sweep cap handed to [`shorten_loop`][triwarp.geodesic_walk.shorten_loop]. Shortening is what
        makes the length test meaningful: a tree-cotree loop around a thin handle can be many times
        the handle's own girth, so an unshortened basis under-reports every tunnel.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Positions of the result. Longer than the input's wherever the cut split a vertex; the
        existing positions are unchanged and no new position is invented, since the rims are filled
        over their own vertices.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer, the cut mesh plus the fill triangles.
    removed : int
        How many loops were cut. Zero means nothing was short enough, and the buffers are the
        input's.

        Returned **unconditionally**, unlike the diagnostic counts on
        [`straighten_boundary`][triwarp.repair.straighten_boundary] and
        [`remove_degree3_vertices`][triwarp.repair.remove_degree3_vertices], which sit behind
        a ``return_count`` keyword. This one is part of the answer rather than a report on it: one
        call removes at most one tunnel per disjoint family, so the documented usage is to loop
        until it reads zero, and a caller who cannot see it cannot use the function correctly.
        [`remesh.intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay]'s iteration count is
        unconditional for the same reason.

    Raises
    ------
    ValueError
        If ``max_length`` is negative, or the mesh has a boundary (a surface with boundary has a
        different homology basis, so "tunnel" is not defined by this test).
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`shorten_loop`][triwarp.geodesic_walk.shorten_loop]
        Makes the length test meaningful, and is where the loops come from.
    [`homology_generators`][triwarp.homology.homology_generators]
    [`fix_self_intersections`][triwarp.repair.fix_self_intersections]
        The other topological repair here: that one removes crossings, this one removes genus.
    """
    require_same_device(vertices=vertices, faces=faces)
    if max_length < 0.0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    device = faces.device
    loops = tw.homology.homology_generators(vertices, faces)
    if not loops:
        return vertices, faces, 0

    shortened, _sweeps = tw.geodesic_walk.shorten_loop(vertices, faces, loops, max_iter=max_iter)
    # One readback per loop, and the loops are the only thing being measured: a basis has 2 * genus
    # of them and each is a handful of indices, so this never scales with the mesh.
    short = sorted(
        ((_cycle_length(vertices, loop), loop) for loop in shortened), key=lambda pair: pair[0]
    )
    selected = _disjoint_loops([loop for length, loop in short if length <= max_length])
    if not selected:
        return vertices, faces, 0

    cut_edges = wp.array(
        np.concatenate([_cycle_edges(loop) for loop in selected]), dtype=wp.int32, device=device
    )
    cut_vertices, cut_faces = tw.seams.cut_along_edges(
        vertices, faces, twt.as_array2d(cut_edges, wp.int32)
    )
    return (
        cut_vertices,
        tw.holes.fill_min_weight(cut_vertices, cut_faces, metric=metric),
        len(selected),
    )


def _disjoint_loops(loops: list[wp.array[wp.int32]]) -> list[wp.array[wp.int32]]:
    """
    Greedily keep the loops that share no vertex, taking them shortest first.

    Cutting along two loops that *cross* is not the same operation as cutting along each in turn:
    the shared vertex is split by both cuts at once, and the genus can stop dropping by one per
    loop, or cutting the whole basis can shatter the surface into several pieces. Keeping the
    selection pairwise disjoint is what makes ``removed`` mean what it says.
    """
    claimed: set[int] = set()
    kept: list[wp.array[wp.int32]] = []
    for loop in loops:
        loop_indices = {int(index) for index in loop.numpy()}
        if loop_indices & claimed:
            continue
        claimed |= loop_indices
        kept.append(loop)
    return kept


def _cycle_length(vertices: wp.array[wp.vec3], loop: wp.array[wp.int32]) -> float:
    """Length of a closed vertex-index cycle, gathered onto its positions."""
    points = wp.empty(int(loop.shape[0]), dtype=wp.vec3, device=vertices.device)
    wp.copy(points, vertices[loop])
    return tw.polyline.polyline_length(points, closed=True)


def _cycle_edges(loop: wp.array[wp.int32]) -> np.ndarray:
    """Pack a closed cycle's edges as ascending ``(k, 2)`` rows, which is what a cut keys on."""
    loop_np = loop.numpy()
    return np.sort(np.stack([loop_np, np.roll(loop_np, -1)], axis=1), axis=1).astype(np.int32)


def flip_t_vertices(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    threshold: float = 40.0,
    max_iter: int = 10,
) -> wp.array[wp.int32]:
    """
    Repair T-vertices by flipping the long edge of each sliver they create.

    A **T-vertex** is a vertex that sits in the interior of a neighbouring triangle's edge rather
    than at one of its corners — the classic symptom of two patches stitched at different
    resolutions. The vertex is topologically fine, but the triangle opposite it is a sliver: its
    apex lies (nearly) on the far edge, which sends its circumradius-to-inradius ratio to infinity
    and makes every cotangent weight, normal and curvature estimate around it unusable.

    The repair is a flip, not a deletion: flipping the sliver's long edge moves the diagonal off the
    T and leaves two well-shaped triangles, with the same vertices and the same face count.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions. Never modified.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    threshold
        Aspect ratio above which a triangle counts as a T-vertex sliver, in the
        ``aspect_ratio`` sense of [`face_quality`][triwarp.triangles.face_quality] (``1`` is
        equilateral, unbounded above). MeshLab's ``meshing_remove_t_vertices`` threshold, whose
        default of ``40`` is this one. Must be positive.
    max_iter
        Maximum number of parallel flip passes. Each pass commits a conflict-free independent set of
        flips; MeshLab's ``repeat=True`` is the same idea serially.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer with the slivers re-triangulated, on ``faces.device`` (a copy; the input is
        not modified).

    Raises
    ------
    RuntimeError
        If ``vertices`` and ``faces`` are not all on one device.

    See Also
    --------
    [`triwarp.remesh.flip_by_objective`][triwarp.remesh.flip_by_objective]
    [`remove_folded_faces`][triwarp.repair.remove_folded_faces]
    [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles]

    Notes
    -----
    A flip cannot fix a T-vertex on the mesh **boundary** or on a non-manifold edge, because there
    is no second triangle to flip against. MeshLab offers an edge *collapse* method for that case;
    here the equivalent is [`collapse_small_triangles`][triwarp.repair.collapse_small_triangles],
    which removes the sliver by merging its short edge instead.
    """
    require_same_device(vertices=vertices, faces=faces)
    return tw.remesh.flip_by_objective(
        vertices, faces, objective="t_vertex", aspect_threshold=threshold, max_iter=max_iter
    )
