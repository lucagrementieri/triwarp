from typing import Any

import warp as wp

from triwarp.kernels.array import binary_search_sorted_contains, cross2
from triwarp.kernels.grouping import pack_edge_key


@wp.func
def halfedge_next(h: wp.int32) -> wp.int32:
    # Next halfedge inside the same face, under the ``h = 3 * f + k`` convention of
    # ``triwarp.halfedge``: index arithmetic, no structure to look up.
    return h - h % 3 + (h + 1) % 3


@wp.kernel
def corner_union_edges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    marked_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_edges: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # Two graph edges per *uncut* interior mesh edge, joining the two face corners that meet at each
    # of its two endpoints. Connected components of that graph are exactly the copies each vertex
    # needs: a vertex whose whole fan is uncut stays one vertex, and every marked edge crossing the
    # fan splits it.
    #
    # Both endpoints matter, and getting only one of them wrong is silent: the fan around a vertex
    # is then connected by half its edges and every vertex splits into two.
    #
    # ``h < twin`` emits each undirected edge once. For the halfedge ``h: u -> v`` with twin
    # ``t: v -> u``, the corners at ``u`` are ``h`` and ``next(t)``, and the corners at ``v`` are
    # ``next(h)`` and ``t`` -- the twin runs the other way, so its *following* halfedge is the one
    # starting where ``h`` does.
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin < 0 or twin < h:
        return
    if binary_search_sorted_contains(
        marked_keys, pack_edge_key(faces[h], faces[halfedge_next(h)], key_base)
    ):
        return
    slot = wp.atomic_add(out_count, 0, 2)
    out_edges[slot, 0] = h
    out_edges[slot, 1] = halfedge_next(twin)
    out_edges[slot + 1, 0] = halfedge_next(h)
    out_edges[slot + 1, 1] = twin


@wp.kernel
def scatter_corner_values(
    faces: wp.array[wp.int32],
    corner_index: wp.array[wp.int32],
    values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # Position (or any per-vertex attribute) of each output copy, gathered through the corner it
    # came from. Every corner in a component writes the *same* value, so the race is benign by
    # construction and no atomics are needed.
    h = wp.int32(wp.tid())
    out_values[corner_index[h]] = values[faces[h]]


@wp.func
def crease_edge_mask(adjacency_angle: wp.float32, threshold: wp.float32) -> wp.bool:
    # Strictly greater, so a threshold of 0 selects every non-coplanar interior edge.
    return adjacency_angle > threshold


@wp.func
def halfedge_opposite_corner(h: wp.int32) -> wp.int32:
    # The corner of ``h``'s face that ``h`` does *not* touch -- igl's ``(i + 2) % 3``.
    return halfedge_next(halfedge_next(h))


@wp.func
def texcoords_differ(
    texcoords: wp.array[wp.vec2], first: wp.int32, second: wp.int32, tolerance_sq: wp.float32
) -> wp.bool:
    # Coordinate comparison, MeshLab's predicate. ``tolerance_sq == 0`` makes this exact inequality,
    # which is what VCG's ``TexCoord2f::operator==`` does.
    return wp.length_sq(texcoords[first] - texcoords[second]) > tolerance_sq


@wp.kernel
def classify_uv_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    out_is_seam: wp.array[wp.bool],
    out_is_boundary: wp.array[wp.bool],
    out_is_foldover: wp.array[wp.bool],
    out_quads: wp.array2d[wp.int32],
) -> None:
    # One thread per halfedge; the pair ``(h, twin)`` is classified by whichever of the two has the
    # smaller index, so each undirected edge is decided exactly once and the compacted output comes
    # out in ascending canonical-halfedge order. Every slot is written on every path, so the caller
    # may allocate the outputs with ``wp.empty``.
    h = wp.int32(wp.tid())
    twin = twins[h]

    out_is_seam[h] = False
    out_is_boundary[h] = False
    out_is_foldover[h] = False
    out_quads[h, 0] = 0
    out_quads[h, 1] = 0
    out_quads[h, 2] = 0
    out_quads[h, 3] = 0

    if twin < 0:
        # A boundary row is just this halfedge's own ``(face, corner)``; ``boundary_face_corners``
        # recovers it from the compacted index, so no quad is needed here.
        out_is_boundary[h] = True
        return
    if twin < h:
        return

    # igl's canonical direction: the halfedge running from the smaller vertex index to the larger.
    # A degenerate edge (both endpoints equal) has no such direction; keeping ``h`` forwards is
    # arbitrary but deterministic.
    forwards = h
    backwards = twin
    if faces[h] > faces[halfedge_next(h)]:
        forwards = twin
        backwards = h

    out_quads[h, 0] = forwards // 3
    out_quads[h, 1] = forwards % 3
    out_quads[h, 2] = backwards // 3
    out_quads[h, 3] = backwards % 3

    # ``backwards`` runs the other way, so the corner sitting on top of ``forwards``' tail is the
    # one *following* ``backwards``, and vice versa.
    tail_forwards = face_texcoords[forwards]
    head_forwards = face_texcoords[halfedge_next(forwards)]
    tail_backwards = face_texcoords[halfedge_next(backwards)]
    head_backwards = face_texcoords[backwards]

    if match_uv:
        is_seam = texcoords_differ(
            texcoords, tail_forwards, tail_backwards, tolerance_sq
        ) or texcoords_differ(texcoords, head_forwards, head_backwards, tolerance_sq)
    else:
        is_seam = tail_forwards != tail_backwards or head_forwards != head_backwards
    out_is_seam[h] = is_seam
    if is_seam:
        return

    # Matched texcoords, so both triangles agree on where the shared edge lands in UV space. They
    # fold over each other exactly when their two opposite corners land on the *same* side of it.
    # Strictly the same side: a collinear corner is a degenerate UV triangle, not a foldover.
    a = texcoords[tail_forwards]
    b = texcoords[head_forwards]
    c_forwards = texcoords[face_texcoords[halfedge_opposite_corner(forwards)]]
    c_backwards = texcoords[face_texcoords[halfedge_opposite_corner(backwards)]]
    orientation_forwards = cross2(a - c_forwards, b - c_forwards)
    orientation_backwards = cross2(a - c_backwards, b - c_backwards)
    out_is_foldover[h] = (orientation_forwards > 0.0 and orientation_backwards > 0.0) or (
        orientation_forwards < 0.0 and orientation_backwards < 0.0
    )


@wp.kernel
def boundary_face_corners(
    halfedges: wp.array[wp.int32], out_face_corners: wp.array2d[wp.int32]
) -> None:
    # Compacted boundary halfedge indices back into ``(face, corner)`` rows, under the
    # ``h = 3 * f + k`` convention.
    i = wp.int32(wp.tid())
    out_face_corners[i, 0] = halfedges[i] // 3
    out_face_corners[i, 1] = halfedges[i] % 3


@wp.kernel
def face_corner_edge_vertices(
    faces: wp.array[wp.int32], face_corners: wp.array2d[wp.int32], out_edges: wp.array2d[wp.int32]
) -> None:
    # ``(face, corner)`` provenance back to the vertex pair it names. Columns 0 and 1 are the
    # forward side of a seam/foldover row and the whole of a boundary row, so one kernel serves all
    # three blocks.
    i = wp.int32(wp.tid())
    h = 3 * face_corners[i, 0] + face_corners[i, 1]
    out_edges[i, 0] = faces[h]
    out_edges[i, 1] = faces[halfedge_next(h)]


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. One overload across **2** module loads: the smallest fork in the package, and
# registered for the same reason the others are -- so that adding a second dtype later cannot
# quietly reintroduce one.
#
# ``cut_mesh_from_seams`` scatters the corner *positions* it is splitting, so the value dtype is the
# vertex dtype and nothing else reaches this kernel.
def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in (wp.vec3, wp.vec3d):
        wp.overload(
            scatter_corner_values,
            [wp.array[wp.int32], wp.array[wp.int32], wp.array[dtype], wp.array[dtype]],
        )


_register_overloads()
