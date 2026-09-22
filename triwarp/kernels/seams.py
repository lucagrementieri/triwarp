from typing import Any

import warp as wp

from triwarp.kernels.array import (
    OverloadTable,
    binary_search_sorted_contains,
    cross2,
    pack_edge_key,
)
from triwarp.kernels.halfedge import halfedge_next, halfedge_prev


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
    # ``h < twin`` emits each undirected edge once. ``halfedge_twins`` pairs halfedges by their
    # *undirected* endpoint set alone (no direction check), so ``twin`` can run either opposite
    # ``h`` (the consistently-wound case) or the same way as ``h`` (an edge-manifold but
    # inconsistently-wound mesh -- this module's own ``uv_seam_edges`` Notes name that as a real,
    # supported divergence, not an excluded input). Both cases have to be handled, or the two
    # corners unioned belong to two different original vertices.
    #
    # For ``h: u -> v``: if ``twin`` runs ``v -> u`` (opposite), the corners at ``u`` are ``h`` and
    # ``next(twin)`` and the corners at ``v`` are ``next(h)`` and ``twin``. If ``twin`` runs
    # ``u -> v`` (same direction as ``h``), the corners at ``u`` are ``h`` and ``twin`` themselves,
    # and at ``v`` are ``next(h)`` and ``next(twin)``.
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin < 0 or twin < h:
        return
    if binary_search_sorted_contains(
        marked_keys, pack_edge_key(faces[h], faces[halfedge_next(h)], key_base)
    ):
        return
    slot = wp.atomic_add(out_count, 0, 2)
    if faces[twin] == faces[h]:
        # Same-direction twin: each halfedge's own corner is at the shared origin ``u``.
        out_edges[slot, 0] = h
        out_edges[slot, 1] = twin
        out_edges[slot + 1, 0] = halfedge_next(h)
        out_edges[slot + 1, 1] = halfedge_next(twin)
    else:
        # Opposite-direction twin (the consistently-wound case).
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
def texcoords_differ(
    texcoords: wp.array[wp.vec2], first: wp.int32, second: wp.int32, tolerance_sq: wp.float32
) -> wp.bool:
    # Coordinate comparison, MeshLab's predicate. ``tolerance_sq == 0`` makes this exact inequality,
    # which is what VCG's ``TexCoord2f::operator==`` does.
    return wp.length_sq(texcoords[first] - texcoords[second]) > tolerance_sq


# The class ``classify_uv_halfedge`` assigns a halfedge. Only the canonical (lower-index) half of an
# interior edge carries a seam or foldover verdict, so every undirected edge is decided once.
UV_EDGE_NONE = wp.constant(wp.int32(0))
UV_EDGE_SEAM = wp.constant(wp.int32(1))
UV_EDGE_BOUNDARY = wp.constant(wp.int32(2))
UV_EDGE_FOLDOVER = wp.constant(wp.int32(3))


@wp.func
def corner_texcoord(
    face_texcoords: wp.array[wp.int32], has_face_texcoords: wp.bool, h: wp.int32
) -> wp.int32:
    # The texcoord index of corner ``h``. Without a corner-to-texcoord table the texcoords are
    # per-corner, so the index *is* the corner and the table is never read -- which is what lets
    # the caller pass ``None`` for it instead of materialising ``arange(3 * n_faces)``.
    if has_face_texcoords:
        return face_texcoords[h]
    return h


@wp.func
def canonical_edge_halfedges(
    faces: wp.array[wp.int32], h: wp.int32, twin: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # igl's canonical direction: the halfedge running from the smaller vertex index to the larger.
    # A degenerate edge (both endpoints equal) has no such direction; keeping ``h`` forwards is
    # arbitrary but deterministic.
    if faces[h] > faces[halfedge_next(h)]:
        return twin, h
    return h, twin


@wp.func
def classify_uv_halfedge(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    h: wp.int32,
) -> wp.int32:
    # The one seam / boundary / foldover decision, shared by the per-edge compaction and the
    # per-vertex mask so the two cannot disagree about which edges are seams. The pair
    # ``(h, twin)`` is classified by whichever of the two has the smaller index; the other half
    # answers ``UV_EDGE_NONE``.
    twin = twins[h]
    if twin < 0:
        return UV_EDGE_BOUNDARY
    if twin < h:
        return UV_EDGE_NONE
    forwards, backwards = canonical_edge_halfedges(faces, h, twin)

    # ``halfedge_twins`` pairs ``h``/``twin`` by their undirected endpoint set alone, with no
    # direction check, so ``backwards`` runs opposite ``forwards`` only on a consistently-wound
    # mesh -- this module's own Notes name inconsistent winding as a real, supported divergence, not
    # an excluded input. When it runs the *other* way, the corner sitting on top of ``forwards``'
    # tail is the one *following* ``backwards``, and vice versa; when it runs the *same* way, that
    # corner is ``backwards`` itself, and the one on top of ``forwards``' head is the one following
    # it. ``faces[backwards] == faces[forwards]`` is exactly the same-direction case, since both
    # then share the same origin vertex.
    tail_forwards = corner_texcoord(face_texcoords, has_face_texcoords, forwards)
    head_forwards = corner_texcoord(face_texcoords, has_face_texcoords, halfedge_next(forwards))
    if faces[backwards] == faces[forwards]:
        tail_backwards = corner_texcoord(face_texcoords, has_face_texcoords, backwards)
        head_backwards = corner_texcoord(
            face_texcoords, has_face_texcoords, halfedge_next(backwards)
        )
    else:
        tail_backwards = corner_texcoord(
            face_texcoords, has_face_texcoords, halfedge_next(backwards)
        )
        head_backwards = corner_texcoord(face_texcoords, has_face_texcoords, backwards)

    if match_uv:
        is_seam = texcoords_differ(
            texcoords, tail_forwards, tail_backwards, tolerance_sq
        ) or texcoords_differ(texcoords, head_forwards, head_backwards, tolerance_sq)
    else:
        is_seam = tail_forwards != tail_backwards or head_forwards != head_backwards
    if is_seam:
        return UV_EDGE_SEAM

    # Matched texcoords, so both triangles agree on where the shared edge lands in UV space. They
    # fold over each other exactly when their two opposite corners land on the *same* side of it.
    # Strictly the same side: a collinear corner is a degenerate UV triangle, not a foldover.
    a = texcoords[tail_forwards]
    b = texcoords[head_forwards]
    c_forwards = texcoords[
        corner_texcoord(face_texcoords, has_face_texcoords, halfedge_prev(forwards))
    ]
    c_backwards = texcoords[
        corner_texcoord(face_texcoords, has_face_texcoords, halfedge_prev(backwards))
    ]
    orientation_forwards = cross2(a - c_forwards, b - c_forwards)
    orientation_backwards = cross2(a - c_backwards, b - c_backwards)
    if (orientation_forwards > 0.0 and orientation_backwards > 0.0) or (
        orientation_forwards < 0.0 and orientation_backwards < 0.0
    ):
        return UV_EDGE_FOLDOVER
    return UV_EDGE_NONE


@wp.kernel
def classify_uv_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    out_flags: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # One thread per halfedge. Row ``k`` of ``out_flags`` is the 0/1 selection of class ``k + 1``
    # (seam, boundary, foldover), laid out so one inclusive scan over the flattened buffer
    # numbers all three blocks at once; ``out_counts`` totals each class so a single readback sizes
    # all three outputs. Every flag is written on every path, so the caller may allocate
    # ``out_flags`` with ``wp.empty``; ``out_counts`` arrives zeroed.
    h = wp.int32(wp.tid())
    kind = classify_uv_halfedge(
        faces, twins, face_texcoords, has_face_texcoords, texcoords, match_uv, tolerance_sq, h
    )
    for k in range(3):
        out_flags[k, h] = wp.where(kind == k + 1, wp.int32(1), wp.int32(0))
    if kind != UV_EDGE_NONE:
        wp.atomic_add(out_counts, kind - 1, 1)


@wp.kernel
def scatter_uv_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    flags: wp.array2d[wp.int32],
    inclusive: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
    out_seams: wp.array2d[wp.int32],
    out_boundaries: wp.array2d[wp.int32],
    out_foldovers: wp.array2d[wp.int32],
) -> None:
    # Compact the three classes into their rows in ascending halfedge order. ``inclusive`` is the
    # inclusive scan of the *flattened* ``flags``, so block ``k``'s positions run on from the
    # totals of the blocks before it; subtracting those totals makes each block zero-based. A seam
    # or foldover row is the ``(face, corner)`` pair of both canonical halfedges; a boundary row is
    # this halfedge's own ``(face, corner)``, under the ``h = 3 * f + k`` convention.
    h = wp.int32(wp.tid())
    if flags[1, h] != 0:
        row = inclusive[1, h] - 1 - counts[0]
        out_boundaries[row, 0] = h // 3
        out_boundaries[row, 1] = h % 3
        return
    is_seam = flags[0, h] != 0
    if not is_seam and flags[2, h] == 0:
        return
    forwards, backwards = canonical_edge_halfedges(faces, h, twins[h])
    if is_seam:
        row = inclusive[0, h] - 1
        out_seams[row, 0] = forwards // 3
        out_seams[row, 1] = forwards % 3
        out_seams[row, 2] = backwards // 3
        out_seams[row, 3] = backwards % 3
    else:
        row = inclusive[2, h] - 1 - counts[0] - counts[1]
        out_foldovers[row, 0] = forwards // 3
        out_foldovers[row, 1] = forwards % 3
        out_foldovers[row, 2] = backwards // 3
        out_foldovers[row, 3] = backwards % 3


@wp.kernel
def mark_uv_seam_vertices(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    face_texcoords: wp.array[wp.int32],
    has_face_texcoords: wp.bool,
    texcoords: wp.array[wp.vec2],
    match_uv: wp.bool,
    tolerance_sq: wp.float32,
    include_boundary: wp.bool,
    out_mask: wp.array[wp.bool],
) -> None:
    # The per-vertex seam mask straight from the classification: both endpoints of every seam
    # halfedge (and of every boundary halfedge, when asked). A seam row's forward halfedge and
    # ``h`` span the same vertex pair, so marking ``h``'s own endpoints is the same answer as
    # decoding the row. Out-of-range endpoints are skipped, as ``array.indices_to_mask`` does;
    # every write stores ``True``, so the race between two halfedges on one vertex is benign.
    h = wp.int32(wp.tid())
    kind = classify_uv_halfedge(
        faces, twins, face_texcoords, has_face_texcoords, texcoords, match_uv, tolerance_sq, h
    )
    if kind == UV_EDGE_SEAM or (include_boundary and kind == UV_EDGE_BOUNDARY):
        n = out_mask.shape[0]
        for endpoint in range(2):
            v = faces[wp.where(endpoint == 0, h, halfedge_next(h))]
            if v >= 0 and v < n:
                out_mask[v] = True


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
# CLAUDE.md section 2.5. Two overloads across **2** module loads: the smallest fork in the package.
#
# ``triwarp.seams.cut_along_edges`` scatters the corner *positions* it is splitting, so the value
# dtype is the vertex dtype it was handed -- ``wp.vec3`` or ``wp.vec3d`` -- and nothing else reaches
# this kernel. Both are registered because the wrapper's own dispatch (``SCATTER_CORNER_VALUES[
# vertices.dtype]``) can reach either one.
# The concrete handle keyed by the value dtype -- see
# [`OverloadTable`][triwarp.kernels.array.OverloadTable].
SCATTER_CORNER_VALUES: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global SCATTER_CORNER_VALUES
    SCATTER_CORNER_VALUES = OverloadTable(
        scatter_corner_values,
        {
            d: [wp.array[wp.int32], wp.array[wp.int32], wp.array[d], wp.array[d]]
            for d in (wp.vec3, wp.vec3d)
        },
    )


_register_overloads()
