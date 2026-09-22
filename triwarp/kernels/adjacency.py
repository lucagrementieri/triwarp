import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT
from triwarp.kernels.array import pack_edge_key
from triwarp.kernels.predicates import vector_angle
from triwarp.kernels.triangles import corner_triple


@wp.func
def write_face_edge_keys(
    faces: wp.array[wp.int32], f: wp.int32, base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # The three undirected edge keys of face ``f``, written at ``3f .. 3f + 2``. Edge ``3f + k``
    # belongs to face ``f``, which is what lets ``edge_pairs_to_face_pairs`` and
    # ``edge_endpoints`` recover everything else from an edge index alone.
    #
    # ``pack_edge_key`` is byte-identical to what ``pack_indices`` produces for the sorted edge row
    # ``[min, max]``, which is what keeps the radix sort's key order -- and so the adjacency row
    # order -- the same as the composed ``faces_to_edges`` + ``pack_indices`` path.
    c = f * 3
    i0 = faces[c + 0]
    i1 = faces[c + 1]
    i2 = faces[c + 2]
    out_keys[c + 0] = pack_edge_key(i0, i1, base)
    out_keys[c + 1] = pack_edge_key(i1, i2, base)
    out_keys[c + 2] = pack_edge_key(i2, i0, base)


@wp.kernel
def face_edge_keys(
    faces: wp.array[wp.int32], base: wp.uint64, out_keys: wp.array[wp.uint64]
) -> None:
    # One launch in place of ``faces_to_edges`` + ``pack_indices``, so the intermediate
    # ``(3F, 2)`` edge rows are never materialized. ``remesh.pass_edge_keys`` is the same kernel
    # over a fixed-capacity buffer, differing only in writing a sentinel key past the live faces.
    write_face_edge_keys(faces, wp.tid(), base, out_keys)


@wp.func
def edge_endpoints(faces: wp.array[wp.int32], edge_index: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The sorted endpoints of edge ``3f + c``, recovered from the edge index alone. Corner ``c`` of
    # face ``f`` spans ``(v[c], v[(c + 1) % 3])``, matching ``kernels/edges.py:faces_to_edges``, so
    # this reproduces exactly the row ``faces_to_edges(sorted=True)`` would have written there.
    face_base = (edge_index // 3) * 3
    corner = edge_index % 3
    a = faces[face_base + corner]
    b = faces[face_base + (corner + 1) % 3]
    return wp.min(a, b), wp.max(a, b)


@wp.func
def write_face_pair(
    edge_groups: wp.array2d[wp.int32], row: wp.int32, out_adjacency: wp.array2d[wp.int32]
) -> None:
    # Two edge indices sharing a key -> the ascending pair of faces owning them. Replaces a gather
    # through a materialized ``edges_face`` table plus an in-place row sort: the owning face of
    # edge ``e`` is just ``e // 3``, and ordering two values needs no sort kernel.
    f0 = edge_groups[row, 0] // 3
    f1 = edge_groups[row, 1] // 3
    out_adjacency[row, 0] = wp.min(f0, f1)
    out_adjacency[row, 1] = wp.max(f0, f1)


@wp.kernel
def edge_pairs_to_face_pairs(
    edge_groups: wp.array2d[wp.int32], out_adjacency: wp.array2d[wp.int32]
) -> None:
    write_face_pair(edge_groups, wp.int32(wp.tid()), out_adjacency)


@wp.kernel
def edge_pairs_to_face_pairs_and_edges(
    faces: wp.array[wp.int32],
    edge_groups: wp.array2d[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # ``edge_pairs_to_face_pairs`` plus the shared edge of each pair, which is the first grouped
    # edge's sorted endpoints -- recovered from its index through ``edge_endpoints``, so no
    # ``(3 * n_faces, 2)`` edge table has to exist to gather it from. Differs from
    # ``edge_pairs_to_face_pairs_and_table_edges`` only in where that row is read: the faces here,
    # a caller's precomputed table there.
    tid = wp.int32(wp.tid())
    write_face_pair(edge_groups, tid, out_adjacency)
    a, b = edge_endpoints(faces, edge_groups[tid, 0])
    out_edges[tid, 0] = a
    out_edges[tid, 1] = b


@wp.kernel
def edge_pairs_to_face_pairs_and_table_edges(
    edge_groups: wp.array2d[wp.int32],
    edges_sorted: wp.array2d[wp.int32],
    out_adjacency: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # The same two answers as ``edge_pairs_to_face_pairs_and_edges`` with the shared edge read from
    # a caller-supplied ``edges_sorted`` row -- in one launch rather than the pair kernel plus a
    # ``wp.clone`` of the strided ``edge_groups[:, 0]`` column and a gather through it.
    tid = wp.int32(wp.tid())
    write_face_pair(edge_groups, tid, out_adjacency)
    first = edge_groups[tid, 0]
    out_edges[tid, 0] = edges_sorted[first, 0]
    out_edges[tid, 1] = edges_sorted[first, 1]


@wp.func
def unshared_vertex(
    v0: wp.int32, v1: wp.int32, v2: wp.int32, e0: wp.int32, e1: wp.int32
) -> wp.int32:
    result = wp.int32(-1)
    count = wp.int32(0)
    if v0 != e0 and v0 != e1:
        result = v0
        count = count + wp.int32(1)
    if v1 != e0 and v1 != e1:
        if count == wp.int32(0):
            result = v1
        count = count + wp.int32(1)
    if v2 != e0 and v2 != e1:
        if count == wp.int32(0):
            result = v2
        count = count + wp.int32(1)
    if count != wp.int32(1):
        return wp.int32(-1)
    return result


@wp.kernel
def face_adjacency_unshared(
    faces: wp.array[wp.int32],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    f0 = face_adjacency[tid, 0] * 3
    f1 = face_adjacency[tid, 1] * 3
    e0 = face_adjacency_edges[tid, 0]
    e1 = face_adjacency_edges[tid, 1]
    out_unshared[tid, 0] = unshared_vertex(faces[f0 + 0], faces[f0 + 1], faces[f0 + 2], e0, e1)
    out_unshared[tid, 1] = unshared_vertex(faces[f1 + 0], faces[f1 + 1], faces[f1 + 2], e0, e1)


@wp.func
def edge_pair_topology(
    faces: wp.array[wp.int32], edge_0: wp.int32, edge_1: wp.int32
) -> tuple[wp.int32, wp.int32, wp.int32, wp.int32, wp.int32, wp.int32]:
    # Everything a manifold edge's two grouped halfedge indices determine: the sorted endpoints of
    # the edge they share, the two faces owning them, and each face's opposite apex. No edge table
    # and no adjacency table is read -- a halfedge index already encodes its face as ``edge // 3``.
    #
    # Shared with ``remesh.emit_flip_topology``, which wants the identical six values and differs
    # only in where it puts them: a scan-derived slot rather than a thread row, plus the adjacency
    # pair itself. It lives here because the quantity is edge topology, not a remeshing step.
    shared_a, shared_b = edge_endpoints(faces, edge_0)
    face_0 = edge_0 // 3
    face_1 = edge_1 // 3
    i0, i1, i2 = corner_triple(faces, face_0)
    j0, j1, j2 = corner_triple(faces, face_1)
    unshared_0 = unshared_vertex(i0, i1, i2, shared_a, shared_b)
    unshared_1 = unshared_vertex(j0, j1, j2, shared_a, shared_b)
    return shared_a, shared_b, face_0, face_1, unshared_0, unshared_1


@wp.kernel
def face_adjacency_unshared_from_edges(
    faces: wp.array[wp.int32], edge_groups: wp.array2d[wp.int32], out_unshared: wp.array2d[wp.int32]
) -> None:
    # Same answer as ``face_adjacency_unshared`` with no edge table and no adjacency table. Column
    # order follows ``edge_pairs_to_face_pairs``, which emits the face pair ascending.
    tid = wp.int32(wp.tid())
    _shared_a, _shared_b, face_0, face_1, unshared_0, unshared_1 = edge_pair_topology(
        faces, edge_groups[tid, 0], edge_groups[tid, 1]
    )
    if face_0 <= face_1:
        out_unshared[tid, 0] = unshared_0
        out_unshared[tid, 1] = unshared_1
    else:
        out_unshared[tid, 0] = unshared_1
        out_unshared[tid, 1] = unshared_0


@wp.kernel
def face_adjacency_angles(
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    out_angles: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    normal_a = face_normals[face_adjacency[tid, 0]]
    normal_b = face_normals[face_adjacency[tid, 1]]
    out_angles[tid] = vector_angle(normal_a, normal_b)


@wp.kernel
def scatter_vertex_faces(
    faces: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_vertex_faces: wp.array[wp.int32],
) -> None:
    # Vertex-to-face CSR payload. Row order is thread-order and therefore arbitrary, which is what
    # the wrapper documents: a rotational order would need halfedge twins and would refuse a
    # vertex-non-manifold mesh, which the decimator that consumes this must not do.
    f = wp.int32(wp.tid())
    for k in range(3):
        v = faces[f * 3 + k]
        out_vertex_faces[offsets[v] + wp.atomic_add(cursor, v, 1)] = f


@wp.func
def unshared_projection(
    vertices: wp.array[wp.vec3], normal: wp.vec3, origin: wp.int32, other: wp.int32
) -> wp.float32:
    # Signed distance of a face pair's second unshared vertex ``other`` above the first face's
    # plane, measured from the shared edge's first endpoint ``origin``: the convexity rule
    # ``face_adjacency_convex`` thresholds and ``curvature.face_pair_dihedrals`` signs its angle by.
    #
    # ``unshared_vertex`` returns -1 for a degenerate second face (it does not have exactly one
    # vertex off the shared edge), and that sentinel is not a valid index into ``vertices`` --
    # reading it would be the out-of-bounds access CLAUDE.md's memory-safety rule forbids. There is
    # no meaningful projection for a degenerate face, so it reports as never locally convex
    # (+inf is never < TOLERANCE_MERGE) rather than being read as an arbitrary finite value.
    if other < wp.int32(0):
        return FLOAT32_INF_CONSTANT
    return wp.dot(vertices[other] - vertices[origin], normal)


@wp.func
def adjacency_projection(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    row: wp.int32,
) -> wp.float32:
    # ``unshared_projection`` for adjacency row ``row``, read from the precomputed tables: the
    # quantity ``face_adjacency_projections`` returns and ``face_adjacency_convex`` thresholds.
    return unshared_projection(
        vertices,
        face_normals[face_adjacency[row, 0]],
        face_adjacency_edges[row, 0],
        face_adjacency_unshared[row, 1],
    )


@wp.kernel
def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    out_projections: wp.array[wp.float32],
) -> None:
    tid = wp.int32(wp.tid())
    out_projections[tid] = adjacency_projection(
        vertices, face_normals, face_adjacency, face_adjacency_edges, face_adjacency_unshared, tid
    )


@wp.kernel
def face_adjacency_convex(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    tolerance: wp.float32,
    out_convex: wp.array[wp.bool],
) -> None:
    # The projection and its ``< tolerance`` threshold in one launch, where the wrapper used to
    # write the ``(m,)`` projections and ``wp.map`` a comparison over them.
    tid = wp.int32(wp.tid())
    projection = adjacency_projection(
        vertices, face_normals, face_adjacency, face_adjacency_edges, face_adjacency_unshared, tid
    )
    out_convex[tid] = projection < tolerance
