from typing import Any

import warp as wp

from triwarp.kernels.array import binary_search_index


@wp.func
def atomic_add_vec3(out_sum: wp.array2d[wp.float32], row: wp.int32, v: wp.vec3) -> None:
    # Component-wise atomic accumulation of a wp.vec3 into row ``row`` of a (n, 3) buffer.
    wp.atomic_add(out_sum, row, 0, v[0])
    wp.atomic_add(out_sum, row, 1, v[1])
    wp.atomic_add(out_sum, row, 2, v[2])


@wp.kernel
def scatter_add(values: wp.array[Any], indices: wp.array[wp.int32], out_sum: wp.array[Any]) -> None:
    # 1D indexed accumulation: out_sum[indices[tid]] += values[tid]. ``Any`` rather than
    # ``wp.Scalar`` because the latter does not instantiate for vectors, and the vector heat
    # method seeds a ``wp.vec2d`` field through exactly this kernel. The widening gives up a
    # compile-time dtype guard: Warp still rejects a mismatched ``values`` / ``out_sum`` pair, but
    # the error arrives from codegen rather than from overload resolution and reads worse.
    tid = wp.int32(wp.tid())
    wp.atomic_add(out_sum, indices[tid], values[tid])


@wp.kernel
def count_occurrences(indices: wp.array[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Histogram of ``indices``: one atomic increment per entry. Launch over ``indices.shape[0]``.
    # Sizing a CSR is what every caller wants it for -- incident faces per vertex (a flat face
    # buffer *is* the corner -> vertex map), face-corners per unique edge, outgoing halfedges per
    # vertex -- so the counts feed a scan and the caller allocates ``out_counts`` zeroed.
    tid = wp.int32(wp.tid())
    wp.atomic_add(out_counts, indices[tid], 1)


@wp.kernel
def count_occurrences_rows(indices: wp.array2d[wp.int32], out_counts: wp.array[wp.int32]) -> None:
    # Row form of ``count_occurrences``: every column of every row increments its own counter.
    # One thread per row, so an edge list ``(n, 2)`` gives each endpoint's degree in one launch.
    row = wp.int32(wp.tid())
    for j in range(indices.shape[1]):
        wp.atomic_add(out_counts, indices[row, j], 1)


@wp.kernel
def scatter_sum_scalar(
    values: wp.array2d[wp.Scalar], indices: wp.array2d[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    tid = wp.tid()
    index = indices[tid]
    for j in range(indices.shape[1]):
        wp.atomic_add(out_sum, index[j], values[tid, j])


@wp.kernel
def scatter_sum_vec(
    values: wp.array[wp.vec3], indices: wp.array2d[wp.int32], out_sum: wp.array2d[wp.float32]
) -> None:
    tid = wp.tid()
    index = indices[tid]
    value = values[tid]
    for j in range(indices.shape[1]):
        atomic_add_vec3(out_sum, index[j], value)


@wp.kernel
def scatter_weighted_sum_vec(
    values: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    weights: wp.array2d[wp.float32],
    out_sum: wp.array2d[wp.float32],
) -> None:
    tid = wp.tid()
    index = indices[tid]
    value = values[tid]
    for j in range(indices.shape[1]):
        atomic_add_vec3(out_sum, index[j], value * weights[tid, j])


@wp.kernel
def scatter_offset_sum(
    values: wp.array[wp.Scalar],
    flat_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_sum: wp.array[wp.Scalar],
) -> None:
    tid = wp.tid()
    in_index = flat_indices[tid]
    value = values[in_index]
    out_index = binary_search_index(offsets, tid) - 1
    wp.atomic_add(out_sum, out_index, value)


@wp.kernel
def scatter_face_thirds(
    faces: wp.array[wp.int32],
    areas: wp.array[wp.Float],
    count: wp.Float,
    out_mass: wp.array[wp.Float],
) -> None:
    # Barycentric (lumped) mass: each face donates ``areas[f] / count`` to each incident vertex.
    # ``areas``, ``count`` and ``out_mass`` share one float dtype so the kernel specialises to
    # float32 (Laplacian) or float64 (geodesic heat method) at launch time.
    f = wp.int32(wp.tid())
    third = areas[f] / count
    wp.atomic_add(out_mass, faces[f * 3 + 0], third)
    wp.atomic_add(out_mass, faces[f * 3 + 1], third)
    wp.atomic_add(out_mass, faces[f * 3 + 2], third)


@wp.kernel
def scatter_face_values_sum_and_valence(
    faces: wp.array[wp.int32],
    face_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    value = face_values[f]
    for j in range(3):
        vertex_index = faces[f * 3 + j]
        wp.atomic_add(out_sum, vertex_index, value)
        wp.atomic_add(out_valence, vertex_index, wp.float32(1.0))


@wp.kernel
def scatter_edges_sum_and_valence(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    edges_orientation: wp.array2d[wp.int32],
    edge_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    for j in range(3):
        if edges_orientation[f, j] < 0:
            continue
        e = edges[f, j]
        vi = faces[f * 3 + (j + 1) % 3]
        vj = faces[f * 3 + (j + 2) % 3]
        value = edge_values[e]
        wp.atomic_add(out_sum, vi, value)
        wp.atomic_add(out_sum, vj, value)
        wp.atomic_add(out_valence, vi, wp.float32(1.0))
        wp.atomic_add(out_valence, vj, wp.float32(1.0))


@wp.kernel
def scatter_index(index: wp.array[wp.int32], out_scattered: wp.array[wp.int32]) -> None:
    tid = wp.int32(wp.tid())
    out_scattered[index[tid]] = tid


@wp.kernel
def scatter_index_where(
    flags: wp.array[wp.int32], inclusive: wp.array[wp.int32], out_scattered: wp.array[wp.int32]
) -> None:
    # ``flags`` is the 0/1 selection array and ``inclusive`` its inclusive prefix sum, so a set
    # position lands at ``inclusive[i] - 1`` (its exclusive-scan value). Reading the flags rather
    # than the original mask is what lets ``flatnonzero`` take non-boolean input from one kernel.
    i = wp.int32(wp.tid())
    if flags[i] != wp.int32(0):
        out_scattered[inclusive[i] - 1] = i


@wp.kernel
def scatter_edge_incidence(
    inverse: wp.array[wp.int32],
    out_edge_face_count: wp.array[wp.int32],
    out_edge_faces: wp.array2d[wp.int32],
) -> None:
    # Face-corners per unique edge *and* the faces themselves, in one pass over the corner ->
    # unique-edge map ``inverse``: corner ``c`` belongs to face ``c // 3``, so no face table is
    # needed. The count is 1 on a boundary edge and 2 on an interior one; a non-manifold edge
    # counts higher and its faces past the second are dropped, which is what the exactly-2 row
    # grouping behind ``adjacency.face_adjacency`` does with them too.
    #
    # Launch over ``inverse.shape[0]`` with both outputs zeroed: the count doubles as the write
    # cursor, which is why this replaces ``count_occurrences`` rather than following it.
    #
    # It lives here rather than in ``kernels/remesh.py``, where the decimation passes first needed
    # it, because ``homology.tree_cotree`` groups the same rows for the same reason -- one grouping
    # answering "how many faces meet along this edge, and which" for the whole package.
    c = wp.int32(wp.tid())
    e = inverse[c]
    slot = wp.atomic_add(out_edge_face_count, e, 1)
    if slot < 2:
        out_edge_faces[e, slot] = c // 3


@wp.kernel
def scatter_group_bounds(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    out_corners: wp.array[wp.float32],
) -> None:
    # Axis-aligned bounds of every face group, six ``float32`` slots per group packed
    # ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]`` -- the ``kernels/bounds.py`` convention, so
    # one ``wp.full(inf)`` seeds both ends and every update is a single ``wp.atomic_min``. Launch
    # over ``n_faces`` with ``out_corners`` sized ``6 * n_groups``; ``groups[f]`` is the group index
    # of face ``f``, which for a connected-component label is a representative face index and so
    # needs ``n_groups == n_faces``. A group no face names keeps its ``inf`` seed, which
    # [`packed_box_diagonals`][triwarp.kernels.bounds.packed_box_diagonals] reads as empty.
    f = wp.int32(wp.tid())
    base = groups[f] * 6
    for c in range(3):
        position = vertices[faces[f * 3 + c]]
        for k in range(3):
            wp.atomic_min(out_corners, base + k, position[k])
            wp.atomic_min(out_corners, base + 3 + k, -position[k])


@wp.kernel
def mark_membership_mask(
    indices: wp.array[wp.int32], n: wp.int32, out_mask: wp.array[wp.bool]
) -> None:
    # Mark out_mask[indices[tid]] = True, skipping out-of-range indices (negative or >= n).
    tid = wp.int32(wp.tid())
    index = indices[tid]
    if index >= wp.int32(0) and index < n:
        out_mask[index] = wp.bool(True)


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured over the suite: 6 overloads created across **11** module loads --
# nearly two rebuilds per overload, this module being reached from 11 wrappers at scattered moments.
#
# Each set is the dtypes its call sites actually build, not a menu: ``scatter_add`` accumulates a
# ``wp.float32`` per-component volume in ``repair`` and a ``wp.vec2d`` tangent field in
# ``heat.vector``, and the two mass/curvature scatters follow their wrapper's precision keyword.
_VALUE_DTYPES = (wp.float32, wp.float64)


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in (*_VALUE_DTYPES, wp.vec2d, wp.vec3):
        wp.overload(scatter_add, [wp.array[dtype], wp.array[wp.int32], wp.array[dtype]])
    for dtype in _VALUE_DTYPES:
        wp.overload(
            scatter_face_thirds, [wp.array[wp.int32], wp.array[dtype], dtype, wp.array[dtype]]
        )
        wp.overload(
            scatter_offset_sum,
            [wp.array[dtype], wp.array[wp.int32], wp.array[wp.int32], wp.array[dtype]],
        )
        wp.overload(scatter_sum_scalar, [wp.array2d[dtype], wp.array2d[wp.int32], wp.array[dtype]])


_register_overloads()
