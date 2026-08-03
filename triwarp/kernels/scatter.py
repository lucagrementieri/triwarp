import warp as wp

from triwarp.kernels.array import binary_search_index


@wp.func
def atomic_add_vec3(out_sum: wp.array2d[wp.float32], row: wp.int32, v: wp.vec3) -> None:
    # Component-wise atomic accumulation of a wp.vec3 into row ``row`` of a (n, 3) buffer.
    wp.atomic_add(out_sum, row, 0, v[0])
    wp.atomic_add(out_sum, row, 1, v[1])
    wp.atomic_add(out_sum, row, 2, v[2])


@wp.kernel
def scatter_add_scalar(
    values: wp.array[wp.Scalar], indices: wp.array[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    # 1D indexed scalar accumulation: out_sum[indices[tid]] += values[tid].
    tid = int(wp.tid())
    wp.atomic_add(out_sum, indices[tid], values[tid])


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
    f = int(wp.tid())
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
    f = int(wp.tid())
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
    f = int(wp.tid())
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
    tid = int(wp.tid())
    out_scattered[index[tid]] = wp.int32(tid)


@wp.kernel
def scatter_index_where(
    flags: wp.array[wp.int32], inclusive: wp.array[wp.int32], out_scattered: wp.array[wp.int32]
) -> None:
    # ``flags`` is the 0/1 selection array and ``inclusive`` its inclusive prefix sum, so a set
    # position lands at ``inclusive[i] - 1`` (its exclusive-scan value). Reading the flags rather
    # than the original mask is what lets ``flatnonzero`` take non-boolean input from one kernel.
    i = int(wp.tid())
    if flags[i] != wp.int32(0):
        out_scattered[inclusive[i] - 1] = wp.int32(i)


@wp.kernel
def mark_membership_mask(
    indices: wp.array[wp.int32], n: wp.int32, out_mask: wp.array[wp.bool]
) -> None:
    # Mark out_mask[indices[tid]] = True, skipping out-of-range indices (negative or >= n).
    tid = int(wp.tid())
    index = indices[tid]
    if index >= wp.int32(0) and index < n:
        out_mask[index] = wp.bool(True)
