import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT


@wp.func
def sort3(a: wp.Scalar, b: wp.Scalar, c: wp.Scalar) -> tuple[wp.Scalar, wp.Scalar, wp.Scalar]:
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c


@wp.func
def tolerance_sign(value: wp.float32) -> wp.int32:
    if value < -TOLERANCE_MERGE_CONSTANT:
        return wp.int32(-1)
    if value > TOLERANCE_MERGE_CONSTANT:
        return wp.int32(1)
    return wp.int32(0)


@wp.kernel
def sub(array: wp.array[wp.Scalar], n: wp.Scalar) -> None:
    i = int(wp.tid())
    array[i] = array[i] - n


@wp.kernel
def init_range(out: wp.array[wp.Int]) -> None:
    i = int(wp.tid())
    out[i] = i


@wp.kernel
def init_range_step(out: wp.array[wp.Int], step: wp.Int) -> None:
    i = int(wp.tid())
    out[i] = i * step


@wp.kernel
def init_sort_pair_indices(
    out: wp.array[wp.Int], n: wp.Int, fill_value: wp.Int
) -> None:
    i = int(wp.tid())
    if i < n:
        out[i] = i
    else:
        out[i] = fill_value


@wp.kernel
def init_repeat_index(out: wp.array[wp.Int], repeats: wp.Int) -> None:
    i = int(wp.tid())
    out[i] = i // repeats


@wp.kernel
def normalize(array: wp.array[wp.vec3]) -> None:
    tid = wp.tid()
    array[tid] = wp.normalize(array[tid])


@wp.kernel
def gather_2d_from_1d(
    array: wp.array[wp.Scalar], indices: wp.array2d[wp.int32], out_gathered: wp.array2d[wp.Scalar]
) -> None:
    i, j = wp.tid()
    index = indices[i, j]
    out_gathered[i, j] = array[index]


@wp.kernel
def gather_rows(
    array: wp.array2d[wp.Scalar], indices: wp.array[wp.int32], out_gathered: wp.array2d[wp.Scalar]
) -> None:
    tid = wp.tid()
    index = indices[tid]
    for j in range(out_gathered.shape[1]):
        out_gathered[tid, j] = array[index, j]


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
        wp.atomic_add(out_sum, index[j], 0, value[0])
        wp.atomic_add(out_sum, index[j], 1, value[1])
        wp.atomic_add(out_sum, index[j], 2, value[2])


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
        wp.atomic_add(out_sum, index[j], 0, value[0] * weights[tid, j])
        wp.atomic_add(out_sum, index[j], 1, value[1] * weights[tid, j])
        wp.atomic_add(out_sum, index[j], 2, value[2] * weights[tid, j])


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
def mark_membership_mask(indices: wp.array[wp.int32], mask: wp.array[wp.bool]) -> None:
    tid = int(wp.tid())
    mask[indices[tid]] = wp.bool(True)


@wp.kernel
def isin_lookup_mask(
    elements: wp.array[wp.int32], membership: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    tid = int(wp.tid())
    out_mask[tid] = membership[elements[tid]]


@wp.kernel
def isin_lookup_sorted(
    elements: wp.array[wp.int32], sorted_test: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    tid = int(wp.tid())
    out_mask[tid] = binary_search_sorted_contains(sorted_test, elements[tid])


@wp.kernel
def scatter_compact_indices(
    mask: wp.array[wp.bool], exclusive_offsets: wp.array[wp.int32], out_indices: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    if mask[i]:
        out_indices[exclusive_offsets[i]] = wp.int32(i)


@wp.func
def vector_angle_vec(a: wp.vec3, b: wp.vec3) -> wp.float32:
    dot = wp.clamp(wp.dot(a, b), -1.0, 1.0)
    return wp.abs(wp.acos(dot))


@wp.kernel
def vector_angle(
    a: wp.array[wp.vec3], b: wp.array[wp.vec3], out_angles: wp.array[wp.float32]
) -> None:
    tid = int(wp.tid())
    out_angles[tid] = vector_angle_vec(a[tid], b[tid])


@wp.func
def array_shift_insert(array: wp.array[wp.Scalar], value: wp.Scalar, index: wp.int32) -> None:
    for i in range(array.shape[0] - 1, index, -1):
        array[i] = array[i - 1]
    array[index] = value


@wp.func
def binary_search_index(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] > value, or len(values) (numpy searchsorted side='right')."""
    n = values.shape[0]
    left = wp.int32(0)
    right = n - 1
    result = n
    while left <= right:
        mid = (left + right) // 2
        if values[mid] > value:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return wp.int32(result)


@wp.func
def binary_search_sorted_contains(values: wp.array[wp.Scalar], value: wp.Scalar) -> bool:
    idx = binary_search_index(values, value)
    return idx > wp.int32(0) and values[idx - wp.int32(1)] == value


@wp.kernel
def map_sorted_inverse(
    data: wp.array[wp.Scalar], sorted_unique: wp.array[wp.Scalar], out_inverse: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    out_inverse[i] = binary_search_index(sorted_unique, data[i]) - wp.int32(1)
