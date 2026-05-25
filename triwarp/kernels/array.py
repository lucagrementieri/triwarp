import warp as wp


@wp.kernel
def sub(array: wp.array[wp.Scalar], n: wp.Scalar) -> None:
    i = int(wp.tid())
    array[i] = array[i] - n


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
def gather_rows(array: wp.array2d[wp.Scalar], indices: wp.array[wp.int32], out_gathered: wp.array2d[wp.Scalar]) -> None:
    tid = wp.tid()
    index = indices[tid]
    out_gathered[tid] = array[index]


@wp.kernel
def scatter_sum_scalar(
    values: wp.array2d[wp.Scalar], indices: wp.array2d[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    tid = wp.tid()
    index = indices[tid]
    for j in range(indices.shape[1]):
        wp.atomic_add(out_sum, index[j], values[tid, j])


@wp.kernel
def scatter_sum_vec(values: wp.array[wp.vec3], indices: wp.array2d[wp.int32], out_sum: wp.array2d[wp.float32]) -> None:
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


@wp.func
def array_shift_insert(array: wp.array[wp.Scalar], value: wp.Scalar, index: wp.int32) -> None:
    for i in range(array.shape[0] - 1, index, -1):
        array[i] = array[i - 1]
    array[index] = value


@wp.func
def linear_search_index(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    n = values.shape[0]
    for slot_index in range(n):
        if value < values[slot_index]:
            return wp.int32(slot_index)
    return wp.int32(n)


@wp.func
def binary_search_index(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    n = values.shape[0]
    left = int(0)
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
