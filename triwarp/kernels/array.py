import warp as wp


@wp.kernel
def normalize(array: wp.array[wp.vec3]) -> None:
    i = wp.tid()
    array[i] = wp.normalize(array[i])


@wp.kernel
def scatter_sum_scalar(
    values: wp.array2d[wp.Scalar], indices: wp.array2d[wp.int32], out_sum: wp.array[wp.Scalar]
) -> None:
    i = wp.tid()
    index = indices[i]
    value = values[i]
    for j in range(indices.shape[1]):
        wp.atomic_add(out_sum, index[j], value[i, j])
        wp.atomic_add(out_sum, index[j], value[i, j])
        wp.atomic_add(out_sum, index[j], value[i, j])


@wp.kernel
def scatter_sum_vec(values: wp.array[wp.vec3], indices: wp.array2d[wp.int32], out_sum: wp.array2d[wp.float32]) -> None:
    i = wp.tid()
    index = indices[i]
    value = values[i]
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
    i = wp.tid()
    index = indices[i]
    value = values[i]
    for j in range(indices.shape[1]):
        wp.atomic_add(out_sum, index[j], 0, value[0] * weights[i, j])
        wp.atomic_add(out_sum, index[j], 1, value[1] * weights[i, j])
        wp.atomic_add(out_sum, index[j], 2, value[2] * weights[i, j])


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
