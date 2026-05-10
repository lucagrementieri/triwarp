import warp as wp


@wp.func
def array_shift_insert(
    array: wp.array[wp.Scalar], value: wp.Scalar, index: wp.int32
) -> None:
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
