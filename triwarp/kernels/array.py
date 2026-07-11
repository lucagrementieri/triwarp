import warp as wp

from triwarp.constants import TILE_1D, TOLERANCE_MERGE_CONSTANT
from triwarp.kernels.reduce import outer_sum_tile


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


@wp.func
def wrap_index(i: wp.int32, n: wp.int32) -> wp.int32:
    # Positive modulo: ``%`` follows C++11 semantics (sign of the dividend).
    return ((i % n) + n) % n


@wp.func
def update_argmin(
    best_value: wp.ref[wp.float32],
    best_index: wp.ref[wp.int32],
    value: wp.float32,
    index: wp.int32,
):
    # Running min-with-index update in place. Callers must be compiled with
    # ``enable_backward=False`` (``wp.ref`` helpers have no adjoint).
    if value < best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> wp.float32:
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def to_vec3(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def square_scalar(value: wp.Scalar) -> wp.Scalar:
    return value * value


@wp.func
def divide_if_positive(value: wp.float32, divisor: wp.float32) -> wp.float32:
    # Guarded division: leave ``value`` unchanged when ``divisor <= 0``.
    if divisor > 0.0:
        return value / divisor
    return value


@wp.kernel
def init_range(out: wp.array[wp.Int]) -> None:
    i = int(wp.tid())
    out[i] = i


@wp.kernel
def init_range_step(out: wp.array[wp.Int], step: wp.Int) -> None:
    i = int(wp.tid())
    out[i] = i * step


@wp.kernel
def init_sort_pair_indices(out: wp.array[wp.Int], n: wp.Int, fill_value: wp.Int) -> None:
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
def gather_1d_skip_negative(
    indices: wp.array[wp.int32], table: wp.array[wp.int32], out_gathered: wp.array[wp.int32]
) -> None:
    tid = int(wp.tid())
    index = indices[tid]
    if index < wp.int32(0):
        out_gathered[tid] = index
    else:
        out_gathered[tid] = table[index]


@wp.kernel
def gather_2d_from_1d(
    array: wp.array[wp.Scalar], indices: wp.array2d[wp.int32], out_gathered: wp.array2d[wp.Scalar]
) -> None:
    i, j = wp.tid()
    index = indices[i, j]
    out_gathered[i, j] = array[index]


@wp.kernel
def gather_vec_skip_negative(
    source: wp.array[wp.vec3], index: wp.array[wp.int32], out_gathered: wp.array[wp.vec3]
) -> None:
    # Gather vectors by index, writing a zero vector wherever the index is negative
    # (missing-correspondence sentinel).
    i = int(wp.tid())
    f = index[i]
    if f >= 0:
        out_gathered[i] = source[f]
    else:
        out_gathered[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def isin_lookup_sorted(
    elements: wp.array[wp.int32], sorted_test: wp.array[wp.int32], out_mask: wp.array[wp.bool]
) -> None:
    tid = int(wp.tid())
    out_mask[tid] = binary_search_sorted_contains(sorted_test, elements[tid])


@wp.func
def vector_angle_vec(a: wp.vec3, b: wp.vec3) -> wp.float32:
    dot = wp.clamp(wp.dot(a, b), -1.0, 1.0)
    return wp.abs(wp.acos(dot))


@wp.func
def is_close_scalar(a: wp.float32, b: wp.float32, rtol: wp.float32, atol: wp.float32) -> wp.bool:
    return wp.abs(a - b) <= atol + rtol * wp.abs(b)


@wp.func
def is_close_vec3(a: wp.vec3, b: wp.vec3, rtol: wp.float32, atol: wp.float32) -> wp.bool:
    close = True
    for k in range(3):
        if wp.abs(a[k] - b[k]) > atol + rtol * wp.abs(b[k]):
            close = False
    return close


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


@wp.kernel
def centered_covariance(
    points: wp.array[wp.vec3], center: wp.array[wp.vec3], out_cov: wp.array[wp.mat33]
) -> None:
    # Scatter matrix C = sum_k outer(x_k - center, x_k - center). With a zero center this is
    # the uncentred Gram matrix G = sum_k outer(x_k, x_k).
    i, t = wp.tid()
    n = points.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    m = outer_sum_tile(points, center[0], offset, remaining)

    if t == 0:
        wp.atomic_add(out_cov, 0, m)
