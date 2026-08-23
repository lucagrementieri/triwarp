from typing import Any

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


@wp.func
def sign_with_tolerance(value: wp.float32, tolerance: wp.float32) -> wp.int32:
    # ``tolerance_sign`` with the dead zone exposed as an argument, for callers whose tolerance is
    # scale-dependent rather than fixed at ``TOLERANCE_MERGE`` (a plane split's snap band scales
    # with the model). Pass ``TOLERANCE_MERGE`` to recover ``tolerance_sign`` exactly.
    if value < -tolerance:
        return wp.int32(-1)
    if value > tolerance:
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def wrap_index(i: wp.int32, n: wp.int32) -> wp.int32:
    # Positive modulo: ``%`` follows C++11 semantics (sign of the dividend).
    return ((i % n) + n) % n


@wp.func
def update_argmin(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running min-with-index update in place. Callers must be compiled with
    # ``enable_backward=False`` (``wp.ref`` helpers have no adjoint). Concrete ``float32``:
    # ``wp.ref[wp.Scalar]`` generics do not instantiate through Warp 1.16 -- re-probed there,
    # still a ``WarpCodegenError`` at kernel parse ("Couldn't find function overload") -- so float64
    # sites keep a hand-written loop; the index/tag stays ``int32``.
    if value < best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running max-with-index mirror of ``update_argmin`` (same ``enable_backward=False`` rule).
    if value > best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax_lowest_index(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # argmax with a lowest-index tie-break (matches the label-vote rule in ``rasterize_labels``).
    if value > best_value or (value == best_value and index < best_index):
        best_value = value
        best_index = index


@wp.func
def update_argmin_pair(
    best_value: wp.ref[wp.float32],
    first: wp.ref[wp.int32],
    second: wp.ref[wp.int32],
    value: wp.float32,
    a: wp.int32,
    b: wp.int32,
):
    # Running argmin carrying a pair of associated indices (shortest-edge endpoints).
    if value < best_value:
        best_value = value
        first = a  # noqa: F841 — writes through the wp.ref parameter
        second = b  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def cross2(a: Any, b: Any) -> wp.Float:
    # 2D cross product (signed parallelogram area). Generic so float32 and float64 call sites share
    # one definition.
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def to_vec3(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def to_vec2d(v: wp.vec2) -> wp.vec2d:
    return wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))


@wp.func
def to_vec2(v: wp.vec2d) -> wp.vec2:
    return wp.vec2(wp.float32(v[0]), wp.float32(v[1]))


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
def init_range(out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i


@wp.kernel
def init_range_step(step: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i * step


@wp.kernel
def init_sort_pair_indices(n: wp.Int, fill_value: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    if i < n:
        out_indices[i] = i
    else:
        out_indices[i] = fill_value


@wp.kernel
def init_repeat_index(repeats: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i // repeats


@wp.kernel
def random_priorities(seed: wp.int32, out_priority: wp.array[wp.uint32]) -> None:
    # A total order on the elements, drawn once for the whole run rather than per round. Every
    # multi-round selection that breaks ties by priority -- blue-noise dart throwing, the
    # maximal-independent-set aggregation -- reads the same order in every round, which is what
    # makes the loop a deterministic function of ``seed`` alone. Drawing per round would make the
    # answer depend on how many rounds the input happened to need.
    i = wp.int32(wp.tid())
    out_priority[i] = wp.randu(wp.rand_init(seed, i))


@wp.kernel
def gather_1d_skip_negative(
    indices: wp.array[wp.int32], table: wp.array[wp.int32], out_gathered: wp.array[wp.int32]
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    if index < wp.int32(0):
        out_gathered[tid] = index
    else:
        out_gathered[tid] = table[index]


@wp.kernel
def sort_rows_insertion(data: wp.array2d[wp.Scalar]) -> None:
    # One thread per row, in-place insertion sort across the row. For the narrow rows this library
    # actually sorts (vertex pairs, triangle corners) that is 1-3 register comparisons, versus a
    # segmented radix sort whose fixed per-segment cost dominates completely at these widths.
    row = wp.int32(wp.tid())
    width = data.shape[1]
    for i in range(1, width):
        value = data[row, i]
        j = i - 1
        while j >= 0 and data[row, j] > value:
            data[row, j + 1] = data[row, j]
            j = j - 1
        data[row, j + 1] = value


@wp.kernel
def gather_vec_skip_negative(
    source: wp.array[wp.vec3], index: wp.array[wp.int32], out_gathered: wp.array[wp.vec3]
) -> None:
    # Gather vectors by index, writing a zero vector wherever the index is negative
    # (missing-correspondence sentinel).
    i = wp.int32(wp.tid())
    f = index[i]
    if f >= 0:
        out_gathered[i] = source[f]
    else:
        out_gathered[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def shifted_index(value: wp.Scalar, offset: wp.Scalar) -> wp.int32:
    # Position of ``value`` in a table anchored at ``offset``. The subtraction happens in the
    # value's own dtype, which is exact for every dtype ``array.isin`` reaches this with: it widens
    # sub-32-bit dtypes first (so the span cannot overflow the type) and only takes the table path
    # when the span is small (so a 64-bit difference cannot overflow either).
    return wp.int32(value - offset)


@wp.kernel
def isin_lookup_sorted(
    elements: wp.array[wp.Scalar], sorted_test: wp.array[wp.Scalar], out_mask: wp.array[wp.bool]
) -> None:
    tid = wp.int32(wp.tid())
    out_mask[tid] = binary_search_sorted_contains(sorted_test, elements[tid])


@wp.func
def mask_not(a: wp.bool) -> wp.bool:
    return not a


@wp.func
def mask_and(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and b


@wp.func
def mask_and_not(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and not b


@wp.func
def complement_flag(a: wp.bool) -> wp.int32:
    # ``1`` where the mask is False, ``0`` where it is True: the scan input for an inverted
    # ``mask_to_index_map`` (free/interior DOFs of a fixed-vertex mask, for instance).
    return wp.where(a, wp.int32(0), wp.int32(1))


@wp.func
def nonzero_flag(value: wp.Scalar) -> wp.int32:
    # ``1`` for any non-zero value, ``0`` otherwise: the scan input that lets ``flatnonzero``
    # accept integer and float arrays as well as masks. A bool array does not need this -- Warp's
    # ``wp.Scalar`` does not instantiate for ``wp.bool``, and ``wp.utils.array_cast`` already
    # produces exactly 0/1 for one, which is why the wrapper keeps that path separate.
    return wp.where(value != type(value)(0), wp.int32(1), wp.int32(0))


@wp.func
def greater(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a > b


@wp.func
def greater_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a >= b


@wp.func
def less(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a < b


@wp.func
def less_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a <= b


@wp.func
def equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a == b


@wp.func
def not_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a != b


@wp.func
def is_positive_finite(value: wp.Float) -> wp.bool:
    return value > type(value)(0) and wp.isfinite(value)


@wp.func
def value_if_positive_finite(value: wp.Float) -> wp.Float:
    # The masked half of a "mean over the positive finite entries" reduction: the excluded entries
    # contribute an exact zero to the sum, so one plain reduction over this and one over the
    # companion mask give the numerator and the denominator without a compaction pass.
    if value > type(value)(0) and wp.isfinite(value):
        return value
    return type(value)(0)


@wp.func
def is_close_scalar(a: wp.Float, b: wp.Float, rtol: wp.Float, atol: wp.Float) -> wp.bool:
    # Generic over the caller's float precision, so ``allclose`` works on float16/32/64 from one
    # definition. The tolerances must arrive at that same precision -- see the wrapper.
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
def binary_search_index_left(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] >= value, or len(values) (numpy searchsorted side='left')."""
    # ``wp.lower_bound`` is the same search, but it clamps its result to ``n - 1``
    # (``warp/native/array.h``), so a value past the last element reads back as the last index
    # instead of ``n``. Both callers (``graph.component_segment_bounds`` probes one past the
    # highest component key, ``holes.rim_opposite_from_table`` probes edges absent from the
    # table) do query past the end, so the fix-up is mandatory, not defensive.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    if index == n - 1 and values[n - 1] < value:
        index = n
    return wp.int32(index)


@wp.func
def binary_search_sorted_contains(values: wp.array[wp.Scalar], value: wp.Scalar) -> bool:
    # ``wp.lower_bound``'s clamp to ``n - 1`` is harmless here: a value past the end lands on the
    # last element, which then compares unequal. ``and`` short-circuits in kernel scope, so the
    # element read is skipped on an empty array.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    return index < n and values[index] == value


@wp.kernel
def map_sorted_inverse(
    data: wp.array[wp.Scalar], sorted_unique: wp.array[wp.Scalar], out_inverse: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    out_inverse[i] = binary_search_index(sorted_unique, data[i]) - wp.int32(1)


@wp.kernel
def mark_rows_present(
    rows: wp.array2d[wp.int32], queries: wp.array2d[wp.int32], out_present: wp.array[wp.bool]
) -> None:
    # Which of a handful of index rows occur in a table of them, decided on device so the caller
    # never reads back a buffer that scales with the mesh. Every thread that matches stores ``True``
    # into its query's slot, so the race is benign and no atomic is needed; ``out_present`` arrives
    # zeroed. Deliberately a scan rather than a hash or a sort: the query count is a handful, the
    # table is read once, and there is nothing to amortize a structure over.
    row, query = wp.tid()
    for column in range(queries.shape[1]):
        if rows[row, column] != queries[query, column]:
            return
    out_present[query] = True


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured over the suite: 12 overloads created across **13** module loads,
# and this module is imported by 25 kernel modules and 15 wrappers, so its rebuilds are felt widely.
#
# The ``init_*`` kernels fill an index buffer and every caller in the package allocates that buffer
# ``wp.int32``; ``wp.Int`` in their annotation is the template, not a menu. The two search kernels
# take the caller's *key* dtype, whose surface is the one ``sortable_dtype`` maps onto.
_INDEX_DTYPES = (wp.int32,)
_KEY_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in _INDEX_DTYPES:
        wp.overload(init_range, [wp.array[dtype]])
        wp.overload(init_range_step, [dtype, wp.array[dtype]])
        wp.overload(init_repeat_index, [dtype, wp.array[dtype]])
        wp.overload(init_sort_pair_indices, [dtype, dtype, wp.array[dtype]])
    for dtype in _KEY_DTYPES:
        wp.overload(isin_lookup_sorted, [wp.array[dtype], wp.array[dtype], wp.array[wp.bool]])
        wp.overload(map_sorted_inverse, [wp.array[dtype], wp.array[dtype], wp.array[wp.int32]])
    # ``sort_rows_insertion`` sorts a rank-2 table in place; ``unique_rows`` and the hashing paths
    # that reach it build that table in the caller's dtype.
    for dtype in (wp.int32, wp.float32, wp.float64):
        wp.overload(sort_rows_insertion, [wp.array2d[dtype]])


_register_overloads()


@wp.func
def pack_edge_key(u: wp.int32, v: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of undirected edge (u, v); matches ``pack_indices`` for a sorted 2-index row."""
    lo = wp.uint64(wp.uint32(wp.min(u, v)))
    hi = wp.uint64(wp.uint32(wp.max(u, v)))
    return lo + hi * base


@wp.func
def pack_farthest_key(distance_sq: wp.float32, index: wp.int32) -> wp.int64:
    # One int64 whose ``wp.atomic_max`` is "largest distance, lowest index on a tie". The IEEE-754
    # bits of a non-negative float increase monotonically with the value, so the high half orders
    # by distance; the low half stores ``index`` complemented within int32 so that a *smaller*
    # index compares *larger*. Both halves are non-negative, so the whole key is, which is what
    # makes ``-1`` a sentinel below every real candidate.
    distance_bits = wp.uint64(wp.uint32(wp.cast(distance_sq, wp.int32)))
    rank = wp.uint64(wp.uint32(wp.int32(2147483647) - index))
    return wp.int64((distance_bits << wp.uint64(32)) | rank)


@wp.func
def pack_ranked_key(value: wp.int32, index: wp.int32) -> wp.int64:
    # The integer sibling of ``pack_farthest_key``: one int64 whose ``wp.atomic_max`` is "largest
    # value, lowest index on a tie". ``value`` must be non-negative (a count, a size, a degree), so
    # the high half orders by it directly with no bit trick; the low half stores ``index``
    # complemented within int32 so that a *smaller* index compares *larger*. Both halves are
    # non-negative, so ``-1`` is a sentinel below every real candidate.
    #
    # Comparing this key is also how a caller applies the result without a readback: recomputing
    # ``pack_ranked_key(value, index)`` in a second kernel and testing it against the reduced
    # maximum identifies the winner on the device.
    return wp.int64(
        (wp.uint64(wp.uint32(value)) << wp.uint64(32))
        | wp.uint64(wp.uint32(wp.int32(2147483647) - index))
    )


@wp.func
def pack_nearest_key(distance: wp.float32, index: wp.int32) -> wp.int64:
    # The ``min``-ordered twin of ``pack_farthest_key``: one int64 whose minimum is "smallest
    # distance, lowest index on a tie". Same monotone IEEE-754 high half (valid because a distance
    # is non-negative), but the index is stored plainly rather than complemented, because a
    # *smaller* index must now compare *smaller*. The key stays non-negative, so it needs no
    # sentinel.
    distance_bits = wp.uint64(wp.uint32(wp.cast(distance, wp.int32)))
    return wp.int64((distance_bits << wp.uint64(32)) | wp.uint64(wp.uint32(index)))
