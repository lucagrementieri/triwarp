"""Global reductions: tiled ``wp.tile_load`` reductions generated from a shared template."""

import warp as wp

# Tile-reduction builtins are resolved only inside kernel source text and are not
# exposed as Python-scope attributes (``wp.tile_max`` etc. raise ``AttributeError``),
# unlike the atomic and scalar builtins (``wp.atomic_max``, ``wp.max`` …). To let the
# kernel factories below *capture* them, pull the concrete ``Function`` objects out of
# Warp's builtin registry. A captured builtin is emitted inline at codegen (same as a
# literal ``wp.tile_sum`` call), so it templates on the tile dtype correctly — wrapping
# it in a ``@wp.func`` instead would erase the dtype and produce ambiguous C++ overloads.
from warp._src.context import builtin_functions as _warp_builtins

from triwarp.constants import TILE_1D, TILE_2D

_tile_min = _warp_builtins["tile_min"]
_tile_max = _warp_builtins["tile_max"]
_tile_sum = _warp_builtins["tile_sum"]


# ---------------------------------------------------------------------------
# Kernel factories.
#
# Each builds one ``@wp.kernel`` capturing a ``(tile_reduce, atomic, scalar)``
# triple. The full-tile branch loads a tile and reduces it with ``tile_reduce``;
# the partial-tile remainder folds elements with ``scalar``; the result is
# committed with ``atomic``. A unique ``name`` per instantiation is REQUIRED:
# Warp keys generated kernels by name, so reusing one name would alias distinct
# reductions to the same compiled kernel.
# ---------------------------------------------------------------------------


def _reduce_1d_tiled(tile_reduce, atomic, scalar, name):
    """axis=None on a 1-D array: one tile per block, atomically fold into slot 0."""

    def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        offset = i * TILE_1D
        remaining = n - offset
        if remaining <= 0:
            return

        if remaining >= TILE_1D:
            tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
            result = tile_reduce(tile)[0]
        else:
            result = values[offset]
            for k in range(1, remaining):
                result = scalar(result, values[offset + k])

        if t == 0:
            atomic(out, 0, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


def _reduce_2d_tiled(tile_reduce, atomic, scalar, name):
    """axis=None on a 2-D array: one 2-D tile per block, atomically fold into slot 0."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_rows = values.shape[0]
        n_cols = values.shape[1]
        row_offset = i * TILE_2D
        col_offset = j * TILE_2D
        if row_offset >= n_rows or col_offset >= n_cols:
            return

        remaining_rows = n_rows - row_offset
        remaining_cols = n_cols - col_offset
        tile_rows = TILE_2D if remaining_rows >= TILE_2D else remaining_rows
        tile_cols = TILE_2D if remaining_cols >= TILE_2D else remaining_cols
        if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
            tile = wp.tile_load(
                values,
                shape=(TILE_2D, TILE_2D),
                offset=(row_offset, col_offset),
                storage="register",
            )
            result = tile_reduce(tile)[0]
        else:
            result = values[row_offset, col_offset]
            for r in range(tile_rows):
                for c in range(tile_cols):
                    if r == 0 and c == 0:
                        continue
                    result = scalar(result, values[row_offset + r, col_offset + c])

        if t == 0:
            atomic(out, 0, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


def _reduce_2d_rows_tiled(tile_reduce, atomic, scalar, name):
    """axis=1: one row per grid index, tile across columns, fold into slot ``i``."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_cols = values.shape[1]
        col_offset = j * TILE_1D
        if col_offset >= n_cols:
            return
        remaining = n_cols - col_offset
        if remaining >= TILE_1D:
            tile = wp.tile_load(
                values, shape=(1, TILE_1D), offset=(i, col_offset), storage="register"
            )
            result = tile_reduce(tile)[0]
        else:
            result = values[i, col_offset]
            for k in range(1, remaining):
                result = scalar(result, values[i, col_offset + k])
        if t == 0:
            atomic(out, i, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


def _reduce_2d_cols_tiled(tile_reduce, atomic, scalar, name):
    """axis=0: one column per grid index, tile across rows, fold into slot ``i``."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_rows = values.shape[0]
        row_offset = j * TILE_1D
        if row_offset >= n_rows:
            return
        remaining = n_rows - row_offset
        if remaining >= TILE_1D:
            tile = wp.tile_load(
                values, shape=(TILE_1D, 1), offset=(row_offset, i), storage="register"
            )
            result = tile_reduce(tile)[0]
        else:
            result = values[row_offset, i]
            for k in range(1, remaining):
                result = scalar(result, values[row_offset + k, i])
        if t == 0:
            atomic(out, i, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


# ---------------------------------------------------------------------------
# Scalar reductions (min / max / sum) over ``wp.Scalar`` arrays.
# ---------------------------------------------------------------------------

min1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "min1d_tiled")
min2d_tiled = _reduce_2d_tiled(_tile_min, wp.atomic_min, wp.min, "min2d_tiled")
min_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_min, wp.atomic_min, wp.min, "min_2d_rows_tiled")
min_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_min, wp.atomic_min, wp.min, "min_2d_cols_tiled")

max1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "max1d_tiled")
max2d_tiled = _reduce_2d_tiled(_tile_max, wp.atomic_max, wp.max, "max2d_tiled")
max_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_max, wp.atomic_max, wp.max, "max_2d_rows_tiled")
max_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_max, wp.atomic_max, wp.max, "max_2d_cols_tiled")

sum1d_tiled = _reduce_1d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum1d_tiled")
sum2d_tiled = _reduce_2d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum2d_tiled")
sum_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_sum, wp.atomic_add, wp.add, "sum_2d_rows_tiled")
sum_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_sum, wp.atomic_add, wp.add, "sum_2d_cols_tiled")

# ---------------------------------------------------------------------------
# Boolean reductions over int32 0/1 masks. ``any`` == OR == max; ``all`` == AND
# == min (a 0/1 mask minimises to 1 iff every element is 1). The 2-D global case
# is handled by the wrapper flattening the mask and reusing the 1-D kernel, so no
# ``*2d_tiled`` bool kernel is needed.
# ---------------------------------------------------------------------------

any_1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "any_1d_tiled")
any_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_max, wp.atomic_max, wp.max, "any_2d_rows_tiled")
any_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_max, wp.atomic_max, wp.max, "any_2d_cols_tiled")

all_1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "all_1d_tiled")
all_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_min, wp.atomic_min, wp.min, "all_2d_rows_tiled")
all_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_min, wp.atomic_min, wp.min, "all_2d_cols_tiled")


# ---------------------------------------------------------------------------
# Min/max together. The dual tile-reduction, dual atomic and two output slots do
# not fit the single-primitive factory template, so these stay hand-written.
# ---------------------------------------------------------------------------


@wp.kernel
def minmax1d_tiled(values: wp.array[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[offset]
        tile_max = values[offset]
        for k in range(1, remaining):
            v = values[offset + k]
            tile_min = wp.min(tile_min, v)
            tile_max = wp.max(tile_max, v)

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)


@wp.kernel
def minmax2d_tiled(values: wp.array2d[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    i, j, t = wp.tid()
    n_rows = values.shape[0]
    n_cols = values.shape[1]
    row_offset = i * TILE_2D
    col_offset = j * TILE_2D
    if row_offset >= n_rows or col_offset >= n_cols:
        return

    remaining_rows = n_rows - row_offset
    remaining_cols = n_cols - col_offset
    tile_rows = TILE_2D if remaining_rows >= TILE_2D else remaining_rows
    tile_cols = TILE_2D if remaining_cols >= TILE_2D else remaining_cols
    if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
        tile = wp.tile_load(
            values, shape=(TILE_2D, TILE_2D), offset=(row_offset, col_offset), storage="register"
        )
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[row_offset, col_offset]
        tile_max = values[row_offset, col_offset]
        for r in range(tile_rows):
            for c in range(tile_cols):
                if r == 0 and c == 0:
                    continue
                v = values[row_offset + r, col_offset + c]
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)


@wp.kernel
def minmax_2d_rows_tiled(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    i, j, t = wp.tid()
    n_cols = values.shape[1]
    col_offset = j * TILE_1D
    if col_offset >= n_cols:
        return
    remaining = n_cols - col_offset
    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=(1, TILE_1D), offset=(i, col_offset), storage="register")
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[i, col_offset]
        tile_max = values[i, col_offset]
        for k in range(1, remaining):
            v = values[i, col_offset + k]
            tile_min = wp.min(tile_min, v)
            tile_max = wp.max(tile_max, v)
    if t == 0:
        wp.atomic_min(out_min, i, tile_min)
        wp.atomic_max(out_max, i, tile_max)


@wp.kernel
def minmax_2d_cols_tiled(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    i, j, t = wp.tid()
    n_rows = values.shape[0]
    row_offset = j * TILE_1D
    if row_offset >= n_rows:
        return
    remaining = n_rows - row_offset
    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=(TILE_1D, 1), offset=(row_offset, i), storage="register")
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[row_offset, i]
        tile_max = values[row_offset, i]
        for k in range(1, remaining):
            v = values[row_offset + k, i]
            tile_min = wp.min(tile_min, v)
            tile_max = wp.max(tile_max, v)
    if t == 0:
        wp.atomic_min(out_min, i, tile_min)
        wp.atomic_max(out_max, i, tile_max)


# ---------------------------------------------------------------------------
# Specialised sum helpers used by other kernel modules and the vec3 / weighted
# reductions. These carry bespoke dtypes (vec3, weight products) that the shared
# scalar template does not cover, so they remain standalone.
# ---------------------------------------------------------------------------


@wp.func
def sum1d_tile(values: wp.array[wp.Scalar], offset: int, remaining: int) -> wp.Scalar:
    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        return wp.tile_sum(tile)[0]

    tile_sum = values[offset]
    for k in range(1, remaining):
        tile_sum += values[offset + k]
    return tile_sum


@wp.func
def weighted_sum1d_tile(
    values: wp.array[wp.float32], weights: wp.array[wp.float32], offset: int, remaining: int
) -> wp.float32:
    if remaining >= TILE_1D:
        v_tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        w_tile = wp.tile_load(weights, shape=TILE_1D, offset=offset, storage="register")
        return wp.tile_sum(v_tile * w_tile)[0]

    tile_sum = wp.float32(0.0)
    for k in range(remaining):
        tile_sum += values[offset + k] * weights[offset + k]
    return tile_sum


@wp.func
def sum_vec3_tile(values: wp.array[wp.vec3], offset: int, remaining: int) -> wp.vec3:
    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        return wp.tile_sum(tile)[0]

    tile_sum = values[offset]
    for k in range(1, remaining):
        tile_sum += values[offset + k]
    return tile_sum


@wp.func
def weighted_sum_vec3_tile(
    values: wp.array[wp.vec3], weights: wp.array[wp.float32], offset: int, remaining: int
) -> wp.vec3:
    if remaining >= TILE_1D:
        v_tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        w_tile = wp.tile_load(weights, shape=TILE_1D, offset=offset, storage="register")
        return wp.tile_sum(v_tile * w_tile)[0]

    tile_sum = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for k in range(remaining):
        tile_sum += weights[offset + k] * values[offset + k]
    return tile_sum


@wp.kernel
def weighted_sum1d_tiled(
    values: wp.array[wp.float32], weights: wp.array[wp.float32], out_sum: wp.array[wp.float32]
) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    tile_sum = weighted_sum1d_tile(values, weights, offset, remaining)

    if t == 0:
        wp.atomic_add(out_sum, 0, tile_sum)


@wp.func
def outer_sum_tile(
    points: wp.array[wp.vec3], center: wp.vec3, offset: int, remaining: int
) -> wp.mat33:
    count = remaining
    if count > TILE_1D:
        count = TILE_1D
    # M = sum_k outer(x_k, x_k) where x_k = points[k] - center
    m = wp.mat33(0.0)
    for k in range(count):
        x = points[offset + k] - center
        m += wp.outer(x, x)
    return m


@wp.func
def cross_outer_sum_tile(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32],
    a_center: wp.vec3,
    b_center: wp.vec3,
    offset: int,
    remaining: int,
) -> wp.mat33:
    count = remaining
    if count > TILE_1D:
        count = TILE_1D
    # masked cross-covariance H = sum_{k: w_k > 0} outer(b_k - b_center, a_k - a_center)
    m = wp.mat33(0.0)
    for k in range(count):
        if weights[offset + k] > wp.float32(0.0):
            ac = a[offset + k] - a_center
            bc = b[offset + k] - b_center
            m += wp.outer(bc, ac)
    return m


@wp.kernel
def sum_vec3_1d_tiled(values: wp.array[wp.vec3], out_sum: wp.array[wp.vec3]) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    tile_sum = sum_vec3_tile(values, offset, remaining)

    if t == 0:
        wp.atomic_add(out_sum, 0, tile_sum)


@wp.kernel
def weighted_sum_vec3_1d_tiled(
    values: wp.array[wp.vec3], weights: wp.array[wp.float32], out_sum: wp.array[wp.vec3]
) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    tile_sum = weighted_sum_vec3_tile(values, weights, offset, remaining)

    if t == 0:
        wp.atomic_add(out_sum, 0, tile_sum)
