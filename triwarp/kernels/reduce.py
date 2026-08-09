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

from triwarp.constants import TILE_1D, TILE_2D, TILES_PER_BLOCK_1D

_tile_min = _warp_builtins["tile_min"]
_tile_max = _warp_builtins["tile_max"]
_tile_sum = _warp_builtins["tile_sum"]


def blocks_1d(n: int) -> int:
    """
    Launch width for the 1-D global reduction kernels in this module.

    Every ``*1d_tiled`` kernel here folds ``TILES_PER_BLOCK_1D`` tiles per block, so its grid is
    ``n / (TILE_1D * TILES_PER_BLOCK_1D)`` and **not** ``n / TILE_1D``. Call this rather than
    open-coding the division: passing the tile count would give each block the same 16 tiles the
    next 15 blocks also claim, folding every element 16 times over -- and, the fold being
    idempotent for the extrema, that would leave ``min`` / ``max`` / ``any`` / ``all`` looking
    correct while ``sum`` silently returned 16x its answer.

    Parameters
    ----------
    n
        Number of elements in the 1-D array being reduced.

    Returns
    -------
    int
        Block count to pass as ``dim`` to ``wp.launch_tiled`` with ``block_dim=TILE_1D``.
    """
    items = TILE_1D * TILES_PER_BLOCK_1D
    return (n + items - 1) // items


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
    """
    axis=None on a 1-D array: ``TILES_PER_BLOCK_1D`` tiles per block, one atomic per block.

    The accumulator is seeded from the block's *first* chunk rather than from an identity, which
    keeps the kernel generic over ``wp.Scalar`` without the wrapper having to pass a per-dtype
    identity value in. Every subsequent chunk folds in with ``scalar``.
    """

    def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base = i * TILES_PER_BLOCK_1D * TILE_1D
        remaining = n - base
        if remaining <= 0:
            return

        # First chunk seeds the accumulator (both branches assign it -- see CLAUDE.md section 5 on
        # Warp's conditional scoping).
        if remaining >= TILE_1D:
            tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
            result = tile_reduce(tile)[0]
        else:
            result = values[base]
            for k in range(1, remaining):
                result = scalar(result, values[base + k])

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                result = scalar(result, tile_reduce(chunk)[0])
            elif rest > 0:
                for k in range(rest):
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


# When the reduced extent is narrower than ``TILE_1D`` the tiled kernels above never take their
# ``tile_load`` branch — all ``TILE_1D`` lanes of every block redundantly run the serial remainder
# loop, a ``TILE_1D``-fold read amplification (measured 6.6 ms -> 0.13 ms for ``max(axis=1)`` on a
# ``(14M, 3)`` table, 49x). These serial variants launch one plain thread per *output* element and
# write directly: no tiles, no atomics, and no init fill needed on the output buffer. The wrapper
# picks them whenever ``reduced extent < TILE_1D``; past that the tiled kernels stay (a tall
# ``(n, 3)`` table reduced along axis=0 has only 3 outputs — 3 serial threads would be 89x slower).


def _reduce_2d_rows_serial(scalar, name):
    """axis=1 with fewer than ``TILE_1D`` columns: one thread per row, direct write."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i = wp.tid()
        n_cols = values.shape[1]
        result = values[i, 0]
        for k in range(1, n_cols):
            result = scalar(result, values[i, k])
        out[i] = result

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


def _reduce_2d_cols_serial(scalar, name):
    """axis=0 with fewer than ``TILE_1D`` rows: one thread per column, coalesced direct write."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        j = wp.tid()
        n_rows = values.shape[0]
        result = values[0, j]
        for k in range(1, n_rows):
            result = scalar(result, values[k, j])
        out[j] = result

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
min_2d_rows_serial = _reduce_2d_rows_serial(wp.min, "min_2d_rows_serial")
min_2d_cols_serial = _reduce_2d_cols_serial(wp.min, "min_2d_cols_serial")

max1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "max1d_tiled")
max2d_tiled = _reduce_2d_tiled(_tile_max, wp.atomic_max, wp.max, "max2d_tiled")
max_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_max, wp.atomic_max, wp.max, "max_2d_rows_tiled")
max_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_max, wp.atomic_max, wp.max, "max_2d_cols_tiled")
max_2d_rows_serial = _reduce_2d_rows_serial(wp.max, "max_2d_rows_serial")
max_2d_cols_serial = _reduce_2d_cols_serial(wp.max, "max_2d_cols_serial")

sum1d_tiled = _reduce_1d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum1d_tiled")
sum2d_tiled = _reduce_2d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum2d_tiled")
sum_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_sum, wp.atomic_add, wp.add, "sum_2d_rows_tiled")
sum_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_sum, wp.atomic_add, wp.add, "sum_2d_cols_tiled")
sum_2d_rows_serial = _reduce_2d_rows_serial(wp.add, "sum_2d_rows_serial")
sum_2d_cols_serial = _reduce_2d_cols_serial(wp.add, "sum_2d_cols_serial")

# ---------------------------------------------------------------------------
# Boolean reductions over int32 0/1 masks. ``any`` == OR == max; ``all`` == AND
# == min (a 0/1 mask minimises to 1 iff every element is 1). The 2-D global case
# is handled by the wrapper flattening the mask and reusing the 1-D kernel, so no
# ``*2d_tiled`` bool kernel is needed.
# ---------------------------------------------------------------------------

any_1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "any_1d_tiled")
any_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_max, wp.atomic_max, wp.max, "any_2d_rows_tiled")
any_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_max, wp.atomic_max, wp.max, "any_2d_cols_tiled")
any_2d_rows_serial = _reduce_2d_rows_serial(wp.max, "any_2d_rows_serial")
any_2d_cols_serial = _reduce_2d_cols_serial(wp.max, "any_2d_cols_serial")

all_1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "all_1d_tiled")
all_2d_rows_tiled = _reduce_2d_rows_tiled(_tile_min, wp.atomic_min, wp.min, "all_2d_rows_tiled")
all_2d_cols_tiled = _reduce_2d_cols_tiled(_tile_min, wp.atomic_min, wp.min, "all_2d_cols_tiled")
all_2d_rows_serial = _reduce_2d_rows_serial(wp.min, "all_2d_rows_serial")
all_2d_cols_serial = _reduce_2d_cols_serial(wp.min, "all_2d_cols_serial")


# ---------------------------------------------------------------------------
# Min/max together. The dual tile-reduction, dual atomic and two output slots do
# not fit the single-primitive factory template, so these stay hand-written.
# ---------------------------------------------------------------------------


@wp.kernel
def minmax1d_tiled(values: wp.array[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    # Same TILES_PER_BLOCK_1D fold as the factory kernels above, with two accumulators seeded from
    # the block's first chunk; two atomics per block instead of two per tile.
    i, t = wp.tid()
    n = values.shape[0]
    base = i * TILES_PER_BLOCK_1D * TILE_1D
    remaining = n - base
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[base]
        tile_max = values[base]
        for k in range(1, remaining):
            v = values[base + k]
            tile_min = wp.min(tile_min, v)
            tile_max = wp.max(tile_max, v)

    for s in range(1, TILES_PER_BLOCK_1D):
        offset = base + s * TILE_1D
        rest = n - offset
        if rest >= TILE_1D:
            chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
            tile_min = wp.min(tile_min, wp.tile_min(chunk)[0])
            tile_max = wp.max(tile_max, wp.tile_max(chunk)[0])
        elif rest > 0:
            for k in range(rest):
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


@wp.kernel
def minmax_2d_rows_serial(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    # axis=1 with fewer than TILE_1D columns: one thread per row, direct writes (see the serial
    # factories above for why the tiled form loses TILE_1D-fold here).
    i = wp.tid()
    n_cols = values.shape[1]
    row_min = values[i, 0]
    row_max = values[i, 0]
    for k in range(1, n_cols):
        v = values[i, k]
        row_min = wp.min(row_min, v)
        row_max = wp.max(row_max, v)
    out_min[i] = row_min
    out_max[i] = row_max


@wp.kernel
def minmax_2d_cols_serial(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    # axis=0 with fewer than TILE_1D rows: one thread per column, coalesced direct writes.
    j = wp.tid()
    n_rows = values.shape[0]
    col_min = values[0, j]
    col_max = values[0, j]
    for k in range(1, n_rows):
        v = values[k, j]
        col_min = wp.min(col_min, v)
        col_max = wp.max(col_max, v)
    out_min[j] = col_min
    out_max[j] = col_max


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
    count = wp.min(remaining, TILE_1D)
    # M = sum_k outer(x_k, x_k) where x_k = points[k] - center
    m = wp.mat33(0.0)
    for k in range(count):
        x = points[offset + k] - center
        m += wp.outer(x, x)
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


@wp.kernel
def minmax_vec3_chunked(points: wp.array[wp.vec3], out_corners: wp.array[wp.float32]) -> None:
    # Component-wise min and max of a ``wp.vec3`` array, in one launch into one buffer.
    #
    # ``out_corners`` is ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]``, *negating* the upper
    # half so a single ``wp.full(6, inf)`` initializes both and every update is an
    # ``atomic_min``. The alternative — separate min and max buffers — needs two allocations, two
    # fills and two readbacks, and at this size the reduction is entirely host-latency-bound.
    #
    # One thread per ``TILE_1D`` points, so the atomics see a few hundred contenders per address
    # rather than one per point.
    chunk = int(wp.tid())
    offset = chunk * TILE_1D
    remaining = points.shape[0] - offset
    if remaining <= 0:
        return
    count = wp.min(remaining, TILE_1D)

    lower = points[offset]
    upper = points[offset]
    for k in range(1, count):
        p = points[offset + k]
        lower = wp.min(lower, p)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, p)

    for c in range(3):
        wp.atomic_min(out_corners, c, lower[c])
        wp.atomic_min(out_corners, 3 + c, -upper[c])
