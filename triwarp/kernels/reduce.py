"""Global reductions: tiled ``wp.tile_load`` + SIMT axis paths."""

import warp as wp

from triwarp.constants import TILE_1D, TILE_2D


@wp.kernel
def max1d_tiled(values: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_max = values[offset]
        for k in range(1, remaining):
            tile_max = wp.max(tile_max, values[offset + k])

    if t == 0:
        wp.atomic_max(out_max, 0, tile_max)


@wp.kernel
def max2d_tiled(values: wp.array2d[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
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
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_max = values[row_offset, col_offset]
        for r in range(tile_rows):
            for c in range(tile_cols):
                if r == 0 and c == 0:
                    continue
                tile_max = wp.max(tile_max, values[row_offset + r, col_offset + c])

    if t == 0:
        wp.atomic_max(out_max, 0, tile_max)


@wp.kernel
def min1d_tiled(values: wp.array[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()
    n = values.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
        tile_min = wp.tile_min(tile)[0]
    else:
        tile_min = values[offset]
        for k in range(1, remaining):
            tile_min = wp.min(tile_min, values[offset + k])

    if t == 0:
        wp.atomic_min(out_min, 0, tile_min)


@wp.kernel
def min2d_tiled(values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
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
    else:
        tile_min = values[row_offset, col_offset]
        for r in range(tile_rows):
            for c in range(tile_cols):
                if r == 0 and c == 0:
                    continue
                tile_min = wp.min(tile_min, values[row_offset + r, col_offset + c])

    if t == 0:
        wp.atomic_min(out_min, 0, tile_min)


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
def any_1d_tiled(mask: wp.array[wp.int32], out: wp.array[wp.int32]) -> None:
    i, t = wp.tid()
    n = mask.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(mask, shape=TILE_1D, offset=offset, storage="register")
        tile_any = wp.tile_max(tile)[0]
    else:
        tile_any = wp.int32(0)
        for k in range(remaining):
            tile_any = wp.max(tile_any, mask[offset + k])

    if t == 0:
        wp.atomic_max(out, 0, tile_any)


@wp.kernel
def all_1d_tiled(mask: wp.array[wp.int32], out: wp.array[wp.int32]) -> None:
    i, t = wp.tid()
    n = mask.shape[0]
    offset = i * TILE_1D
    remaining = n - offset
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(mask, shape=TILE_1D, offset=offset, storage="register")
        tile_all = wp.int32(wp.tile_sum(tile)[0] == TILE_1D)
    else:
        chunk_sum = wp.int32(0)
        for k in range(remaining):
            chunk_sum += mask[offset + k]
        tile_all = wp.int32(chunk_sum == remaining)

    if t == 0:
        wp.atomic_min(out, 0, tile_all)


@wp.kernel
def min_2d_rows(values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
    """axis=1: one thread per row, min across columns."""
    i = int(wp.tid())
    result = values[i, 0]
    for j in range(1, values.shape[1]):
        result = wp.min(result, values[i, j])
    out_min[i] = result


@wp.kernel
def min_2d_cols(values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
    """axis=0: one thread per column, min across rows."""
    i = int(wp.tid())
    result = values[0, i]
    for row in range(1, values.shape[0]):
        result = wp.min(result, values[row, i])
    out_min[i] = result


@wp.kernel
def max_2d_rows(values: wp.array2d[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
    """axis=1: one thread per row, max across columns."""
    i = int(wp.tid())
    result = values[i, 0]
    for j in range(1, values.shape[1]):
        result = wp.max(result, values[i, j])
    out_max[i] = result


@wp.kernel
def max_2d_cols(values: wp.array2d[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
    """axis=0: one thread per column, max across rows."""
    i = int(wp.tid())
    result = values[0, i]
    for row in range(1, values.shape[0]):
        result = wp.max(result, values[row, i])
    out_max[i] = result


@wp.kernel
def minmax_2d_rows(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    """axis=1: one thread per row, min and max across columns."""
    i = int(wp.tid())
    vmin = values[i, 0]
    vmax = values[i, 0]
    for j in range(1, values.shape[1]):
        v = values[i, j]
        vmin = wp.min(vmin, v)
        vmax = wp.max(vmax, v)
    out_min[i] = vmin
    out_max[i] = vmax


@wp.kernel
def minmax_2d_cols(
    values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
) -> None:
    """axis=0: one thread per column, min and max across rows."""
    i = int(wp.tid())
    vmin = values[0, i]
    vmax = values[0, i]
    for row in range(1, values.shape[0]):
        v = values[row, i]
        vmin = wp.min(vmin, v)
        vmax = wp.max(vmax, v)
    out_min[i] = vmin
    out_max[i] = vmax


@wp.kernel
def any_2d_rows(mask: wp.array2d[wp.int32], out: wp.array[wp.bool]) -> None:
    """axis=1: one thread per row, OR-reduce across columns."""
    i = int(wp.tid())
    result = wp.int32(0)
    for j in range(mask.shape[1]):
        result = wp.max(result, mask[i, j])
    out[i] = result != 0


@wp.kernel
def any_2d_cols(mask: wp.array2d[wp.int32], out: wp.array[wp.bool]) -> None:
    """axis=0: one thread per column, OR-reduce across rows."""
    i = int(wp.tid())
    result = wp.int32(0)
    for row in range(mask.shape[0]):
        result = wp.max(result, mask[row, i])
    out[i] = result != 0


@wp.kernel
def all_2d_rows(mask: wp.array2d[wp.int32], out: wp.array[wp.bool]) -> None:
    """axis=1: one thread per row, AND-reduce across columns."""
    i = int(wp.tid())
    result = wp.int32(1)
    for j in range(mask.shape[1]):
        result = wp.min(result, mask[i, j])
    out[i] = result != 0


@wp.kernel
def all_2d_cols(mask: wp.array2d[wp.int32], out: wp.array[wp.bool]) -> None:
    """axis=0: one thread per column, AND-reduce across rows."""
    i = int(wp.tid())
    result = wp.int32(1)
    for row in range(mask.shape[0]):
        result = wp.min(result, mask[row, i])
    out[i] = result != 0
