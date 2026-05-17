"""Global max reductions: SIMT + ``wp.tile`` (doc pattern) vs ``tile_load`` + ``launch_tiled``."""

import warp as wp

from triwarp.constants import TILE_1D, TILE_2D


@wp.kernel
def max1d_tiled(values: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()

    tile = wp.tile_load(values, shape=TILE_1D, offset=i * TILE_1D, storage="register")
    tile_max = wp.tile_max(tile)[0]

    if t == 0:
        wp.atomic_max(out_max, 0, tile_max)


@wp.kernel
def max2d_tiled(values: wp.array2d[wp.Scalar], out_max: wp.array[wp.Scalar]) -> None:
    i, j, t = wp.tid()

    tile = wp.tile_load(values, shape=(TILE_2D, TILE_2D), offset=(i * TILE_2D, j * TILE_2D), storage="register")
    tile_max = wp.tile_max(tile)[0]

    if t == 0:
        wp.atomic_max(out_max, 0, tile_max)


@wp.kernel
def min1d_tiled(values: wp.array[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()

    tile = wp.tile_load(values, shape=TILE_1D, offset=i * TILE_1D, storage="register")
    tile_min = wp.tile_min(tile)[0]

    if t == 0:
        wp.atomic_min(out_min, 0, tile_min)


@wp.kernel
def min2d_tiled(values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar]) -> None:
    i, j, t = wp.tid()

    tile = wp.tile_load(values, shape=(TILE_2D, TILE_2D), offset=(i * TILE_2D, j * TILE_2D), storage="register")
    tile_min = wp.tile_min(tile)[0]

    if t == 0:
        wp.atomic_min(out_min, 0, tile_min)


@wp.kernel
def minmax1d_tiled(values: wp.array[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    i, t = wp.tid()

    tile = wp.tile_load(values, shape=TILE_1D, offset=i * TILE_1D, storage="register")
    tile_min = wp.tile_min(tile)[0]
    tile_max = wp.tile_max(tile)[0]

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)


@wp.kernel
def minmax2d_tiled(values: wp.array2d[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    i, j, t = wp.tid()

    tile = wp.tile_load(values, shape=(TILE_2D, TILE_2D), offset=(i * TILE_2D, j * TILE_2D), storage="register")
    tile_min = wp.tile_min(tile)[0]
    tile_max = wp.tile_max(tile)[0]

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)
