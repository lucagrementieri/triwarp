"""Global reductions on Warp arrays."""

import warp as wp

from triwarp.kernels import reduce as kernel_reduce
from triwarp.constants import TILE_1D, TILE_2D
from typing import overload, Union  # pyright: ignore[reportDeprecated]


@overload
def max_for_dtype(dtype: type[wp.Int]) -> int: ...
@overload
def max_for_dtype(dtype: type[wp.Float]) -> float: ...
def max_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    if not wp.types.type_is_int(dtype):
        return float("inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if not dtype.__name__.lower().startswith("u"):
        bits -= 1
    return (1 << bits) - 1


@overload
def min_for_dtype(dtype: type[wp.Int]) -> int: ...
@overload
def min_for_dtype(dtype: type[wp.Float]) -> float: ...
def min_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    if not wp.types.type_is_int(dtype):
        return float("-inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if dtype.__name__.lower().startswith("u"):
        return 0
    return -(1 << (bits - 1))


@overload
def min(array: Union[wp.array[wp.Int], wp.array2d[wp.Int]]) -> int: ...
@overload
def min(array: Union[wp.array[wp.Float], wp.array2d[wp.Float]]) -> float: ...
def min(array: Union[wp.array[wp.Scalar], wp.array2d[wp.Scalar]]) -> float | int:
    """
    Global minimum of ``array`` (reduce every element to one scalar).

    Tiled kernels on ``array.device`` load fixed-size patches with ``wp.tile_load``,
    reduce each tile with ``wp.tile_min``, and merge tile minima into a length-1
    buffer using ``wp.atomic_min`` (``warp.launch_tiled``; tile widths from
    ``TILE_1D`` / ``TILE_2D`` in :mod:`triwarp.constants`).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.

    Returns
    -------
    float | int
        Smallest element as a Python scalar from ``out.numpy().item()`` (typically
        ``float`` for floating ``array.dtype``, ``int`` for integer dtypes).

    Raises
    ------
    ValueError
        If ``array`` is empty or its rank is not 1 or 2.
    """
    if int(array.size) == 0:
        raise ValueError("min requires a non-empty array.")

    out = wp.full(1, max_for_dtype(array.dtype), dtype=array.dtype, device=array.device)

    if array.ndim == 1:
        n = int(array.shape[0])
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            kernel_reduce.min1d_tiled, dim=[n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
        )
    elif array.ndim == 2:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        wp.launch_tiled(
            kernel_reduce.min2d_tiled,
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )
    else:
        raise ValueError("min requires a 1D or 2D array.")

    return out.numpy().item()


@overload
def max(array: Union[wp.array[wp.Int], wp.array2d[wp.Int]]) -> int: ...
@overload
def max(array: Union[wp.array[wp.Float], wp.array2d[wp.Float]]) -> float: ...
def max(array: Union[wp.array[wp.Scalar], wp.array2d[wp.Scalar]]) -> float | int:
    """
    Global maximum of ``array`` (reduce every element to one scalar).

    Tiled kernels on ``array.device`` load fixed-size patches with ``wp.tile_load``,
    reduce each tile with ``wp.tile_max``, and merge tile maxima into a length-1
    buffer using ``wp.atomic_max`` (``warp.launch_tiled``; tile widths from
    ``TILE_1D`` / ``TILE_2D`` in :mod:`triwarp.constants`).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.

    Returns
    -------
    float | int
        Largest element as a Python scalar from ``out.numpy().item()`` (typically
        ``float`` for floating ``array.dtype``, ``int`` for integer dtypes).

    Raises
    ------
    ValueError
        If ``array`` is empty or its rank is not 1 or 2.
    """
    if int(array.size) == 0:
        raise ValueError("max requires a non-empty array.")

    out = wp.full(1, min_for_dtype(array.dtype), dtype=array.dtype, device=array.device)

    if array.ndim == 1:
        n = int(array.shape[0])
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            kernel_reduce.max1d_tiled, dim=[n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
        )
    elif array.ndim == 2:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        wp.launch_tiled(
            kernel_reduce.max2d_tiled,
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )
    else:
        raise ValueError("max requires a 1D or 2D array.")

    return out.numpy().item()


@overload
def minmax(array: Union[wp.array[wp.Int], wp.array2d[wp.Int]]) -> tuple[int, int]: ...
@overload
def minmax(array: Union[wp.array[wp.Float], wp.array2d[wp.Float]]) -> tuple[float, float]: ...
def minmax(array: Union[wp.array[wp.Scalar], wp.array2d[wp.Scalar]]) -> tuple[float, float] | tuple[int, int]:
    """
    Global minimum and maximum of ``array`` (reduce every element to two scalars).

    Tiled kernels on ``array.device`` load fixed-size patches with ``wp.tile_load``,
    reduce each tile with ``wp.tile_min`` and ``wp.tile_max``, and merge tile minima and maxima into length-1
    buffer using ``wp.atomic_max`` (``warp.launch_tiled``; tile widths from
    ``TILE_1D`` / ``TILE_2D`` in :mod:`triwarp.constants`).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.

    Returns
    -------
    float | int
        Tuple of smallest and largest elements as Python scalars from ``out.numpy().item()`` (typically
        ``float`` for floating ``array.dtype``, ``int`` for integer dtypes).

    Raises
    ------
    ValueError
        If ``array`` is empty or its rank is not 1 or 2.
    """
    if int(array.size) == 0:
        raise ValueError("minmax requires a non-empty array.")

    out = wp.array([max_for_dtype(array.dtype), min_for_dtype(array.dtype)], dtype=array.dtype, device=array.device)

    if array.ndim == 1:
        n = int(array.shape[0])
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            kernel_reduce.minmax1d_tiled, dim=[n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
        )
    elif array.ndim == 2:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        wp.launch_tiled(
            kernel_reduce.minmax2d_tiled,
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )
    else:
        raise ValueError("minmax requires a 1D or 2D array.")

    out_min, out_max = out.list()
    return out_min, out_max
