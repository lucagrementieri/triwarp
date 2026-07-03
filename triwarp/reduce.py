"""Global reductions on Warp arrays."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, cast, overload

import warp as wp

import triwarp.typing as twt
from triwarp.constants import TILE_1D, TILE_2D
from triwarp.kernels import array as kernel_array
from triwarp.kernels import reduce as kernel_reduce


@overload
def min(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def min(array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...) -> float: ...
@overload
def min(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def min(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Minimum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min`` + ``wp.atomic_min``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype using one thread per output element.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["min"]))


@overload
def max(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def max(array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...) -> float: ...
@overload
def max(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def max(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Maximum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_max`` + ``wp.atomic_max``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype using one thread per output element.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["max"]))


@overload
def minmax(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> tuple[int, int]: ...
@overload
def minmax(
    array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...
) -> tuple[float, float]: ...
@overload
def minmax(
    array: twt.Array2dScalar, *, axis: Literal[0, 1]
) -> tuple[twt.Array1dScalar, twt.Array1dScalar]: ...
def minmax(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    """
    Minimum and maximum of ``array``.

    With ``axis=None`` (default), reduces every element to two Python scalars using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min``/``tile_max`` +
    ``wp.atomic_min``/``atomic_max``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    pair of 1D ``wp.array`` buffers (min, max) of the same dtype.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for per-axis 1D
        results (rank-2 input only).

    Returns
    -------
    tuple[float, float] | tuple[int, int] | tuple[wp.array, wp.array]
        ``(min, max)`` as Python scalars when ``axis=None``; pair of 1D arrays
        of length ``n`` (``axis=1``) or ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(
        tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar],
        _reduce_scalar(array, axis, _SCALAR_REDUCE["minmax"]),
    )


def any(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical OR.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` ``wp.bool`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        Length ``array.shape[1]`` when ``axis=0``, else ``array.shape[0]``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["any"])


@overload
def sum(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def sum(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def sum(array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...) -> float: ...
@overload
def sum(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: None = ...) -> int: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> twt.Array1dInt32: ...
def sum(
    array: twt.ScalarArray | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | int | twt.Array1dScalar | twt.Array1dInt32 | wp.vec3:
    """
    Sum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_sum`` + ``wp.atomic_add``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype using one thread per output element.

    For ``wp.bool`` input, counts ``True`` values and returns ``int`` (global) or
    ``wp.int32`` (per-axis).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar or ``wp.bool`` Warp array.
        Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("sum over a vec3 array supports only axis=None.")
        if array.ndim != 1:
            raise ValueError("sum over a vec3 array requires a 1D array.")
        n = int(array.shape[0])
        if n == 0:
            raise ValueError("sum requires a non-empty array.")
        out_vec = wp.zeros(1, dtype=wp.vec3, device=array.device)
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            kernel_reduce.sum_vec3_1d_tiled,
            dim=[n_tiles],
            inputs=[array, out_vec],
            block_dim=TILE_1D,
            device=array.device,
        )
        return wp.vec3(*out_vec.numpy()[0].tolist())
    if array.dtype == wp.bool:
        mask = cast(wp.array[wp.bool], array)
        mask_i32 = _bool_mask_as_int32(mask)
        if mask.ndim == 2 and axis is None:
            mask_i32 = mask_i32.flatten()
        result = _reduce_scalar(cast(twt.ScalarArray, mask_i32), axis, _SCALAR_REDUCE["sum"])
        if axis is None:
            return int(cast(int, result))
        return cast(twt.Array1dInt32, result)
    return cast(
        float | int | twt.Array1dScalar,
        _reduce_scalar(cast(twt.ScalarArray, array), axis, _SCALAR_REDUCE["sum"]),
    )


@overload
def mean(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def mean(array: twt.ScalarArray | wp.array[wp.bool], *, axis: None = ...) -> float: ...
@overload
def mean(
    array: twt.Array2dScalar | wp.array[wp.bool], *, axis: Literal[0, 1]
) -> twt.Array1dFloat32: ...
def mean(
    array: twt.ScalarArray | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | twt.Array1dFloat32 | wp.vec3:
    """
    Arithmetic mean of ``array``.

    Delegates the reduction to the tiled [`sum`][triwarp.reduce.sum], then divides by the
    element count. With ``axis=None`` (default), returns one Python ``float``. With
    ``axis=0`` or ``axis=1`` on a rank-2 input, returns a 1D ``wp.float32`` array.

    The result is always floating point regardless of input dtype. For ``wp.bool``
    input, the mean is the fraction of ``True`` values. For a 1D ``wp.vec3`` input
    the mean is the component-wise average ``wp.vec3`` (``axis=None`` only).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar or ``wp.bool`` Warp array, or
        a rank-1 ``(n,)`` ``wp.vec3`` array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | wp.vec3 | wp.array
        Global ``float`` (scalar/``wp.bool`` input) or ``wp.vec3`` (``wp.vec3``
        input) when ``axis=None``; 1D ``wp.float32`` array of length ``n``
        (``axis=1``) or ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 (or ``wp.vec3``) input.
    """
    if array.dtype == wp.vec3:
        return cast(wp.vec3, sum(array, axis=axis)) / float(int(array.size))
    total = sum(array, axis=axis)
    if axis is None:
        return float(total) / float(int(array.size))
    sums = cast(twt.Array1dScalar, total)
    out = wp.empty(int(sums.shape[0]), dtype=wp.float32, device=array.device)
    wp.utils.array_cast(sums, out)
    wp.launch(
        kernel_array.divide,
        dim=int(out.shape[0]),
        inputs=[out, wp.float32(array.shape[axis])],
        device=array.device,
    )
    return cast(twt.Array1dFloat32, out)


@overload
def weighted_sum(values: twt.Array1dFloat32, weights: twt.Array1dFloat32) -> float: ...
@overload
def weighted_sum(values: wp.array[wp.vec3], weights: twt.Array1dFloat32) -> wp.vec3: ...
def weighted_sum(
    values: twt.Array1dFloat32 | wp.array[wp.vec3], weights: twt.Array1dFloat32
) -> float | wp.vec3:
    """
    Weighted sum ``sum_i weights[i] * values[i]``.

    Parameters
    ----------
    values
        Rank-1 ``(n,)`` ``wp.float32`` or ``wp.vec3`` array. Must be non-empty.
    weights
        Rank-1 ``(n,)`` ``wp.float32`` array of the same length as ``values``.

    Returns
    -------
    float | wp.vec3
        Scalar weighted sum on the host (``float`` for scalar ``values``,
        ``wp.vec3`` for ``wp.vec3`` ``values``).

    Raises
    ------
    ValueError
        If either array is empty or their lengths differ.
    """
    n_values = int(values.shape[0])
    n_weights = int(weights.shape[0])
    if n_values == 0 or n_weights == 0:
        raise ValueError("weighted_sum requires non-empty arrays.")
    if n_values != n_weights:
        raise ValueError("weighted_sum requires values and weights of equal length.")

    n_tiles = (n_values + TILE_1D - 1) // TILE_1D
    if values.dtype == wp.vec3:
        out_vec = wp.zeros(1, dtype=wp.vec3, device=values.device)
        wp.launch_tiled(
            kernel_reduce.weighted_sum_vec3_1d_tiled,
            dim=[n_tiles],
            inputs=[values, weights, out_vec],
            block_dim=TILE_1D,
            device=values.device,
        )
        return wp.vec3(*out_vec.numpy()[0].tolist())

    out = wp.zeros(1, dtype=wp.float32, device=values.device)
    wp.launch_tiled(
        kernel_reduce.weighted_sum1d_tiled,
        dim=[n_tiles],
        inputs=[values, weights, out],
        block_dim=TILE_1D,
        device=values.device,
    )
    return float(out.numpy().item())


def all(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical AND.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` ``wp.bool`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        Length ``array.shape[1]`` when ``axis=0``, else ``array.shape[0]``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["all"])


@overload
def max_for_dtype(dtype: type[wp.Int]) -> int: ...
@overload
def max_for_dtype(dtype: type[wp.Float]) -> float: ...
def max_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    """
    Largest representable value for a Warp scalar type.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``float("inf")`` for floating-point types; the maximum representable
        integer for integer types.
    """
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
    """
    Smallest representable value for a Warp scalar type.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``float("-inf")`` for floating-point types; the minimum representable
        integer for integer types.
    """
    if not wp.types.type_is_int(dtype):
        return float("-inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if dtype.__name__.lower().startswith("u"):
        return 0
    return -(1 << (bits - 1))


def zero_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    """
    Zero value for a Warp scalar type, typed to match Python's ``int``/``float`` split.

    Parameters
    ----------
    dtype
        A Warp integer or floating-point scalar type.

    Returns
    -------
    int | float
        ``0`` for integer types, ``0.0`` for floating-point types.
    """
    if wp.types.type_is_int(dtype):
        return 0
    return 0.0


class _ScalarReduceSpec(NamedTuple):
    name: str
    axis_rows: Callable[..., None]
    axis_cols: Callable[..., None]
    axis_rows_tiled: Callable[..., None]
    axis_cols_tiled: Callable[..., None]
    tiled_1d: Callable[..., None]
    tiled_2d: Callable[..., None]
    init_global: Callable[[type], int | float]
    global_output_slots: int
    dual_axis: bool


class _BoolReduceSpec(NamedTuple):
    name: str
    axis_rows: Callable[..., None]
    axis_cols: Callable[..., None]
    axis_rows_tiled: Callable[..., None]
    axis_cols_tiled: Callable[..., None]
    tiled_1d: Callable[..., None]
    init_global: int


_SCALAR_REDUCE: dict[str, _ScalarReduceSpec] = {
    "min": _ScalarReduceSpec(
        name="min",
        axis_rows=kernel_reduce.min_2d_rows,
        axis_cols=kernel_reduce.min_2d_cols,
        axis_rows_tiled=kernel_reduce.min_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.min_2d_cols_tiled,
        tiled_1d=kernel_reduce.min1d_tiled,
        tiled_2d=kernel_reduce.min2d_tiled,
        init_global=max_for_dtype,
        global_output_slots=1,
        dual_axis=False,
    ),
    "max": _ScalarReduceSpec(
        name="max",
        axis_rows=kernel_reduce.max_2d_rows,
        axis_cols=kernel_reduce.max_2d_cols,
        axis_rows_tiled=kernel_reduce.max_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.max_2d_cols_tiled,
        tiled_1d=kernel_reduce.max1d_tiled,
        tiled_2d=kernel_reduce.max2d_tiled,
        init_global=min_for_dtype,
        global_output_slots=1,
        dual_axis=False,
    ),
    "minmax": _ScalarReduceSpec(
        name="minmax",
        axis_rows=kernel_reduce.minmax_2d_rows,
        axis_cols=kernel_reduce.minmax_2d_cols,
        axis_rows_tiled=kernel_reduce.minmax_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.minmax_2d_cols_tiled,
        tiled_1d=kernel_reduce.minmax1d_tiled,
        tiled_2d=kernel_reduce.minmax2d_tiled,
        init_global=max_for_dtype,
        global_output_slots=2,
        dual_axis=True,
    ),
    "sum": _ScalarReduceSpec(
        name="sum",
        axis_rows=kernel_reduce.sum_2d_rows,
        axis_cols=kernel_reduce.sum_2d_cols,
        axis_rows_tiled=kernel_reduce.sum_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.sum_2d_cols_tiled,
        tiled_1d=kernel_reduce.sum1d_tiled,
        tiled_2d=kernel_reduce.sum2d_tiled,
        init_global=zero_for_dtype,
        global_output_slots=1,
        dual_axis=False,
    ),
}

_BOOL_REDUCE: dict[str, _BoolReduceSpec] = {
    "any": _BoolReduceSpec(
        name="any",
        axis_rows=kernel_reduce.any_2d_rows,
        axis_cols=kernel_reduce.any_2d_cols,
        axis_rows_tiled=kernel_reduce.any_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.any_2d_cols_tiled,
        tiled_1d=kernel_reduce.any_1d_tiled,
        init_global=0,
    ),
    "all": _BoolReduceSpec(
        name="all",
        axis_rows=kernel_reduce.all_2d_rows,
        axis_cols=kernel_reduce.all_2d_cols,
        axis_rows_tiled=kernel_reduce.all_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.all_2d_cols_tiled,
        tiled_1d=kernel_reduce.all_1d_tiled,
        init_global=1,
    ),
}


def _validate_scalar_array(
    array: twt.ScalarArray, spec: _ScalarReduceSpec, axis: Literal[0, 1] | None
) -> None:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")
    if array.ndim == 1 and axis is not None:
        raise ValueError(f"{spec.name} requires axis=None for a 1D array.")
    if array.ndim not in (1, 2):
        raise ValueError(f"{spec.name} requires a 1D or 2D array.")


def _launch_axis_scalar(
    array: twt.Array2dScalar, axis: Literal[0, 1], spec: _ScalarReduceSpec
) -> twt.Array1dScalar | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
    n_col_tiles = (n_cols + TILE_1D - 1) // TILE_1D
    n_row_tiles = (n_rows + TILE_1D - 1) // TILE_1D
    if spec.dual_axis:
        if axis == 1:
            out_min = wp.full(
                n_rows, max_for_dtype(array.dtype), dtype=array.dtype, device=array.device
            )
            out_max = wp.full(
                n_rows, min_for_dtype(array.dtype), dtype=array.dtype, device=array.device
            )
            wp.launch_tiled(
                spec.axis_rows_tiled,
                dim=[n_rows, n_col_tiles],
                inputs=[array, out_min, out_max],
                block_dim=TILE_1D,
                device=array.device,
            )
        else:
            out_min = wp.full(
                n_cols, max_for_dtype(array.dtype), dtype=array.dtype, device=array.device
            )
            out_max = wp.full(
                n_cols, min_for_dtype(array.dtype), dtype=array.dtype, device=array.device
            )
            wp.launch_tiled(
                spec.axis_cols_tiled,
                dim=[n_cols, n_row_tiles],
                inputs=[array, out_min, out_max],
                block_dim=TILE_1D,
                device=array.device,
            )
        return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)

    if axis == 1:
        out = wp.full(n_rows, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)
        wp.launch_tiled(
            spec.axis_rows_tiled,
            dim=[n_rows, n_col_tiles],
            inputs=[array, out],
            block_dim=TILE_1D,
            device=array.device,
        )
    else:
        out = wp.full(n_cols, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)
        wp.launch_tiled(
            spec.axis_cols_tiled,
            dim=[n_cols, n_row_tiles],
            inputs=[array, out],
            block_dim=TILE_1D,
            device=array.device,
        )
    return cast(twt.Array1dScalar, out)


def _launch_global_scalar_tiled(
    array: twt.ScalarArray, spec: _ScalarReduceSpec
) -> float | int | tuple[float, float] | tuple[int, int]:
    if spec.global_output_slots == 2:
        out = wp.array(
            [max_for_dtype(array.dtype), min_for_dtype(array.dtype)],
            dtype=array.dtype,
            device=array.device,
        )
    else:
        out = wp.full(1, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)

    if array.ndim == 1:
        n = int(array.shape[0])
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            spec.tiled_1d,
            dim=[n_tiles],
            inputs=[array, out],
            block_dim=TILE_1D,
            device=array.device,
        )
    else:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        wp.launch_tiled(
            spec.tiled_2d,
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )

    out_np = out.numpy()
    if spec.global_output_slots == 2:
        return cast(tuple[int, int] | tuple[float, float], (out_np[0].item(), out_np[1].item()))
    return cast(float | int, out_np.item())


def _reduce_scalar(
    array: twt.ScalarArray, axis: Literal[0, 1] | None, spec: _ScalarReduceSpec
) -> (
    float
    | int
    | twt.Array1dScalar
    | tuple[twt.Array1dScalar, twt.Array1dScalar]
    | tuple[float, float]
    | tuple[int, int]
):
    _validate_scalar_array(array, spec, axis)

    if array.ndim == 2 and axis is not None:
        return _launch_axis_scalar(array, axis, spec)

    return _launch_global_scalar_tiled(array, spec)


def _bool_mask_as_int32(mask: wp.array[wp.bool]) -> wp.array[wp.int32]:
    out = wp.empty(mask.shape, dtype=wp.int32, device=mask.device)
    wp.utils.array_cast(mask.flatten(), out.flatten())
    return out


def _launch_global_bool_tiled(mask_i32: wp.array[wp.int32], spec: _BoolReduceSpec) -> bool:
    out = wp.full(1, spec.init_global, dtype=wp.int32, device=mask_i32.device)
    n = int(mask_i32.shape[0])
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        spec.tiled_1d,
        dim=[n_tiles],
        inputs=[mask_i32, out],
        block_dim=TILE_1D,
        device=mask_i32.device,
    )
    return bool(out.numpy().item() != 0)


def _reduce_bool(
    array: wp.array[wp.bool], axis: Literal[0, 1] | None, spec: _BoolReduceSpec
) -> wp.array[wp.bool] | bool:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")

    if array.ndim == 1:
        mask_i32 = _bool_mask_as_int32(array)
        return _launch_global_bool_tiled(mask_i32, spec)

    if array.ndim == 2:
        mask_i32 = _bool_mask_as_int32(array)
        n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
        if axis is None:
            return _launch_global_bool_tiled(mask_i32.flatten(), spec)
        n_col_tiles = (n_cols + TILE_1D - 1) // TILE_1D
        n_row_tiles = (n_rows + TILE_1D - 1) // TILE_1D
        if axis == 1:
            out_i32 = wp.full(n_rows, spec.init_global, dtype=wp.int32, device=array.device)
            wp.launch_tiled(
                spec.axis_rows_tiled,
                dim=[n_rows, n_col_tiles],
                inputs=[mask_i32, out_i32],
                block_dim=TILE_1D,
                device=array.device,
            )
        else:
            out_i32 = wp.full(n_cols, spec.init_global, dtype=wp.int32, device=array.device)
            wp.launch_tiled(
                spec.axis_cols_tiled,
                dim=[n_cols, n_row_tiles],
                inputs=[mask_i32, out_i32],
                block_dim=TILE_1D,
                device=array.device,
            )
        out = wp.empty(out_i32.shape[0], dtype=wp.bool, device=array.device)
        wp.utils.array_cast(out_i32, out)
        return out

    raise ValueError(f"{spec.name} requires a 1D or 2D array.")
