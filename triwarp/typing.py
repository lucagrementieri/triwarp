"""
Warp array type aliases and runtime rank checks for Python wrappers.

Use these aliases in ``triwarp`` Python APIs instead of ``wp.array2d[dtype]``, which
static checkers treat as Warp annotation objects (no ``.shape`` / indexing).

Kernels should continue to use ``wp.array2d[dtype]`` in ``@wp.kernel`` signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast, overload

import warp as wp

T = TypeVar("T")

if TYPE_CHECKING:
    Array1dInt32: TypeAlias = wp.array[wp.int32, Literal[1]]
    Array1dFloat32: TypeAlias = wp.array[wp.float32, Literal[1]]
    Array1dFloat64: TypeAlias = wp.array[wp.float64, Literal[1]]
    Array2dInt32: TypeAlias = wp.array[wp.int32, Literal[2]]
    Array2dFloat32: TypeAlias = wp.array[wp.float32, Literal[2]]
    Array2dFloat64: TypeAlias = wp.array[wp.float64, Literal[2]]
    Array3dFloat32: TypeAlias = wp.array[wp.float32, Literal[3]]
    Array3dBool: TypeAlias = wp.array[wp.bool, Literal[3]]

    # wp.Int / wp.Float / wp.Scalar are TypeVars; subscripting wp.array[...] with them
    # yields a generic alias that requires type arguments under basedpyright.
    Array1dInt: TypeAlias = Array1dInt32
    Array2dInt: TypeAlias = Array2dInt32
    Array1dFloat: TypeAlias = Array1dFloat32 | Array1dFloat64
    Array2dFloat: TypeAlias = Array2dFloat32 | Array2dFloat64
    Array1dScalar: TypeAlias = Array1dInt32 | Array1dFloat
    Array2dScalar: TypeAlias = Array2dInt32 | Array2dFloat

    IntArray: TypeAlias = Array1dInt | Array2dInt
    FloatArray: TypeAlias = Array1dFloat | Array2dFloat
    ScalarArray: TypeAlias = Array1dScalar | Array2dScalar

    # Any rank, for the handful of operations that are genuinely rank-agnostic because they
    # flatten and reshape back (``array.isin``, ``array.gather``). Prefer a concrete rank alias
    # everywhere else -- this one deliberately gives up the rank check.
    ArrayNdInt32: TypeAlias = wp.array[wp.int32, Any]
    ArrayNd: TypeAlias = wp.array[Any, Any]
else:
    Array1dInt32 = wp.array
    Array1dFloat32 = wp.array
    Array1dFloat64 = wp.array
    Array2dInt32 = wp.array
    Array2dFloat32 = wp.array
    Array2dFloat64 = wp.array
    Array3dFloat32 = wp.array
    Array3dBool = wp.array
    Array1dInt = wp.array
    Array2dInt = wp.array
    Array1dFloat = wp.array
    Array2dFloat = wp.array
    Array1dScalar = wp.array
    Array2dScalar = wp.array
    IntArray = wp.array
    FloatArray = wp.array
    ScalarArray = wp.array
    ArrayNdInt32 = wp.array
    ArrayNd = wp.array

__all__ = [
    "Array1dFloat",
    "Array1dFloat32",
    "Array1dFloat64",
    "Array1dInt",
    "Array1dInt32",
    "Array1dScalar",
    "Array2dFloat",
    "Array2dFloat32",
    "Array2dFloat64",
    "Array2dInt",
    "Array2dInt32",
    "Array2dScalar",
    "Array3dBool",
    "Array3dFloat32",
    "ArrayNd",
    "ArrayNdInt32",
    "FloatArray",
    "IntArray",
    "ScalarArray",
    "as_array2d_float",
    "as_array2d_float32",
    "as_array2d_int32",
    "as_array3d_bool",
    "as_array3d_float32",
    "dtype_max",
    "dtype_min",
    "dtype_zero",
    "empty_bool_3d",
    "empty_float32_2d",
    "empty_float32_3d",
    "empty_float_2d",
    "empty_int32_2d",
    "ensure_ndim",
]


def ensure_ndim(arr: wp.array[T], ndim: int, *, dtype: type | None = None) -> wp.array[T]:
    """
    Validate the rank (and optionally the dtype) of a Warp array.

    The runtime counterpart to the aliases in this module: `isinstance(x, wp.array2d)` is always
    ``False``, so rank is checked here instead.

    Parameters
    ----------
    arr
        Array to validate.
    ndim
        Required rank.
    dtype
        When given, the required element dtype.

    Returns
    -------
    wp.array
        ``arr`` unchanged, so this can wrap an argument in place.

    Raises
    ------
    TypeError
        If the rank or dtype does not match.
    """
    if int(arr.ndim) != ndim:
        raise TypeError(f"expected {ndim}D array, got ndim={arr.ndim}")
    if dtype is not None and arr.dtype != dtype:
        raise TypeError(f"expected dtype {dtype}, got {arr.dtype}")
    return arr


def as_array2d_int32(arr: wp.array[T]) -> Array2dInt32:
    """
    Validate and narrow a Warp array to [`Array2dInt32`][triwarp.typing.Array2dInt32].

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 ``int32``.

    Returns
    -------
    Array2dInt32
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 ``int32``.
    """
    ensure_ndim(arr, 2, dtype=wp.int32)
    return cast(Array2dInt32, arr)


def as_array2d_float32(arr: wp.array[T]) -> Array2dFloat32:
    """
    Validate and narrow a Warp array to [`Array2dFloat32`][triwarp.typing.Array2dFloat32].

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 ``float32``.

    Returns
    -------
    Array2dFloat32
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 ``float32``.
    """
    ensure_ndim(arr, 2, dtype=wp.float32)
    return cast(Array2dFloat32, arr)


def as_array2d_float(arr: wp.array[T], *, dtype: type = wp.float32) -> Array2dFloat:
    """
    Validate and narrow a Warp array to [`Array2dFloat`][triwarp.typing.Array2dFloat].

    The dtype-parameterized analogue of
    [`as_array2d_float32`][triwarp.typing.as_array2d_float32]: use it for functions whose
    result precision is chosen at call time (``wp.float32`` or ``wp.float64``).

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 with scalar type ``dtype``.
    dtype
        Expected floating-point scalar type: ``wp.float32`` (default) or ``wp.float64``.

    Returns
    -------
    Array2dFloat
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 with scalar type ``dtype``.
    """
    ensure_ndim(arr, 2, dtype=dtype)
    return cast(Array2dFloat, arr)


def as_array3d_float32(arr: wp.array[T]) -> Array3dFloat32:
    """
    Validate and narrow a Warp array to [`Array3dFloat32`][triwarp.typing.Array3dFloat32].

    Parameters
    ----------
    arr
        Warp array expected to be rank-3 ``float32``.

    Returns
    -------
    Array3dFloat32
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-3 ``float32``.
    """
    ensure_ndim(arr, 3, dtype=wp.float32)
    return cast(Array3dFloat32, arr)


def as_array3d_bool(arr: wp.array[T]) -> Array3dBool:
    """
    Validate and narrow a Warp array to [`Array3dBool`][triwarp.typing.Array3dBool].

    Parameters
    ----------
    arr
        Warp array expected to be rank-3 ``wp.bool``.

    Returns
    -------
    Array3dBool
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-3 ``wp.bool``.
    """
    ensure_ndim(arr, 3, dtype=wp.bool)
    return cast(Array3dBool, arr)


@overload
def dtype_max(dtype: type[wp.Int]) -> int: ...
@overload
def dtype_max(dtype: type[wp.Float]) -> float: ...
def dtype_max(dtype: type[wp.Scalar]) -> int | float:
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
def dtype_min(dtype: type[wp.Int]) -> int: ...
@overload
def dtype_min(dtype: type[wp.Float]) -> float: ...
def dtype_min(dtype: type[wp.Scalar]) -> int | float:
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


def dtype_zero(dtype: type[wp.Scalar]) -> int | float:
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


def _shape_2d(shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 2:
        raise ValueError(f"2D shape must have length 2, got {shape!r}")
    return (dims[0], dims[1])


def empty_int32_2d(
    shape: tuple[int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array2dInt32:
    """
    Allocate an uninitialized rank-2 ``int32`` Warp array.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array2dInt32
        Uninitialized ``(rows, cols)`` ``int32`` array on ``device``.
    """
    return cast(Array2dInt32, wp.empty(_shape_2d(shape), dtype=wp.int32, device=device))


def empty_float32_2d(
    shape: tuple[int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array2dFloat32:
    """
    Allocate an uninitialized rank-2 ``float32`` Warp array.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array2dFloat32
        Uninitialized ``(rows, cols)`` ``float32`` array on ``device``.
    """
    return cast(Array2dFloat32, wp.empty(_shape_2d(shape), dtype=wp.float32, device=device))


def _shape_3d(shape: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3:
        raise ValueError(f"3D shape must have length 3, got {shape!r}")
    return (dims[0], dims[1], dims[2])


def empty_bool_3d(
    shape: tuple[int, int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array3dBool:
    """
    Allocate an uninitialized rank-3 ``wp.bool`` Warp array.

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array3dBool
        Uninitialized ``(nx, ny, nz)`` ``wp.bool`` array on ``device``.
    """
    return cast(Array3dBool, wp.empty(_shape_3d(shape), dtype=wp.bool, device=device))


def empty_float32_3d(
    shape: tuple[int, int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array3dFloat32:
    """
    Allocate an uninitialized rank-3 ``float32`` Warp array.

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array3dFloat32
        Uninitialized ``(nx, ny, nz)`` ``float32`` array on ``device``.
    """
    return cast(Array3dFloat32, wp.empty(_shape_3d(shape), dtype=wp.float32, device=device))


def empty_float_2d(
    shape: tuple[int, int] | list[int], *, dtype: type = wp.float32, device: wp.DeviceLike = None
) -> Array2dFloat:
    """
    Allocate an uninitialized rank-2 floating-point Warp array of the given precision.

    The dtype-parameterized analogue of
    [`empty_float32_2d`][triwarp.typing.empty_float32_2d].

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    dtype
        Floating-point scalar type: ``wp.float32`` (default) or ``wp.float64``.
    device
        Target Warp device.

    Returns
    -------
    Array2dFloat
        Uninitialized ``(rows, cols)`` array of scalar type ``dtype`` on ``device``.
    """
    return cast(Array2dFloat, wp.empty(_shape_2d(shape), dtype=dtype, device=device))
