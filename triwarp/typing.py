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
    Array2dVec3: TypeAlias = wp.array[wp.vec3, Literal[2]]
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
    Array2dVec3 = wp.array
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
    "as_array2d",
    "as_array3d",
    "dtype_max",
    "dtype_min",
    "dtype_zero",
    "empty_2d",
    "empty_3d",
    "ensure_ndim",
    "sortable_dtype",
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


@overload
def as_array2d(arr: wp.array[T], dtype: type[wp.int32]) -> Array2dInt32: ...
@overload
def as_array2d(arr: wp.array[T], dtype: type[wp.float32]) -> Array2dFloat32: ...
@overload
def as_array2d(arr: wp.array[T], dtype: type[wp.float64]) -> Array2dFloat64: ...
@overload
def as_array2d(arr: wp.array[T], dtype: type[wp.vec3]) -> Array2dVec3: ...
def as_array2d(arr: wp.array[T], dtype: type) -> Array2dInt32 | Array2dFloat | Array2dVec3:
    """
    Validate and narrow a Warp array to the rank-2 alias for ``dtype``.

    One function for what used to be ``as_array2d_int32`` / ``as_array2d_float32`` /
    ``as_array2d_float``: the dtype selects the return alias through overloads, so a call site keeps
    the narrow type it had -- ``as_array2d(x, wp.int32)`` is an
    [`Array2dInt32`][triwarp.typing.Array2dInt32], not a union.

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 with scalar type ``dtype``.
    dtype
        Expected scalar type: ``wp.int32``, ``wp.float32`` or ``wp.float64``.

    Returns
    -------
    Array2dInt32 | Array2dFloat32 | Array2dFloat64
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 with scalar type ``dtype``.
    """
    ensure_ndim(arr, 2, dtype=dtype)
    return cast(Array2dInt32 | Array2dFloat, arr)


@overload
def as_array3d(arr: wp.array[T], dtype: type[wp.float32]) -> Array3dFloat32: ...
@overload
def as_array3d(arr: wp.array[T], dtype: type[wp.bool]) -> Array3dBool: ...
def as_array3d(arr: wp.array[T], dtype: type) -> Array3dFloat32 | Array3dBool:
    """
    Validate and narrow a Warp array to the rank-3 alias for ``dtype``.

    The rank-3 counterpart of [`as_array2d`][triwarp.typing.as_array2d].

    Parameters
    ----------
    arr
        Warp array expected to be rank-3 with scalar type ``dtype``.
    dtype
        Expected scalar type: ``wp.float32`` or ``wp.bool``.

    Returns
    -------
    Array3dFloat32 | Array3dBool
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-3 with scalar type ``dtype``.
    """
    ensure_ndim(arr, 3, dtype=dtype)
    return cast(Array3dFloat32 | Array3dBool, arr)


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


def sortable_dtype(dtype: type[wp.Scalar]) -> type[wp.Scalar]:
    """
    Same-width dtype that ``warp.utils.radix_sort_pairs`` accepts, preserving ``dtype``'s order.

    The single widening rule for every radix sort in this package. Sorting a sub-32-bit dtype is
    not supported by Warp, and sorting the reinterpreted *bits* of a float or an unsigned value
    gets the order wrong, so callers ask here rather than widening ad hoc.

    The hash table works in one common signed-integer key space (see
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]), which is fine for equality but wrong for
    ordering: negative floats have descending bit patterns, and a ``uint64`` with its top bit set
    reads as a negative ``int64``. Warp sorts ``int32`` / ``int64`` / ``uint32`` / ``uint64`` /
    ``float32`` / ``float64`` keys directly, so the sort is done in this dtype instead of on the
    reinterpreted bits. The set is unchanged through Warp 1.17.0 (re-probed on both devices: every
    narrower width -- ``int8`` / ``uint8`` / ``int16`` / ``uint16`` / ``float16`` -- still raises
    ``Unsupported keys and values data types``), so the widening table below still has a case for
    each of them.

    Parameters
    ----------
    dtype
        Any Warp scalar dtype.

    Returns
    -------
    type[wp.Scalar]
        ``dtype`` itself when Warp can already sort it, otherwise the narrowest same-signedness,
        same-kind dtype it can (``float32`` / ``float64``, ``uint32`` / ``uint64``, ``int32`` /
        ``int64``), chosen by whether ``dtype`` is wider than four bytes.

    See Also
    --------
    [`sort_and_argsort`][triwarp.array.sort_and_argsort]
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
    """
    wide = wp.types.type_size_in_bytes(dtype) > 4
    if wp.types.type_is_float(dtype):
        return wp.float64 if wide else wp.float32
    if dtype.__name__.lower().startswith("u"):
        return wp.uint64 if wide else wp.uint32
    return wp.int64 if wide else wp.int32


@overload
def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type[wp.int32], *, device: wp.DeviceLike = None
) -> Array2dInt32: ...
@overload
def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type[wp.float32], *, device: wp.DeviceLike = None
) -> Array2dFloat32: ...
@overload
def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type[wp.float64], *, device: wp.DeviceLike = None
) -> Array2dFloat64: ...
@overload
def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type[wp.vec3], *, device: wp.DeviceLike = None
) -> Array2dVec3: ...
def empty_2d(
    shape: tuple[int, int] | list[int], dtype: type, *, device: wp.DeviceLike = None
) -> Array2dInt32 | Array2dFloat | Array2dVec3:
    """
    Allocate an uninitialized rank-2 Warp array of the given scalar type.

    One function for what used to be ``empty_int32_2d`` / ``empty_float32_2d`` / ``empty_float_2d``.
    The dtype selects the return alias through overloads, so a call site keeps the narrow type it
    had rather than falling back to a union -- verified with ``reveal_type`` under this repo's
    basedpyright config, which is the only place Warp's stubs resolve.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    dtype
        Scalar type: ``wp.int32``, ``wp.float32`` or ``wp.float64``.
    device
        Target Warp device.

    Returns
    -------
    Array2dInt32 | Array2dFloat32 | Array2dFloat64
        Uninitialized ``(rows, cols)`` array of scalar type ``dtype`` on ``device``.
    """
    return cast(Array2dInt32 | Array2dFloat, wp.empty(_shape_2d(shape), dtype=dtype, device=device))


def _shape_2d(shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 2:
        raise ValueError(f"2D shape must have length 2, got {shape!r}")
    return (dims[0], dims[1])


@overload
def empty_3d(
    shape: tuple[int, int, int] | list[int],
    dtype: type[wp.float32],
    *,
    device: wp.DeviceLike = None,
) -> Array3dFloat32: ...
@overload
def empty_3d(
    shape: tuple[int, int, int] | list[int], dtype: type[wp.bool], *, device: wp.DeviceLike = None
) -> Array3dBool: ...
def empty_3d(
    shape: tuple[int, int, int] | list[int], dtype: type, *, device: wp.DeviceLike = None
) -> Array3dFloat32 | Array3dBool:
    """
    Allocate an uninitialized rank-3 Warp array of the given scalar type.

    The rank-3 counterpart of [`empty_2d`][triwarp.typing.empty_2d].

    Parameters
    ----------
    shape
        ``(nx, ny, nz)`` shape of the allocated array.
    dtype
        Scalar type: ``wp.float32`` or ``wp.bool``.
    device
        Target Warp device.

    Returns
    -------
    Array3dFloat32 | Array3dBool
        Uninitialized ``(nx, ny, nz)`` array of scalar type ``dtype`` on ``device``.
    """
    return cast(
        Array3dFloat32 | Array3dBool, wp.empty(_shape_3d(shape), dtype=dtype, device=device)
    )


def _shape_3d(shape: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 3:
        raise ValueError(f"3D shape must have length 3, got {shape!r}")
    return (dims[0], dims[1], dims[2])
