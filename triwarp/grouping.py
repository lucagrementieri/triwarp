"""Index grouping by equal value/row, plus value and row deduplication for Warp arrays."""

from __future__ import annotations

import math
from typing import Literal, TypeVar, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import (
    bitcast_from_int,
    bitcast_to_int,
    gather,
    init_range,
    init_sort_pair_indices,
    sortable_dtype,
)
from triwarp.kernels import array as kernel_array
from triwarp.kernels import grouping as kernel_grouping

Scalar = TypeVar("Scalar", bound=wp.Scalar)


def group(values: wp.array[wp.Int], length: int) -> twt.Array2dInt32:
    """
    Return index groups of exactly ``length`` entries that share the same value.

    ``values`` are radix-sorted with their original indices; each output row lists
    ``length`` indices whose corresponding entries are equal and form a run of
    precisely that size (runs shorter or longer than ``length`` are omitted). This
    matches [`trimesh.grouping.group`][] with ``min_len == max_len == length`` and
    [`trimesh.grouping.group_rows`][] with ``require_count=length``.

    Parameters
    ----------
    values
        ``(n,)`` device array of integer keys (any ``wp.Int`` dtype).
    length
        Required run length; each returned group contains exactly this many indices.

    Returns
    -------
    twt.Array2dInt32
        ``(g, length)`` array on ``values.device`` where ``g`` is the number of
        groups found. Empty when no run has exactly ``length`` equal neighbors.

    See Also
    --------
    [`group_int_rows`][triwarp.grouping.group_int_rows]
    """
    n = int(values.shape[0])
    device = values.device
    if n < length or length <= 0:
        return twt.as_array2d_int32(twt.empty_int32_2d((0, max(length, 0)), device=device))

    sort_dtype = sortable_dtype(values.dtype)
    values_buffer = wp.empty(2 * n, dtype=sort_dtype, device=device)
    if sort_dtype == values.dtype:
        wp.copy(values_buffer, values, count=n)
    else:
        wp.utils.array_cast(values, values_buffer, count=n)
    indices_buffer = init_sort_pair_indices(n, -1, device)
    wp.utils.radix_sort_pairs(values_buffer, indices_buffer, count=n)

    # Scan + scatter compaction: mark run starts, compact them with flatnonzero, then emit one
    # right-sized row per group (deterministic ascending-value order, no atomic counter and no
    # (n, length) over-allocation).
    is_start = wp.empty(n, dtype=wp.bool, device=device)
    wp.launch(
        kernel_grouping.mark_group_starts,
        dim=n,
        inputs=[values_buffer, wp.int32(n), wp.int32(length), is_start],
        device=device,
    )
    starts = tw.array.flatnonzero(is_start)
    n_groups = int(starts.shape[0])
    if n_groups == 0:
        return twt.as_array2d_int32(twt.empty_int32_2d((0, length), device=device))
    groups = twt.empty_int32_2d((n_groups, length), device=device)
    wp.launch(
        kernel_grouping.emit_groups,
        dim=n_groups,
        inputs=[starts, indices_buffer, groups],
        device=device,
    )
    return twt.as_array2d_int32(groups)


def group_int_rows(
    data: twt.Array2dInt, length: int, max_value: int | None = None, *, validate: bool = True
) -> twt.Array2dInt32:
    """
    Return index groups of exactly ``length`` rows that are identical.

    Each row is hashed with [`hash_indices_rows`][triwarp.grouping.hash_indices_rows], then
    [`group`][triwarp.grouping.group] finds runs of ``length`` equal keys in sorted order. For
    example, ``[[1, 2], [3, 4], [1, 2]]`` with ``length=2`` yields one group ``[[0, 2]]`` (same as
    [`trimesh.grouping.group_rows`][] with ``require_count=2``).

    Parameters
    ----------
    data
        ``(n, w)`` device array of non-negative ``int32`` row values.
    length
        Required number of duplicate rows per group.
    max_value
        Optional exclusive upper bound on entries and radix for row hashing; passed
        through to [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] as ``max_index``. If
        ``None``, inferred
        from ``max(data) + 1``.
    validate
        Forwarded to [`hash_indices_rows`][triwarp.grouping.hash_indices_rows]. ``False`` skips the
        range-check reduction (and its host readback) and requires ``max_value``; see the warning
        there before using it.

    Returns
    -------
    twt.Array2dInt32
        ``(g, length)`` array on ``data.device`` with original row indices per group.

    See Also
    --------
    [`group`][triwarp.grouping.group]
    """
    twt.ensure_ndim(data, 2)
    hashed_rows = hash_indices_rows(data, max_value, validate=validate)
    return group(hashed_rows, length)


@overload
def unique_1d(
    data: wp.array[Scalar],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> wp.array[Scalar]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: bool = False, return_counts: bool = False
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find sorted unique elements of a 1D Warp array (``numpy.unique`` subset).

    Uses an open-addressing hash table: O(n) inserts plus an O(n_unique log n_unique) sort of only
    the unique values.

    ``return_inverse`` maps each input position to the index of its value in the sorted unique
    output.
    ``return_index`` is not supported.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` with any scalar ``dtype``.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique : wp.array
        Sorted unique values (same ``dtype`` and ``device`` as ``data``).
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique`` such
        that conceptually ``unique[inverse[i]] == data[i]`` (float ``NaN`` matches
        NumPy: all ``NaN`` values share one slot).
    counts : wp.array[wp.int32], optional
        Present if ``return_counts=True``. Number of occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-1 or length is ``> 2**30``.
    """
    if int(data.ndim) != 1:
        raise ValueError(f"unique_1d expects a rank-1 array, got ndim={data.ndim}")

    device = data.device
    n = int(data.shape[0])

    if n == 0:
        empty_unique = wp.empty(0, dtype=data.dtype, device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )
    if n > (1 << 30):
        raise ValueError(f"unique_1d requires length <= 2**30, got length {n}")

    log2_capacity = max(3, math.ceil(math.log2(n) + 1))
    mask = wp.int32((1 << log2_capacity) - 1)

    return _unique_hash(
        data, bitcast_to_int(data, n), data.dtype, n, mask, return_inverse, return_counts
    )


@overload
def unique_rows(
    data: wp.array[wp.vec3],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> wp.array[wp.vec3]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> twt.Array2dInt32: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[twt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> twt.Array2dFloat32: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = False,
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True],
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_rows(
    data: wp.array, *, return_inverse: bool = False, return_counts: bool = False
) -> (
    wp.array
    | tuple[wp.array, wp.array[wp.int32]]
    | tuple[wp.array, wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find unique rows of a 2D Warp array (``numpy.unique`` along axis 0).

    Each row is hashed with [`hash_rows`][triwarp.grouping.hash_rows], then deduplicated via the
    same open-addressing hash table as [`unique_1d`][triwarp.grouping.unique_1d]. Unique rows are
    returned in
    sorted hash-key order; the representative row for each key is the first input
    row in that equivalence class.

    ``return_inverse`` maps each input row index to its slot in the sorted unique
    output. ``return_index`` is not supported.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or length-``n`` ``wp.vec3`` array.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique
        ``(n_unique, w)`` or length-``n_unique`` ``wp.vec3`` array on ``data.device``.
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique``.
    counts : wp.array[wp.int32], optional
        Present if ``return_counts=True``. Occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-2 ``int32``/``float32`` or rank-1 ``wp.vec3``.
    """
    device = data.device
    n = int(data.shape[0])
    is_vec3 = data.dtype == wp.vec3

    if n == 0:
        if is_vec3:
            empty_unique = wp.empty(0, dtype=wp.vec3, device=device)
        elif data.dtype == wp.int32:
            empty_unique = twt.empty_int32_2d((0, int(data.shape[1])), device=device)
        else:
            empty_unique = twt.empty_float32_2d((0, int(data.shape[1])), device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )

    row_keys = hash_rows(data)
    keys_result = unique_1d(row_keys, return_inverse=True, return_counts=return_counts)
    if return_counts:
        unique_keys, inverse, counts = keys_result
    else:
        unique_keys, inverse = keys_result
        counts = None

    # The class count is the length of the unique-key array ``unique_1d`` just returned; recovering
    # it as ``reduce.max(inverse) + 1`` would be a whole reduction launch and a host sync for a
    # number already in hand.
    first_idx = first_occurrence_indices(inverse, int(unique_keys.shape[0]))

    if not is_vec3:
        twt.ensure_ndim(data, 2)
    # ``gather`` performs both rank-1 (vec3) and rank-2 (int32/float32 row) gather.
    unique_rows_out = gather(data, first_idx)

    if not return_inverse:
        inverse = None
    return _pack_unique_result(unique_rows_out, inverse=inverse, counts=counts)


@overload
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: Literal[False] = False
) -> wp.array[wp.int32]: ...
@overload
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: Literal[True]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: bool = False
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Find unique triangular faces up to vertex permutation (orientation-agnostic).

    Two faces are equal when they share the same three vertices regardless of order. Each
    face's indices are sorted before deduplication; the representative returned for a class is
    the first occurring face, with its original vertex order preserved.

    Parameters
    ----------
    faces
        Flat ``wp.int32`` triangle index buffer of length ``3 * n_faces`` (common 1D format).
    return_inverse
        If ``True``, also return the inverse mapping from each input face to its slot in the
        unique output.

    Returns
    -------
    unique_faces : wp.array[wp.int32]
        Flat buffer of the unique faces (length ``3 * n_unique``).
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. Length ``n_faces``; ``inverse[i]`` is the unique
        slot of input face ``i``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_faces = wp.empty(0, dtype=wp.int32, device=device)
        if return_inverse:
            return empty_faces, wp.empty(0, dtype=wp.int32, device=device)
        return empty_faces

    faces2d = faces.reshape((-1, 3))
    sorted_faces = twt.empty_int32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_grouping.sort_face_indices,
        dim=n_faces,
        inputs=[faces2d, sorted_faces],
        device=device,
    )
    unique_sorted_faces, inverse = unique_rows(sorted_faces, return_inverse=True)
    # As in ``unique_rows``: the count is a shape, not something to reduce for.
    first = first_occurrence_indices(inverse, int(unique_sorted_faces.shape[0]))
    unique_faces_out = gather(faces2d, first).reshape((-1,))
    if return_inverse:
        return unique_faces_out, inverse
    return unique_faces_out


def first_occurrence_indices(
    inverse: wp.array[wp.int32], n_unique: int | None = None
) -> wp.array[wp.int32]:
    """
    Index of the first element of each equivalence class in an ``inverse`` map.

    The representative-picking half of every deduplication in this package: given the
    ``inverse`` that ``unique_1d`` / ``unique_rows`` return, produce for each class the smallest
    input index belonging to it. Gathering any per-element payload by the result yields one
    representative per class, taken from its **first** occurrence -- which is what makes
    ``unique_faces`` keep the original winding and ``edges_unique`` keep the first-seen edge.

    Parameters
    ----------
    inverse
        Length-``n`` ``wp.int32`` map from each element to its class slot, as returned by
        [`unique_1d`][triwarp.grouping.unique_1d] or
        [`unique_rows`][triwarp.grouping.unique_rows].
    n_unique
        Number of classes (the output length). Defaults to ``int(reduce.max(inverse)) + 1``,
        which costs a device reduction **and** a host synchronization; pass it whenever the
        caller already knows it. It is almost always a ``.shape[0]`` the caller is holding --
        the length of the unique array that came back beside ``inverse``.

    Returns
    -------
    wp.array[wp.int32]
        Length-``n_unique`` array whose entry ``c`` is ``min{i : inverse[i] == c}``. A class with
        no member (only reachable when ``n_unique`` is passed too large) holds the sentinel ``n``.

    See Also
    --------
    [`unique_rows`][triwarp.grouping.unique_rows]
    [`unique_1d`][triwarp.grouping.unique_1d]
    [`gather`][triwarp.array.gather]
        The companion step: gather the payload by these indices to get the representatives.
    """
    device = inverse.device
    n = int(inverse.shape[0])
    if n_unique is None:
        n_unique = int(tw.reduce.max(inverse)) + 1 if n > 0 else 0
    first = wp.full(n_unique, wp.int32(n), dtype=wp.int32, device=device)
    if n > 0 and n_unique > 0:
        wp.launch(
            kernel_grouping.scatter_first_occurrence, dim=n, inputs=[inverse, first], device=device
        )
    return first


@overload
def hash_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: twt.Array2dInt32) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: twt.Array2dFloat32) -> wp.array[wp.uint64]: ...
def hash_rows(
    data: wp.array[wp.vec3] | twt.Array2dInt32 | twt.Array2dFloat32,
) -> wp.array[wp.uint64]:
    """
    Pack each row of a 2D or ``wp.vec3`` array into a single ``uint64`` key.

    Dispatches to [`hash_vector_rows`][triwarp.grouping.hash_vector_rows] for ``wp.vec3``
    (including ``float32`` arrays with width 3) or
    [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] for ``int32`` rows.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or length-``n`` ``wp.vec3`` array.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``data`` has an unsupported dtype or shape.
    """
    if data.dtype == wp.vec3:
        return hash_vector_rows(data)
    twt.ensure_ndim(data, 2)
    if data.dtype == wp.int32:
        return hash_indices_rows(data)
    if data.dtype == wp.float32:
        n = int(data.shape[0])
        if int(data.shape[1]) != 3:
            raise ValueError("float32 hash_rows currently requires width 3")
        vec = wp.empty(n, dtype=wp.vec3, device=data.device)
        wp.utils.array_cast(data, vec)
        return hash_vector_rows(vec)
    raise ValueError(f"hash_rows unsupported dtype {data.dtype}")


def hash_vector_rows(data: wp.array[wp.vec3], epsilon: float = 0.0) -> wp.array[wp.uint64]:
    """
    Pack each ``wp.vec3`` row into a single ``uint64`` key.

    Two schemes, selected by ``epsilon``:

    - ``epsilon > 0.0`` — an **absolute** tolerance. Each coordinate is snapped to a multiple of
      ``epsilon`` measured from the data's own minimum corner, and the resulting integer row is
      packed via [`hash_indices_rows`][triwarp.grouping.hash_indices_rows]. Equal keys mean the rows
      landed in the same grid cell. Snapping relative to the minimum corner rather than to the
      coordinate origin is what lets negative coordinates work at all, and it keeps the scaled
      values at the size of the data's extent, where ``float32`` still resolves ``epsilon``.
    - ``epsilon == 0.0`` — a **relative** bucket. Each coordinate's ``float32`` bits are
      right-shifted by 11, so a key names an interval roughly ``2 ** -12`` (about ``2.4e-4``) wide
      relative to the coordinate's own magnitude. This is *not* an equality test in either
      direction, and callers that need one should pass an explicit ``epsilon``: see Notes.

    Parameters
    ----------
    data
        ``(n,)`` device array of ``wp.vec3`` values.
    epsilon
        Uniqueness tolerance. ``0`` uses the relative bit-truncation bucket; positive values snap
        coordinates to ``round(v / epsilon)`` before packing.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    Notes
    -----
    Both schemes quantize, so both split a pair that straddles a cell boundary no matter how close
    the two values are. For ``epsilon > 0`` the packing is additionally injective only while
    ``radix ** 3`` fits a ``uint64``, where ``radix`` is the widest per-axis extent in cells; beyond
    that the row keys wrap and behave as a hash with a small collision probability rather than an
    exact cell identity. The relative scheme additionally:

    - **collides distinct coordinates** that share a bucket — ``1.0`` and ``1.000244`` produce the
      same key, so a mesh whose vertex spacing is below ``2.4e-4`` relative is merged too
      aggressively;
    - has *unbounded* resolution approaching zero, so ``+1e-6`` and ``-1e-6`` are thousands of
      buckets apart (correct — they are distinct points), but so are ``+1e-40`` and ``-1e-40``,
      which for most purposes are the same point. Only ``+0.0`` and ``-0.0`` are folded together,
      because IEEE-754 defines them as equal.

    Neither is a defect of the packing so much as the nature of a fixed-width key; pass an explicit
    ``epsilon`` when the tolerance has to be one you chose.

    See Also
    --------
    [`hash_indices_rows`][triwarp.grouping.hash_indices_rows]
    [`hash_rows`][triwarp.grouping.hash_rows]
    """
    if data.dtype != wp.vec3:
        raise ValueError(f"data must be a wp.array[wp.vec3], got wp.array[{data.dtype}]")
    n = int(data.shape[0])
    if epsilon > 0.0:
        if n == 0:
            return wp.empty(0, dtype=wp.uint64, device=data.device)
        # `hash_indices_rows` packs each row as digits in a positive radix, so negative cell indices
        # -- which every mesh spanning the origin produces -- cannot be packed. Snapping relative to
        # the data's own minimum corner makes them non-negative *by construction*: subtracting the
        # true minimum cannot give a negative result, so no validation pass or shift is needed. The
        # bounds also supply the radix, and doing it per component keeps that radix as small as the
        # widest single extent rather than the whole diagonal, which matters because the row packing
        # is only injective while ``radix ** 3`` fits a ``uint64``.
        min_bound, max_bound = tw.bounds.aabb_bounds(data)
        rounded = twt.empty_int32_2d((n, 3), device=data.device)
        wp.launch(
            kernel_grouping.round_vec3_scaled,
            dim=n,
            inputs=[data, min_bound, wp.float32(1.0 / epsilon), rounded],
            device=data.device,
        )
        extent = max(float(max_bound[c]) - float(min_bound[c]) for c in range(3)) / epsilon
        # The device rounds a float32 product where this divides in float64; the relative slack plus
        # the half-cell of rounding covers the difference, and an over-wide radix is harmless.
        radix = int(extent * (1.0 + 1e-6)) + 3
        return hash_indices_rows(rounded, max_index=radix, validate=False)
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.map(kernel_grouping.pack_vec3, data, out=hashes)
    return hashes


def hash_indices_rows(
    data: twt.Array2dInt32, max_index: int | None = None, *, validate: bool = True
) -> wp.array[wp.uint64]:
    """
    Pack each row of non-negative ``int32`` values into a single ``uint64`` key.

    Each row is treated as digits in base ``max_index`` (or ``max(data)+1`` when
    ``max_index`` is omitted) and combined into one integer key per row. A global
    min/max over ``data`` (via ``triwarp.reduce.minmax``) validates the range and
    supplies the radix when it is inferred.

    Parameters
    ----------
    data
        ``(n, w)`` device array of ``int32``; every entry must be non-negative. If
        ``max_index`` is given, every entry must satisfy ``entry < max_index``.
    max_index
        Optional exclusive upper bound on entries and radix for packing; must be
        positive when provided. If ``None``, set to ``max(data) + 1`` after validation.
    validate
        When ``True`` (default), a ``triwarp.reduce.minmax`` over ``data`` checks that entries are
        non-negative and below ``max_index``. That reduction ends in a host readback, which
        serialises the device pipeline — measurably so when this is called once per pass inside a
        remeshing loop. Pass ``False``, together with an explicit ``max_index``, to skip it when the
        bound is already guaranteed by construction (mesh edge rows are built from face indices, so
        they are non-negative and below the vertex count by definition).

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``max_index`` is not positive, if ``validate=False`` is passed without a ``max_index``,
        or -- when validating -- if ``data`` is negative or reaches ``max_index``.

    Warnings
    --------
    ``validate=False`` with a ``max_index`` smaller than the true maximum silently produces
    colliding keys, and therefore wrong groupings, rather than raising. Only use it where the bound
    is structurally guaranteed.

    The mixed-radix key is injective only while ``max_index ** w`` fits a ``uint64``; past that the
    positional sum wraps and equal keys no longer imply equal rows. Two 32-bit columns always fit,
    but three columns need ``max_index <= 2 ** (64 / 3)``, about ``2.6e6``. Beyond that the keys
    degrade from an exact row identity into a hash whose collision probability grows with the square
    of the row count -- around ``2e-5`` for 28M rows at a radix of ``1.4e7``. This is deliberately
    not validated: raising would reject meshes that group correctly in practice.

    See Also
    --------
    [`hash_vector_rows`][triwarp.grouping.hash_vector_rows]
    [`hash_rows`][triwarp.grouping.hash_rows]
    """
    twt.ensure_ndim(data, 2, dtype=wp.int32)
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    if not validate:
        if max_index is None:
            raise ValueError("validate=False requires an explicit max_index (the radix to use).")
    else:
        # The min/max is a device reduction with a host readback, so it serialises the pipeline.
        # It is unavoidable when the radix has to be inferred, and skippable via validate=False
        # when the caller already knows the bound.
        min_data, max_data = tw.reduce.minmax(data)
        if min_data < 0:
            raise ValueError(f"data must be non-negative, got a minimum of {min_data}")
        if max_index is not None and max_data >= max_index:
            raise ValueError(
                f"data must be less than max_index {max_index}, got a maximum of {max_data}"
            )
        if max_index is None:
            max_index = max_data + 1
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(
        kernel_grouping.pack_indices,
        dim=data.shape[0],
        inputs=[data, wp.uint64(max_index), hashes],
        device=data.device,
    )
    return hashes


def _unique_hash(
    data: wp.array[Scalar],
    data_int: wp.array[wp.int32] | wp.array[wp.int64],
    original_dtype: type[Scalar],
    n: int,
    mask: wp.int32,
    return_inverse: bool,
    return_counts: bool,
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    # One slot past the table is reserved for the single key that collides with the empty-slot
    # sentinel; see the comment on ``kernel_grouping.hash_insert``.
    cap = int(mask) + 2
    key_dtype = data_int.dtype
    device = data_int.device

    # Phase 1: parallel insert into open-addressing hash table (slot_key 0 = empty).
    slot_key = wp.zeros(cap, dtype=key_dtype, device=device)
    slot_counts = wp.zeros(cap, dtype=wp.int32, device=device)
    wp.launch(
        kernel_grouping.hash_insert,
        dim=n,
        inputs=[data_int, slot_key, slot_counts, mask],
        device=device,
    )

    # Phase 2: mark occupied slots, prefix-scan to get compact positions.
    occ_mask = wp.empty(cap, dtype=wp.int32, device=device)
    wp.map(kernel_grouping.mark_occupied, slot_counts, out=occ_mask)
    scan_pos = wp.empty(cap, dtype=wp.int32, device=device)
    wp.utils.array_scan(occ_mask, scan_pos, inclusive=True)
    wp.map(wp.sub, scan_pos, wp.int32(1), out=scan_pos)
    # ``scan_pos`` is an inclusive scan minus one, so its last element *is* ``n_unique - 1``: one
    # 4-byte tail read sizes the output, where ``reduce.max`` would scan all ``cap`` (~2n) slots.
    # The same idiom as ``array.flatnonzero`` and ``array.counts_to_offsets``.
    n_unique = int(scan_pos[cap - 1 :].numpy()[0]) + 1

    # Phase 3: compact unique keys and their occurrence counts.
    keys_compact = wp.empty(n_unique, dtype=key_dtype, device=device)
    cnts_compact = wp.empty(n_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_grouping.compact_from_table,
        dim=cap,
        inputs=[slot_key, slot_counts, occ_mask, scan_pos, keys_compact, cnts_compact],
        device=device,
    )

    # Phase 4: sort only the n_unique keys (typically n_unique << n), in a dtype that orders them
    # the way the caller's dtype does rather than by their reinterpreted bit pattern.
    sort_dtype = sortable_dtype(original_dtype)
    keys_buf = bitcast_from_int(keys_compact, sort_dtype, count=2 * n_unique)
    perm_buf = init_range(2 * n_unique, device)
    wp.utils.radix_sort_pairs(keys_buf, perm_buf, count=n_unique)

    if sort_dtype == original_dtype:
        unique_values = wp.empty(n_unique, dtype=original_dtype, device=device)
        wp.copy(unique_values, keys_buf, count=n_unique)
    else:
        # A dtype narrower than 32 bits was widened to be sortable; narrow it back.
        unique_values = bitcast_from_int(
            bitcast_to_int(keys_buf, n_unique), original_dtype, count=n_unique
        )

    unique_counts = None
    if return_counts:
        sort_perm = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.copy(sort_perm, perm_buf, count=n_unique)
        unique_counts = gather(cnts_compact, sort_perm)

    unique_inverse = None
    if return_inverse:
        sorted_dense = wp.empty(n_unique, dtype=sort_dtype, device=device)
        wp.copy(sorted_dense, keys_buf, count=n_unique)
        # The binary search has to probe in the same space the keys were sorted in.
        data_sorted_space = (
            data if data.dtype == sort_dtype else bitcast_from_int(data_int, sort_dtype, count=n)
        )
        unique_inverse = wp.empty(n, dtype=wp.int32, device=device)
        wp.launch(
            kernel_array.map_sorted_inverse,
            dim=n,
            inputs=[data_sorted_space, sorted_dense, unique_inverse],
            device=device,
        )

    return _pack_unique_result(unique_values, inverse=unique_inverse, counts=unique_counts)


def _pack_unique_result(
    unique: wp.array,
    *,
    inverse: wp.array[wp.int32] | None = None,
    counts: wp.array[wp.int32] | None = None,
) -> (
    wp.array
    | tuple[wp.array, wp.array[wp.int32]]
    | tuple[wp.array, wp.array[wp.int32], wp.array[wp.int32]]
):
    """Assemble the ``(unique[, inverse][, counts])`` return tuple shared by the unique_* family."""
    if inverse is not None and counts is not None:
        return unique, inverse, counts
    if inverse is not None:
        return unique, inverse
    if counts is not None:
        return unique, counts
    return unique
