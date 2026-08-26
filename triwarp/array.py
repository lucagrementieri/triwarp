"""NumPy-style structural and elementwise ops on Warp arrays (ranges, gather, sort, masks)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TypeVar

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar
from triwarp.kernels import array as kernel_array
from triwarp.kernels import scatter as kernel_scatter

DType = TypeVar("DType")

# Use a direct-index membership table when the value *span* (max - min + 1, over both inputs) is at
# most this multiple of |test_elements|.
_ISIN_MASK_SIZE_FACTOR = 8

# Row width up to which [`sort_rows`][triwarp.array.sort_rows] uses a per-row insertion sort instead
# of a segmented radix sort. Comfortably above every in-library row width (edges 2, corners 3).
SORT_ROWS_INSERTION_MAX_COLS = 8


def arange(n: int, device: wp.DeviceLike, *, dtype: type[wp.Int] = wp.int32) -> wp.array:
    """
    Fill ``out[i] = i`` for ``i`` in ``[0, n)`` (``numpy.arange``).

    Parameters
    ----------
    n
        Number of elements; must be non-negative.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result. Must be able to represent ``n - 1``.

    Returns
    -------
    wp.array
        Length-``n`` array of consecutive indices on ``device``.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``n - 1`` does not fit in ``dtype``.
    """
    dtype = _ensure_int_dtype(dtype)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > 0:
        _check_int_fits(dtype, n - 1, "n")
    out = wp.empty(n, dtype=dtype, device=device)
    if n > 0:
        wp.launch(kernel_array.init_range, dim=n, inputs=[out], device=device)
    return out


def arange_step(
    count: int, step: int, device: wp.DeviceLike, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``out[i] = i * step`` (``numpy.arange(0, count * step, step)``).

    Parameters
    ----------
    count
        Number of elements; must be non-negative.
    step
        Stride between consecutive values; must be non-negative.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result. Must be able to represent ``(count - 1) * step``.

    Returns
    -------
    wp.array
        Length-``count`` array on ``device``.

    Raises
    ------
    ValueError
        If ``count`` or ``step`` is negative, or the largest value does not fit in ``dtype``.
    """
    dtype = _ensure_int_dtype(dtype)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if count > 0:
        _check_int_fits(dtype, (count - 1) * step, "count * step")
        _check_int_fits(dtype, step, "step")
    out = wp.empty(count, dtype=dtype, device=device)
    if count > 0:
        wp.launch(
            kernel_array.init_range_step,
            dim=count,
            inputs=[_int_scalar(dtype, step), out],
            device=device,
        )
    return out


def sort_pair_indices(
    n: int, fill_value: int, device: str, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``[0, 1, ..., n-1, fill_value, ..., fill_value]`` (length ``2 * n``).

    The payload buffer ``warp.utils.radix_sort_pairs`` wants: the first half seeded with the
    identity permutation, the second half (its scratch) filled with a padding value.

    Parameters
    ----------
    n
        Number of real entries; the result has length ``2 * n``.
    fill_value
        Padding written into the upper half.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result.

    Returns
    -------
    wp.array
        Length-``2 * n`` array on ``device``.

    Raises
    ------
    ValueError
        If ``n`` is negative, or ``n - 1`` / ``fill_value`` does not fit in ``dtype``.

    See Also
    --------
    [`sort_and_argsort`][triwarp.array.sort_and_argsort]
    """
    dtype = _ensure_int_dtype(dtype)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n > 0:
        _check_int_fits(dtype, n - 1, "n")
    _check_int_fits(dtype, fill_value, "fill_value")
    out = wp.empty(2 * n, dtype=dtype, device=device)
    if n > 0:
        wp.launch(
            kernel_array.init_sort_pair_indices,
            dim=2 * n,
            inputs=[_int_scalar(dtype, n), _int_scalar(dtype, fill_value), out],
            device=device,
        )
    return out


def repeat_range(
    count: int, repeats: int, device: str, *, dtype: type[wp.Int] = wp.int32
) -> wp.array:
    """
    Fill ``out[i] = i // repeats`` (``numpy.repeat`` of an index range).

    Parameters
    ----------
    count
        Number of elements; must be non-negative.
    repeats
        How many consecutive entries share an index; must be positive.
    device
        Warp device for the result.
    dtype
        Integer dtype of the result.

    Returns
    -------
    wp.array
        Length-``count`` array on ``device``.

    Raises
    ------
    ValueError
        If ``count`` is negative, ``repeats`` is not positive, or the largest value does not fit
        in ``dtype``.
    """
    dtype = _ensure_int_dtype(dtype)
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if count > 0:
        _check_int_fits(dtype, (count - 1) // repeats, "count // repeats")
    out = wp.empty(count, dtype=dtype, device=device)
    if count > 0:
        wp.launch(
            kernel_array.init_repeat_index,
            dim=count,
            inputs=[_int_scalar(dtype, repeats), out],
            device=device,
        )
    return out


def pack_1d_arrays(
    arrays: Sequence[wp.array[wp.Scalar]], *, copy: bool = True
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
    """
    Concatenate several 1-D ``warp.array`` instances into one buffer plus per-segment offsets.

    Segment ``i`` starts at ``offsets[i]`` in ``flat``; its length is ``arrays[i].size``, so it
    occupies ``flat[offsets[i] : offsets[i] + arrays[i].size]``. This is the usual packed
    representation for variable-length per-item lists on the device (no nested arrays).

    Parameters
    ----------
    arrays
        Non-empty sequence of 1-D arrays sharing the same ``dtype`` and ``device``.
    copy
        Keep ``False`` for a read-only result. ``arrays`` that are already consecutive non-empty
        views of one buffer -- what [`split`][triwarp.array.split] returns with ``copy=False`` --
        are then handed back as that buffer's span instead of being copied into a new one, so the
        ``split`` round trip costs nothing at all: measured at **2.505 ms of ``loop_perimeters``'
        2.633 ms on ``dragon``**, 407 ``warp.copy`` launches to move 17 kB. The default copies, so
        writing into ``flat`` is safe; with ``copy=False`` such a write reaches the segments.

    Returns
    -------
    flat
        1-D array of length ``sum(a.size for a in arrays)``, same ``dtype`` and ``device`` as
        the inputs.
    offsets
        Length ``len(arrays)`` (the start offset of each segment, an exclusive scan of the
        segment sizes), ``dtype`` ``wp.int32``, same ``device`` as the inputs. ``offsets[0] == 0``.
        This is *not* a total-terminated CSR array — there is no ``offsets[-1] == flat.size``
        terminator, so the last segment's length must be taken from ``arrays[-1].size`` (or
        ``flat.size - offsets[-1]``).

    Raises
    ------
    ValueError
        If ``arrays`` is empty, any input is not rank-1, or their ``dtype`` differs.

    See Also
    --------
    [`split`][triwarp.array.split]
        The inverse: recovers the per-segment arrays from ``(flat, offsets)``.
    [`concatenate`][triwarp.array.concatenate]
    """
    flat, offsets = _pack_segments(arrays, caller="pack_1d_arrays", copy=copy)
    return flat, wp.array(offsets, dtype=wp.int32, device=flat.device)


def concatenate(arrays: Sequence[wp.array[DType]], *, copy: bool = True) -> wp.array[DType]:
    """
    Concatenate 1-D ``warp.array`` instances in order (``numpy.concatenate``).

    Parameters
    ----------
    arrays
        Non-empty sequence of rank-1 arrays sharing the same ``dtype`` and ``device``.
        Empty segments are allowed.
    copy
        Keep ``False`` for a read-only result, which then costs nothing when the segments already
        tile one buffer -- see [`pack_1d_arrays`][triwarp.array.pack_1d_arrays].

    Returns
    -------
    wp.array
        Contiguous 1-D array of length ``sum(a.size for a in arrays)`` on the input
        device. When ``arrays`` has a single element, that array is returned without
        copying whatever ``copy`` says, since there is nothing to concatenate it with.

    Raises
    ------
    ValueError
        If ``arrays`` is empty, any input is not rank-1, or their ``dtype`` differs.

    See Also
    --------
    [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]
    [`concatenate`][triwarp.combine.concatenate]
        The mesh-level operation of the same name, which joins ``(vertices, faces)`` pairs and
        reindexes as it goes. Both names are required: this one mirrors
        [`numpy.concatenate`][], that one [`trimesh.util.concatenate`][].
    [`numpy.concatenate`][]
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")
    if len(arrays) == 1:
        arr = arrays[0]
        if int(arr.ndim) != 1:
            raise ValueError(f"concatenate requires rank-1 arrays, got ndim={arr.ndim}")
        return arr
    return _pack_segments(arrays, caller="concatenate", copy=copy)[0]


def split(
    array: wp.array[DType], offsets: wp.array[wp.int32], *, copy: bool = False
) -> list[wp.array[DType]]:
    """
    Break a packed 1-D array into its per-segment arrays (``numpy.split``).

    The inverse of [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]: segment ``i`` is
    ``array[offsets[i] : offsets[i + 1]]``, with the last segment running to the end of
    ``array``. One host readback of ``offsets``, then zero copies by default — the segments are
    views into ``array``, which they keep alive.

    Parameters
    ----------
    array
        Rank-1 array to split, any ``dtype``.
    offsets
        Length-``n_segments`` ``wp.int32`` exclusive prefix sum of the segment sizes, starting
        at ``0`` — exactly what [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] and
        [`counts_to_offsets`][triwarp.array.counts_to_offsets] return. The total-terminated
        ``n + 1`` form (``include_total=True``) is also accepted; its trailing entry simply
        yields one final empty segment, so pass the length-``n`` form when that matters.
    copy
        When ``True``, return independent ``wp.clone`` copies instead of views.

    Returns
    -------
    list[wp.array]
        One array per segment, on ``array.device``, in segment order. Empty list when
        ``offsets`` is empty.

    Raises
    ------
    ValueError
        If ``array`` or ``offsets`` is not rank-1, or ``offsets`` is not a non-decreasing
        sequence starting at ``0`` and bounded by ``array``'s length.

    See Also
    --------
    [`pack_1d_arrays`][triwarp.array.pack_1d_arrays]
        The inverse: packs per-segment arrays into one buffer plus these offsets.
    [`split`][triwarp.combine.split]
        The mesh-level operation of the same name, which separates a mesh into connected
        components. Both names are required: this one mirrors [`numpy.split`][], that one
        ``trimesh.Trimesh.split``.
    [`numpy.split`][]
    """
    if int(array.ndim) != 1:
        raise ValueError(f"split requires a rank-1 array, got ndim={array.ndim}")
    if int(offsets.ndim) != 1:
        raise ValueError(f"split requires rank-1 offsets, got ndim={offsets.ndim}")

    n = int(array.shape[0])
    starts = [int(start) for start in offsets.numpy().tolist()]
    if not starts:
        return []
    bounds = [*starts, n]
    if starts[0] != 0 or any(a > b for a, b in itertools.pairwise(bounds)):
        raise ValueError(
            f"offsets must start at 0 and be non-decreasing within [0, {n}], got {starts}"
        )
    # Warp rejects a zero-length slice at the very end of a buffer (``arr[n:n]``) while accepting
    # an interior one, so an empty trailing segment needs its own allocation.
    segments = [
        array[begin:end] if end > begin else wp.empty(0, dtype=array.dtype, device=array.device)
        for begin, end in itertools.pairwise(bounds)
    ]
    return [wp.clone(segment) for segment in segments] if copy else segments


def _pack_segments(
    arrays: Sequence[wp.array[DType]], *, caller: str, copy: bool = True
) -> tuple[wp.array[DType], list[int]]:
    """
    Validate rank-1 segments and copy them into one contiguous buffer.

    The shared body of [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] and
    [`concatenate`][triwarp.array.concatenate], which differ only in whether the caller wants the
    segment offsets back as a device array. Returns them as a Python list so ``concatenate`` pays
    nothing for the offsets it discards.

    With ``copy=False`` the result may be one of the inputs' own storage rather than a fresh
    buffer -- see [`_tiled_span`][triwarp.array._tiled_span] for when, and for why the choice
    cannot be made here.
    """
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty")

    dtype = arrays[0].dtype
    device = arrays[0].device
    sizes = []
    for i, arr in enumerate(arrays):
        if int(arr.ndim) != 1:
            raise ValueError(f"{caller} requires rank-1 arrays, got ndim={arr.ndim} at index {i}")
        if arr.dtype != dtype:
            raise ValueError(
                f"all arrays must have the same dtype, got {dtype} and {arr.dtype} at index {i}"
            )
        sizes.append(int(arr.shape[0]))

    offsets = list(itertools.accumulate(sizes[:-1], initial=0))
    total = offsets[-1] + sizes[-1]
    if not copy:
        already_packed = _tiled_span(arrays, sizes, total)
        if already_packed is not None:
            return already_packed, offsets

    flat = wp.empty(total, dtype=dtype, device=device)
    for arr, offset, n in zip(arrays, offsets, sizes, strict=True):
        if n > 0:
            wp.copy(flat, arr, dest_offset=offset, count=n)
    return flat, offsets


def _tiled_span(
    arrays: Sequence[wp.array[DType]], sizes: Sequence[int], total: int
) -> wp.array[DType] | None:
    """
    Return the span these segments already occupy, when they are consecutive views of one buffer.

    ``split`` and [`pack_1d_arrays`][triwarp.array.pack_1d_arrays] are documented inverses, and the
    round trip is common: [`boundary_loops`][triwarp.boundary.boundary_loops] slices one packed
    buffer into per-loop views and every batched consumer of those loops packs them straight back.
    Copying there rebuilds a buffer that already exists, one ``wp.copy`` per segment -- measured at
    **2.505 ms of ``loop_perimeters``' 2.633 ms on ``dragon``**, 407 launches to move 17 kB, against
    0.023 ms for the launch the function exists for.

    It runs only for a caller that asked (``copy=False``), because a packer that *sometimes* aliases
    is a trap and this one was caught by the suite on its first run: ``combine.concatenate`` adds
    each piece's vertex offset into the packed face buffer **in place**, so a view handed to it
    rewrites the caller's own faces. Which callers write is not inferable from here, so the choice
    stays theirs.

    The gate is *identity* on the base rather than adjacency of the pointers. Two separately
    allocated buffers can land adjacent in Warp's memory pool by luck, and an adjacency test would
    then alias or copy depending on the allocator. Requiring a common base restricts the fast path
    to callers already holding aliases of one allocation -- exactly the ``split`` round trip.

    ``_ref`` is Warp's own back-reference from a slice to the array it keeps alive (Warp 1.16); it
    is read through ``getattr`` and every conclusion drawn from it is re-checked against the public
    ``ptr`` / ``shape`` / ``strides`` / ``dtype`` / ``device``, so a release that drops the
    attribute loses the fast path rather than the correctness. ``None`` when the segments are not
    one buffer's, which is the ordinary case.
    """
    if any(n <= 0 for n in sizes):
        return None  # a zero-length segment has no address to chain through
    base = _view_base(arrays[0])
    if int(base.ndim) != 1 or not base.is_contiguous or base.dtype != arrays[0].dtype:
        return None
    stride = int(base.strides[0])
    cursor = int(arrays[0].ptr)
    for arr, n in zip(arrays, sizes, strict=True):
        if (
            _view_base(arr) is not base
            or not arr.is_contiguous
            or int(arr.strides[0]) != stride
            or int(arr.ptr) != cursor
        ):
            return None
        cursor += n * stride
    start, remainder = divmod(int(arrays[0].ptr) - int(base.ptr), stride)
    if remainder or start < 0 or start + total > int(base.shape[0]):
        return None
    return base[start : start + total]


def _view_base(arr: wp.array[DType]) -> wp.array[DType]:
    """Resolve a slice view to the allocation it reads, or return an owning array unchanged."""
    while (parent := getattr(arr, "_ref", None)) is not None:
        arr = parent
    return arr


def allclose(
    a: wp.array[wp.Float] | wp.array[wp.vec3],
    b: wp.array[wp.Float] | wp.array[wp.vec3],
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """
    Test whether two arrays are element-wise equal within a tolerance (``numpy.allclose``).

    Reduces ``|a - b| <= atol + rtol * |b|`` (element-wise, and component-wise for ``wp.vec3``)
    to a single Python ``bool`` on-device, without copying either array to the host. Matches the
    asymmetric ``numpy.allclose`` / ``torch.allclose`` tolerance convention.

    Parameters
    ----------
    a
        Length-``n`` array of any float dtype (``float16`` / ``float32`` / ``float64``) or of
        ``wp.vec3``, on the target device.
    b
        Array of the same length and dtype as ``a``.
    rtol
        Relative tolerance. Defaults to ``1e-05``. Converted to ``a``'s precision.
    atol
        Absolute tolerance. Defaults to ``1e-08``. Converted to ``a``'s precision, so a
        ``float16`` comparison cannot resolve a tolerance below its own epsilon.

    Returns
    -------
    bool
        ``True`` when every element (every component, for ``wp.vec3``) is within tolerance.
        ``True`` for empty inputs, following the ``numpy.allclose`` convention.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or dtypes.
    """
    if a.dtype != b.dtype:
        raise ValueError(f"allclose requires matching dtypes, got {a.dtype} and {b.dtype}")
    n = int(a.shape[0])
    if n != int(b.shape[0]):
        raise ValueError(f"allclose requires equal lengths, got {n} and {b.shape[0]}")
    if n == 0:
        return True

    mask = wp.empty(n, dtype=wp.bool, device=a.device)
    # ``is_close_scalar`` is generic over ``wp.Float`` and instantiates at the input's precision, so
    # the tolerances have to arrive at that precision too. Vectors get a concrete overload: Warp has
    # no generic vector annotation, and no ``wp.all`` over components to fold one with.
    if a.dtype == wp.vec3:
        wp.map(kernel_array.is_close_vec3, a, b, wp.float32(rtol), wp.float32(atol), out=mask)
    else:
        scalar = a.dtype
        wp.map(kernel_array.is_close_scalar, a, b, scalar(rtol), scalar(atol), out=mask)
    return bool(tw.reduce.all(mask))


def sort_and_argsort(
    keys: wp.array[wp.Scalar], *, fill_value: int = -1
) -> tuple[wp.array[wp.Scalar], wp.array[wp.int32]]:
    """
    Ascending sort of ``keys`` together with the permutation that produced it.

    Both halves of ``numpy.sort`` and ``numpy.argsort`` at once: a radix sort produces the ordered
    keys as a side effect of computing the order, so returning only one of the two would throw work
    away. ``sorted_keys[i] == keys[order[i]]``.

    Wraps ``warp.utils.radix_sort_pairs``, which needs double-width scratch for both the keys and
    the payload; this allocates that scratch, seeds the payload with ``0..n-1`` and hands back
    length-``n`` views of the sorted prefixes.

    Parameters
    ----------
    keys
        Length-``n`` sort keys (any radix-sortable scalar dtype; see
        [`sortable_dtype`][triwarp.typing.sortable_dtype] for which those are).
    fill_value
        Padding written into the upper half of the payload buffer, where the sort's scratch lives.
        Only matters to callers that read past ``n``.

    Returns
    -------
    sorted_keys : wp.array
        Length-``n`` view of the ascending keys.
    order : wp.array[wp.int32]
        Length-``n`` view of the original index of each sorted key.

    Notes
    -----
    Both results are **views** into the scratch buffers, kept alive by the returned arrays. Clone
    them if they must outlive the caller's frame alongside another sort.

    The sort is **stable**: equal keys keep their input order, so ``order`` is ascending within
    each run of duplicate keys. ``warp.utils.radix_sort_pairs`` documents this ("the sort is
    stable and operates in linear time"), it is inherent to its LSD radix passes, and it is
    verified on both devices against ``numpy.argsort(kind="stable")``. Callers may rely on it --
    [`split_batched`][triwarp.combine.split_batched] does, to keep faces ascending within each
    component.

    See Also
    --------
    [`sort_rows`][triwarp.array.sort_rows]
    [`sort_pair_indices`][triwarp.array.sort_pair_indices]
    [`sortable_dtype`][triwarp.typing.sortable_dtype]
    """
    device = keys.device
    n = int(keys.shape[0])
    if n == 0:
        return keys, wp.empty(0, dtype=wp.int32, device=device)
    keys_buffer = wp.empty(2 * n, dtype=keys.dtype, device=device)
    wp.copy(keys_buffer, keys, count=n)
    order_buffer = sort_pair_indices(n, fill_value, device)
    wp.utils.radix_sort_pairs(keys_buffer, order_buffer, count=n)
    return keys_buffer[:n], order_buffer[:n]


def sort_rows(data: twt.Array2dInt32 | twt.Array2dFloat32) -> None:
    """
    Sort each row of a 2D array independently, in place, ascending.

    Each row is treated as its own radix-sort segment, so rows are reordered internally
    but their relative row order is unaffected.

    Parameters
    ----------
    data
        ``(n, w)`` device array sorted in place, row by row.

    Notes
    -----
    Rows no wider than ``SORT_ROWS_INSERTION_MAX_COLS`` are sorted by a per-row insertion sort (one
    thread per row); wider rows fall back to a segmented radix sort. The narrow path is not a
    micro-optimization: ``segmented_sort_pairs`` pays a fixed cost per *segment*, so sorting a
    million two-element rows with it cost ~183 ms against ~0.1 ms for the compare-and-swap the width
    actually needs. Every in-library caller sorts vertex pairs or triangle corners.
    """
    n = data.size
    n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
    if n_rows == 0 or n_cols < 2:
        return
    if n_cols <= SORT_ROWS_INSERTION_MAX_COLS:
        wp.launch(kernel_array.sort_rows_insertion, dim=n_rows, inputs=[data], device=data.device)
        return

    data_buffer = wp.empty(n * 2, dtype=data.dtype, device=data.device)
    wp.copy(data_buffer, data, count=n)
    indices_buffer = sort_pair_indices(n, -1, data.device)
    segment_start_indices = arange_step(n // n_cols + 1, n_cols, data.device)
    wp.utils.segmented_sort_pairs(
        data_buffer, indices_buffer, n, segment_start_indices=segment_start_indices
    )
    wp.copy(data, data_buffer, count=n)


def triplet_buffers(
    n_triplets: int, dtype: type, device: wp.DeviceLike
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array]:
    """
    Uninitialized ``(rows, cols, values)`` COO buffers for one ``bsr_from_triplets`` build.

    Parameters
    ----------
    n_triplets
        Length of each of the three buffers: the number of ``(row, col, value)`` entries the
        writing kernel will emit, counting duplicates, since ``warp.sparse.bsr_from_triplets``
        sums entries that land on the same position.
    dtype
        Element type of the value buffer. A scalar (``wp.float32`` / ``wp.float64``) for a
        1x1-block matrix, or a matrix type (``wp.mat22d``) for a block matrix.
    device
        Warp device for all three buffers.

    Returns
    -------
    rows, cols, values
        Three length-``n_triplets`` arrays on ``device``. The index buffers are ``wp.int32``;
        ``values`` takes ``dtype``. All three are **uninitialized** -- the caller's kernel is
        expected to write every entry.

    Notes
    -----
    ``wp.empty`` rather than ``wp.zeros`` deliberately: a triplet writer fills all three buffers,
    so zeroing them first would be three wasted launches. A kernel that emits *fewer* than
    ``n_triplets`` entries must therefore write an explicit structural zero (typically a
    self-entry) rather than leave a slot untouched, which is also why every operator build in
    this package passes ``prune_numerical_zeros=False`` -- see the note on
    [`index_sparse`][triwarp.array.index_sparse], the one caller that prunes.

    See Also
    --------
    [`index_sparse`][triwarp.array.index_sparse]
    """
    rows = wp.empty(n_triplets, dtype=wp.int32, device=device)
    cols = wp.empty(n_triplets, dtype=wp.int32, device=device)
    values = wp.empty(n_triplets, dtype=dtype, device=device)
    return rows, cols, values


def index_sparse(
    n_rows: int,
    indices: twt.Array2dInt32,
    data: wp.array[wp.Scalar] | None = None,
    dtype: type[wp.Scalar] | None = None,
    *,
    prune_numerical_zeros: bool = True,
) -> wps.BsrMatrix[wp.Scalar]:
    """
    Build a sparse row/column incidence matrix from flat index columns.

    This mirrors ``trimesh.geometry.index_sparse``, but returns a ``warp.sparse.BsrMatrix``
    in 1x1 BSR (CSR) form instead of ``scipy.sparse.coo_matrix``.

    Parameters
    ----------
    n_rows
        Number of matrix rows (e.g. vertex count). Matrix shape is ``(n_rows, len(indices))``.
    indices
        Integer array of shape ``(m, d)`` — typically ``mesh.faces`` with three vertex indices
        per face.
    data
        Optional 1-D array of length ``m * d``. If omitted, ``wp.ones`` is used; see ``dtype``.
    dtype
        Scalar type for ``wp.ones`` when ``data`` is ``None`` (defaults to ``wp.float32`` if
        ``dtype`` is ``None``). When ``data`` and ``dtype`` are provided, the values of the
        matrix are cast to ``dtype``.
    prune_numerical_zeros
        Forwarded to ``warp.sparse.bsr_from_triplets``.

    Returns
    -------
    warp.sparse.BsrMatrix
        Sparse matrix with shape ``(n_rows, len(indices))`` and 1x1 blocks.

    Raises
    ------
    ValueError
        If ``data`` is given and its size differs from ``indices.size``.
    """
    prune_numerical_zeros = prune_numerical_zeros and data is not None
    if data is None:
        data = wp.ones(
            indices.size, dtype=dtype if dtype is not None else wp.float32, device=indices.device
        )
    else:
        if data.size != indices.size:
            raise ValueError(
                f"data must have the same size as indices, got {data.size} and {indices.size}"
            )
        if dtype is not None and data.dtype != dtype:
            data = astype(data, dtype)

    n_cols, n_repeats = indices.shape
    cols = repeat_range(n_cols * n_repeats, n_repeats, indices.device)
    return wps.bsr_from_triplets(
        n_rows,
        indices.shape[0],
        indices.flatten(),
        cols,
        data,
        prune_numerical_zeros=prune_numerical_zeros,
    )


def isin(elements: twt.ArrayNd, test_elements: wp.array[wp.Int]) -> wp.array[wp.bool]:
    """
    Test whether each element appears in ``test_elements`` (``numpy.isin`` for integers).

    Works for every Warp integer dtype -- ``int8`` through ``int64``, ``uint8`` through ``uint64``
    -- and for negative values. Both arrays must share one dtype.

    Two strategies, chosen by the value **span** ``max - min + 1`` taken over both inputs together.
    When the span is modest relative to ``len(test_elements)``, membership is a boolean lookup table
    indexed by ``value - min`` (fast for dense mesh indices). Otherwise ``test_elements`` is sorted
    and each query is a binary search, which keeps memory bounded when the values are sparse in
    their dtype.

    Parameters
    ----------
    elements
        Integer array of **any rank** on the target device. Membership is a per-element predicate,
        so the array is flattened, tested, and the result reshaped back; nothing in the two
        strategies looks at the shape.
    test_elements
        1D array of values to test membership against, of the same dtype as ``elements``.

    Returns
    -------
    wp.array[wp.bool]
        Boolean array with the same shape as ``elements``. All ``False`` when either
        input is empty.

    Raises
    ------
    TypeError
        If either array is not an integer dtype, or the two dtypes differ.

    See Also
    --------
    [`indices_to_mask`][triwarp.array.indices_to_mask]
    [`sortable_dtype`][triwarp.typing.sortable_dtype]
    [`numpy.isin`][]

    Notes
    -----
    Dtypes narrower than four bytes are widened to ``int32`` / ``uint32`` (the
    [`sortable_dtype`][triwarp.typing.sortable_dtype] rule) before either strategy runs: Warp's
    radix sort does not accept them, and neither does the tiled min/max reduction the span needs.

    Two host readbacks, one min/max reduction per input, which is what selects the strategy and
    anchors the table.
    """
    device = elements.device
    dtype = elements.dtype
    if dtype != test_elements.dtype:
        raise TypeError(
            f"isin requires one dtype for both arrays, got {dtype} and {test_elements.dtype}"
        )
    if not wp.types.type_is_int(dtype) or dtype == wp.bool:
        raise TypeError(f"isin requires an integer dtype, got {dtype}")

    k = int(test_elements.shape[0])
    if k == 0 or int(elements.size) == 0:
        return wp.zeros(elements.shape, dtype=wp.bool, device=device)

    is_flat = int(elements.ndim) == 1
    elements_flat = elements if is_flat else elements.flatten()
    # Widen sub-32-bit dtypes once, up front: neither ``reduce.minmax`` nor the radix sort accepts
    # them, and a widened span cannot overflow the type it is measured in (int8's span reaches 256).
    if wp.types.type_size_in_bytes(dtype) < 4:
        wide = twt.sortable_dtype(dtype)
        elements_flat = astype(elements_flat, wide)
        test_elements = astype(test_elements, wide)

    lo_elements, hi_elements = tw.reduce.minmax(elements_flat)
    lo_test, hi_test = tw.reduce.minmax(test_elements)
    offset = min(int(lo_elements), int(lo_test))
    span = max(int(hi_elements), int(hi_test)) - offset + 1
    if span <= _ISIN_MASK_SIZE_FACTOR * k:
        out_flat = _isin_lookup_mask(elements_flat, test_elements, span, offset)
    else:
        out_flat = _isin_lookup_sorted(elements_flat, test_elements)

    return out_flat if is_flat else out_flat.reshape(elements.shape)


def _isin_lookup_mask(
    elements_flat: wp.array[wp.Scalar], test_elements: wp.array[wp.Scalar], span: int, offset: int
) -> wp.array[wp.bool]:
    # The table is anchored at ``offset`` (the global minimum over both inputs) rather than at zero,
    # so it holds negative values and stays span-sized instead of max-sized. Anchoring at zero
    # instead is a *silent wrong answer* for negative input: ``mark_membership_mask`` drops the
    # negative test values as out of range, so they read back as absent.
    device = elements_flat.device
    dtype = elements_flat.dtype
    anchor = dtype(offset)
    test_slots = wp.empty(int(test_elements.shape[0]), dtype=wp.int32, device=device)
    wp.map(kernel_array.shifted_index, test_elements, anchor, out=test_slots)
    element_slots = wp.empty(int(elements_flat.shape[0]), dtype=wp.int32, device=device)
    wp.map(kernel_array.shifted_index, elements_flat, anchor, out=element_slots)

    membership_wp = wp.zeros(span, dtype=wp.bool, device=device)
    wp.launch(
        kernel_scatter.mark_membership_mask,
        dim=int(test_slots.shape[0]),
        inputs=[test_slots, wp.int32(span), membership_wp],
        device=device,
    )
    # ``membership_wp[element_slots]`` gathers the boolean membership flag per element.
    return gather(membership_wp, element_slots)


def _isin_lookup_sorted(
    elements_flat: wp.array[wp.Scalar], test_elements: wp.array[wp.Scalar]
) -> wp.array[wp.bool]:
    device = elements_flat.device
    sorted_test_wp = _sorted_copy(test_elements)
    out_wp = wp.empty(elements_flat.shape, dtype=wp.bool, device=device)
    wp.launch(
        kernel_array.isin_lookup_sorted,
        dim=int(elements_flat.shape[0]),
        inputs=[elements_flat, sorted_test_wp, out_wp],
        device=device,
    )
    return out_wp


def _sorted_copy(values: wp.array[DType]) -> wp.array[DType]:
    """
    Ascending-sorted copy of a 1D scalar array.

    ``sort_and_argsort`` returns a *view* into its own scratch and this outlives the caller's
    frame, so the keys are cloned. The order payload is discarded, which is why the padding value
    it seeds does not matter here.
    """
    if int(values.shape[0]) <= 1:
        return values
    return wp.clone(sort_and_argsort(values)[0])


def flatnonzero(values: wp.array[wp.bool] | wp.array[wp.Scalar]) -> wp.array[wp.int32]:
    """
    Return the indices of the non-zero entries of a 1D array (``numpy.flatnonzero``).

    Takes a boolean mask, which is the common case, or any scalar array — every non-zero value
    selects its index, exactly as ``numpy.flatnonzero`` does, so ``-2`` and ``3`` both count and
    only ``0`` does not.

    Parameters
    ----------
    values
        Length-``n`` ``wp.bool`` mask, or a ``wp.int32`` / float / other scalar array, on the
        target device.

    Returns
    -------
    wp.array[wp.int32]
        Selected indices on ``values.device``, ascending. Empty when nothing is non-zero.

    Raises
    ------
    ValueError
        If ``values`` is not rank-1.

    See Also
    --------
    [`indices_to_mask`][triwarp.array.indices_to_mask]
    [`mask_to_index_map`][triwarp.array.mask_to_index_map]
    [`numpy.flatnonzero`][]
    """
    if int(values.ndim) != 1:
        raise ValueError(f"flatnonzero requires a 1D array, got ndim={values.ndim}")

    device = values.device
    n = int(values.shape[0])
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    flags = wp.empty(n, dtype=wp.int32, device=device)
    if values.dtype == wp.bool:
        # ``array_cast`` already yields exactly 0/1 from a mask, and ``wp.Scalar`` does not
        # instantiate for ``wp.bool`` anyway. For every other dtype the cast would copy the
        # *values*, and the scan below would then sum them instead of counting them.
        wp.utils.array_cast(values, flags)
    else:
        wp.map(kernel_array.nonzero_flag, values, out=flags)

    # Inclusive scan: the total is its last element, so one 4-byte tail read sizes the output
    # (the scatter kernel derives each exclusive position as inclusive[i] - 1).
    inclusive = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    n_out = int(read_scalar(inclusive))

    if n_out == 0:
        return wp.empty(0, dtype=wp.int32, device=device)

    out_indices = wp.empty(n_out, dtype=wp.int32, device=device)
    wp.launch(
        kernel_scatter.scatter_index_where,
        dim=n,
        inputs=[flags, inclusive, out_indices],
        device=device,
    )
    return out_indices


def gather(
    src: wp.array[DType] | twt.ArrayNd, indices: wp.array[wp.int32]
) -> wp.array[DType] | twt.ArrayNd:
    """
    Dense copy of ``src`` gathered along its first axis by ``indices`` (``numpy.take``).

    Warp's ``src[indices]`` fancy indexing yields a ``warp.indexedarray`` view; this
    materializes a contiguous ``warp.array`` (performing the copy) so callers get a real
    array supporting ``.reshape`` and a stable return type. Works for rank-1 sources
    (``src[indices]``) and rank-2 row gather (``src[indices, :]``), with any scalar or vector
    ``dtype``.

    Only the first axis is indexable, which is what every caller in this package needs. To take a
    *column*, materialize it with ``wp.clone(src[:, k])`` — a column view is strided, and Warp's
    fancy indexing silently ignores the stride of an index array.

    Parameters
    ----------
    src
        Rank-1 or rank-2 ``wp.array`` on the target device.
    indices
        1D ``wp.int32`` array of indices into the first axis of ``src``. **Must be contiguous**;
        see the warning above.

    Returns
    -------
    wp.array
        Contiguous gathered copy on ``src.device`` with shape
        ``(len(indices), *src.shape[1:])`` and the same ``dtype`` as ``src``. Empty along the
        first axis when ``indices`` is empty.

    See Also
    --------
    [`index_sparse`][triwarp.array.index_sparse]
    [`remap_indices`][triwarp.array.remap_indices]
        The sentinel-preserving variant for index buffers that may carry ``-1`` entries.
    """
    k = int(indices.shape[0])
    out_shape = (k, *(int(dim) for dim in src.shape[1:]))
    out = wp.empty(out_shape, dtype=src.dtype, device=src.device)
    if k > 0:
        wp.copy(out, src[indices])
    return out


def astype(values: twt.ArrayNd, dtype: type) -> wp.array:
    """
    Element-wise dtype conversion, shape and rank preserved (``numpy.ndarray.astype``).

    The Python-scope counterpart of ``wp.cast``, which exists only inside a kernel. Allocates a
    buffer of ``values``' shape on ``values``' device and fills it with
    ``warp.utils.array_cast`` -- the pair this replaces at twenty-odd call sites.

    Parameters
    ----------
    values
        Rank-1 or rank-2 Warp array of any scalar dtype ``array_cast`` accepts.
    dtype
        Target scalar dtype.

    Returns
    -------
    wp.array
        A new array of ``values``' shape on ``values``' device, with element type ``dtype``.

    Raises
    ------
    ValueError
        If ``values`` is rank-2 and not contiguous, since the rank-2 path flattens.

    Notes
    -----
    ``warp.utils.array_cast``'s kernel is ``dest[i] = dest.dtype(src[i])``, which is a *scalar*
    conversion -- handed a rank-2 array it fails to compile, because ``src[i]`` is a row. So rank-2
    input is cast through paired ``flatten()`` views here rather than at each call site, which is
    what several of them were doing by hand.

    The output keeps ``values``' shape, so this is not a reinterpretation that changes rank: an
    ``(n, 3)`` ``float32`` read as ``(n,)`` ``wp.vec3`` is a different operation and still calls
    ``wp.utils.array_cast`` directly, as [`hash_rows`][triwarp.grouping.hash_rows] and
    [`mean_vertex_normals`][triwarp.vertices.mean_vertex_normals] do.

    See Also
    --------
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
        Reinterpret the *bits* rather than convert the value.
    """
    out = wp.empty(values.shape, dtype=dtype, device=values.device)
    if int(values.ndim) == 1:
        wp.utils.array_cast(values, out)
    else:
        if not values.is_contiguous:
            raise ValueError("astype requires a contiguous array for rank-2 input")
        wp.utils.array_cast(values.flatten(), out.flatten())
    return out


def index_domain_size(indices: twt.IntArray) -> int:
    """
    Size of the domain an index buffer addresses, as ``max(indices) + 1``.

    For a face or edge buffer this is the vertex count, and it follows libigl's
    ``F.maxCoeff() + 1`` convention: one past the largest referenced index, so a mesh with trailing
    unreferenced vertices reports fewer than it has. Accepts any ``wp.int32`` index buffer of any
    shape -- a length-``3 * n_faces`` flat triangle buffer, an ``(n, 2)`` edge array -- and reads
    only the 4-byte maximum back to the host.

    Named for the index buffer rather than for vertices because that is all it sees: it is
    index arithmetic, and nothing about it is geometric.

    Parameters
    ----------
    indices
        A ``wp.int32`` index buffer of any shape (e.g. a flat ``faces`` array or a ``(n, 2)``
        edge array).

    Returns
    -------
    int
        ``max(indices) + 1``, or ``0`` when ``indices`` is empty.
    """
    if int(indices.size) == 0:
        return 0
    # Device-side tiled max: only the 4-byte result crosses to the host, not the whole buffer.
    return int(tw.reduce.max(indices)) + 1


def indices_to_mask(
    indices: wp.array[wp.int32], n: int, *, device: wp.DeviceLike = None
) -> wp.array[wp.bool]:
    """
    Boolean membership mask of length ``n`` marking each value in ``indices`` as ``True``.

    Wraps the ``mark_membership_mask`` scatter kernel: every ``indices[i]`` sets
    ``out_mask[indices[i]] = True``. The inverse of [`flatnonzero`][triwarp.array.flatnonzero].

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of values in ``[0, n)`` to mark. May be empty.
    n
        Length of the returned mask.
    device
        Target Warp device. Defaults to ``indices.device``.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n`` mask, all ``False`` except at positions named by ``indices``.

    See Also
    --------
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`isin`][triwarp.array.isin]
    """
    device = device if device is not None else indices.device
    mask = wp.zeros(n, dtype=wp.bool, device=device)
    k = int(indices.shape[0])
    if k > 0:
        wp.launch(
            kernel_scatter.mark_membership_mask,
            dim=k,
            inputs=[indices, wp.int32(n), mask],
            device=device,
        )
    return mask


def mask_to_index_map(
    mask: wp.array[wp.bool], *, invert: bool = False
) -> tuple[wp.array[wp.int32], int]:
    """
    Compact index map over the ``True`` entries of a boolean mask, plus their count.

    Not [`flatnonzero`][triwarp.array.flatnonzero], though the two come from the same scan:
    ``flatnonzero`` returns *which* elements are selected (length ``count``, values are positions
    into ``mask``), while this returns *where each element lands* in the compacted numbering
    (length ``n``, values are compact ranks). Renumbering consumers — the free/fixed
    degree-of-freedom partitions in [`triwarp.smoothing`][triwarp.smoothing] and
    [`triwarp.linalg`][triwarp.linalg] — need this full-length scatter-side form, which
    ``flatnonzero`` cannot provide without an extra pass.

    Parameters
    ----------
    mask
        Length-``n`` ``wp.bool`` array.
    invert
        When ``True``, map the ``False`` entries instead. Useful for a free/fixed degree-of-freedom
        partition, where the mask marks the *constrained* entries and the compact map is wanted over
        the unconstrained complement (see [`free_partition`][triwarp.linalg.free_partition]).

    Returns
    -------
    index_map : wp.array[wp.int32]
        Length-``n`` array on ``mask.device``: an exclusive scan of the (optionally inverted) mask,
        so ``index_map[i]`` is the compact 0-based rank of element ``i`` among the selected
        entries at or before it (meaningful only where element ``i`` is itself selected).
    count : int
        Total number of selected entries in ``mask``.
    """
    device = mask.device
    n = int(mask.shape[0])
    if n == 0:
        return wp.zeros(0, dtype=wp.int32, device=device), 0
    flags = wp.empty(n, dtype=wp.int32, device=device)
    if invert:
        wp.map(kernel_array.complement_flag, mask, out=flags)
    else:
        wp.utils.array_cast(mask, flags)
    return counts_to_offsets(flags)


def counts_to_offsets(
    counts: wp.array[wp.int32], *, include_total: bool = False
) -> tuple[wp.array[wp.int32], int]:
    """
    Exclusive prefix sum of ``counts``, plus their total.

    The CSR-building step that turns per-element counts into row starts. Done in **one** scan pass
    and one 4-byte host read: the scan runs *inclusive* into the tail of an ``n + 1`` buffer whose
    leading zero is already in place, which makes the first ``n`` entries the exclusive sum and the
    last entry the total. The obvious spelling — one exclusive scan for the offsets and a second
    inclusive scan (or a ``reduce.sum``) for the total — costs a second full pass over ``counts``,
    and reading the total as ``inclusive.numpy()[-1]`` copies the whole array to the host to look at
    one element of it.

    Parameters
    ----------
    counts
        Length-``n`` ``wp.int32`` per-element counts.
    include_total
        Return the length-``n + 1`` CSR form, whose trailing element is ``total``, instead of the
        length-``n`` form. Free — that buffer is what gets built either way — and it is what
        ``warp.utils.segmented_sort_pairs`` and the other segment-bounds consumers want.

    Returns
    -------
    offsets : wp.array[wp.int32]
        Exclusive prefix sum: length ``n`` by default, or ``n + 1`` with ``offsets[n] == total``
        when ``include_total`` is set. The default is a **view** into the ``n + 1`` buffer, which
        the returned array keeps alive. Element ``i`` owns ``[offsets[i], offsets[i] + counts[i])``.
    total : int
        Sum of ``counts``.

    Notes
    -----
    Two offsets conventions coexist in this package: the length-``n`` form, with the total
    implicit, and the length-``n + 1`` form that stores it (``halfedge.vertex_one_rings``,
    ``geodesic_walk.trace_from_vertex``, and every ``segmented_sort_pairs`` caller). Both come
    out of here, so no caller has to append the terminator afterwards.

    **This is for callers that want ``total``**, which it reads back unconditionally -- about
    0.1 ms of host synchronization. A caller that only needs the offsets and already knows its
    buffer size should keep the open-coded ``wp.zeros(n + 1)`` plus a scan into ``[1:]``, as
    ``halfedge.vertex_one_rings`` and ``adjacency.vertex_face_adjacency`` do: both size their
    payload from ``3 * n_faces`` and would gain a synchronization they currently do not have.

    See Also
    --------
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`mask_to_index_map`][triwarp.array.mask_to_index_map]
    """
    n = int(counts.shape[0])
    device = counts.device
    if n == 0:
        return wp.zeros(1 if include_total else 0, dtype=wp.int32, device=device), 0
    # The leading zero from ``wp.zeros`` is the first exclusive offset; the inclusive scan fills the
    # rest, so ``buffer[n]`` is the total and ``buffer[:n]`` the exclusive offsets.
    buffer = wp.zeros(n + 1, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=buffer[1:], inclusive=True)
    return buffer if include_total else buffer[:n], int(read_scalar(buffer))


def remap_indices(indices: wp.array[wp.int32], remap: wp.array[wp.int32]) -> wp.array[wp.int32]:
    """
    Remap an index buffer through a lookup table, passing negative (sentinel) entries through.

    Not a plain [`gather`][triwarp.array.gather]: the ``-1`` slots that mark padded or removed
    entries in an index buffer must survive the remap unchanged, where a gather would read out of
    bounds on them. [`triwarp.repair`][triwarp.repair] relies on this — its functions preserve
    ``-1`` face sentinels through vertex renumbering (see
    ``remove_unreferenced_vertices``, pinned by ``tests/test_repair.py::
    test_remove_unreferenced_sentinel``).

    Parameters
    ----------
    indices
        1D ``wp.int32`` array of indices into ``remap`` (e.g. a flat face buffer). Negative
        entries are passed through unchanged.
    remap
        1D ``wp.int32`` lookup table (e.g. old-to-new vertex index map).

    Returns
    -------
    wp.array[wp.int32]
        Length ``len(indices)`` array on ``indices.device`` with ``out[i] = remap[indices[i]]``
        for non-negative ``indices[i]``, and ``out[i] = indices[i]`` otherwise.

    See Also
    --------
    [`gather`][triwarp.array.gather]
        The sentinel-free form: a dense first-axis gather for index buffers known to be in range.
    """
    n = int(indices.shape[0])
    device = indices.device
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=device)
    out = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_array.gather_1d_skip_negative, dim=n, inputs=[indices, remap, out], device=device
    )
    return out


def trim_to_count(
    counter: wp.array[wp.int32], *buffers: wp.array[DType]
) -> tuple[int, list[wp.array[DType]]]:
    """
    Trim atomic-append output buffers to the number of elements actually written.

    A kernel that emits an unpredictable number of results cannot size its output ahead of
    time. The usual pattern is to over-allocate the output buffers to a safe upper bound and
    have each thread claim its slots with ``wp.atomic_add`` on a shared length-1 ``counter``,
    writing into ``buffer[slot]``. After the launch, only the first ``n_out`` slots hold valid
    data and the tail is uninitialized, but ``n_out`` is known only on the device.

    This finalizes that pattern: it reads the counter back to the host once, then copies the
    valid prefix ``buffer[:n_out]`` of each over-allocated buffer into a freshly allocated,
    exact-size array. Pass every buffer filled by the same counter in one call so they are all
    trimmed to a consistent length.

    Parameters
    ----------
    counter
        Length-1 ``wp.int32`` array holding the final atomic-append count on the target device.
    *buffers
        Over-allocated output buffers to trim, all indexed along their first axis by the same
        counter. Any rank and ``dtype``; trailing dimensions are preserved.

    Returns
    -------
    n_out : int
        The counter value: the number of valid leading elements in each buffer.
    trimmed : list[wp.array]
        One contiguous ``(n_out, *buffer.shape[1:])`` copy per input buffer, in order, each on
        its buffer's device.
    """
    n_out = int(read_scalar(counter, 0))
    trimmed = []
    for buffer in buffers:
        out_shape = (n_out, *(int(dim) for dim in buffer.shape[1:]))
        out = wp.empty(out_shape, dtype=buffer.dtype, device=buffer.device)
        if n_out > 0:
            wp.copy(out, buffer[:n_out])
        trimmed.append(out)
    return n_out, trimmed


def bitcast_to_int(
    data: wp.array[wp.Scalar], count: int | None = None
) -> wp.array[wp.int32] | wp.array[wp.int64]:
    """
    Reinterpret an array's underlying bits as a same-width signed integer dtype.

    32-bit-or-narrower dtypes (``wp.int32``, ``wp.uint32``, ``wp.float32``, and narrower) are
    reinterpreted as ``wp.int32``; wider dtypes (``wp.int64``, ``wp.uint64``, ``wp.float64``) as
    ``wp.int64``. Narrower-than-32-bit floating point values are first upcast to ``wp.float32``
    (a numeric cast, not a bit reinterpretation) so every dtype narrower than 32 bits shares one
    ``wp.int32`` key space. Used by the hashing/uniqueness machinery
    ([`unique_1d`][triwarp.grouping.unique_1d]) to give arbitrary scalar dtypes a common sortable,
    hashable integer key.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` of any scalar dtype.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized (over-allocation for in-place radix-sort scratch).

    Returns
    -------
    wp.array[wp.int32] | wp.array[wp.int64]
        Bit-reinterpreted (or, for sub-32-bit floats, upcast-then-reinterpreted) copy of length
        ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_from_int`][triwarp.array.bitcast_from_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits > 32:
        reinterpreted = wp.empty(count, dtype=wp.int64, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    reinterpreted = wp.empty(count, dtype=wp.int32, device=data.device)
    if wp.types.type_is_float(data.dtype):
        src = data
        if n_bits < 32:
            src = wp.empty(copy_count, dtype=wp.float32, device=data.device)
            wp.utils.array_cast(data, src, count=copy_count)
        wp.copy(reinterpreted, src, count=copy_count)
    else:
        wp.utils.array_cast(data, reinterpreted, count=copy_count)
    return reinterpreted


def bitcast_from_int(
    data: wp.array[wp.int32] | wp.array[wp.int64], dtype: type[wp.Scalar], count: int | None = None
) -> wp.array[wp.Scalar]:
    """
    Inverse of [`bitcast_to_int`][triwarp.array.bitcast_to_int]: recover the original dtype.

    Reinterprets (same-width) or upcasts-then-reinterprets (narrower target) the bits produced
    by ``bitcast_to_int`` back into ``dtype``.

    Parameters
    ----------
    data
        Rank-1 ``wp.array[wp.int32]`` or ``wp.array[wp.int64]``, typically the output of
        [`bitcast_to_int`][triwarp.array.bitcast_to_int].
    dtype
        Target scalar dtype to reinterpret ``data`` as.
    count
        Output length. Defaults to ``data.shape[0]``. When greater than the input length, the
        tail is left uninitialized.

    Returns
    -------
    wp.array[wp.Scalar]
        Array of dtype ``dtype`` and length ``count`` on ``data.device``.

    See Also
    --------
    [`bitcast_to_int`][triwarp.array.bitcast_to_int]
    """
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n_target_bits = wp.types.type_size_in_bytes(dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits == n_target_bits:
        reinterpreted = wp.empty(count, dtype=dtype, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    if wp.types.type_is_float(dtype) and n_bits > n_target_bits:
        wide_dtype = getattr(wp, f"float{n_bits}")
        wide = wp.empty(count, dtype=wide_dtype, device=data.device)
        wp.copy(wide, data, count=copy_count)
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(wide, reinterpreted_casted, count=copy_count)
    else:
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(data, reinterpreted_casted, count=copy_count)
    return reinterpreted_casted


# ---------------------------------------------------------------------------
# Private cross-cutting helpers (used by a single external caller today; promote to public
# only with demonstrated cross-module demand).
# ---------------------------------------------------------------------------


def _ensure_int_dtype(dtype: type) -> type[wp.Int]:
    if not wp.types.type_is_int(dtype):
        raise TypeError(f"dtype must be a Warp integer type, got {dtype!r}")
    return dtype


def _check_int_fits(dtype: type[wp.Int], value: int, name: str) -> None:
    vmin = twt.dtype_min(dtype)
    vmax = twt.dtype_max(dtype)
    if value < vmin or value > vmax:
        raise ValueError(f"{name}={value} is out of range for {dtype} [{vmin}, {vmax}]")


def _int_scalar(dtype: type[wp.Int], value: int) -> wp.Int:
    _check_int_fits(dtype, value, "value")
    return dtype(value)
