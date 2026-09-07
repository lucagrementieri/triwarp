"""
Private device-capability guards and host-readback helpers shared across wrapper modules.

Two of the six members are not about devices, and the name is a historical accident rather than a
claim: ``read_scalar`` is a host-readback helper (which is at least device-adjacent -- it is the
sync) and ``require_nonempty_mesh`` is a plain validation guard, with 19 internal uses between
them. They live here because this is the module wrapper code already imports for shared internals,
not because either consults the device. Noted so a reader grepping for the guard is not surprised
to find it under this name.

``require_same_device`` is the exception: it is squarely about devices, and it is the one member of
this module every public two-or-more-array function in the package now calls. See its own
docstring for why a manual device check, generally discouraged for internal call sites, is load-
bearing at the public boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from triwarp.constants import ITEMS_PER_SLICE_CPU, ITEMS_PER_SLICE_CUDA


def prefers_tiled_reduction(device: wp.DeviceLike) -> bool:
    """
    Whether ``device`` should run the ``wp.tile``-based variant of a global reduction.

    Reductions that land in a single accumulator have two implementations in this package, and the
    choice is forced rather than stylistic. ``wp.launch_tiled`` runs exactly **one** lane per block
    on the Warp CPU device -- ``wp.tid()``'s lane index is always 0 -- so a tile built out of
    *per-lane* values, ``wp.tile(x)``, holds a single element there and any reduction over it
    silently returns one element's worth of answer.

    The distinction matters, because it is *only* the lane-constructed tile that breaks.
    ``wp.tile_load`` reads its whole tile out of an array and is lane-independent, so it totals the
    same value on both devices -- which is why every factory in ``kernels/reduce.py`` may be tiled
    unconditionally while ``kernels/measures.py`` and ``kernels/metrics.py``, which build their
    tiles from a per-thread contribution, must branch here.

    **This is a known platform limitation, not a bug awaiting a report.** ``wp.launch`` documents
    ``block_dim`` as "always 1 for cpu devices" and ``launch_tiled`` forces it, so ``wp.tile(x)``
    correctly forms a one-element tile there; Warp's tiles guide states the consequence outright.
    Upstream tracks closing the gap in two open issues -- NVIDIA/warp#1480 (*CPU/GPU parity for all
    tile code*, which names ``wp.tile(lane_value)`` followed by reductions or scans as an affected
    pattern) and NVIDIA/warp#1638 (*Add efficient CPU block execution with fibers*, the request to
    honour ``block_dim > 1`` on CPU). The branch becomes removable only once CPU blocks run more
    than one logical thread.

    The portable form instead gives each thread a strided slice and one atomic, which is correct on
    both devices but gives up the block shuffle-reduce, which costs real CUDA time once the input
    is large enough to exceed the launch overhead. Below roughly 100k elements both forms sit at the
    launch floor and the difference is negligible.

    So: tiles on CUDA, slices on CPU. Reductions with *many* accumulators do not need this -- one
    per query already fills the device, and there the portable form is the faster one on CUDA too,
    so those have a single implementation.

    Parameters
    ----------
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    bool
        ``True`` for a CUDA device, ``False`` for CPU.

    See Also
    --------
    [`items_per_slice`][triwarp._device.items_per_slice]
    """
    return wp.get_device(device).is_cuda


def items_per_slice(device: wp.DeviceLike) -> int:
    """
    Elements per thread for the lane-free strided-slice reductions on ``device``.

    The optimum splits by device by more than a tolerance in both directions, so the value is
    chosen here rather than read from a single module constant -- see the measurements next to
    [`ITEMS_PER_SLICE_CUDA`][triwarp.constants.ITEMS_PER_SLICE_CUDA].

    Parameters
    ----------
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    int
        Slice length: ``ITEMS_PER_SLICE_CUDA`` on CUDA, ``ITEMS_PER_SLICE_CPU`` on CPU.

    See Also
    --------
    [`prefers_tiled_reduction`][triwarp._device.prefers_tiled_reduction]
    """
    return ITEMS_PER_SLICE_CUDA if wp.get_device(device).is_cuda else ITEMS_PER_SLICE_CPU


def slice_count(count: int, device: wp.DeviceLike) -> int:
    """
    Thread count for a lane-free strided-slice reduction over ``count`` elements.

    The launch dimension that goes with [`items_per_slice`][triwarp._device.items_per_slice]: one
    thread per strided slice, and at least one thread so an empty input still launches a well-formed
    grid. Callers should not divide by ``items_per_slice`` themselves -- that spelling is what this
    exists to hold in one place.

    Parameters
    ----------
    count
        Number of elements to reduce.
    device
        Warp device (or device string) the reduction will run on.

    Returns
    -------
    int
        ``ceil(count / items_per_slice(device))``, floored at 1.

    See Also
    --------
    [`items_per_slice`][triwarp._device.items_per_slice]
    """
    per_slice = items_per_slice(device)
    return max(1, (count + per_slice - 1) // per_slice)


def require_nonempty_mesh(faces: wp.array[wp.int32], name: str) -> None:
    """
    Raise before constructing a ``warp.Mesh`` with zero triangles.

    A ``warp.Mesh`` built with an empty ``indices`` array does not raise, but silently
    corrupts CUDA driver/allocator state: the constructor itself "succeeds", but a later, unrelated
    CUDA allocation anywhere else in the process then fails and cascades into "illegal memory
    access" errors. Every ``wp.Mesh(...)`` call site in this package must call this first instead
    of letting the native constructor run on an empty face buffer.

    Parameters
    ----------
    faces
        Flat ``wp.int32`` triangle index buffer about to be passed to ``warp.Mesh``.
    name
        Name of the calling function, used in the error message.

    Raises
    ------
    ValueError
        If ``faces`` is empty (zero triangles).
    """
    if int(faces.shape[0]) == 0:
        raise ValueError(
            f"{name} cannot build a warp.Mesh with zero triangles: this silently corrupts CUDA "
            "state through Warp 1.17 (see the Warp issue tracker for wp.Mesh + empty BVH)."
        )


def require_same_device(**named: Any) -> None:
    """
    Raise if two or more of the given device-bearing arguments disagree on device.

    Warp does not catch this for you. Since Warp 1.14, ``wp.launch`` accepts a cross-device
    argument list without complaint -- the default ``wp.config.launch_array_access_mode`` is
    ``RELAXED``, which passes pointers straight through -- and the two ways such a call then fails
    are both invisible to a Python ``except`` clause: a CUDA launch reading CPU arrays computes the
    *right* answer and corrupts the host heap later, once those arrays are freed while the kernel
    is still running asynchronously; a CPU launch reading CUDA arrays segfaults immediately, with no
    Python exception at all. Neither is something a caller can catch or debug from the traceback it
    gets, so every public function that accepts more than one device-bearing argument calls this
    first, before any of them reaches a kernel.

    Every argument may be ``None`` -- a caller should pass every device-bearing parameter it
    received unconditionally, including an ``X | None = None`` precomputed-cache argument, rather
    than filtering beforehand. A ``list`` or ``tuple`` argument (a sequence of loops, rings, ...) is
    unpacked element-wise, each labelled ``f"{name}[{i}]"`` in the message, rather than compared as
    one opaque object.

    Parameters
    ----------
    **named
        Every device-bearing argument the caller received (arrays, meshes, BVHs, hash grids,
        volumes, or a list/tuple of any of those), keyed by its own parameter name.

    Raises
    ------
    RuntimeError
        If two of the given arguments report a different ``.device``.
    """
    seen: list[tuple[str, Any]] = []
    for name, value in named.items():
        seen.extend(_named_devices(name, value))
    if not seen:
        return
    first_name, first_device = seen[0]
    for name, device in seen[1:]:
        if device != first_device:
            raise RuntimeError(
                f"triwarp requires every argument to run on one device, but '{first_name}' is on "
                f"{first_device} while '{name}' is on {device}. Move one onto the other's device "
                "(e.g. wp.clone(array, device=...) for a wp.array) before calling this function."
            )


def _named_devices(name: str, value: Any) -> list[tuple[str, Any]]:
    """Flatten ``value`` into ``(label, device)`` pairs, descending into a list/tuple."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        found: list[tuple[str, Any]] = []
        for i, item in enumerate(value):
            found.extend(_named_devices(f"{name}[{i}]", item))
        return found
    device = getattr(value, "device", None)
    return [(name, device)] if device is not None else []


# One scratch buffer per dtype for ``read_scalar`` below, allocated on first use and reused for the
# life of the process. A single element each, so the whole table is a few dozen bytes.
_SCALAR_SCRATCH: dict[type, wp.array] = {}


def read_scalar(arr: wp.array[Any], index: int = -1) -> Any:
    """
    One element of ``arr``, read back to the host as a Python scalar.

    The spelling matters more than it looks. ``int(arr[n - 1 :].numpy()[0])`` -- a natural first
    attempt -- builds a one-element Warp view, then a *fresh* host array for it, then synchronizes.
    Copying into a scratch buffer allocated once avoids that intermediate array; on a host array,
    where ``.numpy()`` is already a zero-copy view of the whole buffer, indexing that view directly
    is cheaper than slicing first. So the device branch is not a portability concession: each
    side's fast path is the other's slow one.

    !!! warning "The scratch must not be pinned"
        A pinned host destination makes ``cudaMemcpyAsync`` genuinely asynchronous, and Warp issues
        the copy without an event or a synchronization, so the read can race the producing kernel
        and silently return the *previous* round's value instead. Pageable memory is documented to
        return only once a device-to-host copy has completed.

    Parameters
    ----------
    arr
        Warp array to read from. Any scalar dtype; one scratch buffer per dtype is cached.
    index
        Element to read, negative from the end as in Python. Defaults to the last element, which
        is what an inclusive scan's total lives in.

    Returns
    -------
    Any
        The element, as the value ``numpy`` gives for that dtype: a Python scalar for the scalar
        dtypes (``int`` for the integer ones, ``float`` for the floating ones), and a **copy** of
        the row for a vector or matrix dtype. Callers wrap it in ``int(...)`` / ``float(...)``
        where a definite type is wanted.

    Notes
    -----
    Not reentrant: the scratch is shared, so two concurrent readbacks of the same dtype from
    different threads would clobber each other. Nothing in this package reads back off-thread.

    **The copy is load-bearing for the non-scalar dtypes, and its absence is silent.** Indexing a
    ``wp.array[wp.vec3]``'s ``.numpy()`` yields a *view*, so without it two sequential reads of the
    same dtype would both alias the one cached scratch row and the first would take the second's
    value -- ``creation.sweep_polygon`` reads a path's two endpoints back to back and would decide
    every open path was closed. On the host branch the view is onto the caller's own buffer, where
    a caller writing through it would corrupt the array. Scalar dtypes are unaffected (``numpy``
    hands back a scalar, which is already a copy), which is exactly why this hides.
    """
    n = int(arr.shape[0])
    slot = index if index >= 0 else n + index
    device = arr.device
    if device is None or not device.is_cuda:
        return _detached(arr.numpy()[slot])
    scratch = _SCALAR_SCRATCH.get(arr.dtype)
    if scratch is None:
        scratch = wp.empty(1, dtype=arr.dtype, device="cpu")
        _SCALAR_SCRATCH[arr.dtype] = scratch
    wp.copy(scratch, arr[slot : slot + 1])
    return _detached(scratch.numpy()[0])


def _detached(value: Any) -> Any:
    """Copy ``value`` when it is a view, so a vector or matrix element outlives the next read."""
    return value.copy() if isinstance(value, np.ndarray) else value
