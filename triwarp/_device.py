"""Private device-capability guards shared across Python wrapper modules."""

from __future__ import annotations

import warp as wp

from triwarp.constants import ITEMS_PER_SLICE_CPU, ITEMS_PER_SLICE_CUDA


def prefers_tiled_reduction(device: wp.DeviceLike) -> bool:
    """
    Whether ``device`` should run the ``wp.tile``-based variant of a global reduction.

    Reductions that land in a single accumulator have two implementations in this package, and the
    choice is forced rather than stylistic. ``wp.launch_tiled`` runs exactly **one** lane per block
    on the Warp CPU device -- ``wp.tid()``'s lane index is always 0 -- through Warp 1.16, so a tile
    built out of *per-lane* values, ``wp.tile(x)``, holds a single element there and any reduction
    over it silently returns one element's worth of answer. Measured on Warp 1.16.0: over 8 blocks
    of 64 ones, ``wp.tile_sum(wp.tile(v))`` totals **8.0 on CPU** against 512.0 on CUDA.

    The distinction matters, because it is *only* the lane-constructed tile that breaks.
    ``wp.tile_load`` reads its whole tile out of an array and is lane-independent, so it totals
    512.0 on both devices -- which is why every factory in ``kernels/reduce.py`` may be tiled
    unconditionally while ``kernels/totals.py`` and ``kernels/distance.py``, which build their tiles
    from a per-thread contribution, must branch here.

    **This is a known platform limitation, not a bug awaiting a report.** ``wp.launch`` documents
    ``block_dim`` as "always 1 for cpu devices" and ``launch_tiled`` forces it, so ``wp.tile(x)``
    correctly forms a one-element tile there; Warp's tiles guide states the consequence outright.
    Upstream tracks closing the gap in two open issues -- NVIDIA/warp#1480 (*CPU/GPU parity for all
    tile code*, which names ``wp.tile(lane_value)`` followed by reductions or scans as an affected
    pattern) and NVIDIA/warp#1638 (*Add efficient CPU block execution with fibers*, the request to
    honour ``block_dim > 1`` on CPU). So do not re-probe this from scratch on the next upgrade and
    do not file it: read those two issues, and expect the branch to become removable only once CPU
    blocks run more than one logical thread.

    The portable form instead gives each thread a strided slice and one atomic, which is correct on
    both devices but gives up the block shuffle-reduce, and that costs real CUDA time once the input
    is large enough to exceed the launch overhead: measured 1.67x on the area-weighted centroid at
    327k faces (16.0 -> 26.7 us of kernel time) and 1.57x on the chamfer loss term at 500k points
    (19.0 -> 29.7 us). Below roughly 100k elements both forms sit at the ~18 us launch floor and the
    difference is unmeasurable.

    So: tiles on CUDA, slices on CPU. Reductions with *many* accumulators do not need this -- one
    per query already fills the device, and there the portable form is the faster one on CUDA too
    (the solid-angle winding sum measured 1.6x faster at 5k queries and 2.0x at 50k), so those have
    a single implementation.

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
    corrupts CUDA driver/allocator state through Warp 1.16: the constructor itself "succeeds", but a
    later, unrelated CUDA allocation anywhere else in the process then fails and cascades into
    "illegal memory access" errors. Every ``wp.Mesh(...)`` call site in this package must call
    this first instead of letting the native constructor run on an empty face buffer.

    Re-verified on Warp 1.16.0 in ten throwaway subprocesses, since the failure lands on a *later*
    allocation and an in-process probe would poison the session: 10/10 aborted with
    ``Warp CUDA error 2: out of memory`` on the first 4 MiB allocation after the empty mesh, on a
    card with 32 GiB free.

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
            "state through Warp 1.16 (see the Warp issue tracker for wp.Mesh + empty BVH)."
        )
