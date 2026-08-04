"""Private device-capability guards shared across Python wrapper modules."""

from __future__ import annotations

import warp as wp

from triwarp.constants import ITEMS_PER_SLICE_CPU, ITEMS_PER_SLICE_CUDA


def prefers_tiled_reduction(device: wp.DeviceLike) -> bool:
    """
    Whether ``device`` should run the ``wp.tile``-based variant of a global reduction.

    Reductions that land in a single accumulator have two implementations in this package, and the
    choice is forced rather than stylistic. ``wp.launch_tiled`` runs exactly **one** lane per block
    on Warp 1.15's CPU device -- ``wp.tid()``'s lane index is always 0 -- so a block-wide
    ``wp.tile_sum`` silently reduces one element per tile there and returns a wrong answer. The
    portable form instead gives each thread a strided slice and one atomic, which is correct on both
    devices but gives up the block shuffle-reduce, and that costs real CUDA time once the input is
    large enough to exceed the launch overhead: measured 1.67x on the area-weighted centroid at 327k
    faces (16.0 -> 26.7 us of kernel time) and 1.57x on the chamfer loss term at 500k points
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


def require_cuda(device: wp.DeviceLike, name: str) -> None:
    """
    Raise unless ``device`` is a CUDA device.

    ``warp.optim.linear.cg`` produces ``NaN`` on the CPU device in Warp 1.14-1.15, so any
    caller whose solve goes through it must reject CPU devices up front instead of returning
    silently wrong results.

    Parameters
    ----------
    device
        Warp device (or device string) to check.
    name
        Name of the calling function, used in the error message.

    Raises
    ------
    NotImplementedError
        If ``device`` resolves to a CPU device.
    """
    if wp.get_device(device).is_cpu:
        raise NotImplementedError(
            f"{name} requires a CUDA device: warp.optim.linear.cg produces NaN on the CPU "
            "device in Warp 1.14-1.15."
        )


def require_nonempty_mesh(faces: wp.array[wp.int32], name: str) -> None:
    """
    Raise before constructing a ``warp.Mesh`` with zero triangles.

    A ``warp.Mesh`` built with an empty ``indices`` array does not raise, but silently
    corrupts CUDA driver/allocator state in Warp 1.15: the constructor itself "succeeds", but a
    later, unrelated CUDA allocation anywhere else in the process then fails and cascades into
    "illegal memory access" errors. Every ``wp.Mesh(...)`` call site in this package must call
    this first instead of letting the native constructor run on an empty face buffer.

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
            "state in Warp 1.15 (see the Warp issue tracker for wp.Mesh + empty BVH)."
        )
