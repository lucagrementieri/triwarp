"""Private device-capability guards shared across Python wrapper modules."""

from __future__ import annotations

import warp as wp


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
