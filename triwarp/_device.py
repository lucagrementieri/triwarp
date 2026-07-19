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
