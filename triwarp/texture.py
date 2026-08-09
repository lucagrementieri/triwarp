"""
Bake per-vertex attributes into UV-space textures and sample them back.

The four functions here operate on an *existing* UV map (from a disk parameterization or a
mesh file, e.g. loaded via [`load_mesh`][triwarp.io.load_mesh]): they rasterize per-vertex
attributes into a square texture and sample a texture back to per-vertex values. They are a
Warp software-rasterizer replacement for an OpenGL/moderngl pipeline, keeping all data on the
input Warp device.

Computing a UV map from mesh connectivity (parameterization) is a separate concern and is not
part of this module.

See [`rasterize_attribute`][triwarp.texture.rasterize_attribute],
[`rasterize_discrete_attribute`][triwarp.texture.rasterize_discrete_attribute],
[`remap_attribute_from_uv`][triwarp.texture.remap_attribute_from_uv], and
[`remap_discrete_attribute_from_uv`][triwarp.texture.remap_discrete_attribute_from_uv].
"""

from __future__ import annotations

from typing import Literal, cast

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import INT32_MAX
from triwarp.kernels import texture as kernel_texture

# Owner sentinel: larger than any face index, so wp.atomic_min lets the lowest-index covering
# face win each pixel. The kernels seed the same slot from ``INT32_MAX_CONSTANT``, so both sides
# read the one constant rather than two copies of the literal.
_OWNER_SENTINEL = INT32_MAX


def _check_uv_in_range(uv: wp.array[wp.vec2]) -> None:
    """Raise if any finite UV lies outside ``[0, 1]`` (non-finite UVs are ignored)."""
    n_vertices = int(uv.shape[0])
    if n_vertices == 0:
        return
    flag = wp.zeros(1, dtype=wp.int32, device=uv.device)
    wp.launch(kernel_texture.check_uv_range, dim=n_vertices, inputs=[uv, flag], device=uv.device)
    if int(flag.numpy()[0]) != 0:
        raise ValueError("UV coordinates must be in the range [0, 1]")


def rasterize_attribute(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], attribute: twt.Array2dFloat32, resolution: int
) -> twt.Array3dFloat32:
    """
    Rasterize a per-vertex attribute into a square UV-space texture.

    Each triangle is rasterized in UV space and its per-vertex attribute is barycentrically
    interpolated at every covered pixel center. Pixels not covered by any triangle stay ``0``.
    Where triangles overlap (degenerate for a valid parameterization), the lowest-index face wins.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates as ``wp.vec2`` in ``[0, 1]``. Non-finite
        rows (e.g. unreferenced vertices left ``NaN`` by a disk parameterization) are ignored;
        by construction no face indexes them.
    faces
        Flat length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    attribute
        ``(n_vertices, n_channels)`` ``float32`` per-vertex attribute to interpolate.
    resolution
        Output image size in pixels (square).

    Returns
    -------
    Array3dFloat32
        ``(resolution, resolution, n_channels)`` ``float32`` image on ``uv.device``. Row ``0``
        corresponds to ``v = 1`` (vertical flip), matching the sampling convention of
        [`remap_attribute_from_uv`][triwarp.texture.remap_attribute_from_uv].

    See Also
    --------
    [`remap_attribute_from_uv`][triwarp.texture.remap_attribute_from_uv]
    [`rasterize_discrete_attribute`][triwarp.texture.rasterize_discrete_attribute]
    """
    twt.ensure_ndim(attribute, 2, dtype=wp.float32)
    n_channels = int(attribute.shape[1])
    _check_rasterize_inputs(uv, int(attribute.shape[0]), resolution)

    device = uv.device
    image = wp.zeros((resolution, resolution, n_channels), dtype=wp.float32, device=device)
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.as_array3d_float32(image)

    owner = _rasterize_owner(uv, faces, resolution)
    wp.launch(
        kernel_texture.rasterize_scatter,
        dim=n_faces,
        inputs=[uv, faces, attribute, n_channels, owner, image],
        device=device,
    )
    return twt.as_array3d_float32(image)


def rasterize_discrete_attribute(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], attribute: wp.array[wp.int32], resolution: int
) -> twt.Array2dInt32:
    """
    Rasterize a per-vertex discrete label into a square UV-space class image.

    At each covered pixel the label with the greatest barycentric weight is emitted (weights of
    vertices sharing a label sum; ties resolve to the lowest label value). This is equivalent to
    one-hot encoding the labels, interpolating, and taking the per-pixel ``argmax``. Pixels not
    covered by any triangle are set to ``-1``.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates as ``wp.vec2`` in ``[0, 1]``. Non-finite
        rows are ignored.
    faces
        Flat length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    attribute
        ``(n_vertices,)`` ``int32`` per-vertex labels, all ``>= 0``.
    resolution
        Output image size in pixels (square).

    Returns
    -------
    Array2dInt32
        ``(resolution, resolution)`` ``int32`` class image with values in ``[-1, n_classes - 1]``
        (``-1`` marks uncovered pixels), on ``uv.device``.

    See Also
    --------
    [`remap_discrete_attribute_from_uv`][triwarp.texture.remap_discrete_attribute_from_uv]
    [`rasterize_attribute`][triwarp.texture.rasterize_attribute]
    """
    twt.ensure_ndim(attribute, 1, dtype=wp.int32)
    n_vertices = int(attribute.shape[0])
    _check_rasterize_inputs(uv, n_vertices, resolution)
    if n_vertices > 0 and tw.reduce.min(cast(twt.Array1dInt32, attribute)) < 0:
        raise ValueError("Attribute values must be greater than or equal to 0")

    device = uv.device
    labels_image = wp.full((resolution, resolution), -1, dtype=wp.int32, device=device)
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return twt.as_array2d_int32(labels_image)

    owner = _rasterize_owner(uv, faces, resolution)
    wp.launch(
        kernel_texture.rasterize_labels,
        dim=n_faces,
        inputs=[uv, faces, attribute, owner, labels_image],
        device=device,
    )
    return twt.as_array2d_int32(labels_image)


def _check_rasterize_inputs(uv: wp.array[wp.vec2], n_vertices: int, resolution: int) -> None:
    """
    Validate the arguments both rasterizers share: resolution, row count and UV range.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates.
    n_vertices
        Row count of the attribute being rasterized.
    resolution
        Output image size in pixels.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive, ``uv`` and the attribute disagree on their row count, or
        any finite UV lies outside ``[0, 1]``.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive")
    if int(uv.shape[0]) != n_vertices:
        raise ValueError(f"uv and attribute row count mismatch: {int(uv.shape[0])} vs {n_vertices}")
    _check_uv_in_range(uv)


def _rasterize_owner(
    uv: wp.array[wp.vec2], faces: wp.array[wp.int32], resolution: int
) -> twt.Array2dInt32:
    """
    Resolve which face owns each pixel: the lowest-index triangle covering it.

    Both rasterizers run this first so their scatter pass can write exactly one face's contribution
    per pixel. Overlap is degenerate for a valid parameterization; where it happens the lowest face
    index wins, which ``wp.atomic_min`` against the sentinel gives for free.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates.
    faces
        Flat length-``3 * n_faces`` ``wp.int32`` triangle index buffer, non-empty.
    resolution
        Output image size in pixels.

    Returns
    -------
    twt.Array2dInt32
        ``(resolution, resolution)`` owning face index, ``_OWNER_SENTINEL`` where uncovered.
    """
    device = uv.device
    owner = wp.full((resolution, resolution), _OWNER_SENTINEL, dtype=wp.int32, device=device)
    wp.launch(
        kernel_texture.rasterize_owner,
        dim=int(faces.shape[0]) // 3,
        inputs=[uv, faces, resolution, owner],
        device=device,
    )
    return twt.as_array2d_int32(owner)


def remap_attribute_from_uv(
    uv: wp.array[wp.vec2],
    image: twt.Array2dFloat32 | twt.Array3dFloat32,
    *,
    order: Literal[0, 1] = 1,
) -> twt.Array2dFloat32:
    """
    Sample a UV-space texture back to per-vertex values.

    Inverse of [`rasterize_attribute`][triwarp.texture.rasterize_attribute].
    Sampling uses the same pixel-center convention as
    [`rasterize_attribute`][triwarp.texture.rasterize_attribute] (``col = u * W - 0.5``,
    ``row = (1 - v) * H - 0.5``) and clamps to the nearest edge pixel, so vertices on the
    coverage boundary read the boundary value instead of blending toward ``0``. Vertices whose
    UV is non-finite are not sampled; their output row is ``NaN`` on every channel.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates as ``wp.vec2`` in ``[0, 1]``. Non-finite
        rows yield ``NaN`` output rows.
    image
        UV-space texture to sample, ``(H, W)`` or ``(H, W, C)`` ``float32``.
    order
        Interpolation order: ``1`` for bilinear (default), ``0`` for nearest-neighbor.

    Returns
    -------
    Array2dFloat32
        ``(n_vertices, C)`` ``float32`` per-vertex values on ``uv.device`` (``C == 1`` for a
        2D input image). Non-finite-UV rows are ``NaN``.

    See Also
    --------
    [`rasterize_attribute`][triwarp.texture.rasterize_attribute]
    """
    if int(image.ndim) == 2:
        height, width = int(image.shape[0]), int(image.shape[1])
        n_channels = 1
        image3d = image.reshape((height, width, 1))
    elif int(image.ndim) == 3:
        n_channels = int(image.shape[2])
        image3d = image
    else:
        raise TypeError(f"image must be (H, W) or (H, W, C), got ndim={image.ndim}")
    twt.ensure_ndim(image3d, 3, dtype=wp.float32)
    _check_uv_in_range(uv)

    device = uv.device
    n_vertices = int(uv.shape[0])
    out_values = twt.empty_float32_2d((n_vertices, n_channels), device=device)
    if n_vertices > 0:
        mode = kernel_texture.SAMPLE_BILINEAR if order == 1 else kernel_texture.SAMPLE_NEAREST
        wp.launch(
            kernel_texture.sample_texture,
            dim=n_vertices,
            inputs=[uv, image3d, n_channels, mode, out_values],
            device=device,
        )
    return twt.as_array2d_float32(out_values)


def remap_discrete_attribute_from_uv(
    uv: wp.array[wp.vec2], class_image: twt.Array2dInt32
) -> twt.Array1dInt32:
    """
    Sample a UV-space class image back to per-vertex labels.

    Inverse of [`rasterize_discrete_attribute`][triwarp.texture.rasterize_discrete_attribute].
    Nearest-neighbor sampling is used so labels are never blended, including the ``-1`` uncovered
    sentinel. Vertices whose UV is non-finite cannot be sampled and map to the same ``-1``
    sentinel.

    Parameters
    ----------
    uv
        ``(n_vertices,)`` per-vertex UV coordinates as ``wp.vec2`` in ``[0, 1]``. Non-finite
        rows yield ``-1``.
    class_image
        ``(H, W)`` ``int32`` class image (values e.g. in ``[-1, n_classes - 1]``).

    Returns
    -------
    Array1dInt32
        ``(n_vertices,)`` ``int32`` per-vertex labels on ``uv.device``. Non-finite-UV rows
        are ``-1``.

    See Also
    --------
    [`rasterize_discrete_attribute`][triwarp.texture.rasterize_discrete_attribute]
    [`remap_attribute_from_uv`][triwarp.texture.remap_attribute_from_uv]
    """
    twt.ensure_ndim(class_image, 2, dtype=wp.int32)
    device = uv.device
    float_image = twt.as_array2d_float32(tw.array.astype(class_image, wp.float32))

    sampled = remap_attribute_from_uv(uv, float_image, order=0)

    n_vertices = int(uv.shape[0])
    out_labels = wp.empty(n_vertices, dtype=wp.int32, device=device)
    # ``sampled`` is ``(n_vertices, 1)`` and contiguous, so ``flatten()`` is a reshape view.
    wp.map(kernel_texture.round_labels, sampled.flatten(), out=out_labels)
    return cast(twt.Array1dInt32, out_labels)
