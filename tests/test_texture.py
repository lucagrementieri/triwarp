"""
Regression tests for ``triwarp.texture``.

The forward rasterizers are compared against a vendored moderngl (OpenGL) reference; the inverse
samplers against ``scipy.ndimage.map_coordinates``. Because OpenGL's top-left edge fill rule
differs subtly from a barycentric pixel-center coverage test, comparisons of the covered/uncovered
mask allow a thin boundary band of disagreement, while interpolated values are compared only where
both rasterizers cover the pixel (the interpolated field is continuous, so values agree there).
"""

from __future__ import annotations

import textwrap

import moderngl
import numpy as np
import numpy.typing as npt
import pytest
import warp as wp
from scipy.ndimage import map_coordinates

import triwarp as tw
from tests.conversions import points_to_warp_uv


# --------------------------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------------------------
def _rasterize_attribute_gl(
    ctx: moderngl.Context,
    uv: npt.NDArray[np.float32],
    faces: npt.NDArray[np.int32],
    attribute: npt.NDArray[np.float32],
    resolution: int,
) -> npt.NDArray[np.float32]:
    """OpenGL reference: interpolate a per-vertex attribute into a UV texture (chunked vec3s)."""
    n_vertices, n_channels = attribute.shape
    n_chunks = (n_channels + 2) // 3
    padded = np.zeros((n_vertices, n_chunks * 3), dtype=np.float32)
    padded[:, :n_channels] = attribute
    chunks_3d = padded.reshape(n_vertices, n_chunks, 3)

    vertex_shader = textwrap.dedent(
        """
        #version 330 core
        in vec2 in_uv;
        in vec3 in_attr;
        out vec3 v_attr;
        void main() {
            gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            v_attr = in_attr;
        }
        """
    )
    fragment_shader = textwrap.dedent(
        """
        #version 330 core
        in vec3 v_attr;
        out vec4 fragColor;
        void main() { fragColor = vec4(v_attr, 1.0); }
        """
    )
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    uv_buffer = ctx.buffer(uv.astype(np.float32).tobytes())
    face_buffer = ctx.buffer(faces.astype(np.int32).tobytes())
    out_frame_buffer = ctx.framebuffer(
        color_attachments=[ctx.texture((resolution, resolution), 4, dtype="f4")]
    )
    color_tex = out_frame_buffer.color_attachments[0]

    out_parts: list[npt.NDArray[np.float32]] = []
    try:
        for i in range(n_chunks):
            start = i * 3
            n_keep = min(3, n_channels - start)
            attribute_buffer = ctx.buffer(chunks_3d[:, i, :].tobytes())
            out_vertex_array = ctx.vertex_array(
                program,
                [(uv_buffer, "2f", "in_uv"), (attribute_buffer, "3f", "in_attr")],
                index_buffer=face_buffer,
                index_element_size=4,
            )
            try:
                out_frame_buffer.use()
                ctx.clear(0.0, 0.0, 0.0, 1.0)
                out_vertex_array.render(moderngl.TRIANGLES)
                raw = color_tex.read()
                plane = np.frombuffer(raw, dtype=np.float32).reshape(resolution, resolution, 4)
                out_parts.append(np.flipud(plane)[:, :, :n_keep])
            finally:
                out_vertex_array.release()
                attribute_buffer.release()
    finally:
        uv_buffer.release()
        face_buffer.release()
        color_tex.release()
        out_frame_buffer.release()
        program.release()
    return np.concatenate(out_parts, axis=-1) if len(out_parts) > 1 else out_parts[0].copy()


def _remap_attribute_scipy(
    uv: npt.NDArray[np.float32], image: npt.NDArray[np.float32], order: int
) -> npt.NDArray[np.float32]:
    """Scipy reference for the pixel-center inverse sampling."""
    image_hwc = image[:, :, None] if image.ndim == 2 else image
    height, width, n_channels = image_hwc.shape
    u = uv[:, 0].astype(np.float64)
    v = uv[:, 1].astype(np.float64)
    rows = (1.0 - v) * height - 0.5
    cols = u * width - 0.5
    out = np.empty((uv.shape[0], n_channels), dtype=np.float32)
    for c in range(n_channels):
        out[:, c] = map_coordinates(
            image_hwc[:, :, c].astype(np.float64),
            [rows, cols],
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return out


# --------------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gl_context():
    """
    Yield a standalone EGL context, or skip naming the *driver* -- never the missing package.

    ``moderngl`` is a hard ``test``-group dependency like ``igl`` and ``pymeshlab``, so it is
    imported plainly at module scope and a missing wheel is a failure, not a skip. What genuinely
    cannot be declared in ``pyproject.toml`` is a working EGL driver, and that is the only thing
    this skip is allowed to be about: the module previously opened with
    ``pytest.importorskip("moderngl")``, which would have deleted all 354 lines of coverage silently
    if the import ever broke.
    """
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
    except Exception as exc:  # noqa: BLE001 - any GL/EGL init failure should skip, not error
        pytest.skip(f"no EGL OpenGL context available: {exc!r}")
    yield ctx
    ctx.release()


def _grid_uv_mesh(
    n: int, pad: float = 0.05
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Triangulated ``n x n`` lattice covering ``[pad, 1 - pad]^2`` in UV space."""
    coords = np.linspace(pad, 1.0 - pad, n)
    uu, vv = np.meshgrid(coords, coords, indexing="xy")
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float32)
    faces: list[int] = []
    for r in range(n - 1):
        for c in range(n - 1):
            i = r * n + c
            faces += [i, i + 1, i + n, i + 1, i + n + 1, i + n]
    return uv, np.array(faces, dtype=np.int32)


def _interior_mask(n: int) -> npt.NDArray[np.bool_]:
    """Lattice vertices not on the outer ring (surrounded by covered pixels)."""
    rows, cols = np.divmod(np.arange(n * n), n)
    return (rows > 0) & (rows < n - 1) & (cols > 0) & (cols < n - 1)


# --------------------------------------------------------------------------------------------
# Forward rasterization vs OpenGL
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_channels", [1, 2, 5])
def test_rasterize_attribute_matches_opengl(gl_context, device: str, n_channels: int):
    rng = np.random.default_rng(20240704 + n_channels)
    uv_np, faces_np = _grid_uv_mesh(n=8)
    # Offset into [1, 2] so covered pixels are never ~0 (distinguishable from uncovered 0).
    attribute_np = (rng.random((uv_np.shape[0], n_channels), dtype=np.float32) + 1.0).astype(
        np.float32
    )
    resolution = 64

    image_gl = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), attribute_np, resolution
    )

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    attribute_wp = wp.array(attribute_np, dtype=wp.float32, device=device)
    image_wp = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution).numpy()

    covered_gl = image_gl[:, :, 0] > 0.5
    covered_wp = image_wp[:, :, 0] > 0.5
    # Coverage masks agree except for a thin boundary band (fill-rule differences).
    assert np.mean(covered_gl == covered_wp) > 0.98
    both = covered_gl & covered_wp
    assert both.sum() > 0
    assert np.allclose(image_wp[both], image_gl[both], rtol=1e-3, atol=1e-3)


def test_rasterize_discrete_attribute_matches_opengl(gl_context, device: str):
    rng = np.random.default_rng(99)
    n = 8
    uv_np, faces_np = _grid_uv_mesh(n=n)
    labels_np = rng.integers(0, 4, size=uv_np.shape[0]).astype(np.int32)
    resolution = 64

    n_classes = int(labels_np.max() + 1)
    one_hot = np.eye(n_classes, dtype=np.float32)[labels_np]
    one_hot_image = _rasterize_attribute_gl(
        gl_context, uv_np, faces_np.reshape(-1, 3), one_hot, resolution
    )
    covered_gl = ~np.all(np.isclose(one_hot_image, 0.0), axis=-1)
    class_gl = np.argmax(one_hot_image, axis=-1).astype(np.int32)
    class_gl[~covered_gl] = -1

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_wp = tw.texture.rasterize_discrete_attribute(
        uv_wp, faces_wp, labels_wp, resolution
    ).numpy()

    covered_wp = class_wp >= 0
    assert np.mean(covered_gl == covered_wp) > 0.98
    both = covered_gl & covered_wp
    # Labels agree on the vast majority of jointly-covered pixels (argmax ties near edges differ).
    assert np.mean(class_wp[both] == class_gl[both]) > 0.98


# --------------------------------------------------------------------------------------------
# Inverse sampling vs scipy
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("n_channels", [1, 3])
def test_remap_attribute_matches_scipy(device: str, order: int, n_channels: int):
    rng = np.random.default_rng(7 * order + n_channels)
    height, width = 40, 48
    image_np = rng.random((height, width, n_channels), dtype=np.float32)
    uv_np = rng.random((200, 2), dtype=np.float32)

    values_scipy = _remap_attribute_scipy(uv_np, image_np, order)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values_wp = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=order).numpy()

    assert values_wp.shape == (200, n_channels)
    assert np.allclose(values_wp, values_scipy, rtol=1e-4, atol=1e-4)


def test_remap_attribute_2d_image(device: str):
    rng = np.random.default_rng(3)
    image_np = rng.random((32, 32), dtype=np.float32)
    uv_np = rng.random((50, 2), dtype=np.float32)
    values_scipy = _remap_attribute_scipy(uv_np, image_np, order=1)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values_wp = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    assert values_wp.shape == (50, 1)
    assert np.allclose(values_wp[:, 0], values_scipy[:, 0], rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------------
# Round-trip identity
# --------------------------------------------------------------------------------------------
def test_rasterize_remap_roundtrip_linear(device: str):
    n = 10
    uv_np, faces_np = _grid_uv_mesh(n=n)
    # Globally-linear field -> bilinear sampling recovers it exactly at interior vertices.
    u = uv_np[:, 0]
    v = uv_np[:, 1]
    attribute_np = np.stack([u, v, u + v], axis=1).astype(np.float32)
    resolution = 128

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    attribute_wp = wp.array(attribute_np, dtype=wp.float32, device=device)
    image = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, resolution)
    recovered = tw.texture.remap_attribute_from_uv(uv_wp, image, order=1).numpy()

    interior = _interior_mask(n)
    assert np.allclose(recovered[interior], attribute_np[interior], rtol=1e-2, atol=1e-2)


def test_rasterize_remap_roundtrip_discrete(device: str):
    rng = np.random.default_rng(11)
    n = 10
    uv_np, faces_np = _grid_uv_mesh(n=n)
    labels_np = rng.integers(0, 5, size=uv_np.shape[0]).astype(np.int32)
    resolution = 128

    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    labels_wp = wp.array(labels_np, dtype=wp.int32, device=device)
    class_image = tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, resolution)
    recovered = tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image).numpy()

    interior = _interior_mask(n)
    assert np.array_equal(recovered[interior], labels_np[interior])


# --------------------------------------------------------------------------------------------
# NaN-UV handling
# --------------------------------------------------------------------------------------------
def test_nan_uv_rows(device: str):
    uv_np = np.array([[0.5, 0.5], [np.nan, np.nan], [0.25, 0.75]], dtype=np.float32)
    image_np = np.random.default_rng(0).random((16, 16, 2), dtype=np.float32)

    uv_wp = points_to_warp_uv(uv_np, device)
    image_wp = wp.array(image_np, dtype=wp.float32, device=device)
    values = tw.texture.remap_attribute_from_uv(uv_wp, image_wp, order=1).numpy()
    assert np.all(np.isnan(values[1]))
    assert not np.any(np.isnan(values[0]))
    assert not np.any(np.isnan(values[2]))

    class_image_np = np.random.default_rng(1).integers(0, 3, size=(16, 16)).astype(np.int32)
    class_image_wp = wp.array(class_image_np, dtype=wp.int32, device=device)
    labels = tw.texture.remap_discrete_attribute_from_uv(uv_wp, class_image_wp).numpy()
    assert labels[1] == -1
    assert labels[0] >= 0
    assert labels[2] >= 0


# --------------------------------------------------------------------------------------------
# Edge cases / validation
# --------------------------------------------------------------------------------------------
def test_empty_faces(device: str):
    uv_wp = wp.array(np.zeros((0, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.float32, device=device)
    image = tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8).numpy()
    assert image.shape == (8, 8, 3)
    assert np.all(image == 0.0)

    labels_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    class_image = tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8).numpy()
    assert class_image.shape == (8, 8)
    assert np.all(class_image == -1)


def test_invalid_resolution(device: str):
    uv_wp = wp.array(np.array([[0.5, 0.5]], dtype=np.float32), dtype=wp.vec2, device=device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((1, 1), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="Resolution must be positive"):
        tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 0)


def test_uv_out_of_range(device: str):
    uv_np = np.array([[0.5, 0.5], [1.5, 0.2]], dtype=np.float32)
    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(np.zeros(0, dtype=np.int32), dtype=wp.int32, device=device)
    attribute_wp = wp.array(np.zeros((2, 1), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tw.texture.rasterize_attribute(uv_wp, faces_wp, attribute_wp, 8)


def test_discrete_negative_labels(device: str):
    uv_np = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
    uv_wp = points_to_warp_uv(uv_np, device)
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    labels_wp = wp.array(np.array([0, -1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        tw.texture.rasterize_discrete_attribute(uv_wp, faces_wp, labels_wp, 8)
