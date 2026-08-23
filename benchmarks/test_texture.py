"""
Benchmarks for ``triwarp.texture``.

Two inverse pairs, with opposite cost drivers:

* **Rasterize** (``rasterize_attribute``, ``rasterize_discrete_attribute``) — scatter: one thread
  per triangle walks the pixels its UV-space bounding box covers and interpolates the attribute
  barycentrically at each pixel center. Cost is *total covered area in pixels*, so it scales with
  ``resolution**2`` far more strongly than with the face count, and it is sensitive to UV-space
  triangle size: many small triangles amortize the per-triangle setup badly, a few large ones fill
  efficiently. The discrete variant does strictly more work per pixel (it accumulates per-label
  weights and takes an argmax instead of a single lerp).
* **Remap** (``remap_attribute_from_uv``, ``remap_discrete_attribute_from_uv``) — gather: one thread
  per *vertex* does a single bilinear (``order=1``) or nearest (``order=0``) texture fetch. Cost is
  the vertex count, essentially independent of resolution beyond cache behaviour, so these are
  ~2 orders of magnitude cheaper than the rasterizers and are launch-latency bound on the smaller
  meshes. ``order=1`` is the ``wp.lerp`` path; ``order=0`` is the same gather without the blend, so
  timing both isolates the interpolation cost.

Axis: the **scan sweep** for the face-count half, plus a **resolution** sweep of 512 against 2048
on every group. Those are the module's two independent sizes and they pull in opposite directions,
so pinning either one hides half the story: the rasterizers should show a ~16x step across the
resolution pair (4x the pixels each way) and the remappers should show none at all beyond cache
effects. A rasterizer that fails to scale quadratically, or a remapper that *does*, is the signal
this pair is here to produce.

UV coordinates
--------------
The scan meshes carry no UV map, and computing a real one per case is not viable here — an LSCM or
harmonic parametrization needs disk topology (the scan meshes are near-closed) and would dominate
the measurement by orders of magnitude, timing the solver instead of the rasterizer.

So UVs are the **vertices' xy coordinates normalized to the unit square**. This is a projection, not
an injective parametrization: on a closed mesh the front and back surfaces land on the same pixels
and the rasterizer's documented "lowest face index wins" rule resolves the overlap. That is
deliberate and it does not distort what is being measured — the rasterizer's cost is the number of
(triangle, covered pixel) pairs it visits, which a projection produces just as faithfully as a true
atlas, and the overlap additionally exercises the contention path that a real atlas would not. It
*would* matter for a correctness comparison, which is why the parity tests in
``tests/test_texture.py`` use synthetic injective UVs instead.

References
----------
**No CPU baseline is registered.** Neither trimesh, libigl nor open3d has a UV-space attribute
rasterizer or sampler: trimesh's ``visual.texture`` only stores and looks up existing image
textures, libigl has no rasterization module in its Python bindings, and open3d's legacy geometry
exposes UVs as mesh data without any bake or resample operation.

The references the *correctness* tests use are not benchmarkable baselines either. The forward
rasterizers are checked against a **moderngl (OpenGL)** reference, which times GPU driver and
context overhead rather than an algorithm, and the inverse samplers against
``scipy.ndimage.map_coordinates``, which samples a plain grid with no notion of UV or mesh
topology — it is the right correctness oracle for the interpolation and the wrong baseline for the
operation as a whole. So these are before/after self-comparisons, which is what the ``wp.lerp`` and
component-reduction batches touching this module need.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase, skip_larger_than

# Texture sizes: 4x the pixels between the two points, so the rasterizers' quadratic term and the
# remappers' independence from resolution both read directly off the pair.
_RESOLUTIONS = [512, 2048]

# Resolution of the cached source textures the remap groups sample from.
_RESOLUTION = 1024

# Number of distinct labels for the discrete pair.
_N_CLASSES = 8

_uv_cache: dict[tuple[str, str], wp.array[wp.vec2]] = {}
_attribute_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}
_labels_cache: dict[tuple[str, str], wp.array[wp.int32]] = {}
_image_cache: dict[tuple[str, str], twt.Array3dFloat32] = {}
_class_image_cache: dict[tuple[str, str], twt.Array2dInt32] = {}


def _uv_wp(bench_case: BenchCase) -> wp.array[wp.vec2]:
    """Vertex xy normalized to the unit square — a projection, not an atlas (see docstring)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _uv_cache:
        xy = bench_case.vertices_np[:, :2]
        span = np.where(xy.max(axis=0) - xy.min(axis=0) > 0.0, xy.max(axis=0) - xy.min(axis=0), 1.0)
        uv = (xy - xy.min(axis=0)) / span
        _uv_cache[key] = wp.array(
            np.ascontiguousarray(uv, dtype=np.float32), dtype=wp.vec2, device=bench_case.device
        )
    return _uv_cache[key]


def _attribute_wp(bench_case: BenchCase) -> twt.Array2dFloat32:
    """``(n_vertices, 3)`` float attribute (the positions themselves) — the usual bake payload."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _attribute_cache:
        _attribute_cache[key] = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _attribute_cache[key]


def _labels_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    """``(n_vertices,)`` int labels cycling through ``_N_CLASSES``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _labels_cache:
        labels = np.arange(bench_case.n_vertices, dtype=np.int32) % _N_CLASSES
        _labels_cache[key] = wp.array(labels, dtype=wp.int32, device=bench_case.device)
    return _labels_cache[key]


def _image_wp(bench_case: BenchCase) -> twt.Array3dFloat32:
    """Rasterize a float texture to sample back, built once outside the timed region."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _image_cache:
        _image_cache[key] = tw.texture.rasterize_attribute(
            _uv_wp(bench_case), bench_case.faces_wp, _attribute_wp(bench_case), _RESOLUTION
        )
    return _image_cache[key]


def _class_image_wp(bench_case: BenchCase) -> twt.Array2dInt32:
    """Rasterize a class image to sample back, built once outside the timed region."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _class_image_cache:
        _class_image_cache[key] = tw.texture.rasterize_discrete_attribute(
            _uv_wp(bench_case), bench_case.faces_wp, _labels_wp(bench_case), _RESOLUTION
        )
    return _class_image_cache[key]


@pytest.mark.benchmark(group="rasterize_attribute")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_rasterize_attribute(bench_case: BenchCase, resolution: int) -> None:
    """Barycentric scatter of a 3-channel attribute, at 512^2 and 2048^2."""
    uv, faces = _uv_wp(bench_case), bench_case.faces_wp
    attribute = _attribute_wp(bench_case)
    image = bench_case.run(lambda: tw.texture.rasterize_attribute(uv, faces, attribute, resolution))
    assert image.shape[:2] == (resolution, resolution)


@pytest.mark.benchmark(group="rasterize_discrete_attribute")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("resolution", _RESOLUTIONS)
def test_rasterize_discrete_attribute(bench_case: BenchCase, resolution: int) -> None:
    """The same scatter, resolving a per-pixel label argmax instead of interpolating."""
    uv, faces = _uv_wp(bench_case), bench_case.faces_wp
    labels = _labels_wp(bench_case)
    class_image = bench_case.run(
        lambda: tw.texture.rasterize_discrete_attribute(uv, faces, labels, resolution)
    )
    assert class_image.shape == (resolution, resolution)


@pytest.mark.benchmark(group="remap_attribute_from_uv_linear")
@pytest.mark.benchlibs("triwarp")
def test_remap_attribute_from_uv_linear(bench_case: BenchCase) -> None:
    """Per-vertex bilinear texture fetch — the ``wp.lerp`` gather path."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 source texture per mesh is cached")
    uv, image = _uv_wp(bench_case), _image_wp(bench_case)
    values = bench_case.run(lambda: tw.texture.remap_attribute_from_uv(uv, image, order=1))
    assert values.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="remap_attribute_from_uv_nearest")
@pytest.mark.benchlibs("triwarp")
def test_remap_attribute_from_uv_nearest(bench_case: BenchCase) -> None:
    """The same gather without the blend: isolates the bilinear interpolation cost."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 source texture per mesh is cached")
    uv, image = _uv_wp(bench_case), _image_wp(bench_case)
    values = bench_case.run(lambda: tw.texture.remap_attribute_from_uv(uv, image, order=0))
    assert values.shape[0] == bench_case.n_vertices


@pytest.mark.benchmark(group="remap_discrete_attribute_from_uv")
@pytest.mark.benchlibs("triwarp")
def test_remap_discrete_attribute_from_uv(bench_case: BenchCase) -> None:
    """Nearest-neighbor label fetch, preserving the ``-1`` uncovered sentinel."""
    skip_larger_than(bench_case, "happy_buddha", "one 1024^2 class image per mesh is cached")
    uv, class_image = _uv_wp(bench_case), _class_image_wp(bench_case)
    labels = bench_case.run(lambda: tw.texture.remap_discrete_attribute_from_uv(uv, class_image))
    assert labels.shape[0] == bench_case.n_vertices
