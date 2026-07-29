"""
Benchmarks for ``triwarp.contour.marching_triangles`` against potpourri3d.

Two groups:

* **scale** -- a coordinate function on the clean size sweep, i.e. one long closed contour, so the
  cost is the per-face crossing pass plus the fixed prologue.
* **level-set size** -- ``sphere_med`` held fixed while the field's frequency is raised, which is
  where the interesting question lives: this function does its *linking* on the host, so the concern
  is whether the cost tracks the number of curves (a per-component host cost, the shape that made
  ``combine.split`` 262x and ``hole_filling.fill_holes_min_weight`` invert) or merely the number of
  segments.

Measured on an RTX 5090 at ``sphere_med`` (V = 40 962, F = 81 920), the answer is the second one:

| field | curves | segments | triwarp | potpourri3d |
|---|---|---|---|---|
| ``plane`` (coordinate) | 1 | 742 | 1.86 ms | 40.6 ms |
| ``wave12`` | 102 | 8 280 | 5.53 ms | 100 ms |
| ``wave40`` | 1 072 | 26 246 | 17.8 ms | 932 ms |

Across those points the curve count grows **1 072x** and the segment count 35x, while triwarp's
time grows 9.6x -- it tracks segments, not curves. The host-side chain walk is therefore not a
per-component cost in practice; what scales is the readback and the numpy successor pass, both
linear in segments. The group stays in the suite as the probe that keeps it that way. potpourri3d
spreads 23x over the same points.

On the ``scale`` axis with one contour, triwarp runs 1.41 / 1.58 / 3.51 ms against potpourri3d's
3.01 / 39.6 / 207 ms -- a 2.1x / 25x / 59x win, widening with size because the reference rebuilds a
halfedge mesh per call while triwarp's fixed prologue is amortized over more faces.

References
----------
**potpourri3d** is the only reference with an equivalent (``pp3d.marching_triangles``), and it is a
faithful one: it interpolates the same piecewise-linear level set and returns the same curve count
on every case here. Two differences to keep in mind when reading the rows:

* Its output is a list of *barycentric* points rather than positions, so a caller wanting geometry
  pays a decode potpourri3d does not, while triwarp's number includes writing ``wp.vec3``
  positions. That decode is in neither timed callable.
* The one-off ``pp3d.marching_triangles`` builds a geometry-central mesh per call, which is the
  reference's "setup inside the timed callable" convention used throughout this suite (triwarp's row
  likewise includes its ``edges_unique_inverse`` prologue).

**trimesh** has ``intersections.mesh_plane``, which contours the *linear* function a plane defines
and cannot take an arbitrary vertex field; it is timed in
[`test_intersection.py`](test_intersection.py) against ``mesh_with_plane``, the function it actually
corresponds to. **libigl** has no marching triangles (``igl.isolines`` exists in the C++ library but
not in the python bindings). **open3d** has neither, and **scipy** nothing on a surface mesh.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp
from conftest import BenchCase

import triwarp as tw

_ISOVALUE = 0.1137
# potpourri3d runs into hundreds of milliseconds on the many-curve fields.
_ROUNDS = 3

# Field frequency -> (name, wavenumber). ``plane`` is the single-contour control.
_FIELDS = {"plane": 0, "wave12": 12, "wave40": 40}

_field_cache: dict[tuple[str, str], np.ndarray] = {}
_field_wp_cache: dict[tuple[str, str, str], wp.array] = {}


def _field_np(bench_case: BenchCase, field: str) -> np.ndarray:
    """Scalar field on this mesh, cached: it is an input, not part of the measurement."""
    key = (bench_case.mesh_name, field)
    if key not in _field_cache:
        vertices = bench_case.vertices_np
        wavenumber = _FIELDS[field]
        if wavenumber == 0:
            values = vertices[:, 2]
        else:
            values = (
                np.sin(wavenumber * vertices[:, 0])
                * np.cos(wavenumber * vertices[:, 1])
                * np.sin(wavenumber * vertices[:, 2])
            )
        _field_cache[key] = np.ascontiguousarray(values, dtype=np.float64)
    return _field_cache[key]


def _field_wp(bench_case: BenchCase, field: str) -> wp.array:
    """Return the same field as a device buffer."""
    key = (bench_case.mesh_name, field, str(bench_case.device))
    if key not in _field_wp_cache:
        _field_wp_cache[key] = wp.array(
            _field_np(bench_case, field), dtype=wp.float64, device=bench_case.device
        )
    return _field_wp_cache[key]


def _run_case(bench_case: BenchCase, field: str) -> None:
    """Extract one level set, in triwarp or potpourri3d."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        values, n_vertices = _field_wp(bench_case, field), bench_case.n_vertices
        curves, _ = bench_case.run(
            lambda: tw.contour.marching_triangles(
                vertices, faces, values, _ISOVALUE, n_vertices=n_vertices
            ),
            rounds=_ROUNDS,
        )
        assert len(curves) > 0
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        values_np = _field_np(bench_case, field)
        curves_pp = bench_case.run(
            lambda: pp3d.marching_triangles(vertices_np, faces_np, values_np, _ISOVALUE),
            rounds=_ROUNDS,
        )
        assert len(curves_pp) > 0


@pytest.mark.benchmark(group="marching_triangles")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
def test_marching_triangles(bench_case: BenchCase) -> None:
    """One long closed contour of a coordinate function, over the clean size sweep."""
    _run_case(bench_case, "plane")


@pytest.mark.benchmark(group="marching_triangles_curves")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
@pytest.mark.parametrize("field", list(_FIELDS))
def test_marching_triangles_curves(bench_case: BenchCase, field: str) -> None:
    """One mesh, level sets from 1 to ~1 000 curves, to see whether linking cost shows up."""
    _run_case(bench_case, field)
