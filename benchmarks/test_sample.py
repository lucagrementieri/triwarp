"""
Benchmarks for ``triwarp.sample.sample_surface_blue_noise`` (Bridson dart throwing).

The radius targets ~2,000 samples (same helper formula as ``tests/test_sample.py``). Wall time
is dominated by host orchestration (per-cell seeding launches and per-round syncs before the
fix), which is exactly what this measures. Capped at ``bunny``.
"""

from __future__ import annotations

import math

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_TARGET_SAMPLES = 2_000
_SEED = 11

_radius_cache: dict[str, float] = {}


def _radius_for_mesh(bench_case: BenchCase) -> float:
    """Blue-noise radius targeting ~2k samples (mirrors tests/test_sample.py)."""
    if bench_case.mesh_name not in _radius_cache:
        area = float(tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False).area)
        _radius_cache[bench_case.mesh_name] = math.sqrt(
            (area * 0.5 / (_TARGET_SAMPLES * 0.6162910373)) / math.pi
        )
    return _radius_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="blue_noise")
@pytest.mark.benchlibs("triwarp")
def test_sample_surface_blue_noise(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    radius = _radius_for_mesh(bench_case)
    points, face_index = bench_case.run(
        lambda: tw.sample.sample_surface_blue_noise(vertices, faces, radius, seed=_SEED)
    )
    assert points.shape[0] > 0
    assert face_index.shape == points.shape
