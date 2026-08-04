"""
Benchmarks for ``triwarp.sample.sample_surface_blue_noise`` (parallel dart throwing).

The radius targets ~2,000 samples (same helper formula as ``tests/test_sample.py``). Capped at
``bunny``.

**This group is 95 % device-bound, and reading that correctly is what closed it.** The Bridson
active-list implementation that used to be here measured 260 ms on ``bunny_decimated`` at the 2k
radius with 246 ms of it in kernels across 98 rounds -- 69 % in one kernel (``bridson_propose``) and
28 % in the pruning pass behind it. Four micro-optimizations of that hot kernel had already been
tried and every one lost, because the cost was structural: an active parent must enumerate a
``9x9x9`` shell of background cells every round to find a child in its ``[r, 2r]`` annulus, and the
round count is set by how the front advances rather than by the work. Replacing it with
randomized-priority selection over the whole pool -- 27 cells, a handful of rounds -- is **6.0x at
the 2k radius and 10.6x at half of it**, with *tighter* coverage than either reference. A fifth
micro-optimization was measured on the way out and is worth recording as a null: making the shell
permutation lazy (a partial Fisher-Yates, drawing only the prefix the loop consumes, where the
eager one shuffled up to 728 entries to use the first) is a **wash**, so the shuffle was never
the cost either.

**open3d**'s ``sample_points_poisson_disk`` is the reference: the same blue-noise / Poisson-disk
surface sampling problem, parametrized by sample *count* rather than by radius, so it is given
``_TARGET_SAMPLES`` — the count triwarp's radius is derived to produce. Open3D's implementation
starts from a dense uniform sample and eliminates points down to the target (Yuksel's sample
elimination), where triwarp reduces a dense pool by randomized priority; the comparison is of
cost per sample delivered, not of identical work.

**pymeshlab**'s ``generate_sampling_poisson_disk`` is the only reference that can be given the
*radius* rather than a count: ``radius=PureValue(r)`` overrides ``samplenum`` outright, so both
sides receive the identical parameter and the radius sweep this group is built around maps across
libraries for the first time. Its algorithm is Corsini et al.'s *hierarchical* dart throwing, which
is neither triwarp's flat-grid parallel dart throwing nor open3d's sample elimination -- three
implementations, three schemes, one parametrization. It is also the closest of the three in output:
same exact minimum distance, coverage within 3 %. It pushes the sample cloud onto the MeshSet as
a new mesh, so the set is rebuilt per round.

MeshLab's uniform ``generate_sampling_montecarlo`` is deliberately **not** a row here: it is not a
blue-noise sampler at all (no minimum-distance guarantee), so it would be a floor rather than a
comparison. ``generate_sampling_volumetric`` and ``generate_simplified_point_cloud`` are likewise
different problems.
"""

from __future__ import annotations

import math

import pymeshlab as ml
import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_TARGET_SAMPLES = 2_000
_SEED = 11

# Radius multipliers applied to the ~2k-sample baseline. Halving the radius multiplies the
# background grid's cells by 8 and quadruples the samples that fit, so this is the module's dominant
# knob -- the output count is *derived* from the radius, never requested.
_RADIUS_SCALES = [1.0, 0.5]

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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES, ids=["r1", "rhalf"])
def test_sample_surface_blue_noise(bench_case: BenchCase, radius_scale: float) -> None:
    """
    Maximal Poisson-disk selection from a dense pool, on a background grid sized by the radius.

    Halving the radius is 8x the cells and 4x the output, so the pair should show a large,
    superlinear step -- and it is now a *mild* one (44 -> 54 ms on ``bunny_decimated``), because the
    round count no longer grows with it. open3d is parametrized by *count* rather than radius, so
    its two rows are matched to the sample count each radius implies rather than to the radius.
    """
    skip_larger_than(bench_case, "bunny")
    # Halving the radius quadruples the samples that fit (area / radius^2).
    target = int(_TARGET_SAMPLES / (radius_scale * radius_scale))
    if bench_case.kind == "pymeshlab":
        # MeshLab takes *either* a count or an explicit radius, so this is the one blue-noise
        # reference that can be matched to triwarp's actual parameter: ``radius=PureValue(r)``
        # overrides ``samplenum`` and is fed the identical radius. It pushes a new point-cloud mesh
        # onto the set, so the MeshSet is rebuilt per round.
        radius = radius_scale * _radius_for_mesh(bench_case)
        bench_case.run(
            lambda: bench_case.new_meshset_pml().generate_sampling_poisson_disk(
                radius=ml.PureValue(radius)
            )
        )
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        radius = radius_scale * _radius_for_mesh(bench_case)
        points, face_index = bench_case.run(
            lambda: tw.sample.sample_surface_blue_noise(vertices, faces, radius, seed=_SEED)
        )
        assert points.shape[0] > 0
        assert face_index.shape == points.shape
    else:  # open3d takes a target count instead of a radius; sampling does not mutate the mesh
        mesh_o3d = bench_case.mesh_o3d
        cloud = bench_case.run(lambda: mesh_o3d.sample_points_poisson_disk(number_of_points=target))
        assert len(cloud.points) == target
