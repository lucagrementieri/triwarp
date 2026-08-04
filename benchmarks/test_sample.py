"""
Benchmarks for ``triwarp.sample``: uniform surface sampling and blue-noise selection.

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

**pymeshlab**'s ``generate_sampling_poisson_disk`` and **libigl**'s ``igl.blue_noise`` are the two
references that can be given the *radius* rather than a count (``radius=PureValue(r)`` overrides
MeshLab's ``samplenum`` outright; igl's third positional argument *is* ``r``), so they receive the
identical parameter and the radius sweep this group is built around maps across libraries. MeshLab's
algorithm is Corsini et al.'s *hierarchical* dart throwing and igl's is Bridson active-list dart
throwing -- **four implementations, four schemes, one parametrization**, with triwarp's randomized-
priority selection and open3d's sample elimination as the other two. MeshLab is the closest of the
three in output: same exact minimum distance, coverage within 3 %. It pushes the sample cloud onto
the MeshSet as a new mesh, so the set is rebuilt per round.

**libigl is where this port came from, and the row inverted when the algorithm changed.** The
``30x`` oversampling factor ``sample_surface_blue_noise`` draws its pool at is
``igl::blue_noise``'s, and while triwarp ran Bridson too, igl was *faster*: 78.4 ms against 98.5 on
``bunny`` at ``4 * mean_edge``.
After ``ae26e8f`` the same pair reads **18.0 ms against 73.8** — and the margin
**grows as the radius falls** (4.1x at ``4 * mean_edge``, 5.4x at the 2k radius, 9.0x at half of it,
24x on ``bunny_decimated`` at half), because igl's serial cost is per accepted sample where
triwarp's is per round. igl also returns 2-7% *fewer* samples at the same radius, so the ratios are
a lower bound per sample delivered — and its *quality* is the best of the three references
(``tests/test_sample.py`` measures its coverage gap at 1.073 r against triwarp's 1.103 and MeshLab's
1.112), so this is not speed bought with quality on either side. It gets ``rounds=3``: 278 ms at the
2k radius on ``bunny`` and 1 196 at half of it.

MeshLab's uniform ``generate_sampling_montecarlo`` is deliberately **not** a row in ``blue_noise``:
it is not a blue-noise sampler at all (no minimum-distance guarantee), so it would be a floor
rather than a comparison. It belongs to the same question as the ``sample_surface`` group below,
where it is not a row either -- for a different reason, given there.
``generate_sampling_volumetric`` and ``generate_simplified_point_cloud`` are likewise different
problems.
"""

from __future__ import annotations

import math

import igl
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

# Sample counts for the uniform group: a decade apart, since the cost is linear in the count and
# the mesh contributes only the one-off area CDF.
_UNIFORM_COUNTS = [10_000, 100_000]

_radius_cache: dict[str, float] = {}


def _radius_for_mesh(bench_case: BenchCase) -> float:
    """Blue-noise radius targeting ~2k samples (mirrors tests/test_sample.py)."""
    if bench_case.mesh_name not in _radius_cache:
        area = float(tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False).area)
        _radius_cache[bench_case.mesh_name] = math.sqrt(
            (area * 0.5 / (_TARGET_SAMPLES * 0.6162910373)) / math.pi
        )
    return _radius_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="sample_surface")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh")
@pytest.mark.parametrize("count", _UNIFORM_COUNTS, ids=["n10k", "n100k"])
def test_sample_surface(bench_case: BenchCase, count: int) -> None:
    """
    Uniform area-weighted surface sampling: the dense pool every blue-noise sampler starts from.

    All three libraries take the same two parameters (a count and a seed) and return the same two
    things (positions and the face index each sample landed on), so this is the module's one group
    where nothing has to be matched up -- ``igl.random_points_on_mesh(n, V, F, seed)`` and
    ``tm.sample.sample_surface(mesh, n, seed=)`` are the same call as triwarp's.

    The axis is the count, and it separates the two sides cleanly: **the references are linear in it
    and triwarp is flat.** Measured on ``bunny``, medians: igl 21.1 -> 55.2 ms and trimesh
    25.8 -> 55.0 from 10k to 100k, against triwarp's **260.8 -> 252.8 µs** — a decade more samples
    for no more time, because at these counts triwarp's row is the two launches and the area CDF
    rather than the sampling. So read this group as 80-200x, and read the *slope* as the statement:
    the crossover where triwarp's per-sample cost becomes visible is above 100 000 samples.

    open3d's ``sample_points_uniformly`` and MeshLab's ``generate_sampling_montecarlo`` are the same
    operation but return a bare point cloud with no face index, so they would need a closest-point
    decode before they could be asserted against the area law -- a transform on the *reference* to
    make it comparable, which is what the two references above avoid.

    trimesh rebuilds its ``tm.Trimesh`` inside the timed callable, as the other trimesh rows in the
    suite do, because the area CDF is cached on the mesh object and reusing it would time a
    lookup.
    """
    skip_larger_than(bench_case, "bunny", "the CPU references are single-threaded per sample")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points, face_index = bench_case.run(
            lambda: tw.sample.sample_surface(vertices, faces, count, seed=_SEED)
        )
        assert points.shape == (count,)
        assert face_index.shape == (count,)
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        _barycentric, face_index_igl, points_igl = bench_case.run(
            lambda: igl.random_points_on_mesh(count, vertices_np, faces_np, _SEED)
        )
        assert points_igl.shape == (count, 3)
        assert face_index_igl.shape == (count,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        points_tm, face_index_tm = bench_case.run(
            lambda: tm.sample.sample_surface(
                tm.Trimesh(vertices_np, faces_np, process=False), count, seed=_SEED
            )
        )
        assert points_tm.shape == (count, 3)
        assert face_index_tm.shape == (count,)


@pytest.mark.benchmark(group="blue_noise")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES, ids=["r1", "rhalf"])
def test_sample_surface_blue_noise(bench_case: BenchCase, radius_scale: float) -> None:
    """
    Maximal Poisson-disk selection from a dense pool, on a background grid sized by the radius.

    Halving the radius is 8x the cells and 4x the output, so the pair should show a large,
    superlinear step -- and it is now a *mild* one (44 -> 54 ms on ``bunny_decimated``), because the
    round count no longer grows with it. open3d is parametrized by *count* rather than radius, so
    its two rows are matched to the sample count each radius implies rather than to the radius.

    **libigl is the reference this port was written from** -- ``sample_surface_blue_noise`` still
    sizes its pool at the ``30x`` oversampling factor ``igl::blue_noise`` uses -- and it takes the
    radius directly, so it and MeshLab both receive triwarp's own parameter. It is Bridson
    active-list dart throwing, which is what triwarp *was* before ``ae26e8f``: four schemes across
    four libraries on one parametrization.
    """
    skip_larger_than(bench_case, "bunny")
    # Halving the radius quadruples the samples that fit (area / radius^2).
    target = int(_TARGET_SAMPLES / (radius_scale * radius_scale))
    if bench_case.kind == "igl":
        radius = radius_scale * _radius_for_mesh(bench_case)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        _barycentric, _face_index, points_igl = bench_case.run(
            lambda: igl.blue_noise(vertices_np, faces_np, radius), rounds=3
        )
        assert points_igl.shape[0] > 0
        return
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
