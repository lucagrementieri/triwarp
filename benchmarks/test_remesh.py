"""
Benchmarks for ``triwarp.remesh``.

Covers the four public entry points. ``subdivide`` / ``subdivide_to_size`` are single-shot and
compared against the ``trimesh`` reference; ``flip_to_delaunay`` and ``isotropic_remesh`` are the
iterative paths and have no CPU reference, so they are timed for ``triwarp`` only.

Sizing is derived from the mesh's own mean edge length (computed once from the NumPy source so
every library gets the *same* target), which keeps the amount of work proportional to the mesh
rather than to an absolute length that would explode on one mesh and no-op on another.

``flip_to_delaunay`` mutates its face buffer in place, so the timed callable clones it — the clone
is a single device copy and is negligible against the flip passes it feeds.

Caps: ``isotropic_remesh`` is capped at ``bunny`` (it runs ~1.2 s there, and ~10x that on
``dragon``, which would dominate the whole suite); the split paths are capped at ``dragon`` because
a 1:4 subdivision of ``happy_buddha`` / ``lucy`` does not fit a sane memory budget. The ``trimesh``
reference for ``subdivide_to_size`` is capped at ``bunny`` on top of that: it takes ~11 s on
``dragon``, which was 70% of this module's total runtime for a ratio the smaller meshes already
establish.
"""

from __future__ import annotations

import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# Iterations for the full remeshing pipeline. The default is 10; 3 keeps the case under a couple
# of seconds while still exercising the split/collapse/flip/smooth/reproject loop several times.
_REMESH_ITERATIONS = 3

# Split target as a fraction of the mean edge length: below 1.0 so roughly every edge splits once.
_SPLIT_FRACTION = 0.7


@pytest.mark.benchmark(group="subdivide")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_subdivide(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(lambda: tw.remesh.subdivide(vertices, faces))
        assert int(new_faces.shape[0]) == 4 * int(faces.shape[0])
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        _new_vertices, new_faces = bench_case.run(lambda: tm.remesh.subdivide(vertices, faces))
        assert new_faces.shape[0] == 4 * faces.shape[0]


@pytest.mark.benchmark(group="subdivide_to_size")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_subdivide_to_size(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    max_edge = _SPLIT_FRACTION * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(
            lambda: tw.remesh.subdivide_to_size(vertices, faces, max_edge)
        )
        assert int(new_faces.shape[0]) >= int(faces.shape[0])
    else:
        skip_larger_than(bench_case, "bunny", "trimesh subdivide_to_size takes ~11 s on dragon")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.remesh.subdivide_to_size(vertices, faces, max_edge))
        assert result[1].shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="flip_to_delaunay")
@pytest.mark.benchlibs("triwarp")
def test_flip_to_delaunay(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    # ``flip_to_delaunay`` rewrites the winding in place, so each round needs a fresh buffer;
    # without the clone every round after the first would start from an already-Delaunay mesh.
    def run() -> wp.array[wp.int32]:
        return tw.remesh.flip_to_delaunay(vertices, wp.clone(faces), max_iter=100)

    flipped = bench_case.run(run)
    assert int(flipped.shape[0]) == int(faces.shape[0])


@pytest.mark.benchmark(group="isotropic_remesh")
@pytest.mark.benchlibs("triwarp")
def test_isotropic_remesh(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny", "isotropic_remesh above bunny dominates the suite")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    target = bench_case.mean_edge
    _new_vertices, new_faces = bench_case.run(
        lambda: tw.remesh.isotropic_remesh(
            vertices, faces, target_length=target, iterations=_REMESH_ITERATIONS
        )
    )
    assert int(new_faces.shape[0]) > 0
