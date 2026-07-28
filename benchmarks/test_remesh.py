"""
Benchmarks for ``triwarp.remesh``.

Three axes, because the four entry points fail to scale for three different reasons:

* ``subdivide`` on the **scan sweep** -- exactly 4x the faces, one pass, no data dependence. This
  is the module's throughput baseline and the only group here that a face count fully explains.
* ``subdivide_to_size`` on **scale**, sweeping ``max_edge`` -- output size grows as
  ``(L_max / max_edge)^2`` while the *pass count* grows only as ``log2(L_max / max_edge)``, so
  halving the target is roughly four times the output for one extra pass. Edge-length *variance*
  matters too: a single long edge forces another pass over the whole mesh.
* ``flip_to_delaunay`` and ``isotropic_remesh`` on **quality** -- the iterative paths, where the
  number of rounds is set by how far the input is from the fixed point rather than by its size.
  ``saddle_graded`` exists for exactly this: identical connectivity to ``saddle``, worst aspect
  ratio 4 719 against 1.6, so it is badly non-Delaunay and full of slivers while ``saddle`` is
  nearly converged already. A near-Delaunay mesh flips in one or two rounds; a bad one runs all
  ``max_iter`` of them.

Sizing is derived from the mesh's own mean edge length (computed once from the NumPy source so
every library gets the *same* target), which keeps the amount of work proportional to the mesh
rather than to an absolute length that would explode on one mesh and no-op on another.

``flip_to_delaunay`` mutates its face buffer in place, so the timed callable clones it -- the clone
is a single device copy and is negligible against the flip passes it feeds.

References
----------
``subdivide`` has an exact open3d equivalent in ``subdivide_midpoint(1)`` -- same 1:4 midpoint
split, and it returns a new mesh rather than mutating, so the shared mesh is reusable -- and a
trimesh one in ``remesh.subdivide``. ``subdivide_to_size`` has a trimesh counterpart.

Open3D has no ``subdivide_to_size`` (its ``subdivide_midpoint`` takes an iteration count, not an
edge-length target, so it cannot split adaptively), no edge-flip pass, and no isotropic remesher --
``simplify_quadric_decimation`` is decimation, which is the opposite operation. trimesh has neither
of the iterative paths either, so those two groups are triwarp-only.

Caps: ``isotropic_remesh`` is capped at ``bunny`` on the scan sweep (it runs ~1.2 s there and ~10x
that on ``dragon``, which would dominate the whole suite); the split paths are capped at ``dragon``
because a 1:4 subdivision of ``happy_buddha`` / ``lucy`` does not fit a sane memory budget. The
``trimesh`` reference for ``subdivide_to_size`` is capped at ``bunny`` on top of that: it takes
~11 s on ``dragon``, for a ratio the smaller meshes already establish.
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

# Split targets as a fraction of the mean edge length. Both are below 1.0 so edges actually split;
# halving the fraction quadruples the output for one extra pass, which is the point of the pair.
_SPLIT_FRACTIONS = [0.7, 0.35]

# The iterative paths run to seconds a call.
_ROUNDS = 3


@pytest.mark.benchmark(group="subdivide")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_subdivide(bench_case: BenchCase) -> None:
    """Exactly 4x the faces in one pass: the module's clean throughput baseline."""
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(lambda: tw.remesh.subdivide(vertices, faces))
        assert int(new_faces.shape[0]) == 4 * int(faces.shape[0])
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        _new_vertices, new_faces = bench_case.run(lambda: tm.remesh.subdivide(vertices, faces))
        assert new_faces.shape[0] == 4 * faces.shape[0]
    else:  # open3d midpoint subdivision returns a new mesh, so the shared one is reusable
        mesh_o3d = bench_case.mesh_o3d
        n_faces = bench_case.n_faces
        subdivided = bench_case.run(lambda: mesh_o3d.subdivide_midpoint(number_of_iterations=1))
        assert len(subdivided.triangles) == 4 * n_faces


@pytest.mark.benchmark(group="subdivide_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("split_fraction", _SPLIT_FRACTIONS)
def test_subdivide_to_size(bench_case: BenchCase, split_fraction: float) -> None:
    """Adaptive splitting to an edge-length target: quadratic in output, logarithmic in passes."""
    max_edge = split_fraction * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(
            lambda: tw.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert int(new_faces.shape[0]) >= int(faces.shape[0])
    else:
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("trimesh subdivide_to_size takes tens of seconds at this size")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(
            lambda: tm.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert result[1].shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="flip_to_delaunay")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
def test_flip_to_delaunay(bench_case: BenchCase) -> None:
    """Flip rounds until convergence: driven by distance from Delaunay, not by size."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    # ``flip_to_delaunay`` rewrites the winding in place, so each round needs a fresh buffer;
    # without the clone every round after the first would start from an already-Delaunay mesh.
    def run() -> wp.array[wp.int32]:
        return tw.remesh.flip_to_delaunay(vertices, wp.clone(faces), max_iter=100)

    flipped = bench_case.run(run, rounds=_ROUNDS)
    assert int(flipped.shape[0]) == int(faces.shape[0])


@pytest.mark.benchmark(group="isotropic_remesh")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
def test_isotropic_remesh(bench_case: BenchCase) -> None:
    """Five stages x ``iterations``, on a nearly-converged mesh and a badly conditioned one."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    target = bench_case.mean_edge
    _new_vertices, new_faces = bench_case.run(
        lambda: tw.remesh.isotropic_remesh(
            vertices, faces, target_length=target, iterations=_REMESH_ITERATIONS
        ),
        rounds=_ROUNDS,
    )
    assert int(new_faces.shape[0]) > 0
