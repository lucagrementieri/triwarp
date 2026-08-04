"""
Benchmarks for ``triwarp.vertices``: the vertex-normal weightings and the angle defect.

Two axes, because the module has two different kinds of function:

* **scan sweep** for ``n_vertices`` and ``mean_vertex_normals`` -- one pass over ``3F`` indices,
  pure throughput, and the place ``lucy`` (28M faces) earns its keep: before the device-reduce fix
  ``n_vertices`` copied the whole 336 MB face buffer to the host just to take a max.
* **valence** for everything that *accumulates* per vertex. All of these do ``3F`` atomic adds
  into ``V`` slots, so in principle a mesh with a few very-high-valence hubs serializes where a
  regular one does not. ``fan_hub`` is the extreme: identical vertex *and* face count to
  ``sphere_med``, but its cone apex and base centre have valence 40 960 against a uniform 6.

``average_onto_vertices`` and ``transfer_onto_vertices`` used to live here and are now in
[`test_interpolation.py`](test_interpolation.py), where ``triwarp.interpolation`` does; their group
names are unchanged, since the group name is the cross-suite key every ``parity`` marker cites.

Measured medians (RTX 5090, ``--device=cuda``): ``area_weighted_vertex_normals`` 0.22 ms against
0.39 ms, ``average_onto_vertices`` 0.1 ms against 0.2 ms. **Under 2x** -- which is the useful
result. Two vertices absorbing 40 960 atomics each cost less than doubling, so CUDA's atomic
aggregation is doing its job and contention is not a hot spot worth engineering around. The axis
stays because it is the only thing that would catch a regression here (a switch to
sort-and-segment-reduce, say, would show up as a large swing), not because it currently hurts.

References
----------
**trimesh**'s area-weighted ``vertex_normals``, rebuilt inside the timed callable because trimesh
caches it. **open3d**'s ``compute_vertex_normals`` is also area-weighted and is triwarp's closest
analogue; it writes the result into the mesh but recomputes on every call rather than caching, so
the shared mesh stays honest across rounds. ``igl.per_vertex_normals`` was dropped: it segfaults
flakily when invoked late in a session that mixes Warp CUDA/CPU JIT with the other native
libraries.

``n_vertices`` has no open3d equivalent worth timing: open3d stores the vertex count explicitly, so
``len(mesh.vertices)`` is O(1) and does not measure the max-reduce triwarp performs.

**pymeshlab**'s ``weightmode`` enum is what makes it useful here: ``compute_normal_per_vertex``
implements four weighting schemes behind one filter, two of which are exactly triwarp's --
``'Simple Average'`` is ``mean_vertex_normals`` and ``'By Area'`` is
``area_weighted_vertex_normals``, so the pair also isolates what the area weight costs on the
reference's side (1.15 -> 1.34 ms). Both write only the vertex-normal attribute and are idempotent,
so they run against the shared MeshSet with no build inside the timed region.

**libigl** answers ``vertex_defects`` (``igl.gaussian_curvature`` is the pointwise angle defect, not
the ball-integrated measure of the same name) and **trimesh** answers it too, so that group is the
module's only three-way one.

The reference **agrees with this module's headline result independently**: across the valence axis
it reads 1.34 against 1.22 ms for the area-weighted normals -- i.e. also flat, and if anything
marginally *faster* on the hub mesh. Two implementations with nothing in common both saying valence
is not a cost driver here is a stronger statement than triwarp's own under-2x spread was on its own.
"""

from __future__ import annotations

from typing import cast

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt

_face_data_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.float32]]] = {}


def _face_normals_and_areas(
    bench_case: BenchCase,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """Precomputed per-face normals and areas -- optional *inputs*, not part of the operation."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_data_cache:
        _face_data_cache[key] = tw.triangles.face_normals_and_areas(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _face_data_cache[key]


@pytest.mark.benchmark(group="n_vertices")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_n_vertices(bench_case: BenchCase) -> None:
    """
    A max-reduce over ``3F`` indices; the scan sweep is here for ``lucy``'s 84M of them.

    Below roughly ``10 ** 3`` indices this row reports the ~340 µs wrapper floor and nothing else
    (see ``test_creation::test_box``), so the small end of the axis loses to ``faces.max()`` by up
    to two orders of magnitude while ``lucy`` wins by 259x. Read the whole axis, not one point: the
    crossover, not either endpoint, is what this group establishes.
    """
    if bench_case.kind == "triwarp":
        faces = cast(twt.Array1dInt32, bench_case.faces_wp)
        result = bench_case.run(lambda: tw.vertices.n_vertices(faces))
        assert result == bench_case.n_vertices
    else:  # numpy reference: what trimesh-style code does on host arrays
        faces = bench_case.faces_np
        result = bench_case.run(lambda: int(faces.max()) + 1)
        assert result == bench_case.n_vertices


@pytest.mark.benchmark(group="mean_vertex_normals")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_mean_vertex_normals(bench_case: BenchCase) -> None:
    """The unweighted scatter, on the scan sweep: the throughput baseline for the group below."""
    if bench_case.kind == "pymeshlab":  # 'Simple Average' is exactly the unweighted scheme
        meshset_pml = bench_case.meshset_pml
        bench_case.run(lambda: meshset_pml.compute_normal_per_vertex(weightmode="Simple Average"))
        assert meshset_pml.current_mesh().vertex_normal_matrix().shape[0] == bench_case.n_vertices
        return
    face_normals, _areas = _face_normals_and_areas(bench_case)
    n_vertices = bench_case.n_vertices
    faces = bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.vertices.mean_vertex_normals(n_vertices, faces, face_normals)
    )
    assert result.shape == (n_vertices,)


@pytest.mark.noparity(
    "trimesh",
    oracle="open3d",
    reason="D2 a different weighting: Trimesh.vertex_normals is *angle*-weighted, not "
    "area-weighted -- it matches triwarp's angle_weighted_vertex_normals to 4.3e-7 and "
    "differs from the area-weighted answer by up to 0.072 on half_torus. trimesh has no "
    "area-weighted vertex "
    "normal, so this row prices 'compute vertex normals' generally; open3d and pymeshlab compute "
    "the same scheme and are the oracles, in tests/test_vertices.py::"
    "test_vertex_normal_weightings_match_open3d_and_pymeshlab.",
)
@pytest.mark.benchmark(group="area_weighted_vertex_normals")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
def test_area_weighted_vertex_normals(bench_case: BenchCase) -> None:
    """Area-weighted scatter, uniform valence 6 against two 40 960-valence hubs."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":  # 'By Area' is triwarp's weighting exactly
        meshset_pml = bench_case.meshset_pml
        bench_case.run(lambda: meshset_pml.compute_normal_per_vertex(weightmode="By Area"))
        assert meshset_pml.current_mesh().vertex_normal_matrix().shape == (n_vertices, 3)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(
            lambda: tw.vertices.area_weighted_vertex_normals(n_vertices, vertices, faces)
        )
        assert result.shape == (n_vertices,)
    elif bench_case.kind == "trimesh":
        # trimesh vertex_normals are area-weighted; rebuild inside (cached property)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).vertex_normals)
        assert result.shape == (n_vertices, 3)
        assert np.isfinite(result).any()
    else:
        # open3d writes the normals into the mesh, but recomputes them on every call rather than
        # caching (measured: identical cost on the second call), so the shared mesh is reusable.
        mesh_o3d = bench_case.mesh_o3d
        result_o3d = bench_case.run(mesh_o3d.compute_vertex_normals)
        assert np.asarray(result_o3d.vertex_normals).shape == (n_vertices, 3)


@pytest.mark.benchmark(group="vertex_defects")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_vertex_defects(bench_case: BenchCase) -> None:
    """
    The angle defect ``2π - Σθ``: the accumulation axis's cheapest member.

    The same ``3F``-into-``V`` scatter as the normals above, over scalars instead of vectors.

    triwarp takes the per-face angles as an *argument*, so the row includes computing them (the
    references do the same internally and there is no way to hand them either one). All three
    agree element-wise including at boundary vertices -- ``igl.gaussian_curvature`` is the
    pointwise angle defect, **not** the ball-integrated Cohen-Steiner/Morvan measure
    ``curvature.discrete_gaussian_curvature`` computes, and confusing the two is the trap this
    group's name is chosen against.

    ``igl.gaussian_curvature`` is one of the ``(V, F)``-family functions, so it returns ``len(V)``
    rows rather than ``F.max() + 1`` and lines up with triwarp on a mesh with unreferenced
    vertices. That is why it can be a row here where ``igl.adjacency_matrix``-family functions need
    a padding transform.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        defects = bench_case.run(
            lambda: tw.vertices.vertex_defects(
                n_vertices, faces, tw.triangles.face_angles(vertices, faces)
            )
        )
        assert defects.shape == (n_vertices,)
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        defects_igl = bench_case.run(lambda: igl.gaussian_curvature(vertices_np, faces_np))
        assert defects_igl.ravel().shape == (n_vertices,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        defects_tm = bench_case.run(
            lambda: tm.curvature.vertex_defects(tm.Trimesh(vertices_np, faces_np, process=False))
        )
        assert defects_tm.shape == (n_vertices,)


@pytest.mark.benchmark(group="area_weighted_vertex_normals_precomputed")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp")
def test_area_weighted_vertex_normals_precomputed(bench_case: BenchCase) -> None:
    """
    The same scatter with ``face_normals`` / ``face_areas`` supplied: the warm half of the cost.

    Passing ``None`` for either makes the function recompute a full per-face pass first. Neither
    reference library has an equivalent -- both always recompute -- so this is triwarp-only, and
    its gap against the group above is what a caller saves by keeping the face data around.
    """
    face_normals, face_areas = _face_normals_and_areas(bench_case)
    n_vertices = bench_case.n_vertices
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.vertices.area_weighted_vertex_normals(
            n_vertices, vertices, faces, face_normals=face_normals, face_areas=face_areas
        )
    )
    assert result.shape == (n_vertices,)
