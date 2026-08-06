"""
Benchmarks for ``triwarp.visibility``: how far the surface is from a point.

Five functions, three shapes of work, and the split is what the rows are for.

- **Ray bundles.** ``ambient_occlusion`` traces a hemisphere lattice outward per point and
  ``shape_diameter`` fires the same lattice inward and then trims the distances. Both are
  embarrassingly parallel and both are among the slowest per-vertex filters MeshLab ships, so they
  are the module's clearest GPU-versus-one-core rows. ``rays`` is the axis rather than the mesh:
  cost is exactly linear in it on both sides, and the sweep pins that. Read the two groups against
  each other at equal ray counts -- the gap is what ``shape_diameter``'s two extra passes over its
  ``(n_vertices, n_rays)`` distance scratch cost, and it should be small, because the rays dominate.
- **Iterated closest-point.** ``max_tangent_sphere`` shrinks a sphere until nothing but the surface
  touches it, so it pays a full BVH closest-point pass *per iteration* (up to 100) plus an 8-byte
  convergence readback each time -- deliberately, since an extra iteration costs far more than the
  readback. The ``_reach`` row runs it outward, which is the branch where the first ray escapes and
  the packed support-argmax passes over the vertex cloud run instead.
- **One ray, or one sphere.** ``thickness`` is a dispatcher, and its two methods are two orders of
  magnitude apart by construction. It sits on the **depth** axis because both pay for surface
  crossings: a ray through ``shells_8`` meets 16 of them against ``sphere_med``'s 2, and the sphere
  method's convergence depends on local thickness, which nested shells make small.

``volumetric_obscurance`` has **no group**. It shares ``ambient_occlusion``'s kernel and differs
only in a per-hit ``exp(-tau * t)`` factor, so a group over it would re-measure the same axis;
MeshLab's
``compute_scalar_by_volumetric_obscurance`` would nonetheless be a real second reference, and that
is a benchmark gap rather than an API one. Recorded in ``benchmarks/README.md``.

Two caveats for reading the pymeshlab rows. ``compute_scalar_by_shape_diameter_function_per_vertex``
is the most expensive per-vertex filter MeshLab ships (674 ms on ``bunny``) and its
``cone_amplitude`` parameter is a **no-op** in the 2025.07 build -- byte-identical output at 90 and
120 degrees -- so its cone is whatever it is. Both filters write only the vertex scalar attribute,
so the MeshSet is shared rather than rebuilt per call. Open3D has no equivalent for anything here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_QUERY_SEED = 42
_N_QUERIES = 10_000

_surface_cache: dict[tuple[str, str], wp.array] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}
_normals_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}


def _mesh_wp(bench_case: BenchCase) -> wp.Mesh:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mesh_cache:
        _mesh_cache[key] = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    return _mesh_cache[key]


def _surface_points_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """10k on-surface points (subsampled mesh vertices, no jitter) for tangent-sphere queries."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _surface_cache:
        rng = np.random.default_rng(_QUERY_SEED)
        vertices = bench_case.vertices_np
        idx = rng.choice(vertices.shape[0], size=min(_N_QUERIES, vertices.shape[0]), replace=False)
        _surface_cache[key] = wp.array(
            np.ascontiguousarray(vertices[idx], dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _surface_cache[key]


# Rays per point for the bundle groups. MeshLab's default is 64; 256 shows the cost is exactly
# linear in it on both sides, which is the whole shape of these two groups.
_N_RAYS_SWEEP = [64, 256]


def _vertex_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Smooth outward normals over the mesh's own vertices -- an *input* of the bundle queries."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_cache:
        _normals_cache[key] = tw.vertices.area_weighted_vertex_normals(
            bench_case.n_vertices, bench_case.vertices_wp, bench_case.faces_wp
        )
    return _normals_cache[key]


@pytest.mark.benchmark(group="ambient_occlusion")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("n_rays", _N_RAYS_SWEEP)
def test_ambient_occlusion(bench_case: BenchCase, n_rays: int) -> None:
    """A hemisphere ray bundle per vertex: the embarrassingly parallel case, against one core."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab traces the whole bundle on one core")
        meshset_pml = bench_case.meshset_pml  # writes only the vertex scalar
        bench_case.run(lambda: meshset_pml.compute_scalar_ambient_occlusion(rays=n_rays), rounds=3)
        assert meshset_pml.current_mesh().vertex_scalar_array().shape == (n_vertices,)
        return
    mesh, points = _mesh_wp(bench_case), bench_case.vertices_wp
    normals = _vertex_normals_wp(bench_case)
    occlusion = bench_case.run(
        lambda: tw.visibility.ambient_occlusion(mesh, points, normals=normals, n_rays=n_rays)
    )
    assert occlusion.shape == (n_vertices,)


@pytest.mark.benchmark(group="shape_diameter")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("n_rays", _N_RAYS_SWEEP)
def test_shape_diameter(bench_case: BenchCase, n_rays: int) -> None:
    """
    The same bundle fired inward, plus the trimming passes over a ``(n_vertices, n_rays)`` scratch.

    Read against ``ambient_occlusion``: identical ray count, and the gap between the two rows is
    what the two extra passes over the distance scratch cost. It should be small -- the rays
    dominate -- and if it ever is not, the scratch is the thing to attack.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "674 ms a call on bunny at the default ray count")
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_shape_diameter_function_per_vertex(rays=n_rays),
            rounds=3,
        )
        assert meshset_pml.current_mesh().vertex_scalar_array().shape == (n_vertices,)
        return
    mesh, points = _mesh_wp(bench_case), bench_case.vertices_wp
    normals = _vertex_normals_wp(bench_case)
    diameter = bench_case.run(
        lambda: tw.visibility.shape_diameter(mesh, points, normals=normals, n_rays=n_rays)
    )
    assert diameter.shape == (n_vertices,)


@pytest.mark.benchmark(group="thickness_interior")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("method", ["ray", "max_sphere"])
def test_thickness_interior(bench_case: BenchCase, method: Literal["ray", "max_sphere"]) -> None:
    """
    Interior thickness by one ray cast against up to 100 shrinking-sphere iterations.

    On the **depth** axis because both methods pay for surface crossings: a ray through
    ``shells_8`` meets 16 of them against ``sphere_med``'s 2, and the sphere method's convergence
    depends on local thickness, which nested shells make small. The two rows are roughly two
    orders of magnitude apart by construction -- the point is whether that ratio holds when the
    geometry stops being convex.
    """
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    result = bench_case.run(lambda: tw.visibility.thickness(mesh, points, method=method))
    assert result.shape == points.shape


@pytest.mark.benchmark(group="max_tangent_sphere_reach")
@pytest.mark.benchlibs("triwarp")
def test_max_tangent_sphere_reach(bench_case: BenchCase) -> None:
    """Exterior tangent spheres: exercises the ``init_sphere_radii`` inf-distance branch."""
    skip_larger_than(bench_case, "dragon")
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    _, radii = bench_case.run(lambda: tw.visibility.max_tangent_sphere(mesh, points, inwards=False))
    assert radii.shape == points.shape
