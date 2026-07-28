"""
Benchmarks for ``triwarp.creation``: parametric primitive generation.

The only benchmarks in the suite with **no input mesh**, so they use the ``bench_lib`` fixture
rather than ``bench_case`` — see the mesh-free path in ``conftest.pytest_generate_tests``. Work is
sized by resolution instead of by mesh: ``sections`` for the revolution primitives and
``subdivisions`` for the icosphere. The largest points are chosen so the device has real work
(``sections=4096`` on a 33-point torus profile is ~135k vertices / 270k faces; ``subdivisions=7``
is ~164k faces), while staying inside the scale where the absolute degenerate-triangle threshold in
``revolve`` is meaningful.

References
----------
**trimesh** is the direct port source, so it is apples-to-apples for every function here: identical
profile, identical section count, identical face template. The only structural difference is the
final clean-up — trimesh discovers coincident vertices by hashing positions inside
``Trimesh(process=True)``, while triwarp derives them from the profile. That step is part of what is
being measured on both sides, not factored out.

**open3d**'s factory methods cover four of them, with caveats that matter for reading the numbers:

- ``create_cylinder`` / ``create_cone`` take a ``resolution`` (sections) *and* a ``split`` count
  that subdivides the wall along its axis; ``split=1`` is the closest match to a triwarp cylinder.
- ``create_torus`` takes both radial and tubular resolutions, matching ``major``/``minor_sections``.
- ``create_sphere`` is a UV sphere, so it is the counterpart of ``uv_sphere``, not of ``icosphere``.
  Its ``resolution`` is the *latitude* count and it derives longitude as ``2 * resolution``, so it
  is given ``sections // 2`` to land on the same face count.
- ``create_icosahedron`` exists but is never subdivided, so ``icosphere`` has **no** open3d
  counterpart, and neither do ``revolve``, ``annulus``, ``capsule``, ``extrude_polygon``,
  ``sweep_polygon``, ``truncated_prisms``, ``axis`` or ``random_soup``.

open3d builds these on the CPU in C++ with per-vertex loops, so at low resolution it wins on launch
latency and at high resolution the comparison is the intended one: loop-per-vertex versus one kernel
launch per buffer.

Measured medians
----------------
RTX 5090 / Warp 1.15, ``--device=cuda``. ``sections`` are 32 / 512 / 4096 unless noted.

| case | triwarp-cuda | trimesh | open3d |
|---|---|---|---|
| ``revolve`` (64-point profile) | 313 / 317 / 317 µs | 826 µs / 12.4 / 118 ms | — |
| ``cylinder`` | 342 / 343 / 338 µs | 217 / 573 µs / 4.1 ms | 3.0 / 21.9 / 165 µs |
| ``cone`` | 351 / 353 / 356 µs | 201 / 447 µs / 2.5 ms | 1.6 / 11.8 / 77.5 µs |
| ``annulus`` | 338 / 338 / 338 µs | 237 / 847 µs / 6.2 ms | — |
| ``torus`` (32 minor) | 361 / 355 / 362 µs | 493 µs / 5.8 / 52.1 ms | 11.4 / 167 µs / 1.3 ms |
| ``uv_sphere`` | 576 / 560 / 550 µs | 789 µs / 8.3 / 46.1 ms | 10.1 µs / 2.4 / 319 ms |
| ``icosphere`` (3 / 5 / 7) | 2.7 / 4.4 / 6.3 ms | 429 µs / 2.7 / 65.7 ms | — |
| ``truncated_prisms`` (1k / 256k) | 70 / 164 µs | 180 µs / 112 ms | — |
| ``random_soup`` (1k / 256k) | 92 / 90 µs | 1.2 / 310 ms | — |
| ``sweep_polygon`` (64-gon, 4k path) | 2.1 ms | 13.6 ms | — |

The shape to read here is that **triwarp is flat in resolution** — every revolution primitive costs
the same at 32 sections as at 4096, because the work is two kernel launches over one buffer each.
The ~340 µs floor is host-side allocation and launch overhead shared with every triwarp wrapper,
not anything specific to ``creation``: the profile round trip and the NumPy prologue account for
~75 µs of it and the rest is Warp's launch path. So trimesh wins below roughly 250 sections and
loses by 12-150x above it, and open3d's tight C++ loops win outright at small sizes while still
falling behind above a few thousand sections (its ``create_sphere`` at 4096 is a 580x outlier).

``icosphere`` is the exception to the flat profile, because it is genuinely iterative: each of the
``subdivisions`` passes is a full ``subdivide`` (edge dedup, a sort, a scan), so its cost grows with
the subdivision count rather than being one launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchLibrary

import triwarp as tw

if TYPE_CHECKING:
    import open3d as o3d

# Section counts spanning launch-latency-bound to bandwidth-bound.
_SECTIONS = [32, 512, 4096]
_SUBDIVISIONS = [3, 5, 7]

# Sweep path and profile sizes for the polygon-based generators.
_SWEEP_RING = 64
_SWEEP_PATH = 4096


def _o3d_mesh(bench_lib: BenchLibrary) -> type[o3d.geometry.TriangleMesh]:
    """Legacy open3d ``TriangleMesh`` class, imported lazily like the other open3d references."""
    del bench_lib
    import open3d as o3d

    return o3d.geometry.TriangleMesh


def _ring_wp(n: int, device: str) -> wp.array[wp.vec2]:
    """Regular ``n``-gon as a ``wp.vec2`` ring on ``device``."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    return wp.array(
        np.ascontiguousarray(
            np.column_stack((np.cos(angle_np), np.sin(angle_np))), dtype=np.float32
        ),
        dtype=wp.vec2,
        device=device,
    )


def _ring_np(n: int) -> np.ndarray:
    angle_np = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack((np.cos(angle_np), np.sin(angle_np)))


def _helix_np(n: int) -> np.ndarray:
    """Open helix path: enough turning that every slice frame differs."""
    t_np = np.linspace(0.0, 6.0 * np.pi, n)
    return np.column_stack((5.0 * np.cos(t_np), 5.0 * np.sin(t_np), t_np))


@pytest.mark.benchmark(group="box")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_box(bench_lib: BenchLibrary) -> None:
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(lambda: tw.creation.box(extents=(1.0, 2.0, 3.0), device=device))
        assert int(faces_wp.shape[0]) == 36
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(lambda: tm.creation.box(extents=[1.0, 2.0, 3.0]))
        assert len(mesh_tm.faces) == 12
    else:
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(lambda: mesh_class.create_box(1.0, 2.0, 3.0))
        assert len(mesh_o3d.triangles) == 12


@pytest.mark.benchmark(group="icosphere")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
def test_icosphere(bench_lib: BenchLibrary, subdivisions: int) -> None:
    # No open3d counterpart: create_icosahedron is never subdivided.
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.icosphere(subdivisions=subdivisions, device=device)
        )
        assert int(faces_wp.shape[0]) // 3 == 20 * 4**subdivisions
    else:
        mesh_tm = bench_lib.run(lambda: tm.creation.icosphere(subdivisions=subdivisions))
        assert len(mesh_tm.faces) == 20 * 4**subdivisions


@pytest.mark.benchmark(group="uv_sphere")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_uv_sphere(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.uv_sphere(count=(32, sections // 2), device=device)
        )
        assert int(faces_wp.shape[0]) > 0
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(lambda: tm.creation.uv_sphere(count=[32, sections // 2]))
        assert len(mesh_tm.faces) > 0
    else:
        # open3d's resolution is the latitude count and longitude is derived as 2 * resolution.
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(lambda: mesh_class.create_sphere(1.0, resolution=sections // 2))
        assert len(mesh_o3d.triangles) > 0


@pytest.mark.benchmark(group="cylinder")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_cylinder(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.cylinder(radius=1.0, height=2.0, sections=sections, device=device)
        )
        assert int(faces_wp.shape[0]) // 3 == 4 * sections
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(
            lambda: tm.creation.cylinder(radius=1.0, height=2.0, sections=sections)
        )
        assert len(mesh_tm.faces) == 4 * sections
    else:
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(
            lambda: mesh_class.create_cylinder(1.0, 2.0, resolution=sections, split=1)
        )
        assert len(mesh_o3d.triangles) == 4 * sections


@pytest.mark.benchmark(group="cone")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_cone(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.cone(radius=1.0, height=2.0, sections=sections, device=device)
        )
        assert int(faces_wp.shape[0]) // 3 == 2 * sections
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(lambda: tm.creation.cone(radius=1.0, height=2.0, sections=sections))
        assert len(mesh_tm.faces) == 2 * sections
    else:
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(
            lambda: mesh_class.create_cone(1.0, 2.0, resolution=sections, split=1)
        )
        assert len(mesh_o3d.triangles) == 2 * sections


@pytest.mark.benchmark(group="annulus")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_annulus(bench_lib: BenchLibrary, sections: int) -> None:
    # No open3d counterpart: there is no annular-cylinder factory.
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.annulus(0.5, 1.0, height=2.0, sections=sections, device=device)
        )
        assert int(faces_wp.shape[0]) // 3 == 8 * sections
    else:
        mesh_tm = bench_lib.run(
            lambda: tm.creation.annulus(0.5, 1.0, height=2.0, sections=sections)
        )
        assert len(mesh_tm.faces) == 8 * sections


@pytest.mark.benchmark(group="torus")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_torus(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(
            lambda: tw.creation.torus(1.0, 0.25, major_sections=sections, device=device)
        )
        assert int(faces_wp.shape[0]) // 3 == 2 * 32 * sections
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(lambda: tm.creation.torus(1.0, 0.25, major_sections=sections))
        assert len(mesh_tm.faces) == 2 * 32 * sections
    else:
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(
            lambda: mesh_class.create_torus(
                1.0, 0.25, radial_resolution=sections, tubular_resolution=32
            )
        )
        assert len(mesh_o3d.triangles) == 2 * 32 * sections


@pytest.mark.benchmark(group="revolve")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_revolve(bench_lib: BenchLibrary, sections: int) -> None:
    # The shared engine, timed directly on a 64-point profile so the per-slice work dominates the
    # fixed host prologue. No open3d counterpart.
    profile_np = np.column_stack(
        (1.0 + 0.25 * np.cos(np.linspace(0.0, np.pi, 64)), np.linspace(-1.0, 1.0, 64))
    )
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        profile_wp = wp.array(
            np.ascontiguousarray(profile_np, dtype=np.float32), dtype=wp.vec2, device=device
        )
        _, faces_wp = bench_lib.run(lambda: tw.creation.revolve(profile_wp, sections=sections))
        assert int(faces_wp.shape[0]) // 3 == 2 * 63 * sections
    else:
        mesh_tm = bench_lib.run(lambda: tm.creation.revolve(profile_np, sections=sections))
        assert len(mesh_tm.faces) == 2 * 63 * sections


@pytest.mark.benchmark(group="extrude_polygon")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("ring_size", [64, 1024])
def test_extrude_polygon(bench_lib: BenchLibrary, ring_size: int) -> None:
    # Dominated by the ear-clipping triangulation of the ring, which is the interesting part: a
    # convex ring takes triwarp's single-fan fast path. No open3d counterpart.
    shapely = pytest.importorskip("shapely.geometry")
    if bench_lib.kind == "triwarp":
        ring_wp = _ring_wp(ring_size, str(bench_lib.device))
        _, faces_wp = bench_lib.run(lambda: tw.creation.extrude_polygon(ring_wp, 1.0))
        assert int(faces_wp.shape[0]) // 3 == 2 * (ring_size - 2) + 2 * ring_size
    else:
        polygon = shapely.Polygon(_ring_np(ring_size))
        mesh_tm = bench_lib.run(lambda: tm.creation.extrude_polygon(polygon, 1.0))
        assert len(mesh_tm.faces) > 0


@pytest.mark.benchmark(group="sweep_polygon")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_sweep_polygon(bench_lib: BenchLibrary) -> None:
    # A long helix so the per-slice frame construction and wall emission dominate the one-off
    # triangulation. No open3d counterpart.
    shapely = pytest.importorskip("shapely.geometry")
    path_np = _helix_np(_SWEEP_PATH)
    if bench_lib.kind == "triwarp":
        device = str(bench_lib.device)
        ring_wp = _ring_wp(_SWEEP_RING, device)
        path_wp = wp.array(
            np.ascontiguousarray(path_np, dtype=np.float32), dtype=wp.vec3, device=device
        )
        _, faces_wp = bench_lib.run(lambda: tw.creation.sweep_polygon(ring_wp, path_wp))
        assert int(faces_wp.shape[0]) // 3 > 0
    else:
        polygon = shapely.Polygon(_ring_np(_SWEEP_RING))
        mesh_tm = bench_lib.run(lambda: tm.creation.sweep_polygon(polygon, path_np))
        assert len(mesh_tm.faces) > 0


@pytest.mark.benchmark(group="truncated_prisms")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("face_count", [1_024, 262_144])
def test_truncated_prisms(bench_lib: BenchLibrary, face_count: int) -> None:
    # Perfectly per-triangle parallel, so this is the cleanest kernel-versus-NumPy comparison in the
    # module. The input soup is built outside the timed region: it is the input, not the operation.
    # No open3d counterpart.
    triangles_np = np.random.default_rng(0).random((face_count, 3, 3)) + np.array([0.0, 0.0, 1.0])
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        vertices_wp = wp.array(
            np.ascontiguousarray(triangles_np.reshape(-1, 3), dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        )
        faces_wp = wp.array(
            np.arange(3 * face_count, dtype=np.int32), dtype=wp.int32, device=device
        )
        _, out_faces_wp = bench_lib.run(lambda: tw.creation.truncated_prisms(vertices_wp, faces_wp))
        assert int(out_faces_wp.shape[0]) // 3 == 8 * face_count
    else:
        mesh_tm = bench_lib.run(lambda: tm.creation.truncated_prisms(triangles_np))
        assert len(mesh_tm.faces) == 8 * face_count


@pytest.mark.benchmark(group="random_soup")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("face_count", [1_024, 262_144])
def test_random_soup(bench_lib: BenchLibrary, face_count: int) -> None:
    # Pure generation: Warp's per-thread RNG against NumPy's global stream. Values differ by
    # construction (see the note in triwarp.creation.random_soup); only the cost is comparable.
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        vertices_wp, _ = bench_lib.run(
            lambda: tw.creation.random_soup(face_count, seed=0, device=device)
        )
        assert int(vertices_wp.shape[0]) == 3 * face_count
    else:
        mesh_tm = bench_lib.run(lambda: tm.creation.random_soup(face_count))
        assert len(mesh_tm.faces) == face_count
