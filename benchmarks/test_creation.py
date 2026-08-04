"""
Benchmarks for ``triwarp.creation``: parametric primitive generation.

The only benchmarks in the suite with **no input mesh**, so they use the ``bench_lib`` fixture
rather than ``bench_case`` — see the mesh-free path in ``conftest.pytest_generate_tests``. Their
axis is **resolution**, the mesh-free member of the axis set: work is sized by ``sections`` for the
revolution primitives, ``subdivisions`` for the icosphere and ``face_count`` for the per-triangle
generators, all as a plain ``pytest.mark.parametrize`` because there is no mesh for ``benchaxis`` to
select. The largest points are chosen so the device has real work
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

**pymeshlab** covers five, and it is the only reference for ``icosphere``:

- ``create_sphere(subdiv=)`` is "a regular subdivision of an icosahedron" -- so unlike open3d's
  ``create_sphere`` it pairs with ``icosphere``, **not** with ``uv_sphere``, and it takes the same
  subdivision count. That closes the gap open3d left.
- ``create_torus(hsubdiv=, vsubdiv=)``, ``create_annulus(sides=)`` and ``create_cone(subdiv=)`` map
  directly onto the section counts, and the annulus is the only annular factory outside trimesh.
- ``create_cube(size=)`` is the ``box`` counterpart, with one caveat: it takes a single scale factor
  rather than three extents, so it builds a cube where the other three build a 1x2x3 box. Twelve
  triangles either way, which is all that group measures.

- ``create_tetrahedron`` / ``create_octahedron`` / ``create_dodecahedron`` are where triwarp's
  three constant Platonic tables came from, and the *only* reference for them: trimesh has no
  Platonic solid but the icosahedron and igl has no generators at all. They take no parameters on
  either side, so those rows are pure fixed cost, like ``box``.
- ``create_grid(numvertx=, numverty=)`` and ``create_sphere_cap(angle=, subdiv=)`` are the two open
  primitives, and both have a real axis: the grid's vertex counts and the cap's subdivision level.
  ``create_sphere_cap`` takes the full aperture in degrees where triwarp takes the polar half-angle
  in radians, so its ``angle`` is twice triwarp's -- both rows generate the identical lattice.

There is no ``uv_sphere``, ``revolve``, ``capsule``, ``extrude_polygon``, ``sweep_polygon``,
``truncated_prisms``, ``axis`` or ``random_soup`` counterpart -- MeshLab's parametric primitives are
a fixed catalogue, not a profile-and-sweep toolkit, which is the structural difference this module's
open3d note already makes.

Every ``create_*`` filter pushes a new mesh onto the MeshSet, so each row builds a fresh
``ml.MeshSet`` inside the timed callable. That construction is ~30 us empty, so unlike the
mesh-driven modules the build is not a meaningful share of these rows.

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
| ``icosphere`` (3 / 5 / 7) | 245 / 253 / 345 µs | 472 µs / 2.7 / 67.1 ms | — |
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

``icosphere`` **used to be** the exception to the flat profile, and this group is what retired it.
It was iterative — one full ``subdivide`` per level (edge dedup, a radix sort, a count readback,
~17 launches), so its cost grew with the subdivision count rather than being one launch: 4.8 / 7.9 /
6.9 ms at levels 3 / 5 / 7, the non-monotonicity itself the tell that host cost and not the 164k
output faces was being measured. Generating the connectivity in closed form instead (see the Notes
of [`triwarp.creation.icosphere`][]) makes it ``subdivisions + 2`` launches and puts it back on the
floor with everything else: **19-31x** faster, and it now beats trimesh at every level rather than
losing 6x at level 3 (245 µs against 472, and 345 µs against 67.1 ms at level 7, a 195x win).

``triangulate_polygon`` is the remaining exception, and the module's worked example of a cost
that is not where it looks. It delegates to ``polyline.triangulate_polyline``, a **parallel**
multi-round ear clipper — every launch in the round loop is ``dim=n``, and a round clips a whole
*independent set* of ears at once. What is serial is the **round count**: a round costs four
``dim=n`` launches whatever it clips, so the only thing that matters is how many ears survive per
round. Since the round loop moved onto the device (``wp.capture_while``, so no readback per round)
that costs 14 µs a round at 64 points, and the group measures **2.2 / 4.5 ms** against 4.5 / 6.9
before — 2.07x and 1.53x.

Read the residual against its attribution rather than against the round loop, because the round loop
is no longer the cost: at 64 points the clip itself is 0.22 ms of the 2.2, and **1.05 ms is the
prologue** — ``open_polyline``'s ``is_closed`` (0.24), ``polyline_normal`` (0.35) and
``polyline_centroid`` (0.25), three reductions that each end in a host readback because their result
is a Python-scope ``wp.vec3``, plus the reflex-count readback. That share is *flat in n*, so it is
the whole gap to trimesh's 0.12 ms at 64 points and none of it at 1 024.

The round *count* is what this group caught first. ``select_independent`` used to rank competing
ear candidates by their raw ring index, which on an alternating star lets ear ``i - 2`` suppress
ear ``i`` for every ``i``, so exactly **one** ear was clipped per round and the loop ran to its
``n`` cap: 6.1 ms at 64 points and **141 ms** at 1 024, growing as ``n^1.12`` (rounds proportional
to ``n`` times a slowly growing per-round cost) rather than the ``O(L^2)`` a serial clipper would
give. Ranking by a bijective hash of the ring index makes it the textbook maximal-independent-set
rule, which retires a constant fraction per round: 16 and 30 rounds, the latter instead of 1 022.

``extrude_polygon`` only ever exercises the *convex* fast path (a single fan), which is why the ear
clipper is benchmarked directly on a non-convex star ring — the same reasoning that puts
``polyline.simplify_polyline`` in its own group in [`test_polyline.py`](test_polyline.py). It is
also why this row does **not** move with anything done to the ear loop (measured flat, 0.81-1.05x
across an A/B of the device-driven round loop): a convex ring reaches the fan and returns, so
``extrude_polygon`` is a floor row wearing a triangulator's name, and its loss to trimesh at 64
points is the ~340 µs wrapper floor plus the same flat prologue.

Deliberately not benchmarked
----------------------------
``box``, ``icosahedron`` and ``axis`` are constant tables, so they measure only the per-wrapper
floor — which is worth measuring exactly once, and ``test_box`` is where that happens (see its
comment). ``capsule`` and ``extrude_triangulation`` are the same ``revolve`` and triangulation
engines behind a different profile, already covered by ``cylinder`` / ``uv_sphere`` and
``extrude_polygon`` respectively.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pymeshlab as ml
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

# Ring sizes for the ear-clipping triangulator. Kept modest because the *round count*, not the
# per-round work, is what grows -- and because the two points are what showed the round count was
# growing linearly rather than logarithmically in the ring size.
_STAR_RINGS = [64, 1024]


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


def _star_np(n: int) -> np.ndarray:
    """Non-convex star ring: alternating radii, so ear clipping cannot take the single-fan path."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, 0.45)
    return np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))


def _star_wp(n: int, device: str) -> wp.array[wp.vec2]:
    return wp.array(
        np.ascontiguousarray(_star_np(n), dtype=np.float32), dtype=wp.vec2, device=device
    )


def _helix_np(n: int) -> np.ndarray:
    """Open helix path: enough turning that every slice frame differs."""
    t_np = np.linspace(0.0, 6.0 * np.pi, n)
    return np.column_stack((5.0 * np.cos(t_np), 5.0 * np.sin(t_np), t_np))


def _new_cube_pml() -> ml.MeshSet:
    """Build a fresh MeshSet carrying MeshLab's unit cube; every ``create_*`` pushes a mesh."""
    meshset_pml = ml.MeshSet()
    meshset_pml.create_cube(size=1.0)
    return meshset_pml


@pytest.mark.benchmark(group="box")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
def test_box(bench_lib: BenchLibrary) -> None:
    """
    The suite's launch-overhead calibration probe.

    ``box`` is a 12-triangle constant table, so there is nothing to sweep and nothing to scale: what
    this group measures is the fixed host-side cost of *any* triwarp wrapper call — allocation plus
    Warp's launch path, of which the NumPy prologue is ~75 us. Measured at ~340 us on an RTX 5090.

    That number is the baseline every other group should be read against: a group sitting at the
    floor across its whole axis is reporting launch overhead rather than an algorithm, and its
    inputs are too small to tell anyone anything.

    **Read it from a full-suite run, not from this module alone.** Being the first group in the
    file, it absorbs each library's one-time initialization -- measured at 52 ms for triwarp and
    102 ms for trimesh when ``test_creation.py`` runs by itself, against ~340 us and ~200 us once
    anything else has already imported and JITed. That is a property of first-touch cost, not of
    ``box``, and it applies to whichever group happens to run first in any module.
    """
    if bench_lib.kind == "pymeshlab":  # a cube, not a 1x2x3 box: one scale factor is all it takes
        assert _new_cube_pml().current_mesh().face_number() == 12
        bench_lib.run(_new_cube_pml)
        return
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


@pytest.mark.benchmark(group="platonic_solids")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize(
    ("builder", "filter_name", "n_faces"),
    [
        ("tetrahedron", "create_tetrahedron", 4),
        ("octahedron", "create_octahedron", 8),
        ("dodecahedron", "create_dodecahedron", 36),
    ],
)
def test_platonic_solids(
    bench_lib: BenchLibrary, builder: str, filter_name: str, n_faces: int
) -> None:
    """Constant tables on both sides, so this is ``box``'s fixed-cost probe once per solid."""
    if bench_lib.kind == "pymeshlab":

        def solid_pml() -> int:
            meshset_pml = ml.MeshSet()
            getattr(meshset_pml, filter_name)()
            return meshset_pml.current_mesh().face_number()

        assert bench_lib.run(solid_pml) == n_faces
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(lambda: getattr(tw.creation, builder)(device=device))
    assert int(faces_wp.shape[0]) // 3 == n_faces


@pytest.mark.benchmark(group="grid")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("count", [32, 512])
def test_grid(bench_lib: BenchLibrary, count: int) -> None:
    """A ``count x count`` vertex lattice: one NumPy prologue against MeshLab's per-vertex loop."""
    n_faces = 2 * (count - 1) ** 2
    if bench_lib.kind == "pymeshlab":

        def grid_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_grid(numvertx=count, numverty=count)
            return meshset_pml.current_mesh().face_number()

        assert bench_lib.run(grid_pml) == n_faces
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(lambda: tw.creation.grid(count=(count, count), device=device))
    assert int(faces_wp.shape[0]) // 3 == n_faces


@pytest.mark.benchmark(group="sphere_cap")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("subdivisions", [3, 6])
def test_sphere_cap(bench_lib: BenchLibrary, subdivisions: int) -> None:
    """
    The concentric-ring lattice, whose cost is a Python loop over ``2 ** subdivisions`` rings.

    Unlike the revolution primitives this one is *not* flat in resolution on the triwarp side: the
    ring loop is host-side NumPy and grows linearly in the ring count, so read the slope here as the
    prologue's rather than the device's. MeshLab's own generator is a per-vertex C++ loop.
    """
    n_faces = 6 * (2**subdivisions) ** 2
    if bench_lib.kind == "pymeshlab":

        def cap_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_sphere_cap(angle=60.0, subdiv=subdivisions)
            return meshset_pml.current_mesh().face_number()

        assert bench_lib.run(cap_pml) == n_faces
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(
        lambda: tw.creation.sphere_cap(
            angle=math.radians(30.0), subdivisions=subdivisions, device=device
        )
    )
    assert int(faces_wp.shape[0]) // 3 == n_faces


@pytest.mark.benchmark(group="icosphere")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
def test_icosphere(bench_lib: BenchLibrary, subdivisions: int) -> None:
    # No open3d counterpart: create_icosahedron is never subdivided. MeshLab's create_sphere is
    # exactly this scheme, so it is the only reference this group has beyond trimesh.
    #
    # Since the connectivity became closed-form this row is *also* a floor row at every subdivision
    # level it is asked for -- 245 / 253 / 345 us at 3 / 5 / 7 is one launch per level over buffers
    # that reach 164k faces, so the axis reports the wrapper floor rather than the output size.
    if bench_lib.kind == "pymeshlab":
        if subdivisions > 8:
            pytest.skip("MeshLab's create_sphere caps subdiv at 8")

        def sphere_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_sphere(radius=1.0, subdiv=subdivisions)
            return meshset_pml.current_mesh().face_number()

        assert bench_lib.run(sphere_pml) == 20 * 4**subdivisions
        return
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
    """
    UV sphere at matched tessellation -- open3d's ``resolution`` is neither axis on its own.

    Measured, exactly, at every point on this axis: ``create_sphere(resolution=r)`` produces the
    same vertex and face counts as ``uv_sphere(count=(2 * r, r // 2))``, so ``resolution`` is *half*
    the longitude count and *twice* the latitude count. Pairing it with ``count=(32, r)``, as this
    group used to, compared meshes of different sizes -- and the gap widened along the axis, because
    open3d's face count is quadratic in ``resolution`` while a fixed 32-longitude sweep is linear:
    15 360 triwarp faces against 65 024 open3d ones at ``sections=256``, a 4.2x mismatch reported as
    a speed ratio.

    ``tests/test_creation.py::test_uv_sphere_matches_open3d`` pins the mapping so it cannot drift
    back.
    """
    count = (2 * sections, sections // 2)
    if bench_lib.kind == "triwarp":
        device = bench_lib.device
        _, faces_wp = bench_lib.run(lambda: tw.creation.uv_sphere(count=count, device=device))
        assert int(faces_wp.shape[0]) > 0
    elif bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(lambda: tm.creation.uv_sphere(count=list(count)))
        assert len(mesh_tm.faces) > 0
    else:
        mesh_class = _o3d_mesh(bench_lib)
        mesh_o3d = bench_lib.run(lambda: mesh_class.create_sphere(1.0, resolution=sections))
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_cone(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "pymeshlab":

        def cone_pml() -> None:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_cone(r0=1.0, r1=0.0, h=2.0, subdiv=sections)

        bench_lib.run(cone_pml)
        return
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


@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D6 different primitive: MeshLab's create_annulus builds a flat holed *disk* while "
    "triwarp's annulus is an annular *cylinder* with height, so the two do not bound the same "
    "solid and MeshLab emits fewer faces at the same side count. trimesh is the oracle here and "
    "is asserted in tests/test_creation.py::test_annulus.",
)
@pytest.mark.benchmark(group="annulus")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_annulus(bench_lib: BenchLibrary, sections: int) -> None:
    # No open3d counterpart: there is no annular-cylinder factory. MeshLab's annulus is a flat holed
    # disk rather than triwarp's annular *cylinder*, so it builds fewer faces at the same section
    # count -- a floor for this row rather than an equivalent.
    if bench_lib.kind == "pymeshlab":

        def annulus_pml() -> None:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_annulus(internalradius=0.5, externalradius=1.0, sides=sections)

        bench_lib.run(annulus_pml)
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_torus(bench_lib: BenchLibrary, sections: int) -> None:
    if bench_lib.kind == "pymeshlab":

        def torus_pml() -> int:
            meshset_pml = ml.MeshSet()
            meshset_pml.create_torus(hradius=1.0, vradius=0.25, hsubdiv=sections, vsubdiv=32)
            return meshset_pml.current_mesh().face_number()

        assert bench_lib.run(torus_pml) == 2 * 32 * sections
        return
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


@pytest.mark.benchmark(group="triangulate_polygon")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("ring_size", _STAR_RINGS)
def test_triangulate_polygon(bench_lib: BenchLibrary, ring_size: int) -> None:
    """
    Ear clipping on a *non-convex* ring: the only group here that reaches the clipper at all.

    ``extrude_polygon`` above hands the triangulator a convex ring, which takes the single-fan fast
    path and never reaches the ear loop. A star ring forces it. Each round is fully parallel (four
    ``dim=n`` launches) and the round loop itself runs on device, so what this group measures is
    **how many rounds the independent-set rule needs**: 16 and 30, against 62 and 1 022 before the
    ranking hash.

    At ``ring_size=64`` that is no longer the dominant term -- the clip is 0.22 ms of a 2.2 ms call
    and the flat plane-fitting prologue is 1.05 -- so read the small point as a floor row and the
    large one as a rounds ratio. No open3d counterpart.
    """
    shapely = pytest.importorskip("shapely.geometry")
    if bench_lib.kind == "triwarp":
        ring_wp = _star_wp(ring_size, str(bench_lib.device))
        _vertices, faces_wp = bench_lib.run(lambda: tw.creation.triangulate_polygon(ring_wp))
        assert int(faces_wp.shape[0]) // 3 == ring_size - 2
    else:
        polygon = shapely.Polygon(_star_np(ring_size))
        _vertices, faces_tm = bench_lib.run(lambda: tm.creation.triangulate_polygon(polygon))
        assert faces_tm.shape[0] > 0


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


@pytest.mark.noparity(
    "trimesh",
    reason="D5 stochastic with no shared invariant: both draw n random triangles in the unit "
    "cube, but triwarp takes a seed and trimesh does not, so no two runs can be aligned. Only "
    "the shape, the bounds and the cost are comparable, and the first two are asserted in "
    "tests/test_creation.py::test_random_soup without needing trimesh.",
)
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
