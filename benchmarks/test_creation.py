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
  counterpart, and neither do ``revolve``, ``annulus``, ``capsule``, ``sweep_polygon``,
  ``truncated_prisms``, ``axis`` or ``random_soup``.
- ``extrude_polygon`` **does** have one, in the tensor API: ``o3d.t.geometry.TriangleMesh``'s
  ``extrude_linear``. An earlier version of this list said otherwise. It walls an *already
  triangulated* mesh rather than triangulating a ring, so its row is the walls alone and the cap fan
  is built outside the timed callable -- and its faces must be ``Int32``/``Int64``, where
  ``RaycastingScene.add_triangles`` takes ``UInt32``. pyvista's ``extrude(capping=True)`` is the
  third implementation and does triangulate the cap, like trimesh and triwarp.

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
  Platonic solid but the icosahedron. **libigl has exactly one**, ``igl.icosahedron()``, which is
  why the ``platonic_solids`` group carries an ``icosahedron`` case that igl and trimesh answer and
  MeshLab's three others they do not. They take no parameters on any side, so those rows are pure
  fixed cost, like ``box``.
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

**pyvista** (VTK 9.6) is the reference for the four parametric-surface groups and for nothing else
here -- ``pv.Sphere`` / ``Cube`` / ``Icosphere`` come in their own frames and scales, so the
Platonic and revolution rows keep trimesh, open3d and MeshLab. ``pv.Parametric*`` is triwarp's
source for those surfaces and evaluates the identical map, but it takes a different route to the
mesh: a per-point C++ loop, then ``vtkCleanPolyData`` welding the raw lattice by *distance*, where
triwarp identifies the seam combinatorially and never allocates the duplicates. ``clean=True`` is
passed explicitly on every row because pyvista's own default differs per surface, and an unwelded
surface is a different (and cheaper) thing to time.

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

The parametric-surface groups, whose axis is instead ``u_res = v_res`` at 40 / 160 / 640 (RTX 5090 /
Warp 1.17), against pyvista rather than trimesh:

| case | triwarp-cuda | pyvista |
|---|---|---|
| ``parametric_surface`` (``boy``) | 0.28 / 0.64 / 0.70 ms | 2.5 / 20.4 / 1 271 ms |
| ``parametric_surface`` (``dini``) | 0.25 / 0.65 / 0.84 ms | 2.4 / 16.5 / 1 003 ms |
| ``super_ellipsoid`` (40 / 640) | 0.38 / 33.9 ms | 3.2 / 1 168 ms |
| ``super_toroid`` (40 / 640) | 0.37 / 31.4 ms | 3.2 / 706 ms |
| ``random_hills`` (40 / 640) | 0.42 / 44.3 ms | 3.7 / 1 104 ms |

These used to be the one family in the module that was **not** flat in resolution, because the
slope was host-side: at 640 (409k samples) ``_parametric_lattice``'s NumPy was 51 of the 55 ms
against 4 ms for the launch. The lattice now runs on the device above
``creation._PARAMETRIC_LATTICE_DEVICE_FROM`` samples and the family is flat too -- 42.7 -> 0.70 ms
at 640, **61x**, and 2.26x at 160 -- which takes the gap to VTK from 7-23x to roughly 60-1800x. The
40 column is unchanged because it sits below the gate: the device path costs a flat ~0.66 ms of
launches and readbacks whatever the resolution, so it *loses* 2.4x on a small lattice, and the
dispatch is what keeps both ends. ``super_ellipsoid`` / ``super_toroid`` / ``random_hills`` share
the lattice and move with it at 640; their 40 columns likewise do not.

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

``extrude_polygon`` only ever exercises the *convex* fast path (a single fan), which is why the ear
clipper is benchmarked directly on a non-convex star ring — the same reasoning that puts
``polyline.polyline_simplify`` in its own group in [`test_polyline.py`](test_polyline.py). It is
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

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.utils as p3d_utils
import pyvista as pv
import shapely.geometry as sg
import torch
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchLibrary

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


def _new_cube_pml() -> ml.MeshSet:
    """Build a fresh MeshSet carrying MeshLab's unit cube; every ``create_*`` pushes a mesh."""
    meshset_pml = ml.MeshSet()
    meshset_pml.create_cube(size=1.0)
    return meshset_pml


@pytest.mark.benchmark(group="box")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
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
    if bench_lib.kind == "meshlib":
        # ``makeCube`` takes a size and a **base corner**, not a centre, so a centred box needs
        # ``base = -size / 2``; the same table and the same 12 triangles (tests/test_creation.py).
        faces_ml = bench_lib.run(
            lambda: mm.makeCube(mm.Vector3f(1.0, 2.0, 3.0), mm.Vector3f(-0.5, -1.0, -1.5))
        )
        assert faces_ml.topology.numValidFaces() == 12
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "open3d", "pymeshlab")
@pytest.mark.parametrize(
    ("builder", "filter_name", "n_faces"),
    [
        ("tetrahedron", "create_tetrahedron", 4),
        ("octahedron", "create_octahedron", 8),
        ("dodecahedron", "create_dodecahedron", 36),
        ("icosahedron", "create_icosahedron", 20),
    ],
)
def test_platonic_solids(
    bench_lib: BenchLibrary, builder: str, filter_name: str, n_faces: int
) -> None:
    """
    Constant tables on every side, so this is ``box``'s fixed-cost probe once per solid.

    Coverage is uneven by necessity and the ``builder`` id says which library can answer: MeshLab
    has all four, open3d lacks only the dodecahedron, and **igl and trimesh have only the
    icosahedron** (``igl.icosahedron`` is libigl's one and only generator of this kind), so each
    row skips what its library cannot build rather than substituting a different solid.
    """
    if bench_lib.kind in {"igl", "trimesh"} and builder != "icosahedron":
        pytest.skip(f"neither igl nor trimesh has a {builder}")
    if bench_lib.kind == "open3d":
        if builder == "dodecahedron":
            pytest.skip("open3d has no create_dodecahedron")
        import open3d as o3d

        mesh_o3d = bench_lib.run(getattr(o3d.geometry.TriangleMesh, filter_name))
        assert len(mesh_o3d.triangles) == n_faces
        return
    if bench_lib.kind == "igl":
        vertices_igl, faces_igl = bench_lib.run(igl.icosahedron)
        assert faces_igl.shape == (n_faces, 3)
        assert vertices_igl.shape == (12, 3)
        return
    if bench_lib.kind == "trimesh":
        mesh_tm = bench_lib.run(tm.creation.icosahedron)
        assert mesh_tm.faces.shape == (n_faces, 3)
        return
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
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab")
@pytest.mark.parametrize("count", [32, 512])
def test_grid(bench_lib: BenchLibrary, count: int) -> None:
    """
    A ``count x count`` vertex lattice: two closed-form kernels against two per-vertex loops.

    ``igl.triangulated_grid(nx, ny)`` is the same lattice with the same diagonal, and returns **2D**
    ``(n, 2)`` vertices in the unit square -- so it carries no extents, no centring and no third
    coordinate, which is the class-B transform the parity test applies. That also makes it the
    cheapest of the three: it writes two floats per vertex where the others write three.

    This group is what caught the host build: at 512 it was 78 % NumPy prologue, and the lattice
    is a closed-form parallel map rather than the host-sequential assembly the other templates in
    that module are. Building it on the device is bit-identical and **33x** at 512 (6.77 -> 0.205
    ms), which inverts the row from a 1.84x loss against igl to a large win; ``count=32`` stays a
    wrapper-floor row, gaining only 1.34x. See [`triwarp.creation.grid`][] for the full table.
    """
    n_faces = 2 * (count - 1) ** 2
    if bench_lib.kind == "igl":
        vertices_igl, faces_igl = bench_lib.run(lambda: igl.triangulated_grid(count, count))
        assert faces_igl.shape == (n_faces, 3)
        assert vertices_igl.shape == (count * count, 2)
        return
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
    The concentric-ring lattice: one thread per vertex and one per triangle, both closed forms.

    Flat in resolution on the triwarp side, like the revolution primitives. It was not: the lattice
    was a host-side NumPy loop over ``2 ** subdivisions`` rings and grew quadratically in the ring
    count, so the slope here used to be the prologue's rather than the device's. MeshLab's own
    generator is a per-vertex C++ loop and still carries that slope, which is what widens the gap
    with ``subdivisions``.
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
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab", "pytorch3d")
@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
def test_icosphere(bench_lib: BenchLibrary, subdivisions: int) -> None:
    # No open3d counterpart: create_icosahedron is never subdivided. MeshLab's create_sphere is
    # exactly this scheme, so it is the only reference this group has beyond trimesh.
    #
    # Since the connectivity became closed-form this row is *also* a floor row at every subdivision
    # level it is asked for -- 245 / 253 / 345 us at 3 / 5 / 7 is one launch per level over buffers
    # that reach 164k faces, so the axis reports the wrapper floor rather than the output size.
    if bench_lib.kind == "pytorch3d":
        device = bench_lib.torch_device
        sphere_p3d = bench_lib.run(
            lambda: p3d_utils.ico_sphere(subdivisions, device=torch.device(device))
        )
        assert sphere_p3d.faces_packed().shape[0] == 20 * 4**subdivisions
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "meshlib")
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

    meshlib's ``makeUVSphere`` needs a mapping of its own and a different one: its
    ``verticalResolution`` counts interior latitude **rings**, not profile points, so the match is
    ``makeUVSphere(1, sections, 2 * sections - 2)``. At that pairing it is the same mesh down to the
    vertex -- same counts, a position bijection at 4.7e-07 -- which
    ``tests/test_creation.py::test_uv_sphere_matches_meshlib`` pins for the same reason.

    **pytorch3d**'s ``utils.ico_sphere`` is the same construction from the same base table -- the
    positions correspond one-to-one and agree to 5.8e-05, which is only pytorch3d's table being
    *written* to four decimals (``tests/test_creation.py::test_icosphere_matches_pytorch3d``). It
    is the only reference here that builds on the GPU, and its implementation is the one thing this
    row prices that the others do not: it subdivides **iteratively** through ``SubdivideMeshes``,
    one pass per level, where triwarp's connectivity is closed-form and costs one launch whatever
    the level. So expect its column to grow with the level where triwarp's is flat -- that contrast
    is the point of the row.
    """
    count = (2 * sections, sections // 2)
    if bench_lib.kind == "meshlib":
        mesh_ml = bench_lib.run(lambda: mm.makeUVSphere(1.0, sections, 2 * sections - 2))
        assert mesh_ml.topology.numValidFaces() == 2 * sections * (2 * sections - 2)
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "meshlib")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_cylinder(bench_lib: BenchLibrary, sections: int) -> None:
    """
    A ring table plus two caps, at three resolutions.

    meshlib's ``makeCylinder`` builds the same mesh -- same counts, volume and area
    (``tests/test_creation.py``) -- **base-anchored** at ``z = 0`` where triwarp centres it, which
    is a frame convention rather than a difference in the table. Its radius defaults to 0.1 rather
    than 1, so the row passes it explicitly.
    """
    if bench_lib.kind == "meshlib":
        mesh_ml = bench_lib.run(lambda: mm.makeCylinder(1.0, 2.0, sections))
        assert mesh_ml.topology.numValidFaces() == 4 * sections
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_cone(bench_lib: BenchLibrary, sections: int) -> None:
    """
    A ring, an apex and one cap. meshlib's ``makeCone`` agrees exactly, frame included.

    Unlike its cylinder, this one is base-anchored on *both* sides, so nothing has to be
    reconciled -- same counts, volume, area and bounding box (``tests/test_creation.py``). Its
    radius also defaults to 0.1 rather than 1 and is passed explicitly.
    """
    if bench_lib.kind == "meshlib":
        mesh_ml = bench_lib.run(lambda: mm.makeCone(1.0, 2.0, sections))
        assert mesh_ml.topology.numValidFaces() == 2 * sections
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib", "pytorch3d")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_torus(bench_lib: BenchLibrary, sections: int) -> None:
    """
    Two nested rings. meshlib's ``makeTorus`` matches exactly, counts and frame alike.

    Its resolutions are positional -- ``(primaryRadius, secondaryRadius, primaryResolution,
    secondaryResolution)`` -- so the minor resolution is passed as the 32 the other rows fix.

    **pytorch3d**'s ``utils.torus`` takes the **minor** radius first and its ``sides`` / ``rings``
    are the minor and major loop counts, the reverse of triwarp's ``(major_sections,
    minor_sections)`` -- the mapping is pinned in
    ``tests/test_creation.py::test_torus_matches_pytorch3d``. It builds the vertex table in a
    **Python double loop** and only the tensor conversion is native, so its column is the honest
    cost of that and grows with the section count faster than any other row here; read it as the
    per-vertex Python floor rather than as a GPU row, even though the buffers land on the device.
    """
    if bench_lib.kind == "pytorch3d":
        if sections > 512:
            pytest.skip("pytorch3d builds the torus in a Python double loop over the vertices")
        device = bench_lib.torch_device
        torus_p3d = bench_lib.run(
            lambda: p3d_utils.torus(0.25, 1.0, 32, sections, device=torch.device(device))
        )
        assert torus_p3d.faces_packed().shape[0] == 2 * 32 * sections
        return
    if bench_lib.kind == "meshlib":
        mesh_ml = bench_lib.run(lambda: mm.makeTorus(1.0, 0.25, sections, 32))
        assert mesh_ml.topology.numValidFaces() == 2 * 32 * sections
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "meshlib")
@pytest.mark.parametrize("sections", _SECTIONS)
def test_revolve(bench_lib: BenchLibrary, sections: int) -> None:
    # The shared engine, timed directly on a 64-point profile so the per-slice work dominates the
    # fixed host prologue. No open3d counterpart; meshlib's ``makeSolidOfRevolution`` is the third
    # sweep and builds the identical mesh -- same counts, area and bounding box
    # (``tests/test_creation.py``). Its profile is a ``std_vector_Vector2_float``, filled per point
    # outside the timed callable like every other row's input.
    profile_np = np.column_stack(
        (1.0 + 0.25 * np.cos(np.linspace(0.0, np.pi, 64)), np.linspace(-1.0, 1.0, 64))
    )
    if bench_lib.kind == "meshlib":
        profile_ml = mm.std_vector_Vector2_float()
        for point_np in profile_np:
            profile_ml.append(mm.Vector2f(float(point_np[0]), float(point_np[1])))
        mesh_ml = bench_lib.run(lambda: mm.makeSolidOfRevolution(profile_ml, sections))
        assert mesh_ml.topology.numValidFaces() == 2 * 63 * sections
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "pyvista", "open3d")
@pytest.mark.parametrize("ring_size", [64, 1024])
def test_extrude_polygon(bench_lib: BenchLibrary, ring_size: int) -> None:
    """
    Cap, extrude and wall a closed ring: dominated by the cap triangulation.

    A convex ring takes triwarp's single-fan fast path, which is what makes the ear clipper the
    interesting part of the other rows rather than of this one.

    Three references, and the split between them is **who triangulates the cap**:

    * **trimesh** ``creation.extrude_polygon`` takes a shapely polygon and triangulates it itself,
      so its row includes the cap -- the same work triwarp's does.
    * **pyvista** ``extrude((0, 0, h), capping=True)`` takes a ``PolyData`` whose single polygon
      *cell* is the cap, so VTK triangulates on the way out; ``.triangulate()`` is inside the row
      because without it the result is polygons rather than triangles and the counts are not
      comparable.
    * **open3d** ``extrude_linear`` takes an **already triangulated** disc, so its row is the walls
      alone and is the lower bound of the three. Its faces must be ``Int32``/``Int64`` -- a
      ``UInt32`` tensor raises, although ``RaycastingScene.add_triangles`` accepts one, which is the
      same two-conventions-in-one-API note ``benchmarks/test_ray.py`` records.

    All three land on the same mesh: measured on a 32-gon, **64** vertices and **124** faces from
    every one, watertight with chi = 2 (``tests/test_creation.py``).
    """
    if bench_lib.kind == "pyvista":
        # ``_ring_np`` is the 2-D ring triwarp's ``wp.vec2`` signature takes; both references
        # want 3-D points.
        ring_np = np.column_stack([_ring_np(ring_size), np.zeros(ring_size)])
        polygon_pv = pv.PolyData(ring_np, faces=np.hstack([[ring_size], np.arange(ring_size)]))
        extruded_pv = bench_lib.run(
            lambda: polygon_pv.extrude((0.0, 0.0, 1.0), capping=True).triangulate()
        )
        assert extruded_pv.n_cells == 2 * (ring_size - 2) + 2 * ring_size
        return
    if bench_lib.kind == "open3d":
        import open3d as o3d

        ring_np = np.column_stack([_ring_np(ring_size), np.zeros(ring_size)])
        # Already-triangulated cap: extrude_linear walls a mesh, it does not triangulate a ring.
        fan_np = np.array([[0, i, i + 1] for i in range(1, ring_size - 1)], dtype=np.int32)
        disc_o3d = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.ascontiguousarray(ring_np, dtype=np.float64)),
            o3d.core.Tensor(np.ascontiguousarray(fan_np)),
        )
        extruded_o3d = bench_lib.run(lambda: disc_o3d.extrude_linear([0.0, 0.0, 1.0]))
        assert int(extruded_o3d.triangle.indices.shape[0]) == 2 * (ring_size - 2) + 2 * ring_size
        return
    if bench_lib.kind == "triwarp":
        ring_wp = _ring_wp(ring_size, str(bench_lib.device))
        _, faces_wp = bench_lib.run(lambda: tw.creation.extrude_polygon(ring_wp, 1.0))
        assert int(faces_wp.shape[0]) // 3 == 2 * (ring_size - 2) + 2 * ring_size
    else:
        polygon = sg.Polygon(_ring_np(ring_size))
        mesh_tm = bench_lib.run(lambda: tm.creation.extrude_polygon(polygon, 1.0))
        assert len(mesh_tm.faces) > 0


@pytest.mark.benchmark(group="sweep_polygon")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_sweep_polygon(bench_lib: BenchLibrary) -> None:
    # A long helix so the per-slice frame construction and wall emission dominate the one-off
    # triangulation. No open3d counterpart.
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
        polygon = sg.Polygon(_ring_np(_SWEEP_RING))
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


@pytest.mark.benchmark(group="parametric_surface")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("surface", ["boy", "dini"])
@pytest.mark.parametrize("resolution", [40, 160, 640])
def test_parametric_surface(bench_lib: BenchLibrary, surface: str, resolution: int) -> None:
    """
    Analytic surface evaluation on a ``resolution ** 2`` lattice, against VTK's own generator.

    Two surfaces, one per cost class: ``boy`` is a twisted wrap with two collapsed pole rows, so its
    lattice does the most identification work, and ``dini`` is a plain open patch that does none.
    The axis is the resolution, quadratic in both.

    **The triwarp side is host-bound above ~25k samples, and the module docstring's floor does not
    apply here.** Measured on an RTX 5090: 51 of the 55 ms at ``resolution=640`` is
    ``_parametric_lattice``'s NumPy — a ``meshgrid``, the canonicalising ``where`` chain and one
    ``np.unique`` over ``resolution ** 2`` keys — against 4 ms for the launch that evaluates the
    map. Read a change in this group as a change to the host prologue unless the sample count is
    small. VTK evaluates its map in a per-point C++ loop and then *welds by distance*, which is the
    part triwarp does combinatorially and for free.
    """
    if bench_lib.kind == "pyvista":
        name = "ParametricBoy" if surface == "boy" else "ParametricDini"
        mesh_pv = bench_lib.run(
            lambda: getattr(pv, name)(u_res=resolution, v_res=resolution, clean=True)
        )
        assert mesh_pv.n_faces > 0
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(
        lambda: tw.creation.parametric_surface(surface, resolution, resolution, device=device)  # type: ignore[arg-type]
    )
    assert int(faces_wp.shape[0]) // 3 > 0


@pytest.mark.benchmark(group="super_ellipsoid")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("resolution", [40, 640])
def test_super_ellipsoid(bench_lib: BenchLibrary, resolution: int) -> None:
    """The superquadric sphere: the same lattice plus two signed powers per coordinate."""
    if bench_lib.kind == "pyvista":
        mesh_pv = bench_lib.run(
            lambda: pv.ParametricSuperEllipsoid(u_res=resolution, v_res=resolution, clean=True)
        )
        assert mesh_pv.n_faces > 0
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(
        lambda: tw.creation.super_ellipsoid(
            u_resolution=resolution, v_resolution=resolution, device=device
        )
    )
    assert int(faces_wp.shape[0]) // 3 > 0


@pytest.mark.benchmark(group="super_toroid")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("resolution", [40, 640])
def test_super_toroid(bench_lib: BenchLibrary, resolution: int) -> None:
    """The superquadric torus: both directions wrap, so no cell is dropped at any resolution."""
    if bench_lib.kind == "pyvista":
        mesh_pv = bench_lib.run(
            lambda: pv.ParametricSuperToroid(u_res=resolution, v_res=resolution, clean=True)
        )
        assert mesh_pv.n_faces > 0
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(
        lambda: tw.creation.super_toroid(
            u_resolution=resolution, v_resolution=resolution, device=device
        )
    )
    assert int(faces_wp.shape[0]) // 3 > 0


@pytest.mark.noparity(
    "pyvista",
    reason="D5 stochastic with no shared invariant: VTK draws each hill's amplitude and both "
    "variances from its own generator and offsets them on a coarse internal grid, where "
    "random_hills takes all three as parameters and draws only the centres, through NumPy's "
    "stream rather than VTK's. So no seed pairs the two height fields and only the lattice, the "
    "domain and the cost are comparable -- the first two are asserted in "
    "tests/test_creation.py::test_random_hills without needing VTK.",
)
@pytest.mark.benchmark(group="random_hills")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("resolution", [40, 640])
def test_random_hills(bench_lib: BenchLibrary, resolution: int) -> None:
    """The height field: the plain grid lattice, plus a 30-term sum per vertex on the device."""
    if bench_lib.kind == "pyvista":
        mesh_pv = bench_lib.run(
            lambda: pv.ParametricRandomHills(u_res=resolution, v_res=resolution, clean=True)
        )
        assert mesh_pv.n_faces > 0
        return
    device = bench_lib.device
    _, faces_wp = bench_lib.run(
        lambda: tw.creation.random_hills(
            seed=0, u_resolution=resolution, v_resolution=resolution, device=device
        )
    )
    assert int(faces_wp.shape[0]) // 3 > 0


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
