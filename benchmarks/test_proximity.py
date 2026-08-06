"""
Benchmarks for ``triwarp.proximity`` hot paths.

Covers winding number, signed distance, tangent spheres and geodesic-ball queries. The AABB
reduction moved to [`test_bounds.py`](test_bounds.py), where ``triwarp.bounds`` lives.
``winding_number`` is O(n_queries x n_faces) even in the tiled variant, so ``lucy`` is skipped;
the pinned serial (``tiled=False``) path is additionally capped at ``bunny`` because one thread
per query walking every face takes minutes beyond that.

Read ``winding_number`` and ``signed_distance_on_mesh[winding]`` together: both answer an
inside/outside question from solid angle, but the first accumulates it exactly over every face while
the second lets Warp's BVH traversal approximate it and keeps only the sign. The gap between them is
the cost of needing the winding *value* rather than just its sign.

Open3D has no equivalent for anything in this module: it has no
generalized winding number — its inside/outside test is raycasting-based
(``RaycastingScene.compute_occupancy``), a different algorithm answering a coarser question — and no
tangent-sphere, local-thickness or geodesic-ball query at all.

``signed_distance_on_mesh`` has **two** references, and they are the only two that exist: libigl's
``igl.signed_distance``, whose ``sign_type`` axis maps onto triwarp's ``sign_mode`` one-for-one, and
pymeshlab's, whose sign rule is a third algorithm and therefore appears once. See
``test_signed_distance_on_mesh`` for the igl mapping and its numbers.

**One thing libigl's signed distance does that no assert may ignore:** for both winding-based sign
types it returns ``(1 - 2w) * d`` rather than ``sign(1 - 2w) * d``, with ``w`` the *continuous*
winding number. So its magnitude is only ``|d|`` where ``w`` is exactly 0 or 1, and near the surface
it is scaled down — measured on ``bunny_decimated``, ``|S|`` deviates from the pseudonormal type's
by **2.7e-2 of the bbox diagonal** for both ``WINDING_NUMBER`` and ``FAST_WINDING_NUMBER``, while
the pseudonormal type agrees with triwarp to **8e-8**. That is why the parity oracle in
``tests/test_proximity.py`` is the pseudonormal type and why the winding row here is a *cost*
comparison only.

**And the Barnes-Hut approximation does pay**, which is worth recording because a 5 000-query probe
had suggested otherwise. At this module's 10 000 queries on ``bunny``, ``igl.fast_winding_number``
is **76.5 ms against ``igl.winding_number``'s 272.5** (25.1 against 65.5 on ``bunny_decimated``) —
3.6x and 2.6x — for a maximum winding deviation of 0.004. It is not a row of its own because triwarp
exposes no approximate-winding entry point to put on the other side of it (``winding_number`` is the
exact sum; the Barnes-Hut walk exists only inside
``signed_distance_on_mesh(sign_mode="winding")``), but it is the number to weigh a fast-winding port
against.

``compute_scalar_by_distance_from_another_mesh_per_vertex(signeddist=True)`` (MeshLab's Distance
from Reference Mesh) measures every vertex of one mesh against another, so the query points go in as
a second, face-less mesh and the answer comes back on their vertex scalar attribute. Three things to
read its row against:

- **Its sign is a third algorithm.** MeshLab takes the dot product with the reference normal at the
  closest point — neither triwarp's 5-ray parity test nor its Barnes-Hut winding accumulation. So it
  appears once rather than twice, in the ``parity`` row.
- **Its per-query cost grows with the reference mesh.** At a fixed 10 000 queries it costs
  **206 / 627 / 7 678 ms** on bunny_decimated / bunny / dragon — 20, 63 and 768 µs per query — while
  being cleanly linear in the query count at a fixed mesh (69 ms at 1 k, 642 at 10 k, 6 125 at 100 k
  on bunny). A closest-point query that is *not* sublinear in the face count is the opposite of what
  triwarp's BVH does — measured **95x** and **235x** against it on bunny_decimated and bunny, a
  ratio that widens with the mesh — which is why it is capped at ``bunny`` with ``rounds=3`` rather
  than allowed to spend 85 s on dragon.
- It writes only the vertex scalar, so the two-mesh MeshSet is built once and shared.

``shape_diameter`` is the one group in this module whose reference is *not* faster than a
millisecond and not close either: ``compute_scalar_by_shape_diameter_function_per_vertex`` is the
most expensive per-vertex filter MeshLab ships (674 ms on ``bunny``), because it traces 64 rays from
every vertex on one core. That is exactly the shape of work a GPU should win outright, which is why
the port exists. Two caveats for reading its row: its ``cone_amplitude`` parameter is a **no-op** in
the 2025.07 build (byte-identical output at 90 and 120 degrees), so its cone is whatever it is, and
it writes only the vertex scalar, so the MeshSet is shared. The ``rays`` sweep is the axis, not the
mesh: the cost is exactly linear in it on both sides, and the pair pins that.
"""

from __future__ import annotations

from typing import Literal

import igl
import numpy as np
import pymeshlab as ml
import pytest
import warp as wp
from conftest import BenchCase, BenchLibrary, skip_larger_than
from scipy.spatial import Delaunay

import triwarp as tw

_QUERY_SEED = 42
_N_QUERIES = 10_000

# Query counts for ``winding_number``, the one genuinely O(queries x faces) function here: the
# other half of its product, swept independently of the mesh.
_N_QUERIES_SWEEP = [10_000, 100_000]

_query_cache: dict[tuple[str, str, int], wp.array] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}
_pml_distance_cache: dict[tuple[str, str], ml.MeshSet] = {}


def _query_points_np(bench_case: BenchCase, count: int = _N_QUERIES) -> np.ndarray:
    """Query points: subsampled vertices jittered by 10% of the bbox diagonal."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    idx = rng.integers(0, vertices.shape[0], size=count)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return vertices[idx] + rng.normal(scale=0.1 * diagonal, size=(count, 3))


def _query_points_wp(bench_case: BenchCase, count: int = _N_QUERIES) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device), count)
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_query_points_np(bench_case, count), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _query_cache[key]


def _mesh_wp(bench_case: BenchCase) -> wp.Mesh:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mesh_cache:
        _mesh_cache[key] = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    return _mesh_cache[key]


@pytest.mark.benchmark(group="winding_number")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.parametrize("n_queries", _N_QUERIES_SWEEP)
def test_winding_number(bench_case: BenchCase, n_queries: int) -> None:
    """
    Exact winding number: no BVH, every query sums over every face.

    The one genuinely ``O(queries x faces)`` function in the module, so both sizes are swept --
    the mesh by the registry and the query count here. A 10x step in queries that is not a 10x
    step in time would mean the launch is not saturating the device.
    """
    skip_larger_than(bench_case, "happy_buddha", "O(queries x faces): lucy is untenable")
    if n_queries > _N_QUERIES:
        # 100k queries against dragon is 8.7e10 pair evaluations, and against happy_buddha 1.1e11.
        # The 10x query step is measurable on the medium meshes and the product is what it says.
        skip_larger_than(bench_case, "bunny", "the wide query sweep is only tenable up to bunny")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points = _query_points_wp(bench_case, n_queries)
        result = bench_case.run(lambda: tw.proximity.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)
    else:  # igl exact generalized winding number
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        points = _query_points_np(bench_case, n_queries)
        result = bench_case.run(lambda: igl.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)


@pytest.mark.benchmark(group="winding_number_serial")
@pytest.mark.benchlibs("triwarp")
def test_winding_number_serial(bench_case: BenchCase) -> None:
    """Pinned ``tiled=False`` reference path: one thread per query loops over every face."""
    skip_larger_than(bench_case, "bunny", "serial winding takes minutes beyond bunny")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    result = bench_case.run(
        lambda: tw.proximity.winding_number(vertices, faces, points, tiled=False)
    )
    assert result.shape == (_N_QUERIES,)


def _distance_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the reference mesh at id 0 and the query points as a face-less mesh at id 1, once.

    The filter writes only mesh 1's vertex scalar attribute and leaves both geometries alone, so
    sharing is sound (verified: repeated calls return bit-identical scalars at the same cost).
    """
    key = (bench_case.mesh_name, "pml")
    if key not in _pml_distance_cache:
        meshset_pml = bench_case.new_meshset_pml()
        meshset_pml.add_mesh(
            ml.Mesh(vertex_matrix=np.ascontiguousarray(_query_points_np(bench_case)))
        )
        _pml_distance_cache[key] = meshset_pml
    return _pml_distance_cache[key]


@pytest.mark.benchmark(group="signed_distance_on_mesh")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab")
@pytest.mark.parametrize("sign_mode", ["parity", "winding"])
def test_signed_distance_on_mesh(
    bench_case: BenchCase, sign_mode: Literal["parity", "winding"]
) -> None:
    """
    The two sign modes, on the same closest-point query.

    ``"winding"`` walks the BVH accumulating solid angle (Barnes-Hut, ``accuracy=2.0``) instead of
    casting 5 perturbed parity rays, and needs a ``wp.Mesh`` carrying the per-node solid-angle
    expansion — so the ``wp.Mesh`` build inside the timed region differs between the two, which is
    intentional: it is part of what the mode costs. Both include that build because
    ``signed_distance_on_mesh`` constructs its own mesh (it takes vertex/face arrays, not a
    ``wp.Mesh``), so there is no way for a caller to hoist it.

    **libigl is the only reference whose sign axis maps onto triwarp's**, which is why it appears
    twice where pymeshlab appears once: ``SIGNED_DISTANCE_TYPE_PSEUDONORMAL`` against ``"parity"``
    and ``SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER`` against ``"winding"`` — the second is the same
    Barnes-Hut family triwarp's mode is. Its AABB tree is built per call, as triwarp's ``wp.Mesh``
    is, and it gets ``rounds=3`` like the pymeshlab row.

    In-harness medians at 10 000 queries, igl rows run in isolation: **64 / 150 ms on
    ``bunny_decimated`` and 394 / 520 on ``bunny``** against triwarp's 6.3 / 6.6 and 4.3 / 4.0 — so
    10-100x, and note that **the mode ratio disagrees between the two sides**: igl's winding sign
    costs 2.3x its pseudonormal one where triwarp's two modes are within 1.3x of each other,
    because the solid-angle walk rides the BVH traversal triwarp is already doing. Read igl's
    *medians* here, not its minima: the pseudonormal row spreads 225-399 ms on ``bunny``.
    """
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl rebuilds a single-threaded AABB tree per call")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        points_np = _query_points_np(bench_case)
        sign_type = (
            igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            if sign_mode == "parity"
            else igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
        )
        distance_igl, _, _, _ = bench_case.run(
            lambda: igl.signed_distance(points_np, vertices_np, faces_np, sign_type), rounds=3
        )
        assert distance_igl.shape == (_N_QUERIES,)
        return

    if bench_case.kind == "pymeshlab":
        if sign_mode != "parity":
            pytest.skip("MeshLab signs by the closest-point normal: a third mode, so one row only")
        skip_larger_than(bench_case, "bunny", "768 us per query on dragon: 85 s for one row")
        meshset_pml = _distance_meshset_pml(bench_case)
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_distance_from_another_mesh_per_vertex(
                measuremesh=1, refmesh=0, signeddist=True
            ),
            rounds=3,
        )
        assert meshset_pml.mesh(1).vertex_scalar_array().shape == (_N_QUERIES,)
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    distance = bench_case.run(
        lambda: tw.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode=sign_mode)
    )
    assert distance.shape == (_N_QUERIES,)


_LATTICE_ROWS = [26, 80, 240]
_lattice_cache: dict[int, tuple[np.ndarray, Delaunay, np.ndarray]] = {}


def _triangular_lattice_2d(rows: int) -> tuple[np.ndarray, Delaunay, np.ndarray]:
    """
    Build lattice points, their Delaunay triangulation and a query cloud overhanging it by 10%.

    Cached per size: the triangulation is the benchmark's *input*, so building it inside the timed
    region would price scipy's Delaunay rather than either library's point location.
    """
    if rows not in _lattice_cache:
        cols = rows + 4
        height = np.sqrt(3.0) / 2.0
        points_np = np.array(
            [(col + 0.5 * (row % 2), row * height) for row in range(rows) for col in range(cols)],
            dtype=np.float64,
        )
        triangulation_sp = Delaunay(points_np)
        extent_np = points_np.max(axis=0) - points_np.min(axis=0)
        rng = np.random.default_rng(4)
        queries_np = (
            points_np.min(axis=0) - 0.05 * extent_np + rng.random((_N_QUERIES, 2)) * 1.1 * extent_np
        )
        _lattice_cache[rows] = (points_np, triangulation_sp, queries_np)
    return _lattice_cache[rows]


@pytest.mark.benchmark(group="containing_faces_2d")
@pytest.mark.benchlibs("triwarp", "scipy")
@pytest.mark.parametrize("rows", _LATTICE_ROWS, ids=["lattice26", "lattice80", "lattice240"])
def test_containing_faces_2d(bench_lib: BenchLibrary, rows: int) -> None:
    """
    Point location in a planar triangulation: BVH candidate, then a barycentric sign test.

    The only group in this module with **no input mesh** -- a 2D triangulation is not one of the
    registry's surfaces -- so it takes ``bench_lib`` and builds its own: a triangular lattice
    whose Delaunay triangulation is equilateral throughout, swept over three sizes. The lattice
    is not arbitrary. Any Delaunay of a bounded point set has needle triangles on its hull, and
    a query within float32 noise of two needles' shared edge resolves to neither, so a random-
    point triangulation would make this row's own correctness assert flaky for a reason that has
    nothing to do with cost.

    **The comparison is deliberately unfavourable to triwarp.**
    ``scipy.spatial.Delaunay.find_simplex`` is timed on a triangulation built *outside* the
    timed region, while triwarp's row includes building its BVH on every call -- there is no
    prebuilt-index entry point on this side. Read the row as a floor on triwarp's margin, not as
    a like-for-like split; scipy is doing strictly less work per call.
    """
    points_np, triangulation_sp, queries_np = _triangular_lattice_2d(rows)
    if bench_lib.kind == "scipy":
        located_sp = bench_lib.run(lambda: triangulation_sp.find_simplex(queries_np))
        assert located_sp.shape == (_N_QUERIES,)
        assert (located_sp >= 0).any()
        return

    device = bench_lib.device
    vertices_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32).ravel(),
        dtype=wp.int32,
        device=device,
    )
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    located_wp = bench_lib.run(
        lambda: tw.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp)
    )
    # Exact on this lattice, which is why the fixture is a lattice; see the docstring.
    assert np.array_equal(located_wp.numpy(), triangulation_sp.find_simplex(queries_np))
