"""
Benchmarks for ``triwarp.mesh.Trimesh``: what a cold cached property costs.

Axis: **cache state**, which is not a mesh property at all. ``Trimesh`` is a lazily-computed
container -- first access to ``warp_mesh`` or ``face_adjacency`` or ``boundary_loops`` runs the real
work, every access after that is a dict lookup. So the honest question about this class is not "how
does it scale" (each property scales the way its own module's benchmark says it does) but "which
properties are expensive enough that a caller needs to know they are being computed", and the
warm/cold gap is the only way to see that.

Each group therefore times two rows on the same mesh:

* **cold** -- a freshly constructed ``Trimesh``, built *inside* the timed callable, so every round
  pays the full computation. Construction itself is two array assignments and is negligible.
* **warm** -- one shared instance whose property was already forced, so the row measures the dict
  lookup and nothing else. It exists as the floor: a warm row that is not ~0 means the property is
  not actually being cached.

The ratio between them is what a caller saves by holding onto the mesh, and it is the number to
quote when deciding whether an API should take a ``Trimesh`` or raw buffers.

Cache invalidation is a third row where it applies. ``with_vertices`` keeps every topology cache and
only drops the geometric ones, ``with_faces`` drops everything; that distinction is the reason
``with_vertices`` exists, so it is measured rather than asserted.

Properties chosen
-----------------
The five that are expensive for genuinely different reasons: ``warp_mesh`` (a BVH build),
``vertex_normals`` (an atomic scatter), ``face_adjacency`` (an edge sort plus manifold-pair
grouping), ``boundary_loops`` (pointer-jumping list ranking plus per-loop host work) and
``is_watertight`` (which composes an edge test with a self-intersection pass over a fresh BVH, and
is by far the priciest). Timing all of them would just re-run the rest of the suite through a
different door.

References
----------
**trimesh**'s ``Trimesh`` is the model this class mirrors, and it caches the same way, so the cold
rows are directly comparable -- the trimesh side is likewise rebuilt inside the timed callable. It
has no ``with_vertices`` / ``with_faces`` equivalent (assigning to ``mesh.vertices`` invalidates
everything), so the invalidation group is triwarp-only. open3d and libigl have no caching container
at all: open3d recomputes on request and libigl is free functions over raw arrays.
"""

from __future__ import annotations

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# ``is_watertight`` composes a self-intersection pass over a fresh BVH; a few rounds is enough.
_ROUNDS = 3

_warm_cache: dict[tuple[str, str], tw.Trimesh] = {}


def _cold(bench_case: BenchCase) -> tw.Trimesh:
    """Construct a fresh ``Trimesh`` with an empty cache -- two array assignments, no work."""
    return tw.Trimesh(bench_case.vertices_wp, bench_case.faces_wp)


def _warm(bench_case: BenchCase, name: str) -> tw.Trimesh:
    """Return a shared ``Trimesh`` with ``name`` already forced into its cache."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _warm_cache:
        _warm_cache[key] = _cold(bench_case)
    mesh = _warm_cache[key]
    getattr(mesh, name)
    return mesh


def _time_property(bench_case: BenchCase, name: str, *, warm: bool, rounds: int = 10) -> None:
    """Time one cached property, either from a fresh mesh or from a pre-forced shared one."""
    if warm:
        mesh = _warm(bench_case, name)
        assert bench_case.run(lambda: getattr(mesh, name), rounds=rounds) is not None
    else:
        assert bench_case.run(lambda: getattr(_cold(bench_case), name), rounds=rounds) is not None


@pytest.mark.benchmark(group="mesh_warp_mesh")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_warp_mesh(bench_case: BenchCase, warm: bool) -> None:
    """``warp_mesh``: a BVH build over all faces, or a dict lookup."""
    skip_larger_than(bench_case, "happy_buddha")
    _time_property(bench_case, "warp_mesh", warm=warm)


@pytest.mark.benchmark(group="mesh_vertex_normals")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_vertex_normals(bench_case: BenchCase, warm: bool) -> None:
    """``vertex_normals``: face normals and areas, then an atomic scatter into the vertices."""
    if bench_case.kind == "triwarp":
        _time_property(bench_case, "vertex_normals", warm=warm)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if warm:
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        assert mesh_tm.vertex_normals is not None
        assert bench_case.run(lambda: mesh_tm.vertex_normals) is not None
    else:
        assert (
            bench_case.run(lambda: tm.Trimesh(vertices_np, faces_np, process=False).vertex_normals)
            is not None
        )


@pytest.mark.benchmark(group="mesh_face_adjacency")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_face_adjacency(bench_case: BenchCase, warm: bool) -> None:
    """``face_adjacency``: a radix sort over ``3F`` edge keys plus manifold-pair grouping."""
    if bench_case.kind == "triwarp":
        _time_property(bench_case, "face_adjacency", warm=warm)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if warm:
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        assert mesh_tm.face_adjacency is not None
        assert bench_case.run(lambda: mesh_tm.face_adjacency) is not None
    else:
        assert (
            bench_case.run(lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency)
            is not None
        )


@pytest.mark.benchmark(group="mesh_boundary_loops")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_boundary_loops(bench_case: BenchCase, warm: bool) -> None:
    """
    ``boundary_loops`` through the container, on the loop axis.

    The one property whose cold cost depends on something other than size, so it gets the ``loops``
    axis rather than the scan sweep -- six rows that show the caching gap and the loop-count gap at
    once. [`test_boundary.py`](test_boundary.py) measures the underlying function.
    """
    _time_property(bench_case, "boundary_loops", warm=warm)


@pytest.mark.benchmark(group="mesh_is_watertight")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_is_watertight(bench_case: BenchCase, warm: bool) -> None:
    """``is_watertight``: the priciest property, since it builds a BVH for self-intersection."""
    skip_larger_than(bench_case, "dragon", "the self-intersection pass dominates beyond dragon")
    _time_property(bench_case, "is_watertight", warm=warm, rounds=_ROUNDS)


@pytest.mark.benchmark(group="mesh_invalidation")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("mode", ["with_vertices", "with_faces"], ids=["keepstopo", "dropall"])
def test_invalidation(bench_case: BenchCase, mode: str) -> None:
    """
    Rebuilding ``face_adjacency`` after a geometry change against after a topology change.

    ``with_vertices`` keeps the topology caches, so the adjacency should come back free;
    ``with_faces`` drops everything and pays the edge sort again. That gap is the entire reason
    ``with_vertices`` exists as a separate method, so it is worth a measurement rather than a
    comment.
    """
    skip_larger_than(bench_case, "happy_buddha")
    mesh = _warm(bench_case, "face_adjacency")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    def run() -> object:
        derived = (
            mesh.with_vertices(vertices) if mode == "with_vertices" else mesh.with_faces(faces)
        )
        return derived.face_adjacency

    assert bench_case.run(run) is not None
