"""
Benchmarks for ``triwarp.selection``: submesh extraction and vertex-selection morphology.

Two axes, and neither is the face count:

* **hops**, for the morphology pair. ``expand_vertex_mask`` / ``shrink_vertex_mask`` run a host loop
  of one full edge pass per hop, so their cost is ``hops x |E|`` -- completely independent of how
  many vertices are actually selected. That makes them the cleanest pure-iteration-count group in
  the suite: the slope between 1 and 8 hops should be exactly 8, and anything else means the pass
  is not doing constant work per hop.
* **components**, for ``exclude_fully_selected_components``. It runs a connected-components pass and
  then an all-selected test per component, so it inherits the component-count sensitivity that
  [`test_combine.py`](test_combine.py) documents for ``split``.

``submesh_from_face_indices`` gets a ``unique_indices`` sweep instead. That flag skips a dedup sort
the caller can promise is unnecessary, and ``combine.split`` relies on the fast path -- so the gap
between the two rows is exactly what ``split`` saves per component, which is worth knowing given
that ``split``'s per-component host sequence is the package's largest single slowness path.

References
----------
**trimesh**'s ``Trimesh.submesh`` is the reference for the extraction group; it takes a sequence of
face-index groups and returns a list of meshes, so it is given a single group to match triwarp's
single submesh.

Neither trimesh nor open3d nor libigl has selection *morphology* -- growing or shrinking a mask
across the mesh graph. **pymeshlab** does, and is the only reference the morphology pair has:
``apply_selection_dilatation`` / ``apply_selection_erosion``, MeshLab's Dilate / Erode Selection.

Two differences to read the rows against, neither of them correctable:

- **It is face morphology, not vertex morphology.** MeshLab dilates the selected *face* set (via
  VCGlib's loose vertex-from-face / face-from-vertex pair), so seeding it needs
  ``compute_selection_by_condition_per_face`` and a vertex selection handed to it is simply cleared.
  triwarp grows a vertex mask over the unique-edge table. Same operation class, same asymptotic work
  -- one full pass over the elements per hop -- on a different element type.
- **One filter call is one hop**, so the reference is a host loop of ``hops`` calls, which is
  structurally what ``expand_vertex_mask``'s own per-hop launch loop does.

The MeshSet is *shared* here rather than rebuilt per round, which is the exception to the rule in
``BenchCase.new_meshset_pml``: these filters touch only the selected bit, and their cost does not
depend on how much is selected (measured flat at 0.86-1.21 ms over 120 consecutive dilatations
carrying ``sphere_med`` from 0.9% to 86% selected). Rebuilding instead would put a 22 ms MeshSet
build on a 1 ms filter and flatten the slope this group exists to measure. It reads
**1.03 -> 7.45 ms** for 1 -> 8 dilatations and **1.19 -> 9.54 ms** for 1 -> 8 erosions -- slopes of
7.3x and 8.0x, i.e. the reference confirms independently that a hop is constant work. triwarp runs
0.061 -> 0.30 ms and 0.13 -> 0.39 ms against it, so 17x and 9x at one hop and 25x at eight.

trimesh's ``graph.connected_component_labels`` could reproduce
``exclude_fully_selected_components`` in several steps, but not as one call, so that group remains a
before/after self-comparison.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from conftest import BenchCase
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw

_SEED = 5

# Hop counts for the morphology pair: cost is hops x edges, so the slope should be exactly 8.
_HOPS = [1, 8]

# Fraction of vertices seeded into the mask before growing it.
_SEED_FRACTION = 0.01

# Fraction of faces extracted into the submesh.
_SUBMESH_FRACTION = 0.5

_mask_cache: dict[tuple[str, str], wp.array] = {}
_edges_cache: dict[tuple[str, str], wp.array] = {}
_indices_cache: dict[tuple[str, str], tuple] = {}


def _seed_mask(bench_case: BenchCase) -> wp.array[wp.bool]:
    """Sparse boolean vertex mask -- 1% of the vertices, at a fixed seed."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mask_cache:
        rng = np.random.default_rng(_SEED)
        n = bench_case.n_vertices
        mask_np = np.zeros(n, dtype=bool)
        mask_np[rng.choice(n, size=max(1, int(n * _SEED_FRACTION)), replace=False)] = True
        _mask_cache[key] = wp.array(mask_np, dtype=wp.bool, device=bench_case.device)
    return _mask_cache[key]


def _seed_mask_np(bench_case: BenchCase) -> np.ndarray:
    """Build the same 1% seed as [`_seed_mask`], as a host bool array for the meshlib rows."""
    rng = np.random.default_rng(_SEED)
    n = bench_case.n_vertices
    mask_np = np.zeros(n, dtype=bool)
    mask_np[rng.choice(n, size=max(1, int(n * _SEED_FRACTION)), replace=False)] = True
    return mask_np


def _unique_edges(bench_case: BenchCase) -> wp.array:
    """Build the unique edge table once: an *input*, so morphology never re-times the sort."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edges_cache:
        _edges_cache[key] = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )[0]
    return _edges_cache[key]


def _seeded_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Return the shared MeshSet with a small face selection already established.

    Seeded *outside* the timed callable, which the module docstring justifies: MeshLab's dilate and
    erode cost the same whatever fraction is selected, so what the rounds start from does not change
    what they measure. The condition picks a polar cap rather than the random 1% ``_seed_mask``
    uses -- MeshLab has no way to set a selection array directly, only to derive one.
    """
    meshset_pml = bench_case.meshset_pml
    meshset_pml.compute_selection_by_condition_per_face(condselect="(z0 > 0.98)")
    return meshset_pml


@pytest.mark.benchmark(group="expand_vertex_mask")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
@pytest.mark.parametrize("hops", _HOPS)
def test_expand_vertex_mask(bench_case: BenchCase, hops: int) -> None:
    """
    One full edge pass per hop, regardless of selection size: the slope should be exactly 8.

    meshlib's ``expand`` takes the hop count itself and dilates a ``VertBitSet`` over the *same*
    vertex neighbourhood, agreeing element for element (``tests/test_selection.py``), so unlike the
    pymeshlab row it needs no face round trip. It mutates the bitset in place, so a fresh one is
    built inside the timed callable from the shared seed -- the bitset build is a bit-pack of the
    seed array and is the row's floor.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        seed_np = np.ascontiguousarray(_seed_mask_np(bench_case))

        def expand_ml() -> int:
            region_ml = mn.vertBitSetFromBools(seed_np)
            mm.expand(mesh_ml.topology, region_ml, hops)
            return region_ml.count()

        assert bench_case.run(expand_ml) > 0
        return
    if bench_case.kind == "triwarp":
        faces, mask = bench_case.faces_wp, _seed_mask(bench_case)
        edges = _unique_edges(bench_case)
        grown = bench_case.run(
            lambda: tw.selection.expand_vertex_mask(faces, mask, hops, unique_edges=edges)
        )
        assert grown.shape == mask.shape
    else:  # one Dilate Selection call per hop, on the face set
        meshset_pml = _seeded_meshset_pml(bench_case)

        def dilate_pml() -> int:
            for _ in range(hops):
                meshset_pml.apply_selection_dilatation()
            return meshset_pml.current_mesh().selected_face_number()

        assert bench_case.run(dilate_pml) > 0


@pytest.mark.noparity(
    "pymeshlab",
    reason="D4: MeshLab's Erode Selection removes a face when *any* of its vertices is on the "
    "selection boundary, so reading the vertex selection back gives the vertices of the "
    "surviving faces, not an eroded vertex set. Measured on a 9x9 grid: 51 / 39 / 25 vertices "
    "after 1 / 2 / 3 erosions where shrink_vertex_mask gives 19 / 7 / 1. Dilation does map and is "
    "covered; erosion is a different operation, and meshlib's shrink -- which erodes a VertBitSet "
    "by one-ring layers and agrees element for element -- is the oracle for this function, in "
    "tests/test_selection.py::test_expand_and_shrink_vertex_mask_match_meshlib.",
)
@pytest.mark.benchmark(group="shrink_vertex_mask")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
@pytest.mark.parametrize("hops", _HOPS)
def test_shrink_vertex_mask(bench_case: BenchCase, hops: int) -> None:
    """
    The erosion counterpart, on the same input: should match ``expand`` row for row.

    meshlib is the reference the exemption above says this group lacked: ``shrink`` erodes a
    ``VertBitSet`` by one-ring layers, which is triwarp's operation and not MeshLab's face-based
    one, and the two agree element for element at every hop count. The mask is pre-grown outside
    the timed callable on both sides, as the triwarp row does, so the rounds erode the same set.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        seed_np = np.ascontiguousarray(_seed_mask_np(bench_case))
        grown_ml = mn.vertBitSetFromBools(seed_np)
        mm.expand(mesh_ml.topology, grown_ml, max(_HOPS))
        grown_np = np.ascontiguousarray(mn.getNumpyBitSet(grown_ml))

        def shrink_ml() -> int:
            region_ml = mn.vertBitSetFromBools(grown_np)
            mm.shrink(mesh_ml.topology, region_ml, hops)
            return region_ml.count()

        assert bench_case.run(shrink_ml) >= 0
        return
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        edges = _unique_edges(bench_case)
        # Grow first so there is something left to erode after 8 hops.
        mask = tw.selection.expand_vertex_mask(
            faces, _seed_mask(bench_case), max(_HOPS), unique_edges=edges
        )
        shrunk = bench_case.run(
            lambda: tw.selection.shrink_vertex_mask(faces, mask, hops, unique_edges=edges)
        )
        assert shrunk.shape == mask.shape
    else:  # dilate well past the erosion depth first, so there is something left to erode
        meshset_pml = _seeded_meshset_pml(bench_case)
        for _ in range(2 * max(_HOPS)):
            meshset_pml.apply_selection_dilatation()

        def erode_pml() -> None:
            for _ in range(hops):
                meshset_pml.apply_selection_erosion()

        bench_case.run(erode_pml)


def _face_indices(bench_case: BenchCase) -> tuple:
    """Half the faces as a ``(indices_wp, indices_np)`` pair, already unique and sorted."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _indices_cache:
        count = max(1, int(bench_case.n_faces * _SUBMESH_FRACTION))
        indices_np = np.arange(count, dtype=np.int32)
        _indices_cache[key] = (
            wp.array(indices_np, dtype=wp.int32, device=bench_case.device),
            indices_np,
        )
    return _indices_cache[key]


@pytest.mark.benchmark(group="submesh_from_face_indices")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("unique_indices", [False, True], ids=["dedup", "presorted"])
def test_submesh_from_face_indices(bench_case: BenchCase, unique_indices: bool) -> None:
    """
    Face gather plus a vertex remap, with and without the dedup sort.

    ``unique_indices=True`` is the promise ``combine.split`` makes on every component, so the gap
    between these two rows is what the fast path buys there -- and ``split``'s per-component host
    sequence is the largest single slowness path the axis set has surfaced.
    """
    if bench_case.kind == "triwarp":
        indices_wp, _indices_np = _face_indices(bench_case)
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _sub_vertices, sub_faces = bench_case.run(
            lambda: tw.selection.submesh_from_face_indices(
                vertices, faces, indices_wp, unique_indices=unique_indices
            )
        )
        assert int(sub_faces.shape[0]) == 3 * int(indices_wp.shape[0])
    else:
        if unique_indices:
            pytest.skip("trimesh's submesh has no presorted fast path to compare against")
        _indices_wp, indices_np = _face_indices(bench_case)
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        parts = bench_case.run(lambda: mesh_tm.submesh([indices_np], append=False))
        assert len(parts[0].faces) == indices_np.shape[0]


@pytest.mark.benchmark(group="exclude_fully_selected_components")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp")
def test_exclude_fully_selected_components(bench_case: BenchCase) -> None:
    """A components pass plus an all-selected reduce per component: inherits ``split``'s axis."""
    faces, n_vertices = bench_case.faces_wp, bench_case.n_vertices
    edges = _unique_edges(bench_case)
    mask = _seed_mask(bench_case)
    kept = bench_case.run(
        lambda: tw.selection.exclude_fully_selected_components(
            faces, mask, n_vertices, unique_edges=edges
        )
    )
    assert kept.shape == mask.shape


_FACE_REGION_FRACTION = 0.25


def _seed_face_region(bench_case: BenchCase) -> np.ndarray:
    """Mask a contiguous quarter of the face buffer as the region -- one seam, not a scatter."""
    mask_np = np.zeros(bench_case.n_faces, dtype=bool)
    mask_np[: int(bench_case.n_faces * _FACE_REGION_FRACTION)] = True
    return mask_np


@pytest.mark.benchmark(group="region_boundary_edges")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
def test_region_boundary_edges(bench_case: BenchCase) -> None:
    """
    The interior seam around a face region: one edge pass, independent of the region's size.

    meshlib's ``findRegionBoundaryUndirectedEdgesInsideMesh`` is the function triwarp's is named
    after and the only reference that has it -- pinned edge-for-edge in
    tests/test_selection.py::test_region_boundary_edges. Its answer is an ``UndirectedEdgeBitSet``
    rather than an ``(k, 2)`` array, so this row times the seam pass on both sides but not the
    decoding, which is test-side. Pure, so the mesh is built once outside the timed callable.

    pyvista has no seam filter and reaches the same edges by **construction**: extract the region as
    a sub-surface, then take that surface's boundary edges. So its row is doing strictly more -- a
    cell extraction and a surface pass before the edge walk, 6.8 ms here -- and its answer is a
    superset, since a region touching the mesh's own rim contributes those edges too. The transform
    is subtracting them, and at that it is exact (``tests/test_selection.py``).
    """
    region_np = _seed_face_region(bench_case)
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        region_pv = np.flatnonzero(region_np)

        def region_edges_pv() -> pv.PolyData:
            surface_pv = mesh_pv.extract_cells(region_pv).extract_surface(
                algorithm="dataset_surface"
            )
            return surface_pv.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                non_manifold_edges=False,
                manifold_edges=False,
            )

        assert bench_case.run(region_edges_pv).n_cells > 0
        return
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        region_ml = mn.faceBitSetFromBools(region_np)
        bits_ml = bench_case.run(
            lambda: mm.findRegionBoundaryUndirectedEdgesInsideMesh(mesh_ml.topology, region_ml)
        )
        assert bits_ml.count() > 0
        return
    faces = bench_case.faces_wp
    region_wp = wp.array(region_np, dtype=wp.bool, device=bench_case.device)
    n_vertices = bench_case.n_vertices
    edges = bench_case.run(
        lambda: tw.selection.region_boundary_edges(faces, region_wp, n_vertices=n_vertices)
    )
    assert int(edges.shape[0]) > 0
