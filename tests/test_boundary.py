"""Regression tests for ``triwarp.boundary`` against Trimesh (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp

import triwarp as tw
from tests.comparisons import assert_cyclic_permutation_equal
from tests.conversions import trimesh_to_pymeshlab

# Open-surface fixtures that actually have a boundary (watertight solids do not).
OPEN_MESHES = ["hemisphere", "half_torus"]


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_edges", "trimesh")
def test_boundary_edges(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    boundary_edges_wp = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)

    assert np.array_equal(
        _lexsort_rows(boundary_edges_wp.numpy()), _lexsort_rows(boundary_edges_tm)
    )


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_oriented_boundary_edges(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    oriented_edges_tm = mesh_tm.edges[_boundary_indices_tm(mesh_tm)]
    oriented_edges_wp = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)

    # Directed edges: compare as a set without sorting within each row.
    assert np.array_equal(
        _lexsort_rows(oriented_edges_wp.numpy()), _lexsort_rows(oriented_edges_tm)
    )


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_edges", "pymeshlab")
def test_boundary_vertex_indices(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: MeshLab marks the boundary *vertices* where triwarp returns the edge pairs.

    ``compute_selection_from_mesh_border`` does the same find-the-boundary pass and stops one step
    earlier, writing a per-vertex bool selection rather than the edges. Two named transforms make
    them comparable: the reference is read off ``vertex_selection_array()`` (the filter returns
    ``None``), and triwarp's edge pairs are projected down with ``np.unique`` -- which is exactly
    what [`boundary_vertex_indices`][triwarp.boundary.boundary_vertex_indices] computes, so the
    projection is a function under test rather than test-side glue.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    vertex_indices_tm = np.unique(boundary_edges_tm)
    vertex_indices_wp = tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_from_mesh_border()
    selection_pml = np.asarray(meshset_pml.current_mesh().vertex_selection_array())

    assert np.array_equal(vertex_indices_wp.numpy(), vertex_indices_tm)
    assert np.array_equal(np.flatnonzero(selection_pml), vertex_indices_wp.numpy())
    # And the edges themselves project onto the same vertex set.
    edges_wp = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(np.unique(edges_wp.numpy()), np.flatnonzero(selection_pml))


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_vertices(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    vertices_tm = mesh_tm.vertices[np.unique(boundary_edges_tm)]
    vertices_wp = tw.boundary.boundary_vertices(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(vertices_wp.numpy(), vertices_tm, rtol=1e-4, atol=1e-4)


def test_boundary_precomputed_edges(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = hemisphere
    edges_sorted_wp = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)
    edges_wp = tw.edges.faces_to_edges(mesh_wp.indices)

    # Boundary row order is non-deterministic (group compacts via an atomic counter), so
    # the precomputed-edge path must yield the same edge *set* as the derived path.
    boundary_default = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    boundary_precomputed = tw.boundary.boundary_edges(
        mesh_wp.points, mesh_wp.indices, edges_sorted=edges_sorted_wp
    )
    assert np.array_equal(
        _lexsort_rows(boundary_default.numpy()), _lexsort_rows(boundary_precomputed.numpy())
    )

    oriented_default = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)
    oriented_precomputed = tw.boundary.oriented_boundary_edges(
        mesh_wp.points, mesh_wp.indices, edges_sorted=edges_sorted_wp, edges=edges_wp
    )
    assert np.array_equal(
        _lexsort_rows(oriented_default.numpy()), _lexsort_rows(oriented_precomputed.numpy())
    )

    indices_default = tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices)
    indices_precomputed = tw.boundary.boundary_vertex_indices(
        mesh_wp.points, mesh_wp.indices, edges_sorted=edges_sorted_wp
    )
    assert np.array_equal(indices_default.numpy(), indices_precomputed.numpy())


def test_boundary_watertight(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    assert tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape == (0, 2)
    assert tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices).shape == (0, 2)
    assert tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).shape == (0,)
    assert tw.boundary.boundary_vertices(mesh_wp.points, mesh_wp.indices).shape == (0,)


def test_boundary_empty(device: str) -> None:
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    assert tw.boundary.boundary_edges(vertices_wp, faces_wp).shape == (0, 2)
    assert tw.boundary.oriented_boundary_edges(vertices_wp, faces_wp).shape == (0, 2)
    assert tw.boundary.boundary_vertex_indices(vertices_wp, faces_wp).shape == (0,)
    assert tw.boundary.boundary_vertices(vertices_wp, faces_wp).shape == (0,)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_loops", "igl")
def test_boundary_loops(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loops_igl = igl.boundary_loop_all(mesh_tm.faces.astype(np.int64))
    loops_wp = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)

    assert len(loops_wp) == len(loops_igl)
    for loop_wp, loop_igl in zip(loops_wp, loops_igl, strict=True):
        assert np.array_equal(loop_wp.numpy(), np.asarray(loop_igl))


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("boundary_loops", "trimesh")
def test_boundary_loops_matches_trimesh_outline(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``Trimesh.outline()`` returns the same loops as ``Path3D`` entities.

    Three named transforms, all conventions rather than results. The entities index into
    ``Path3D.vertices``, which is the mesh's own vertex array unchanged, so no remapping is needed.
    A closed entity **repeats its first point** as the last one, so that trailing duplicate is
    dropped. And neither the loop *order* within the list (triwarp ranks by length, trimesh by
    traversal) nor the starting point and direction *within* a loop are defined by either library,
    so the lists are paired by lowest vertex index and compared with
    [`tests.comparisons.assert_cyclic_permutation_equal`][].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    outline_tm = mesh_tm.outline()
    assert np.allclose(outline_tm.vertices, mesh_tm.vertices)
    loops_tm = []
    for entity in outline_tm.entities:
        assert bool(entity.closed), "an open outline entity means the fixture is not a clean rim"
        points = np.asarray(entity.points)
        assert points[0] == points[-1]
        loops_tm.append(points[:-1])
    loops_wp = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)

    assert len(loops_wp) == len(loops_tm)
    for loop_wp, loop_tm in zip(
        sorted((loop.numpy() for loop in loops_wp), key=lambda loop: int(loop.min())),
        sorted(loops_tm, key=lambda loop: int(loop.min())),
        strict=True,
    ):
        assert_cyclic_permutation_equal(loop_wp, loop_tm)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_loops_batched_matches_boundary_loops(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loops_wp = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    flat_wp, offsets_wp, sizes_wp = tw.boundary.boundary_loops_batched(
        mesh_wp.points, mesh_wp.indices
    )

    offsets_np, sizes_np = offsets_wp.numpy(), sizes_wp.numpy()
    assert len(loops_wp) == offsets_np.shape[0]
    assert int(flat_wp.shape[0]) == int(sizes_np.sum())
    for i, loop_wp in enumerate(loops_wp):
        begin = int(offsets_np[i])
        assert np.array_equal(loop_wp.numpy(), flat_wp.numpy()[begin : begin + int(sizes_np[i])])


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_loops_copy_detaches_from_packed_buffer(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    # The default is a view into one shared buffer; ``copy=True`` must give independent storage.
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    views = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    copies = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices, copy=True)

    assert len(views) == len(copies)
    for view_wp, copy_wp in zip(views, copies, strict=True):
        assert np.array_equal(view_wp.numpy(), copy_wp.numpy())
    if len(views) > 1:
        assert views[0].ptr != views[1].ptr
        # Adjacent views share one allocation; the copies do not.
        assert views[1].ptr - views[0].ptr == 4 * int(views[0].shape[0])


def test_boundary_loops_non_manifold_terminates(device: str) -> None:
    # A "bowtie" (two triangles sharing a single pinch vertex) has a vertex-non-manifold boundary:
    # the boundary successor chain is not one simple cycle, so vertex 2 gets two outgoing edges and
    # only one survives (last write wins). boundary_loops must still terminate -- the successor walk
    # in ``rank_loop_positions`` is bounded -- rather than spin forever on the device (regression).
    vertices_wp = wp.array(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 2, 3, 4], dtype=np.int32), dtype=wp.int32, device=device)

    loops_wp = tw.boundary.boundary_loops(vertices_wp, faces_wp)

    # Completes without hanging; the boundary vertices are distributed across the returned loops.
    assert sum(int(loop.shape[0]) for loop in loops_wp) >= 1


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_loop(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    loop_igl = igl.boundary_loop(mesh_tm.faces.astype(np.int64))
    loop_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)

    assert np.array_equal(loop_wp.numpy(), loop_igl)


def test_boundary_loops_watertight(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    assert tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices) == []
    assert tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices).shape == (0,)


def test_boundary_loops_empty(device: str) -> None:
    vertices_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)

    assert tw.boundary.boundary_loops(vertices_wp, faces_wp) == []
    assert tw.boundary.boundary_loop(vertices_wp, faces_wp).shape == (0,)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_ears(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces.astype(np.int64)

    ear_igl, ear_opp_igl = igl.ears(faces_np)
    ear_wp, ear_opp_wp = tw.boundary.ears(mesh_wp.indices)

    pairs_igl = np.stack([ear_igl, ear_opp_igl], axis=1)
    pairs_wp = np.stack([ear_wp.numpy(), ear_opp_wp.numpy()], axis=1)
    assert np.array_equal(_lexsort_rows(pairs_wp), _lexsort_rows(pairs_igl))

    oriented_boundary = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)
    boundary_set = {tuple(row) for row in oriented_boundary.numpy()}
    directed_edges = tw.edges.faces_to_edges(mesh_wp.indices).numpy()
    for face_idx, opp in zip(ear_wp.numpy(), ear_opp_wp.numpy(), strict=True):
        f = int(face_idx)
        for local_edge in ((int(opp) + 1) % 3, (int(opp) + 2) % 3):
            edge = tuple(directed_edges[3 * f + local_edge])
            assert edge in boundary_set


def test_ears_watertight(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    ear_wp, ear_opp_wp = tw.boundary.ears(mesh_wp.indices)
    assert ear_wp.shape == (0,)
    assert ear_opp_wp.shape == (0,)


def test_ears_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    ear_wp, ear_opp_wp = tw.boundary.ears(faces_wp)
    assert ear_wp.shape == (0,)
    assert ear_opp_wp.shape == (0,)


def _lexsort_rows(rows: np.ndarray) -> np.ndarray:
    """Sort ``(n, 2)`` rows lexicographically (rows kept intact) for set comparison."""
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def _boundary_indices_tm(mesh_tm: tm.Trimesh) -> np.ndarray:
    return tm_grouping.group_rows(mesh_tm.edges_sorted, require_count=1)
