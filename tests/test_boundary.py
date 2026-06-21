"""Regression tests for ``triwarp.boundary`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp

import triwarp as tw

# Open-surface fixtures that actually have a boundary (watertight solids do not).
OPEN_MESHES = ["hemisphere", "half_torus"]


def _lexsort_rows(rows: np.ndarray) -> np.ndarray:
    """Sort ``(n, 2)`` rows lexicographically (rows kept intact) for set comparison."""
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def _boundary_indices_tm(mesh_tm: tm.Trimesh) -> np.ndarray:
    return tm_grouping.group_rows(mesh_tm.edges_sorted, require_count=1)


# ---------------------------------------------------------------------------
# boundary_edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_edges(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    boundary_edges_wp = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)

    assert np.array_equal(
        _lexsort_rows(boundary_edges_wp.numpy()), _lexsort_rows(boundary_edges_tm)
    )


# ---------------------------------------------------------------------------
# oriented_boundary_edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_oriented_boundary_edges(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    oriented_edges_tm = mesh_tm.edges[_boundary_indices_tm(mesh_tm)]
    oriented_edges_wp = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)

    # Directed edges: compare as a set without sorting within each row.
    assert np.array_equal(
        _lexsort_rows(oriented_edges_wp.numpy()), _lexsort_rows(oriented_edges_tm)
    )


# ---------------------------------------------------------------------------
# boundary_vertex_indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_vertex_indices(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    vertex_indices_tm = np.unique(boundary_edges_tm)
    vertex_indices_wp = tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices)

    assert np.array_equal(vertex_indices_wp.numpy(), vertex_indices_tm)


# ---------------------------------------------------------------------------
# boundary_vertices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_vertices(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    boundary_edges_tm = mesh_tm.edges_sorted[_boundary_indices_tm(mesh_tm)]
    vertices_tm = mesh_tm.vertices[np.unique(boundary_edges_tm)]
    vertices_wp = tw.boundary.boundary_vertices(mesh_wp.points, mesh_wp.indices)

    assert np.allclose(vertices_wp.numpy(), vertices_tm, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# precomputed edges fast path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


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
