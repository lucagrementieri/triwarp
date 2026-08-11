"""
Structural tests for ``triwarp.homology``.

This is the one module in the port with no reference implementation on either side: potpourri3d does
not bind geometry-central's homology code, and neither trimesh nor libigl computes a homology basis.
So the invariants stand in for an oracle, and between them they pin the result down tightly: the
*number* of loops is forced by the Euler characteristic, and each loop has to be a simple closed
walk along real mesh edges. A basis is not unique, so nothing checks *which* loops come back.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def _is_simple_edge_cycle(loop: np.ndarray, edges_tm: set[tuple[int, int]]) -> bool:
    """Whether a vertex sequence closes up along mesh edges without repeating a vertex."""
    if len(loop) < 3 or len(set(loop.tolist())) != len(loop):
        return False
    closed = np.append(loop, loop[0])
    return all(tuple(sorted((int(a), int(b)))) in edges_tm for a, b in itertools.pairwise(closed))


@pytest.mark.parametrize(
    ("mesh_name", "genus"),
    # ``bohemian_dome`` is genus 1 like the torus but *self-intersecting*, so the generators are
    # found on a surface whose embedding gives no hint of where they run.
    [("icosahedron", 0), ("torus", 1), ("bohemian_dome", 1), ("genus_two", 2)],
)
def test_homology_generator_count_is_twice_the_genus(
    request: pytest.FixtureRequest, mesh_name: str, genus: int, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    loops = tw.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    # The genus the fixture is built for, and the genus the mesh actually has, must agree first.
    assert tw.totals.euler_characteristic(mesh_wp.indices) == 2 - 2 * genus
    assert len(loops) == 2 * genus


@pytest.mark.parametrize("mesh_name", ["torus", "genus_two"])
def test_homology_generators_are_simple_closed_edge_cycles(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_tm = {tuple(sorted(edge)) for edge in mesh_tm.edges_unique.tolist()}
    loops = tw.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    assert len(loops) > 0
    for loop in loops:
        assert _is_simple_edge_cycle(loop.numpy(), edges_tm)


def test_homology_generators_are_not_contractible(
    torus: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    mesh_tm, mesh_wp = torus
    loops = tw.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    # A contractible loop bounds a disk, so cutting the *faces* along it would split the mesh in
    # two. Removing a genuine generator's vertices instead leaves the surface in one piece: the
    # standard cheap witness that a cycle wraps a handle rather than a disk.
    faces_np = np.asarray(mesh_tm.faces)
    for loop in loops:
        on_loop = np.zeros(len(mesh_tm.vertices), dtype=bool)
        on_loop[loop.numpy()] = True
        kept = faces_np[~on_loop[faces_np].any(axis=1)]
        assert len(kept) > 0
        components = tm.graph.connected_components(
            tm.geometry.faces_to_edges(kept), nodes=np.unique(kept)
        )
        assert len(components) == 1


def test_homology_generators_reject_a_boundary(
    hemisphere: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    _, mesh_wp = hemisphere
    with pytest.raises(ValueError, match="closed surface"):
        tw.homology.homology_generators(mesh_wp.points, mesh_wp.indices)


def test_tree_cotree_partitions_the_edges(torus: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    mesh_tm, mesh_wp = torus
    unique_edges, generator_edges, parents = tw.homology.tree_cotree(
        mesh_wp.points, mesh_wp.indices
    )

    n_vertices = len(mesh_tm.vertices)
    n_faces = len(mesh_tm.faces)
    n_edges = int(unique_edges.shape[0])
    # The decomposition is a partition: primal tree (V - 1 edges), dual tree (F - 1), generators.
    assert int(generator_edges.shape[0]) == n_edges - (n_vertices - 1) - (n_faces - 1)
    # Every vertex but the root has a parent, i.e. the primal tree spans the mesh.
    assert int((parents.numpy() < 0).sum()) == 1
    # And each generator edge is a real edge of the mesh.
    edges_tm = {tuple(sorted(edge)) for edge in mesh_tm.edges_unique.tolist()}
    for edge in generator_edges.numpy():
        assert tuple(sorted(edge.tolist())) in edges_tm


def test_homology_generators_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.homology.homology_generators(vertices_wp, faces_wp) == []
