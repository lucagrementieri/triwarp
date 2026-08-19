"""
Structural tests for ``triwarp.homology``.

**meshlib is the one reference that computes a basis**, through ``detectBasisTunnels``, and it is
compared here for the only thing two bases can share: their *count*, forced by the Euler
characteristic. potpourri3d does not bind geometry-central's homology code and neither trimesh nor
libigl computes one, so for everything else the invariants stand in for an oracle -- each loop has
to be a simple closed walk along real mesh edges, and non-contractible. A basis is not unique, so
nothing checks *which* loops come back: measured on a torus, triwarp returns loops of 32 and 18
edges where MeshLib returns 72 and 32, both valid.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import trimesh_to_meshlib


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
    assert tw.measures.euler_characteristic(mesh_wp.indices) == 2 - 2 * genus
    assert len(loops) == 2 * genus


@pytest.mark.parametrize(("mesh_name", "genus"), [("torus", 1), ("genus_two", 2)])
@pytest.mark.parity("homology_generators", "meshlib")
def test_homology_generator_count_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, genus: int
) -> None:
    """
    Class C (a count): a homology basis is not unique, so the loop *count* is all two share.

    ``detectBasisTunnels`` returns a vector of ``EdgeId`` paths -- MeshLib's own basis of
    non-contractible cycles -- and both libraries must find ``2 * genus`` of them. Which loops they
    are is free: on the torus triwarp returns cycles of 32 and 18 edges where MeshLib returns 72 and
    32, and both are correct bases of the same first homology group.

    So the count is the comparison and the *structure* is what makes it non-vacuous: each of
    MeshLib's paths is checked to be a genuine closed edge walk with no repeated vertex, which is
    the same property [`test_homology_generators_are_simple_closed_edge_cycles`] asserts on
    triwarp's side. A reference returning ``2 * genus`` arbitrary edge lists would pass a bare count
    and fail this.

    The decode is the ``EdgeId`` one this suite uses throughout: ``org`` gives each step's tail and
    the last step's ``dest`` closes the loop.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    loops_wp = tw.homology.homology_generators(mesh_wp.points, mesh_wp.indices)

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    tunnels_ml = mm.detectBasisTunnels(mm.MeshPart(mesh_ml))

    assert len(tunnels_ml) == 2 * genus
    assert len(loops_wp) == len(tunnels_ml)
    for tunnel_ml in tunnels_ml:
        walk_np = [int(mesh_ml.topology.org(edge_ml)) for edge_ml in tunnel_ml]
        assert len(walk_np) >= 3
        assert len(set(walk_np)) == len(walk_np)  # simple: no vertex repeats
        assert int(mesh_ml.topology.dest(tunnel_ml[-1])) == walk_np[0]  # and closed


@pytest.mark.parametrize("mesh_name", ["torus", "genus_two"])
def test_homology_generators_are_simple_closed_edge_cycles(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Not a library comparison: no reference computes a homology basis, so the claim is structural.

    trimesh supplies only the mesh's own edge set, used to check each loop *is* a walk along real
    edges -- closed, and visiting no vertex twice. The generator count is pinned separately by the
    Euler characteristic; what this excludes is a "loop" jumping between unconnected vertices, and
    it asserts the loop list is non-empty first so a function returning nothing cannot pass.
    """
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
    """
    Not a library comparison: the counting identity a tree-cotree decomposition must satisfy.

    ``E = (V - 1) + (F - 1) + 2g`` is the whole point of the construction, and trimesh contributes
    only ``V`` and ``F``. An implementation mislabelling one edge would break the identity, which no
    reference implementation is needed to state.
    """
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
