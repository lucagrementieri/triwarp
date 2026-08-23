"""Regression tests for ``triwarp.halfedge`` against Trimesh (CPU reference)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


# ---------------------------------------------------------------------------
# halfedge_twins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_halfedge_twins_are_a_symmetric_pairing(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Not a library comparison: trimesh has no halfedge structure, so the oracle is the algebra.

    Three properties that together pin the pairing without a reference implementation -- it is an
    involution, no halfedge is its own twin, and twins run over the same undirected edge (that last
    is where ``trimesh.geometry.faces_to_edges`` comes in, as a *definition* of the edge a halfedge
    spans rather than as a second answer).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins_wp = tw.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))

    twins = twins_wp.numpy()
    assert twins.shape == (3 * len(mesh_tm.faces),)
    interior = np.flatnonzero(twins >= 0)
    # Involution: crossing an edge twice returns to the same halfedge.
    assert np.array_equal(twins[twins[interior]], interior)
    # A halfedge is never its own twin, and twins run over the same undirected edge.
    halfedge_endpoints_tm = tm.geometry.faces_to_edges(mesh_tm.faces)
    assert np.array_equal(
        np.sort(halfedge_endpoints_tm[interior], axis=1),
        np.sort(halfedge_endpoints_tm[twins[interior]], axis=1),
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_halfedge_twins_has_no_boundary_on_closed_mesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins_wp = tw.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))
    assert (twins_wp.numpy() >= 0).all()


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_halfedge_twins_boundary_matches_oriented_boundary_edges(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    twins = tw.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=len(mesh_tm.vertices)).numpy()

    halfedge_endpoints_tm = tm.geometry.faces_to_edges(mesh_tm.faces)
    boundary_from_twins = halfedge_endpoints_tm[twins < 0]
    boundary_wp = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices)

    assert len(boundary_from_twins) > 0
    assert {tuple(edge) for edge in boundary_from_twins} == {
        tuple(edge) for edge in boundary_wp.numpy()
    }


def test_halfedge_twins_rejects_non_manifold_edge(device: str) -> None:
    # Three triangles hinged on the edge (0, 1): "the" opposite halfedge is not defined.
    faces_wp = wp.array(
        np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32), dtype=wp.int32, device=device
    )
    with pytest.raises(ValueError, match="edge-manifold"):
        tw.halfedge.halfedge_twins(faces_wp, n_vertices=5)


def test_halfedge_twins_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.halfedge.halfedge_twins(faces_wp, n_vertices=0).shape == (0,)


# ---------------------------------------------------------------------------
# vertex_one_rings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_one_ring_sizes_match_incident_face_counts(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A after a named transform (Class B): ring sizes are the incident-face-corner counts.

    One outgoing halfedge per incident corner, so ``np.bincount`` over the flat face buffer is the
    reference -- exact, no tolerance. The second assert is what makes it a *partition*: every
    halfedge appears in exactly one ring, which a size check alone would not catch.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    offsets_wp, ring_wp, _ = tw.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=n_vertices)

    # One outgoing halfedge per incident face-corner.
    incident_faces_tm = np.bincount(mesh_tm.faces.reshape(-1), minlength=n_vertices)
    assert np.array_equal(np.diff(offsets_wp.numpy()), incident_faces_tm)
    assert np.array_equal(np.sort(ring_wp.numpy()), np.arange(3 * len(mesh_tm.faces)))


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_one_ring_neighbor_counts_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class B: ``Trimesh.vertex_neighbors`` counts, after adding one per boundary vertex.

    The transform is the definitional difference between the two structures, not a fudge: a ring
    holds one halfedge per incident *face*, which is one fewer than the neighbour count exactly when
    the fan is open. Folding ``is_boundary`` into the comparison means it also tests that flag,
    which is why it is asserted here as an addend rather than masked out.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    offsets_wp, _, is_boundary_wp = tw.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )

    # A ring holds one halfedge per incident face, so it is one short of the neighbor count at a
    # boundary vertex (whose fan is open) and equal to it in the interior.
    neighbors_tm = np.array([len(neighbors) for neighbors in mesh_tm.vertex_neighbors])
    ring_sizes = np.diff(offsets_wp.numpy())
    assert np.array_equal(ring_sizes + is_boundary_wp.numpy().astype(np.int32), neighbors_tm)


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_one_rings_are_rotationally_ordered(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Not a library comparison: the *ordering* claim, which no reference exposes.

    trimesh's ``vertex_neighbors`` is a set, so it can check the ring's membership but never that
    consecutive entries rotate around the vertex. This asserts that directly -- successive faces
    share an edge *through this vertex* -- and then that one more rotation closes an interior fan
    and falls off an open one, which is the same predicate ``is_boundary`` reports.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    offsets_wp, ring_wp, is_boundary_wp = tw.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )
    twins = tw.halfedge.halfedge_twins(mesh_wp.indices, n_vertices=n_vertices).numpy()
    offsets, ring, is_boundary = offsets_wp.numpy(), ring_wp.numpy(), is_boundary_wp.numpy()

    for vertex in range(n_vertices):
        ring_halfedges = ring[offsets[vertex] : offsets[vertex + 1]]
        # Every entry leaves this vertex.
        assert np.array_equal(
            mesh_tm.faces.reshape(-1)[ring_halfedges], np.full(len(ring_halfedges), vertex)
        )
        # Consecutive entries lie in faces sharing an edge *at this vertex*, i.e. the walk rotates
        # around the vertex rather than wandering over the surface.
        for current, following in itertools.pairwise(ring_halfedges):
            shared = set(mesh_tm.faces[current // 3]) & set(mesh_tm.faces[following // 3])
            assert len(shared) == 2
            assert vertex in shared
        # One more rotation from the last entry closes an interior fan and falls off an open one.
        last = ring_halfedges[-1]
        previous = last + 2 if last % 3 == 0 else last - 1
        assert (twins[previous] == ring_halfedges[0]) != bool(is_boundary[vertex])


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_vertex_one_rings_boundary_flags_match_trimesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class B: trimesh's multiplicity-1 edge grouping, reduced to a per-vertex boolean.

    ``group_rows(..., require_count=1)`` gives the boundary *edges*; the named transform is taking
    the unique vertices they touch. Only the open fixtures, because the flag is uniformly ``False``
    on a closed mesh and the comparison would hold for a constant.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    _, _, is_boundary_wp = tw.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=n_vertices)

    boundary_vertices_tm = np.zeros(n_vertices, dtype=bool)
    boundary_vertices_tm[
        np.unique(mesh_tm.edges[tm.grouping.group_rows(mesh_tm.edges_sorted, require_count=1)])
    ] = True
    assert np.array_equal(is_boundary_wp.numpy(), boundary_vertices_tm)


def test_vertex_one_rings_isolated_vertex_is_empty(device: str) -> None:
    # Vertex 3 is unreferenced: it gets an empty ring rather than a bogus one.
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp, ring_wp, is_boundary_wp = tw.halfedge.vertex_one_rings(faces_wp, n_vertices=4)

    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1, 2, 3, 3]))
    assert np.array_equal(np.sort(ring_wp.numpy()), np.array([0, 1, 2]))
    assert np.array_equal(is_boundary_wp.numpy(), np.array([True, True, True, False]))


def test_vertex_one_rings_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp, ring_wp, is_boundary_wp = tw.halfedge.vertex_one_rings(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert ring_wp.shape == (0,)
    assert is_boundary_wp.shape == (0,)
