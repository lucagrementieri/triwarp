"""Regression tests for ``triwarp.adjacency`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "half_torus", "hemisphere"]


def _adjacency_order(adjacency_np: np.ndarray) -> np.ndarray:
    """Row order that sorts ``(f0, f1)`` adjacency pairs canonically."""
    return np.lexsort((adjacency_np[:, 1], adjacency_np[:, 0]))


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("face_adjacency", "trimesh")
def test_face_adjacency(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Class A: face pairs and their shared edges, elementwise after a canonical row sort."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    adjacency_edges_tm = mesh_tm.face_adjacency_edges
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )

    order_tm = _adjacency_order(adjacency_tm)
    order_wp = _adjacency_order(adjacency_wp.numpy())
    assert np.array_equal(adjacency_wp.numpy()[order_wp], adjacency_tm[order_tm])
    assert np.array_equal(adjacency_edges_wp.numpy()[order_wp], adjacency_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_face_adjacency_n_vertices_matches_inferred(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """Supplying the hash radix skips a ``reduce.minmax`` readback; the result must not move."""
    _, mesh_wp = request.getfixturevalue(mesh_name)
    inferred_wp = tw.adjacency.face_adjacency(mesh_wp.indices)
    supplied_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )
    assert np.array_equal(inferred_wp.numpy(), supplied_wp.numpy())


def test_face_adjacency_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(faces_wp, return_edges=True)
    assert adjacency_wp.shape == (0, 2)
    assert adjacency_edges_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("face_adjacency_unshared", "trimesh")
def test_face_adjacency_unshared(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Class A: the off-edge corner of each adjacent face, elementwise after a row sort."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    unshared_tm = mesh_tm.face_adjacency_unshared.astype(np.int32)

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_precomputed_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    order_tm = _adjacency_order(adjacency_tm)
    order_wp = _adjacency_order(adjacency_wp.numpy())
    assert np.array_equal(unshared_precomputed_wp.numpy()[order_wp], unshared_tm[order_tm])

    # The table-free path must agree **row for row**, not merely as a set: callers pair its output
    # with a separately-computed face_adjacency, so a permutation between the two would silently
    # mis-associate every row.
    unshared_wp = tw.adjacency.face_adjacency_unshared(mesh_wp.indices)
    assert np.array_equal(unshared_wp.numpy(), unshared_precomputed_wp.numpy())


def test_face_adjacency_unshared_duplicate_faces(device: str) -> None:
    """
    Two coincident triangles: the answer follows the *recorded shared edge*, not a set difference.

    The pair meets across all three of its edges, so three adjacency rows are reported and each
    one's unshared vertex is the corner off *that* edge -- ``[[2, 2], [1, 1], [0, 0]]``, which is
    what ``trimesh.graph.face_adjacency_unshared`` returns for this mesh. A "vertex of one face
    absent from the other" rule would give ``-1`` three times, since no vertex of either face is
    absent from the other; this is the only input class where the two rules diverge, and it is why
    the table-free kernel derives the shared edge from the *edge* index rather than the face pair.

    Both the tabled and table-free paths are checked, since only the former existed when this
    behaviour was first pinned.
    """
    faces_np = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(faces_wp, return_edges=True)
    unshared_tabled_wp = tw.adjacency.face_adjacency_unshared(
        faces_wp, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(faces_wp)

    assert adjacency_wp.shape == (3, 2)
    assert np.array_equal(adjacency_wp.numpy(), np.tile(np.array([0, 1], dtype=np.int32), (3, 1)))
    # The off-edge corner of {0, 1, 2}, computed independently in NumPy, for both faces of the pair.
    off_edge_np = np.array(
        [[int(3 - edge[0] - edge[1])] * 2 for edge in adjacency_edges_wp.numpy()], dtype=np.int32
    )
    assert np.array_equal(unshared_tabled_wp.numpy(), off_edge_np)
    assert np.array_equal(unshared_wp.numpy(), off_edge_np)


def test_face_adjacency_unshared_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unshared_wp = tw.adjacency.face_adjacency_unshared(faces_wp)
    assert unshared_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("face_adjacency_angles", "trimesh")
def test_face_adjacency_angles(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: equal after indexing both sides by their ``(f0, f1)`` pair.

    The two implementations emit adjacency rows in different orders (sort-key order here, edge-list
    order in trimesh), so the angle arrays are matched through the face pair they belong to rather
    than positionally.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    angles_tm = mesh_tm.face_adjacency_angles

    adjacency_wp = tw.adjacency.face_adjacency(mesh_wp.indices)
    angles_wp = tw.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )

    angles_wp_lookup = {
        (int(row[0]), int(row[1])): float(angles_wp.numpy()[i])
        for i, row in enumerate(adjacency_wp.numpy())
    }
    angles_tm_lookup = {
        (int(row[0]), int(row[1])): float(angles_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert angles_wp_lookup.keys() == angles_tm_lookup.keys()
    for key, angle_tm in angles_tm_lookup.items():
        assert np.isclose(angles_wp_lookup[key], angle_tm, rtol=1e-4, atol=5e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_angles_precomputed(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """Passing precomputed face normals must not change the angles."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp = tw.adjacency.face_adjacency(mesh_wp.indices)
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    angles_all_wp = tw.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )
    angles_precomputed_wp = tw.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp, face_normals=face_normals_wp
    )
    assert np.allclose(angles_all_wp.numpy(), angles_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5)

    angles_tm_lookup = {
        (int(row[0]), int(row[1])): float(mesh_tm.face_adjacency_angles[i])
        for i, row in enumerate(mesh_tm.face_adjacency)
    }
    for i, row in enumerate(adjacency_wp.numpy()):
        angle_tm = angles_tm_lookup[(int(row[0]), int(row[1]))]
        assert np.isclose(float(angles_precomputed_wp.numpy()[i]), angle_tm, rtol=1e-4, atol=5e-4)


def test_face_adjacency_angles_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    angles_wp = tw.adjacency.face_adjacency_angles(vertices_wp, faces_wp)
    assert angles_wp.shape == (0,)
