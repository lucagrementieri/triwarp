"""Regression tests for ``triwarp.edges`` against Trimesh (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp

import triwarp as tw
from tests.comparisons import assert_unordered_rows_equal
from tests.conversions import trimesh_to_pymeshlab

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _faces_np_to_wp(faces_np: np.ndarray, device: str) -> wp.array:
    return wp.array(faces_np.flatten().astype(np.int32), dtype=wp.int32, device=device)


def _vertices_np_to_wp(vertices_np: np.ndarray, device: str) -> wp.array:
    return wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
    )


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


@pytest.mark.parity("faces_to_edges", "trimesh")
def test_edges(device: str) -> None:
    rng = np.random.default_rng(0)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp)
    assert np.array_equal(edges_wp.numpy(), edges_np)


@pytest.mark.parity("faces_to_edges_sorted", "trimesh")
def test_edges_sorted(device: str) -> None:
    rng = np.random.default_rng(1)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp, sorted=True)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.faces_to_edges(faces_wp).shape == (0, 2)
    assert tw.edges.faces_to_edges(faces_wp, sorted=True).shape == (0, 2)


# ---------------------------------------------------------------------------
# edges_face
# ---------------------------------------------------------------------------


@pytest.mark.parity("edges_face", "trimesh")
def test_edges_face(device: str) -> None:
    rng = np.random.default_rng(3)
    n_faces = 24
    faces_np = rng.integers(0, 50, size=(n_faces, 3), dtype=np.int32)

    faces_wp = _faces_np_to_wp(faces_np, device)
    face_idx_wp = tw.edges.edges_face(faces_wp)

    # each face f contributes edges at positions 3*f, 3*f+1, 3*f+2
    expected = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    assert np.array_equal(face_idx_wp.numpy(), expected)


def test_edges_face_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.edges_face(faces_wp).shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_unique", "trimesh")
def test_edges_unique(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]

    unique_edges_wp, _ = tw.edges.edges_unique(mesh_wp.indices)
    unique_edges_wp_np = unique_edges_wp.numpy()

    order_tm = np.lexsort((unique_edges_tm[:, 1], unique_edges_tm[:, 0]))
    order_wp = np.lexsort((unique_edges_wp_np[:, 1], unique_edges_wp_np[:, 0]))
    assert np.array_equal(unique_edges_wp_np[order_wp], unique_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_unique_inverse", "trimesh")
def test_edges_unique_inverse(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices)
    edges_sorted_wp = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)

    # unique_edges[inverse] must reconstruct edges_sorted
    reconstructed = unique_edges_wp.numpy()[inverse_wp.numpy()]
    assert np.array_equal(reconstructed, edges_sorted_wp.numpy())


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_unique_manifold", "potpourri3d")
def test_edges_unique_matches_potpourri3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The unique undirected edge set against geometry-central's, which lists it in its own order.

    ``benchmarks/test_edges.py`` calls this row "a timing comparison, not a parity one" because
    ``pp3d.edges`` returns geometry-central's internal halfedge ordering. That is a statement about
    *order*, and sorting dissolves it -- the sets themselves must match exactly, so this is class B
    with a lexsort, at full tolerance rather than a weakened one.

    Both sides are canonicalised twice over: ``np.sort(..., axis=1)`` because the pair is undirected
    and the two libraries need not agree on which endpoint comes first, then a row lexsort. Uses the
    manifold fixtures for the reason the benchmark draws this row on the synthetic ``scale`` axis --
    ``pp3d.edges`` raises ``GC_SAFETY_ASSERT FAILURE ... unreferenced vertex`` on any mesh carrying
    an unreferenced vertex, which every scan mesh does.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    edges_pp = np.asarray(
        pp3d.edges(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
        )
    )

    unique_edges_wp, _ = tw.edges.edges_unique(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )
    assert_unordered_rows_equal(np.sort(unique_edges_wp.numpy(), axis=1), np.sort(edges_pp, axis=1))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_unique", "igl")
@pytest.mark.parity("edges_unique_manifold", "igl")
@pytest.mark.parity("edges_unique_inverse", "igl")
def test_edges_unique_and_inverse_match_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The unique undirected edge list and its inverse map, against ``igl.unique_edge_map``.

    igl returns ``(E, uE, EMAP, uEC, uEE)``; ``uE`` is the unique undirected list and ``EMAP`` sends
    each of the ``3 * n_faces`` directed edges to its row in ``uE``. Both libraries are free to
    order
    ``uE`` however they like, so this is class B twice over rather than a weakened comparison:

    - the edge *sets* are compared after sorting each pair and then lexsorting the rows;
    - the inverse maps index into two differently ordered tables *and* are indexed by two different
      directed-edge orderings, so they need both ends aligned. Composing each map with its own table
      removes the first difference; the second is a fixed permutation, because igl stacks its
      directed edges by column (``[F[:,1:3]; F[:,[2,0]]; F[:,0:2]]``, so entry ``f + k * n_faces``)
      while triwarp interleaves them per face (``3f+0 = (v0,v1)``, ``3f+1 = (v1,v2)``,
      ``3f+2 = (v2,v0)``). Verified exact on icosahedron, icosphere(2) and box.

    Composing and then permuting is what makes this a real check of the map rather than of the edge
    set: it would fail on an off-by-one, on a permuted table, or on two directed edges assigned to
    the wrong unique row -- none of which a bare count of distinct labels would catch.

    ``edges_unique_manifold`` is the same call on the clean synthetic meshes, which is where the
    benchmark draws it so potpourri3d can run alongside; the fixtures here are all manifold.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_wp.points.shape[0])
    faces_np = mesh_tm.faces.astype(np.int64)

    _e_igl, unique_edges_igl, inverse_igl = igl.unique_edge_map(faces_np)[:3]
    unique_edges_igl = np.asarray(unique_edges_igl)
    inverse_igl = np.asarray(inverse_igl).ravel()

    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices, n_vertices=n_vertices)
    unique_edges_np = unique_edges_wp.numpy()

    # The standalone entry point is what the ``edges_unique_inverse`` benchmark group times; it must
    # return the same map the combined call does.
    standalone_wp = tw.edges.edges_unique_inverse(mesh_wp.indices, n_vertices=n_vertices)
    assert np.array_equal(standalone_wp.numpy(), inverse_wp.numpy())

    assert_unordered_rows_equal(np.sort(unique_edges_np, axis=1), np.sort(unique_edges_igl, axis=1))

    # Compose each inverse map with its own table, then reorder igl's directed edges into triwarp's
    # per-face interleaving (see the docstring) so the two are indexed the same way.
    n_faces = len(faces_np)
    order_igl = np.empty(3 * n_faces, dtype=np.int64)
    order_igl[0::3] = np.arange(n_faces) + 2 * n_faces  # (v0, v1)
    order_igl[1::3] = np.arange(n_faces) + 0 * n_faces  # (v1, v2)
    order_igl[2::3] = np.arange(n_faces) + 1 * n_faces  # (v2, v0)

    resolved_wp = np.sort(unique_edges_np[inverse_wp.numpy()], axis=1)
    resolved_igl = np.sort(unique_edges_igl[inverse_igl][order_igl], axis=1)
    assert np.array_equal(resolved_wp, resolved_igl)


def test_edges_unique_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unique_wp, inverse_wp = tw.edges.edges_unique(faces_wp)
    assert unique_wp.shape == (0, 2)
    assert inverse_wp.shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique_inverse (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_unique_inverse_standalone(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)

    _unique_edges_wp, inverse_from_unique = tw.edges.edges_unique(mesh_wp.indices)
    inverse_standalone = tw.edges.edges_unique_inverse(mesh_wp.indices)
    assert np.array_equal(inverse_from_unique.numpy(), inverse_standalone.numpy())


# ---------------------------------------------------------------------------
# edges_unique_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_unique_length", "trimesh")
def test_edges_unique_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(
        verts_np[unique_edges_tm[:, 1]] - verts_np[unique_edges_tm[:, 0]], axis=1
    )

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_unique_length(vertices_wp, mesh_wp.indices)
    lengths_wp_np = lengths_wp.numpy()

    # lengths are unordered — sort both for comparison
    assert np.allclose(np.sort(lengths_wp_np), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)


def test_edges_unique_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(7)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = _vertices_np_to_wp(verts_np, device)

    unique_edges_wp, _ = tw.edges.edges_unique(faces_wp)
    lengths_via_precomputed = tw.edges.edges_unique_length(
        vertices_wp, faces_wp, unique_edges=unique_edges_wp
    )
    lengths_fresh = tw.edges.edges_unique_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# edges_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("edges_length", "trimesh")
def test_edges_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    edges_np = mesh_tm.edges
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(verts_np[edges_np[:, 1]] - verts_np[edges_np[:, 0]], axis=1)

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_length(vertices_wp, mesh_wp.indices)

    assert np.allclose(np.sort(lengths_wp.numpy()), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)


def test_edges_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(8)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = _vertices_np_to_wp(verts_np, device)

    edges_in_wp = tw.edges.faces_to_edges(faces_wp)
    lengths_via_precomputed = tw.edges.edges_length(vertices_wp, faces_wp, edges_in=edges_in_wp)
    lengths_fresh = tw.edges.edges_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


def test_edges_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.edges_length(verts_wp, faces_wp).shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("mean_edge_length", "trimesh")
def test_mean_edge_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The **per-face** average: every face contributes all three of its edges.

    An interior edge is therefore counted twice and a boundary edge once. The closing block asserts
    this really is a different number from
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] on the open fixtures -- the
    two functions exist because libigl needs both, so a change collapsing them into one must fail.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    verts_np = mesh_tm.vertices.astype(np.float64)
    triangles_np = verts_np[mesh_tm.faces]
    per_face_np = float(np.linalg.norm(triangles_np - triangles_np[:, [1, 2, 0]], axis=2).mean())

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    assert np.allclose(
        tw.edges.mean_edge_length(vertices_wp, mesh_wp.indices), per_face_np, rtol=1e-4, atol=1e-4
    )

    # The unique-edge average is the same number only when there is no boundary.
    unique_wp = tw.edges.mean_unique_edge_length(vertices_wp, mesh_wp.indices)
    n_boundary = int(tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0])
    if n_boundary == 0:
        assert np.isclose(unique_wp, per_face_np, rtol=1e-6)
    else:
        assert not np.isclose(unique_wp, per_face_np, rtol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("mean_unique_edge_length", "igl", "pymeshlab", "trimesh")
@pytest.mark.parity("mean_edge_length", "igl")
@pytest.mark.parity("edges_length", "igl")
def test_edge_length_averages_match_their_references(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Both edge averages, each against the libigl function it is meant to reproduce.

    libigl carries two, and triwarp now exposes one for each:
    [`mean_edge_length`][triwarp.edges.mean_edge_length] is
    ``CurvatureCalculator::getAverageEdge`` (each edge once per incident face), which
    ``igl::principal_curvature`` uses to set its sphere radius;
    [`mean_unique_edge_length`][triwarp.edges.mean_unique_edge_length] is ``igl::avg_edge_length``
    (each edge once), which ``igl::heat_geodesics`` uses to set its timestep.

    They agree exactly on a closed mesh -- every edge has two incident faces there -- and diverge
    otherwise: measured **0.452405 against 0.449910** on ``half_torus`` (64 boundary edges) and
    **0.293087 against 0.291590** on ``hemisphere`` (24), while ``icosahedron`` reads 1.051462
    either way. The final block asserts that divergence, which is what stops the two being quietly
    swapped for one another.

    The per-face length *table* is class B: igl's ``(n_faces, 3)`` uses its opposite-edge corner
    convention against triwarp's flat face-order buffer, so rows are sorted before comparing.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = mesh_tm.faces.astype(np.int64)

    lengths_igl = np.asarray(igl.edge_lengths(vertices_np, faces_np))
    lengths_wp = tw.edges.edges_length(mesh_wp.points, mesh_wp.indices).numpy().reshape(-1, 3)
    assert np.allclose(
        np.sort(lengths_wp, axis=1), np.sort(lengths_igl, axis=1), rtol=1e-4, atol=1e-5
    )

    # Per-face average == igl's curvature-side getAverageEdge, i.e. edge_lengths().mean().
    per_face_wp = float(tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices))
    assert np.isclose(per_face_wp, float(lengths_igl.mean()), rtol=1e-4)

    # Unique-edge average == igl::avg_edge_length == MeshLab's avg_edge_length.
    unique_wp = float(tw.edges.mean_unique_edge_length(mesh_wp.points, mesh_wp.indices))
    assert np.isclose(unique_wp, float(igl.avg_edge_length(vertices_np, faces_np)), rtol=1e-4)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_geometric_measures()
    assert np.isclose(unique_wp, float(measures_pml["avg_edge_length"]), rtol=1e-4)

    # And the numpy dedup, which is the formula the trimesh benchmark row uses.
    edges_np = np.unique(np.sort(tm.geometry.faces_to_edges(mesh_tm.faces), axis=1), axis=0)
    unique_np = float(
        np.linalg.norm(vertices_np[edges_np[:, 1]] - vertices_np[edges_np[:, 0]], axis=1).mean()
    )
    assert np.isclose(unique_wp, unique_np, rtol=1e-4)

    # The two averages coincide only when there is no boundary.
    n_boundary = int(tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).shape[0])
    if n_boundary == 0:
        assert np.isclose(per_face_wp, unique_wp, rtol=1e-4)
    else:
        assert not np.isclose(per_face_wp, unique_wp, rtol=1e-4)


def test_mean_edge_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.mean_edge_length(verts_wp, faces_wp) == 0.0


# --- face_edge_lengths ----------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", _MESHES)
def test_face_edge_lengths_are_the_opposite_edges(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()

    triangles = np.asarray(mesh_tm.vertices)[np.asarray(mesh_tm.faces)]
    expected = np.stack(
        [
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        ],
        axis=1,
    )
    assert np.allclose(lengths, expected, rtol=1e-5, atol=1e-5)
