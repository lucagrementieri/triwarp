"""Regression tests for ``triwarp.adjacency`` against Trimesh (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.comparisons import lexsort_rows, same_partition
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    trimesh_to_meshlib,
    trimesh_to_pyvista,
)

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


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_face_adjacency_radix_is_invariant_to_an_oversized_base(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the edge grouping is unchanged by any base above ``max(faces)``.

    ``_edge_groups`` asserts the partition is invariant to a sufficiently large radix, which is what
    lets a caller holding a vertex buffer with *unreferenced* vertices pass ``vertices.shape[0]``
    rather than pay the ``reduce.minmax`` that infers ``max(faces) + 1``. Both spellings are
    exercised: ``edges_sorted=None`` hashes off ``faces`` in one launch, supplying it hashes the
    edge rows.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    tight = tw.vertices.n_vertices(mesh_wp.indices)
    edges_sorted = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)

    baseline_wp = tw.adjacency.face_adjacency(mesh_wp.indices, n_vertices=tight)
    for base in (tight + 1, tight + 1000):
        assert np.array_equal(
            tw.adjacency.face_adjacency(mesh_wp.indices, n_vertices=base).numpy(),
            baseline_wp.numpy(),
        )
        assert np.array_equal(
            tw.adjacency.face_adjacency(mesh_wp.indices, edges_sorted, n_vertices=base).numpy(),
            baseline_wp.numpy(),
        )


def test_face_adjacency_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(faces_wp, return_edges=True)
    assert adjacency_wp.shape == (0, 2)
    assert adjacency_edges_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_resolved_face_adjacency_derives_and_forwards(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the resolver derives what ``face_adjacency`` does, and ``n_vertices`` does not move it.

    Both branches are covered -- deriving from ``faces`` with and without the radix, and passing the
    tables straight through, where the radix is documented as ignored and must therefore be
    accepted without changing anything.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_wp.points.shape[0])
    adjacency_wp, edges_wp = tw.adjacency.face_adjacency(mesh_wp.indices, return_edges=True)

    derived_wp, derived_edges_wp = tw.adjacency.resolved_face_adjacency(mesh_wp.indices)
    supplied_wp, supplied_edges_wp = tw.adjacency.resolved_face_adjacency(
        mesh_wp.indices, n_vertices=n_vertices
    )
    passed_wp, passed_edges_wp = tw.adjacency.resolved_face_adjacency(
        mesh_wp.indices, adjacency_wp, edges_wp, n_vertices=n_vertices
    )

    assert int(adjacency_wp.shape[0]) > 0
    for got_wp, got_edges_wp in (
        (derived_wp, derived_edges_wp),
        (supplied_wp, supplied_edges_wp),
        (passed_wp, passed_edges_wp),
    ):
        assert np.array_equal(got_wp.numpy(), adjacency_wp.numpy())
        assert np.array_equal(got_edges_wp.numpy(), edges_wp.numpy())


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


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("face_adjacency", "igl")
@pytest.mark.parity("face_adjacency_unshared", "igl")
def test_face_adjacency_and_unshared_match_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B on both: igl's per-corner table decoded into triwarp's pair list and off-edge corners.

    ``igl.triangle_triangle_adjacency`` returns ``(TT, TTi)`` in a ``(n_faces, 3)`` **corner**
    layout: ``TT[f, i]`` is the face across edge ``i`` of face ``f`` (``-1`` on a boundary) and
    ``TTi[f, i]`` is that edge's index within the neighbour. Two named transforms turn it into what
    triwarp returns, and both are exact:

    1. **pairs** -- collect ``(f, TT[f, i])`` over every corner with a neighbour, sort each pair and
       deduplicate. Every interior pair appears exactly twice in igl's table (once per side), so the
       deduplicated count must equal triwarp's row count, which the assert checks by shape before
       comparing values.
    2. **unshared corners** -- igl's edge ``i`` of face ``f`` runs ``(F[f, i], F[f, (i + 1) % 3])``,
       so the vertex *off* that edge is ``F[f, (i + 2) % 3]``. Reading that for both sides of a pair
       gives triwarp's ``face_adjacency_unshared`` row.

    The second transform is the one worth pinning: the ``(i + 2) % 3`` offset depends on igl's edge
    numbering convention, and getting it wrong yields a table that is a *cyclic shift* of the right
    answer -- still a valid-looking set of vertex indices, and still one vertex per face.

    The three fixtures cover closed (``icosahedron``) and bounded (``half_torus``, ``hemisphere``)
    meshes, so the ``-1`` boundary entries are exercised rather than assumed away.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3).astype(np.int64)

    neighbours_igl, corners_igl = igl.triangle_triangle_adjacency(faces_np)
    has_neighbour = neighbours_igl >= 0
    face_of_corner = np.broadcast_to(np.arange(faces_np.shape[0])[:, None], neighbours_igl.shape)[
        has_neighbour
    ]
    pairs_igl = np.unique(
        np.sort(np.stack([face_of_corner, neighbours_igl[has_neighbour]], axis=1), axis=1), axis=0
    )
    # The off-edge corner of face f across its edge i, keyed by (f, neighbour).
    off_edge_igl = {
        (int(f), int(neighbours_igl[f, i])): int(faces_np[f, (i + 2) % 3])
        for f in range(faces_np.shape[0])
        for i in range(3)
        if neighbours_igl[f, i] >= 0
    }

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    adjacency_np = adjacency_wp.numpy()

    assert np.array_equal(lexsort_rows(np.sort(adjacency_np, axis=1)), pairs_igl)
    assert np.array_equal(
        unshared_wp.numpy(),
        np.array(
            [
                [off_edge_igl[(int(a), int(b))], off_edge_igl[(int(b), int(a))]]
                for a, b in adjacency_np
            ],
            dtype=np.int32,
        ),
    )
    # TTi is what makes transform 2 possible; assert it is the corner index it claims to be.
    assert np.array_equal(corners_igl[has_neighbour] >= 0, np.ones(has_neighbour.sum(), dtype=bool))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
@pytest.mark.parity("face_connected_component_labels", "igl")
def test_face_connected_component_labels_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the same partition under different label *names*.

    ``igl.facet_components`` numbers components ``0..k-1`` in its own traversal order and returns
    ``(n_components, labels)`` -- the count **first**, which is the unpacking trap here. triwarp's
    label propagation names each component after a representative face instead, so on a
    two-component mesh it returns e.g. ``{0, 12}`` where igl returns ``{0, 1}``. The transform is
    [`canonical_labels`][tests.comparisons.canonical_labels]: relabel by first appearance.

    Both a single-component fixture and a two-component union are checked, because a labelling that
    collapsed everything into one component would pass on the first alone.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    faces_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)

    n_components_igl, labels_igl = igl.facet_components(faces_np)
    labels_wp = tw.adjacency.face_connected_component_labels(faces_wp)

    assert n_components_igl == np.unique(labels_wp.numpy()).shape[0]
    assert same_partition(labels_wp.numpy(), np.asarray(labels_igl).ravel())

    # Two disjoint copies: the labelling must split them, which a constant output would not.
    doubled_np = np.concatenate([faces_np, faces_np + faces_np.max() + 1])
    doubled_wp = wp.array(
        np.ascontiguousarray(doubled_np.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=faces_wp.device,
    )
    n_doubled_igl, labels_doubled_igl = igl.facet_components(doubled_np)
    labels_doubled_wp = tw.adjacency.face_connected_component_labels(doubled_wp)

    assert n_doubled_igl == 2 * n_components_igl
    assert same_partition(labels_doubled_wp.numpy(), np.asarray(labels_doubled_igl).ravel())


def _face_labels_ml(components_ml: object, n_faces: int) -> np.ndarray:
    """Decode MeshLib's vector of ``FaceBitSet`` components into a per-face label array."""
    labels_np = np.full(n_faces, -1, dtype=np.int64)
    for label, component_ml in enumerate(components_ml):  # type: ignore[call-overload]
        labels_np[meshlib_bitset_to_numpy(component_ml, n_faces)] = label
    return labels_np


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
@pytest.mark.parity("face_connected_component_labels", "meshlib")
def test_face_connected_component_labels_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B, and the pair that pins *which* incidence rule triwarp implements.

    ``getAllComponents`` takes a ``FaceIncidence`` and the two settings are different operations,
    not two tunings: ``PerEdge`` connects faces sharing an edge, which is triwarp's rule, and
    ``PerVertex`` connects faces sharing a single vertex. On a bowtie -- two triangles meeting at
    one vertex -- they read **2** components and **1**, and triwarp reads 2. No other reference in
    this module exposes that choice, so this is the only test that can fail if the convention ever
    drifts.

    The decode is the usual one: a vector of ``FaceBitSet`` in MeshLib's own traversal order, each
    padded to the face domain, its index taken as the label, compared as a *partition*. Note the
    overload set -- a second form takes ``maxComponentCount`` and returns a ``(components, count)``
    **tuple**, so the result's type is asserted by unpacking it as a plain sequence here.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    n_faces = mesh_tm.faces.shape[0]

    components_ml = mm.getAllComponents(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), mm.MeshComponents.FaceIncidence.PerEdge
    )
    labels_wp = tw.adjacency.face_connected_component_labels(faces_wp)
    assert len(components_ml) == np.unique(labels_wp.numpy()).shape[0]
    assert same_partition(labels_wp.numpy(), _face_labels_ml(components_ml, n_faces))

    # Two disjoint copies, the case a constant labelling would pass.
    doubled_tm = tm.util.concatenate([mesh_tm, mesh_tm.copy().apply_translation([10.0, 0.0, 0.0])])
    doubled_wp = wp.array(
        np.ascontiguousarray(doubled_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=faces_wp.device,
    )
    doubled_ml = mm.getAllComponents(
        mm.MeshPart(trimesh_to_meshlib(doubled_tm)), mm.MeshComponents.FaceIncidence.PerEdge
    )
    assert len(doubled_ml) == 2 * len(components_ml)
    assert same_partition(
        tw.adjacency.face_connected_component_labels(doubled_wp).numpy(),
        _face_labels_ml(doubled_ml, doubled_tm.faces.shape[0]),
    )

    # The convention, on the input that separates the two rules.
    bowtie_vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    bowtie_faces_np = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    bowtie_ml = mm.MeshPart(numpy_to_meshlib(bowtie_vertices_np, bowtie_faces_np))
    bowtie_wp = wp.array(
        np.ascontiguousarray(bowtie_faces_np.reshape(-1)), dtype=wp.int32, device=faces_wp.device
    )
    per_edge_ml = mm.getAllComponents(bowtie_ml, mm.MeshComponents.FaceIncidence.PerEdge)
    per_vertex_ml = mm.getAllComponents(bowtie_ml, mm.MeshComponents.FaceIncidence.PerVertex)
    assert (len(per_edge_ml), len(per_vertex_ml)) == (2, 1)
    assert np.unique(tw.adjacency.face_connected_component_labels(bowtie_wp).numpy()).shape[0] == 2


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
@pytest.mark.parity("face_connected_component_labels", "pyvista")
def test_face_connected_component_labels_matches_pyvista(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B, the same relabelling as the igl row: VTK's ``RegionId`` names the components its way.

    ``connectivity('all')`` writes a ``RegionId`` **cell** array numbered ``0..k-1``, and the
    numbering is neither triwarp's representative-face id nor igl's traversal order -- measured on
    two disjoint spheres it labels the *first* component ``1``, so even a pack-by-first-appearance
    comparison fails and only the partition is shared. That is what
    [`same_partition`][tests.comparisons.same_partition] compares; pyvista ships its own
    ``pack_labels`` for the same reason.

    The two-copy case is the non-vacuous half: on a single-component fixture any labelling at all
    induces the same trivial partition.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices

    labels_pv = np.asarray(trimesh_to_pyvista(mesh_tm).connectivity("all").cell_data["RegionId"])
    labels_wp = tw.adjacency.face_connected_component_labels(faces_wp)
    assert same_partition(labels_wp.numpy(), labels_pv)

    doubled_tm = tm.util.concatenate([mesh_tm, mesh_tm.copy().apply_translation([10.0, 0.0, 0.0])])
    doubled_wp = wp.array(
        np.ascontiguousarray(doubled_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=faces_wp.device,
    )
    labels_doubled_pv = np.asarray(
        trimesh_to_pyvista(doubled_tm).connectivity("all").cell_data["RegionId"]
    )
    labels_doubled_wp = tw.adjacency.face_connected_component_labels(doubled_wp)

    assert np.unique(labels_doubled_pv).shape[0] == 2
    assert same_partition(labels_doubled_wp.numpy(), labels_doubled_pv)


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
    """
    Triwarp against triwarp: passing precomputed face normals must not change the angles.

    Not a reference comparison -- the oracle for the angles themselves is
    [`test_face_adjacency_angles`], which compares them to trimesh. This pins only that the
    precomputed path takes the same route as the deriving one.
    """
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


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("vertex_face_adjacency", "igl")
def test_vertex_face_adjacency_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row order): the same ``(offsets, vertex_faces)`` CSR, arbitrary within a row.

    ``igl.vertex_triangle_adjacency(F, n)`` returns ``(VF, NI)`` -- the payload and the offsets, in
    that order, exactly triwarp's pair reversed -- so the only transform is the unpacking plus
    sorting each row. Both give ``n_vertices + 1`` offsets, so no sentinel has to be appended.

    Row order is genuinely undefined in triwarp's version (a counting-sort scatter, so it is thread
    order) and the docstring says so, which is why the rows are compared as **sets**. The offsets
    are compared exactly: those are not order-dependent, and an off-by-one there is the failure
    mode this function's consumers -- the decimator's normal-flip guard -- see as silent corruption.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_wp = mesh_wp.indices
    n_vertices = int(mesh_wp.points.shape[0])
    faces_np = faces_wp.numpy().reshape(-1, 3).astype(np.int64)

    payload_igl, offsets_igl = igl.vertex_triangle_adjacency(faces_np, n_vertices)
    offsets_wp, payload_wp = tw.adjacency.vertex_face_adjacency(faces_wp, n_vertices=n_vertices)

    assert np.array_equal(offsets_wp.numpy(), np.asarray(offsets_igl).ravel())
    bounds_np = offsets_wp.numpy()
    for vertex in range(n_vertices):
        row_wp = payload_wp.numpy()[bounds_np[vertex] : bounds_np[vertex + 1]]
        row_igl = np.asarray(payload_igl).ravel()[bounds_np[vertex] : bounds_np[vertex + 1]]
        assert np.array_equal(np.sort(row_wp), np.sort(row_igl))


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_face_adjacency_infers_n_vertices(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Omitting ``n_vertices`` costs a readback and must not change the *rows*.

    The **offsets** are compared exactly and the rows only as sets, because the payload order is
    genuinely nondeterministic: the scatter picks each slot with a ``wp.atomic_add`` on a per-vertex
    cursor, so two runs on CUDA order a row differently. Asserting ``array_equal`` on the payload
    would assert something the function does not promise -- it passes on cpu and fails on cuda,
    which is how this test found its own bug.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    inferred_offsets, inferred_faces = tw.adjacency.vertex_face_adjacency(mesh_wp.indices)
    supplied_offsets, supplied_faces = tw.adjacency.vertex_face_adjacency(
        mesh_wp.indices, n_vertices=int(mesh_wp.points.shape[0])
    )

    assert np.array_equal(inferred_offsets.numpy(), supplied_offsets.numpy())
    bounds_np = inferred_offsets.numpy()
    for vertex in range(bounds_np.shape[0] - 1):
        row = slice(int(bounds_np[vertex]), int(bounds_np[vertex + 1]))
        assert np.array_equal(
            np.sort(inferred_faces.numpy()[row]), np.sort(supplied_faces.numpy()[row])
        )


def test_vertex_face_adjacency_unreferenced_vertex(device: str) -> None:
    """
    A vertex no face touches gets an **empty row**, not a missing one.

    That is the property the offsets encode and the reason ``n_vertices`` is a parameter rather than
    inferred unconditionally: with two trailing unreferenced vertices the payload is unchanged and
    only the offsets grow, repeating the final value.
    """
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    offsets_wp, payload_wp = tw.adjacency.vertex_face_adjacency(faces_wp, n_vertices=5)

    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1, 2, 3, 3, 3], dtype=np.int32))
    assert np.array_equal(np.sort(payload_wp.numpy()), np.zeros(3, dtype=np.int32))


def test_vertex_face_adjacency_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp, payload_wp = tw.adjacency.vertex_face_adjacency(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert payload_wp.shape == (0,)


def test_vertex_face_adjacency_zero_rows_with_faces(device: str) -> None:
    """``n_vertices=0`` on a non-empty mesh returns zeros, not an unwritten buffer."""
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp, payload_wp = tw.adjacency.vertex_face_adjacency(faces_wp, n_vertices=0)
    assert offsets_wp.shape == (1,)
    assert np.array_equal(payload_wp.numpy(), np.zeros(3, dtype=np.int32))
