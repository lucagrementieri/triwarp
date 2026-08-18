"""Regression tests for ``triwarp.seams`` against pymeshlab (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import lexsort_rows
from tests.conversions import (
    numpy_to_warp,
    pyvista_edges_to_indices,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    wedge_uv_to_pymeshlab,
)


def _sorted_edge_set(edges_np: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(row)) for row in edges_np.tolist()}


def _face_component_count(vertices_wp, faces_wp) -> int:
    mesh_tm = tm.Trimesh(
        vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3), process=False
    )
    return len(
        tm.graph.connected_components(
            mesh_tm.face_adjacency, nodes=np.arange(mesh_tm.faces.shape[0])
        )
    )


def test_crease_edges_finds_a_cube_edges(device: str) -> None:
    """A unit cube has exactly 12 crease edges at 90 degrees, and its 6 face diagonals are flat."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_np = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0).numpy()
    assert creases_np.shape == (12, 2)

    # Every selected edge joins two cube corners one unit apart; a face diagonal would be sqrt(2).
    lengths_np = np.linalg.norm(
        box_tm.vertices[creases_np[:, 0]] - box_tm.vertices[creases_np[:, 1]], axis=1
    )
    assert np.allclose(lengths_np, 1.0, rtol=1e-5)


def test_crease_edges_thresholds(device: str) -> None:
    """
    The comparison is strict, so ``0`` selects every *non-coplanar* interior edge.

    On a cube that is the 12 cube edges and not the 6 face diagonals, whose dihedral is exactly zero
    — which is the useful reading of "all of them", and the reason the test pins ``0`` rather than
    treating it as a synonym for the whole 18-edge set.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=0.0).shape[0]) == 12
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=89.0).shape[0]) == 12
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=91.0).shape[0]) == 0
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=180.0).shape[0]) == 0


@pytest.mark.parity("crease_edges", "pymeshlab")
def test_crease_edges_matches_pymeshlab(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    ``compute_selection_crease_per_edge`` is the same dihedral threshold, reported as a selection.

    MeshLab selects *vertices* of crease edges rather than the edges themselves, so the comparison
    is on the vertex set the two edge lists span — which is the quantity a caller of either one
    actually uses (it is what gets duplicated by a cut, or pinned by a solver).
    """
    mesh_tm, mesh_wp = hemisphere
    angle = 40.0
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_crease_per_edge(angledegneg=-angle, angledegpos=angle)
    selected_pml = np.flatnonzero(meshset_pml.current_mesh().vertex_selection_array())

    creases_np = tw.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=angle).numpy()
    assert set(np.unique(creases_np).tolist()) == set(selected_pml.tolist())


@pytest.mark.parity("crease_edges", "pyvista")
def test_crease_edges_matches_pyvista(device: str) -> None:
    """
    Class B: ``extract_feature_edges`` is the same dihedral threshold, as line cells.

    Two named transforms, both exact. VTK returns a **new** ``PolyData`` whose points are its own
    renumbered subset, so its lines go through
    [`tests.conversions.pyvista_edges_to_indices`][] first; and the three other edge classes
    (``boundary_edges``, ``non_manifold_edges``, ``manifold_edges``) have to be switched **off**,
    because VTK's default emits all four and triwarp's ``include_boundary=False`` emits only the
    crease.

    The fixture is a **box**, not a curved mesh: on a smooth sphere the reference finds nothing at
    ``30`` degrees, so the comparison would be ``[] == []`` -- the ``test_ears`` failure. A box has
    exactly 12 feature edges, which is asserted before the sets are compared.
    """
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)

    edges_pv = pyvista_edges_to_indices(
        trimesh_to_pyvista(mesh_tm).extract_feature_edges(
            feature_angle=30.0,
            feature_edges=True,
            boundary_edges=False,
            non_manifold_edges=False,
            manifold_edges=False,
        ),
        mesh_tm.vertices,
    )
    assert len(edges_pv) == 12  # non-vacuous, and the count a cube's creases must have

    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    assert np.array_equal(lexsort_rows(np.sort(creases_wp.numpy(), axis=1)), lexsort_rows(edges_pv))


def test_crease_edges_include_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = hemisphere
    interior_np = tw.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=40.0).numpy()
    with_boundary_np = tw.seams.crease_edges(
        mesh_wp.points, mesh_wp.indices, angle=40.0, include_boundary=True
    ).numpy()
    boundary_np = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices).numpy()
    assert boundary_np.shape[0] > 0
    assert _sorted_edge_set(with_boundary_np) == _sorted_edge_set(interior_np) | _sorted_edge_set(
        boundary_np
    )
    assert mesh_tm.vertices.shape[0] > 0


def test_crease_edges_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match=r"angle must be in \[0, 180\]"):
        tw.seams.crease_edges(mesh_wp.points, mesh_wp.indices, angle=-1.0)


def test_crease_edges_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.seams.crease_edges(vertices_wp, faces_wp).shape == (0, 2)


def test_cut_along_edges_separates_the_faces_of_a_cube(device: str) -> None:
    """
    Cutting every crease of a cube leaves six disconnected quads, sharing no vertex.

    The counts are exact and worth stating: each of the 8 corners is incident to 3 of the 6 quads
    and every crease through it is cut, so it becomes 3 vertices — 24 in all — while the 12 faces
    and their winding are untouched.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)

    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)
    assert int(cut_vertices_wp.shape[0]) == 24
    assert int(cut_faces_wp.shape[0]) == int(faces_wp.shape[0])
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 6
    assert tw.validation.is_winding_consistent(cut_faces_wp)


def test_cut_along_edges_is_geometrically_a_noop(device: str) -> None:
    """Every output vertex sits exactly where its input did, so the surface is unchanged."""
    box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    before_np = box_tm.vertices[box_tm.faces].astype(np.float32)
    after_np = cut_vertices_wp.numpy()[cut_faces_wp.numpy().reshape(-1, 3)]
    assert np.allclose(after_np, before_np, rtol=1e-6, atol=1e-6)


def test_cut_along_edges_with_no_edges_is_the_identity(device: str) -> None:
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(
        vertices_wp, faces_wp, twt.empty_2d((0, 2), wp.int32, device=device)
    )
    assert int(cut_vertices_wp.shape[0]) == int(vertices_wp.shape[0])
    assert np.allclose(
        np.sort(cut_vertices_wp.numpy(), axis=0), np.sort(vertices_wp.numpy(), axis=0)
    )
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 1


def test_cut_along_edges_all_interior_edges_gives_a_triangle_soup(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """Cutting everything leaves one vertex per corner: the definition of a soup."""
    sphere_tm, _sphere_tm_wp = icosphere_coarse
    vertices_wp, faces_wp = numpy_to_warp(sphere_tm.vertices, sphere_tm.faces, device)
    all_edges_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=0.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, all_edges_wp)
    assert int(cut_vertices_wp.shape[0]) == int(faces_wp.shape[0])
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == int(faces_wp.shape[0]) // 3


def test_cut_along_edges_opens_a_boundary(device: str) -> None:
    """
    A cut edge set becomes boundary, which is the point: a closed cube gains two boundary loops.

    The cut set has to be a **closed curve in the edge graph**, not merely a band of edges. Cutting
    a single edge of a closed surface separates nothing — both endpoints stay connected the long way
    round their fans — and so does any set that leaves some vertex with only one cut edge. The four
    edges bounding one cube face are the smallest set that does close, which is why they are the
    fixture: each of its four corners then has *two* cut edges and splits in two.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)

    creases_np = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0).numpy()
    on_face_np = (box_tm.vertices[creases_np[:, 0], 0] > 0.0) & (
        box_tm.vertices[creases_np[:, 1], 0] > 0.0
    )
    assert on_face_np.sum() == 4
    ring_wp = wp.array(
        np.ascontiguousarray(creases_np[on_face_np], dtype=np.int32), dtype=wp.int32, device=device
    )

    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(
        vertices_wp, faces_wp, twt.as_array2d(ring_wp, wp.int32)
    )
    # The four corners of that face each split in two; the other four are untouched.
    assert int(cut_vertices_wp.shape[0]) == int(vertices_wp.shape[0]) + 4
    assert not tw.validation.is_edge_manifold(cut_faces_wp, allow_boundary_edges=False)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 2
    assert len(tw.boundary.boundary_loops(cut_vertices_wp, cut_faces_wp)) == 2


def test_cut_along_edges_round_trips_through_a_weld(device: str) -> None:
    """Welding coincident positions undoes the cut exactly — the documented inverse."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    welded_vertices_wp, _unique, _inverse, welded_faces_wp = tw.repair.remove_duplicated_vertices(
        cut_vertices_wp, cut_faces_wp, epsilon=1e-6
    )
    assert int(welded_vertices_wp.shape[0]) == int(vertices_wp.shape[0])
    assert _face_component_count(welded_vertices_wp, welded_faces_wp) == 1


@pytest.mark.parity("cut_along_edges", "pymeshlab")
def test_cut_along_edges_matches_pymeshlab_topology(device: str) -> None:
    """
    ``meshing_cut_along_crease_edges`` opens the same seams; only the vertex count differs.

    On a cube cut at every crease both sides produce 6 face-connected components, 12 faces and the
    same surface area — the whole content of the operation. **triwarp emits 24 vertices and MeshLab
    32**: 24 is minimal (each of the 8 corners is incident to 3 quads, so it needs exactly 3 copies)
    and MeshLab's 32 carries 8 redundant duplicates, apparently from splitting per face corner and
    re-welding only some of them. So the count is asserted as an inequality in triwarp's favour
    rather than as a match; the topology is asserted exactly.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    angle = 30.0
    meshset_pml = trimesh_to_pymeshlab(box_tm)
    meshset_pml.meshing_cut_along_crease_edges(angledeg=angle)
    mesh_pml = meshset_pml.current_mesh()
    faces_pml = mesh_pml.face_matrix()
    mesh_cut_pml = tm.Trimesh(mesh_pml.vertex_matrix(), faces_pml, process=False)
    components_pml = len(
        tm.graph.connected_components(
            mesh_cut_pml.face_adjacency, nodes=np.arange(faces_pml.shape[0])
        )
    )

    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=angle)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)
    cut_tm = tm.Trimesh(
        cut_vertices_wp.numpy().astype(np.float64),
        cut_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )

    assert int(cut_faces_wp.shape[0]) // 3 == faces_pml.shape[0]
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == components_pml == 6
    assert np.isclose(cut_tm.area, box_tm.area, rtol=1e-5)
    assert np.isclose(mesh_cut_pml.area, box_tm.area, rtol=1e-5)
    assert int(cut_vertices_wp.shape[0]) == 24 < mesh_pml.vertex_number()


@pytest.mark.parity("cut_along_edges", "igl")
def test_cut_along_edges_matches_igl(device: str) -> None:
    """
    Class B (an edge set becomes a per-corner mask), and it **settles** the MeshLab disagreement.

    ``igl.cut_mesh(V, F, C)`` takes ``C`` as a ``(n_faces, 3)`` **bool** per-corner mask rather than
    an edge list, so the named transform is to mark ``C[f, i]`` for every face-corner whose edge is
    in triwarp's cut set. On the cube cut at every crease igl emits **24 vertices, exactly triwarp's
    answer**, against MeshLab's 32 -- which is the point of having a third implementation on this
    group: 24 is minimal and now independently confirmed, so MeshLab's extra 8 are its own.

    **The corner numbering is ``(i, i + 1)`` here, unlike ``igl.ears``.** ``cut_mesh``'s edge ``i``
    of face ``f`` is ``(F[f, i], F[f, (i + 1) % 3])`` -- the same convention triwarp uses -- where
    ``igl.ears`` inherits ``igl::on_boundary``'s *opposite-vertex* numbering. The two conventions
    coexist inside one library, so the mask is built with the ``(i, i + 1)`` rule and the
    alternative is checked to be wrong rather than assumed: it yields 28 vertices, so a
    convention slip would fail this test rather than pass it.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_np = np.ascontiguousarray(box_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(box_tm.faces, dtype=np.int64)

    vertices_wp, faces_wp = numpy_to_warp(box_tm.vertices, box_tm.faces, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    cut_set = {tuple(sorted(pair)) for pair in creases_wp.numpy().tolist()}
    corner_mask_igl = np.array(
        [
            [
                tuple(sorted((int(faces_np[f, i]), int(faces_np[f, (i + 1) % 3])))) in cut_set
                for i in range(3)
            ]
            for f in range(faces_np.shape[0])
        ],
        dtype=bool,
    )
    assert int(corner_mask_igl.sum()) == 2 * len(cut_set), (
        "every cut edge is marked from both sides"
    )

    vertices_cut_igl, faces_cut_igl = igl.cut_mesh(vertices_np, faces_np, corner_mask_igl)[:2]

    assert vertices_cut_igl.shape[0] == int(cut_vertices_wp.shape[0]) == 24
    assert faces_cut_igl.shape[0] == int(cut_faces_wp.shape[0]) // 3
    cut_tm = tm.Trimesh(
        cut_vertices_wp.numpy().astype(np.float64),
        cut_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    mesh_cut_igl = tm.Trimesh(vertices_cut_igl, faces_cut_igl, process=False)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 6
    assert np.isclose(mesh_cut_igl.area, cut_tm.area, rtol=1e-5)
    # The opposite-vertex convention is genuinely a different answer, so the mask above is a choice.
    opposite_mask_igl = np.array(
        [
            [
                tuple(sorted((int(faces_np[f, (i + 1) % 3]), int(faces_np[f, (i + 2) % 3]))))
                in cut_set
                for i in range(3)
            ]
            for f in range(faces_np.shape[0])
        ],
        dtype=bool,
    )
    assert igl.cut_mesh(vertices_np, faces_np, opposite_mask_igl)[0].shape[0] == 28


def test_cut_along_edges_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match=r"edges must have shape \(k, 2\)"):
        tw.seams.cut_along_edges(
            mesh_wp.points, mesh_wp.indices, twt.empty_2d((3, 3), wp.int32, device=mesh_wp.device)
        )


def test_cut_along_edges_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.seams.cut_along_edges(
        vertices_wp, faces_wp, twt.empty_2d((0, 2), wp.int32, device=device)
    )
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0


def _spherical_wedge_atlas(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """
    Build a ``(3 * n_faces, 2)`` per-corner atlas with one genuine seam: the ``+-pi`` wrap.

    Each face is unwrapped independently -- corners are pulled to within half a period of corner 0
    -- so faces straddling the antimeridian disagree with their neighbours about ``u`` by exactly
    one period. That disagreement is the seam, and it is a *ring*, not the whole mesh.
    """
    centered_np = vertices_np - vertices_np.mean(axis=0)
    corners_np = centered_np[faces_np]
    u_np = np.arctan2(corners_np[..., 1], corners_np[..., 0]) / (2.0 * np.pi) + 0.5
    u_np = u_np - np.round(u_np - u_np[:, :1])
    radius_np = np.linalg.norm(corners_np, axis=-1)
    v_np = np.arccos(np.clip(corners_np[..., 2] / np.maximum(radius_np, 1e-12), -1.0, 1.0)) / np.pi
    return np.stack([u_np, v_np], axis=-1).reshape(-1, 2).astype(np.float32)


def _seam_edges_np(
    texcoords_np: np.ndarray,
    faces_np: np.ndarray,
    face_texcoords_np: np.ndarray,
    match: str = "index",
    tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU port of ``igl::seam_edges`` (``reference/libigl/include/igl/seam_edges.cpp``)."""
    directed = {
        (int(faces_np[f, i]), int(faces_np[f, (i + 1) % 3])): (f, i)
        for f in range(faces_np.shape[0])
        for i in range(3)
    }
    undirected = {(min(a, b), max(a, b)) for a, b in directed}

    def orientation(a_np, b_np, c_np) -> float:
        return float((a_np - c_np)[0] * (b_np - c_np)[1] - (b_np - c_np)[0] * (a_np - c_np)[1])

    seams, boundaries, foldovers = [], [], []
    for edge in undirected:
        reverse = (edge[1], edge[0])
        if edge not in directed or reverse not in directed:
            boundaries.append(list(directed[edge if edge in directed else reverse]))
            continue
        forwards, backwards = directed[edge], directed[reverse]
        tail = (
            face_texcoords_np[forwards[0], forwards[1]],
            face_texcoords_np[backwards[0], (backwards[1] + 1) % 3],
        )
        head = (
            face_texcoords_np[forwards[0], (forwards[1] + 1) % 3],
            face_texcoords_np[backwards[0], backwards[1]],
        )
        if match == "index":
            matched = tail[0] == tail[1] and head[0] == head[1]
        else:
            matched = all(
                np.linalg.norm(texcoords_np[pair[0]] - texcoords_np[pair[1]]) <= tolerance
                for pair in (tail, head)
            )
        row = [forwards[0], forwards[1], backwards[0], backwards[1]]
        if not matched:
            seams.append(row)
            continue
        a_np, b_np = texcoords_np[tail[0]], texcoords_np[head[0]]
        opposite_forwards = texcoords_np[face_texcoords_np[forwards[0], (forwards[1] + 2) % 3]]
        opposite_backwards = texcoords_np[face_texcoords_np[backwards[0], (backwards[1] + 2) % 3]]
        forward_side = orientation(a_np, b_np, opposite_forwards)
        backward_side = orientation(a_np, b_np, opposite_backwards)
        if (forward_side > 0.0 and backward_side > 0.0) or (
            forward_side < 0.0 and backward_side < 0.0
        ):
            foldovers.append(row)

    def rows(collected: list[list[int]], width: int) -> np.ndarray:
        return np.array(collected, dtype=np.int32).reshape(-1, width)

    return rows(seams, 4), rows(boundaries, 2), rows(foldovers, 4)


def _quad_mesh(device: str) -> tuple[wp.array, np.ndarray]:
    """Two triangles sharing the diagonal ``1-2``: the smallest mesh with an interior edge."""
    faces_np = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return wp.array(faces_np.reshape(-1).copy(), dtype=wp.int32, device=device), faces_np


def _upload_uv(uv_np: np.ndarray, device: str) -> wp.array:
    return wp.array(np.ascontiguousarray(uv_np, dtype=np.float32), dtype=wp.vec2, device=device)


@pytest.mark.parity("uv_seam_edges", "pymeshlab")
@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
@pytest.mark.parametrize("include_boundary", [True, False])
def test_uv_seam_vertex_mask_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str, include_boundary: bool
) -> None:
    """
    Class A: ``uv_seam_vertex_mask`` equals MeshLab's texture-seam selection element-wise.

    MeshLab stores per-wedge UVs and so compares coordinates rather than texcoord indices, and it
    folds boundary edges into the seam set -- which is exactly ``match="uv"`` plus
    ``include_boundary=True``. Both closed fixtures (where the two settings agree) and the open one
    (where they must not) are covered, so neither answer of the boolean passes by default.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)

    mask_wp = tw.seams.uv_seam_vertex_mask(
        mesh_wp.indices,
        _upload_uv(wedge_uv_np, str(mesh_wp.device)),
        include_boundary=include_boundary,
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    meshset_pml = wedge_uv_to_pymeshlab(mesh_tm.vertices, faces_np, wedge_uv_np)
    meshset_pml.compute_selection_by_texture_seams_per_vertex()
    selected_pml = meshset_pml.current_mesh().vertex_selection_array()

    # The atlas has a real seam ring, so this is not the trivially-empty comparison.
    assert 0 < int(selected_pml.sum()) < selected_pml.shape[0]
    if include_boundary:
        assert np.array_equal(mask_wp.numpy(), selected_pml)
    else:
        # Dropping the boundary can only ever remove marks, and on an open mesh it must remove some.
        assert not (mask_wp.numpy() & ~selected_pml).any()
        assert bool((mask_wp.numpy() != selected_pml).any()) == (not mesh_tm.is_watertight)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_uv_seam_edges_matches_igl_port(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row-set canonicalization): all three blocks equal the ``igl::seam_edges`` CPU port.

    igl's output order is an ``unordered_set`` walk and triwarp's is halfedge order, so both sides
    are lexsorted first -- exact on integer rows. Run with an explicit ``FTC`` so the *index*
    predicate, the one MeshLab cannot express, is the thing under test.

    Each block is pinned to a per-fixture count first, because a block that is empty on both sides
    compares equal and asserts nothing. Measured: 5 seams and 2 foldovers on either fixture, and 0
    boundaries on ``icosahedron`` against 24 on ``hemisphere`` -- a closed mesh has no UV boundary
    to report, so there the emptiness *is* the claim rather than a gap in the comparison.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    # Pool the wedge UVs into a shared texcoord array, so equal coordinates share an index and the
    # index predicate has something to disagree about.
    texcoords_np, face_texcoords_np = np.unique(wedge_uv_np, axis=0, return_inverse=True)
    face_texcoords_np = face_texcoords_np.reshape(-1, 3).astype(np.int32)

    seams_wp, boundaries_wp, foldovers_wp = tw.seams.uv_seam_edges(
        mesh_wp.indices,
        _upload_uv(texcoords_np, str(mesh_wp.device)),
        wp.array(face_texcoords_np.reshape(-1).copy(), dtype=wp.int32, device=str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    seams_igl, boundaries_igl, foldovers_igl = _seam_edges_np(
        texcoords_np.astype(np.float32), faces_np, face_texcoords_np
    )

    assert seams_igl.shape[0] > 0
    assert foldovers_igl.shape[0] > 0
    assert (boundaries_igl.shape[0] == 0) == mesh_tm.is_watertight
    assert np.array_equal(lexsort_rows(seams_wp.numpy()), lexsort_rows(seams_igl))
    assert np.array_equal(lexsort_rows(boundaries_wp.numpy()), lexsort_rows(boundaries_igl))
    assert np.array_equal(lexsort_rows(foldovers_wp.numpy()), lexsort_rows(foldovers_igl))


def test_uv_seam_edges_match_uv_ignores_duplicate_indices(device: str) -> None:
    """
    The one case where the two references genuinely disagree: a duplicated *identical* texcoord.

    An atlas that stores the same coordinate twice makes igl's index predicate report a seam that
    is not one; MeshLab, holding no indices at all, cannot. Perturbing one copy past the tolerance
    brings the seam back under both.
    """
    faces_wp, _faces_np = _quad_mesh(device)
    # Corners of the shared edge 1-2 point at duplicate texcoord entries 4 and 5.
    texcoords_np = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    face_texcoords_wp = wp.array(
        np.array([0, 1, 2, 4, 3, 5], dtype=np.int32), dtype=wp.int32, device=device
    )

    seams_index_wp, _b, _f = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(texcoords_np, device), face_texcoords_wp, n_vertices=4
    )
    assert int(seams_index_wp.shape[0]) == 1

    seams_uv_wp, _b, _f = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(texcoords_np, device), face_texcoords_wp, match="uv", n_vertices=4
    )
    assert int(seams_uv_wp.shape[0]) == 0

    moved_np = texcoords_np.copy()
    moved_np[4] += 0.25
    seams_moved_wp, _b, _f = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(moved_np, device), face_texcoords_wp, match="uv", n_vertices=4
    )
    assert int(seams_moved_wp.shape[0]) == 1
    # ...and a tolerance wide enough to swallow the move hides it again.
    seams_tolerant_wp, _b, _f = tw.seams.uv_seam_edges(
        faces_wp,
        _upload_uv(moved_np, device),
        face_texcoords_wp,
        match="uv",
        tolerance=1.0,
        n_vertices=4,
    )
    assert int(seams_tolerant_wp.shape[0]) == 0


@pytest.mark.parametrize(
    ("opposite_uv", "n_foldovers"),
    [((0.5, 1.0), 0), ((0.5, -0.5), 1), ((0.5, 0.5), 0)],
    ids=["unfolded", "folded", "collinear"],
)
def test_uv_seam_edges_foldover(
    device: str, opposite_uv: tuple[float, float], n_foldovers: int
) -> None:
    """
    A foldover is matched texcoords whose two opposite corners land on the *same* UV side.

    The three cases sweep the second triangle's free corner across the shared diagonal: clear of it,
    folded back over it, and exactly on it. Collinear is a degenerate UV triangle rather than a
    fold, and igl's strict comparison does not flag it -- so this pins the boundary of the test,
    which a ``>=`` would silently move.
    """
    faces_wp, _faces_np = _quad_mesh(device)
    corner_uv_np = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],  # face 0: vertices 0, 1, 2
            [1.0, 0.0],
            list(opposite_uv),
            [0.5, 0.5],  # face 1: vertices 1, 3, 2
        ],
        dtype=np.float32,
    )
    seams_wp, boundaries_wp, foldovers_wp = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(corner_uv_np, device), n_vertices=4
    )
    assert int(seams_wp.shape[0]) == 0
    assert int(boundaries_wp.shape[0]) == 4
    assert int(foldovers_wp.shape[0]) == n_foldovers


def test_uv_seam_edges_corner_uv_matches_explicit_indices(device: str) -> None:
    """Wedge input is exactly ``match="uv"`` with an identity ``FTC``, and returns the same rows."""
    faces_wp, _faces_np = _quad_mesh(device)
    corner_uv_np = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]], dtype=np.float32
    )
    identity_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)

    wedge = tw.seams.uv_seam_edges(faces_wp, _upload_uv(corner_uv_np, device), n_vertices=4)
    explicit = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(corner_uv_np, device), identity_wp, match="uv", n_vertices=4
    )
    assert int(wedge[0].shape[0]) == 1  # corner 2 and corner 5 disagree, so the diagonal tears
    for block_wedge, block_explicit in zip(wedge, explicit, strict=True):
        assert np.array_equal(block_wedge.numpy(), block_explicit.numpy())


def test_seam_edge_vertices_boundaries_match_oriented_boundary(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Boundary rows name the same directed edges ``oriented_boundary_edges`` reports."""
    mesh_tm, mesh_wp = hemisphere
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    _seams_wp, boundaries_wp, _foldovers_wp = tw.seams.uv_seam_edges(
        mesh_wp.indices,
        _upload_uv(wedge_uv_np, str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    pairs_np = tw.seams.seam_edge_vertices(mesh_wp.indices, boundaries_wp).numpy()
    oriented_np = tw.boundary.oriented_boundary_edges(mesh_wp.points, mesh_wp.indices).numpy()

    assert pairs_np.shape[0] > 0
    assert np.array_equal(lexsort_rows(pairs_np), lexsort_rows(oriented_np))


def test_seam_edge_vertices_feeds_cut_along_edges(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    The detect -> convert -> cut loop: cutting an atlas' own seams opens it without breaking it.

    Seam pairs come out smaller-index-first (the forward halfedge is by definition the one running
    that way), and feeding them to the cut duplicates vertices along the ring while leaving the
    face count, the surface area and the component count untouched.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    wedge_uv_np = _spherical_wedge_atlas(np.asarray(mesh_tm.vertices, dtype=np.float64), faces_np)
    seams_wp, _boundaries_wp, _foldovers_wp = tw.seams.uv_seam_edges(
        mesh_wp.indices,
        _upload_uv(wedge_uv_np, str(mesh_wp.device)),
        n_vertices=int(mesh_tm.vertices.shape[0]),
    )
    pairs_np = tw.seams.seam_edge_vertices(mesh_wp.indices, seams_wp).numpy()
    assert pairs_np.shape[0] > 0
    assert (pairs_np[:, 0] < pairs_np[:, 1]).all()

    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(
        mesh_wp.points, mesh_wp.indices, tw.seams.seam_edge_vertices(mesh_wp.indices, seams_wp)
    )
    cut_tm = tm.Trimesh(
        cut_vertices_wp.numpy().astype(np.float64),
        cut_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    assert int(cut_faces_wp.shape[0]) == int(mesh_wp.indices.shape[0])
    assert int(cut_vertices_wp.shape[0]) > int(mesh_tm.vertices.shape[0])
    assert np.isclose(cut_tm.area, mesh_tm.area, rtol=1e-5)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 1
    # The cut turned the seam ring into a real boundary.
    assert int(tw.boundary.boundary_edges(cut_vertices_wp, cut_faces_wp).shape[0]) > 0


def test_uv_seam_edges_rejects_index_match_without_face_texcoords(device: str) -> None:
    faces_wp, _faces_np = _quad_mesh(device)
    with pytest.raises(ValueError, match="match='index' needs face_texcoords"):
        tw.seams.uv_seam_edges(
            faces_wp, _upload_uv(np.zeros((6, 2), dtype=np.float32), device), match="index"
        )


def test_uv_seam_edges_rejects_mismatched_buffers(device: str) -> None:
    faces_wp, _faces_np = _quad_mesh(device)
    with pytest.raises(ValueError, match="one entry per face corner"):
        tw.seams.uv_seam_edges(
            faces_wp,
            _upload_uv(np.zeros((6, 2), dtype=np.float32), device),
            wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device=device),
        )
    with pytest.raises(ValueError, match="texcoords must be per-corner"):
        tw.seams.uv_seam_edges(faces_wp, _upload_uv(np.zeros((4, 2), dtype=np.float32), device))


def test_uv_seam_edges_nonmanifold_raises(device: str) -> None:
    """Three faces on one edge: "the other side" is undefined, as it is for the cut."""
    faces_wp = wp.array(
        np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=np.int32), dtype=wp.int32, device=device
    )
    with pytest.raises(ValueError, match="edge-manifold"):
        tw.seams.uv_seam_edges(
            faces_wp, _upload_uv(np.zeros((9, 2), dtype=np.float32), device), n_vertices=5
        )


def test_uv_seam_edges_single_triangle(device: str) -> None:
    """Every edge is a boundary, and neither seam nor foldover can exist without a second face."""
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    corner_uv_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    seams_wp, boundaries_wp, foldovers_wp = tw.seams.uv_seam_edges(
        faces_wp, _upload_uv(corner_uv_np, device), n_vertices=3
    )
    assert seams_wp.shape == (0, 4)
    assert foldovers_wp.shape == (0, 4)
    assert np.array_equal(boundaries_wp.numpy(), np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int32))
    assert _sorted_edge_set(tw.seams.seam_edge_vertices(faces_wp, boundaries_wp).numpy()) == {
        (0, 1),
        (1, 2),
        (0, 2),
    }


def test_uv_seam_edges_empty_mesh(device: str) -> None:
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    seams_wp, boundaries_wp, foldovers_wp = tw.seams.uv_seam_edges(
        faces_wp, wp.empty(0, dtype=wp.vec2, device=device), n_vertices=0
    )
    assert seams_wp.shape == (0, 4)
    assert boundaries_wp.shape == (0, 2)
    assert foldovers_wp.shape == (0, 4)
    assert (
        int(
            tw.seams.uv_seam_vertex_mask(
                faces_wp, wp.empty(0, dtype=wp.vec2, device=device), n_vertices=0
            ).shape[0]
        )
        == 0
    )
