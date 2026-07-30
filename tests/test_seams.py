"""Regression tests for ``triwarp.seams`` against pymeshlab (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import trimesh_to_pymeshlab


def _upload(mesh_tm: tm.Trimesh, device: str):
    return (
        wp.array(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
        ),
        wp.array(
            np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
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
    vertices_wp, faces_wp = _upload(box_tm, device)
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
    vertices_wp, faces_wp = _upload(box_tm, device)
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=0.0).shape[0]) == 12
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=89.0).shape[0]) == 12
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=91.0).shape[0]) == 0
    assert int(tw.seams.crease_edges(vertices_wp, faces_wp, angle=180.0).shape[0]) == 0


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
    vertices_wp, faces_wp = _upload(box_tm, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)

    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)
    assert int(cut_vertices_wp.shape[0]) == 24
    assert int(cut_faces_wp.shape[0]) == int(faces_wp.shape[0])
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 6
    assert tw.validation.is_winding_consistent(cut_faces_wp)


def test_cut_along_edges_is_geometrically_a_noop(device: str) -> None:
    """Every output vertex sits exactly where its input did, so the surface is unchanged."""
    box_tm = tm.creation.box(extents=[1.0, 2.0, 3.0])
    vertices_wp, faces_wp = _upload(box_tm, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    before_np = box_tm.vertices[box_tm.faces].astype(np.float32)
    after_np = cut_vertices_wp.numpy()[cut_faces_wp.numpy().reshape(-1, 3)]
    assert np.allclose(after_np, before_np, rtol=1e-6, atol=1e-6)


def test_cut_along_edges_with_no_edges_is_the_identity(device: str) -> None:
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = _upload(box_tm, device)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(
        vertices_wp, faces_wp, twt.empty_int32_2d((0, 2), device=device)
    )
    assert int(cut_vertices_wp.shape[0]) == int(vertices_wp.shape[0])
    assert np.allclose(
        np.sort(cut_vertices_wp.numpy(), axis=0), np.sort(vertices_wp.numpy(), axis=0)
    )
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 1


def test_cut_along_edges_all_interior_edges_gives_a_triangle_soup(device: str) -> None:
    """Cutting everything leaves one vertex per corner: the definition of a soup."""
    sphere_tm = tm.creation.icosphere(subdivisions=2)
    vertices_wp, faces_wp = _upload(sphere_tm, device)
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
    vertices_wp, faces_wp = _upload(box_tm, device)
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
        vertices_wp, faces_wp, twt.as_array2d_int32(ring_wp)
    )
    # The four corners of that face each split in two; the other four are untouched.
    assert int(cut_vertices_wp.shape[0]) == int(vertices_wp.shape[0]) + 4
    assert not tw.validation.is_edge_manifold(cut_faces_wp, allow_boundary_edges=False)
    assert _face_component_count(cut_vertices_wp, cut_faces_wp) == 2
    assert len(tw.boundary.boundary_loops(cut_vertices_wp, cut_faces_wp)) == 2


def test_cut_along_edges_round_trips_through_a_weld(device: str) -> None:
    """Welding coincident positions undoes the cut exactly — the documented inverse."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices_wp, faces_wp = _upload(box_tm, device)
    creases_wp = tw.seams.crease_edges(vertices_wp, faces_wp, angle=30.0)
    cut_vertices_wp, cut_faces_wp = tw.seams.cut_along_edges(vertices_wp, faces_wp, creases_wp)

    welded_vertices_wp, _unique, _inverse, welded_faces_wp = tw.repair.remove_duplicated_vertices(
        cut_vertices_wp, cut_faces_wp, epsilon=1e-6
    )
    assert int(welded_vertices_wp.shape[0]) == int(vertices_wp.shape[0])
    assert _face_component_count(welded_vertices_wp, welded_faces_wp) == 1


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

    vertices_wp, faces_wp = _upload(box_tm, device)
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


def test_cut_along_edges_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match=r"edges must have shape \(k, 2\)"):
        tw.seams.cut_along_edges(
            mesh_wp.points, mesh_wp.indices, twt.empty_int32_2d((3, 3), device=mesh_wp.device)
        )


def test_cut_along_edges_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.seams.cut_along_edges(
        vertices_wp, faces_wp, twt.empty_int32_2d((0, 2), device=device)
    )
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0
