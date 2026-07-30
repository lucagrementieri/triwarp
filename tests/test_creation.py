"""Regression tests for ``triwarp.creation`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import cKDTree

import triwarp as tw


def _mesh(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> tm.Trimesh:
    """Wrap a triwarp result as a `trimesh.Trimesh` for its measures, without reprocessing it."""
    return tm.Trimesh(
        vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3), process=False
    )


def _assert_same_faces(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """
    Assert the triwarp result is the same mesh as ``mesh_tm`` up to vertex and face order.

    Vertex order genuinely differs — triwarp compacts its buffers rather than reproducing trimesh's
    merge order — so the comparison matches face *centroid* sets and requires the match to be a
    bijection. A plain lexsort is too brittle here: triwarp works in ``float32`` and trimesh in
    ``float64``, so near-tied coordinates sort differently on the two sides.
    """
    vertices_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    assert vertices_np.shape[0] == mesh_tm.vertices.shape[0], (
        f"vertex count mismatch: got {vertices_np.shape[0]}, expected {mesh_tm.vertices.shape[0]}"
    )
    assert faces_np.shape[0] == mesh_tm.faces.shape[0], (
        f"face count mismatch: got {faces_np.shape[0]}, expected {mesh_tm.faces.shape[0]}"
    )
    centroids_wp = vertices_np[faces_np].mean(axis=1)
    centroids_tm = mesh_tm.vertices[mesh_tm.faces].mean(axis=1)
    distance_np, match_np = cKDTree(centroids_tm).query(centroids_wp)
    assert distance_np.max() < 1e-5, f"face centroids differ by up to {distance_np.max():.3e}"
    assert len(set(match_np.tolist())) == len(match_np), "face centroid match is not a bijection"


def _assert_same_solid(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """
    Assert the triwarp result is the same solid as ``mesh_tm`` without pinning its triangulation.

    Used wherever the result contains an ear-clipped cap: triwarp's clipper and trimesh's earcut
    pick different (equally valid) diagonals, so the face *count* and the enclosed volume agree
    while individual cap triangles do not.
    """
    assert int(vertices_wp.shape[0]) == mesh_tm.vertices.shape[0]
    assert int(faces_wp.shape[0]) // 3 == mesh_tm.faces.shape[0]
    assert np.isclose(_mesh(vertices_wp, faces_wp).volume, mesh_tm.volume, rtol=1e-4)
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds, mesh_tm.bounds, rtol=1e-5, atol=1e-5)


def _assert_closed(vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32]) -> None:
    """
    Assert the mesh bounds a volume with no cracks: trimesh's ``is_watertight`` semantics.

    Deliberately not [`triwarp.validation.is_watertight`][], which follows Open3D and also
    requires the mesh not to self-intersect. That extra check reports the coplanar edge-adjacent
    triangles of a flat cap as intersecting — it says ``True`` for the output of
    ``trimesh.creation.annulus`` too — so it cannot tell a correct cap from a broken one here.
    """
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert tw.validation.is_winding_consistent(faces_wp)
    assert _mesh(vertices_wp, faces_wp).volume > 0.0


def _assert_same_vertices_and_faces(
    vertices_wp: wp.array[wp.vec3], faces_wp: wp.array[wp.int32], mesh_tm: tm.Trimesh
) -> None:
    """Assert an exact face-set match after remapping triwarp's vertices onto trimesh's."""
    faces_np = faces_wp.numpy().reshape(-1, 3)
    distance_np, remap_np = cKDTree(mesh_tm.vertices).query(vertices_wp.numpy().astype(np.float64))
    assert distance_np.max() < 1e-5, f"vertices differ by up to {distance_np.max():.3e}"
    mapped_np = np.sort(remap_np[faces_np], axis=1)
    reference_np = np.sort(mesh_tm.faces, axis=1)
    assert np.array_equal(
        mapped_np[np.lexsort(mapped_np.T[::-1])], reference_np[np.lexsort(reference_np.T[::-1])]
    )


def _ring(points_np: np.ndarray, device: str) -> wp.array[wp.vec2]:
    return wp.array(np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device)


def _mat44(matrix_np: np.ndarray) -> wp.mat44:
    return wp.mat44(*matrix_np.flatten().tolist())


# A rectangle and a non-convex L, as counter-clockwise 2D rings.
_SQUARE_RING = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
_L_RING = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])


# --- table primitives -------------------------------------------------------------------


def test_box(device: str) -> None:
    _assert_same_vertices_and_faces(*tw.creation.box(device=device), tm.creation.box())
    _assert_same_vertices_and_faces(
        *tw.creation.box(extents=(1.0, 2.0, 3.0), device=device),
        tm.creation.box(extents=[1.0, 2.0, 3.0]),
    )


def test_box_bounds(device: str) -> None:
    bounds_np = np.array([[-1.0, 0.0, 2.0], [3.0, 1.0, 5.0]])
    vertices_wp, faces_wp = tw.creation.box(bounds=bounds_np, device=device)
    _assert_same_vertices_and_faces(vertices_wp, faces_wp, tm.creation.box(bounds=bounds_np))
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds, bounds_np, rtol=1e-5, atol=1e-5)


def test_box_transform(device: str) -> None:
    matrix_np = tm.transformations.rotation_matrix(np.deg2rad(37.0), [1.0, 2.0, 3.0])
    matrix_np[:3, 3] = np.array([1.0, -2.0, 0.5])
    _assert_same_vertices_and_faces(
        *tw.creation.box(extents=(1.0, 2.0, 3.0), transform=_mat44(matrix_np), device=device),
        tm.creation.box(extents=[1.0, 2.0, 3.0], transform=matrix_np),
    )


def test_box_mirror_transform_keeps_outward_winding(device: str) -> None:
    mirror_np = np.eye(4)
    mirror_np[0, 0] = -1.0
    vertices_wp, faces_wp = tw.creation.box(
        extents=(1.0, 2.0, 3.0), transform=_mat44(mirror_np), device=device
    )
    # A negative-determinant transform reverses winding, so the faces have to be flipped back.
    assert _mesh(vertices_wp, faces_wp).volume > 0.0
    assert tw.validation.is_volume(vertices_wp, faces_wp)


def test_box_invalid(device: str) -> None:
    bounds_np = np.zeros((2, 3))
    with pytest.raises(ValueError, match="bounds overrides"):
        tw.creation.box(extents=(1.0, 1.0, 1.0), bounds=bounds_np, device=device)
    with pytest.raises(ValueError, match="bounds must be"):
        tw.creation.box(bounds=np.zeros((3, 3)), device=device)
    with pytest.raises(ValueError, match="extents must be"):
        tw.creation.box(extents=np.zeros(4), device=device)


def test_icosahedron(device: str) -> None:
    _assert_same_vertices_and_faces(
        *tw.creation.icosahedron(device=device), tm.creation.icosahedron()
    )


@pytest.mark.parametrize(
    ("builder", "filter_name", "n_vertices", "n_faces"),
    [
        ("tetrahedron", "create_tetrahedron", 4, 4),
        ("octahedron", "create_octahedron", 6, 8),
        ("dodecahedron", "create_dodecahedron", 20, 36),
    ],
)
def test_platonic_solids_match_pymeshlab(
    device: str, builder: str, filter_name: str, n_vertices: int, n_faces: int
) -> None:
    """
    The three tables came from MeshLab, so they must still be it up to the unit-sphere scaling.

    trimesh has no tetrahedron / octahedron / dodecahedron, and igl has no generators at all, so
    pymeshlab is the only reference here.
    """
    vertices_wp, faces_wp = getattr(tw.creation, builder)(device=device)
    assert int(vertices_wp.shape[0]) == n_vertices
    assert int(faces_wp.shape[0]) // 3 == n_faces
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)
    _assert_closed(vertices_wp, faces_wp)

    meshset_pml = ml.MeshSet()
    getattr(meshset_pml, filter_name)()
    vertices_pml = meshset_pml.current_mesh().vertex_matrix()
    vertices_pml = vertices_pml / np.linalg.norm(vertices_pml, axis=1, keepdims=True)
    _assert_same_vertices_and_faces(
        vertices_wp,
        faces_wp,
        tm.Trimesh(vertices_pml, meshset_pml.current_mesh().face_matrix(), process=False),
    )


@pytest.mark.parametrize("count", [(2, 2), (3, 7), (10, 10)])
def test_grid(device: str, count: tuple[int, int]) -> None:
    vertices_wp, faces_wp = tw.creation.grid(count=count, extents=(2.0, 3.0), device=device)
    assert int(vertices_wp.shape[0]) == count[0] * count[1]
    assert int(faces_wp.shape[0]) // 3 == 2 * (count[0] - 1) * (count[1] - 1)

    mesh_tm = _mesh(vertices_wp, faces_wp)
    assert np.allclose(mesh_tm.bounds, [[-1.0, -1.5, 0.0], [1.0, 1.5, 0.0]], rtol=1e-5, atol=1e-5)
    assert np.isclose(mesh_tm.area, 6.0, rtol=1e-5)
    # Flat, wound outward along +Z, and one boundary loop around the rim.
    assert np.allclose(mesh_tm.face_normals, [0.0, 0.0, 1.0], rtol=1e-5, atol=1e-5)
    assert tw.validation.is_winding_consistent(faces_wp)
    assert len(tw.boundary.boundary_loops(vertices_wp, faces_wp)) == 1


def test_grid_matches_pymeshlab(device: str) -> None:
    """MeshLab's ``create_grid`` is the same lattice, in its uncentered form."""
    vertices_wp, faces_wp = tw.creation.grid(
        count=(10, 8), extents=(0.3, 0.5), center=False, device=device
    )
    meshset_pml = ml.MeshSet()
    meshset_pml.create_grid(numvertx=10, numverty=8, absscalex=0.3, absscaley=0.5, center=False)
    mesh_pml = meshset_pml.current_mesh()
    assert int(vertices_wp.shape[0]) == mesh_pml.vertex_number()
    assert int(faces_wp.shape[0]) // 3 == mesh_pml.face_number()
    # MeshLab lays the patch out along -X; compare the shape after mirroring that back.
    vertices_pml = mesh_pml.vertex_matrix() * np.array([-1.0, 1.0, 1.0])
    assert np.allclose(
        np.sort(vertices_wp.numpy().astype(np.float64), axis=0),
        np.sort(vertices_pml, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grid_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="count must be at least 2"):
        tw.creation.grid(count=(1, 4), device=device)
    with pytest.raises(ValueError, match="extents must be non-negative"):
        tw.creation.grid(extents=(-1.0, 1.0), device=device)


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3, 4])
def test_sphere_cap(device: str, subdivisions: int) -> None:
    angle, radius = np.deg2rad(35.0), 1.5
    vertices_wp, faces_wp = tw.creation.sphere_cap(
        angle=angle, subdivisions=subdivisions, radius=radius, device=device
    )
    n_rings = 2**subdivisions
    assert int(vertices_wp.shape[0]) == 1 + 3 * n_rings * (n_rings + 1)
    assert int(faces_wp.shape[0]) // 3 == 6 * n_rings**2

    # Every vertex on the sphere, the apex at the pole, and the rim at exactly ``angle``.
    vertices_np = vertices_wp.numpy().astype(np.float64)
    assert np.allclose(np.linalg.norm(vertices_np, axis=1), radius, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertices_np[0], [0.0, 0.0, radius], rtol=1e-5, atol=1e-5)
    polar_np = np.arccos(np.clip(vertices_np[:, 2] / radius, -1.0, 1.0))
    assert np.isclose(polar_np.max(), angle, rtol=1e-5, atol=1e-5)

    # An open disc: one boundary loop, of exactly the rim's ``6 * n_rings`` vertices.
    loops = tw.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops) == 1
    assert int(loops[0].shape[0]) == 6 * n_rings
    assert tw.validation.is_winding_consistent(faces_wp)
    # Wound outward: the area-weighted normal of a cap around +Z points along +Z.
    normals_wp, areas_wp = tw.triangles.face_normals_and_areas(vertices_wp, faces_wp)
    assert (normals_wp.numpy() * areas_wp.numpy()[:, None]).sum(axis=0)[2] > 0.0

    # Area of a spherical cap of half-angle ``angle``, approached from below by the inscribed mesh.
    exact_area = 2.0 * np.pi * radius**2 * (1.0 - np.cos(angle))
    assert _mesh(vertices_wp, faces_wp).area <= exact_area * (1.0 + 1e-6)
    if subdivisions >= 3:
        assert _mesh(vertices_wp, faces_wp).area > exact_area * 0.99


def test_sphere_cap_matches_pymeshlab_size(device: str) -> None:
    """MeshLab's ``create_sphere_cap`` builds the same lattice; ``angle`` is its full aperture."""
    vertices_wp, faces_wp = tw.creation.sphere_cap(
        angle=np.deg2rad(30.0), subdivisions=3, device=device
    )
    meshset_pml = ml.MeshSet()
    meshset_pml.create_sphere_cap(angle=60.0, subdiv=3)
    mesh_pml = meshset_pml.current_mesh()
    assert int(vertices_wp.shape[0]) == mesh_pml.vertex_number()
    assert int(faces_wp.shape[0]) // 3 == mesh_pml.face_number()
    # MeshLab puts the rim plane at z = 0 rather than centering the sphere; shift it back and the
    # two caps are the same surface (its lattice rings are rotated in azimuth, so compare radii).
    vertices_pml = mesh_pml.vertex_matrix() + np.array([0.0, 0.0, np.cos(np.deg2rad(30.0))])
    assert np.allclose(
        np.sort(np.linalg.norm(vertices_wp.numpy().astype(np.float64), axis=1)),
        np.sort(np.linalg.norm(vertices_pml, axis=1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sphere_cap_invalid(device: str) -> None:
    with pytest.raises(ValueError, match=r"angle must be in \(0, pi\)"):
        tw.creation.sphere_cap(angle=0.0, device=device)
    with pytest.raises(ValueError, match=r"angle must be in \(0, pi\)"):
        tw.creation.sphere_cap(angle=np.pi, device=device)
    with pytest.raises(ValueError, match="subdivisions must be non-negative"):
        tw.creation.sphere_cap(subdivisions=-1, device=device)


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
def test_icosphere(device: str, subdivisions: int) -> None:
    vertices_wp, faces_wp = tw.creation.icosphere(subdivisions=subdivisions, device=device)
    _assert_same_vertices_and_faces(
        vertices_wp, faces_wp, tm.creation.icosphere(subdivisions=subdivisions)
    )
    assert int(faces_wp.shape[0]) // 3 == 20 * 4**subdivisions
    assert int(vertices_wp.shape[0]) == 10 * 4**subdivisions + 2
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_icosphere_radius(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.icosphere(subdivisions=3, radius=2.5, device=device)
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 2.5, rtol=1e-5, atol=1e-5)
    exact_volume = 4.0 / 3.0 * np.pi * 2.5**3
    assert abs(_mesh(vertices_wp, faces_wp).volume - exact_volume) / exact_volume < 0.01


# --- revolution primitives --------------------------------------------------------------


def test_uv_sphere(device: str) -> None:
    _assert_same_faces(*tw.creation.uv_sphere(device=device), tm.creation.uv_sphere())
    vertices_wp, _ = tw.creation.uv_sphere(radius=3.0, device=device)
    assert np.allclose(np.linalg.norm(vertices_wp.numpy(), axis=1), 3.0, rtol=1e-5, atol=1e-5)


def test_uv_sphere_explicit_count_doubles_longitude(device: str) -> None:
    # trimesh doubles count[1] only when count is passed explicitly; the port keeps that asymmetry.
    vertices_wp, faces_wp = tw.creation.uv_sphere(count=(16, 16), device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.uv_sphere(count=[16, 16]))
    assert int(vertices_wp.shape[0]) == 14 * 32 + 2


def test_capsule(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.capsule(height=2.0, radius=0.5, device=device)
    mesh_tm = tm.creation.capsule(height=2.0, radius=0.5)
    _assert_same_faces(vertices_wp, faces_wp, mesh_tm)
    # Centered on the origin, spanning +-(height / 2 + radius) along Z.
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds[:, 2], [-1.5, 1.5], rtol=1e-5, atol=1e-4)


def test_cylinder(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.cylinder(radius=1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.cylinder(radius=1.0, height=2.0))
    # 32 sections truncate the circle, so the volume is the inscribed prism's, not pi * r^2 * h.
    inscribed = 0.5 * 32 * np.sin(2.0 * np.pi / 32) * 2.0
    assert np.isclose(_mesh(vertices_wp, faces_wp).volume, inscribed, rtol=1e-4)


def test_cylinder_segment(device: str) -> None:
    segment_np = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vertices_wp, faces_wp = tw.creation.cylinder(radius=0.5, segment=segment_np, device=device)
    mesh_tm = tm.creation.cylinder(radius=0.5, segment=segment_np)
    _assert_same_faces(vertices_wp, faces_wp, mesh_tm)
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds, mesh_tm.bounds, rtol=1e-5, atol=1e-5)


def test_cylinder_requires_height_or_segment(device: str) -> None:
    with pytest.raises(ValueError, match="height or segment"):
        tw.creation.cylinder(radius=1.0, device=device)
    with pytest.raises(ValueError, match="segment must be"):
        tw.creation.cylinder(radius=1.0, segment=np.zeros((3, 3)), device=device)


def test_cone(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.cone(radius=1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.cone(radius=1.0, height=2.0))
    # 32 rim vertices plus the apex and the base center; the two fans need the vertex collapse.
    assert int(vertices_wp.shape[0]) == 34
    _assert_closed(vertices_wp, faces_wp)


def test_annulus(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.annulus(0.5, 1.0, height=2.0, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.annulus(0.5, 1.0, height=2.0))
    # The closing point of the annulus profile has to collapse, or the inner-wall seam stays open.
    _assert_closed(vertices_wp, faces_wp)
    assert tw.validation.euler_characteristic(faces_wp) == 0


def test_annulus_zero_inner_radius_is_a_cylinder(device: str) -> None:
    _assert_same_faces(
        *tw.creation.annulus(0.0, 1.0, height=2.0, device=device),
        tm.creation.cylinder(radius=1.0, height=2.0),
    )


def test_annulus_requires_height_or_segment(device: str) -> None:
    with pytest.raises(ValueError, match="height or segment"):
        tw.creation.annulus(0.5, 1.0, device=device)


def test_torus(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.torus(1.0, 0.25, device=device)
    _assert_same_faces(vertices_wp, faces_wp, tm.creation.torus(1.0, 0.25))
    assert int(faces_wp.shape[0]) // 3 == 2 * 32 * 32
    exact_volume = 2.0 * np.pi**2 * 1.0 * 0.25**2
    assert abs(_mesh(vertices_wp, faces_wp).volume - exact_volume) / exact_volume < 0.02


_CLOSED_BUILDERS = {
    "uv_sphere": lambda device: tw.creation.uv_sphere(device=device),
    "capsule": lambda device: tw.creation.capsule(device=device),
    "cylinder": lambda device: tw.creation.cylinder(radius=1.0, height=2.0, device=device),
    "cone": lambda device: tw.creation.cone(radius=1.0, height=2.0, device=device),
    "annulus": lambda device: tw.creation.annulus(0.5, 1.0, height=2.0, device=device),
    "torus": lambda device: tw.creation.torus(1.0, 0.25, device=device),
    "icosphere": lambda device: tw.creation.icosphere(subdivisions=2, device=device),
    "box": lambda device: tw.creation.box(device=device),
}


@pytest.mark.parametrize("name", sorted(_CLOSED_BUILDERS))
def test_closed_primitives_are_volumes(device: str, name: str) -> None:
    # The single assertion that pins the vertex collapse: without it the apex, pole and
    # closed-profile vertices stay duplicated per slice and nothing here is watertight.
    vertices_wp, faces_wp = _CLOSED_BUILDERS[name](device)
    _assert_closed(vertices_wp, faces_wp)
    assert tw.validation.is_volume(vertices_wp, faces_wp)


@pytest.mark.parametrize("name", sorted(_CLOSED_BUILDERS))
def test_primitives_are_deterministic(device: str, name: str) -> None:
    first_v, first_f = _CLOSED_BUILDERS[name](device)
    second_v, second_f = _CLOSED_BUILDERS[name](device)
    assert np.array_equal(first_v.numpy(), second_v.numpy())
    assert np.array_equal(first_f.numpy(), second_f.numpy())


def test_revolve_matches_trimesh(device: str) -> None:
    profile_np = np.array([[0.25, 0.0], [1.0, 0.0], [1.0, 1.0], [0.25, 1.0], [0.25, 0.0]])
    _assert_same_faces(
        *tw.creation.revolve(_ring(profile_np, device), sections=24),
        tm.creation.revolve(profile_np, sections=24),
    )


@pytest.mark.parametrize("cap", [False, True])
def test_revolve_partial_revolution(device: str, cap: bool) -> None:
    profile_np = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0], [0.5, 0.0]])
    vertices_wp, faces_wp = tw.creation.revolve(
        _ring(profile_np, device), angle=np.pi, cap=cap, sections=16
    )
    mesh_tm = tm.creation.revolve(profile_np, angle=np.pi, cap=cap, sections=16)
    closed = tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert closed is cap
    if cap:
        # The cap winding has to come out facing away from the solid, not into it.
        _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
        _assert_closed(vertices_wp, faces_wp)
        assert tw.validation.is_volume(vertices_wp, faces_wp)
    else:
        _assert_same_faces(vertices_wp, faces_wp, mesh_tm)


def test_revolve_invalid(device: str) -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        tw.creation.revolve(_ring(np.zeros((1, 2)), device))
    with pytest.raises(ValueError, match="sections must be at least 1"):
        tw.creation.revolve(_ring(_SQUARE_RING, device), sections=0)


def test_revolve_absolute_tolerance_is_scale_dependent(device: str) -> None:
    # Documented limitation: the degenerate-triangle filter compares an absolute area against
    # 1e-8, so a large enough sphere keeps its (near-)zero-area polar triangles. Recorded here so
    # the threshold in revolve's Notes stays honest rather than asserted as desirable.
    small_v, small_f = tw.creation.uv_sphere(radius=1.0, count=(8, 8), device=device)
    large_v, large_f = tw.creation.uv_sphere(radius=1.0e5, count=(8, 8), device=device)
    _assert_closed(small_v, small_f)
    assert int(large_f.shape[0]) >= int(small_f.shape[0])
    assert np.allclose(np.linalg.norm(large_v.numpy(), axis=1), 1.0e5, rtol=1e-5, atol=1.0)


# --- extrusion and polygons -------------------------------------------------------------


@pytest.mark.parametrize("ring_name", ["square", "L"])
@pytest.mark.parametrize("height", [0.5, -0.5])
def test_extrude_polygon(device: str, ring_name: str, height: float) -> None:
    shapely = pytest.importorskip("shapely.geometry")
    ring_np = _SQUARE_RING if ring_name == "square" else _L_RING
    vertices_wp, faces_wp = tw.creation.extrude_polygon(_ring(ring_np, device), height)
    mesh_tm = tm.creation.extrude_polygon(shapely.Polygon(ring_np), height)
    assert int(vertices_wp.shape[0]) == 2 * ring_np.shape[0]
    # Both signs of height must give an outward-facing solid of the same volume.
    _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
    _assert_closed(vertices_wp, faces_wp)


def test_extrude_polygon_mid_plane(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.extrude_polygon(
        _ring(_SQUARE_RING, device), 1.0, mid_plane=True
    )
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds[:, 2], [-0.5, 0.5], atol=1e-5)


def test_extrude_triangulation_recovers_subdivided_boundary(device: str) -> None:
    # A boundary edge split by an extra collinear vertex still has to become two wall quads, which
    # is why the boundary is recovered from the triangulation rather than taken from the input ring.
    ring_np = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(ring_np, device))
    solid_v, solid_f = tw.creation.extrude_triangulation(vertices_wp, faces_wp, 0.5)
    _assert_closed(solid_v, solid_f)
    assert int(solid_f.shape[0]) // 3 == 2 * 3 + 2 * 5
    assert np.isclose(_mesh(solid_v, solid_f).volume, 1.0, rtol=1e-4)


def test_extrude_triangulation_invalid(device: str) -> None:
    ring_wp, faces_wp = tw.creation.triangulate_polygon(_ring(_SQUARE_RING, device))
    with pytest.raises(ValueError, match="height must be nonzero"):
        tw.creation.extrude_triangulation(ring_wp, faces_wp, 0.0)
    with pytest.raises(ValueError, match="multiple of 3"):
        tw.creation.extrude_triangulation(ring_wp, faces_wp[:2].contiguous(), 1.0)


@pytest.mark.parametrize("ring_name", ["square", "L"])
def test_triangulate_polygon(device: str, ring_name: str) -> None:
    ring_np = _SQUARE_RING if ring_name == "square" else _L_RING
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(ring_np, device))
    assert int(vertices_wp.shape[0]) == ring_np.shape[0]
    assert int(faces_wp.shape[0]) // 3 == ring_np.shape[0] - 2
    # No Steiner points, and the triangles must tile the polygon exactly.
    triangles_np = vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)]
    edge_a, edge_b = (
        triangles_np[:, 1] - triangles_np[:, 0],
        triangles_np[:, 2] - triangles_np[:, 0],
    )
    area_np = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]).sum()
    exact_area = 2.0 if ring_name == "square" else 3.0
    assert np.isclose(area_np, exact_area, rtol=1e-5)


def _star_ring(n: int, inner: float = 0.45) -> np.ndarray:
    """Alternating-radius star: every other vertex is reflex, so no ear has an ear-free ring-2."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, inner)
    return np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))


@pytest.mark.parametrize("n", [16, 64, 512])
def test_triangulate_polygon_star(device: str, n: int) -> None:
    """
    A star ring is the worst case for the ear clipper's independent-set rule.

    Half its vertices are reflex and the convex ones alternate, so competing ears sit exactly two
    apart around the ring -- the configuration that made a raw-index rank clip one ear per round.
    """
    ring_np = _star_ring(n)
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(ring_np, device))
    assert int(vertices_wp.shape[0]) == n
    assert int(faces_wp.shape[0]) // 3 == n - 2

    triangles_np = vertices_wp.numpy().astype(np.float64)[faces_wp.numpy().reshape(-1, 3)]
    edge_a = triangles_np[:, 1] - triangles_np[:, 0]
    edge_b = triangles_np[:, 2] - triangles_np[:, 0]
    signed_np = 0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    # Consistent winding: every triangle turns the same way as the ring, so no signed area flips.
    assert np.all(signed_np > 0.0) or np.all(signed_np < 0.0)
    # And they tile the star exactly (shoelace over the ring).
    shoelace = 0.5 * abs(
        np.dot(ring_np[:, 0], np.roll(ring_np[:, 1], -1))
        - np.dot(np.roll(ring_np[:, 0], -1), ring_np[:, 1])
    )
    assert np.isclose(np.abs(signed_np).sum(), shoelace, rtol=1e-5)


def test_triangulate_polygon_near_collinear(device: str) -> None:
    # A ring whose interior vertices are almost on the line back to the start: every ear test is
    # decided by a near-zero cross product, so this is where a ranking change could stall.
    n = 64
    x_np = np.linspace(0.0, 1.0, n - 1)
    ring_np = np.vstack(
        (np.column_stack((x_np, 1e-7 * np.sin(np.pi * x_np))), np.array([[0.5, -0.25]]))
    )
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(ring_np, device))
    assert int(vertices_wp.shape[0]) == n
    # A degenerate ring may yield a partial triangulation, but never more than n - 2 faces and
    # never a hang: the round cap is the guarantee being checked here.
    assert 0 < int(faces_wp.shape[0]) // 3 <= n - 2


def test_triangulate_polygon_drops_repeated_closing_point(device: str) -> None:
    closed_np = np.vstack((_SQUARE_RING, _SQUARE_RING[:1]))
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(closed_np, device))
    assert int(vertices_wp.shape[0]) == _SQUARE_RING.shape[0]
    assert int(faces_wp.shape[0]) // 3 == _SQUARE_RING.shape[0] - 2


def test_triangulate_polygon_too_few_points(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.triangulate_polygon(_ring(np.zeros((2, 2)), device))
    assert int(vertices_wp.shape[0]) == 2
    assert int(faces_wp.shape[0]) == 0


_SWEEP_PATHS = {
    "straight": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
    "bent": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 2.0, 2.0]]),
    "closed": np.column_stack(
        (
            np.cos(2.0 * np.pi * np.arange(9) / 8),
            np.sin(2.0 * np.pi * np.arange(9) / 8),
            np.zeros(9),
        )
    )
    * 2.0,
}


@pytest.mark.parametrize("path_name", sorted(_SWEEP_PATHS))
def test_sweep_polygon(device: str, path_name: str) -> None:
    shapely = pytest.importorskip("shapely.geometry")
    ring_np = np.array([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]])
    path_np = _SWEEP_PATHS[path_name]
    vertices_wp, faces_wp = tw.creation.sweep_polygon(
        _ring(ring_np, device), wp.array(path_np.astype(np.float32), dtype=wp.vec3, device=device)
    )
    mesh_tm = tm.creation.sweep_polygon(shapely.Polygon(ring_np), path_np)
    _assert_same_solid(vertices_wp, faces_wp, mesh_tm)
    _assert_closed(vertices_wp, faces_wp)
    assert _mesh(vertices_wp, faces_wp).body_count == 1


def test_sweep_polygon_angles_roll_the_profile(device: str) -> None:
    shapely = pytest.importorskip("shapely.geometry")
    ring_np = np.array([[-0.5, -0.1], [0.5, -0.1], [0.5, 0.1], [-0.5, 0.1]])
    # A quarter turn spread over four segments. Concentrating the same twist in a single segment
    # sweeps the profile through itself, and both libraries then report a negative volume for the
    # self-intersecting result — so the roll is kept gentle enough for the solid to stay valid.
    path_np = np.column_stack((np.zeros(5), np.zeros(5), np.linspace(0.0, 1.0, 5)))
    path_wp = wp.array(path_np.astype(np.float32), dtype=wp.vec3, device=device)
    angles_np = np.linspace(0.0, np.pi / 2.0, 5)
    straight_v, _ = tw.creation.sweep_polygon(_ring(ring_np, device), path_wp)
    twisted_v, twisted_f = tw.creation.sweep_polygon(
        _ring(ring_np, device),
        path_wp,
        angles=wp.array(angles_np.astype(np.float32), device=device),
    )
    assert not np.allclose(straight_v.numpy(), twisted_v.numpy(), atol=1e-3)
    _assert_closed(twisted_v, twisted_f)
    _assert_same_solid(
        twisted_v,
        twisted_f,
        tm.creation.sweep_polygon(shapely.Polygon(ring_np), path_np, angles=angles_np),
    )


def test_sweep_polygon_open_path_without_caps(device: str) -> None:
    ring_np = np.array([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]])
    path_wp = wp.array(_SWEEP_PATHS["straight"].astype(np.float32), dtype=wp.vec3, device=device)
    _, faces_wp = tw.creation.sweep_polygon(_ring(ring_np, device), path_wp, cap=False)
    assert not tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert int(faces_wp.shape[0]) // 3 == 2 * 2 * 4


def test_sweep_polygon_invalid(device: str) -> None:
    ring_wp = _ring(_SQUARE_RING, device)
    single_wp = wp.array(np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="at least 2 points"):
        tw.creation.sweep_polygon(ring_wp, single_wp)
    path_wp = wp.array(_SWEEP_PATHS["straight"].astype(np.float32), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="one entry per path point"):
        tw.creation.sweep_polygon(
            ring_wp, path_wp, angles=wp.zeros(2, dtype=wp.float32, device=device)
        )


# --- composites -------------------------------------------------------------------------


def _triangle_soup(device: str, seed: int = 7) -> tuple[np.ndarray, wp.array, wp.array]:
    triangles_np = np.random.default_rng(seed).random((5, 3, 3)) + np.array([0.0, 0.0, 1.0])
    vertices_wp = wp.array(
        np.ascontiguousarray(triangles_np.reshape(-1, 3), dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.arange(15, dtype=np.int32), dtype=wp.int32, device=device)
    return triangles_np, vertices_wp, faces_wp


def test_truncated_prisms(device: str) -> None:
    triangles_np, vertices_wp, faces_wp = _triangle_soup(device)
    prism_v, prism_f = tw.creation.truncated_prisms(vertices_wp, faces_wp)
    mesh_tm = tm.creation.truncated_prisms(triangles_np)
    assert int(prism_v.shape[0]) == 6 * 5
    assert int(prism_f.shape[0]) // 3 == 8 * 5
    assert np.isclose(_mesh(prism_v, prism_f).volume, mesh_tm.volume, rtol=1e-4)
    assert _mesh(prism_v, prism_f).body_count == 5


def test_truncated_prisms_plane(device: str) -> None:
    triangles_np, vertices_wp, faces_wp = _triangle_soup(device)
    origin_np, normal_np = np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0])
    prism_v, prism_f = tw.creation.truncated_prisms(
        vertices_wp,
        faces_wp,
        origin=wp.vec3(*origin_np.tolist()),
        normal=wp.vec3(*normal_np.tolist()),
    )
    mesh_tm = tm.creation.truncated_prisms(triangles_np, origin=origin_np, normal=normal_np)
    assert np.isclose(_mesh(prism_v, prism_f).volume, mesh_tm.volume, rtol=1e-4)


def test_truncated_prisms_reversed_winding(device: str) -> None:
    # A source triangle facing the plane needs its prism's winding reversed, or the body comes out
    # inside-out with negative volume.
    triangles_np, _, faces_wp = _triangle_soup(device)
    flipped_np = np.ascontiguousarray(triangles_np[:, ::-1, :])
    vertices_wp = wp.array(
        np.ascontiguousarray(flipped_np.reshape(-1, 3), dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    prism_v, prism_f = tw.creation.truncated_prisms(vertices_wp, faces_wp)
    assert _mesh(prism_v, prism_f).volume > 0.0
    assert np.isclose(
        _mesh(prism_v, prism_f).volume, tm.creation.truncated_prisms(flipped_np).volume, rtol=1e-4
    )


def test_truncated_prisms_requires_normal_with_origin(device: str) -> None:
    _, vertices_wp, faces_wp = _triangle_soup(device)
    with pytest.raises(ValueError, match="normal is required"):
        tw.creation.truncated_prisms(vertices_wp, faces_wp, origin=wp.vec3(0.0, 0.0, 0.0))


def test_axis(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.axis(device=device)
    ball_v, ball_f = tw.creation.icosphere(radius=0.04, device=device)
    shaft_v, shaft_f = tw.creation.cylinder(radius=0.008, height=0.4, device=device)
    assert int(vertices_wp.shape[0]) == int(ball_v.shape[0]) + 3 * int(shaft_v.shape[0])
    assert int(faces_wp.shape[0]) == int(ball_f.shape[0]) + 3 * int(shaft_f.shape[0])
    # One shaft runs out to axis_length along each of X, Y and Z.
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds[1], 0.4, rtol=1e-3, atol=1e-3)
    assert np.allclose(_mesh(vertices_wp, faces_wp).bounds[0], -0.04, rtol=1e-3, atol=1e-3)


def test_axis_transform(device: str) -> None:
    matrix_np = tm.transformations.rotation_matrix(np.deg2rad(90.0), [1.0, 0.0, 0.0])
    matrix_np[:3, 3] = np.array([1.0, 0.0, 0.0])
    vertices_wp, faces_wp = tw.creation.axis(transform=_mat44(matrix_np), device=device)
    expected_np = tm.transform_points(
        tw.creation.axis(device=device)[0].numpy().astype(np.float64), matrix_np
    )
    assert np.allclose(vertices_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)
    assert int(faces_wp.shape[0]) > 0


def test_random_soup(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.random_soup(50, seed=3, device=device)
    assert int(vertices_wp.shape[0]) == 150
    assert np.array_equal(faces_wp.numpy(), np.arange(150, dtype=np.int32))
    assert vertices_wp.numpy().min() >= -0.5
    assert vertices_wp.numpy().max() <= 0.5
    assert np.array_equal(
        vertices_wp.numpy(), tw.creation.random_soup(50, seed=3, device=device)[0].numpy()
    )
    assert not np.array_equal(
        vertices_wp.numpy(), tw.creation.random_soup(50, seed=4, device=device)[0].numpy()
    )


def test_empty_results(device: str) -> None:
    vertices_wp, faces_wp = tw.creation.random_soup(0, seed=1, device=device)
    assert int(vertices_wp.shape[0]) == 0
    assert int(faces_wp.shape[0]) == 0

    empty_v = wp.empty(0, dtype=wp.vec3, device=device)
    empty_f = wp.empty(0, dtype=wp.int32, device=device)
    prism_v, prism_f = tw.creation.truncated_prisms(empty_v, empty_f)
    assert int(prism_v.shape[0]) == 0
    assert int(prism_f.shape[0]) == 0

    solid_v, solid_f = tw.creation.extrude_triangulation(
        wp.empty(0, dtype=wp.vec2, device=device), empty_f, 1.0
    )
    assert int(solid_v.shape[0]) == 0
    assert int(solid_f.shape[0]) == 0
