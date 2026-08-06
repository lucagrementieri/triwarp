"""Regression tests for ``triwarp.combine``."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def _fillable_loops(vertices: wp.array, faces: wp.array) -> list[wp.array]:
    return [loop for loop in tw.boundary.boundary_loops(vertices, faces) if int(loop.shape[0]) >= 3]


def _loop_sizes_of(vertices: wp.array, faces: wp.array) -> list[int]:
    return [int(loop.shape[0]) for loop in _fillable_loops(vertices, faces)]


def _loop_sizes(mesh_wp: wp.Mesh) -> list[int]:
    return _loop_sizes_of(mesh_wp.points, mesh_wp.indices)


def _cone(
    n: int,
    apex_z: float,
    rim_z: float,
    radius: float = 1.0,
    phase: float = 0.0,
    center_x: float = 0.0,
):
    """
    Open triangle-fan cone: an apex plus one rim circle. Its boundary is the rim loop.

    Returns ``(vertices_np, faces_np)`` with vertex 0 the apex and vertices ``1..n`` the rim,
    wound so the surface is consistently oriented. ``center_x`` shifts the rim laterally, which
    makes the two-rim correspondence non-monotone and exercises the LIS correction.
    """
    angles = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.column_stack(
        [center_x + radius * np.cos(angles), radius * np.sin(angles), np.full(n, rim_z)]
    )
    vertices = np.vstack([[center_x, 0.0, apex_z], rim]).astype(np.float64)
    apex_above_rim = apex_z > rim_z
    faces = np.empty((n, 3), dtype=np.int32)
    for i in range(n):
        first, second = 1 + i, 1 + (i + 1) % n
        faces[i] = (0, first, second) if apex_above_rim else (0, second, first)
    return vertices, faces.reshape(-1)


def _cone_wp(device: str, **kwargs):
    vertices_np, faces_np = _cone(**kwargs)
    vertices_wp = wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    return vertices_np, faces_np, vertices_wp, faces_wp


def _capsule_halves(device: str, n_a: int, n_b: int, phase: float = 0.0, offset: float = 0.0):
    """
    Two open cones whose rims are separated in ``z`` (a real frustum band, no overlap).

    ``offset`` shifts the top rim laterally so the rim-to-rim correspondence is non-monotone,
    exercising the LIS correction; the band stays non-degenerate.
    """
    bottom = _cone_wp(device, n=n_a, apex_z=-1.0, rim_z=0.0)
    top = _cone_wp(device, n=n_b, apex_z=1.5, rim_z=0.5, phase=phase, center_x=offset)
    return bottom, top


def _sorted_triangle_rows(faces_flat: np.ndarray) -> np.ndarray:
    triangles = np.sort(faces_flat.reshape(-1, 3), axis=1)
    return triangles[np.lexsort(triangles.T[::-1])]


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"),
    [
        (8, 8, 0.0, 0.0),
        (16, 11, 0.3, 0.0),
        (7, 13, 0.7, 0.0),
        (24, 5, 1.1, 0.0),
        (17, 11, 0.9, 1.2),
    ],
)
def test_stitch_watertight(device: str, n_a: int, n_b: int, phase: float, offset: float) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, n_a, n_b, phase, offset)

    assert not tw.validation.is_watertight(va, fa)
    assert not tw.validation.is_watertight(vb, fb)

    new_vertices, new_faces = tw.combine.stitch(va, fa, vb, fb)

    assert tw.validation.is_watertight(new_vertices, new_faces)
    assert tw.validation.is_winding_consistent(new_faces)

    # A + B faces plus one bridge triangle per rim edge on each side.
    n_new_faces = (int(new_faces.shape[0]) - int(fa.shape[0]) - int(fb.shape[0])) // 3
    assert n_new_faces == n_a + n_b
    assert int(new_vertices.shape[0]) == int(va.shape[0]) + int(vb.shape[0])


def test_stitch_argument_order_invariant(device: str) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, 16, 11, phase=0.3)

    vertices_ab, faces_ab = tw.combine.stitch(va, fa, vb, fb)
    vertices_ba, faces_ba = tw.combine.stitch(vb, fb, va, fa)

    # The larger loop is always A, so swapping the arguments yields the same mesh.
    assert np.array_equal(vertices_ab.numpy(), vertices_ba.numpy())
    assert np.array_equal(
        _sorted_triangle_rows(faces_ab.numpy()), _sorted_triangle_rows(faces_ba.numpy())
    )


def test_stitch_requires_single_boundary_watertight(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    _, _, va, fa = _cone_wp(device=str(mesh_wp.device), n=10, apex_z=-1.0, rim_z=0.0)

    # A watertight mesh has no boundary loop, so it cannot be stitched.
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.combine.stitch(mesh_wp.points, mesh_wp.indices, va, fa)


def test_stitch_requires_single_boundary_multi(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = half_torus
    _, _, va, fa = _cone_wp(device=str(mesh_wp.device), n=10, apex_z=-1.0, rim_z=0.0)

    assert len(_loop_sizes(mesh_wp)) >= 2
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.combine.stitch(mesh_wp.points, mesh_wp.indices, va, fa)


# --- Minimum-weight stitching (``stitch_min_weight``) ------------------------------------------

STITCH_METRICS = ["complex_stitch", "edge_length_stitch", "vertical"]
# complex_stitch (aspect + dihedral) and vertical (area/normal) are winding-invariant, so MeshLib's
# calcCombinedFillMetric re-scores them exactly; edge_length_stitch's |c-a| term is winding-order
# sensitive, so it is checked structurally only.
STITCH_COST_METRICS = ["complex_stitch", "vertical"]


def _meshlib_stitch_band(
    va_np: np.ndarray, fa_np: np.ndarray, vb_np: np.ndarray, fb_np: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MeshLib ``stitchHoles`` band for the same metric; returns ``(verts, orig_faces, band)``."""
    mr = pytest.importorskip("meshlib.mrmeshpy")
    mn = pytest.importorskip("meshlib.mrmeshnumpy")
    make_metric = {
        "complex_stitch": lambda m: mr.getComplexStitchMetric(m),
        "edge_length_stitch": lambda m: mr.getEdgeLengthStitchMetric(m),
        "vertical": lambda m: mr.getVerticalStitchMetric(m, mr.Vector3f(0.0, 0.0, 1.0)),
    }[metric]
    verts = np.ascontiguousarray(np.vstack([va_np, vb_np]), dtype=np.float32)
    orig = np.ascontiguousarray(np.vstack([fa_np, fb_np + len(va_np)]), dtype=np.int32)
    mesh = mn.meshFromFacesVerts(orig, verts)
    edges = mesh.topology.findHoleRepresentiveEdges()
    params = mr.StitchHolesParams()
    params.metric = make_metric(mesh)
    mr.stitchHoles(mesh, edges[0], edges[1], params)
    faces_out = mn.getNumpyFaces(mesh.topology)
    original = {tuple(sorted(int(x) for x in t)) for t in orig}
    band = np.array(
        [t for t in faces_out if tuple(sorted(int(x) for x in t)) not in original], np.int32
    )
    return verts, orig, band


def _meshlib_stitch_cost(
    verts: np.ndarray, orig: np.ndarray, band: np.ndarray, metric: str
) -> float:
    mr = pytest.importorskip("meshlib.mrmeshpy")
    mn = pytest.importorskip("meshlib.mrmeshnumpy")
    make_metric = {
        "complex_stitch": lambda m: mr.getComplexStitchMetric(m),
        "vertical": lambda m: mr.getVerticalStitchMetric(m, mr.Vector3f(0.0, 0.0, 1.0)),
    }[metric]
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    mesh_orig = mn.meshFromFacesVerts(np.ascontiguousarray(orig, np.int32), verts)
    metric_obj = make_metric(mesh_orig)
    full = np.ascontiguousarray(np.vstack([orig, band]), dtype=np.int32)
    mesh_full = mn.meshFromFacesVerts(full, verts)
    region_bools = np.zeros(len(full), dtype=bool)
    region_bools[len(orig) :] = True
    region = mn.faceBitSetFromBools(region_bools)
    return mr.calcCombinedFillMetric(mesh_full, region, metric_obj)


@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11), (8, 8)])
@pytest.mark.parametrize("metric", STITCH_METRICS)
def test_stitch_min_weight_watertight(device: str, n_a: int, n_b: int, metric: str) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, n_a, n_b)

    new_vertices, new_faces = tw.combine.stitch_min_weight(va, fa, vb, fb, metric=metric)

    # A band of exactly n_a + n_b triangles over the existing vertices, closing the two rims.
    n_band = (int(new_faces.shape[0]) - int(fa.shape[0]) - int(fb.shape[0])) // 3
    assert n_band == n_a + n_b
    assert int(new_vertices.shape[0]) == int(va.shape[0]) + int(vb.shape[0])
    assert tw.validation.is_winding_consistent(new_faces)
    filled_tm = tm.Trimesh(
        vertices=new_vertices.numpy(), faces=new_faces.numpy().reshape(-1, 3), process=False
    )
    assert filled_tm.is_watertight


@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11)])
@pytest.mark.parametrize("metric", STITCH_COST_METRICS)
def test_stitch_min_weight_matches_meshlib(device: str, n_a: int, n_b: int, metric: str) -> None:
    (va_np, fa_np, va, fa), (vb_np, fb_np, vb, fb) = _capsule_halves(device, n_a, n_b)

    new_vertices, new_faces = tw.combine.stitch_min_weight(va, fa, vb, fb, metric=metric)
    n_orig = int(fa.shape[0]) + int(fb.shape[0])  # flat length of the two original face buffers
    band_tw = new_faces.numpy()[n_orig:].reshape(-1, 3)
    fa_rows, fb_rows = fa_np.reshape(-1, 3), fb_np.reshape(-1, 3)
    orig_tw = np.vstack([fa_rows, fb_rows + len(va_np)]).astype(np.int32)

    verts_ml, orig_ml, band_ml = _meshlib_stitch_band(va_np, fa_rows, vb_np, fb_rows, metric)

    # triwarp reaches MeshLib's exhaustive stitchHoles optimum (compare cost, not exact triangles).
    cost_tw = _meshlib_stitch_cost(new_vertices.numpy(), orig_tw, band_tw, metric)
    cost_ml = _meshlib_stitch_cost(verts_ml, orig_ml, band_ml, metric)
    assert np.isclose(cost_tw, cost_ml, rtol=3e-3, atol=1e-3)


def test_stitch_min_weight_requires_single_boundary(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    _, _, vb, fb = _cone_wp(device=mesh_wp.device, n=8, apex_z=1.0, rim_z=0.5)
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.combine.stitch_min_weight(mesh_wp.points, mesh_wp.indices, vb, fb)


def test_stitch_min_weight_rejects_unknown_metric(device: str) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, 8, 8)
    with pytest.raises(ValueError, match="metric must be one of"):
        tw.combine.stitch_min_weight(va, fa, vb, fb, metric="bogus")


# ---------------------------------------------------------------------------
# stitch_smooth (MeshLib stitchHolesNicely)
# ---------------------------------------------------------------------------


def _skip_cpu(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("stitch_smooth subdivision/smoothing requires CUDA (warp.optim.linear.cg)")


def _hemisphere_pair(device: str):
    """Two facing hemispheres (single boundary loop each) for stitch tests."""
    meshes = []
    for z_sign, z_off in ((1.0, 0.6), (-1.0, -0.6)):
        sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
        cap = sphere.slice_plane(
            plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, z_sign]), cap=False
        )
        cap.merge_vertices()
        cap.apply_translation([0.0, 0.0, z_off])
        v = wp.array(
            np.ascontiguousarray(cap.vertices.astype(np.float64)), dtype=wp.vec3, device=device
        )
        f = wp.array(
            np.ascontiguousarray(cap.faces.astype(np.int32).reshape(-1)),
            dtype=wp.int32,
            device=device,
        )
        meshes.append((v, f))
    return meshes[0], meshes[1]


def test_stitch_smooth_watertight(device: str):
    _skip_cpu(device)
    (va, fa), (vb, fb) = _hemisphere_pair(device)
    n_v0 = int(va.shape[0]) + int(vb.shape[0])

    new_vertices, new_faces = tw.combine.stitch_smooth(va, fa, vb, fb)
    verts_np = new_vertices.numpy()
    mesh_tm = tm.Trimesh(verts_np, new_faces.numpy().reshape(-1, 3), process=False)

    assert len(_loop_sizes_of(new_vertices, new_faces)) == 0
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent
    # Original vertices are the concatenated prefix, unchanged.
    assert np.allclose(verts_np[:n_v0], np.concatenate([va.numpy(), vb.numpy()]), atol=1e-6)


def test_stitch_smooth_requires_single_loop(device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.combine.stitch_smooth(mesh_wp.points, mesh_wp.indices, mesh_wp.points, mesh_wp.indices)
