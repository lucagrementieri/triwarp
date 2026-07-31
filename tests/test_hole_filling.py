"""Regression tests for ``triwarp.hole_filling``."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_open3d, trimesh_to_pymeshlab
from triwarp.hole_filling import _non_increasing_indices

# Open-surface fixtures that actually have a boundary to fill.
OPEN_MESHES = ["hemisphere", "half_torus"]


def _fillable_loops(vertices: wp.array, faces: wp.array) -> list[wp.array]:
    return [loop for loop in tw.boundary.boundary_loops(vertices, faces) if int(loop.shape[0]) >= 3]


def _loop_sizes_of(vertices: wp.array, faces: wp.array) -> list[int]:
    return [int(loop.shape[0]) for loop in _fillable_loops(vertices, faces)]


def _loop_perimeters_of(vertices: wp.array, faces: wp.array) -> list[float]:
    return [
        tw.polyline.closed_polyline_length(tw.array.gather(vertices, loop))
        for loop in _fillable_loops(vertices, faces)
    ]


def _loop_sizes(mesh_wp: wp.Mesh) -> list[int]:
    return _loop_sizes_of(mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_fill_holes_fan_watertight(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    assert not tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices)

    filled_faces = tw.hole_filling.fill_holes_fan(mesh_wp.points, mesh_wp.indices)

    assert tw.validation.is_watertight(mesh_wp.points, filled_faces)
    assert tw.validation.is_winding_consistent(filled_faces)

    # A fan adds B - 2 triangles per loop; cross-check against the trimesh reference count.
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(size - 2 for size in _loop_sizes(mesh_wp))

    n_faces_before = len(mesh_tm.faces)
    tm_repair.fill_holes(mesh_tm, use_fan=True)
    assert n_new_faces == len(mesh_tm.faces) - n_faces_before


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_fill_holes_cone_watertight(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)

    loop_sizes = _loop_sizes(mesh_wp)
    n_vertices_before = int(mesh_wp.points.shape[0])

    new_vertices, filled_faces = tw.hole_filling.fill_holes_cone(mesh_wp.points, mesh_wp.indices)

    assert tw.validation.is_watertight(new_vertices, filled_faces)
    assert tw.validation.is_winding_consistent(filled_faces)

    # A cone adds one centroid vertex and B triangles per loop.
    assert int(new_vertices.shape[0]) == n_vertices_before + len(loop_sizes)
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(loop_sizes)

    # Every face index references a valid (original or new centroid) vertex.
    faces_np = filled_faces.numpy()
    assert faces_np.min() >= 0
    assert faces_np.max() < int(new_vertices.shape[0])


def test_fill_holes_centroid_position(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = hemisphere

    loops = tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices)
    loop_vertices_np = mesh_wp.points.numpy()[loops[0].numpy()]
    centroid_expected = loop_vertices_np.mean(axis=0)

    new_vertices, _ = tw.hole_filling.fill_holes_cone(mesh_wp.points, mesh_wp.indices)
    centroid_wp = new_vertices.numpy()[int(mesh_wp.points.shape[0])]

    assert np.allclose(centroid_wp, centroid_expected, rtol=1e-4, atol=1e-4)


def test_fill_holes_watertight_mesh_unchanged(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron

    filled_faces = tw.hole_filling.fill_holes_fan(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())

    new_vertices, cone_faces = tw.hole_filling.fill_holes_cone(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(cone_faces.numpy(), mesh_wp.indices.numpy())
    assert int(new_vertices.shape[0]) == int(mesh_wp.points.shape[0])


def test_fill_holes_fan_preserve_largest(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = half_torus

    loop_sizes = _loop_sizes(mesh_wp)
    perimeters = _loop_perimeters_of(mesh_wp.points, mesh_wp.indices)
    # The fixture must have several holes for "preserve the largest" to be meaningful.
    assert len(loop_sizes) >= 2

    # The preserved loop is the one with the greatest perimeter, not the most vertices.
    preserved = int(np.argmax(perimeters))
    preserved_size = loop_sizes[preserved]

    filled_faces = tw.hole_filling.fill_holes_fan(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )

    # Every hole but the longest-perimeter one is fanned (B - 2 triangles each); it stays open.
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(loop_sizes) - 2 * len(loop_sizes) - (preserved_size - 2)

    remaining_perimeters = _loop_perimeters_of(mesh_wp.points, filled_faces)
    assert len(remaining_perimeters) == 1
    assert np.isclose(remaining_perimeters[0], perimeters[preserved], rtol=1e-5, atol=1e-5)


def test_fill_holes_cone_preserve_largest(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = half_torus

    loop_sizes = _loop_sizes(mesh_wp)
    perimeters = _loop_perimeters_of(mesh_wp.points, mesh_wp.indices)
    assert len(loop_sizes) >= 2
    n_vertices_before = int(mesh_wp.points.shape[0])

    preserved = int(np.argmax(perimeters))
    preserved_size = loop_sizes[preserved]

    new_vertices, filled_faces = tw.hole_filling.fill_holes_cone(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )

    # One centroid per filled hole (all but the longest-perimeter one); B triangles per filled hole.
    assert int(new_vertices.shape[0]) == n_vertices_before + len(loop_sizes) - 1
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(loop_sizes) - preserved_size

    remaining_perimeters = _loop_perimeters_of(new_vertices, filled_faces)
    assert len(remaining_perimeters) == 1
    assert np.isclose(remaining_perimeters[0], perimeters[preserved], rtol=1e-5, atol=1e-5)


def test_fill_holes_preserve_largest_single_hole_unchanged(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = hemisphere

    # With one hole, preserving the largest leaves nothing to fill.
    assert len(_loop_sizes(mesh_wp)) == 1

    filled_faces = tw.hole_filling.fill_holes_fan(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())

    new_vertices, cone_faces = tw.hole_filling.fill_holes_cone(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    assert np.array_equal(cone_faces.numpy(), mesh_wp.indices.numpy())
    assert int(new_vertices.shape[0]) == int(mesh_wp.points.shape[0])


def test_fill_holes_empty_mesh(device: str) -> None:
    vertices = wp.empty(0, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)

    assert int(tw.hole_filling.fill_holes_fan(vertices, faces).shape[0]) == 0

    new_vertices, new_faces = tw.hole_filling.fill_holes_cone(vertices, faces)
    assert int(new_vertices.shape[0]) == 0
    assert int(new_faces.shape[0]) == 0


# --- Minimum-weight hole triangulation (``fill_holes_min_weight``) ---------------------------

_BAD = 1e10  # kernel ``BAD_METRIC``


def _circumcircle_diameter(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.dot(b - a, b - a))
    ca = float(np.dot(a - c, a - c))
    bc = float(np.dot(c - b, c - b))
    if ab <= 0.0:
        return float(np.sqrt(ca))
    if ca <= 0.0:
        return float(np.sqrt(bc))
    if bc <= 0.0:
        return float(np.sqrt(ab))
    n = np.cross(b - a, c - a)
    f = float(np.dot(n, n))
    if f <= 0.0:
        return np.inf
    return float(np.sqrt(ab * ca * bc / f))


def _triangle_aspect_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    bc = float(np.linalg.norm(c - b))
    ca = float(np.linalg.norm(a - c))
    ab = float(np.linalg.norm(b - a))
    half_perimeter = (bc + ca + ab) * 0.5
    den = 8.0 * (half_perimeter - bc) * (half_perimeter - ca) * (half_perimeter - ab)
    if den <= 0.0:
        return np.inf
    return bc * ca * ab / den


FILL_METRICS = [
    "plane_normalized",
    "min_area",
    "circumscribed",
    "plane",
    "min_tri_angle",
    "edge_length",
    "universal",
    "max_dihedral",
    "complex_fill",
]
_COMBINE_MAX = {"max_dihedral"}  # metrics that accumulate with max instead of sum


def _min_triangle_angle_sin(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.linalg.norm(b - a))
    ca = float(np.linalg.norm(a - c))
    bc = float(np.linalg.norm(c - b))
    if ab <= 0.0 or ca <= 0.0 or bc <= 0.0:
        return 0.0
    return float(np.linalg.norm(np.cross(b - a, c - a)) * min(ab, ca, bc) / (ab * ca * bc))


def _dihedral(left: np.ndarray, right: np.ndarray, edge: np.ndarray) -> float:
    edge_dir = edge / np.linalg.norm(edge)
    return float(np.arctan2(np.dot(edge_dir, np.cross(left, right)), np.dot(left, right)))


def _tri_term(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, normal: np.ndarray, char_area: float, metric: str
) -> float:
    """Per-triangle term, pure-NumPy mirror of ``kernels.hole_filling.triangle_fill_metric``."""
    a, b, c = (x.astype(np.float32) for x in (a, b, c))
    if metric == "min_area":
        return float(np.linalg.norm(np.cross(b - a, c - a)))
    if metric == "circumscribed" or metric == "universal":
        return _circumcircle_diameter(a, b, c)
    if metric == "min_tri_angle":
        return float(np.exp(25.0 * (0.86602540378443864676 - _min_triangle_angle_sin(a, b, c))))
    if metric == "edge_length" or metric == "max_dihedral":
        return 0.0
    if metric == "complex_fill":
        aspect_ratio = _triangle_aspect_ratio(a, b, c)
        if aspect_ratio > _BAD:
            return _BAD
        return aspect_ratio + 100.0 * float(np.linalg.norm(np.cross(b - a, c - a))) * char_area
    if metric == "plane":
        if float(np.dot(normal.astype(np.float32), np.cross(b - a, c - a))) < 0.0:
            return _BAD
        return _circumcircle_diameter(a, b, c)
    # plane_normalized
    face_norm = np.cross(b - a, c - a)
    face_dbl_area_sq = float(np.dot(face_norm, face_norm))
    if face_dbl_area_sq == 0.0:
        return _BAD
    dot_res = float(np.dot(normal.astype(np.float32), face_norm))
    if dot_res < 0.0 or dot_res * dot_res * 4.0 < face_dbl_area_sq:
        return _BAD
    aspect_ratio = _triangle_aspect_ratio(a, b, c)
    if aspect_ratio > _BAD:
        return _BAD
    return _circumcircle_diameter(a, b, c) * aspect_ratio


def _edge_term(
    a: np.ndarray, b: np.ndarray, lft: np.ndarray, rgt: np.ndarray, metric: str
) -> float:
    """Per-edge term, pure-NumPy mirror of ``kernels.hole_filling.fill_edge_term``."""
    a, b, lft, rgt = (x.astype(np.float32) for x in (a, b, lft, rgt))
    if metric == "edge_length":
        return float(np.linalg.norm(b - a))
    ab = b - a
    if metric == "universal":
        norm_l = np.cross(lft - a, ab)
        norm_r = np.cross(ab, rgt - a)
        dbl_area = float(np.linalg.norm(norm_l) + np.linalg.norm(norm_r))
        return float(np.sqrt(dbl_area) * np.exp(5.0 * abs(_dihedral(norm_l, norm_r, ab))))
    if metric == "max_dihedral":
        return abs(_dihedral(np.cross(lft - a, ab), np.cross(ab, rgt - a), ab))
    if metric == "complex_fill":
        norm_a = np.cross(rgt - b, -ab)
        norm_c = np.cross(lft - a, ab)
        denom = float(np.linalg.norm(norm_a) * np.linalg.norm(norm_c))
        if denom == 0.0:
            return _BAD
        cos_ac = float(np.dot(norm_a, norm_c) / denom)
        if cos_ac <= -1.0:
            return _BAD
        return ((1.0 - cos_ac) / (1.0 + cos_ac)) ** 4
    return 0.0


def _total_fill_metric(
    vertices: wp.array, faces: wp.array, fill_flat: np.ndarray, metric: str
) -> float:
    """
    Score a fill triangulation exactly as ``fill_dp_span`` accumulates it.

    Validated against MeshLib's own ``calcCombinedFillMetric``: per-triangle terms plus per-edge
    (dihedral) terms — interior edges use both adjacent fill apexes, rim edges the existing face's
    opposite vertex (``smoothBd``).
    """
    from collections import defaultdict

    vertices_np = vertices.numpy()
    fill = fill_flat.reshape(-1, 3)
    is_max = metric in _COMBINE_MAX

    loops = _fillable_loops(vertices, faces)
    loop_of: dict[int, int] = {}
    normals: list[np.ndarray] = []
    char_areas: list[float] = []
    for li, loop_wp in enumerate(loops):
        loop = loop_wp.numpy()
        for vtx in loop:
            loop_of[int(vtx)] = li
        loop_pos = tw.array.gather(vertices, loop_wp)
        normals.append(np.asarray(tw.polyline.polyline_normal(loop_pos)))
        pos = vertices_np[loop]
        rim = np.roll(pos, -1, axis=0) - pos
        max_sq = float(np.max(np.einsum("ij,ij->i", rim, rim)))
        char_areas.append(1.0 / max_sq if max_sq > 0.0 else 1.0)

    def edge_map(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = defaultdict(list)
        for tri in triangles:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v, w in ((a, b, c), (b, c, a), (a, c, b)):
                out[(u, v) if u < v else (v, u)].append(w)
        return out

    fill_apex = edge_map(fill)
    orig_third = edge_map(faces.numpy().reshape(-1, 3))

    total = 0.0
    for tri in fill:
        li = loop_of[int(tri[0])]
        a, b, c = (vertices_np[int(x)] for x in tri)
        # Emitted triangles are reverse-wound vs. the DP scoring; take the accepted orientation.
        term = min(
            _tri_term(a, b, c, normals[li], char_areas[li], metric),
            _tri_term(a, c, b, normals[li], char_areas[li], metric),
        )
        total = max(total, term) if is_max else total + term
    for edge, apexes in fill_apex.items():
        pa, pb = vertices_np[edge[0]], vertices_np[edge[1]]
        if len(apexes) == 2:
            left, right = vertices_np[apexes[0]], vertices_np[apexes[1]]
        elif len(apexes) == 1:
            others = orig_third.get(edge, [])
            if len(others) != 1:
                continue
            left, right = vertices_np[others[0]], vertices_np[apexes[0]]
        else:
            continue
        term = _edge_term(pa, pb, left, right, metric)
        total = max(total, term) if is_max else total + term
    return total


def _meshlib_fill_triangles(vertices: wp.array, faces: wp.array, metric: str) -> np.ndarray:
    """
    Fill triangles produced by MeshLib's ``fillHole`` for the same metric, as a flat vertex buffer.

    ``maxPolygonSubdivisions`` is raised so MeshLib runs the exhaustive DP (no large-hole
    sub-sampling), matching [`fill_holes_min_weight`]'s full search. fillHole reuses existing
    vertices, so the new faces are those not present in the original triangle set.
    """
    mr = pytest.importorskip("meshlib.mrmeshpy")
    mn = pytest.importorskip("meshlib.mrmeshnumpy")
    make_metric = {
        "plane_normalized": lambda m, e: mr.getPlaneNormalizedFillMetric(m, e),
        "min_area": lambda m, e: mr.getMinAreaMetric(m),
        "circumscribed": lambda m, e: mr.getCircumscribedMetric(m),
        "plane": lambda m, e: mr.getPlaneFillMetric(m, e),
        "min_tri_angle": lambda m, e: mr.getMinTriAngleMetric(m),
        "edge_length": lambda m, e: mr.getEdgeLengthFillMetric(m),
        "universal": lambda m, e: mr.getUniversalMetric(m),
        "max_dihedral": lambda m, e: mr.getMaxDihedralAngleMetric(m),
        "complex_fill": lambda m, e: mr.getComplexFillMetric(m, e),
    }[metric]
    vertices_np = np.ascontiguousarray(vertices.numpy(), dtype=np.float32)
    faces_np = np.ascontiguousarray(faces.numpy().reshape(-1, 3), dtype=np.int32)
    mesh = mn.meshFromFacesVerts(faces_np, vertices_np)
    params = mr.FillHoleParams()
    params.maxPolygonSubdivisions = 1000
    for edge in mesh.topology.findHoleRepresentiveEdges():
        params.metric = make_metric(mesh, edge)
        mr.fillHole(mesh, edge, params)
    faces_out = mn.getNumpyFaces(mesh.topology)
    original = {tuple(sorted(int(x) for x in tri)) for tri in faces_np}
    fill = [tri for tri in faces_out if tuple(sorted(int(x) for x in tri)) not in original]
    return np.asarray(fill, dtype=np.int32).reshape(-1)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parametrize("metric", FILL_METRICS)
def test_fill_holes_min_weight_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str, metric: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    n_orig = int(mesh_wp.indices.shape[0])

    tw_fill = tw.hole_filling.fill_holes_min_weight(
        mesh_wp.points, mesh_wp.indices, metric=metric
    ).numpy()[n_orig:]
    ml_fill = _meshlib_fill_triangles(mesh_wp.points, mesh_wp.indices, metric)

    # Same triangle count and same achieved optimum as MeshLib's exhaustive fillHole (the exact
    # triangulation can differ under ties / MeshLib's tie-breaking, so compare the cost).
    assert len(tw_fill) // 3 == len(ml_fill) // 3
    tw_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, tw_fill, metric)
    ml_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, ml_fill, metric)
    assert np.isclose(tw_total, ml_total, rtol=2e-3, atol=1e-3)


def test_fill_metric_scorer_matches_meshlib(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Anchor ``_total_fill_metric`` to MeshLib's own ``calcCombinedFillMetric`` (single-hole mesh).

    Validates that the test-side scorer used by ``test_fill_holes_min_weight_matches_meshlib`` truly
    computes each MeshLib metric, for the metrics that expose a triangle term
    (``calcCombinedFillMetric`` cannot score the edge-only metrics — it always calls
    ``triangleMetric``, which is empty for them).
    """
    mr = pytest.importorskip("meshlib.mrmeshpy")
    mn = pytest.importorskip("meshlib.mrmeshnumpy")
    _, mesh_wp = hemisphere
    faces_np = np.ascontiguousarray(mesh_wp.indices.numpy().reshape(-1, 3), dtype=np.int32)
    verts_np = np.ascontiguousarray(mesh_wp.points.numpy(), dtype=np.float32)
    make_metric = {
        "plane_normalized": lambda m, e: mr.getPlaneNormalizedFillMetric(m, e),
        "min_area": lambda m, e: mr.getMinAreaMetric(m),
        "circumscribed": lambda m, e: mr.getCircumscribedMetric(m),
        "plane": lambda m, e: mr.getPlaneFillMetric(m, e),
        "min_tri_angle": lambda m, e: mr.getMinTriAngleMetric(m),
        "universal": lambda m, e: mr.getUniversalMetric(m),
        "complex_fill": lambda m, e: mr.getComplexFillMetric(m, e),
    }
    for metric, factory in make_metric.items():
        fill = (
            tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices, metric=metric)
            .numpy()[faces_np.size :]
            .reshape(-1, 3)
        )
        mesh_orig = mn.meshFromFacesVerts(faces_np, verts_np)
        metric_obj = factory(mesh_orig, mesh_orig.topology.findHoleRepresentiveEdges()[0])
        full = np.vstack([faces_np, fill]).astype(np.int32)
        mesh_full = mn.meshFromFacesVerts(np.ascontiguousarray(full, np.int32), verts_np)
        region_bools = np.zeros(len(full), dtype=bool)
        region_bools[len(faces_np) :] = True
        region = mn.faceBitSetFromBools(region_bools)
        mr_cost = mr.calcCombinedFillMetric(mesh_full, region, metric_obj)
        my_cost = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fill.reshape(-1), metric)
        assert np.isclose(my_cost, mr_cost, rtol=2e-3, atol=1e-3), metric


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parametrize("metric", FILL_METRICS)
def test_fill_holes_min_weight_watertight(
    request: pytest.FixtureRequest, mesh_name: str, metric: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    loop_sizes = _loop_sizes(mesh_wp)

    filled_faces = tw.hole_filling.fill_holes_min_weight(
        mesh_wp.points, mesh_wp.indices, metric=metric
    )

    # No vertices added; every hole sealed with exactly B - 2 triangles.
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(size - 2 for size in loop_sizes)

    # Topologically closed and consistently wound (triwarp's own is_watertight false-positives on
    # coplanar/curved caps via its self-intersection test, so trimesh is the watertight oracle).
    assert tw.validation.is_edge_manifold(filled_faces, allow_boundary_edges=False)
    assert tw.validation.is_winding_consistent(filled_faces)
    assert len(_loop_sizes_of(mesh_wp.points, filled_faces)) == 0
    filled_tm = tm.Trimesh(
        vertices=mesh_wp.points.numpy(), faces=filled_faces.numpy().reshape(-1, 3), process=False
    )
    assert filled_tm.is_watertight
    assert filled_tm.is_winding_consistent


def _sealed_volume(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """
    Enclosed volume of a sealed mesh, after canonicalising the winding.

    The ``fix_winding`` pass is load-bearing for Open3D: ``fill_holes`` emits its cap triangles
    wound
    *against* the rest of the mesh, so a raw signed-volume read of its output is meaningless (-1.06
    on ``hemisphere``, where the true volume is 2.02). Repairing the winding first is the named
    transform that makes the three libraries' volumes comparable.
    """
    mesh_tm = tm.Trimesh(vertices_np, np.asarray(faces_np).reshape(-1, 3), process=False)
    tm_repair.fix_winding(mesh_tm)
    return float(mesh_tm.volume)


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("fill_holes_min_weight", "open3d", "pymeshlab")
def test_fill_holes_min_weight_matches_open3d_and_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C: three hole fillers pick three different triangulations of the same loop.

    All three seal the identical boundary with the identical budget -- ``B - 2`` triangles over the
    existing vertices only -- but no two agree triangle for triangle, so there is no correspondence
    to recover and the comparison has to be on the sealed *surface* plus the achieved optimum.
    Open3D's is read through the tensor API (``o3d.t`` ``fill_holes``) and MeshLab's through
    ``meshing_close_holes``, whose default ``maxholesize=30`` would close nothing on these rims, so
    the cap is lifted exactly as the benchmark lifts it.

    Three asserts. The counts are class A: same vertex count, same face count, and every added
    triangle drawn only from the boundary loops. The **volume** is class B under the named
    [`_sealed_volume`][tests.test_hole_filling._sealed_volume] transform -- these rims are planar,
    so the enclosed volume is triangulation-invariant and all three must agree exactly. And the
    **cost** is the optimality claim: triwarp runs the exact minimum-weight DP, so its
    ``plane_normalized`` cost must be no worse than either reference's triangulation of the same
    loop.

    **Bug class excluded:** a DP that finds a valid but *suboptimal* triangulation -- the failure
    the volume and count asserts are both blind to, since every triangulation of a planar rim has
    the same volume and the same triangle count.

    **Mutation probe, measured:** on ``hemisphere`` the costs are triwarp **290.7**, MeshLab 1 821.1
    (6.3x) and Open3D 827.3 (2.8x); on ``half_torus`` **686.0** against 2 740.6 (4.0x) and 1 127.9
    (1.6x). The bound is not one any answer passes: this module's own
    [`fill_holes_fan`][triwarp.hole_filling.fill_holes_fan] emits the same ``B - 2`` triangles over
    the same vertices and scores 860.1 and 3 509.5 -- it *fails* against Open3D on both fixtures.
    ``zeros_like`` and "return the input faces" fail at the face-count assert.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_tm = tm.Trimesh(
        vertices=mesh_wp.points.numpy().astype(np.float64),
        faces=mesh_wp.indices.numpy().reshape(-1, 3),
        process=False,
    )
    vertices_np = mesh_tm.vertices
    n_original = int(mesh_wp.indices.shape[0])

    filled_wp = tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices).numpy()

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    statistics_pml = meshset_pml.meshing_close_holes(maxholesize=1_000_000)
    assert statistics_pml["closed_holes"] > 0
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())
    assert np.allclose(meshset_pml.current_mesh().vertex_matrix(), vertices_np, atol=1e-6)

    # The tensor mesh must be held in a name: chaining ``from_legacy(...).fill_holes()`` lets the
    # temporary be collected and the result's ``positions`` then read freed memory (observed as
    # 2052.1 and 4.4e-41 in the first rows) rather than raising.
    tensor_o3d = o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_open3d(mesh_tm))
    mesh_o3d = tensor_o3d.fill_holes()
    faces_o3d = mesh_o3d.triangle["indices"].numpy()
    assert np.allclose(mesh_o3d.vertex["positions"].numpy(), vertices_np, atol=1e-6)

    loop_vertices = {
        int(vertex)
        for loop_wp in _fillable_loops(mesh_wp.points, mesh_wp.indices)
        for vertex in loop_wp.numpy()
    }
    cost_wp = _total_fill_metric(
        mesh_wp.points, mesh_wp.indices, filled_wp[n_original:], "plane_normalized"
    )
    for faces_ref in (faces_pml, faces_o3d):
        assert faces_ref.shape[0] == filled_wp.shape[0] // 3
        added_ref = np.asarray(faces_ref).reshape(-1)[n_original:].astype(np.int32)
        assert set(np.unique(added_ref).tolist()) <= loop_vertices
        assert np.isclose(
            _sealed_volume(vertices_np, filled_wp),
            _sealed_volume(vertices_np, faces_ref),
            rtol=1e-5,
        )
        cost_ref = _total_fill_metric(
            mesh_wp.points, mesh_wp.indices, added_ref, "plane_normalized"
        )
        assert cost_wp <= cost_ref, f"the minimum-weight DP scored worse: {cost_wp} > {cost_ref}"


def _punched_sphere(n_holes: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Icosphere with ``n_holes`` scattered single-triangle holes: many independent small loops."""
    mesh_tm = tm.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(seed)
    # Only drop faces whose one-rings are disjoint, so each hole stays a separate 3-vertex loop.
    blocked: set[int] = set()
    dropped: list[int] = []
    adjacency = mesh_tm.face_adjacency
    neighbors: dict[int, set[int]] = {face: set() for face in range(len(mesh_tm.faces))}
    for left, right in adjacency:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    for face in rng.permutation(len(mesh_tm.faces)):
        face = int(face)
        if len(dropped) == n_holes or face in blocked:
            continue
        dropped.append(face)
        blocked.add(face)
        blocked.update(neighbors[face])
        for near in neighbors[face]:
            blocked.update(neighbors[near])
    keep = np.setdiff1d(np.arange(len(mesh_tm.faces)), np.asarray(dropped))
    return mesh_tm.vertices, mesh_tm.faces[keep]


@pytest.mark.parametrize("metric", ["plane_normalized", "min_area", "universal"])
def test_fill_holes_min_weight_batched_equals_per_loop(device: str, metric: str) -> None:
    """
    Filling many holes in one call must match filling them one at a time.

    The interval DP is solved for every loop in the same launches over one ragged table, so this is
    the regression that would catch a loop bleeding into its neighbour's block -- an off-by-one in
    ``dp_offsets``, a chord marked against the wrong loop, or a min-area fallback firing for the
    whole batch instead of the loops that needed it.
    """
    vertices_np, faces_np = _punched_sphere(24)
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    loops = _fillable_loops(vertices_wp, faces_wp)
    assert len(loops) >= 8

    batched_np = tw.hole_filling.fill_loops(vertices_wp, faces_wp, loops, metric, True).numpy()
    per_loop = [
        tw.hole_filling.fill_loops(vertices_wp, faces_wp, [loop], metric, True).numpy()[
            int(faces_wp.shape[0]) :
        ]
        for loop in loops
    ]
    expected_np = np.concatenate([faces_wp.numpy(), *per_loop])
    assert np.array_equal(batched_np, expected_np)


def test_fill_holes_min_weight_optimal_vs_fan(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = hemisphere
    # Single-loop fixture: the DP minimizes the plane-normalized objective over all triangulations,
    # and the fan is one such triangulation, so the min-weight total must not exceed the fan's.
    n_orig = int(mesh_wp.indices.shape[0])
    fan_fill = tw.hole_filling.fill_holes_fan(mesh_wp.points, mesh_wp.indices).numpy()[n_orig:]
    mw_fill = tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices).numpy()[
        n_orig:
    ]

    fan_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, fan_fill, "plane_normalized")
    mw_total = _total_fill_metric(mesh_wp.points, mesh_wp.indices, mw_fill, "plane_normalized")
    assert mw_total <= fan_total + 1e-4


def test_fill_holes_min_weight_avoids_multiple_edges(device: str) -> None:
    # Two triangles sharing edge (0, 2); the boundary loop 0-1-2-3 has 0-2 as a pre-existing chord.
    vertices = wp.array(
        [[0.0, 0.0, 0.0], [0.5, 2.0, 0.0], [1.0, 0.0, 0.0], [0.5, -2.0, 0.0]],
        dtype=wp.vec3,
        device=device,
    )
    faces = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device)

    def edge_face_count(faces_flat: np.ndarray, u: int, v: int) -> int:
        return int(sum({u, v} <= set(tri) for tri in faces_flat.reshape(-1, 3).tolist()))

    resolved = tw.hole_filling.fill_holes_min_weight(vertices, faces, resolve_multiple_edges=True)
    # The forbidden diagonal (0, 2) keeps its two original faces; the other diagonal is used.
    assert edge_face_count(resolved.numpy(), 0, 2) == 2
    assert tw.validation.is_edge_manifold(resolved, allow_boundary_edges=True)

    unresolved = tw.hole_filling.fill_holes_min_weight(
        vertices, faces, resolve_multiple_edges=False
    )
    # Free to reuse the geometrically preferred diagonal (0, 2), creating a non-manifold edge.
    assert edge_face_count(unresolved.numpy(), 0, 2) == 4
    assert not tw.validation.is_edge_manifold(unresolved, allow_boundary_edges=True)


def test_fill_holes_min_weight_watertight_mesh_unchanged(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    filled_faces = tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())


def test_fill_holes_min_weight_preserve_largest(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = half_torus
    loop_sizes = _loop_sizes(mesh_wp)
    perimeters = _loop_perimeters_of(mesh_wp.points, mesh_wp.indices)
    assert len(loop_sizes) >= 2
    preserved_size = loop_sizes[int(np.argmax(perimeters))]

    filled_faces = tw.hole_filling.fill_holes_min_weight(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    n_new_faces = (int(filled_faces.shape[0]) - int(mesh_wp.indices.shape[0])) // 3
    assert n_new_faces == sum(size - 2 for size in loop_sizes) - (preserved_size - 2)
    assert len(_loop_sizes_of(mesh_wp.points, filled_faces)) == 1


def test_fill_holes_min_weight_empty_mesh(device: str) -> None:
    vertices = wp.empty(0, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)
    assert int(tw.hole_filling.fill_holes_min_weight(vertices, faces).shape[0]) == 0


def test_fill_holes_min_weight_rejects_unknown_metric(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = hemisphere
    with pytest.raises(ValueError, match="metric must be one of"):
        tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices, metric="bogus")


# --- Boundary triangulation engine (``triangulate_boundaries``) ------------------------------


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


def _triangulate_boundaries_np(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    loop_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
    loop_b: np.ndarray,
) -> np.ndarray:
    """
    Pure-NumPy port of the boundary zippering, used as the CPU reference for the kernels.

    Mirrors ``triwarp.hole_filling.triangulate_boundaries`` (itself the port of promesh's
    ``triangulate_boundaries``). Perimeters are computed in ``float32`` so the argmin tie-breaks
    match the Warp kernels. Returns the flat ``(3 * n_faces,)`` face buffer.
    """
    vertices_a = vertices_a.astype(np.float32)
    vertices_b = vertices_b.astype(np.float32)
    n, m = loop_a.size, loop_b.size
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n

    flipped_a = loop_a[::-1]
    loop_b_shifted = loop_b + len(vertices_a)
    a_pos = vertices_a[flipped_a]
    b_pos = vertices_b[loop_b]

    difference = a_pos[:, None, :] - b_pos[None, :, :]
    distances = np.sqrt((difference**2).sum(-1)).astype(np.float32)
    perimeters = distances + np.roll(distances, -1, axis=0)

    shift_a, shift_b = np.unravel_index(int(np.argmin(perimeters)), perimeters.shape)
    flipped_a = np.roll(flipped_a, -shift_a)
    loop_b_shifted = np.roll(loop_b_shifted, -shift_b)
    perimeters = np.roll(np.roll(perimeters, -shift_a, axis=0), -shift_b, axis=1)
    edge = np.argmin(perimeters, axis=1)

    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        flipped_a = np.roll(flipped_a, trailing)
        edge = np.roll(edge, trailing)
        perimeters = np.roll(perimeters, trailing, axis=0)

    if not np.all(np.diff(edge) >= 0):
        edge = np.append(edge, loop_b.size)
        perimeters = np.vstack([perimeters, perimeters[0]])
        unsorted_indices = _non_increasing_indices(edge)
        stable = np.delete(np.arange(edge.size), unsorted_indices)
        next_indices = stable[np.searchsorted(stable, unsorted_indices)]
        for index, next_index in zip(unsorted_indices, next_indices, strict=True):
            edge[index] = (
                int(np.argmin(perimeters[index, edge[index - 1] : edge[next_index] + 1]))
                + edge[index - 1]
            )
        edge = edge[:-1]

    window_a = np.lib.stride_tricks.sliding_window_view(
        np.append(flipped_a, flipped_a[0]), window_shape=2
    )
    bridge_a = np.column_stack([window_a, loop_b_shifted[edge]])
    window_b = np.lib.stride_tricks.sliding_window_view(
        np.append(loop_b_shifted, loop_b_shifted[0]), window_shape=2
    )
    apex = flipped_a[np.searchsorted(edge, np.arange(loop_b.size), side="right") % flipped_a.size]
    bridge_b = np.column_stack([np.fliplr(window_b), apex])

    faces = np.vstack(
        [faces_a.reshape(-1, 3), faces_b.reshape(-1, 3) + len(vertices_a), bridge_a, bridge_b]
    )
    return faces.reshape(-1).astype(np.int32)


def _sorted_triangle_rows(faces_flat: np.ndarray) -> np.ndarray:
    triangles = np.sort(faces_flat.reshape(-1, 3), axis=1)
    return triangles[np.lexsort(triangles.T[::-1])]


# Equal-count regular rims give a circulant perimeter matrix whose global argmin is tied by
# rotational symmetry; Warp's float32 reduction may pick a different (equally valid) minimum than
# NumPy, so the exact-match regression uses only tie-free asymmetric rims. The last case is
# laterally offset, which forces the non-monotone LIS correction path.
@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"),
    [(16, 11, 0.3, 0.0), (7, 13, 0.7, 0.0), (24, 5, 1.1, 0.0), (17, 11, 0.9, 1.2)],
)
def test_triangulate_boundaries_matches_numpy(
    device: str, n_a: int, n_b: int, phase: float, offset: float
) -> None:
    bottom, top = _capsule_halves(device, n_a, n_b, phase, offset)
    va_np, fa_np, va, fa = bottom
    vb_np, fb_np, vb, fb = top

    loop_a = tw.boundary.boundary_loop(va, fa)
    loop_b = tw.boundary.boundary_loop(vb, fb)

    _, faces_wp = tw.hole_filling.triangulate_boundaries(va, fa, loop_a, vb, fb, loop_b)
    faces_np = _triangulate_boundaries_np(
        va_np, fa_np, loop_a.numpy(), vb_np, fb_np, loop_b.numpy()
    )

    assert np.array_equal(_sorted_triangle_rows(faces_wp.numpy()), _sorted_triangle_rows(faces_np))


def test_triangulate_boundaries_rejects_small_loop(device: str) -> None:
    _, _, va, fa = _cone_wp(device=device, n=8, apex_z=-1.0, rim_z=0.0)
    _, _, vb, fb = _cone_wp(device=device, n=8, apex_z=1.0, rim_z=0.5)

    loop_a = tw.boundary.boundary_loop(va, fa)
    tiny_loop = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="at least 3 vertices"):
        tw.hole_filling.triangulate_boundaries(va, fa, loop_a, vb, fb, tiny_loop)


def test_non_increasing_indices() -> None:
    # The longest non-decreasing subsequence keeps the repeated 1s and 4s; only 5 (at index 4)
    # falls outside it, so its index is flagged for correction.
    numbers = np.array([0, 1, 1, 2, 5, 3, 4, 4, 7], dtype=np.int64)
    assert np.array_equal(_non_increasing_indices(numbers), np.array([4]))

    # A strictly sorted sequence needs no correction.
    assert _non_increasing_indices(np.arange(6, dtype=np.int64)).size == 0


# ---------------------------------------------------------------------------
# fill_holes_nicely (MeshLib fillHoleNicely)
# ---------------------------------------------------------------------------


def _skip_cpu(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("fill_holes_nicely subdivision/smoothing requires CUDA (warp.optim.linear.cg)")


def _mesh_volume_area(vertices_np: np.ndarray, faces_np: np.ndarray) -> tuple[float, float]:
    mesh = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(mesh.volume), float(mesh.area)


def _meshlib_fill_nicely_volume(
    vertices_np: np.ndarray, faces_np: np.ndarray, max_edge: float
) -> float:
    mm = pytest.importorskip("meshlib.mrmeshpy")
    mn = pytest.importorskip("meshlib.mrmeshnumpy")
    mesh = mn.meshFromFacesVerts(
        np.ascontiguousarray(faces_np.astype(np.int32)),
        np.ascontiguousarray(vertices_np.astype(np.float32)),
    )
    settings = mm.FillHoleNicelySettings()
    settings.subdivideSettings.maxEdgeLen = max_edge
    for edge in mesh.topology.findHoleRepresentiveEdges():
        mm.fillHoleNicely(mesh, edge, settings)
    verts = mn.getNumpyVerts(mesh)
    faces = mn.getNumpyFaces(mesh.topology)
    return float(tm.Trimesh(verts, faces, process=False).volume)


def test_fill_holes_nicely_invariants(device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _skip_cpu(device)
    _, mesh_wp = hemisphere
    n_v0 = int(mesh_wp.points.shape[0])

    new_vertices, new_faces, patch = tw.hole_filling.fill_holes_nicely(
        mesh_wp.points, mesh_wp.indices, return_patch=True
    )
    verts_np = new_vertices.numpy()
    faces_np = new_faces.numpy().reshape(-1, 3)

    assert len(_loop_sizes_of(new_vertices, new_faces)) == 0
    mesh_tm = tm.Trimesh(verts_np, faces_np, process=False)
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent
    assert tw.validation.is_edge_manifold(new_faces)
    assert int(patch.numpy().sum()) > 0
    assert np.allclose(verts_np[:n_v0], mesh_wp.points.numpy(), atol=1e-6)


def test_fill_holes_nicely_triangulate_only(device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    new_vertices, new_faces = tw.hole_filling.fill_holes_nicely(
        mesh_wp.points, mesh_wp.indices, triangulate_only=True
    )
    expected_faces = tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(new_faces.numpy(), expected_faces.numpy())
    assert np.array_equal(new_vertices.numpy(), mesh_wp.points.numpy())


def test_fill_holes_nicely_statistics_vs_meshlib(
    device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]
):
    _skip_cpu(device)
    pytest.importorskip("meshlib.mrmeshpy")
    _, mesh_wp = hemisphere
    vertices_np = mesh_wp.points.numpy().astype(np.float64)
    faces_np = mesh_wp.indices.numpy().reshape(-1, 3)
    max_edge = 0.3

    new_vertices, new_faces = tw.hole_filling.fill_holes_nicely(
        mesh_wp.points, mesh_wp.indices, max_edge=max_edge
    )
    volume_tw, _ = _mesh_volume_area(new_vertices.numpy(), new_faces.numpy().reshape(-1, 3))
    volume_ml = _meshlib_fill_nicely_volume(vertices_np, faces_np, max_edge)
    assert np.isclose(volume_tw, volume_ml, rtol=0.05)


def test_fill_holes_nicely_natural_smooth(device: str):
    _skip_cpu(device)
    sphere = tm.creation.icosphere(subdivisions=3, radius=1.0)
    hemi = sphere.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    hemi.merge_vertices()
    vertices_np = np.ascontiguousarray(hemi.vertices.astype(np.float64))
    faces_np = np.ascontiguousarray(hemi.faces.astype(np.int32).reshape(-1))
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    n_v0 = len(vertices_np)

    verts_off = tw.hole_filling.fill_holes_nicely(v_wp, f_wp, natural_smooth=False)[0].numpy()
    verts_on = tw.hole_filling.fill_holes_nicely(v_wp, f_wp, natural_smooth=True)[0].numpy()

    # naturalSmooth grows a collar past the rim, so some original vertices move.
    disp = np.linalg.norm(verts_off[:n_v0] - verts_on[:n_v0], axis=1)
    assert int((disp > 1e-5).sum()) > 0

    _, new_faces = tw.hole_filling.fill_holes_nicely(v_wp, f_wp, natural_smooth=True)
    mesh_tm = tm.Trimesh(verts_on, new_faces.numpy().reshape(-1, 3), process=False)
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent


def test_fill_holes_nicely_watertight_unchanged(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
):
    _, mesh_wp = icosahedron
    new_vertices, new_faces = tw.hole_filling.fill_holes_nicely(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(new_vertices.numpy(), mesh_wp.points.numpy())
    assert np.array_equal(new_faces.numpy(), mesh_wp.indices.numpy())


def test_fill_holes_nicely_rejects_unknown(device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = hemisphere
    with pytest.raises(ValueError, match="metric must be one of"):
        tw.hole_filling.fill_holes_nicely(mesh_wp.points, mesh_wp.indices, metric="nope")
    with pytest.raises(ValueError, match="edge_weights must be"):
        tw.hole_filling.fill_holes_nicely(mesh_wp.points, mesh_wp.indices, edge_weights="nope")
