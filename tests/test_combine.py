"""Regression tests for ``triwarp.combine``."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.comparisons import boundary_loop_sizes, lexsort_rows
from tests.conversions import numpy_to_meshlib
from triwarp.combine import _non_increasing_indices


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
        lexsort_rows(np.sort(faces_ab.numpy().reshape(-1, 3), axis=1)),
        lexsort_rows(np.sort(faces_ba.numpy().reshape(-1, 3), axis=1)),
    )


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"), [(16, 11, 0.3, 0.0), (17, 11, 0.9, 0.4)]
)
@pytest.mark.parity(
    "stitch",
    "meshlib",
    benchmarked=False,
    reason="MeshLib has no zipper. Its two-argument stitchHoles finds the holes itself and then "
    "runs the same minimum-weight DP its three-argument form runs, which is already timed in the "
    "stitch_min_weight group -- a row here would price that DP under the zipper's name and read as "
    "the zipper being 100x slower than it is.",
)
def test_stitch_reaches_meshlibs_optimum(
    device: str, n_a: int, n_b: int, phase: float, offset: float
) -> None:
    """
    Class C (a derived scalar): the greedy zipper's band, scored by MeshLib's own metric.

    ``stitch`` walks the two rims from the cheapest starting correspondence and never reconsiders,
    where MeshLib's two-argument ``stitchHoles`` -- the overload that finds the holes itself, so
    triwarp's loop *pairing* is compared too and not only its triangulation -- searches the whole
    space. The zipper is therefore an upper bound on the optimum by construction, and the question
    a comparison can answer is how loose a bound.

    Measured over eight rim configurations: **the zipper reaches the optimum exactly (ratio 1.0000)
    on seven of them**, and costs 18.9% more on the one whose top rim is shifted sideways, where the
    rim-to-rim correspondence stops being monotone and greed is provably not enough. Those are the
    two cases parametrized here, so the test covers both the tight branch and the loose one rather
    than only the flattering half.

    What it excludes is a zipper that emits a *valid but badly shaped* band -- a slipped
    correspondence costs several times the optimum, not 19% -- and what it cannot see is a different
    tie-break at equal cost, which is real: 4 of the 8 bands differ from MeshLib's triangle for
    triangle at an identical score, so comparing triangles rather than cost would fail on a
    symmetric rim. The cost helper is shared with the minimum-weight section below.
    """
    (va_np, fa_np, va, fa), (vb_np, fb_np, vb, fb) = _capsule_halves(
        device, n_a, n_b, phase, offset
    )

    new_vertices, new_faces = tw.combine.stitch(va, fa, vb, fb)
    n_orig = int(fa.shape[0]) + int(fb.shape[0])
    band_tw = new_faces.numpy()[n_orig:].reshape(-1, 3)
    # ``stitch`` puts the larger-boundary mesh first, so rebuild the prefix in the order it used.
    fa_rows, fb_rows = fa_np.reshape(-1, 3), fb_np.reshape(-1, 3)
    if n_a >= n_b:
        orig_tw = np.vstack([fa_rows, fb_rows + len(va_np)]).astype(np.int32)
    else:
        orig_tw = np.vstack([fb_rows, fa_rows + len(vb_np)]).astype(np.int32)

    verts_ml = np.ascontiguousarray(np.vstack([va_np, vb_np]), dtype=np.float32)
    orig_ml = np.ascontiguousarray(np.vstack([fa_rows, fb_rows + len(va_np)]), dtype=np.int32)
    mesh_ml = numpy_to_meshlib(verts_ml, orig_ml)
    assert len(mesh_ml.topology.findHoleRepresentiveEdges()) == 2  # the auto-detect's input
    assert mm.stitchHoles(mesh_ml, mm.StitchHolesParams())  # the two-argument overload, and it ran
    faces_out_ml = mn.getNumpyFaces(mesh_ml.topology)
    original = {tuple(sorted(int(x) for x in row)) for row in orig_ml}
    band_ml = np.array(
        [row for row in faces_out_ml if tuple(sorted(int(x) for x in row)) not in original],
        dtype=np.int32,
    )

    # Both close the two rims with one triangle per rim edge and no new vertex.
    assert len(band_tw) == len(band_ml) == n_a + n_b
    assert int(new_vertices.shape[0]) == len(va_np) + len(vb_np)
    assert tw.validation.is_watertight(new_vertices, new_faces)

    cost_tw = _meshlib_stitch_cost(new_vertices.numpy(), orig_tw, band_tw, "complex_stitch")
    cost_ml = _meshlib_stitch_cost(verts_ml, orig_ml, band_ml, "complex_stitch")
    assert cost_ml > 0.0  # non-vacuity: a zero optimum would make any ratio pass
    assert cost_tw >= cost_ml * (1.0 - 1e-6)  # the greedy band cannot beat the exhaustive one
    assert cost_tw < 1.25 * cost_ml  # measured 1.0000 and 1.1889 on these two configurations


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

    assert len(boundary_loop_sizes(mesh_wp.indices.numpy().reshape(-1, 3))) >= 2
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
    make_metric = {
        "complex_stitch": lambda m: mm.getComplexStitchMetric(m),
        "edge_length_stitch": lambda m: mm.getEdgeLengthStitchMetric(m),
        "vertical": lambda m: mm.getVerticalStitchMetric(m, mm.Vector3f(0.0, 0.0, 1.0)),
    }[metric]
    verts = np.ascontiguousarray(np.vstack([va_np, vb_np]), dtype=np.float32)
    orig = np.ascontiguousarray(np.vstack([fa_np, fb_np + len(va_np)]), dtype=np.int32)
    mesh = numpy_to_meshlib(verts, orig)
    edges = mesh.topology.findHoleRepresentiveEdges()
    params = mm.StitchHolesParams()
    params.metric = make_metric(mesh)
    mm.stitchHoles(mesh, edges[0], edges[1], params)
    faces_out = mn.getNumpyFaces(mesh.topology)
    original = {tuple(sorted(int(x) for x in t)) for t in orig}
    band = np.array(
        [t for t in faces_out if tuple(sorted(int(x) for x in t)) not in original], np.int32
    )
    return verts, orig, band


def _meshlib_stitch_cost(
    verts: np.ndarray, orig: np.ndarray, band: np.ndarray, metric: str
) -> float:
    make_metric = {
        "complex_stitch": lambda m: mm.getComplexStitchMetric(m),
        "vertical": lambda m: mm.getVerticalStitchMetric(m, mm.Vector3f(0.0, 0.0, 1.0)),
    }[metric]
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    mesh_orig = numpy_to_meshlib(verts, orig)
    metric_obj = make_metric(mesh_orig)
    full = np.ascontiguousarray(np.vstack([orig, band]), dtype=np.int32)
    mesh_full = numpy_to_meshlib(verts, full)
    region_bools = np.zeros(len(full), dtype=bool)
    region_bools[len(orig) :] = True
    region = mn.faceBitSetFromBools(region_bools)
    return mm.calcCombinedFillMetric(mesh_full, region, metric_obj)


@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11), (8, 8)])
@pytest.mark.parametrize("metric", STITCH_METRICS)
def test_stitch_min_weight_watertight(device: str, n_a: int, n_b: int, metric: str) -> None:
    """
    Not a library comparison: the band's exact triangle count, with trimesh as the oracle.

    Two rims of ``n_a`` and ``n_b`` vertices close with exactly ``n_a + n_b`` triangles and no new
    vertices, whatever the metric chooses -- an arithmetic reference, not another implementation.
    The metric's *choice* among those triangulations is what
    [`test_stitch_min_weight_matches_meshlib`] compares.
    """
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


@pytest.mark.parity("stitch_min_weight", "meshlib")
@pytest.mark.parametrize(("n_a", "n_b"), [(9, 13), (16, 11)])
@pytest.mark.parametrize("metric", STITCH_COST_METRICS)
def test_stitch_min_weight_matches_meshlib(device: str, n_a: int, n_b: int, metric: str) -> None:
    """
    Class C (a derived scalar): the band's *cost* matches MeshLib's optimum, not its triangles.

    Both sides minimize the same objective over the same rims, and several triangulations can reach
    the optimum -- so the comparable quantity is the cost, evaluated by one shared function on both
    answers. What this excludes is triwarp settling for a worse triangulation; what it cannot see is
    a different tie-break at equal cost, which is the point of comparing costs.
    """
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
    """
    Not a library comparison: the invariants ``stitch_smooth`` must hold, with trimesh as oracle.

    Watertight, winding-consistent, no boundary loop left, and the two inputs' vertices unmoved in
    the prefix -- that last is what separates *stitching* from *remeshing the whole thing*.
    """
    (va, fa), (vb, fb) = _hemisphere_pair(device)
    n_v0 = int(va.shape[0]) + int(vb.shape[0])

    new_vertices, new_faces = tw.combine.stitch_smooth(va, fa, vb, fb)
    verts_np = new_vertices.numpy()
    mesh_tm = tm.Trimesh(verts_np, new_faces.numpy().reshape(-1, 3), process=False)

    assert boundary_loop_sizes(new_faces.numpy().reshape(-1, 3)) == []
    assert mesh_tm.is_watertight
    assert mesh_tm.is_winding_consistent
    # Original vertices are the concatenated prefix, unchanged.
    assert np.allclose(verts_np[:n_v0], np.concatenate([va.numpy(), vb.numpy()]), atol=1e-6)


@pytest.mark.parity(
    "stitch_smooth",
    "meshlib",
    benchmarked=False,
    reason="stitch_smooth has no benchmark group: the band it refines is the one "
    "stitch_min_weight already times, plus fill_smooth's finisher, and a group here "
    "would re-measure both under a third name. stitchHolesNicely is nonetheless the "
    "only reference that performs the whole three-stage operation, which is why the "
    "comparison lives here.",
)
def test_stitch_smooth_statistics_vs_meshlib(device: str):
    """
    Class C (a derived scalar): the enclosed volume, because the two bands share no vertices.

    ``stitchHolesNicely`` is the same three stages in the same order -- minimum-weight band, refine
    to a target edge length, smooth the new interior into both surrounding surfaces -- and it takes
    the same knobs (``subdivideSettings.maxEdgeLen``, ``maxEdgeSplits``, ``smoothCurvature``), which
    is what makes this comparable at all. But each side subdivides on its own schedule, so no vertex
    correspondence exists and the volume is the strongest shared quantity, exactly as
    [`test_fill_smooth_statistics_vs_meshlib`] argues for the single-hole case.

    Measured on two facing hemispheres at ``max_edge=0.15``: volume within **1.9%** and area within
    **1.2%**, with 2 384 faces against MeshLib's 2 368 -- so the refinement lands within 0.7% of the
    same triangle budget from the same target length. At ``max_edge=0.3`` the volumes are 3.8%
    apart, which is the refinement schedule diverging where there are fewer splits to average over,
    and is why the finer target is the one asserted.

    What the volume excludes is a band that bulges, collapses or pinches while still being
    watertight; what it cannot see is a band that is smooth in the wrong place.
    """
    (va, fa), (vb, fb) = _hemisphere_pair(device)
    max_edge = 0.15

    new_vertices, new_faces = tw.combine.stitch_smooth(va, fa, vb, fb, max_edge=max_edge)
    mesh_tw = tm.Trimesh(new_vertices.numpy(), new_faces.numpy().reshape(-1, 3), process=False)
    assert mesh_tw.is_watertight

    verts_ml = np.ascontiguousarray(np.vstack([va.numpy(), vb.numpy()]), dtype=np.float32)
    orig_ml = np.ascontiguousarray(
        np.vstack([fa.numpy().reshape(-1, 3), fb.numpy().reshape(-1, 3) + int(va.shape[0])]),
        dtype=np.int32,
    )
    mesh_ml = numpy_to_meshlib(verts_ml, orig_ml)
    holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
    assert len(holes_ml) == 2
    settings_ml = mm.StitchHolesNicelySettings()
    settings_ml.triangulateParams.metric = mm.getComplexStitchMetric(mesh_ml)
    settings_ml.subdivideSettings.maxEdgeLen = max_edge
    settings_ml.subdivideSettings.maxEdgeSplits = 1000
    settings_ml.smoothCurvature = True
    patch_ml = mm.stitchHolesNicely(mesh_ml, holes_ml[0], holes_ml[1], settings_ml)
    n_patch_ml = int(mn.getNumpyBitSet(patch_ml).sum())  # before pack(): the returned ids are stale
    mesh_ml.pack()  # mandatory before reading topology back
    mesh_ref = tm.Trimesh(
        mn.getNumpyVerts(mesh_ml), mn.getNumpyFaces(mesh_ml.topology), process=False
    )

    # Non-vacuity: both sides really refined the band rather than returning the plain stitch, whose
    # band is one triangle per rim edge -- an order of magnitude fewer faces than this.
    n_input_faces = int(fa.shape[0]) // 3 + int(fb.shape[0]) // 3
    assert n_patch_ml > 100
    assert mesh_ref.is_watertight
    assert len(mesh_ref.faces) > n_input_faces + 100
    assert len(mesh_tw.faces) > n_input_faces + 100
    assert np.isclose(mesh_tw.volume, mesh_ref.volume, rtol=0.05)
    assert np.isclose(mesh_tw.area, mesh_ref.area, rtol=0.05)


def test_stitch_smooth_requires_single_loop(device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.combine.stitch_smooth(mesh_wp.points, mesh_wp.indices, mesh_wp.points, mesh_wp.indices)


# ---------------------------------------------------------------------------
# The loop-level engines (``stitch_loops`` / ``stitch_loops_min_weight``)
# ---------------------------------------------------------------------------


def _stitch_loops_np(
    vertices_a: np.ndarray,
    faces_a: np.ndarray,
    loop_a: np.ndarray,
    vertices_b: np.ndarray,
    faces_b: np.ndarray,
    loop_b: np.ndarray,
) -> np.ndarray:
    """
    Pure-NumPy port of the boundary zippering, used as the CPU reference for the kernels.

    Mirrors ``triwarp.combine.stitch_loops`` (itself the port of promesh's
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


@pytest.mark.parametrize(
    ("n_a", "n_b", "phase", "offset"),
    [(16, 11, 0.3, 0.0), (7, 13, 0.7, 0.0), (24, 5, 1.1, 0.0), (17, 11, 0.9, 1.2)],
)
def test_stitch_loops_matches_numpy(
    device: str, n_a: int, n_b: int, phase: float, offset: float
) -> None:
    bottom, top = _capsule_halves(device, n_a, n_b, phase, offset)
    va_np, fa_np, va, fa = bottom
    vb_np, fb_np, vb, fb = top

    loop_a = tw.boundary.boundary_loop(va, fa)
    loop_b = tw.boundary.boundary_loop(vb, fb)

    _, faces_wp = tw.combine.stitch_loops(va, fa, loop_a, vb, fb, loop_b)
    faces_np = _stitch_loops_np(va_np, fa_np, loop_a.numpy(), vb_np, fb_np, loop_b.numpy())

    assert np.array_equal(
        lexsort_rows(np.sort(faces_wp.numpy().reshape(-1, 3), axis=1)),
        lexsort_rows(np.sort(faces_np.reshape(-1, 3), axis=1)),
    )


def test_stitch_loops_rejects_small_loop(device: str) -> None:
    _, _, va, fa = _cone_wp(device=device, n=8, apex_z=-1.0, rim_z=0.0)
    _, _, vb, fb = _cone_wp(device=device, n=8, apex_z=1.0, rim_z=0.5)

    loop_a = tw.boundary.boundary_loop(va, fa)
    tiny_loop = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="at least 3 vertices"):
        tw.combine.stitch_loops(va, fa, loop_a, vb, fb, tiny_loop)


def test_non_increasing_indices() -> None:
    # The longest non-decreasing subsequence keeps the repeated 1s and 4s; only 5 (at index 4)
    # falls outside it, so its index is flagged for correction.
    numbers = np.array([0, 1, 1, 2, 5, 3, 4, 4, 7], dtype=np.int64)
    assert np.array_equal(_non_increasing_indices(numbers), np.array([4]))

    # A strictly sorted sequence needs no correction.
    assert _non_increasing_indices(np.arange(6, dtype=np.int64)).size == 0
