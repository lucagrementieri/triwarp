"""Regression tests for ``triwarp.convex`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import points_to_open3d, points_to_pymeshlab


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_projections(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (dict index): the projection is keyed by its adjacency *pair*, not by row position.

    triwarp and trimesh both return one projection per adjacent face pair, but in different row
    orders, and the value only means anything paired with its own row -- so both sides are
    indexed into a dict by ``(face_a, face_b)`` before comparing. The key-set assert is what
    makes that sound: it fails if the two disagree about *which* pairs are adjacent, which a
    value comparison over a shared key subset would hide.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    projections_tm = mesh_tm.face_adjacency_projections

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    projections_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )

    adjacency_wp_np = adjacency_wp.numpy()
    projections_wp_np = projections_wp.numpy()
    projections_wp_lookup = {
        (int(row[0]), int(row[1])): float(projections_wp_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    projections_tm_lookup = {
        (int(row[0]), int(row[1])): float(projections_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert projections_wp_lookup.keys() == projections_tm_lookup.keys()
    for key, projection_tm in projections_tm_lookup.items():
        projection_wp = projections_wp_lookup[key]
        assert np.isclose(projection_wp, projection_tm, rtol=1e-4, atol=5e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_projections_precomputed(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Triwarp against triwarp: the precomputed-normals path must give the same projections.

    The oracle for the values is [`test_face_adjacency_projections`] above, against trimesh;
    this pins only that supplying ``face_adjacency_unshared`` and ``face_normals`` takes the
    same route as deriving them.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    projections_all_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    projections_precomputed_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
        face_normals=face_normals_wp,
    )
    assert np.allclose(
        projections_all_wp.numpy(), projections_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5
    )

    adjacency_tm = mesh_tm.face_adjacency
    projections_tm = mesh_tm.face_adjacency_projections
    adjacency_wp_np = adjacency_wp.numpy()
    projections_precomputed_np = projections_precomputed_wp.numpy()
    projections_tm_lookup = {
        (int(row[0]), int(row[1])): float(projections_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    projections_precomputed_lookup = {
        (int(row[0]), int(row[1])): float(projections_precomputed_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    for key, projection_tm in projections_tm_lookup.items():
        assert np.isclose(projections_precomputed_lookup[key], projection_tm, rtol=1e-4, atol=5e-4)


def test_face_adjacency_projections_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    projections_wp = tw.convex.face_adjacency_projections(vertices_wp, faces_wp)
    assert projections_wp.shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("face_adjacency_convex", "trimesh")
def test_face_adjacency_convex(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (dict index): the per-pair convexity flag, keyed like the projections above.

    Same transform and the same reason as [`test_face_adjacency_projections`]. Non-vacuous by
    fixture choice rather than by an assert: ``icosahedron`` is convex at every edge and
    ``half_torus`` is not, so the boolean is exercised both ways across the parametrisation.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    convex_tm = mesh_tm.face_adjacency_convex

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    convex_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )

    adjacency_wp_np = adjacency_wp.numpy()
    convex_wp_np = convex_wp.numpy()
    convex_wp_lookup = {
        (int(row[0]), int(row[1])): bool(convex_wp_np[i]) for i, row in enumerate(adjacency_wp_np)
    }
    convex_tm_lookup = {
        (int(row[0]), int(row[1])): bool(convex_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert convex_wp_lookup.keys() == convex_tm_lookup.keys()
    for key, is_convex_tm in convex_tm_lookup.items():
        assert convex_wp_lookup[key] == is_convex_tm


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_convex_precomputed(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Triwarp against triwarp: the precomputed path must give the same convexity flags.

    Oracle is [`test_face_adjacency_convex`]; this pins the precomputed-argument route only.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    convex_all_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    convex_precomputed_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
        face_normals=face_normals_wp,
    )
    assert np.array_equal(convex_all_wp.numpy(), convex_precomputed_wp.numpy())

    adjacency_tm = mesh_tm.face_adjacency
    convex_tm = mesh_tm.face_adjacency_convex
    adjacency_wp_np = adjacency_wp.numpy()
    convex_precomputed_np = convex_precomputed_wp.numpy()
    convex_tm_lookup = {
        (int(row[0]), int(row[1])): bool(convex_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    convex_precomputed_lookup = {
        (int(row[0]), int(row[1])): bool(convex_precomputed_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    for key, is_convex_tm in convex_tm_lookup.items():
        assert convex_precomputed_lookup[key] == is_convex_tm


def test_face_adjacency_convex_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    convex_wp = tw.convex.face_adjacency_convex(vertices_wp, faces_wp)
    assert convex_wp.shape == (0,)


@pytest.mark.parity("convex_subset_mask", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parity("convex_subset", "trimesh", "open3d", "pymeshlab")
def test_convex_subset_mask_against_the_three_qhull_backends(device: str) -> None:
    """
    Class C (soundness plus a recall bound), against exact qhull.

    ``benchmarks/test_convex.py`` says of these three rows that "this is not a parity comparison":
    trimesh, Open3D and pymeshlab all run **qhull** and return the exact hull as a *mesh*, while
    ``convex_subset`` returns an approximate *vertex subset* from a direction sweep. That rules
    out equality -- it does not rule out a test. Two properties are checkable and are exactly
    what an approximate hull filter has to guarantee:

    - **soundness**, asserted exactly: every point triwarp selects must be a true hull vertex. This
      is the half that catches a real bug -- an implementation that returned interior points, or the
      whole cloud, fails immediately, and no tolerance is involved. Note this is the *fixture's*
      guarantee, not the function's: it holds because 500 standard-normal points are in general
      position, so no support direction ties. The invariant that holds for every input is the weaker
      "every selected point lies on the hull boundary" -- on a cloud with coplanar ties (a grid over
      each face of a cube) the mask also selects face-edge midpoints, which are boundary points but
      not hull vertices, and this assertion would fail there by design.
    - **recall**, asserted with a bound: measured **27 of 31** hull vertices at
      ``n_directions=256`` (0.871) against all three references, which agree with each other on the
      hull exactly. The 0.70 floor leaves room for a different direction set without admitting a
      filter that has stopped finding most of the hull. The gap is not slack in the test -- the four
      missed vertices have normal cones spanning 3e-5 to 2e-3 of the sphere, so no direction sample
      this size is expected to find them; recall reaches 1.00 on this cloud at 16384 directions.

    Marked for all three libraries deliberately: they compute the identical answer here, so one
    assertion covers all three rows, and confirming they agree is itself worth a line -- it says the
    benchmark's three qhull rows are pricing wrappers around one algorithm, not three algorithms.

    Both hull entry points are checked against the references here, which is why the marker names
    ``convex_subset`` as well as ``convex_subset_mask``: the two benchmark groups time the same
    approximation against the same qhull bar, so one comparison is the honest place for both claims.
    The *mask against subset* equality below is triwarp-against-triwarp -- the mask is the entry
    point carrying the oracle, and ``test_convex_subset_points`` pins the same pair on positions.
    """
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    selected = set(
        np.flatnonzero(tw.convex.convex_subset_mask(points_wp, n_directions=256).numpy()).tolist()
    )

    def hull_indices(hull_vertices: np.ndarray, tolerance: float = 1e-9) -> set[int]:
        """Map a hull's vertex positions back onto indices into the input cloud."""
        distance_np, index_np = scipy.spatial.cKDTree(points_np).query(np.asarray(hull_vertices))
        assert distance_np.max() < tolerance  # every returned position is one of the inputs
        return set(index_np[distance_np < tolerance].tolist())

    # The compacted entry point resolves back to the identical index set, so the assertions below
    # speak for both benchmark groups rather than only the mask one. Its positions come back
    # ``float32`` where the three references hand back the ``float64`` inputs verbatim, so the
    # lookup needs a float32-scale tolerance: measured 1.2e-07 of round-trip error against a
    # minimum inter-point spacing of 0.046 in this cloud, so 1e-5 is unambiguous by ~4 600x.
    assert (
        hull_indices(tw.convex.convex_subset(points_wp, n_directions=256).numpy(), 1e-5) == selected
    )

    hull_tm = hull_indices(tm.points.PointCloud(points_np).convex_hull.vertices)
    mesh_o3d, _kept = points_to_open3d(points_np).compute_convex_hull()
    hull_o3d = hull_indices(np.asarray(mesh_o3d.vertices))
    meshset_pml = points_to_pymeshlab(points_np)
    meshset_pml.generate_convex_hull()
    hull_pml = hull_indices(meshset_pml.current_mesh().vertex_matrix())

    # The three qhull wrappers agree, so any one of them is "the" exact hull.
    assert hull_tm == hull_o3d == hull_pml
    assert len(hull_tm) > 0

    for name, hull in (("trimesh", hull_tm), ("open3d", hull_o3d), ("pymeshlab", hull_pml)):
        assert selected <= hull, f"{name}: selected a point that is not a hull vertex"
        assert len(selected & hull) / len(hull) > 0.70, f"{name}: recall too low"


def test_convex_subset_mask_sound(device: str) -> None:
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.convex_subset_mask(points_wp, n_directions=256)
    selected = np.flatnonzero(mask_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_convex_subset_mask_scale_invariant(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    scaled_np = points_np * 1e4

    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    scaled_wp = wp.array(np.ascontiguousarray(scaled_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.convex_subset_mask(points_wp, n_directions=256)
    mask_scaled_wp = tw.convex.convex_subset_mask(scaled_wp, n_directions=256)
    assert np.array_equal(mask_wp.numpy(), mask_scaled_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(scaled_np)
    selected = np.flatnonzero(mask_scaled_wp.numpy())
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_convex_subset_recall(device: str) -> None:
    rng = np.random.default_rng(2)
    points_np = rng.standard_normal((200, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.convex_subset_mask(points_wp, n_directions=4096)
    selected = set(np.flatnonzero(mask_wp.numpy()).tolist())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert selected == set(hull_scipy.vertices.tolist())


def test_convex_subset_points(device: str) -> None:
    rng = np.random.default_rng(4)
    points_np = rng.standard_normal((300, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.convex_subset_mask(points_wp, n_directions=256)
    subset_wp = tw.convex.convex_subset(points_wp, n_directions=256)

    selected = np.flatnonzero(mask_wp.numpy())
    expected_points = points_np[np.sort(selected)]
    assert np.allclose(subset_wp.numpy(), expected_points, rtol=1e-5, atol=1e-5)


def test_convex_subset_mask_empty(device: str) -> None:
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    mask_wp = tw.convex.convex_subset_mask(points_wp)
    assert mask_wp.shape == (0,)


def _cloud(kind: str, n: int, seed: int) -> np.ndarray:
    """Build one of the point distributions the superset filter behaves differently on."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((n, 3))
    if kind == "cube":
        return rng.random((n, 3))
    direction_np = rng.standard_normal((n, 3))
    direction_np /= np.linalg.norm(direction_np, axis=1, keepdims=True)
    return direction_np * rng.random((n, 1)) ** (1.0 / 3.0)


# Fraction of the cloud the filter is allowed to keep, per distribution, at ``subdivisions=3`` on
# 20k points. Measured 0.40% / 3.60% / 1.46% for gaussian / ball / cube; these bounds sit ~3x above
# that, which is what makes the containment assertion below non-vacuous -- an all-``True`` mask
# (the trivially correct superset) keeps 100% and fails every one of them.
_MAX_KEPT_FRACTION = {"gaussian": 0.012, "ball": 0.11, "cube": 0.05}


@pytest.mark.parametrize("kind", ["gaussian", "ball", "cube"])
@pytest.mark.parity("convex_superset_mask", "scipy")
def test_convex_superset_mask_contains_the_exact_hull(device: str, kind: str) -> None:
    """
    Class B: exact containment of the reference hull's vertex set.

    The one named transform is reading [`scipy.spatial.ConvexHull`][]'s hull vertex *indices* as a
    boolean mask.

    Containment rather than equality is the point, not a weakening: ``convex_superset_mask`` is
    defined as a conservative filter, and "no hull vertex is ever discarded" is the whole contract.
    The assertion is exact -- no tolerance -- and it is the assertion that fails if the tetrahedron
    interior test is ever wrong in the unsafe direction.

    Containment alone would be vacuous (an all-``True`` mask satisfies it), so the second assertion
    bounds how much the filter keeps. Both must hold: the first catches a filter that discards too
    much, the second a filter that discards too little. Parametrized over three distributions
    because selectivity varies by two orders of magnitude between them -- near-spherical clouds are
    the easy case and flat-faced ones the hard case -- so a single fixture would hide a regression
    on the others.
    """
    points_np = _cloud(kind, 20_000, seed=11)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_np = tw.convex.convex_superset_mask(points_wp, subdivisions=3).numpy()
    kept = set(np.flatnonzero(mask_np).tolist())
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    assert hull_scipy <= kept, f"{kind}: discarded {len(hull_scipy - kept)} true hull vertices"
    assert mask_np.mean() < _MAX_KEPT_FRACTION[kind], f"{kind}: filter kept {mask_np.mean():.3%}"


@pytest.mark.parametrize("kind", ["gaussian", "cube"])
def test_convex_superset_mask_tightens_with_subdivisions(device: str, kind: str) -> None:
    """More directions wrap the hull more closely, and the guarantee holds at every level."""
    points_np = _cloud(kind, 5_000, seed=12)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    counts = []
    for subdivisions in (0, 1, 2, 3):
        mask_np = tw.convex.convex_superset_mask(points_wp, subdivisions=subdivisions).numpy()
        assert hull_scipy <= set(np.flatnonzero(mask_np).tolist())
        counts.append(int(mask_np.sum()))

    assert counts == sorted(counts, reverse=True), f"not monotone in subdivisions: {counts}"
    assert counts[-1] < counts[0]
    assert counts[-1] >= len(hull_scipy)


def test_convex_superset_mask_contains_the_subset_mask(device: str) -> None:
    """The two one-sided filters bracket the hull: subset ``<=`` hull vertices ``<=`` superset."""
    points_np = _cloud("gaussian", 5_000, seed=13)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    subset_np = tw.convex.convex_subset_mask(points_wp, n_directions=256).numpy()
    superset_np = tw.convex.convex_superset_mask(points_wp, subdivisions=3).numpy()
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    assert set(np.flatnonzero(subset_np).tolist()) <= hull_scipy
    assert hull_scipy <= set(np.flatnonzero(superset_np).tolist())
    assert np.array_equal(subset_np & superset_np, subset_np)


def test_convex_superset_mask_scale_invariant(device: str) -> None:
    """The flatness and margin tests are relative, so scaling the cloud cannot change the mask."""
    points_np = _cloud("gaussian", 5_000, seed=14)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    scaled_wp = wp.array(np.ascontiguousarray(points_np * 1e4), dtype=wp.vec3, device=device)

    assert np.array_equal(
        tw.convex.convex_superset_mask(points_wp).numpy(),
        tw.convex.convex_superset_mask(scaled_wp).numpy(),
    )


@pytest.mark.parametrize("kind", ["coplanar", "collinear", "identical", "three_points"])
def test_convex_superset_mask_degenerate_keeps_everything(device: str, kind: str) -> None:
    """
    Degenerate clouds have no non-flat tetrahedron, so nothing is certified interior.

    Keeping every point is the conservative answer and a valid (if useless) superset -- the failure
    mode this guards against is the opposite one, where a flat tetrahedron's ill-conditioned inverse
    reports arbitrary points as interior and discards hull vertices.
    """
    rng = np.random.default_rng(15)
    if kind == "coplanar":
        points_np = np.column_stack([rng.standard_normal((500, 2)), np.zeros(500)])
    elif kind == "collinear":
        points_np = np.outer(np.linspace(0.0, 1.0, 100), np.array([1.0, 2.0, 3.0]))
    elif kind == "identical":
        points_np = np.ones((50, 3))
    else:
        points_np = rng.standard_normal((3, 3))

    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    assert tw.convex.convex_superset_mask(points_wp).numpy().all()


def test_convex_superset_mask_empty(device: str) -> None:
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    mask_wp = tw.convex.convex_superset_mask(points_wp)
    assert mask_wp.shape == (0,)


@pytest.mark.parametrize("mask_device", ["cpu", "cuda:0"])
def test_support_sweep_agrees_across_devices(mask_device: str) -> None:
    """
    Both hull filters must agree on CPU and CUDA, which is not automatic.

    ``wp.launch_tiled`` runs exactly **one** lane per block on Warp 1.16's CPU backend -- the lane
    index from ``wp.tid()`` is always 0 -- so the block-wide ``wp.tile_max`` reduction the support
    sweep originally used silently reduced over one point per 64-point tile there. That returned an
    under-estimated support maximum, which made ``convex_subset_mask`` mark interior points (its
    soundness assertion above fails outright on CPU) and made ``convex_superset_mask`` build a
    shrunken shell. Both kernels are now lane-free; this pins that, on the device where every
    ``device``-fixture test is silent because the fixture prefers ``cuda:0``.
    """
    if mask_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    points_np = _cloud("gaussian", 5_000, seed=16)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=mask_device)
    hull_scipy = set(scipy.spatial.ConvexHull(points_np).vertices.tolist())

    subset_np = tw.convex.convex_subset_mask(points_wp, n_directions=128).numpy()
    superset_np = tw.convex.convex_superset_mask(points_wp, subdivisions=2).numpy()

    assert set(np.flatnonzero(subset_np).tolist()) <= hull_scipy
    assert hull_scipy <= set(np.flatnonzero(superset_np).tolist())
    # The tile bug's signature was a wildly less selective filter, not a wrong-shaped one.
    assert superset_np.mean() < 0.05
