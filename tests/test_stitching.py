"""Regression tests for ``triwarp.stitching`` hole filling and boundary stitching."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp

import triwarp as tw
from triwarp.stitching import _non_increasing_indices

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

    assert not tw.characteristics.is_watertight(mesh_wp.points, mesh_wp.indices)

    filled_faces = tw.stitching.fill_holes_fan(mesh_wp.points, mesh_wp.indices)

    assert tw.characteristics.is_watertight(mesh_wp.points, filled_faces)
    assert tw.characteristics.is_winding_consistent(filled_faces)

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

    new_vertices, filled_faces = tw.stitching.fill_holes_cone(mesh_wp.points, mesh_wp.indices)

    assert tw.characteristics.is_watertight(new_vertices, filled_faces)
    assert tw.characteristics.is_winding_consistent(filled_faces)

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

    new_vertices, _ = tw.stitching.fill_holes_cone(mesh_wp.points, mesh_wp.indices)
    centroid_wp = new_vertices.numpy()[int(mesh_wp.points.shape[0])]

    assert np.allclose(centroid_wp, centroid_expected, rtol=1e-4, atol=1e-4)


def test_fill_holes_watertight_mesh_unchanged(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron

    filled_faces = tw.stitching.fill_holes_fan(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())

    new_vertices, cone_faces = tw.stitching.fill_holes_cone(mesh_wp.points, mesh_wp.indices)
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

    filled_faces = tw.stitching.fill_holes_fan(
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

    new_vertices, filled_faces = tw.stitching.fill_holes_cone(
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

    filled_faces = tw.stitching.fill_holes_fan(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    assert np.array_equal(filled_faces.numpy(), mesh_wp.indices.numpy())

    new_vertices, cone_faces = tw.stitching.fill_holes_cone(
        mesh_wp.points, mesh_wp.indices, preserve_largest_hole=True
    )
    assert np.array_equal(cone_faces.numpy(), mesh_wp.indices.numpy())
    assert int(new_vertices.shape[0]) == int(mesh_wp.points.shape[0])


def test_fill_holes_empty_mesh(device: str) -> None:
    vertices = wp.empty(0, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)

    assert int(tw.stitching.fill_holes_fan(vertices, faces).shape[0]) == 0

    new_vertices, new_faces = tw.stitching.fill_holes_cone(vertices, faces)
    assert int(new_vertices.shape[0]) == 0
    assert int(new_faces.shape[0]) == 0


# --- Boundary stitching (``triangulate_boundaries`` / ``stitch``) ----------------------------


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

    Mirrors ``triwarp.stitching.triangulate_boundaries`` (itself the port of promesh's
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
    apex = flipped_a[
        np.searchsorted(edge, np.arange(loop_b.size), side="right") % flipped_a.size
    ]
    bridge_b = np.column_stack([np.fliplr(window_b), apex])

    faces = np.vstack(
        [
            faces_a.reshape(-1, 3),
            faces_b.reshape(-1, 3) + len(vertices_a),
            bridge_a,
            bridge_b,
        ]
    )
    return faces.reshape(-1).astype(np.int32)


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

    assert not tw.characteristics.is_watertight(va, fa)
    assert not tw.characteristics.is_watertight(vb, fb)

    new_vertices, new_faces = tw.stitching.stitch(va, fa, vb, fb)

    assert tw.characteristics.is_watertight(new_vertices, new_faces)
    assert tw.characteristics.is_winding_consistent(new_faces)

    # A + B faces plus one bridge triangle per rim edge on each side.
    n_new_faces = (int(new_faces.shape[0]) - int(fa.shape[0]) - int(fb.shape[0])) // 3
    assert n_new_faces == n_a + n_b
    assert int(new_vertices.shape[0]) == int(va.shape[0]) + int(vb.shape[0])


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

    _, faces_wp = tw.stitching.triangulate_boundaries(va, fa, loop_a, vb, fb, loop_b)
    faces_np = _triangulate_boundaries_np(
        va_np, fa_np, loop_a.numpy(), vb_np, fb_np, loop_b.numpy()
    )

    assert np.array_equal(
        _sorted_triangle_rows(faces_wp.numpy()), _sorted_triangle_rows(faces_np)
    )


def test_stitch_argument_order_invariant(device: str) -> None:
    (_, _, va, fa), (_, _, vb, fb) = _capsule_halves(device, 16, 11, phase=0.3)

    vertices_ab, faces_ab = tw.stitching.stitch(va, fa, vb, fb)
    vertices_ba, faces_ba = tw.stitching.stitch(vb, fb, va, fa)

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
        tw.stitching.stitch(mesh_wp.points, mesh_wp.indices, va, fa)


def test_stitch_requires_single_boundary_multi(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = half_torus
    _, _, va, fa = _cone_wp(device=str(mesh_wp.device), n=10, apex_z=-1.0, rim_z=0.0)

    assert len(_loop_sizes(mesh_wp)) >= 2
    with pytest.raises(ValueError, match="exactly one boundary loop"):
        tw.stitching.stitch(mesh_wp.points, mesh_wp.indices, va, fa)


def test_triangulate_boundaries_rejects_small_loop(device: str) -> None:
    _, _, va, fa = _cone_wp(device=device, n=8, apex_z=-1.0, rim_z=0.0)
    _, _, vb, fb = _cone_wp(device=device, n=8, apex_z=1.0, rim_z=0.5)

    loop_a = tw.boundary.boundary_loop(va, fa)
    tiny_loop = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="at least 3 vertices"):
        tw.stitching.triangulate_boundaries(va, fa, loop_a, vb, fb, tiny_loop)


def test_non_increasing_indices() -> None:
    # The longest non-decreasing subsequence keeps the repeated 1s and 4s; only 5 (at index 4)
    # falls outside it, so its index is flagged for correction.
    numbers = np.array([0, 1, 1, 2, 5, 3, 4, 4, 7], dtype=np.int64)
    assert np.array_equal(_non_increasing_indices(numbers), np.array([4]))

    # A strictly sorted sequence needs no correction.
    assert _non_increasing_indices(np.arange(6, dtype=np.int64)).size == 0
