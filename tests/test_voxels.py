"""
Tests for ``triwarp.voxels``.

Four references, each where it is the closest thing to an oracle: **open3d** for the two
voxelizers, the down-sample and the membership query (it computes the same tri-box accept set and
the same half-voxel anchor); **scipy.ndimage** through trimesh's ``voxel.morphology`` for dilation,
erosion and hole filling, which are exactly its binary morphology; **trimesh** for
``fill_orthographic`` and ``multibox``; and **igl** for ``grid`` and
``unique_sparse_voxel_corners``.

Two conventions matter throughout:

- **Every fixture is translated by a non-round offset.** open3d floors cell coordinates in
  ``float64`` and triwarp in ``float32``, so a vertex sitting exactly on a cell plane can land in
  different cells. Nothing here is allowed to sit on that edge.
- **Row order is leaf-major**, not lexicographic, so every set comparison sorts both sides. The
  order is nonetheless deterministic and identical on CPU and CUDA, which
  ``test_cell_order_is_deterministic`` and ``test_cell_order_matches_across_devices`` assert
  directly — the device-agnostic design rests on it.

The module runs on the ordinary ``device`` fixture like every other one: ``allocate_by_voxels``
gained a CPU path in Warp 1.16, so nothing here is CUDA-only.
"""

from __future__ import annotations

import igl
import numpy as np
import open3d as o3d
import pytest
import scipy.ndimage as ndi
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.comparisons import lexsort_rows
from tests.conversions import (
    meshlib_bitset_to_numpy,
    numpy_to_meshlib,
    numpy_to_meshlib_bitset,
    numpy_to_warp,
    points_to_meshlib,
    trimesh_to_open3d,
    trimesh_to_pyvista,
)

# A translation with no round coordinate, so no vertex of any fixture lands on a cell plane.
_OFFSET = np.array([0.137, -0.219, 0.331])

# igl's ``unique_sparse_voxel_corners`` numbers a cell's corners in yxz binary-counting order and
# reads column 0 of its subscripts as *y*; this is the resulting permutation of triwarp's
# ``c = 4 dx + 2 dy + dz`` columns (it is its own inverse).
_IGL_CORNER_ORDER = [1, 0, 2, 3, 5, 4, 6, 7]


@pytest.fixture
def sphere(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> tuple[tm.Trimesh, wp.array, wp.array]:
    """
    Move the shared ``icosphere`` to a non-round offset, as trimesh plus its Warp buffers.

    The translation is the point of the local fixture: a voxel grid built around a sphere centred
    on the origin can pass while every index is off by half a cell, because the offsets cancel.
    """
    mesh_tm, _mesh_wp = icosphere
    mesh_tm.apply_translation(_OFFSET)
    return (mesh_tm, *numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device))


def _cloud(mesh_tm: tm.Trimesh, device: str, n: int = 20000) -> tuple[np.ndarray, wp.array]:
    """Sample the surface of ``mesh_tm`` deterministically, as NumPy and Warp."""
    points_np = tm.sample.sample_surface(mesh_tm, n, seed=7)[0]
    return points_np, wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device
    )


def _dense(grid: wp.Volume, origin_cell, shape) -> np.ndarray:
    """Dense occupancy of ``grid`` over an explicit cell box, as NumPy."""
    return tw.voxels.to_dense(grid, origin_cell=origin_cell, shape=shape)[0].numpy()


# ---------------------------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parity("voxelize_mesh", "open3d")
def test_voxelize_mesh_matches_open3d(sphere, device: str):
    """
    Class A: the accepted cell sets are identical, compared after ``lexsort_rows`` on both sides.

    open3d walks a ``round((max - min) / voxel_size) + 2`` candidate window and triwarp the exact
    inclusive AABB window, but a cell outside a triangle's AABB cannot overlap the triangle, so the
    windows differ only in rejected candidates. Both then run Moller's 13-axis test with no epsilon.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.13
    origin = mesh_tm.vertices.min(axis=0) - 0.5 * voxel_size
    grid = tw.voxels.voxelize_mesh(
        vertices_wp, faces_wp, voxel_size, origin=wp.vec3(*origin.tolist())
    )

    grid_o3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        trimesh_to_open3d(mesh_tm),
        voxel_size=voxel_size,
        min_bound=origin,
        max_bound=origin
        + np.ceil((mesh_tm.vertices.max(axis=0) - origin) / voxel_size) * voxel_size,
    )
    cells_o3d = np.array([voxel.grid_index for voxel in grid_o3d.get_voxels()], dtype=np.int32)

    assert cells_o3d.shape[0] > 0
    assert np.array_equal(lexsort_rows(tw.voxels.cells(grid).numpy()), lexsort_rows(cells_o3d))


@pytest.mark.parity("voxelize_mesh", "meshlib")
def test_voxelize_mesh_matches_meshlib(sphere, device: str):
    """
    Class C (two representations): MeshLib returns a **distance field**, not an occupancy set.

    ``meshToVolume`` builds an OpenVDB narrow band -- unsigned distances in *voxel* units, clamped
    at ``surfaceOffset`` -- so there is no cell set to compare with. What is comparable is the
    geometry both encode, and the two directions of that are exact statements rather than
    tolerances:

    - a cell triwarp accepts contains a piece of the surface, so its **centre** is within half a
      cell diagonal of it: ``distance <= sqrt(3)/2`` voxels;
    - a grid node whose distance is under **half** a voxel has its closest surface point inside the
      surrounding cell, so triwarp must have accepted that cell.

    Measured on the translated ``icosphere(3)`` at a 0.1 voxel: all 1 898 accepted cells sample at
    most **0.806** against the 0.866 bound, and all 1 226 nodes under 0.5 land inside an accepted
    cell -- both at 100 %, in both directions.

    Three facts about the grid, none of them documented and all of them load-bearing. The field is
    read through ``vdbVolumeToSimpleVolume`` and ``mn.getNumpy3Darray``; its samples sit on grid
    **nodes**, not cell centres; and its origin is ``bbox.min - surfaceOffset * voxel``, which is
    also why ``dims`` comes out as the mesh's extent in cells plus ``2 * surfaceOffset``.

    **Bug class excluded:** a grid anchored differently -- the failure mode every voxel comparison
    in this module is written against. **Mutation probe, measured:** displacing the sample points by
    half a voxel puts **19.3 %** of the cells outside the bound and takes the maximum from 0.806 to
    1.661, so the assert bites at exactly the error it is for.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size, surface_offset = 0.1, 3.0

    grid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    centres_np = tw.voxels.cell_centers(grid).numpy().astype(np.float64)

    vertices_np = vertices_wp.numpy().astype(np.float64)
    mesh_ml = numpy_to_meshlib(vertices_np, faces_wp.numpy().reshape(-1, 3))
    params_ml = mm.MeshToVolumeParams()
    params_ml.voxelSize = mm.Vector3f(voxel_size, voxel_size, voxel_size)
    params_ml.surfaceOffset = surface_offset
    field_np = mn.getNumpy3Darray(
        mm.vdbVolumeToSimpleVolume(mm.meshToVolume(mm.MeshPart(mesh_ml), params_ml))
    )
    origin_np = vertices_np.min(axis=0) - surface_offset * voxel_size

    # Non-vacuity: a narrow band around the surface, not a constant field.
    assert field_np.min() < 0.1 < field_np.max()
    assert centres_np.shape[0] > 100

    # 1. Every accepted cell's centre is within half a cell diagonal of the surface.
    sampled_np = ndi.map_coordinates(
        field_np, ((centres_np - origin_np) / voxel_size).T, order=1, mode="nearest"
    )
    assert sampled_np.max() <= np.sqrt(3.0) / 2.0

    # 2. Every node closer than half a voxel lies inside a cell triwarp accepted.
    near_np = np.argwhere(field_np < 0.5)
    assert near_np.shape[0] > 100
    node_positions_np = origin_np + near_np * voxel_size
    occupied_wp = tw.voxels.occupancy_at_points(
        grid,
        wp.array(
            np.ascontiguousarray(node_positions_np, dtype=np.float32), dtype=wp.vec3, device=device
        ),
    )
    assert occupied_wp.numpy().all()


@pytest.mark.parity("voxelize_mesh", "pyvista")
def test_voxelize_mesh_solid_contains_the_pyvista_mask(sphere, device: str):
    """
    Class B, and the named transform is the **sample convention**, which is the whole finding.

    ``voxelize_binary_mask`` is *solid*, so it maps to ``mode="solid"`` and not to the default
    surface mode -- measured 17 256 filled of 32 768 on a 32-cube against triwarp's 4 496 surface
    cells. But it is not the same set even then: VTK writes a **point** mask, testing each lattice
    *point* against the closed surface, while triwarp accepts a **cell** the closed triangle meets.
    So VTK's set is contained in triwarp's and the difference is exactly the boundary shell, which
    is what the asserts below state: **0** cells are pyvista-only, 2 480 are triwarp-only, and every
    one of those 2 480 is in triwarp's own surface voxelization (of 4 760 surface cells).

    Reading that containment as a disagreement is the trap; a comparison that expected equality
    would fail by 12.7% of the cells on a correct implementation. The grid is also cell-centred on
    VTK's side -- origin ``-0.96875`` at spacing ``0.0625`` for a unit sphere -- so triwarp's origin
    is shifted by half a voxel to put the two lattices in register.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    dimensions = (32, 32, 32)

    mask_pv = trimesh_to_pyvista(mesh_tm).voxelize_binary_mask(dimensions=dimensions)
    spacing_np = np.asarray(mask_pv.spacing)
    origin_np = np.asarray(mask_pv.origin)
    mask_np = np.asarray(mask_pv.point_data["mask"]).astype(bool).reshape(dimensions, order="F")
    assert 0 < int(mask_np.sum()) < mask_np.size, "the reference filled some cells, not all"

    origin_wp = wp.vec3(*(origin_np - 0.5 * spacing_np).tolist())
    solid = tw.voxels.voxelize_mesh(
        vertices_wp, faces_wp, float(spacing_np[0]), origin=origin_wp, mode="solid"
    )
    surface = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, float(spacing_np[0]), origin=origin_wp)

    def dense(grid: wp.Volume) -> np.ndarray:
        cells_np = tw.voxels.cells(grid).numpy()
        inside_np = np.all((cells_np >= 0) & (cells_np < np.array(dimensions)), axis=1)
        occupied_np = np.zeros(dimensions, dtype=bool)
        kept_np = cells_np[inside_np]
        occupied_np[kept_np[:, 0], kept_np[:, 1], kept_np[:, 2]] = True
        return occupied_np

    solid_np, surface_np = dense(solid), dense(surface)
    assert not (mask_np & ~solid_np).any(), "VTK's point mask must be contained in the solid set"
    excess_np = solid_np & ~mask_np
    assert excess_np.any(), "and the two conventions must actually differ, or the row is vacuous"
    assert not (excess_np & ~surface_np).any(), "every extra cell is a boundary-shell cell"


def test_voxelize_mesh_solid_is_sealed(sphere, device: str):
    """``mode="solid"`` fills the interior, and no 6-connected path leaves it."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    surface = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.12)
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.12, mode="solid")
    assert int(tw.voxels.cells(solid).shape[0]) > int(tw.voxels.cells(surface).shape[0])

    # The padded shell of the solid's own bounding box must stay empty, and the interior must be
    # separated from it: filling again changes nothing.
    refilled = tw.voxels.fill_holes(solid)
    lower, extent = _bounds_of(solid)
    assert np.array_equal(_dense(solid, lower, extent), _dense(refilled, lower, extent))
    assert tw.voxels.occupancy_at_points(
        solid, wp.array([wp.vec3(*_OFFSET.tolist())], dtype=wp.vec3, device=device)
    ).numpy()[0]


def test_voxelize_mesh_rejects_too_many_candidates(sphere, device: str):
    """The candidate guard raises before allocating, and names ``voxel_size``."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    with pytest.raises(ValueError, match="voxel_size"):
        tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 1e-4, max_candidates=1000)


@pytest.mark.parity("voxelize_points", "open3d")
def test_voxelize_points_matches_open3d(sphere, device: str):
    """
    Class A: same occupied cells for the same cloud and pitch, both sides ``lexsort``ed.

    Also pins the default anchor: triwarp's ``origin`` default is open3d's
    ``min_bound - 0.5 * voxel_size``, so the two agree cell for cell without either being told the
    other's transform.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)
    voxel_size = 0.13

    grid = tw.voxels.voxelize_points(points_wp, voxel_size)
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
    cells_o3d = np.array([voxel.grid_index for voxel in grid_o3d.get_voxels()], dtype=np.int32)

    assert cells_o3d.shape[0] > 0
    assert np.array_equal(lexsort_rows(tw.voxels.cells(grid).numpy()), lexsort_rows(cells_o3d))
    assert np.allclose(np.asarray(grid_o3d.origin), tw.voxels.grid_transform(grid)[1], atol=1e-6)


@pytest.mark.parity("voxel_down_sample", "open3d")
def test_voxel_down_sample_matches_open3d(sphere, device: str):
    """
    Class B: both sides re-keyed by the integer cell before their positions are compared.

    The positions themselves are never ``lexsort``ed — two cell means that tie in ``float32`` and
    differ in the 16th digit in ``float64`` would order differently and fail by the full coordinate
    range. The integer cell is the correspondence; the positions are then compared in place.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)
    voxel_size = 0.13

    grid = tw.voxels.voxelize_points(points_wp, voxel_size)
    _size, origin = tw.voxels.grid_transform(grid)
    pooled_wp = tw.voxels.voxel_down_sample(points_wp, voxel_size)
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    pooled_o3d = np.asarray(cloud_o3d.voxel_down_sample(voxel_size).points)

    assert pooled_o3d.shape[0] > 0
    assert pooled_wp.shape[0] == pooled_o3d.shape[0]
    key_o3d = tw.voxels.cell_indices(
        wp.array(np.ascontiguousarray(pooled_o3d, dtype=np.float32), dtype=wp.vec3, device=device),
        voxel_size,
        origin=origin,
    ).numpy()
    key_wp = tw.voxels.cells(grid).numpy()
    order_o3d = np.lexsort(key_o3d.T[::-1])
    order_wp = np.lexsort(key_wp.T[::-1])
    assert np.array_equal(key_o3d[order_o3d], key_wp[order_wp])
    assert np.allclose(pooled_wp.numpy()[order_wp], pooled_o3d[order_o3d], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("voxel_size", [0.13, 0.26])
@pytest.mark.parity("voxel_down_sample", "meshlib")
def test_voxel_down_sample_matches_meshlib(sphere, device: str, voxel_size: float):
    """
    Class C (a count statistic): ``pointGridSampling`` *selects* a point, it does not pool one.

    The difference is the operation, not the parameter: MeshLib returns a ``VertBitSet`` picking one
    surviving input point per occupied cell, where ``voxel_down_sample`` returns the cell **mean**,
    a position that need not be an input point at all. So the positions are not comparable and the
    shared quantity is how many cells the cloud occupies.

    Measured on the translated icosphere cloud, triwarp against MeshLib: 429 / 464 at a 5 % voxel
    and 136 / 128 at 10 %. The residual is grid *anchoring* -- neither library documents where cell
    zero starts -- so the bound is 1.3x, the same one ``cluster_decimate`` carries against
    ``verticesGridSampling`` for the same reason.

    **Mutation probe, measured:** the count falls ~3x per doubling of the voxel size on both sides,
    so a factor-of-two error in the cell size lands far outside the 1.3x band that anchoring
    accounts for. open3d carries the exact positional comparison for this group.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device)

    pooled_wp = tw.voxels.voxel_down_sample(points_wp, voxel_size)
    cloud_ml = points_to_meshlib(points_np)
    sampled_ml = mm.pointGridSampling(mm.PointCloudPart(cloud_ml), voxel_size)

    n_cells_ml = sampled_ml.count()
    n_cells_wp = int(pooled_wp.shape[0])
    assert 0 < n_cells_ml < points_np.shape[0]  # non-vacuity: it really sampled down
    assert 0 < n_cells_wp < points_np.shape[0]
    assert 1.0 / 1.3 < n_cells_wp / n_cells_ml < 1.3


def test_voxel_down_sample_pooling_modes(sphere, device: str):
    """``min`` / ``max`` bracket the cell mean, and ``sum`` is that mean times the count."""
    mesh_tm, _vertices_wp, _faces_wp = sphere
    _points_np, points_wp = _cloud(mesh_tm, device, n=4000)
    voxel_size = 0.2

    lowest = tw.voxels.voxel_down_sample(points_wp, voxel_size, pooling="min").numpy()
    highest = tw.voxels.voxel_down_sample(points_wp, voxel_size, pooling="max").numpy()
    average, inverse = tw.voxels.voxel_down_sample(
        points_wp, voxel_size, pooling="mean", return_inverse=True
    )
    total = tw.voxels.voxel_down_sample(points_wp, voxel_size, pooling="sum").numpy()

    assert (lowest <= average.numpy() + 1e-6).all()
    assert (average.numpy() <= highest + 1e-6).all()
    counts = np.bincount(inverse.numpy(), minlength=average.shape[0])
    assert counts.min() > 0
    assert np.allclose(total, average.numpy() * counts[:, None], rtol=1e-5, atol=1e-5)


def test_voxel_down_sample_mean_is_reproducible(sphere, device: str):
    """The deterministic branch is bitwise stable: no float atomics in the ``mean`` reduce."""
    mesh_tm, _vertices_wp, _faces_wp = sphere
    _points_np, points_wp = _cloud(mesh_tm, device, n=8000)
    first = tw.voxels.voxel_down_sample(points_wp, 0.15).numpy()
    second = tw.voxels.voxel_down_sample(points_wp, 0.15).numpy()
    assert np.array_equal(first, second)


def test_pool_by_voxel_ignores_outside_points(device: str):
    """Points outside the grid contribute nothing, and voxels with no point pool to zero."""
    cells_np = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int32)
    grid = tw.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 1.0, wp.vec3(0.0, 0.0, 0.0)
    )
    points_np = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [-9.0, -9.0, -9.0]])
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    pooled = tw.voxels.pool_by_voxel(grid, points_wp, points_wp).numpy()

    slot = tw.voxels.cells(grid).numpy().tolist().index([0, 0, 0])
    other = 1 - slot
    assert np.allclose(pooled[slot], [0.5, 0.5, 0.5])
    assert np.array_equal(pooled[other], np.zeros(3))


# ---------------------------------------------------------------------------------------------
# Grid <-> world
# ---------------------------------------------------------------------------------------------


def test_cells_round_trip_through_from_cells(device: str):
    """
    The half-voxel translation, pinned directly.

    NanoVDB centres voxel ``i`` on index-space coordinate ``i``, so a transform anchored at
    ``origin`` rather than ``origin + 0.5 * voxel_size`` returns every cell shifted by ``+1`` on
    every axis. The set below includes negatives and a coordinate past the first 8-cubed leaf, so
    that failure mode cannot hide.
    """
    cells_np = np.array([[0, 0, 0], [3, -4, 5], [-7, 2, 9], [63, 63, 63]], dtype=np.int32)
    grid = tw.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.25, wp.vec3(1.3, -2.7, 0.4)
    )
    assert np.array_equal(lexsort_rows(tw.voxels.cells(grid).numpy()), lexsort_rows(cells_np))

    voxel_size, origin = tw.voxels.grid_transform(grid)
    assert voxel_size == pytest.approx(0.25)
    assert np.allclose(origin, [1.3, -2.7, 0.4], atol=1e-6)


def test_cell_slots_are_the_row_indices(device: str):
    """
    The volume's linear index *is* the row index of [`cells`][triwarp.voxels.cells].

    Every payload array in the module depends on it, so probing the cells back must give
    ``0..n-1`` in order.
    """
    rng = np.random.default_rng(11)
    cells_np = np.unique(rng.integers(-6, 7, size=(400, 3)).astype(np.int32), axis=0)
    grid = tw.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
    )
    voxels = tw.voxels.cells(grid)
    assert int(voxels.shape[0]) == cells_np.shape[0]
    assert tw.voxels.occupancy_at_cells(grid, voxels).numpy().all()

    centers = tw.voxels.cell_centers(grid)
    assert tw.voxels.occupancy_at_points(grid, centers).numpy().all()
    # The centre of row k must probe back to row k, which is the statement above.
    pooled = tw.voxels.pool_by_voxel(grid, centers, centers).numpy()
    assert np.allclose(pooled, centers.numpy(), rtol=1e-6, atol=1e-6)


def test_cell_order_is_deterministic(device: str):
    """
    Class of bug this excludes: a build order that depends on the input order.

    Two builds from differently shuffled cell rows return byte-identical
    [`cells`][triwarp.voxels.cells]. The order is *leaf-major*, checked here on a multi-leaf set by
    an explicit example rather than by global sortedness, which is false for it.
    """
    rng = np.random.default_rng(3)
    cells_np = np.unique(rng.integers(-9, 10, size=(500, 3)).astype(np.int32), axis=0)
    first = tw.voxels.cells(
        tw.voxels.from_cells(
            wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
        )
    ).numpy()
    shuffled = cells_np[rng.permutation(cells_np.shape[0])]
    second = tw.voxels.cells(
        tw.voxels.from_cells(
            wp.array(np.ascontiguousarray(shuffled), dtype=wp.int32, device=device),
            0.5,
            wp.vec3(0.0, 0.0, 0.0),
        )
    ).numpy()
    assert np.array_equal(first, second)

    # Leaf-major: an 8-cubed leaf's cells come out together, ahead of the next leaf's, even though
    # a globally lexicographic order would interleave them.
    block = np.array([[0, 0, 0], [1, 1, 1], [8, 0, 0], [0, 0, 8]], dtype=np.int32)
    rows = tw.voxels.cells(
        tw.voxels.from_cells(
            wp.array(block, dtype=wp.int32, device=device), 1.0, wp.vec3(0.0, 0.0, 0.0)
        )
    ).numpy()
    leaves = rows // 8
    assert np.array_equal(leaves, lexsort_rows(leaves))
    assert not np.array_equal(rows, lexsort_rows(rows))


def test_cell_order_matches_across_devices():
    """
    The claim the whole device-agnostic design rests on: CPU and CUDA return the same permutation.

    Reported as a skip rather than passing silently when only one device is available — a
    one-device run would leave the claim untested while looking green.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both a CPU and a CUDA device to compare the two row orders")
    rng = np.random.default_rng(5)
    cells_np = np.ascontiguousarray(
        np.unique(rng.integers(-9, 10, size=(499, 3)).astype(np.int32), axis=0)
    )
    rows = [
        tw.voxels.cells(
            tw.voxels.from_cells(
                wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
            )
        ).numpy()
        for device in ("cpu", "cuda:0")
    ]
    assert np.array_equal(rows[0], rows[1])


def test_cells_sorted_order_is_the_unique_rows_order(device: str):
    """``order="sorted"`` reproduces ``grouping.unique_rows``'s row order exactly."""
    rng = np.random.default_rng(17)
    cells_np = np.ascontiguousarray(rng.integers(0, 12, size=(600, 3)).astype(np.int32))
    cells_wp = wp.array(cells_np, dtype=wp.int32, device=device)
    grid = tw.voxels.from_cells(cells_wp, 0.5, wp.vec3(0.0, 0.0, 0.0))
    assert np.array_equal(
        tw.voxels.cells(grid, order="sorted").numpy(), tw.grouping.unique_rows(cells_wp).numpy()
    )


def test_cells_sorted_order_handles_negative_cells(device: str):
    """The shift that makes the packed key well defined is order-preserving, negatives included."""
    cells_np = np.array([[0, 0, 0], [3, -4, 5], [-7, 2, 9], [-7, 2, 8]], dtype=np.int32)
    grid = tw.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.5, wp.vec3(0.0, 0.0, 0.0)
    )
    rows = tw.voxels.cells(grid, order="sorted").numpy()
    # Ascending in the packed key means lexicographic with x as the *last* tiebreak.
    assert np.array_equal(rows, rows[np.lexsort((rows[:, 0], rows[:, 1], rows[:, 2]))])


@pytest.mark.parity("occupancy_at_points", "open3d")
def test_occupancy_at_points_matches_open3d(sphere, device: str):
    """
    Class A: identical boolean masks against ``check_if_included``.

    The query set holds points both inside and outside, so a constant answer cannot pass.
    """
    mesh_tm, _vertices_wp, _faces_wp = sphere
    points_np, points_wp = _cloud(mesh_tm, device, n=5000)
    voxel_size = 0.15
    grid = tw.voxels.voxelize_points(points_wp, voxel_size)

    queries_np = np.concatenate([points_np[:400], points_np[:400] + 5.0])
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
    inside_o3d = np.array(
        grid_o3d.check_if_included(o3d.utility.Vector3dVector(queries_np)), dtype=bool
    )

    assert inside_o3d.any()
    assert not inside_o3d.all()
    assert np.array_equal(tw.voxels.occupancy_at_points(grid, queries_wp).numpy(), inside_o3d)


@pytest.mark.parity("grid_points", "igl")
def test_grid_points_matches_igl(device: str):
    """
    Class B: ``igl.grid`` emits the same lattice over the unit cube.

    Compared as sets, because igl's flattening order is its own.
    """
    shape = (4, 5, 6)
    lattice_wp = tw.voxels.grid_points(
        shape, bounds=(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0)), device=device
    ).numpy()
    lattice_igl = igl.grid(np.array(shape))

    assert lattice_igl.shape == lattice_wp.shape
    assert np.allclose(
        lexsort_rows(np.round(lattice_igl, 9)), lexsort_rows(np.round(lattice_wp, 9)), atol=1e-6
    )
    # C order, z fastest, and the two corners land on ``bounds``.
    assert np.allclose(lattice_wp[0], [0.0, 0.0, 0.0])
    assert np.allclose(lattice_wp[-1], [1.0, 1.0, 1.0])
    assert np.allclose(lattice_wp[1], [0.0, 0.0, 0.2])


def test_grid_points_defaults_to_index_space(device: str):
    """``bounds=None`` gives lattice indices, matching ``reconstruction.marching_cubes``."""
    lattice = tw.voxels.grid_points((2, 3, 4), device=device).numpy()
    expected = np.stack(np.meshgrid(*(np.arange(n) for n in (2, 3, 4)), indexing="ij"), -1)
    assert np.array_equal(lattice, expected.reshape(-1, 3).astype(np.float32))


# ---------------------------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------------------------


@pytest.mark.parity("dilate", "trimesh")
def test_dilate_matches_scipy(sphere, device: str):
    """
    Class A: dense occupancy against ``scipy.ndimage.binary_dilation``.

    Its default structure is what trimesh's ``morphology.binary_dilation`` passes.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)
    assert occupancy_np.any()

    padded_lower = tuple(c - 1 for c in lower)
    padded_shape = tuple(n + 2 for n in extent)
    dilated = _dense(tw.voxels.dilate(grid), padded_lower, padded_shape)
    assert np.array_equal(dilated, ndi.binary_dilation(np.pad(occupancy_np, 1)))


@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_erode_matches_scipy(sphere, device: str, connectivity: int):
    """Class A: erosion against ``scipy.ndimage.binary_erosion`` at the matching structure rank."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.14, mode="solid")
    lower, extent = _bounds_of(solid)
    occupancy_np = _dense(solid, lower, extent)
    rank = {6: 1, 18: 2, 26: 3}[connectivity]

    eroded = _dense(tw.voxels.erode(solid, connectivity=connectivity), lower, extent)
    reference = ndi.binary_erosion(
        occupancy_np, structure=ndi.generate_binary_structure(3, rank), border_value=0
    )
    assert reference.any()
    assert np.array_equal(eroded, reference)


def _voxel_mask_ml(occupancy_np: np.ndarray) -> tuple[mm.VoxelBitSet, mm.VolumeIndexer]:
    """
    Load a dense occupancy array into a ``VoxelBitSet`` plus the ``VolumeIndexer`` addressing it.

    Both MeshLab morphology calls take the pair and **mutate the bitset in place**, returning
    ``None``; the indexer is what carries the dimensions, and it fixes the bit order --
    ``VolumeIndexer::toPos`` decodes ``x + dims.x * y + dims.x * dims.y * z``, so ``x`` runs
    fastest and the dense array flattens with ``order="F"``.

    The load itself is one ``np.packbits`` through
    [`numpy_to_meshlib_bitset`][tests.conversions.numpy_to_meshlib_bitset]; ``VoxelBitSet``'s
    converting constructor then copies the bits and the size out of the untyped set.
    """
    dims_ml = mm.Vector3i(*(int(n) for n in occupancy_np.shape))
    indexer_ml = mm.VolumeIndexer(dims_ml)
    mask_ml = mm.VoxelBitSet(numpy_to_meshlib_bitset(occupancy_np.ravel(order="F")))
    return mask_ml, indexer_ml


def _dense_from_mask_ml(mask_ml: mm.VoxelBitSet, shape) -> np.ndarray:
    """
    Read a ``VoxelBitSet`` back as a dense bool array of ``shape``.

    ``mn.getNumpyBitSet`` is declared over ``const MR::BitSet&`` and ``VoxelBitSet`` derives from
    it, so pybind11 upcasts and the whole set comes back in one call, flat in ``VoxelId`` order --
    the inverse of the ``order="F"`` flattening above, hence ``reshape(shape[::-1]).T``.
    """
    return meshlib_bitset_to_numpy(mask_ml, int(np.prod(shape))).reshape(shape[::-1]).T


@pytest.mark.parity("dilate", "meshlib")
@pytest.mark.parity(
    "erode",
    "meshlib",
    benchmarked=False,
    reason="erosion has no benchmark group of its own: it is dilation's complement over the same "
    "candidate buffer, and a second row would price the same pass under another name. The dilate "
    "group carries the timed MeshLib row, and test_erode_matches_scipy is the scipy oracle.",
)
def test_dilate_and_erode_match_meshlib(sphere, device: str):
    """
    Class A on the occupancy: ``expandVoxelsMask`` / ``shrinkVoxelsMask`` are the 6-neighbour forms.

    MeshLib exposes no connectivity switch -- its neighbourhood is the six face-adjacent voxels,
    which is triwarp's *default* and not its 18- or 26-connected settings, so the comparison is
    pinned at ``connectivity=6`` and the other two are excluded by construction rather than by
    tolerance. Verified on a 2x2x2 block: 8 voxels dilate to 32 under both, where triwarp's
    26-connected setting gives 64.

    Two interface facts the helpers above encode: both calls **mutate the bitset and return
    ``None``**, and the bitset is addressed in ``VoxelId`` order (``x`` fastest) rather than in the
    dense array's C order. Neither direction needs a per-cell Python loop -- ``BitSet.fromBlocks``
    loads the packed ``uint64`` blocks and ``mn.getNumpyBitSet`` upcasts the ``VoxelBitSet`` back to
    a flat bool array -- which is what lets ``benchmarks/test_voxels.py`` time the morphology itself
    with the load hoisted into ``pedantic``'s untimed ``setup``.

    The dense box is padded by one cell on every side so a dilation has somewhere to go; both sides
    see the identical box, so the comparison is element-wise over the whole array.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.2, mode="solid")
    lower, extent = _bounds_of(solid)
    padded_lower = tuple(c - 1 for c in lower)
    padded_shape = tuple(n + 2 for n in extent)
    occupancy_np = _dense(solid, padded_lower, padded_shape)
    assert occupancy_np.any()

    dilated_np = _dense(tw.voxels.dilate(solid), padded_lower, padded_shape)
    eroded_np = _dense(tw.voxels.erode(solid), padded_lower, padded_shape)

    mask_ml, indexer_ml = _voxel_mask_ml(occupancy_np)
    assert mask_ml.count() == int(occupancy_np.sum())  # non-vacuity: the load round-trips
    assert mm.expandVoxelsMask(mask_ml, indexer_ml, 1) is None  # mutates, returns nothing
    assert np.array_equal(_dense_from_mask_ml(mask_ml, padded_shape), dilated_np)

    mask_ml, indexer_ml = _voxel_mask_ml(occupancy_np)
    mm.shrinkVoxelsMask(mask_ml, indexer_ml, 1)
    assert np.array_equal(_dense_from_mask_ml(mask_ml, padded_shape), eroded_np)
    assert eroded_np.sum() < occupancy_np.sum() < dilated_np.sum()


def test_surface_voxels_is_the_erosion_complement(sphere, device: str):
    """``surface_voxels`` is exactly ``grid`` minus ``erode(grid)``, by definition of the shell."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.14, mode="solid")
    lower, extent = _bounds_of(solid)
    occupancy_np = _dense(solid, lower, extent)
    eroded = _dense(tw.voxels.erode(solid), lower, extent)
    shell = _dense(tw.voxels.surface_voxels(solid), lower, extent)
    assert shell.any()
    assert eroded.any()
    assert np.array_equal(shell, occupancy_np & ~eroded)


@pytest.mark.parity("fill_holes", "trimesh")
def test_fill_holes_matches_scipy(cave_cube, device: str):
    """
    Class A: dense occupancy against ``scipy.ndimage.binary_fill_holes``.

    Run on ``cave_cube``, a box with a box-shaped void, so the fill is emphatically not a no-op --
    a fixture with nothing to fill would pass while testing nothing.
    """
    mesh_tm, mesh_wp = cave_cube
    grid = tw.voxels.voxelize_mesh(mesh_wp.points, mesh_wp.indices, 0.02)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)

    filled = _dense(tw.voxels.fill_holes(grid), lower, extent)
    reference = ndi.binary_fill_holes(occupancy_np)
    assert reference.sum() > occupancy_np.sum()
    assert np.array_equal(filled, reference)
    assert mesh_tm.is_watertight


@pytest.mark.parity("fill_orthographic", "trimesh")
def test_fill_orthographic_matches_trimesh(sphere, device: str):
    """Class A: dense occupancy against ``trimesh.voxel.ops.fill_orthographic``."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    grid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, 0.16)
    lower, extent = _bounds_of(grid)
    occupancy_np = _dense(grid, lower, extent)

    filled = _dense(tw.voxels.fill_orthographic(grid), lower, extent)
    reference = tm.voxel.ops.fill_orthographic(occupancy_np)
    assert reference.sum() > occupancy_np.sum()
    assert np.array_equal(filled, reference)


# ---------------------------------------------------------------------------------------------
# resolve_voxel_grid
# ---------------------------------------------------------------------------------------------


def test_resolve_voxel_grid_defaults_are_one_percent_and_a_half_cell_below(device: str) -> None:
    """
    Class A: the package's single definition of an unspecified voxel grid, against the arithmetic.

    A cell of ``1 %`` of the bounding-box diagonal, anchored half a cell below the lower corner --
    Open3D's anchor, and the reason ``voxelize_points``, ``voxel_down_sample`` and
    ``cluster_decimate`` agree cell for cell. Both defaults resolve independently, so each is
    checked with the other supplied: a rule applied only when *both* are ``None`` would pass a test
    that always omits both.

    ``atol`` rather than exact: ``origin`` is a ``wp.vec3``, so the returned corner is ``float32``
    where the arithmetic here is ``float64`` (measured gap 7e-10).
    """
    points_np = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    diagonal = float(np.linalg.norm(points_np[1] - points_np[0]))

    voxel_size, origin = tw.voxels.resolve_voxel_grid(points_wp)

    assert voxel_size == pytest.approx(0.01 * diagonal)
    assert np.allclose(list(origin), -0.5 * voxel_size, rtol=0, atol=1e-7)

    # Each default resolves on its own, not only when both are missing.
    size_only, given_origin = tw.voxels.resolve_voxel_grid(points_wp, origin=wp.vec3(9.0, 9.0, 9.0))
    assert size_only == pytest.approx(0.01 * diagonal)
    assert list(given_origin) == [9.0, 9.0, 9.0]
    given_size, origin_only = tw.voxels.resolve_voxel_grid(points_wp, 0.25)
    assert given_size == 0.25
    assert np.allclose(list(origin_only), -0.125, rtol=0, atol=1e-7)


def test_resolve_voxel_grid_passes_both_arguments_through(device: str) -> None:
    """With both supplied it reads no bounds at all, so it must return them unchanged."""
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )

    voxel_size, origin = tw.voxels.resolve_voxel_grid(points_wp, 0.25, wp.vec3(1.0, 2.0, 3.0))

    assert voxel_size == 0.25
    assert list(origin) == [1.0, 2.0, 3.0]


def test_resolve_voxel_grid_empty_input_takes_a_unit_diagonal(device: str) -> None:
    """An empty set has no bounding box, so the documented fallback is a diagonal of one."""
    voxel_size, origin = tw.voxels.resolve_voxel_grid(wp.empty(0, dtype=wp.vec3, device=device))

    assert voxel_size == pytest.approx(0.01)
    assert np.allclose(list(origin), -0.005, rtol=0, atol=1e-7)


def test_resolve_voxel_grid_rejects_a_non_positive_size_and_names_its_caller(device: str) -> None:
    """
    The ``caller`` argument exists so the message names the public entry point, not this helper.

    Both halves are asserted, because a message that silently ignored ``caller`` would still look
    right at the one call site that does not pass it.
    """
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )

    with pytest.raises(ValueError, match="resolve_voxel_grid requires voxel_size > 0"):
        tw.voxels.resolve_voxel_grid(points_wp, 0.0)
    with pytest.raises(ValueError, match="cluster_decimate requires voxel_size > 0"):
        tw.voxels.resolve_voxel_grid(points_wp, -1.0, caller="cluster_decimate")


# ---------------------------------------------------------------------------------------------
# Conversion and meshing
# ---------------------------------------------------------------------------------------------


def test_dense_round_trip_with_negative_cells(device: str):
    """``from_dense(to_dense(g))`` is the identity, and ``origin_cell`` is what makes it one."""
    cells_np = np.array([[-3, -3, -3], [0, 1, 2], [-1, 4, -2], [5, -5, 5]], dtype=np.int32)
    grid = tw.voxels.from_cells(
        wp.array(cells_np, dtype=wp.int32, device=device), 0.4, wp.vec3(-1.0, 2.0, 0.5)
    )
    occupancy, origin_cell = tw.voxels.to_dense(grid)
    assert origin_cell == (-3, -5, -3)

    voxel_size, origin = tw.voxels.grid_transform(grid)
    rebuilt = tw.voxels.from_dense(occupancy, voxel_size, origin, origin_cell=origin_cell)
    assert np.array_equal(lexsort_rows(tw.voxels.cells(rebuilt).numpy()), lexsort_rows(cells_np))


def test_to_field_round_trips_through_marching_cubes(sphere, cave_cube, device: str):
    """
    The ``bounds`` handoff, end to end and on two topologies.

    This is the mistake the module is most exposed to: the same tuple meaning two different things
    to producer and consumer, i.e. an ``(n-1)``-vs-``n`` spacing error or a C-order transposition.
    Either one moves the surface bodily, which the two-way surface-distance bound catches; the
    volume check pins the scale.

    ``cave_cube`` is voxelized in ``"surface"`` mode rather than ``"solid"``, because it is
    precisely a mesh with an enclosed cavity and ``"solid"`` fills that cavity by definition — its
    inner shell is what makes it worth testing here, so it has to survive.
    """
    mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.1
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size, mode="solid")
    field, bounds = tw.voxels.to_field(solid)
    vertices_out, faces_out = tw.reconstruction.marching_cubes(field, 0.5, bounds=bounds)
    surface = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    tm.repair.fix_normals(surface)

    n_voxels = int(tw.voxels.cells(solid).shape[0])
    assert surface.is_watertight
    # The iso-0.5 crossing of a 0/1 field on cell centres lands on the cells' shared faces, so the
    # extracted volume is the occupied cell volume (a little under it, because marching cubes
    # bevels the convex corners). Against the *mesh* volume the bound would have to be 30 %;
    # against its own cells it is tight, which is what makes it a check on ``bounds``.
    assert surface.volume == pytest.approx(n_voxels * voxel_size**3, rel=0.12)
    _assert_surfaces_within(surface, mesh_tm, 2.0 * voxel_size)

    # The cavity fixture, surface mode, both shells present.
    mesh_tm, mesh_wp = cave_cube
    voxel_size = 0.03
    shell = tw.voxels.voxelize_mesh(mesh_wp.points, mesh_wp.indices, voxel_size)
    field, bounds = tw.voxels.to_field(shell)
    vertices_out, faces_out = tw.reconstruction.marching_cubes(field, 0.5, bounds=bounds)
    shell_out = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    assert shell_out.faces.shape[0] > 0
    _assert_surfaces_within(shell_out, mesh_tm, 2.0 * voxel_size)


def test_grid_points_round_trips_through_marching_cubes(icosahedron, device: str):
    """Not a library comparison: the ``bounds`` handoff, sampling an SDF and re-extracting."""
    mesh_tm, mesh_wp = icosahedron
    lower, upper = tw.bounds.aabb_bounds(mesh_wp.points)
    pad = 0.1 * float(wp.length(upper - lower))
    bounds = (
        wp.vec3(*(lower[axis] - pad for axis in range(3))),
        wp.vec3(*(upper[axis] + pad for axis in range(3))),
    )
    shape = (40, 40, 40)
    samples = tw.voxels.grid_points(shape, bounds=bounds, device=device)
    distance = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, samples, sign_mode="winding"
    )
    vertices_out, faces_out = tw.reconstruction.marching_cubes(
        distance.reshape(shape), 0.0, bounds=bounds
    )
    surface = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=True)
    tm.repair.fix_normals(surface)

    assert surface.is_watertight
    assert abs(surface.volume - mesh_tm.volume) / mesh_tm.volume < 0.05


@pytest.mark.parity("to_boxes", "trimesh")
def test_to_boxes_matches_trimesh_multibox(sphere, device: str):
    """
    Class B: ``cull_internal=False`` against ``trimesh.voxel.ops.multibox``.

    Not the triangle centroids: the two split each cube face into triangles along different
    diagonals, so those differ by a third of a cell while the *surface* is identical. And not
    ``lexsort`` on the corner positions either — that is the measured false negative on float rows
    with ties; a ``cKDTree`` bijection is used instead.
    """
    from scipy.spatial import cKDTree

    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    grid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    vertices_out, faces_out = tw.voxels.to_boxes(grid, cull_internal=False)
    boxes_wp = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=False)
    boxes_tm = tm.voxel.ops.multibox(tw.voxels.cell_centers(grid).numpy(), pitch=voxel_size)

    assert boxes_tm.faces.shape[0] > 0
    assert boxes_wp.faces.shape[0] == boxes_tm.faces.shape[0]
    assert boxes_wp.area == pytest.approx(boxes_tm.area, rel=1e-5)

    # multibox translates a template cube per centre, so its shared corners are only equal to
    # within float rounding and ``np.unique`` will not collapse them. Set equality is therefore
    # stated as a two-way nearest-neighbour bound against triwarp's exactly-shared corners.
    corners_wp = np.unique(boxes_wp.vertices[boxes_wp.faces.reshape(-1)], axis=0)
    corners_tm = boxes_tm.vertices[boxes_tm.faces.reshape(-1)]
    assert cKDTree(corners_wp).query(corners_tm)[0].max() < 1e-5
    assert cKDTree(corners_tm).query(corners_wp)[0].max() < 1e-5
    assert boxes_wp.volume == pytest.approx(boxes_tm.volume, rel=1e-5)


def test_to_boxes_culled_is_a_closed_outward_shell(sphere, device: str):
    """The culled shell is watertight and its enclosed volume is the occupied cell volume."""
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    solid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size, mode="solid")
    vertices_out, faces_out = tw.voxels.to_boxes(solid)
    shell = tm.Trimesh(vertices_out.numpy(), faces_out.numpy().reshape(-1, 3), process=False)
    shell.remove_unreferenced_vertices()

    n_voxels = int(tw.voxels.cells(solid).shape[0])
    assert shell.is_watertight
    # Positive volume is the winding claim: an inward-wound shell would report the negative.
    assert shell.volume == pytest.approx(n_voxels * voxel_size**3, rel=1e-5)


@pytest.mark.parity("voxel_corners", "igl")
def test_voxel_corners_matches_igl(sphere, device: str):
    """
    Class B: ``igl.unique_sparse_voxel_corners`` after two named transforms.

    It reads column 0 of a subscript as *y*, and it numbers a cell's corners in yxz
    binary-counting order.

    Its packing radix is ``2 ** depth + 1``, so ``depth`` must be large enough to hold the largest
    cell index; at the documented ``depth=0`` every subscript aliases and it returns a handful of
    corners for hundreds of cells.
    """
    _mesh_tm, vertices_wp, faces_wp = sphere
    voxel_size = 0.2
    grid = tw.voxels.voxelize_mesh(vertices_wp, faces_wp, voxel_size)
    _size, origin = tw.voxels.grid_transform(grid)
    cells_np = tw.voxels.cells(grid).numpy()
    corners_wp, cell_corners_wp = tw.voxels.voxel_corners(grid)

    depth = int(np.ceil(np.log2(cells_np.max() + 2)))
    corners_igl, indices_igl, positions_igl = igl.unique_sparse_voxel_corners(
        np.asarray(origin, dtype=np.float64),
        voxel_size * (1 << depth),
        depth,
        cells_np[:, [1, 0, 2]].astype(np.int64),
    )
    corners_igl = corners_igl[:, [1, 0, 2]].astype(np.int32)

    assert corners_igl.shape[0] > 0
    assert np.array_equal(lexsort_rows(corners_igl), lexsort_rows(corners_wp.numpy()))
    # Per-cell indices agree once igl's corner numbering is permuted onto triwarp's.
    ours = corners_wp.numpy()[cell_corners_wp.numpy()]
    theirs = corners_igl[indices_igl]
    assert np.array_equal(ours, theirs[:, _IGL_CORNER_ORDER, :])
    assert np.allclose(
        positions_igl, corners_igl * voxel_size + np.asarray(origin, dtype=np.float64), atol=1e-6
    )


# ---------------------------------------------------------------------------------------------
# Contracts: empty inputs, rebuildable grids, the lazy fem import
# ---------------------------------------------------------------------------------------------


def test_every_entry_point_survives_an_empty_input(device: str):
    """
    An empty input returns an empty result rather than raising.

    Warp 1.16 raises ``Failed to create volume`` on a zero-point build (1.15 aborted the process),
    so this is a contract every entry point has to hold up on its own.
    """
    no_points = wp.zeros(0, dtype=wp.vec3, device=device)
    no_faces = wp.zeros(0, dtype=wp.int32, device=device)
    empty_cells = wp.zeros((0, 3), dtype=wp.int32, device=device)

    grids = [
        tw.voxels.voxelize_points(no_points),
        tw.voxels.voxelize_mesh(no_points, no_faces),
        tw.voxels.from_cells(empty_cells, 1.0, wp.vec3(0.0, 0.0, 0.0)),
    ]
    for grid in grids:
        assert int(tw.voxels.cells(grid).shape[0]) == 0
        assert int(tw.voxels.cell_centers(grid).shape[0]) == 0
        assert int(tw.voxels.dilate(grid).get_active_stats().voxel_count) == 0
        assert int(tw.voxels.erode(grid).get_active_stats().voxel_count) == 0
        assert int(tw.voxels.surface_voxels(grid).get_active_stats().voxel_count) == 0
        assert int(tw.voxels.fill_holes(grid).get_active_stats().voxel_count) == 0
        assert int(tw.voxels.fill_orthographic(grid).get_active_stats().voxel_count) == 0
        assert int(tw.voxels.pool_by_voxel(grid, no_points, no_points).shape[0]) == 0
        assert int(tw.voxels.voxel_corners(grid)[0].shape[0]) == 0
        assert int(tw.voxels.to_boxes(grid)[1].shape[0]) == 0
    assert int(tw.voxels.voxel_down_sample(no_points).shape[0]) == 0


def test_cells_reports_the_active_length_of_a_rebuildable_grid(device: str):
    """
    A rebuildable grid reports its length through ``get_active_stats``, not ``get_voxel_count``.

    The trap that would otherwise produce a plausible wrong answer: ``get_voxel_count`` reports the
    reserved **capacity**, and the tail of ``get_voxels`` is ``[0, 0, 0]`` padding. Nothing in this
    module builds such a grid, but a caller can hand one in.
    """
    cells_np = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=np.int32)
    grid = wp.Volume.allocate_by_voxels(
        wp.array(cells_np, dtype=wp.int32, device=device),
        voxel_size=1.0,
        translation=(0.5, 0.5, 0.5),
        max_active_voxels=64,
        max_leaf_nodes=64,
        max_lower_nodes=64,
        max_upper_nodes=64,
        device=device,
    )
    assert grid.is_rebuildable
    grid.rebuild(wp.array(cells_np[:2], dtype=wp.int32, device=device))

    assert grid.get_voxel_count() > 2  # the capacity, padded with [0, 0, 0]
    rows = tw.voxels.cells(grid).numpy()
    assert rows.shape[0] == 2
    assert np.array_equal(lexsort_rows(rows), lexsort_rows(cells_np[:2]))


def test_voxels_imports_warp_fem_lazily():
    """
    ``warp.fem`` is imported inside [`voxel_corners`][triwarp.voxels.voxel_corners], not at scope.

    That import costs ~0.15 s, which every caller of the module would otherwise pay.

    Checked statically rather than through ``sys.modules``: ``triwarp/kernels/curvature.py``
    imports ``warp.fem.linalg`` at module scope, which drags the whole ``warp.fem`` package in
    regardless of what this module does, so a runtime probe could never fail.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(tw.voxels))
    module_scope_imports = [
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    ] + [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module]
    assert not any(name.startswith("warp.fem") for name in module_scope_imports)
    # ...and it really is imported somewhere, so this is not vacuous.
    assert "import warp.fem" in inspect.getsource(tw.voxels)


def test_invalid_arguments_raise(device: str):
    """Each documented ``ValueError`` / ``TypeError`` is reachable."""
    points = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    empty_cells = wp.zeros((0, 3), dtype=wp.int32, device=device)
    grid = tw.voxels.voxelize_points(points, 1.0)

    with pytest.raises(ValueError, match="voxel_size"):
        tw.voxels.voxelize_points(points, -1.0)
    with pytest.raises(ValueError, match="mode"):
        tw.voxels.voxelize_mesh(points, wp.zeros(0, dtype=wp.int32, device=device), 1.0, mode="x")
    with pytest.raises(ValueError, match="order"):
        tw.voxels.cells(grid, order="lexicographic")
    with pytest.raises(ValueError, match="pooling"):
        tw.voxels.pool_by_voxel(grid, points, points, pooling="median")
    with pytest.raises(ValueError, match="connectivity"):
        tw.voxels.dilate(grid, connectivity=7)
    with pytest.raises(ValueError, match="three columns"):
        tw.voxels.from_cells(wp.zeros((0, 2), dtype=wp.int32, device=device), 1.0, wp.vec3())
    with pytest.raises(ValueError, match="voxel_size"):
        tw.voxels.from_cells(empty_cells, 0.0, wp.vec3())
    with pytest.raises(ValueError, match="shape"):
        tw.voxels.grid_points((0, 1, 1), device=device)
    with pytest.raises(ValueError, match="pad"):
        tw.voxels.to_field(grid, pad=-1)
    with pytest.raises(TypeError, match="index"):
        tw.voxels.cells(wp.Volume.load_from_numpy(np.ones((2, 2, 2), dtype=np.float32)))


def _assert_surfaces_within(surface_a: tm.Trimesh, surface_b: tm.Trimesh, tolerance: float) -> None:
    """Both one-sided surface distances between two meshes are under ``tolerance``."""
    samples_a = tm.sample.sample_surface(surface_a, 2000, seed=1)[0]
    samples_b = tm.sample.sample_surface(surface_b, 2000, seed=1)[0]
    assert tm.proximity.closest_point(surface_b, samples_a)[1].max() < tolerance
    assert tm.proximity.closest_point(surface_a, samples_b)[1].max() < tolerance


def _bounds_of(grid: wp.Volume) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Tight ``(lower_cell, extent)`` cell box of a grid, for the dense comparisons."""
    rows = tw.voxels.cells(grid).numpy()
    lower = rows.min(axis=0)
    upper = rows.max(axis=0)
    return tuple(int(x) for x in lower), tuple(int(x) for x in upper - lower + 1)
