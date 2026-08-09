"""
What the whole mesh adds up to: the volume it encloses, its moments and its Euler characteristic.

Every function here reduces the entire surface to a single host-side number (or a small tuple of
them), which is why they are gathered rather than left beside the per-element fields they sum over.
Each therefore costs at least one host readback and forces a device synchronisation -- call them
once and keep the answer, and reach for [`triwarp.reduce`][triwarp.reduce] when a reduction should
stay on the device.

Three of the four are **mass properties of the enclosed solid**, and all three are only meaningful
on a closed, consistently wound surface -- [`is_volume`][triwarp.validation.is_volume] is the check
for that, and [`make_volume`][triwarp.repair.make_volume] the repair.
[`volume`][triwarp.totals.volume] is the signed volume,
[`moments`][triwarp.totals.moments] adds the centre of mass and the inertia tensor, and
[`surface_centroid`][triwarp.totals.surface_centroid] is the one member that is a *normalized*
total rather than a sum: the area-weighted mean of the per-face centroids. It is a property of the
**shell**, so it differs from ``moments``' centre of mass on any solid whose mass is not distributed
like its surface -- and it needs no watertightness, since a surface has a centroid whether or not it
bounds anything.

[`euler_characteristic`][triwarp.totals.euler_characteristic] is the odd one out: a topological
invariant rather than a measurement, counted off the index buffer with no geometry involved. It sits
here because it is a whole-mesh integer, and it is what
[`homology_generators`][triwarp.homology.homology_generators] derives the genus from.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
from triwarp._device import prefers_tiled_reduction, slice_count
from triwarp.constants import TILE_1D
from triwarp.kernels import totals as kernel_totals


def volume(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Signed volume enclosed by the mesh.

    Sum of per-face signed tetrahedron volumes measured from the origin
    (``dot(v0, cross(v1, v2)) / 6``); for a closed, consistently wound surface this is
    independent of the reference point, and its sign follows the orientation of the face
    normals (positive for outward-facing normals). For an open or inconsistently wound mesh
    the result is not a meaningful volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    float
        Signed volume in ``float32``. ``0.0`` for an empty mesh.

    See Also
    --------
    [`face_signed_volumes`][triwarp.triangles.face_signed_volumes]
        The per-face primitive this sums.
    [`is_volume`][triwarp.validation.is_volume]
        Whether the sum means anything on this mesh.
    [`make_volume`][triwarp.repair.make_volume]
        Make it mean something.
    [`moments`][triwarp.totals.moments]
        The same volume in float64, plus the centre of mass and inertia.
    [`trimesh.Trimesh.volume`][]
    """
    if int(faces.shape[0]) == 0:
        return 0.0
    return tw.reduce.sum(tw.triangles.face_signed_volumes(vertices, faces))


def surface_centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.vec3:
    """
    Area-weighted centroid of the mesh surface: the mean of the per-face centroids by area.

    A *normalized* total rather than a sum, and a property of the shell alone -- unlike
    [`moments`][triwarp.totals.moments]' centre of mass it needs no watertightness and no consistent
    winding, because a surface has a centroid whether or not it bounds a solid. The two differ on
    any solid whose mass is not distributed like its surface.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.vec3
        Area-weighted mean of per-face centroids. All-``NaN`` when ``faces`` is empty.

    See Also
    --------
    [`moments`][triwarp.totals.moments]
        The centre of mass of the enclosed *solid*, which this is not.
    [`face_centroids`][triwarp.triangles.face_centroids]
        The per-face barycentres this averages.
    [`centroid`][triwarp.points.centroid]
        The unweighted mean of a point cloud, and the only other ``centroid`` in the package.
    [`trimesh.Trimesh.centroid`][]
    """
    f = faces.shape[0] // 3
    if f == 0:
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    device = vertices.device
    out_centroid = wp.zeros(3, dtype=wp.float32, device=device)
    out_total_area = wp.zeros(1, dtype=wp.float32, device=device)
    if prefers_tiled_reduction(device):
        wp.launch_tiled(
            kernel_totals.centroid_tiled,
            dim=[(f + TILE_1D - 1) // TILE_1D],
            inputs=[vertices, faces, wp.int32(f), out_centroid, out_total_area],
            block_dim=TILE_1D,
            device=device,
        )
    else:
        n_slices = slice_count(f, device)
        wp.launch(
            kernel_totals.centroid_sliced,
            dim=n_slices,
            inputs=[vertices, faces, wp.int32(f), wp.int32(n_slices), out_centroid, out_total_area],
            device=device,
        )
    # Two unavoidable readbacks: the return type is a host-side wp.vec3, so the sums have to
    # cross to the host to be divided.
    weighted = out_centroid.numpy()
    total_area = float(out_total_area.numpy()[0])
    return wp.vec3(
        float(weighted[0]) / total_area,
        float(weighted[1]) / total_area,
        float(weighted[2]) / total_area,
    )


def moments(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[float, wp.vec3, wp.mat33d]:
    """
    Mass properties of the solid bounded by the mesh: volume, centre of mass and inertia tensor.

    Integrates over the enclosed solid at unit density by summing the contribution of every
    (origin, face) tetrahedron, so the result is only meaningful for a **closed, consistently
    wound** surface -- the same precondition [`volume`][triwarp.totals.volume] carries, and
    [`is_volume`][triwarp.validation.is_volume] is the check for it.

    The integrals accumulate in ``float64`` even though the positions are ``float32``: the second
    moments scale as ``length ** 5``, so a ``float32`` sum loses their low digits on any sizeable
    mesh.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    volume : float
        Signed volume, identical to [`volume`][triwarp.totals.volume] up to its ``float32``
        accumulation.
    center_of_mass : wp.vec3
        Volume centroid, i.e. the first moment divided by the volume. This is **not**
        [`surface_centroid`][triwarp.totals.surface_centroid], the area-weighted centre of the
        *surface*; the two differ on any solid whose mass is not distributed like its shell.
    inertia : wp.mat33d
        ``(3, 3)`` inertia tensor about the centre of mass, at unit density. ``float64``, like the
        integrals it is assembled from -- a ``wp.mat33`` would discard exactly the low digits this
        function accumulates in double precision to keep.

    Notes
    -----
    ``igl.moments`` returns ``(m0, m1, m2)`` where ``m1`` is the first moment -- the centre of mass
    times the mass -- rather than the centre of mass itself, and ``m2`` is already referred to the
    centre of mass (verified against a translated mesh, not assumed). This function returns the
    decoded forms, so the transform between the two is ``m1 / m0``.

    Three host readbacks are unavoidable here: all three returns are host-side values, so the ten
    accumulated sums have to cross the device boundary to be combined.

    See Also
    --------
    [`volume`][triwarp.totals.volume]
    [`surface_centroid`][triwarp.totals.surface_centroid]
    [`is_volume`][triwarp.validation.is_volume]
    [`trimesh.Trimesh.moment_inertia`][]
    ``igl.moments``
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0.0, wp.vec3(float("nan"), float("nan"), float("nan")), wp.mat33d()

    volumes = wp.empty(n_faces, dtype=wp.float64, device=device)
    first = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    squares = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    products = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_totals.moment_integrands,
        dim=n_faces,
        inputs=[vertices, faces, volumes, first, squares, products],
        device=device,
    )

    # Four device reductions, each returning its total to the host: every return here is a
    # host-side value, so ten sums have to cross -- but only the ten, not the per-face integrands.
    # ``wp.utils.array_sum`` reduces a ``wp.vec3d`` array componentwise, so the three vector groups
    # need no kernel of their own. Reading them with ``.numpy().sum(axis=0)`` instead copied three
    # ``(n_faces,)`` ``vec3d`` buffers -- 72 bytes per face -- to add them on the host: measured
    # back-to-back, 3.08 ms -> 0.18 ms (17x) on ``bunny`` and 33.6 -> 0.19 (174x) on ``dragon`` on
    # CUDA, and 2.54 -> 0.55 (4.7x) on CPU.
    total_volume = float(wp.utils.array_sum(volumes))
    first_moment = np.asarray(list(wp.utils.array_sum(first)), dtype=np.float64)
    integral_squares = np.asarray(list(wp.utils.array_sum(squares)), dtype=np.float64)
    integral_products = np.asarray(list(wp.utils.array_sum(products)), dtype=np.float64)

    center = first_moment / total_volume if total_volume != 0.0 else np.full(3, np.nan)

    # Inertia about the origin from the raw integrals, then shifted to the centre of mass. Nine
    # scalars of host float64 arithmetic, packed into a wp.mat33d so no caller needs numpy to read
    # the answer.
    x2, y2, z2 = integral_squares
    xy, xz, yz = integral_products
    inertia = np.array(
        [[y2 + z2, -xy, -xz], [-xy, x2 + z2, -yz], [-xz, -yz, x2 + y2]], dtype=np.float64
    )
    if total_volume != 0.0:
        shift = total_volume * (float(center @ center) * np.eye(3) - np.outer(center, center))
        inertia = inertia - shift

    return total_volume, wp.vec3(*center.tolist()), wp.mat33d(*inertia.ravel().tolist())


def euler_characteristic(faces: wp.array[wp.int32]) -> int:
    """
    Euler characteristic ``V - E + F`` of the mesh (topological invariant).

    Counts distinct referenced vertices, unique undirected edges, and faces, matching
    [`trimesh.Trimesh.euler_number`][] (which uses referenced vertices, ``edges_unique``, and
    faces). For a closed genus-``g`` surface this equals ``2 - 2 * g``.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    int
        ``(#distinct referenced vertices) - (#unique edges) + (#faces)``. ``0`` for an empty mesh.

    See Also
    --------
    [`edges_unique`][triwarp.edges.edges_unique]
    [`trimesh.Trimesh.euler_number`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0

    n_referenced = int(tw.grouping.unique_1d(faces).shape[0])
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=tw.vertices.n_vertices(faces))
    n_edges = int(unique_edges.shape[0])
    return n_referenced - n_edges + n_faces
