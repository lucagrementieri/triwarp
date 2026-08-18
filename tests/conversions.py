"""Shared mesh format conversions for tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pyvista as pv
import scipy.sparse as sp
import trimesh as tm
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree


def trimesh_to_warp(mesh: tm.Trimesh, device: str) -> wp.Mesh:
    vertices = wp.array(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces = wp.array(
        np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return wp.Mesh(points=vertices, indices=faces)


def numpy_to_warp(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str
) -> tuple[wp.array, wp.array]:
    """
    Upload a NumPy mesh as triwarp's ``(wp.array[wp.vec3], flat wp.array[wp.int32])`` pair.

    The single most duplicated helper in this suite: six private copies across six modules and 54
    call sites, differing only in where the ``float32`` cast sat. It is separate from
    [`trimesh_to_warp`][tests.conversions.trimesh_to_warp], which returns a ``wp.Mesh`` (a BVH
    build), because most tests want the raw buffers a triwarp wrapper takes and never touch a
    ``wp.Mesh``.

    Positions land as **float32**: that is what ``wp.vec3`` holds, and it is the reason a comparison
    against a float64 reference bottoms out around 1e-7 rather than at machine epsilon.

    Parameters
    ----------
    vertices_np
        ``(n, 3)`` positions, any float dtype.
    faces_np
        Vertex indices, ``(n_faces, 3)`` or already flat -- reshaped to triwarp's flat buffer
        either way.
    device
        Warp device for both arrays.

    See Also
    --------
    [`warp_to_trimesh`][tests.conversions.warp_to_trimesh]
        The inverse, for reading a triwarp result back out.
    [`numpy_to_warp_uv`][tests.conversions.numpy_to_warp_uv]
        The ``wp.vec2`` form, for the parametrization tests' 2-D vertex buffers.
    """
    return (
        wp.array(np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device),
        wp.array(
            np.ascontiguousarray(np.asarray(faces_np).reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def numpy_to_warp_uv(
    uv_np: np.ndarray, faces_np: np.ndarray, device: str
) -> tuple[wp.array, wp.array]:
    """
    Upload a 2-D vertex buffer and its faces as ``(wp.array[wp.vec2], flat wp.array[wp.int32])``.

    The [`numpy_to_warp`][tests.conversions.numpy_to_warp] of the parametrization tests, whose
    "vertices" are UV coordinates in the plane. Separate rather than a ``dtype=`` switch because the
    two are never interchangeable at a call site: a function taking a UV atlas will not accept
    positions, and a silently-wrong vector width is exactly the kind of mistake a shared helper
    should make impossible.
    """
    return (
        wp.array(np.ascontiguousarray(uv_np, dtype=np.float32), dtype=wp.vec2, device=device),
        wp.array(
            np.ascontiguousarray(np.asarray(faces_np).reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def trimesh_to_pymeshlab(mesh: tm.Trimesh, scalars: np.ndarray | None = None) -> ml.MeshSet:
    """
    Wrap a ``tm.Trimesh`` in a fresh single-mesh ``pymeshlab.MeshSet``.

    MeshLab wants float64 positions; the ``(n_faces, 3)`` index array goes in as-is. The returned
    MeshSet is **not** reusable across filters: almost every one of them mutates ``current_mesh()``
    in place, so build a new one per comparison rather than threading one through a test.

    ``scalars`` seeds the per-vertex scalar attribute, which the ``*_per_vertex`` scalar filters
    read and write in place: they take no array argument, so a comparison against one has to put
    its input here rather than pass it.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
            **(
                {}
                if scalars is None
                else {"v_scalar_array": np.ascontiguousarray(scalars, dtype=np.float64)}
            ),
        )
    )
    return meshset


def warp_to_pymeshlab(vertices_wp: wp.array, faces_wp: wp.array) -> ml.MeshSet:
    """
    Read triwarp's ``(vertices, flat faces)`` pair back into a ``pymeshlab.MeshSet``.

    The face buffer is triwarp's flat one, so it is reshaped to ``(n_faces, 3)`` here; use this when
    the mesh under test is a triwarp *output* rather than one of the ``tests/conftest.py`` fixtures
    (which already carry a ``tm.Trimesh`` for ``trimesh_to_pymeshlab``).
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64),
            np.ascontiguousarray(faces_wp.numpy().reshape(-1, 3), dtype=np.int32),
        )
    )
    return meshset


def wedge_uv_to_pymeshlab(
    vertices_np: np.ndarray, faces_np: np.ndarray, wedge_uv_np: np.ndarray
) -> ml.MeshSet:
    """
    Build a MeshSet carrying a **per-wedge** (per-corner) texture atlas.

    MeshLab stores UVs on face corners rather than on vertices, which is why it has no texcoord
    *index* buffer at all and why its seam predicate compares coordinates. ``w_tex_coords_matrix``
    takes exactly triwarp's per-corner layout, ``(3 * n_faces, 2)`` in ``3 * f + k`` order, so any
    numpy atlas can drive ``compute_selection_by_texture_seams_per_vertex`` without going through a
    file.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            vertex_matrix=np.ascontiguousarray(vertices_np, dtype=np.float64),
            face_matrix=np.ascontiguousarray(faces_np, dtype=np.int32),
            w_tex_coords_matrix=np.ascontiguousarray(wedge_uv_np, dtype=np.float64),
        )
    )
    return meshset


def points_to_pymeshlab(points_np: np.ndarray, normals_np: np.ndarray | None = None) -> ml.MeshSet:
    """
    Wrap a bare point cloud in a **face-less** single-mesh ``pymeshlab.MeshSet``.

    A handful of MeshLab filters require a mesh with vertices and no faces --
    ``compute_normal_for_point_clouds`` refuses anything else, and the query layer of
    ``compute_scalar_by_distance_from_another_mesh_per_vertex`` wants the sample points on their own
    -- which neither [`trimesh_to_pymeshlab`][tests.conversions.trimesh_to_pymeshlab] nor
    [`warp_to_pymeshlab`][tests.conversions.warp_to_pymeshlab] can build. Same per-filter freshness
    rule as those two: one MeshSet, one filter call.
    """
    meshset = ml.MeshSet()
    vertices = np.ascontiguousarray(points_np, dtype=np.float64)
    if normals_np is None:
        meshset.add_mesh(ml.Mesh(vertices))
    else:
        meshset.add_mesh(
            ml.Mesh(vertices, v_normals_matrix=np.ascontiguousarray(normals_np, dtype=np.float64))
        )
    return meshset


def trimesh_to_open3d(mesh: tm.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Wrap a ``tm.Trimesh`` in a legacy ``open3d.geometry.TriangleMesh``.

    Open3D wants float64 positions and **int32** faces (``Vector3iVector`` silently misreads a
    wider dtype). Every array goes through ``np.array`` rather than ``np.ascontiguousarray``, which
    is load-bearing: the pybind11 ``Vector3dVector`` cast raises ``ValueError: array is not
    writeable`` on a read-only input, and ``ascontiguousarray`` returns an already-contiguous array
    untouched -- so trimesh's cached ``vertex_normals`` would fail where ``vertices`` succeeded.

    Unlike the pymeshlab helpers this result is safe to reuse across calls *in tests*: the
    per-filter freshness rule that forces ``benchmarks/conftest.py`` to rebuild its MeshSet exists
    because a benchmark applies the same mutating call ten times in a row, which no test does. A
    test that calls a genuinely mutating method twice should still build twice.
    """
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32)),
    )


def points_to_open3d(
    points_np: np.ndarray, normals_np: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    """
    Wrap a point cloud, and optionally its normals, in an ``open3d.geometry.PointCloud``.

    ``np.array`` rather than ``np.ascontiguousarray`` for the same writeability reason as
    [`trimesh_to_open3d`][tests.conversions.trimesh_to_open3d].
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points_np, dtype=np.float64))
    if normals_np is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.array(normals_np, dtype=np.float64))
    return cloud


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> tm.Trimesh:
    """Read a legacy open3d mesh back out, unprocessed so the topology survives the round trip."""
    return tm.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles), process=False
    )


def trimesh_to_open3d_t(mesh: tm.Trimesh) -> o3d.t.geometry.TriangleMesh:
    """
    Wrap a ``tm.Trimesh`` in a tensor-API ``open3d.t.geometry.TriangleMesh``.

    The result must be **bound to a name** by the caller before anything is chained off it:
    ``o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()`` lets the temporary be collected
    mid-expression and the result reads freed memory -- garbage floats rather than an exception.
    Routing every tensor-mesh construction through this helper keeps that binding explicit. For
    ``RaycastingScene`` work, bind this result, then ``scene.add_triangles(mesh_t)``.
    """
    return o3d.t.geometry.TriangleMesh.from_legacy(trimesh_to_open3d(mesh))


def faces_igl(mesh: tm.Trimesh) -> np.ndarray:
    """
    Faces as the ``(n_faces, 3)`` int64 array the libigl bindings expect.

    libigl's Eigen templates are instantiated for 64-bit indices, so handing them trimesh's native
    dtype works by luck rather than contract; several functions crash on int32.

    See Also
    --------
    [`mesh_igl`][tests.conversions.mesh_igl]
        Both halves at once, for the majority of igl calls that take ``(V, F)``.
    """
    return mesh.faces.astype(np.int64)


def mesh_igl(mesh: tm.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(vertices float64 (n, 3), faces int64 (n_faces, 3))``: the libigl calling convention.

    Most of the bound surface takes both arrays together, so the pair is the useful unit and
    [`faces_igl`][tests.conversions.faces_igl] is the F-only special case (``igl.adjacency_matrix``,
    ``igl.vertex_components``, ``igl.is_vertex_manifold``, which size their output by
    ``F.max() + 1`` rather than by ``len(V)``).

    !!! danger "Never pair these arrays with a *different* mesh's faces"
        libigl bounds-checks nothing: ``igl.cotmatrix(V, F)`` with one index past the end of ``V``
        is a SIGSEGV with no traceback, not an exception. Taking both halves from one mesh in one
        call is the point of this helper.
    """
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def trimesh_to_pyvista(mesh: tm.Trimesh) -> pv.PolyData:
    """
    Wrap a ``tm.Trimesh`` as a ``pyvista.PolyData``, preserving float64 positions exactly.

    Goes through ``PolyData.from_regular_faces`` rather than the padded ``[3, i, j, k]`` cell array
    the ``PolyData(points, faces)`` constructor wants: building that padding costs 2.34 ms on a
    40 962-vertex mesh against **0.184 ms** for the classmethod, and it is the floor under every
    pyvista benchmark row.

    pyvista round-trips float64 exactly (measured error ``0.0`` on ``[1/3, pi, e]``), so where a
    comparison against triwarp shows a ~1e-7 residual the float32 floor is *triwarp's* ``wp.vec3``
    vertex buffer, not the reference's storage.
    """
    return pv.PolyData.from_regular_faces(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32),
    )


def points_to_pyvista(points_np: np.ndarray) -> pv.PolyData:
    """
    Wrap an ``(n, 3)`` point array as a ``pyvista.PolyData`` cloud.

    The bare constructor gives one vertex cell per point, which is what the cloud filters
    (``select_interior_points``, ``compute_implicit_distance``, ``interpolate``, ``delaunay_2d``)
    expect; passing a face array would make them read cell centres instead.
    """
    return pv.PolyData(np.ascontiguousarray(points_np, dtype=np.float64))


def pyvista_edges_to_indices(edges_pv: pv.PolyData, vertices_np: np.ndarray) -> np.ndarray:
    """
    Read a pyvista edge extraction's line cells back as ``(n, 2)`` indices, min-first per row.

    ``extract_feature_edges`` returns a **new** ``PolyData`` carrying only the points its lines
    touch, renumbered -- so its ``lines`` array indexes that new set and not the mesh it came from.
    Measured: on a unit box the extraction happens to keep all 8 points and still orders them
    differently, so a comparison that skips this step fails on a mesh where the *counts* match,
    which reads as a real disagreement. ``extract_all_edges`` does keep the numbering, but it goes
    through the same path here so no caller has to remember which filter is which.

    The mapping is by position and exact: VTK copies the coordinates through unchanged, so the
    nearest-neighbour distance is ``0.0`` and the assert below is a bijection check rather than a
    tolerance.
    """
    lines_np = np.asarray(edges_pv.lines).reshape(-1, 3)[:, 1:]
    distances_np, indices_np = cKDTree(np.ascontiguousarray(vertices_np, dtype=np.float64)).query(
        np.asarray(edges_pv.points)
    )
    assert float(np.max(distances_np, initial=0.0)) == 0.0
    return np.sort(indices_np[lines_np], axis=1)


def numpy_to_meshlib(vertices_np: np.ndarray, faces_np: np.ndarray) -> mm.Mesh:
    """
    Build a ``meshlib.mrmeshpy.Mesh`` from a NumPy ``(vertices, faces)`` pair.

    **Vertices first**, like every other converter in this module -- which is the point of routing
    through it, because ``mn.meshFromFacesVerts`` itself takes *faces* first. That swap is the one
    thing to get wrong by hand: the two arrays differ in shape only when the counts differ, so a
    reversed call on a mesh with as many faces as vertices builds silent nonsense rather than
    raising. Both dtypes are permissive on this wheel -- int32 and int64 faces, float32 and float64
    positions all accepted -- so the arrays go through as the caller holds them, and ``faces_np``
    may be ``(n_faces, 3)`` or already flat.

    The result is **not** reusable across calls: almost every MeshLib free function mutates its
    ``Mesh`` in place and returns a status, a count or a bitset of new elements, so build a fresh
    one per comparison -- the ``trimesh_to_pymeshlab`` rule rather than the ``trimesh_to_open3d``
    one. A mutating call also invalidates the lazily-built AABB tree cached on the mesh.

    !!! warning "The vertex buffer is sized by the faces, not by ``len(V)``"
        A **trailing** unreferenced vertex is dropped outright (163 positions in, 162 back), while
        an **interior** one is kept in the buffer and excluded from ``numValidVerts`` (163 in, 163
        back, ``numValidVerts`` 162). So never assume ``getNumpyVerts(...).shape[0] == len(V)``, and
        never hand MeshLib a compacted ``V`` with the original ``F``.

    See Also
    --------
    [`meshlib_to_trimesh`][tests.conversions.meshlib_to_trimesh]
        The inverse, which packs before reading the topology back.
    [`trimesh_to_meshlib`][tests.conversions.trimesh_to_meshlib]
        The same conversion from a ``tests/conftest.py`` fixture.
    [`warp_to_meshlib`][tests.conversions.warp_to_meshlib]
        The same conversion from a triwarp output.
    """
    return mn.meshFromFacesVerts(
        np.ascontiguousarray(np.asarray(faces_np).reshape(-1, 3)), np.ascontiguousarray(vertices_np)
    )


def trimesh_to_meshlib(mesh: tm.Trimesh) -> mm.Mesh:
    """
    Wrap a ``tm.Trimesh`` in a ``meshlib.mrmeshpy.Mesh``.

    Thin front end on [`numpy_to_meshlib`][tests.conversions.numpy_to_meshlib]; its argument-order,
    freshness and vertex-buffer-sizing notes all apply here.
    """
    return numpy_to_meshlib(mesh.vertices, mesh.faces)


def warp_to_meshlib(vertices_wp: wp.array, faces_wp: wp.array) -> mm.Mesh:
    """
    Read triwarp's ``(vertices, flat faces)`` pair into a ``meshlib.mrmeshpy.Mesh``.

    Use this when the mesh under test is a triwarp *output* rather than one of the
    ``tests/conftest.py`` fixtures (which already carry a ``tm.Trimesh`` for
    [`trimesh_to_meshlib`][tests.conversions.trimesh_to_meshlib]). Same freshness rule as
    [`numpy_to_meshlib`][tests.conversions.numpy_to_meshlib], which does the work.
    """
    return numpy_to_meshlib(vertices_wp.numpy(), faces_wp.numpy())


def points_to_meshlib(points_np: np.ndarray, normals_np: np.ndarray | None = None) -> mm.PointCloud:
    """
    Wrap a bare point cloud, and optionally its normals, in a ``meshlib.mrmeshpy.PointCloud``.

    Several oracles read the normals and quietly do something else without them --
    ``makeOrientedNormals``, ``triangulatePointCloud`` and ``findOutliers`` all consult
    ``cloud.normals`` -- so pass them whenever the triwarp side had them. ``findOutliers`` does
    worse than quietly: at its default ``mask`` of ``All`` it **segfaults** on a cloud with no
    normals, because that set includes the ``AwayNormal`` criterion.
    """
    if normals_np is None:
        return mn.pointCloudFromPoints(np.ascontiguousarray(points_np))
    return mn.pointCloudFromPoints(
        np.ascontiguousarray(points_np), np.ascontiguousarray(normals_np)
    )


def meshlib_to_trimesh(mesh_ml: mm.Mesh, *, pack: bool = True) -> tm.Trimesh:
    """
    Read a MeshLib mesh back into a ``tm.Trimesh``, packing the deleted elements away first.

    ``pack()`` is **mandatory** after anything that deletes, and skipping it is silent. Measured
    after ``decimateMesh(maxDeletedFaces=200)`` on a 320-face mesh: ``numValidFaces`` reads 120,
    ``topology.faceSize()`` reads 320, and ``getNumpyFaces`` returns **319** rows --
    ``last_valid_face_id + 1`` -- of which **199 are ``[0, 0, 0]``**, degenerate triangles on vertex
    0. ``getNumpyVerts`` still returns all 162 positions while ``numValidVerts`` is 62. No
    exception and no warning. After ``pack()`` the same reads give 120 faces and 62 vertices.

    Pass ``pack=False`` only to observe that state deliberately; note it *mutates* ``mesh_ml``
    either way, since packing renumbers in place.

    ``process=False`` for the same reason
    [`warp_to_trimesh`][tests.conversions.warp_to_trimesh] uses it: trimesh's default processing
    welds coincident vertices, which would undo the identification the comparison is testing.
    """
    if pack:
        mesh_ml.pack()
    return tm.Trimesh(
        vertices=mn.getNumpyVerts(mesh_ml).astype(np.float64),
        faces=mn.getNumpyFaces(mesh_ml.topology).astype(np.int64),
        process=False,
    )


def meshlib_scalars_to_numpy(scalars_ml: object) -> np.ndarray:
    """
    Read a MeshLib scalar container (``VertScalars`` / ``FaceScalars`` / ...) as a float64 array.

    Neither of the two obvious routes works. ``mn.toNumpyArray`` binds only ``VertCoords`` /
    ``FaceNormals`` / ``std_vector_Vector3_float`` and raises a clear ``TypeError`` for anything
    else, which is safe; but ``np.asarray(vert_scalars)`` returns a **0-d ``object`` array** rather
    than raising, so a comparison written that way fails several lines later in whatever NumPy call
    comes next, with nothing pointing at the cause.

    Iteration is the working route. Bitsets have their own reader,
    [`meshlib_bitset_to_numpy`][tests.conversions.meshlib_bitset_to_numpy], because their size is
    *not* always the element domain.
    """
    return np.fromiter(iter(scalars_ml), np.float64, scalars_ml.size())  # type: ignore[attr-defined]


def meshlib_bitset_to_numpy(bitset_ml: object, size: int) -> np.ndarray:
    """
    Read a MeshLib bitset as a ``bool`` array of exactly ``size`` entries.

    ``mn.getNumpyBitSet`` returns the bitset at *its own* length, and whether that is the element
    domain depends on which function produced it. A bitset MeshLib sized against the mesh comes
    back domain-sized -- ``getBoundaryVerts`` on a 162-vertex mesh gives 162 entries even when one
    bit is set -- but a bitset built by *insertion* is only as long as its highest set bit needs:
    ``findSelfCollidingTrianglesBS`` on a 640-face pair of overlapping spheres returns **608**
    entries (last colliding face 607) and an *empty* array on a clean mesh, where the comparison
    wants 640 and 640.

    Neither shape raises, and both break a ``np.array_equal`` against a triwarp mask by shape
    rather than by value, so every bitset in this suite is read through here with the domain size
    stated by the caller. A bitset *longer* than ``size`` is a genuine domain mismatch (the wrong
    element type, or a converter that dropped elements) and raises.
    """
    flags_np = mn.getNumpyBitSet(bitset_ml)
    if flags_np.shape[0] > size:
        raise ValueError(f"bitset holds {flags_np.shape[0]} bits, more than the {size} elements")
    return np.pad(flags_np, (0, size - flags_np.shape[0]))


def warp_to_trimesh(vertices_wp: wp.array, faces_wp: wp.array) -> tm.Trimesh:
    """
    Read a triwarp ``(vertices, flat faces)`` pair back into a ``tm.Trimesh``.

    ``process=False`` is not optional: trimesh's default processing merges coincident vertices, and
    the meshes that most need this conversion are the ones whose *identification* is the thing under
    test -- a Moebius band's seam is a single shared index by construction, and a fixture that gets
    re-welded on the way in is no longer the fixture that was built.
    """
    return tm.Trimesh(
        vertices=vertices_wp.numpy().astype(np.float64),
        faces=faces_wp.numpy().reshape(-1, 3).astype(np.int64),
        process=False,
    )


def bsr_to_dense(matrix: object, n_vertices: int) -> np.ndarray:
    """
    Densify a ``BsrMatrix``, reading only the entries its offsets actually address.

    ``BsrMatrix.values`` is allocated at the *triplet* count and its ``nnz`` is an upper bound until
    synchronized, so the tail of that buffer is uninitialized scratch. Comparing two matrices'
    ``values`` arrays directly reads that scratch and is flaky by construction; the row offsets are
    the only safe way in.
    """
    offsets = matrix.offsets.numpy()  # type: ignore[attr-defined]
    columns = matrix.columns.numpy()  # type: ignore[attr-defined]
    values = matrix.values.numpy()  # type: ignore[attr-defined]
    dense = np.zeros((n_vertices, n_vertices))
    for row in range(n_vertices):
        for slot in range(offsets[row], offsets[row + 1]):
            dense[row, columns[slot]] = values[slot]
    return dense


def bsr_to_csr(matrix: object) -> sp.csr_matrix:
    """
    Convert a ``BsrMatrix`` to a scipy CSR, at its own declared shape.

    The scipy-side counterpart of [`bsr_to_dense`][tests.conversions.bsr_to_dense], for the operator
    comparisons that want a sparse matrix rather than a dense block -- the Crouzeix-Raviart pair is
    ``(n_edges, n_edges)`` and the LSCM Hessian ``(2n, 2n)``, both too large to densify comfortably
    on the larger fixtures. Reads through the row offsets for the same reason ``bsr_to_dense`` does:
    ``values`` is allocated at the triplet count and its tail is uninitialized scratch.
    """
    nrow = int(matrix.nrow)  # type: ignore[attr-defined]
    ncol = int(matrix.ncol)  # type: ignore[attr-defined]
    return sp.csr_matrix(
        (
            matrix.values.numpy(),  # type: ignore[attr-defined]
            matrix.columns.numpy(),  # type: ignore[attr-defined]
            matrix.offsets.numpy(),  # type: ignore[attr-defined]
        ),
        shape=(nrow, ncol),
    )
