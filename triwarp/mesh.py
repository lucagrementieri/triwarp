"""A cached, composition-based triangle mesh container (mirrors `trimesh.Trimesh`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, NoReturn, TypeVar, cast, overload

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh

_R = TypeVar("_R")

# Cached quantities that depend only on `faces` (topology), not on vertex positions. A
# functional update that keeps the same faces and the same vertex count (`with_vertices`)
# can carry these forward instead of recomputing them. Every new cached property below must
# be added to this set (if faces-only) or left out of it (if it also depends on `vertices`).
# `face_adjacency_unshared` is faces-only and belongs here; its neighbour `face_adjacency_angles`
# reads `face_normals` and is therefore correctly absent. The halfedge trio -- `halfedge_twins`,
# `vertex_one_rings`, `vertex_face_adjacency` -- is pure connectivity and belongs here too, while
# every discrete operator (`cotmatrix` through `vector_heat_operators`) is assembled from the
# connectivity *and* the geometry and must not: `tests/test_mesh.py` pins both halves of that rule,
# this set by parametrizing over it and its complement by name.
_TOPOLOGY_KEYS: frozenset[str] = frozenset(
    {
        "edges",
        "edges_sorted",
        "edges_face",
        "edges_unique",
        "edges_unique_inverse",
        "face_adjacency",
        "face_adjacency_edges",
        "face_adjacency_unshared",
        "face_connected_component_labels",
        "vertex_face_adjacency",
        "halfedge_twins",
        "vertex_one_rings",
        "boundary_edges",
        "oriented_boundary_edges",
        "boundary_loops",
        "boundary_vertex_indices",
        "euler_characteristic",
        "is_edge_manifold",
        "is_vertex_manifold",
        "is_winding_consistent",
        "is_orientable",
    }
)


class _CachedProperty(Generic[_R]):
    """
    Read-only descriptor storing its value in the owning `Trimesh`'s shared ``_cache`` dict.

    Unlike `functools.cached_property`, the value lives in a dict shared across all cached
    properties of the instance rather than in a dedicated instance attribute. This lets one
    property computation stash extra by-product values under sibling keys (e.g. computing
    `Trimesh.face_normals` also caches `Trimesh.face_areas`), and lets `Trimesh.invalidate`
    clear every cached value in a single ``dict.clear()``.
    """

    def __init__(self, func: Callable[[Trimesh], _R]) -> None:
        self._func = func
        self._key = func.__name__
        self.__doc__ = func.__doc__

    @overload
    def __get__(self, obj: None, objtype: type | None = None) -> _CachedProperty[_R]: ...
    @overload
    def __get__(self, obj: Trimesh, objtype: type | None = None) -> _R: ...
    def __get__(self, obj: Trimesh | None, objtype: type | None = None) -> _CachedProperty[_R] | _R:
        if obj is None:
            return self
        if self._key not in obj._cache:
            obj._cache[self._key] = self._func(obj)
        return cast(_R, obj._cache[self._key])

    def __set__(self, obj: Trimesh, value: object) -> NoReturn:
        raise AttributeError(f"'{self._key}' is read-only; call invalidate() after in-place edits")


class Trimesh:
    """
    A triangle mesh with lazily cached derived quantities.

    Holds a ``vertices``/``faces`` pair (the same flat-``int32`` convention used throughout
    `triwarp`) and computes derived geometry, topology, and validity predicates on first
    access, caching the result. `Trimesh` is frozen: there are no setters, so a cached value
    can never silently go stale from a Python-level mutation. Composition rather than
    inheritance is used for the ``warp.Mesh`` BVH — building it eagerly on every mesh would
    waste GPU memory on meshes that never issue a ray or proximity query, so it is built lazily
    behind [`warp_mesh`][triwarp.mesh.Trimesh.warp_mesh] on first use instead.

    Every cached property returns the same object on each access — an array, a tuple of arrays or
    a sparse matrix, aliased across repeated accesses and (for `warp_mesh`) with the mesh's own
    `vertices`/`faces` buffers — so callers must not mutate a returned buffer in place. That
    matters most where they are passed *into* the free functions as precomputed arguments: a
    wrapper that rewrites such a buffer copies it first
    ([`filter_normals`][triwarp.smoothing.filter_normals] is the one that does). If a buffer is
    deliberately mutated in place by a kernel, call
    [`invalidate`][triwarp.mesh.Trimesh.invalidate] afterward to drop every cached value
    (including the BVH); otherwise use
    [`with_vertices`][triwarp.mesh.Trimesh.with_vertices] /
    [`with_faces`][triwarp.mesh.Trimesh.with_faces], which return a new `Trimesh` and carry
    forward whichever cached values are still valid.

    Scalar-valued properties (`area`, `centroid`, `bounds`, `enclosing_diagonal`,
    `mean_edge_length`, `euler_characteristic`, and every `is_*` predicate) synchronize the result
    from device to host on first access; the synchronized Python value is then cached like any
    other property.

    Most of these are also what the free functions accept as an optional precomputed argument, so
    the cache is worth more than the repeat accesses on this class: pass `edges_sorted`,
    `face_adjacency`, `halfedge_twins`, `vertex_one_rings`, `vertex_face_adjacency`,
    `face_normals` / `face_areas`, `cotmatrix_entries`, `bounds`, `laplacian_operator` or an
    operator bundle into the wrapper that takes it and the whole assembly is skipped. The discrete
    operators at the bottom of the class (`cotmatrix` through `vector_heat_operators`) are the
    heaviest of these and the reason a solver run over one mesh should go through a `Trimesh`.

    Measured on ``icosphere(5)`` (10 242 vertices) on an RTX 5090, interleaved, with the cache
    built *inside* the timed callable so the first-use cost is paid in every row:

    | workflow | raw | cached | |
    |---|---|---|---|
    | `heat_geodesic` from 5 source sets | 60.0 ms | 47.3 ms | 1.27x |
    | `filter_laplacian`, 3 calls | 4.18 ms | 3.11 ms | 1.35x |
    | `vertex_tangent_frames`, 4 calls | 3.74 ms | 1.14 ms | 3.30x |

    The gain is the assembly's share of the call, so it grows with how often the mesh is reused and
    shrinks where the *solve* dominates -- which is why the heat row is the smallest of the three
    despite caching the most expensive object here.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Triangle indices as ``wp.int32``, either flat (length ``3 * n_faces``) or shape
        ``(n_faces, 3)`` (flattened on construction).
    initial_cache
        Optional pre-populated cache entries, keyed by property name. Used by
        [`from_warp_mesh`][triwarp.mesh.Trimesh.from_warp_mesh] to seed `warp_mesh` without
        rebuilding its BVH, and by
        [`with_vertices`][triwarp.mesh.Trimesh.with_vertices] to carry forward
        topology-only caches.

    Raises
    ------
    TypeError
        If ``vertices`` is not a rank-1 ``wp.vec3`` array, or ``faces`` is not rank-1 or
        rank-2 ``wp.int32`` after flattening.
    ValueError
        If a rank-2 ``faces`` array does not have shape ``(n_faces, 3)``, or the flattened
        face buffer's size is not a multiple of 3.

    Examples
    --------
    ```python
    mesh = tw.Trimesh.from_warp_mesh(warp_mesh)
    mesh.face_normals  # computed once, cached
    mesh.face_areas  # already cached as a by-product of face_normals
    hits = tw.ray.intersects_any(mesh.warp_mesh, origins, directions)
    ```

    See Also
    --------
    [`trimesh.Trimesh`][]
    """

    __slots__ = ("_cache", "_faces", "_vertices")

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        *,
        initial_cache: dict[str, object] | None = None,
    ) -> None:
        """Construct a `Trimesh` from a vertex buffer and a face index buffer."""
        twt.ensure_ndim(vertices, 1, dtype=wp.vec3)
        if int(faces.ndim) == 2:
            if int(faces.shape[1]) != 3:
                raise ValueError(f"2D faces must have shape (n_faces, 3), got {tuple(faces.shape)}")
            faces = faces.reshape((-1,))
        twt.ensure_ndim(faces, 1, dtype=wp.int32)
        if int(faces.size) % 3 != 0:
            raise ValueError(f"faces size must be a multiple of 3, got {int(faces.size)}")

        self._vertices = vertices.contiguous()
        self._faces = faces.contiguous()
        self._cache: dict[str, object] = dict(initial_cache) if initial_cache is not None else {}

    @classmethod
    def from_warp_mesh(cls, mesh: wp.Mesh) -> Trimesh:
        """
        Wrap an existing `warp.Mesh`, seeding `warp_mesh` so its BVH is never rebuilt.

        Parameters
        ----------
        mesh
            Source mesh; ``mesh.points`` and ``mesh.indices`` become the new `Trimesh`'s
            `vertices` and `faces`.

        Returns
        -------
        Trimesh
            New instance whose `warp_mesh` property is `mesh` itself.
        """
        return cls(mesh.points, mesh.indices, initial_cache={"warp_mesh": mesh})

    @property
    def vertices(self) -> wp.array[wp.vec3]:
        """``(n_vertices,)`` mesh vertex positions."""
        return self._vertices

    @property
    def faces(self) -> wp.array[wp.int32]:
        """Length-``3 * n_faces`` flat ``wp.int32`` triangle index buffer."""
        return self._faces

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the `vertices` buffer (its length, not the referenced count)."""
        return int(self._vertices.shape[0])

    @property
    def n_faces(self) -> int:
        """Number of triangles (``faces.size // 3``)."""
        return int(self._faces.shape[0]) // 3

    @property
    def device(self) -> wp.Device:
        """Warp device holding `vertices` and `faces`."""
        return cast(wp.Device, self._vertices.device)

    @_CachedProperty
    def warp_mesh(self) -> wp.Mesh:
        """
        Build the `vertices` / `faces` pair into a `warp.Mesh`, constructing its BVH.

        Raises
        ------
        ValueError
            If the mesh has zero faces — building a ``warp.Mesh`` with an empty BVH silently
            corrupts CUDA state through Warp 1.16.

        See Also
        --------
        [`triwarp.ray`][triwarp.ray]
        [`triwarp.proximity`][triwarp.proximity]
        """
        require_nonempty_mesh(self._faces, "Trimesh.warp_mesh")
        return wp.Mesh(points=self._vertices, indices=self._faces)

    @_CachedProperty
    def face_normals(self) -> wp.array[wp.vec3]:
        """
        Length-``n_faces`` unit face normals (zero for degenerate faces).

        Computed together with [`face_areas`][triwarp.mesh.Trimesh.face_areas]; whichever is
        accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.face_normals`][]
        """
        normals, areas = tw.triangles.face_normals_and_areas(self._vertices, self._faces)
        self._cache["face_areas"] = areas
        return normals

    @_CachedProperty
    def face_areas(self) -> wp.array[wp.float32]:
        """
        Length-``n_faces`` triangle areas.

        Computed together with [`face_normals`][triwarp.mesh.Trimesh.face_normals]; whichever
        is accessed first also caches the other.

        See Also
        --------
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
        [`trimesh.Trimesh.area_faces`][]
        """
        _ = self.face_normals
        return cast("wp.array[wp.float32]", self._cache["face_areas"])

    @_CachedProperty
    def area(self) -> float:
        """
        Total surface area (sum of `face_areas`).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`trimesh.Trimesh.area`][]
        """
        return tw.reduce.sum(cast("twt.Array1dFloat32", self.face_areas))

    @_CachedProperty
    def face_angles(self) -> twt.Array2dFloat32:
        """
        Shape ``(n_faces, 3)`` interior angles in radians, aligned with each face's corners.

        See Also
        --------
        [`triwarp.triangles.face_angles`][]
        [`trimesh.Trimesh.face_angles`][]
        """
        return tw.triangles.face_angles(self._vertices, self._faces)

    @_CachedProperty
    def centroid(self) -> wp.vec3:
        """
        Area-weighted centroid of the mesh surface (all-``NaN`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.measures.surface_centroid`][]
        [`trimesh.Trimesh.centroid`][]
        """
        return tw.measures.surface_centroid(self._vertices, self._faces)

    @_CachedProperty
    def bounds(self) -> tuple[wp.vec3, wp.vec3]:
        """
        Axis-aligned bounding box of `vertices`, as ``(min_bound, max_bound)``.

        Pass it to any wrapper taking a ``bounds=`` argument --
        [`marching_cubes`][triwarp.levelset.marching_cubes],
        [`signed_distance_grid`][triwarp.proximity.signed_distance_grid],
        [`query_nearest`][triwarp.neighbors.query_nearest],
        [`grid_points`][triwarp.voxels.grid_points] -- so the box is reduced once per mesh rather
        than once per call.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. ``(+inf, -inf)`` for an empty
        mesh, the [`aabb`][triwarp.bounds.aabb] convention.

        See Also
        --------
        [`triwarp.bounds.aabb`][]
        [`enclosing_diagonal`][triwarp.mesh.Trimesh.enclosing_diagonal]
        """
        return tw.bounds.aabb(self._vertices)

    @_CachedProperty
    def enclosing_diagonal(self) -> float:
        """
        Diagonal length of `bounds`: this package's default mesh-query search radius.

        Notes
        -----
        Costs no device work of its own, unlike
        [`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal], which reduces the box again --
        it reads the cached `bounds` on the host. ``inf`` for an empty mesh, that function's
        convention.

        A query wrapper's own default is the diagonal of the box around the mesh **and the query
        points**, which this is not; it is the right radius only when the queries lie inside the
        mesh's own box.

        See Also
        --------
        [`triwarp.bounds.enclosing_diagonal`][]
        [`bounds`][triwarp.mesh.Trimesh.bounds]
        ``trimesh.Trimesh.scale``
            The same quantity under trimesh's name for it (no Sphinx inventory entry to link).
        """
        lower, upper = self.bounds
        return float(wp.length(upper - lower))

    @_CachedProperty
    def vertex_normals(self) -> wp.array[wp.vec3]:
        """
        Length-``n_vertices`` angle-weighted unit vertex normals.

        Matches `trimesh`'s default vertex-normal weighting (interior-angle weighted).

        See Also
        --------
        [`vertices.vertex_normals`][triwarp.vertices.vertex_normals] at ``weighting="angle"``
        [`trimesh.Trimesh.vertex_normals`][]
        """
        return tw.vertices.vertex_normals(
            self._vertices,
            self._faces,
            weighting="angle",
            face_normals=self.face_normals,
            face_weights=self.face_angles,
        )

    @_CachedProperty
    def vertex_defects(self) -> wp.array[wp.float32]:
        """
        Length-``n_vertices`` discrete angle defect (``2π`` minus incident corner angles).

        See Also
        --------
        [`triwarp.vertices.vertex_defects`][]
        """
        return tw.vertices.vertex_defects(self.n_vertices, self._faces, self.face_angles)

    @_CachedProperty
    def nondegenerate_faces(self) -> wp.array[wp.bool]:
        """
        Length-``n_faces`` mask; ``True`` where the triangle has non-zero area.

        Named for ``trimesh.Trimesh.nondegenerate_faces`` rather than for the function it calls:
        matching trimesh's property names is this facade's whole job, so the property keeps the
        spelling a reader arrives with while the module-level function is
        [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask].

        See Also
        --------
        [`face_nondegenerate_mask`][triwarp.triangles.face_nondegenerate_mask]
        """
        return tw.triangles.face_nondegenerate_mask(self._vertices, self._faces)

    @_CachedProperty
    def mean_edge_length(self) -> float:
        """
        Mean length over all ``3 * n_faces`` per-face edges (``0.0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.edges.mean_edge_length`][]
        """
        if self.n_faces == 0:
            return 0.0
        lengths = tw.edges.edges_length(self._vertices, self._faces, edges_in=self.edges)
        return tw.reduce.mean(cast("twt.Array1dFloat32", lengths))

    @_CachedProperty
    def edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_faces * 3, 2)`` directed triangle edges, in face-corner order.

        See Also
        --------
        [`triwarp.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges`][]
        """
        return tw.edges.faces_to_edges(self._faces, sorted=False)

    @_CachedProperty
    def edges_sorted(self) -> twt.Array2dInt32:
        """
        Shape ``(n_faces * 3, 2)`` edges from `edges`, each row sorted (smaller index first).

        See Also
        --------
        [`triwarp.edges.faces_to_edges`][]
        [`trimesh.Trimesh.edges_sorted`][]
        """
        return tw.edges.faces_to_edges(self._faces, sorted=True)

    @_CachedProperty
    def edges_face(self) -> wp.array[wp.int32]:
        """
        Length-``n_faces * 3`` face index for each row of `edges` / `edges_sorted`.

        See Also
        --------
        [`triwarp.edges.edges_face`][]
        [`trimesh.Trimesh.edges_face`][]
        """
        return tw.edges.edges_face(self._faces)

    @_CachedProperty
    def edges_unique(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` unique undirected edges, ``m <= n_faces * 3``.

        Also caches [`edges_unique_inverse`][triwarp.mesh.Trimesh.edges_unique_inverse] as a
        by-product.

        See Also
        --------
        [`triwarp.edges.edges_unique`][]
        [`trimesh.Trimesh.edges_unique`][]
        """
        unique, inverse = tw.edges.edges_unique(
            self._faces, edges_sorted=self.edges_sorted, n_vertices=self.n_vertices
        )
        self._cache["edges_unique_inverse"] = inverse
        return unique

    @_CachedProperty
    def edges_unique_inverse(self) -> wp.array[wp.int32]:
        """
        Length-``n_faces * 3`` inverse mapping into `edges_unique` (reconstructs `edges_sorted`).

        See Also
        --------
        [`triwarp.edges.edges_unique_inverse`][]
        [`trimesh.Trimesh.edges_unique_inverse`][]
        """
        _ = self.edges_unique
        return cast("wp.array[wp.int32]", self._cache["edges_unique_inverse"])

    @_CachedProperty
    def edges_unique_length(self) -> wp.array[wp.float32]:
        """
        Length-``m`` Euclidean length of each `edges_unique` row.

        See Also
        --------
        [`triwarp.edges.edges_unique_length`][]
        [`trimesh.Trimesh.edges_unique_length`][]
        """
        return tw.edges.edges_unique_length(
            self._vertices, self._faces, unique_edges=self.edges_unique
        )

    @_CachedProperty
    def face_adjacency(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` face index pairs that share an undirected edge.

        Also caches [`face_adjacency_edges`][triwarp.mesh.Trimesh.face_adjacency_edges] as a
        by-product.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency`][]
        [`trimesh.graph.face_adjacency`][]
        """
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            self._faces,
            edges_sorted=self.edges_sorted,
            return_edges=True,
            # The mesh knows its own vertex count, so the row-hash radix never has to be inferred
            # -- that inference is a device reduction ending in a host readback (1.25-1.74x on the
            # whole call, measured in benchmarks/test_adjacency.py).
            n_vertices=int(self._vertices.shape[0]),
        )
        self._cache["face_adjacency_edges"] = adjacency_edges
        return adjacency

    @_CachedProperty
    def face_adjacency_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` shared vertex pair for each `face_adjacency` row.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency`][]
        """
        _ = self.face_adjacency
        return cast("twt.Array2dInt32", self._cache["face_adjacency_edges"])

    @_CachedProperty
    def face_adjacency_unshared(self) -> twt.Array2dInt32:
        """
        Shape ``(m, 2)`` vertex on each adjacent face not on their shared edge.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_unshared`][]
        [`trimesh.graph.face_adjacency_unshared`][]
        """
        return tw.adjacency.face_adjacency_unshared(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
        )

    @_CachedProperty
    def face_adjacency_angles(self) -> wp.array[wp.float32]:
        """
        Length-``m`` unsigned angle in radians between each `face_adjacency` pair.

        See Also
        --------
        [`triwarp.adjacency.face_adjacency_angles`][]
        [`trimesh.Trimesh.face_adjacency_angles`][]
        """
        return tw.adjacency.face_adjacency_angles(
            self._vertices,
            self._faces,
            face_adjacency=self.face_adjacency,
            face_normals=self.face_normals,
        )

    @_CachedProperty
    def face_connected_component_labels(self) -> wp.array[wp.int32]:
        """
        Length-``n_faces`` connected-component label per face (face-adjacency graph).

        Notes
        -----
        Calls
        [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
        directly rather than going through
        [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels], so
        that the cached `face_adjacency` is reused instead of rebuilt. Same labelling engine, same
        answer.

        See Also
        --------
        [`triwarp.adjacency.face_connected_component_labels`][]
        """
        return tw.graph.connected_component_labels_from_edges(
            self.face_adjacency, node_count=self.n_faces
        )

    @_CachedProperty
    def vertex_face_adjacency(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
        """
        Incidence CSR of the faces touching each vertex, as ``(vertex_faces, offsets)``.

        Each row is a *set*: use `vertex_one_rings` where the rotational order around the vertex
        is what matters, at the price of needing an edge-manifold mesh.

        See Also
        --------
        [`triwarp.adjacency.vertex_face_adjacency`][]
        [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings]
        """
        return tw.adjacency.vertex_face_adjacency(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def halfedge_twins(self) -> wp.array[wp.int32]:
        """
        Length-``3 * n_faces`` opposite halfedge of every halfedge, or ``-1`` on a boundary.

        Halfedge ``h = 3 * f + k`` runs from ``faces[3f + k]`` to ``faces[3f + (k + 1) % 3]``, so
        ``next`` and ``prev`` are index arithmetic and this array is all a walk needs to cross an
        edge.

        Raises
        ------
        ValueError
            Propagated from [`halfedge_twins`][triwarp.halfedge.halfedge_twins] when an undirected
            edge carries three or more halfedges, i.e. the mesh is not edge-manifold.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`triwarp.halfedge.halfedge_twins`][]
        [`vertex_one_rings`][triwarp.mesh.Trimesh.vertex_one_rings]
        """
        return tw.halfedge.halfedge_twins(self._faces, n_vertices=self.n_vertices)

    @_CachedProperty
    def vertex_one_rings(self) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]]:
        """
        Counter-clockwise outgoing halfedges per vertex: ``(ring_halfedges, offsets, is_boundary)``.

        Built from the cached `halfedge_twins`, so accessing either first pays for that array once.

        Raises
        ------
        ValueError
            Propagated from [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings] when a vertex's
            rotation closes before its whole fan is covered (a pinched, vertex-non-manifold
            vertex), or from `halfedge_twins` on an edge-non-manifold mesh.

        Notes
        -----
        Triggers a device-to-host synchronization on first access (that manifoldness check).

        See Also
        --------
        [`triwarp.halfedge.vertex_one_rings`][]
        [`halfedge_twins`][triwarp.mesh.Trimesh.halfedge_twins]
        [`vertex_face_adjacency`][triwarp.mesh.Trimesh.vertex_face_adjacency]
        """
        return tw.halfedge.vertex_one_rings(
            self._faces, twins=self.halfedge_twins, n_vertices=self.n_vertices
        )

    @_CachedProperty
    def boundary_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_boundary, 2)`` undirected boundary edges (each row sorted, min-first).

        See Also
        --------
        [`triwarp.boundary.boundary_edges`][]
        """
        return tw.boundary.boundary_edges(self._vertices, self._faces, self.edges_sorted)

    @_CachedProperty
    def oriented_boundary_edges(self) -> twt.Array2dInt32:
        """
        Shape ``(n_boundary, 2)`` directed boundary edges, preserving face winding.

        See Also
        --------
        [`triwarp.boundary.oriented_boundary_edges`][]
        """
        return tw.boundary.oriented_boundary_edges(
            self._vertices, self._faces, self.edges_sorted, self.edges
        )

    @_CachedProperty
    def boundary_loops(self) -> list[wp.array[wp.int32]]:
        """
        One ordered vertex-index array per boundary loop (empty list if the mesh is closed).

        See Also
        --------
        [`triwarp.boundary.boundary_loops`][]
        """
        return tw.boundary.boundary_loops(
            self._vertices, self._faces, self.edges_sorted, self.edges
        )

    @_CachedProperty
    def boundary_vertex_indices(self) -> wp.array[wp.int32]:
        """
        Sorted unique vertex indices lying on the mesh boundary.

        See Also
        --------
        [`triwarp.boundary.boundary_vertex_indices`][]
        """
        return tw.boundary.boundary_vertex_indices(self._vertices, self._faces, self.edges_sorted)

    @_CachedProperty
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` (``0`` for an empty mesh).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Equivalent to
        [`euler_characteristic`][triwarp.measures.euler_characteristic], recomposed here to
        reuse the cached `edges_unique` count instead of recomputing it.

        See Also
        --------
        [`triwarp.measures.euler_characteristic`][]
        [`trimesh.Trimesh.euler_number`][]
        """
        if self.n_faces == 0:
            return 0
        n_referenced = int(tw.grouping.unique_1d(self._faces).shape[0])
        return n_referenced - int(self.edges_unique.shape[0]) + self.n_faces

    @_CachedProperty
    def is_edge_manifold(self) -> bool:
        """
        Whether every undirected edge is shared by one or two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.validation.is_edge_manifold`][]
        """
        return tw.validation.is_edge_manifold(
            self._faces, edges_sorted=self.edges_sorted, n_vertices=self.n_vertices
        )

    @_CachedProperty
    def is_vertex_manifold(self) -> bool:
        """
        Whether every referenced vertex has a single edge-connected fan of faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.validation.is_vertex_manifold`][]
        """
        return tw.validation.is_vertex_manifold(
            self._faces,
            face_adjacency=self.face_adjacency,
            face_adjacency_edges=self.face_adjacency_edges,
        )

    @_CachedProperty
    def is_winding_consistent(self) -> bool:
        """
        Whether every shared edge is traversed in opposite directions by its two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Equivalent to
        [`is_winding_consistent`][triwarp.validation.is_winding_consistent], recomposed here to
        reuse the cached `edges` / `edges_sorted`.

        See Also
        --------
        [`triwarp.validation.is_winding_consistent`][]
        [`trimesh.Trimesh.is_winding_consistent`][]
        """
        if self.n_faces == 0:
            return True
        mask = tw.validation.edge_winding_consistent_mask(
            self._faces, edges=self.edges, edges_sorted=self.edges_sorted
        )
        if int(mask.shape[0]) == 0:
            return True
        return bool(tw.reduce.all(mask))

    @_CachedProperty
    def is_orientable(self) -> bool:
        """
        Whether the faces admit a consistent orientation (allowing per-face flips).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Unlike the other
        adjacency-based predicates on this class, this does not reuse `face_adjacency`: the
        underlying propagation engine rebuilds it internally and does not accept a
        precomputed adjacency.

        See Also
        --------
        [`triwarp.validation.is_orientable`][]
        """
        return tw.validation.is_orientable(self._faces)

    @_CachedProperty
    def is_watertight(self) -> bool:
        """
        Whether the mesh bounds a closed volume with no self-intersections.

        Follows Open3D's ``IsWatertight`` semantics (edge-manifold-closed, vertex-manifold, and
        not self-intersecting) — **not** the same predicate as `trimesh.Trimesh.is_watertight`,
        which only checks that every edge is shared by exactly two faces.

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Passes the cached `warp_mesh`,
        so the self-intersection broad phase reuses that BVH instead of building a second one --
        the same reuse [`is_self_intersecting`][triwarp.mesh.Trimesh.is_self_intersecting] gets
        for free by taking a `wp.Mesh` directly.

        See Also
        --------
        [`triwarp.validation.is_watertight`][]
        [`is_volume`][triwarp.mesh.Trimesh.is_volume]
        """
        if self.n_faces == 0:
            return True
        return tw.validation.is_watertight(
            self._vertices, self._faces, edges_sorted=self.edges_sorted, mesh=self.warp_mesh
        )

    @_CachedProperty
    def is_self_intersecting(self) -> bool:
        """
        Whether any two non-adjacent triangles of the mesh intersect.

        Raises
        ------
        ValueError
            If the mesh has zero faces (see [`warp_mesh`][triwarp.mesh.Trimesh.warp_mesh]).

        Notes
        -----
        Triggers a device-to-host synchronization on first access. Reuses the cached
        `warp_mesh` BVH for its broad phase.

        See Also
        --------
        [`triwarp.validation.is_self_intersecting`][]
        """
        return tw.validation.is_self_intersecting(self.warp_mesh)

    @_CachedProperty
    def is_volume(self) -> bool:
        """
        Whether the mesh is a valid closed volume with outward-facing normals.

        Watertight in the trimesh sense (every edge shared by exactly two faces),
        winding-consistent, and enclosing a positive signed volume.

        Notes
        -----
        Triggers a device-to-host synchronization on first access.

        See Also
        --------
        [`triwarp.validation.is_volume`][]
        [`trimesh.Trimesh.is_volume`][]
        """
        return tw.validation.is_volume(
            self._vertices, self._faces, edges=self.edges, edges_sorted=self.edges_sorted
        )

    # Discrete operators: the assemblies every solve over this mesh shares. Heavier than everything
    # above (a sparse build each, and the two heat bundles several) and reached by fewer callers,
    # so they sit last -- but they are also where the cache pays most, since each is what a whole
    # family of wrappers accepts as its precomputed argument.

    @_CachedProperty
    def cotmatrix_entries(self) -> twt.Array2dFloat32:
        """
        Shape ``(n_faces, 3)`` per-triangle half-cotangent weights, in igl's edge order.

        The ``float32`` table, and that costs a ``float64`` consumer nothing: the free function
        computes these weights in ``float32`` -- the vertex precision -- whatever dtype is asked
        for, and casts on write, so a ``float64`` request would return this same table widened.
        Every consumer accepts either precision and casts to the *matrix* dtype in one build, so
        pass this to [`cotmatrix`][triwarp.laplacian.cotmatrix],
        [`connection_laplacian`][triwarp.laplacian.connection_laplacian],
        [`heat_operators`][triwarp.heat.heat_operators] or
        [`crouzeix_raviart_cotmatrix`][triwarp.energies.crouzeix_raviart_cotmatrix] at either
        precision -- which is what this class's own ``float32`` `cotmatrix` and ``float64``
        `heat_operators` both do.

        See Also
        --------
        [`triwarp.laplacian.cotmatrix_entries`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.cotmatrix_entries(self._vertices, self._faces)

    @_CachedProperty
    def cotmatrix(self) -> wps.BsrMatrix[wp.float32]:
        """
        Cotangent stiffness matrix: the ``float32`` discrete Laplace-Beltrami operator.

        Assembled from the cached `cotmatrix_entries`. Diagonal entries are negative and each row
        sums to zero, so ``-L`` is positive semi-definite on a closed mesh.

        Notes
        -----
        ``float32``, which is the right precision for a *product* (an energy, a residual, a filter
        weight) and not for a solve -- the heat solvers and
        [`triwarp.parametrization`][triwarp.parametrization] assemble their own ``float64``
        operators, and `heat_operators` caches the one they share.

        See Also
        --------
        [`triwarp.laplacian.cotmatrix`][]
        [`cotmatrix_entries`][triwarp.mesh.Trimesh.cotmatrix_entries]
        [`mass_matrix_entries`][triwarp.mesh.Trimesh.mass_matrix_entries]
        [`triwarp.energies.k_harmonic`][]
            Takes this operator and `mass_matrix_entries` as its two arguments.
        """
        return tw.laplacian.cotmatrix(
            self._vertices, self._faces, cot_entries=self.cotmatrix_entries
        )

    @_CachedProperty
    def mass_matrix_entries(self) -> wp.array[wp.float32]:
        """
        Length-``n_vertices`` barycentric lumped mass: a third of each incident triangle's area.

        Built from the cached `face_areas`, so this is a scatter and nothing else.

        Notes
        -----
        ``float32``. A ``float64`` solve wants
        [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries] at that dtype, which
        accumulates in the requested precision rather than casting this -- the two are not the same
        array widened.

        See Also
        --------
        [`triwarp.laplacian.mass_matrix_entries`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.mass_matrix_entries(
            self._vertices, self._faces, face_areas=self.face_areas
        )

    @_CachedProperty
    def laplacian_operator(self) -> wps.BsrMatrix[wp.float32]:
        """
        Row-normalized 1-ring averaging operator (the uniform / umbrella Laplacian).

        Exactly what every position filter in [`triwarp.smoothing`][triwarp.smoothing] builds for
        itself when its ``laplacian_operator=`` argument is ``None``, so passing this hoists the
        assembly out of a multi-filter or multi-call pass.

        Notes
        -----
        The **directed** ``mesh.edges`` adjacency, the shared default.
        [`filter_neighborhood_average`][triwarp.smoothing.filter_neighborhood_average] is the one
        filter that wants the *symmetric* adjacency instead
        ([`laplacian`][triwarp.laplacian.laplacian] at ``symmetric=True``) and must not be handed
        this one; the two differ only on a mesh with an open boundary.

        See Also
        --------
        [`triwarp.laplacian.laplacian`][]
        [`triwarp.smoothing.filter_laplacian`][]
        [`cotmatrix`][triwarp.mesh.Trimesh.cotmatrix]
        """
        return tw.laplacian.laplacian(self._vertices, self._faces)

    @_CachedProperty
    def vertex_tangent_frames(
        self,
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]:
        """
        Orthonormal tangent frame at every vertex as ``(basis_x, basis_y, normal)``.

        The gauge every 2-D tangent quantity on this mesh is measured in. Built from the cached
        `vertex_normals` and `vertex_one_rings`, and its third element **is** `vertex_normals`.

        Raises
        ------
        ValueError
            Propagated from `vertex_one_rings` on a non-manifold mesh.

        See Also
        --------
        [`triwarp.tangent_space.vertex_tangent_frames`][]
        [`vector_heat_operators`][triwarp.mesh.Trimesh.vector_heat_operators]
        """
        return tw.tangent_space.vertex_tangent_frames(
            self._vertices, self._faces, normals=self.vertex_normals, rings=self.vertex_one_rings
        )

    @_CachedProperty
    def heat_operators(self) -> tw.heat.HeatOperators:
        """
        Source-independent operator bundle for the heat method, at the default diffusion time.

        Pass it to [`heat_geodesic`][triwarp.heat.heat_geodesic] or
        [`geodesic_path`][triwarp.geodesic_walk.geodesic_path] through their ``operators=``
        argument: everything in the bundle depends on the mesh alone, so distance from many
        different source sets costs one assembly.

        Notes
        -----
        At ``t = None`` (the squared mean unique-edge length, ``igl::heat_geodesics``' default) and
        ``use_robust=False``. A mesh with degenerate triangles, or a caller sweeping ``t``, wants
        [`heat_operators`][triwarp.heat.heat_operators] directly -- and can still pass this
        class's `cotmatrix_entries` into it.

        See Also
        --------
        [`triwarp.heat.heat_operators`][]
        [`vector_heat_operators`][triwarp.mesh.Trimesh.vector_heat_operators]
        """
        return tw.heat.heat_operators(
            self._vertices, self._faces, cot_entries=self.cotmatrix_entries
        )

    @_CachedProperty
    def vector_heat_operators(self) -> tw.heat.VectorHeatOperators:
        """
        Operator bundle for the vector heat method, at the default diffusion time.

        ``(vector_system, scalar, frames)``: the ``2 x 2``-block connection system, the scalar
        `heat_operators` and the `vertex_tangent_frames`. Pass it to
        [`transport_tangent_vectors`][triwarp.heat.transport_tangent_vectors],
        [`log_map`][triwarp.heat.log_map],
        [`extend_scalar`][triwarp.heat.extend_scalar] or
        [`heat_signed_distance`][triwarp.heat.heat_signed_distance] through their
        ``operators=`` argument.

        Its last two fields are this class's own `heat_operators` and `vertex_tangent_frames`, so
        the three properties share one assembly however they are reached.

        Notes
        -----
        At ``t = None``, the same default `heat_operators` uses -- which is load-bearing rather
        than incidental: [`log_map`][triwarp.heat.log_map]'s radius is asserted to *be* the
        [`heat_geodesic`][triwarp.heat.heat_geodesic] distance, so the two systems must
        share a diffusion time.

        See Also
        --------
        [`triwarp.heat.vector_heat_operators`][]
        [`heat_operators`][triwarp.mesh.Trimesh.heat_operators]
        [`vertex_tangent_frames`][triwarp.mesh.Trimesh.vertex_tangent_frames]
        """
        return tw.heat.vector_heat_operators(
            self._vertices,
            self._faces,
            scalar_operators=self.heat_operators,
            frames=self.vertex_tangent_frames,
        )

    def with_vertices(self, new_vertices: wp.array[wp.vec3]) -> Trimesh:
        """
        Return a new `Trimesh` with different vertex positions but the same topology.

        Cached quantities that depend only on `faces` (edges, adjacency, boundary,
        manifold/orientation predicates) are carried forward; quantities that depend on
        vertex positions (normals, areas, angles, the BVH, ...) are dropped and recomputed
        lazily on the new instance. `faces` is shared (aliased), not copied.

        Parameters
        ----------
        new_vertices
            ``(n_vertices,)`` replacement vertex positions; must have the same length as the
            current `vertices`.

        Returns
        -------
        Trimesh
            New instance sharing `faces` with the topology-only caches carried forward.

        Raises
        ------
        ValueError
            If ``new_vertices`` has a different vertex count than the current mesh.
        """
        if int(new_vertices.shape[0]) != self.n_vertices:
            raise ValueError(
                f"with_vertices requires the same vertex count ({self.n_vertices}), "
                f"got {int(new_vertices.shape[0])}"
            )
        survived = {key: value for key, value in self._cache.items() if key in _TOPOLOGY_KEYS}
        return Trimesh(new_vertices, self._faces, initial_cache=survived)

    def with_faces(self, new_faces: wp.array[wp.int32]) -> Trimesh:
        """
        Return a new `Trimesh` with different faces; no cached quantities are carried forward.

        Parameters
        ----------
        new_faces
            Replacement triangle indices, flat or ``(n_faces, 3)``.

        Returns
        -------
        Trimesh
            New instance sharing `vertices` with an empty cache.
        """
        return Trimesh(self._vertices, new_faces)

    def invalidate(self) -> None:
        """
        Clear every cached value, including the `warp_mesh` BVH.

        Call this after mutating `vertices` or `faces` in place from a kernel — `Trimesh` has
        no way to detect such an edit on its own, so cached values would otherwise silently
        go stale.
        """
        self._cache.clear()
