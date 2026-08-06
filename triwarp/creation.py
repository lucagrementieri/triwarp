"""
Parametric mesh generators: boxes, Platonic solids, surfaces of revolution and extrusions.

Every function returns a bare ``(vertices, faces)`` pair using triwarp's flat ``(3 * n_faces,)``
face layout, so results feed straight into the rest of the package or into
[`Trimesh`][triwarp.mesh.Trimesh] via ``tw.Trimesh(*tw.creation.box())``.

[`revolve`][triwarp.creation.revolve] is the engine behind most of the module:
[`uv_sphere`][triwarp.creation.uv_sphere], [`capsule`][triwarp.creation.capsule],
[`cylinder`][triwarp.creation.cylinder], [`cone`][triwarp.creation.cone],
[`annulus`][triwarp.creation.annulus] and [`torus`][triwarp.creation.torus] each build a small 2D
profile and sweep it around the Z axis. [`box`][triwarp.creation.box] and the four Platonic solids
([`tetrahedron`][triwarp.creation.tetrahedron], [`octahedron`][triwarp.creation.octahedron],
[`icosahedron`][triwarp.creation.icosahedron], [`dodecahedron`][triwarp.creation.dodecahedron]) are
constant tables, and [`icosphere`][triwarp.creation.icosphere] refines one of them in closed form.
[`grid`][triwarp.creation.grid] and
[`sphere_cap`][triwarp.creation.sphere_cap] are the two *open* primitives — a flat patch and a
curved one, each with exactly one boundary loop.

Differences from `trimesh.creation` that apply module-wide:

- No ``**kwargs`` passthrough, no ``metadata`` and no colors — triwarp has no visual layer, so
  `axis` takes no ``origin_color`` and no function records a ``{"shape": ...}`` tag.
- Where trimesh relies on ``Trimesh(process=True)`` merging coincident vertices to make a result
  watertight, the duplicates are avoided or collapsed structurally instead — exactly, and without
  a position hash. See the Notes of [`revolve`][triwarp.creation.revolve] and
  [`extrude_triangulation`][triwarp.creation.extrude_triangulation]. Results that trimesh builds
  with ``process=False`` keep their duplicates here too.
- Polygon inputs are ``wp.vec2`` rings rather than ``shapely`` polygons; interior rings (holes)
  are not supported. See [`triangulate_polygon`][triwarp.creation.triangulate_polygon].

!!! note "Every function here costs at least ~340 µs, and most of them cost exactly that"

    A triwarp wrapper call carries a fixed host-side cost — allocation plus Warp's launch path,
    ~75 µs of it a NumPy prologue — measured at **~340 µs on an RTX 5090** by
    ``benchmarks/test_creation.py::test_box``, which builds a 12-triangle constant table and so
    measures nothing else. That floor is a property of the wrapper layer, not of this module.

    It dominates this module more than any other, because these functions are *small*: `box`,
    `axis` and the four Platonic solids are constant tables, and every
    [`revolve`][triwarp.creation.revolve]-based primitive is two launches over one buffer each, so
    each is flat in its section count from 32 sections to 4096. Below roughly ``10 ** 3``
    elements of output, a timing of any of them reports launch overhead and nothing else, and a
    CPU library returning the same mesh from a per-vertex loop wins outright; above a few thousand
    the flatness is what wins, by 12-150x. Build primitives once and reuse them rather than calling
    these in a loop — that is the only thing the floor actually asks of a caller.
"""

from __future__ import annotations

import math
import secrets

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TOLERANCE_MERGE
from triwarp.kernels import creation as kernel_creation
from triwarp.kernels import registration as kernel_registration

# Default number of pie wedges per full revolution, matching trimesh.
DEFAULT_SECTIONS = 32

# Absolute tolerance trimesh uses to decide that a revolution angle closes the loop.
_CLOSED_ANGLE_ATOL = 1e-10

_UNIT_X = np.array([1.0, 0.0, 0.0])
_UNIT_Y = np.array([0.0, 1.0, 0.0])
_UNIT_Z = np.array([0.0, 0.0, 1.0])

# Unit cube in [0, 1]^3 with outward-facing winding, from trimesh's baked
# ``resources/creation.json``. Kept verbatim so `box` is vertex-for-vertex identical to
# ``trimesh.creation.box``.
_BOX_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_BOX_FACES = np.array(
    [
        [1, 3, 0],
        [4, 1, 0],
        [0, 3, 2],
        [2, 4, 0],
        [1, 7, 3],
        [5, 1, 4],
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6],
    ],
    dtype=np.int32,
).reshape(-1)

# Regular icosahedron on the unit sphere: the cyclic permutations of (+-1, +-phi, 0) normalized
# by sqrt(1 + phi^2). Also from trimesh's ``resources/creation.json``.
_ICOSAHEDRON_VERTICES = np.array(
    [
        [-0.5257311121191336, 0.85065080835204, 0.0],
        [0.5257311121191336, 0.85065080835204, 0.0],
        [-0.5257311121191336, -0.85065080835204, 0.0],
        [0.5257311121191336, -0.85065080835204, 0.0],
        [0.0, -0.5257311121191336, 0.85065080835204],
        [0.0, 0.5257311121191336, 0.85065080835204],
        [0.0, -0.5257311121191336, -0.85065080835204],
        [0.0, 0.5257311121191336, -0.85065080835204],
        [0.85065080835204, 0.0, -0.5257311121191336],
        [0.85065080835204, 0.0, 0.5257311121191336],
        [-0.85065080835204, 0.0, -0.5257311121191336],
        [-0.85065080835204, 0.0, 0.5257311121191336],
    ],
    dtype=np.float64,
)
_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int32,
).reshape(-1)


def _icosphere_face_table() -> np.ndarray:
    """
    Per-base-face topology of the icosahedron, as the ``(20, 9)`` int table the icosphere needs.

    Columns are ``(a, b, c)`` — the face's corner vertices, which are also their own global
    indices — then the base-edge id of each of the three sides ``a-b``, ``b-c``, ``c-a``, then a
    flag per side saying whether that side runs from the edge's higher-numbered endpoint to its
    lower one. Base edges are numbered by first appearance in ``_ICOSAHEDRON_FACES``.

    The two flags together are what makes the numbering *shared*: the ``n - 1`` interior points of
    a base edge are stored once, in the direction of its lower-numbered endpoint, and each of the
    two faces holding that edge walks them in whichever direction its own winding needs.
    """
    faces_np = _ICOSAHEDRON_FACES.reshape(-1, 3)
    edge_ids: dict[tuple[int, int], int] = {}
    table_np = np.empty((faces_np.shape[0], 9), dtype=np.int32)
    for f, (a, b, c) in enumerate(faces_np.tolist()):
        table_np[f, :3] = (a, b, c)
        for side, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            key = (min(u, v), max(u, v))
            table_np[f, 3 + side] = edge_ids.setdefault(key, len(edge_ids))
            table_np[f, 6 + side] = u > v
    return table_np


# Built once: the icosahedron's topology is a constant, so the only per-call cost is the upload.
_ICOSPHERE_FACE_TABLE = _icosphere_face_table()


# The remaining three Platonic solids, as MeshLab's ``create_tetrahedron`` /
# ``create_octahedron`` / ``create_dodecahedron`` tables normalized onto the unit sphere (MeshLab
# emits them at whatever circumradius the integer coordinates give). Kept verbatim, exactly as
# ``_BOX_VERTICES`` is kept verbatim from trimesh, so both the vertex order and the pentagon-fan
# triangulation of the dodecahedron match the reference they came from.
_TETRAHEDRON_VERTICES = np.array(
    [
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
    ],
    dtype=np.float64,
)
_TETRAHEDRON_FACES = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]], dtype=np.int32).reshape(
    -1
)

_OCTAHEDRON_VERTICES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_OCTAHEDRON_FACES = np.array(
    [[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [3, 1, 5], [3, 5, 4], [3, 4, 2], [3, 2, 1]],
    dtype=np.int32,
).reshape(-1)

# 20 vertices: the 8 cube corners at 1/sqrt(3) plus the 12 golden rectangle points. The 12
# pentagonal faces are each fanned from one cube corner, giving 36 triangles.
_DODECAHEDRON_VERTICES = np.array(
    [
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
        [-0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
        [0.0, 0.35682208977309, 0.9341723589627156],
        [0.0, 0.35682208977309, -0.9341723589627156],
        [0.0, -0.35682208977309, 0.9341723589627156],
        [0.0, -0.35682208977309, -0.9341723589627156],
        [0.35682208977309, 0.9341723589627156, 0.0],
        [0.35682208977309, -0.9341723589627156, 0.0],
        [-0.35682208977309, 0.9341723589627156, 0.0],
        [-0.35682208977309, -0.9341723589627156, 0.0],
        [0.9341723589627156, 0.0, 0.35682208977309],
        [0.9341723589627156, 0.0, -0.35682208977309],
        [-0.9341723589627156, 0.0, 0.35682208977309],
        [-0.9341723589627156, 0.0, -0.35682208977309],
    ],
    dtype=np.float64,
)
_DODECAHEDRON_FACES = np.array(
    [
        [0, 8, 10],
        [0, 10, 2],
        [0, 2, 16],
        [0, 16, 17],
        [0, 17, 1],
        [0, 1, 12],
        [0, 12, 14],
        [0, 14, 4],
        [0, 4, 8],
        [5, 14, 12],
        [5, 12, 1],
        [5, 1, 9],
        [5, 19, 18],
        [5, 18, 4],
        [5, 4, 14],
        [5, 9, 11],
        [5, 11, 7],
        [5, 7, 19],
        [3, 11, 9],
        [3, 9, 1],
        [3, 1, 17],
        [3, 13, 15],
        [3, 15, 7],
        [3, 7, 11],
        [3, 17, 16],
        [3, 16, 2],
        [3, 2, 13],
        [6, 18, 19],
        [6, 19, 7],
        [6, 7, 15],
        [6, 15, 13],
        [6, 13, 2],
        [6, 2, 10],
        [6, 10, 8],
        [6, 8, 4],
        [6, 4, 18],
    ],
    dtype=np.int32,
).reshape(-1)


def box(
    extents: tuple[float, float, float] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    bounds: np.ndarray | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a cuboid as 8 vertices and 12 triangles.

    Parameters
    ----------
    extents
        ``(3,)`` edge lengths. The box is centered on the origin. Defaults to a unit cube.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when the transform has negative determinant, so
        normals keep pointing outward.
    bounds
        ``(2, 3)`` axis-aligned corners. Overrides ``extents`` and ``transform``, and yields a
        box spanning exactly those corners rather than one centered on the origin.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 8 vertices and 36 flat face indices.

    Raises
    ------
    ValueError
        If ``bounds`` is combined with ``extents`` or ``transform``, or if either has the wrong
        shape.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.box`][]
    """
    vertices_np = _BOX_VERTICES.copy()

    if bounds is not None:
        if transform is not None or extents is not None:
            raise ValueError("bounds overrides extents/transform: pass only one")
        bounds_np = np.asanyarray(bounds, dtype=np.float64)
        if bounds_np.shape != (2, 3):
            raise ValueError(f"bounds must be (2, 3) float, got {bounds_np.shape}")
        vertices_np *= np.ptp(bounds_np, axis=0)
        vertices_np += bounds_np[0]
    elif extents is not None:
        extents_np = np.asanyarray(extents, dtype=np.float64)
        if extents_np.shape != (3,):
            raise ValueError(f"extents must be (3,) float, got {extents_np.shape}")
        vertices_np -= 0.5
        vertices_np *= extents_np
    else:
        vertices_np -= 0.5

    vertices = _upload_vertices(vertices_np, device)
    faces = wp.array(_BOX_FACES, dtype=wp.int32, device=device)
    return _apply_transform(vertices, faces, transform)


def grid(
    count: tuple[int, int] = (10, 10),
    extents: tuple[float, float] = (1.0, 1.0),
    center: bool = True,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a flat, regularly triangulated rectangle in the ``z = 0`` plane.

    The cheapest mesh with interior vertices and a boundary loop, which is what makes it the
    default fixture for anything that needs one: parametrization, boundary conditions, texture
    baking. Every quad cell is split by the same diagonal, so both triangles of a cell are
    right-angled — be aware that this makes every diagonal's cotangent weight exactly zero, which
    some solvers are sensitive to.

    Parameters
    ----------
    count
        ``(nx, ny)`` **vertex** counts along X and Y; each must be at least 2. The result has
        ``nx * ny`` vertices and ``2 * (nx - 1) * (ny - 1)`` triangles.
    extents
        ``(width, height)`` total span along X and Y.
    center
        When ``True`` (the default) the patch spans ``[-extents / 2, extents / 2]``; when
        ``False`` its lower corner sits at the origin, matching MeshLab's ``create_grid``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when the transform has negative determinant.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, wound so the normals point along ``+Z``.

    Raises
    ------
    ValueError
        If either entry of ``count`` is less than 2, or either entry of ``extents`` is negative.

    See Also
    --------
    [`box`][triwarp.creation.box]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    """
    nx, ny = int(count[0]), int(count[1])
    if nx < 2 or ny < 2:
        raise ValueError(f"count must be at least 2 along each axis, got {count}")
    width, height = float(extents[0]), float(extents[1])
    if width < 0.0 or height < 0.0:
        raise ValueError(f"extents must be non-negative, got {extents}")

    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    if center:
        x = x - 0.5 * width
        y = y - 0.5 * height
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    vertices_np = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.zeros(nx * ny, dtype=np.float64))
    )

    # Row-major layout (X is the slow axis), so a cell's four corners are ``corner``,
    # ``corner + ny`` (next X) and ``+ 1`` (next Y). The two triangles are wound
    # counter-clockwise seen from +Z.
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    corner = (i * ny + j).ravel()
    faces_np = np.empty((corner.shape[0], 2, 3), dtype=np.int32)
    faces_np[:, 0] = np.column_stack((corner, corner + ny, corner + ny + 1))
    faces_np[:, 1] = np.column_stack((corner, corner + ny + 1, corner + 1))

    vertices = _upload_vertices(vertices_np, device)
    faces = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    return _apply_transform(vertices, faces, transform)


def icosahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular icosahedron on the unit sphere, centered on the origin.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 12 vertices and 20 triangles.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`trimesh.creation.icosahedron`][]
    """
    vertices = _upload_vertices(_ICOSAHEDRON_VERTICES, device)
    faces = wp.array(_ICOSAHEDRON_FACES, dtype=wp.int32, device=device)
    return vertices, faces


def tetrahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular tetrahedron on the unit sphere, centered on the origin.

    The four vertices are the even-parity cube corners ``(+-1, +-1, +-1) / sqrt(3)``. This is the
    coarsest closed triangle mesh there is, which makes it the natural stress case for anything
    that assumes a vertex has a large ring: every vertex has valence 3 and every face borders every
    other.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 4 vertices and 4 triangles.

    See Also
    --------
    [`octahedron`][triwarp.creation.octahedron]
    [`icosahedron`][triwarp.creation.icosahedron]
    """
    return (
        _upload_vertices(_TETRAHEDRON_VERTICES, device),
        wp.array(_TETRAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def octahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular octahedron on the unit sphere, centered on the origin.

    The six vertices are the ``+-`` unit axis directions, so every face is an octant of the
    coordinate frame — the shape to reach for when a test wants a closed mesh whose faces are
    exactly axis-aligned.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 6 vertices and 8 triangles.

    See Also
    --------
    [`tetrahedron`][triwarp.creation.tetrahedron]
    [`icosahedron`][triwarp.creation.icosahedron]
    """
    return (
        _upload_vertices(_OCTAHEDRON_VERTICES, device),
        wp.array(_OCTAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def dodecahedron(device: wp.DeviceLike = None) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a regular dodecahedron on the unit sphere, centered on the origin.

    triwarp is triangle-only, so the twelve pentagonal faces arrive triangulated: each pentagon is
    fanned from one of the eight cube-corner vertices, giving 36 triangles over 20 vertices with no
    added centroid. The fan is MeshLab's, so this is vertex- and face-for-face identical to
    ``create_dodecahedron`` up to the uniform scaling onto the unit sphere.

    Because the triangulation is a fan rather than a symmetric split, the triangles are *not*
    congruent and the mesh is not a good uniform-sampling proxy — use
    [`icosphere`][triwarp.creation.icosphere] for that.

    Parameters
    ----------
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with 20 vertices and 36 triangles.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`octahedron`][triwarp.creation.octahedron]
    """
    return (
        _upload_vertices(_DODECAHEDRON_VERTICES, device),
        wp.array(_DODECAHEDRON_FACES, dtype=wp.int32, device=device),
    )


def icosphere(
    subdivisions: int = 3, radius: float = 1.0, device: wp.DeviceLike = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a geodesic sphere by recursively subdividing an icosahedron.

    Each refinement level splits every triangle into four and projects the new vertices back onto
    the sphere, so the result has ``20 * 4 ** subdivisions`` faces and ``10 * 4 ** subdivisions +
    2`` vertices — the same mesh [`trimesh.creation.icosphere`][] builds. Triangles stay far more
    uniform than [`uv_sphere`][triwarp.creation.uv_sphere]'s, at roughly an order of magnitude
    more work.

    Parameters
    ----------
    subdivisions
        Number of refinement levels. Face count grows as ``4 ** subdivisions``, so values above
        ~7 get expensive. Values ``<= 0`` return a plain icosahedron scaled to ``radius``.
    radius
        Sphere radius.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Notes
    -----
    The connectivity is generated in closed form rather than by iterating
    [`subdivide`][triwarp.remesh.subdivide]: every vertex of the refined mesh is addressed directly
    by its barycentric coordinates within one of the 20 base faces (see
    ``kernels.creation.icosphere_vertex_index``), so the whole face buffer is written by **one**
    kernel and the vertices by one launch per refinement level. That is ``subdivisions + 2``
    launches and no host synchronization, against ~17 launches per level — an edge dedup, a radix
    sort and a count readback each time — for the iterated form.

    Measured back to back on an RTX 5090 (`benchmarks/test_creation.py`'s ``icosphere`` group,
    medians from ``--benchmark-json``): **19-31x**, and flat where the iterated form was not.

    | ``subdivisions`` | iterated | closed form | |
    |---|---|---|---|
    | 3 (1 280 faces) | 4.76 ms | **245 µs** | 19.4x |
    | 5 (20 480 faces) | 7.86 ms | **253 µs** | 31.1x |
    | 7 (327 680 faces) | 6.87 ms | **345 µs** | 22.3x |

    The iterated column being *non-monotonic* is the tell that it was measuring host cost rather
    than the output: seven `subdivide` passes cost more than five of them, and neither cost is about
    the faces. What is left is the ~340 µs wrapper floor, so this now beats
    [`trimesh.creation.icosphere`][] at every level (472 µs at 3, 67.1 ms at 7) instead of losing 6x
    at level 3.

    The *geometry* is still the recursive one, level by level, because that is what the reference
    produces: a vertex is the projected midpoint of two vertices of the previous level, which is
    not the same point as the projection of the corresponding barycentric point of the base face
    (the two differ by a few percent of the edge length). Vertex *order* differs from both trimesh
    and the former iterated implementation — vertices come out as the 12 base corners, then the
    interior points of each base edge, then the interior points of each base face.

    See Also
    --------
    [`icosahedron`][triwarp.creation.icosahedron]
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.creation.icosphere`][]
    """
    levels = max(0, int(subdivisions))
    n = 1 << levels
    radius_f = wp.float32(float(radius))

    table = wp.array(_ICOSPHERE_FACE_TABLE, dtype=wp.int32, device=device)
    corners = _upload_vertices(_ICOSAHEDRON_VERTICES, device)
    vertices = wp.empty(10 * 4**levels + 2, dtype=wp.vec3, device=corners.device)
    # The 12 base corners occupy the first block of the numbering, so scaling them to `radius` is
    # the whole of level 0.
    wp.map(kernel_creation.project_to_radius, corners, radius_f, out=vertices[:12])
    for level in range(1, levels + 1):
        wp.launch(
            kernel_creation.icosphere_generation,
            dim=(20, (1 << level) + 1, (1 << level) + 1),
            inputs=[table, wp.int32(n), wp.int32(1 << level), radius_f, vertices],
            device=vertices.device,
        )

    faces = wp.empty(20 * n * n * 3, dtype=wp.int32, device=vertices.device)
    wp.launch(
        kernel_creation.icosphere_faces,
        dim=(20, n, n),
        inputs=[table, wp.int32(n), faces],
        device=vertices.device,
    )
    return vertices, faces


def uv_sphere(
    radius: float = 1.0,
    count: tuple[int, int] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a latitude/longitude sphere centered on the origin.

    Much cheaper than [`icosphere`][triwarp.creation.icosphere], at the cost of triangles that
    shrink toward the poles.

    Parameters
    ----------
    radius
        Sphere radius.
    count
        ``(2,)`` number of latitude and longitude lines. Defaults to ``(32, 64)``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Notes
    -----
    The longitude count is doubled when ``count`` is given explicitly but not when it is left at
    the default, so ``count=(32, 64)`` yields 128 longitude sections while ``count=None`` yields
    64. This asymmetry is inherited from [`trimesh.creation.uv_sphere`][] and kept for parity.

    The two pole points are snapped to exactly ``(0, -radius)`` and ``(0, radius)`` in the 2D
    profile. ``sin(pi)`` is ``1.2e-16`` rather than zero, and without the snap the pole vertices
    of adjacent slices differ by ``1.2e-16 * radius`` — enough that the weld and the
    degenerate-triangle filter in [`revolve`][triwarp.creation.revolve] both start to fail at
    large radii. The correction is far below ``float32`` resolution.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`capsule`][triwarp.creation.capsule]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.uv_sphere`][]
    """
    if count is None:
        latitude, longitude = 32, 64
    else:
        counts = np.asanyarray(count, dtype=np.int64)
        if counts.shape != (2,):
            raise ValueError(f"count must be (2,) int, got {counts.shape}")
        counts += counts % 2
        latitude, longitude = int(counts[0]), int(counts[1]) * 2

    radius_f = abs(float(radius))
    theta = np.linspace(0.0, np.pi, num=latitude)
    profile = np.column_stack((np.sin(theta), -np.cos(theta))) * radius_f
    # Snap the poles: sin(0) is exact but sin(pi) is not (see Notes).
    profile[0] = (0.0, -radius_f)
    profile[-1] = (0.0, radius_f)

    return revolve(_upload_profile(profile, device), sections=longitude, transform=transform)


def sphere_cap(
    angle: float = math.pi / 6.0,
    subdivisions: int = 3,
    radius: float = 1.0,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a spherical cap around ``+Z``: an open disc of a sphere, with one boundary loop.

    A triangular lattice of ``2 ** subdivisions`` concentric rings is projected onto the sphere, so
    ring ``r`` carries ``6 * r`` vertices evenly spaced in azimuth at polar angle
    ``angle * r / n_rings``. Unlike a sliced [`icosphere`][triwarp.creation.icosphere] the rim is a
    clean circle of exactly ``6 * n_rings`` vertices, which is what makes this the fixture of choice
    for boundary-condition work on a curved surface.

    Parameters
    ----------
    angle
        Polar half-angle of the cap in **radians**, measured from ``+Z``. Must be in ``(0, pi)``
        exclusive: at ``pi`` the whole rim collapses onto the south pole and every rim triangle
        would be degenerate. MeshLab's ``create_sphere_cap`` takes the full aperture in degrees
        instead, so its ``angle=60`` is ``math.radians(30)`` here.
    subdivisions
        Number of refinement passes; the lattice has ``2 ** subdivisions`` rings, giving
        ``1 + 3 * n * (n + 1)`` vertices and ``6 * n ** 2`` triangles for ``n = 2 **
        subdivisions``. Must be ``>= 0``.
    radius
        Sphere radius.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, wound so the normals point away from the sphere
        center. Vertex 0 is the apex at ``(0, 0, radius)`` and the last ``6 * n`` vertices are the
        rim, in azimuthal order.

    Raises
    ------
    ValueError
        If ``angle`` is outside ``(0, pi)`` or ``subdivisions`` is negative.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`uv_sphere`][triwarp.creation.uv_sphere]
    """
    if not 0.0 < angle < math.pi:
        raise ValueError(f"angle must be in (0, pi) radians, got {angle}")
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be non-negative, got {subdivisions}")

    n_rings = 2 ** int(subdivisions)
    # ``ring_start[r]`` is where ring r's ``6 * r`` vertices begin; ring 0 is the single apex.
    ring_start = [0] + [1 + 3 * r * (r - 1) for r in range(1, n_rings + 1)]

    vertices_np = np.empty((1 + 3 * n_rings * (n_rings + 1), 3), dtype=np.float64)
    vertices_np[0] = (0.0, 0.0, radius)
    for r in range(1, n_rings + 1):
        theta = angle * r / n_rings
        phi = 2.0 * math.pi * np.arange(6 * r) / (6 * r)
        vertices_np[ring_start[r] : ring_start[r] + 6 * r] = np.column_stack(
            (
                radius * math.sin(theta) * np.cos(phi),
                radius * math.sin(theta) * np.sin(phi),
                np.full(6 * r, radius * math.cos(theta)),
            )
        )

    # Stitch ring r-1 to ring r: six sectors, and in each the outer ring carries one more vertex
    # than the inner one. That extra vertex is what turns the strip into ``2 * r - 1`` triangles --
    # ``6 * r`` outward-pointing (one per outer edge) and ``6 * (r - 1)`` inward-pointing (one per
    # inner edge) -- rather than an even fan, and it is why the total lands on exactly ``6 * n **
    # 2``.
    faces_np = np.empty((6 * n_rings * n_rings, 3), dtype=np.int32)
    written = 0
    for r in range(1, n_rings + 1):
        outer_base, inner_base = ring_start[r], ring_start[r - 1]
        inner_count = 6 * (r - 1) if r > 1 else 1
        outer_index = np.arange(6 * r)
        sector, step = np.divmod(outer_index, r)
        faces_np[written : written + 6 * r] = np.column_stack(
            (
                outer_base + outer_index,
                outer_base + (outer_index + 1) % (6 * r),
                inner_base + (sector * (r - 1) + step) % inner_count,
            )
        )
        written += 6 * r
        if r > 1:
            inner_index = np.arange(inner_count)
            sector, step = np.divmod(inner_index, r - 1)
            faces_np[written : written + inner_count] = np.column_stack(
                (
                    inner_base + inner_index,
                    outer_base + (sector * r + step + 1) % (6 * r),
                    inner_base + (inner_index + 1) % inner_count,
                )
            )
            written += inner_count

    return (
        _upload_vertices(vertices_np, device),
        wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device),
    )


def capsule(
    height: float = 1.0,
    radius: float = 1.0,
    count: tuple[int, int] | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a capsule: a cylinder along Z closed by a hemisphere at each end.

    The result is centered on the origin and spans ``z`` in
    ``[-height / 2 - radius, height / 2 + radius]``; ``height`` is the center-to-center distance
    between the two hemispheres.

    Parameters
    ----------
    height
        Center-to-center distance of the two hemispheres.
    radius
        Radius of the cylinder and of both hemispheres.
    count
        ``(2,)`` number of sections along latitude and longitude. Defaults to ``(32, 64)``. Both
        entries are rounded up to even, which is what keeps the cylindrical wall vertical.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Notes
    -----
    Unlike [`uv_sphere`][triwarp.creation.uv_sphere], the longitude count is *not* doubled here.
    [`trimesh.creation.capsule`][]'s docstring describes the geometry as having one hemisphere at
    the origin and the other at ``height``; the implementation (and this port) centers it
    instead.

    See Also
    --------
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.capsule`][]
    """
    counts = np.array([32, 64], dtype=np.int64) if count is None else np.asanyarray(count, np.int64)
    if counts.shape != (2,):
        raise ValueError(f"count must be (2,) int, got {counts.shape}")
    counts = counts + counts % 2
    latitude, longitude = int(counts[0]), int(counts[1])

    height_f = abs(float(height))
    radius_f = abs(float(radius))
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, latitude)
    profile = np.column_stack((np.cos(theta), np.sin(theta))) * radius_f
    half = len(profile) // 2
    profile[:half, 1] -= height_f / 2.0
    profile[half:, 1] += height_f / 2.0
    # Snap the poles the way uv_sphere does: cos(+-pi/2) is 6.1e-17, not zero.
    profile[0] = (0.0, -height_f / 2.0 - radius_f)
    profile[-1] = (0.0, height_f / 2.0 + radius_f)

    return revolve(_upload_profile(profile, device), sections=longitude, transform=transform)


def cylinder(
    radius: float,
    height: float | None = None,
    sections: int | None = None,
    segment: np.ndarray | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a closed cylinder along Z, centered on the origin.

    Parameters
    ----------
    radius
        Cylinder radius.
    height
        Cylinder height. Required unless ``segment`` is passed.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    segment
        ``(2, 3)`` axis endpoints. Overrides both ``height`` and ``transform``, placing the
        cylinder along the segment.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`annulus`][triwarp.creation.annulus]
    [`cone`][triwarp.creation.cone]
    [`capsule`][triwarp.creation.capsule]
    [`trimesh.creation.cylinder`][]
    """
    if segment is not None:
        transform, height = _segment_to_cylinder(segment)
    if height is None:
        raise ValueError("either height or segment must be passed")

    half = abs(float(height)) / 2.0
    radius_f = float(radius)
    profile = np.array(
        [[0.0, -half], [radius_f, -half], [radius_f, half], [0.0, half]], dtype=np.float64
    )
    return revolve(_upload_profile(profile, device), sections=sections, transform=transform)


def cone(
    radius: float,
    height: float,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a closed cone along Z with its base on the ``z = 0`` plane.

    Parameters
    ----------
    radius
        Radius of the cone at its widest (the base).
    height
        Height of the cone; the apex sits at ``(0, 0, height)``.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    See Also
    --------
    [`cylinder`][triwarp.creation.cylinder]
    [`trimesh.creation.cone`][]
    """
    profile = np.array([[0.0, 0.0], [float(radius), 0.0], [0.0, float(height)]], dtype=np.float64)
    return revolve(_upload_profile(profile, device), sections=sections, transform=transform)


def annulus(
    r_min: float,
    r_max: float,
    height: float | None = None,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    segment: np.ndarray | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create an annular cylinder (a tube with flat ends) along Z, centered on the origin.

    Parameters
    ----------
    r_min
        Inner radius. When smaller than ``1e-8`` this delegates to
        [`cylinder`][triwarp.creation.cylinder], which has a different topology (no inner wall).
    r_max
        Outer radius.
    height
        Height of the annular cylinder. Required unless ``segment`` is passed.
    sections
        Number of pie wedges around the revolution. Defaults to 32.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    segment
        ``(2, 3)`` axis endpoints. Overrides both ``height`` and ``transform``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``.

    Raises
    ------
    ValueError
        If neither ``height`` nor ``segment`` is given, or ``segment`` is not ``(2, 3)``.

    See Also
    --------
    [`cylinder`][triwarp.creation.cylinder]
    [`torus`][triwarp.creation.torus]
    [`trimesh.creation.annulus`][]
    """
    if segment is not None:
        transform, height = _segment_to_cylinder(segment)
    if height is None:
        raise ValueError("either height or segment must be passed")

    r_min_f = abs(float(r_min))
    if r_min_f < TOLERANCE_MERGE:
        return cylinder(
            radius=r_max, height=height, sections=sections, transform=transform, device=device
        )

    r_max_f = abs(float(r_max))
    half = abs(float(height)) / 2.0
    # Counter-clockwise rectangle with the first point repeated: the duplicate closes the profile
    # (and therefore the inner-wall/cap seam) once revolve's weld runs.
    profile = np.array(
        [[r_min_f, -half], [r_max_f, -half], [r_max_f, half], [r_min_f, half], [r_min_f, -half]],
        dtype=np.float64,
    )
    return revolve(_upload_profile(profile, device), sections=sections, transform=transform)


def torus(
    major_radius: float,
    minor_radius: float,
    major_sections: int = 32,
    minor_sections: int = 32,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a torus around Z, centered on the origin.

    Parameters
    ----------
    major_radius
        Distance from the center of the torus to the center of the tube.
    minor_radius
        Radius of the tube.
    major_sections
        Number of sections around the major radius.
    minor_sections
        Number of sections around the minor radius.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``2 * major_sections * minor_sections`` triangles.

    Notes
    -----
    The closing point of the tube profile is snapped to equal its first point exactly. ``sin(2 *
    pi)`` evaluates to ``-2.4e-16``, so without the snap the minor seam would only close through
    [`revolve`][triwarp.creation.revolve]'s absolute weld tolerance and would come apart at large
    radii.

    See Also
    --------
    [`annulus`][triwarp.creation.annulus]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.torus`][]
    """
    minor_f = float(minor_radius)
    phi = np.linspace(0.0, 2.0 * np.pi, int(minor_sections) + 1, endpoint=True)
    profile = np.column_stack((minor_f * np.cos(phi), minor_f * np.sin(phi)))
    profile += (float(major_radius), 0.0)
    profile[-1] = profile[0]

    return revolve(_upload_profile(profile, device), sections=major_sections, transform=transform)


def revolve(
    linestring: wp.array[wp.vec2],
    angle: float | None = None,
    cap: bool = False,
    sections: int | None = None,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Revolve a 2D profile around the 2D Y axis, which becomes the 3D Z axis.

    This is the shared engine behind every radially symmetric primitive in this module. The 2D X
    component of ``linestring`` is used as the revolution radius and the 2D Y component as the
    height along Z.

    Parameters
    ----------
    linestring
        ``(n,)`` ordered 2D profile points, ``n >= 2``. A closed profile should be wound
        counter-clockwise for outward-facing normals.
    angle
        Angle in radians to revolve through. ``None`` (the default) is a full revolution.
    cap
        For a partial revolution, triangulate the two end faces so the result is a closed volume.
        Ignored for a full revolution, which needs no caps. Requires a simple (non
        self-intersecting) profile — see
        [`triangulate_polygon`][triwarp.creation.triangulate_polygon].
    sections
        Number of pie wedges around the revolution. Defaults to 32 per full revolution, scaled by
        ``angle``.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when its determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``linestring.device``.

    Raises
    ------
    ValueError
        If ``linestring`` is not a rank-1 ``wp.vec2`` array with at least 2 points, or
        ``sections`` resolves to less than 1.

    Notes
    -----
    Two clean-up steps are part of the algorithm, not optional polish:

    - **Degenerate triangles are dropped.** Any template triangle whose area is at most ``1e-8``
      is discarded, which is how the polar triangles of a sphere, the apex and base fans of a
      cone, and the axis triangles of a cylinder disappear without being special-cased. The
      threshold is absolute, exactly as trimesh's ``tol.merge`` is, so a sphere's polar triangles
      stop registering as degenerate somewhere above ``radius = 1e4``. Build near unit scale and
      pass a scaling ``transform`` if you need the result large.
    - **Coincident vertices are collapsed**, which is what makes the result watertight: a cone
      apex, a sphere pole, and the closing point of a closed profile each appear once *per slice*
      beforehand. `trimesh` discovers these by position hashing (the ``merge_vertices`` pass of
      ``Trimesh(process=True)``); this port derives them from the profile instead — a profile
      point within ``1e-8`` of the axis becomes a single shared vertex, and a profile whose last
      point repeats its first shares that column. That is exact and scale independent, where a
      position hash both misses ``+0.0``/``-0.0`` pole pairs and starts merging genuinely distinct
      vertices at high section counts.

    A consequence of deriving the collapse rather than measuring it: a profile that repeats an
    *interior* point, or whose ends are close but not equal, keeps its duplicate vertices (the
    zero-area faces between them are still dropped, so they are simply unreferenced).

    See Also
    --------
    [`uv_sphere`][triwarp.creation.uv_sphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`torus`][triwarp.creation.torus]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`trimesh.creation.revolve`][]
    """
    twt.ensure_ndim(linestring, 1, dtype=wp.vec2)
    device = linestring.device
    per = int(linestring.shape[0])
    if per < 2:
        raise ValueError(f"linestring must have at least 2 points, got {per}")

    full = 2.0 * math.pi
    closed = angle is None or abs(float(angle) - full) <= _CLOSED_ANGLE_ATOL
    span = full if angle is None else float(angle)
    if sections is None:
        sections = int(span / full * DEFAULT_SECTIONS)
    # trimesh converts the wedge count to a point count here.
    n_points = int(sections) + 1
    n_slices = n_points - 1
    if n_slices < 1:
        raise ValueError(f"sections must be at least 1, got {sections}")

    # Everything that depends only on the profile is decided on the host, from at most a few dozen
    # points: which template triangles survive, and where each (slice, profile point) pair lands in
    # the merged output. The two kernels below then write their final layout directly, with no
    # compaction pass and no device-to-host synchronization.
    profile_np = linestring.numpy().astype(np.float64)
    keep_np = _revolve_kept_template(profile_np, span / float(n_slices))
    column_np, offsets_np, on_axis_np, n_vertices = _revolve_vertex_layout(
        profile_np, n_slices if closed else n_points
    )
    n_keep = int(keep_np.shape[0])
    layout = (
        wp.array(column_np, dtype=wp.int32, device=device),
        wp.array(offsets_np, dtype=wp.int32, device=device),
        wp.array(on_axis_np, dtype=wp.bool, device=device),
    )
    n_kept_slices = n_slices if closed else n_points

    vertices = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_creation.revolve_vertices,
        dim=(n_kept_slices, per),
        inputs=[
            linestring,
            wp.float32(span),
            wp.int32(n_points),
            wp.int32(n_kept_slices),
            *layout,
            vertices,
        ],
        device=device,
    )

    cap_faces = None
    n_cap = 0
    if cap and not closed:
        profile_3d = wp.empty(per, dtype=wp.vec3, device=device)
        wp.map(kernel_creation.lift_vec2, linestring, wp.float32(0.0), out=profile_3d)
        # Ear clipping introduces no new vertices, so its indices address profile points directly --
        # the guarantee trimesh gets from ``triangulate_polygon(force_vertices=True)``.
        cap_faces = tw.polyline.triangulate_polyline(profile_3d).reshape((-1,))
        n_cap = int(cap_faces.shape[0]) // 3

    faces = wp.empty((n_slices * n_keep + 2 * n_cap) * 3, dtype=wp.int32, device=device)
    if n_keep > 0:
        wp.launch(
            kernel_creation.revolve_faces,
            dim=(n_slices, n_keep),
            inputs=[
                wp.array(keep_np, dtype=wp.int32, device=device),
                wp.int32(per),
                wp.int32(n_keep),
                wp.int32(n_kept_slices),
                *layout,
                faces[: n_slices * n_keep * 3],
            ],
            device=device,
        )
    if cap_faces is not None and n_cap > 0:
        base = n_slices * n_keep * 3
        for slice_index, reverse, offset in (
            (0, False, base),
            (n_kept_slices - 1, True, base + n_cap * 3),
        ):
            wp.launch(
                kernel_creation.revolve_cap_faces,
                dim=n_cap,
                inputs=[
                    cap_faces,
                    wp.int32(slice_index),
                    reverse,
                    wp.int32(n_kept_slices),
                    *layout,
                    faces[offset : offset + n_cap * 3],
                ],
                device=device,
            )

    return _apply_transform(vertices, faces, transform)


def _revolve_kept_template(profile_np: np.ndarray, step: float) -> np.ndarray:
    """
    Template triangles of one revolution slice that are not degenerate.

    trimesh drops any triangle of the slice-0 template whose area is at most ``1e-8``, which is how
    the polar triangles of a sphere, the apex and base fans of a cone, and the axis triangles of a
    cylinder disappear without being special-cased. Every slice is a rotation of slice 0, so one
    mask covers them all, and the whole test is ``2 * (per - 1)`` triangles — cheap enough to keep
    on the host in ``float64``, which also matches the reference's precision.
    """
    per = profile_np.shape[0]
    radius_np, height_np = profile_np[:, 0], profile_np[:, 1]
    grid_np = np.vstack(
        (
            np.column_stack((radius_np, np.zeros(per), height_np)),
            np.column_stack((np.cos(step) * radius_np, np.sin(step) * radius_np, height_np)),
        )
    )
    segment_np = np.arange(per - 1)
    triangles_np = np.empty((2 * (per - 1), 3), dtype=np.int64)
    triangles_np[0::2] = np.column_stack((segment_np, segment_np + per, segment_np + 1))
    triangles_np[1::2] = np.column_stack((segment_np + 1, segment_np + per, segment_np + per + 1))
    corner_np = grid_np[triangles_np]
    areas_np = 0.5 * np.linalg.norm(
        np.cross(corner_np[:, 1] - corner_np[:, 0], corner_np[:, 2] - corner_np[:, 0]), axis=1
    )
    return np.flatnonzero(areas_np > TOLERANCE_MERGE).astype(np.int32)


def _revolve_vertex_layout(
    profile_np: np.ndarray, n_kept_slices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Per-profile-point tables placing each revolved vertex at its final, already-merged index.

    Returns ``(column, offsets, on_axis, n_vertices)``. ``column[i]`` is the profile point ``i``
    shares its position with (itself, except that a profile repeating its first point as its last
    folds that column onto ``0``), ``on_axis[j]`` marks a point within ``1e-8`` of the revolution
    axis — one vertex for the whole revolution rather than one per slice — and ``offsets[j]`` is
    where the owning column's block starts.

    This replaces the position-hash vertex merge trimesh gets from ``Trimesh(process=True)``. It is
    exact and scale independent, where a position hash both misses ``+0.0``/``-0.0`` pole pairs and
    starts merging genuinely distinct vertices at high section counts.
    """
    per = profile_np.shape[0]
    on_axis_np = np.abs(profile_np[:, 0]) <= TOLERANCE_MERGE
    column_np = np.arange(per, dtype=np.int32)
    if per > 2 and bool(np.all(np.abs(profile_np[0] - profile_np[-1]) <= TOLERANCE_MERGE)):
        column_np[per - 1] = 0
    owns_np = column_np == np.arange(per)
    counts_np = np.where(on_axis_np, 1, n_kept_slices) * owns_np
    offsets_np = np.concatenate(([0], np.cumsum(counts_np)[:-1])).astype(np.int32)
    return column_np, offsets_np, on_axis_np, int(counts_np.sum())


def extrude_triangulation(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    height: float,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude a 2D triangulation along Z into a watertight mesh.

    The input triangulation becomes the two caps; the walls are raised from its boundary edges,
    which are recovered from the triangulation itself rather than assumed, so a subdivided outline
    works.

    Parameters
    ----------
    vertices
        ``(n,)`` 2D vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer into ``vertices``.
    height
        Distance to extrude along Z. May be negative; the triangulation is re-wound to agree with
        its sign so the result always has positive volume.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``. Face winding is reversed when its determinant is negative.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``2 * n`` vertices, on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``vertices`` is not a rank-1 ``wp.vec2`` array, ``faces`` is not a flat multiple of 3,
        or ``abs(height)`` is at most ``1e-8``.

    Notes
    -----
    Unlike [`trimesh.creation.extrude_triangulation`][], which samples the first ten triangles to
    decide the input winding, this averages the signed area of *every* triangle. The answer is the
    same for any consistently wound triangulation and does not depend on which triangles happen to
    come first.

    The walls index the cap vertices directly instead of carrying their own four-vertex quad per
    boundary edge, so the output is watertight with no vertex-merge pass (trimesh relies on
    ``Trimesh(process=True)`` to fuse the three blocks) and has exactly ``2 * n`` vertices.

    See Also
    --------
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`sweep_polygon`][triwarp.creation.sweep_polygon]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.extrude_triangulation`][]
    """
    twt.ensure_ndim(vertices, 1, dtype=wp.vec2)
    twt.ensure_ndim(faces, 1, dtype=wp.int32)
    device = vertices.device
    n = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if int(faces.shape[0]) % 3 != 0:
        raise ValueError(f"faces size must be a multiple of 3, got {int(faces.shape[0])}")
    height_f = float(height)
    if abs(height_f) < TOLERANCE_MERGE:
        raise ValueError(f"height must be nonzero, got {height_f}")

    if n_faces == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)

    # Re-wind the triangulation to match the sign of the extrusion, so both caps and the walls end
    # up facing outward.
    areas = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_creation.triangulation_signed_areas,
        dim=n_faces,
        inputs=[vertices, faces, areas],
        device=device,
    )
    if math.copysign(1.0, tw.reduce.mean(areas)) != math.copysign(1.0, height_f):
        flipped = wp.empty_like(faces)
        wp.launch(
            kernel_creation.reverse_face_winding,
            dim=n_faces,
            inputs=[faces, flipped],
            device=device,
        )
        faces = flipped

    bottom = wp.empty(2 * n, dtype=wp.vec3, device=device)
    wp.map(kernel_creation.lift_vec2, vertices, wp.float32(0.0), out=bottom[:n])
    wp.map(kernel_creation.lift_vec2, vertices, wp.float32(height_f), out=bottom[n:])

    boundary = tw.boundary.oriented_boundary_edges(bottom[:n].contiguous(), faces)
    n_boundary = int(boundary.shape[0])

    out_faces = wp.empty((2 * n_faces + 2 * n_boundary) * 3, dtype=wp.int32, device=device)
    # Bottom cap winding is reversed; the top cap keeps it and is offset by one vertex block.
    wp.launch(
        kernel_creation.offset_cap_faces,
        dim=n_faces,
        inputs=[faces, wp.int32(0), True, out_faces[: n_faces * 3]],
        device=device,
    )
    wp.launch(
        kernel_creation.offset_cap_faces,
        dim=n_faces,
        inputs=[faces, wp.int32(n), False, out_faces[n_faces * 3 : 2 * n_faces * 3]],
        device=device,
    )
    if n_boundary > 0:
        wp.launch(
            kernel_creation.extrude_wall_faces,
            dim=n_boundary,
            inputs=[boundary, wp.int32(n), out_faces[2 * n_faces * 3 :]],
            device=device,
        )

    return _apply_transform(bottom, out_faces, transform)


def extrude_polygon(
    polygon: wp.array[wp.vec2],
    height: float,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    mid_plane: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude a simple 2D polygon along Z into a watertight prism.

    Triangulates the ring with [`triangulate_polygon`][triwarp.creation.triangulate_polygon] and
    hands the result to [`extrude_triangulation`][triwarp.creation.extrude_triangulation].

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped. Interior rings
        (holes) are not supported.
    height
        Distance to extrude along Z.
    transform
        Transform applied after construction, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    mid_plane
        Center the extrusion on ``z = 0`` instead of starting it there.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``polygon.device``.

    See Also
    --------
    [`triangulate_polygon`][triwarp.creation.triangulate_polygon]
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`sweep_polygon`][triwarp.creation.sweep_polygon]
    [`trimesh.creation.extrude_polygon`][]
    """
    ring, faces = triangulate_polygon(polygon)
    if mid_plane:
        translation = np.eye(4)
        translation[2, 3] = abs(float(height)) / -2.0
        if transform is None:
            transform = wp.mat44(*translation.flatten().tolist())
        else:
            composed = _transform_to_numpy(transform).dot(translation)
            transform = wp.mat44(*composed.flatten().tolist())
    return extrude_triangulation(ring, faces, height, transform=transform)


def triangulate_polygon(polygon: wp.array[wp.vec2]) -> tuple[wp.array[wp.vec2], wp.array[wp.int32]]:
    """
    Triangulate a simple 2D polygon by ear clipping, adding no new vertices.

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped.

    Returns
    -------
    ring : wp.array[wp.vec2]
        The input ring with any repeated closing point removed; the vertices ``faces`` indexes.
    faces : wp.array[wp.int32]
        Length-``3 * (n - 2)`` flat triangle index buffer into ``ring``.

    Notes
    -----
    Differs from [`trimesh.creation.triangulate_polygon`][] in three ways, all of which follow from
    replacing the CPU polygon libraries (``mapbox_earcut`` / ``manifold3d`` / ``triangle``) with
    triwarp's own GPU ear clipper, [`triangulate_polyline`][triwarp.polyline.triangulate_polyline]:

    - The input is a ``wp.vec2`` ring, not a ``shapely.geometry.Polygon``, and **interior rings
      (holes) are not supported**.
    - No Steiner points are ever inserted, which is trimesh's ``force_vertices=True`` contract
      rather than its default.
    - There is no ``engine`` selection, and ``faces`` is flat rather than ``(m, 3)``, matching the
      face layout used throughout triwarp.

    The ring must be a simple (non self-intersecting) polygon. A degenerate ring yields a partial
    triangulation with fewer than ``n - 2`` triangles rather than raising.

    See Also
    --------
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`triangulate_polyline`][triwarp.polyline.triangulate_polyline]
    [`trimesh.creation.triangulate_polygon`][]
    """
    twt.ensure_ndim(polygon, 1, dtype=wp.vec2)
    device = polygon.device
    n = int(polygon.shape[0])
    if n < 3:
        return polygon, wp.empty(0, dtype=wp.int32, device=device)

    lifted = wp.empty(n, dtype=wp.vec3, device=device)
    wp.map(kernel_creation.lift_vec2, polygon, wp.float32(0.0), out=lifted)
    opened = tw.polyline.open_polyline(lifted)
    faces = tw.polyline.triangulate_polyline(opened).reshape((-1,))
    # open_polyline only ever drops a repeated final point, so the matching 2D ring is a prefix.
    return polygon[: int(opened.shape[0])].contiguous(), faces


def sweep_polygon(
    polygon: wp.array[wp.vec2],
    path: wp.array[wp.vec3],
    angles: wp.array[wp.float32] | None = None,
    cap: bool = True,
    connect: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Sweep a simple 2D polygon along a 3D path.

    One copy of the polygon is placed at every path vertex, on the plane bisecting the two
    adjacent path segments, and consecutive copies are bridged into walls.

    Parameters
    ----------
    polygon
        ``(n,)`` 2D ring vertices, in order. A repeated closing point is dropped. Interior rings
        (holes) are not supported.
    path
        ``(m,)`` path vertices, ``m >= 2``. The path counts as closed when its first and last
        vertex coincide.
    angles
        ``(m,)`` roll of the polygon about the path tangent at each path vertex, in radians.
        Defaults to no roll.
    cap
        Triangulate the two ends. Ignored when the path is closed and ``connect`` is set, which
        needs no caps.
    connect
        For a closed path, fuse the last slice back onto the first so the sweep is a single
        watertight body.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``polygon.device``.

    Raises
    ------
    ValueError
        If ``path`` has fewer than 2 vertices, ``angles`` does not match ``path`` in length, or the
        polygon's triangulation is not bounded by exactly one edge per ring vertex (which means the
        ring is not a simple polygon).

    See Also
    --------
    [`extrude_polygon`][triwarp.creation.extrude_polygon]
    [`revolve`][triwarp.creation.revolve]
    [`trimesh.creation.sweep_polygon`][]
    """
    twt.ensure_ndim(path, 1, dtype=wp.vec3)
    device = polygon.device
    n_path = int(path.shape[0])
    if n_path < 2:
        raise ValueError(f"path must have at least 2 points, got {n_path}")
    if angles is None:
        angles = wp.zeros(n_path, dtype=wp.float32, device=device)
    elif int(angles.shape[0]) != n_path:
        raise ValueError(
            f"angles must have one entry per path point ({n_path}), got {angles.shape}"
        )

    ring, cap_faces = triangulate_polygon(polygon)
    stride = int(ring.shape[0])
    # oriented_boundary_edges only uses the vertex count (as its row-hash base), and the ring's own
    # 3D positions are never needed here, so a zero buffer of the right length is enough.
    boundary = tw.boundary.oriented_boundary_edges(
        wp.zeros(stride, dtype=wp.vec3, device=device), cap_faces
    )
    n_boundary = int(boundary.shape[0])
    if n_boundary != stride:
        raise ValueError(
            f"polygon must be a simple ring: its triangulation has {n_boundary} boundary edges "
            f"for {stride} vertices"
        )

    path_np = path.numpy().astype(np.float64)
    closed = bool(np.linalg.norm(path_np[0] - path_np[-1]) < TOLERANCE_MERGE)
    connect_closed = closed and connect

    normals = wp.empty(n_path, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_creation.sweep_plane_normals,
        dim=n_path,
        inputs=[path, connect_closed, normals],
        device=device,
    )
    transforms = wp.empty(n_path, dtype=wp.mat44, device=device)
    wp.launch(
        kernel_creation.sweep_transforms,
        dim=n_path,
        inputs=[path, angles, normals, transforms],
        device=device,
    )

    # A connected closed path drops its duplicate final slice and wraps onto the first instead.
    n_slices = n_path - 1
    n_vertices = (n_slices if connect_closed else n_path) * stride
    vertices = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_creation.sweep_slice_vertices,
        dim=(n_vertices // stride, stride),
        inputs=[ring, transforms, wp.int32(stride), vertices],
        device=device,
    )

    n_cap = 0 if connect_closed or not cap else int(cap_faces.shape[0]) // 3
    faces = wp.empty((2 * n_slices * n_boundary + 2 * n_cap) * 3, dtype=wp.int32, device=device)
    wp.launch(
        kernel_creation.sweep_wall_faces,
        dim=(n_slices, n_boundary),
        inputs=[
            boundary,
            wp.int32(stride),
            wp.int32(n_vertices),
            faces[: 2 * n_slices * n_boundary * 3],
        ],
        device=device,
    )
    if n_cap > 0:
        base = 2 * n_slices * n_boundary * 3
        wp.launch(
            kernel_creation.offset_cap_faces,
            dim=n_cap,
            inputs=[cap_faces, wp.int32(0), True, faces[base : base + n_cap * 3]],
            device=device,
        )
        wp.launch(
            kernel_creation.offset_cap_faces,
            dim=n_cap,
            inputs=[
                cap_faces,
                wp.int32(stride * n_slices),
                False,
                faces[base + n_cap * 3 : base + 2 * n_cap * 3],
            ],
            device=device,
        )

    return vertices, faces


def truncated_prisms(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    origin: wp.vec3 | None = None,
    normal: wp.vec3 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Extrude every triangle of a mesh down onto a plane, as one watertight prism each.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    origin
        Point on the truncation plane. ``None`` truncates against the ``z = 0`` plane.
    normal
        Unit normal of the truncation plane. Required when ``origin`` is given.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``6 * n_faces`` vertices and ``8 * n_faces`` triangles, on
        ``vertices.device``.

    Raises
    ------
    ValueError
        If ``faces`` is not a flat multiple of 3, or ``origin`` is given without ``normal``.

    Notes
    -----
    Takes an indexed mesh rather than [`trimesh.creation.truncated_prisms`][]'s ``(n, 3, 3)``
    triangle soup, matching the ``(vertices, faces)`` convention used throughout triwarp. Vertices
    are *not* merged, so each prism stays a separate body — the same choice trimesh makes with
    ``process=False``.

    See Also
    --------
    [`extrude_triangulation`][triwarp.creation.extrude_triangulation]
    [`trimesh.creation.truncated_prisms`][]
    """
    twt.ensure_ndim(faces, 1, dtype=wp.int32)
    device = vertices.device
    if int(faces.shape[0]) % 3 != 0:
        raise ValueError(f"faces size must be a multiple of 3, got {int(faces.shape[0])}")
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)

    if origin is None:
        transform_np = np.eye(4)
    else:
        if normal is None:
            raise ValueError("normal is required when origin is given")
        transform_np = _plane_transform(np.array(origin), np.array(normal))

    out_vertices = wp.empty(6 * n_faces, dtype=wp.vec3, device=device)
    out_faces = wp.empty(24 * n_faces, dtype=wp.int32, device=device)
    wp.launch(
        kernel_creation.truncated_prism_geometry,
        dim=n_faces,
        inputs=[
            vertices,
            faces,
            wp.mat44(*transform_np.flatten().tolist()),
            wp.mat44(*np.linalg.inv(transform_np).flatten().tolist()),
            out_vertices,
            out_faces,
        ],
        device=device,
    )
    return out_vertices, out_faces


def _plane_transform(origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Transform moving the plane through ``origin`` with ``normal`` onto the XY plane."""
    transform = _align_vectors(np.asanyarray(normal, dtype=np.float64).reshape(3), _UNIT_Z)
    transform[:3, 3] = -transform.dot(
        np.append(np.asanyarray(origin, dtype=np.float64).reshape(3), 1.0)
    )[:3]
    return transform


def axis(
    origin_size: float = 0.04,
    transform: wp.mat44 | wp.array[wp.mat44] | None = None,
    axis_radius: float | None = None,
    axis_length: float | None = None,
    device: wp.DeviceLike = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create an XYZ axis marker: a ball at the origin and one cylinder along each axis.

    Parameters
    ----------
    origin_size
        Radius of the ball marking the origin. The other defaults are derived from it.
    transform
        Transform applied to the whole marker, as a ``(1,)`` ``wp.mat44`` array or a scalar
        ``wp.mat44``.
    axis_radius
        Radius of the three axis cylinders. Defaults to ``origin_size / 5``.
    axis_length
        Length of the three axis cylinders. Defaults to ``origin_size * 10``.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` on ``device``, with the ball and the three cylinders concatenated in
        X, Y, Z order after the ball.

    Notes
    -----
    [`trimesh.creation.axis`][] colors the ball white and the cylinders red/green/blue for X/Y/Z.
    triwarp carries no visual attributes, so this returns geometry only and takes no
    ``origin_color``; the pieces are concatenated in a documented order so a caller can assign
    per-face colors itself.

    See Also
    --------
    [`icosphere`][triwarp.creation.icosphere]
    [`cylinder`][triwarp.creation.cylinder]
    [`concatenate`][triwarp.combine.concatenate]
    [`trimesh.creation.axis`][]
    """
    origin_size_f = float(origin_size)
    if axis_radius is None:
        axis_radius = origin_size_f / 5.0
    if axis_length is None:
        axis_length = origin_size_f * 10.0

    # Each cylinder is built centered on the origin, so it has to be pushed out along its own axis
    # by half its length. The marker transform is applied once, to the assembled result.
    shift = np.eye(4)
    shift[2, 3] = float(axis_length) / 2.0

    parts = [icosphere(radius=origin_size_f, device=device)]
    for direction in (_UNIT_X, _UNIT_Y, _UNIT_Z):
        placement = _align_vectors(_UNIT_Z, direction).dot(shift)
        parts.append(
            cylinder(
                radius=float(axis_radius),
                height=float(axis_length),
                transform=wp.mat44(*placement.flatten().tolist()),
                device=device,
            )
        )
    return _apply_transform(*tw.combine.concatenate(parts), transform)


def random_soup(
    face_count: int = 100, seed: int | None = None, device: wp.DeviceLike = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Create a soup of random, unconnected triangles inside the unit cube around the origin.

    Parameters
    ----------
    face_count
        Number of triangles. Each gets its own three vertices, so nothing is shared.
    seed
        RNG seed. A random one is drawn when omitted.
    device
        Warp device for the result. Defaults to the current device.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(vertices, faces)`` with ``3 * face_count`` vertices in ``[-0.5, 0.5] ** 3`` and
        ``faces == arange(3 * face_count)``.

    Notes
    -----
    Takes an explicit ``seed`` where [`trimesh.creation.random_soup`][] draws from NumPy's global
    RNG, so results are reproducible. The point values come from Warp's per-thread generator and do
    not match NumPy's stream.

    See Also
    --------
    [`trimesh.creation.random_soup`][]
    """
    n = 3 * max(int(face_count), 0)
    vertices = wp.empty(n, dtype=wp.vec3, device=device)
    if n > 0:
        resolved_seed = secrets.randbelow(2**31) if seed is None else int(seed)
        wp.launch(
            kernel_creation.random_soup_vertices,
            dim=n,
            inputs=[wp.int32(resolved_seed), vertices],
            device=device,
        )
    return vertices, tw.array.init_range(n, str(vertices.device))


# --- private helpers ---------------------------------------------------------------------


def _segment_to_cylinder(segment: np.ndarray) -> tuple[wp.mat44, float]:
    """Convert a 3D line segment to the transform and height of a Z-extruded origin cylinder."""
    segment_np = np.asanyarray(segment, dtype=np.float64)
    if segment_np.shape != (2, 3):
        raise ValueError(f"segment must be (2, 3) float, got {segment_np.shape}")
    vector = segment_np[1] - segment_np[0]
    height = float(np.linalg.norm(vector))
    matrix = _align_vectors(np.array([0.0, 0.0, 1.0]), vector)
    # Compose translation-to-midpoint with the rotation.
    matrix[:3, 3] = segment_np[0] + vector * 0.5
    return wp.mat44(*matrix.flatten().tolist()), height


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Homogeneous rotation taking unit vector ``a`` onto ``b`` (``trimesh.geometry`` port)."""
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]
    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0
    matrix = np.eye(4)
    matrix[:3, :3] = bu.dot(au.T)
    return matrix


def _apply_transform(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    transform: wp.mat44 | wp.array[wp.mat44] | None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Transform ``vertices`` in place, reversing face winding when the transform is a mirror."""
    if transform is None:
        return vertices, faces

    device = vertices.device
    matrix = (
        transform
        if isinstance(transform, wp.array)
        else wp.array([transform], dtype=wp.mat44, device=device)
    )
    wp.launch(
        kernel_registration.apply_transform_mat44,
        dim=int(vertices.shape[0]),
        inputs=[vertices, matrix, vertices],
        device=device,
    )
    if float(np.linalg.det(_transform_to_numpy(transform)[:3, :3])) < 0.0:
        wp.launch(
            kernel_creation.reverse_face_winding,
            dim=int(faces.shape[0]) // 3,
            inputs=[faces, faces],
            device=device,
        )
    return vertices, faces


def _transform_to_numpy(transform: wp.mat44 | wp.array[wp.mat44]) -> np.ndarray:
    """
    Read a transform parameter back as a host ``(4, 4)`` array.

    Needed for the two decisions that cannot be made on the device: whether the transform reverses
    face winding, and composing it with the ``mid_plane`` offset in
    [`extrude_polygon`][triwarp.creation.extrude_polygon]. A ``wp.array`` argument costs one
    device-to-host synchronization here.
    """
    if isinstance(transform, wp.array):
        return transform.numpy().reshape(4, 4).astype(np.float64)
    return np.array(transform, dtype=np.float64).reshape(4, 4)


def _upload_vertices(vertices_np: np.ndarray, device: wp.DeviceLike) -> wp.array[wp.vec3]:
    """Upload an ``(n, 3)`` host array as a ``wp.vec3`` device buffer."""
    return wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
    )


def _upload_profile(profile_np: np.ndarray, device: wp.DeviceLike) -> wp.array[wp.vec2]:
    """
    Upload an ``(n, 2)`` revolution profile as a ``wp.vec2`` device buffer.

    Profiles are built on the host in ``float64`` and downcast here. Evaluating them on the
    device in ``float32`` instead would put the closing point of a closed profile roughly
    ``1e-7`` away from its first point — outside [`revolve`][triwarp.creation.revolve]'s weld
    tolerance — while ``float64`` keeps the gap near ``1e-16``. The profiles are tiny; the
    per-slice work they drive is what runs on the device.
    """
    return wp.array(
        np.ascontiguousarray(profile_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
