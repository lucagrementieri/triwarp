"""Mesh file loading via the optional ``meshio`` dependency."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from triwarp._device import require_nonempty_mesh
from triwarp.mesh import Trimesh

if TYPE_CHECKING:
    import meshio

# Per-vertex / per-face attribute column names as written by common mesh formats.
_NORMAL_COLUMNS = ("nx", "ny", "nz")
_UV_COLUMN_SETS = (("u", "v"), ("s", "t"), ("texture_u", "texture_v"))
_COLOR_COLUMNS = ("red", "green", "blue")
_COLOR_COLUMNS_RGBA = ("red", "green", "blue", "alpha")


def _import_meshio():
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "triwarp.io requires meshio. Install it with `pip install triwarp[io]`."
        ) from exc
    return meshio


def load_mesh_data(path: str | Path, *, device: wp.DeviceLike = None) -> dict[str, wp.array[Any]]:
    """
    Load every mesh attribute ``meshio`` can read from a file into Warp arrays.

    Reads ``path`` once with ``meshio`` and returns a dictionary whose keys are present
    only when the corresponding data exists in the file. ``vertices`` is always present;
    everything else is optional.

    Parameters
    ----------
    path
        Path to a mesh file in any format supported by ``meshio`` (PLY, STL, OBJ, OFF,
        VTK, glTF, ...).
    device
        Warp device for the returned arrays. Defaults to the current Warp device.

    Returns
    -------
    dict
        Mapping with these possible keys:

        - ``vertices`` : ``wp.array[wp.vec3]`` of ``float32`` vertex positions (always present).
        - ``faces`` : flat ``wp.array[wp.int32]`` of length ``3 * n_faces`` (triangle cells only).
        - ``vertex_normals`` : ``wp.array[wp.vec3]`` from ``nx, ny, nz`` per-vertex data.
        - ``uv`` : ``wp.array[wp.vec2]`` from ``u, v`` (or ``s, t`` / ``texture_u, texture_v``).
        - ``colors`` : ``wp.array[wp.vec3]`` or ``wp.array[wp.vec4]`` from ``red, green, blue``
          (and optional ``alpha``); integer channels are normalized to ``[0, 1]``.
        - ``face_normals`` : ``wp.array[wp.vec3]`` — only when the file stored per-face normals.

    See Also
    --------
    [`load_mesh`][triwarp.io.load_mesh]

    Notes
    -----
    Only triangle cells are consumed; non-triangle cell blocks are ignored. ``meshio`` reads
    attributes verbatim and never computes them, so ``face_normals`` appears only when the
    file itself contained per-face normals.
    """
    meshio = _import_meshio()
    mesh = meshio.read(path)

    result: dict[str, wp.array[Any]] = {}

    points = np.ascontiguousarray(mesh.points, dtype=np.float32)
    result["vertices"] = wp.array(points, dtype=wp.vec3, device=device)

    faces_np = mesh.cells_dict.get("triangle")
    if faces_np is not None and len(faces_np) > 0:
        faces_flat = np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32)
        result["faces"] = wp.array(faces_flat, dtype=wp.int32, device=device)

    point_data = mesh.point_data

    normals = _stack_columns(point_data, _NORMAL_COLUMNS)
    if normals is not None:
        result["vertex_normals"] = wp.array(
            np.ascontiguousarray(normals, dtype=np.float32), dtype=wp.vec3, device=device
        )

    for names in _UV_COLUMN_SETS:
        uv = _stack_columns(point_data, names)
        if uv is not None:
            result["uv"] = wp.array(
                np.ascontiguousarray(uv, dtype=np.float32), dtype=wp.vec2, device=device
            )
            break

    colors_raw = _stack_columns(point_data, _COLOR_COLUMNS_RGBA)
    color_dtype = wp.vec4
    if colors_raw is None:
        colors_raw = _stack_columns(point_data, _COLOR_COLUMNS)
        color_dtype = wp.vec3
    if colors_raw is not None:
        colors = colors_raw.astype(np.float32)
        if np.issubdtype(colors_raw.dtype, np.integer):
            colors /= 255.0
        result["colors"] = wp.array(
            np.ascontiguousarray(colors, dtype=np.float32), dtype=color_dtype, device=device
        )

    face_normals = _face_normals_from_cell_data(mesh)
    if face_normals is not None:
        result["face_normals"] = wp.array(
            np.ascontiguousarray(face_normals, dtype=np.float32), dtype=wp.vec3, device=device
        )

    return result


def load_mesh(path: str | Path, *, device: wp.DeviceLike = None) -> wp.Mesh:
    """
    Load a triangle mesh file into a ``warp.Mesh``.

    Thin wrapper over [`load_mesh_data`][triwarp.io.load_mesh_data] that keeps only the
    vertices and triangle faces and wraps them in a ``warp.Mesh`` that owns its buffers.

    Parameters
    ----------
    path
        Path to a mesh file in any format supported by ``meshio``.
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    warp.Mesh
        Mesh with ``vec3`` points and flat ``int32`` indices.

    Raises
    ------
    ValueError
        If the file contains no triangle faces (e.g. a point cloud or a triangle-strip
        PLY that ``meshio`` cannot decode).

    See Also
    --------
    [`load_mesh_data`][triwarp.io.load_mesh_data]
    """
    data = load_mesh_data(path, device=device)
    if "faces" not in data:
        raise ValueError(f"Mesh file {path!r} has no triangle faces; cannot build a wp.Mesh.")
    require_nonempty_mesh(data["faces"], "load_mesh")
    # ``load_mesh_data`` allocated both buffers on the line above and nothing else holds
    # them, so the mesh can own them directly.
    return wp.Mesh(points=data["vertices"], indices=data["faces"])


def mesh_from_numpy(
    vertices: np.ndarray, faces: np.ndarray, *, device: wp.DeviceLike = None
) -> Trimesh:
    """
    Build a [`Trimesh`][triwarp.mesh.Trimesh] from NumPy vertex and face arrays.

    Parameters
    ----------
    vertices
        ``(n_vertices, 3)`` vertex positions, any NumPy float dtype.
    faces
        ``(n_faces, 3)`` (or already-flat length-``3 * n_faces``) triangle vertex indices,
        any NumPy integer dtype.
    device
        Warp device for the returned mesh. Defaults to the current Warp device.

    Returns
    -------
    Trimesh
        New mesh with ``vec3`` vertices and a flat ``int32`` face buffer, owning freshly
        allocated Warp arrays (no aliasing with ``vertices`` / ``faces``).

    See Also
    --------
    [`load_mesh`][triwarp.io.load_mesh]
    """
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return Trimesh(vertices_wp, faces_wp)


def _stack_columns(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    """Stack scalar ``point_data``/``cell_data`` columns into ``(n, len(names))`` or ``None``."""
    if not all(name in data for name in names):
        return None
    return np.column_stack([np.asarray(data[name]) for name in names])


def _face_normals_from_cell_data(mesh: meshio.Mesh) -> np.ndarray | None:
    """Read ``nx, ny, nz`` cell data aligned with the triangle cell block, if present."""
    cell_data = mesh.cell_data
    if not cell_data:
        return None
    tri_index = None
    for i, block in enumerate(mesh.cells):
        if block.type == "triangle":
            tri_index = i
            break
    if tri_index is None:
        return None
    columns = []
    for name in _NORMAL_COLUMNS:
        if name not in cell_data:
            return None
        columns.append(np.asarray(cell_data[name][tri_index]))
    return np.column_stack(columns)
