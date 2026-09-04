"""
Triangular mesh utilities on NVIDIA Warp.

Submodules and ``Trimesh`` resolve **lazily**, through the :pep:`562` module ``__getattr__``
below, so ``import triwarp`` costs almost nothing and a caller pays only for what it reaches.

Every ``@wp.kernel`` decoration parses its function and builds an ``Adjoint`` at *import* time, so
eagerly importing every public module would pull in every kernel module along with it and make
``import triwarp`` pay for all of them up front. Importing one submodule instead does not avoid
this either: Python imports a parent package before its child, so ``import triwarp.edges`` still
runs this file first. Lazy resolution is what keeps both cheap: ``import triwarp as tw`` (how most
wrappers open) is nearly free, and ``tw.laplacian`` costs one ``__getattr__`` on first touch, after
which it is a plain global lookup because the resolved module is cached into this module's
namespace.

Two things this deliberately does not change. Nothing is imported at a *different* time relative to
its own kernel module, so a kernel module's overload registration still runs before the first
launch through it -- a wrapper cannot be reached without importing it. And ``wp.load_module`` /
``wp.force_load`` are still not called here: registration is not compilation, and eager loading at
import would defeat the whole point of this module.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Re-imported for the type checker and for IDE completion only; at runtime these names come
    # from ``__getattr__``. basedpyright reads this branch, so the public surface stays typed.
    from triwarp import (
        adjacency,
        array,
        boundary,
        bounds,
        combine,
        constants,
        creation,
        curvature,
        edges,
        energies,
        geodesic_walk,
        graph,
        grouping,
        halfedge,
        heat,
        holes,
        homology,
        interpolation,
        intersection,
        io,
        laplacian,
        levelset,
        linalg,
        measures,
        mesh,
        metrics,
        neighbors,
        parametrization,
        points,
        polyline,
        proximity,
        ray,
        reconstruction,
        reduce,
        registration,
        remesh,
        repair,
        sample,
        seams,
        selection,
        smoothing,
        tangent_space,
        texture,
        transform,
        triangles,
        typing,
        validation,
        vertices,
        visibility,
        voxels,
    )
    from triwarp.mesh import Trimesh

__all__ = [
    "Trimesh",
    "adjacency",
    "array",
    "boundary",
    "bounds",
    "combine",
    "constants",
    "creation",
    "curvature",
    "edges",
    "energies",
    "geodesic_walk",
    "graph",
    "grouping",
    "halfedge",
    "heat",
    "holes",
    "homology",
    "interpolation",
    "intersection",
    "io",
    "laplacian",
    "levelset",
    "linalg",
    "measures",
    "mesh",
    "metrics",
    "neighbors",
    "parametrization",
    "points",
    "polyline",
    "proximity",
    "ray",
    "reconstruction",
    "reduce",
    "registration",
    "remesh",
    "repair",
    "sample",
    "seams",
    "selection",
    "smoothing",
    "tangent_space",
    "texture",
    "transform",
    "triangles",
    "typing",
    "validation",
    "vertices",
    "visibility",
    "voxels",
]

# The submodules ``__getattr__`` will resolve. ``Trimesh`` is not one of them -- it is a class in
# ``triwarp.mesh`` and is handled separately below, because importing it eagerly here is exactly
# the whole-package pull this file exists to avoid.
_LAZY_SUBMODULES = frozenset(__all__) - {"Trimesh"}


def __getattr__(name: str) -> object:
    """
    Resolve a submodule, or ``Trimesh``, on first access (:pep:`562`).

    The result is cached in this module's namespace, so only the first access reaches here and
    every later ``triwarp.<name>`` is an ordinary attribute lookup.

    Parameters
    ----------
    name
        The attribute being looked up.

    Returns
    -------
    object
        The imported submodule, or the ``Trimesh`` class.

    Raises
    ------
    AttributeError
        If ``name`` is not a public submodule or ``Trimesh``. Raising this rather than letting an
        ``ImportError`` escape is what keeps ``hasattr`` and ``getattr(..., default)`` working.
    """
    if name == "Trimesh":
        value: object = importlib.import_module("triwarp.mesh").Trimesh
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"triwarp.{name}")
    else:
        raise AttributeError(f"module 'triwarp' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """
    Public names, whether or not they have been resolved yet.

    Without this, ``dir(triwarp)`` would list only the submodules some caller happened to touch,
    which is what makes a lazy package hard to explore interactively.

    Returns
    -------
    list of str
        Every name in ``__all__``, sorted.
    """
    return sorted(__all__)
