"""
Visibility fields over a surface: how open each point is to the space around it.

Both functions here integrate the same thing — a bundle of rays over the outward hemisphere at each
point, weighted by Lambert's cosine law — and differ only in what a blocked ray costs.
[`ambient_occlusion`][triwarp.shading.ambient_occlusion] charges a hit its full weight however far
away it is; [`volumetric_obscurance`][triwarp.shading.volumetric_obscurance] discounts it by
``exp(-tau * distance)``, so a distant wall barely darkens a point and only nearby geometry does.
Ambient occlusion is the ``tau -> 0`` limit of obscurance, which is why they share a kernel.

These are *illumination* fields, and the counterpart to
[`shape_diameter`][triwarp.proximity.shape_diameter], which fires the same kind of bundle **inward**
to measure local thickness instead. Both are embarrassingly parallel and both are among the slowest
filters MeshLab ships, which is the whole reason they are here.

Neither is normalized against a scene: the value at a point depends only on the mesh, so it is
comparable across meshes and across resolutions. On a **convex** closed surface no ray can return,
so every point reads exactly ``0``.
"""

from __future__ import annotations

from typing import Literal

import warp as wp

import triwarp as tw
from triwarp.bounds import enclosing_diagonal
from triwarp.kernels import shading as kernel_shading
from triwarp.proximity import normals_at_closest_faces

# Ray-origin offset along the normal, as a fraction of the query AABB diagonal. Without it every
# ray would hit the surface it started on; the value is small enough not to shadow a real occluder
# and large enough to clear float32 error on the starting triangle.
_SURFACE_OFFSET = 1e-4

_WEIGHT_MODES: dict[str, wp.int32] = {
    "cosine": kernel_shading.WEIGHT_COSINE,
    "uniform": kernel_shading.WEIGHT_UNIFORM,
}

RayWeight = Literal["cosine", "uniform"]
"""Weighting of a ray; see [`ambient_occlusion`][triwarp.shading.ambient_occlusion]."""


def ambient_occlusion(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    weight: RayWeight = "cosine",
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Fraction of the outward hemisphere at each point that is blocked by the mesh itself.

    A Fibonacci hemisphere lattice is rotated into each point's tangent frame and every direction is
    traced against the mesh; the result is the weighted share of directions that hit something. This
    is MeshLab's ``compute_scalar_ambient_occlusion`` and libigl's ``igl::ambient_occlusion``, which
    differ from each other exactly in the ``weight`` argument below.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``). The occluder *and* the surface being shaded.
    points
        ``(m,)`` positions to shade, normally the mesh's own vertices.
    normals
        ``(m,)`` outward unit normals, defining which hemisphere is "outward" at each point. When
        ``None`` they are taken from the closest face of ``mesh``
        ([`normals_at_closest_faces`][triwarp.proximity.normals_at_closest_faces]), which is right
        for points on the surface and meaningless for points far off it — pass them explicitly in
        that case. For a smooth result on the mesh's own vertices, pass
        [`area_weighted_vertex_normals`][triwarp.vertices.area_weighted_vertex_normals] instead:
        face normals make the field piecewise constant across each vertex's ring.
    n_rays
        Directions per point. Error falls as ``1 / sqrt(n_rays)``; MeshLab's default is ``64``,
        which is this one. Cost is exactly linear in it.
    weight
        How a direction is weighted in the integral:

        - ``"cosine"`` (default) — by ``dot(direction, normal)``, which is the physically correct
          weighting for ambient irradiance and MeshLab's. Grazing directions barely matter.
        - ``"uniform"`` — every direction counts once, which is libigl's convention and reads as a
          solid-angle fraction rather than an irradiance one.
    max_t
        Maximum ray length. Anything beyond it does not occlude. When ``None``, the diagonal of the
        AABB enclosing the mesh and the query points, so nothing in the mesh is missed.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` occlusion in ``[0, 1]`` on ``points.device``: ``0`` where the hemisphere is fully
        open and ``1`` where it is fully blocked. **Exactly ``0`` everywhere on a convex closed
        mesh**, which is the cheapest available sanity check on a result.

    Raises
    ------
    ValueError
        If ``n_rays < 1``, ``weight`` is not one of the two names, or ``normals`` has a different
        length from ``points``.

    See Also
    --------
    [`volumetric_obscurance`][triwarp.shading.volumetric_obscurance]
    [`triwarp.proximity.shape_diameter`][triwarp.proximity.shape_diameter]
    [`triwarp.sample.sample_fibonacci_hemisphere`][triwarp.sample.sample_fibonacci_hemisphere]

    Notes
    -----
    MeshLab reports the *complement* of this (higher is more exposed) and leaves it unnormalized —
    its number is roughly ``(1 - occlusion) * n_rays / 4`` for a cosine-weighted bundle over
    ``n_rays`` whole-sphere directions. Its direction set is also its own, so the two agree in
    distribution and in ranking rather than value by value.
    """
    return _occlusion_bundle(mesh, points, normals, n_rays, weight, 0.0, max_t, "ambient_occlusion")


def volumetric_obscurance(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    tau: float = 0.1,
    weight: RayWeight = "cosine",
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Distance-attenuated ambient occlusion: an occluder at range ``t`` counts ``exp(-tau * t)``.

    Iones et al.'s obscurance, and MeshLab's ``compute_scalar_by_volumetric_obscurance``. It exists
    because binary [`ambient_occlusion`][triwarp.shading.ambient_occlusion] treats a wall across the
    room like a crevice wall a millimetre away, which darkens the interior of any closed room
    uniformly and hides exactly the small-scale detail the field is usually wanted for. Attenuating
    by distance keeps the response local.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(m,)`` positions to shade.
    normals
        ``(m,)`` outward unit normals; see
        [`ambient_occlusion`][triwarp.shading.ambient_occlusion] for the default.
    n_rays
        Directions per point.
    tau
        Attenuation rate, in inverse length units of the mesh — so it is **not** scale-invariant,
        and a mesh scaled by ``k`` wants ``tau / k`` for the same result. MeshLab's default is
        ``0.1``, which suits a mesh of extent order 1. As ``tau -> 0`` this becomes
        [`ambient_occlusion`][triwarp.shading.ambient_occlusion]; as ``tau -> inf`` everything reads
        ``0``. Must be positive.
    weight
        Ray weighting, as in [`ambient_occlusion`][triwarp.shading.ambient_occlusion].
    max_t
        Maximum ray length.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` obscurance-weighted occlusion in ``[0, 1]`` on ``points.device``, ``0`` where
        nothing nearby blocks. Exactly ``0`` on a convex closed mesh.

    Raises
    ------
    ValueError
        If ``tau <= 0``, ``n_rays < 1``, ``weight`` is unknown, or ``normals`` is the wrong length.

    See Also
    --------
    [`ambient_occlusion`][triwarp.shading.ambient_occlusion]
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}; use ambient_occlusion for the limit")
    return _occlusion_bundle(
        mesh, points, normals, n_rays, weight, tau, max_t, "volumetric_obscurance"
    )


def _occlusion_bundle(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3] | None,
    n_rays: int,
    weight: RayWeight,
    tau: float,
    max_t: float | None,
    name: str,
) -> wp.array[wp.float32]:
    """Shared hemisphere-bundle trace behind both public functions; ``tau <= 0`` means binary."""
    if n_rays < 1:
        raise ValueError(f"{name} requires n_rays >= 1, got {n_rays}")
    if weight not in _WEIGHT_MODES:
        raise ValueError(f"weight must be 'cosine' or 'uniform', got {weight!r}")

    device = points.device
    m = int(points.shape[0])
    out_occlusion = wp.zeros(m, dtype=wp.float32, device=device)
    if m == 0:
        return out_occlusion

    if normals is None:
        normals = normals_at_closest_faces(mesh, points)
    elif int(normals.shape[0]) != m:
        raise ValueError(
            f"normals must have one entry per point, got {normals.shape[0]} for {m} points"
        )

    diagonal = enclosing_diagonal(mesh.points, points)
    directions = tw.sample.sample_fibonacci_hemisphere(n_rays, device=device)
    wp.launch(
        kernel_shading.obscurance,
        dim=m,
        inputs=[
            mesh.id,
            points,
            normals,
            directions,
            wp.float32(tau),
            _WEIGHT_MODES[weight],
            wp.float32(max_t if max_t is not None else diagonal),
            wp.float32(_SURFACE_OFFSET * max(diagonal, 1e-12)),
            out_occlusion,
        ],
        device=device,
    )
    return out_occlusion
