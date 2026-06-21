"""Point cloud registration (Procrustes analysis) on NVIDIA Warp."""

from __future__ import annotations

from typing import Literal, overload

import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels import registration as kernel_registration


@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: Literal[False] = False,
) -> wp.array[wp.mat44]: ...
@overload
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: Literal[True] = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float]: ...
def procrustes(
    a: wp.array[wp.vec3],
    b: wp.array[wp.vec3],
    weights: wp.array[wp.float32] | None = None,
    reflection: bool = True,
    translation: bool = True,
    scale: bool = True,
    return_cost: bool = True,
) -> tuple[wp.array[wp.mat44], wp.array[wp.vec3], float] | wp.array[wp.mat44]:
    """
    Find the optimal transform mapping point cloud *a* onto *b*.

    Warp port of ``trimesh.registration.procrustes``. Accepts and returns
    GPU-resident arrays; no NumPy in the hot path.

    Parameters
    ----------
    a, b:
        Corresponding point clouds, shape ``(n,)``, dtype ``wp.vec3``.
    weights:
        Per-point weights, shape ``(n,)``, dtype ``wp.float32``.
        Uniform weights are assumed when ``None``.
    reflection:
        Allow reflections in the rotation component.
    translation:
        Allow translation.
    scale:
        Allow uniform scaling.
    return_cost:
        When ``True`` (default) also return the transformed points and cost.

    Returns
    -------
    matrix:
        ``wp.array[wp.mat44]`` of shape ``(1,)`` encoding the transform.
    transformed:
        ``wp.array[wp.vec3]`` of shape ``(n,)`` — image of *a* under the transform.
        Only returned when ``return_cost=True``.
    cost:
        Weighted sum of squared distances between *transformed* and *b*.
        Only returned when ``return_cost=True``.
    """
    n = a.shape[0]
    device = a.device

    if weights is None:
        weights = wp.full(n, wp.float32(1.0), dtype=wp.float32, device=device)

    # --- Phase 1a: weighted sums for centroids ---
    w_sum = wp.zeros(1, dtype=wp.float32, device=device)
    a_sum = wp.zeros(1, dtype=wp.vec3, device=device)
    b_sum = wp.zeros(1, dtype=wp.vec3, device=device)
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_registration.accumulate_weighted_sums,
        dim=[n_tiles],
        inputs=[a, b, weights, w_sum, a_sum, b_sum],
        block_dim=TILE_1D,
        device=device,
    )

    # --- Phase 1b: weighted scale^2 and cross-covariance ---
    a_scale_sq = wp.zeros(1, dtype=wp.float32, device=device)
    b_scale_sq = wp.zeros(1, dtype=wp.float32, device=device)
    cov = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch_tiled(
        kernel_registration.accumulate_scale_and_cov,
        dim=[n_tiles],
        inputs=[
            a,
            b,
            weights,
            w_sum,
            a_sum,
            b_sum,
            translation,
            a_scale_sq,
            b_scale_sq,
            cov,
        ],
        block_dim=TILE_1D,
        device=device,
    )

    # --- Phase 2: SVD and 4×4 matrix construction (single thread) ---
    out_matrix = wp.zeros(1, dtype=wp.mat44, device=device)
    wp.launch(
        kernel_registration.build_procrustes_matrix,
        dim=1,
        inputs=[
            w_sum,
            a_sum,
            b_sum,
            a_scale_sq,
            b_scale_sq,
            cov,
            reflection,
            translation,
            scale,
            out_matrix,
        ],
        device=device,
    )

    if not return_cost:
        return out_matrix

    # --- Phase 3a: apply transform to all points ---
    out_transformed = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_registration.apply_transform_mat44,
        dim=n,
        inputs=[a, out_matrix, out_transformed],
        device=device,
    )

    # --- Phase 3b: weighted cost reduction ---
    out_cost = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_registration.accumulate_cost,
        dim=n,
        inputs=[out_transformed, b, weights, w_sum, out_cost],
        device=device,
    )

    cost = float(out_cost.numpy()[0])
    return out_matrix, out_transformed, cost
