"""
Heat-diffusion methods on triangle meshes.

Three solvers that share one idea: diffuse a quantity over the surface for a short time ``t``, then
recover the answer from the *direction* of the resulting field rather than its magnitude. Short-time
heat flow approximates the geodesic kernel, so a single sparse solve carries information that a
combinatorial shortest-path search would have to walk edge by edge. Each solver differs only in what
it diffuses and how it reads the result back:

- [`distance`][triwarp.heat.distance] — diffuse a scalar indicator from source vertices, normalize
  its gradient, and integrate that unit field back with a Poisson solve. This is the heat method of
  Crane et al. (``igl::heat_geodesics``, ``potpourri3d.MeshHeatMethodDistanceSolver``).
- [`vector`][triwarp.heat.vector] — diffuse *tangent vectors* through the connection Laplacian,
  which transports each vector into its neighbour's frame before differencing. Gives parallel
  transport, nearest-source extension and the logarithmic map
  (``potpourri3d.MeshVectorHeatSolver``).
- [`signed`][triwarp.heat.signed] — diffuse the normals of a set of oriented curves, then solve a
  Poisson problem against that field to get a *signed* distance whose zero set is the curves
  (``potpourri3d.MeshSignedHeatSolver``).

All three run in ``float64``, because the diffused field decays exponentially and underflows
``float32``. They run on either device.

The operators these solvers assemble are **not** here — the cotangent and connection Laplacians live
in [`triwarp.laplacian`][triwarp.laplacian], tangent frames in
[`triwarp.tangent_space`][triwarp.tangent_space], and the batched conjugate-gradient machinery in
[`triwarp.linalg`][triwarp.linalg]. This package is the three algorithms only.
"""

from . import distance, signed, vector

__all__ = ["distance", "signed", "vector"]
