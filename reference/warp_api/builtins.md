# Warp Built-In Functions (kernel scope)

Source: https://nvidia.github.io/warp/stable/language_reference/builtins.html (Warp 1.15.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

These are callable inside `@wp.kernel` / `@wp.func` as `wp.<name>(...)`.

## Scalar Math
- `abs(x)` — absolute value of `x`.
- `acos(x)` — arccos of `x` in radians (auto-clamps to [-1,1]).
- `asin(x)` — arcsin of `x` in radians (auto-clamps to [-1,1]).
- `atan(x)` — arctangent of `x` in radians.
- `atan2(x, y)` — 2-argument arctangent of `(x, y)` in radians.
- `cbrt(x)` — cube root of `x`.
- `ceil(x)` — smallest integer >= `x`.
- `clamp(x, low, high)` — clamp `x` to [low, high].
- `copysign(x, y)` — magnitude of `x` with sign of `y`.
- `cos(x)` — cosine of `x` (radians).
- `cosh(x)` — hyperbolic cosine of `x`.
- `degrees(x)` — radians -> degrees.
- `erf(x)` — error function.
- `erfc(x)` — complementary error function.
- `erfcinv(x)` — inverse complementary error function.
- `erfinv(x)` — inverse error function.
- `exp(x)` — e^x.
- `floor(x)` — largest integer <= `x`.
- `frac(x)` — fractional part of `x`.
- `isfinite(a)` — check if `a` is finite.
- `isinf(a)` — check if `a` is +/- infinity.
- `isnan(a)` — check if `a` is NaN.
- `log(x)` — natural log (x>0).
- `log2(x)` — base-2 log (x>0).
- `log10(x)` — base-10 log (x>0).
- `max(a, b)` — maximum.
- `min(a, b)` — minimum.
- `nonzero(x)` — 1.0 if `x` != 0 else 0.0.
- `pow(x, y)` — `x` raised to `y`.
- `radians(x)` — degrees -> radians.
- `rint(x)` — nearest integer, ties to even.
- `round(x)` — nearest integer, ties away from zero.
- `sign(x)` — sign of `x`.
- `sin(x)` — sine of `x` (radians).
- `sinh(x)` — hyperbolic sine of `x`.
- `sqrt(x)` — square root (x>0).
- `step(x)` — 1.0 if `x`<0.0 else 0.0.
- `tan(x)` — tangent of `x` (radians).
- `tanh(x)` — hyperbolic tangent of `x`.
- `trunc(x)` — nearest integer closer to zero than `x`.

## Vector / Matrix Math
- `argmax(a)` — index of max element of vector `a`.
- `argmin(a)` — index of min element of vector `a`.
- `cross(a, b)` — cross product of two 3D vectors.
- `cw_div(a, b)` — component-wise division.
- `cw_mul(a, b)` — component-wise product.
- `ddot(a, b)` — double dot product of two matrices.
- `determinant(a)` — determinant of matrix `a`.
- `diag(vec)` — matrix with `vec` on the diagonal.
- `dot(a, b)` — dot product.
- `eig3(A)` — eigendecomposition of 3x3 matrix.
- `get_diag(mat)` — vector of diagonal elements of square matrix.
- `identity(n, dtype)` — (n,n) identity matrix.
- `inverse(a)` — inverse of matrix `a`.
- `inverse_approx(a)` — inverse using approximate GPU intrinsics.
- `length(a)` — length of `a`.
- `length_sq(a)` — squared length of `a`.
- `matrix(...)` — construct a matrix.
- `matrix_from_cols(...)` — matrix with each vector arg as a column.
- `matrix_from_rows(...)` — matrix with each vector arg as a row.
- `norm_huber(v, delta)` — Huber norm of vector `v`.
- `norm_l1(v)` — L1 norm.
- `norm_l2(v)` — L2 norm.
- `norm_pseudo_huber(v, delta)` — pseudo-Huber norm.
- `normalize(a)` — normalized `a`.
- `outer(a, b)` — outer product `a*b^T`.
- `qr3(A)` — QR decomposition of 3x3 matrix.
- `skew(vec)` — skew-symmetric 3x3 matrix for 3D vector.
- `smooth_normalize(v)` — normalize using pseudo-Huber norm.
- `svd2(A)` — SVD of 2x2 matrix.
- `svd3(A)` — SVD of 3x3 matrix.
- `trace(a)` — trace of matrix `a`.
- `transpose(a)` — transpose of matrix `a`.
- `vector(length, dtype)` — construct a vector.

## Quaternion Math
- `quat_from_axis_angle(axis, angle)` — quaternion from axis + angle (radians).
- `quat_from_euler(angles, axes)` — quaternion from Euler angles and axis sequence.
- `quat_from_matrix(mat)` — quaternion from a matrix.
- `quat_identity()` — identity quaternion.
- `quat_inverse(q)` — quaternion conjugate.
- `quat_rotate(q, v)` — rotate vector by quaternion.
- `quat_rotate_inv(q, v)` — rotate vector by inverse quaternion.
- `quat_rpy(roll, pitch, yaw)` — quaternion from roll/pitch/yaw (radians).
- `quat_slerp(q1, q2, t)` — interpolate between two quaternions.
- `quat_to_axis_angle(q)` — extract axis and angle from quaternion.
- `quat_to_euler(q, axes)` — quaternion -> Euler angles.
- `quat_to_matrix(q)` — quaternion -> 3x3 rotation matrix.
- `quat_to_rpy(q)` — quaternion -> roll-pitch-yaw (ZYX).
- `quat_twist(q, axis)` — twist quaternion around `axis`.
- `quat_twist_angle(q, axis)` — twist magnitude around `axis`.
- `quaternion(...)` — construct a quaternion.

## Transformations
- `transform_compose(position, rotation, scale)` — compose 4x4 from pos/quat/scale.
- `transform_decompose(xform)` — decompose 4x4 into pos/quat/scale.
- `transform_from_matrix(mat)` — transform from 4x4 matrix.
- `transform_get_rotation(xform)` — rotational part of transform.
- `transform_get_translation(xform)` — translational part of transform.
- `transform_identity()` — identity transform.
- `transform_inverse(xform)` — inverse of transform.
- `transform_multiply(t1, t2)` — multiply two rigid-body transforms.
- `transform_point(xform, p)` — apply transform to a point.
- `transform_set_rotation(xform, rot)` — set rotational part.
- `transform_set_translation(xform, trans)` — set translational part.
- `transform_to_matrix(xform)` — transform -> 4x4 matrix.
- `transform_vector(xform, v)` — apply transform to a vector.
- `transformation(...)` — construct a transformation.

## Spatial Math
- `spatial_adjoint(top, bottom)` — 6x6 spatial inertial matrix from two 3x3 blocks.
- `spatial_bottom(s)` — bottom part of a 6D screw vector.
- `spatial_cross(a, b)` — cross product of two 6D screw vectors.
- `spatial_cross_dual(a, b)` — dual cross product of two 6D screw vectors.
- `spatial_dot(a, b)` — dot product of two 6D screw vectors.
- `spatial_jacobian(chain)` — spatial Jacobian for a kinematic chain.
- `spatial_mass(chain)` — composite rigid-body mass matrix.
- `spatial_top(s)` — top part of a 6D screw vector.
- `spatial_vector(...)` — construct a 6D screw vector.
- `transform_twist(s, xform)` — transform a spatial twist between frames.
- `transform_wrench(w, xform)` — transform a spatial wrench between frames.
- `velocity_at_point(s, offset)` — linear velocity of an offset point on a rigid body.

## Tile Primitives
- `tile(...)` — construct a tile from per-thread values.
- `tile_arange(start, stop, dtype)` — tile of linearly spaced elements.
- `tile_argmax(t)` / `tile_argmin(t)` — cooperative index of max/min element.
- `tile_assign(dest, offset, src)` — assign a tile to a subrange of dest.
- `tile_astype(t, dtype)` — same data, different dtype.
- `tile_atomic_add(a, offset, t)` — atomically add a tile onto array `a`.
- `tile_atomic_add_indexed(a, axis, indices, t)` — atomic add with indexed storage.
- `tile_axpy(alpha, src, dest)` — scale `src` by `alpha`, accumulate into `dest`.
- `tile_broadcast(t)` — broadcast a tile.
- `tile_bvh_query_aabb(bvh, lower, upper)` — block AABB query against BVH.
- `tile_bvh_query_next(query)` — next bound in block BVH query.
- `tile_bvh_query_ray(bvh, start, dir, max_t)` — block ray query against BVH.
- `tile_cholesky(A)` / `tile_cholesky_inplace(A)` — Cholesky factorization.
- `tile_cholesky_solve(L, y)` / `_inplace` — solve `Ax=y` from Cholesky factor.
- `tile_diag_add(mat, d)` — add square matrix and diagonal `d` (1D tile).
- `tile_dot(a, b)` — dot product of two tiles.
- `tile_empty(shape, dtype)` — uninitialized tile.
- `tile_extract(t, indices)` — extract a single element.
- `tile_fft(t)` / `tile_ifft(t)` — forward/inverse FFT along last dim.
- `tile_from_thread(value, shape, dtype)` — tile filled with value from a thread.
- `tile_full(value, shape, dtype)` — tile filled with value.
- `tile_load(arr, offset)` — load a tile from global memory.
- `tile_load_indexed(arr, axis, indices)` — load tile mapped by index tile.
- `tile_lower_solve(L, y)` / `_inplace` — solve `Lz=y` (lower triangular).
- `tile_map(func, t)` — apply a function to tile elements.
- `tile_matmul(a, b)` — matrix product `a*b`.
- `tile_max(t)` / `tile_min(t)` — cooperative max/min of tile elements.
- `tile_mesh_query_aabb(mesh, lower, upper)` / `_next` — block mesh AABB query.
- `tile_ones(shape, dtype)` / `tile_zeros(shape, dtype)` — one/zero-initialized tile.
- `tile_query_valid(query)` — whether block BVH query has remaining results.
- `tile_randf(shape, dtype)` / `tile_randi(shape, dtype)` — tile of randoms.
- `tile_reduce(func, t)` — custom reduction across a tile.
- `tile_reshape(t, shape)` — reshaped view.
- `tile_scan_exclusive(t, func)` / `tile_scan_inclusive(t, func)` — prefix sums.
- `tile_scan_max_inclusive(t)` / `tile_scan_min_inclusive(t)` — inclusive max/min scan.
- `tile_scatter_add(t, indices)` — scatter-add per-thread value into shared tile.
- `tile_scatter_masked(t, indices, value)` — write value into shared tile.
- `tile_sort(keys, values)` — cooperative sort by keys.
- `tile_squeeze(t)` — squeezed view.
- `tile_stack(dtype, capacity)` — block stack in shared memory.
- `tile_stack_clear/count/pop/push(...)` — tile-stack operations.
- `tile_store(arr, offset, t)` / `tile_store_indexed(...)` — store a tile.
- `tile_sum(t)` — cooperative sum of tile elements.
- `tile_transpose(t)` — transpose a tile.
- `tile_upper_solve(U, z)` / `_inplace` — solve `Ux=z` (upper triangular).
- `tile_view(t, offset, shape)` — slice [offset, offset+shape].
- `untile(t)` — convert a tile back to per-thread values.

## Geometry
- `bvh_get_group_root(bvh, group)` — root of a group in a BVH.
- `bvh_query_aabb(bvh, lower, upper)` / `_tiled` — AABB query against BVH.
- `bvh_query_next(query)` / `_tiled` — next bound returned by query.
- `bvh_query_ray(bvh, start, dir, max_t)` / `_tiled` — ray query against BVH.
- `closest_point_edge_edge(p1, q1, p2, q2, epsilon)` — closest points between two edges. Takes the
  four **endpoints** (not directions) plus a degeneracy tolerance, and returns
  `vec3(s, t, d)`: the barycentric weight along each edge and the distance between the closest
  points. `vec3`/`float32` only — there is no float64 overload.
- `hash_grid_point_id(grid, index)` — index of a point in the HashGrid.
- `hash_grid_query(grid, point)` / `hash_grid_query_next(query)` — HashGrid point query.
- `intersect_tri_tri(v0,v1,v2,u0,u1,u2)` — triangle/triangle intersection (Möller).
- `mesh_eval_face_normal(mesh, face_index)` — face normal.
- `mesh_eval_position(mesh, face_index, bary)` — position from face + barycentrics.
- `mesh_eval_velocity(mesh, face_index, bary)` — velocity from face + barycentrics.
- `mesh_get(id)` — retrieve mesh by index.
- `mesh_get_group_root(mesh, group)` — root of a group in a Mesh.
- `mesh_get_index(mesh, fv_index)` — point index from face-vertex index.
- `mesh_get_point(mesh, index)` — point of the mesh by index.
- `mesh_get_velocity(mesh, index)` — velocity of the mesh by index.
- `mesh_query_aabb(mesh, lower, upper)` / `_next` / `_tiled` / `_next_tiled` — AABB query.
- `mesh_query_furthest_point_no_sign(id, point)` — furthest point on mesh.
- `mesh_query_point(id, point)` — closest point on mesh.
- `mesh_query_point_no_sign(id, point)` — closest point, no sign.
- `mesh_query_point_sign_normal(id, point)` — closest point, sign via normal.
- `mesh_query_point_sign_parity(id, point)` — closest point, sign via parity.
- `mesh_query_point_sign_winding_number(id, point)` — closest point, sign via winding number.
- `mesh_query_ray(id, start, dir, max_t)` — closest ray hit (< max_t).
- `mesh_query_ray_anyhit(id, start, dir, max_t)` — any ray hit.
- `mesh_query_ray_count_intersections(id, start, dir, max_t)` — count ray/mesh intersections.

## Volumes
- `volume_index_to_world(volume, uvw)` / `_dir` — index space -> world space (point/dir).
- `volume_lookup(volume, i, j, k)` and typed `_f` / `_i` / `_v` / `_index` — query a voxel value.
- `volume_sample(volume, uvw)` and typed `_f` / `_i` / `_v` / `_index` — sample volume.
- `volume_sample_grad(volume, uvw)` and `_f` / `_index` — sample value + gradient.
- `volume_store(volume, i, j, k, value)` and `_f` / `_i` / `_v` — store a voxel value.
- `volume_world_to_index(volume, xyz)` / `_dir` — world space -> index space (point/dir).

## Textures
- `texture_sample(texture, u)` — sample 1D texture at U coordinate.

## Random
- `curlnoise(x)` — divergence-free vector field from Perlin noise.
- `noise(x)` — non-periodic Perlin-style noise.
- `pnoise(x)` — periodic Perlin-style noise.
- `poisson(lambda)` — sample from a Poisson distribution.
- `rand_init(seed)` — initialize an RNG state.
- `randf()` / `randi(max)` / `randu(max)` — random float / int / unsigned int.
- `randn()` — sample N(0,1).
- `sample_cdf(cdf, u)` — inverse-transform sample a CDF.
- `sample_triangle(u, v)` — uniformly sample a triangle.
- `sample_unit_cube(u,v,w)` / `sample_unit_sphere(u,v,w)` — sample unit volume.
- `sample_unit_disk(u,v)` / `sample_unit_ring(u,v)` / `sample_unit_square(u,v)` — sample unit 2D region.
- `sample_unit_hemisphere(u,v)` / `_surface` — sample unit hemisphere (volume/surface).
- `sample_unit_sphere_surface(u,v)` — sample unit sphere surface.

## Utility
- `array(ptr, shape, dtype)` — construct an array from pointer/shape/dtype.
- `atomic_add/sub/min/max/and/or/xor(arr, i, value)` — atomic ops returning old value.
- `atomic_cas(arr, i, compare, value)` — atomic compare-and-swap.
- `atomic_exch(arr, i, value)` — atomic exchange.
- `block_dim()` — number of threads in the current block.
- `breakpoint()` — trigger a debugger breakpoint.
- `cast(x, dtype)` — reinterpret value as a different type (bit-preserving).
- `expect_near(a, b, tolerance)` — print error if `a`/`b` differ by > tolerance.
- `len(a)` — length of `a`.
- `lerp(a, b, t)` — linear interpolation `a*(1-t)+b*t`.
- `lower_bound(arr, value)` / `lower_bound(arr, begin, end, value)` — index of the first element
  >= `value` in sorted `arr` (optionally within `[begin, end)`). **Clamped to `end - 1`**: a `value`
  past the last element returns the last index, *not* `n` like `numpy.searchsorted`.
- `print(x)` / `printf(fmt, ...)` — print to stdout.
- `select(cond, if_true, if_false)` / `where(cond, if_true, if_false)` — branchless select.
- `smoothstep(a, b, x)` — cubic Hermite interpolation.
- `tid()` — current thread index/indices (kernel scope only).
- `zeros(shape, dtype)` — zero-initialized fixed-size (stack) array.

## Operators
- `add/sub/mul/div(a, b)` — arithmetic.
- `div_approx(a, b)` — division via approximate GPU intrinsics.
- `floordiv(a, b)` — floor division.
- `mod(a, b)` — modulo (C++11 truncated division — sign of dividend).
- `neg(x)` / `pos(x)` — negate / pass through.
- `bit_and/bit_or/bit_xor(a, b)` — bitwise ops.
- `invert(a)` — bitwise complement.
- `lshift(a, b)` / `rshift(a, b)` — bit shifts.
- `unot(a)` — logical NOT.

## Code Generation
- `static(expr)` — evaluate a static Python expression and inline its result.
