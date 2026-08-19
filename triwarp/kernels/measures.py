import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.triangles import face_normals_and_area, face_vertices_vec3d


@wp.kernel
def centroid_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    out_weighted_centroid: wp.array[wp.float32],
    out_total_area: wp.array[wp.float32],
) -> None:
    # CUDA path, launched via wp.launch_tiled with block_dim=TILE_1D: each lane computes one face's
    # area-weighted centroid contribution, the block reduces each component cooperatively with
    # wp.tile_sum, and lane 0 commits four atomics per block (out-of-range lanes contribute zero).
    # CPU MUST NOT use this: `wp.launch_tiled` runs one lane per block there, so the block reduction
    # would see one face per tile. See `centroid_sliced` and `_device.prefers_tiled_reduction`.
    i, t = wp.tid()
    f = i * TILE_1D + t
    contrib = wp.vec3(0.0, 0.0, 0.0)
    area = wp.float32(0.0)
    if f < n_faces:
        triangle_face = faces[f * 3 : (f + 1) * 3]
        _, area = face_normals_and_area(vertices, triangle_face)
        contrib = (
            vertices[triangle_face[0]] + vertices[triangle_face[1]] + vertices[triangle_face[2]]
        ) * (area / 3.0)
    sum_x = wp.tile_sum(wp.tile(contrib[0]))
    sum_y = wp.tile_sum(wp.tile(contrib[1]))
    sum_z = wp.tile_sum(wp.tile(contrib[2]))
    area_sum = wp.tile_sum(wp.tile(area))
    if t == 0:
        wp.tile_atomic_add(out_weighted_centroid, sum_x, (0,))
        wp.tile_atomic_add(out_weighted_centroid, sum_y, (1,))
        wp.tile_atomic_add(out_weighted_centroid, sum_z, (2,))
        wp.tile_atomic_add(out_total_area, area_sum, (0,))


@wp.kernel
def centroid_sliced(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    n_slices: wp.int32,
    out_weighted_centroid: wp.array[wp.float32],
    out_total_area: wp.array[wp.float32],
) -> None:
    # Portable path: one thread per face slice walks a strided slice, accumulates locally and
    # commits four atomics. Lane-free, so it is correct on the CPU device where `centroid_tiled`
    # is not; it gives up the block shuffle-reduce and measures 1.67x slower on CUDA at 327k faces
    # (16.0 -> 26.7 us), which is why both exist. That bug was invisible on a symmetric mesh, whose
    # every-Nth-face centroid is still the true centroid.
    j = wp.tid()
    total = wp.vec3(0.0, 0.0, 0.0)
    area_total = wp.float32(0.0)
    for f in range(j, n_faces, n_slices):
        triangle_face = faces[f * 3 : (f + 1) * 3]
        _, area = face_normals_and_area(vertices, triangle_face)
        total = total + (
            vertices[triangle_face[0]] + vertices[triangle_face[1]] + vertices[triangle_face[2]]
        ) * (area / 3.0)
        area_total = area_total + area
    wp.atomic_add(out_weighted_centroid, 0, total[0])
    wp.atomic_add(out_weighted_centroid, 1, total[1])
    wp.atomic_add(out_weighted_centroid, 2, total[2])
    wp.atomic_add(out_total_area, 0, area_total)


@wp.kernel
def moment_integrands(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_volume: wp.array[wp.float64],
    out_first: wp.array[wp.vec3d],
    out_squares: wp.array[wp.vec3d],
    out_products: wp.array[wp.vec3d],
) -> None:
    # Per-face contribution to the mass integrals of the solid bounded by the mesh, taking each
    # face with the origin as a tetrahedron. Everything accumulates in float64: the second moments
    # scale as length^5, so a float32 sum over a large mesh loses the answer's low digits well
    # before the reduction finishes.
    #
    # For the tetrahedron (0, a, b, c) with det = dot(a, cross(b, c)):
    #   ∫dV      = det / 6
    #   ∫x dV    = det * (a.x + b.x + c.x) / 24
    #   ∫x^2 dV  = det * (a.x^2 + b.x^2 + c.x^2 + a.x b.x + a.x c.x + b.x c.x) / 60
    #   ∫xy dV   = det * (2(a.x a.y + b.x b.y + c.x c.y)
    #                     + a.x b.y + b.x a.y + a.x c.y + c.x a.y + b.x c.y + c.x b.y) / 120
    f = wp.int32(wp.tid())
    a, b, c = face_vertices_vec3d(vertices, faces, f)
    det = wp.dot(a, wp.cross(b, c))

    out_volume[f] = det / wp.float64(6.0)
    out_first[f] = det * (a + b + c) / wp.float64(24.0)
    out_squares[f] = (
        det
        * wp.vec3d(
            a[0] * a[0] + b[0] * b[0] + c[0] * c[0] + a[0] * b[0] + a[0] * c[0] + b[0] * c[0],
            a[1] * a[1] + b[1] * b[1] + c[1] * c[1] + a[1] * b[1] + a[1] * c[1] + b[1] * c[1],
            a[2] * a[2] + b[2] * b[2] + c[2] * c[2] + a[2] * b[2] + a[2] * c[2] + b[2] * c[2],
        )
        / wp.float64(60.0)
    )
    # (xy, xz, yz), in the same order the wrapper reads them back.
    out_products[f] = (
        det
        * wp.vec3d(
            wp.float64(2.0) * (a[0] * a[1] + b[0] * b[1] + c[0] * c[1])
            + a[0] * b[1]
            + b[0] * a[1]
            + a[0] * c[1]
            + c[0] * a[1]
            + b[0] * c[1]
            + c[0] * b[1],
            wp.float64(2.0) * (a[0] * a[2] + b[0] * b[2] + c[0] * c[2])
            + a[0] * b[2]
            + b[0] * a[2]
            + a[0] * c[2]
            + c[0] * a[2]
            + b[0] * c[2]
            + c[0] * b[2],
            wp.float64(2.0) * (a[1] * a[2] + b[1] * b[2] + c[1] * c[2])
            + a[1] * b[2]
            + b[1] * a[2]
            + a[1] * c[2]
            + c[1] * a[2]
            + b[1] * c[2]
            + c[1] * b[2],
        )
        / wp.float64(120.0)
    )
