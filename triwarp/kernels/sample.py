import warp as wp


@wp.kernel
def sample_surface(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cdf: wp.array[wp.float32],
    seed: int,
    out_points: wp.array[wp.vec3],
    out_face_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    fi = int(wp.sample_cdf(state, cdf))

    v0 = vertices[faces[fi * 3]]
    v1 = vertices[faces[fi * 3 + 1]]
    v2 = vertices[faces[fi * 3 + 2]]

    uv = wp.sample_triangle(state)
    w = 1.0 - uv.x - uv.y

    out_face_indices[tid] = fi
    out_points[tid] = v0 * w + v1 * uv.x + v2 * uv.y


@wp.kernel
def signed_tet_volumes(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.vec3,
    out_volumes: wp.array[wp.float32],
) -> None:
    fi = int(wp.tid())
    v0 = vertices[faces[fi * 3]] - center
    v1 = vertices[faces[fi * 3 + 1]] - center
    v2 = vertices[faces[fi * 3 + 2]] - center
    out_volumes[fi] = wp.dot(v0, wp.cross(v1, v2)) / 6.0


@wp.kernel
def sample_volume_tet(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.vec3,
    cdf: wp.array[wp.float32],
    seed: int,
    out_points: wp.array[wp.vec3],
) -> None:
    tid = int(wp.tid())
    state = wp.rand_init(seed, tid)
    fi = int(wp.sample_cdf(state, cdf))

    v0 = vertices[faces[fi * 3]]
    v1 = vertices[faces[fi * 3 + 1]]
    v2 = vertices[faces[fi * 3 + 2]]

    # Uniform sampling in tet (center, v0, v1, v2) via order statistics of
    # 3 U(0,1) samples. Sorted values s1 ≤ s2 ≤ s3 give spacings
    # (s1, s2-s1, s3-s2, 1-s3) as barycentric coords for (center, v0, v1, v2):
    # P = center*(1-s3) + v0*s1 + v1*(s2-s1) + v2*(s3-s2)
    a = wp.randf(state)
    b = wp.randf(state)
    c = wp.randf(state)

    s1 = wp.min(a, wp.min(b, c))
    s3 = wp.max(a, wp.max(b, c))
    s2 = a + b + c - s1 - s3

    out_points[tid] = center * (1.0 - s3) + v0 * s1 + v1 * (s2 - s1) + v2 * (s3 - s2)
