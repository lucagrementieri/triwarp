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
