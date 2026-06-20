import numpy as np
import trimesh.points as tm
import warp as wp

import triwarp.points as tw


def test_point_plane_distance(device: str) -> None:
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((50, 3))
    plane_normal_np = rng.standard_normal(3)
    plane_origin_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np, plane_origin_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distances_wp = tw.point_plane_distance(
        points_wp, wp.vec3(*plane_normal_np.tolist()), wp.vec3(*plane_origin_np.tolist())
    )

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)


def test_point_plane_distance_no_origin(device: str) -> None:
    rng = np.random.default_rng(1)
    points_np = rng.standard_normal((30, 3))
    plane_normal_np = rng.standard_normal(3)

    distances_tm = tm.point_plane_distance(points_np, plane_normal_np)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distances_wp = tw.point_plane_distance(points_wp, wp.vec3(*plane_normal_np.tolist()))

    assert np.allclose(distances_wp.numpy(), distances_tm, rtol=1e-5, atol=1e-5)
