import time

import numpy as np
import warp as wp

import triwarp.unique as tu

wp.init()
rng = np.random.default_rng(42)

for device in ["cpu", "cuda:0"]:
    print(f"\n=== {device} ===")
    for n_unique_frac in [0.001, 0.01, 0.1, 0.5, 1.0]:
        n = 1_000_000
        n_unique = max(1, int(n * n_unique_frac))
        for dtype in [np.int32, np.int64]:
            data_np = rng.choice(n_unique, size=n).astype(dtype)
            data_wp = wp.array(data_np, dtype=getattr(wp, dtype.__name__), device=device)
            # warm up (also triggers JIT)
            tu.unique_1d(data_wp, return_inverse=True, return_counts=True)
            wp.synchronize_device(device)
            N = 20
            t0 = time.perf_counter()
            for _ in range(N):
                u, inv, cnt = tu.unique_1d(data_wp, return_inverse=True, return_counts=True)
            wp.synchronize_device(device)
            ms = (time.perf_counter() - t0) / N * 1000
            print(
                f"  {dtype.__name__:6s}  n={n}  n_unique={n_unique:7d} ({n_unique_frac * 100:5.1f}%)  {ms:7.2f} ms"
            )
