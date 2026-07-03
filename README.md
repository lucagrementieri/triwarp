# triwarp

Triangular mesh library based on [NVIDIA Warp](https://github.com/NVIDIA/warp), providing
GPU-accelerated mesh geometry, connectivity, and processing kernels with a
[trimesh](https://trimesh.org)-inspired API.

## Install

```bash
uv add triwarp
```

or with `pip`:

```bash
pip install triwarp
```

## Quickstart

```python
import numpy as np
import warp as wp

import triwarp as tw

vertices = wp.array(
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
    dtype=wp.vec3,
)
faces = wp.array(np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32)

edges = tw.edges.faces_to_edges(faces)
print(edges.numpy())
```

## Documentation

Full API reference: <https://lucagrementieri.github.io/triwarp/>

## Development

```bash
uv sync --extra dev --extra test --extra docs
uv run pytest
uv run mkdocs serve
```
