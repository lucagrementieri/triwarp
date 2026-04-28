import pytest
import warp as wp


@pytest.fixture
def device():
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"
