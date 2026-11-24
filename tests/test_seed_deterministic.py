import os

import torch

from phystta.utils.seed import set_global_seed


def test_set_global_seed_enables_deterministic_mode_and_env():
    set_global_seed(123, deterministic=True, benchmark=False)
    assert os.environ.get("PYTHONHASHSEED") == "123"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}
    assert torch.are_deterministic_algorithms_enabled()


def test_set_global_seed_reproducible_cpu_tensor():
    set_global_seed(777, deterministic=True, benchmark=False)
    a = torch.randn(4, 4)
    set_global_seed(777, deterministic=True, benchmark=False)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)
