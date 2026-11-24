import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by CUDA deterministic GEMM kernels.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark and not deterministic)
    # Keep numerical semantics stable across platforms where possible.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        # Some environments/operators do not support strict deterministic mode.
        # In that case we keep seeded behavior with cudnn deterministic flags.
        pass


def seed_worker(worker_id: int, base_seed: Optional[int] = None) -> None:
    if base_seed is None:
        base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
