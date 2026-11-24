import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> Tuple[int, int]:
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
