from __future__ import annotations

from pathlib import Path

import torch

from phystta.intervention.eraser import (
    LinearSubspaceEraser,
    TaskPreservingEraser,
    load_eraser,
    save_eraser,
)


def test_head_subspace_soft_interpolates_between_hard_and_raw():
    # Base eraser removes the first feature (classifier-visible direction).
    base = LinearSubspaceEraser(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    task_basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    hard = TaskPreservingEraser(
        base,
        task_basis,
        mode="head_subspace",
        preserve_strength=1.0,
    )
    soft = TaskPreservingEraser(
        base,
        task_basis,
        mode="head_subspace_soft",
        preserve_strength=0.75,
    )

    x = torch.tensor(
        [
            [2.0, 0.0],  # fully task-axis
            [0.0, 3.0],  # orthogonal to task axis
            [2.0, 3.0],  # mixed
        ],
        dtype=torch.float32,
    )

    raw = base(x)
    y_hard = hard(x)
    y_soft = soft(x)

    expected = 0.75 * y_hard + 0.25 * raw
    assert torch.allclose(y_soft, expected, atol=1e-6)

    # On the task axis, hard preserve keeps the feature, raw eraser removes it,
    # and soft preserve must lie strictly between them.
    assert torch.allclose(y_hard[0], x[0], atol=1e-6)
    assert torch.allclose(raw[0], torch.tensor([0.0, 0.0]), atol=1e-6)
    assert 0.0 < float(y_soft[0, 0]) < float(y_hard[0, 0])

    # When preserve_strength=1, soft mode should match hard mode exactly.
    soft_full = TaskPreservingEraser(
        base,
        task_basis,
        mode="head_subspace_soft",
        preserve_strength=1.0,
    )
    assert torch.allclose(soft_full(x), y_hard, atol=1e-6)


def test_task_preserving_eraser_roundtrip_preserve_strength(tmp_path: Path):
    base = LinearSubspaceEraser(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    task_basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    eraser = TaskPreservingEraser(
        base,
        task_basis,
        mode="head_subspace_soft",
        preserve_strength=0.75,
        meta={"note": "soft-test"},
    )

    path = tmp_path / "soft_task_preserve.pt"
    save_eraser(str(path), eraser)
    loaded = load_eraser(str(path))

    assert isinstance(loaded, TaskPreservingEraser)
    assert loaded.mode == "head_subspace_soft"
    assert abs(float(loaded.preserve_strength) - 0.75) < 1e-12
    assert loaded.meta["task_preserve"] == "head_subspace_soft"
    assert abs(float(loaded.meta["preserve_strength"]) - 0.75) < 1e-12

    x = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(loaded(x), eraser(x), atol=1e-6)


def test_task_preserving_eraser_rejects_invalid_preserve_strength():
    base = LinearSubspaceEraser(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    task_basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    for bad in (-0.1, 1.1):
        try:
            TaskPreservingEraser(
                base,
                task_basis,
                mode="head_subspace_soft",
                preserve_strength=bad,
            )
        except ValueError as exc:
            assert "preserve_strength" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid preserve_strength")
