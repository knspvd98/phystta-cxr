from __future__ import annotations

import torch

from phystta.intervention.eraser import LinearSubspaceEraser, TaskPreservingEraser
from scripts.fit_eraser import (
    _resolve_task_preserve_strength,
    _wrap_task_preserving_eraser,
)


def test_resolve_task_preserve_strength_defaults():
    assert abs(_resolve_task_preserve_strength("head_subspace_soft", None) - 0.75) < 1e-12
    assert abs(_resolve_task_preserve_strength("head_subspace", None) - 1.0) < 1e-12
    assert abs(_resolve_task_preserve_strength("residualize", None) - 1.0) < 1e-12


def test_resolve_task_preserve_strength_respects_explicit_value():
    assert abs(_resolve_task_preserve_strength("head_subspace_soft", 0.6) - 0.6) < 1e-12
    assert abs(_resolve_task_preserve_strength("head_subspace", 0.9) - 0.9) < 1e-12


def test_resolve_task_preserve_strength_rejects_out_of_range():
    for bad in (-0.01, 1.01):
        try:
            _resolve_task_preserve_strength("head_subspace_soft", bad)
        except ValueError as exc:
            assert "preserve_strength" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid preserve_strength")


def test_wrap_task_preserving_eraser_soft_sets_mode_meta_and_strength():
    base = LinearSubspaceEraser(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    meta = {"shift_type": "heel", "eraser_type": "steering"}

    wrapped = _wrap_task_preserving_eraser(
        base,
        basis,
        task_preserve="head_subspace_soft",
        preserve_strength=None,
        meta=meta,
    )

    assert isinstance(wrapped, TaskPreservingEraser)
    assert wrapped.mode == "head_subspace_soft"
    assert abs(float(wrapped.preserve_strength) - 0.75) < 1e-12
    assert wrapped.meta["task_preserve"] == "head_subspace_soft"
    assert abs(float(wrapped.meta["preserve_strength"]) - 0.75) < 1e-12
    assert wrapped.meta["shift_type"] == "heel"


def test_wrap_task_preserving_eraser_hard_mode_keeps_strength_one():
    base = LinearSubspaceEraser(torch.tensor([[1.0], [0.0]], dtype=torch.float32))
    basis = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    wrapped = _wrap_task_preserving_eraser(
        base,
        basis,
        task_preserve="head_subspace",
        preserve_strength=None,
        meta={},
    )

    assert isinstance(wrapped, TaskPreservingEraser)
    assert wrapped.mode == "head_subspace"
    assert abs(float(wrapped.preserve_strength) - 1.0) < 1e-12
    assert wrapped.meta["task_preserve"] == "head_subspace"
    assert abs(float(wrapped.meta["preserve_strength"]) - 1.0) < 1e-12
