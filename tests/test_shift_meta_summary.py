import torch

from phystta.shifts import ShiftManager


def test_batch_meta_summary_consistent():
    mgr = ShiftManager(seed=0)
    x = torch.rand(2, 1, 32, 32)
    intensities = {"poisson": 1.0}
    _, meta = mgr.apply_batch_scenario(x, intensities)
    per = meta.get("per_sample", [])
    summary = meta.get("summary", {})
    destructive = sum(1 for m in per if m.get("destructive", False))
    assert len(per) == x.shape[0]
    assert summary.get("destructive_count") == destructive
