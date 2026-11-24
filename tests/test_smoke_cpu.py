import os
import pytest

from phystta.config.loader import load_config
from phystta.data import DataEngine


def test_smoke_cpu():
    cfg = load_config("configs/representation_surgery.yaml")
    # skip if dataset root is not available
    if not cfg.data.sources or any(not s.root.exists() for s in cfg.data.sources):
        pytest.skip("dataset roots not available")
    if any(s.metadata_csv is None for s in cfg.data.sources):
        pytest.skip("metadata_csv not configured")
    engine = DataEngine(cfg.data)
    engine.build_index()
    engine.create_splits()
    loaders = engine.build_dataloaders(batch_size=2, splits=[cfg.data.eval_split])
    batch = next(iter(loaders[cfg.data.eval_split]))
    assert "image" in batch and "labels" in batch
