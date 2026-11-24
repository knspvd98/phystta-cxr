import sys
import types

import pytest

from phystta.config.loader import load_config
from phystta.models.encoder_factory import EncoderFactory


def test_hf_encoder_offline_requires_local_cache(monkeypatch):
    cfg = load_config("configs/representation_surgery.yaml")
    cfg.runtime.allow_download = False
    cfg.model.encoder.encoder_source = "hf_clip_or_vit"
    cfg.model.encoder.encoder_path = None
    cfg.model.encoder.model_id = "nonexistent/offline-model"
    cfg.model.encoder.revision = "0123456789abcdef0123456789abcdef01234567"

    hub = types.ModuleType("huggingface_hub")

    def _snapshot_download(*args, **kwargs):
        raise OSError("not found in local cache")

    hub.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(RuntimeError, match="Download is disabled"):
        EncoderFactory.from_config(cfg)


def test_openclip_offline_requires_local_weights(monkeypatch):
    cfg = load_config("configs/representation_surgery.yaml")
    cfg.runtime.allow_download = False
    cfg.model.encoder.encoder_source = "openclip"
    cfg.model.encoder.architecture = "ViT-B-32"
    cfg.model.encoder.encoder_path = None
    cfg.model.encoder.pretrained = True
    cfg.model.encoder.model_id = "openai"

    # Provide a stub module so import succeeds before the offline guard check.
    open_clip_stub = types.ModuleType("open_clip")
    open_clip_stub.create_model = lambda *args, **kwargs: None
    open_clip_stub.create_model_and_transforms = lambda *args, **kwargs: (None, None, None)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip_stub)

    with pytest.raises(RuntimeError, match="openclip download is disabled"):
        EncoderFactory.from_config(cfg)
