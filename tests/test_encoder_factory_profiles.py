import pytest
import torch.nn as nn

from phystta.config.loader import load_config
from phystta.models import encoder_factory as ef
from phystta.models.encoder_factory import EncoderFactory


def test_baseline_profile_architecture_is_used(monkeypatch):
    cfg = load_config("configs/representation_surgery_2026.yaml")
    # cfg.model.architecture is clip-vit..., profile architecture is densenet121
    assert cfg.model.architecture != "densenet121"
    for p in cfg.model.encoder_profiles or []:
        if p.name == "baseline_cnn":
            p.encoder.encoder_path = None
            p.encoder.pretrained = False

    captured = {}

    class _Dummy(nn.Module):
        def forward(self, x):
            return x

    def _fake_cnn(architecture, num_classes, in_chans=1, pretrained=False, dropout=0.0, use_medical_pretrain=False):
        captured["architecture"] = architecture
        captured["pretrained"] = pretrained
        captured["use_medical_pretrain"] = use_medical_pretrain
        return _Dummy()

    monkeypatch.setattr(ef, "CNNClassifier", _fake_cnn)
    EncoderFactory.from_config(cfg, profile="baseline_cnn")

    assert captured["architecture"] == "densenet121"


def test_baseline_offline_requires_path_or_pretrained_false():
    cfg = load_config("configs/representation_surgery.yaml")
    cfg.runtime.allow_download = False
    cfg.model.encoder.encoder_source = "baseline_cnn"
    cfg.model.encoder.architecture = "densenet121"
    if cfg.model.encoder.input_adapter is not None:
        cfg.model.encoder.input_adapter.repeat_to_3 = False
    cfg.model.encoder.encoder_path = None
    cfg.model.encoder.pretrained = True

    with pytest.raises(RuntimeError, match="download is disabled"):
        EncoderFactory.from_config(cfg)
