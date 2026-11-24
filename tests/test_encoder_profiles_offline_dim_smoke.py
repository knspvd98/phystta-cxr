import textwrap
from pathlib import Path

import torch

from phystta.config.loader import load_config
from phystta.models.encoder_factory import EncoderFactory


def _save_local_encoder(path: Path) -> None:
    model = torch.nn.Module()
    model.backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
    torch.save(model, path)


def test_offline_encoder_profiles_same_embedding_dim(tmp_path):
    ckpt_a = tmp_path / "enc_a.pt"
    ckpt_b = tmp_path / "enc_b.pt"
    ckpt_c = tmp_path / "enc_c.pt"
    _save_local_encoder(ckpt_a)
    _save_local_encoder(ckpt_b)
    _save_local_encoder(ckpt_c)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            f"""
            data:
              require_real_data: false
              preprocess:
                image_size: 32
                normalize_mean: 0.0
                normalize_std: 1.0
              sources:
                - name: dummy
                  root: /tmp
            runtime:
              allow_download: false
            model:
              num_classes: 1
              encoder:
                encoder_source: eva_x
                encoder_path: {ckpt_a}
                input_adapter:
                  repeat_to_3: false
                  image_size: 32
                  mean: [0.0]
                  std: [1.0]
              encoder_profiles:
                - name: p1
                  encoder:
                    encoder_source: eva_x
                    encoder_path: {ckpt_a}
                    input_adapter:
                      repeat_to_3: false
                      image_size: 32
                      mean: [0.0]
                      std: [1.0]
                - name: p2
                  encoder:
                    encoder_source: eva_x
                    encoder_path: {ckpt_b}
                    input_adapter:
                      repeat_to_3: false
                      image_size: 32
                      mean: [0.0]
                      std: [1.0]
                - name: p3
                  encoder:
                    encoder_source: eva_x
                    encoder_path: {ckpt_c}
                    input_adapter:
                      repeat_to_3: false
                      image_size: 32
                      mean: [0.0]
                      std: [1.0]
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))
    x = torch.rand(2, 1, 32, 32)
    dims = []
    for profile in ["p1", "p2", "p3"]:
        enc = EncoderFactory.from_config(cfg, profile=profile)
        with torch.no_grad():
            z = enc.encode(x)
        dims.append(int(z.shape[1]))
    assert len(set(dims)) == 1
