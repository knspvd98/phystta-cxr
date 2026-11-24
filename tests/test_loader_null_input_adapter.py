import textwrap

from phystta.config.loader import load_config


def test_loader_accepts_null_input_adapter(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              sources:
                - name: mini
                  root: /tmp
                  label_cols: [No Finding]
            model:
              encoder:
                encoder_source: hf_clip_or_vit
                model_id: openai/clip-vit-base-patch32
                input_adapter: null
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.model.encoder.input_adapter is None
