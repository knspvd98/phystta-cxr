import textwrap

import pytest

from phystta.config.loader import load_config


def test_architecture_study_loaded_from_typed_schema(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              sources:
                - name: dummy
                  root: /tmp
                  source_type: chexpert
              split_fractions:
                train_head: 0.5
                train_eraser: 0.2
                calib: 0.1
                eval: 0.2
            model:
              encoder_profiles:
                - name: p1
                  encoder:
                    encoder_source: baseline_cnn
                    architecture: densenet121
            architecture_study:
              encoder_profiles: [p1]
              interventions: [erasure]
              interaction_pairs:
                - [no_eraser, eraser]
              reporting:
                clean_intensity: 0.0
            """
        )
    )
    cfg = load_config(str(cfg_path))
    assert cfg.architecture_study.encoder_profiles == ["p1"]
    assert cfg.architecture_study.interventions == ["erasure"]
    assert cfg.architecture_study.reporting.clean_intensity == 0.0


def test_paper_tier_rejects_missing_timm_path_when_selected(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              sources:
                - name: dummy
                  root: /tmp
                  source_type: chexpert
              split_fractions:
                train_head: 0.5
                train_eraser: 0.2
                calib: 0.1
                eval: 0.2
              prebuilt_index_csv: /tmp/prebuilt_index.csv
              prebuilt_index_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
              split_group_key: global_patient_id
              head_split: train_head
              eraser_split: train_eraser
              calib_split: calib
              eval_split: eval
            model:
              encoder:
                encoder_source: hf_clip_or_vit
                model_id: openai/clip-vit-base-patch32
                revision: 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
              encoder_profiles:
                - name: timm_vit
                  encoder:
                    encoder_source: timm_vit
                    architecture: vit_base_patch16_224
                    encoder_path: /tmp/missing_timm_vit.pth
            runtime:
              freeze_artifacts: true
              allow_download: false
            shift_id:
              calibration: platt
            intervention:
              use_realism: true
              realism_path: outputs/representation_surgery/realism_calibrator.json
              intensity_grid: [0.0]
              group_key: global_patient_id
            erasure_eval:
              intensity_grid: [0.0]
              modes: [clinical, causal_invariance, safety, fairness]
            experiment:
              tier: paper
            architecture_study:
              encoder_profiles: [timm_vit]
            """
        )
    )
    with pytest.raises(ValueError, match="encoder_path does not exist"):
        load_config(str(cfg_path))
