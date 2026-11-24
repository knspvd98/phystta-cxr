from phystta.config.loader import load_config


def test_representation_surgery_2026_paper_variant_contract():
    cfg = load_config("configs/representation_surgery_2026_paper.yaml")
    assert cfg.experiment.tier == "paper"
    assert cfg.runtime.allow_download is False
    assert cfg.runtime.device == "cuda"
    assert cfg.data.prebuilt_index_csv is not None
    sha = str(cfg.data.prebuilt_index_sha256 or "")
    assert len(sha) == 64
    assert set(sha) != {"0"}
    assert cfg.vlm.model_path is None
