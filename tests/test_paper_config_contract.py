from phystta.config.loader import load_config


def test_representation_surgery_2026_paper_contract():
    cfg = load_config("configs/representation_surgery_2026.yaml")
    assert cfg.experiment.tier == "paper"
    assert cfg.runtime.freeze_artifacts is True
    assert cfg.runtime.allow_download is False

    # manual grids must be disabled under realism in paper-tier
    assert cfg.intervention.use_realism is True
    assert cfg.intervention.intensity_grid in ([], [0.0])
    assert cfg.erasure_eval.intensity_grid in ([], [0.0])
    assert cfg.data.split_group_key == "global_patient_id"
    assert cfg.data.prebuilt_index_csv is not None
    assert set(cfg.erasure_eval.modes or []) >= {"clinical", "causal_invariance", "safety", "fairness"}
    assert len(cfg.erasure_eval.fairness_groups) > 0

    assert cfg._is_immutable_revision(cfg.model.encoder.revision)

    prof_map = {p.name: p.encoder for p in (cfg.model.encoder_profiles or [])}
    for pname in cfg.architecture_study.encoder_profiles or []:
        enc = prof_map[pname]
        if enc.encoder_source.startswith("hf_") and enc.model_id:
            assert cfg._is_immutable_revision(enc.revision), f"profile={pname} has non-immutable revision"
        if enc.encoder_source == "baseline_cnn":
            assert enc.encoder_path is not None

    # Architecture matrix coverage for dissertation-grade comparative analysis:
    # at least 2 CNN, 2 ViT, and 2 foundation-style profiles.
    tags = cfg.architecture_study.profile_tags or {}
    selected = cfg.architecture_study.encoder_profiles or []
    paradigms = [str(tags.get(p, {}).get("paradigm", "")) for p in selected]
    assert sum(p == "CNN" for p in paradigms) >= 2
    assert sum(p == "ViT" for p in paradigms) >= 2
    assert sum("Foundation" in p or "CLIP" in p for p in paradigms) >= 2
