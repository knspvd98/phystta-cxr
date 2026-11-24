import numpy as np
import pandas as pd

from phystta.studies import (
    infer_paradigm,
    infer_pretraining,
    build_interaction_table,
    summarize_architecture_matrix,
    build_robustness_table,
)


def _sample_metrics_df() -> pd.DataFrame:
    rows = [
        # clean
        {"profile": "cnn", "paradigm": "CNN", "pretraining": "ImageNet", "shift": "heel", "intensity": 0.0, "method": "no_eraser", "auroc_macro": 0.80, "ece": 0.08},
        {"profile": "cnn", "paradigm": "CNN", "pretraining": "ImageNet", "shift": "heel", "intensity": 0.0, "method": "eraser", "auroc_macro": 0.79, "ece": 0.09},
        # shifted
        {"profile": "cnn", "paradigm": "CNN", "pretraining": "ImageNet", "shift": "heel", "intensity": 0.4, "method": "no_eraser", "auroc_macro": 0.70, "ece": 0.12},
        {"profile": "cnn", "paradigm": "CNN", "pretraining": "ImageNet", "shift": "heel", "intensity": 0.4, "method": "eraser", "auroc_macro": 0.75, "ece": 0.10},
        # second architecture
        {"profile": "clip", "paradigm": "CLIP/Foundation", "pretraining": "Foundation", "shift": "heel", "intensity": 0.0, "method": "no_eraser", "auroc_macro": 0.85, "ece": 0.06},
        {"profile": "clip", "paradigm": "CLIP/Foundation", "pretraining": "Foundation", "shift": "heel", "intensity": 0.5, "method": "no_eraser", "auroc_macro": 0.81, "ece": 0.09},
    ]
    return pd.DataFrame(rows)


def test_infer_paradigm_and_pretraining():
    assert infer_paradigm("baseline_cnn") == "CNN"
    assert infer_paradigm("timm_vit") == "ViT"
    assert infer_paradigm("hf_clip_or_vit") == "CLIP/Foundation"
    assert infer_pretraining("hf_clip_or_vit", "openai/clip-vit-base-patch32") == "Foundation"


def test_build_interaction_table_delta():
    df = _sample_metrics_df()
    out = build_interaction_table(df)
    assert not out.empty
    row = out[(out["profile"] == "cnn") & (np.isclose(out["intensity"].astype(float), 0.4))].iloc[0]
    # eraser - no_eraser = 0.75 - 0.70
    assert np.isclose(row["delta_auroc_macro"], 0.05)


def test_build_robustness_table_signs():
    df = _sample_metrics_df()
    out = build_robustness_table(df, metric_cols=["auroc_macro", "ece"], clean_intensity=0.0)
    assert not out.empty

    # For AUROC (higher better): degradation = clean - worst_shift
    row_auc = out[(out["profile"] == "cnn") & (out["method"] == "no_eraser") & (out["metric"] == "auroc_macro")].iloc[0]
    assert np.isclose(row_auc["degradation"], 0.10)
    assert np.isclose(row_auc["relative_degradation"], 0.10 / 0.80)

    # For ECE (lower better): degradation = worst_shift - clean
    row_ece = out[(out["profile"] == "cnn") & (out["method"] == "no_eraser") & (out["metric"] == "ece")].iloc[0]
    assert np.isclose(row_ece["degradation"], 0.04)
    assert np.isclose(row_ece["relative_degradation"], 0.04 / 0.08)


def test_summarize_architecture_matrix():
    df = _sample_metrics_df()
    out = summarize_architecture_matrix(df)
    assert not out.empty
    assert "auroc_macro_mean" in out.columns
