from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from phystta.cli.main import PIPELINE_NOTES, SCRIPT_MAP
from phystta.data import canonical_external_label_space, harmonize_external_labels, build_label_projection_manifest
from phystta.shifts.realism import build_realism_validity_summary
from phystta.viz import (
    build_erasure_eval_figures,
    build_qualitative_shift_triptych,
    build_realism_calibration_figures,
    build_safety_policy_figures,
)


def test_external_label_harmonization_preserves_target_order_and_surfaces_nih_proxies():
    target = canonical_external_label_space()
    index_df = pd.DataFrame(
        [
            {
                "source": "chexpert",
                **{label: 0.0 for label in target},
            },
            {
                "source": "nih",
                "Atelectasis": 0.0,
                "Cardiomegaly": 1.0,
                "Effusion": 1.0,
                "Infiltration": 1.0,
                "Mass": 0.0,
                "Nodule": 1.0,
                "Pneumonia": 0.0,
                "Pneumothorax": 0.0,
                "Consolidation": 0.0,
                "Edema": 0.0,
                "Emphysema": 0.0,
                "Fibrosis": 0.0,
                "Pleural_Thickening": 1.0,
                "Hernia": 0.0,
                "labels_from_findings_raw": "Effusion|Infiltration|Nodule",
            },
        ]
    )
    source_map = {
        "chexpert": SimpleNamespace(label_cols=list(target)),
        "nih": SimpleNamespace(
            label_cols=[
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia",
            ]
        ),
    }

    out_df, manifest = harmonize_external_labels(
        index_df,
        source_map=source_map,
        target_labels=target,
        eval_sources=["chexpert", "nih"],
    )

    nih_row = out_df[out_df["source"] == "nih"].iloc[0]
    assert manifest["target_label_order"] == target
    assert manifest["sources"]["nih"]["validation_ok"] is True
    assert "Lung Opacity" in manifest["sources"]["nih"]["approximate_labels"]
    assert nih_row["Pleural Effusion"] == 1.0
    assert nih_row["Lung Opacity"] == 1.0
    assert nih_row["Lung Lesion"] == 1.0
    assert nih_row["Pleural Other"] == 1.0
    assert nih_row["Support Devices"] == -1.0
    assert nih_row["No Finding"] == 0.0


def test_external_label_harmonization_rejects_unmapped_target_label():
    with pytest.raises(ValueError, match="No NIH harmonization rule defined"):
        harmonize_external_labels(
            pd.DataFrame([{"source": "nih", "labels_from_findings_raw": "No Finding"}]),
            source_map={"nih": SimpleNamespace(label_cols=["Atelectasis"])},
            target_labels=canonical_external_label_space() + ["Portable Marker"],
            eval_sources=["nih"],
        )


def test_vindr_projection_manifest_is_explicitly_partial():
    manifest = build_label_projection_manifest("vindr", target_labels=canonical_external_label_space())
    assert manifest["source"] == "vindr"
    assert "Lung Lesion" in manifest["approximate_labels"]
    assert "No Finding" in manifest["unsupported_labels"]
    assert "Support Devices" in manifest["unsupported_labels"]


def test_vindr_harmonization_uses_explicit_partial_mapping():
    target = canonical_external_label_space()
    index_df = pd.DataFrame(
        [
            {
                "source": "vindr",
                "Cardiomegaly": 1.0,
                "Lung Opacity": 1.0,
                "Nodule/Mass": 1.0,
                "Other lesion": 0.0,
                "Consolidation": 0.0,
                "Atelectasis": 0.0,
                "Pneumothorax": 0.0,
                "Pleural effusion": 1.0,
                "Pleural thickening": 1.0,
            }
        ]
    )
    out_df, manifest = harmonize_external_labels(
        index_df,
        source_map={"vindr": SimpleNamespace(label_cols=[
            "Cardiomegaly",
            "Lung Opacity",
            "Nodule/Mass",
            "Other lesion",
            "Consolidation",
            "Atelectasis",
            "Pneumothorax",
            "Pleural effusion",
            "Pleural thickening",
        ])},
        target_labels=target,
        eval_sources=["vindr"],
    )
    row = out_df.iloc[0]
    assert row["Cardiomegaly"] == 1.0
    assert row["Lung Opacity"] == 1.0
    assert row["Lung Lesion"] == 1.0
    assert row["Pleural Effusion"] == 1.0
    assert row["Pleural Other"] == 1.0
    assert row["No Finding"] == -1.0
    assert "Lung Lesion" in manifest["sources"]["vindr"]["approximate_labels"]


def test_realism_validity_summary_contract_contains_required_fields():
    summary = build_realism_validity_summary(
        is_graded=False,
        level_separation_margin=0.01,
        n_pairs_total=120,
        n_pairs_accepted=42,
        pair_quality_distribution={"same_view_low_risk": 42, "clinical_change_suspected": 8},
        accepted_view_families=["frontal"],
        accepted_procedure_contexts=["portable_like"],
        accepted_orientation_contexts=["upright"],
        pair_acceptance_funnel={
            "all_candidate_pairs": 120,
            "valid_temporal_pairs": 90,
            "quality_accepted_pairs": 42,
            "calibrator_used_pairs": 33,
        },
    )

    assert summary["is_graded"] is False
    assert summary["n_pairs_total"] == 120
    assert summary["accepted_view_families"] == ["frontal"]
    assert summary["pair_acceptance_funnel"]["calibrator_used_pairs"] == 33


def test_atomic_figure_pack_writes_required_outputs(tmp_path: Path):
    metrics_df = pd.DataFrame(
        [
            {
                "method": method,
                "shift": shift,
                "intensity": realism,
                "realism_level": realism,
                "auroc_macro": 0.82 - 0.05 * realism + (0.02 if method == "eraser" else 0.0),
                "auprc_macro": 0.61 - 0.04 * realism + (0.01 if method == "eraser" else 0.0),
                "ece": 0.10 + 0.04 * realism - (0.01 if method == "eraser" else 0.0),
                "brier": 0.18 + 0.03 * realism - (0.01 if method == "eraser" else 0.0),
                "risk_at_coverage": 0.12 + 0.05 * realism - (0.01 if method == "eraser" else 0.0),
                "expected_harm": 0.25 + 0.06 * realism - (0.02 if method == "eraser" else 0.0),
                "coverage_realized": 0.8,
                "abstain_rate": 0.2,
            }
            for method in ["no_eraser", "eraser"]
            for shift in ["heel", "scatter"]
            for realism in [0.0, 0.5, 1.0]
        ]
    )
    reliability_df = pd.DataFrame(
        [
            {
                "scenario_label": "heel:realism_1.00",
                "shift": "heel",
                "method": method,
                "intensity": 1.0,
                "realism_level": 1.0,
                "x_value": 1.0,
                "confidence": conf,
                "accuracy": acc,
                "count": 10,
            }
            for method, conf, acc in [
                ("no_eraser", 0.6, 0.5),
                ("no_eraser", 0.8, 0.65),
                ("eraser", 0.6, 0.58),
                ("eraser", 0.8, 0.74),
            ]
        ]
    )

    outputs = build_erasure_eval_figures(
        metrics_df,
        out_dir=tmp_path,
        scenario_mode="atomic_calibrated_eval",
        reliability_df=reliability_df,
    )

    fig_dir = tmp_path / "figures" / "atomic_calibrated"
    assert (fig_dir / "atomic_calibrated_main.png").exists()
    assert (fig_dir / "atomic_calibrated_main.pdf").exists()
    assert (fig_dir / "atomic_calibrated_by_family.png").exists()
    assert (fig_dir / "atomic_main_metrics_small_multiples.png").exists()
    assert (fig_dir / "atomic_main_metrics_small_multiples_tidy.csv").exists()
    assert (fig_dir / "atomic_worst_family_overlay.png").exists()
    assert (fig_dir / "atomic_worst_family_overlay_tidy.csv").exists()
    assert (fig_dir / "atomic_summary_spread_overlay.png").exists()
    assert (fig_dir / "atomic_summary_spread_overlay_tidy.csv").exists()
    assert (fig_dir / "atomic_metrics_tidy.csv").exists()
    assert (fig_dir / "atomic_reliability_selected.png").exists()
    assert "atomic_calibrated_main" in outputs
    assert "atomic_main_metrics_small_multiples" in outputs
    assert "atomic_worst_family_overlay" in outputs
    assert "atomic_summary_spread_overlay" in outputs
    assert not (tmp_path / "figures" / "mixed_stress" / "mixed_stress_summary.png").exists()


def test_mixed_figure_pack_stays_separate_from_atomic(tmp_path: Path):
    metrics_df = pd.DataFrame(
        [
            {
                "method": method,
                "shift": "mixed",
                "profile": profile,
                "auroc_macro": 0.7,
                "ece": 0.12,
                "risk_at_coverage": 0.2,
                "expected_harm": 0.3,
            }
            for method in ["no_eraser", "eraser"]
            for profile in ["portable_push", "high_noise_combo"]
        ]
    )

    outputs = build_erasure_eval_figures(
        metrics_df,
        out_dir=tmp_path,
        scenario_mode="mixed_stress_eval",
        reliability_df=None,
    )

    fig_dir = tmp_path / "figures" / "mixed_stress"
    assert (fig_dir / "mixed_stress_summary.png").exists()
    assert (fig_dir / "mixed_stress_metrics_tidy.csv").exists()
    assert "mixed_stress_summary" in outputs
    assert not (tmp_path / "figures" / "atomic_calibrated" / "atomic_calibrated_main.png").exists()


def test_safety_and_realism_figure_helpers_emit_required_artifacts(tmp_path: Path):
    scenario_trace_df = pd.DataFrame(
        [
            {
                "scenario_id": "heel:0.5",
                "decision": "apply_eraser",
                "reason": "covariate_shift",
                "harm_no": 0.3,
                "harm_eraser": 0.2,
                "harm_recalibrated": 0.25,
                "coverage": 0.78,
                "accepted_samples": 78,
                "abstained_samples": 22,
            },
            {
                "scenario_id": "scatter:1.0",
                "decision": "abstain",
                "reason": "high_uncertainty",
                "harm_no": 0.45,
                "harm_eraser": 0.41,
                "harm_recalibrated": 0.38,
                "coverage": 0.0,
                "accepted_samples": 0,
                "abstained_samples": 100,
            },
        ]
    )
    op_target_df = pd.DataFrame(
        [
            {
                "label": "Edema",
                "delta_sensitivity_vs_calib": 0.03,
                "delta_specificity_vs_calib": -0.02,
                "delta_ppv_vs_calib": 0.01,
                "delta_npv_vs_calib": 0.00,
            },
            {
                "label": "Pneumothorax",
                "delta_sensitivity_vs_calib": -0.04,
                "delta_specificity_vs_calib": 0.03,
                "delta_ppv_vs_calib": -0.01,
                "delta_npv_vs_calib": 0.02,
            },
        ]
    )
    op_df = pd.DataFrame(
        [
            {"label": "Edema", "delta_harm_vs_calib": 0.02},
            {"label": "Pneumothorax", "delta_harm_vs_calib": -0.01},
        ]
    )
    bootstrap_df = pd.DataFrame(
        [
            {
                "shift": "heel",
                "intensity": 0.5,
                "realism_level": 0.5,
                "decision": "apply_eraser",
                "policy_harm_mean": 0.22,
                "policy_harm_ci_low": 0.18,
                "policy_harm_ci_high": 0.27,
                "policy_regret_mean": 0.03,
                "policy_regret_ci_low": 0.01,
                "policy_regret_ci_high": 0.06,
            },
            {
                "shift": "scatter",
                "intensity": 1.0,
                "realism_level": 1.0,
                "decision": "abstain",
                "policy_harm_mean": 0.40,
                "policy_harm_ci_low": 0.36,
                "policy_harm_ci_high": 0.45,
                "policy_regret_mean": 0.08,
                "policy_regret_ci_low": 0.05,
                "policy_regret_ci_high": 0.11,
            },
        ]
    )
    safety_outputs = build_safety_policy_figures(
        out_dir=tmp_path,
        scenario_trace_df=scenario_trace_df,
        op_target_df=op_target_df,
        op_df=op_df,
        bootstrap_df=bootstrap_df,
    )
    assert (tmp_path / "figures" / "safety_policy" / "policy_decision_summary.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_decision_distribution.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_reason_heatmap.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_harm_delta_summary.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_coverage_abstain_summary.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_operating_point_deltas.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_harm_deltas.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_bootstrap_summary.png").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_bootstrap_tidy.csv").exists()
    assert (tmp_path / "figures" / "safety_policy" / "policy_reason_heatmap_tidy.csv").exists()
    assert "policy_decision_summary" in safety_outputs
    assert "policy_reason_heatmap" in safety_outputs

    report = {
        "n_pairs_before_filter": 120,
        "n_pairs_after_filter": 80,
        "pair_quality_counts": {"technical_dominant": 42},
        "realism_validity_summary": {
            "pair_acceptance_funnel": {
                "all_candidate_pairs": 120,
                "valid_temporal_pairs": 80,
                "quality_accepted_pairs": 42,
                "calibrator_used_pairs": 33,
            }
        },
        "heel": {"mapping": {"levels": [0.0, 0.5, 1.0]}, "is_graded": True, "level_separation_margin": 0.15, "validity_flags": ["valid"]},
        "scatter": {"mapping": {"levels": [0.0, 0.5, 1.0]}, "is_graded": False, "level_separation_margin": 0.0, "validity_flags": ["non_graded"]},
    }
    pairs_df = pd.DataFrame(
        [
            {"acquisition_context_risk": 0.1, "clinical_change_risk": 0.12, "pair_quality_bucket": "technical_dominant"},
            {"acquisition_context_risk": 0.6, "clinical_change_risk": 0.7, "pair_quality_bucket": "ambiguous"},
        ]
    )
    realism_outputs = build_realism_calibration_figures(out_dir=tmp_path, report=report, pairs_df=pairs_df)
    assert (tmp_path / "figures" / "realism" / "realism_pair_acceptance_funnel.png").exists()
    assert (tmp_path / "figures" / "realism" / "realism_pair_risk_histogram.png").exists()
    assert (tmp_path / "figures" / "realism" / "realism_pair_risk_ecdf.png").exists()
    assert (tmp_path / "figures" / "realism" / "realism_pair_risk_ecdf_tidy.csv").exists()
    assert (tmp_path / "figures" / "realism" / "realism_validity.png").exists()
    assert "realism_validity" in realism_outputs
    assert "realism_pair_risk_histogram" in realism_outputs
    assert "realism_pair_risk_ecdf" in realism_outputs


def test_cli_marks_public_mainline_and_historical_aliases():
    assert SCRIPT_MAP["public-mainline"] == "scripts/run_public_mainline.py"
    assert "historical" in PIPELINE_NOTES["paper-suite"]


def test_qualitative_shift_triptych_builder_writes_expected_outputs(tmp_path: Path):
    base = pd.Series(np.linspace(0.0, 1.0, num=64), dtype=float).to_numpy().reshape(8, 8)
    cases = [
        {
            "case_id": "cxr_001",
            "shift_family": "heel",
            "intensity": 0.5,
            "clean": base,
            "shifted": np.clip(base + 0.12, 0.0, 1.0),
            "erased": np.clip(base + 0.04, 0.0, 1.0),
            "pred_before": "Edema 0.62",
            "pred_after": "Edema 0.48",
            "harm_delta": -0.08,
        },
        {
            "case_id": "cxr_002",
            "shift_family": "scatter",
            "intensity": 1.0,
            "clean": base[::-1],
            "shifted": np.clip(base[::-1] * 0.85 + 0.10, 0.0, 1.0),
            "erased": np.clip(base[::-1] * 0.92 + 0.04, 0.0, 1.0),
            "caption_lines": ["portable_like", "frontal"],
        },
    ]

    outputs = build_qualitative_shift_triptych(
        out_dir=tmp_path,
        cases=cases,
        output_name="qualitative_shift_triptych",
        include_difference=True,
    )

    fig_dir = tmp_path / "figures" / "qualitative"
    assert (fig_dir / "qualitative_shift_triptych.png").exists()
    assert (fig_dir / "qualitative_shift_triptych.pdf").exists()
    assert (fig_dir / "qualitative_shift_triptych_manifest.csv").exists()
    manifest_df = pd.read_csv(fig_dir / "qualitative_shift_triptych_manifest.csv")
    assert set(manifest_df["case_id"].astype(str)) == {"cxr_001", "cxr_002"}
    assert "qualitative_shift_triptych" in outputs
    assert "qualitative_shift_triptych_manifest" in outputs
