from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_summary_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    realism_dir = tmp_path / "realism"
    confounding_dir = tmp_path / "confounding"
    erasure_dir = tmp_path / "erasure"
    safety_dir = tmp_path / "safety"
    for path in [realism_dir, confounding_dir, erasure_dir, safety_dir]:
        path.mkdir(parents=True, exist_ok=True)

    (realism_dir / "realism_calibrator.json").write_text("{}", encoding="utf-8")
    (realism_dir / "realism_calibrator.report.json").write_text(
        json.dumps({"realism_validity_summary": {"is_graded": True}}, indent=2),
        encoding="utf-8",
    )
    (realism_dir / "pair_quality_audit.json").write_text(json.dumps({"technical_dominant": 3}, indent=2), encoding="utf-8")

    (confounding_dir / "confounding_summary.json").write_text(
        json.dumps({"n_samples": 12, "active_shift_id_group_key": "view_family"}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{"audit_family": "context_from_embeddings", "target": "view_family", "auroc_macro": 0.8}]).to_csv(
        confounding_dir / "confounding_predictability.csv",
        index=False,
    )
    pd.DataFrame([{"audit_family": "shift_id_proxy", "target": "view_family", "balanced_accuracy": 0.75}]).to_csv(
        confounding_dir / "shift_id_proxy_audit.csv",
        index=False,
    )
    pd.DataFrame([{"audit_family": "pair_pathology_delta", "target": "Edema", "auroc": 0.55}]).to_csv(
        confounding_dir / "technical_pair_pathology_deltas.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"method": "no_eraser", "shift": "heel", "realism_level": 0.2, "auroc_macro": 0.79, "ece": 0.13, "risk_at_coverage": 0.17, "expected_harm": 0.26, "coverage_realized": 0.8, "abstain_rate": 0.2},
            {"method": "eraser", "shift": "heel", "realism_level": 0.2, "auroc_macro": 0.81, "ece": 0.11, "risk_at_coverage": 0.15, "expected_harm": 0.22, "coverage_realized": 0.82, "abstain_rate": 0.18},
        ]
    ).to_csv(erasure_dir / "metrics.csv", index=False)
    pd.DataFrame([{"diagnostic": "probe_spuriousness_summary", "severity_shortcut_auroc": 0.61}]).to_csv(
        erasure_dir / "guarded_erasure_diagnostics.csv",
        index=False,
    )
    pd.DataFrame([{"strength": 0.5, "invariance_delta_vs_clean": 0.12}]).to_csv(
        erasure_dir / "erasure_tradeoff.csv",
        index=False,
    )
    pd.DataFrame([{"coverage": 0.8, "risk": 0.13, "risk_ci_low": 0.10, "risk_ci_high": 0.16}]).to_csv(
        erasure_dir / "risk_coverage_curves.csv",
        index=False,
    )

    pd.DataFrame([{"coverage": 0.8, "risk_at_coverage": 0.11, "harm": 0.18, "policy_regret": 0.03}]).to_csv(
        safety_dir / "safety.csv",
        index=False,
    )
    pd.DataFrame([{"decision": "apply_eraser", "reason": "covariate", "coverage": 0.8, "policy_regret": 0.03}]).to_csv(
        safety_dir / "policy_decisions.csv",
        index=False,
    )
    pd.DataFrame([{"decision": "apply_eraser", "reason": "covariate"}]).to_csv(
        safety_dir / "scenario_trace.csv",
        index=False,
    )
    pd.DataFrame([{"policy_regret_ci_low": 0.01, "policy_regret_ci_high": 0.05}]).to_csv(
        safety_dir / "policy_bootstrap.csv",
        index=False,
    )
    (safety_dir / "policy.json").write_text(
        json.dumps(
            {
                "contract_status": "incomplete",
                "incomplete_fields": ["source_label_priors_nonfinite"],
                "paper_tier_ready": False,
                "source_label_priors_status": {
                    "field": "source_label_priors",
                    "present": True,
                    "length": 1,
                    "n_nonfinite": 1,
                    "all_finite": False,
                    "fallback_applied": True,
                },
                "class_thresholds_status": {
                    "field": "class_thresholds",
                    "present": True,
                    "length": 1,
                    "n_nonfinite": 0,
                    "all_finite": True,
                },
                "label_space": "classifier_label_space",
                "label_cols": ["No Finding"],
                "source_label_priors": [0.5],
                "class_thresholds": [0.7],
                "coverage": 0.8,
                "harm": {"no_eraser": 0.2, "eraser": 0.18},
                "shift_harm": {"heel": {"no_eraser": 0.22, "eraser": 0.19}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (safety_dir / "policy_thresholds.json").write_text(
        json.dumps(
            {
                "class_thresholds": [0.7],
                "class_thresholds_status": {
                    "field": "class_thresholds",
                    "present": True,
                    "length": 1,
                    "n_nonfinite": 0,
                    "all_finite": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (safety_dir / "policy_operating_points_summary.json").write_text(
        json.dumps(
            {
                "n_rows": 1,
                "labels": ["No Finding"],
                "drift_stats": [{"label": "No Finding", "delta_fnr_vs_calib_mean": 0.02}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (safety_dir / "policy_operating_points_targeted_summary.json").write_text(
        json.dumps(
            {
                "n_rows": 1,
                "target_sensitivities": [0.9],
                "target_specificities": [0.9],
                "drift_stats": [
                    {
                        "label": "No Finding",
                        "target_kind": "sensitivity",
                        "target_value": 0.9,
                        "delta_sensitivity_vs_calib_mean": 0.01,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (safety_dir / "policy_regret_summary.json").write_text(
        json.dumps({"regret_mean": 0.03, "regret_boot_mean": 0.03}, indent=2),
        encoding="utf-8",
    )
    return realism_dir, confounding_dir, erasure_dir, safety_dir


def _write_tier_config(repo: Path, tmp_path: Path, tier: str) -> Path:
    base_cfg = (repo / "configs" / "mimic_public_acquisition_atomic.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / f"summary_{tier}.yaml"
    cfg_path.write_text(base_cfg + f"\nexperiment:\n  tier: {tier}\n", encoding="utf-8")
    return cfg_path


def test_build_public_mainline_summary_reporting_contract(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    realism_dir, confounding_dir, erasure_dir, safety_dir = _write_summary_artifacts(tmp_path)
    dev_cfg = _write_tier_config(repo, tmp_path, "dev")
    paper_cfg = repo / "configs" / "representation_surgery_2026_paper.yaml"
    out_dir_dev = tmp_path / "summary_dev"
    out_dir_paper = tmp_path / "summary_paper"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    dev_cmd = [
        sys.executable,
        str(repo / "scripts" / "build_public_mainline_summary.py"),
        "--config",
        str(dev_cfg),
        "--out_dir",
        str(out_dir_dev),
        "--realism_dir",
        str(realism_dir),
        "--confounding_dir",
        str(confounding_dir),
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
    ]
    subprocess.run(dev_cmd, check=True, cwd=str(repo), env=env)

    payload = json.loads((out_dir_dev / "public_mainline_summary.json").read_text(encoding="utf-8"))
    assert payload["reporting_contract"]["status"] == "incomplete"
    assert payload["reporting_contract"]["paper_tier_ready"] is False
    assert payload["reporting_contract"]["policy_contract_status"] == "incomplete"
    assert "source_label_priors_nonfinite" in payload["reporting_contract"]["incomplete_fields"]
    assert "policy_json" in payload["artifact_status"]
    assert "policy_operating_points_summary" in payload["artifact_status"]

    paper_cmd = [
        sys.executable,
        str(repo / "scripts" / "build_public_mainline_summary.py"),
        "--config",
        str(paper_cfg),
        "--out_dir",
        str(out_dir_paper),
        "--realism_dir",
        str(realism_dir),
        "--confounding_dir",
        str(confounding_dir),
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
    ]
    proc = subprocess.run(paper_cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "reporting_contract.status='ok'" in (proc.stdout + proc.stderr)
