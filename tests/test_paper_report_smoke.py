import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_paper_report_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    er_dir = tmp_path / "erasure"
    sa_dir = tmp_path / "safety"
    ra_dir = tmp_path / "real_anchor"
    out_dir = tmp_path / "report_out"
    er_dir.mkdir(parents=True, exist_ok=True)
    sa_dir.mkdir(parents=True, exist_ok=True)
    ra_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.2, "auroc_macro": 0.71, "auprc_macro": 0.44, "expected_harm": 0.12}]
    ).to_csv(er_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "sex_fnr_gap": 0.03, "sex_worst_auroc": 0.68, "sex_ece_gap": 0.01}]
    ).to_csv(er_dir / "fairness.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "flip_rate": 0.09, "js_div": 0.02, "sensitivity_slope": 0.12}]
    ).to_csv(er_dir / "causal_invariance.csv", index=False)

    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.2, "coverage": 0.8, "expected_harm": 0.11, "risk_at_coverage": 0.08, "ece": 0.05}]
    ).to_csv(sa_dir / "safety.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.2, "decision": "no_eraser"}]
    ).to_csv(sa_dir / "policy_decisions.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.2, "policy_harm_mean": 0.1, "policy_regret_mean": 0.02}]
    ).to_csv(sa_dir / "policy_bootstrap.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "label": "Finding", "delta_fnr_vs_calib": -0.01, "delta_harm_vs_calib": -0.02}]
    ).to_csv(sa_dir / "policy_operating_points.csv", index=False)
    pd.DataFrame(
        [
            {
                "shift": "heel",
                "label": "Finding",
                "target_kind": "sensitivity",
                "target_value": 0.9,
                "delta_sensitivity_vs_calib": -0.01,
                "delta_specificity_vs_calib": 0.02,
                "delta_ppv_vs_calib": 0.01,
                "delta_npv_vs_calib": -0.01,
            }
        ]
    ).to_csv(sa_dir / "policy_operating_points_targeted.csv", index=False)
    (ra_dir / "principal_angles.json").write_text(json.dumps({"median_train_vs_synth_deg": 18.3}), encoding="utf-8")
    (ra_dir / "falsification_report.json").write_text(json.dumps({"n_pairs_train": 20}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "build_paper_report.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--out_dir",
        str(out_dir),
        "--erasure_dir",
        str(er_dir),
        "--safety_dir",
        str(sa_dir),
        "--real_anchor_dir",
        str(ra_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    report_path = out_dir / "paper_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config"]["tier"] == "paper"
    assert report["missing_artifacts"] == []
    assert report["artifact_presence"]["policy_bootstrap_csv"] is True
    assert "claims" in report
    assert "paper_gate" in report
    assert "coverage_floor" in report["claims"]
