import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from phystta.config.loader import load_config


def _write_profile_artifacts(root: Path, profile: str) -> None:
    eval_dir = root / profile / "eval"
    safety_dir = root / profile / "safety"
    eval_dir.mkdir(parents=True, exist_ok=True)
    safety_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(
        [
            {"shift": "heel", "intensity": 0.0, "method": "no_eraser", "auroc_macro": 0.82, "ece": 0.08},
            {"shift": "heel", "intensity": 0.0, "method": "eraser", "auroc_macro": 0.81, "ece": 0.09},
            {"shift": "heel", "intensity": 0.5, "method": "no_eraser", "auroc_macro": 0.74, "ece": 0.14},
            {"shift": "heel", "intensity": 0.5, "method": "eraser", "auroc_macro": 0.78, "ece": 0.11},
        ]
    )
    metrics.to_csv(eval_dir / "metrics.csv", index=False)

    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.5, "group": "sex", "worst_group_auroc_macro": 0.70}]
    ).to_csv(eval_dir / "fairness.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.5, "metric": "dose_response_slope", "value": -0.15}]
    ).to_csv(eval_dir / "causal_invariance.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.5, "decision": "no_eraser", "coverage": 0.9, "auroc_macro": 0.74}]
    ).to_csv(safety_dir / "safety.csv", index=False)
    pd.DataFrame(
        [{"shift": "heel", "intensity": 0.5, "decision": "no_eraser", "reason": "legacy_policy"}]
    ).to_csv(safety_dir / "policy_decisions.csv", index=False)


def test_architecture_matrix_generates_full_artifact_set(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo / "configs" / "representation_surgery_2026.yaml"))
    out_dir = tmp_path / "arch"
    out_dir.mkdir(parents=True, exist_ok=True)

    for profile in cfg.architecture_study.encoder_profiles or []:
        _write_profile_artifacts(out_dir, str(profile))

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_architecture_matrix.py"),
        "--config",
        str(repo / "configs" / "representation_surgery_2026.yaml"),
        "--out_dir",
        str(out_dir),
        "--aggregate_only",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    required = [
        "architecture_matrix_metrics.csv",
        "architecture_interactions.csv",
        "architecture_robustness.csv",
        "architecture_robustness_summary.csv",
        "architecture_summary.csv",
        "architecture_fairness.csv",
        "architecture_causal.csv",
        "architecture_safety.csv",
        "architecture_policy_decisions.csv",
        "architecture_matrix_report.json",
        "architecture_protocol_manifest.json",
    ]
    for name in required:
        assert (out_dir / name).exists(), f"missing artifact: {name}"
