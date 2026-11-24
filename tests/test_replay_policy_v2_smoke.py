from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_config(path: Path, *, tier: str) -> None:
    path.write_text(
        f"""
experiment:
  tier: {tier}
safety_policy:
  abstain_cost: 0.25
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_safety_bundle(path: Path, *, policy_contract_status: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "shift_family": "heel",
                "decision": "abstain",
                "reason": "high_uncertainty",
                "harm_no": 0.40,
                "harm_eraser": 0.20,
                "harm_recalibrated": 0.35,
                "p_none": 0.10,
                "p_covariate": 0.80,
                "p_prevalence": 0.10,
                "p_mixed": 0.00,
                "uncertainty": 0.90,
            },
            {
                "shift_family": "none",
                "decision": "no_eraser",
                "reason": "no_shift",
                "harm_no": 0.10,
                "harm_eraser": 0.30,
                "harm_recalibrated": 0.25,
                "p_none": 0.90,
                "p_covariate": 0.05,
                "p_prevalence": 0.03,
                "p_mixed": 0.02,
                "uncertainty": 0.05,
            },
            {
                "shift_family": "scatter",
                "decision": "abstain",
                "reason": "high_uncertainty",
                "harm_no": 0.40,
                "harm_eraser": 0.38,
                "harm_recalibrated": 0.15,
                "p_none": 0.10,
                "p_covariate": 0.15,
                "p_prevalence": 0.75,
                "p_mixed": 0.00,
                "uncertainty": 0.88,
            },
            {
                "shift_family": "mixed",
                "decision": "abstain",
                "reason": "high_uncertainty",
                "harm_no": 0.35,
                "harm_eraser": 0.40,
                "harm_recalibrated": 0.50,
                "p_none": 0.15,
                "p_covariate": 0.10,
                "p_prevalence": 0.10,
                "p_mixed": 0.65,
                "uncertainty": 0.95,
            },
            {
                "shift_family": "heel",
                "decision": "apply_eraser",
                "reason": "lower_expected_harm",
                "harm_no": 0.35,
                "harm_eraser": 0.12,
                "harm_recalibrated": 0.20,
                "p_none": 0.05,
                "p_covariate": 0.85,
                "p_prevalence": 0.08,
                "p_mixed": 0.02,
                "uncertainty": 0.15,
            },
        ]
    ).to_csv(path / "scenario_trace.csv", index=False)
    (path / "policy.json").write_text(
        json.dumps(
            {
                "contract_status": policy_contract_status,
                "paper_tier_ready": policy_contract_status == "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"oracle_action": "apply_eraser", "oracle_harm": 0.20, "uncertainty_veto": True},
            {"oracle_action": "no_eraser", "oracle_harm": 0.10, "uncertainty_veto": False},
            {"oracle_action": "recalibrate", "oracle_harm": 0.15, "uncertainty_veto": True},
            {"oracle_action": "abstain", "oracle_harm": 0.25, "uncertainty_veto": False},
            {"oracle_action": "apply_eraser", "oracle_harm": 0.12, "uncertainty_veto": False},
        ]
    ).to_csv(path / "oracle_policy_gap_trace.csv", index=False)
    (path / "oracle_policy_gap_summary.json").write_text(
        json.dumps({"n_scenarios": 5}, indent=2),
        encoding="utf-8",
    )


def test_replay_policy_v2_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "cfg.yaml"
    safety_dir = tmp_path / "safety"
    out_dir = tmp_path / "replay_out"

    _write_config(cfg_path, tier="dev")
    _write_safety_bundle(safety_dir, policy_contract_status="ok")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "replay_policy_v2.py"),
        "--config",
        str(cfg_path),
        "--safety_dir",
        str(safety_dir),
        "--out_dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    trace_path = out_dir / "policy_v2_replay_trace.csv"
    summary_path = out_dir / "policy_v2_replay_summary.json"
    md_path = out_dir / "policy_v2_replay.md"
    assert trace_path.exists()
    assert summary_path.exists()
    assert md_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selector_summaries = summary["selector_summaries"]
    selector_names = [row["selector_name"] for row in selector_summaries]

    assert "expected_harm_no_veto" in selector_names
    assert any(name == "expected_harm_with_risk_premium" for name in selector_names)

    best = summary["best_selector_by_mean_harm"]
    assert best["beats_current_rate"] > 0
    assert best["uncertainty_veto_reduction_rate"] > 0

    paper_cfg_path = tmp_path / "cfg_paper.yaml"
    paper_safety_dir = tmp_path / "safety_paper"
    paper_out_dir = tmp_path / "replay_out_paper"
    _write_config(paper_cfg_path, tier="paper")
    _write_safety_bundle(paper_safety_dir, policy_contract_status="incomplete")

    paper_cmd = [
        sys.executable,
        str(repo / "scripts" / "replay_policy_v2.py"),
        "--config",
        str(paper_cfg_path),
        "--safety_dir",
        str(paper_safety_dir),
        "--out_dir",
        str(paper_out_dir),
    ]
    proc = subprocess.run(paper_cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
