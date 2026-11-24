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
                "p_covariate": 0.60,
                "p_prevalence": 0.20,
                "p_mixed": 0.10,
                "uncertainty": 0.95,
            },
            {
                "shift_family": "none",
                "decision": "no_eraser",
                "reason": "no_shift",
                "harm_no": 0.10,
                "harm_eraser": 0.30,
                "harm_recalibrated": 0.25,
                "p_none": 0.80,
                "p_covariate": 0.10,
                "p_prevalence": 0.05,
                "p_mixed": 0.05,
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
                "p_covariate": 0.20,
                "p_prevalence": 0.60,
                "p_mixed": 0.10,
                "uncertainty": 0.92,
            },
            {
                "shift_family": "mixed",
                "decision": "abstain",
                "reason": "high_uncertainty",
                "harm_no": 0.35,
                "harm_eraser": 0.40,
                "harm_recalibrated": 0.50,
                "p_none": 0.10,
                "p_covariate": 0.10,
                "p_prevalence": 0.10,
                "p_mixed": 0.70,
                "uncertainty": 0.97,
            },
        ]
    ).to_csv(path / "scenario_trace.csv", index=False)
    pd.DataFrame(
        [
            {"decision": "abstain"},
            {"decision": "no_eraser"},
            {"decision": "abstain"},
            {"decision": "abstain"},
        ]
    ).to_csv(path / "policy_decisions.csv", index=False)
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
    (path / "policy_regret_summary.json").write_text(
        json.dumps({"regret_mean": 0.0375}, indent=2),
        encoding="utf-8",
    )
    (path / "shift_id_report.json").write_text(
        json.dumps({"status": "ok"}, indent=2),
        encoding="utf-8",
    )


def test_decompose_oracle_policy_gap_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "cfg.yaml"
    safety_dir = tmp_path / "safety"
    out_dir = tmp_path / "oracle_gap_out"

    _write_config(cfg_path, tier="dev")
    _write_safety_bundle(safety_dir, policy_contract_status="ok")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "decompose_oracle_policy_gap.py"),
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

    trace_path = out_dir / "oracle_policy_gap_trace.csv"
    summary_path = out_dir / "oracle_policy_gap_summary.json"
    md_path = out_dir / "oracle_policy_gap.md"
    assert trace_path.exists()
    assert summary_path.exists()
    assert md_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trace = pd.read_csv(trace_path)

    assert summary["n_scenarios"] == 4
    assert summary["uncertainty_veto_count"] == 2
    assert summary["eraser_oracle_count"] == 1
    assert summary["eraser_missed_count"] == 1
    assert summary["recalibrate_oracle_count"] == 1
    assert summary["recalibrate_missed_count"] == 1
    assert summary["oracle_abstain_count"] == 1

    assert ((trace["oracle_action"] == "apply_eraser") & (trace["uncertainty_veto"] == True)).any()
    assert ((trace["oracle_action"] == "recalibrate") & (trace["recalibrate_missed"] == True)).any()

    paper_cfg_path = tmp_path / "cfg_paper.yaml"
    paper_safety_dir = tmp_path / "safety_paper"
    paper_out_dir = tmp_path / "oracle_gap_out_paper"
    _write_config(paper_cfg_path, tier="paper")
    _write_safety_bundle(paper_safety_dir, policy_contract_status="incomplete")

    paper_cmd = [
        sys.executable,
        str(repo / "scripts" / "decompose_oracle_policy_gap.py"),
        "--config",
        str(paper_cfg_path),
        "--safety_dir",
        str(paper_safety_dir),
        "--out_dir",
        str(paper_out_dir),
    ]
    proc = subprocess.run(paper_cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    assert proc.returncode != 0
