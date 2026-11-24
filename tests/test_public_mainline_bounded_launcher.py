from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from phystta.cli.main import PIPELINE_NOTES, SCRIPT_MAP
from phystta.config.loader import load_config
from phystta.intervention.eraser import LinearSubspaceEraser, load_eraser_collection, resolve_eraser_for_shift, save_eraser
from phystta.utils import preprocess_signature


def test_public_mainline_cli_points_to_bounded_launcher():
    assert SCRIPT_MAP["public-mainline"] == "scripts/run_public_mainline.py"
    assert "bounded" in PIPELINE_NOTES["public-mainline"]


def test_load_eraser_collection_supports_bank_manifest(tmp_path: Path):
    heel_path = tmp_path / "heel.pt"
    scatter_path = tmp_path / "scatter.pt"
    save_eraser(str(heel_path), LinearSubspaceEraser(torch.eye(4, 1, dtype=torch.float32)))
    save_eraser(str(scatter_path), LinearSubspaceEraser(torch.eye(4, 1, dtype=torch.float32)))
    manifest_path = tmp_path / "eraser_bank_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fit_mode": "one_eraser_per_shift_family",
                "erasers": [
                    {"shift_family": "heel", "eraser_path": str(heel_path)},
                    {"shift_family": "scatter", "eraser_path": str(scatter_path)},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    collection, meta = load_eraser_collection(manifest_path)
    assert meta["kind"] == "bank"
    assert sorted(meta["shift_families"]) == ["heel", "scatter"]
    assert resolve_eraser_for_shift(collection, "heel") is collection["heel"]
    assert resolve_eraser_for_shift(collection, "scatter") is collection["scatter"]


def test_run_public_mainline_dry_run_writes_bounded_plan(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    plan_path = out_dir / "public_mainline_plan.json"
    preflight_path = out_dir / "public_mainline_preflight.json"
    effective_cfg_path = out_dir / "public_mainline.effective.yaml"
    assert plan_path.exists()
    assert preflight_path.exists()
    assert effective_cfg_path.exists()

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    joined = "\n".join(" ".join(cmd) for cmd in payload["commands"])
    assert "collect_probes.py" in joined
    assert "fit_eraser.py" in joined
    assert "calibrate_physics_realism.py" in joined
    assert "run_confounding_audit.py" in joined
    assert "run_erasure_eval.py" in joined
    assert "run_safety_policy_eval.py" in joined
    assert "build_public_mainline_summary.py" in joined
    assert payload["preflight"]["errors"], "dry-run should preserve preflight blockers instead of hiding them"
    assert payload["canonical_contract"] == "docs/canonical_public_mimic_experiment_contract.md"
    assert payload["logs_dir"] == str(out_dir / "launcher_logs")
    assert payload["stage_state_dir"] == str(out_dir / ".stage_state")
    assert preflight["logs_dir"] == str(out_dir / "launcher_logs")
    assert preflight["stage_state_dir"] == str(out_dir / ".stage_state")
    selected = [row for row in payload["stages"] if row["selected"]]
    assert selected
    assert all("log_path" in row and "state_path" in row for row in selected)
    assert all(Path(row["log_path"]).parent == out_dir / "launcher_logs" for row in selected)
    assert all(Path(row["state_path"]).parent == out_dir / ".stage_state" for row in selected)


def test_run_public_mainline_dry_run_supports_stage_range_and_path_overrides(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--dataset_root",
        str(tmp_path / "pod_mimic"),
        "--head_path",
        str(tmp_path / "heads" / "head.pt"),
        "--head_meta_path",
        str(tmp_path / "heads" / "head.meta.json"),
        "--index_dir",
        str(tmp_path / "indexes"),
        "--cache_dir",
        str(tmp_path / "cache"),
        "--start_from",
        "confounding_audit",
        "--stop_after",
        "summary",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    payload = json.loads((out_dir / "public_mainline_plan.json").read_text(encoding="utf-8"))
    assert payload["selected_stages"] == ["confounding_audit", "erasure_eval", "safety_eval", "summary"]
    stages = {row["stage"]: row for row in payload["stages"]}
    assert stages["collect_probes"]["planned_action"] == "not_selected"
    assert stages["summary"]["planned_action"] == "run"
    effective_cfg = load_config(str(out_dir / "public_mainline.effective.yaml"))
    assert str(effective_cfg.data.sources[0].root) == str(tmp_path / "pod_mimic")
    assert str(effective_cfg.model.encoder.head_path) == str(tmp_path / "heads" / "head.pt")
    assert str(effective_cfg.data.index_dir) == str(tmp_path / "indexes")
    assert str(effective_cfg.data.cache_dir) == str(tmp_path / "cache")


def test_run_public_mainline_resume_skips_success_marked_stage(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out"
    probes_dir = out_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"))
    torch.save(
        {
            "probes": {
                "features_clean": torch.zeros((1, 2), dtype=torch.float32),
                "features_shifted": torch.zeros((1, 2), dtype=torch.float32),
                "intensity": torch.zeros((1,), dtype=torch.float32),
                "patient_id": ["p0"],
                "group_id": ["p0"],
                "study_id": ["s0"],
                "image_id": ["i0"],
                "profile": ["heel"],
                "shift_type": ["heel"],
                "labels": None,
            },
            "meta": {
                "preprocess_signature": str(preprocess_signature(cfg)),
                "group_valid": "True",
                "shift_family": "heel,scatter,exposure_poisson,motion_blur",
                "source_split": str(cfg.data.eraser_split),
                "consistency_ok": "True",
            },
        },
        probes_dir / "probes.pt",
    )
    (probes_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    stage_state_dir = out_dir / ".stage_state"
    stage_state_dir.mkdir(parents=True, exist_ok=True)
    (stage_state_dir / "collect_probes.json").write_text(
        json.dumps(
            {
                "stage": "collect_probes",
                "status": "success",
                "expected_outputs": [
                    str(probes_dir / "probes.pt"),
                    str(probes_dir / "artifact_manifest.json"),
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--resume",
        "--stop_after",
        "collect_probes",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    payload = json.loads((out_dir / "public_mainline_plan.json").read_text(encoding="utf-8"))
    row = next(item for item in payload["stages"] if item["stage"] == "collect_probes")
    assert row["planned_action"] == "resume_skip"


def test_run_public_mainline_dry_run_surfaces_missing_dependency_for_late_start(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--start_from",
        "erasure_eval",
        "--stop_after",
        "erasure_eval",
        "--dry_run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    payload = json.loads((out_dir / "public_mainline_preflight.json").read_text(encoding="utf-8"))
    joined = " | ".join(payload["errors"])
    assert "Selected stage 'erasure_eval' requires missing input artifact" in joined


def test_build_public_mainline_summary_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    realism_dir = tmp_path / "realism"
    confounding_dir = tmp_path / "confounding"
    erasure_dir = tmp_path / "erasure"
    safety_dir = tmp_path / "safety"
    out_dir = tmp_path / "summary"
    launcher_root = tmp_path / "launcher"
    for path in [realism_dir, confounding_dir, erasure_dir, safety_dir]:
        path.mkdir(parents=True, exist_ok=True)
    launcher_root.mkdir(parents=True, exist_ok=True)
    (launcher_root / ".stage_state").mkdir(parents=True, exist_ok=True)
    (launcher_root / "launcher_logs").mkdir(parents=True, exist_ok=True)

    (realism_dir / "realism_calibrator.json").write_text("{}", encoding="utf-8")
    (realism_dir / "realism_calibrator.report.json").write_text(
        json.dumps(
            {
                "realism_validity_summary": {"is_graded": True, "n_pairs_total": 20, "n_pairs_accepted": 8},
                "pair_quality_counts": {"technical_dominant": 8, "ambiguous": 3},
                "physical_dependence_warnings": ["exposure_poisson is dose-coupled"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (realism_dir / "pair_quality_audit.json").write_text(json.dumps({"technical_dominant": 8}, indent=2), encoding="utf-8")

    (confounding_dir / "confounding_summary.json").write_text(
        json.dumps({"n_samples": 40, "active_shift_id_group_key": "view_family"}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"audit_family": "context_from_embeddings", "target": "view_family", "auroc_macro": 0.82},
            {"audit_family": "labels_from_context", "target": "Edema", "balanced_accuracy": 0.61},
        ]
    ).to_csv(confounding_dir / "confounding_predictability.csv", index=False)
    pd.DataFrame([{"audit_family": "shift_id_proxy", "target": "view_family", "balanced_accuracy": 0.77}]).to_csv(
        confounding_dir / "shift_id_proxy_audit.csv", index=False
    )
    pd.DataFrame([{"audit_family": "pair_pathology_delta", "target": "Edema", "auroc": 0.55}]).to_csv(
        confounding_dir / "technical_pair_pathology_deltas.csv", index=False
    )

    pd.DataFrame(
        [
            {"method": "no_eraser", "shift": "heel", "realism_level": 0.0, "auroc_macro": 0.80, "ece": 0.12, "risk_at_coverage": 0.15, "expected_harm": 0.24, "coverage_realized": 0.8, "abstain_rate": 0.2},
            {"method": "eraser", "shift": "heel", "realism_level": 0.0, "auroc_macro": 0.82, "ece": 0.10, "risk_at_coverage": 0.13, "expected_harm": 0.21, "coverage_realized": 0.82, "abstain_rate": 0.18},
        ]
    ).to_csv(erasure_dir / "metrics.csv", index=False)
    pd.DataFrame([{"diagnostic": "probe_spuriousness_summary", "severity_shortcut_auroc": 0.61}]).to_csv(
        erasure_dir / "guarded_erasure_diagnostics.csv", index=False
    )
    pd.DataFrame([{"coverage": 0.8, "risk": 0.13, "risk_ci_low": 0.10, "risk_ci_high": 0.16}]).to_csv(
        erasure_dir / "risk_coverage_curves.csv", index=False
    )
    pd.DataFrame([{"strength": 0.5, "invariance_delta_vs_clean": 0.12}]).to_csv(
        erasure_dir / "erasure_tradeoff.csv", index=False
    )

    pd.DataFrame([{"coverage": 0.8, "risk_at_coverage": 0.11, "harm": 0.18, "policy_regret": 0.03}]).to_csv(
        safety_dir / "safety.csv", index=False
    )
    pd.DataFrame([{"decision": "apply_eraser", "reason": "covariate"}]).to_csv(
        safety_dir / "scenario_trace.csv", index=False
    )
    pd.DataFrame([{"policy_regret_ci_low": 0.01, "policy_regret_ci_high": 0.05}]).to_csv(
        safety_dir / "policy_bootstrap.csv", index=False
    )
    (safety_dir / "policy_regret_summary.json").write_text(
        json.dumps({"regret_boot_mean": 0.03, "regret_boot_ci_high": 0.05}, indent=2),
        encoding="utf-8",
    )
    (launcher_root / "public_mainline_preflight.json").write_text(json.dumps({"errors": [], "notes": ["ok"]}, indent=2), encoding="utf-8")
    (launcher_root / "public_mainline_plan.json").write_text(json.dumps({"selected_stages": ["collect_probes", "summary"]}, indent=2), encoding="utf-8")
    (launcher_root / "public_mainline_manifest.json").write_text(json.dumps({"completion_timestamp_utc": "2026-01-01T00:00:00Z"}, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "build_public_mainline_summary.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--realism_dir",
        str(realism_dir),
        "--confounding_dir",
        str(confounding_dir),
        "--erasure_dir",
        str(erasure_dir),
        "--safety_dir",
        str(safety_dir),
        "--launcher_root",
        str(launcher_root),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    summary_path = out_dir / "public_mainline_summary.json"
    artifact_index_path = out_dir / "artifact_index.json"
    assert summary_path.exists()
    assert artifact_index_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["realism_evidence"]["realism_validity_summary"]["is_graded"] is True
    assert "context_from_embeddings" in json.dumps(payload["confounding_evidence"]["top_predictability_rows"])
    assert "decision_counts" in payload["safety_evidence"]
    assert payload["launcher_bundle"]["preflight"]["errors"] == []
    assert payload["launcher_artifact_status"]["stage_state_dir"]["exists"] is True
    assert payload["launcher_artifact_status"]["launcher_logs_dir"]["exists"] is True


def test_intervention_gate_evaluation_accepts_classifier_relevant_row():
    from scripts.run_public_mainline import _evaluate_intervention_gate

    verdict = _evaluate_intervention_gate(
        [
            {
                "bank": "candidate",
                "score": 1.0,
                "mean_nonzero_logit_delta": 2e-4,
                "mean_nonzero_prob_delta": 5e-5,
                "mean_zero_logit_damage": 1e-4,
                "mean_zero_prob_damage": 1e-5,
            }
        ],
        min_nonzero_logit_delta=1e-5,
        min_nonzero_prob_delta=1e-6,
        max_zero_logit_damage=5e-3,
        max_zero_prob_damage=5e-4,
    )

    assert verdict["passed"] is True
    assert verdict["reason"] == "ok"
    assert verdict["selected_bank"] == "candidate"


def test_run_public_mainline_dry_run_exposes_intervention_gate_config(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out_gate"
    dummy_head = tmp_path / "head.pt"
    dummy_head.write_bytes(b"stub")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--dry_run",
        "--adjudicate_after_fit",
        "--adjudication_head",
        str(dummy_head),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    plan_path = out_dir / "public_mainline_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert payload["intervention_gate"]["enabled"] is True
    assert payload["intervention_gate"]["head"] == str(dummy_head)
    assert payload["artifact_paths"]["adjudication"] == str(out_dir / "adjudication")
    assert payload["intervention_gate"]["thresholds"]["min_nonzero_logit_delta"] == 1e-5
    assert payload["intervention_gate"]["thresholds"]["min_nonzero_prob_delta"] == 1e-6


def test_build_intervention_gate_decision_payload_surfaces_recommended_bank_metadata(tmp_path: Path):
    from scripts.run_public_mainline import _build_intervention_gate_decision_payload

    decision_path = tmp_path / "gate_decision_reason.json"
    gate_payload = {
        "status": "success",
        "reason": "ok",
        "timestamp_utc": "2026-04-01T00:00:00Z",
        "summary_json": "/tmp/delta_summary.json",
        "log_path": "/tmp/intervention_gate.log",
        "eraser_target": "/tmp/eraser_bank_manifest.json",
        "recommended_bank_metadata": {
            "bank": "soft_recovery",
            "task_preserve": "head_subspace_soft",
            "preserve_strength": 0.75,
            "recovery_enabled": True,
            "recovery_source": "flag_defaults",
        },
        "verdict": {
            "passed": True,
            "reason": "ok",
            "selected_bank": "soft_recovery",
            "selected_row": {"bank": "soft_recovery", "score": 1.23},
            "thresholds": {
                "min_nonzero_logit_delta": 1e-5,
                "min_nonzero_prob_delta": 1e-6,
                "max_zero_logit_damage": 5e-3,
                "max_zero_prob_damage": 5e-4,
            },
        },
    }

    payload = _build_intervention_gate_decision_payload(
        gate_payload=gate_payload,
        decision_json_path=decision_path,
    )

    assert payload["decision_json"] == str(decision_path)
    assert payload["status"] == "success"
    assert payload["reason"] == "ok"
    assert payload["recommended_bank"] == "soft_recovery"
    assert payload["recommended_bank_metadata"]["task_preserve"] == "head_subspace_soft"
    assert abs(float(payload["recommended_bank_metadata"]["preserve_strength"]) - 0.75) < 1e-12
    assert payload["selected_row"]["bank"] == "soft_recovery"


def test_run_public_mainline_dry_run_exposes_intervention_gate_decision_artifact(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "public_mainline_out_gate_reason"
    dummy_head = tmp_path / "head.pt"
    dummy_head.write_bytes(b"stub")

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_public_mainline.py"),
        "--config",
        str(repo / "configs" / "mimic_public_acquisition_atomic.yaml"),
        "--out_dir",
        str(out_dir),
        "--dry_run",
        "--adjudicate_after_fit",
        "--adjudication_head",
        str(dummy_head),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(repo), env=env)

    plan_path = out_dir / "public_mainline_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert payload["artifact_paths"]["intervention_gate_decision"] == str(
        out_dir / "adjudication" / "gate_decision_reason.json"
    )
    assert payload["intervention_gate"]["decision_json"] == str(
        out_dir / "adjudication" / "gate_decision_reason.json"
    )


def test_selected_stage_required_artifacts_ignore_downstream_full_pipeline_outputs():
    from scripts.run_public_mainline import _compute_artifact_contract

    stage_results = [
        {
            "stage": "fit_eraser",
            "status": "success",
            "required_inputs": ["/tmp/probes.pt"],
            "expected_outputs": ["/tmp/eraser_bank_manifest.json"],
        }
    ]
    required_artifacts = [
        "/tmp/probes.pt",
        "/tmp/eraser_bank_manifest.json",
        "/tmp/confounding/confounding_summary.json",
        "/tmp/erasure/metrics.csv",
        "/tmp/safety/safety.csv",
        "/tmp/summary/public_mainline_summary.json",
    ]

    contract = _compute_artifact_contract(
        required_artifacts=required_artifacts,
        stage_results=stage_results,
    )

    assert contract["selected_required_artifacts"] == [
        "/tmp/probes.pt",
        "/tmp/eraser_bank_manifest.json",
    ]
    assert contract["missing_selected_artifacts"] == [
        "/tmp/probes.pt",
        "/tmp/eraser_bank_manifest.json",
    ]
    assert "/tmp/summary/public_mainline_summary.json" in contract["missing_full_artifacts"]


def test_compute_artifact_contract_partial_scope_can_be_selected_success(tmp_path: Path):
    from scripts.run_public_mainline import _compute_artifact_contract

    probes = tmp_path / "probes.pt"
    manifest = tmp_path / "eraser_bank_manifest.json"
    probes.write_bytes(b"ok")
    manifest.write_text("{}", encoding="utf-8")

    stage_results = [
        {
            "stage": "fit_eraser",
            "status": "success",
            "required_inputs": [str(probes)],
            "expected_outputs": [str(manifest)],
        }
    ]
    required_artifacts = [
        str(probes),
        str(manifest),
        str(tmp_path / "confounding" / "confounding_summary.json"),
        str(tmp_path / "erasure" / "metrics.csv"),
        str(tmp_path / "safety" / "safety.csv"),
        str(tmp_path / "summary" / "public_mainline_summary.json"),
    ]

    contract = _compute_artifact_contract(
        required_artifacts=required_artifacts,
        stage_results=stage_results,
    )

    assert contract["missing_selected_artifacts"] == []
    assert str(tmp_path / "summary" / "public_mainline_summary.json") in contract["missing_full_artifacts"]


def test_evaluate_summary_scientific_contract_accepts_ok_summary():
    from scripts.run_public_mainline import _evaluate_summary_scientific_contract

    payload = {
        "reporting_contract": {
            "status": "ok",
            "paper_tier_ready": True,
        },
        "policy_contract_status": "ok",
        "safety_contract_status": "ok",
    }

    verdict = _evaluate_summary_scientific_contract(payload)

    assert verdict["checked"] is True
    assert verdict["status"] == "ok"
    assert verdict["reason"] == "ok"
    assert verdict["paper_tier_ready"] is True
    assert verdict["reporting_contract_status"] == "ok"


def test_evaluate_summary_scientific_contract_rejects_incomplete_summary():
    from scripts.run_public_mainline import _evaluate_summary_scientific_contract

    payload = {
        "reporting_contract": {
            "status": "incomplete",
            "paper_tier_ready": False,
        },
        "policy_contract_status": "incomplete",
    }

    verdict = _evaluate_summary_scientific_contract(payload)

    assert verdict["checked"] is True
    assert verdict["status"] == "failed"
    assert "reporting_contract.status=incomplete" in verdict["reason"]
    assert "paper_tier_ready=False" in verdict["reason"]
    assert "policy_contract_status=incomplete" in verdict["reason"]


def test_resolve_scientific_contract_skips_selected_stage_scope(tmp_path: Path):
    from scripts.run_public_mainline import _resolve_scientific_contract

    summary_path = tmp_path / "public_mainline_summary.json"
    verdict = _resolve_scientific_contract(
        contract_scope="selected_stages",
        summary_path=summary_path,
    )

    assert verdict["checked"] is False
    assert verdict["status"] == "not_applicable"
    assert verdict["summary_path"] == str(summary_path)
