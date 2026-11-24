#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from phystta.config.loader import load_config
from phystta.utils.report_contract import contract_status


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _artifact_status(paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "exists": bool(path.exists()),
        }
        for name, path in paths.items()
    }


def _collect_nonfinite_json_fields(value: Any, prefix: str) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            out.extend(_collect_nonfinite_json_fields(nested, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        for idx, nested in enumerate(value):
            out.extend(_collect_nonfinite_json_fields(nested, f"{prefix}[{idx}]"))
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            out.append(prefix)
    return out


def _collect_nonfinite_csv_fields(df: pd.DataFrame | None, artifact_name: str) -> list[str]:
    if df is None or df.empty:
        return []
    out: list[str] = []
    for col in df.columns:
        series = df[col]
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() == 0:
            continue
        mask = series.notna() & ~np.isfinite(converted)
        if bool(mask.any()):
            out.append(f"{artifact_name}.{col}")
    return out


def _top_confounding_rows(df: pd.DataFrame | None, limit: int = 8) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    score_col = None
    for candidate in ["auroc_macro", "auroc", "balanced_accuracy", "macro_f1"]:
        if candidate in df.columns:
            score_col = candidate
            break
    if score_col is None:
        return df.head(limit).to_dict(orient="records")
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.sort_values(score_col, ascending=False, na_position="last")
    cols = [c for c in ["audit_family", "target", "context_key", score_col, f"{score_col}_ci_low", f"{score_col}_ci_high"] if c in work.columns]
    if not cols:
        cols = list(work.columns[: min(len(work.columns), 6)])
    return work[cols].head(limit).to_dict(orient="records")


def _guarded_summary(df: pd.DataFrame | None) -> Dict[str, Any]:
    if df is None or df.empty or "diagnostic" not in df.columns:
        return {}
    out: Dict[str, Any] = {}
    for diag in ["probe_spuriousness_summary", "eraser_stability_summary"]:
        rows = df[df["diagnostic"].astype(str) == diag]
        if rows.empty:
            continue
        rec = rows.iloc[0].to_dict()
        out[diag] = {str(k): v for k, v in rec.items() if k not in {"diagnostic"}}
    return out


def _erasure_delta_summary(df: pd.DataFrame | None) -> Dict[str, Any]:
    if df is None or df.empty or "method" not in df.columns:
        return {}
    work = df[df["method"].isin(["no_eraser", "eraser"])].copy()
    if work.empty:
        return {}
    index_cols = [c for c in ["shift", "realism_level", "intensity", "profile"] if c in work.columns]
    pivot = work.pivot_table(index=index_cols, columns="method", aggfunc="mean")
    out: Dict[str, Any] = {}
    for metric in ["auroc_macro", "auprc_macro", "ece", "brier", "risk_at_coverage", "expected_harm", "coverage_realized", "abstain_rate"]:
        if (metric, "no_eraser") not in pivot.columns or (metric, "eraser") not in pivot.columns:
            continue
        delta = pivot[(metric, "eraser")] - pivot[(metric, "no_eraser")]
        delta = pd.to_numeric(delta, errors="coerce").dropna()
        if delta.empty:
            continue
        out[metric] = {
            "mean_delta_eraser_minus_no": float(delta.mean()),
            "min_delta_eraser_minus_no": float(delta.min()),
            "max_delta_eraser_minus_no": float(delta.max()),
        }
    return out


def _safety_summary(safety_df: pd.DataFrame | None, trace_df: pd.DataFrame | None, regret_json: Dict[str, Any] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if safety_df is not None and not safety_df.empty:
        for col in ["coverage", "risk_at_coverage", "harm", "policy_regret"]:
            if col in safety_df.columns:
                vals = pd.to_numeric(safety_df[col], errors="coerce").dropna()
                if not vals.empty:
                    out[col] = {
                        "mean": float(vals.mean()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    }
    if trace_df is not None and not trace_df.empty and "decision" in trace_df.columns:
        out["decision_counts"] = {
            str(k): int(v) for k, v in trace_df["decision"].astype(str).value_counts().to_dict().items()
        }
    if regret_json:
        out["policy_regret_summary"] = regret_json
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build bounded public-mainline summary bundle.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--realism_dir", required=True)
    ap.add_argument("--confounding_dir", required=True)
    ap.add_argument("--erasure_dir", required=True)
    ap.add_argument("--safety_dir", required=True)
    ap.add_argument("--launcher_root", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    realism_dir = Path(args.realism_dir).expanduser()
    confounding_dir = Path(args.confounding_dir).expanduser()
    erasure_dir = Path(args.erasure_dir).expanduser()
    safety_dir = Path(args.safety_dir).expanduser()
    launcher_root = Path(args.launcher_root).expanduser() if args.launcher_root else None

    artifact_paths = {
        "realism_calibrator": realism_dir / "realism_calibrator.json",
        "realism_report": realism_dir / "realism_calibrator.report.json",
        "pair_quality_audit": realism_dir / "pair_quality_audit.json",
        "confounding_summary": confounding_dir / "confounding_summary.json",
        "confounding_predictability": confounding_dir / "confounding_predictability.csv",
        "shift_id_proxy_audit": confounding_dir / "shift_id_proxy_audit.csv",
        "technical_pair_deltas": confounding_dir / "technical_pair_pathology_deltas.csv",
        "erasure_metrics": erasure_dir / "metrics.csv",
        "erasure_guarded": erasure_dir / "guarded_erasure_diagnostics.csv",
        "erasure_tradeoff": erasure_dir / "erasure_tradeoff.csv",
        "risk_coverage_curves": erasure_dir / "risk_coverage_curves.csv",
        "safety_metrics": safety_dir / "safety.csv",
        "scenario_trace": safety_dir / "scenario_trace.csv",
        "policy_bootstrap": safety_dir / "policy_bootstrap.csv",
        "policy_json": safety_dir / "policy.json",
        "policy_thresholds": safety_dir / "policy_thresholds.json",
        "policy_operating_points_summary": safety_dir / "policy_operating_points_summary.json",
        "policy_operating_points_targeted_summary": safety_dir / "policy_operating_points_targeted_summary.json",
        "policy_decisions": safety_dir / "policy_decisions.csv",
        "policy_regret_summary": safety_dir / "policy_regret_summary.json",
    }
    launcher_artifacts = {}
    if launcher_root is not None:
        launcher_artifacts = {
            "preflight": launcher_root / "public_mainline_preflight.json",
            "plan": launcher_root / "public_mainline_plan.json",
            "manifest": launcher_root / "public_mainline_manifest.json",
            "effective_config": launcher_root / "public_mainline.effective.yaml",
            "stage_state_dir": launcher_root / ".stage_state",
            "launcher_logs_dir": launcher_root / "launcher_logs",
        }
    artifact_status = _artifact_status(artifact_paths)
    launcher_artifact_status = _artifact_status(launcher_artifacts) if launcher_artifacts else {}

    realism_report = _safe_read_json(artifact_paths["realism_report"])
    pair_quality_audit = _safe_read_json(artifact_paths["pair_quality_audit"])
    confounding_summary = _safe_read_json(artifact_paths["confounding_summary"])
    confounding_predictability = _safe_read_csv(artifact_paths["confounding_predictability"])
    shift_id_proxy = _safe_read_csv(artifact_paths["shift_id_proxy_audit"])
    pair_deltas = _safe_read_csv(artifact_paths["technical_pair_deltas"])
    erasure_metrics = _safe_read_csv(artifact_paths["erasure_metrics"])
    guarded_df = _safe_read_csv(artifact_paths["erasure_guarded"])
    tradeoff_df = _safe_read_csv(artifact_paths["erasure_tradeoff"])
    risk_curve_df = _safe_read_csv(artifact_paths["risk_coverage_curves"])
    safety_df = _safe_read_csv(artifact_paths["safety_metrics"])
    trace_df = _safe_read_csv(artifact_paths["scenario_trace"])
    bootstrap_df = _safe_read_csv(artifact_paths["policy_bootstrap"])
    policy_json = _safe_read_json(artifact_paths["policy_json"])
    policy_thresholds = _safe_read_json(artifact_paths["policy_thresholds"])
    policy_op_summary = _safe_read_json(artifact_paths["policy_operating_points_summary"])
    policy_op_targeted_summary = _safe_read_json(artifact_paths["policy_operating_points_targeted_summary"])
    policy_decisions_df = _safe_read_csv(artifact_paths["policy_decisions"])
    regret_json = _safe_read_json(artifact_paths["policy_regret_summary"])

    reporting_artifact_names = [
        "policy_json",
        "policy_thresholds",
        "policy_operating_points_summary",
        "policy_operating_points_targeted_summary",
        "policy_decisions",
        "policy_regret_summary",
    ]
    missing_reporting_artifacts = [
        name for name in reporting_artifact_names if not artifact_status.get(name, {}).get("exists", False)
    ]
    policy_contract_status = str((policy_json or {}).get("contract_status", "incomplete"))
    policy_incomplete_fields = [
        str(v) for v in (policy_json or {}).get("incomplete_fields", []) if str(v)
    ]
    decision_label_source = str((policy_json or {}).get("decision_label_source", "predicted"))
    decision_label_diagnostic_only = bool((policy_json or {}).get("decision_label_diagnostic_only", False))
    critical_nan_fields = sorted(
        set(
            _collect_nonfinite_json_fields(policy_json, "policy_json")
            + _collect_nonfinite_json_fields(policy_thresholds, "policy_thresholds")
            + _collect_nonfinite_json_fields(policy_op_summary, "policy_operating_points_summary")
            + _collect_nonfinite_json_fields(
                policy_op_targeted_summary,
                "policy_operating_points_targeted_summary",
            )
            + _collect_nonfinite_json_fields(regret_json, "policy_regret_summary")
            + _collect_nonfinite_csv_fields(policy_decisions_df, "policy_decisions")
        )
    )
    reporting_incomplete_fields = sorted(
        set(
            policy_incomplete_fields
            + [f"missing:{name}" for name in missing_reporting_artifacts]
            + [f"critical_nan:{name}" for name in critical_nan_fields]
            + ([] if policy_contract_status == "ok" else [f"policy_contract:{policy_contract_status}"])
            + ([] if decision_label_source == "predicted" else [f"decision_label_source:{decision_label_source}"])
            + ([] if not decision_label_diagnostic_only else ["decision_label_diagnostic_only"])
        )
    )
    reporting_contract_payload = {
        "status": contract_status([], reporting_incomplete_fields),
        "policy_contract_status": policy_contract_status,
        "incomplete_fields": reporting_incomplete_fields,
        "critical_nan_fields": critical_nan_fields,
        "decision_label_source": decision_label_source,
        "decision_label_diagnostic_only": bool(decision_label_diagnostic_only),
        "paper_tier_ready": bool(
            not missing_reporting_artifacts
            and policy_contract_status == "ok"
            and not critical_nan_fields
            and decision_label_source == "predicted"
            and not decision_label_diagnostic_only
        ),
        "missing_reporting_artifacts": missing_reporting_artifacts,
    }

    summary = {
        "config_path": str(Path(args.config).expanduser()),
        "canonical_contract": "docs/canonical_public_mimic_experiment_contract.md",
        "artifact_status": artifact_status,
        "artifact_integrity_pass": bool(all(cell["exists"] for cell in artifact_status.values())),
        "launcher_artifact_status": launcher_artifact_status,
        "reporting_contract": reporting_contract_payload,
        "experiment": {
            "pair_mode": str(getattr(cfg.intervention, "pair_mode", "unknown")),
            "scenario_mode": str(getattr(cfg.erasure_eval, "scenario_mode", "unknown")),
            "shift_families": list(cfg.intervention.shift_types),
            "realism_levels": list(getattr(cfg.intervention, "realism_levels", [])),
            "canonical_calibrated_shift_set": list(cfg.erasure_eval.shift_types),
            "ready_but_not_in_canonical_calibrated_set": [s for s in ["collimation_crop"] if s not in set(cfg.erasure_eval.shift_types)],
        },
        "realism_evidence": {
            "realism_validity_summary": (realism_report or {}).get("realism_validity_summary"),
            "pair_quality_counts": (realism_report or {}).get("pair_quality_counts"),
            "pair_quality_audit": pair_quality_audit,
            "physical_dependence_warnings": (realism_report or {}).get("physical_dependence_warnings", []),
        },
        "confounding_evidence": {
            "summary": confounding_summary,
            "top_predictability_rows": _top_confounding_rows(confounding_predictability),
            "top_shift_id_proxy_rows": _top_confounding_rows(shift_id_proxy),
            "top_pair_delta_rows": _top_confounding_rows(pair_deltas),
        },
        "guarded_erasure_evidence": {
            "summary_rows": _guarded_summary(guarded_df),
            "tradeoff_rows": int(len(tradeoff_df)) if tradeoff_df is not None else 0,
        },
        "erasure_eval_evidence": {
            "delta_summary": _erasure_delta_summary(erasure_metrics),
            "risk_curve_rows": int(len(risk_curve_df)) if risk_curve_df is not None else 0,
        },
        "safety_evidence": _safety_summary(safety_df, trace_df, regret_json),
    }
    if launcher_root is not None:
        summary["launcher_bundle"] = {
            "root": str(launcher_root),
            "preflight": _safe_read_json(launcher_artifacts["preflight"]),
            "plan": _safe_read_json(launcher_artifacts["plan"]),
            "manifest": _safe_read_json(launcher_artifacts["manifest"]),
        }
    if bootstrap_df is not None and not bootstrap_df.empty:
        boot_cols = [c for c in ["policy_harm_ci_low", "policy_harm_ci_high", "policy_regret_ci_low", "policy_regret_ci_high"] if c in bootstrap_df.columns]
        if boot_cols:
            summary["safety_evidence"]["bootstrap_columns"] = boot_cols
    if str(getattr(getattr(cfg, "experiment", None), "tier", "dev")) == "paper" and reporting_contract_payload["status"] != "ok":
        raise RuntimeError(
            "build_public_mainline_summary requires reporting_contract.status='ok' for experiment.tier=paper"
        )

    artifact_index = {
        "realism_dir": str(realism_dir),
        "confounding_dir": str(confounding_dir),
        "erasure_dir": str(erasure_dir),
        "safety_dir": str(safety_dir),
        "summary_dir": str(out_dir),
    }
    if launcher_root is not None:
        artifact_index["launcher_root"] = str(launcher_root)
    (out_dir / "public_mainline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "artifact_index.json").write_text(json.dumps(artifact_index, indent=2), encoding="utf-8")
    print(f"Saved bounded public-mainline summary to {out_dir}")


if __name__ == "__main__":
    main()
