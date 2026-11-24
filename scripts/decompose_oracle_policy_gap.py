#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


REQUIRED_TRACE_COLUMNS = {
    "shift_family",
    "decision",
    "reason",
    "harm_no",
    "harm_eraser",
    "harm_recalibrated",
}
ROUTE_COLUMNS = ["p_none", "p_covariate", "p_prevalence", "p_mixed"]
CANONICAL_ACTIONS = ["abstain", "no_eraser", "apply_eraser", "recalibrate"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decompose oracle-vs-policy gap from existing safety artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--safety_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected YAML object at {path}, got {type(data).__name__}.")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object at {path}, got {type(data).__name__}.")
    return data


def _load_required_csv(path: Path, artifact_name: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact: {artifact_name} at {path}")
    return pd.read_csv(path)


def _load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_optional_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical_decision_name(value: Any) -> str | None:
    if not _nonempty_text(value):
        return None
    text = str(value).strip().lower()
    mapping = {
        "abstain": "abstain",
        "no_eraser": "no_eraser",
        "no-eraser": "no_eraser",
        "apply_eraser": "apply_eraser",
        "eraser": "apply_eraser",
        "recalibrate": "recalibrate",
        "recalibrated": "recalibrate",
    }
    return mapping.get(text)


def _to_finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _argmin_finite(mapping: Dict[str, float], allowed_actions: list[str] | None = None) -> tuple[str | None, float]:
    keys = allowed_actions if allowed_actions is not None else list(mapping.keys())
    best_action = None
    best_harm = float("nan")
    for action in keys:
        harm = _to_finite_or_nan(mapping.get(action))
        if not math.isfinite(harm):
            continue
        if best_action is None or harm < best_harm:
            best_action = action
            best_harm = harm
    return best_action, best_harm


def _route_proxy(row: pd.Series) -> tuple[str | None, float, float]:
    if not set(ROUTE_COLUMNS).issubset(set(row.index)):
        return None, float("nan"), float("nan")
    probs = np.asarray([_to_finite_or_nan(row[col]) for col in ROUTE_COLUMNS], dtype=float)
    if not np.isfinite(probs).any():
        return None, float("nan"), float("nan")
    probs = np.where(np.isfinite(probs), probs, 0.0)
    total = float(probs.sum())
    if total <= 0.0:
        return None, float("nan"), float("nan")
    probs = probs / total
    idx = int(np.argmax(probs))
    entropy = float(-np.sum([p * math.log(p) for p in probs if p > 0.0]))
    return ROUTE_COLUMNS[idx].replace("p_", ""), float(probs[idx]), entropy


def _mean_by(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if group_col not in df.columns or value_col not in df.columns:
        return out
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    for key, group in work.groupby(group_col, dropna=False):
        vals = group[value_col].dropna()
        if vals.empty:
            continue
        out[str(key)] = float(vals.mean())
    return out


def _markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Oracle Policy Gap",
        "",
        f"- analysis_contract_status: {summary['analysis_contract_status']}",
        f"- policy_contract_status: {summary['policy_contract_status']}",
        f"- paper_tier_ready: {str(summary['paper_tier_ready']).lower()}",
        f"- n_scenarios: {summary['n_scenarios']}",
        f"- chosen_equals_oracle: {summary['chosen_equals_oracle_count']} ({summary['chosen_equals_oracle_rate']:.3f})",
        f"- uncertainty_veto_count: {summary['uncertainty_veto_count']} ({summary['uncertainty_veto_rate']:.3f})",
        f"- eraser_oracle_count: {summary['eraser_oracle_count']}",
        f"- eraser_missed_count: {summary['eraser_missed_count']}",
        f"- recalibrate_oracle_count: {summary['recalibrate_oracle_count']}",
        f"- recalibrate_missed_count: {summary['recalibrate_missed_count']}",
        f"- mean_policy_gap: {summary['mean_policy_gap']:.6f}",
        "",
        "## Warnings",
    ]
    if summary["warnings"]:
        lines.extend(f"- {item}" for item in summary["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parser().parse_args()
    print(
        f"[decompose_oracle_policy_gap] stage_start config={args.config} safety_dir={args.safety_dir} out_dir={args.out_dir}",
        flush=True,
    )

    config_path = Path(args.config).expanduser().resolve()
    safety_dir = Path(args.safety_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    tier = str(cfg.get("experiment", {}).get("tier", "dev"))
    abstain_cost = cfg.get("safety_policy", {}).get("abstain_cost")
    if abstain_cost is None:
        raise RuntimeError("config.safety_policy.abstain_cost is required for oracle-policy gap analysis.")
    abstain_cost = float(abstain_cost)

    scenario_trace_path = safety_dir / "scenario_trace.csv"
    policy_json_path = safety_dir / "policy.json"
    policy_decisions_path = safety_dir / "policy_decisions.csv"
    regret_summary_path = safety_dir / "policy_regret_summary.json"
    shift_id_report_path = safety_dir / "shift_id_report.json"

    trace_df = _load_required_csv(scenario_trace_path, "scenario_trace.csv")
    missing_cols = sorted(REQUIRED_TRACE_COLUMNS - set(trace_df.columns))
    if missing_cols:
        raise RuntimeError(f"scenario_trace.csv is missing required columns: {missing_cols}")

    if not policy_json_path.exists():
        raise RuntimeError(f"Missing required artifact: policy.json at {policy_json_path}")
    policy_json = _load_json(policy_json_path)
    policy_contract_status = str(policy_json.get("contract_status", "incomplete"))

    warnings: list[str] = []
    if tier == "paper" and policy_contract_status != "ok":
        raise RuntimeError(
            f"decompose_oracle_policy_gap requires policy.json contract_status='ok' for experiment.tier=paper; got {policy_contract_status!r}."
        )
    if tier != "paper" and policy_contract_status != "ok":
        warnings.append(f"policy_contract:{policy_contract_status}")

    policy_decisions_df = _load_optional_csv(policy_decisions_path)
    regret_summary = _load_optional_json(regret_summary_path)
    shift_id_report = _load_optional_json(shift_id_report_path)
    if policy_decisions_df is not None and len(policy_decisions_df) != len(trace_df):
        warnings.append("policy_decisions_row_count_mismatch")
    if regret_summary is not None and "regret_mean" not in regret_summary:
        warnings.append("policy_regret_summary_missing_regret_mean")
    if shift_id_report is not None and not isinstance(shift_id_report, dict):
        warnings.append("shift_id_report_invalid")

    rows: list[dict[str, Any]] = []
    unknown_seen = False
    route_available = set(ROUTE_COLUMNS).issubset(set(trace_df.columns))
    for idx, row in trace_df.iterrows():
        harm_mapping = {
            "abstain": float(abstain_cost),
            "no_eraser": _to_finite_or_nan(row["harm_no"]),
            "apply_eraser": _to_finite_or_nan(row["harm_eraser"]),
            "recalibrate": _to_finite_or_nan(row["harm_recalibrated"]),
        }
        oracle_action, oracle_harm = _argmin_finite(harm_mapping, CANONICAL_ACTIONS)
        best_nonabstain_action, best_nonabstain_harm = _argmin_finite(
            harm_mapping,
            ["no_eraser", "apply_eraser", "recalibrate"],
        )

        chosen_action = canonical_decision_name(row["decision"])
        decision_unknown = chosen_action is None
        if decision_unknown:
            unknown_seen = True
            chosen_action = "unknown"
            chosen_harm = float("nan")
        else:
            chosen_harm = _to_finite_or_nan(harm_mapping.get(chosen_action))

        predicted_route = None
        predicted_route_confidence = float("nan")
        predicted_route_entropy = float("nan")
        if route_available:
            predicted_route, predicted_route_confidence, predicted_route_entropy = _route_proxy(row)

        policy_gap = float("nan")
        if math.isfinite(chosen_harm) and math.isfinite(oracle_harm):
            policy_gap = float(chosen_harm - oracle_harm)

        reason = str(row["reason"])
        shift_family = str(row["shift_family"])
        uncertainty_veto = bool(
            chosen_action == "abstain"
            and reason == "high_uncertainty"
            and oracle_action is not None
            and oracle_action != "abstain"
        )
        eraser_oracle = bool(oracle_action == "apply_eraser")
        eraser_missed = bool(eraser_oracle and chosen_action != "apply_eraser")
        recalibrate_oracle = bool(oracle_action == "recalibrate")
        recalibrate_missed = bool(recalibrate_oracle and chosen_action != "recalibrate")
        oracle_matches_decision = bool(chosen_action == oracle_action)

        rows.append(
            {
                "shift_family": shift_family,
                "decision": str(row["decision"]),
                "reason": reason,
                "chosen_action": chosen_action,
                "chosen_harm": chosen_harm,
                "oracle_action": oracle_action,
                "oracle_harm": oracle_harm,
                "policy_gap": policy_gap,
                "decision_unknown": decision_unknown,
                "uncertainty_veto": uncertainty_veto,
                "eraser_oracle": eraser_oracle,
                "eraser_missed": eraser_missed,
                "recalibrate_oracle": recalibrate_oracle,
                "recalibrate_missed": recalibrate_missed,
                "oracle_matches_decision": oracle_matches_decision,
                "best_nonabstain_action": best_nonabstain_action,
                "best_nonabstain_harm": best_nonabstain_harm,
                "predicted_route": predicted_route,
                "predicted_route_confidence": predicted_route_confidence,
                "predicted_route_entropy": predicted_route_entropy,
            }
        )

    if unknown_seen:
        warnings.append("decision_unknown")

    out_trace = pd.DataFrame(rows)
    out_trace.to_csv(out_dir / "oracle_policy_gap_trace.csv", index=False)
    print(f"[decompose_oracle_policy_gap] trace_saved path={out_dir / 'oracle_policy_gap_trace.csv'}", flush=True)

    policy_gap_vals = pd.to_numeric(out_trace["policy_gap"], errors="coerce").dropna()
    oracle_action_counts = {
        str(k): int(v) for k, v in out_trace["oracle_action"].fillna("unknown").astype(str).value_counts().to_dict().items()
    }
    decision_counts = {
        str(k): int(v) for k, v in out_trace["chosen_action"].fillna("unknown").astype(str).value_counts().to_dict().items()
    }
    summary: Dict[str, Any] = {
        "analysis_contract_status": "incomplete" if warnings else "ok",
        "policy_contract_status": policy_contract_status,
        "paper_tier_ready": bool(policy_contract_status == "ok" and not warnings),
        "abstain_cost": float(abstain_cost),
        "n_scenarios": int(len(out_trace)),
        "oracle_action_counts": oracle_action_counts,
        "decision_counts": decision_counts,
        "chosen_equals_oracle_count": int(out_trace["oracle_matches_decision"].astype(bool).sum()),
        "chosen_equals_oracle_rate": float(out_trace["oracle_matches_decision"].astype(bool).mean()),
        "oracle_abstain_count": int((out_trace["oracle_action"] == "abstain").sum()),
        "uncertainty_veto_count": int(out_trace["uncertainty_veto"].astype(bool).sum()),
        "uncertainty_veto_rate": float(out_trace["uncertainty_veto"].astype(bool).mean()),
        "eraser_oracle_count": int(out_trace["eraser_oracle"].astype(bool).sum()),
        "eraser_missed_count": int(out_trace["eraser_missed"].astype(bool).sum()),
        "recalibrate_oracle_count": int(out_trace["recalibrate_oracle"].astype(bool).sum()),
        "recalibrate_missed_count": int(out_trace["recalibrate_missed"].astype(bool).sum()),
        "mean_policy_gap": float(policy_gap_vals.mean()) if not policy_gap_vals.empty else float("nan"),
        "max_policy_gap": float(policy_gap_vals.max()) if not policy_gap_vals.empty else float("nan"),
        "mean_policy_gap_by_reason": _mean_by(out_trace, "reason", "policy_gap"),
        "mean_policy_gap_by_shift_family": _mean_by(out_trace, "shift_family", "policy_gap"),
        "warnings": warnings,
    }
    if route_available:
        summary["predicted_route_counts"] = {
            str(k): int(v)
            for k, v in out_trace["predicted_route"].fillna("unknown").astype(str).value_counts().to_dict().items()
        }
        summary["mean_predicted_route_confidence_by_reason"] = _mean_by(
            out_trace,
            "reason",
            "predicted_route_confidence",
        )

    if regret_summary is not None and "regret_mean" in regret_summary:
        regret_mean = _to_finite_or_nan(regret_summary.get("regret_mean"))
        if math.isfinite(regret_mean) and math.isfinite(summary["mean_policy_gap"]):
            if abs(regret_mean - float(summary["mean_policy_gap"])) > 1e-9:
                warnings.append("policy_regret_summary_mean_mismatch")
                summary["analysis_contract_status"] = "incomplete"
                summary["paper_tier_ready"] = False
                summary["warnings"] = warnings

    (out_dir / "oracle_policy_gap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[decompose_oracle_policy_gap] summary_saved path={out_dir / 'oracle_policy_gap_summary.json'}", flush=True)
    (out_dir / "oracle_policy_gap.md").write_text(_markdown_summary(summary), encoding="utf-8")
    print(f"[decompose_oracle_policy_gap] markdown_saved path={out_dir / 'oracle_policy_gap.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"[decompose_oracle_policy_gap] failure_context argv={' '.join(__import__('sys').argv[1:])} error={type(exc).__name__}:{exc}",
            flush=True,
        )
        raise
