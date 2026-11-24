#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


ROUTE_COLUMNS = ["p_none", "p_covariate", "p_prevalence", "p_mixed"]
CANONICAL_ACTIONS = ["abstain", "no_eraser", "apply_eraser", "recalibrate"]
REQUIRED_TRACE_COLUMNS = {"decision", "reason", "harm_no", "harm_eraser", "harm_recalibrated"}


def _progress(message: str) -> None:
    print(f"[policy-v2-replay] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay scaffold for policy-v2 selectors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--safety_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--uncertainty_lambda_grid", default="0.0,0.1,0.25,0.5")
    parser.add_argument("--route_gamma_grid", default="0.0,0.1,0.25")
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


def _parse_grid(text: str, field_name: str) -> List[float]:
    values = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    if not values:
        raise RuntimeError(f"{field_name} must contain at least one float.")
    out = []
    for value in values:
        try:
            out.append(float(value))
        except Exception as exc:
            raise RuntimeError(f"{field_name} contains non-float value {value!r}.") from exc
    return out


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


def _argmin_finite(mapping: Dict[str, float]) -> tuple[str | None, float]:
    best_action = None
    best_value = float("nan")
    for action in CANONICAL_ACTIONS:
        value = _to_finite_or_nan(mapping.get(action))
        if not math.isfinite(value):
            continue
        if best_action is None or value < best_value:
            best_action = action
            best_value = value
    return best_action, best_value


def _oracle_from_mapping(mapping: Dict[str, float]) -> tuple[str | None, float]:
    return _argmin_finite(mapping)


def _maybe_route_penalty(prob_value: float, route_gamma: float) -> float:
    if not math.isfinite(prob_value):
        return 0.0
    return float(route_gamma * (1.0 - prob_value))


def _mean_of(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _selector_key(selector_name: str, uncertainty_lambda: float, route_gamma: float) -> tuple[str, float, float]:
    return (str(selector_name), float(uncertainty_lambda), float(route_gamma))


def _selector_record(frame: pd.DataFrame, selector_name: str, uncertainty_lambda: float, route_gamma: float) -> Dict[str, Any]:
    chosen_equals_oracle = (frame["chosen_action"].astype(str) == frame["oracle_action"].astype(str)).astype(float)
    current_uncertainty_veto = frame["current_uncertainty_veto"].astype(bool)
    replay_nonabstain = frame["chosen_action"].astype(str) != "abstain"
    return {
        "selector_name": selector_name,
        "uncertainty_lambda": float(uncertainty_lambda),
        "route_gamma": float(route_gamma),
        "n_rows": int(len(frame)),
        "mean_harm": _mean_of(frame["chosen_cost"]),
        "mean_policy_gap_vs_oracle": _mean_of(frame["policy_gap_vs_oracle"]),
        "mean_harm_improvement_vs_current": _mean_of(frame["harm_improvement_vs_current"]),
        "beats_current_rate": float(frame["beats_current_harm"].astype(float).mean()),
        "abstain_rate": float(frame["abstains"].astype(float).mean()),
        "eraser_rate": float(frame["uses_eraser"].astype(float).mean()),
        "recalibrate_rate": float(frame["uses_recalibrate"].astype(float).mean()),
        "chosen_equals_oracle_rate": float(chosen_equals_oracle.mean()),
        "uncertainty_veto_reduction_rate": float((current_uncertainty_veto & replay_nonabstain).astype(float).mean()),
    }


def _markdown_summary(summary: Dict[str, Any]) -> str:
    current = summary["current_policy_summary"]
    best_harm = summary["best_selector_by_mean_harm"]
    best_gap = summary["best_selector_by_mean_policy_gap"]
    lines = [
        "# Policy V2 Replay",
        "",
        f"- analysis_contract_status: {summary['analysis_contract_status']}",
        f"- policy_contract_status: {summary['policy_contract_status']}",
        f"- paper_tier_ready: {str(summary['paper_tier_ready']).lower()}",
        f"- current_policy_mean_harm: {current['mean_harm']:.6f}",
        f"- best_selector_by_mean_harm: {best_harm['selector_name']} "
        f"(lambda={best_harm['uncertainty_lambda']}, gamma={best_harm['route_gamma']}, mean_harm={best_harm['mean_harm']:.6f})",
        f"- best_selector_by_mean_policy_gap: {best_gap['selector_name']} "
        f"(lambda={best_gap['uncertainty_lambda']}, gamma={best_gap['route_gamma']}, mean_gap={best_gap['mean_policy_gap_vs_oracle']:.6f})",
        f"- best_selector_abstain_delta_vs_current: {current['abstain_rate'] - best_harm['abstain_rate']:.6f}",
        f"- best_selector_uncertainty_veto_reduction_rate: {best_harm['uncertainty_veto_reduction_rate']:.6f}",
        f"- promising_selector_exists: {str(summary['promising_selector_exists']).lower()}",
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

    config_path = Path(args.config).expanduser().resolve()
    safety_dir = Path(args.safety_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"start config={config_path} safety_dir={safety_dir} out_dir={out_dir}")

    cfg = _load_yaml(config_path)
    tier = str(cfg.get("experiment", {}).get("tier", "dev"))
    abstain_cost = cfg.get("safety_policy", {}).get("abstain_cost")
    if abstain_cost is None:
        raise RuntimeError("config.safety_policy.abstain_cost is required for policy-v2 replay.")
    abstain_cost = float(abstain_cost)

    uncertainty_lambda_grid = _parse_grid(args.uncertainty_lambda_grid, "uncertainty_lambda_grid")
    route_gamma_grid = _parse_grid(args.route_gamma_grid, "route_gamma_grid")

    trace_df = _load_required_csv(safety_dir / "scenario_trace.csv", "scenario_trace.csv")
    missing_cols = sorted(REQUIRED_TRACE_COLUMNS - set(trace_df.columns))
    if missing_cols:
        raise RuntimeError(f"scenario_trace.csv is missing required columns: {missing_cols}")
    _progress(f"loaded scenario_trace rows={len(trace_df)}")

    policy_json_path = safety_dir / "policy.json"
    if not policy_json_path.exists():
        raise RuntimeError(f"Missing required artifact: policy.json at {policy_json_path}")
    policy_json = _load_json(policy_json_path)
    policy_contract_status = str(policy_json.get("contract_status", "incomplete"))

    warnings: list[str] = []
    if tier == "paper" and policy_contract_status != "ok":
        raise RuntimeError(
            f"replay_policy_v2 requires policy.json contract_status='ok' for experiment.tier=paper; got {policy_contract_status!r}."
        )
    if tier != "paper" and policy_contract_status != "ok":
        warnings.append(f"policy_contract:{policy_contract_status}")
    _progress(f"policy_contract_status={policy_contract_status} tier={tier}")

    oracle_trace_df = _load_optional_csv(safety_dir / "oracle_policy_gap_trace.csv")
    oracle_summary = _load_optional_json(safety_dir / "oracle_policy_gap_summary.json")
    if oracle_summary is not None:
        n_scenarios = oracle_summary.get("n_scenarios")
        if n_scenarios is not None and int(n_scenarios) != int(len(trace_df)):
            warnings.append("oracle_policy_gap_summary_n_scenarios_mismatch")

    use_oracle_trace = bool(
        oracle_trace_df is not None
        and len(oracle_trace_df) == len(trace_df)
        and {"oracle_action", "oracle_harm"}.issubset(set(oracle_trace_df.columns))
    )
    if oracle_trace_df is not None and not use_oracle_trace:
        warnings.append("oracle_policy_gap_trace_unusable")

    unknown_decision_seen = False
    uncertainty_missing = "uncertainty" not in trace_df.columns
    if uncertainty_missing:
        warnings.append("uncertainty_missing")
    route_available = {"p_covariate", "p_prevalence"}.issubset(set(trace_df.columns))

    replay_rows: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    _progress(
        "grid_search "
        + f"uncertainty_lambdas={len(uncertainty_lambda_grid)} route_gammas={len(route_gamma_grid)}"
    )
    for idx, row in trace_df.iterrows():
        shift_family = str(row["shift_family"]) if "shift_family" in row else "unknown"
        reason = str(row["reason"])
        uncertainty = _to_finite_or_nan(row["uncertainty"]) if "uncertainty" in row else 0.0
        p_covariate = _to_finite_or_nan(row["p_covariate"]) if "p_covariate" in row else float("nan")
        p_prevalence = _to_finite_or_nan(row["p_prevalence"]) if "p_prevalence" in row else float("nan")

        harm_mapping = {
            "abstain": float(abstain_cost),
            "no_eraser": _to_finite_or_nan(row["harm_no"]),
            "apply_eraser": _to_finite_or_nan(row["harm_eraser"]),
            "recalibrate": _to_finite_or_nan(row["harm_recalibrated"]),
        }

        if use_oracle_trace:
            oracle_action = canonical_decision_name(oracle_trace_df.iloc[idx]["oracle_action"])
            oracle_harm = _to_finite_or_nan(oracle_trace_df.iloc[idx]["oracle_harm"])
            if oracle_action is None or not math.isfinite(oracle_harm):
                oracle_action, oracle_harm = _oracle_from_mapping(harm_mapping)
        else:
            oracle_action, oracle_harm = _oracle_from_mapping(harm_mapping)

        current_action = canonical_decision_name(row["decision"])
        if current_action is None:
            current_action = "unknown"
            unknown_decision_seen = True
            current_cost = float("nan")
        else:
            current_cost = _to_finite_or_nan(harm_mapping.get(current_action))
        current_gap = (
            float(current_cost - oracle_harm)
            if math.isfinite(current_cost) and math.isfinite(oracle_harm)
            else float("nan")
        )
        current_uncertainty_veto = bool(
            current_action == "abstain"
            and reason == "high_uncertainty"
            and oracle_action is not None
            and oracle_action != "abstain"
        )

        current_rows.append(
            {
                "row_id": idx,
                "shift_family": shift_family,
                "reason": reason,
                "current_action": current_action,
                "current_cost": current_cost,
                "current_policy_gap_vs_oracle": current_gap,
                "current_uncertainty_veto": current_uncertainty_veto,
                "oracle_action": oracle_action,
                "oracle_harm": oracle_harm,
            }
        )

        selector_specs: list[tuple[str, float, float, Dict[str, float]]] = []
        selector_specs.append(("current_policy", 0.0, 0.0, dict(harm_mapping)))
        selector_specs.append(("expected_harm_no_veto", 0.0, 0.0, dict(harm_mapping)))
        for uncertainty_lambda in uncertainty_lambda_grid:
            for route_gamma in route_gamma_grid:
                premium = float(uncertainty_lambda) * (uncertainty if math.isfinite(uncertainty) else 0.0)
                selector_specs.append(
                    (
                        "expected_harm_with_risk_premium",
                        float(uncertainty_lambda),
                        float(route_gamma),
                        {
                            "abstain": float(abstain_cost),
                            "no_eraser": (
                                harm_mapping["no_eraser"] + premium
                                if math.isfinite(harm_mapping["no_eraser"])
                                else float("nan")
                            ),
                            "apply_eraser": (
                                harm_mapping["apply_eraser"]
                                + premium
                                + _maybe_route_penalty(p_covariate, float(route_gamma))
                                if math.isfinite(harm_mapping["apply_eraser"])
                                else float("nan")
                            ),
                            "recalibrate": (
                                harm_mapping["recalibrate"]
                                + premium
                                + _maybe_route_penalty(p_prevalence, float(route_gamma))
                                if math.isfinite(harm_mapping["recalibrate"])
                                else float("nan")
                            ),
                        },
                    )
                )

        for selector_name, uncertainty_lambda, route_gamma, selector_mapping in selector_specs:
            if selector_name == "current_policy":
                chosen_action = current_action
                chosen_cost = current_cost
            else:
                chosen_action, chosen_cost = _argmin_finite(selector_mapping)
                if chosen_action is None:
                    chosen_action = "unknown"
            chosen_gap = (
                float(chosen_cost - oracle_harm)
                if math.isfinite(chosen_cost) and math.isfinite(oracle_harm)
                else float("nan")
            )
            replay_rows.append(
                {
                    "row_id": idx,
                    "shift_family": shift_family,
                    "reason": reason,
                    "selector_name": selector_name,
                    "uncertainty_lambda": float(uncertainty_lambda),
                    "route_gamma": float(route_gamma),
                    "chosen_action": chosen_action,
                    "chosen_cost": chosen_cost,
                    "oracle_action": oracle_action,
                    "oracle_harm": oracle_harm,
                    "policy_gap_vs_oracle": chosen_gap,
                    "current_action": current_action,
                    "current_cost": current_cost,
                    "current_policy_gap_vs_oracle": current_gap,
                    "current_uncertainty_veto": current_uncertainty_veto,
                    "improves_over_current": bool(
                        math.isfinite(chosen_gap) and math.isfinite(current_gap) and chosen_gap < current_gap
                    ),
                    "beats_current_harm": bool(
                        math.isfinite(chosen_cost) and math.isfinite(current_cost) and chosen_cost < current_cost
                    ),
                    "harm_improvement_vs_current": (
                        float(current_cost - chosen_cost)
                        if math.isfinite(chosen_cost) and math.isfinite(current_cost)
                        else float("nan")
                    ),
                    "abstains": bool(chosen_action == "abstain"),
                    "uses_eraser": bool(chosen_action == "apply_eraser"),
                    "uses_recalibrate": bool(chosen_action == "recalibrate"),
                }
            )

    if unknown_decision_seen:
        warnings.append("decision_unknown")

    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_csv(out_dir / "policy_v2_replay_trace.csv", index=False)

    selector_summaries: list[dict[str, Any]] = []
    for (selector_name, uncertainty_lambda, route_gamma), frame in replay_df.groupby(
        ["selector_name", "uncertainty_lambda", "route_gamma"],
        dropna=False,
        sort=False,
    ):
        selector_summaries.append(
            _selector_record(
                frame,
                str(selector_name),
                float(uncertainty_lambda),
                float(route_gamma),
            )
        )

    current_policy_summary = next(
        row for row in selector_summaries if row["selector_name"] == "current_policy"
    )
    best_selector_by_mean_harm = min(
        selector_summaries,
        key=lambda row: (math.inf if not math.isfinite(float(row["mean_harm"])) else float(row["mean_harm"]), row["selector_name"]),
    )
    best_selector_by_mean_policy_gap = min(
        selector_summaries,
        key=lambda row: (
            math.inf if not math.isfinite(float(row["mean_policy_gap_vs_oracle"])) else float(row["mean_policy_gap_vs_oracle"]),
            row["selector_name"],
        ),
    )
    promising_selector_exists = bool(
        float(best_selector_by_mean_harm["mean_harm_improvement_vs_current"]) > 0.0
        and float(best_selector_by_mean_harm["mean_policy_gap_vs_oracle"]) < float(current_policy_summary["mean_policy_gap_vs_oracle"])
    )

    analysis_contract_status = "incomplete" if warnings else "ok"
    summary = {
        "analysis_contract_status": analysis_contract_status,
        "policy_contract_status": policy_contract_status,
        "paper_tier_ready": bool(policy_contract_status == "ok" and analysis_contract_status == "ok"),
        "current_policy_summary": current_policy_summary,
        "best_selector_by_mean_harm": best_selector_by_mean_harm,
        "best_selector_by_mean_policy_gap": best_selector_by_mean_policy_gap,
        "promising_selector_exists": promising_selector_exists,
        "selector_summaries": selector_summaries,
        "warnings": warnings,
    }

    (out_dir / "policy_v2_replay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "policy_v2_replay.md").write_text(_markdown_summary(summary), encoding="utf-8")
    _progress("saved policy_v2_replay_summary.json and policy_v2_replay.md")


if __name__ == "__main__":
    main()
