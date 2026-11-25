#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from phystta.config.loader import load_config
from phystta.utils.io import metric_report_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _summary_from_metrics(df: pd.DataFrame | None, metric_cols: list[str]) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    out: Dict[str, float] = {}
    for c in metric_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) > 0:
                out[f"{c}_mean"] = float(vals.mean())
                out[f"{c}_min"] = float(vals.min())
                out[f"{c}_max"] = float(vals.max())
    return out


def _claim(status: str, message: str, details: Dict[str, Any]) -> Dict[str, Any]:
    if status not in {"pass", "fail", "not_evaluable"}:
        raise ValueError(f"invalid claim status: {status}")
    return {
        "status": status,
        "passed": status == "pass",
        "message": message,
        "details": details,
    }


def _evaluate_claims(
    cfg,
    required_artifacts: Dict[str, bool],
    er_metrics: pd.DataFrame | None,
    er_fairness: pd.DataFrame | None,
    sa_metrics: pd.DataFrame | None,
    sa_boot: pd.DataFrame | None,
    ra_fals: Dict[str, Any] | None,
    coverage_floor_scale: float,
    auroc_drop_tol: float,
    fnr_gap_increase_tol: float,
    regret_ci_cap: float | None,
    anchor_angle_margin: float,
) -> Dict[str, Any]:
    claims: Dict[str, Dict[str, Any]] = {}
    missing = sorted([k for k, ok in required_artifacts.items() if not ok])
    if missing:
        claims["artifact_completeness"] = _claim(
            "fail",
            "Missing required artifacts",
            {"missing": missing},
        )
    else:
        claims["artifact_completeness"] = _claim("pass", "All required artifacts are present", {"missing": []})

    # Coverage floor on accepted predictions under policy.
    if sa_metrics is None or sa_metrics.empty or "coverage" not in sa_metrics.columns:
        claims["coverage_floor"] = _claim("not_evaluable", "coverage column missing in safety.csv", {})
    else:
        cov = pd.to_numeric(sa_metrics["coverage"], errors="coerce").dropna()
        if len(cov) == 0:
            claims["coverage_floor"] = _claim("not_evaluable", "coverage has no numeric rows", {})
        else:
            floor = float(cfg.safety_policy.coverage_budget) * float(coverage_floor_scale)
            min_cov = float(cov.min())
            status = "pass" if min_cov >= floor else "fail"
            claims["coverage_floor"] = _claim(
                status,
                f"Coverage floor check: min={min_cov:.4f}, required>={floor:.4f}",
                {"min_coverage": min_cov, "required_floor": floor},
            )

    # Risk cap using risk_at_coverage where available.
    if sa_metrics is None or sa_metrics.empty or "risk_at_coverage" not in sa_metrics.columns:
        claims["risk_cap"] = _claim("not_evaluable", "risk_at_coverage column missing in safety.csv", {})
    else:
        risk = pd.to_numeric(sa_metrics["risk_at_coverage"], errors="coerce").dropna()
        if len(risk) == 0:
            claims["risk_cap"] = _claim("not_evaluable", "risk_at_coverage has no numeric rows", {})
        else:
            cap = float(cfg.safety_policy.risk_target) if getattr(cfg.safety_policy, "risk_target", None) is not None else 0.05
            p95 = float(risk.quantile(0.95))
            status = "pass" if p95 <= cap else "fail"
            claims["risk_cap"] = _claim(
                status,
                f"Risk cap check: p95={p95:.4f}, cap<={cap:.4f}",
                {"risk_p95": p95, "risk_cap": cap},
            )

    # Worst-group non-degradation: compare eraser vs no_eraser when fairness.csv has method columns.
    if er_fairness is None or er_fairness.empty:
        claims["worst_group_non_degradation"] = _claim("not_evaluable", "fairness.csv missing or empty", {})
    elif "method" not in er_fairness.columns:
        claims["worst_group_non_degradation"] = _claim("not_evaluable", "fairness.csv has no method column", {})
    elif not {"worst_group_auroc_macro", "fnr_gap_macro"}.issubset(set(er_fairness.columns)):
        claims["worst_group_non_degradation"] = _claim(
            "not_evaluable",
            "fairness.csv missing worst_group_auroc_macro/fnr_gap_macro",
            {},
        )
    else:
        fair = er_fairness.copy()
        keys = [k for k in ["shift", "intensity", "group"] if k in fair.columns]
        pivot_auc = fair.pivot_table(index=keys, columns="method", values="worst_group_auroc_macro", aggfunc="mean")
        pivot_fnr = fair.pivot_table(index=keys, columns="method", values="fnr_gap_macro", aggfunc="mean")
        if not {"no_eraser", "eraser"}.issubset(set(pivot_auc.columns)) or not {"no_eraser", "eraser"}.issubset(set(pivot_fnr.columns)):
            claims["worst_group_non_degradation"] = _claim(
                "not_evaluable",
                "fairness.csv does not contain both no_eraser and eraser rows",
                {},
            )
        else:
            delta_auc = (pivot_auc["eraser"] - pivot_auc["no_eraser"]).dropna()
            delta_fnr = (pivot_fnr["eraser"] - pivot_fnr["no_eraser"]).dropna()
            min_delta_auc = float(delta_auc.min()) if len(delta_auc) else float("nan")
            max_delta_fnr = float(delta_fnr.max()) if len(delta_fnr) else float("nan")
            ok_auc = bool(len(delta_auc) > 0 and min_delta_auc >= -float(auroc_drop_tol))
            ok_fnr = bool(len(delta_fnr) > 0 and max_delta_fnr <= float(fnr_gap_increase_tol))
            status = "pass" if (ok_auc and ok_fnr) else "fail"
            claims["worst_group_non_degradation"] = _claim(
                status,
                f"Worst-group check: min(eraser-no_eraser AUROC)={min_delta_auc:.4f}, max(eraser-no_eraser FNR gap)={max_delta_fnr:.4f}",
                {
                    "min_delta_worst_group_auroc": min_delta_auc,
                    "max_delta_fnr_gap": max_delta_fnr,
                    "auroc_drop_tol": float(auroc_drop_tol),
                    "fnr_gap_increase_tol": float(fnr_gap_increase_tol),
                },
            )

    # Policy regret CI cap using bootstrap artifact.
    if sa_boot is None or sa_boot.empty or "policy_regret_ci_high" not in sa_boot.columns:
        claims["policy_regret_ci_cap"] = _claim("not_evaluable", "policy_bootstrap.csv missing policy_regret_ci_high", {})
    else:
        regret_high = pd.to_numeric(sa_boot["policy_regret_ci_high"], errors="coerce").dropna()
        if len(regret_high) == 0:
            claims["policy_regret_ci_cap"] = _claim("not_evaluable", "policy_regret_ci_high has no numeric rows", {})
        else:
            cap = float(cfg.safety_policy.abstain_cost) if regret_ci_cap is None else float(regret_ci_cap)
            max_high = float(regret_high.max())
            status = "pass" if max_high <= cap else "fail"
            claims["policy_regret_ci_cap"] = _claim(
                status,
                f"Policy regret CI cap: max CI high={max_high:.4f}, cap<={cap:.4f}",
                {"max_policy_regret_ci_high": max_high, "regret_ci_cap": cap},
            )

    # Real-anchor consistency sanity check.
    if not ra_fals:
        claims["real_anchor_consistency"] = _claim("not_evaluable", "falsification_report.json missing", {})
    else:
        a = ra_fals.get("median_angle_train_vs_synth_deg")
        b = ra_fals.get("median_angle_train_vs_holdout_deg")
        if a is None or b is None:
            claims["real_anchor_consistency"] = _claim(
                "not_evaluable",
                "falsification report missing median angles",
                {},
            )
        else:
            a = float(a)
            b = float(b)
            # Synth should not diverge from train more than holdout by a large margin.
            ok = a <= (b + float(anchor_angle_margin))
            status = "pass" if ok else "fail"
            claims["real_anchor_consistency"] = _claim(
                status,
                f"Real-anchor angle check: train-vs-synth={a:.3f}, train-vs-holdout={b:.3f}, margin={anchor_angle_margin:.3f}",
                {
                    "median_angle_train_vs_synth_deg": a,
                    "median_angle_train_vs_holdout_deg": b,
                    "angle_margin_deg": float(anchor_angle_margin),
                },
            )

    failed = sorted([k for k, v in claims.items() if v["status"] == "fail"])
    not_eval = sorted([k for k, v in claims.items() if v["status"] == "not_evaluable"])
    gate_pass = len(failed) == 0 and len(missing) == 0
    return {
        "claims": claims,
        "paper_gate": {
            "pass": gate_pass,
            "failed_claims": failed,
            "not_evaluable_claims": not_eval,
            "missing_artifacts": missing,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build consolidated paper-tier report from run artifacts.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--erasure_dir", required=True)
    ap.add_argument("--safety_dir", required=True)
    ap.add_argument("--real_anchor_dir", required=True)
    ap.add_argument("--vlm_dir", default=None)
    ap.add_argument("--coverage_floor_scale", type=float, default=0.9)
    ap.add_argument("--auroc_drop_tol", type=float, default=0.01)
    ap.add_argument("--fnr_gap_increase_tol", type=float, default=0.02)
    ap.add_argument("--regret_ci_cap", type=float, default=None)
    ap.add_argument("--anchor_angle_margin", type=float, default=15.0)
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = load_config(str(cfg_path))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    er_dir = Path(args.erasure_dir).expanduser()
    sa_dir = Path(args.safety_dir).expanduser()
    ra_dir = Path(args.real_anchor_dir).expanduser()
    vlm_dir = Path(args.vlm_dir).expanduser() if args.vlm_dir else None
    open_data_fallback = bool(getattr(cfg.experiment, "open_data_fallback", False))

    er_metrics = _safe_read_csv(er_dir / "metrics.csv")
    er_fairness = _safe_read_csv(er_dir / "fairness.csv")
    er_causal = _safe_read_csv(er_dir / "causal_invariance.csv")
    sa_metrics = _safe_read_csv(sa_dir / "safety.csv")
    sa_policy = _safe_read_csv(sa_dir / "policy_decisions.csv")
    sa_boot = _safe_read_csv(sa_dir / "policy_bootstrap.csv")
    sa_op = _safe_read_csv(sa_dir / "policy_operating_points.csv")
    sa_op_targeted = _safe_read_csv(sa_dir / "policy_operating_points_targeted.csv")
    ra_angles = _safe_read_json(ra_dir / "principal_angles.json")
    ra_fals = _safe_read_json(ra_dir / "falsification_report.json")
    vlm_metrics = _safe_read_csv(vlm_dir / "vlm_metrics.csv") if vlm_dir else None
    vlm_summary = _safe_read_json(vlm_dir / "vlm_summary.json") if vlm_dir else None

    required_artifacts = {
        "erasure_metrics_csv": (er_dir / "metrics.csv").exists(),
        "erasure_fairness_csv": (er_dir / "fairness.csv").exists(),
        "erasure_causal_csv": (er_dir / "causal_invariance.csv").exists(),
        "safety_csv": (sa_dir / "safety.csv").exists(),
        "policy_decisions_csv": (sa_dir / "policy_decisions.csv").exists(),
        "policy_bootstrap_csv": (sa_dir / "policy_bootstrap.csv").exists(),
        "policy_operating_points_csv": (sa_dir / "policy_operating_points.csv").exists(),
        "policy_operating_points_targeted_csv": (sa_dir / "policy_operating_points_targeted.csv").exists(),
    }
    if not open_data_fallback:
        required_artifacts.update(
            {
                "real_anchor_angles_json": (ra_dir / "principal_angles.json").exists(),
                "real_anchor_falsification_json": (ra_dir / "falsification_report.json").exists(),
            }
        )
    if vlm_dir is not None:
        required_artifacts.update(
            {
                "vlm_metrics_csv": (vlm_dir / "vlm_metrics.csv").exists(),
                "vlm_summary_json": (vlm_dir / "vlm_summary.json").exists(),
            }
        )
    missing = sorted([k for k, ok in required_artifacts.items() if not ok])

    report = {
        "config": {
            "path": str(cfg_path),
            "sha256": _sha256(cfg_path),
            "tier": cfg.experiment.tier,
            "runtime": {
                "allow_download": cfg.runtime.allow_download,
                "freeze_artifacts": cfg.runtime.freeze_artifacts,
                "seed": cfg.runtime.seed,
            },
        },
        "run_mode": {
            "name": "open_data_fallback" if open_data_fallback else "paper_tier",
            "full_paper_tier": not open_data_fallback,
            "real_pair_realism_enabled": bool(cfg.intervention.use_realism and not open_data_fallback),
            "real_anchor_enabled": not open_data_fallback,
            "synthetic_intervention_grid_only": bool(open_data_fallback),
            "not_equivalent_to_credentialed_paper_tier": bool(open_data_fallback),
            "notes": (
                [
                    "Open-data fallback run: real-pair realism is disabled.",
                    "Open-data fallback run: real-anchor evaluation is skipped.",
                    "Open-data fallback run: synthetic intervention grid only.",
                    "Open-data fallback run: not equivalent to the credentialed paper-tier suite.",
                ]
                if open_data_fallback
                else []
            ),
        },
        "artifact_paths": {
            "erasure_dir": str(er_dir),
            "safety_dir": str(sa_dir),
            "real_anchor_dir": str(ra_dir),
            "vlm_dir": str(vlm_dir) if vlm_dir else None,
        },
        "artifact_presence": required_artifacts,
        "missing_artifacts": missing,
        "summaries": {
            "erasure": _summary_from_metrics(er_metrics, ["auroc_macro", "auprc_macro", "expected_harm", "ece", "brier"]),
            "fairness": _summary_from_metrics(er_fairness, ["fnr_gap_macro", "worst_group_auroc_macro", "ece_gap"]),
            "causal": _summary_from_metrics(er_causal, ["flip_rate", "js_div", "sensitivity_slope"]),
            "safety": _summary_from_metrics(sa_metrics, ["coverage", "expected_harm", "risk_at_coverage", "ece"]),
            "policy_bootstrap": _summary_from_metrics(sa_boot, ["policy_harm_mean", "policy_regret_mean"]),
            "policy_operating_points": _summary_from_metrics(sa_op, ["delta_fnr_vs_calib", "delta_harm_vs_calib"]),
            "policy_operating_points_targeted": _summary_from_metrics(
                sa_op_targeted,
                ["delta_sensitivity_vs_calib", "delta_specificity_vs_calib", "delta_ppv_vs_calib", "delta_npv_vs_calib"],
            ),
            "vlm": _summary_from_metrics(vlm_metrics, ["auroc_no", "auroc_eraser", "auprc_no", "auprc_eraser"]),
        },
        "policy_counts": {
            "n_rows": int(len(sa_policy)) if sa_policy is not None else 0,
            "actions": (
                sa_policy["decision"].value_counts(dropna=False).to_dict()
                if sa_policy is not None and "decision" in sa_policy.columns
                else {}
            ),
        },
        "real_anchor": {
            "angles": ra_angles or {},
            "falsification": ra_fals or {},
        },
        "vlm": vlm_summary or {},
    }
    claims = _evaluate_claims(
        cfg=cfg,
        required_artifacts=required_artifacts,
        er_metrics=er_metrics,
        er_fairness=er_fairness,
        sa_metrics=sa_metrics,
        sa_boot=sa_boot,
        ra_fals=ra_fals,
        coverage_floor_scale=args.coverage_floor_scale,
        auroc_drop_tol=args.auroc_drop_tol,
        fnr_gap_increase_tol=args.fnr_gap_increase_tol,
        regret_ci_cap=args.regret_ci_cap,
        anchor_angle_margin=args.anchor_angle_margin,
    )
    report.update(claims)
    if open_data_fallback and "paper_gate" in report:
        report["paper_gate"]["pass"] = False
        report["paper_gate"]["run_mode"] = "open_data_fallback"
        report["paper_gate"]["message"] = "Open-data fallback runs are not full credentialed paper-tier validations."

    (out_dir / "paper_report.json").write_text(metric_report_json(report), encoding="utf-8")
    print(f"Saved consolidated report: {out_dir / 'paper_report.json'}")


if __name__ == "__main__":
    main()
