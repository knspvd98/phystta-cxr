#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

# Keep matplotlib/font cache writable in restricted environments (CI/sandbox).
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplcfg_")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdgcache_")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_table(df: pd.DataFrame | None, path: Path) -> bool:
    if df is None or df.empty:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return True


def _to_canvas_xy(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> tuple[int, int]:
    left, right, top, bottom = 50, 780, 40, 460
    xn = 0.0 if xmax <= xmin else (x - xmin) / (xmax - xmin)
    yn = 0.0 if ymax <= ymin else (y - ymin) / (ymax - ymin)
    px = int(left + xn * (right - left))
    py = int(bottom - yn * (bottom - top))
    return px, py


def _init_plot_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (820, 500), color="white")
    draw = ImageDraw.Draw(img)
    draw.line((50, 40, 50, 460), fill="black", width=2)
    draw.line((50, 460, 780, 460), fill="black", width=2)
    return img, draw


def _plot_risk_coverage(safety_df: pd.DataFrame | None, out_path: Path) -> bool:
    if safety_df is None or safety_df.empty:
        return False
    if not {"coverage", "risk_at_coverage"}.issubset(set(safety_df.columns)):
        return False
    df = safety_df.copy()
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["risk_at_coverage"] = pd.to_numeric(df["risk_at_coverage"], errors="coerce")
    df = df.dropna(subset=["coverage", "risk_at_coverage"])
    if df.empty:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img, draw = _init_plot_canvas()
    xmin, xmax = float(df["coverage"].min()), float(df["coverage"].max())
    ymin, ymax = float(df["risk_at_coverage"].min()), float(df["risk_at_coverage"].max())
    if abs(xmax - xmin) < 1e-8:
        xmin, xmax = min(xmin, 0.0), max(xmax, 1.0)
    if abs(ymax - ymin) < 1e-8:
        ymin, ymax = min(ymin, 0.0), max(ymax, 1.0)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    if "shift" in df.columns:
        groups = list(df.groupby("shift"))
    else:
        groups = [("all", df)]
    for i, (_, sub) in enumerate(groups):
        sub = sub.sort_values("coverage")
        pts = [
            _to_canvas_xy(
                float(row["coverage"]),
                float(row["risk_at_coverage"]),
                xmin,
                xmax,
                ymin,
                ymax,
            )
            for _, row in sub.iterrows()
        ]
        if len(pts) >= 2:
            draw.line(pts, fill=palette[i % len(palette)], width=3)
        for pt in pts:
            draw.ellipse((pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3), fill=palette[i % len(palette)])
    img.save(out_path)
    return True


def _plot_worst_group(fairness_df: pd.DataFrame | None, out_path: Path) -> bool:
    if fairness_df is None or fairness_df.empty:
        return False
    if not {"group", "method", "fnr_gap_macro"}.issubset(set(fairness_df.columns)):
        return False
    df = fairness_df.copy()
    df["fnr_gap_macro"] = pd.to_numeric(df["fnr_gap_macro"], errors="coerce")
    df = df.dropna(subset=["fnr_gap_macro"])
    if df.empty:
        return False
    pivot = (
        df.groupby(["group", "method"], dropna=False)["fnr_gap_macro"]
        .mean()
        .reset_index()
        .pivot(index="group", columns="method", values="fnr_gap_macro")
    )
    if pivot.empty:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img, draw = _init_plot_canvas()
    vals = pivot.fillna(0.0).to_numpy(dtype=float).reshape(-1)
    if vals.size == 0:
        return False
    ymin, ymax = float(min(vals.min(), 0.0)), float(max(vals.max(), 1e-6))
    if abs(ymax - ymin) < 1e-8:
        ymax = ymin + 1.0
    n_bars = vals.size
    bar_w = max(2, int(700 / max(n_bars, 1)))
    for i, v in enumerate(vals):
        x0 = 60 + i * bar_w
        x1 = x0 + int(bar_w * 0.8)
        y0 = _to_canvas_xy(0.0, v, 0.0, 1.0, ymin, ymax)[1]
        y_base = _to_canvas_xy(0.0, 0.0, 0.0, 1.0, ymin, ymax)[1]
        y_top, y_bot = sorted((y0, y_base))
        draw.rectangle((x0, y_top, x1, y_bot), fill="#2c7fb8")
    img.save(out_path)
    return True


def _plot_sim_real_gap(anchor_fals: dict | None, out_path: Path) -> bool:
    if not anchor_fals:
        return False
    a = anchor_fals.get("median_angle_train_vs_synth_deg")
    b = anchor_fals.get("median_angle_train_vs_holdout_deg")
    if a is None or b is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img, draw = _init_plot_canvas()
    vals = [float(a), float(b)]
    ymin, ymax = float(min(vals + [0.0])), float(max(vals + [1.0]))
    x_positions = [220, 520]
    for i, v in enumerate(vals):
        x0 = x_positions[i] - 50
        x1 = x_positions[i] + 50
        y0 = _to_canvas_xy(0.0, v, 0.0, 1.0, ymin, ymax)[1]
        y_base = _to_canvas_xy(0.0, 0.0, 0.0, 1.0, ymin, ymax)[1]
        y_top, y_bot = sorted((y0, y_base))
        draw.rectangle((x0, y_top, x1, y_bot), fill=["#2c7fb8", "#f03b20"][i])
    img.save(out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dissertation-ready tables and figures from PhysTTA artifacts")
    ap.add_argument("--erasure_dir", required=True)
    ap.add_argument("--safety_dir", required=True)
    ap.add_argument("--real_anchor_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--vlm_dir", default=None)
    args = ap.parse_args()

    er_dir = Path(args.erasure_dir).expanduser()
    sa_dir = Path(args.safety_dir).expanduser()
    ra_dir = Path(args.real_anchor_dir).expanduser()
    ext_dir = Path(args.external_dir).expanduser() if args.external_dir else None
    vlm_dir = Path(args.vlm_dir).expanduser() if args.vlm_dir else None
    out_dir = Path(args.out_dir).expanduser()
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _safe_read_csv(er_dir / "metrics.csv")
    fairness = _safe_read_csv(er_dir / "fairness.csv")
    causal = _safe_read_csv(er_dir / "causal_invariance.csv")
    safety = _safe_read_csv(sa_dir / "safety.csv")
    safety_boot = _safe_read_csv(sa_dir / "policy_bootstrap.csv")
    anchor_fals = _safe_read_json(ra_dir / "falsification_report.json")
    external = _safe_read_csv(ext_dir / "external_policy_metrics.csv") if ext_dir else None
    vlm_metrics = _safe_read_csv(vlm_dir / "vlm_metrics.csv") if vlm_dir else None
    vlm_summary = _safe_read_json(vlm_dir / "vlm_summary.json") if vlm_dir else None

    generated_tables: List[str] = []
    generated_plots: List[str] = []
    missing_inputs: List[str] = []

    if metrics is not None and not metrics.empty:
        cols = [c for c in ["method", "shift", "intensity", "realism_level", "auroc_macro", "auprc_macro", "ece", "risk_at_coverage"] if c in metrics.columns]
        clinical_summary = (
            metrics[cols]
            .groupby([c for c in ["method", "shift"] if c in cols], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        if _save_table(clinical_summary, tables_dir / "clinical_summary.csv"):
            generated_tables.append(str(tables_dir / "clinical_summary.csv"))
    else:
        missing_inputs.append(str(er_dir / "metrics.csv"))

    if fairness is not None and not fairness.empty:
        cols = [c for c in ["method", "shift", "group", "worst_group_auroc_macro", "fnr_gap_macro", "ece_gap"] if c in fairness.columns]
        fairness_summary = (
            fairness[cols]
            .groupby([c for c in ["method", "shift", "group"] if c in cols], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        if _save_table(fairness_summary, tables_dir / "fairness_summary.csv"):
            generated_tables.append(str(tables_dir / "fairness_summary.csv"))
    else:
        missing_inputs.append(str(er_dir / "fairness.csv"))

    if causal is not None and not causal.empty:
        cols = [c for c in ["method", "shift", "effect_type", "intensity", "realism_level", "mean_effect", "slope", "p_value"] if c in causal.columns]
        causal_summary = causal[cols] if cols else causal
        if _save_table(causal_summary, tables_dir / "causal_summary.csv"):
            generated_tables.append(str(tables_dir / "causal_summary.csv"))
    else:
        missing_inputs.append(str(er_dir / "causal_invariance.csv"))

    if safety is not None and not safety.empty:
        cols = [c for c in ["shift", "intensity", "decision", "coverage", "expected_harm", "risk_at_coverage", "ece"] if c in safety.columns]
        safety_summary = (
            safety[cols]
            .groupby([c for c in ["shift", "decision"] if c in cols], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        if _save_table(safety_summary, tables_dir / "safety_summary.csv"):
            generated_tables.append(str(tables_dir / "safety_summary.csv"))
    else:
        missing_inputs.append(str(sa_dir / "safety.csv"))

    if safety_boot is not None and not safety_boot.empty:
        if _save_table(safety_boot, tables_dir / "policy_bootstrap_summary.csv"):
            generated_tables.append(str(tables_dir / "policy_bootstrap_summary.csv"))
    else:
        missing_inputs.append(str(sa_dir / "policy_bootstrap.csv"))

    if external is not None and not external.empty:
        cols = [c for c in ["shift", "decision", "coverage", "risk_at_coverage", "auroc_macro", "auprc_macro"] if c in external.columns]
        ext_summary = (
            external[cols]
            .groupby([c for c in ["shift", "decision"] if c in cols], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        if _save_table(ext_summary, tables_dir / "external_summary.csv"):
            generated_tables.append(str(tables_dir / "external_summary.csv"))

    if vlm_metrics is not None and not vlm_metrics.empty:
        cols = [c for c in ["shift", "intensity", "profile", "auroc_no", "auroc_eraser", "auprc_no", "auprc_eraser", "flip_rate_no", "flip_rate_eraser"] if c in vlm_metrics.columns]
        vlm_summary_df = (
            vlm_metrics[cols]
            .groupby([c for c in ["shift", "profile"] if c in cols], dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        if _save_table(vlm_summary_df, tables_dir / "vlm_summary.csv"):
            generated_tables.append(str(tables_dir / "vlm_summary.csv"))
    elif vlm_dir is not None:
        missing_inputs.append(str(vlm_dir / "vlm_metrics.csv"))

    if _plot_risk_coverage(safety, plots_dir / "risk_coverage.png"):
        generated_plots.append(str(plots_dir / "risk_coverage.png"))
    if _plot_worst_group(fairness, plots_dir / "worst_group_fnr_gap.png"):
        generated_plots.append(str(plots_dir / "worst_group_fnr_gap.png"))
    if _plot_sim_real_gap(anchor_fals, plots_dir / "sim_real_gap.png"):
        generated_plots.append(str(plots_dir / "sim_real_gap.png"))

    report = {
        "inputs": {
            "erasure_dir": str(er_dir),
            "safety_dir": str(sa_dir),
            "real_anchor_dir": str(ra_dir),
            "external_dir": str(ext_dir) if ext_dir else None,
            "vlm_dir": str(vlm_dir) if vlm_dir else None,
        },
        "generated_tables": generated_tables,
        "generated_plots": generated_plots,
        "missing_inputs": sorted(set(missing_inputs)),
        "vlm": vlm_summary or {},
    }
    (out_dir / "dissertation_artifacts_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved dissertation artifacts report: {out_dir / 'dissertation_artifacts_report.json'}")


if __name__ == "__main__":
    main()
