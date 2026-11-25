from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    DIVERGING_CMAP,
    FIGSIZE_DOUBLE,
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    METRIC_LINESTYLES,
    METHOD_LINESTYLES,
    SEQUENTIAL_CMAP,
    SEMANTIC_PALETTE,
    SHIFT_FAMILY_PALETTE,
    apply_paper_style,
    ensure_parent,
    new_subplots,
)


def save_figure_bundle(fig: plt.Figure, base_path: Path) -> None:
    apply_paper_style()
    ensure_parent(base_path)
    fig.savefig(base_path.with_suffix(".png"))
    fig.savefig(base_path.with_suffix(".pdf"))
    plt.close(fig)


def _scenario_mode_dir(out_dir: Path, scenario_mode: str) -> Path:
    if scenario_mode == "mixed_stress_eval":
        return out_dir / "figures" / "mixed_stress"
    return out_dir / "figures" / "atomic_calibrated"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _ordered_shift_families(values: Iterable[object]) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    canonical = [key for key in SHIFT_FAMILY_PALETTE if key in present]
    extras = sorted({v for v in present if v not in SHIFT_FAMILY_PALETTE})
    return canonical + extras


def _ordered_scenario_ids(values: Iterable[object]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        label = str(value)
        if label not in seen:
            seen.add(label)
            unique.append(label)

    def _parse(label: str) -> tuple[int, int, str, float, str]:
        if ":" in label:
            shift, suffix = label.split(":", 1)
        else:
            shift, suffix = label, ""
        try:
            numeric = float(suffix)
            numeric_flag = 0
        except ValueError:
            numeric = float("inf")
            numeric_flag = 1
        shift_rank = _ordered_shift_families([shift]).index(shift) if shift in SHIFT_FAMILY_PALETTE else len(SHIFT_FAMILY_PALETTE)
        return (shift_rank, numeric_flag, shift, numeric, label)

    return sorted(unique, key=_parse)


def _annotate_heatmap(ax, matrix: np.ndarray, fmt: str = ".0f") -> None:
    if matrix.size == 0:
        return
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    max_dim = max(int(matrix.shape[0]), int(matrix.shape[1]))
    if max_dim <= 5:
        fontsize = 8
    elif max_dim <= 8:
        fontsize = 7
    elif max_dim <= 12:
        fontsize = 6
    else:
        fontsize = 5
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if norm(float(value)) > 0.55 else "#18212F"
            ax.text(col, row, format(float(value), fmt), ha="center", va="center", fontsize=fontsize, color=text_color)


def _add_bar_value_labels(ax, *, fmt: str = "{:.2f}", orientation: str = "vertical") -> None:
    for patch in ax.patches:
        if orientation == "horizontal":
            value = patch.get_width()
            if not np.isfinite(value):
                continue
            y = patch.get_y() + patch.get_height() / 2.0
            offset = 0.01 * max(1.0, abs(value))
            x = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(x, y, fmt.format(value), va="center", ha=ha, fontsize=8, color="#18212F")
        else:
            value = patch.get_height()
            if not np.isfinite(value):
                continue
            x = patch.get_x() + patch.get_width() / 2.0
            offset = 0.01 * max(1.0, abs(value))
            y = value + offset if value >= 0 else value - offset
            va = "bottom" if value >= 0 else "top"
            ax.text(x, y, fmt.format(value), ha="center", va=va, fontsize=8, color="#18212F")


def _make_ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    arr = np.sort(arr)
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, y


def _build_worst_family_summary(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["method", "x_value"], dropna=False)
    for (method, x_value), sub in grouped:
        row = {"method": str(method), "x_value": float(x_value)}
        for metric in metrics:
            values = _coerce_numeric(sub.get(metric, pd.Series([], dtype=float)))
            valid = sub.assign(_metric=values).dropna(subset=["_metric"])
            if valid.empty:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_worst"] = float("nan")
                row[f"{metric}_worst_shift"] = "unknown"
                continue
            row[f"{metric}_mean"] = float(valid["_metric"].mean())
            worst_idx = valid["_metric"].idxmax()
            row[f"{metric}_worst"] = float(valid.loc[worst_idx, "_metric"])
            row[f"{metric}_worst_shift"] = str(valid.loc[worst_idx, "shift"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "x_value"]).reset_index(drop=True)


def _build_spread_summary(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["method", "x_value"], dropna=False)
    for (method, x_value), sub in grouped:
        row = {
            "method": str(method),
            "x_value": float(x_value),
            "n_shift_families": int(sub["shift"].astype(str).nunique()),
        }
        for metric in metrics:
            values = _coerce_numeric(sub.get(metric, pd.Series([], dtype=float)))
            valid = sub.assign(_metric=values).dropna(subset=["_metric"]).sort_values(["_metric", "shift"])
            if valid.empty:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_best"] = float("nan")
                row[f"{metric}_worst"] = float("nan")
                row[f"{metric}_best_shift"] = "unknown"
                row[f"{metric}_worst_shift"] = "unknown"
                continue
            row[f"{metric}_mean"] = float(valid["_metric"].mean())
            best_idx = valid["_metric"].idxmin()
            worst_idx = valid["_metric"].idxmax()
            row[f"{metric}_best"] = float(valid.loc[best_idx, "_metric"])
            row[f"{metric}_worst"] = float(valid.loc[worst_idx, "_metric"])
            row[f"{metric}_best_shift"] = str(valid.loc[best_idx, "shift"])
            row[f"{metric}_worst_shift"] = str(valid.loc[worst_idx, "shift"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "x_value"]).reset_index(drop=True)


def _as_display_image(array_like: object) -> np.ndarray:
    image = np.asarray(array_like, dtype=float)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in (2, 3):
        raise ValueError(f"qualitative image must be 2D or 3D after squeezing; got shape {image.shape}")
    return image


def _prepare_erasure_plot_df(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = metrics_df.copy()
    df = df[df["method"].isin(["no_eraser", "eraser"])].copy()
    realism = _coerce_numeric(df.get("realism_level", pd.Series([], dtype=float)))
    intensity = _coerce_numeric(df.get("intensity", pd.Series([], dtype=float)))
    if realism.notna().any():
        df["x_value"] = realism
        x_label = "Realism Level"
    else:
        df["x_value"] = intensity
        x_label = "Effective Intensity"
    return df[df["x_value"].notna()].copy(), x_label


def _aggregate_metric_curves(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for (method, x_value), sub in df.groupby(["method", "x_value"], dropna=False):
        row = {"method": method, "x_value": float(x_value), "n_shift_families": int(sub["shift"].nunique())}
        for metric in metrics:
            vals = _coerce_numeric(sub.get(metric, pd.Series([], dtype=float))).dropna()
            row[metric] = float(vals.mean()) if not vals.empty else float("nan")
            row[f"{metric}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
            low_col = f"{metric}_ci_low"
            high_col = f"{metric}_ci_high"
            if low_col in sub.columns and high_col in sub.columns:
                low_vals = _coerce_numeric(sub[low_col]).dropna()
                high_vals = _coerce_numeric(sub[high_col]).dropna()
                row[low_col] = float(low_vals.mean()) if not low_vals.empty else float("nan")
                row[high_col] = float(high_vals.mean()) if not high_vals.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "x_value"]).reset_index(drop=True)


def _plot_metric_panel(ax, agg_df: pd.DataFrame, metric_a: str, metric_b: str, y_label: str) -> None:
    for method in ["no_eraser", "eraser"]:
        sub = agg_df[agg_df["method"] == method].sort_values("x_value")
        if sub.empty:
            continue
        color = SEMANTIC_PALETTE[method]
        ax.plot(
            sub["x_value"],
            sub[metric_a],
            color=color,
            linestyle=METRIC_LINESTYLES.get(metric_a, "-"),
            marker="o",
            label=f"{method} · {metric_a}",
        )
        low_col = f"{metric_a}_ci_low"
        high_col = f"{metric_a}_ci_high"
        if low_col in sub.columns and high_col in sub.columns:
            low = _coerce_numeric(sub[low_col]).to_numpy(dtype=float)
            high = _coerce_numeric(sub[high_col]).to_numpy(dtype=float)
            if np.isfinite(low).any() and np.isfinite(high).any():
                ax.fill_between(sub["x_value"], low, high, color=color, alpha=0.14)
        if metric_b in sub.columns:
            ax.plot(
                sub["x_value"],
                sub[metric_b],
                color=color,
                linestyle=METRIC_LINESTYLES.get(metric_b, "--"),
                marker="s",
                alpha=0.95,
                label=f"{method} · {metric_b}",
            )
            low_col = f"{metric_b}_ci_low"
            high_col = f"{metric_b}_ci_high"
            if low_col in sub.columns and high_col in sub.columns:
                low = _coerce_numeric(sub[low_col]).to_numpy(dtype=float)
                high = _coerce_numeric(sub[high_col]).to_numpy(dtype=float)
                if np.isfinite(low).any() and np.isfinite(high).any():
                    ax.fill_between(sub["x_value"], low, high, color=color, alpha=0.10)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", ncol=2)


def _scenario_sort_key(label: object) -> tuple[int, int, str, float, str]:
    ordered = _ordered_scenario_ids([label])
    if ordered:
        mapped = ordered[0]
    else:
        mapped = str(label)
    if ":" in mapped:
        shift, suffix = mapped.split(":", 1)
    else:
        shift, suffix = mapped, ""
    try:
        numeric = float(suffix)
        numeric_flag = 0
    except ValueError:
        numeric = float("inf")
        numeric_flag = 1
    shift_rank = _ordered_shift_families([shift]).index(shift) if shift in SHIFT_FAMILY_PALETTE else len(SHIFT_FAMILY_PALETTE)
    return (shift_rank, numeric_flag, shift, numeric, mapped)


def _plot_grouped_bars(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    categories = list(dict.fromkeys(df["scenario_id"].astype(str).tolist()))
    positions = np.arange(len(categories))
    width = 0.36
    for idx, method in enumerate(["no_eraser", "eraser"]):
        sub = df[df["method"] == method].sort_values("scenario_id")
        if sub.empty:
            continue
        vals = []
        for category in categories:
            row = sub[sub["scenario_id"].astype(str) == category]
            if row.empty or metric not in row.columns:
                vals.append(float("nan"))
            else:
                vals.append(float(_coerce_numeric(row[metric]).iloc[0]))
        offset = (-0.5 + idx) * width
        ax.bar(positions + offset, vals, width=width, color=SEMANTIC_PALETTE[method], label=method)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend(loc="best")


def build_erasure_eval_figures(
    metrics_df: pd.DataFrame,
    *,
    out_dir: Path,
    scenario_mode: str,
    reliability_df: pd.DataFrame | None = None,
    risk_curve_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    apply_paper_style()
    fig_dir = _scenario_mode_dir(out_dir, scenario_mode)
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if scenario_mode == "mixed_stress_eval":
        plot_df = metrics_df.copy()
        plot_df["scenario_id"] = plot_df.get("profile", plot_df.get("intensity", "unknown")).astype(str)
        plot_df.to_csv(fig_dir / "mixed_stress_metrics_tidy.csv", index=False)
        outputs["mixed_stress_metrics_tidy"] = str(fig_dir / "mixed_stress_metrics_tidy.csv")
        fig, axes = new_subplots(2, 2, kind="wide", figsize=FIGSIZE_WIDE)
        _plot_grouped_bars(axes[0, 0], plot_df, "auroc_macro", "AUROC by Mixed Profile")
        _plot_grouped_bars(axes[0, 1], plot_df, "ece", "ECE by Mixed Profile")
        _plot_grouped_bars(axes[1, 0], plot_df, "risk_at_coverage", "Selective Risk by Mixed Profile")
        _plot_grouped_bars(axes[1, 1], plot_df, "expected_harm", "Expected Harm by Mixed Profile")
        save_figure_bundle(fig, fig_dir / "mixed_stress_summary")
        outputs["mixed_stress_summary"] = str(fig_dir / "mixed_stress_summary.png")
        return outputs

    plot_df, x_label = _prepare_erasure_plot_df(metrics_df)
    plot_df.to_csv(fig_dir / "atomic_metrics_tidy.csv", index=False)
    outputs["atomic_metrics_tidy"] = str(fig_dir / "atomic_metrics_tidy.csv")
    agg = _aggregate_metric_curves(
        plot_df,
        metrics=[
            "auroc_macro",
            "auprc_macro",
            "ece",
            "brier",
            "risk_at_coverage",
            "expected_harm",
            "coverage_realized",
            "abstain_rate",
        ],
    )
    agg.to_csv(fig_dir / "atomic_metrics_aggregated.csv", index=False)
    outputs["atomic_metrics_aggregated"] = str(fig_dir / "atomic_metrics_aggregated.csv")

    fig, axes = new_subplots(2, 2, kind="wide", figsize=FIGSIZE_WIDE)
    _plot_metric_panel(axes[0, 0], agg, "auroc_macro", "auprc_macro", "Panel A · AUROC / AUPRC")
    _plot_metric_panel(axes[0, 1], agg, "ece", "brier", "Panel B · ECE / Brier")
    _plot_metric_panel(axes[1, 0], agg, "risk_at_coverage", "expected_harm", "Panel C · Risk / Harm")
    _plot_metric_panel(axes[1, 1], agg, "coverage_realized", "abstain_rate", "Panel D · Coverage / Abstain")
    for ax in axes.ravel():
        ax.set_xlabel(x_label)
    save_figure_bundle(fig, fig_dir / "atomic_calibrated_main")
    outputs["atomic_calibrated_main"] = str(fig_dir / "atomic_calibrated_main.png")

    shift_order = _ordered_shift_families(plot_df["shift"].tolist())
    fig, axes = new_subplots(2, 2, kind="wide", figsize=FIGSIZE_WIDE)
    for ax, metric, title in zip(
        axes.ravel(),
        ["auroc_macro", "ece", "risk_at_coverage", "expected_harm"],
        ["AUROC by Family", "ECE by Family", "Selective Risk by Family", "Expected Harm by Family"],
    ):
        for shift in shift_order:
            for method in ["no_eraser", "eraser"]:
                sub = plot_df[(plot_df["shift"].astype(str) == shift) & (plot_df["method"] == method)].sort_values("x_value")
                if sub.empty or metric not in sub.columns:
                    continue
                ax.plot(
                    sub["x_value"],
                    sub[metric],
                    color=SHIFT_FAMILY_PALETTE.get(shift, "#4E5969"),
                    linestyle=METHOD_LINESTYLES.get(method, "-"),
                    marker="o",
                    alpha=0.92,
                    label=f"{shift} · {method}",
                )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.legend(loc="best", ncol=2)
    save_figure_bundle(fig, fig_dir / "atomic_calibrated_by_family")
    outputs["atomic_calibrated_by_family"] = str(fig_dir / "atomic_calibrated_by_family.png")

    metric_panels = [
        ("expected_harm", "Expected Harm"),
        ("risk_at_coverage", "Selective Risk"),
        ("ece", "ECE"),
        ("brier", "Brier"),
    ]
    small_rows = []
    if shift_order:
        fig, axes = new_subplots(
            len(shift_order),
            len(metric_panels),
            figsize=(4.0 * len(metric_panels), 2.8 * max(1, len(shift_order))),
            squeeze=False,
        )
        for row_idx, shift in enumerate(shift_order):
            shift_df = plot_df[plot_df["shift"].astype(str) == shift].copy().sort_values(["method", "x_value"])
            for col_idx, (metric, title) in enumerate(metric_panels):
                ax = axes[row_idx, col_idx]
                for method in ["no_eraser", "eraser"]:
                    sub = shift_df[shift_df["method"] == method].sort_values("x_value")
                    if sub.empty or metric not in sub.columns:
                        continue
                    ax.plot(
                        sub["x_value"],
                        sub[metric],
                        color=SEMANTIC_PALETTE[method],
                        linestyle=METHOD_LINESTYLES.get(method, "-"),
                        marker="o",
                        label=method,
                    )
                    for _, row in sub.iterrows():
                        small_rows.append(
                            {
                                "shift": shift,
                                "method": method,
                                "x_value": float(row["x_value"]),
                                "metric": metric,
                                "metric_value": float(pd.to_numeric(row[metric], errors="coerce")),
                            }
                        )
                if row_idx == 0:
                    ax.set_title(title)
                if row_idx == len(shift_order) - 1:
                    ax.set_xlabel(x_label)
                if col_idx == 0:
                    ax.set_ylabel(shift.replace("_", " "))
                if row_idx == 0 and col_idx == len(metric_panels) - 1:
                    ax.legend(loc="upper left", title="Method")
        small_df = pd.DataFrame(small_rows)
        small_df.to_csv(fig_dir / "atomic_main_metrics_small_multiples_tidy.csv", index=False)
        outputs["atomic_main_metrics_small_multiples_tidy"] = str(fig_dir / "atomic_main_metrics_small_multiples_tidy.csv")
        save_figure_bundle(fig, fig_dir / "atomic_main_metrics_small_multiples")
        outputs["atomic_main_metrics_small_multiples"] = str(fig_dir / "atomic_main_metrics_small_multiples.png")

    worst_df = _build_worst_family_summary(plot_df, metrics=["expected_harm", "risk_at_coverage"])
    worst_df.to_csv(fig_dir / "atomic_worst_family_overlay_tidy.csv", index=False)
    outputs["atomic_worst_family_overlay_tidy"] = str(fig_dir / "atomic_worst_family_overlay_tidy.csv")
    if not worst_df.empty:
        fig, axes = new_subplots(1, 2, kind="double", figsize=FIGSIZE_DOUBLE)
        for ax, metric, title in zip(
            np.atleast_1d(axes).reshape(-1),
            ["expected_harm", "risk_at_coverage"],
            ["Expected Harm Mean vs Worst Family", "Selective Risk Mean vs Worst Family"],
        ):
            for method in ["no_eraser", "eraser"]:
                sub = worst_df[worst_df["method"] == method].sort_values("x_value")
                if sub.empty:
                    continue
                color = SEMANTIC_PALETTE[method]
                ax.plot(sub["x_value"], sub[f"{metric}_mean"], color=color, marker="o", label=f"{method} mean")
                ax.plot(
                    sub["x_value"],
                    sub[f"{metric}_worst"],
                    color=color,
                    linestyle="--",
                    marker="s",
                    alpha=0.95,
                    label=f"{method} worst-family",
                )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend(loc="best", ncol=1)
        save_figure_bundle(fig, fig_dir / "atomic_worst_family_overlay")
        outputs["atomic_worst_family_overlay"] = str(fig_dir / "atomic_worst_family_overlay.png")

    spread_df = _build_spread_summary(plot_df, metrics=["expected_harm", "risk_at_coverage"])
    spread_df.to_csv(fig_dir / "atomic_summary_spread_overlay_tidy.csv", index=False)
    outputs["atomic_summary_spread_overlay_tidy"] = str(fig_dir / "atomic_summary_spread_overlay_tidy.csv")
    if not spread_df.empty:
        fig, axes = new_subplots(1, 2, kind="double", figsize=FIGSIZE_DOUBLE)
        for ax, metric, title in zip(
            np.atleast_1d(axes).reshape(-1),
            ["expected_harm", "risk_at_coverage"],
            ["Expected Harm Mean + Family Spread", "Selective Risk Mean + Family Spread"],
        ):
            for method in ["no_eraser", "eraser"]:
                sub = spread_df[spread_df["method"] == method].sort_values("x_value")
                if sub.empty:
                    continue
                color = SEMANTIC_PALETTE[method]
                x_values = _coerce_numeric(sub["x_value"]).to_numpy(dtype=float)
                mean_values = _coerce_numeric(sub[f"{metric}_mean"]).to_numpy(dtype=float)
                best_values = _coerce_numeric(sub[f"{metric}_best"]).to_numpy(dtype=float)
                worst_values = _coerce_numeric(sub[f"{metric}_worst"]).to_numpy(dtype=float)
                if np.isfinite(best_values).any() and np.isfinite(worst_values).any():
                    ax.fill_between(
                        x_values,
                        best_values,
                        worst_values,
                        color=color,
                        alpha=0.12,
                    )
                ax.plot(
                    x_values,
                    mean_values,
                    color=color,
                    marker="o",
                    label=f"{method} mean",
                )
                ax.plot(
                    x_values,
                    worst_values,
                    color=color,
                    linestyle="--",
                    marker="s",
                    alpha=0.95,
                    label=f"{method} worst-family",
                )
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend(loc="best", ncol=1, title="Summary")
        save_figure_bundle(fig, fig_dir / "atomic_summary_spread_overlay")
        outputs["atomic_summary_spread_overlay"] = str(fig_dir / "atomic_summary_spread_overlay.png")

    delta_rows = []
    no_df = agg[agg["method"] == "no_eraser"].set_index("x_value")
    er_df = agg[agg["method"] == "eraser"].set_index("x_value")
    for x_value in sorted(set(no_df.index) & set(er_df.index)):
        delta_rows.append(
            {
                "x_value": float(x_value),
                "delta_ece": float(er_df.at[x_value, "ece"] - no_df.at[x_value, "ece"]),
                "delta_brier": float(er_df.at[x_value, "brier"] - no_df.at[x_value, "brier"]),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(fig_dir / "atomic_calibration_deltas_tidy.csv", index=False)
    outputs["atomic_calibration_deltas_tidy"] = str(fig_dir / "atomic_calibration_deltas_tidy.csv")
    if not delta_df.empty:
        fig, ax = new_subplots(1, 1, kind="double")
        ax.axhline(0.0, color="#30343F", linewidth=1.0, alpha=0.8)
        ax.plot(delta_df["x_value"], delta_df["delta_ece"], color=SEMANTIC_PALETTE["eraser"], marker="o", label="Delta ECE")
        ax.plot(delta_df["x_value"], delta_df["delta_brier"], color=SEMANTIC_PALETTE["recalibrated"], marker="s", label="Delta Brier")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Eraser minus no_eraser")
        ax.set_title("Calibration Delta Summary")
        ax.legend(loc="best")
        save_figure_bundle(fig, fig_dir / "atomic_calibration_delta_summary")
        outputs["atomic_calibration_delta_summary"] = str(fig_dir / "atomic_calibration_delta_summary.png")

    if reliability_df is not None and not reliability_df.empty:
        reliability_df.to_csv(fig_dir / "atomic_reliability_tidy.csv", index=False)
        outputs["atomic_reliability_tidy"] = str(fig_dir / "atomic_reliability_tidy.csv")
        selected = []
        clean_rows = reliability_df[_coerce_numeric(reliability_df.get("x_value", pd.Series([], dtype=float))).fillna(np.inf) == 0.0]
        if not clean_rows.empty:
            selected.append(str(clean_rows.iloc[0]["scenario_label"]))
        for shift in sorted({str(v) for v in reliability_df["shift"].dropna().tolist()}):
            sub = reliability_df[reliability_df["shift"].astype(str) == shift].copy()
            if sub.empty:
                continue
            x_vals = _coerce_numeric(sub["x_value"])
            sub = sub.assign(_x=x_vals)
            sub = sub[sub["_x"].notna()]
            if sub.empty:
                continue
            selected.append(str(sub.sort_values("_x").iloc[-1]["scenario_label"]))
        selected = list(dict.fromkeys(selected))
        ncols = min(3, max(1, len(selected)))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = new_subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows))
        axes_arr = np.atleast_1d(axes).reshape(-1)
        for ax, scenario_label in zip(axes_arr, selected):
            sub = reliability_df[reliability_df["scenario_label"].astype(str) == scenario_label]
            for method in ["no_eraser", "eraser"]:
                cur = sub[sub["method"] == method]
                if cur.empty:
                    continue
                ax.plot(cur["confidence"], cur["accuracy"], marker="o", color=SEMANTIC_PALETTE[method], label=method)
            ax.plot([0, 1], [0, 1], linestyle="--", color="#404552", linewidth=1.0, alpha=0.8)
            ax.set_title(str(scenario_label))
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.legend(loc="best")
        for ax in axes_arr[len(selected):]:
            ax.axis("off")
        save_figure_bundle(fig, fig_dir / "atomic_reliability_selected")
        outputs["atomic_reliability_selected"] = str(fig_dir / "atomic_reliability_selected.png")

    if risk_curve_df is not None and not risk_curve_df.empty:
        curve_df = risk_curve_df.copy()
        curve_df.to_csv(fig_dir / "atomic_risk_coverage_tidy.csv", index=False)
        outputs["atomic_risk_coverage_tidy"] = str(fig_dir / "atomic_risk_coverage_tidy.csv")
        fig, axes = new_subplots(1, 2, kind="double")
        for ax, method in zip(axes, ["no_eraser", "eraser"]):
            sub = curve_df[curve_df["method"] == method].copy()
            if sub.empty:
                ax.axis("off")
                continue
            group_col = "scenario_label" if "scenario_label" in sub.columns else "shift"
            for scenario_label, cur in sub.groupby(group_col, dropna=False):
                cur = cur.sort_values("coverage")
                if cur.empty:
                    continue
                shift_name = str(cur["shift"].iloc[0]) if "shift" in cur.columns and not cur.empty else str(scenario_label)
                color = SHIFT_FAMILY_PALETTE.get(shift_name, "#4E5969")
                ax.plot(
                    cur["coverage"],
                    cur["risk"],
                    color=color,
                    label=str(scenario_label),
                )
                if "risk_ci_low" in cur.columns and "risk_ci_high" in cur.columns:
                    low = _coerce_numeric(cur["risk_ci_low"]).to_numpy(dtype=float)
                    high = _coerce_numeric(cur["risk_ci_high"]).to_numpy(dtype=float)
                    if np.isfinite(low).any() and np.isfinite(high).any():
                        ax.fill_between(cur["coverage"], low, high, color=color, alpha=0.12)
            ax.set_title(f"Risk-Coverage · {method}")
            ax.set_xlabel("Coverage")
            ax.set_ylabel("Selective Risk")
            ax.legend(loc="best", ncol=2)
        save_figure_bundle(fig, fig_dir / "atomic_risk_coverage_curves")
        outputs["atomic_risk_coverage_curves"] = str(fig_dir / "atomic_risk_coverage_curves.png")

    return outputs


def build_safety_policy_figures(
    *,
    out_dir: Path,
    scenario_trace_df: pd.DataFrame | None = None,
    op_target_df: pd.DataFrame | None = None,
    op_df: pd.DataFrame | None = None,
    bootstrap_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    apply_paper_style()
    fig_dir = out_dir / "figures" / "safety_policy"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if scenario_trace_df is not None and not scenario_trace_df.empty:
        trace_df = scenario_trace_df.copy()
        trace_df.to_csv(fig_dir / "policy_scenario_trace_tidy.csv", index=False)
        outputs["policy_scenario_trace_tidy"] = str(fig_dir / "policy_scenario_trace_tidy.csv")
        decision_share = (
            trace_df.groupby(["scenario_id", "decision"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        decision_share["fraction"] = decision_share.groupby("scenario_id")["count"].transform(lambda s: s / max(float(s.sum()), 1.0))
        reason_counts = (
            trace_df.groupby(["scenario_id", "reason"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        harm_df = trace_df.groupby("scenario_id", dropna=False).agg(
            {
                "harm_no": "mean",
                "harm_eraser": "mean",
                "harm_recalibrated": "mean",
                "coverage": "mean",
                "accepted_samples": "sum",
                "abstained_samples": "sum",
            }
        ).reset_index()
        harm_df["delta_harm_eraser_vs_no"] = _coerce_numeric(harm_df["harm_eraser"]) - _coerce_numeric(harm_df["harm_no"])
        harm_df["delta_harm_recalibrated_vs_no"] = _coerce_numeric(harm_df["harm_recalibrated"]) - _coerce_numeric(harm_df["harm_no"])
        harm_df["abstain_fraction"] = _coerce_numeric(harm_df["abstained_samples"]) / (
            _coerce_numeric(harm_df["accepted_samples"]) + _coerce_numeric(harm_df["abstained_samples"])
        ).replace(0.0, np.nan)
        scenario_order = _ordered_scenario_ids(harm_df["scenario_id"].astype(str).tolist())
        decision_share["scenario_id"] = decision_share["scenario_id"].astype(str)
        decision_share["scenario_id"] = pd.Categorical(decision_share["scenario_id"], categories=scenario_order, ordered=True)
        decision_share = decision_share.sort_values(["scenario_id", "decision"]).reset_index(drop=True)
        harm_df["scenario_id"] = harm_df["scenario_id"].astype(str)
        harm_df["scenario_id"] = pd.Categorical(harm_df["scenario_id"], categories=scenario_order, ordered=True)
        harm_df = harm_df.sort_values("scenario_id").reset_index(drop=True)
        reason_order = (
            reason_counts.groupby("reason", dropna=False)["count"].sum().sort_values(ascending=False).index.astype(str).tolist()
        )
        reason_pivot = (
            reason_counts.assign(
                scenario_id=reason_counts["scenario_id"].astype(str),
                reason=reason_counts["reason"].astype(str),
            )
            .pivot(index="reason", columns="scenario_id", values="count")
            .fillna(0.0)
            .reindex(index=reason_order, columns=scenario_order, fill_value=0.0)
        )
        decision_share.to_csv(fig_dir / "policy_decision_distribution_tidy.csv", index=False)
        reason_counts.to_csv(fig_dir / "policy_reason_counts_tidy.csv", index=False)
        harm_df.to_csv(fig_dir / "policy_harm_trace_tidy.csv", index=False)
        decision_share.to_csv(fig_dir / "policy_decision_distribution_ordered_tidy.csv", index=False)
        reason_pivot.reset_index().melt(id_vars="reason", var_name="scenario_id", value_name="count").to_csv(
            fig_dir / "policy_reason_heatmap_tidy.csv",
            index=False,
        )
        outputs["policy_decision_distribution_tidy"] = str(fig_dir / "policy_decision_distribution_tidy.csv")
        outputs["policy_reason_counts_tidy"] = str(fig_dir / "policy_reason_counts_tidy.csv")
        outputs["policy_harm_trace_tidy"] = str(fig_dir / "policy_harm_trace_tidy.csv")
        outputs["policy_decision_distribution_ordered_tidy"] = str(fig_dir / "policy_decision_distribution_ordered_tidy.csv")
        outputs["policy_reason_heatmap_tidy"] = str(fig_dir / "policy_reason_heatmap_tidy.csv")

        def _draw_decision_distribution(ax) -> None:
            ordered_scenarios = [str(v) for v in harm_df["scenario_id"].astype(str).tolist()]
            bottom_map = {scenario: 0.0 for scenario in ordered_scenarios}
            decision_colors = {
                "no_eraser": SEMANTIC_PALETTE["no_eraser"],
                "apply_eraser": SEMANTIC_PALETTE["eraser"],
                "recalibrate": SEMANTIC_PALETTE["recalibrated"],
                "abstain": SEMANTIC_PALETTE["abstain"],
            }
            for decision in ["no_eraser", "apply_eraser", "recalibrate", "abstain"]:
                sub = decision_share[decision_share["decision"].astype(str) == decision]
                vals = []
                for scenario in ordered_scenarios:
                    row = sub[sub["scenario_id"].astype(str) == scenario]
                    vals.append(float(row["fraction"].iloc[0]) if not row.empty else 0.0)
                ax.bar(
                    np.arange(len(ordered_scenarios)),
                    vals,
                    bottom=[bottom_map[s] for s in ordered_scenarios],
                    color=decision_colors.get(decision, "#777777"),
                    label=decision,
                )
                for scenario, val in zip(ordered_scenarios, vals):
                    bottom_map[scenario] += val
            ax.set_xticks(np.arange(len(ordered_scenarios)))
            ax.set_xticklabels(ordered_scenarios, rotation=30, ha="right")
            ax.set_title("Decision Distribution by Scenario")
            ax.set_ylabel("Fraction")
            ax.legend(loc="best", title="Decision")

        def _draw_reason_heatmap(ax) -> None:
            matrix = reason_pivot.to_numpy(dtype=float)
            im = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP)
            ax.set_xticks(np.arange(len(reason_pivot.columns)))
            ax.set_xticklabels(reason_pivot.columns.tolist(), rotation=30, ha="right")
            ax.set_yticks(np.arange(len(reason_pivot.index)))
            ax.set_yticklabels(reason_pivot.index.tolist())
            ax.set_title("Reason-Code Heatmap")
            _annotate_heatmap(ax, matrix)
            fig = ax.figure
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def _draw_harm_delta(ax) -> None:
            x = np.arange(len(harm_df))
            ordered_scenarios = harm_df["scenario_id"].astype(str).tolist()
            ax.bar(x - 0.18, harm_df["delta_harm_eraser_vs_no"], width=0.36, color=SEMANTIC_PALETTE["eraser"], label="eraser - no_eraser")
            ax.bar(x + 0.18, harm_df["delta_harm_recalibrated_vs_no"], width=0.36, color=SEMANTIC_PALETTE["recalibrated"], label="recalibrated - no_eraser")
            ax.axhline(0.0, color="#30343F", linewidth=1.0, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(ordered_scenarios, rotation=30, ha="right")
            ax.set_title("Regret / Harm Delta Summary")
            ax.set_ylabel("Delta harm")
            ax.legend(loc="best")

        def _draw_coverage_abstain(ax) -> None:
            x = np.arange(len(harm_df))
            ordered_scenarios = harm_df["scenario_id"].astype(str).tolist()
            ax.plot(x, harm_df["coverage"], color=SEMANTIC_PALETTE["coverage"], marker="o", label="coverage")
            ax.plot(x, harm_df["abstain_fraction"], color=SEMANTIC_PALETTE["abstain"], marker="s", label="abstain fraction")
            ax.set_xticks(x)
            ax.set_xticklabels(ordered_scenarios, rotation=30, ha="right")
            ax.set_title("Coverage / Abstain Rate")
            ax.set_ylabel("Fraction")
            ax.legend(loc="best")

        fig, axes = new_subplots(2, 2, kind="wide", figsize=FIGSIZE_WIDE)
        bottom_map = {scenario: 0.0 for scenario in scenario_order}
        _draw_decision_distribution(axes[0, 0])
        _draw_reason_heatmap(axes[0, 1])
        _draw_harm_delta(axes[1, 0])
        _draw_coverage_abstain(axes[1, 1])
        save_figure_bundle(fig, fig_dir / "policy_decision_summary")
        outputs["policy_decision_summary"] = str(fig_dir / "policy_decision_summary.png")

        fig, ax = new_subplots(1, 1, kind="double")
        _draw_decision_distribution(ax)
        save_figure_bundle(fig, fig_dir / "policy_decision_distribution")
        outputs["policy_decision_distribution"] = str(fig_dir / "policy_decision_distribution.png")

        fig, ax = new_subplots(1, 1, kind="double")
        _draw_reason_heatmap(ax)
        save_figure_bundle(fig, fig_dir / "policy_reason_heatmap")
        outputs["policy_reason_heatmap"] = str(fig_dir / "policy_reason_heatmap.png")

        fig, ax = new_subplots(1, 1, kind="double")
        _draw_harm_delta(ax)
        save_figure_bundle(fig, fig_dir / "policy_harm_delta_summary")
        outputs["policy_harm_delta_summary"] = str(fig_dir / "policy_harm_delta_summary.png")

        fig, ax = new_subplots(1, 1, kind="double")
        _draw_coverage_abstain(ax)
        save_figure_bundle(fig, fig_dir / "policy_coverage_abstain_summary")
        outputs["policy_coverage_abstain_summary"] = str(fig_dir / "policy_coverage_abstain_summary.png")

    if op_target_df is not None and not op_target_df.empty:
        target_df = op_target_df.copy()
        target_df.to_csv(fig_dir / "policy_operating_points_targeted_tidy.csv", index=False)
        outputs["policy_operating_points_targeted_tidy"] = str(fig_dir / "policy_operating_points_targeted_tidy.csv")
        summary = target_df.groupby("label", dropna=False)[
            [
                "delta_sensitivity_vs_calib",
                "delta_specificity_vs_calib",
                "delta_ppv_vs_calib",
                "delta_npv_vs_calib",
            ]
        ].agg(["mean", "std"])
        labels = list(summary.index.astype(str))
        fig, axes = new_subplots(2, 2, kind="wide", figsize=FIGSIZE_WIDE)
        panels = [
            ("delta_sensitivity_vs_calib", "Delta Sensitivity"),
            ("delta_specificity_vs_calib", "Delta Specificity"),
            ("delta_ppv_vs_calib", "Delta PPV"),
            ("delta_npv_vs_calib", "Delta NPV"),
        ]
        positions = np.arange(len(labels))
        for ax, (metric, title) in zip(axes.ravel(), panels):
            means = summary[(metric, "mean")].to_numpy(dtype=float)
            stds = summary[(metric, "std")].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(means, positions, xerr=stds, fmt="o", color=SEMANTIC_PALETTE["eraser"], ecolor="#7F8798", capsize=3)
            ax.axvline(0.0, color="#30343F", linewidth=1.0, alpha=0.8)
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.set_title(title)
        save_figure_bundle(fig, fig_dir / "policy_operating_point_deltas")
        outputs["policy_operating_point_deltas"] = str(fig_dir / "policy_operating_point_deltas.png")

    if op_df is not None and not op_df.empty and "delta_harm_vs_calib" in op_df.columns:
        harm_by_label = op_df.groupby("label", dropna=False)["delta_harm_vs_calib"].agg(["mean", "std"]).reset_index()
        harm_by_label.to_csv(fig_dir / "policy_operating_points_harm_tidy.csv", index=False)
        outputs["policy_operating_points_harm_tidy"] = str(fig_dir / "policy_operating_points_harm_tidy.csv")
        fig, ax = new_subplots(1, 1, kind="double", figsize=(8.5, 6.0))
        y = np.arange(len(harm_by_label))
        ax.errorbar(
            harm_by_label["mean"].to_numpy(dtype=float),
            y,
            xerr=harm_by_label["std"].fillna(0.0).to_numpy(dtype=float),
            fmt="o",
            color=SEMANTIC_PALETTE["recalibrated"],
            ecolor="#8891A1",
            capsize=3,
        )
        ax.axvline(0.0, color="#30343F", linewidth=1.0, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(harm_by_label["label"].astype(str).tolist())
        ax.set_title("Delta Harm by Label")
        ax.set_xlabel("Mean delta_harm_vs_calib")
        save_figure_bundle(fig, fig_dir / "policy_harm_deltas")
        outputs["policy_harm_deltas"] = str(fig_dir / "policy_harm_deltas.png")

    if bootstrap_df is not None and not bootstrap_df.empty:
        boot_df = bootstrap_df.copy()
        if "scenario_id" not in boot_df.columns:
            boot_df["scenario_id"] = boot_df.apply(
                lambda row: f"{row.get('shift', 'unknown')}:{row.get('realism_level', row.get('intensity', 'unknown'))}",
                axis=1,
            )
        ordered_ids = _ordered_scenario_ids(boot_df["scenario_id"].astype(str).tolist())
        boot_df["scenario_id"] = pd.Categorical(boot_df["scenario_id"].astype(str), categories=ordered_ids, ordered=True)
        boot_df = boot_df.sort_values("scenario_id").reset_index(drop=True)
        boot_df.to_csv(fig_dir / "policy_bootstrap_tidy.csv", index=False)
        outputs["policy_bootstrap_tidy"] = str(fig_dir / "policy_bootstrap_tidy.csv")
        fig, axes = new_subplots(1, 2, kind="double")
        scenario_order = [str(v) for v in boot_df["scenario_id"].astype(str).tolist()]
        x = np.arange(len(scenario_order))
        harm_mean = _coerce_numeric(boot_df["policy_harm_mean"]).to_numpy(dtype=float)
        harm_low = _coerce_numeric(boot_df["policy_harm_ci_low"]).to_numpy(dtype=float)
        harm_high = _coerce_numeric(boot_df["policy_harm_ci_high"]).to_numpy(dtype=float)
        axes[0].errorbar(
            x,
            harm_mean,
            yerr=[
                harm_mean - harm_low,
                harm_high - harm_mean,
            ],
            fmt="o",
            color=SEMANTIC_PALETTE["eraser"],
            ecolor="#8B93A3",
            capsize=3,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenario_order, rotation=30, ha="right")
        axes[0].set_title("Policy Harm With Bootstrap CI")
        axes[0].set_ylabel("Expected harm")
        regret_mean = _coerce_numeric(boot_df["policy_regret_mean"]).to_numpy(dtype=float)
        regret_low = _coerce_numeric(boot_df["policy_regret_ci_low"]).to_numpy(dtype=float)
        regret_high = _coerce_numeric(boot_df["policy_regret_ci_high"]).to_numpy(dtype=float)
        axes[1].errorbar(
            x,
            regret_mean,
            yerr=[
                regret_mean - regret_low,
                regret_high - regret_mean,
            ],
            fmt="o",
            color=SEMANTIC_PALETTE["recalibrated"],
            ecolor="#8B93A3",
            capsize=3,
        )
        axes[1].axhline(0.0, color="#30343F", linewidth=1.0, alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(scenario_order, rotation=30, ha="right")
        axes[1].set_title("Policy Regret With Bootstrap CI")
        axes[1].set_ylabel("Regret")
        save_figure_bundle(fig, fig_dir / "policy_bootstrap_summary")
        outputs["policy_bootstrap_summary"] = str(fig_dir / "policy_bootstrap_summary.png")

    return outputs


def build_shift_monitor_figures(*, out_dir: Path, monitor_df: pd.DataFrame | None = None, drift_start: int | None = None) -> dict[str, str]:
    apply_paper_style()
    fig_dir = out_dir / "figures" / "monitoring"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if monitor_df is None or monitor_df.empty:
        return outputs
    monitor_df = monitor_df.copy()
    monitor_df.to_csv(fig_dir / "monitor_events_tidy.csv", index=False)
    outputs["monitor_events_tidy"] = str(fig_dir / "monitor_events_tidy.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for score_key in sorted({str(v) for v in monitor_df["score_key"].dropna().tolist()}):
        sub = monitor_df[monitor_df["score_key"].astype(str) == score_key]
        color = SHIFT_FAMILY_PALETTE.get(score_key, "#4E5969")
        ax.plot(sub["step"], sub["score"], marker="o", linewidth=1.8, color=color, label=f"{score_key} score")
        if "threshold" in sub.columns:
            ax.plot(sub["step"], sub["threshold"], linestyle="--", linewidth=1.2, color=color, alpha=0.6)
    if drift_start is not None:
        ax.axvspan(float(drift_start), float(monitor_df["step"].max()), color="#F3C37A", alpha=0.18)
    ax.set_title("Shift Monitor Trajectories")
    ax.set_xlabel("Batch Step")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    save_figure_bundle(fig, fig_dir / "monitor_score_trajectories")
    outputs["monitor_score_trajectories"] = str(fig_dir / "monitor_score_trajectories.png")
    return outputs


def build_realism_calibration_figures(
    *,
    out_dir: Path,
    report: Mapping[str, object],
    pairs_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    apply_paper_style()
    fig_dir = out_dir / "figures" / "realism"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    funnel = pd.DataFrame(
        [
            {"stage": "all_candidate_pairs", "count": int(report.get("n_pairs_before_filter", 0))},
            {"stage": "valid_temporal_pairs", "count": int(report.get("n_pairs_after_filter", 0))},
            {"stage": "quality_accepted_pairs", "count": int(report.get("pair_quality_counts", {}).get("technical_dominant", 0))},
            {"stage": "calibrator_used_pairs", "count": int(report.get("realism_validity_summary", {}).get("pair_acceptance_funnel", {}).get("calibrator_used_pairs", 0))},
        ]
    )
    funnel.to_csv(fig_dir / "realism_pair_acceptance_funnel_tidy.csv", index=False)
    outputs["realism_pair_acceptance_funnel_tidy"] = str(fig_dir / "realism_pair_acceptance_funnel_tidy.csv")
    fig, ax = new_subplots(1, 1, kind="double", figsize=(8.0, 4.6))
    ax.bar(funnel["stage"], funnel["count"], color=["#B8BEC9", "#8FA3BF", "#5B7DB1", "#C84C1A"])
    ax.set_title("Pair Acceptance Funnel")
    ax.set_ylabel("Pairs")
    ax.tick_params(axis="x", rotation=25)
    _add_bar_value_labels(ax, fmt="{:.0f}", orientation="vertical")
    save_figure_bundle(fig, fig_dir / "realism_pair_acceptance_funnel")
    outputs["realism_pair_acceptance_funnel"] = str(fig_dir / "realism_pair_acceptance_funnel.png")

    if pairs_df is not None and not pairs_df.empty and "acquisition_context_risk" in pairs_df.columns:
        risk_df = pairs_df.copy()
        risk_df["acceptance"] = np.where(
            risk_df.get("pair_quality_bucket", pd.Series(["ambiguous"] * len(risk_df))).astype(str) == "technical_dominant",
            "accepted",
            "rejected",
        )
        risk_df[["acquisition_context_risk", "clinical_change_risk", "acceptance"]].to_csv(
            fig_dir / "realism_pair_risk_tidy.csv",
            index=False,
        )
        outputs["realism_pair_risk_tidy"] = str(fig_dir / "realism_pair_risk_tidy.csv")
        fig, axes = new_subplots(1, 2, kind="double", figsize=(11.5, 4.4))
        for ax, metric in zip(axes, ["acquisition_context_risk", "clinical_change_risk"]):
            for acceptance, color in [("accepted", SEMANTIC_PALETTE["eraser"]), ("rejected", SEMANTIC_PALETTE["no_eraser"])]:
                vals = _coerce_numeric(risk_df.loc[risk_df["acceptance"] == acceptance, metric]).dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                ax.hist(vals, bins=20, alpha=0.55, density=True, color=color, label=acceptance)
            ax.set_title(metric.replace("_", " ").title())
            ax.legend(loc="best")
        save_figure_bundle(fig, fig_dir / "realism_pair_risk_histogram")
        outputs["realism_pair_risk_histogram"] = str(fig_dir / "realism_pair_risk_histogram.png")

        ecdf_rows = []
        fig, axes = new_subplots(1, 2, kind="double", figsize=(11.5, 4.4))
        for ax, metric in zip(axes, ["acquisition_context_risk", "clinical_change_risk"]):
            for acceptance, color in [("accepted", SEMANTIC_PALETTE["eraser"]), ("rejected", SEMANTIC_PALETTE["no_eraser"])]:
                vals = _coerce_numeric(risk_df.loc[risk_df["acceptance"] == acceptance, metric]).dropna().to_numpy(dtype=float)
                xs, ys = _make_ecdf(vals)
                if xs.size == 0:
                    continue
                ax.step(xs, ys, where="post", color=color, label=acceptance)
                for x_val, y_val in zip(xs, ys):
                    ecdf_rows.append(
                        {
                            "metric": metric,
                            "acceptance": acceptance,
                            "value": float(x_val),
                            "ecdf": float(y_val),
                        }
                    )
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Risk")
            ax.set_ylabel("ECDF")
            ax.legend(loc="best", title="Pair status")
        pd.DataFrame(ecdf_rows).to_csv(fig_dir / "realism_pair_risk_ecdf_tidy.csv", index=False)
        outputs["realism_pair_risk_ecdf_tidy"] = str(fig_dir / "realism_pair_risk_ecdf_tidy.csv")
        save_figure_bundle(fig, fig_dir / "realism_pair_risk_ecdf")
        outputs["realism_pair_risk_ecdf"] = str(fig_dir / "realism_pair_risk_ecdf.png")

    validity_rows = []
    for shift_name, payload in report.items():
        if not isinstance(payload, Mapping) or "mapping" not in payload:
            continue
        validity_rows.append(
            {
                "shift_family": str(shift_name),
                "is_graded": bool(payload.get("is_graded", False)),
                "level_separation_margin": float(payload.get("level_separation_margin", float("nan"))),
                "validity_flags": ",".join(str(v) for v in payload.get("validity_flags", [])),
            }
        )
    validity_df = pd.DataFrame(validity_rows)
    if not validity_df.empty:
        validity_df = validity_df.sort_values(["level_separation_margin", "shift_family"], ascending=[False, True]).reset_index(drop=True)
        validity_df["validity_rank"] = np.arange(1, len(validity_df) + 1, dtype=int)
        validity_df.to_csv(fig_dir / "realism_validity_tidy.csv", index=False)
        outputs["realism_validity_tidy"] = str(fig_dir / "realism_validity_tidy.csv")
        fig, ax = new_subplots(1, 1, kind="double", figsize=(8.0, 4.6))
        colors = [SEMANTIC_PALETTE["recalibrated"] if ok else SEMANTIC_PALETTE["abstain"] for ok in validity_df["is_graded"]]
        ax.bar(validity_df["shift_family"], validity_df["level_separation_margin"], color=colors)
        ax.set_title("Realism Validity by Shift Family")
        ax.set_ylabel("Level Separation Margin")
        ax.tick_params(axis="x", rotation=20)
        save_figure_bundle(fig, fig_dir / "realism_validity")
        outputs["realism_validity"] = str(fig_dir / "realism_validity.png")

    return outputs


def build_confounding_audit_figures(
    *,
    out_dir: Path,
    predictability_df: pd.DataFrame | None = None,
    shift_id_df: pd.DataFrame | None = None,
    pair_delta_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    apply_paper_style()
    fig_dir = out_dir / "figures" / "confounding"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    if predictability_df is not None and not predictability_df.empty:
        pred_df = predictability_df.copy()
        pred_df.to_csv(fig_dir / "confounding_predictability_tidy.csv", index=False)
        outputs["confounding_predictability_tidy"] = str(fig_dir / "confounding_predictability_tidy.csv")
        fig, axes = new_subplots(1, 2, kind="wide", figsize=(13.0, 4.8))
        context_df = pred_df[pred_df["audit_family"].isin(["context_from_embeddings", "context_from_labels"])].copy()
        if not context_df.empty:
            context_df["target_label"] = context_df["audit_family"].astype(str) + " · " + context_df["target"].astype(str)
            context_df = context_df.sort_values(["audit_family", "auroc_macro", "target_label"], ascending=[True, True, True])
            y_pos = np.arange(len(context_df))
            axes[0].barh(
                y_pos,
                _coerce_numeric(context_df["auroc_macro"]).fillna(0.0),
                color=[SEMANTIC_PALETTE["eraser"] if fam == "context_from_embeddings" else SEMANTIC_PALETTE["coverage"] for fam in context_df["audit_family"].astype(str)],
            )
            if "auroc_macro_ci_low" in context_df.columns and "auroc_macro_ci_high" in context_df.columns:
                vals = _coerce_numeric(context_df["auroc_macro"]).fillna(0.0).to_numpy(dtype=float)
                lo = _coerce_numeric(context_df["auroc_macro_ci_low"]).to_numpy(dtype=float)
                hi = _coerce_numeric(context_df["auroc_macro_ci_high"]).to_numpy(dtype=float)
                lo = np.where(np.isfinite(lo), lo, vals)
                hi = np.where(np.isfinite(hi), hi, vals)
                axes[0].errorbar(
                    vals,
                    y_pos,
                    xerr=np.vstack([np.maximum(vals - lo, 0.0), np.maximum(hi - vals, 0.0)]),
                    fmt="none",
                    ecolor="#30343F",
                    alpha=0.7,
                    capsize=2.0,
                )
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(context_df["target_label"].astype(str))
            axes[0].set_title("Context Predictability")
            axes[0].set_xlabel("Macro AUROC")
            _add_bar_value_labels(axes[0], fmt="{:.2f}", orientation="horizontal")
        else:
            axes[0].axis("off")
        severity_df = pred_df[pred_df["audit_family"].isin(["labels_from_context", "severity_from_context"])].copy()
        if not severity_df.empty:
            metric_col = "auroc" if "auroc" in severity_df.columns else "balanced_accuracy"
            severity_df["target_label"] = severity_df["audit_family"].astype(str) + " · " + severity_df["target"].astype(str)
            severity_df = severity_df.sort_values(["audit_family", metric_col, "target_label"], ascending=[True, True, True])
            y_pos = np.arange(len(severity_df))
            axes[1].barh(y_pos, _coerce_numeric(severity_df[metric_col]).fillna(0.0), color=SEMANTIC_PALETTE["no_eraser"])
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(severity_df["target_label"].astype(str))
            axes[1].set_title("Label / Severity Predictability from Context")
            axes[1].set_xlabel(metric_col.replace("_", " ").title())
            _add_bar_value_labels(axes[1], fmt="{:.2f}", orientation="horizontal")
        else:
            axes[1].axis("off")
        save_figure_bundle(fig, fig_dir / "confounding_predictability_summary")
        outputs["confounding_predictability_summary"] = str(fig_dir / "confounding_predictability_summary.png")

    if shift_id_df is not None and not shift_id_df.empty:
        sid_df = shift_id_df.copy()
        sid_df.to_csv(fig_dir / "shift_id_proxy_audit_tidy.csv", index=False)
        outputs["shift_id_proxy_audit_tidy"] = str(fig_dir / "shift_id_proxy_audit_tidy.csv")
        fig, axes = new_subplots(1, 2, kind="double")
        sid_df["scenario_kind"] = pd.Categorical(
            sid_df["scenario_kind"].astype(str),
            categories=_ordered_scenario_ids(sid_df["scenario_kind"].astype(str).tolist()),
            ordered=True,
        )
        sid_df = sid_df.sort_values("scenario_kind").reset_index(drop=True)
        scenarios = sid_df["scenario_kind"].astype(str).tolist()
        x = np.arange(len(scenarios))
        bottom = np.zeros(len(scenarios), dtype=float)
        for cls, color in [
            ("p_none", SEMANTIC_PALETTE["no_eraser"]),
            ("p_covariate", SEMANTIC_PALETTE["eraser"]),
            ("p_prevalence", SEMANTIC_PALETTE["coverage"]),
            ("p_mixed", SEMANTIC_PALETTE["recalibrated"]),
        ]:
            if cls not in sid_df.columns:
                continue
            vals = _coerce_numeric(sid_df[cls]).fillna(0.0).to_numpy(dtype=float)
            axes[0].bar(x, vals, bottom=bottom, color=color, label=cls.replace("p_", ""))
            bottom += vals
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenarios, rotation=20, ha="right")
        axes[0].set_ylabel("Posterior")
        axes[0].set_title("Shift-ID Proxy Audit")
        axes[0].legend(loc="best")
        axes[1].bar(x, _coerce_numeric(sid_df.get("uncertainty", pd.Series([0.0] * len(sid_df)))).fillna(0.0), color=SEMANTIC_PALETTE["abstain"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(scenarios, rotation=20, ha="right")
        axes[1].set_ylabel("Uncertainty")
        axes[1].set_title("Shift-ID Uncertainty on Proxy Scenarios")
        save_figure_bundle(fig, fig_dir / "shift_id_proxy_audit")
        outputs["shift_id_proxy_audit"] = str(fig_dir / "shift_id_proxy_audit.png")

    if pair_delta_df is not None and not pair_delta_df.empty:
        pair_df = pair_delta_df.copy().sort_values("mean_abs_delta", ascending=False)
        pair_df.to_csv(fig_dir / "technical_pair_pathology_deltas_tidy.csv", index=False)
        outputs["technical_pair_pathology_deltas_tidy"] = str(fig_dir / "technical_pair_pathology_deltas_tidy.csv")
        top_df = pair_df.head(12).iloc[::-1]
        fig, ax = new_subplots(1, 1, kind="double", figsize=(8.6, 4.8))
        vals = _coerce_numeric(top_df["mean_abs_delta"]).fillna(0.0).to_numpy(dtype=float)
        ax.barh(np.arange(len(top_df)), vals, color=SEMANTIC_PALETTE["coverage"])
        if "mean_abs_delta_ci_low" in top_df.columns and "mean_abs_delta_ci_high" in top_df.columns:
            lo = _coerce_numeric(top_df["mean_abs_delta_ci_low"]).to_numpy(dtype=float)
            hi = _coerce_numeric(top_df["mean_abs_delta_ci_high"]).to_numpy(dtype=float)
            lo = np.where(np.isfinite(lo), lo, vals)
            hi = np.where(np.isfinite(hi), hi, vals)
            ax.errorbar(
                vals,
                np.arange(len(top_df)),
                xerr=np.vstack([np.maximum(vals - lo, 0.0), np.maximum(hi - vals, 0.0)]),
                fmt="none",
                ecolor="#30343F",
                alpha=0.7,
                capsize=2.0,
            )
        ax.set_yticks(np.arange(len(top_df)))
        ax.set_yticklabels(top_df["label"].astype(str))
        ax.set_xlabel("Mean Absolute Pair Delta")
        ax.set_title("Technical-Dominant Pair Pathology Drift")
        _add_bar_value_labels(ax, fmt="{:.2f}", orientation="horizontal")
        save_figure_bundle(fig, fig_dir / "technical_pair_pathology_deltas")
        outputs["technical_pair_pathology_deltas"] = str(fig_dir / "technical_pair_pathology_deltas.png")

    return outputs


def build_qualitative_shift_triptych(
    *,
    out_dir: Path,
    cases: Sequence[Mapping[str, object]],
    output_name: str = "qualitative_shift_triptych",
    include_difference: bool = False,
) -> dict[str, str]:
    apply_paper_style()
    fig_dir = out_dir / "figures" / "qualitative"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    if not cases:
        return outputs

    n_cols = 4 if include_difference else 3
    panel_titles = ["Clean", "Shifted", "Erased"] + (["Erased - Clean"] if include_difference else [])
    fig, axes = new_subplots(
        len(cases),
        n_cols,
        figsize=(3.0 * n_cols + 0.8, 2.8 * max(1, len(cases))),
        squeeze=False,
    )
    manifest_rows = []
    for row_idx, case in enumerate(cases):
        clean = _as_display_image(case["clean"])
        shifted = _as_display_image(case["shifted"])
        erased = _as_display_image(case["erased"])
        base_images = [clean, shifted, erased]
        panels = list(base_images)
        if include_difference:
            panels.append(np.asarray(erased, dtype=float) - np.asarray(clean, dtype=float))
        base_stack = np.stack([np.asarray(img, dtype=float) for img in base_images], axis=0)
        base_finite = base_stack[np.isfinite(base_stack)]
        vmin = float(base_finite.min()) if base_finite.size else 0.0
        vmax = float(base_finite.max()) if base_finite.size else 1.0
        diff_abs = float(np.nanmax(np.abs(panels[-1]))) if include_difference and np.isfinite(panels[-1]).any() else 1.0

        case_id = str(case.get("case_id", case.get("sample_id", f"case_{row_idx + 1:02d}")))
        shift_family = str(case.get("shift_family", case.get("shift", "unknown")))
        intensity = case.get("intensity", case.get("realism_level", "unknown"))
        metadata_lines = [
            case_id,
            f"shift={shift_family}",
            f"level={intensity}",
        ]
        for extra_key in ["pred_before", "pred_after", "harm_delta"]:
            if extra_key in case and case[extra_key] not in (None, ""):
                metadata_lines.append(f"{extra_key}={case[extra_key]}")
        extra_lines = case.get("caption_lines", [])
        if isinstance(extra_lines, Sequence) and not isinstance(extra_lines, (str, bytes)):
            metadata_lines.extend(str(line) for line in extra_lines if str(line))
        else:
            if extra_lines:
                metadata_lines.append(str(extra_lines))

        manifest_rows.append(
            {
                "case_id": case_id,
                "shift_family": shift_family,
                "intensity": intensity,
                "pred_before": case.get("pred_before", ""),
                "pred_after": case.get("pred_after", ""),
                "harm_delta": case.get("harm_delta", ""),
                "caption": " | ".join(metadata_lines[3:]),
            }
        )

        for col_idx, image in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_facecolor("white")
            if np.asarray(image).ndim == 2:
                if include_difference and col_idx == n_cols - 1:
                    ax.imshow(image, cmap=DIVERGING_CMAP, vmin=-diff_abs, vmax=diff_abs)
                else:
                    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.clip(np.asarray(image), 0.0, 1.0))
            if row_idx == 0:
                ax.set_title(panel_titles[col_idx])
        axes[row_idx, 0].set_ylabel(
            "\n".join(metadata_lines),
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
            fontsize=8,
            color="#18212F",
        )

    manifest_path = fig_dir / f"{output_name}_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    outputs[f"{output_name}_manifest"] = str(manifest_path)
    save_figure_bundle(fig, fig_dir / output_name)
    outputs[output_name] = str(fig_dir / f"{output_name}.png")
    return outputs
