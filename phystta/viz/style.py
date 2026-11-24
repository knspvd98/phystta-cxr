from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SEMANTIC_PALETTE = {
    "no_eraser": "#5B6578",
    "eraser": "#C84C1A",
    "recalibrated": "#1E7F5C",
    "abstain": "#9B1C31",
    "coverage": "#2457A6",
}

SHIFT_FAMILY_PALETTE = {
    "heel": "#C07A14",
    "scatter": "#2D6FA3",
    "exposure": "#3B8C6E",
    "poisson": "#6C7A1A",
    "exposure_poisson": "#2F8F78",
    "motion_blur": "#7A5E2A",
    "collimation_crop": "#9A6A36",
    "mixed": "#A44920",
    "mtf": "#6A4C93",
    "glare": "#885C3A",
}

CATEGORY_COLORS = SEMANTIC_PALETTE
SEQUENTIAL_CMAP = "cividis"
DIVERGING_CMAP = "RdBu_r"

METHOD_LINESTYLES = {
    "no_eraser": "-",
    "eraser": "-",
    "recalibrated": "--",
}

METRIC_LINESTYLES = {
    "auroc_macro": "-",
    "auprc_macro": "--",
    "ece": "-",
    "brier": "--",
    "risk_at_coverage": "-",
    "expected_harm": "--",
    "coverage_realized": "-",
    "abstain_rate": "--",
}

FIGSIZE_SINGLE = (6.4, 4.2)
FIGSIZE_DOUBLE = (10.8, 4.8)
FIGSIZE_WIDE = (12.5, 7.8)
FIGSIZE_TALL = (7.0, 9.0)


def _resolve_figsize(kind: str | None = None, figsize: tuple[float, float] | None = None) -> tuple[float, float]:
    if figsize is not None:
        return figsize
    presets = {
        "single": FIGSIZE_SINGLE,
        "double": FIGSIZE_DOUBLE,
        "wide": FIGSIZE_WIDE,
        "tall": FIGSIZE_TALL,
    }
    return presets.get(str(kind or "single"), FIGSIZE_SINGLE)


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.04,
            "figure.constrained_layout.w_pad": 0.04,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D6D8DE",
            "grid.alpha": 0.35,
            "axes.edgecolor": "#3B4352",
            "axes.linewidth": 1.0,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.titlepad": 10.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.title_fontsize": 9.0,
            "legend.labelspacing": 0.35,
            "legend.handlelength": 2.2,
            "legend.borderaxespad": 0.35,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.5,
            "errorbar.capsize": 3.0,
            "font.family": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_figure(*, kind: str | None = None, figsize: tuple[float, float] | None = None, **kwargs):
    apply_paper_style()
    resolved = _resolve_figsize(kind=kind, figsize=figsize)
    if "layout" not in kwargs and "constrained_layout" not in kwargs:
        kwargs["layout"] = "constrained"
    try:
        return plt.figure(figsize=resolved, **kwargs)
    except TypeError:
        kwargs.pop("layout", None)
        kwargs.setdefault("constrained_layout", True)
        return plt.figure(figsize=resolved, **kwargs)


def new_subplots(*args, kind: str | None = None, figsize: tuple[float, float] | None = None, **kwargs):
    apply_paper_style()
    resolved = _resolve_figsize(kind=kind, figsize=figsize)
    if "layout" not in kwargs and "constrained_layout" not in kwargs:
        kwargs["layout"] = "constrained"
    try:
        return plt.subplots(*args, figsize=resolved, **kwargs)
    except TypeError:
        kwargs.pop("layout", None)
        kwargs.setdefault("constrained_layout", True)
        return plt.subplots(*args, figsize=resolved, **kwargs)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
