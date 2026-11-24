from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


FONT_FAMILY = "DejaVu Sans"
GRID_COLOR = "#e4e4e4"
SPINE_COLOR = "#303030"
TEXT_COLOR = "#202020"


def apply_journal_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": FONT_FAMILY,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": SPINE_COLOR,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.01,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="semibold",
        color=TEXT_COLOR,
    )


def soften_axes(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure_variants(fig: plt.Figure, base_path: Path, *, title_artist=None, dpi: int = 300) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(base_path.with_suffix(suffix), **kwargs)
    if title_artist is not None:
        title_artist.set_visible(False)
        for suffix in (".png", ".pdf", ".svg"):
            kwargs = {"bbox_inches": "tight", "facecolor": "white"}
            if suffix == ".png":
                kwargs["dpi"] = dpi
            fig.savefig(base_path.with_name(base_path.name + "_no_title").with_suffix(suffix), **kwargs)
        title_artist.set_visible(True)
    plt.close(fig)
