#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import add_panel_label, apply_journal_style, save_figure_variants, soften_axes


SHIFT_FAMILIES = [
    "Clean",
    "Heel-\nlike",
    "Scatter-\nlike",
    "Exposure",
    "Poisson",
    "Exposure+\nPoisson",
    "Motion\nblur",
    "Mixed\nstress",
]

AUROC = [0.846, 0.836, 0.829, 0.823, 0.819, 0.807, 0.802, 0.791]
EXPECTED_HARM = [0.081, 0.087, 0.090, 0.095, 0.097, 0.106, 0.110, 0.118]

LINE_COLOR = "#3f3f3f"
MARKER_FACE = "white"
GROUP_SHADE = "#f7f7f7"
SEPARATOR_COLOR = "#cdcdcd"
GROUP_TEXT = "#767676"


def add_group_structure(ax: plt.Axes, y_top: float) -> None:
    ax.axvspan(-0.45, 2.45, color=GROUP_SHADE, zorder=0)
    ax.axvline(2.5, color=SEPARATOR_COLOR, linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
    ax.text(
        1.0,
        y_top,
        "Principal technical shifts",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=GROUP_TEXT,
    )
    ax.text(
        5.1,
        y_top,
        "Supportive stress families",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=GROUP_TEXT,
    )


def draw_panel(
    ax: plt.Axes,
    values: list[float],
    ylabel: str,
    panel_label: str,
    ylim: tuple[float, float],
) -> None:
    x = list(range(len(SHIFT_FAMILIES)))
    ax.plot(
        x,
        values,
        color=LINE_COLOR,
        linewidth=1.35,
        marker="o",
        markersize=4.1,
        markerfacecolor=MARKER_FACE,
        markeredgecolor=LINE_COLOR,
        markeredgewidth=0.9,
        zorder=3,
    )
    ax.set_xlim(-0.35, len(SHIFT_FAMILIES) - 0.65)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFT_FAMILIES)
    ax.set_ylabel(ylabel)
    soften_axes(ax, grid_axis="y")
    add_panel_label(ax, panel_label)
    margin = (ylim[1] - ylim[0]) * 0.035
    add_group_structure(ax, ylim[1] - margin)


def build_figure(add_title: bool) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.45), constrained_layout=False)
    draw_panel(
        axes[0],
        AUROC,
        ylabel="AUROC",
        panel_label="A",
        ylim=(0.786, 0.852),
    )
    draw_panel(
        axes[1],
        EXPECTED_HARM,
        ylabel="Expected harm",
        panel_label="B",
        ylim=(0.078, 0.1215),
    )
    title_artist = None
    if add_title:
        title_artist = fig.suptitle(
            "Degradation of reliability metrics across the principal technical shift families",
            fontsize=9.8,
            y=0.965,
        )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.205, top=0.84 if add_title else 0.90, wspace=0.18)
    return fig, title_artist


def main() -> None:
    apply_journal_style()

    out_dir = Path("outputs/paper_figures/reliability_shift_degradation")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "reliability_shift_degradation"

    fig, title_artist = build_figure(add_title=True)
    save_figure_variants(fig, base, title_artist=title_artist)


if __name__ == "__main__":
    main()
