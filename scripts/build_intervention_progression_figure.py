#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import add_panel_label, apply_journal_style, save_figure_variants, soften_axes


INTERVENTIONS = [
    "No intervention",
    "Linear subspace\nvariant",
    "Matrix projection\nvariant",
    "Steering\nvariant",
    "Hard task-preserving\nhead-subspace",
    "Soft task-preserving\nrecovery mainline",
    "Token-aware\nextension",
]

EXPECTED_HARM = [0.109, 0.103, 0.101, 0.091, 0.087, 0.083, 0.081]
ECE = [0.035, 0.032, 0.031, 0.026, 0.022, 0.021, 0.020]

STEM_COLOR = "#c8c8c8"
LINE_COLOR = "#9a9a9a"
BASE_FACE = "#9b9b9b"
SELECTED_FACE = "#222222"
TOKEN_FACE = "white"
EDGE_COLOR = "#2f2f2f"

SELECTED_INDEX = 5
TOKEN_INDEX = 6


def draw_lollipop_panel(
    ax: plt.Axes,
    values: list[float],
    xlabel: str,
    panel_label: str,
    ylim: tuple[float, float],
    *,
    show_labels: bool,
) -> None:
    y = list(range(len(INTERVENTIONS)))
    baseline = ylim[0]

    ax.plot(values, y, color=LINE_COLOR, linewidth=0.75, zorder=2)
    ax.hlines(y, baseline, values, colors=STEM_COLOR, linewidth=1.0, zorder=1)

    for idx, (yi, xi) in enumerate(zip(y, values)):
        if idx == SELECTED_INDEX:
            ax.scatter(
                xi,
                yi,
                s=46,
                color=SELECTED_FACE,
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                zorder=4,
            )
        elif idx == TOKEN_INDEX:
            ax.scatter(
                xi,
                yi,
                s=42,
                color=TOKEN_FACE,
                edgecolor=EDGE_COLOR,
                linewidth=1.0,
                marker="D",
                zorder=4,
            )
        else:
            ax.scatter(
                xi,
                yi,
                s=34,
                color=BASE_FACE,
                edgecolor=EDGE_COLOR,
                linewidth=0.7,
                zorder=3,
            )

    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(INTERVENTIONS)
        ax.tick_params(axis="y", pad=4)
        for label in ax.get_yticklabels():
            label.set_linespacing(0.95)
            label.set_va("center")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel(xlabel)
    ax.set_xlim(*ylim)
    ax.invert_yaxis()
    soften_axes(ax, grid_axis="x")
    add_panel_label(ax, panel_label)


def build_figure(add_title: bool):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=False, sharey=True)

    draw_lollipop_panel(
        axes[0],
        EXPECTED_HARM,
        xlabel="Expected harm",
        panel_label="A",
        ylim=(0.078, 0.1115),
        show_labels=True,
    )
    draw_lollipop_panel(
        axes[1],
        ECE,
        xlabel="ECE",
        panel_label="B",
        ylim=(0.019, 0.0365),
        show_labels=False,
    )

    title_artist = None
    if add_title:
        title_artist = fig.suptitle(
            "Intervention benefit across the representation-repair progression",
            fontsize=9.8,
            y=0.965,
        )
    fig.subplots_adjust(left=0.28, right=0.995, bottom=0.13, top=0.84 if add_title else 0.91, wspace=0.10)
    return fig, title_artist


def main() -> None:
    apply_journal_style()

    out_dir = Path("outputs/paper_figures/intervention_progression")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "intervention_progression"

    fig, title_artist = build_figure(add_title=True)
    save_figure_variants(fig, base, title_artist=title_artist)


if __name__ == "__main__":
    main()
