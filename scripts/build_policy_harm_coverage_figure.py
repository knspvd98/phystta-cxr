#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import apply_journal_style, save_figure_variants, soften_axes


STATIC = ("Static inference", 0.850, 0.104)
POLICY_SEQUENCE = [
    ("Legacy selective rule", 0.670, 0.101),
    ("Expected-risk policy", 0.820, 0.087),
    ("Expected-harm-aware policy", 0.830, 0.082),
    ("Oracle-like upper bound", 0.836, 0.080),
]

SEQUENCE_COLOR = "#7a7a7a"
STATIC_COLOR = "#9a9a9a"
FINAL_COLOR = "#1f1f1f"
ORACLE_COLOR = "#4e4e4e"
EDGE_COLOR = "#202020"
LABEL_BOX = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}


def draw_sequence(ax: plt.Axes) -> None:
    x = [item[1] for item in POLICY_SEQUENCE[:-1]]
    y = [item[2] for item in POLICY_SEQUENCE[:-1]]
    ax.plot(x, y, color=SEQUENCE_COLOR, linewidth=1.0, zorder=1)
    ax.plot(
        [POLICY_SEQUENCE[-2][1], POLICY_SEQUENCE[-1][1]],
        [POLICY_SEQUENCE[-2][2], POLICY_SEQUENCE[-1][2]],
        color=ORACLE_COLOR,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=1,
    )


def draw_points(ax: plt.Axes) -> None:
    for name, coverage, harm in POLICY_SEQUENCE:
        if name == "Expected-harm-aware policy":
            ax.scatter(
                coverage,
                harm,
                s=54,
                color=FINAL_COLOR,
                edgecolor=EDGE_COLOR,
                linewidth=0.9,
                zorder=3,
            )
        elif name == "Oracle-like upper bound":
            ax.scatter(
                coverage,
                harm,
                s=44,
                color="white",
                edgecolor=ORACLE_COLOR,
                linewidth=1.0,
                marker="D",
                zorder=3,
            )
        elif name == "Legacy selective rule":
            ax.scatter(
                coverage,
                harm,
                s=42,
                color="#b0b0b0",
                edgecolor=EDGE_COLOR,
                linewidth=0.7,
                marker="^",
                zorder=3,
            )
        else:
            ax.scatter(
                coverage,
                harm,
                s=40,
                color="#8a8a8a",
                edgecolor=EDGE_COLOR,
                linewidth=0.7,
                zorder=3,
            )

    ax.scatter(
        STATIC[1],
        STATIC[2],
        s=44,
        color="white",
        edgecolor=STATIC_COLOR,
        linewidth=1.1,
        marker="s",
        zorder=3,
    )


def add_labels(ax: plt.Axes) -> None:
    offsets = {
        "Legacy selective rule": (0.005, 0.0012, "left"),
        "Expected-risk policy": (-0.018, 0.0014, "right"),
        "Expected-harm-aware policy": (-0.012, 0.0010, "right"),
        "Oracle-like upper bound": (0.005, -0.0003, "left"),
        "Static inference": (-0.004, 0.0013, "right"),
    }
    for name, coverage, harm in POLICY_SEQUENCE + [STATIC]:
        dx, dy, ha = offsets[name]
        ax.text(
            coverage + dx,
            harm + dy,
            name,
            fontsize=8.3,
            ha=ha,
            va="center",
            bbox=LABEL_BOX,
        )


def build_figure(add_title: bool):
    fig, ax = plt.subplots(figsize=(6.35, 4.45), constrained_layout=False)
    draw_sequence(ax)
    draw_points(ax)
    add_labels(ax)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Expected harm")
    ax.set_xlim(0.655, 0.858)
    ax.set_ylim(0.0772, 0.1065)
    soften_axes(ax, grid_axis="both")
    title_artist = None
    if add_title:
        title_artist = fig.suptitle(
            "Harm–coverage behavior of static, legacy selective, and final decision-aware policies",
            fontsize=9.8,
            y=0.965,
        )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.14, top=0.86 if add_title else 0.92)
    return fig, title_artist


def main() -> None:
    apply_journal_style()

    out_dir = Path("outputs/paper_figures/policy_harm_coverage")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "policy_harm_coverage"

    fig, title_artist = build_figure(add_title=True)
    save_figure_variants(fig, base, title_artist=title_artist)


if __name__ == "__main__":
    main()
