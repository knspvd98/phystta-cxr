#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phystta.shifts.physics import HeelEffectShift, ScatterShift


DEFAULT_IMAGE_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_0023_0.png"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_figures" / "figure2_technical_shifts"
DEFAULT_SOURCE_DIR = DEFAULT_OUTPUT_DIR / "source_image"
DEFAULT_SOURCE_PATH = DEFAULT_SOURCE_DIR / "MCUCXR_0023_0.png"

HEEL_INTENSITY = 0.6
SCATTER_INTENSITY = 0.4

WINDOW_PERCENTILES = (0.005, 0.995)
FOREGROUND_QUANTILE = 0.85
FOREGROUND_THRESHOLD_FRACTION = 0.12
PADDING_FRACTION = 0.035
MAX_WORKING_HEIGHT = 1536
MAX_EXPORT_HEIGHT = 1400
FIGURE2_SHOWN_EXAMPLES = ["clean", "heel-like", "scatter-like"]
FIGURE2_NOT_SHOWN_EXAMPLES = ["broader stress", "mixed stress"]
FIGURE2_METHODS_SENTENCE = "Figure 2 shows examples of clean, heel-like, and scatter-like images."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a publication-style technical shift figure from one real chest radiograph."
    )
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="Public source image URL.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Local path for the downloaded or already-available source image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figure panels, composite outputs, and metadata.",
    )
    return parser.parse_args()


def download_if_needed(image_url: str, input_path: Path) -> Path:
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        urllib.request.urlretrieve(image_url, input_path)
    return input_path


def load_grayscale(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def resize_float_uniform(image: np.ndarray, *, max_height: int) -> np.ndarray:
    if image.shape[0] <= max_height:
        return image
    scale = max_height / float(image.shape[0])
    new_width = max(1, int(round(image.shape[1] * scale)))
    pil_image = Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resized = pil_image.resize((new_width, max_height), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def compute_crop_box(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape
    row_signal = np.quantile(image, FOREGROUND_QUANTILE, axis=1)
    col_signal = np.quantile(image, FOREGROUND_QUANTILE, axis=0)
    row_thresh = max(0.03, float(row_signal.max()) * FOREGROUND_THRESHOLD_FRACTION)
    col_thresh = max(0.03, float(col_signal.max()) * FOREGROUND_THRESHOLD_FRACTION)
    rows = np.flatnonzero(row_signal > row_thresh)
    cols = np.flatnonzero(col_signal > col_thresh)
    if rows.size == 0 or cols.size == 0:
        return (0, 0, w, h)
    pad_y = int(round(PADDING_FRACTION * h))
    pad_x = int(round(PADDING_FRACTION * w))
    y0 = max(0, int(rows[0]) - pad_y)
    y1 = min(h, int(rows[-1]) + pad_y + 1)
    x0 = max(0, int(cols[0]) - pad_x)
    x1 = min(w, int(cols[-1]) + pad_x + 1)
    return (x0, y0, x1, y1)


def crop_image(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    return image[y0:y1, x0:x1]


def apply_heel_like_from_repo(image: np.ndarray, *, intensity: float) -> np.ndarray:
    strength = max(0.0, min(1.0, float(intensity)))
    gamma = 0.45
    coord = np.linspace(-1.0, 1.0, image.shape[1], dtype=np.float32)
    t = np.clip((coord + 1.0) / 2.0, 0.0, 1.0)
    field = 1.0 - strength * np.power(t, gamma)
    return np.clip(image * field[None, :], 0.0, 1.0)


def apply_scatter_like_from_repo(image: np.ndarray, *, intensity: float) -> np.ndarray:
    scatter_ratio = max(0.0, min(1.2, float(intensity) * 1.2))
    scatter_sigma = 6.0 + 12.0 * scatter_ratio
    attenuation = -np.log(np.clip(image, 1e-6, 1.0))
    scatter = scatter_ratio * gaussian_filter(attenuation, sigma=scatter_sigma, mode="nearest") * image
    return np.clip(image + scatter, 0.0, 1.0)


def render_display_uint8(image: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    windowed = np.clip((image - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.round(windowed * 255.0).astype(np.uint8)


def resize_uniform(image: np.ndarray, *, max_height: int) -> np.ndarray:
    if image.shape[0] <= max_height:
        return image
    scale = max_height / float(image.shape[0])
    new_width = max(1, int(round(image.shape[1] * scale)))
    pil_image = Image.fromarray(image, mode="L")
    resized = pil_image.resize((new_width, max_height), resample=Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def save_panel_png(panel: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel, mode="L").save(out_path)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="black",
        bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def build_composite(
    panels: Iterable[dict[str, object]],
    output_dir: Path,
    *,
    vmin: float,
    vmax: float,
) -> list[Path]:
    panel_list = list(panels)
    n_panels = len(panel_list)
    fig_width = 2.8 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 3.45), facecolor="white")
    if n_panels == 1:
        axes = [axes]
    for ax, panel in zip(axes, panel_list):
        ax.imshow(panel["image"], cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor("#3a3a3a")
        add_panel_label(ax, str(panel["label"]))
        ax.text(
            0.5,
            -0.08,
            str(panel["caption"]),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#202020",
        )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.985, bottom=0.19, wspace=0.055)
    base = output_dir / "figure2_technical_shift_panel"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path, svg_path]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    input_path = download_if_needed(args.image_url, args.input_path.resolve())
    source_clean = load_grayscale(input_path)
    clean = resize_float_uniform(source_clean, max_height=MAX_WORKING_HEIGHT)
    crop_box = compute_crop_box(clean)

    heel_shift = HeelEffectShift()
    scatter_shift = ScatterShift()
    heel = apply_heel_like_from_repo(clean, intensity=HEEL_INTENSITY)
    scatter = apply_scatter_like_from_repo(clean, intensity=SCATTER_INTENSITY)

    clean_crop = crop_image(clean, crop_box)
    heel_crop = crop_image(heel, crop_box)
    scatter_crop = crop_image(scatter, crop_box)

    vmin = float(np.quantile(clean_crop, WINDOW_PERCENTILES[0]))
    vmax = float(np.quantile(clean_crop, WINDOW_PERCENTILES[1]))

    panel_specs = [
        {
            "label": "A",
            "caption": "Clean reference",
            "slug": "clean_reference",
            "image": clean_crop,
        },
        {
            "label": "B",
            "caption": "Heel-like shift",
            "slug": "heel_like_shift",
            "image": heel_crop,
        },
        {
            "label": "C",
            "caption": "Scatter-like shift",
            "slug": "scatter_like_shift",
            "image": scatter_crop,
        },
    ]

    saved_panel_paths: list[Path] = []
    for panel in panel_specs:
        display_uint8 = render_display_uint8(panel["image"], vmin=vmin, vmax=vmax)
        display_uint8 = resize_uniform(display_uint8, max_height=MAX_EXPORT_HEIGHT)
        out_path = panel_dir / f"panel_{panel['label']}_{panel['slug']}.png"
        save_panel_png(display_uint8, out_path)
        saved_panel_paths.append(out_path)
        panel["image"] = np.asarray(display_uint8, dtype=np.float32) / 255.0

    composite_paths = build_composite(panel_specs, output_dir, vmin=0.0, vmax=1.0)

    metadata = {
        "figure_title": "Examples of structured technical image shifts in chest radiography",
        "figure_spec": {
            "panel_count": len(panel_specs),
            "shown_examples": FIGURE2_SHOWN_EXAMPLES,
            "not_shown_examples": FIGURE2_NOT_SHOWN_EXAMPLES,
        },
        "source_image": {
            "dataset": "Montgomery County Chest X-ray Set",
            "local_path": str(input_path),
            "url": args.image_url,
            "filename": input_path.name,
            "original_shape_hw": [int(source_clean.shape[0]), int(source_clean.shape[1])],
            "working_shape_hw": [int(clean.shape[0]), int(clean.shape[1])],
        },
        "crop_box_xyxy": {
            "x0": int(crop_box[0]),
            "y0": int(crop_box[1]),
            "x1": int(crop_box[2]),
            "y1": int(crop_box[3]),
        },
        "crop_shape_hw": {
            "height": int(clean_crop.shape[0]),
            "width": int(clean_crop.shape[1]),
        },
        "display_window_quantiles": {
            "lower_quantile": WINDOW_PERCENTILES[0],
            "upper_quantile": WINDOW_PERCENTILES[1],
            "vmin": vmin,
            "vmax": vmax,
        },
        "export_scaling": {
            "max_working_height": MAX_WORKING_HEIGHT,
            "max_height": MAX_EXPORT_HEIGHT,
            "applied_uniformly_to_all_panels": True,
        },
        "parameters": {
            "heel_like_shift": {
                "intensity": HEEL_INTENSITY,
                "physical_params": heel_shift.physical_params(HEEL_INTENSITY),
            },
            "scatter_like_shift": {
                "intensity": SCATTER_INTENSITY,
                "physical_params": scatter_shift.physical_params(SCATTER_INTENSITY),
            },
        },
        "mixed_panel": {
            "included": False,
            "reason": (
                "Omitted to keep a tighter three-panel scientific comparison focused on the two "
                "principal technical perturbation families."
            ),
        },
        "outputs": {
            "individual_panels": [str(path) for path in saved_panel_paths],
            "composite_outputs": [str(path) for path in composite_paths],
        },
        "manuscript_alignment": {
            "figure_role": "Three-panel examples of the principal technical shift families only.",
            "methods_text_recommendation": FIGURE2_METHODS_SENTENCE,
            "forbidden_mentions_for_figure_text": FIGURE2_NOT_SHOWN_EXAMPLES,
            "rationale": (
                "Figure 2 intentionally omits broader or mixed stress so the visual comparison stays "
                "focused on the clean reference and the two principal shift families actually shown."
            ),
        },
        "preferred_caption": (
            "Figure 2. Examples of the principal technical image shifts applied to the same chest "
            "radiograph. Panel A shows the clean reference image, panel B a heel-like shift, "
            "and panel C a scatter-like shift."
        ),
    }
    metadata_path = output_dir / "figure2_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    note_path = output_dir / "figure2_notes.txt"
    note_path.write_text(
        "\n".join(
            [
                "Input image: Montgomery County Chest X-ray Set / MCUCXR_0023_0.png",
                f"Source URL: {args.image_url}",
                f"Crop box (x0, y0, x1, y1): {crop_box}",
                f"Display window quantiles: {WINDOW_PERCENTILES} -> ({vmin:.6f}, {vmax:.6f})",
                f"Heel-like intensity: {HEEL_INTENSITY}",
                f"Scatter-like intensity: {SCATTER_INTENSITY}",
                "Mixed stress: omitted for a cleaner three-panel figure.",
                "Figure role: examples of clean, heel-like, and scatter-like images only.",
                "Manuscript alignment: do not describe Figure 2 as showing broader stress or mixed stress.",
                f"Suggested Methods wording: {FIGURE2_METHODS_SENTENCE}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"metadata": str(metadata_path), "outputs": metadata["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
