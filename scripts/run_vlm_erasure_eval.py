#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.shifts import ShiftManager
from phystta.intervention import load_eraser
from phystta.shifts.realism import load_calibrator, intensity_grid_from_calibrator
from phystta.vlm import load_vlm
from phystta.metrics import auroc_per_class, auprc_per_class, macro_average
from phystta.utils import set_global_seed, sample_ids_from_batch


def tensor_to_pil(x: torch.Tensor):
    from PIL import Image
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def load_raw_image(dataset, image_path: str, source_type: str | None = None) -> torch.Tensor:
    x = dataset._load_image(image_path)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if source_type == "lidc_drr":
        mu_max = getattr(dataset.preprocess_cfg, "mu_max", 6.0) if getattr(dataset, "preprocess_cfg", None) else 6.0
        x = dataset._mu_to_intensity(x, mu_max=mu_max)
    return x


def json_to_probs(j: dict, labels: list[str], uncertain_value: float = 0.5) -> np.ndarray:
    out = []
    for k in labels:
        v = j.get(k, "uncertain") if j else "uncertain"
        v = str(v).lower()
        if v in {"present", "yes", "positive", "true", "1"}:
            out.append(1.0)
        elif v in {"absent", "no", "negative", "false", "0"}:
            out.append(0.0)
        else:
            out.append(float(uncertain_value))
    return np.array(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM erasure eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    if not hasattr(cfg, "vlm") or cfg.vlm is None or not cfg.vlm.enabled:
        print("VLM config missing; skipping VLM eval")
        return
    if not cfg.vlm.model_path and not cfg.vlm.model_id:
        print("VLM model_id/model_path missing; skipping VLM eval")
        return

    try:
        vlm = load_vlm(
            model_id=cfg.vlm.model_id,
            model_path=str(cfg.vlm.model_path) if cfg.vlm.model_path else None,
            model_type=cfg.vlm.model_type,
            device=device,
            allow_download=cfg.runtime.allow_download,
            revision=cfg.vlm.revision,
            offline=cfg.vlm.offline,
        )
    except Exception as e:
        print(f"VLM load failed: {e}")
        return

    eraser = load_eraser(args.eraser)
    if hasattr(eraser, "to"):
        eraser = eraser.to(device)
    if hasattr(eraser, "eval"):
        eraser.eval()

    data_engine = DataEngine(cfg.data)
    data_engine.build_index()
    data_engine.create_splits()
    split = args.split if args.split is not None else cfg.data.eval_split
    loaders = data_engine.build_dataloaders(batch_size=1, num_workers=cfg.train.num_workers, splits=[split])
    loader = loaders[split]
    dataset = loader.dataset
    source_type_lookup = None
    if hasattr(dataset, "df") and "source_type" in dataset.df.columns:
        source_type_lookup = dict(zip(dataset.df["image_path"].tolist(), dataset.df["source_type"].tolist()))

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    shift_types = [args.shift_type] if args.shift_type else cfg.erasure_eval.shift_types
    if cfg.intervention.use_realism:
        if args.intensity_grid:
            raise RuntimeError("intensity_grid overridden by realism calibrator; remove --intensity_grid")
        if cfg.intervention.realism_path is None:
            raise RuntimeError("use_realism=true but realism_path not set")
        calib = load_calibrator(str(cfg.intervention.realism_path))
        intensity_map = intensity_grid_from_calibrator(calib, shift_types)
    else:
        grid = [float(v) for v in args.intensity_grid.split(",")] if args.intensity_grid else cfg.erasure_eval.intensity_grid
        intensity_map = {st: grid for st in shift_types}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "vlm_outputs.jsonl"
    rows = []
    invalid = 0
    total = 0

    labels = data_engine.label_cols
    prompt = cfg.vlm.prompt_template
    uncertain_value = cfg.vlm.uncertain_value

    for shift_type in shift_types:
        baseline_json_no = None
        baseline_json_er = None
        if shift_type == "mixed":
            profiles = cfg.intervention.mixed_profiles or []
            scenarios = []
            for idx, profile in enumerate(profiles):
                name = profile.get("name", f"mixed_{idx}")
                if "intensities" in profile:
                    intens = profile["intensities"]
                elif "components" in profile:
                    intens = profile["components"]
                else:
                    intens = {k: v for k, v in profile.items() if k != "name"}
                scenarios.append((name, intens))
        else:
            grid = intensity_map.get(shift_type)
            if grid is None:
                raise RuntimeError(f"No calibrated intensity grid for shift_type={shift_type}")
            scenarios = [(float(intensity), None) for intensity in grid]
        for scenario in scenarios:
            intensity = scenario[0]
            preds_no = []
            preds_er = []
            targets = []
            json_no_list = []
            json_er_list = []
            for i, batch in enumerate(loader):
                if i >= args.max_images:
                    break
                image_path = batch["image_path"][0]
                source_type = source_type_lookup.get(image_path) if source_type_lookup is not None else None
                x = load_raw_image(dataset, image_path, source_type=source_type).unsqueeze(0)
                y = batch["labels"].numpy()
                if shift_type == "mixed":
                    sample_ids = sample_ids_from_batch(batch)
                    x_shift, _meta = shift_manager.apply_batch_scenario(x, scenario[1], sample_ids=sample_ids)
                else:
                    sample_ids = sample_ids_from_batch(batch)
                    x_shift = shift_manager.apply_batch(x, shift_type, float(intensity), sample_ids=sample_ids)
                img = tensor_to_pil(x_shift[0])
                text_no, norm_no, valid_no = vlm.generate_json(img, prompt, labels, eraser=None, max_retries=cfg.vlm.json_max_retries)
                text_er, norm_er, valid_er = vlm.generate_json(img, prompt, labels, eraser=eraser, max_retries=cfg.vlm.json_max_retries)
                if not (valid_no and valid_er):
                    invalid += 1
                total += 1
                preds_no.append(json_to_probs(norm_no, labels, uncertain_value=uncertain_value))
                preds_er.append(json_to_probs(norm_er, labels, uncertain_value=uncertain_value))
                targets.append(y[0])
                json_no_list.append(norm_no)
                json_er_list.append(norm_er)
                jsonl_path.open("a").write(json.dumps({
                    "shift": shift_type,
                    "intensity": intensity,
                    "text_no": text_no,
                    "text_eraser": text_er,
                    "json_no": norm_no,
                    "json_eraser": norm_er,
                    "valid_no": bool(valid_no),
                    "valid_eraser": bool(valid_er),
                }) + "\n")

            if preds_no:
                probs_no = np.stack(preds_no, axis=0)
                probs_er = np.stack(preds_er, axis=0)
                targets = np.stack(targets, axis=0)
                auroc_no = macro_average(auroc_per_class(targets, probs_no))
                auroc_er = macro_average(auroc_per_class(targets, probs_er))
                auprc_no = macro_average(auprc_per_class(targets, probs_no))
                auprc_er = macro_average(auprc_per_class(targets, probs_er))
                row = {
                    "shift": shift_type,
                    "intensity": intensity,
                    "auroc_no": auroc_no,
                    "auroc_eraser": auroc_er,
                    "auprc_no": auprc_no,
                    "auprc_eraser": auprc_er,
                }
                if shift_type == "mixed":
                    row["profile"] = intensity
                if baseline_json_no is None:
                    baseline_json_no = json_no_list
                    baseline_json_er = json_er_list
                else:
                    if baseline_json_no is not None:
                        flips = []
                        for b, c in zip(baseline_json_no, json_no_list):
                            flips.append(any(b[k] != c[k] for k in labels))
                        row["flip_rate_no"] = float(np.mean(flips))
                    if baseline_json_er is not None:
                        flips = []
                        for b, c in zip(baseline_json_er, json_er_list):
                            flips.append(any(b[k] != c[k] for k in labels))
                        row["flip_rate_eraser"] = float(np.mean(flips))
                rows.append(row)

    invalid_rate = invalid / max(total, 1)
    (out_dir / "vlm_summary.json").write_text(json.dumps({"invalid_rate": invalid_rate, "valid_rate": 1.0 - invalid_rate}, indent=2))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "vlm_metrics.csv", index=False)


if __name__ == "__main__":
    main()
