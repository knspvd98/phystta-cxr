#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.models import EncoderFactory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark loader-only and encoder-forward throughput on the same runtime data path as the future run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch_sizes", default="4,8,16")
    parser.add_argument("--num_workers_grid", default="0,2,4")
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--extra_metadata_csv", default=None)
    return parser


def _parse_csv_ints(value: str, field: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise RuntimeError(f"{field} must contain at least one integer.")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise RuntimeError(f"{field} must be a comma-separated integer list.") from exc


def _apply_path_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if not getattr(cfg.data, "sources", None):
        return
    src = cfg.data.sources[0]
    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root).expanduser()
        src.root = dataset_root
        src.image_root = Path(args.image_root).expanduser() if args.image_root is not None else dataset_root
        if args.metadata_csv is not None:
            src.metadata_csv = Path(args.metadata_csv).expanduser()
        elif src.metadata_csv is not None:
            src.metadata_csv = dataset_root / Path(src.metadata_csv).name
        if args.extra_metadata_csv is not None:
            src.extra_metadata_csv = Path(args.extra_metadata_csv).expanduser()
        elif src.extra_metadata_csv is not None:
            src.extra_metadata_csv = dataset_root / Path(src.extra_metadata_csv).name
    else:
        if args.image_root is not None:
            src.image_root = Path(args.image_root).expanduser()
        if args.metadata_csv is not None:
            src.metadata_csv = Path(args.metadata_csv).expanduser()
        if args.extra_metadata_csv is not None:
            src.extra_metadata_csv = Path(args.extra_metadata_csv).expanduser()


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _p95(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 95))


def _probe_loader_only(loader: Any, *, split: str, batch_size: int, num_workers: int, max_batches: int) -> Dict[str, Any]:
    latencies: List[float] = []
    n_images = 0
    iterator = iter(loader)
    started = time.perf_counter()
    n_batches = 0
    for _ in range(max_batches):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_end = time.perf_counter()
        latencies.append(batch_end - batch_start)
        n_batches += 1
        n_images += int(batch["image"].shape[0])
    elapsed = time.perf_counter() - started
    return {
        "mode": "loader_only",
        "split": split,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "n_batches": int(n_batches),
        "n_images": int(n_images),
        "elapsed_s": float(elapsed),
        "images_per_sec": float(n_images / elapsed) if elapsed > 0 and n_images > 0 else float("nan"),
        "first_batch_s": float(latencies[0]) if latencies else float("nan"),
        "mean_batch_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
        "p95_batch_latency_s": _p95(latencies),
        "status": "ok",
    }


def _probe_encoder_forward(
    loader: Any,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    device: str,
    encoder: Any,
) -> Dict[str, Any]:
    latencies: List[float] = []
    n_images = 0
    iterator = iter(loader)
    started = time.perf_counter()
    n_batches = 0
    with torch.inference_mode():
        for _ in range(max_batches):
            batch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            images = batch["image"].to(device, non_blocking=str(device).startswith("cuda"))
            _ = encoder.encode(images)
            _sync_if_cuda(device)
            batch_end = time.perf_counter()
            latencies.append(batch_end - batch_start)
            n_batches += 1
            n_images += int(images.shape[0])
    elapsed = time.perf_counter() - started
    return {
        "mode": "encoder_forward",
        "split": split,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "n_batches": int(n_batches),
        "n_images": int(n_images),
        "elapsed_s": float(elapsed),
        "images_per_sec": float(n_images / elapsed) if elapsed > 0 and n_images > 0 else float("nan"),
        "first_batch_s": float(latencies[0]) if latencies else float("nan"),
        "mean_batch_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
        "p95_batch_latency_s": _p95(latencies),
        "status": "ok",
    }


def _error_row(mode: str, split: str, batch_size: int, num_workers: int, exc: Exception) -> Dict[str, Any]:
    return {
        "mode": mode,
        "split": split,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "n_batches": 0,
        "n_images": 0,
        "elapsed_s": float("nan"),
        "images_per_sec": float("nan"),
        "first_batch_s": float("nan"),
        "mean_batch_latency_s": float("nan"),
        "p95_batch_latency_s": float("nan"),
        "status": f"error:{type(exc).__name__}:{exc}",
    }


def _best_row(frame: pd.DataFrame, mode: str) -> Dict[str, Any] | None:
    if frame.empty:
        return None
    subset = frame[(frame["mode"] == mode) & (frame["status"] == "ok")].copy()
    if subset.empty:
        return None
    subset["images_per_sec"] = pd.to_numeric(subset["images_per_sec"], errors="coerce")
    subset = subset[np.isfinite(subset["images_per_sec"].to_numpy(dtype=float))]
    if subset.empty:
        return None
    row = subset.sort_values("images_per_sec", ascending=False).iloc[0]
    return {
        "mode": str(row["mode"]),
        "split": str(row["split"]),
        "batch_size": int(row["batch_size"]),
        "num_workers": int(row["num_workers"]),
        "images_per_sec": float(row["images_per_sec"]),
        "status": str(row["status"]),
    }


def _markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Throughput Probe",
        "",
        f"- status: {payload['status']}",
        f"- config_path: {payload['config_path']}",
        f"- split: {payload['split']}",
        f"- rows_n: {payload['rows_n']}",
        f"- recommended_mode: {payload.get('recommended_mode')}",
        f"- recommended_batch_size: {payload.get('recommended_batch_size')}",
        f"- recommended_num_workers: {payload.get('recommended_num_workers')}",
    ]
    if payload.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in payload["warnings"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parser().parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested device='cuda' but CUDA is not available.")

    batch_sizes = _parse_csv_ints(args.batch_sizes, "batch_sizes")
    num_workers_grid = _parse_csv_ints(args.num_workers_grid, "num_workers_grid")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    _apply_path_overrides(cfg, args)
    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()

    split_names = set(data_engine.index["split"].astype(str).tolist()) if data_engine.index is not None else set()
    if str(args.split) not in split_names:
        raise RuntimeError(f"Requested split {args.split!r} does not exist in config-derived splits: {sorted(split_names)}")

    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    encoder = EncoderFactory.from_config(cfg)
    if hasattr(encoder, "model"):
        encoder.model.to(args.device)
        encoder.model.eval()

    for batch_size in batch_sizes:
        for num_workers in num_workers_grid:
            loaders = data_engine.build_dataloaders(
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                splits=[str(args.split)],
            )
            loader = loaders[str(args.split)]
            try:
                rows.append(
                    _probe_loader_only(
                        loader,
                        split=str(args.split),
                        batch_size=int(batch_size),
                        num_workers=int(num_workers),
                        max_batches=int(args.max_batches),
                    )
                )
            except Exception as exc:
                warnings.append(f"loader_only_failed:b{batch_size}:w{num_workers}:{type(exc).__name__}")
                rows.append(_error_row("loader_only", str(args.split), int(batch_size), int(num_workers), exc))
            try:
                rows.append(
                    _probe_encoder_forward(
                        loader,
                        split=str(args.split),
                        batch_size=int(batch_size),
                        num_workers=int(num_workers),
                        max_batches=int(args.max_batches),
                        device=str(args.device),
                        encoder=encoder,
                    )
                )
            except Exception as exc:
                warnings.append(f"encoder_forward_failed:b{batch_size}:w{num_workers}:{type(exc).__name__}")
                rows.append(_error_row("encoder_forward", str(args.split), int(batch_size), int(num_workers), exc))

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "throughput_probe_rows.csv", index=False)

    best_loader = _best_row(frame, "loader_only")
    best_encoder = _best_row(frame, "encoder_forward")
    if best_encoder is not None:
        recommendation = best_encoder
        recommended_mode = "encoder_forward"
    elif best_loader is not None:
        recommendation = best_loader
        recommended_mode = "loader_only"
    else:
        recommendation = None
        recommended_mode = None

    payload = {
        "status": "ok" if recommendation is not None else "error",
        "config_path": str(Path(args.config).expanduser().resolve()),
        "split": str(args.split),
        "rows_n": int(len(frame)),
        "recommended_mode": recommended_mode,
        "recommended_batch_size": int(recommendation["batch_size"]) if recommendation is not None else None,
        "recommended_num_workers": int(recommendation["num_workers"]) if recommendation is not None else None,
        "best_loader_only": best_loader,
        "best_encoder_forward": best_encoder,
        "warnings": warnings,
    }

    (out_dir / "throughput_probe_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "throughput_probe.md").write_text(_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
