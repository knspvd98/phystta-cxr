#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow direct script execution without requiring manual PYTHONPATH setup.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from phystta.config.loader import load_config, save_config
from phystta.models.encoder_factory import EncoderFactory
from phystta.models.cnn import CNNClassifier
from phystta.utils import set_global_seed


def _select_profiles(cfg, mode: str) -> List[str]:
    if mode == "none":
        return []
    if mode == "study":
        return list(cfg.architecture_study.encoder_profiles or [])
    if mode == "all":
        return [p.name for p in (cfg.model.encoder_profiles or [])]
    raise ValueError(f"Unknown profile mode: {mode}")


def _collect_encoders(cfg, profile_mode: str) -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = [("__default__", cfg.model.encoder)]
    for profile in _select_profiles(cfg, profile_mode):
        enc = EncoderFactory.resolve_encoder_config(cfg, profile=profile)
        out.append((profile, enc))
    uniq = {}
    for name, enc in out:
        uniq[name] = enc
    return list(uniq.items())


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap local model cache for paper-tier offline runs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--profiles", default="study", choices=["none", "study", "all"])
    ap.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache dir")
    ap.add_argument(
        "--cnn_weights_dir",
        default="outputs/weights",
        help="Where to export baseline_cnn weights when --download is enabled and encoder_path is missing.",
    )
    ap.add_argument("--out_report", default="outputs/bootstrap/model_cache_report.json")
    ap.add_argument(
        "--set_config",
        action="store_true",
        help="Persist resolved local encoder_path values for exported baseline_cnn weights back into config.",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Allow network downloads for missing HF snapshots. "
        "Default is verify-only (local_files_only).",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.download and not cfg.runtime.allow_download:
        raise RuntimeError(
            "cache_models --download requires runtime.allow_download=true. "
            "For paper-tier use verify-only mode (omit --download)."
        )
    report_path = Path(args.out_report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    errors: List[str] = []
    updates: Dict[str, str] = {}
    cnn_weights_dir = Path(args.cnn_weights_dir).expanduser()
    if args.download and not args.dry_run:
        cnn_weights_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)

    for name, enc in _collect_encoders(cfg, args.profiles):
        src = str(getattr(enc, "encoder_source", ""))
        model_id = getattr(enc, "model_id", None)
        revision = getattr(enc, "revision", None)
        row: Dict[str, object] = {
            "profile": name,
            "encoder_source": src,
            "model_id": model_id,
            "revision": revision,
            "status": "skipped",
        }
        if src == "hf_clip_or_vit" and model_id:
            if getattr(cfg.experiment, "tier", "dev") == "paper" and not cfg._is_immutable_revision(revision):
                row["status"] = "error"
                row["error"] = "non_immutable_revision"
                errors.append(f"{name}: non-immutable revision for model_id={model_id}")
                items.append(row)
                continue
            if args.dry_run:
                row["status"] = "planned_download" if args.download else "planned_verify"
            else:
                try:
                    from huggingface_hub import snapshot_download
                    local_dir = snapshot_download(
                        repo_id=model_id,
                        revision=revision,
                        cache_dir=args.cache_dir,
                        local_files_only=not args.download,
                    )
                    row["status"] = "downloaded" if args.download else "verified"
                    row["local_dir"] = str(local_dir)
                except Exception as e:
                    err_type = type(e).__name__
                    err_msg = f"{err_type}: {e}"
                    if not args.download and ("LocalEntryNotFoundError" in err_type or "local" in str(e).lower()):
                        row["status"] = "missing"
                        row["error"] = err_msg
                    else:
                        row["status"] = "error"
                        row["error"] = err_msg
                        errors.append(f"{name}: {err_msg}")
        elif src == "openclip":
            row["status"] = "manual"
            row["note"] = "Provide local encoder_path for offline paper-tier execution."
        elif src == "baseline_cnn":
            arch = str(getattr(enc, "architecture", None) or cfg.model.architecture)
            local_path = Path(enc.encoder_path).expanduser() if getattr(enc, "encoder_path", None) is not None else None
            if local_path is None:
                local_path = cnn_weights_dir / f"{name}_{arch}.pt"
            use_medical_pretrain = bool(cfg.model.use_medical_pretrain and args.download)
            row["architecture"] = arch
            row["local_path"] = str(local_path)
            row["use_medical_pretrain"] = use_medical_pretrain
            if args.dry_run:
                if local_path.exists():
                    row["status"] = "planned_verify"
                else:
                    row["status"] = "planned_export" if args.download else "planned_missing"
            else:
                if local_path.exists():
                    row["status"] = "verified"
                elif not args.download:
                    row["status"] = "missing"
                    row["error"] = "baseline_cnn weights not found locally; rerun with --download to export."
                else:
                    try:
                        model = CNNClassifier(
                            architecture=arch,
                            num_classes=cfg.model.num_classes,
                            in_chans=cfg.model.in_chans,
                            pretrained=True,
                            dropout=cfg.model.dropout,
                            use_medical_pretrain=use_medical_pretrain,
                        )
                        torch.save({"model": model.state_dict()}, local_path)
                        row["status"] = "exported"
                    except Exception as e:
                        row["status"] = "error"
                        row["error"] = f"{type(e).__name__}: {e}"
                        errors.append(f"{name}: {row['error']}")
            updates[name] = str(local_path)
        elif src in {"eva_x", "chexfound", "llava_rad_encoder", "timm_vit"}:
            row["status"] = "local_or_builtin"
        else:
            row["status"] = "unsupported_source"
        items.append(row)

    if args.set_config and updates and not args.dry_run:
        cfg_to_save = load_config(args.config)
        for profile_name, enc_path in updates.items():
            if profile_name == "__default__":
                if cfg_to_save.model.encoder.encoder_source == "baseline_cnn":
                    cfg_to_save.model.encoder.encoder_path = Path(enc_path)
                continue
            for p in cfg_to_save.model.encoder_profiles or []:
                if p.name == profile_name and p.encoder.encoder_source == "baseline_cnn":
                    p.encoder.encoder_path = Path(enc_path)
                    break
        save_config(args.config, cfg_to_save)

    payload = {
        "config": str(Path(args.config).expanduser()),
        "experiment_tier": cfg.experiment.tier,
        "dry_run": bool(args.dry_run),
        "download": bool(args.download),
        "cache_dir": args.cache_dir,
        "cnn_weights_dir": str(cnn_weights_dir),
        "set_config": bool(args.set_config),
        "items": items,
        "errors": errors,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved cache bootstrap report: {report_path}")
    if errors and not args.dry_run:
        raise RuntimeError("Model cache bootstrap failed. See report for details.")


if __name__ == "__main__":
    main()
