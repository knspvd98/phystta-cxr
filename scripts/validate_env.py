#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

from phystta.config.loader import load_config


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "scipy",
    "sklearn",
    "PIL",
    "yaml",
]

OPTIONAL_MODULES = [
    "iterstrat",         # iterative-stratification
    "huggingface_hub",   # HF cache/bootstrap
    "transformers",
    "timm",
    "open_clip",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_imports(modules: list[str], errors: list[str], out: dict, required: bool) -> None:
    for name in modules:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            out[name] = f"missing: {type(e).__name__}: {e}"
            if required:
                errors.append(f"Missing required module '{name}': {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate runtime environment and paper-tier reproducibility preconditions.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/bootstrap/env_validation.json")
    ap.add_argument("--strict_optional", action="store_true", help="Treat optional modules as required.")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    payload: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(Path(args.config).expanduser()),
        "modules": {},
    }

    _check_imports(REQUIRED_MODULES, errors, payload["modules"], required=True)
    _check_imports(OPTIONAL_MODULES, errors if args.strict_optional else warnings, payload["modules"], required=args.strict_optional)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        errors.append(f"Config load failed: {type(e).__name__}: {e}")
        cfg = None

    if cfg is not None:
        payload["experiment_tier"] = cfg.experiment.tier
        payload["allow_download"] = bool(cfg.runtime.allow_download)
        payload["freeze_artifacts"] = bool(cfg.runtime.freeze_artifacts)
        payload["deterministic"] = bool(cfg.runtime.deterministic)
        payload["split_group_key"] = cfg.data.split_group_key

        if cfg.data.prebuilt_index_csv is not None:
            idx = Path(cfg.data.prebuilt_index_csv).expanduser()
            payload["prebuilt_index_csv"] = str(idx)
            if not idx.exists():
                errors.append(f"Prebuilt index not found: {idx}")
            else:
                got = _sha256(idx)
                payload["prebuilt_index_sha256_actual"] = got
                payload["prebuilt_index_sha256_expected"] = cfg.data.prebuilt_index_sha256
                if cfg.data.prebuilt_index_sha256 and got != cfg.data.prebuilt_index_sha256:
                    errors.append(
                        f"Prebuilt index hash mismatch: expected={cfg.data.prebuilt_index_sha256} got={got}"
                    )

        if cfg.experiment.tier == "paper" and cfg.runtime.allow_download:
            errors.append("paper-tier requires runtime.allow_download=false.")
        if cfg.experiment.tier == "paper" and not cfg.runtime.deterministic:
            errors.append("paper-tier requires runtime.deterministic=true.")
        if cfg.experiment.tier == "paper" and "missing" in str(payload["modules"].get("iterstrat", "missing")):
            errors.append(
                "paper-tier requires iterative-stratification (iterstrat) for patient-level multilabel splits. "
                "Install: pip install iterative-stratification"
            )
        if cfg.experiment.tier == "paper" and cfg.vlm.model_path is not None and "/path/to/" in str(cfg.vlm.model_path):
            errors.append(
                f"paper-tier forbids placeholder vlm.model_path '{cfg.vlm.model_path}'. "
                "Set vlm.model_path=null or provide a real local path."
            )

    payload["errors"] = errors
    payload["warnings"] = warnings
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved env validation report: {out_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
