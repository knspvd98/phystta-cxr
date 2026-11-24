from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _adapter_dict(enc_cfg) -> Dict[str, Any]:
    ia = getattr(enc_cfg, "input_adapter", None)
    if ia is None:
        return {}
    return {
        "repeat_to_3": bool(getattr(ia, "repeat_to_3", False)),
        "image_size": int(getattr(ia, "image_size", 0)) if getattr(ia, "image_size", None) is not None else None,
        "mean": list(getattr(ia, "mean", []) or []),
        "std": list(getattr(ia, "std", []) or []),
    }


def build_encoder_fingerprint(enc_cfg, feat_dim: Optional[int] = None) -> Dict[str, Any]:
    fp = {
        "encoder_source": getattr(enc_cfg, "encoder_source", None),
        "architecture": getattr(enc_cfg, "architecture", None),
        "model_id": getattr(enc_cfg, "model_id", None),
        "encoder_name": getattr(enc_cfg, "encoder_name", None),
        "revision": getattr(enc_cfg, "revision", None),
        "pretrained": getattr(enc_cfg, "pretrained", None),
        "use_medical_pretrain": getattr(enc_cfg, "use_medical_pretrain", None),
        "effective_medical_pretrain": getattr(enc_cfg, "effective_medical_pretrain", None),
        "encoder_path": str(getattr(enc_cfg, "encoder_path", "")) if getattr(enc_cfg, "encoder_path", None) is not None else None,
        "input_adapter": _adapter_dict(enc_cfg),
    }
    if feat_dim is not None:
        fp["feat_dim"] = int(feat_dim)
    return fp


def build_head_meta(
    *,
    enc_cfg,
    feat_dim: int,
    num_classes: int,
    label_cols: List[str],
    group_key: str,
    encoder_profile: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    return {
        "feat_dim": int(feat_dim),
        "num_classes": int(num_classes),
        "label_cols": list(label_cols),
        "group_key": str(group_key),
        "encoder_profile": encoder_profile,
        "seed": int(seed),
        "fingerprint": build_encoder_fingerprint(enc_cfg, feat_dim=feat_dim),
    }


def infer_head_meta_paths(head_path: Path) -> List[Path]:
    return [
        head_path.with_suffix(".meta.json"),
        head_path.with_name(head_path.stem + "_meta.json"),
        head_path.parent / "head_meta.json",
    ]


def load_head_meta(head_path: Path, explicit_meta: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    if explicit_meta is not None:
        candidates.append(Path(explicit_meta))
    candidates.extend(infer_head_meta_paths(Path(head_path)))
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def validate_head_meta(
    meta: Dict[str, Any],
    *,
    enc_cfg,
    expected_num_classes: Optional[int] = None,
    expected_label_cols: Optional[List[str]] = None,
) -> List[str]:
    errs: List[str] = []
    if not isinstance(meta, dict):
        return ["head meta is not a JSON object"]
    if expected_num_classes is not None and int(meta.get("num_classes", -1)) != int(expected_num_classes):
        errs.append(f"num_classes mismatch: meta={meta.get('num_classes')} expected={expected_num_classes}")
    if expected_label_cols is not None:
        mcols = meta.get("label_cols")
        if isinstance(mcols, list) and list(mcols) != list(expected_label_cols):
            errs.append("label_cols mismatch between head meta and current dataset config")
    fp = meta.get("fingerprint", {})
    if not isinstance(fp, dict):
        errs.append("missing fingerprint in head meta")
        return errs
    expected_fp = build_encoder_fingerprint(enc_cfg)
    for key in [
        "encoder_source",
        "architecture",
        "model_id",
        "revision",
        "pretrained",
        "use_medical_pretrain",
        "effective_medical_pretrain",
        "encoder_path",
    ]:
        mv = fp.get(key)
        ev = expected_fp.get(key)
        if mv != ev:
            errs.append(f"encoder fingerprint mismatch for '{key}': meta={mv} expected={ev}")
    if fp.get("input_adapter", {}) != expected_fp.get("input_adapter", {}):
        errs.append("encoder fingerprint mismatch for 'input_adapter'")
    if "feat_dim" in fp and "feat_dim" in meta and int(fp["feat_dim"]) != int(meta["feat_dim"]):
        errs.append("head meta inconsistent: fingerprint.feat_dim != meta.feat_dim")
    return errs
