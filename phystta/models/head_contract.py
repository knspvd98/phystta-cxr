from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import load_torch_artifact
from .encoder_factory import EncoderFactory
from .head_meta import load_head_meta, validate_head_meta


@dataclass(frozen=True)
class HeadContract:
    enc_cfg: Any
    head_path: Optional[Path]
    head_meta: Optional[Dict[str, Any]]
    head_artifact: Any = None


def _infer_head_feat_dim(obj: Any) -> Optional[int]:
    if obj is None:
        return None

    in_features = getattr(obj, "in_features", None)
    if in_features is not None:
        try:
            return int(in_features)
        except Exception:
            pass

    weight = getattr(obj, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) == 2:
        try:
            return int(shape[1])
        except Exception:
            pass

    if hasattr(obj, "state_dict"):
        try:
            return _infer_head_feat_dim(obj.state_dict())
        except Exception:
            pass

    if isinstance(obj, MappingABC):
        preferred_keys = [
            "weight",
            "classifier.weight",
            "head.weight",
            "linear.weight",
            "fc.weight",
        ]
        for key in preferred_keys:
            tensor = obj.get(key)
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) == 2:
                try:
                    return int(shape[1])
                except Exception:
                    pass
        for tensor in obj.values():
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) == 2:
                try:
                    return int(shape[1])
                except Exception:
                    pass

    return None



def _backfill_legacy_false_fingerprint_fields(meta: Dict[str, Any], *, enc_cfg: Any) -> Dict[str, Any]:
    fp = meta.get("fingerprint")
    legacy_fp = meta.get("encoder_fingerprint")

    if isinstance(fp, dict):
        fp_out = dict(fp)
    elif isinstance(legacy_fp, dict):
        fp_out = dict(legacy_fp)
    else:
        return meta

    changed = False

    for key in ("use_medical_pretrain", "effective_medical_pretrain"):
        current_val = fp_out.get(key, None)
        expected = getattr(enc_cfg, key, False)
        if current_val is None and expected in (False, None):
            fp_out[key] = False
            changed = True

    meta_out = dict(meta)

    # Всегда канонизируем в top-level fingerprint.
    meta_out["fingerprint"] = fp_out
    meta_out.pop("encoder_fingerprint", None)

    if not changed and isinstance(fp, dict):
        return meta

    return meta_out


def resolve_head_contract(
    cfg,
    *,
    label_cols: Optional[List[str]],
    encoder_profile: Optional[str] = None,
    require_head_path: bool = True,
    strict_meta: Optional[bool] = None,
    load_artifact: bool = True,
) -> HeadContract:
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=encoder_profile)
    raw_head_path = getattr(enc_cfg, "head_path", None)

    if raw_head_path is None:
        if require_head_path:
            raise RuntimeError(
                "Configured encoder profile has no head_path; this execution path requires an explicit trained head artifact."
            )
        return HeadContract(
            enc_cfg=enc_cfg,
            head_path=None,
            head_meta=None,
            head_artifact=None,
        )

    head_path = Path(raw_head_path).expanduser()
    if not head_path.exists():
        raise RuntimeError(f"Head artifact not found: {head_path}")

    if strict_meta is None:
        strict_meta = bool(
            getattr(cfg.runtime, "freeze_artifacts", False)
            or getattr(getattr(cfg, "experiment", None), "tier", "dev") == "paper"
        )
    else:
        strict_meta = bool(strict_meta)

    explicit_meta = getattr(enc_cfg, "head_meta_path", None)
    meta = load_head_meta(
        head_path,
        explicit_meta=Path(explicit_meta).expanduser() if explicit_meta is not None else None,
    )

    if meta is None:
        if strict_meta:
            raise RuntimeError(
                f"Head metadata not found for {head_path}. "
                "Provide encoder.head_meta_path or save a sidecar meta json."
            )
        return HeadContract(
            enc_cfg=enc_cfg,
            head_path=head_path,
            head_meta=None,
            head_artifact=load_torch_artifact(head_path, map_location='cpu') if load_artifact else None,
        )

    meta = _backfill_legacy_false_fingerprint_fields(meta, enc_cfg=enc_cfg)

    errs = validate_head_meta(
        meta,
        enc_cfg=enc_cfg,
        expected_num_classes=cfg.model.num_classes,
        expected_label_cols=label_cols,
    )

    head_artifact = None
    if load_artifact:
        head_artifact = load_torch_artifact(head_path, map_location="cpu")
        inferred_feat_dim = _infer_head_feat_dim(head_artifact)
        meta_feat_dim = meta.get("feat_dim")
        if inferred_feat_dim is not None and meta_feat_dim is not None:
            if int(inferred_feat_dim) != int(meta_feat_dim):
                errs.append(
                    f"head artifact feat_dim mismatch: artifact={inferred_feat_dim} meta={meta_feat_dim}"
                )

    if errs:
        raise RuntimeError("Head metadata validation failed: " + "; ".join(errs))

    return HeadContract(
        enc_cfg=enc_cfg,
        head_path=head_path,
        head_meta=meta,
        head_artifact=head_artifact,
    )
