from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phystta.intervention.eraser import load_eraser_collection, resolve_eraser_for_shift
from phystta.intervention.probes import load_probes
from phystta.utils import load_torch_artifact


def _parse_bank_arg(values: Sequence[str]) -> List[Tuple[str, str]]:
    banks: List[Tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"Bank spec must be NAME=PATH, got: {raw!r}"
            )
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid bank spec: {raw!r}")
        banks.append((name, path))
    if not banks:
        raise ValueError("At least one --bank NAME=PATH is required.")
    return banks


def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.as_tensor(x, dtype=torch.float32)


def _try_load_probes(path: str) -> Dict:
    probe_path = Path(path)
    try:
        payload = load_probes(str(probe_path))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    artifact = load_torch_artifact(str(probe_path), map_location="cpu")
    if isinstance(artifact, dict):
        if "probes" in artifact and isinstance(artifact["probes"], dict):
            return artifact["probes"]
        if "features_clean" in artifact and "features_shifted" in artifact:
            return artifact
    raise RuntimeError(f"Could not load probes from {probe_path}")


def _first_present(d: Dict, keys: Iterable[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _normalize_string_array(x, n_expected: int, name: str) -> np.ndarray:
    if x is None:
        raise RuntimeError(f"Missing required per-sample field: {name}")
    arr = np.asarray(x)
    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"Field {name!r} has length {arr.shape[0]}, expected {n_expected}"
        )
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr.astype(str)
    return arr.astype(str)


def _normalize_float_array(x, n_expected: int, name: str) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"Field {name!r} has length {arr.shape[0]}, expected {n_expected}"
        )
    return arr


def _extract_probe_fields(probes: Dict) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    features_clean = _ensure_tensor(
        _first_present(
            probes,
            (
                "features_clean",
                "clean_features",
            ),
        )
    )
    features_shifted = _ensure_tensor(
        _first_present(
            probes,
            (
                "features_shifted",
                "shifted_features",
            ),
        )
    )
    if features_clean.ndim != 2 or features_shifted.ndim != 2:
        raise RuntimeError("Probe features must be rank-2 tensors [N, D].")
    if features_clean.shape != features_shifted.shape:
        raise RuntimeError(
            f"Probe feature shape mismatch: clean={tuple(features_clean.shape)} shifted={tuple(features_shifted.shape)}"
        )

    n = int(features_clean.shape[0])

    family = _normalize_string_array(
        _first_present(
            probes,
            (
                "shift_family",
                "shift_families",
                "shift_type",
                "shift_types",
                "technical_shift",
                "technical_shift_type",
            ),
        ),
        n,
        "shift_family",
    )

    intensity = _normalize_float_array(
        _first_present(
            probes,
            (
                "shift_intensity",
                "shift_intensities",
                "intensity",
                "intensities",
                "realism_level",
                "realism_levels",
            ),
        ),
        n,
        "shift_intensity",
    )

    return features_clean, features_shifted, family, intensity


def _load_head_module(head_path: str, head_meta_path: Optional[str]) -> nn.Module:
    artifact = load_torch_artifact(head_path, map_location="cpu")

    if isinstance(artifact, nn.Module):
        artifact.eval()
        return artifact

    if isinstance(artifact, dict):
        for key in ("head", "model", "module"):
            candidate = artifact.get(key)
            if isinstance(candidate, nn.Module):
                candidate.eval()
                return candidate

    meta = {}
    if head_meta_path:
        meta = json.loads(Path(head_meta_path).read_text(encoding="utf-8"))

    state_dict = None
    if isinstance(artifact, dict):
        if isinstance(artifact.get("state_dict"), dict):
            state_dict = artifact["state_dict"]
        elif "weight" in artifact:
            state_dict = artifact

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Could not resolve a linear head from {head_path}")

    weight_tensor = None
    bias_tensor = None
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 2 and key.endswith("weight"):
            weight_tensor = value.detach().cpu().float()
            bias_key = key[:-6] + "bias"
            if bias_key in state_dict and torch.is_tensor(state_dict[bias_key]):
                bias_tensor = state_dict[bias_key].detach().cpu().float()
            break

    if weight_tensor is None:
        raise RuntimeError(f"Could not find linear weight tensor in {head_path}")

    out_features = int(weight_tensor.shape[0])
    in_features = int(weight_tensor.shape[1])

    if meta:
        feat_dim = int(meta.get("feat_dim", in_features))
        num_classes = int(meta.get("num_classes", out_features))
        if feat_dim != in_features or num_classes != out_features:
            raise RuntimeError(
                f"Head/head_meta mismatch: weight={tuple(weight_tensor.shape)} "
                f"meta.feat_dim={feat_dim} meta.num_classes={num_classes}"
            )

    head = nn.Linear(in_features, out_features, bias=bias_tensor is not None)
    with torch.no_grad():
        head.weight.copy_(weight_tensor)
        if bias_tensor is not None:
            head.bias.copy_(bias_tensor)
    head.eval()
    return head


@torch.inference_mode()
def _batched_logits_probs(
    head: nn.Module,
    clean: torch.Tensor,
    shifted: torch.Tensor,
    erased: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if not (clean.shape == shifted.shape == erased.shape):
        raise RuntimeError("clean/shifted/erased features must have identical shapes")

    head = head.to(device)
    n = int(clean.shape[0])

    feat_delta_sum = 0.0
    logit_delta_sum = 0.0
    prob_delta_sum = 0.0
    shifted_vs_clean_logit_sum = 0.0
    erased_vs_clean_logit_sum = 0.0
    shifted_vs_clean_prob_sum = 0.0
    erased_vs_clean_prob_sum = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        c = clean[start:end].to(device, non_blocking=True)
        s = shifted[start:end].to(device, non_blocking=True)
        e = erased[start:end].to(device, non_blocking=True)

        lc = head(c)
        ls = head(s)
        le = head(e)

        pc = torch.sigmoid(lc)
        ps = torch.sigmoid(ls)
        pe = torch.sigmoid(le)

        feat_delta_sum += float((e - s).abs().sum().item())
        logit_delta_sum += float((le - ls).abs().sum().item())
        prob_delta_sum += float((pe - ps).abs().sum().item())

        shifted_vs_clean_logit_sum += float((ls - lc).abs().sum().item())
        erased_vs_clean_logit_sum += float((le - lc).abs().sum().item())

        shifted_vs_clean_prob_sum += float((ps - pc).abs().sum().item())
        erased_vs_clean_prob_sum += float((pe - pc).abs().sum().item())

    denom_feat = float(n * clean.shape[1])
    denom_logit = float(n * head.out_features)
    denom_prob = float(n * head.out_features)

    eps = 1e-12
    shifted_vs_clean_logit = shifted_vs_clean_logit_sum / max(denom_logit, eps)
    erased_vs_clean_logit = erased_vs_clean_logit_sum / max(denom_logit, eps)
    shifted_vs_clean_prob = shifted_vs_clean_prob_sum / max(denom_prob, eps)
    erased_vs_clean_prob = erased_vs_clean_prob_sum / max(denom_prob, eps)

    return {
        "feature_mean_abs_delta": feat_delta_sum / max(denom_feat, eps),
        "logit_mean_abs_delta": logit_delta_sum / max(denom_logit, eps),
        "prob_mean_abs_delta": prob_delta_sum / max(denom_prob, eps),
        "shifted_vs_clean_logit_abs": shifted_vs_clean_logit,
        "erased_vs_clean_logit_abs": erased_vs_clean_logit,
        "shifted_vs_clean_prob_abs": shifted_vs_clean_prob,
        "erased_vs_clean_prob_abs": erased_vs_clean_prob,
        "recovery_fraction_logit": (
            1.0 - (erased_vs_clean_logit / max(shifted_vs_clean_logit, eps))
            if shifted_vs_clean_logit > eps
            else float("nan")
        ),
        "recovery_fraction_prob": (
            1.0 - (erased_vs_clean_prob / max(shifted_vs_clean_prob, eps))
            if shifted_vs_clean_prob > eps
            else float("nan")
        ),
    }


def _mask_for_subset(
    family: np.ndarray,
    intensity: Optional[np.ndarray],
    family_name: str,
    intensity_value: Optional[float],
    intensity_tol: float,
) -> np.ndarray:
    mask = family == family_name
    if intensity_value is not None:
        if intensity is None:
            return np.zeros_like(mask, dtype=bool)
        mask = mask & np.isclose(intensity, intensity_value, atol=intensity_tol, rtol=0.0)
    return mask


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _first_dict(x) -> Dict:
    return x if isinstance(x, dict) else {}


def _extract_bank_identity(
    *,
    bank_name: str,
    bank_path: str,
    collection_meta: Dict,
    eraser,
) -> Dict[str, object]:
    bundle_meta = _first_dict(collection_meta)
    eraser_meta = _first_dict(getattr(eraser, "meta", None))

    task_preserve = eraser_meta.get("task_preserve", bundle_meta.get("task_preserve"))
    preserve_strength = eraser_meta.get("preserve_strength", bundle_meta.get("preserve_strength"))

    recovery = eraser_meta.get("recovery_objective", bundle_meta.get("recovery_objective"))
    recovery = _first_dict(recovery)

    recovery_enabled = recovery.get("enabled")
    recovery_source = recovery.get("source")
    recovery_weights = _first_dict(recovery.get("weights"))

    logit_w = recovery_weights.get("logit_recovery_weight")
    prob_w = recovery_weights.get("prob_recovery_weight")
    clean_w = recovery_weights.get("clean_identity_weight")

    descriptor_parts = [str(bank_name)]
    if task_preserve:
        descriptor_parts.append(f"tp={task_preserve}")
    if preserve_strength is not None:
        try:
            descriptor_parts.append(f"ps={float(preserve_strength):.2f}")
        except Exception:
            descriptor_parts.append(f"ps={preserve_strength}")
    if recovery_enabled is True:
        descriptor_parts.append(f"rec={recovery_source or 'on'}")
    elif recovery_enabled is False:
        descriptor_parts.append("rec=off")

    return {
        "bank": str(bank_name),
        "bank_path": str(bank_path),
        "bank_kind": str(bundle_meta.get("kind", "unknown")),
        "task_preserve": (None if task_preserve is None else str(task_preserve)),
        "preserve_strength": (_safe_float(preserve_strength)),
        "recovery_enabled": (None if recovery_enabled is None else bool(recovery_enabled)),
        "recovery_source": (None if recovery_source is None else str(recovery_source)),
        "recovery_logit_weight": _safe_float(logit_w),
        "recovery_prob_weight": _safe_float(prob_w),
        "recovery_clean_weight": _safe_float(clean_w),
        "bank_descriptor": " | ".join(descriptor_parts),
    }


def _uniform_value(series: pd.Series):
    vals = [v for v in series.tolist() if pd.notna(v)]
    if not vals:
        return None
    first = vals[0]
    if isinstance(first, (bool, np.bool_)):
        return bool(first) if all(bool(v) == bool(first) for v in vals) else None
    if isinstance(first, (int, float, np.integer, np.floating)):
        return (
            float(first)
            if all(np.isclose(float(v), float(first), atol=1e-12, rtol=0.0) for v in vals)
            else None
        )
    return first if all(str(v) == str(first) for v in vals) else None


def _build_recommended_bank_metadata(summary_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if summary_df.empty:
        return None
    row = summary_df.iloc[0].to_dict()
    return {
        "bank": row.get("bank"),
        "bank_kind": row.get("bank_kind"),
        "task_preserve": row.get("task_preserve"),
        "preserve_strength": row.get("preserve_strength"),
        "recovery_enabled": row.get("recovery_enabled"),
        "recovery_source": row.get("recovery_source"),
        "recovery_logit_weight": row.get("recovery_logit_weight"),
        "recovery_prob_weight": row.get("recovery_prob_weight"),
        "recovery_clean_weight": row.get("recovery_clean_weight"),
        "bank_descriptor": row.get("bank_descriptor"),
        "score": row.get("score"),
    }


def _aggregate_bank(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for bank, g in df.groupby("bank"):
        nz = g[g["intensity"].fillna(-1.0) > 0.0]
        z = g[g["intensity"].fillna(-1.0) == 0.0]

        mean_nonzero_logit_recovery = float(nz["recovery_fraction_logit"].mean()) if len(nz) else float("nan")
        mean_nonzero_prob_recovery = float(nz["recovery_fraction_prob"].mean()) if len(nz) else float("nan")
        mean_nonzero_logit_delta = float(nz["logit_mean_abs_delta"].mean()) if len(nz) else float("nan")
        mean_nonzero_prob_delta = float(nz["prob_mean_abs_delta"].mean()) if len(nz) else float("nan")
        mean_zero_logit_damage = float(z["erased_vs_clean_logit_abs"].mean()) if len(z) else float("nan")
        mean_zero_prob_damage = float(z["erased_vs_clean_prob_abs"].mean()) if len(z) else float("nan")

        score = (
            np.nan_to_num(mean_nonzero_logit_recovery, nan=0.0)
            + np.nan_to_num(mean_nonzero_prob_recovery, nan=0.0)
            + 0.25 * np.nan_to_num(mean_nonzero_logit_delta, nan=0.0)
            + 0.25 * np.nan_to_num(mean_nonzero_prob_delta, nan=0.0)
            - np.nan_to_num(mean_zero_logit_damage, nan=0.0)
            - np.nan_to_num(mean_zero_prob_damage, nan=0.0)
        )

        rows.append(
            {
                "bank": bank,
                "bank_path": _uniform_value(g["bank_path"]),
                "bank_kind": _uniform_value(g["bank_kind"]),
                "task_preserve": _uniform_value(g["task_preserve"]),
                "preserve_strength": _uniform_value(g["preserve_strength"]),
                "recovery_enabled": _uniform_value(g["recovery_enabled"]),
                "recovery_source": _uniform_value(g["recovery_source"]),
                "recovery_logit_weight": _uniform_value(g["recovery_logit_weight"]),
                "recovery_prob_weight": _uniform_value(g["recovery_prob_weight"]),
                "recovery_clean_weight": _uniform_value(g["recovery_clean_weight"]),
                "bank_descriptor": _uniform_value(g["bank_descriptor"]),
                "n_rows": int(len(g)),
                "mean_nonzero_logit_recovery": mean_nonzero_logit_recovery,
                "mean_nonzero_prob_recovery": mean_nonzero_prob_recovery,
                "mean_nonzero_logit_delta": mean_nonzero_logit_delta,
                "mean_nonzero_prob_delta": mean_nonzero_prob_delta,
                "mean_zero_logit_damage": mean_zero_logit_damage,
                "mean_zero_prob_damage": mean_zero_prob_damage,
                "score": float(score),
            }
        )
    out = pd.DataFrame(rows).sort_values(["score", "bank"], ascending=[False, True]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjudicate existing eraser banks on bounded probe subsets.")
    parser.add_argument("--probes", required=True, help="Path to probes artifact (.pt).")
    parser.add_argument("--head", required=True, help="Path to trained linear head artifact (.pt).")
    parser.add_argument("--head_meta", default=None, help="Optional path to head_meta.json.")
    parser.add_argument(
        "--bank",
        action="append",
        required=True,
        help="Bank spec NAME=PATH, where PATH points to eraser_bank_manifest.json or a single eraser artifact.",
    )
    parser.add_argument(
        "--families",
        default="heel,scatter",
        help="Comma-separated shift families to evaluate.",
    )
    parser.add_argument(
        "--intensities",
        default="0.0,0.5,1.0",
        help="Comma-separated intensities to evaluate. Use 'all' to ignore intensity filtering.",
    )
    parser.add_argument("--intensity_tol", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes = _try_load_probes(args.probes)
    features_clean, features_shifted, family, intensity = _extract_probe_fields(probes)
    head = _load_head_module(args.head, args.head_meta)

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    if not families:
        raise RuntimeError("No families requested.")

    requested_intensities: List[Optional[float]]
    if args.intensities.strip().lower() == "all":
        requested_intensities = [None]
    else:
        requested_intensities = [float(x.strip()) for x in args.intensities.split(",") if x.strip()]

    banks = _parse_bank_arg(args.bank)
    device = torch.device(args.device)

    rows: List[Dict[str, object]] = []

    for bank_name, bank_path in banks:
        collection, meta = load_eraser_collection(bank_path)
        for fam in families:
            eraser = resolve_eraser_for_shift(collection, fam)
            if eraser is None:
                continue
            eraser = eraser.to(device).eval()

            for intensity_value in requested_intensities:
                mask = _mask_for_subset(
                    family=family,
                    intensity=intensity,
                    family_name=fam,
                    intensity_value=intensity_value,
                    intensity_tol=float(args.intensity_tol),
                )
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue

                idx_t = torch.as_tensor(idx, dtype=torch.long)
                clean_subset = features_clean.index_select(0, idx_t)
                shifted_subset = features_shifted.index_select(0, idx_t)

                with torch.inference_mode():
                    erased_subset = []
                    for start in range(0, len(idx), args.batch_size):
                        end = min(start + args.batch_size, len(idx))
                        s = shifted_subset[start:end].to(device, non_blocking=True)
                        e = eraser(s).detach().cpu()
                        erased_subset.append(e)
                    erased_subset = torch.cat(erased_subset, dim=0)

                metrics = _batched_logits_probs(
                    head=head,
                    clean=clean_subset,
                    shifted=shifted_subset,
                    erased=erased_subset,
                    batch_size=args.batch_size,
                    device=device,
                )

                bank_identity = _extract_bank_identity(
                    bank_name=bank_name,
                    bank_path=str(bank_path),
                    collection_meta=meta,
                    eraser=eraser,
                )

                rows.append(
                    {
                        **bank_identity,
                        "family": fam,
                        "intensity": (None if intensity_value is None else float(intensity_value)),
                        "n_samples": int(len(idx)),
                        **metrics,
                    }
                )

    if not rows:
        raise RuntimeError("No adjudication rows were produced. Check family/intensity filters and probe contents.")

    detail_df = pd.DataFrame(rows).sort_values(
        ["bank", "family", "intensity"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary_df = _aggregate_bank(detail_df)

    detail_path = out_dir / "delta_summary.csv"
    summary_path = out_dir / "delta_summary_by_bank.csv"
    json_path = out_dir / "delta_summary.json"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    recommended_bank = None
    if len(summary_df):
        recommended_bank = str(summary_df.iloc[0]["bank"])

    payload = {
        "recommended_bank": recommended_bank,
        "recommended_bank_metadata": _build_recommended_bank_metadata(summary_df),
        "families": families,
        "intensities": (
            "all"
            if requested_intensities == [None]
            else [float(x) for x in requested_intensities]
        ),
        "n_rows": int(len(detail_df)),
        "summary_by_bank": summary_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {detail_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {json_path}")
    if recommended_bank is not None:
        print(f"recommended_bank={recommended_bank}")


if __name__ == "__main__":
    main()
