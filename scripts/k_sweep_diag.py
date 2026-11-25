#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.intervention import load_eraser_collection, resolve_eraser_for_shift
from phystta.metrics import (
    auprc_per_class,
    auroc_per_class,
    expected_calibration_error,
    macro_average,
)
from phystta.metrics.risk import risk_coverage_inputs_multilabel
from phystta.utils.io import metric_report_json
from phystta.models import build_encoder, resolve_head_contract
from phystta.shifts import ShiftManager
from phystta.shifts.realism import load_calibrator, scenario_grid_from_calibrator
from phystta.utils import load_torch_artifact, require_group_key, sample_ids_from_batch, set_global_seed


VARIANT_ORDER = ["legacy", "k2", "k4", "k8"]
SAMPLE_ID_KEYS = ("global_image_id", "dicom_id", "study_id", "image_path")


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _configured_label_cols(cfg: Any) -> List[str]:
    if getattr(cfg.data, "label_cols", None):
        return list(cfg.data.label_cols)
    label_cols = sorted({col for src in cfg.data.sources for col in (src.label_cols or [])})
    return list(label_cols)


def _normalize_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _materialize_head_module(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        return obj["model"]
    raise RuntimeError(f"Unsupported head artifact type for k_sweep_diag: {type(obj).__name__}")


def _contract_snapshot(name: str, contract: Any) -> Dict[str, Any]:
    meta = contract.head_meta or {}
    return {
        "variant": name,
        "encoder_source": getattr(contract.enc_cfg, "encoder_source", None),
        "architecture": getattr(contract.enc_cfg, "architecture", None),
        "model_id": getattr(contract.enc_cfg, "model_id", None),
        "head_path": _normalize_path(contract.head_path),
        "feat_dim": meta.get("feat_dim"),
        "num_classes": meta.get("num_classes"),
        "label_cols": list(meta.get("label_cols", [])),
        "group_key": meta.get("group_key"),
    }


def _raise_contract_mismatch(reference: Dict[str, Any], current: Dict[str, Any]) -> None:
    diffs: List[str] = []
    for key in (
        "encoder_source",
        "architecture",
        "model_id",
        "head_path",
        "feat_dim",
        "num_classes",
        "label_cols",
        "group_key",
    ):
        if reference.get(key) != current.get(key):
            diffs.append(
                f"{key}: {reference['variant']}={reference.get(key)!r} "
                f"vs {current['variant']}={current.get(key)!r}"
            )
    if diffs:
        raise RuntimeError(
            "Head/encoder contract mismatch across variants:\n" + "\n".join(diffs)
        )


def _prepare_group_lookup(eval_df: pd.DataFrame, group_key: str) -> Dict[str, Dict[str, str]]:
    lookups: Dict[str, Dict[str, str]] = {}
    if group_key not in eval_df.columns:
        return lookups

    for sample_key in SAMPLE_ID_KEYS:
        if sample_key not in eval_df.columns:
            continue
        mapping: Dict[str, str] = {}
        ambiguous: set[str] = set()
        for _, row in eval_df[[sample_key, group_key]].iterrows():
            sample_value = row[sample_key]
            if pd.isna(sample_value):
                continue
            sample_id = str(sample_value)
            group_value = row[group_key]
            group_str = "__missing__" if pd.isna(group_value) or str(group_value) == "" else str(group_value)
            if sample_id in mapping and mapping[sample_id] != group_str:
                ambiguous.add(sample_id)
            else:
                mapping[sample_id] = group_str
        for sample_id in ambiguous:
            mapping.pop(sample_id, None)
        lookups[sample_key] = mapping
    return lookups


def _resolve_group_values(
    batch: Dict[str, Any],
    group_key: str,
    sample_ids: Optional[List[str]],
    lookups: Dict[str, Dict[str, str]],
    warnings: set[str],
) -> List[str]:
    batch_size = int(batch["image"].shape[0])

    if group_key in batch:
        values = []
        missing = False
        for value in batch[group_key]:
            value_str = str(value) if value is not None else ""
            if value_str == "" or value_str.lower() == "nan":
                values.append("__missing__")
                missing = True
            else:
                values.append(value_str)
        if missing:
            warnings.add("Some group values were missing in the batch; group_value='__missing__' was used.")
        return values

    source_key = next((key for key in SAMPLE_ID_KEYS if key in batch), None)
    if source_key is None or sample_ids is None:
        warnings.add("Sample IDs could not be mapped to groups; group_value='__missing__' was used.")
        return ["__missing__"] * batch_size

    mapping = lookups.get(source_key, {})
    values = []
    missing = False
    for sample_id in sample_ids:
        group_value = mapping.get(str(sample_id))
        if group_value is None:
            missing = True
            values.append("__missing__")
        else:
            values.append(group_value)
    if missing:
        warnings.add("Some sample IDs could not be mapped to groups; group_value='__missing__' was used.")
    return values


def _eval_metrics(probs: np.ndarray, labels: np.ndarray, coverage: float) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    valid = labels >= 0
    probs_flat = probs[valid].reshape(-1)
    labels_flat = labels[valid].reshape(-1)

    if probs_flat.size == 0:
        return {
            "ece": float("nan"),
            "auroc_macro": float("nan"),
            "auprc_macro": float("nan"),
            "risk_at_coverage": float("nan"),
            "coverage_realized": float("nan"),
            "abstain_rate": float("nan"),
        }

    metrics = {
        "ece": float(expected_calibration_error(probs_flat, labels_flat, n_bins=15)),
    }

    try:
        metrics["auroc_macro"] = float(macro_average(auroc_per_class(labels, probs)))
        metrics["auprc_macro"] = float(macro_average(auprc_per_class(labels, probs)))
    except Exception:
        metrics["auroc_macro"] = float("nan")
        metrics["auprc_macro"] = float("nan")

    errors, confidences = risk_coverage_inputs_multilabel(probs, labels, threshold=0.5)
    observed = np.isfinite(errors) & np.isfinite(confidences)
    errors = errors[observed]
    confidences = confidences[observed]
    if confidences.size == 0:
        metrics["risk_at_coverage"] = float("nan")
        metrics["coverage_realized"] = float("nan")
        metrics["abstain_rate"] = float("nan")
        return metrics

    target_coverage = float(np.clip(coverage, 0.0, 1.0))
    threshold = float(np.quantile(confidences, max(0.0, 1.0 - target_coverage)))
    accept = confidences >= threshold
    metrics["coverage_realized"] = float(np.mean(accept))
    metrics["abstain_rate"] = float(1.0 - metrics["coverage_realized"])
    metrics["risk_at_coverage"] = float(np.mean(errors[accept])) if np.any(accept) else float("nan")
    return metrics


def _mean(values: Iterable[float], *, default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.mean())


def _median(values: Iterable[float], *, default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _std(values: Iterable[float], *, default: float = float("nan")) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(arr.std(ddof=0))


def _summary_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _aggregate_scenario_rows(
    *,
    variant: str,
    scenario: Dict[str, float | str],
    sample_rows: List[Dict[str, Any]],
    metrics_shift: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, Any]:
    frame = pd.DataFrame(sample_rows)
    group_frame = (
        frame.groupby("group_value", dropna=False)
        .agg(
            group_prob_recovery_fraction=("prob_recovered", "mean"),
            group_logit_repair_mae=("logit_delta_shift_after_mae", "mean"),
        )
        .reset_index()
    )

    return {
        "variant": variant,
        "shift_family": str(scenario["shift_family"]),
        "realism_level": float(scenario["realism_level"]),
        "intensity": float(scenario["intensity"]),
        "n_samples": int(len(frame)),
        "n_groups": int(frame["group_value"].nunique(dropna=False)),
        "feat_delta_clean_shift_mean": _mean(frame["feat_delta_clean_shift"]),
        "feat_delta_clean_shift_median": _median(frame["feat_delta_clean_shift"]),
        "feat_delta_shift_after_mean": _mean(frame["feat_delta_shift_after"]),
        "feat_delta_shift_after_median": _median(frame["feat_delta_shift_after"]),
        "logit_delta_clean_shift_mae_mean": _mean(frame["logit_delta_clean_shift_mae"]),
        "logit_delta_shift_after_mae_mean": _mean(frame["logit_delta_shift_after_mae"]),
        "prob_delta_clean_shift_mae_mean": _mean(frame["prob_delta_clean_shift_mae"]),
        "prob_delta_shift_after_mae_mean": _mean(frame["prob_delta_shift_after_mae"]),
        "logit_delta_clean_after_mae_mean": _mean(frame["logit_delta_clean_after_mae"]),
        "prob_delta_clean_after_mae_mean": _mean(frame["prob_delta_clean_after_mae"]),
        "logit_clean_repair_gain_mean": _mean(frame["logit_clean_repair_gain"]),
        "prob_clean_repair_gain_mean": _mean(frame["prob_clean_repair_gain"]),
        "logit_delta_clean_afterclean_mae_mean": _mean(frame["logit_delta_clean_afterclean_mae"]),
        "prob_delta_clean_afterclean_mae_mean": _mean(frame["prob_delta_clean_afterclean_mae"]),
        "prob_recovery_fraction_mean": _mean(frame["prob_recovered"]),
        "ece_shift": float(metrics_shift["ece"]),
        "ece_after": float(metrics_after["ece"]),
        "auroc_macro_shift": float(metrics_shift["auroc_macro"]),
        "auroc_macro_after": float(metrics_after["auroc_macro"]),
        "auprc_macro_shift": float(metrics_shift["auprc_macro"]),
        "auprc_macro_after": float(metrics_after["auprc_macro"]),
        "risk_at_coverage_shift": float(metrics_shift["risk_at_coverage"]),
        "risk_at_coverage_after": float(metrics_after["risk_at_coverage"]),
        "coverage_realized_shift": float(metrics_shift["coverage_realized"]),
        "coverage_realized_after": float(metrics_after["coverage_realized"]),
        "abstain_rate_shift": float(metrics_shift["abstain_rate"]),
        "abstain_rate_after": float(metrics_after["abstain_rate"]),
        "group_prob_recovery_fraction_mean": _mean(group_frame["group_prob_recovery_fraction"]),
        "group_prob_recovery_fraction_std": _std(group_frame["group_prob_recovery_fraction"], default=0.0),
        "group_logit_repair_mae_mean": _mean(group_frame["group_logit_repair_mae"]),
    }


def _build_scenarios(
    cfg: Any,
    families: List[str],
    levels: List[float],
    warnings: set[str],
) -> List[Dict[str, float | str]]:
    scenarios: List[Dict[str, float | str]] = [
        {
            "shift_family": "none",
            "realism_level": 0.0,
            "intensity": 0.0,
        }
    ]

    realism_path = getattr(cfg.intervention, "realism_path", None)
    if realism_path is not None and Path(realism_path).exists():
        calibrator = load_calibrator(str(realism_path))
        selected_levels = {round(float(level), 6) for level in levels}
        scenario_map = scenario_grid_from_calibrator(calibrator, families)
        for family in families:
            for realism_level, intensity in scenario_map.get(family, []):
                if round(float(realism_level), 6) in selected_levels:
                    scenarios.append(
                        {
                            "shift_family": str(family),
                            "realism_level": float(realism_level),
                            "intensity": float(intensity),
                        }
                    )
    else:
        warnings.add("Realism calibrator missing; falling back to direct intensity values from --levels.")
        for family in families:
            for level in levels:
                scenarios.append(
                    {
                        "shift_family": str(family),
                        "realism_level": float(level),
                        "intensity": float(level),
                    }
                )
    return scenarios


def _run_variant_scenarios(
    *,
    variant: str,
    cfg: Any,
    eraser_collection: Dict[str, nn.Module],
    scenarios: List[Dict[str, float | str]],
    loader: Any,
    encoder: nn.Module,
    head: nn.Module,
    group_key: str,
    group_lookups: Dict[str, Dict[str, str]],
    coverage: float,
    device: torch.device,
    max_batches: Optional[int],
    warnings: set[str],
) -> List[Dict[str, Any]]:
    shift_manager = ShiftManager(seed=int(cfg.shifts.seed))
    rows: List[Dict[str, Any]] = []

    for eraser in eraser_collection.values():
        eraser.to(device)
        eraser.eval()

    for scenario in scenarios:
        shift_family = str(scenario["shift_family"])
        intensity = float(scenario["intensity"])
        scenario_eraser = None if shift_family == "none" else resolve_eraser_for_shift(eraser_collection, shift_family)
        sample_rows: List[Dict[str, Any]] = []
        probs_shift_list: List[np.ndarray] = []
        probs_after_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        batch_count = 0

        with torch.no_grad():
            for batch in loader:
                batch_count += 1

                x = batch["image"].to(device)
                y = batch["labels"].to(device)
                sample_ids_seq = sample_ids_from_batch(batch)
                if sample_ids_seq is not None:
                    sample_ids = [str(item) for item in sample_ids_seq]
                else:
                    sample_ids = [f"{variant}:{shift_family}:batch{batch_count}:idx{i}" for i in range(x.shape[0])]

                if shift_family == "none":
                    x_shift = x
                else:
                    x_shift, _ = shift_manager.apply_batch_with_meta(
                        x,
                        shift_family,
                        intensity,
                        sample_ids=sample_ids,
                    )

                z_clean = encoder.encode(x)
                logits_clean = head(z_clean)
                probs_clean = torch.sigmoid(logits_clean)

                z_shift = encoder.encode(x_shift)
                logits_shift = head(z_shift)
                probs_shift = torch.sigmoid(logits_shift)

                if scenario_eraser is None:
                    z_after = z_shift
                    z_after_clean = z_clean
                else:
                    z_after = scenario_eraser(z_shift)
                    z_after_clean = scenario_eraser(z_clean)

                logits_after = head(z_after)
                probs_after = torch.sigmoid(logits_after)

                logits_after_clean = head(z_after_clean)
                probs_after_clean = torch.sigmoid(logits_after_clean)

                group_values = _resolve_group_values(
                    batch,
                    group_key,
                    sample_ids,
                    group_lookups,
                    warnings,
                )

                z_clean_np = z_clean.detach().cpu().numpy()
                z_shift_np = z_shift.detach().cpu().numpy()
                z_after_np = z_after.detach().cpu().numpy()
                logits_clean_np = logits_clean.detach().cpu().numpy()
                logits_shift_np = logits_shift.detach().cpu().numpy()
                logits_after_np = logits_after.detach().cpu().numpy()
                logits_after_clean_np = logits_after_clean.detach().cpu().numpy()
                probs_clean_np = probs_clean.detach().cpu().numpy()
                probs_shift_np = probs_shift.detach().cpu().numpy()
                probs_after_np = probs_after.detach().cpu().numpy()
                probs_after_clean_np = probs_after_clean.detach().cpu().numpy()
                labels_np = y.detach().cpu().numpy()
                n_labels_valid = (labels_np >= 0).sum(axis=1)

                probs_shift_list.append(probs_shift_np)
                probs_after_list.append(probs_after_np)
                labels_list.append(labels_np)

                for idx in range(labels_np.shape[0]):
                    clean_shift_logit = np.abs(logits_shift_np[idx] - logits_clean_np[idx])
                    after_clean_logit = np.abs(logits_after_np[idx] - logits_clean_np[idx])
                    clean_afterclean_logit = np.abs(logits_after_clean_np[idx] - logits_clean_np[idx])
                    clean_shift_prob = np.abs(probs_shift_np[idx] - probs_clean_np[idx])
                    after_clean_prob = np.abs(probs_after_np[idx] - probs_clean_np[idx])
                    clean_afterclean_prob = np.abs(probs_after_clean_np[idx] - probs_clean_np[idx])
                    sample_rows.append(
                        {
                            "variant": variant,
                            "shift_family": shift_family,
                            "realism_level": float(scenario["realism_level"]),
                            "intensity": intensity,
                            "sample_id": sample_ids[idx],
                            "group_value": group_values[idx],
                            "feat_delta_clean_shift": float(np.linalg.norm(z_shift_np[idx] - z_clean_np[idx], ord=2)),
                            "feat_delta_shift_after": float(np.linalg.norm(z_after_np[idx] - z_shift_np[idx], ord=2)),
                            "logit_delta_clean_shift_mae": float(np.mean(clean_shift_logit)),
                            "logit_delta_shift_after_mae": float(np.mean(np.abs(logits_after_np[idx] - logits_shift_np[idx]))),
                            "prob_delta_clean_shift_mae": float(np.mean(clean_shift_prob)),
                            "prob_delta_shift_after_mae": float(np.mean(np.abs(probs_after_np[idx] - probs_shift_np[idx]))),
                            "prob_recovered": float(np.mean(after_clean_prob < clean_shift_prob)),
                            "logit_delta_clean_after_mae": float(np.mean(after_clean_logit)),
                            "prob_delta_clean_after_mae": float(np.mean(after_clean_prob)),
                            "logit_clean_repair_gain": float(np.mean(clean_shift_logit - after_clean_logit)),
                            "prob_clean_repair_gain": float(np.mean(clean_shift_prob - after_clean_prob)),
                            "logit_delta_clean_afterclean_mae": float(np.mean(clean_afterclean_logit)),
                            "prob_delta_clean_afterclean_mae": float(np.mean(clean_afterclean_prob)),
                            "n_labels_valid": int(n_labels_valid[idx]),
                        }
                    )

                if max_batches is not None and batch_count >= int(max_batches):
                    break

        probs_shift_all = np.concatenate(probs_shift_list, axis=0)
        probs_after_all = np.concatenate(probs_after_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)
        metrics_shift = _eval_metrics(probs_shift_all, labels_all, coverage=coverage)
        metrics_after = _eval_metrics(probs_after_all, labels_all, coverage=coverage)
        rows.append(
            _aggregate_scenario_rows(
                variant=variant,
                scenario=scenario,
                sample_rows=sample_rows,
                metrics_shift=metrics_shift,
                metrics_after=metrics_after,
            )
        )

    return rows


def _build_summary_rows(scenario_df: pd.DataFrame) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        variant_df = scenario_df[scenario_df["variant"] == variant].reset_index(drop=True)
        shifted_df = variant_df[variant_df["shift_family"] != "none"].reset_index(drop=True)

        mean_prob_recovery_fraction = _summary_mean(shifted_df["prob_recovery_fraction_mean"]) if not shifted_df.empty else 0.0
        mean_logit_repair_gain = _summary_mean(shifted_df["logit_clean_repair_gain_mean"]) if not shifted_df.empty else 0.0
        mean_prob_repair_gain = _summary_mean(shifted_df["prob_clean_repair_gain_mean"]) if not shifted_df.empty else 0.0
        mean_ece_delta = _summary_mean(shifted_df["ece_shift"] - shifted_df["ece_after"]) if not shifted_df.empty else 0.0
        mean_auroc_delta = _summary_mean(shifted_df["auroc_macro_after"] - shifted_df["auroc_macro_shift"]) if not shifted_df.empty else 0.0
        mean_auprc_delta = _summary_mean(shifted_df["auprc_macro_after"] - shifted_df["auprc_macro_shift"]) if not shifted_df.empty else 0.0
        clean_disturbance_prob_mae = _summary_mean(shifted_df["prob_delta_clean_afterclean_mae_mean"]) if not shifted_df.empty else 0.0
        clean_disturbance_logit_mae = _summary_mean(shifted_df["logit_delta_clean_afterclean_mae_mean"]) if not shifted_df.empty else 0.0

        winner_score = (
            2.0 * mean_prob_recovery_fraction
            + 2.0 * mean_prob_repair_gain
            + 1.5 * mean_logit_repair_gain
            + 1.0 * mean_ece_delta
            + 0.5 * mean_auroc_delta
            + 0.5 * mean_auprc_delta
            - 2.0 * clean_disturbance_prob_mae
            - 1.0 * clean_disturbance_logit_mae
        )

        summary_rows.append(
            {
                "variant": variant,
                "shifted_scenarios_n": int(len(shifted_df)),
                "mean_prob_recovery_fraction": float(mean_prob_recovery_fraction),
                "mean_logit_repair_gain": float(mean_logit_repair_gain),
                "mean_prob_repair_gain": float(mean_prob_repair_gain),
                "mean_ece_delta": float(mean_ece_delta),
                "mean_auroc_delta": float(mean_auroc_delta),
                "mean_auprc_delta": float(mean_auprc_delta),
                "clean_disturbance_prob_mae": float(clean_disturbance_prob_mae),
                "clean_disturbance_logit_mae": float(clean_disturbance_logit_mae),
                "winner_score": float(winner_score),
            }
        )
    return summary_rows


def _winner_payload(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_by_variant = {row["variant"]: row for row in summary_rows}
    winner_row = max(summary_rows, key=lambda row: float(row["winner_score"]))
    legacy_row = summary_by_variant["legacy"]
    nonlegacy_rows = [row for row in summary_rows if row["variant"] != "legacy"]
    best_nonlegacy = max(nonlegacy_rows, key=lambda row: float(row["winner_score"])) if nonlegacy_rows else None

    all_low_recovery = all(float(row["mean_prob_recovery_fraction"]) < 0.55 for row in summary_rows)
    all_nonpositive_repair = all(float(row["mean_prob_repair_gain"]) <= 0.0 for row in summary_rows)
    excessive_clean_disturbance = bool(
        best_nonlegacy is not None
        and float(best_nonlegacy["clean_disturbance_prob_mae"])
        > float(legacy_row["clean_disturbance_prob_mae"]) * 1.5
    )
    go_online = bool(
        winner_row["variant"] == "legacy"
        or all_low_recovery
        or all_nonpositive_repair
        or excessive_clean_disturbance
    )

    if winner_row["variant"] == "legacy":
        reason = "legacy remains the highest-scoring option under the bounded winner_score rule."
    elif all_nonpositive_repair:
        reason = "no variant shows positive probability-side repair gain, so online adaptor remains required."
    elif all_low_recovery:
        reason = "all variants stay below the 0.55 recovery threshold, so online adaptor remains required."
    elif excessive_clean_disturbance:
        reason = "the best non-legacy option disturbs clean predictions too much relative to legacy."
    else:
        reason = "the best non-legacy option wins on shifted recovery without excessive clean disturbance."

    return {
        "winner": str(winner_row["variant"]),
        "winner_score": float(winner_row["winner_score"]),
        "go_online_adaptor": bool(go_online),
        "reason": reason,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded k-sweep diagnostic over explicit eraser artifacts.")
    parser.add_argument("--legacy_config", required=True)
    parser.add_argument("--legacy_eraser", required=True)
    parser.add_argument("--k2_config", required=True)
    parser.add_argument("--k2_eraser", required=True)
    parser.add_argument("--k4_config", required=True)
    parser.add_argument("--k4_eraser", required=True)
    parser.add_argument("--k8_config", required=True)
    parser.add_argument("--k8_eraser", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--families", default="heel,scatter")
    parser.add_argument("--levels", default="0.5,1.0")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size_override", type=int, default=None)
    parser.add_argument("--split_override", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = _parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)

    families = _parse_csv_list(args.families)
    levels = _parse_csv_floats(args.levels)
    device = torch.device(args.device)
    warnings: set[str] = set()

    cfgs = {
        "legacy": load_config(args.legacy_config),
        "k2": load_config(args.k2_config),
        "k4": load_config(args.k4_config),
        "k8": load_config(args.k8_config),
    }

    contracts = {
        name: resolve_head_contract(
            cfg,
            label_cols=_configured_label_cols(cfg),
            require_head_path=True,
            strict_meta=True,
            load_artifact=True,
        )
        for name, cfg in cfgs.items()
    }

    reference_snapshot = _contract_snapshot("legacy", contracts["legacy"])
    for name in ("k2", "k4", "k8"):
        _raise_contract_mismatch(reference_snapshot, _contract_snapshot(name, contracts[name]))

    legacy_cfg = cfgs["legacy"]
    legacy_contract = contracts["legacy"]
    label_cols = list(legacy_contract.head_meta["label_cols"])
    group_key = str(legacy_contract.head_meta["group_key"])

    encoder = build_encoder(legacy_cfg)
    encoder.to(device)
    encoder.eval()

    head_obj = legacy_contract.head_artifact
    if head_obj is None:
        head_obj = load_torch_artifact(legacy_contract.head_path, map_location="cpu")
    head = _materialize_head_module(head_obj)
    head.to(device)
    head.eval()

    # Route any derived indexes/split manifests into out_dir to keep main artifacts read-only.
    legacy_cfg.data.index_dir = out_dir / "runtime_indexes"
    legacy_cfg.data.cache_dir = out_dir / "runtime_cache"

    data_engine = DataEngine(legacy_cfg.data)
    data_engine.build_index()
    data_engine.create_splits()
    eval_split = str(args.split_override or legacy_cfg.data.eval_split)
    eval_df = data_engine.get_split_df(eval_split)
    resolved_group_key = require_group_key(eval_df, preferred=legacy_cfg.intervention.group_key)
    batch_size = int(args.batch_size_override or legacy_cfg.train.batch_size)
    loader = data_engine.build_dataloaders(
        batch_size=batch_size,
        num_workers=int(legacy_cfg.train.num_workers),
        splits=[eval_split],
    )[eval_split]
    group_lookups = _prepare_group_lookup(eval_df, resolved_group_key)

    eraser_collections = {
        "legacy": load_eraser_collection(args.legacy_eraser)[0],
        "k2": load_eraser_collection(args.k2_eraser)[0],
        "k4": load_eraser_collection(args.k4_eraser)[0],
        "k8": load_eraser_collection(args.k8_eraser)[0],
    }

    scenarios = _build_scenarios(legacy_cfg, families, levels, warnings)
    coverage = float(getattr(legacy_cfg.erasure_eval, "coverage", 0.8))

    scenario_rows: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        scenario_rows.extend(
            _run_variant_scenarios(
                variant=variant,
                cfg=cfgs[variant],
                eraser_collection=eraser_collections[variant],
                scenarios=scenarios,
                loader=loader,
                encoder=encoder,
                head=head,
                group_key=resolved_group_key,
                group_lookups=group_lookups,
                coverage=coverage,
                device=device,
                max_batches=args.max_batches,
                warnings=warnings,
            )
        )

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(out_dir / "scenario_rows.csv", index=False)

    summary_rows = _build_summary_rows(scenario_df)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "k_sweep_summary.csv", index=False)
    (out_dir / "k_sweep_summary.json").write_text(metric_report_json(summary_rows), encoding="utf-8")

    winner = _winner_payload(summary_rows)
    (out_dir / "winner.txt").write_text(
        "\n".join(
            [
                f"winner={winner['winner']}",
                f"winner_score={winner['winner_score']}",
                f"go_online_adaptor={str(winner['go_online_adaptor']).lower()}",
                f"reason={winner['reason']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "canonical_contract": "phase2_k_sweep_diag_v1",
        "legacy_config": _normalize_path(args.legacy_config),
        "k2_config": _normalize_path(args.k2_config),
        "k4_config": _normalize_path(args.k4_config),
        "k8_config": _normalize_path(args.k8_config),
        "legacy_eraser": _normalize_path(args.legacy_eraser),
        "k2_eraser": _normalize_path(args.k2_eraser),
        "k4_eraser": _normalize_path(args.k4_eraser),
        "k8_eraser": _normalize_path(args.k8_eraser),
        "resolved_head_path": _normalize_path(legacy_contract.head_path),
        "label_cols": list(label_cols),
        "group_key": resolved_group_key,
        "eval_split": eval_split,
        "families": list(families),
        "levels": [float(level) for level in levels],
        "device": str(device),
        "seed": int(args.seed),
        "max_batches": int(args.max_batches) if args.max_batches is not None else None,
        "batch_size": int(batch_size),
        "encoder_fingerprint": dict(legacy_contract.head_meta["fingerprint"]),
        "head_fingerprint_status": "matched",
        "warnings": sorted(warnings),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
