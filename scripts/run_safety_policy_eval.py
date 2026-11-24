#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import torch

from phystta.config.loader import load_config
from phystta.data import DataEngine
from phystta.shifts import ShiftManager
from phystta.models import EncoderFactory, build_encoder_fingerprint, resolve_head_contract
from phystta.intervention import (
    eraser_fingerprint_mismatches,
    load_eraser_collection,
    resolve_eraser_for_shift,
    TokenEraserWrapper,
    EntropySteeringAdaptor,
    SteeringEraser,
    TaskPreservingEraser,
)
from phystta.metrics import (
    expected_calibration_error,
    auroc_per_class,
    auprc_per_class,
    macro_average,
    selective_risk,
    optimal_thresholds,
    report_at_operating_points,
    operating_point_stats,
    threshold_at_sensitivity,
    threshold_at_specificity,
)
from phystta.metrics import worst_group_auc, underdiagnosis_gap, ece_gap, bin_age
from phystta.safety import (
    ShiftIdentifierV2,
    SafetyPolicy,
    SafetyPolicyConfig,
    LabelShiftRecalibrator,
    misidentification_cost,
    sample_harm_multilabel,
    expected_harm_multilabel,
    policy_regret,
    regret_by_shift,
    regret_decomposition,
)
from phystta.safety.decision import build_decision_policy_from_config, decision_policy_contract_payload
from phystta.safety.recalibration import StratifiedLabelShiftRecalibrator
from phystta.shifts.realism import (
    calibrator_fingerprint_mismatches,
    intensity_grid_from_calibrator,
    load_calibrator,
    scenario_grid_from_calibrator,
)
from phystta.utils import (
    set_global_seed,
    tensor_stats,
    preprocess_signature,
    require_group_key,
    cluster_bootstrap_mean,
    sample_ids_from_batch,
    load_torch_artifact,
    write_reprovenance_bundle,
    sha256_file,
)
from phystta.viz import build_safety_policy_figures
from phystta.utils.report_contract import (
    contract_status,
    require_finite_scalar,
    serialize_operating_point_row,
    vector_finiteness_status,
)


def eval_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage: float = 0.8,
    thresholds: np.ndarray | float | None = None,
) -> dict:
    if thresholds is None:
        raise ValueError("eval_metrics requires explicit thresholds; fixed 0.5 defaults are disallowed.")
    valid = labels >= 0
    probs_flat = probs[valid].reshape(-1)
    labels_flat = labels[valid].reshape(-1)
    if probs.ndim == 2:
        c = probs.shape[1]
        th = np.asarray(thresholds, dtype=float).reshape(-1)
        if th.size == 1:
            th = np.full(c, float(th[0]), dtype=float)
        if th.size != c:
            raise ValueError(f"threshold length mismatch in eval_metrics: got {th.size}, expected {c}")
        preds = (probs >= th.reshape(1, -1))[valid].astype(float)
    else:
        t = float(np.asarray(thresholds, dtype=float).reshape(-1)[0])
        preds = (probs_flat >= t).astype(float)
    errors = preds != labels_flat
    conf = np.maximum(probs_flat, 1 - probs_flat)
    metrics = {
        "ece": expected_calibration_error(probs_flat, labels_flat, n_bins=15),
        "risk_at_coverage": selective_risk(errors.astype(float), conf, coverage=coverage),
    }
    try:
        auroc = auroc_per_class(labels, probs)
        auprc = auprc_per_class(labels, probs)
        metrics["auroc_macro"] = macro_average(auroc)
        metrics["auprc_macro"] = macro_average(auprc)
    except Exception:
        metrics["auroc_macro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
    return metrics


def _experiment_tier(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "experiment", None), "tier", "dev"))


def _resolve_decision_label_source(policy_cfg: Any) -> str:
    source = str(getattr(policy_cfg, "decision_label_source", "predicted"))
    if source not in {"predicted", "oracle_shift_family"}:
        raise RuntimeError(f"Unsupported safety_policy.decision_label_source={source!r}")
    return source


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _estimate_source_label_priors(labels: np.ndarray) -> list[float]:
    priors: list[float] = []
    for c in range(labels.shape[1]):
        yc = labels[:, c]
        valid = yc >= 0
        if valid.sum() == 0:
            priors.append(float("nan"))
        else:
            priors.append(float(np.mean(yc[valid] == 1)))
    return priors


def _serialize_json_value(
    value: Any,
    field_name: str,
    *,
    paper_tier: bool,
    incomplete_fields: list[str],
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _serialize_json_value(
                nested,
                f"{field_name}.{key}",
                paper_tier=paper_tier,
                incomplete_fields=incomplete_fields,
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _serialize_json_value(
                nested,
                f"{field_name}[{idx}]",
                paper_tier=paper_tier,
                incomplete_fields=incomplete_fields,
            )
            for idx, nested in enumerate(value)
        ]
    try:
        return require_finite_scalar(value, field_name)
    except RuntimeError:
        if paper_tier:
            raise
        _append_unique(incomplete_fields, field_name)
        return None


def _sanitize_numeric_vector(
    values: Any,
    field_name: str,
    *,
    expected_length: int | None,
    fallback_value: float | None,
    paper_tier: bool,
    incomplete_fields: list[str],
) -> tuple[list[Any], dict[str, Any]]:
    status = vector_finiteness_status(values, field_name)
    out: list[Any] = []
    fallback_applied = False
    raw_values = [] if values is None else np.asarray(values).reshape(-1).tolist()
    target_length = expected_length if expected_length is not None else len(raw_values)
    if expected_length is not None:
        status["expected_length"] = int(expected_length)
        if status["present"] and status["length"] != int(expected_length):
            if paper_tier:
                raise RuntimeError(
                    f"Field '{field_name}' length mismatch: got {status['length']}, expected {expected_length}."
                )
            _append_unique(incomplete_fields, f"{field_name}_length_mismatch")
            fallback_applied = True
    if not status["present"] or target_length == 0:
        if paper_tier:
            raise RuntimeError(f"Field '{field_name}' is missing.")
        _append_unique(incomplete_fields, f"{field_name}_missing")
        status["fallback_applied"] = bool(fallback_value is not None and target_length > 0)
        if fallback_value is not None:
            status["fallback_value"] = float(fallback_value)
            return [float(fallback_value) for _ in range(target_length)], status
        return [], status
    for idx in range(target_length):
        value = raw_values[idx] if idx < len(raw_values) else None
        if value is None:
            if paper_tier:
                raise RuntimeError(f"Field '{field_name}[{idx}]' is missing.")
            _append_unique(incomplete_fields, f"{field_name}_missing")
            fallback_applied = True
            out.append(float(fallback_value) if fallback_value is not None else None)
            continue
        try:
            out.append(require_finite_scalar(value, f"{field_name}[{idx}]"))
        except RuntimeError:
            if paper_tier:
                raise
            _append_unique(incomplete_fields, f"{field_name}_nonfinite")
            fallback_applied = True
            out.append(float(fallback_value) if fallback_value is not None else None)
    status["fallback_applied"] = fallback_applied
    if fallback_applied and fallback_value is not None:
        status["fallback_value"] = float(fallback_value)
    return out, status


def _mark_nonfinite_fields(
    row: dict[str, Any],
    fields: list[str],
    *,
    prefix: str,
    paper_tier: bool,
    incomplete_fields: list[str],
) -> None:
    for field in fields:
        if field not in row:
            continue
        try:
            require_finite_scalar(row[field], f"{prefix}.{field}")
        except RuntimeError:
            if paper_tier:
                raise
            _append_unique(incomplete_fields, f"{prefix}.{field}")


def _serialize_op_summary_row(
    row: dict[str, Any],
    *,
    required_fields: tuple[str, ...],
    prefix: str,
    paper_tier: bool,
    incomplete_fields: list[str],
) -> dict[str, Any]:
    try:
        return serialize_operating_point_row(row, required_fields=required_fields)
    except RuntimeError:
        if paper_tier:
            raise
        out: dict[str, Any] = {}
        for key, value in row.items():
            field_name = f"{prefix}.{key}"
            try:
                if key in {"label", "target_kind"}:
                    out[key] = str(require_finite_scalar(value, field_name))
                else:
                    out[key] = require_finite_scalar(value, field_name)
            except RuntimeError:
                _append_unique(incomplete_fields, field_name)
                out[key] = None
        for field in required_fields:
            if field not in out:
                _append_unique(incomplete_fields, f"{prefix}.{field}")
                out[field] = None
        return out


def _threshold_vector_at_target(
    labels: np.ndarray,
    probs: np.ndarray,
    mode: str,
    target: float,
) -> np.ndarray:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("threshold selection expects labels/probs with shape [N,C]")
    out = []
    for i in range(y.shape[1]):
        yi = y[:, i]
        pi = p[:, i]
        if mode == "target_sensitivity":
            th = threshold_at_sensitivity(yi, pi, target)
        elif mode == "target_specificity":
            th = threshold_at_specificity(yi, pi, target)
        else:
            raise ValueError(f"unsupported threshold mode: {mode}")
        if not np.isfinite(th):
            th = 0.5
        out.append(float(np.clip(th, 0.0, 1.0)))
    return np.asarray(out, dtype=float)


def _collect_strata_strings(batch, strata_keys, keep_mask=None):
    keys = list(strata_keys or [])
    if not keys:
        n = int(len(batch["image"])) if "image" in batch else 0
        if keep_mask is not None:
            n = int(np.asarray(keep_mask, dtype=bool).sum())
        return np.asarray(["unknown"] * n, dtype=object)

    cols = []
    base_n = None
    for key in keys:
        if key in batch:
            raw = [str(v) for v in batch[key]]
        else:
            if base_n is None:
                if "image" in batch and hasattr(batch["image"], "shape"):
                    base_n = int(batch["image"].shape[0])
                else:
                    base_n = 0
            raw = ["unknown"] * base_n
        if keep_mask is not None:
            raw = [v for v, keep in zip(raw, keep_mask) if keep]
        cols.append((key, raw))

    n = len(cols[0][1]) if cols else 0
    out = []
    for i in range(n):
        out.append("|".join(f"{k}={vals[i]}" for k, vals in cols))
    return np.asarray(out, dtype=object)


def per_label_operating_stats(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    label_cols: list[str],
    cost_fn: float,
    cost_fp: float,
    class_cost_fn: list[float] | None = None,
    class_cost_fp: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    th = np.asarray(thresholds, dtype=float).reshape(-1)
    if p.ndim != 2 or y.ndim != 2 or p.shape != y.shape:
        raise ValueError("per_label_operating_stats expects probs/labels as [N,C] with matching shapes")
    if th.size == 1:
        th = np.full(p.shape[1], float(th[0]), dtype=float)
    if th.size != p.shape[1]:
        raise ValueError(f"threshold length mismatch: got {th.size}, expected {p.shape[1]}")
    if class_cost_fn is None:
        cost_fn_vec = np.full(p.shape[1], float(cost_fn), dtype=float)
    else:
        cost_fn_vec = np.asarray(class_cost_fn, dtype=float).reshape(-1)
        if cost_fn_vec.size == 1:
            cost_fn_vec = np.full(p.shape[1], float(cost_fn_vec[0]), dtype=float)
        if cost_fn_vec.size != p.shape[1]:
            raise ValueError(f"class_cost_fn length mismatch: got {cost_fn_vec.size}, expected {p.shape[1]}")
    if class_cost_fp is None:
        cost_fp_vec = np.full(p.shape[1], float(cost_fp), dtype=float)
    else:
        cost_fp_vec = np.asarray(class_cost_fp, dtype=float).reshape(-1)
        if cost_fp_vec.size == 1:
            cost_fp_vec = np.full(p.shape[1], float(cost_fp_vec[0]), dtype=float)
        if cost_fp_vec.size != p.shape[1]:
            raise ValueError(f"class_cost_fp length mismatch: got {cost_fp_vec.size}, expected {p.shape[1]}")
    out: dict[str, dict[str, float]] = {}
    for c in range(p.shape[1]):
        name = label_cols[c] if c < len(label_cols) else f"class_{c}"
        yc = y[:, c]
        pc = p[:, c]
        valid = yc >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            out[name] = {
                "n_valid": 0.0,
                "threshold": float(th[c]),
                "prevalence": float("nan"),
                "fnr": float("nan"),
                "fpr": float("nan"),
                "harm": float("nan"),
            }
            continue
        yy = yc[valid].astype(int)
        pred = (pc[valid] >= float(th[c])).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum())
        tn = int(((pred == 0) & (yy == 0)).sum())
        fp = int(((pred == 1) & (yy == 0)).sum())
        fn = int(((pred == 0) & (yy == 1)).sum())
        pos = tp + fn
        neg = tn + fp
        fnr = float(fn / pos) if pos > 0 else float("nan")
        fpr = float(fp / neg) if neg > 0 else float("nan")
        harm = float((cost_fn_vec[c] * fn + cost_fp_vec[c] * fp) / max(n_valid, 1))
        out[name] = {
            "n_valid": float(n_valid),
            "threshold": float(th[c]),
            "prevalence": float(np.mean(yy == 1)),
            "fnr": fnr,
            "fpr": fpr,
            "harm": harm,
        }
    return out


def _build_shift_id_physics_scenarios(
    *,
    shift_id: ShiftIdentifierV2,
    feats_ref: np.ndarray,
    probs_ref: np.ndarray,
    calib_loader,
    encoder,
    head,
    shift_manager: ShiftManager,
    shift_types: list[str],
    scenario_map: dict,
    mixed_profiles: list[dict] | None,
    device: str,
    seed: int,
    max_samples: int | None = None,
    log_fn=None,
) -> list[tuple[str, dict[str, float]]]:
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    scenarios: list[tuple[str, dict[str, float]]] = []
    rng = np.random.default_rng(seed)
    n = len(feats_ref)
    _log(f"shift_id_physics_begin n_ref={n} shift_types={list(shift_types)} max_samples={max_samples}")
    if n >= 2:
        idx = rng.permutation(n)
        half = n // 2
        if half > 0 and half < n:
            scores_none = shift_id.compute_scores(
                feats_ref[idx[:half]],
                probs_ref[idx[:half]],
                feats_ref[idx[half:]],
                probs_ref[idx[half:]],
            )
            scenarios.append(("none", scores_none))
            _log("shift_id_physics_none_done")

    with torch.no_grad():
        for shift_type in shift_types:
            if shift_type == "mixed":
                continue
            grid = scenario_map.get(shift_type) or []
            if len(grid) == 0:
                _log(f"shift_id_physics_skip shift_type={shift_type} reason=no_grid")
                continue
            _, intensity = sorted(grid, key=lambda pair: float(pair[0]))[-1]
            _log(f"shift_id_physics_shift_begin shift_type={shift_type} intensity={float(intensity)}")
            feats_t = []
            probs_t = []
            n_collected = 0
            batch_idx = 0
            for batch in calib_loader:
                batch_idx += 1
                x = batch["image"].to(device)
                sample_ids = sample_ids_from_batch(batch)
                x_shift, _ = shift_manager.apply_batch_with_meta(x, shift_type, float(intensity), sample_ids=sample_ids)
                z = encoder.encode(x_shift)
                logits = head(z)
                feats_t.append(z.cpu().numpy())
                probs_batch = torch.sigmoid(logits).cpu().numpy()
                probs_t.append(probs_batch)
                n_collected += int(probs_batch.shape[0])
                if batch_idx == 1 or batch_idx % 200 == 0:
                    _log(f"shift_id_physics_shift_progress shift_type={shift_type} batches={batch_idx} collected={n_collected}")
                if max_samples is not None and n_collected >= int(max_samples):
                    break
            if feats_t:
                f_t = np.concatenate(feats_t, axis=0)
                p_t = np.concatenate(probs_t, axis=0)
                scenarios.append((str(shift_type), shift_id.compute_scores(feats_ref, probs_ref, f_t, p_t)))
                _log(f"shift_id_physics_shift_done shift_type={shift_type} samples={n_collected}")

        if "mixed" in shift_types and mixed_profiles:
            profile = mixed_profiles[0]
            if "intensities" in profile:
                intens = profile["intensities"]
            elif "components" in profile:
                intens = profile["components"]
            else:
                intens = {k: v for k, v in profile.items() if k != "name"}
            _log(f"shift_id_physics_mixed_begin profile_keys={sorted(list(intens.keys()))}")
            feats_t = []
            probs_t = []
            n_collected = 0
            batch_idx = 0
            for batch in calib_loader:
                batch_idx += 1
                x = batch["image"].to(device)
                sample_ids = sample_ids_from_batch(batch)
                x_shift, _ = shift_manager.apply_batch_scenario(x, intens, sample_ids=sample_ids)
                z = encoder.encode(x_shift)
                logits = head(z)
                feats_t.append(z.cpu().numpy())
                probs_batch = torch.sigmoid(logits).cpu().numpy()
                probs_t.append(probs_batch)
                n_collected += int(probs_batch.shape[0])
                if batch_idx == 1 or batch_idx % 200 == 0:
                    _log(f"shift_id_physics_mixed_progress batches={batch_idx} collected={n_collected}")
                if max_samples is not None and n_collected >= int(max_samples):
                    break
            if feats_t:
                f_t = np.concatenate(feats_t, axis=0)
                p_t = np.concatenate(probs_t, axis=0)
                scenarios.append(("mixed", shift_id.compute_scores(feats_ref, probs_ref, f_t, p_t)))
                _log(f"shift_id_physics_mixed_done samples={n_collected}")
    _log(f"shift_id_physics_end scenarios={len(scenarios)}")
    return scenarios


def _shift_label_for_policy(label_space: str, shift_type: str) -> str:
    st = str(shift_type)
    if label_space == "physics":
        return st
    if st == "mixed":
        return "mixed"
    if st in {"none", "prevalence"}:
        return st
    return "covariate"


def _estimate_calib_shift_harm_table(
    *,
    calib_loader,
    encoder,
    head,
    encode_erased,
    reference_shift_family: str,
    shift_manager: ShiftManager,
    shift_types: list[str],
    scenario_map: dict,
    mixed_profiles: list[dict] | None,
    clinical_thresholds: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    class_cost_fn: list[float] | None,
    class_cost_fp: list[float] | None,
    critical_idx: list[int] | None,
    device: str,
    label_space: str,
) -> dict[str, dict[str, float]]:
    """Estimate expected harm per predicted shift label using calib data only."""
    accum: dict[str, dict[str, list[float]]] = {}
    debug_rows: list[dict[str, object]] = []

    log_fn = getattr(_estimate_calib_shift_harm_table, "logger", None)
    if log_fn is None:
        def log_fn(msg: str) -> None:
            print(msg, flush=True)

    progress_every = int(getattr(_estimate_calib_shift_harm_table, "progress_every", 200))
    checkpoint_every = int(getattr(_estimate_calib_shift_harm_table, "checkpoint_every", 1))
    checkpoint_path = getattr(_estimate_calib_shift_harm_table, "checkpoint_path", None)
    resume_enabled = bool(getattr(_estimate_calib_shift_harm_table, "resume_enabled", False))

    completed: set[tuple[str, str]] = set()

    if resume_enabled and checkpoint_path is not None:
        try:
            ckpt_obj = Path(checkpoint_path)
            if ckpt_obj.exists():
                payload = json.loads(ckpt_obj.read_text())
                debug_rows = list(payload.get("debug_rows", []))
                for item in payload.get("completed", []):
                    completed.add((str(item["shift_type"]), str(item["scenario_name"])))
                log_fn(
                    f"shift_harm_function_resume checkpoint={ckpt_obj} "
                    f"completed={len(completed)} debug_rows={len(debug_rows)}"
                )
        except Exception as e:
            log_fn(f"shift_harm_function_resume_warn error={e}")

    for row in debug_rows:
        label = str(row["label_key"])
        cell = accum.setdefault(label, {"no_eraser": [], "eraser": []})
        cell["no_eraser"].append(float(row["h_no"]))
        cell["eraser"].append(float(row["h_er"]))

    _estimate_calib_shift_harm_table.last_debug_rows = debug_rows

    def _save_ckpt(current: dict | None = None) -> None:
        if checkpoint_path is None:
            return
        payload = {
            "status": "in_progress",
            "completed": [
                {"shift_type": st, "scenario_name": sc}
                for st, sc in sorted(completed)
            ],
            "current": current,
            "debug_rows": debug_rows,
            "saved_at": time.time(),
        }
        Path(checkpoint_path).write_text(json.dumps(payload, indent=2))

    def _add(label: str, h_no: float, h_er: float) -> None:
        cell = accum.setdefault(label, {"no_eraser": [], "eraser": []})
        cell["no_eraser"].append(float(h_no))
        cell["eraser"].append(float(h_er))

    def _expected(probs: np.ndarray, labels: np.ndarray) -> float:
        return expected_harm_multilabel(
            probs,
            labels,
            thresholds=clinical_thresholds,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            critical_idx=critical_idx if critical_idx else None,
            class_cost_fn=class_cost_fn,
            class_cost_fp=class_cost_fp,
        )

    table_start = time.time()

    with torch.no_grad():
        scenario_key = ("none", "none")
        if scenario_key in completed:
            log_fn("shift_harm_scenario_skip shift=none scenario=none reason=resume_complete")
        else:
            probs_none = []
            probs_none_er = []
            labels_none = []
            batch_idx = 0
            scenario_start = time.time()
            log_fn("shift_harm_scenario_begin shift=none scenario=none")
            for batch in calib_loader:
                batch_idx += 1
                x = batch["image"].to(device)
                y = batch["labels"].to(device)
                logits_no = head(encoder.encode(x))
                logits_er = head(encode_erased(x, reference_shift_family))
                probs_none.append(torch.sigmoid(logits_no).cpu().numpy())
                probs_none_er.append(torch.sigmoid(logits_er).cpu().numpy())
                labels_none.append(y.cpu().numpy())
                if batch_idx == 1 or batch_idx % progress_every == 0:
                    log_fn(
                        f"shift_harm_progress shift=none scenario=none "
                        f"batches={batch_idx} rows={len(debug_rows)} "
                        f"elapsed_sec={time.time() - scenario_start:.2f}"
                    )
            if probs_none:
                p_no = np.concatenate(probs_none, axis=0)
                p_er = np.concatenate(probs_none_er, axis=0)
                y_np = np.concatenate(labels_none, axis=0)
                h_no = _expected(p_no, y_np)
                h_er = _expected(p_er, y_np)
                row = {
                    "shift_type": "none",
                    "scenario_name": "none",
                    "label_key": "none",
                    "n_samples": int(y_np.shape[0]),
                    "h_no": float(h_no),
                    "h_er": float(h_er),
                    "delta_harm": float(h_er - h_no),
                    "mean_abs_prob_delta": float(np.mean(np.abs(p_er - p_no))),
                }
                debug_rows.append(row)
                _estimate_calib_shift_harm_table.last_debug_rows = debug_rows
                _add("none", h_no, h_er)
                completed.add(scenario_key)
                log_fn(
                    f"shift_harm_scenario_done shift=none scenario=none "
                    f"rows={len(debug_rows)} sec={time.time() - scenario_start:.2f}"
                )
                if len(completed) % checkpoint_every == 0:
                    _save_ckpt(current=None)
                    log_fn(
                        f"shift_harm_checkpoint_saved path={checkpoint_path} "
                        f"completed={len(completed)} rows={len(debug_rows)}"
                    )

        for shift_type in shift_types:
            if shift_type == "mixed":
                profiles = mixed_profiles or []
                scenarios = [(profile.get("name", f"mixed_{idx}"), profile) for idx, profile in enumerate(profiles)]
            else:
                grid = scenario_map.get(shift_type) or []
                scenarios = [(float(intensity), None) for _level, intensity in grid]
            if not scenarios:
                continue

            for scenario_name, profile in scenarios:
                scenario_key = (str(shift_type), str(scenario_name))
                if scenario_key in completed:
                    log_fn(
                        f"shift_harm_scenario_skip shift={shift_type} "
                        f"scenario={scenario_name} reason=resume_complete"
                    )
                    continue

                probs_no = []
                probs_er = []
                labels_np = []
                batch_idx = 0
                scenario_start = time.time()
                log_fn(f"shift_harm_scenario_begin shift={shift_type} scenario={scenario_name}")

                for batch in calib_loader:
                    batch_idx += 1
                    x = batch["image"].to(device)
                    y = batch["labels"].to(device)
                    sample_ids = sample_ids_from_batch(batch)
                    if shift_type == "mixed":
                        if profile is None:
                            continue
                        intens = profile.get("intensities") or profile.get("components") or {
                            k: v for k, v in profile.items() if k != "name"
                        }
                        x_shift, _ = shift_manager.apply_batch_scenario(x, intens, sample_ids=sample_ids)
                    else:
                        x_shift, _ = shift_manager.apply_batch_with_meta(
                            x, shift_type, float(scenario_name), sample_ids=sample_ids
                        )
                    logits_no = head(encoder.encode(x_shift))
                    logits_er = head(encode_erased(x_shift, shift_type))
                    probs_no.append(torch.sigmoid(logits_no).cpu().numpy())
                    probs_er.append(torch.sigmoid(logits_er).cpu().numpy())
                    labels_np.append(y.cpu().numpy())

                    if batch_idx == 1 or batch_idx % progress_every == 0:
                        log_fn(
                            f"shift_harm_progress shift={shift_type} scenario={scenario_name} "
                            f"batches={batch_idx} rows={len(debug_rows)} "
                            f"elapsed_sec={time.time() - scenario_start:.2f}"
                        )
                        _save_ckpt(
                            current={
                                "shift_type": str(shift_type),
                                "scenario_name": str(scenario_name),
                                "batch_idx": int(batch_idx),
                            }
                        )

                if not probs_no:
                    continue

                p_no = np.concatenate(probs_no, axis=0)
                p_er = np.concatenate(probs_er, axis=0)
                y_np = np.concatenate(labels_np, axis=0)
                label_key = _shift_label_for_policy(label_space, shift_type)
                h_no = _expected(p_no, y_np)
                h_er = _expected(p_er, y_np)

                row = {
                    "shift_type": str(shift_type),
                    "scenario_name": str(scenario_name),
                    "label_key": str(label_key),
                    "n_samples": int(y_np.shape[0]),
                    "h_no": float(h_no),
                    "h_er": float(h_er),
                    "delta_harm": float(h_er - h_no),
                    "mean_abs_prob_delta": float(np.mean(np.abs(p_er - p_no))),
                }
                debug_rows.append(row)
                _estimate_calib_shift_harm_table.last_debug_rows = debug_rows
                _add(label_key, h_no, h_er)

                completed.add(scenario_key)
                log_fn(
                    f"shift_harm_scenario_done shift={shift_type} scenario={scenario_name} "
                    f"rows={len(debug_rows)} sec={time.time() - scenario_start:.2f}"
                )
                if len(completed) % checkpoint_every == 0:
                    _save_ckpt(current=None)
                    log_fn(
                        f"shift_harm_checkpoint_saved path={checkpoint_path} "
                        f"completed={len(completed)} rows={len(debug_rows)}"
                    )

    out: dict[str, dict[str, float]] = {}
    for label, values in accum.items():
        if not values["no_eraser"] or not values["eraser"]:
            continue
        out[label] = {
            "no_eraser": float(np.mean(values["no_eraser"])),
            "eraser": float(np.mean(values["eraser"])),
        }
        out[label]["recalibrate"] = out[label]["no_eraser"]
    if label_space != "physics" and "prevalence" not in out and "none" in out:
        out["prevalence"] = dict(out["none"])

    if checkpoint_path is not None:
        Path(checkpoint_path).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "completed": [
                        {"shift_type": st, "scenario_name": sc}
                        for st, sc in sorted(completed)
                    ],
                    "debug_rows": debug_rows,
                    "saved_at": time.time(),
                },
                indent=2,
            )
        )

    log_fn(f"shift_harm_table_function_done rows={len(debug_rows)} sec={time.time() - table_start:.2f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety policy eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--eraser", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shift_type", default=None)
    parser.add_argument("--intensity_grid", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder_profile", default=None)
    parser.add_argument("--save_shift_id", default=None)
    parser.add_argument("--save_policy", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--max_ref", type=int, default=None, help="Optional max calibration/reference samples (group-aware).")
    parser.add_argument("--max_calib", type=int, default=None, help="Alias for --max_ref.")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_num_workers", type=int, default=None)
    parser.add_argument("--ref_progress_every", type=int, default=50)
    parser.add_argument("--target_sensitivities", default="0.90")
    parser.add_argument("--target_specificities", default="0.90")
    parser.add_argument(
        "--clinical_threshold_source",
        default="cost_optimal",
        choices=["cost_optimal", "target_sensitivity", "target_specificity"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force_rebuild_reference", action="store_true")
    parser.add_argument("--shift_harm_progress_every", type=int, default=200)
    parser.add_argument("--shift_harm_checkpoint_every", type=int, default=1)
    args = parser.parse_args()
    if args.max_calib is not None:
        args.max_ref = args.max_calib if args.max_ref is None else min(args.max_ref, args.max_calib)

    cfg = load_config(args.config)
    experiment_tier = _experiment_tier(cfg)
    paper_tier = experiment_tier == "paper"
    policy_contract_errors: list[str] = []
    policy_contract_incomplete_fields: list[str] = []
    seed = args.seed if args.seed is not None else cfg.runtime.seed
    device = args.device or cfg.runtime.device
    set_global_seed(seed, deterministic=cfg.runtime.deterministic, benchmark=cfg.runtime.benchmark)
    target_sensitivities = [float(v.strip()) for v in args.target_sensitivities.split(",") if v.strip()] if args.target_sensitivities else []
    target_specificities = [float(v.strip()) for v in args.target_specificities.split(",") if v.strip()] if args.target_specificities else []

    data_engine = DataEngine(cfg.data).attach_runtime_contract(cfg)
    data_engine.build_index()
    data_engine.create_splits()
    label_cols = data_engine.label_cols
    critical_labels = cfg.safety_policy.critical_labels if hasattr(cfg.safety_policy, "critical_labels") else None
    critical_idx = []
    if critical_labels:
        for name in critical_labels:
            if name in label_cols:
                critical_idx.append(label_cols.index(name))

    calib_split = cfg.data.calib_split
    eval_split = cfg.data.eval_split
    calib_df = data_engine.get_split_df(calib_split)
    eval_df = data_engine.get_split_df(eval_split)
    require_group_key(calib_df, preferred=cfg.intervention.group_key)
    eval_group_key = require_group_key(eval_df, preferred=cfg.intervention.group_key)

    eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else int(cfg.train.batch_size)
    eval_num_workers = int(args.eval_num_workers) if args.eval_num_workers is not None else int(cfg.train.num_workers)
    loaders = data_engine.build_dataloaders(
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        splits=[calib_split, eval_split],
    )
    calib_loader = loaders[calib_split]
    eval_loader = loaders[eval_split]

    shift_types = [args.shift_type] if args.shift_type else cfg.erasure_eval.shift_types
    enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile=args.encoder_profile)
    expected_calib_fp = build_encoder_fingerprint(enc_cfg)
    expected_calib_fp["preprocess_signature"] = preprocess_signature(cfg)
    if cfg.intervention.use_realism:
        if args.intensity_grid:
            raise RuntimeError("intensity_grid overridden by realism calibrator; remove --intensity_grid")
        if cfg.intervention.realism_path is None:
            raise RuntimeError("use_realism=true but realism_path not set")
        calib = load_calibrator(str(cfg.intervention.realism_path))
        mismatches = calibrator_fingerprint_mismatches(calib, expected_calib_fp)
        if mismatches:
            msg = (
                "Realism calibrator fingerprint mismatch against active encoder/preprocess: "
                + "; ".join(mismatches)
            )
            if paper_tier:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        intensity_map = intensity_grid_from_calibrator(calib, shift_types, graded_only=True)
        scenario_map = scenario_grid_from_calibrator(calib, shift_types, graded_only=True)
    else:
        grid = [float(v) for v in args.intensity_grid.split(",")] if args.intensity_grid else cfg.erasure_eval.intensity_grid
        intensity_map = {st: grid for st in shift_types}
        scenario_map = {st: [(float("nan"), float(i)) for i in grid] for st in shift_types}

    shift_manager = ShiftManager(seed=cfg.shifts.seed)
    encoder = EncoderFactory.from_config(cfg, profile=args.encoder_profile)
    if hasattr(encoder, "model"):
        encoder.model.to(device)

    head = None
    strict_meta = bool(cfg.runtime.freeze_artifacts or paper_tier)
    head_contract = resolve_head_contract(
        cfg,
        label_cols=data_engine.label_cols,
        encoder_profile=args.encoder_profile,
        require_head_path=True,
        strict_meta=strict_meta,
        load_artifact=True,
    )
    head = head_contract.head_artifact
    if hasattr(head, "to"):
        head.to(device)
    if hasattr(head, "eval"):
        head.eval()

    eraser_collection, eraser_bundle_meta = load_eraser_collection(args.eraser)
    for shift_family, eraser_obj in eraser_collection.items():
        eraser_mismatches = eraser_fingerprint_mismatches(eraser_obj, expected_calib_fp)
        if eraser_mismatches:
            label = "default" if shift_family == "__default__" else str(shift_family)
            msg = (
                f"Eraser fingerprint mismatch for shift_family='{label}' against active encoder/preprocess: "
                + "; ".join(eraser_mismatches)
            )
            if paper_tier:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")
        if hasattr(eraser_obj, "to"):
            eraser_obj = eraser_obj.to(device)
            eraser_collection[shift_family] = eraser_obj
        if hasattr(eraser_obj, "eval"):
            eraser_obj.eval()
    token_wrappers: dict[str, TokenEraserWrapper] = {}
    token_mode_enabled = bool(getattr(cfg, "eraser", None) and cfg.eraser.token_mode != "none")
    eraser_reference_shift_family = next((str(st) for st in shift_types if str(st) != "mixed"), "none")

    def _get_eraser_for_shift(shift_family: str):
        key = eraser_reference_shift_family if str(shift_family) in {"", "none"} and eraser_bundle_meta.get("kind") == "bank" else shift_family
        return resolve_eraser_for_shift(eraser_collection, str(key))

    entropy_adaptor = None
    if getattr(cfg, "eraser", None) and cfg.eraser.online_enabled:
        if eraser_bundle_meta.get("kind") == "bank":
            raise RuntimeError("online_enabled is not compatible with eraser bank manifests in safety_policy_eval.")
        if cfg.eraser.token_mode != "none":
            raise RuntimeError("online_enabled not compatible with token_mode in safety_policy_eval.")
        eraser = _get_eraser_for_shift(eraser_reference_shift_family)
        if cfg.eraser.online_mode != "entropy":
            raise RuntimeError("safety_policy_eval supports online_mode=entropy only.")
        base = eraser.eraser if isinstance(eraser, TaskPreservingEraser) else eraser
        if not isinstance(base, SteeringEraser):
            raise RuntimeError("online_mode=entropy requires SteeringEraser.")
        entropy_adaptor = EntropySteeringAdaptor(
            base,
            head,
            steps=cfg.eraser.online_entropy_steps,
            lr=cfg.eraser.online_entropy_lr,
            guard_eps=cfg.eraser.online_guard_eps,
        ).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    def log(msg: str):
        print(msg)
        log_path.open("a").write(msg + "\n")

    def _pick_reference_groups(df, group_key: str, max_samples: int | None) -> set[str] | None:
        if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
            return None
        selected: set[str] = set()
        seen = 0
        for gid in df[group_key].astype(str).tolist():
            selected.add(str(gid))
            seen += 1
            if seen >= max_samples:
                break
        return selected

    ref_groups = _pick_reference_groups(calib_df, eval_group_key, args.max_ref)

    def encode_erased(x: torch.Tensor, shift_family: str | None = None) -> torch.Tensor:
        if entropy_adaptor is not None:
            z = encoder.encode(x)
            return entropy_adaptor(z)
        eraser = _get_eraser_for_shift(shift_family or eraser_reference_shift_family)
        if token_mode_enabled:
            key = str(shift_family or eraser_reference_shift_family)
            if key not in token_wrappers:
                token_wrappers[key] = TokenEraserWrapper(
                    encoder,
                    eraser,
                    mode=cfg.eraser.token_mode,
                    token_layers=cfg.eraser.token_layers,
                ).to(device)
            return token_wrappers[key](x)
        return eraser(encoder.encode(x))

    log(f"preprocess_signature={preprocess_signature(cfg)}")
    try:
        batch0 = next(iter(eval_loader))
        raw_stats = tensor_stats(batch0["image"])
        adapted = encoder.preprocess(batch0["image"].to(device))
        adapted_stats = tensor_stats(adapted.detach().cpu())
        log(f"raw_stats={raw_stats}")
        log(f"adapter_stats={adapted_stats}")
    except Exception as e:
        log(f"stats_error={e}")

    # calibration features/probs/labels
    feats_ref = []
    probs_ref = []
    labels_ref = []
    group_ref = []
    strata_ref = []
    shift_group_key = str(getattr(cfg.shift_id, "group_key", "view_family"))
    ckpt_dir = out_dir / "_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ref_ckpt = ckpt_dir / "reference_tensors.npz"
    shift_harm_ckpt = ckpt_dir / "shift_harm_table.json"
    ref_progress_every = int(args.ref_progress_every)
    print(
        f"safety_runtime eval_batch_size={eval_batch_size} eval_num_workers={eval_num_workers} "
        f"ref_progress_every={ref_progress_every}",
        flush=True,
    )
    log(f"safety_runtime eval_batch_size={eval_batch_size} eval_num_workers={eval_num_workers} ref_progress_every={ref_progress_every}")

    resume_enabled = bool(args.resume)

    if resume_enabled and ref_ckpt.exists() and not args.force_rebuild_reference:
        arr = np.load(ref_ckpt, allow_pickle=True)
        feats_ref = arr["feats_ref"]
        probs_ref = arr["probs_ref"]
        labels_ref = arr["labels_ref"]

        strata_ref = arr["strata_ref"]
        if len(strata_ref) == 0:
            strata_ref = None

        group_ref = arr["group_ref"]
        if len(group_ref) == 0:
            group_ref = None

        source_label_priors_raw = _estimate_source_label_priors(labels_ref)

        state_path = ckpt_dir / "reference_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except Exception:
                state = {}
        else:
            state = {}

        log(
            f"reference_collection_resume samples={labels_ref.shape[0]} "
            f"checkpoint={ref_ckpt} state={state.get('status', 'unknown')}"
        )
    else:
        log("reference_collection_begin")
        ref_batch_idx = 0
        with torch.no_grad():
            for batch in calib_loader:
                ref_batch_idx += 1
                x = batch["image"].to(device)
                y = batch["labels"].to(device)
                z = encoder.encode(x)
                logits = head(z)
                if eval_group_key in batch:
                    sample_groups = [str(v) for v in batch[eval_group_key]]
                elif "global_patient_id" in batch:
                    sample_groups = [str(v) for v in batch["global_patient_id"]]
                else:
                    sample_groups = [str(v) for v in batch["patient_id"]]
                keep_mask = None
                if ref_groups is not None:
                    keep_mask = np.asarray([g in ref_groups for g in sample_groups], dtype=bool)
                    if not bool(keep_mask.any()):
                        continue
                    keep_idx = torch.as_tensor(keep_mask, dtype=torch.bool, device=z.device)
                    z = z[keep_idx]
                    logits = logits[keep_idx]
                    y = y[keep_idx]
                    sample_groups = [g for g, keep in zip(sample_groups, keep_mask) if keep]
                feats_ref.append(z.cpu().numpy())
                probs_ref.append(torch.sigmoid(logits).cpu().numpy())
                labels_ref.append(y.cpu().numpy())
                strata_ref.extend(
                    _collect_strata_strings(
                        batch,
                        getattr(cfg.safety_policy, "strata_keys", ["view_family", "procedure_context", "orientation_context"]),
                        keep_mask=keep_mask,
                    ).tolist()
                )

                group_vals = None
                if shift_group_key in batch:
                    raw = [str(v) for v in batch[shift_group_key]]
                    group_vals = [v for v, keep in zip(raw, keep_mask)] if keep_mask is not None else raw
                else:
                    for key in ("view_family", "view", "procedure_context", "orientation_context"):
                        if key in batch:
                            raw = [str(v) for v in batch[key]]
                            group_vals = [v for v, keep in zip(raw, keep_mask)] if keep_mask is not None else raw
                            break
                if group_vals is not None:
                    group_ref.extend(group_vals)
                if ref_batch_idx == 1 or ref_batch_idx % ref_progress_every == 0:
                    log(
                        f"reference_progress batches={ref_batch_idx} "
                        f"kept_samples={sum(arr.shape[0] for arr in labels_ref)} groups={len(group_ref)}"
                    )
                if ref_groups is not None and args.max_ref is not None and len(group_ref) >= int(args.max_ref):
                    log(f"reference_budget_early_stop target={int(args.max_ref)} reached_groups={len(ref_groups)} kept_so_far={len(group_ref)}")
                    break
        feats_ref = np.concatenate(feats_ref, axis=0)
        probs_ref = np.concatenate(probs_ref, axis=0)
        labels_ref = np.concatenate(labels_ref, axis=0)
        strata_ref = np.asarray(strata_ref, dtype=object) if len(strata_ref) > 0 else None
        source_label_priors_raw = _estimate_source_label_priors(labels_ref)
        group_ref = np.array(group_ref) if len(group_ref) > 0 else None
        if ref_groups is not None:
            log(f"reference_budget groups={len(ref_groups)} kept_samples={len(labels_ref)}")
        np.savez_compressed(
            ref_ckpt,
            feats_ref=feats_ref,
            probs_ref=probs_ref,
            labels_ref=labels_ref,
            strata_ref=np.asarray(strata_ref, dtype=object) if strata_ref is not None else np.asarray([], dtype=object),
            group_ref=np.asarray(group_ref, dtype=object) if group_ref is not None else np.asarray([], dtype=object),
        )
        (ckpt_dir / "reference_state.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "samples": int(labels_ref.shape[0]),
                    "ref_groups_present": bool(ref_groups is not None),
                },
                indent=2,
            )
        )
        log(f"reference_collection_done samples={labels_ref.shape[0]} checkpoint={ref_ckpt}")

    log("trace_before_shift_id_init")
    shift_id = ShiftIdentifierV2(
        score_family=cfg.shift_id.score_family,
        cost_matrix=getattr(cfg.shift_id, "cost_matrix", None),
        calibration=getattr(cfg.shift_id, "calibration", "none"),
    )
    log("trace_after_shift_id_init")
    shift_id_label_space = str(getattr(cfg.shift_id, "label_space", "nature"))
    log(f"trace_shift_id_label_space value={shift_id_label_space}")
    shift_id_report_scenarios: list[tuple[str, dict[str, float]]] | None = None
    shift_id_scenarios_ckpt = ckpt_dir / "shift_id_physics_scenarios.json"
    if shift_id_label_space == "physics":
        if bool(args.resume) and shift_id_scenarios_ckpt.exists():
            raw = json.loads(shift_id_scenarios_ckpt.read_text())
            shift_id_report_scenarios = [
                (str(item["label"]), {str(k): float(v) for k, v in dict(item["scores"]).items()})
                for item in raw
            ]
            log(f"shift_id_physics_resume scenarios={len(shift_id_report_scenarios)} checkpoint={shift_id_scenarios_ckpt}")
        else:
            log("trace_before_build_shift_id_physics_scenarios")
            shift_id_report_scenarios = _build_shift_id_physics_scenarios(
                shift_id=shift_id,
                feats_ref=feats_ref,
                probs_ref=probs_ref,
                calib_loader=calib_loader,
                encoder=encoder,
                head=head,
                shift_manager=shift_manager,
                shift_types=list(shift_types),
                scenario_map=scenario_map,
                mixed_profiles=list(cfg.intervention.mixed_profiles or []),
                device=device,
                seed=seed,
                max_samples=args.max_ref,
                log_fn=log,
            )
            shift_id_scenarios_ckpt.write_text(
                json.dumps(
                    [{"label": label, "scores": {str(k): float(v) for k, v in scores.items()}} for label, scores in shift_id_report_scenarios],
                    indent=2,
                )
            )
            log(f"trace_after_build_shift_id_physics_scenarios n={0 if shift_id_report_scenarios is None else len(shift_id_report_scenarios)} checkpoint={shift_id_scenarios_ckpt}")
        log("trace_before_shift_id_calibrate_from_scores")
        shift_id.calibrate_from_scores(shift_id_report_scenarios, seed=seed)
        log("trace_after_shift_id_calibrate_from_scores")
    else:
        shift_id.calibrate(
            feats_ref,
            probs_ref,
            labels_ref,
            group_ref,
            batch_sizes=getattr(cfg.shift_id, "batch_sizes", None),
            n_splits=getattr(cfg.shift_id, "n_splits", 1),
        )
    shift_id_classes = [str(v) for v in list(getattr(shift_id, "classes_", []))]
    covariate_label_names = [str(s) for s in shift_types if str(s) not in {"none", "mixed", "prevalence"}]

    log("trace_before_policy_cfg")
    policy_cfg = SafetyPolicyConfig(
        apply_on_covariate=cfg.safety_policy.apply_on_covariate,
        abstain_on_mixed=cfg.safety_policy.abstain_on_mixed,
        abstain_on_unknown=cfg.safety_policy.abstain_on_unknown,
        cost_fn=cfg.safety_policy.cost_fn if hasattr(cfg.safety_policy, "cost_fn") else 10.0,
        cost_fp=cfg.safety_policy.cost_fp if hasattr(cfg.safety_policy, "cost_fp") else 1.0,
        class_cost_fn=getattr(cfg.safety_policy, "class_cost_fn", None),
        class_cost_fp=getattr(cfg.safety_policy, "class_cost_fp", None),
        coverage_budget=cfg.safety_policy.coverage_budget if hasattr(cfg.safety_policy, "coverage_budget") else cfg.erasure_eval.coverage,
        critical_labels=critical_labels,
        conformal=getattr(cfg.safety_policy, "conformal", False),
        risk_target=getattr(cfg.safety_policy, "risk_target", 0.05),
        conformal_alpha=getattr(cfg.safety_policy, "conformal_alpha", 0.05),
        stratify_technical=getattr(cfg.safety_policy, "stratify_technical", False),
        strata_keys=getattr(cfg.safety_policy, "strata_keys", ["view_family", "procedure_context", "orientation_context"]),
        shared_threshold=getattr(cfg.safety_policy, "shared_threshold", True),
        use_decision_policy=getattr(cfg.safety_policy, "use_decision_policy", False),
        uncertainty_abstain_threshold=getattr(cfg.safety_policy, "uncertainty_abstain_threshold", 0.35),
        mixed_abstain_threshold=getattr(cfg.safety_policy, "mixed_abstain_threshold", 0.40),
        prevalence_action=getattr(cfg.safety_policy, "prevalence_action", "recalibrate"),
        recalibration_mode=getattr(cfg.safety_policy, "recalibration_mode", "global"),
        stratified_recalibration_min_samples=getattr(cfg.safety_policy, "stratified_recalibration_min_samples", 64),
        abstain_cost=getattr(cfg.safety_policy, "abstain_cost", 1.0),
        policy_variant=getattr(cfg.safety_policy, "policy_variant", "legacy_v1"),
        uncertainty_lambda=getattr(cfg.safety_policy, "uncertainty_lambda", 0.0),
        route_gamma=getattr(cfg.safety_policy, "route_gamma", 0.0),
        decision_label_source=getattr(cfg.safety_policy, "decision_label_source", "predicted"),
    )
    log("trace_after_policy_cfg")
    log("trace_before_resolve_decision_label_source")
    decision_label_source = _resolve_decision_label_source(policy_cfg)
    log(f"trace_after_resolve_decision_label_source value={decision_label_source}")
    decision_label_diagnostic_only = bool(decision_label_source != "predicted")
    log(f"trace_after_decision_label_diagnostic_only value={decision_label_diagnostic_only}")
    log("trace_before_policy_init")
    policy = SafetyPolicy(policy_cfg)
    log("trace_after_policy_init")
    covariate_labels = None
    if shift_id_label_space == "physics":
        covariate_labels = covariate_label_names
    log(f"trace_before_build_decision_policy label_space={shift_id_label_space} covariate_labels={covariate_labels}")
    decision_policy, decision_policy_variant = build_decision_policy_from_config(
        policy_cfg,
        covariate_labels=covariate_labels,
    )
    log(f"trace_after_build_decision_policy variant={decision_policy_variant} decision_policy_is_none={decision_policy is None}")
    decision_policy_contract = decision_policy_contract_payload(decision_policy, decision_policy_variant)
    log("trace_after_decision_policy_contract")
    prevalence_recalibrator = None
    decision_policy_uses_recalibration = bool(
        decision_policy_variant == "expected_risk_v2"
        or (
            decision_policy is not None
            and getattr(decision_policy, "prevalence_action", None) == "recalibrate"
        )
    )
    if decision_policy_uses_recalibration:
        log(
            f"prevalence_recalibration_gate_begin "
            f"mode={getattr(policy_cfg, 'recalibration_mode', 'global')} "
            f"stratify_technical={bool(getattr(policy_cfg, 'stratify_technical', False))} "
            f"strata_ref_present={strata_ref is not None} "
            f"labels_ref_shape={tuple(labels_ref.shape)}"
        )
        source_status_precheck = vector_finiteness_status(source_label_priors_raw, "source_label_priors")
        log(
            f"prevalence_recalibration_source_status "
            f"present={source_status_precheck['present']} "
            f"n_nonfinite={source_status_precheck['n_nonfinite']}"
        )
        if not source_status_precheck["present"] or source_status_precheck["n_nonfinite"] > 0:
            problem = (
                "source_label_priors_missing"
                if not source_status_precheck["present"]
                else "source_label_priors_nonfinite"
            )
            if paper_tier:
                raise RuntimeError(
                    "Safety policy prevalence recalibration requires finite source_label_priors for paper tier; "
                    f"got {problem}."
                )
            _append_unique(policy_contract_incomplete_fields, problem)
        if (
            getattr(policy_cfg, "recalibration_mode", "global") == "stratified"
            and bool(getattr(policy_cfg, "stratify_technical", False))
            and strata_ref is not None
        ):
            log(
                f"prevalence_recalibration_fit_begin kind=stratified "
                f"min_samples={int(getattr(policy_cfg, 'stratified_recalibration_min_samples', 64))} "
                f"strata_n={len(strata_ref)}"
            )
            prevalence_recalibrator = StratifiedLabelShiftRecalibrator.fit_from_labels_and_strata(
                labels_ref,
                strata_ref,
                min_samples=int(getattr(policy_cfg, "stratified_recalibration_min_samples", 64)),
            )
            log("prevalence_recalibration_fit_done kind=stratified")
        else:
            log("prevalence_recalibration_fit_begin kind=global")
            prevalence_recalibrator = LabelShiftRecalibrator.fit_from_labels(labels_ref)
            log("prevalence_recalibration_fit_done kind=global")

    has_covariate_channel = (
        any(lbl in shift_id_classes for lbl in covariate_label_names)
        if shift_id_label_space == "physics"
        else ("covariate" in shift_id_classes)
    )
    shift_id_runtime_meta = {
        "label_space": shift_id_label_space,
        "group_key": shift_group_key,
        "classes": shift_id_classes,
        "covariate_label_names": covariate_label_names,
        "has_covariate_channel": bool(has_covariate_channel),
        "use_decision_policy": bool(policy_cfg.use_decision_policy),
        "apply_on_covariate": bool(policy_cfg.apply_on_covariate),
        "prevalence_action": str(policy_cfg.prevalence_action),
        "decision_policy_variant": str(decision_policy_contract["decision_policy_variant"]),
        "decision_policy_active": bool(decision_policy_contract["decision_policy_active"]),
        "decision_path_source": str(decision_policy_contract["decision_path_source"]),
        "decision_label_source": decision_label_source,
        "decision_label_diagnostic_only": bool(decision_label_diagnostic_only),
    }
    (out_dir / "shift_id_runtime_meta.json").write_text(json.dumps(shift_id_runtime_meta, indent=2))

    if policy_cfg.apply_on_covariate and not policy_cfg.use_decision_policy:
        raise RuntimeError(
            "safety_policy.apply_on_covariate=true but use_decision_policy=false; refusing legacy_policy fallback"
        )
    if (
        policy_cfg.use_decision_policy
        and policy_cfg.apply_on_covariate
        and not has_covariate_channel
        and decision_policy_variant == "expected_risk_v2"
    ):
        raise RuntimeError(
            f"Decision policy enabled but no covariate channel available for shift_id_label_space={shift_id_label_space}; classes={shift_id_classes}"
        )
    # calibrate thresholds for coverage on calib split
    log("trace_before_probs_er_ckpt")
    probs_er_ckpt = ckpt_dir / "probs_er_ref.npz"
    log(f"trace_after_probs_er_ckpt path={probs_er_ckpt} exists={probs_er_ckpt.exists()}")
    probs_er_state = ckpt_dir / "probs_er_state.json"
    log(f"trace_after_probs_er_state path={probs_er_state} exists={probs_er_state.exists()}")

    probs_no = probs_ref
    log(f"trace_before_erased_branch resume={bool(args.resume)} probs_no_n={len(probs_no)}")
    if bool(args.resume) and probs_er_ckpt.exists():
        arr = np.load(probs_er_ckpt, allow_pickle=True)
        probs_er = arr["probs_er"]
        state = {}
        if probs_er_state.exists():
            try:
                state = json.loads(probs_er_state.read_text())
            except Exception:
                state = {}
        log(
            f"erased_calib_resume samples={probs_er.shape[0]} "
            f"checkpoint={probs_er_ckpt} state={state.get('status', 'unknown')}"
        )
    else:
        log(f"erased_calib_begin target_samples={len(probs_no)}")
        with torch.no_grad():
            probs_er = []
            erased_batch_idx = 0
            collected_er = 0
            for batch in calib_loader:
                erased_batch_idx += 1
                x = batch["image"].to(device)
                if eval_group_key in batch:
                    sample_groups = [str(v) for v in batch[eval_group_key]]
                elif "global_patient_id" in batch:
                    sample_groups = [str(v) for v in batch["global_patient_id"]]
                else:
                    sample_groups = [str(v) for v in batch["patient_id"]]
                keep_mask = None
                if ref_groups is not None:
                    keep_mask = np.asarray([g in ref_groups for g in sample_groups], dtype=bool)
                    if not bool(keep_mask.any()):
                        continue
                z_e = encode_erased(x, eraser_reference_shift_family)
                if keep_mask is not None:
                    keep_idx = torch.as_tensor(keep_mask, dtype=torch.bool, device=z_e.device)
                    z_e = z_e[keep_idx]
                logits = head(z_e)
                probs_chunk = torch.sigmoid(logits).cpu().numpy()
                probs_er.append(probs_chunk)
                collected_er += int(probs_chunk.shape[0])

                if erased_batch_idx == 1 or erased_batch_idx % int(args.ref_progress_every) == 0:
                    log(
                        f"erased_calib_progress batches={erased_batch_idx} "
                        f"collected={collected_er} target={len(probs_no)}"
                    )

                if args.max_ref is not None and collected_er >= len(probs_no):
                    log(f"erased_calib_early_stop target={len(probs_no)} collected={collected_er}")
                    break

        probs_er = np.concatenate(probs_er, axis=0)
        np.savez_compressed(probs_er_ckpt, probs_er=probs_er)
        probs_er_state.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "samples": int(probs_er.shape[0]),
                },
                indent=2,
            )
        )
        log(f"erased_calib_done samples={probs_er.shape[0]} checkpoint={probs_er_ckpt}")
    log(
        f"post_erased_calib_begin probs_no_n={probs_no.shape[0]} "
        f"probs_er_n={probs_er.shape[0]}"
    )
    log(
        f"post_erased_calib_done probs_no_n={probs_no.shape[0]} "
        f"probs_er_n={probs_er.shape[0]}"
    )
    # build technical strata for conformal calibration
    group_labels = None
    if policy_cfg.stratify_technical:
        group_labels = np.asarray(strata_ref, dtype=object) if strata_ref is not None else None
    threshold_meta = {
        "source": args.clinical_threshold_source,
        "cost_fn": float(policy_cfg.cost_fn),
        "cost_fp": float(policy_cfg.cost_fp),
    }
    if args.clinical_threshold_source == "cost_optimal":
        clinical_thresholds = optimal_thresholds(
            labels_ref,
            probs_no,
            cost_fn=policy_cfg.cost_fn,
            cost_fp=policy_cfg.cost_fp,
        )
    elif args.clinical_threshold_source == "target_sensitivity":
        if not target_sensitivities:
            raise RuntimeError("--clinical_threshold_source=target_sensitivity requires --target_sensitivities")
        target_val = float(target_sensitivities[0])
        clinical_thresholds = _threshold_vector_at_target(
            labels_ref,
            probs_no,
            mode="target_sensitivity",
            target=target_val,
        )
        threshold_meta["target_kind"] = "sensitivity"
        threshold_meta["target_value"] = target_val
    else:
        if not target_specificities:
            raise RuntimeError("--clinical_threshold_source=target_specificity requires --target_specificities")
        target_val = float(target_specificities[0])
        clinical_thresholds = _threshold_vector_at_target(
            labels_ref,
            probs_no,
            mode="target_specificity",
            target=target_val,
        )
        threshold_meta["target_kind"] = "specificity"
        threshold_meta["target_value"] = target_val
    log(
        f"policy_calibrate_begin probs_no_shape={tuple(probs_no.shape)} "
        f"probs_er_shape={tuple(probs_er.shape)} "
        f"labels_ref_shape={tuple(labels_ref.shape)} "
        f"group_labels_n={0 if group_labels is None else len(group_labels)}"
    )
    policy.calibrate(
        probs_no,
        probs_er,
        labels_ref,
        coverage=policy_cfg.coverage_budget,
        critical_idx=critical_idx,
        group_labels=group_labels,
        class_thresholds=clinical_thresholds,
    )
    log("policy_calibrate_done")
    # Estimate per-label harms strictly on calib split (no eval-label leakage).
    shift_harm_loader = calib_loader
    if args.max_ref is not None:
        shift_harm_batches = []
        shift_harm_n = 0
        for batch in calib_loader:
            shift_harm_batches.append(batch)
            if "labels" in batch:
                shift_harm_n += int(batch["labels"].shape[0])
            elif "image" in batch:
                shift_harm_n += int(batch["image"].shape[0])
            if shift_harm_n >= int(args.max_ref):
                break
        log(f"shift_harm_loader_materialized batches={len(shift_harm_batches)} samples={shift_harm_n}")
        shift_harm_loader = shift_harm_batches
    log(
        f"shift_harm_guard resume={bool(args.resume)} "
        f"shift_harm_ckpt_exists={shift_harm_ckpt.exists()} path={shift_harm_ckpt}"
    )
    if bool(args.resume) and shift_harm_ckpt.exists():
        log("shift_harm_table_resume_begin")
        policy.shift_harm = json.loads(shift_harm_ckpt.read_text())
        log(f"shift_harm_table_resume path={shift_harm_ckpt}")
        shift_harm_debug_rows = []
        shift_harm_debug_ckpt = ckpt_dir / "shift_harm_debug.json"
        if shift_harm_debug_ckpt.exists():
            try:
                shift_harm_debug_rows = json.loads(shift_harm_debug_ckpt.read_text())
            except Exception:
                shift_harm_debug_rows = []
        (ckpt_dir / "shift_harm_state.json").write_text(
            json.dumps({"status": "complete", "path": str(shift_harm_ckpt), "resumed": True}, indent=2)
        )
        log(f"shift_harm_checkpoint_loaded path={shift_harm_ckpt}")
        log(f"shift_harm_debug_loaded path={shift_harm_debug_ckpt} rows={len(shift_harm_debug_rows)}")
    else:
        if shift_harm_ckpt.exists():
            try:
                shift_harm_ckpt.unlink()
            except Exception:
                pass
        shift_harm_debug_ckpt = ckpt_dir / "shift_harm_debug.json"
        if shift_harm_debug_ckpt.exists():
            try:
                shift_harm_debug_ckpt.unlink()
            except Exception:
                pass
        _estimate_calib_shift_harm_table.logger = log
        _estimate_calib_shift_harm_table.progress_every = int(args.shift_harm_progress_every)
        _estimate_calib_shift_harm_table.checkpoint_every = int(args.shift_harm_checkpoint_every)
        _estimate_calib_shift_harm_table.checkpoint_path = shift_harm_ckpt
        _estimate_calib_shift_harm_table.resume_enabled = bool(args.resume)

        log("shift_harm_table_begin")
        policy.shift_harm = _estimate_calib_shift_harm_table(
            calib_loader=shift_harm_loader,
            encoder=encoder,
            head=head,
            encode_erased=encode_erased,
            reference_shift_family=eraser_reference_shift_family,
            shift_manager=shift_manager,
            shift_types=list(shift_types),
            scenario_map=scenario_map,
            mixed_profiles=list(cfg.intervention.mixed_profiles or []),
            clinical_thresholds=clinical_thresholds,
            cost_fn=policy_cfg.cost_fn,
            cost_fp=policy_cfg.cost_fp,
            class_cost_fn=policy_cfg.class_cost_fn,
            class_cost_fp=policy_cfg.class_cost_fp,
            critical_idx=critical_idx,
            device=device,
            label_space=shift_id_label_space,
        )
        log("shift_harm_table_done")
        shift_harm_ckpt.write_text(json.dumps(policy.shift_harm, indent=2))
        shift_harm_debug_rows = getattr(_estimate_calib_shift_harm_table, "last_debug_rows", [])
        shift_harm_debug_ckpt.write_text(json.dumps(shift_harm_debug_rows, indent=2))
        (ckpt_dir / "shift_harm_state.json").write_text(
            json.dumps({"status": "complete", "path": str(shift_harm_ckpt)}, indent=2)
        )
        log(f"shift_harm_checkpoint_saved path={shift_harm_ckpt}")
        log(f"shift_harm_debug_saved path={shift_harm_debug_ckpt} rows={len(shift_harm_debug_rows)}")
    log(f"shift_harm_debug_saved path={shift_harm_debug_ckpt} rows={len(shift_harm_debug_rows)}")
    source_priors_hard_fail = bool(paper_tier and decision_policy_uses_recalibration)
    source_label_priors, source_label_priors_status = _sanitize_numeric_vector(
        source_label_priors_raw,
        "source_label_priors",
        expected_length=len(label_cols),
        fallback_value=0.5,
        paper_tier=source_priors_hard_fail,
        incomplete_fields=policy_contract_incomplete_fields,
    )
    class_thresholds, class_thresholds_status = _sanitize_numeric_vector(
        clinical_thresholds,
        "class_thresholds",
        expected_length=len(label_cols),
        fallback_value=0.5,
        paper_tier=paper_tier,
        incomplete_fields=policy_contract_incomplete_fields,
    )
    log("shift_id_v2 calibrated")
    if args.save_shift_id:
        shift_id.save(args.save_shift_id)
        log(f"Saved shift_id model to {args.save_shift_id}")
    policy_artifact_payload = {
        "thresholds": _serialize_json_value(
            policy.thresholds,
            "thresholds",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "class_thresholds": class_thresholds,
        "class_thresholds_meta": _serialize_json_value(
            threshold_meta,
            "class_thresholds_meta",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "eraser_selection_mode": str(eraser_bundle_meta.get("kind", "single")),
        "eraser_reference_shift_family": str(eraser_reference_shift_family),
        "source_label_priors": source_label_priors,
        "coverage": require_finite_scalar(policy_cfg.coverage_budget, "coverage"),
        "harm": _serialize_json_value(
            policy.harm,
            "harm",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "shift_harm": _serialize_json_value(
            policy.shift_harm,
            "shift_harm",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "source_label_priors_status": source_label_priors_status,
        "class_thresholds_status": class_thresholds_status,
        "label_space": "classifier_label_space",
        "label_cols": [str(v) for v in label_cols],
        "class_cost_fn": _serialize_json_value(
            policy_cfg.class_cost_fn,
            "class_cost_fn",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "class_cost_fp": _serialize_json_value(
            policy_cfg.class_cost_fp,
            "class_cost_fp",
            paper_tier=paper_tier,
            incomplete_fields=policy_contract_incomplete_fields,
        ),
        "recalibration_mode": str(getattr(policy_cfg, "recalibration_mode", "global")),
        "recalibration_counts_by_stratum": None,
        "target_priors_by_stratum_present": False,
        "shift_id_calibration_report_present": False,
        "decision_label_source": decision_label_source,
        "decision_label_diagnostic_only": bool(decision_label_diagnostic_only),
    }
    policy_artifact_payload.update(decision_policy_contract)

    # shift-id report (confusion matrix + ECE)
    shift_id_calibration_report = None
    try:
        scenarios = list(shift_id_report_scenarios or [])
        if not scenarios:
            rng = np.random.default_rng(seed)
            n = len(feats_ref)
            idx = rng.permutation(n)
            half = n // 2
            scores_none = shift_id.compute_scores(feats_ref[idx[:half]], probs_ref[idx[:half]], feats_ref[idx[half:]], probs_ref[idx[half:]])
            scenarios.append(("none", scores_none))
            if group_ref is not None:
                uniq, counts = np.unique(group_ref, return_counts=True)
                valid = [(u, c) for u, c in zip(uniq.tolist(), counts.tolist()) if u != "unknown" and c > 5]
                valid = sorted(valid, key=lambda x: (-x[1], str(x[0])))
                if len(valid) >= 2:
                    g1, g2 = valid[0][0], valid[1][0]
                    m1 = group_ref == g1
                    m2 = group_ref == g2
                    if m1.sum() > 5 and m2.sum() > 5:
                        scores_cov = shift_id.compute_scores(feats_ref[m1], probs_ref[m1], feats_ref[m2], probs_ref[m2])
                        scenarios.append(("covariate", scores_cov))
            if labels_ref is not None and labels_ref.ndim == 2:
                prevalences = labels_ref.mean(axis=0)
                label_idx = int(np.nanargmax(prevalences))
                y = labels_ref[:, label_idx]
                pos = np.where(y == 1)[0]
                neg = np.where(y == 0)[0]
                if len(pos) > 5 and len(neg) > 5:
                    targ = np.concatenate([pos, rng.choice(pos, size=len(pos), replace=True), neg])
                    targ = targ[rng.permutation(len(targ))][:min(len(targ), n)]
                    scores_prev = shift_id.compute_scores(feats_ref, probs_ref, feats_ref[targ], probs_ref[targ])
                    scenarios.append(("prevalence", scores_prev))
                    if group_ref is not None:
                        uniq, counts = np.unique(group_ref, return_counts=True)
                        valid = [(u, c) for u, c in zip(uniq.tolist(), counts.tolist()) if u != "unknown" and c > 5]
                        valid = sorted(valid, key=lambda x: (-x[1], str(x[0])))
                        if valid:
                            g1 = valid[0][0]
                            mask = group_ref == g1
                            mixed_idx = targ[:mask.sum()] if mask.sum() > 0 else targ
                            scores_mix = shift_id.compute_scores(feats_ref[mask], probs_ref[mask], feats_ref[mixed_idx], probs_ref[mixed_idx])
                            scenarios.append(("mixed", scores_mix))
        y_true = []
        y_pred = []
        conf_true = []
        correct_true = []
        for label, scores in scenarios:
            probs = shift_id.predict_proba_from_scores(scores)
            pred = shift_id._decision_from_probs(probs)
            y_true.append(label)
            y_pred.append(pred)
            if label in shift_id.classes_:
                idx = shift_id.classes_.index(label)
                conf_true.append(probs[idx])
                correct_true.append(float(label == pred))
        # confusion matrix
        labels_list = shift_id.classes_
        mat = {l: {k: 0 for k in labels_list} for l in labels_list}
        for t, p in zip(y_true, y_pred):
            if t in mat and p in mat[t]:
                mat[t][p] += 1
        # simple ECE for type probs
        ece = float("nan")
        if conf_true:
            conf_true = np.array(conf_true)
            correct = np.array(correct_true, dtype=float)
            bins = np.linspace(0, 1, 6)
            ece_val = 0.0
            for i in range(len(bins) - 1):
                mask = (conf_true >= bins[i]) & (conf_true < bins[i + 1])
                if mask.sum() == 0:
                    continue
                acc = correct[mask].mean()
                conf = conf_true[mask].mean()
                ece_val += (mask.sum() / len(conf_true)) * abs(acc - conf)
            ece = float(ece_val)
        misid = misidentification_cost(y_true, y_pred, labels_list)
        if bool(getattr(cfg.shift_id, "report_calibration_metrics", True)) and scenarios:
            shift_id_calibration_report = shift_id.calibration_report_from_scores(scenarios)
        (out_dir / "shift_id_report.json").write_text(
            json.dumps(
                {
                    "label_space": shift_id_label_space,
                    "group_key": shift_group_key,
                    "classes": labels_list,
                    "confusion": mat,
                    "ece": ece,
                    "misidentification_cost": misid,
                    "calibration_report": shift_id_calibration_report,
                },
                indent=2,
            )
        )
    except Exception as e:
        log(f"shift_id_report_error={e}")
    policy_artifact_payload["shift_id_calibration_report_present"] = bool(shift_id_calibration_report is not None)

    rows = []
    decisions = []
    harm_no_hist = []
    harm_er_hist = []
    harm_rec_hist = []
    op_rows = []
    op_target_rows = []
    bootstrap_rows = []
    scenario_trace_rows = []
    op_df = None
    op_target_df = None
    regret_vals_all = []
    regret_groups_all = []
    regret_vals_by_shift: dict[str, list[np.ndarray]] = {}
    regret_groups_by_shift: dict[str, list[np.ndarray]] = {}
    regret_vals_by_action: dict[str, list[np.ndarray]] = {}
    regret_groups_by_action: dict[str, list[np.ndarray]] = {}
    calib_label_stats = per_label_operating_stats(
        probs_ref,
        labels_ref,
        clinical_thresholds,
        label_cols=label_cols,
        cost_fn=policy_cfg.cost_fn,
        cost_fp=policy_cfg.cost_fp,
        class_cost_fn=policy_cfg.class_cost_fn,
        class_cost_fp=policy_cfg.class_cost_fp,
    )
    calib_target_defs = report_at_operating_points(
        labels_ref,
        probs_ref,
        label_names=label_cols,
        target_sensitivities=target_sensitivities,
        target_specificities=target_specificities,
    )
    label_to_idx = {name: idx for idx, name in enumerate(label_cols)}
    artifact_recalibration_counts_by_stratum = None
    artifact_target_priors_by_stratum_present = False

    for shift_type in shift_types:
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
            grid = scenario_map.get(shift_type)
            if grid is None:
                raise RuntimeError(f"No calibrated intensity grid for shift_type={shift_type}")
            scenarios = [(float(intensity), None, float(level)) for level, intensity in grid]
        for scenario in scenarios:
            intensity = scenario[0]
            realism_level = scenario[2] if len(scenario) > 2 else float("nan")
            feats = []
            probs = []
            probs_er = []
            labels = []
            fairness_group_keys = ["sex", "race", "age"]
            cache_keys = list(dict.fromkeys(fairness_group_keys + list(policy_cfg.strata_keys or [])))
            group_cache = {key: [] for key in cache_keys}
            meta_flag = False
            with torch.no_grad():
                eval_group_ids = []
                n_eval_collectedd = 0
                eval_limit = int(args.max_ref) if args.max_ref is not None else None
                batch_idx = 0
                eval_iter = iter(eval_loader)
                while True:
                    t_fetch0 = time.perf_counter()
                    log(f"eval_fetch_begin shift_type={shift_type} scenario={intensity} batch_idx={batch_idx + 1} collected={n_eval_collectedd}")
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        log(f"eval_fetch_eof shift_type={shift_type} scenario={intensity} batch_idx={batch_idx} collected={n_eval_collectedd}")
                        break
                    t_fetch1 = time.perf_counter()
                    batch_idx += 1
                    log(
                        f"eval_fetch_done shift_type={shift_type} scenario={intensity} "
                        f"batch_idx={batch_idx} collected={n_eval_collectedd} fetch_s={t_fetch1 - t_fetch0:.3f}"
                    )

                    t_batch0 = time.perf_counter()
                    x = batch["image"].to(device)
                    y = batch["labels"].to(device)
                    y_np = y.cpu().numpy()
                    t_after_load = time.perf_counter()

                    if eval_group_key in batch:
                        eval_group_ids.extend([str(v) for v in batch[eval_group_key]])
                    elif "global_patient_id" in batch:
                        eval_group_ids.extend([str(v) for v in batch["global_patient_id"]])
                    else:
                        eval_group_ids.extend([str(v) for v in batch["patient_id"]])

                    if shift_type == "mixed":
                        sample_ids = sample_ids_from_batch(batch)
                        x_shift, _meta = shift_manager.apply_batch_scenario(x, scenario[1], sample_ids=sample_ids)
                        if _meta and _meta.get("summary", {}).get("destructive_count", 0) > 0:
                            meta_flag = True
                    else:
                        sample_ids = sample_ids_from_batch(batch)
                        x_shift, _meta = shift_manager.apply_batch_with_meta(x, shift_type, float(intensity), sample_ids=sample_ids)
                        if _meta and _meta.get("summary", {}).get("destructive_count", 0) > 0:
                            meta_flag = True
                    t_after_shift = time.perf_counter()

                    z = encoder.encode(x_shift)
                    logits = head(z)
                    feats_batch = z.cpu().numpy()
                    probs_batch = torch.sigmoid(logits).cpu().numpy()
                    t_after_base = time.perf_counter()

                    logits_e = head(encode_erased(x_shift, shift_type))
                    probs_er_batch = torch.sigmoid(logits_e).cpu().numpy()
                    t_after_erased = time.perf_counter()

                    feats.append(feats_batch)
                    probs.append(probs_batch)
                    probs_er.append(probs_er_batch)
                    labels.append(y_np)

                    for key in group_cache.keys():
                        if key in batch:
                            group_cache[key].extend(list(batch[key]))
                        else:
                            group_cache[key].extend(["unknown"] * int(y_np.shape[0]))

                    n_eval_collectedd += int(y_np.shape[0])
                    log(
                        f"eval_batch_done shift_type={shift_type} scenario={intensity} "
                        f"batch_idx={batch_idx} batch_n={int(y_np.shape[0])} collected={n_eval_collectedd} "
                        f"load_s={t_after_load - t_batch0:.3f} shift_s={t_after_shift - t_after_load:.3f} "
                        f"base_s={t_after_base - t_after_shift:.3f} erased_s={t_after_erased - t_after_base:.3f} "
                        f"total_s={t_after_erased - t_batch0:.3f}"
                    )
                    if eval_limit is not None and n_eval_collectedd >= eval_limit:
                        log(f"eval_budget_early_stop shift_type={shift_type} scenario={intensity} collected={n_eval_collectedd}")
                        break

            feats = np.concatenate(feats, axis=0)
            probs = np.concatenate(probs, axis=0)
            probs_er = np.concatenate(probs_er, axis=0)
            labels = np.concatenate(labels, axis=0)

            if args.max_ref is not None:
                limit = int(args.max_ref)
                feats = feats[:limit]
                probs = probs[:limit]
                probs_er = probs_er[:limit]
                labels = labels[:limit]
                eval_group_ids = eval_group_ids[:limit]
                for key in list(group_cache.keys()):
                    group_cache[key] = group_cache[key][:limit]
            shift_info = shift_id.predict(feats_ref, probs_ref, feats, probs)
            if shift_id_label_space == "physics":
                shift_info["p_covariate"] = float(
                    sum(float(shift_info.get(f"p_{lbl}", 0.0)) for lbl in covariate_label_names)
                )
            destructive = meta_flag
            probs_rec = probs
            target_priors_est = None
            target_priors_by_stratum = None
            recalibration_counts_by_stratum = None
            eval_strata = None
            if bool(getattr(policy_cfg, "stratify_technical", False)):
                eval_strata = np.asarray(
                    [
                        "|".join(
                            f"{k}={str(group_cache.get(k, ['unknown'] * len(eval_group_ids))[i])}"
                            for k in policy_cfg.strata_keys
                        )
                        for i in range(len(eval_group_ids))
                    ],
                    dtype=object,
                )
            if prevalence_recalibrator is not None:
                if isinstance(prevalence_recalibrator, StratifiedLabelShiftRecalibrator):
                    probs_rec, target_priors_by_stratum, recalibration_counts_by_stratum = prevalence_recalibrator.adjust(
                        probs,
                        eval_strata if eval_strata is not None else np.asarray(["unknown"] * len(probs), dtype=object),
                    )
                    artifact_target_priors_by_stratum_present = True
                    if artifact_recalibration_counts_by_stratum is None:
                        artifact_recalibration_counts_by_stratum = {
                            str(k): int(v) for k, v in recalibration_counts_by_stratum.items()
                        }
                else:
                    probs_rec, target_priors_est = prevalence_recalibrator.adjust(probs)
            harm_no_all = expected_harm_multilabel(
                probs,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            harm_er_all = expected_harm_multilabel(
                probs_er,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            harm_rec_all = expected_harm_multilabel(
                probs_rec,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            if decision_policy is not None:
                # Decision must not use eval labels; use harms calibrated on calib split.
                pred_label = str(shift_info.get("label", "unknown"))
                oracle_label = str(shift_type)
                if decision_label_source == "predicted":
                    decision_label = pred_label
                elif decision_label_source == "oracle_shift_family":
                    decision_label = oracle_label
                else:
                    raise RuntimeError(f"Unsupported safety_policy.decision_label_source={decision_label_source!r}")
                label_harm = policy.shift_harm.get(decision_label, {})
                harm_no_decision = float(label_harm.get("no_eraser", policy.harm.get("no_eraser", float("nan"))))
                harm_er_decision = float(label_harm.get("eraser", policy.harm.get("eraser", float("nan"))))
                harm_rec_decision = float(label_harm.get("recalibrate", harm_no_decision))
                trace = decision_policy.decide(
                    shift_info,
                    harm_no=harm_no_decision,
                    harm_er=harm_er_decision,
                    harm_rec=harm_rec_decision,
                )
                decision = trace.action
                reason = trace.reason
            else:
                trace = None
                pred_label = str(shift_info.get("label", "unknown"))
                oracle_label = str(shift_type)
                if decision_label_source == "predicted":
                    decision_label = pred_label
                elif decision_label_source == "oracle_shift_family":
                    decision_label = oracle_label
                else:
                    raise RuntimeError(f"Unsupported safety_policy.decision_label_source={decision_label_source!r}")
                decision = policy.decide(shift_info, destructive=destructive)
                reason = "legacy_policy"

            sample_harm_no = sample_harm_multilabel(
                probs,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            sample_harm_er = sample_harm_multilabel(
                probs_er,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            sample_harm_rec = sample_harm_multilabel(
                probs_rec,
                labels,
                thresholds=clinical_thresholds,
                cost_fn=policy_cfg.cost_fn,
                cost_fp=policy_cfg.cost_fp,
                critical_idx=critical_idx if critical_idx else None,
                class_cost_fn=policy_cfg.class_cost_fn,
                class_cost_fp=policy_cfg.class_cost_fp,
            )
            if decision == "apply_eraser":
                sample_harm_policy = sample_harm_er
            elif decision == "recalibrate":
                sample_harm_policy = sample_harm_rec
            elif decision == "no_eraser":
                sample_harm_policy = sample_harm_no
            else:
                sample_harm_policy = np.full_like(sample_harm_no, float(policy_cfg.abstain_cost))
            sample_harm_oracle = np.minimum.reduce(
                [
                    sample_harm_no,
                    sample_harm_er,
                    sample_harm_rec,
                    np.full_like(sample_harm_no, float(policy_cfg.abstain_cost)),
                ]
            )
            sample_regret = sample_harm_policy - sample_harm_oracle
            if len(eval_group_ids) != len(sample_harm_policy):
                eval_group_ids = [str(i) for i in range(len(sample_harm_policy))]
            if shift_type == "mixed":
                scenario_name = str(intensity)
            else:
                scenario_name = f"{float(intensity):.6f}"
            scenario_key = f"{shift_type}|{scenario_name}"
            sample_groups = np.asarray([f"{scenario_key}|{g}" for g in eval_group_ids], dtype=object)
            regret_vals_all.append(np.asarray(sample_regret, dtype=float))
            regret_groups_all.append(sample_groups)
            regret_vals_by_shift.setdefault(str(shift_type), []).append(np.asarray(sample_regret, dtype=float))
            regret_groups_by_shift.setdefault(str(shift_type), []).append(sample_groups)
            regret_vals_by_action.setdefault(str(decision), []).append(np.asarray(sample_regret, dtype=float))
            regret_groups_by_action.setdefault(str(decision), []).append(sample_groups)
            boot_harm = cluster_bootstrap_mean(
                sample_harm_policy,
                np.asarray(eval_group_ids, dtype=object),
                n_boot=args.bootstrap_samples,
                seed=seed + 1009,
            )
            boot_regret = cluster_bootstrap_mean(
                sample_regret,
                np.asarray(eval_group_ids, dtype=object),
                n_boot=args.bootstrap_samples,
                seed=seed + 1013,
            )
            decision_row = {
                "shift": shift_type,
                "intensity": intensity,
                **shift_info,
                "decision": decision,
                "reason": reason,
                "decision_policy_variant": str(decision_policy_contract["decision_policy_variant"]),
                "decision_policy_active": bool(decision_policy_contract["decision_policy_active"]),
                "decision_path_source": str(decision_policy_contract["decision_path_source"]),
                "decision_label_source": decision_label_source,
                "decision_label": str(decision_label),
                "predicted_shift_label": str(pred_label),
                "oracle_shift_family": str(oracle_label),
                "decision_label_matches_oracle": bool(decision_label == oracle_label),
                "decision_label_matches_predicted": bool(decision_label == pred_label),
                "decision_label_diagnostic_only": bool(decision_label_diagnostic_only),
                "class_cost_fn": policy_cfg.class_cost_fn,
                "class_cost_fp": policy_cfg.class_cost_fp,
                "recalibration_mode": str(getattr(policy_cfg, "recalibration_mode", "global")),
                "recalibration_counts_by_stratum": recalibration_counts_by_stratum,
                "target_priors_by_stratum_present": bool(target_priors_by_stratum is not None),
                "shift_id_calibration_report_present": bool(shift_id_calibration_report is not None),
            }
            decision_row["realism_level"] = realism_level
            decision_row["decision_harm_source"] = "calib_policy_harm" if decision_policy is not None else "legacy_policy"
            if decision_policy is not None:
                decision_row["decision_harm_no_calib"] = float(harm_no_decision)
                decision_row["decision_harm_eraser_calib"] = float(harm_er_decision)
                decision_row["decision_harm_recalibrate_calib"] = float(harm_rec_decision)
                decision_row["p_none"] = float(trace.p_none)
                decision_row["p_covariate"] = float(trace.p_covariate)
                decision_row["p_prevalence"] = float(trace.p_prevalence)
                decision_row["p_mixed"] = float(trace.p_mixed)
                decision_row["uncertainty"] = float(trace.uncertainty)
            else:
                decision_row["p_none"] = float(shift_info.get("p_none", float("nan")))
                decision_row["p_covariate"] = float(shift_info.get("p_covariate", float("nan")))
                decision_row["p_prevalence"] = float(shift_info.get("p_prevalence", float("nan")))
                decision_row["p_mixed"] = float(shift_info.get("p_mixed", float("nan")))
                decision_row["uncertainty"] = float(shift_info.get("uncertainty", float("nan")))
            decision_row.update(
                policy_regret(
                    decision,
                    harm_no_all,
                    harm_er_all,
                    abstain_cost=policy_cfg.abstain_cost,
                    harm_recalibrate=harm_rec_all,
                )
            )
            if target_priors_est is not None:
                decision_row["recalibrated_target_prevalence_mean"] = float(np.mean(target_priors_est))
            if target_priors_by_stratum is not None:
                decision_row["recalibration_counts_by_stratum"] = {
                    str(k): int(v) for k, v in (recalibration_counts_by_stratum or {}).items()
                }
            decision_row.update(
                {
                    "policy_harm_boot_mean": boot_harm["mean"],
                    "policy_harm_boot_ci_low": boot_harm["ci_low"],
                    "policy_harm_boot_ci_high": boot_harm["ci_high"],
                    "policy_regret_boot_mean": boot_regret["mean"],
                    "policy_regret_boot_ci_low": boot_regret["ci_low"],
                    "policy_regret_boot_ci_high": boot_regret["ci_high"],
                }
            )
            if shift_type == "mixed":
                decision_row["profile"] = intensity
            _mark_nonfinite_fields(
                decision_row,
                [
                    "policy_harm",
                    "oracle_harm",
                    "policy_regret",
                    "policy_harm_boot_mean",
                    "policy_harm_boot_ci_low",
                    "policy_harm_boot_ci_high",
                    "policy_regret_boot_mean",
                    "policy_regret_boot_ci_low",
                    "policy_regret_boot_ci_high",
                ],
                prefix="policy_decisions",
                paper_tier=paper_tier,
                incomplete_fields=policy_contract_incomplete_fields,
            )
            decisions.append(decision_row)
            harm_no_hist.append(harm_no_all)
            harm_er_hist.append(harm_er_all)
            harm_rec_hist.append(harm_rec_all)
            bootstrap_rows.append(
                {
                    "shift": shift_type,
                    "intensity": intensity,
                    "realism_level": realism_level,
                    "decision": decision,
                    "policy_harm_mean": boot_harm["mean"],
                    "policy_harm_ci_low": boot_harm["ci_low"],
                    "policy_harm_ci_high": boot_harm["ci_high"],
                    "policy_regret_mean": boot_regret["mean"],
                    "policy_regret_ci_low": boot_regret["ci_low"],
                    "policy_regret_ci_high": boot_regret["ci_high"],
                    "n": boot_harm["n"],
                    "n_clusters": boot_harm["n_clusters"],
                }
            )

            if decision == "apply_eraser":
                probs = probs_er
            elif decision == "recalibrate":
                probs = probs_rec

            accepted_probs = None
            accepted_labels = None
            if decision == "abstain":
                row = {"shift": shift_type, "intensity": intensity, "realism_level": realism_level, "decision": decision, "coverage": 0.0}
                for lname, base in calib_label_stats.items():
                    op_rows.append(
                        {
                            "shift": shift_type,
                            "intensity": intensity,
                            "realism_level": realism_level,
                            "decision": decision,
                            "label": lname,
                            "threshold": float(base.get("threshold", float("nan"))),
                            "accepted_n_valid": 0.0,
                            "accepted_prevalence": float("nan"),
                            "accepted_fnr": float("nan"),
                            "accepted_fpr": float("nan"),
                            "accepted_harm": float("nan"),
                            "calib_fnr": float(base.get("fnr", float("nan"))),
                            "calib_fpr": float(base.get("fpr", float("nan"))),
                            "calib_harm": float(base.get("harm", float("nan"))),
                            "delta_fnr_vs_calib": float("nan"),
                            "delta_fpr_vs_calib": float("nan"),
                            "delta_harm_vs_calib": float("nan"),
                        }
                    )
            else:
                action = "eraser" if decision == "apply_eraser" else "no_eraser"
                # apply stratified thresholds if available
                group_labels_eval = None
                if policy_cfg.stratify_technical:
                    group_labels_eval = eval_strata
                mask = policy.apply_threshold(probs, action, group_labels=group_labels_eval)
                if mask.sum() == 0:
                    row = {"shift": shift_type, "intensity": intensity, "realism_level": realism_level, "decision": decision, "coverage": 0.0}
                    for lname, base in calib_label_stats.items():
                        op_rows.append(
                            {
                                "shift": shift_type,
                                "intensity": intensity,
                                "realism_level": realism_level,
                                "decision": decision,
                                "label": lname,
                                "threshold": float(base.get("threshold", float("nan"))),
                                "accepted_n_valid": 0.0,
                                "accepted_prevalence": float("nan"),
                                "accepted_fnr": float("nan"),
                                "accepted_fpr": float("nan"),
                                "accepted_harm": float("nan"),
                                "calib_fnr": float(base.get("fnr", float("nan"))),
                                "calib_fpr": float(base.get("fpr", float("nan"))),
                                "calib_harm": float(base.get("harm", float("nan"))),
                                "delta_fnr_vs_calib": float("nan"),
                                "delta_fpr_vs_calib": float("nan"),
                                "delta_harm_vs_calib": float("nan"),
                            }
                        )
                else:
                    accepted_probs = probs[mask]
                    accepted_labels = labels[mask]
                    m = eval_metrics(
                        accepted_probs,
                        accepted_labels,
                        coverage=cfg.erasure_eval.coverage,
                        thresholds=clinical_thresholds,
                    )
                    harm = expected_harm_multilabel(
                        accepted_probs,
                        accepted_labels,
                        critical_idx=critical_idx if critical_idx else None,
                        thresholds=clinical_thresholds,
                        cost_fn=policy_cfg.cost_fn,
                        cost_fp=policy_cfg.cost_fp,
                        class_cost_fn=policy_cfg.class_cost_fn,
                        class_cost_fp=policy_cfg.class_cost_fp,
                    )
                    row = {"shift": shift_type, "intensity": intensity, "realism_level": realism_level, "decision": decision, "coverage": float(mask.mean()), "expected_harm": harm, **m}
                    scen_label_stats = per_label_operating_stats(
                        accepted_probs,
                        accepted_labels,
                        clinical_thresholds,
                        label_cols=label_cols,
                        cost_fn=policy_cfg.cost_fn,
                        cost_fp=policy_cfg.cost_fp,
                        class_cost_fn=policy_cfg.class_cost_fn,
                        class_cost_fp=policy_cfg.class_cost_fp,
                    )
                    for lname, stats in scen_label_stats.items():
                        base = calib_label_stats.get(lname, {})
                        b_fnr = float(base.get("fnr", float("nan")))
                        b_fpr = float(base.get("fpr", float("nan")))
                        b_harm = float(base.get("harm", float("nan")))
                        s_fnr = float(stats.get("fnr", float("nan")))
                        s_fpr = float(stats.get("fpr", float("nan")))
                        s_harm = float(stats.get("harm", float("nan")))
                        op_rows.append(
                            {
                                "shift": shift_type,
                                "intensity": intensity,
                                "realism_level": realism_level,
                                "decision": decision,
                                "label": lname,
                                "threshold": float(stats.get("threshold", float("nan"))),
                                "accepted_n_valid": float(stats.get("n_valid", 0.0)),
                                "accepted_prevalence": float(stats.get("prevalence", float("nan"))),
                                "accepted_fnr": s_fnr,
                                "accepted_fpr": s_fpr,
                                "accepted_harm": s_harm,
                                "calib_fnr": b_fnr,
                                "calib_fpr": b_fpr,
                                "calib_harm": b_harm,
                                "delta_fnr_vs_calib": s_fnr - b_fnr if np.isfinite(s_fnr) and np.isfinite(b_fnr) else float("nan"),
                                "delta_fpr_vs_calib": s_fpr - b_fpr if np.isfinite(s_fpr) and np.isfinite(b_fpr) else float("nan"),
                                "delta_harm_vs_calib": s_harm - b_harm if np.isfinite(s_harm) and np.isfinite(b_harm) else float("nan"),
                            }
                        )

                # fairness if metadata present
                for gname in fairness_group_keys:
                    gvals = group_cache.get(gname, [])
                    if len(gvals) == 0:
                        continue
                    if gname == "age":
                        import pandas as _pd
                        gvals_series = _pd.to_numeric(_pd.Series(gvals), errors="coerce")
                        gvals_mask = gvals_series.notna().to_numpy()
                        gvals_binned = np.full(len(gvals_series), "unknown", dtype=object)
                        if bool(gvals_mask.any()):
                            gvals_valid = gvals_series.to_numpy(dtype=float)[gvals_mask]
                            gvals_binned[gvals_mask] = np.asarray(
                                bin_age(gvals_valid, cfg.erasure_eval.age_bins),
                                dtype=object,
                            )
                        gvals = gvals_binned
                    else:
                        gvals = np.array(gvals).astype(str)
                    row.update({f"{gname}_worst_auroc": worst_group_auc(labels, probs, gvals, min_group_size=cfg.metrics.min_group_size)["worst_group_auroc_macro"]})
                    row.update({f"{gname}_fnr_gap": underdiagnosis_gap(labels, probs, gvals, threshold=clinical_thresholds, min_group_size=cfg.metrics.min_group_size)["fnr_gap_macro"]})
                    row.update({f"{gname}_ece_gap": ece_gap(labels, probs, gvals, n_bins=cfg.metrics.calibration_bins, min_group_size=cfg.metrics.min_group_size)["ece_gap"]})

            _mark_nonfinite_fields(
                row,
                ["coverage", "expected_harm", "ece", "risk_at_coverage", "auroc_macro", "auprc_macro"],
                prefix="safety",
                paper_tier=paper_tier,
                incomplete_fields=policy_contract_incomplete_fields,
            )

            for base in calib_target_defs:
                lname = str(base["label"])
                lidx = label_to_idx.get(lname)
                if lidx is None:
                    continue
                if accepted_probs is None or accepted_labels is None:
                    scen_stats = {
                        "n_valid": 0.0,
                        "threshold": float(base["threshold"]),
                        "sensitivity": float("nan"),
                        "specificity": float("nan"),
                        "ppv": float("nan"),
                        "npv": float("nan"),
                        "prevalence": float("nan"),
                    }
                else:
                    scen_stats = operating_point_stats(
                        accepted_labels[:, lidx],
                        accepted_probs[:, lidx],
                        threshold=float(base["threshold"]),
                    )
                rec = {
                    "shift": shift_type,
                    "intensity": intensity,
                    "realism_level": realism_level,
                    "decision": decision,
                    "label": lname,
                    "target_kind": str(base["target_kind"]),
                    "target_value": float(base["target_value"]),
                    "threshold": float(base["threshold"]),
                    "accepted_n_valid": float(scen_stats["n_valid"]),
                    "calib_n_valid": float(base.get("n_valid", float("nan"))),
                }
                for key in ["sensitivity", "specificity", "ppv", "npv", "prevalence"]:
                    s_val = float(scen_stats.get(key, float("nan")))
                    b_val = float(base.get(key, float("nan")))
                    rec[f"accepted_{key}"] = s_val
                    rec[f"calib_{key}"] = b_val
                    rec[f"delta_{key}_vs_calib"] = (
                        s_val - b_val if np.isfinite(s_val) and np.isfinite(b_val) else float("nan")
                    )
                op_target_rows.append(rec)

            if shift_type == "mixed":
                row["profile"] = intensity
            rows.append(row)
            accepted_samples = 0 if decision == "abstain" else int(round(float(row.get("coverage", 0.0)) * len(labels)))
            abstained_samples = int(len(labels) - accepted_samples)
            scenario_trace_rows.append(
                {
                    "scenario_id": f"{shift_type}:{intensity}",
                    "shift_family": shift_type,
                    "intensity_or_realism_level": float(realism_level) if np.isfinite(realism_level) else intensity,
                    "pair_mode": str(getattr(cfg.intervention, "pair_mode", "unknown")),
                    "is_calibrated_atomic": bool(shift_type != "mixed" and cfg.intervention.use_realism and np.isfinite(realism_level)),
                    "is_mixed_stress": bool(shift_type == "mixed"),
                    "decision": decision,
                    "reason": reason,
                    "p_none": float(decision_row.get("p_none", float("nan"))),
                    "p_covariate": float(decision_row.get("p_covariate", float("nan"))),
                    "p_prevalence": float(decision_row.get("p_prevalence", float("nan"))),
                    "p_mixed": float(decision_row.get("p_mixed", float("nan"))),
                    "uncertainty": float(decision_row.get("uncertainty", float("nan"))),
                    "harm_no": float(harm_no_all),
                    "harm_eraser": float(harm_er_all),
                    "harm_recalibrated": float(harm_rec_all),
                    "coverage": float(row.get("coverage", 0.0)),
                    "active_group_key": str(eval_group_key),
                    "active_strata": ",".join(policy_cfg.strata_keys) if policy_cfg.stratify_technical else "",
                    "accepted_samples": accepted_samples,
                    "abstained_samples": abstained_samples,
                    "decision_policy_variant": str(decision_policy_contract["decision_policy_variant"]),
                    "decision_policy_active": bool(decision_policy_contract["decision_policy_active"]),
                    "decision_path_source": str(decision_policy_contract["decision_path_source"]),
                    "decision_label_source": decision_label_source,
                    "decision_label": str(decision_label),
                    "predicted_shift_label": str(pred_label),
                    "oracle_shift_family": str(oracle_label),
                    "decision_label_matches_oracle": bool(decision_label == oracle_label),
                    "decision_label_matches_predicted": bool(decision_label == pred_label),
                    "decision_label_diagnostic_only": bool(decision_label_diagnostic_only),
                }
            )
            log(f"policy {shift_type}@{intensity} decision={decision} metrics={row}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "safety.csv", index=False)
    pd.DataFrame(decisions).to_csv(out_dir / "policy_decisions.csv", index=False)
    if scenario_trace_rows:
        pd.DataFrame(scenario_trace_rows).to_csv(out_dir / "scenario_trace.csv", index=False)
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(out_dir / "policy_bootstrap.csv", index=False)
    op_df = None
    if op_rows:
        op_df = pd.DataFrame(op_rows)
        op_df.to_csv(out_dir / "policy_operating_points.csv", index=False)
        drift_df = (
            op_df.groupby("label", dropna=False)[["delta_fnr_vs_calib", "delta_fpr_vs_calib", "delta_harm_vs_calib"]]
            .agg(["mean", "std", "max", "min"])
            .reset_index()
        )
        drift_rows = []
        for _, r in drift_df.iterrows():
            label_key = "label" if "label" in drift_df.columns else ("label", "")
            row = {"label": str(r[label_key])}
            for base in ["delta_fnr_vs_calib", "delta_fpr_vs_calib", "delta_harm_vs_calib"]:
                for stat in ["mean", "std", "max", "min"]:
                    row[f"{base}_{stat}"] = float(r[(base, stat)])
            drift_rows.append(
                _serialize_op_summary_row(
                    row,
                    required_fields=("label",),
                    prefix="policy_operating_points_summary",
                    paper_tier=paper_tier,
                    incomplete_fields=policy_contract_incomplete_fields,
                )
            )
        (out_dir / "policy_operating_points_summary.json").write_text(
            json.dumps(
                _serialize_json_value(
                    {
                    "n_rows": int(len(op_df)),
                    "labels": sorted({str(x) for x in op_df["label"].dropna().tolist()}),
                    "drift_stats": drift_rows,
                    },
                    "policy_operating_points_summary",
                    paper_tier=paper_tier,
                    incomplete_fields=policy_contract_incomplete_fields,
                ),
                indent=2,
            )
        )
    op_target_df = None
    if op_target_rows:
        op_target_df = pd.DataFrame(op_target_rows)
        op_target_df.to_csv(out_dir / "policy_operating_points_targeted.csv", index=False)
        drift_cols = [
            "delta_sensitivity_vs_calib",
            "delta_specificity_vs_calib",
            "delta_ppv_vs_calib",
            "delta_npv_vs_calib",
        ]
        drift_summary = (
            op_target_df.groupby(["label", "target_kind", "target_value"], dropna=False)[drift_cols]
            .agg(["mean", "std", "max", "min"])
            .reset_index()
        )
        rows_target = []
        for _, r in drift_summary.iterrows():
            rec = {
                "label": str(r["label"]),
                "target_kind": str(r["target_kind"]),
                "target_value": float(r["target_value"].iloc[0] if hasattr(r["target_value"], "iloc") else r["target_value"]),
            }
            for col in drift_cols:
                for stat in ["mean", "std", "max", "min"]:
                    rec[f"{col}_{stat}"] = float(r[(col, stat)])
            rows_target.append(
                _serialize_op_summary_row(
                    rec,
                    required_fields=("label", "target_kind", "target_value"),
                    prefix="policy_operating_points_targeted_summary",
                    paper_tier=paper_tier,
                    incomplete_fields=policy_contract_incomplete_fields,
                )
            )
        (out_dir / "policy_operating_points_targeted_summary.json").write_text(
            json.dumps(
                _serialize_json_value(
                    {
                    "n_rows": int(len(op_target_df)),
                    "target_sensitivities": [float(v) for v in target_sensitivities],
                    "target_specificities": [float(v) for v in target_specificities],
                    "drift_stats": rows_target,
                    },
                    "policy_operating_points_targeted_summary",
                    paper_tier=paper_tier,
                    incomplete_fields=policy_contract_incomplete_fields,
                ),
                indent=2,
            )
        )
    summary = regret_by_shift(
        decisions,
        harm_no_hist,
        harm_er_hist,
        abstain_cost=policy_cfg.abstain_cost,
        harms_recalibrate=harm_rec_hist,
    )
    decomposition = regret_decomposition(
        decisions,
        harm_no_hist,
        harm_er_hist,
        abstain_cost=policy_cfg.abstain_cost,
        harms_recalibrate=harm_rec_hist,
    )
    if regret_vals_all:
        boot_overall = cluster_bootstrap_mean(
            np.concatenate(regret_vals_all, axis=0),
            np.concatenate(regret_groups_all, axis=0),
            n_boot=args.bootstrap_samples,
            seed=seed + 3001,
        )
        boot_by_shift = {
            k: cluster_bootstrap_mean(
                np.concatenate(v, axis=0),
                np.concatenate(regret_groups_by_shift[k], axis=0),
                n_boot=args.bootstrap_samples,
                seed=seed + 3011 + i,
            )
            for i, (k, v) in enumerate(sorted(regret_vals_by_shift.items()))
        }
        boot_by_action = {
            k: cluster_bootstrap_mean(
                np.concatenate(v, axis=0),
                np.concatenate(regret_groups_by_action[k], axis=0),
                n_boot=args.bootstrap_samples,
                seed=seed + 3111 + i,
            )
            for i, (k, v) in enumerate(sorted(regret_vals_by_action.items()))
        }
    else:
        boot_overall = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0.0, "n_clusters": 0.0, "n_boot": float(args.bootstrap_samples)}
        boot_by_shift = {}
        boot_by_action = {}
    summary.update(
        {
            "regret_boot_mean": float(boot_overall["mean"]),
            "regret_boot_ci_low": float(boot_overall["ci_low"]),
            "regret_boot_ci_high": float(boot_overall["ci_high"]),
            "regret_boot_n": int(boot_overall["n"]),
            "regret_boot_n_clusters": int(boot_overall["n_clusters"]),
        }
    )
    _mark_nonfinite_fields(
        summary,
        ["regret_mean", "regret_p95", "regret_boot_mean", "regret_boot_ci_low", "regret_boot_ci_high"],
        prefix="policy_regret_summary",
        paper_tier=paper_tier,
        incomplete_fields=policy_contract_incomplete_fields,
    )
    decomposition["bootstrap_ci"] = {
        "overall": boot_overall,
        "by_shift": boot_by_shift,
        "by_action": boot_by_action,
    }
    (out_dir / "policy_regret_summary.json").write_text(
        json.dumps(
            _serialize_json_value(
                summary,
                "policy_regret_summary",
                paper_tier=paper_tier,
                incomplete_fields=policy_contract_incomplete_fields,
            ),
            indent=2,
        )
    )
    (out_dir / "policy_regret_decomposition.json").write_text(
        json.dumps(
            _serialize_json_value(
                decomposition,
                "policy_regret_decomposition",
                paper_tier=paper_tier,
                incomplete_fields=policy_contract_incomplete_fields,
            ),
            indent=2,
        )
    )
    if decomposition.get("by_shift"):
        pd.DataFrame(
            [{"shift": k, **v} for k, v in decomposition["by_shift"].items()]
        ).to_csv(out_dir / "policy_regret_by_shift.csv", index=False)
    if decomposition.get("by_action"):
        pd.DataFrame(
            [{"action": k, **v} for k, v in decomposition["by_action"].items()]
        ).to_csv(out_dir / "policy_regret_by_action.csv", index=False)
    policy_artifact_payload["recalibration_counts_by_stratum"] = _serialize_json_value(
        artifact_recalibration_counts_by_stratum,
        "recalibration_counts_by_stratum",
        paper_tier=paper_tier,
        incomplete_fields=policy_contract_incomplete_fields,
    )
    policy_artifact_payload["target_priors_by_stratum_present"] = bool(artifact_target_priors_by_stratum_present)
    policy_artifact_payload["contract_status"] = contract_status(
        policy_contract_errors,
        policy_contract_incomplete_fields,
    )
    policy_artifact_payload["incomplete_fields"] = sorted(policy_contract_incomplete_fields)
    policy_artifact_payload["paper_tier_ready"] = policy_artifact_payload["contract_status"] == "ok"
    if args.save_policy:
        Path(args.save_policy).write_text(json.dumps(policy_artifact_payload, indent=2), encoding="utf-8")
        log(f"Saved policy thresholds to {args.save_policy}")
    (out_dir / "policy_thresholds.json").write_text(
        json.dumps(policy_artifact_payload, indent=2),
        encoding="utf-8",
    )
    figure_outputs = build_safety_policy_figures(
        out_dir=out_dir,
        scenario_trace_df=pd.DataFrame(scenario_trace_rows) if scenario_trace_rows else None,
        op_target_df=op_target_df,
        op_df=op_df,
        bootstrap_df=pd.DataFrame(bootstrap_rows) if bootstrap_rows else None,
    )
    source_capabilities = {
        str(name): snap.as_dict() for name, snap in (data_engine.source_capabilities or {}).items()
    }
    write_reprovenance_bundle(
        out_dir,
        cfg=cfg,
        config_path=args.config,
        artifact_manifest={
            "artifact_type": "safety_policy_eval",
            "pair_mode": getattr(cfg.intervention, "pair_mode", None),
            "pair_filter_stats": None,
            "shift_families": shift_types,
            "mixed_profiles": cfg.intervention.mixed_profiles,
            "source_capabilities": source_capabilities,
            "encoder_fingerprint": expected_calib_fp,
            "preprocess_signature": preprocess_signature(cfg),
            "figure_outputs": figure_outputs,
            "eraser_selection_mode": eraser_bundle_meta.get("kind", "single"),
            "eraser_shift_families": sorted(str(v) for v in eraser_collection.keys() if str(v) != "__default__"),
            "eraser_reference_shift_family": eraser_reference_shift_family,
            "head_sha256": sha256_file(Path(enc_cfg.head_path)) if enc_cfg.head_path and Path(enc_cfg.head_path).exists() else None,
            "realism_sha256": sha256_file(Path(cfg.intervention.realism_path))
            if cfg.intervention.realism_path is not None and Path(cfg.intervention.realism_path).exists()
            else None,
        },
        input_artifact_paths={
            "config": args.config,
            "eraser": args.eraser,
            "head": enc_cfg.head_path,
            "realism": cfg.intervention.realism_path,
        },
    )
    log(f"Saved safety outputs to {out_dir}")


if __name__ == "__main__":
    main()
