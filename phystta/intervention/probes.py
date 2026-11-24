from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils import hash_code, sample_ids_from_batch, load_torch_artifact


class ProbeWriter:
    def __init__(self, out_path: str, chunk_size: int = 50000, token_budget: Optional[int] = None):
        self.out_path = Path(out_path)
        self.chunk_size = int(chunk_size)
        self.token_budget = token_budget
        self.chunk_dir = self.out_path.with_suffix(".chunks")
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_paths: List[str] = []
        self.buffer: Dict[str, List] = {}
        self.buffer_count = 0
        self.total_count = 0
        self.chunk_idx = 0

    def _append_buffer(self, key: str, value):
        if key not in self.buffer:
            self.buffer[key] = []
        self.buffer[key].append(value)

    def _concat(self, items):
        if not items:
            return None
        first = items[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(items, dim=0)
        if isinstance(first, np.ndarray):
            return np.concatenate(items, axis=0)
        if isinstance(first, list):
            out = []
            for v in items:
                out.extend(v)
            return np.array(out)
        return np.array(items)

    def _flush(self):
        if self.buffer_count == 0:
            return
        out = {k: self._concat(v) for k, v in self.buffer.items()}
        validate_probe_payload(out)
        chunk_path = self.chunk_dir / f"chunk_{self.chunk_idx:04d}.pt"
        torch.save({"probes": out}, chunk_path)
        self.chunk_paths.append(str(chunk_path))
        self.chunk_idx += 1
        self.buffer = {}
        self.buffer_count = 0

    def append(self, probes: Dict[str, torch.Tensor]) -> bool:
        n = int(probes["features_shifted"].shape[0])
        if self.token_budget is not None and self.total_count >= self.token_budget:
            return True
        if self.token_budget is not None and self.total_count + n > self.token_budget:
            keep = self.token_budget - self.total_count
            if keep <= 0:
                return True
            # trim
            for k, v in probes.items():
                if isinstance(v, torch.Tensor):
                    probes[k] = v[:keep]
                elif isinstance(v, np.ndarray):
                    probes[k] = v[:keep]
                elif isinstance(v, list):
                    probes[k] = v[:keep]
            n = int(keep)
        for k, v in probes.items():
            self._append_buffer(k, v)
        self.buffer_count += n
        self.total_count += n
        if self.buffer_count >= self.chunk_size:
            self._flush()
        return self.token_budget is not None and self.total_count >= self.token_budget

    def finalize(self, meta: Dict[str, str]) -> Dict[str, str]:
        self._flush()
        manifest = {
            "probes_chunks": [str(Path(p).name) for p in self.chunk_paths],
            "chunk_dir": str(self.chunk_dir),
            "meta": meta,
            "chunked": True,
            "n_samples": int(self.total_count),
        }
        torch.save(manifest, self.out_path)
        return manifest


def _field_length(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if isinstance(value, np.ndarray):
        return int(value.shape[0])
    if isinstance(value, list):
        return len(value)
    try:
        return len(value)
    except Exception:
        return None


def probe_length_report(probes: Dict[str, object]) -> Dict[str, object]:
    lengths = {
        key: _field_length(value)
        for key, value in probes.items()
        if key
        in {
            "features_clean",
            "features_shifted",
            "patient_id",
            "group_id",
            "study_id",
            "shift_type",
            "profile",
            "intensity",
            "labels",
            "image_id",
        }
    }
    present_lengths = {k: v for k, v in lengths.items() if v is not None}
    unique = sorted(set(present_lengths.values()))
    feature_dim = None
    if isinstance(probes.get("features_clean"), torch.Tensor) and probes["features_clean"].ndim >= 2:
        feature_dim = int(probes["features_clean"].shape[-1])
    return {
        "n_samples": int(lengths.get("features_clean") or lengths.get("features_shifted") or 0),
        "lengths_by_field": present_lengths,
        "consistency_ok": len(unique) <= 1,
        "feature_dim": feature_dim,
    }


def validate_probe_payload(probes: Dict[str, object]) -> Dict[str, object]:
    report = probe_length_report(probes)
    if not report["consistency_ok"]:
        raise RuntimeError(
            f"Probe artifact length mismatch; refusing to save inconsistent payload. lengths_by_field={report['lengths_by_field']}"
        )
    return report

def _git_hash() -> str:
    import subprocess
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        return out.decode().strip()
    except Exception:
        return "unknown"


def collect_probes(
    loader,
    shift_manager,
    shift_type: str,
    intensity_grid: List[float],
    encoder,
    max_images: Optional[int] = None,
    device: str = "cpu",
    seed: int = 2024,
    mixed_profiles: Optional[List[Dict[str, float]]] = None,
    token_mode: str = "none",
    token_layers: int = 1,
    token_sampling: str = "all",
    token_sample_per_image: Optional[int] = None,
    token_budget: Optional[int] = None,
    writer: Optional[ProbeWriter] = None,
    group_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    encoder.eval()
    features_clean = []
    features_shifted = []
    intensities = []
    patient_ids = []
    group_ids = []
    study_ids = []
    labels = []
    image_ids = []
    profiles = []
    shift_types_out = []
    n_images = 0
    torch.manual_seed(seed)

    def _select_tokens(tokens: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if tokens.dim() != 3:
            raise RuntimeError("encode_tokens must return [N,T,D]")
        if token_mode == "cls":
            return tokens[:, 0], 1
        if token_mode == "mean":
            return tokens.mean(dim=1), 1
        if token_mode in {"patch", "all"}:
            tok = tokens[:, 1:] if token_mode == "patch" and tokens.shape[1] > 1 else tokens
            n_tokens = tok.shape[1]
            if token_sample_per_image is not None and token_sample_per_image < n_tokens:
                if token_sampling == "random":
                    idx = torch.randperm(n_tokens, device=tok.device)[:token_sample_per_image]
                    tok = tok[:, idx]
                elif token_sampling == "topk_norm":
                    norms = torch.norm(tok, dim=-1)
                    idx = torch.topk(norms, k=token_sample_per_image, dim=1).indices
                    tok = torch.gather(tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
                else:
                    tok = tok[:, :token_sample_per_image]
                n_tokens = tok.shape[1]
            return tok.reshape(-1, tok.shape[-1]), n_tokens
        raise RuntimeError(f"Unsupported token_mode: {token_mode}")

    def _repeat_list(vals, repeat: int):
        out = []
        for v in vals:
            out.extend([v] * repeat)
        return out

    def _repeat_labels(lbl: torch.Tensor, repeat: int) -> torch.Tensor:
        return lbl.repeat_interleave(repeat, dim=0)

    def _trim_batch(batch_obj, keep: int):
        trimmed = {}
        base_len = None
        try:
            if "image" in batch_obj and hasattr(batch_obj["image"], "shape"):
                base_len = int(batch_obj["image"].shape[0])
        except Exception:
            base_len = None

        for k, v in batch_obj.items():
            try:
                if isinstance(v, torch.Tensor):
                    trimmed[k] = v[:keep]
                elif isinstance(v, np.ndarray):
                    trimmed[k] = v[:keep]
                elif isinstance(v, list):
                    trimmed[k] = v[:keep]
                elif isinstance(v, tuple):
                    trimmed[k] = v[:keep]
                elif (
                    base_len is not None
                    and not isinstance(v, (str, bytes))
                    and hasattr(v, "__len__")
                    and len(v) == base_len
                ):
                    trimmed[k] = v[:keep]
                else:
                    trimmed[k] = v
            except Exception:
                trimmed[k] = v
        return trimmed

    def _group_values(batch_obj, n: int) -> List[str]:
        if group_key and group_key in batch_obj:
            vals = batch_obj[group_key]
        elif "global_patient_id" in batch_obj:
            vals = batch_obj["global_patient_id"]
        elif "patient_id" in batch_obj:
            vals = batch_obj["patient_id"]
        elif "study_id" in batch_obj:
            vals = batch_obj["study_id"]
        else:
            vals = ["unknown"] * n
        return [str(v) for v in vals]

    use_tokens = token_mode != "none"
    if use_tokens and not hasattr(encoder, "encode_tokens"):
        raise RuntimeError("token_mode requires encoder.encode_tokens support.")

    with torch.no_grad():
        for batch in loader:
            if max_images is not None:
                remaining = int(max_images - n_images)
                if remaining <= 0:
                    break
                try:
                    batch_n = int(batch["image"].shape[0])
                except Exception:
                    batch_n = None
                if batch_n is not None and batch_n > remaining:
                    batch = _trim_batch(batch, remaining)

            x = batch["image"].to(device)
            if max_images is not None:
                remaining = max_images - n_images
                if remaining <= 0:
                    break
                if x.shape[0] > remaining:
                    x = x[:remaining]
                    for key in ["patient_id", "study_id", "labels"]:
                        if key in batch and batch[key] is not None:
                            batch[key] = batch[key][:remaining]
            if use_tokens:
                tokens_clean = encoder.encode_tokens(x, token_layers=token_layers)
                z_clean, rep = _select_tokens(tokens_clean)
            else:
                z_clean = encoder.encode(x)
                rep = 1
            if shift_type == "mixed":
                if not mixed_profiles:
                    raise RuntimeError("mixed shift requested but mixed_profiles not provided")
                for idx, profile in enumerate(mixed_profiles):
                    name = profile.get("name", f"mixed_{idx}")
                    if "intensities" in profile:
                        intens = profile["intensities"]
                    elif "components" in profile:
                        intens = profile["components"]
                    else:
                        intens = {k: v for k, v in profile.items() if k != "name"}
                    sample_ids = sample_ids_from_batch(batch)
                    x_shifted, _meta = shift_manager.apply_batch_scenario(x, intens, sample_ids=sample_ids)
                    if use_tokens:
                        tokens_shifted = encoder.encode_tokens(x_shifted, token_layers=token_layers)
                        z_shifted, rep_shift = _select_tokens(tokens_shifted)
                    else:
                        z_shifted = encoder.encode(x_shifted)
                        rep_shift = 1
                    chunk = {
                        "features_clean": z_clean.detach().cpu(),
                        "features_shifted": z_shifted.detach().cpu(),
                        "intensity": torch.full((z_shifted.shape[0],), 1.0),
                        "shift_type": np.array(["mixed"] * z_shifted.shape[0]),
                        "patient_id": np.array(_repeat_list(_group_values(batch, x.shape[0]), rep_shift)),
                        "group_id": np.array(_repeat_list(_group_values(batch, x.shape[0]), rep_shift)),
                        "study_id": np.array(_repeat_list(batch.get("study_id", ["unknown"] * x.shape[0]), rep_shift)),
                        "image_id": np.array(_repeat_list(sample_ids if sample_ids is not None else ["unknown"] * x.shape[0], rep_shift)),
                        "profile": np.array([name] * z_shifted.shape[0]),
                        "labels": _repeat_labels(batch["labels"].detach().cpu(), rep_shift) if "labels" in batch else None,
                    }
                    if writer:
                        stop = writer.append(chunk)
                        if stop:
                            return {"count": writer.total_count}
                    else:
                        features_clean.append(chunk["features_clean"])
                        features_shifted.append(chunk["features_shifted"])
                        intensities.append(chunk["intensity"])
                        profiles.extend(chunk["profile"].tolist())
                        shift_types_out.extend(chunk["shift_type"].tolist())
                        patient_ids.extend(chunk["patient_id"].tolist())
                        group_ids.extend(chunk["group_id"].tolist())
                        study_ids.extend(chunk["study_id"].tolist())
                        image_ids.extend(chunk["image_id"].tolist())
                        if chunk["labels"] is not None:
                            labels.append(chunk["labels"])
            else:
                for intensity in intensity_grid:
                    sample_ids = sample_ids_from_batch(batch)
                    x_shifted = shift_manager.apply_batch(x, shift_type, float(intensity), sample_ids=sample_ids)
                    if use_tokens:
                        tokens_shifted = encoder.encode_tokens(x_shifted, token_layers=token_layers)
                        z_shifted, rep_shift = _select_tokens(tokens_shifted)
                    else:
                        z_shifted = encoder.encode(x_shifted)
                        rep_shift = 1
                    chunk = {
                        "features_clean": z_clean.detach().cpu(),
                        "features_shifted": z_shifted.detach().cpu(),
                        "intensity": torch.full((z_shifted.shape[0],), float(intensity)),
                        "shift_type": np.array([shift_type] * z_shifted.shape[0]),
                        "patient_id": np.array(_repeat_list(_group_values(batch, x.shape[0]), rep_shift)),
                        "group_id": np.array(_repeat_list(_group_values(batch, x.shape[0]), rep_shift)),
                        "study_id": np.array(_repeat_list(batch.get("study_id", ["unknown"] * x.shape[0]), rep_shift)),
                        "image_id": np.array(_repeat_list(sample_ids if sample_ids is not None else ["unknown"] * x.shape[0], rep_shift)),
                        "profile": np.array([shift_type] * z_shifted.shape[0]),
                        "labels": _repeat_labels(batch["labels"].detach().cpu(), rep_shift) if "labels" in batch else None,
                    }
                    if writer:
                        stop = writer.append(chunk)
                        if stop:
                            return {"count": writer.total_count}
                    else:
                        features_clean.append(chunk["features_clean"])
                        features_shifted.append(chunk["features_shifted"])
                        intensities.append(chunk["intensity"])
                        profiles.extend(chunk["profile"].tolist())
                        shift_types_out.extend(chunk["shift_type"].tolist())
                        patient_ids.extend(chunk["patient_id"].tolist())
                        group_ids.extend(chunk["group_id"].tolist())
                        study_ids.extend(chunk["study_id"].tolist())
                        image_ids.extend(chunk["image_id"].tolist())
                        if chunk["labels"] is not None:
                            labels.append(chunk["labels"])
            n_images += x.shape[0]
            if max_images is not None and n_images >= max_images:
                break
    if writer:
        return {"count": writer.total_count}
    features_clean = torch.cat(features_clean, dim=0)
    features_shifted = torch.cat(features_shifted, dim=0)
    intensities = torch.cat(intensities, dim=0)
    labels_t = torch.cat(labels, dim=0) if labels else None
    return {
        "features_clean": features_clean,
        "features_shifted": features_shifted,
        "intensity": intensities,
        "shift_type": np.array(shift_types_out),
        "patient_id": np.array(patient_ids),
        "group_id": np.array(group_ids) if len(group_ids) > 0 else np.array(patient_ids),
        "study_id": np.array(study_ids),
        "image_id": np.array(image_ids),
        "profile": np.array(profiles),
        "labels": labels_t,
    }


def save_probes(path: str, probes: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    report = validate_probe_payload(probes)
    out = {
        "probes": probes,
        "meta": {
            **meta,
            "n_samples": str(report["n_samples"]),
            "lengths_by_field": str(report["lengths_by_field"]),
            "consistency_ok": str(report["consistency_ok"]),
            "feature_dim": str(report["feature_dim"]),
        },
    }
    torch.save(out, path)


def load_probes(path: str) -> Dict[str, torch.Tensor]:
    payload = load_torch_artifact(path, map_location="cpu")
    if "probes_chunks" in payload:
        chunk_dir = Path(payload.get("chunk_dir", Path(path).with_suffix(".chunks")))
        chunks = payload["probes_chunks"]
        feats_clean = []
        feats_shift = []
        intensity = []
        patient_id = []
        group_id = []
        study_id = []
        image_id = []
        profile = []
        shift_type = []
        labels = []
        for name in chunks:
            chunk_path = chunk_dir / name
            data = load_torch_artifact(chunk_path, map_location="cpu")["probes"]
            feats_clean.append(data["features_clean"])
            feats_shift.append(data["features_shifted"])
            intensity.append(data["intensity"])
            if "shift_type" in data:
                shift_type.append(data["shift_type"])
            patient_id.append(data["patient_id"])
            if "group_id" in data:
                group_id.append(data["group_id"])
            study_id.append(data["study_id"])
            image_id.append(data["image_id"])
            profile.append(data["profile"])
            if data.get("labels") is not None:
                labels.append(data["labels"])
        out = {
            "features_clean": torch.cat(feats_clean, dim=0),
            "features_shifted": torch.cat(feats_shift, dim=0),
            "intensity": torch.cat(intensity, dim=0),
            "shift_type": np.concatenate(shift_type, axis=0) if shift_type else None,
            "patient_id": np.concatenate(patient_id, axis=0),
            "group_id": np.concatenate(group_id, axis=0) if group_id else np.concatenate(patient_id, axis=0),
            "study_id": np.concatenate(study_id, axis=0),
            "image_id": np.concatenate(image_id, axis=0),
            "profile": np.concatenate(profile, axis=0),
            "labels": torch.cat(labels, dim=0) if labels else None,
        }
        validate_probe_payload(out)
        return out
    validate_probe_payload(payload["probes"])
    return payload["probes"]


def build_meta(cfg, split: str, shift_type: str, intensity_grid: List[float], encoder_name: str, count: int, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    meta = {
        "dataset": ",".join(cfg.data.train_sources or []),
        "split": split,
        "shift_type": shift_type,
        "intensity_grid": str(intensity_grid),
        "encoder": encoder_name,
        "count": str(count),
        "seed": str(cfg.runtime.seed),
        "code_hash": hash_code(Path(__file__).resolve().parents[1]),
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label_cols": ",".join(cfg.data.sources[0].label_cols) if cfg.data.sources else "",
    }
    if extra:
        meta.update(extra)
    return meta
