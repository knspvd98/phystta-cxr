from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .index import DatasetIndexBuilder, LeakageDetector
from .capabilities import (
    capability_snapshot_from_frame,
    capability_snapshot_from_source_config,
    pair_mode_support_status,
    validate_declared_capabilities,
    validate_required_columns,
)
from .templates import resolve_dataset_source, expected_files_message
from .split import patient_level_split, patient_level_split_custom
from .datasets import XRayDataset
from .transforms import build_preprocess_transform
from ..config import DataConfig, KaggleConfig, DatasetSourceConfig, DatasetCapabilitySnapshot
from .kaggle import download_all
from ..utils import seed_worker


class DataEngine:
    PAIR_MODE_REQUIREMENTS = {
        "strict_device_change": ("supports_temporal_pairs", "supports_device_metadata"),
        "temporal_acquisition_context": (
            "supports_temporal_pairs",
            "supports_view_metadata",
            "supports_procedure_metadata",
        ),
        "temporal_same_view": (
            "supports_temporal_pairs",
            "supports_view_metadata",
            "supports_orientation_metadata",
            "supports_procedure_metadata",
        ),
    }

    def __init__(self, cfg: DataConfig, logger=None, kaggle_cfg: Optional[KaggleConfig] = None):
        self.cfg = cfg
        self.kaggle_cfg = kaggle_cfg
        self.logger = logger
        self.index: Optional[pd.DataFrame] = None
        self.splits: Dict[str, Dict[str, List[str]]] = {}
        self.label_cols: List[str] = []
        self.index_paths: Dict[str, Path] = {}
        self.source_map: Dict[str, DatasetSourceConfig] = {s.domain_label or s.name: s for s in cfg.sources}
        self.source_capabilities: Dict[str, DatasetCapabilitySnapshot] = {}
        self.source_columns: Dict[str, List[str]] = {}
        self._shift_id_cfg = None
        self._safety_policy_cfg = None
        self._erasure_eval_cfg = None
        self._intervention_cfg = None

    def _configured_label_cols(self) -> List[str]:
        if self.cfg.label_cols:
            return list(self.cfg.label_cols)
        return sorted({c for src in self.cfg.sources for c in (src.label_cols or [])})

    def attach_runtime_contract(self, cfg_bundle) -> "DataEngine":
        self._shift_id_cfg = getattr(cfg_bundle, "shift_id", None)
        self._safety_policy_cfg = getattr(cfg_bundle, "safety_policy", None)
        self._erasure_eval_cfg = getattr(cfg_bundle, "erasure_eval", None)
        self._intervention_cfg = getattr(cfg_bundle, "intervention", None)
        return self

    def _refresh_source_metadata(self) -> None:
        self.source_capabilities = {}
        self.source_columns = {}
        if self.index is None or "source" not in self.index.columns:
            return
        for source, df_src in self.index.groupby("source"):
            source_name = str(source)
            src_cfg = self.source_map.get(source_name)
            snapshot = capability_snapshot_from_frame(source_name, df_src, src_cfg)
            self.source_capabilities[source_name] = snapshot
            self.source_columns[source_name] = list(snapshot.available_columns)
            if src_cfg is not None:
                validate_declared_capabilities(source_name, src_cfg, snapshot)

    def _validate_runtime_semantics(self) -> None:
        if self.index is None:
            return
        eval_modes = set(self.cfg_split_eval_modes())
        fairness_enabled = "fairness" in eval_modes
        for source, df_src in self.index.groupby("source"):
            source_name = str(source)
            snapshot = self.source_capabilities.get(source_name)
            if snapshot is None:
                continue
            if hasattr(self.cfg, "split_group_key") and self.cfg.split_group_key:
                validate_required_columns(
                    df=df_src,
                    source=source_name,
                    keys=[self.cfg.split_group_key],
                    context="data.split_group_key",
                )
            shift_id_cfg = getattr(self, "_shift_id_cfg", None)
            if shift_id_cfg is not None and getattr(shift_id_cfg, "group_key", None):
                validate_required_columns(
                    df=df_src,
                    source=source_name,
                    keys=[str(shift_id_cfg.group_key)],
                    context="shift_id.group_key",
                )
            safety_cfg = getattr(self, "_safety_policy_cfg", None)
            if safety_cfg is not None and bool(getattr(safety_cfg, "stratify_technical", False)):
                validate_required_columns(
                    df=df_src,
                    source=source_name,
                    keys=[str(k) for k in getattr(safety_cfg, "strata_keys", [])],
                    context="safety_policy.strata_keys",
                )
            erasure_eval_cfg = getattr(self, "_erasure_eval_cfg", None)
            if fairness_enabled and erasure_eval_cfg is not None and getattr(erasure_eval_cfg, "fairness_groups", None):
                validate_required_columns(
                    df=df_src,
                    source=source_name,
                    keys=[str(k) for k in getattr(erasure_eval_cfg, "fairness_groups", [])],
                    context="erasure_eval.fairness_groups",
                )

    def cfg_split_eval_modes(self) -> List[str]:
        erasure_eval = getattr(self, "_erasure_eval_cfg", None)
        if erasure_eval is None:
            return []
        if getattr(erasure_eval, "modes", None):
            return list(erasure_eval.modes)
        if getattr(erasure_eval, "mode", None):
            return [str(erasure_eval.mode)]
        return []

    def build_index(self, force: bool = False) -> pd.DataFrame:
        if self.cfg.require_real_data:
            for src in self.cfg.sources:
                if getattr(src, "source_type", "cxr") == "lidc_drr":
                    raise RuntimeError("Synthetic DRR sources are запрещены for main pipeline. Set data.require_real_data=false only for research isolation.")
        split_group_col = self.cfg.split_group_key or "patient_id"
        if self.cfg.prebuilt_index_csv is not None:
            idx_path = Path(self.cfg.prebuilt_index_csv).expanduser()
            if not idx_path.exists():
                raise RuntimeError(
                    f"Prebuilt index not found: {idx_path}. "
                    "Run scripts/build_data_index.py first or provide valid data.prebuilt_index_csv."
                )
            if self.cfg.prebuilt_index_sha256:
                import hashlib
                digest = hashlib.sha256(idx_path.read_bytes()).hexdigest()
                if digest != self.cfg.prebuilt_index_sha256:
                    raise RuntimeError(
                        f"Prebuilt index hash mismatch for {idx_path}: expected {self.cfg.prebuilt_index_sha256}, got {digest}."
                    )
            index = pd.read_csv(idx_path)
            required_cols = {"image_path", "source", "patient_id"}
            missing = required_cols - set(index.columns)
            if missing:
                raise RuntimeError(f"Prebuilt index missing required columns {sorted(missing)}: {idx_path}")
            index = DatasetIndexBuilder.harmonize_frame(index)
            self.label_cols = self._configured_label_cols()
            if self.cfg.require_labels and not self.label_cols:
                raise RuntimeError("label_cols is empty. Provide explicit label columns in data.sources or data.label_cols.")
            for col in self.label_cols:
                if col not in index.columns:
                    raise RuntimeError(f"Label column missing in prebuilt index: {col}")
                index[col] = index[col].fillna(-1.0)
            if "global_patient_id" not in index.columns and {"source", "patient_id"}.issubset(index.columns):
                index["global_patient_id"] = index["source"].astype(str) + ":" + index["patient_id"].astype(str)
            if "global_image_id" not in index.columns:
                index["global_image_id"] = self._build_global_image_ids(index)
            filter_group_col = split_group_col if split_group_col in index.columns else "patient_id"
            if self.cfg.min_patient_images > 1:
                counts = index[filter_group_col].astype(str).value_counts()
                valid = counts[counts >= self.cfg.min_patient_images].index
                index = index[index[filter_group_col].astype(str).isin(valid)]
            if self.cfg.max_patient_images is not None:
                index = (
                    index.sort_values(["source", "image_path"], kind="stable")
                    .groupby(filter_group_col, group_keys=False)
                    .head(self.cfg.max_patient_images)
                    .reset_index(drop=True)
                )
            self.index = index.reset_index(drop=True)
            for s in set(self.index["source"].astype(str).tolist()):
                self.index_paths[s] = idx_path
            self._refresh_source_metadata()
            self._validate_runtime_semantics()
            return self.index
        # enforce explicit label whitelist
        for src in self.cfg.sources:
            if self.cfg.label_cols:
                src.label_cols = list(self.cfg.label_cols)
            if self.cfg.require_labels and (not src.label_cols):
                raise RuntimeError(f"{src.name}: label_cols must be explicitly provided to avoid label leakage.")
        if self.kaggle_cfg and self.kaggle_cfg.auto_download:
            download_all(self.kaggle_cfg, logger=self.logger)
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        builder = DatasetIndexBuilder(self.cfg)
        for src in self.cfg.sources:
            dataset_id = src.domain_label or src.name
            index_path = self.cfg.index_dir / f"{dataset_id}_index.csv"
            if index_path.exists() and not force:
                df_src = pd.read_csv(index_path)
                df_src = DatasetIndexBuilder.harmonize_frame(df_src)
            else:
                src_resolved = self._resolve_source(src)
                df_src = builder._build_source(src_resolved)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                df_src.to_csv(index_path, index=False)
                self.source_map[dataset_id] = src_resolved
            frames.append(df_src)
            self.index_paths[dataset_id] = index_path
        index = pd.concat(frames, ignore_index=True)
        self.label_cols = self._configured_label_cols()
        if self.cfg.require_labels and not self.label_cols:
            raise RuntimeError("label_cols is empty. Provide explicit label columns in data.sources or data.label_cols.")
        # ensure all label cols exist
        for col in self.label_cols:
            if col not in index.columns:
                raise RuntimeError(f"Label column missing in index: {col}")
        for col in self.label_cols:
            if col in index.columns:
                index[col] = index[col].fillna(-1.0)
        # global patient id to avoid cross-source collisions
        if "source" in index.columns and "patient_id" in index.columns:
            index["global_patient_id"] = index["source"].astype(str) + ":" + index["patient_id"].astype(str)
        if "global_image_id" not in index.columns:
            index["global_image_id"] = self._build_global_image_ids(index)
        filter_group_col = split_group_col if split_group_col in index.columns else "patient_id"
        if self.cfg.min_patient_images > 1:
            counts = index[filter_group_col].astype(str).value_counts()
            valid = counts[counts >= self.cfg.min_patient_images].index
            index = index[index[filter_group_col].astype(str).isin(valid)]
        if self.cfg.max_patient_images is not None:
            index = (
                index.sort_values(["source", "image_path"], kind="stable")
                .groupby(filter_group_col, group_keys=False)
                .head(self.cfg.max_patient_images)
                .reset_index(drop=True)
            )
        self.index = DatasetIndexBuilder.harmonize_frame(index.reset_index(drop=True))
        self._refresh_source_metadata()
        self._validate_runtime_semantics()
        return self.index

    @staticmethod
    def _build_global_image_ids(index: pd.DataFrame) -> pd.Series:
        source = index.get("source", pd.Series(["unknown"] * len(index))).astype(str)
        patient = index.get("patient_id", pd.Series(["unknown"] * len(index))).astype(str)
        study = index.get("study_id", pd.Series(["unknown"] * len(index))).astype(str)
        image_name = index.get("image_path", pd.Series(["unknown"] * len(index))).astype(str).map(lambda p: Path(p).name)
        return source + "|" + patient + "|" + study + "|" + image_name

    def _resolve_source(self, src: DatasetSourceConfig) -> DatasetSourceConfig:
        if src.metadata_csv is not None and Path(src.metadata_csv).exists():
            return src
        if not src.root.exists():
            raise FileNotFoundError(f"Dataset root not found for {src.name}: {src.root}. {expected_files_message(src.name)}")
        return resolve_dataset_source(src.name, str(src.root), label_cols=src.label_cols or None)

    def get_source_capability_snapshot(self, source: str) -> Optional[DatasetCapabilitySnapshot]:
        return self.source_capabilities.get(str(source))

    def dataset_summary(self) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        summaries = []
        for dataset_id, df in self.index.groupby("source"):
            src = self.source_map.get(dataset_id)
            caps = self.source_capabilities.get(str(dataset_id))
            summaries.append({
                "dataset_id": dataset_id,
                "root": str(src.root) if src else "",
                "index_path": str(self.index_paths.get(dataset_id, "")),
                "n_samples": int(len(df)),
                "n_patients": int(df["patient_id"].nunique()),
                "capabilities": caps.as_dict() if caps is not None else {},
            })
        return summaries

    def pair_metadata_status(
        self,
        pair_mode: Optional[str] = None,
        require_resolution: bool = False,
    ) -> List[Dict[str, Any]]:
        statuses = []
        if self.index is not None and self.source_capabilities:
            for dataset_id, snapshot in sorted(self.source_capabilities.items()):
                status = {
                    "source": str(dataset_id),
                    "supports_real_pairs": bool(snapshot.supports_temporal_pairs),
                    "supports_device_pairs": bool(
                        snapshot.supports_temporal_pairs and snapshot.supports_device_metadata
                    ),
                    "supports_context_pairs": bool(
                        snapshot.supports_temporal_pairs
                        and snapshot.supports_view_metadata
                        and snapshot.supports_procedure_metadata
                    ),
                    "capabilities": snapshot.as_dict(),
                }
                status.update(pair_mode_support_status(snapshot, pair_mode, require_resolution=require_resolution))
                statuses.append(status)
            return statuses

        for src in self.cfg.sources:
            dataset_id = src.domain_label or src.name
            snapshot = capability_snapshot_from_source_config(src)
            status = {
                "source": str(dataset_id),
                "supports_real_pairs": bool(snapshot.supports_temporal_pairs),
                "supports_device_pairs": bool(snapshot.supports_temporal_pairs and snapshot.supports_device_metadata),
                "supports_context_pairs": bool(
                    snapshot.supports_temporal_pairs and snapshot.supports_view_metadata and snapshot.supports_procedure_metadata
                ),
                "capabilities": snapshot.as_dict(),
            }
            status.update(pair_mode_support_status(snapshot, pair_mode, require_resolution=require_resolution))
            statuses.append(status)
        return statuses

    def create_splits(self) -> None:
        if self.index is None:
            raise RuntimeError("Index not built")
        split_group_key = self.cfg.split_group_key or "patient_id"
        if split_group_key not in self.index.columns:
            raise RuntimeError(
                f"Configured data.split_group_key='{split_group_key}' is missing from index. "
                f"Available columns: {sorted(self.index.columns)}"
            )
        self.splits = {}
        # guard: if custom split names are configured, require split_fractions
        default_names = {"train", "val", "test"}
        requested = {self.cfg.head_split, self.cfg.eraser_split, self.cfg.calib_split, self.cfg.eval_split}
        if getattr(self.cfg, "head_val_split", None):
            requested.add(str(self.cfg.head_val_split))
        if self.cfg.split_fractions is None and not requested.issubset(default_names):
            raise RuntimeError("Custom splits require data.split_fractions. Define fractions for train_head/train_eraser/calib/eval.")
        split_maps = {}
        for source, df_src in self.index.groupby("source"):
            if self.cfg.split_fractions:
                missing = requested - set(self.cfg.split_fractions.keys())
                if missing:
                    raise RuntimeError(f"split_fractions missing required splits: {missing}")
                ids = patient_level_split_custom(
                    df_src,
                    self.label_cols,
                    seed=self.cfg.split_seed,
                    fractions=self.cfg.split_fractions,
                    stratify=len(self.label_cols) > 0,
                    group_col=split_group_key,
                )
            else:
                ids = patient_level_split(
                    df_src,
                    self.label_cols,
                    seed=self.cfg.split_seed,
                    train_frac=self.cfg.train_frac,
                    val_frac=self.cfg.val_frac,
                    test_frac=self.cfg.test_frac,
                    stratify=True,
                    group_col=split_group_key,
                )
            self.splits[source] = ids
            split_map = {}
            for name, pid_list in ids.items():
                for pid in pid_list:
                    split_map[str(pid)] = name
            split_maps[source] = split_map
        # Apply splits to index
        split_col = []
        for _, row in self.index.iterrows():
            src = row["source"]
            pid = str(row[split_group_key])
            split = split_maps[src].get(pid)
            if split is None:
                # fallback to last split name
                split = list(self.splits[src].keys())[-1]
            split_col.append(split)
        self.index = self.index.copy()
        self.index["split"] = split_col
        # Leakage check
        for src, ids in self.splits.items():
            split_names = list(ids.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    a = set(str(v) for v in ids[split_names[i]])
                    b = set(str(v) for v in ids[split_names[j]])
                    if a & b:
                        raise ValueError(
                            f"Group leakage detected between splits {split_names[i]} and {split_names[j]} "
                            f"for source={src} using key={split_group_key}"
                        )

        # Global leakage check across sources for configured split group key.
        if not (
            split_group_key == "patient_id"
            and len(self.cfg.sources) > 1
            and bool(getattr(self.cfg, "single_source_mode", False))
        ):
            global_groups: Dict[str, set[str]] = {}
            for split_name, group_df in self.index.groupby("split"):
                global_groups[split_name] = set(group_df[split_group_key].astype(str).tolist())
            split_names = sorted(global_groups.keys())
            for i in range(len(split_names)):
                for j in range(i + 1, len(split_names)):
                    a = global_groups[split_names[i]]
                    b = global_groups[split_names[j]]
                    if a & b:
                        raise ValueError(
                            f"Global group leakage detected between splits {split_names[i]} and {split_names[j]} "
                            f"using key={split_group_key}"
                        )

        # Save split manifest for reproducibility
        import json, hashlib
        manifest = {
            "split_seed": self.cfg.split_seed,
            "split_group_key": split_group_key,
            "sources": self.splits,
        }
        payload = json.dumps(manifest, sort_keys=True)
        manifest["hash"] = hashlib.sha256(payload.encode()).hexdigest()
        manifest_path = self.cfg.index_dir / "split_manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception as e:
            raise RuntimeError(f"Failed to write split manifest to {manifest_path}") from e

    def get_split_df(self, split: str, sources: Optional[List[str]] = None) -> pd.DataFrame:
        if self.index is None or "split" not in self.index.columns:
            raise RuntimeError("Splits not created")
        df = self.index[self.index["split"] == split]
        if sources:
            df = df[df["source"].isin(sources)]
        return df.reset_index(drop=True)

    def build_dataloaders(self, batch_size: int, num_workers: int = 4, splits: Optional[List[str]] = None, sources_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, DataLoader]:
        if self.index is None:
            raise RuntimeError("Index not built")
        if "split" not in self.index.columns:
            self.create_splits()
        train_sources = self.cfg.train_sources
        val_sources = self.cfg.val_sources
        test_sources = self.cfg.test_sources
        base_transform = build_preprocess_transform(self.cfg.preprocess)
        loaders = {}
        if splits is None:
            if self.cfg.split_fractions:
                splits = list(self.cfg.split_fractions.keys())
            else:
                splits = ["train", "val", "test"]
        for split in splits:
            if sources_map and split in sources_map:
                sources = sources_map[split]
            elif split == "train":
                sources = train_sources
            elif split == "val":
                sources = val_sources
            elif split == "test":
                sources = test_sources
            else:
                sources = None
            df_split = self.get_split_df(split, sources)
            if df_split.empty:
                raise RuntimeError(f"Split '{split}' is empty. Check split_fractions and dataset coverage.")
            transform = base_transform
            dataset = XRayDataset(df_split, self.label_cols, transforms=transform, model_representation=self.cfg.model_representation, preprocess_cfg=self.cfg.preprocess)
            gen = torch.Generator()
            # Keep per-split shuffling deterministic across repeated runs.
            gen.manual_seed(int(self.cfg.split_seed) + (abs(hash(str(split))) % (2**16)))
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split.startswith("train"),
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=gen,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=4 if num_workers > 0 else None,
                drop_last=False,
            )
        return loaders

    def build_paired_index(
        self,
        source: str,
        max_hours: float = 24.0,
        group_key: Optional[str] = None,
        pair_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        if self.index is None:
            raise RuntimeError("Index not built")
        df = self.index[self.index["source"] == source].copy()
        if df.empty:
            raise RuntimeError(f"No rows found in index for source={source}")
        from ..causal.objectives import build_within_patient_pairs

        selected_mode = pair_mode or str(getattr(self._intervention_cfg, "pair_mode", "")) or "temporal_same_view"
        snapshot = self.source_capabilities.get(str(source))
        if (pair_mode is not None or self._intervention_cfg is not None) and selected_mode in self.PAIR_MODE_REQUIREMENTS and snapshot is not None:
            missing_caps = [
                cap for cap in self.PAIR_MODE_REQUIREMENTS[selected_mode]
                if not bool(getattr(snapshot, cap, False))
            ]
            if missing_caps:
                raise RuntimeError(
                    f"{source}: pair_mode='{selected_mode}' is unsupported for this source. "
                    f"Missing capabilities: {missing_caps}. Snapshot: {snapshot.as_dict()}"
                )
        out = build_within_patient_pairs(
            df,
            patient_col="patient_id",
            time_col="study_datetime",
            source_col="source",
            pair_mode=selected_mode,
            max_hours=max_hours,
        )
        if out.empty:
            return out
        if group_key is None or group_key in out.columns:
            return out
        if group_key == "patient_id":
            return out
        if group_key == "global_patient_id":
            if "source_a" in out.columns and "patient_id" in out.columns:
                out["global_patient_id"] = out["source_a"].astype(str) + ":" + out["patient_id"].astype(str)
                return out
            raise RuntimeError("Paired index cannot construct global_patient_id: missing source_a/patient_id columns.")
        raise RuntimeError(
            f"Paired index does not contain requested group_key='{group_key}'. "
            f"Available columns: {sorted(out.columns)}"
        )
