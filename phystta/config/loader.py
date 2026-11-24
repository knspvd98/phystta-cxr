from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, get_origin, get_args, Union, Sequence, List

from .schema import (
    ArchitectureReportingConfig,
    ArchitectureStudyConfig,
    ConfigBundle,
    DatasetSourceConfig,
    DataConfig,
    PreprocessConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    ShiftConfig,
    MetricsConfig,
    VizConfig,
    RuntimeConfig,
    MonitoringConfig,
    OutputConfig,
    LoggingConfig,
    ExperimentConfig,
    KaggleConfig,
    KaggleDatasetConfig,
    EncoderConfig,
    InputAdapterConfig,
    EncoderProfile,
    PairQualityRulesConfig,
    InterventionConfig,
    ErasureEvalConfig,
    SafetyPolicyConfig,
    ShiftIdConfig,
    VLMConfig,
    EraserConfig,
)


def _unwrap_optional(tp):
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_path_type(tp) -> bool:
    tp = _unwrap_optional(tp)
    return tp is Path


def _is_list_path_type(tp) -> bool:
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)
    if origin in (list, List, Sequence):
        args = get_args(tp)
        if len(args) == 1:
            return _is_path_type(args[0])
    return False


def _coerce_value(value: Any, tp) -> Any:
    if value is None:
        return None
    if _is_path_type(tp):
        return Path(value).expanduser()
    if _is_list_path_type(tp):
        return [Path(v).expanduser() for v in value]
    return value


def _coerce_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in values:
            out[f.name] = _coerce_value(values[f.name], f.type)
    return out


def _update_dataclass(dc, values: Dict[str, Any]):
    for f in fields(dc):
        if f.name not in values:
            continue
        v = values[f.name]
        if isinstance(getattr(dc, f.name), list) and f.name == "sources":
            sources = []
            for src in v:
                sources.append(DatasetSourceConfig(**_coerce_kwargs(DatasetSourceConfig, src)))
            setattr(dc, f.name, sources)
        else:
            setattr(dc, f.name, _coerce_value(v, f.type))
    return dc


def from_dict(cfg: Dict[str, Any]) -> ConfigBundle:
    bundle = ConfigBundle()
    if "tta" in cfg:
        raise ValueError("Legacy TTA config is not supported in 2026 pivot. Remove 'tta' from YAML.")
    if "model" in cfg:
        for k in cfg["model"].keys():
            if k.startswith("vae_"):
                raise ValueError(f"Legacy VAE field '{k}' is not supported in 2026 pivot.")
    if "train" in cfg and "use_augmix" in cfg["train"]:
        raise ValueError("Legacy train.use_augmix is not supported in 2026 pivot.")
    if "intervention" in cfg and isinstance(cfg["intervention"], dict):
        if "max_samples" in cfg["intervention"]:
            raise ValueError("intervention.max_samples is not supported; rename to intervention.max_images")
    if "data" in cfg:
        data_cfg = DataConfig()
        data_cfg = _update_dataclass(data_cfg, cfg["data"])
        if "preprocess" in cfg["data"]:
            data_cfg.preprocess = _update_dataclass(PreprocessConfig(), cfg["data"]["preprocess"])
        bundle.data = data_cfg
    if "model" in cfg and cfg["model"] is not None:
        model_cfg = _update_dataclass(ModelConfig(), cfg["model"])
        model_in = cfg["model"]
        if isinstance(model_in.get("encoder"), dict):
            model_cfg.encoder = _update_dataclass(EncoderConfig(), model_in["encoder"])
            if model_in["encoder"].get("input_adapter") is not None:
                model_cfg.encoder.input_adapter = _update_dataclass(
                    InputAdapterConfig(), model_in["encoder"]["input_adapter"]
                )
        profiles_in = model_in.get("encoder_profiles")
        if profiles_in is not None:
            profiles = []
            for p in profiles_in:
                enc_in = p.get("encoder", p)
                enc = _update_dataclass(EncoderConfig(), enc_in)
                if enc_in.get("input_adapter") is not None:
                    enc.input_adapter = _update_dataclass(InputAdapterConfig(), enc_in["input_adapter"])
                profiles.append(EncoderProfile(name=p.get("name", "profile"), encoder=enc))
            model_cfg.encoder_profiles = profiles
        bundle.model = model_cfg
    if "optim" in cfg:
        bundle.optim = _update_dataclass(OptimConfig(), cfg["optim"])
    if "train" in cfg:
        bundle.train = _update_dataclass(TrainConfig(), cfg["train"])
    if "shifts" in cfg:
        bundle.shifts = _update_dataclass(ShiftConfig(), cfg["shifts"])
    if "metrics" in cfg:
        bundle.metrics = _update_dataclass(MetricsConfig(), cfg["metrics"])
    if "shift_id" in cfg:
        bundle.shift_id = _update_dataclass(ShiftIdConfig(), cfg["shift_id"])
    if "safety_policy" in cfg:
        bundle.safety_policy = _update_dataclass(SafetyPolicyConfig(), cfg["safety_policy"])
    if "vlm" in cfg:
        bundle.vlm = _update_dataclass(VLMConfig(), cfg["vlm"])
    if "viz" in cfg:
        bundle.viz = _update_dataclass(VizConfig(), cfg["viz"])
    if "runtime" in cfg:
        bundle.runtime = _update_dataclass(RuntimeConfig(), cfg["runtime"])
    if "monitoring" in cfg:
        bundle.monitoring = _update_dataclass(MonitoringConfig(), cfg["monitoring"])
    if "output" in cfg:
        bundle.output = _update_dataclass(OutputConfig(), cfg["output"])
    if "logging" in cfg:
        bundle.logging = _update_dataclass(LoggingConfig(), cfg["logging"])
    if "experiment" in cfg:
        bundle.experiment = _update_dataclass(ExperimentConfig(), cfg["experiment"])
    if "architecture_study" in cfg:
        arch_cfg = _update_dataclass(ArchitectureStudyConfig(), cfg["architecture_study"])
        if "reporting" in cfg["architecture_study"]:
            arch_cfg.reporting = _update_dataclass(ArchitectureReportingConfig(), cfg["architecture_study"]["reporting"])
        bundle.architecture_study = arch_cfg
    if "kaggle" in cfg:
        kaggle_cfg = KaggleConfig()
        if "auto_download" in cfg["kaggle"]:
            kaggle_cfg.auto_download = bool(cfg["kaggle"]["auto_download"])
        datasets = []
        for ds in cfg["kaggle"].get("datasets", []):
            datasets.append(KaggleDatasetConfig(**_coerce_kwargs(KaggleDatasetConfig, ds)))
        kaggle_cfg.datasets = datasets
        bundle.kaggle = kaggle_cfg
    if "intervention" in cfg:
        bundle.intervention = _update_dataclass(InterventionConfig(), cfg["intervention"])
        if "pair_quality" in cfg["intervention"]:
            bundle.intervention.pair_quality = _update_dataclass(
                PairQualityRulesConfig(), cfg["intervention"]["pair_quality"]
            )
    if "eraser" in cfg:
        bundle.eraser = _update_dataclass(EraserConfig(), cfg["eraser"])
    if "erasure_eval" in cfg:
        bundle.erasure_eval = _update_dataclass(ErasureEvalConfig(), cfg["erasure_eval"])
    bundle.resolve()
    bundle.validate()
    return bundle


def load_config(path: str) -> ConfigBundle:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required to load YAML configs") from e
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    return from_dict(data)


def save_config(path: str, cfg: ConfigBundle) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.as_dict()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise RuntimeError("PyYAML is required to save YAML configs") from e
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        path.write_text(json.dumps(data, indent=2))
