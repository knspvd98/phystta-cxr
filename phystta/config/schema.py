from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DatasetSourceConfig:
    name: str
    root: Path
    source_type: str = "cxr"  # cxr | lidc_drr
    metadata_csv: Optional[Path] = None
    extra_metadata_csv: Optional[Path] = None
    extra_metadata_join_cols: Optional[List[str]] = None
    image_root: Optional[Path] = None
    path_col: str = "path"
    patient_id_col: str = "patient_id"
    study_id_col: Optional[str] = None
    view_col: Optional[str] = None
    sex_col: Optional[str] = None
    age_col: Optional[str] = None
    race_col: Optional[str] = None
    ethnicity_col: Optional[str] = None
    label_cols: List[str] = field(default_factory=list)
    labels_from_findings_col: Optional[str] = None
    label_delim: str = "|"
    allowed_views: Optional[List[str]] = None
    require_exists: bool = True
    domain_label: Optional[str] = None
    patient_id_from_path_regex: Optional[str] = None
    study_id_from_path_regex: Optional[str] = None
    manufacturer_col: Optional[str] = None
    station_name_col: Optional[str] = None
    study_time_col: Optional[str] = None
    study_date_col: Optional[str] = None
    study_datetime_col: Optional[str] = None
    dicom_id_col: Optional[str] = None
    view_family_col: Optional[str] = None
    performed_procedure_step_description_col: Optional[str] = None
    procedure_code_meaning_col: Optional[str] = None
    view_code_meaning_col: Optional[str] = None
    patient_orientation_code_meaning_col: Optional[str] = None
    rows_col: Optional[str] = None
    columns_col: Optional[str] = None
    boxes_csv: Optional[Path] = None
    supports_temporal_pairs: Optional[bool] = None
    supports_view_metadata: Optional[bool] = None
    supports_demographics: Optional[bool] = None
    supports_orientation_metadata: Optional[bool] = None
    supports_resolution_metadata: Optional[bool] = None
    supports_boxes: Optional[bool] = None
    supports_device_metadata: Optional[bool] = None
    supports_procedure_metadata: Optional[bool] = None

    def resolve(self) -> "DatasetSourceConfig":
        self.root = Path(self.root).expanduser()
        if self.metadata_csv is not None:
            self.metadata_csv = Path(self.metadata_csv).expanduser()
        if self.extra_metadata_csv is not None:
            self.extra_metadata_csv = Path(self.extra_metadata_csv).expanduser()
        if self.boxes_csv is not None:
            self.boxes_csv = Path(self.boxes_csv).expanduser()
        if self.image_root is None:
            self.image_root = self.root
        else:
            self.image_root = Path(self.image_root).expanduser()
        if self.domain_label is None:
            self.domain_label = self.name
        return self


@dataclass
class DatasetCapabilitySnapshot:
    source: str
    supports_temporal_pairs: bool = False
    supports_view_metadata: bool = False
    supports_demographics: bool = False
    supports_orientation_metadata: bool = False
    supports_resolution_metadata: bool = False
    supports_boxes: bool = False
    supports_device_metadata: bool = False
    supports_procedure_metadata: bool = False
    available_columns: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessConfig:
    image_size: int = 224
    center_crop: Optional[int] = None
    normalize_mean: float = 0.0
    normalize_std: float = 1.0
    grayscale: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0
    equalize_hist: bool = False
    dataset_output: str = "0_1"  # 0_1 only; no normalize at dataset level
    fixed_min: Optional[float] = None
    fixed_max: Optional[float] = None
    dicom_max: float = 4095.0
    mu_max: float = 6.0


@dataclass
class DRRConfig:
    energies_keV: List[float] = field(default_factory=lambda: [30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    spectrum_weights: Optional[List[float]] = None
    kvp: int = 120
    sid_mm: float = 1200.0
    sad_mm: float = 1000.0
    detector_pixels: int = 224
    pixel_spacing_mm: float = 1.0
    scatter_model: str = "none"
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    input_representation: str = "mu"  # mu | intensity
    output_representation: str = "mu"  # mu | intensity


@dataclass
class DataConfig:
    sources: List[DatasetSourceConfig] = field(default_factory=list)
    label_cols: Optional[List[str]] = None
    split_seed: int = 2024
    split_group_key: str = "patient_id"
    single_source_mode: bool = False
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    min_patient_images: int = 1
    max_patient_images: Optional[int] = None
    cache_dir: Path = Path("./cache")
    index_dir: Path = Path("./data_indexes")
    prebuilt_index_csv: Optional[Path] = None
    prebuilt_index_sha256: Optional[str] = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    require_real_data: bool = True
    train_sources: Optional[List[str]] = None
    val_sources: Optional[List[str]] = None
    test_sources: Optional[List[str]] = None
    drr: Optional[DRRConfig] = None
    model_representation: str = "intensity"  # intensity | mu
    split_fractions: Optional[Dict[str, float]] = None
    head_split: str = "train_head"
    head_val_split: Optional[str] = None
    eraser_split: str = "train_eraser"
    calib_split: str = "calib"
    eval_split: str = "eval"
    require_labels: bool = True

    def resolve(self) -> "DataConfig":
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.index_dir = Path(self.index_dir).expanduser()
        if self.prebuilt_index_csv is not None:
            self.prebuilt_index_csv = Path(self.prebuilt_index_csv).expanduser()
        self.sources = [src.resolve() for src in self.sources]
        if self.drr is None:
            self.drr = DRRConfig()
        return self


@dataclass
class ModelConfig:
    architecture: str = "densenet121"
    num_classes: int = 14
    in_chans: int = 1
    use_medical_pretrain: bool = True
    dropout: float = 0.0
    head: str = "linear"  # linear | evidential
    checkpoint_path: Optional[Path] = None
    ssl_checkpoint: Optional[Path] = None
    family: str = "cnn"  # cnn | transformer | ssl | evidential
    encoder: "EncoderConfig" = field(default_factory=lambda: EncoderConfig())
    encoder_profiles: Optional[List["EncoderProfile"]] = None


@dataclass
class EncoderConfig:
    encoder_source: str = "baseline_cnn"
    encoder_name: Optional[str] = None
    encoder_path: Optional[Path] = None
    revision: Optional[str] = None
    pretrained: Optional[bool] = None
    head_path: Optional[Path] = None
    head_meta_path: Optional[Path] = None
    head_attr: Optional[str] = None
    architecture: Optional[str] = None
    model_id: Optional[str] = None
    input_adapter: Optional["InputAdapterConfig"] = None


@dataclass
class InputAdapterConfig:
    repeat_to_3: bool = True
    image_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    std: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class EncoderProfile:
    name: str
    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class PairQualityRulesConfig:
    max_delta_hours: float = 24.0
    label_change_weight: float = 0.45
    view_change_weight: float = 0.20
    context_change_weight: float = 0.20
    sentinel_change_weight: float = 0.15
    procedure_change_weight: float = 0.15
    orientation_change_weight: float = 0.15
    resolution_jump_weight: float = 0.10
    support_device_change_weight: float = 0.10
    technical_dominant_max_risk: float = 0.25
    same_view_context_shift_max_risk: float = 0.45
    clinical_change_high_risk: float = 0.65
    resolution_ratio_tolerance: float = 0.20


@dataclass
class InterventionConfig:
    shift_types: List[str] = field(default_factory=lambda: ["heel", "scatter", "exposure", "poisson"])
    intensity_grid: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6])
    mixed_profiles: Optional[List[Dict[str, float]]] = None
    use_realism: bool = False
    realism_path: Optional[Path] = None
    realism_levels: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    max_images: int = 500
    k: int = 8
    fit_method: str = "probe_xfit"  # svd | leace | probe_clf | probe_xfit
    probe_target: str = "presence"  # presence | intensity
    group_key: str = "global_patient_id"
    anchor_real: bool = False
    min_overlap: float = 0.7
    causal_check: bool = False
    ate_threshold: float = 0.05
    positivity_min_samples: int = 50
    real_pairs_max_hours: float = 24.0
    pair_quality: PairQualityRulesConfig = field(default_factory=PairQualityRulesConfig)
    pair_mode: str = "temporal_same_view"  # strict_device_change | temporal_acquisition_context | temporal_same_view
    token_budget: Optional[int] = None
    probe_chunk_size: int = 50000
    token_sampling: str = "all"  # all | random | topk_norm
    token_sample_per_image: Optional[int] = None


@dataclass
class EraserConfig:
    type: str = "steering"  # linear | kernel | steering
    task_preserve: str = "none"  # none | head_subspace | residualize
    k: int = 8
    fit_mode: str = "one_eraser_per_shift_family"  # single | one_eraser_per_shift_family
    task_subspace_k: Optional[int] = None
    critical_labels: Optional[List[str]] = None
    rff_dim: int = 512
    rff_sigma: Optional[float] = None
    ridge_lambda: float = 1e-3
    steering_k: int = 8
    steering_hidden: int = 256
    steering_steps: int = 200
    steering_lr: float = 1e-3
    steering_eps: float = 0.01
    token_mode: str = "none"  # none | cls | patch | all
    token_layers: int = 1
    online_enabled: bool = False
    online_mode: str = "mixture"  # mixture | fast
    online_bank: Optional[List[Path]] = None
    online_weights_path: Optional[Path] = None
    online_hidden: int = 32
    online_ridge_lambda: float = 1e-2
    online_guard_eps: float = 0.01
    online_scores: str = "mmd"  # mmd | energy | c2st
    online_entropy_steps: int = 8
    online_entropy_lr: float = 1e-2
    logit_recovery_weight: float = 0.0
    prob_recovery_weight: float = 0.0
    clean_identity_weight: float = 0.0


@dataclass
class ErasureEvalConfig:
    shift_types: List[str] = field(default_factory=lambda: ["heel"])
    intensity_grid: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6])
    mode: str = "all"  # probing | clinical | safety | all
    modes: Optional[List[str]] = None
    scenario_mode: str = "atomic_calibrated_eval"  # atomic_calibrated_eval | mixed_stress_eval
    coverage: float = 0.8
    clinical_classes: Optional[List[str]] = None
    split: str = "test"
    fairness_groups: List[str] = field(default_factory=list)
    age_bins: List[int] = field(default_factory=lambda: [40, 60])
    bootstrap_samples: int = 200
    tradeoff_strengths: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    guarded_probe_strata: List[str] = field(default_factory=lambda: ["view_family", "procedure_context", "orientation_context"])
    guarded_probe_min_samples: int = 24
    stability_resamples: int = 12


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Sequence[float] = (0.9, 0.999)


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    mixed_precision: bool = True
    grad_clip: float = 1.0
    num_workers: int = 4
    batch_sizes: Optional[List[int]] = None


@dataclass
class ShiftConfig:
    types: List[str] = field(default_factory=lambda: ["none", "heel", "scatter", "mtf", "glare", "exposure", "poisson", "exposure_poisson", "motion_blur", "collimation_crop", "protocol"])
    intensities: Dict[str, List[float]] = field(default_factory=lambda: {
        "none": [0.0],
        "heel": [0.0, 0.2, 0.4, 0.6],
        "scatter": [0.0, 0.2, 0.4, 0.6],
        "mtf": [0.0, 0.2, 0.4, 0.6],
        "glare": [0.0, 0.2, 0.4, 0.6],
        "exposure": [0.0, 0.2, 0.4, 0.6],
        "poisson": [0.0, 0.2, 0.4, 0.6],
        "exposure_poisson": [0.0, 0.2, 0.4, 0.6],
        "motion_blur": [0.0, 0.2, 0.4, 0.6],
        "collimation_crop": [0.0, 0.2, 0.4, 0.6],
        "protocol": [0.0, 0.3, 0.6],
    })
    combine: bool = True
    seed: int = 2024
    max_composition_order: int = 2


@dataclass
class MetricsConfig:
    calibration_bins: int = 15
    risk_coverage_points: int = 50
    min_group_size: int = 50
    cost_ratio_grid: List[float] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])


@dataclass
class SafetyPolicyConfig:
    apply_on_covariate: bool = True
    abstain_on_mixed: bool = True
    abstain_on_unknown: bool = True
    cost_fn: float = 10.0
    cost_fp: float = 1.0
    class_cost_fn: Optional[List[float]] = None
    class_cost_fp: Optional[List[float]] = None
    coverage_budget: float = 0.8
    critical_labels: Optional[List[str]] = None
    conformal: bool = False
    risk_target: float = 0.05
    conformal_alpha: float = 0.05
    stratify_technical: bool = False
    strata_keys: List[str] = field(default_factory=lambda: ["view_family", "procedure_context", "orientation_context"])
    shared_threshold: bool = True
    use_decision_policy: bool = False
    uncertainty_abstain_threshold: float = 0.35
    mixed_abstain_threshold: float = 0.40
    prevalence_action: str = "recalibrate"  # recalibrate | no_eraser | abstain
    recalibration_mode: str = "global"  # global | stratified
    stratified_recalibration_min_samples: int = 64
    abstain_cost: float = 1.0
    policy_variant: str = "legacy_v1"  # legacy_v1 | expected_risk_v2
    uncertainty_lambda: float = 0.0
    route_gamma: float = 0.0
    decision_label_source: str = "predicted"  # predicted | oracle_shift_family


@dataclass
class ShiftIdConfig:
    score_family: List[str] = field(default_factory=lambda: ["mmd", "ks", "cond_mmd"])
    batch_sizes: List[int] = field(default_factory=lambda: [100, 250, 500])
    cost_matrix: Optional[Dict[str, Dict[str, float]] | List[List[float]]] = None
    calibration: str = "platt"
    report_calibration_metrics: bool = True
    n_splits: int = 3
    label_space: str = "nature"  # nature | physics
    group_key: str = "view_family"


@dataclass
class VLMConfig:
    enabled: bool = False
    model_type: str = "llava"
    model_id: Optional[str] = None
    model_path: Optional[Path] = None
    revision: Optional[str] = None
    offline: bool = False
    prompt_template: str = "Return a JSON dict with findings for each pathology."
    uncertain_value: float = 0.5
    json_max_retries: int = 2


@dataclass
class VizConfig:
    save_dir: Path = Path("./outputs/figures")
    dpi: int = 200
    format: str = "png"
    cam_layer: str = "encoder"
    audit_attention: bool = False
    max_audit_cases: int = 20
    audit_conf_threshold: float = 0.9
    enable: bool = True


@dataclass
class RuntimeConfig:
    seed: int = 2024
    device: str = "cuda"
    deterministic: bool = True
    benchmark: bool = False
    freeze_artifacts: bool = False
    allow_download: bool = True


@dataclass
class MonitoringConfig:
    enabled: bool = False
    warmup_batches: int = 10
    alert_quantile: float = 0.99
    min_consecutive_alerts: int = 2
    scores: List[str] = field(default_factory=lambda: ["mmd", "ks", "cond_mmd"])
    max_batches: Optional[int] = None


@dataclass
class OutputConfig:
    root: Path = Path("./outputs")
    run_name: Optional[str] = None
    save_checkpoints: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    enable_wandb: bool = False
    project: str = "phystta"
    entity: Optional[str] = None


@dataclass
class ExperimentConfig:
    reps: int = 3
    samples_per_condition: Optional[int] = None
    factorial: bool = True
    architectures: Optional[List[str]] = None
    shift_types: Optional[List[str]] = None
    tier: str = "dev"  # dev | paper
    open_data_fallback: bool = False


@dataclass
class ArchitectureReportingConfig:
    clean_intensity: float = 0.0


@dataclass
class ArchitectureStudyConfig:
    encoder_profiles: Optional[List[str]] = None
    interventions: List[str] = field(default_factory=lambda: ["erasure", "policy"])
    modes: Optional[List[str]] = None
    eraser_paths: Dict[str, str] = field(default_factory=dict)
    interaction_pairs: List[List[str]] = field(default_factory=lambda: [["no_eraser", "eraser"]])
    profile_tags: Dict[str, Dict[str, str]] = field(default_factory=dict)
    reporting: ArchitectureReportingConfig = field(default_factory=ArchitectureReportingConfig)


@dataclass
class KaggleDatasetConfig:
    dataset: str
    dest: Path
    unzip: bool = True
    competition: bool = False
    files: Optional[List[str]] = None


@dataclass
class KaggleConfig:
    auto_download: bool = False
    datasets: List[KaggleDatasetConfig] = field(default_factory=list)


@dataclass
class ConfigBundle:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    shifts: ShiftConfig = field(default_factory=ShiftConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    shift_id: ShiftIdConfig = field(default_factory=ShiftIdConfig)
    safety_policy: SafetyPolicyConfig = field(default_factory=SafetyPolicyConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    architecture_study: ArchitectureStudyConfig = field(default_factory=ArchitectureStudyConfig)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    eraser: EraserConfig = field(default_factory=EraserConfig)
    erasure_eval: ErasureEvalConfig = field(default_factory=ErasureEvalConfig)

    def resolve(self) -> "ConfigBundle":
        self.data.resolve()
        self.viz.save_dir = Path(self.viz.save_dir).expanduser()
        self.output.root = Path(self.output.root).expanduser()
        if self.output.run_name is None:
            self.output.run_name = os.environ.get("PHYSTTA_RUN", "run")
        for ds in self.kaggle.datasets:
            ds.dest = Path(ds.dest).expanduser()
        if self.vlm.model_path is not None:
            self.vlm.model_path = Path(self.vlm.model_path).expanduser()
        if self.intervention.realism_path is not None:
            self.intervention.realism_path = Path(self.intervention.realism_path).expanduser()
        if self.eraser.online_bank:
            self.eraser.online_bank = [Path(p).expanduser() for p in self.eraser.online_bank]
        if self.eraser.online_weights_path is not None:
            self.eraser.online_weights_path = Path(self.eraser.online_weights_path).expanduser()
        return self

    def as_dict(self) -> Dict[str, Any]:
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        return _convert(asdict(self))

    @staticmethod
    def _is_immutable_revision(revision: Optional[str]) -> bool:
        if revision is None:
            return False
        rev = revision.strip()
        if not rev:
            return False
        if rev.lower() in {"main", "master", "head", "latest"}:
            return False
        # Immutable commit hash (short/long hex)
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", rev):
            return True
        # Explicit immutable tag reference (Hugging Face style)
        if re.fullmatch(r"refs/tags/[A-Za-z0-9._-]+", rev):
            return True
        # Semantic version tags (v1.2.3, 1.2.3-rc1)
        if re.fullmatch(r"v?\d+(?:\.\d+){1,3}(?:[-._]?[A-Za-z0-9]+)?", rev):
            return True
        return False

    @staticmethod
    def _is_manual_grid_disabled(grid: Optional[List[float]]) -> bool:
        if grid is None:
            return True
        if len(grid) == 0:
            return True
        if len(grid) == 1:
            try:
                return abs(float(grid[0])) <= 1e-12
            except Exception:
                return False
        return False

    @staticmethod
    def _head_meta_candidates(head_path: Optional[Path], explicit_meta: Optional[Path] = None) -> List[Path]:
        out: List[Path] = []
        if explicit_meta is not None:
            out.append(Path(explicit_meta))
        if head_path is None:
            return out
        hp = Path(head_path)
        out.extend(
            [
                hp.with_suffix(".meta.json"),
                hp.with_name(hp.stem + "_meta.json"),
                hp.parent / "head_meta.json",
            ]
        )
        return out

    @classmethod
    def _head_meta_exists_if_head_exists(cls, head_path: Optional[Path], explicit_meta: Optional[Path] = None) -> bool:
        if head_path is None:
            return True
        hp = Path(head_path)
        if not hp.exists():
            # allow config validation before artifacts are produced
            return True
        return any(p.exists() for p in cls._head_meta_candidates(hp, explicit_meta))

    def validate(self) -> None:
        if not self.data.sources:
            raise ValueError("No dataset sources configured. Provide LIDC/chexpert/mimic sources.")
        if self.shift_id.label_space not in {"nature", "physics"}:
            raise ValueError("shift_id.label_space must be one of {'nature','physics'}.")
        if len(self.data.sources) > 1 and self.data.split_group_key == "patient_id" and not self.data.single_source_mode:
            raise ValueError(
                "Multi-source setup requires data.split_group_key=global_patient_id to avoid patient_id collisions "
                "(or explicitly set data.single_source_mode=true if you accept this risk)."
            )
        if not self.data.split_fractions:
            total = self.data.train_frac + self.data.val_frac + self.data.test_frac
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Split fractions must sum to 1.0, got {total:.3f}")
        for src in self.data.sources:
            if self.data.require_real_data:
                if src.root is None:
                    raise ValueError(f"Dataset root is required for {src.name}")
                if src.source_type == "lidc_drr" or src.name.lower() in {"lidc", "lidc_drr", "lidc-idri"}:
                    raise ValueError("Synthetic DRR sources are disabled when require_real_data=true (lidc_drr).")
        if getattr(self.experiment, "tier", "dev") == "paper":
            if self.vlm.model_path is not None and "/path/to/" in str(self.vlm.model_path):
                raise ValueError(
                    f"experiment.tier=paper forbids placeholder vlm.model_path '{self.vlm.model_path}'. "
                    "Set vlm.model_path=null when disabled, or provide a real local path."
                )
            if self.data.prebuilt_index_csv is None and not any(src.metadata_csv is not None for src in self.data.sources):
                raise ValueError(
                    "experiment.tier=paper requires deterministic indexing: set data.prebuilt_index_csv "
                    "or provide explicit metadata_csv for each data source."
                )
            if self.data.prebuilt_index_csv is not None and not self.data.prebuilt_index_sha256:
                raise ValueError(
                    "experiment.tier=paper with data.prebuilt_index_csv requires data.prebuilt_index_sha256 "
                    "for immutable index reproducibility."
                )
            if not self.intervention.use_realism or self.intervention.realism_path is None:
                raise ValueError("experiment.tier=paper requires intervention.use_realism=true and realism_path set.")
            if not self._is_manual_grid_disabled(self.intervention.intensity_grid):
                raise ValueError(
                    "experiment.tier=paper with intervention.use_realism=true forbids manual intervention.intensity_grid; "
                    "set it to [] or [0.0]."
                )
            if not self._is_manual_grid_disabled(self.erasure_eval.intensity_grid):
                raise ValueError(
                    "experiment.tier=paper with intervention.use_realism=true forbids manual erasure_eval.intensity_grid; "
                    "set it to [] or [0.0]."
                )
            if self.intervention.group_key != "global_patient_id":
                raise ValueError("experiment.tier=paper requires intervention.group_key=global_patient_id.")
            if self.data.split_group_key != "global_patient_id":
                raise ValueError("experiment.tier=paper requires data.split_group_key=global_patient_id.")
            if not self.runtime.freeze_artifacts:
                raise ValueError("experiment.tier=paper requires runtime.freeze_artifacts=true.")
            if self.runtime.allow_download:
                raise ValueError("experiment.tier=paper requires runtime.allow_download=false (no dynamic checkpoint downloads).")
            if not self.data.split_fractions:
                raise ValueError("experiment.tier=paper requires explicit data.split_fractions for head/eraser/calib/eval splits.")
            required_splits = {self.data.head_split, self.data.eraser_split, self.data.calib_split, self.data.eval_split}
            missing = required_splits - set(self.data.split_fractions.keys())
            if missing:
                raise ValueError(f"experiment.tier=paper missing split_fractions for: {sorted(missing)}")
            if self.shift_id.calibration not in {"platt", "temperature"}:
                raise ValueError("experiment.tier=paper requires shift_id.calibration in {'platt','temperature'}.")
            if self.shift_id.label_space == "nature" and self.shift_id.group_key in {"patient_id", "global_patient_id"}:
                raise ValueError(
                    "experiment.tier=paper with shift_id.label_space='nature' forbids shift_id.group_key in "
                    "{'patient_id','global_patient_id'} to avoid patient-identity leakage. "
                    "Use an acquisition-context key like view/view_family/procedure_context/orientation_context."
                )
            eval_modes = set(self.erasure_eval.modes or [self.erasure_eval.mode])
            required_modes = {"clinical", "causal_invariance", "safety", "fairness"}
            if not required_modes.issubset(eval_modes):
                raise ValueError(
                    "experiment.tier=paper requires erasure_eval.modes to include clinical, causal_invariance, safety, and fairness."
                )
            profile_map = {p.name: p.encoder for p in (self.model.encoder_profiles or [])}
            # Enforce placeholder-free paths on every declared profile in paper-tier, even if not selected.
            for p in self.model.encoder_profiles or []:
                if p.encoder.encoder_path is not None and "/path/to/" in str(p.encoder.encoder_path):
                    raise ValueError(
                        f"experiment.tier=paper: profile '{p.name}' uses placeholder encoder_path '{p.encoder.encoder_path}'."
                    )
                if p.encoder.head_path is not None and "/path/to/" in str(p.encoder.head_path):
                    raise ValueError(
                        f"experiment.tier=paper: profile '{p.name}' uses placeholder head_path '{p.encoder.head_path}'."
                    )
            study_profiles = self.architecture_study.encoder_profiles or []
            for profile_name in study_profiles:
                if profile_name not in profile_map:
                    raise ValueError(f"experiment.tier=paper: architecture_study profile '{profile_name}' is not declared in model.encoder_profiles.")
                enc = profile_map[profile_name]
                if enc.encoder_source == "timm_vit" and enc.encoder_path is None:
                    raise ValueError(
                        f"experiment.tier=paper: profile '{profile_name}' uses timm_vit and requires explicit local encoder_path when allow_download=false."
                    )
                if enc.encoder_source == "timm_vit" and enc.encoder_path is not None and not Path(enc.encoder_path).exists():
                    raise ValueError(
                        f"experiment.tier=paper: profile '{profile_name}' timm_vit encoder_path does not exist: {enc.encoder_path}"
                    )
                if enc.encoder_path is not None and "/path/to/" in str(enc.encoder_path):
                    raise ValueError(
                        f"experiment.tier=paper: profile '{profile_name}' uses placeholder encoder_path '{enc.encoder_path}'."
                    )
                if enc.encoder_source in {"hf_clip_or_vit"} and enc.model_id and not self._is_immutable_revision(enc.revision):
                    raise ValueError(
                        f"experiment.tier=paper: profile '{profile_name}' requires immutable pinned encoder.revision (commit/tag) for model_id='{enc.model_id}'."
                    )
                if not self._head_meta_exists_if_head_exists(enc.head_path, enc.head_meta_path):
                    raise ValueError(
                        f"experiment.tier=paper: profile '{profile_name}' has head_path={enc.head_path} but no head metadata sidecar "
                        "(expected *.meta.json, *_meta.json, head_meta.json, or explicit head_meta_path)."
                    )
                if enc.encoder_source == "baseline_cnn":
                    if enc.encoder_path is None:
                        raise ValueError(
                            f"experiment.tier=paper: profile '{profile_name}' baseline_cnn requires explicit encoder_path "
                            "(random/unpinned CNN weights are not allowed in paper-tier)."
                        )
                    if "/path/to/" in str(enc.encoder_path):
                        raise ValueError(
                            f"experiment.tier=paper: profile '{profile_name}' baseline_cnn uses placeholder encoder_path '{enc.encoder_path}'."
                        )
                if enc.encoder_source == "openclip" and not self.runtime.allow_download:
                    if enc.encoder_path is None and enc.pretrained is not False:
                        raise ValueError(
                            f"experiment.tier=paper: profile '{profile_name}' openclip with allow_download=false requires encoder_path or encoder.pretrained=false."
                        )
            if len(self.erasure_eval.fairness_groups or []) == 0:
                raise ValueError("experiment.tier=paper with fairness mode requires non-empty erasure_eval.fairness_groups.")
            main_enc = self.model.encoder
            if main_enc.encoder_source in {"hf_clip_or_vit"} and main_enc.model_id and not self._is_immutable_revision(main_enc.revision):
                raise ValueError(
                    f"experiment.tier=paper: model.encoder requires immutable pinned revision (commit/tag) for model_id='{main_enc.model_id}'."
                )
            if not self._head_meta_exists_if_head_exists(main_enc.head_path, main_enc.head_meta_path):
                raise ValueError(
                    "experiment.tier=paper: model.encoder has head_path but no head metadata sidecar "
                    "(expected *.meta.json, *_meta.json, head_meta.json, or explicit head_meta_path)."
                )
            if main_enc.encoder_path is not None and "/path/to/" in str(main_enc.encoder_path):
                raise ValueError(
                    f"experiment.tier=paper: model.encoder uses placeholder encoder_path '{main_enc.encoder_path}'."
                )
            if main_enc.encoder_source == "baseline_cnn":
                if main_enc.encoder_path is None:
                    raise ValueError(
                        "experiment.tier=paper: model.encoder baseline_cnn requires explicit encoder_path "
                        "(random/unpinned CNN weights are not allowed in paper-tier)."
                    )
            if main_enc.encoder_source == "openclip" and not self.runtime.allow_download:
                if main_enc.encoder_path is None and main_enc.pretrained is not False:
                    raise ValueError(
                        "experiment.tier=paper: model.encoder openclip with allow_download=false requires encoder_path or encoder.pretrained=false."
                    )
        if self.vlm is not None and self.vlm.enabled:
            if not self.vlm.model_id and self.vlm.model_path is None:
                raise ValueError("vlm.enabled=true requires vlm.model_id or vlm.model_path.")
            if self.vlm.offline and self.runtime.allow_download:
                raise ValueError("vlm.offline=true requires runtime.allow_download=false.")
            if self.vlm.model_path is not None and "/path/to/" in str(self.vlm.model_path):
                raise ValueError(f"vlm.enabled=true uses placeholder model_path '{self.vlm.model_path}'.")
            if (
                getattr(self.experiment, "tier", "dev") == "paper"
                and self.vlm.model_path is None
                and not self._is_immutable_revision(self.vlm.revision)
            ):
                raise ValueError(
                    "experiment.tier=paper with vlm.enabled=true and remote model_id requires immutable vlm.revision (commit/tag)."
                )
        if self.monitoring.enabled:
            if self.monitoring.warmup_batches < 2:
                raise ValueError("monitoring.warmup_batches must be >=2 when monitoring.enabled=true")
            if not (0.5 <= self.monitoring.alert_quantile < 1.0):
                raise ValueError("monitoring.alert_quantile must be in [0.5,1.0).")
        pq = self.intervention.pair_quality
        if pq.max_delta_hours <= 0:
            raise ValueError("intervention.pair_quality.max_delta_hours must be > 0.")
        if not (0.0 <= pq.technical_dominant_max_risk <= 1.0):
            raise ValueError("intervention.pair_quality.technical_dominant_max_risk must be in [0,1].")
        for name in [
            "label_change_weight",
            "view_change_weight",
            "context_change_weight",
            "sentinel_change_weight",
            "procedure_change_weight",
            "orientation_change_weight",
            "resolution_jump_weight",
            "support_device_change_weight",
        ]:
            val = float(getattr(pq, name))
            if val < 0:
                raise ValueError(f"intervention.pair_quality.{name} must be >= 0.")
        if self.intervention.pair_mode not in {
            "strict_device_change",
            "temporal_acquisition_context",
            "temporal_same_view",
        }:
            raise ValueError(
                "intervention.pair_mode must be one of {'strict_device_change','temporal_acquisition_context','temporal_same_view'}."
            )
        if self.eraser.fit_mode not in {"single", "one_eraser_per_shift_family"}:
            raise ValueError("eraser.fit_mode must be one of {'single','one_eraser_per_shift_family'}.")
        if self.erasure_eval.scenario_mode not in {"atomic_calibrated_eval", "mixed_stress_eval"}:
            raise ValueError(
                "erasure_eval.scenario_mode must be one of {'atomic_calibrated_eval','mixed_stress_eval'}."
            )
