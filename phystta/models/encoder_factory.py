from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..utils import load_torch_artifact, load_torch_state_dict

try:
    from .cnn import CNNClassifier  # type: ignore
except Exception:  # pragma: no cover - handled lazily in baseline branch
    CNNClassifier = None  # type: ignore[assignment]


@contextmanager
def _hf_offline_env(enabled: bool):
    if not enabled:
        yield
        return
    keys = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    prev = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k in keys:
            if prev[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev[k]


def _require_hf_local_snapshot(model_id: str, revision: Optional[str]) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Download is disabled (runtime.allow_download=false) and huggingface_hub is not available to verify local cache. "
            "Install huggingface_hub or provide model.encoder.encoder_path."
        ) from e
    try:
        return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Download is disabled (runtime.allow_download=false) and model_id='{model_id}'"
            f"{f' revision={revision}' if revision else ''} is not available in local HF cache. "
            "Pre-download this exact revision or set model.encoder.encoder_path to local weights."
        ) from e


class EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_adapter=None, mode: str = "default"):
        super().__init__()
        self.model = model
        self.input_adapter = input_adapter
        self.mode = mode
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._dim: Optional[int] = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        elif hasattr(self.model, "forward_with_features"):
            out = self.model.forward_with_features(x)
            feats = out[-1] if isinstance(out, (tuple, list)) else out
        elif hasattr(self.model, "get_image_features"):
            feats = self.model.get_image_features(pixel_values=x)
        elif hasattr(self.model, "encode_image"):
            feats = self.model.encode_image(x)
        elif hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=x)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            else:
                feats = out.last_hidden_state[:, 0]
        elif hasattr(self.model, "encoder"):
            feats = self.model.encoder(x)
        elif hasattr(self.model, "backbone"):
            feats = self.model.backbone(x)
        else:
            raise RuntimeError("EncoderWrapper could not find feature extractor. Tried: forward_features, forward_with_features, encoder, backbone.")
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        if self._dim is None:
            self._dim = feats.shape[1]
        return feats

    def encode_tokens(self, x: torch.Tensor, token_layers: int = 1) -> torch.Tensor:
        x = self.preprocess(x)
        # HuggingFace vision models
        if hasattr(self.model, "vision_model"):
            out = self.model.vision_model(pixel_values=x, output_hidden_states=token_layers > 1)
            if token_layers > 1:
                hs = out.hidden_states[-token_layers:]
                tokens = torch.stack(hs, dim=0).mean(0)
            else:
                tokens = out.last_hidden_state
            return tokens
        # timm ViT-style
        if all(hasattr(self.model, attr) for attr in ["patch_embed", "cls_token", "pos_embed", "pos_drop", "blocks", "norm"]):
            tokens = self._timm_tokens(x, token_layers=token_layers)
            return tokens
        raise RuntimeError("encode_tokens unsupported for this encoder. Requires HF vision_model or timm ViT-style blocks.")

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # default: CLS token
        if tokens.dim() != 3:
            raise RuntimeError("tokens must be [N,T,D]")
        return tokens[:, 0]

    def _timm_tokens(self, x: torch.Tensor, token_layers: int = 1) -> torch.Tensor:
        model = self.model
        x = model.patch_embed(x)
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        if getattr(model, "dist_token", None) is not None:
            dist = model.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        else:
            x = torch.cat((cls, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        tokens_stack = []
        for idx, blk in enumerate(model.blocks):
            x = blk(x)
            if token_layers > 0 and idx >= len(model.blocks) - token_layers:
                tokens_stack.append(x)
        x = model.norm(x)
        if token_layers > 1 and tokens_stack:
            return torch.stack(tokens_stack, dim=0).mean(0)
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_adapter is not None:
            return self.input_adapter(x)
        return x

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("EncoderWrapper.dim not initialized; call encode once.")
        return self._dim


class EncoderFactory:
    @staticmethod
    def resolve_encoder_config(cfg, profile: Optional[str] = None):
        enc_cfg = cfg.model.encoder
        if profile and cfg.model.encoder_profiles:
            match = [p for p in cfg.model.encoder_profiles if p.name == profile]
            if not match:
                raise RuntimeError(f"Encoder profile '{profile}' not found in config.")
            enc_cfg = match[0].encoder
        # Persist requested medical-pretrain intent in the resolved encoder config so
        # fingerprint metadata can report whether CheXpert/XRV weights were requested.
        if getattr(enc_cfg, "encoder_source", None) == "baseline_cnn":
            requested_medical = bool(getattr(cfg.model, "use_medical_pretrain", False))
            setattr(enc_cfg, "use_medical_pretrain", requested_medical)
            arch = str(getattr(enc_cfg, "architecture", None) or getattr(cfg.model, "architecture", ""))
            if getattr(enc_cfg, "pretrained", None) is None:
                pretrained_hint = bool(getattr(cfg.runtime, "allow_download", False) and getattr(enc_cfg, "encoder_path", None) is None)
            else:
                pretrained_hint = bool(getattr(enc_cfg, "pretrained"))
            effective_medical = bool(requested_medical and pretrained_hint and arch.lower().startswith("densenet"))
            setattr(enc_cfg, "effective_medical_pretrain", effective_medical)
        else:
            setattr(enc_cfg, "use_medical_pretrain", False)
            setattr(enc_cfg, "effective_medical_pretrain", False)
        return enc_cfg

    @staticmethod
    def from_config(cfg, profile: Optional[str] = None) -> EncoderWrapper:
        enc_cfg = EncoderFactory.resolve_encoder_config(cfg, profile)
        source = enc_cfg.encoder_source
        adapter = None
        # double normalization guard
        if cfg.data.preprocess.normalize_mean != 0.0 or cfg.data.preprocess.normalize_std != 1.0:
            raise RuntimeError("Double normalization detected: dataset preprocess normalizes. Remove normalize_mean/std from preprocess.")
        if enc_cfg.input_adapter is not None:
            ia = enc_cfg.input_adapter
            def _adapter(x):
                if x.dim() != 4:
                    raise RuntimeError("input_adapter expects [N,C,H,W]")
                if ia.repeat_to_3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if ia.image_size and (x.shape[-1] != ia.image_size or x.shape[-2] != ia.image_size):
                    x = torch.nn.functional.interpolate(x, size=(ia.image_size, ia.image_size), mode="bilinear", align_corners=False)
                c = x.shape[1]
                mean_list = list(ia.mean)
                std_list = list(ia.std)
                if len(mean_list) == 1 and c > 1:
                    mean_list = mean_list * c
                if len(std_list) == 1 and c > 1:
                    std_list = std_list * c
                if len(mean_list) != c or len(std_list) != c:
                    raise RuntimeError(f"input_adapter mean/std length must match channels (C={c}).")
                mean = torch.tensor(mean_list, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
                std = torch.tensor(std_list, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
                x = (x - mean) / std
                return x
            adapter = _adapter
        if source == "baseline_cnn":
            global CNNClassifier
            if CNNClassifier is None:
                try:
                    from .cnn import CNNClassifier as _CNNClassifier
                    CNNClassifier = _CNNClassifier
                except Exception as e:
                    raise RuntimeError(
                        "baseline_cnn requested but phystta.models.cnn.CNNClassifier is unavailable. "
                        "Install/fix the CNN backend module or use a non-CNN encoder profile."
                    ) from e
            arch = enc_cfg.architecture or cfg.model.architecture
            if enc_cfg.pretrained is None:
                pretrained = bool(cfg.runtime.allow_download and enc_cfg.encoder_path is None)
            else:
                pretrained = bool(enc_cfg.pretrained)
            if not cfg.runtime.allow_download and pretrained and enc_cfg.encoder_path is None:
                raise RuntimeError(
                    "baseline_cnn download is disabled (runtime.allow_download=false). "
                    "Set encoder.pretrained=false or provide encoder.encoder_path."
                )
            use_medical = bool(
                getattr(enc_cfg, "use_medical_pretrain", getattr(cfg.model, "use_medical_pretrain", False))
                and pretrained
                and str(arch).lower().startswith("densenet")
            )
            setattr(enc_cfg, "effective_medical_pretrain", use_medical)
            model = CNNClassifier(
                arch,
                num_classes=cfg.model.num_classes,
                in_chans=cfg.model.in_chans,
                pretrained=pretrained,
                dropout=cfg.model.dropout,
                use_medical_pretrain=use_medical,
            )
            if enc_cfg.encoder_path is not None:
                sd = load_torch_state_dict(enc_cfg.encoder_path, map_location="cpu", key="model")
                model.load_state_dict(sd, strict=False)
            return EncoderWrapper(model, input_adapter=adapter)
        if source == "timm_vit":
            if enc_cfg.input_adapter is None or not enc_cfg.input_adapter.repeat_to_3:
                raise RuntimeError("timm_vit requires 3-channel input. Enable input_adapter.repeat_to_3.")
            try:
                import timm
            except Exception as e:
                raise RuntimeError("timm is required for timm_vit encoders.") from e
            arch = enc_cfg.architecture or cfg.model.architecture
            path = enc_cfg.encoder_path
            if path is None:
                if not cfg.runtime.allow_download:
                    raise RuntimeError("timm_vit requires encoder_path or allow_download=true.")
                model = timm.create_model(arch, pretrained=True, num_classes=0)
            else:
                model = timm.create_model(arch, pretrained=False, num_classes=0)
                sd = load_torch_state_dict(path, map_location="cpu", key="model")
                model.load_state_dict(sd, strict=False)
            return EncoderWrapper(model, input_adapter=adapter)
        if source == "hf_clip_or_vit":
            if enc_cfg.input_adapter is None or not enc_cfg.input_adapter.repeat_to_3:
                raise RuntimeError("hf_clip_or_vit requires 3-channel input. Enable input_adapter.repeat_to_3.")
            model_id = enc_cfg.model_id or enc_cfg.encoder_name
            path = enc_cfg.encoder_path
            if model_id is None and path is None:
                raise RuntimeError("hf_clip_or_vit requires model_id or encoder_path (local).")
            if path is not None:
                model_ref = str(path)
            else:
                model_ref = model_id
                if not cfg.runtime.allow_download:
                    model_ref = _require_hf_local_snapshot(model_id=model_id, revision=enc_cfg.revision)
            try:
                from transformers import AutoModel, CLIPModel
            except Exception as e:
                raise RuntimeError("transformers not installed. Install transformers to use hf_clip_or_vit.") from e
            load_kwargs = {
                "local_files_only": not cfg.runtime.allow_download,
            }
            if path is None and enc_cfg.revision:
                load_kwargs["revision"] = enc_cfg.revision
            with _hf_offline_env(enabled=not cfg.runtime.allow_download):
                try:
                    model = CLIPModel.from_pretrained(model_ref, **load_kwargs)
                except Exception:
                    model = AutoModel.from_pretrained(model_ref, **load_kwargs)
            return EncoderWrapper(model, input_adapter=adapter)
        if source == "openclip":
            if enc_cfg.input_adapter is None or not enc_cfg.input_adapter.repeat_to_3:
                raise RuntimeError("openclip requires 3-channel input. Enable input_adapter.repeat_to_3.")
            try:
                import open_clip
            except Exception as e:
                raise RuntimeError("open_clip_torch is required for encoder_source=openclip.") from e
            model_name = enc_cfg.architecture or "ViT-B-32"
            if not cfg.runtime.allow_download and enc_cfg.encoder_path is None and enc_cfg.pretrained is not False:
                raise RuntimeError(
                    "openclip download is disabled (runtime.allow_download=false). "
                    "Set encoder.pretrained=false or provide encoder.encoder_path."
                )
            if enc_cfg.pretrained is False and enc_cfg.encoder_path is None:
                model = open_clip.create_model(model_name=model_name, pretrained=None, device="cpu")
            else:
                pretrained_ref = str(enc_cfg.encoder_path) if enc_cfg.encoder_path is not None else (enc_cfg.model_id or "openai")
                model, _, _ = open_clip.create_model_and_transforms(
                    model_name=model_name,
                    pretrained=pretrained_ref,
                    device="cpu",
                )
            return EncoderWrapper(model, input_adapter=adapter)
        if source in {"eva_x", "chexfound", "llava_rad_encoder"}:
            path = enc_cfg.encoder_path
            if path is None or not Path(path).exists():
                raise RuntimeError(f"{source} requires local encoder_path to weights. Provide model.encoder.encoder_path or use baseline_cnn.")
            obj = load_torch_artifact(path, map_location="cpu")
            if isinstance(obj, nn.Module):
                return EncoderWrapper(obj, input_adapter=adapter)
            if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
                return EncoderWrapper(obj["model"], input_adapter=adapter)
            raise RuntimeError(f"Unsupported encoder checkpoint format for {source}. Provide a torch nn.Module or a dict with 'model'.")
        raise RuntimeError(f"Unknown encoder source {source}. Use baseline_cnn, timm_vit, or hf_clip_or_vit.")


def build_encoder(cfg, profile: Optional[str] = None) -> EncoderWrapper:
    """Backward-compatible helper for callers expecting function-style factory."""
    return EncoderFactory.from_config(cfg, profile=profile)
