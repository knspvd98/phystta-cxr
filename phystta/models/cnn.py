from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel


class CNNClassifier(BaseModel):
    def __init__(
        self,
        architecture: str,
        num_classes: int,
        in_chans: int = 1,
        pretrained: bool = True,
        dropout: float = 0.0,
        use_medical_pretrain: bool = False,
    ):
        super().__init__(num_classes=num_classes, in_chans=in_chans)
        self.architecture = architecture
        self.use_medical_pretrain = use_medical_pretrain
        self.dropout = dropout
        self.backbone, feature_dim = self._build_backbone(architecture, in_chans, pretrained, use_medical_pretrain)
        self.head = nn.Linear(feature_dim, num_classes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _build_backbone(self, architecture: str, in_chans: int, pretrained: bool, use_medical_pretrain: bool):
        if use_medical_pretrain:
            try:
                import torchxrayvision as xrv
            except Exception as e:
                raise RuntimeError(
                    "use_medical_pretrain=true requires torchxrayvision. "
                    "Install dependency 'torchxrayvision' from requirements.lock."
                ) from e
            if architecture.lower().startswith("densenet"):
                model = xrv.models.DenseNet(weights="chexpert")
                model.op_threshs = None
                model = model.features
                feature_dim = 1024
                return model, feature_dim
            raise RuntimeError(
                f"use_medical_pretrain=true is currently supported only for densenet backbones, got '{architecture}'."
            )
        try:
            import timm
            model = timm.create_model(architecture, pretrained=pretrained, in_chans=in_chans, num_classes=0)
            feature_dim = model.num_features
            return model, feature_dim
        except Exception:
            try:
                import torchvision.models as tvm
                if architecture == "densenet121":
                    model = tvm.densenet121(pretrained=pretrained)
                    feature_dim = model.classifier.in_features
                    model.classifier = nn.Identity()
                    return model, feature_dim
                if architecture == "resnet50":
                    model = tvm.resnet50(pretrained=pretrained)
                    feature_dim = model.fc.in_features
                    model.fc = nn.Identity()
                    return model, feature_dim
            except Exception as e:
                raise RuntimeError(f"Backbone {architecture} not available") from e
        raise RuntimeError(f"Backbone {architecture} not available")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        features = self.drop(features)
        return features

    def forward_with_features(self, x: torch.Tensor):
        features = self.forward_features(x)
        logits = self.head(features)
        return logits, features

    def get_norm_params(self):
        params = []
        for name, param in self.named_parameters():
            if "norm" in name.lower() or "bn" in name.lower():
                params.append(param)
        return params
