from __future__ import annotations

from pathlib import Path
from typing import Optional

from .vlm_wrapper import VLMWrapper


def load_vlm(
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_type: str = "llava",
    device: str = "cpu",
    allow_download: bool = True,
    revision: Optional[str] = None,
    offline: bool = False,
) -> VLMWrapper:
    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"VLM weights not found at {model_path}")
        source = str(path)
        local_files_only = True
    elif model_id:
        source = model_id
        local_files_only = offline or not allow_download
    else:
        raise RuntimeError("VLM requires model_id or model_path")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
    except Exception as e:
        raise RuntimeError("transformers is required for VLM") from e
    processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only, revision=revision)
    if model_type == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(source, local_files_only=local_files_only, revision=revision)
    else:
        model = AutoModelForVision2Seq.from_pretrained(source, local_files_only=local_files_only, revision=revision)
    return VLMWrapper(model, processor, device=device)
