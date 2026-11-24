from __future__ import annotations

import json
from typing import Any, Dict, Optional

import torch


class VLMWrapper:
    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _find_vision_tower(self):
        for attr in ["vision_tower", "vision_encoder", "vision_model"]:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        if hasattr(self.model, "model"):
            inner = self.model.model
            for attr in ["vision_tower", "vision_encoder", "vision_model"]:
                if hasattr(inner, attr):
                    return getattr(inner, attr)
        return None

    def _hook_eraser(self, eraser):
        tower = self._find_vision_tower()
        if tower is None:
            raise RuntimeError("Unsupported VLM architecture: vision tower not found")

        def _hook(module, inputs, output):
            out = output
            if isinstance(output, (tuple, list)):
                out = output[0]
            if hasattr(output, "last_hidden_state"):
                out = output.last_hidden_state
            if out.dim() == 3:
                b, t, d = out.shape
                flat = out.reshape(-1, d)
                flat = eraser(flat)
                out = flat.reshape(b, t, d)
            elif out.dim() == 2:
                out = eraser(out)
            if isinstance(output, (tuple, list)):
                return (out,) + tuple(output[1:])
            if hasattr(output, "last_hidden_state"):
                output.last_hidden_state = out
                return output
            return out

        handle = tower.register_forward_hook(_hook)
        return handle

    def generate(self, image, prompt: str, eraser=None, max_new_tokens: int = 256) -> str:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        handle = None
        if eraser is not None:
            handle = self._hook_eraser(eraser)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        if handle is not None:
            handle.remove()
        text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return text

    @staticmethod
    def validate_json(data: Dict[str, Any], keys: list[str]) -> bool:
        if data is None:
            return False
        for k in keys:
            if k not in data:
                return False
            val = str(data.get(k, "")).strip().lower()
            if val not in {"present", "absent", "uncertain"}:
                return False
        return True

    @staticmethod
    def _repair_prompt(keys: list[str]) -> str:
        key_list = ", ".join(keys)
        return (
            "Return ONLY a valid JSON object with exactly these keys: "
            f"{key_list}. Values must be one of: present, absent, uncertain. No extra text."
        )

    def generate_json(
        self,
        image,
        prompt: str,
        keys: list[str],
        eraser=None,
        max_new_tokens: int = 256,
        max_retries: int = 2,
    ) -> tuple[str, Dict[str, str], bool]:
        last_text = ""
        for attempt in range(max_retries + 1):
            text = self.generate(image, prompt, eraser=eraser, max_new_tokens=max_new_tokens)
            last_text = text
            data = self.extract_json(text)
            norm = self.normalize_json(data, keys)
            if self.validate_json(norm, keys):
                return text, norm, True
            prompt = self._repair_prompt(keys)
        data = self.extract_json(last_text)
        norm = self.normalize_json(data, keys)
        return last_text, norm, False

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        if text is None:
            return None
        # Try to find a balanced JSON object substring.
        start_positions = [i for i, ch in enumerate(text) if ch == "{"]
        for start in start_positions:
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start: idx + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
        return None

    @staticmethod
    def normalize_json(data: Dict[str, Any], keys: list[str]) -> Dict[str, str]:
        norm = {}
        for k in keys:
            val = data.get(k, "uncertain") if data else "uncertain"
            if isinstance(val, (int, float)):
                val = "present" if val >= 0.5 else "absent"
            val = str(val).strip().lower()
            if val in {"yes", "present", "positive", "1", "true"}:
                norm[k] = "present"
            elif val in {"no", "absent", "negative", "0", "false"}:
                norm[k] = "absent"
            else:
                norm[k] = "uncertain"
        return norm
