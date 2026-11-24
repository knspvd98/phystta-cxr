import textwrap

import pytest

from phystta.config.loader import load_config
from phystta.models import EncoderFactory


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_double_normalization_guard(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        data:
          require_real_data: false
          sources:
            - name: dummy
              root: /tmp
              source_type: chexpert
          preprocess:
            normalize_mean: 0.5
            normalize_std: 0.5
        model:
          encoder:
            encoder_source: baseline_cnn
            input_adapter:
              repeat_to_3: false
              mean: [0.5]
              std: [0.5]
        """,
    )
    cfg = load_config(cfg_path)
    with pytest.raises(RuntimeError, match="Double normalization"):
        EncoderFactory.from_config(cfg)
