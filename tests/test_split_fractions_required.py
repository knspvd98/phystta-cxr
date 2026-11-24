import textwrap

import pytest

from phystta.config.loader import load_config
from phystta.data import DataEngine
import pandas as pd


def _write(tmp_path, content: str) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_custom_splits_require_fractions(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        data:
          sources:
            - name: mimic
              root: /tmp
          head_split: train_head
          eraser_split: train_eraser
          calib_split: calib
          eval_split: eval
        """,
    )
    cfg = load_config(cfg_path)
    engine = DataEngine(cfg.data)
    # minimal index stub to pass "index built" check
    engine.index = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "source": ["mimic", "mimic"],
        "image_path": ["/tmp/a.png", "/tmp/b.png"],
    })
    with pytest.raises(RuntimeError, match="split_fractions"):
        engine.create_splits()
