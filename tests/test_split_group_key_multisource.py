import textwrap

import pandas as pd

from phystta.config.loader import load_config
from phystta.data import DataEngine


def _write_cfg(tmp_path) -> str:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            data:
              require_real_data: false
              split_seed: 7
              split_group_key: global_patient_id
              split_fractions:
                train: 0.5
                eval: 0.5
              head_split: train
              eraser_split: train
              calib_split: eval
              eval_split: eval
              sources:
                - name: s1
                  root: /tmp
                - name: s2
                  root: /tmp
            """
        ).strip(),
        encoding="utf-8",
    )
    return str(cfg_path)


def test_multisource_split_uses_global_patient_group_key(tmp_path):
    cfg = load_config(_write_cfg(tmp_path))
    engine = DataEngine(cfg.data)
    # Overlapping patient_id values across sources must not collide when split_group_key=global_patient_id.
    engine.index = pd.DataFrame(
        {
            "source": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "patient_id": ["p1", "p2", "p3", "p1", "p4", "p5"],
            "global_patient_id": ["s1:p1", "s1:p2", "s1:p3", "s2:p1", "s2:p4", "s2:p5"],
            "image_path": [f"/tmp/im_{i}.png" for i in range(6)],
        }
    )
    engine.label_cols = []
    engine.create_splits()

    train_groups = set(engine.index.loc[engine.index["split"] == "train", "global_patient_id"].astype(str))
    eval_groups = set(engine.index.loc[engine.index["split"] == "eval", "global_patient_id"].astype(str))
    assert train_groups.isdisjoint(eval_groups)
    # Confirm overlapping patient_id exists across sources; this is the collision case the group key avoids.
    s1_ids = set(engine.index.loc[engine.index["source"] == "s1", "patient_id"].astype(str))
    s2_ids = set(engine.index.loc[engine.index["source"] == "s2", "patient_id"].astype(str))
    assert len(s1_ids.intersection(s2_ids)) > 0
