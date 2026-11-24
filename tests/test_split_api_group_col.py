import pandas as pd

from phystta.data.split import patient_level_split


def _synthetic_df() -> pd.DataFrame:
    rows = []
    for src in ["s1", "s2"]:
        for pid in ["p1", "p2", "p3", "p4", "p5"]:
            gid = f"{src}:{pid}"
            for t in range(2):
                rows.append(
                    {
                        "source": src,
                        "patient_id": pid,
                        "global_patient_id": gid,
                        "image_path": f"/tmp/{src}_{pid}_{t}.png",
                        "No Finding": float((hash(gid) + t) % 2),
                    }
                )
    return pd.DataFrame(rows)


def test_patient_level_split_uses_group_col_and_is_reproducible():
    df = _synthetic_df()
    kwargs = dict(
        df=df,
        label_cols=["No Finding"],
        seed=42,
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        stratify=False,
        group_col="global_patient_id",
    )
    s1 = patient_level_split(**kwargs)
    s2 = patient_level_split(**kwargs)
    assert s1 == s2
    train = set(s1["train"])
    val = set(s1["val"])
    test = set(s1["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert all(":" in g for g in train | val | test)
