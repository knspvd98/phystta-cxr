import pandas as pd

from phystta.causal.pair_quality import (
    PairQualityConfig,
    annotate_pair_quality,
    ensure_pairs_group_key,
    split_pairs_by_quality,
    summarize_pair_quality,
)
from phystta.shifts.realism import RealismCalibrator


def test_pair_quality_splits_technical_vs_ambiguous():
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "image_path_a": "a1",
                "image_path_b": "b1",
                "delta_hours": 2.0,
                "view_a": "AP",
                "view_b": "AP",
                "manufacturer_a": "m1",
                "manufacturer_b": "m2",
                "station_name_a": "s1",
                "station_name_b": "s2",
            },
            {
                "patient_id": "p2",
                "image_path_a": "a2",
                "image_path_b": "b2",
                "delta_hours": 20.0,
                "view_a": "AP",
                "view_b": "PA",
                "manufacturer_a": "m1",
                "manufacturer_b": "m2",
                "station_name_a": "s1",
                "station_name_b": "s3",
            },
        ]
    )
    index_df = pd.DataFrame(
        [
            {"image_path": "a1", "Pneumonia": 0.0},
            {"image_path": "b1", "Pneumonia": 0.0},
            {"image_path": "a2", "Pneumonia": 0.0},
            {"image_path": "b2", "Pneumonia": 1.0},
        ]
    )
    out = annotate_pair_quality(
        pairs,
        index_df=index_df,
        label_cols=["Pneumonia"],
        cfg=PairQualityConfig(technical_dominant_max_risk=0.25),
    )
    split = split_pairs_by_quality(out)
    assert len(split["technical_dominant"]) == 1
    assert len(split["ambiguous"]) == 1


def test_ensure_pairs_group_key_builds_global_patient_id():
    pairs = pd.DataFrame(
        [
            {"patient_id": "p1", "source_a": "mimic", "image_path_a": "a", "image_path_b": "b"},
            {"patient_id": "p2", "source_a": "mimic", "image_path_a": "c", "image_path_b": "d"},
        ]
    )
    out = ensure_pairs_group_key(pairs, group_key="global_patient_id", default_source="mimic")
    assert "global_patient_id" in out.columns
    assert set(out["global_patient_id"].astype(str)) == {"mimic:p1", "mimic:p2"}


def test_pair_quality_hard_gates_clinical_pairs_from_realism_mapping():
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "stable_1",
                "patient_id": "p1",
                "image_path_a": "a1",
                "image_path_b": "b1",
                "delta_hours": 2.0,
                "view_a": "AP",
                "view_b": "AP",
            },
            {
                "pair_id": "stable_2",
                "patient_id": "p2",
                "image_path_a": "a2",
                "image_path_b": "b2",
                "delta_hours": 2.0,
                "view_a": "AP",
                "view_b": "AP",
            },
            {
                "pair_id": "ambiguous_1",
                "patient_id": "p3",
                "image_path_a": "a3",
                "image_path_b": "b3",
                "delta_hours": 2.0,
                "view_a": "AP",
                "view_b": "PA",
            },
        ]
    )
    index_df = pd.DataFrame(
        [
            {"image_path": "a1", "Pneumonia": 0.0},
            {"image_path": "b1", "Pneumonia": 0.0},
            {"image_path": "a2", "Pneumonia": 0.0},
            {"image_path": "b2", "Pneumonia": 0.0},
            {"image_path": "a3", "Pneumonia": 0.0},
            {"image_path": "b3", "Pneumonia": 1.0},
        ]
    )
    pair_to_dist = {"stable_1": 0.11, "stable_2": 0.12, "ambiguous_1": 0.90}

    strict = annotate_pair_quality(
        pairs,
        index_df=index_df,
        label_cols=["Pneumonia"],
        cfg=PairQualityConfig(technical_dominant_max_risk=0.25),
    )
    loose = annotate_pair_quality(
        pairs,
        index_df=index_df,
        label_cols=["Pneumonia"],
        cfg=PairQualityConfig(technical_dominant_max_risk=0.95),
    )
    strict_td = split_pairs_by_quality(strict)["technical_dominant"]
    loose_td = split_pairs_by_quality(loose)["technical_dominant"]
    assert set(strict_td["pair_id"].astype(str)) == {"stable_1", "stable_2"}
    assert set(loose_td["pair_id"].astype(str)) == {"stable_1", "stable_2"}
    assert strict.loc[strict["pair_id"] == "ambiguous_1", "pair_quality"].item() == "clinical_change_suspected"
    assert loose.loc[loose["pair_id"] == "ambiguous_1", "pair_quality"].item() == "clinical_change_suspected"

    strict_real = strict_td["pair_id"].map(pair_to_dist).to_numpy(dtype=float)
    loose_real = loose_td["pair_id"].map(pair_to_dist).to_numpy(dtype=float)
    synth = {
        0.0: pd.Series([0.08, 0.12]).to_numpy(dtype=float),
        0.5: pd.Series([0.30, 0.36]).to_numpy(dtype=float),
        1.0: pd.Series([0.82, 0.96]).to_numpy(dtype=float),
    }

    calibrator = RealismCalibrator(distance_metric="l2")
    m_strict = calibrator.fit(
        "heel",
        strict_real,
        synth_dists_by_intensity=synth,
        realism_levels=[0.9],
        method="quantile",
    )
    m_loose = calibrator.fit(
        "scatter",
        loose_real,
        synth_dists_by_intensity=synth,
        realism_levels=[0.9],
        method="quantile",
    )
    # Hard clinical/view gates keep the realism anchor set stable even if the numeric threshold is loosened.
    assert abs(float(m_loose.intensities[0]) - float(m_strict.intensities[0])) < 1e-6


def test_pair_quality_context_shift_is_not_technical_dominant():
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "stable",
                "patient_id": "p1",
                "image_path_a": "a1",
                "image_path_b": "b1",
                "delta_hours": 3.0,
                "view_a": "PA",
                "view_b": "PA",
                "view_family_a": "frontal",
                "view_family_b": "frontal",
                "procedure_context_a": "portable_like",
                "procedure_context_b": "portable_like",
                "orientation_context_a": "upright",
                "orientation_context_b": "upright",
            },
            {
                "pair_id": "procedure_jump",
                "patient_id": "p2",
                "image_path_a": "a2",
                "image_path_b": "b2",
                "delta_hours": 3.0,
                "view_a": "PA",
                "view_b": "PA",
                "view_family_a": "frontal",
                "view_family_b": "frontal",
                "procedure_context_a": "portable_like",
                "procedure_context_b": "standard_pa",
                "orientation_context_a": "upright",
                "orientation_context_b": "upright",
            },
        ]
    )
    out = annotate_pair_quality(pairs, cfg=PairQualityConfig(technical_dominant_max_risk=0.25))
    by_id = out.set_index("pair_id")
    assert by_id.loc["stable", "pair_quality_bucket"] == "technical_dominant"
    assert by_id.loc["procedure_jump", "pair_quality_bucket"] == "ambiguous"
    assert by_id.loc["procedure_jump", "pair_quality"] == "same_view_context_shift"

    summary = summarize_pair_quality(out)
    assert summary["bucket_counts"]["technical_dominant"] == 1
    assert summary["bucket_counts"]["ambiguous"] == 1
    assert summary["quality_reason_counts"]["procedure_change"] == 1
