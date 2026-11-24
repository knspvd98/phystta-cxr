import numpy as np

from phystta.shifts.realism import RealismCalibrator, scenario_grid_from_calibrator


def test_scenario_grid_from_calibrator_returns_level_intensity_pairs():
    calib = RealismCalibrator()
    real = np.array([0.1, 0.2, 0.3, 0.4])
    synth = {
        0.0: np.array([0.0, 0.02, 0.01]),
        0.3: np.array([0.12, 0.11, 0.10]),
        0.6: np.array([0.25, 0.28, 0.26]),
    }
    levels = [0.0, 0.5, 1.0]
    calib.fit("heel", real_dists=real, synth_dists_by_intensity=synth, realism_levels=levels, method="quantile")

    grid = scenario_grid_from_calibrator(calib, ["heel"])
    assert "heel" in grid
    assert len(grid["heel"]) == 3
    mapped_levels = [lv for lv, _ in grid["heel"]]
    assert mapped_levels == levels
