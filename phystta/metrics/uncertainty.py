from __future__ import annotations

import numpy as np


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(probs + 1e-8) + (1 - probs) * np.log(1 - probs + 1e-8))


def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
    mean_prob = mc_probs.mean(axis=0)
    entropy_mean = predictive_entropy(mean_prob)
    entropy_each = predictive_entropy(mc_probs)
    return entropy_mean - entropy_each.mean(axis=0)


def variance_from_mc(mc_probs: np.ndarray) -> np.ndarray:
    return mc_probs.var(axis=0)
