from __future__ import annotations

import numpy as np


def compare_final_positions(samples_true: np.ndarray, samples_pred: np.ndarray):
    return {
        "true_mean": samples_true.mean(axis=0),
        "pred_mean": samples_pred.mean(axis=0),
        "true_cov": np.cov(samples_true.T),
        "pred_cov": np.cov(samples_pred.T),
    }
