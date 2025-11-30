"""Tools that mirror the old evaluation helpers for comparing trajectory distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    """Summary statistics comparing two final-position ensembles."""

    true_mean: np.ndarray
    pred_mean: np.ndarray
    true_cov: np.ndarray
    pred_cov: np.ndarray

    def mean_distance(self) -> float:
        return float(np.linalg.norm(self.true_mean - self.pred_mean))

    def cov_distance(self) -> float:
        return float(np.linalg.norm(self.true_cov - self.pred_cov))

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_dist": self.mean_distance(),
            "cov_dist": self.cov_distance(),
        }


def compare_final_positions(samples_true: np.ndarray, samples_pred: np.ndarray) -> ComparisonResult:
    """Compare the Gaussian summaries implied by two final-position point clouds."""

    true_mean = np.mean(samples_true, axis=0)
    pred_mean = np.mean(samples_pred, axis=0)
    true_cov = np.cov(samples_true.T)
    pred_cov = np.cov(samples_pred.T)
    return ComparisonResult(
        true_mean=true_mean,
        pred_mean=pred_mean,
        true_cov=true_cov,
        pred_cov=pred_cov,
    )