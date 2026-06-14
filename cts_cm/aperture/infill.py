from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cts_cm.aperture.frames import FloatArray

BoolArray = NDArray[np.bool_]


def _design(matrix: FloatArray, target: int, mask: BoolArray) -> tuple[FloatArray, BoolArray]:
    others = np.delete(matrix, target, axis=1)
    rows: BoolArray = ~mask[:, target]
    intercept = np.ones((others.shape[0], 1))
    return np.hstack([intercept, others]), rows


@dataclass
class ChainedImputer:
    n_imputations: int = 10
    n_rounds: int = 5
    seed: int = 0

    def impute(self, data: FloatArray) -> list[FloatArray]:
        mask = np.isnan(data)
        column_means = np.where(mask.any(axis=0), np.nanmean(data, axis=0), 0.0)
        completions: list[FloatArray] = []
        for draw in range(self.n_imputations):
            gen = np.random.default_rng(self.seed + draw)
            current = data.copy()
            for column in range(current.shape[1]):
                missing = mask[:, column]
                current[missing, column] = column_means[column]
            for _ in range(self.n_rounds):
                for column in range(current.shape[1]):
                    missing = mask[:, column]
                    if not missing.any():
                        continue
                    design, observed = _design(current, column, mask)
                    coef, residual, _, _ = np.linalg.lstsq(
                        design[observed], current[observed, column], rcond=None
                    )
                    prediction = design[missing] @ coef
                    spread = float(np.sqrt(residual.mean())) if residual.size else 1.0
                    current[missing, column] = prediction + spread * gen.standard_normal(
                        int(missing.sum())
                    )
            completions.append(current)
        return completions


def rubin_pool(estimates: Sequence[float], variances: Sequence[float]) -> tuple[float, float]:
    count = len(estimates)
    point = float(np.mean(estimates))
    within = float(np.mean(variances))
    between = float(np.var(estimates, ddof=1)) if count > 1 else 0.0
    total = within + (1.0 + 1.0 / count) * between
    return point, total
