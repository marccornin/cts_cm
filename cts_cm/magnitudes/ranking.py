from __future__ import annotations

import numpy as np

from cts_cm.aperture.frames import FloatArray


def c_index(score: FloatArray, severity: FloatArray) -> float:
    severity_diff = severity[:, None] - severity[None, :]
    score_diff = score[:, None] - score[None, :]
    comparable = severity_diff > 0
    total = float(comparable.sum())
    if total == 0.0:
        return float("nan")
    concordant = float((comparable & (score_diff > 0)).sum())
    tied = float((comparable & (score_diff == 0)).sum())
    return (concordant + 0.5 * tied) / total


def calibration(predicted: FloatArray, observed: FloatArray) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(predicted), predicted])
    coef, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    return slope, intercept
