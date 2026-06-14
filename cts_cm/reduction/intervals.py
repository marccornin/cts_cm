from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cts_cm.aperture.frames import FloatArray, IntArray


@dataclass
class SplitConformal:
    alpha: float = 0.05

    def calibrate(self, prediction: FloatArray, truth: FloatArray) -> float:
        scores = np.abs(truth - prediction)
        n = scores.shape[0]
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        return float(np.quantile(scores, min(level, 1.0), method="higher"))

    def interval(self, prediction: FloatArray, radius: float) -> tuple[FloatArray, FloatArray]:
        return prediction - radius, prediction + radius


def coverage_report(
    prediction: FloatArray,
    truth: FloatArray,
    radius: float,
    groups: IntArray | None = None,
) -> dict[str, float]:
    covered = np.abs(truth - prediction) <= radius
    report: dict[str, float] = {
        "coverage": float(covered.mean()),
        "median_width": float(2.0 * radius),
    }
    if groups is not None:
        for value in np.unique(groups):
            mask = groups == value
            report[f"coverage_g{int(value)}"] = float(covered[mask].mean())
    return report
