from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from cts_cm.aperture.frames import FloatArray

Edge = tuple[str, str]


def benjamini_hochberg(
    pvalues: Sequence[float], alpha: float = 0.05
) -> tuple[NDArray[np.bool_], FloatArray]:
    values = np.asarray(pvalues, dtype=np.float64)
    n = values.shape[0]
    order = np.argsort(values)
    ranks = np.arange(1, n + 1)
    ordered = values[order]
    adjusted_ordered = np.minimum.accumulate((ordered * n / ranks)[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    rejected = adjusted <= alpha
    return rejected, adjusted


def e_value(risk_ratio: float) -> float:
    oriented = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return float(oriented + np.sqrt(oriented * (oriented - 1.0)))


def structural_hamming_distance(left: set[Edge], right: set[Edge]) -> int:
    return len(left ^ right)


def wald_heterogeneity(
    estimates: Sequence[float], standard_errors: Sequence[float]
) -> tuple[float, float]:
    effect = np.asarray(estimates, dtype=np.float64)
    variance = np.asarray(standard_errors, dtype=np.float64) ** 2
    weights = 1.0 / variance
    pooled = float((weights * effect).sum() / weights.sum())
    statistic = float((weights * (effect - pooled) ** 2).sum())
    degrees = effect.shape[0] - 1
    pvalue = float(stats.chi2.sf(statistic, degrees))
    return statistic, pvalue


def placebo_pvalue(observed: float, null_samples: Sequence[float]) -> float:
    draws = np.asarray(null_samples, dtype=np.float64)
    extreme = float((np.abs(draws) >= abs(observed)).sum())
    return float((extreme + 1.0) / (draws.shape[0] + 1.0))
