from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cts_cm.aperture.frames import FloatArray


def posterior_bootstrap(
    values: FloatArray,
    statistic: Callable[[FloatArray], float],
    n_boot: int = 1000,
    seed: int = 0,
    weights: FloatArray | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    gen = np.random.default_rng(seed)
    n = values.shape[0]
    probabilities = None if weights is None else weights / float(weights.sum())
    samples = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        index = gen.choice(n, size=n, replace=True, p=probabilities)
        samples[draw] = statistic(values[index])
    point = statistic(values)
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return point, low, high
