from __future__ import annotations

import numpy as np

from cts_cm.reduction.resampling import posterior_bootstrap


def _mean(values: np.ndarray) -> float:
    return float(values.mean())


def test_bootstrap_is_deterministic_under_seed() -> None:
    values = np.random.default_rng(0).normal(size=500)
    first = posterior_bootstrap(values, _mean, n_boot=200, seed=42)
    second = posterior_bootstrap(values, _mean, n_boot=200, seed=42)
    assert first == second


def test_interval_brackets_point_estimate() -> None:
    values = np.random.default_rng(1).normal(loc=0.3, size=800)
    point, low, high = posterior_bootstrap(values, _mean, n_boot=400, seed=7)
    assert low <= point <= high
    assert high - low > 0.0


def test_posterior_weights_shift_distribution() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0])
    weights = np.array([0.01, 0.01, 0.01, 0.97])
    _, weighted_low, _ = posterior_bootstrap(values, _mean, n_boot=300, seed=3, weights=weights)
    _, uniform_low, _ = posterior_bootstrap(values, _mean, n_boot=300, seed=3)
    assert weighted_low > uniform_low
