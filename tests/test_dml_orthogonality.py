from __future__ import annotations

import numpy as np

from cts_cm.instruments.spectroscopy import DoubleMLEstimator, GBNuisance


def _scenario(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    gen = np.random.default_rng(seed)
    n = 2000
    covariates = gen.normal(size=(n, 3))
    confounding = covariates[:, 0] + 0.5 * covariates[:, 1] ** 2
    treatment = 0.8 * covariates[:, 0] + gen.normal(size=n)
    theta = 1.5
    outcome = theta * treatment + confounding + gen.normal(size=n)
    return outcome, treatment, covariates, theta


def test_dml_recovers_known_effect() -> None:
    outcome, treatment, covariates, theta = _scenario(0)
    estimator = DoubleMLEstimator(GBNuisance(max_depth=3, rounds=120, seed=0), n_folds=3, seed=0)
    estimate, se = estimator.ate(outcome, treatment, covariates)
    assert abs(estimate - theta) < 0.2
    assert se > 0.0


def test_dml_beats_naive_under_confounding() -> None:
    outcome, treatment, covariates, theta = _scenario(1)
    estimator = DoubleMLEstimator(GBNuisance(max_depth=3, rounds=120, seed=1), n_folds=3, seed=1)
    estimate, _ = estimator.ate(outcome, treatment, covariates)
    naive = estimator.naive_slope(outcome, treatment)
    assert abs(estimate - theta) < abs(naive - theta)
