from __future__ import annotations

import numpy as np
import pytest

from cts_cm.instruments.spectroscopy import GBNuisance, HonestCausalForest, metalearner


def _binary_scenario(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    n = 800
    covariates = gen.normal(size=(n, 4))
    propensity = 1.0 / (1.0 + np.exp(-covariates[:, 0]))
    treatment = gen.binomial(1, propensity).astype(np.float64)
    effect = 0.5 + covariates[:, 1]
    outcome = effect * treatment + covariates[:, 0] + gen.normal(size=n)
    return outcome, treatment, covariates


@pytest.mark.parametrize("name", ["S", "T", "X", "DR", "R"])
def test_metalearner_returns_per_row_cate(name: str) -> None:
    outcome, treatment, covariates = _binary_scenario()
    cate = metalearner(
        name, outcome, treatment, covariates, GBNuisance(max_depth=3, rounds=60), n_folds=3
    )
    assert cate.shape == (outcome.shape[0],)
    assert np.isfinite(cate).all()


def test_honest_forest_effect_shape() -> None:
    outcome, treatment, covariates = _binary_scenario(2)
    forest = HonestCausalForest(GBNuisance(max_depth=3, rounds=60), n_trees=128, seed=2).fit(
        outcome, treatment, covariates
    )
    effect = forest.effect(covariates)
    assert effect.shape == (outcome.shape[0],)
    assert np.isfinite(effect).all()


def test_unknown_metalearner_rejected() -> None:
    outcome, treatment, covariates = _binary_scenario(3)
    with pytest.raises(ValueError):
        metalearner("Z", outcome, treatment, covariates, GBNuisance(rounds=10))
