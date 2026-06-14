from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold

from cts_cm.aperture.frames import FloatArray

Regressor = Callable[[], Any]


@dataclass
class GBNuisance:
    max_depth: int = 6
    lr: float = 0.1
    rounds: int = 500
    subsample: float = 0.8
    seed: int = 0

    def regressor(self) -> Any:
        return xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.lr,
            n_estimators=self.rounds,
            subsample=self.subsample,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=self.seed,
            verbosity=0,
        )


def _fit_predict(
    factory: Regressor, features: FloatArray, target: FloatArray, query: FloatArray
) -> FloatArray:
    model = factory()
    model.fit(features, target)
    return np.asarray(model.predict(query), dtype=np.float64)


def _cross_fit(
    factory: Regressor, features: FloatArray, target: FloatArray, n_folds: int, seed: int
) -> FloatArray:
    out = np.zeros(target.shape[0], dtype=np.float64)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(features):
        out[test_idx] = _fit_predict(
            factory, features[train_idx], target[train_idx], features[test_idx]
        )
    return out


@dataclass
class DoubleMLEstimator:
    nuisance: GBNuisance
    n_folds: int = 5
    seed: int = 0

    def ate(
        self, outcome: FloatArray, treatment: FloatArray, covariates: FloatArray
    ) -> tuple[float, float]:
        factory: Regressor = self.nuisance.regressor
        outcome_hat = _cross_fit(factory, covariates, outcome, self.n_folds, self.seed)
        treatment_hat = _cross_fit(factory, covariates, treatment, self.n_folds, self.seed + 1)
        outcome_res = outcome - outcome_hat
        treatment_res = treatment - treatment_hat
        denom = float((treatment_res**2).mean())
        theta = float((treatment_res * outcome_res).mean() / denom)
        influence = treatment_res * (outcome_res - theta * treatment_res) / denom
        se = float(np.sqrt((influence**2).mean() / outcome.shape[0]))
        return theta, se

    def naive_slope(self, outcome: FloatArray, treatment: FloatArray) -> float:
        centered_t = treatment - float(treatment.mean())
        centered_y = outcome - float(outcome.mean())
        return float((centered_t * centered_y).sum() / (centered_t**2).sum())


def _propensity(
    covariates: FloatArray,
    treatment: FloatArray,
    factory: Regressor,
    n_folds: int,
    seed: int,
    trim: float,
) -> FloatArray:
    raw = _cross_fit(factory, covariates, treatment, n_folds, seed)
    return np.clip(raw, trim, 1.0 - trim)


def _t_learner(
    outcome: FloatArray, treatment: FloatArray, covariates: FloatArray, factory: Regressor
) -> tuple[FloatArray, FloatArray]:
    treated = treatment > 0.5
    mu1 = _fit_predict(factory, covariates[treated], outcome[treated], covariates)
    mu0 = _fit_predict(factory, covariates[~treated], outcome[~treated], covariates)
    return mu0, mu1


def metalearner(
    name: str,
    outcome: FloatArray,
    treatment: FloatArray,
    covariates: FloatArray,
    nuisance: GBNuisance,
    n_folds: int = 5,
    seed: int = 0,
    trim: float = 0.01,
) -> FloatArray:
    factory: Regressor = nuisance.regressor
    key = name.upper()
    if key == "S":
        design = np.column_stack([covariates, treatment])
        treated_design = np.column_stack([covariates, np.ones(treatment.shape[0])])
        control_design = np.column_stack([covariates, np.zeros(treatment.shape[0])])
        model = factory()
        model.fit(design, outcome)
        high = np.asarray(model.predict(treated_design), dtype=np.float64)
        low = np.asarray(model.predict(control_design), dtype=np.float64)
        return high - low
    mu0, mu1 = _t_learner(outcome, treatment, covariates, factory)
    if key == "T":
        return mu1 - mu0
    propensity = _propensity(covariates, treatment, factory, n_folds, seed, trim)
    if key == "X":
        treated = treatment > 0.5
        imputed_treated = outcome[treated] - mu0[treated]
        imputed_control = mu1[~treated] - outcome[~treated]
        tau1 = _fit_predict(factory, covariates[treated], imputed_treated, covariates)
        tau0 = _fit_predict(factory, covariates[~treated], imputed_control, covariates)
        return propensity * tau0 + (1.0 - propensity) * tau1
    if key == "DR":
        pseudo = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / propensity
            - (1.0 - treatment) * (outcome - mu0) / (1.0 - propensity)
        )
        return _fit_predict(factory, covariates, pseudo, covariates)
    if key == "R":
        outcome_hat = _cross_fit(factory, covariates, outcome, n_folds, seed + 2)
        outcome_res = outcome - outcome_hat
        treatment_res = treatment - propensity
        weight = treatment_res**2 + 1e-8
        pseudo = outcome_res / (treatment_res + np.sign(treatment_res) * 1e-3 + 1e-9)
        model = ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1
        )
        model.fit(covariates, pseudo, sample_weight=weight)
        return np.asarray(model.predict(covariates), dtype=np.float64)
    raise ValueError(f"unknown meta-learner {name}")


@dataclass
class HonestCausalForest:
    nuisance: GBNuisance
    n_trees: int = 2000
    min_leaf: int = 5
    honest_frac: float = 0.5
    max_features: float = 0.5
    n_folds: int = 5
    seed: int = 0
    _forest: ExtraTreesRegressor | None = None

    def fit(
        self, outcome: FloatArray, treatment: FloatArray, covariates: FloatArray
    ) -> HonestCausalForest:
        factory: Regressor = self.nuisance.regressor
        gen = np.random.default_rng(self.seed)
        order = gen.permutation(outcome.shape[0])
        split = int(self.honest_frac * order.shape[0])
        nuisance_idx, leaf_idx = order[:split], order[split:]
        mu0 = _fit_predict(
            factory, covariates[nuisance_idx], outcome[nuisance_idx], covariates[leaf_idx]
        )
        propensity = np.clip(
            _fit_predict(
                factory, covariates[nuisance_idx], treatment[nuisance_idx], covariates[leaf_idx]
            ),
            0.01,
            0.99,
        )
        residual_t = treatment[leaf_idx] - propensity
        pseudo = (outcome[leaf_idx] - mu0) * residual_t / (residual_t**2 + 1e-6)
        forest = ExtraTreesRegressor(
            n_estimators=self.n_trees,
            min_samples_leaf=self.min_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=1,
        )
        forest.fit(covariates[leaf_idx], pseudo)
        self._forest = forest
        return self

    def effect(self, covariates: FloatArray) -> FloatArray:
        assert self._forest is not None
        return np.asarray(self._forest.predict(covariates), dtype=np.float64)
