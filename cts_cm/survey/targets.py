from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xgboost as xgb

from cts_cm.almanac.tables import Settings
from cts_cm.aperture.frames import CohortFrame, EffectEstimate, FloatArray, IntArray
from cts_cm.instruments.interferometry import InterventionalMediation
from cts_cm.instruments.spectroscopy import DoubleMLEstimator, GBNuisance, metalearner


def severity_composite(responsibilities: FloatArray) -> FloatArray:
    ladder = np.arange(responsibilities.shape[1], dtype=np.float64)
    raw = responsibilities @ ladder
    return (raw - float(raw.mean())) / (float(raw.std()) or 1.0)


def shifting_effect(probability_x: FloatArray, probability_xprime: FloatArray) -> FloatArray:
    difference: FloatArray = probability_x - probability_xprime
    return difference


def _nuisance(settings: Settings) -> GBNuisance:
    return GBNuisance(
        max_depth=settings.dml.xgb_max_depth,
        lr=settings.dml.xgb_lr,
        rounds=settings.dml.xgb_rounds,
        seed=settings.runtime.seed,
    )


def _is_binary(treatment: FloatArray) -> bool:
    return bool(np.isin(np.unique(treatment), (0.0, 1.0)).all())


def _binarize(treatment: FloatArray) -> FloatArray:
    if _is_binary(treatment):
        return treatment
    return (treatment >= float(np.median(treatment))).astype(np.float64)


def _contrast(treatment: FloatArray) -> tuple[float, float]:
    if _is_binary(treatment):
        return 1.0, 0.0
    mean = float(treatment.mean())
    spread = float(treatment.std()) or 1.0
    return mean + spread, mean - spread


@dataclass
class MembershipShift:
    nuisance: GBNuisance
    n_groups: int

    def __post_init__(self) -> None:
        self._model: Any | None = None

    def fit(
        self, group: IntArray, treatment: FloatArray, covariates: FloatArray
    ) -> MembershipShift:
        design = np.column_stack([treatment, covariates])
        model = xgb.XGBClassifier(
            max_depth=self.nuisance.max_depth,
            learning_rate=self.nuisance.lr,
            n_estimators=self.nuisance.rounds,
            objective="multi:softprob",
            num_class=self.n_groups,
            n_jobs=1,
            random_state=self.nuisance.seed,
            verbosity=0,
        )
        model.fit(design, group)
        self._model = model
        return self

    def do_probability(self, treatment_value: float, covariates: FloatArray) -> FloatArray:
        assert self._model is not None
        n = covariates.shape[0]
        design = np.column_stack([np.full(n, treatment_value), covariates])
        proba = np.asarray(self._model.predict_proba(design), dtype=np.float64)
        averaged: FloatArray = proba.mean(axis=0)
        return averaged

    def shift(
        self, treatment_x: float, treatment_xprime: float, covariates: FloatArray
    ) -> FloatArray:
        return shifting_effect(
            self.do_probability(treatment_x, covariates),
            self.do_probability(treatment_xprime, covariates),
        )


def population_effects(
    cohort: CohortFrame, outcome: FloatArray, settings: Settings
) -> dict[str, EffectEstimate]:
    estimator = DoubleMLEstimator(
        nuisance=_nuisance(settings), n_folds=settings.dml.n_folds, seed=settings.runtime.seed
    )
    results: dict[str, EffectEstimate] = {}
    for factor in settings.risk_factors:
        treatment = cohort.treatments[factor]
        theta, se = estimator.ate(outcome, treatment, cohort.covariates)
        naive = estimator.naive_slope(outcome, treatment)
        attenuation = (
            float((abs(naive) - abs(theta)) / abs(naive) * 100.0)
            if abs(naive) > 1e-9
            else float("nan")
        )
        results[factor] = EffectEstimate(
            factor=factor,
            theta=theta,
            se=se,
            ci_low=theta - 1.96 * se,
            ci_high=theta + 1.96 * se,
            naive=naive,
            attenuation=attenuation,
        )
    return results


def trajectory_effects(
    cohort: CohortFrame, outcome: FloatArray, groups: IntArray, settings: Settings
) -> dict[str, dict[int, float]]:
    nuisance = _nuisance(settings)
    matrix: dict[str, dict[int, float]] = {}
    for factor in settings.risk_factors:
        exposure = _binarize(cohort.treatments[factor])
        cate = metalearner(
            settings.forest.metalearner,
            outcome,
            exposure,
            cohort.covariates,
            nuisance,
            n_folds=settings.dml.n_folds,
            seed=settings.runtime.seed,
        )
        per_group: dict[int, float] = {}
        for group in np.unique(groups):
            per_group[int(group)] = float(cate[groups == group].mean())
        matrix[factor] = per_group
    return matrix


def pathway_decomposition(
    cohort: CohortFrame, outcome: FloatArray, settings: Settings
) -> dict[str, dict[str, float]]:
    mediation = InterventionalMediation(
        nuisance=_nuisance(settings),
        mc_draws=settings.mediation.mc_draws,
        seed=settings.runtime.seed,
    )
    shares: dict[str, dict[str, float]] = {}
    for factor in settings.risk_factors:
        column = cohort.treatments[factor]
        high, low = _contrast(column)
        shares[factor] = mediation.decompose(
            column, cohort.mechanical, cohort.metabolic, cohort.covariates, outcome, high, low
        )
    return shares


def prescribe(shares: dict[str, float]) -> str:
    pathways = {key: shares[key] for key in ("mechanical", "metabolic") if key in shares}
    return max(pathways, key=lambda key: pathways[key])
