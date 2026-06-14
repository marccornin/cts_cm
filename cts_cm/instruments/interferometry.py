from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cts_cm.aperture.frames import FloatArray
from cts_cm.instruments.spectroscopy import GBNuisance


def _predict(model: Any, features: FloatArray) -> FloatArray:
    return np.asarray(model.predict(features), dtype=np.float64)


@dataclass
class InterventionalMediation:
    nuisance: GBNuisance
    mc_draws: int = 200
    seed: int = 0

    def _fit_mediators(
        self, treatment: FloatArray, covariates: FloatArray, mediators: FloatArray
    ) -> tuple[list[Any], FloatArray]:
        design = np.column_stack([treatment, covariates])
        models: list[Any] = []
        spreads = np.zeros(mediators.shape[1], dtype=np.float64)
        for column in range(mediators.shape[1]):
            model = self.nuisance.regressor()
            model.fit(design, mediators[:, column])
            residual = mediators[:, column] - _predict(model, design)
            spreads[column] = float(residual.std()) or 1.0
            models.append(model)
        return models, spreads

    def _draw(
        self,
        models: Sequence[Any],
        spreads: FloatArray,
        treatment_value: float,
        covariates: FloatArray,
        gen: np.random.Generator,
    ) -> FloatArray:
        n = covariates.shape[0]
        design = np.column_stack([np.full(n, treatment_value), covariates])
        columns = [
            _predict(model, design) + spreads[idx] * gen.standard_normal(n)
            for idx, model in enumerate(models)
        ]
        return np.column_stack(columns)

    def decompose(
        self,
        treatment: FloatArray,
        mechanical: FloatArray,
        metabolic: FloatArray,
        covariates: FloatArray,
        outcome: FloatArray,
        high: float = 1.0,
        low: float = 0.0,
    ) -> dict[str, float]:
        gen = np.random.default_rng(self.seed)
        outcome_model = self.nuisance.regressor()
        observed = np.column_stack([treatment, mechanical, metabolic, covariates])
        outcome_model.fit(observed, outcome)

        mech_models, mech_spread = self._fit_mediators(treatment, covariates, mechanical)
        metab_models, metab_spread = self._fit_mediators(treatment, covariates, metabolic)

        def evaluate(t: float, m1: FloatArray, m2: FloatArray) -> FloatArray:
            n = covariates.shape[0]
            design = np.column_stack([np.full(n, t), m1, m2, covariates])
            return _predict(outcome_model, design)

        mech_total = 0.0
        metab_total = 0.0
        direct_total = 0.0
        for _ in range(self.mc_draws):
            m1_high = self._draw(mech_models, mech_spread, high, covariates, gen)
            m1_low = self._draw(mech_models, mech_spread, low, covariates, gen)
            m2_high = self._draw(metab_models, metab_spread, high, covariates, gen)
            m2_low = self._draw(metab_models, metab_spread, low, covariates, gen)
            mech_total += float(
                (evaluate(high, m1_high, m2_low) - evaluate(high, m1_low, m2_low)).mean()
            )
            metab_total += float(
                (evaluate(high, m1_low, m2_high) - evaluate(high, m1_low, m2_low)).mean()
            )
            direct_total += float(
                (evaluate(high, m1_low, m2_low) - evaluate(low, m1_low, m2_low)).mean()
            )

        mechanical_effect = mech_total / self.mc_draws
        metabolic_effect = metab_total / self.mc_draws
        direct_effect = direct_total / self.mc_draws
        total = mechanical_effect + metabolic_effect + direct_effect
        scale = total if abs(total) > 1e-9 else 1.0
        return {
            "mechanical": mechanical_effect / scale,
            "metabolic": metabolic_effect / scale,
            "direct": direct_effect / scale,
            "total_effect": total,
        }


def mediation_sensitivity(
    shares: dict[str, float], rho_grid: Sequence[float]
) -> dict[float, dict[str, float]]:
    table: dict[float, dict[str, float]] = {}
    for rho in rho_grid:
        attenuation = 1.0 - rho
        adjusted_metabolic = shares["metabolic"] * attenuation
        adjusted_mechanical = shares["mechanical"] * attenuation
        residual = 1.0 - adjusted_metabolic - adjusted_mechanical
        table[rho] = {
            "mechanical": adjusted_mechanical,
            "metabolic": adjusted_metabolic,
            "direct": residual,
        }
    return table
