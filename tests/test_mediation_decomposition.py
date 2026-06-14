from __future__ import annotations

import numpy as np

from cts_cm.aperture.frames import CohortFrame
from cts_cm.instruments.interferometry import InterventionalMediation, mediation_sensitivity
from cts_cm.instruments.spectroscopy import GBNuisance


def _composite(cohort: CohortFrame) -> np.ndarray:
    assert cohort.group is not None
    severity = cohort.group.astype(np.float64)
    return (severity - severity.mean()) / severity.std()


def test_bmi_pathway_is_metabolic_dominant(cohort: CohortFrame) -> None:
    outcome = _composite(cohort)
    mediation = InterventionalMediation(GBNuisance(max_depth=3, rounds=60), mc_draws=24, seed=0)
    bmi = cohort.treatments["bmi"]
    high, low = float(bmi.mean() + bmi.std()), float(bmi.mean() - bmi.std())
    shares = mediation.decompose(
        bmi, cohort.mechanical, cohort.metabolic, cohort.covariates, outcome, high, low
    )
    assert shares["metabolic"] > shares["mechanical"]
    total = shares["mechanical"] + shares["metabolic"] + shares["direct"]
    assert abs(total - 1.0) < 1e-6


def test_sensitivity_attenuates_pathways() -> None:
    base = {"mechanical": 0.25, "metabolic": 0.70, "direct": 0.05}
    table = mediation_sensitivity(base, (0.0, 0.5))
    assert table[0.0]["metabolic"] == base["metabolic"]
    assert table[0.5]["metabolic"] < base["metabolic"]
    for shares in table.values():
        assert abs(sum(shares.values()) - 1.0) < 1e-9
