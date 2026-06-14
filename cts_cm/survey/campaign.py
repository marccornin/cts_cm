from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cts_cm.almanac.tables import Settings
from cts_cm.aperture.frames import CohortFrame, FloatArray
from cts_cm.aperture.sources import load_cohort
from cts_cm.dome.logbook import RunLedger, get_logger
from cts_cm.dome.seeds import set_seed
from cts_cm.instruments.lightcurves import TrajectoryMixture, select_group_count
from cts_cm.instruments.spectroscopy import GBNuisance
from cts_cm.magnitudes.ranking import c_index
from cts_cm.reduction.intervals import SplitConformal, coverage_report
from cts_cm.reduction.portability import mr_triangulation, transfer_degradation
from cts_cm.survey.targets import (
    pathway_decomposition,
    population_effects,
    prescribe,
    severity_composite,
    trajectory_effects,
)

_MR_ODDS_RATIOS: dict[str, float] = {
    "bmi": 1.91,
    "physical_activity": 0.85,
    "occupational_loading": 1.52,
    "metabolic_syndrome": 1.15,
}


@dataclass
class Report:
    name: str
    group_count: int
    group_bic: dict[int, float]
    group_sizes: dict[int, float]
    population: dict[str, dict[str, float]]
    trajectory: dict[str, dict[int, float]]
    pathways: dict[str, dict[str, float]]
    prescriptions: dict[str, str]
    conformal: dict[str, float]
    external: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


def _predictor(cohort: CohortFrame, outcome: FloatArray, settings: Settings) -> Any:
    nuisance = GBNuisance(
        max_depth=settings.dml.xgb_max_depth,
        lr=settings.dml.xgb_lr,
        rounds=settings.dml.xgb_rounds,
        seed=settings.runtime.seed,
    )
    features = np.column_stack([cohort.treatments[name] for name in settings.risk_factors])
    design = np.column_stack([features, cohort.covariates])
    model = nuisance.regressor()
    model.fit(design, outcome)
    return model


def _score(model: Any, cohort: CohortFrame, settings: Settings) -> FloatArray:
    features = np.column_stack([cohort.treatments[name] for name in settings.risk_factors])
    design = np.column_stack([features, cohort.covariates])
    return np.asarray(model.predict(design), dtype=np.float64)


def run_all(settings: Settings) -> Report:
    logger = get_logger("cts_cm.survey", settings.runtime.log_level)
    set_seed(settings.runtime.seed)
    ledger = RunLedger(output_dir=settings.runtime.output_dir, seed=settings.runtime.seed)

    cohort = load_cohort(
        cohort="synthetic_oai",
        n=settings.data.n,
        n_timepoints=settings.data.n_timepoints,
        horizon_years=settings.data.horizon_years,
        seed=settings.runtime.seed,
    )
    logger.info("loaded %s cohort with %d participants", cohort.cohort, cohort.n)

    best_k, bic_scores = select_group_count(
        cohort.pain,
        cohort.times,
        grid=settings.trajectory.group_grid,
        poly_degree=settings.trajectory.poly_degree,
        n_starts=settings.trajectory.n_starts,
        seed=settings.runtime.seed,
        min_group_frac=settings.trajectory.min_group_frac,
        min_posterior=settings.trajectory.min_posterior,
    )
    mixture = TrajectoryMixture(
        n_groups=best_k,
        poly_degree=settings.trajectory.poly_degree,
        n_starts=settings.trajectory.n_starts,
        seed=settings.runtime.seed,
    ).fit(cohort.pain, cohort.times)
    responsibilities = mixture.responsibilities(cohort.pain, cohort.times)
    groups = responsibilities.argmax(axis=1).astype(np.int64)
    outcome = severity_composite(responsibilities)
    sizes = {int(k): float((groups == k).mean()) for k in range(best_k)}

    population = population_effects(cohort, outcome, settings)
    population_serialised = {factor: _estimate_dict(est) for factor, est in population.items()}
    trajectory = trajectory_effects(cohort, outcome, groups, settings)
    pathways = pathway_decomposition(cohort, outcome, settings)
    prescriptions = {factor: prescribe(shares) for factor, shares in pathways.items()}

    conformal = _conformal_report(cohort, outcome, groups, settings)
    external = _external_report(cohort, outcome, population_serialised, settings)

    report = Report(
        name=settings.name,
        group_count=best_k,
        group_bic={int(k): float(v) for k, v in bic_scores.items()},
        group_sizes=sizes,
        population=population_serialised,
        trajectory=trajectory,
        pathways=pathways,
        prescriptions=prescriptions,
        conformal=conformal,
        external=external,
    )
    ledger.log_result("report", asdict(report))
    ledger.checkpoint("report.json")
    logger.info("groups=%d main BMI theta=%.3f", best_k, population_serialised["bmi"]["theta"])
    return report


def _estimate_dict(estimate: Any) -> dict[str, float]:
    return {
        "theta": estimate.theta,
        "se": estimate.se,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "naive": estimate.naive,
        "attenuation": estimate.attenuation,
    }


def _conformal_report(
    cohort: CohortFrame, outcome: FloatArray, groups: Any, settings: Settings
) -> dict[str, float]:
    gen = np.random.default_rng(settings.runtime.seed)
    order = gen.permutation(cohort.n)
    calib_size = int(settings.conformal.calib_frac * cohort.n)
    calib_idx, train_idx = order[:calib_size], order[calib_size:]
    train_cohort = _subset(cohort, train_idx)
    model = _predictor(train_cohort, outcome[train_idx], settings)
    calib_cohort = _subset(cohort, calib_idx)
    predicted = _score(model, calib_cohort, settings)
    conformer = SplitConformal(alpha=settings.conformal.alpha)
    radius = conformer.calibrate(predicted, outcome[calib_idx])
    return coverage_report(predicted, outcome[calib_idx], radius, groups[calib_idx])


def _external_report(
    cohort: CohortFrame,
    outcome: FloatArray,
    population: dict[str, dict[str, float]],
    settings: Settings,
) -> dict[str, float]:
    model = _predictor(cohort, outcome, settings)
    internal = c_index(_score(model, cohort, settings), _severity(cohort))
    most = load_cohort(
        cohort="synthetic_most",
        n=settings.data.n // 2,
        n_timepoints=settings.data.n_timepoints,
        horizon_years=settings.data.horizon_years,
        seed=settings.runtime.seed + 7,
        shift=0.4,
    )
    external = c_index(_score(model, most, settings), _severity(most))
    concordant = 0
    for factor, odds in _MR_ODDS_RATIOS.items():
        if mr_triangulation(float(np.exp(population[factor]["theta"])), odds):
            concordant += 1
    return {
        "internal_c_index": internal,
        "external_c_index": external,
        "degradation_pct": transfer_degradation(internal, external),
        "mr_concordance": float(concordant) / len(_MR_ODDS_RATIOS),
    }


def _severity(cohort: CohortFrame) -> FloatArray:
    if cohort.group is not None:
        return cohort.group.astype(np.float64)
    return cohort.covariates[:, 3].astype(np.float64)


def _subset(cohort: CohortFrame, index: Any) -> CohortFrame:
    return CohortFrame(
        cohort=cohort.cohort,
        times=cohort.times,
        pain=cohort.pain[index],
        treatments={name: column[index] for name, column in cohort.treatments.items()},
        mechanical=cohort.mechanical[index],
        metabolic=cohort.metabolic[index],
        covariates=cohort.covariates[index],
        group=None if cohort.group is None else cohort.group[index],
    )
