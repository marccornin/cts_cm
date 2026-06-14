from __future__ import annotations

import numpy as np
from cyclopts import App

from cts_cm.almanac.reader import load_settings
from cts_cm.almanac.tables import Settings
from cts_cm.aperture.sources import load_cohort
from cts_cm.instruments.lightcurves import TrajectoryMixture, select_group_count
from cts_cm.survey.campaign import Report, run_all

app = App(name="cts-cm", help="CTS-CM trajectory-shifting causal pipeline")

_DEFAULT = "configs/experiment/main.yaml"


def _settings(config: str, overrides: list[str] | None) -> Settings:
    return load_settings(config, overrides or [])


@app.command(name="synthesize")
def synthesize(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    settings = _settings(config, set_)
    cohort = load_cohort(
        cohort="synthetic_oai",
        n=settings.data.n,
        n_timepoints=settings.data.n_timepoints,
        horizon_years=settings.data.horizon_years,
        seed=settings.runtime.seed,
    )
    print(f"cohort={cohort.cohort} n={cohort.n} timepoints={cohort.n_timepoints}")
    if cohort.group is not None:
        for value in np.unique(cohort.group):
            print(f"  group {int(value)}: {float((cohort.group == value).mean()):.3f}")


@app.command(name="fit-trajectories")
def fit_trajectories(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    settings = _settings(config, set_)
    cohort = load_cohort(
        cohort="synthetic_oai",
        n=settings.data.n,
        n_timepoints=settings.data.n_timepoints,
        horizon_years=settings.data.horizon_years,
        seed=settings.runtime.seed,
    )
    best_k, scores = select_group_count(
        cohort.pain,
        cohort.times,
        grid=settings.trajectory.group_grid,
        poly_degree=settings.trajectory.poly_degree,
        n_starts=settings.trajectory.n_starts,
        seed=settings.runtime.seed,
    )
    model = TrajectoryMixture(
        n_groups=best_k, poly_degree=settings.trajectory.poly_degree, seed=settings.runtime.seed
    ).fit(cohort.pain, cohort.times)
    groups = model.predict(cohort.pain, cohort.times)
    print(f"selected_groups={best_k}")
    for k, value in scores.items():
        print(f"  bic[{k}]={value:.1f}")
    for value in np.unique(groups):
        print(f"  group {int(value)}: {float((groups == value).mean()):.3f}")


def _print_population(report: Report) -> None:
    print("population average treatment effects:")
    for factor, values in report.population.items():
        print(
            f"  {factor}: theta={values['theta']:.3f} "
            f"ci=[{values['ci_low']:.3f},{values['ci_high']:.3f}] "
            f"attenuation={values['attenuation']:.1f}%"
        )


def _print_pathways(report: Report) -> None:
    print("pathway decomposition:")
    for factor, shares in report.pathways.items():
        print(
            f"  {factor}: mechanical={shares['mechanical']:.3f} "
            f"metabolic={shares['metabolic']:.3f} direct={shares['direct']:.3f} "
            f"-> {report.prescriptions[factor]}"
        )


def _print_validation(report: Report) -> None:
    print("validation:")
    print(f"  conformal coverage={report.conformal['coverage']:.3f}")
    print(
        f"  c-index internal={report.external['internal_c_index']:.3f} "
        f"external={report.external['external_c_index']:.3f} "
        f"degradation={report.external['degradation_pct']:.1f}% "
        f"mr_concordance={report.external['mr_concordance']:.2f}"
    )


@app.command(name="estimate")
def estimate(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    _print_population(run_all(_settings(config, set_)))


@app.command(name="mediate")
def mediate(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    _print_pathways(run_all(_settings(config, set_)))


@app.command(name="validate")
def validate(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    _print_validation(run_all(_settings(config, set_)))


@app.command(name="run-all")
def run_all_command(config: str = _DEFAULT, set_: list[str] | None = None) -> None:
    report = run_all(_settings(config, set_))
    _print_population(report)
    _print_pathways(report)
    _print_validation(report)
    print(f"groups={report.group_count}")


def main(argv: list[str] | None = None) -> int:
    app(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
