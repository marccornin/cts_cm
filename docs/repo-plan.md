# Repo Plan

## Organising metaphor

The package is laid out as an astronomical observatory that surveys patient pain trajectories the
way a sky survey records light curves. Cohorts enter through the `aperture`; the `instruments` record
and reduce each signal (the five estimation stages); the `reduction` stage attaches uncertainty and
checks that signals transport between fields (cohorts); `magnitudes` are the measured quantities
(metrics); the `dome` keeps the run reproducible (seeding, atomic logbook); the `survey` plans and
executes an observing campaign (the estimands and orchestration); the `observer` is the control
console (CLI); the `almanac` holds the reference tables (typed configuration). "Trajectory" maps to a
light curve, the time-series the instruments fit.

## Directory tree

    cts_cm/
      __init__.py
      almanac/
        __init__.py
        tables.py          # frozen dataclasses: Settings and nested blocks
        reader.py          # typed YAML -> dataclass loader, CLI key=value overrides
      aperture/
        __init__.py
        frames.py          # typed containers: CohortFrame, EffectEstimate, array aliases
        sources.py         # CohortSource Protocol + CsvCohort + cohort descriptors
        simulator.py       # deterministic SyntheticCohort matching Tables I-IV moments
        infill.py          # ChainedImputer (MICE-style) + rubin_pool
      instruments/
        __init__.py
        lightcurves.py     # Component 1: TrajectoryMixture (EM), BIC selection
        astrometry.py      # Component 2: CausalGraph, stability selection, adjustment sets
        spectroscopy.py    # Component 3: DoubleMLEstimator, GBNuisance, meta-learners, forest
        interferometry.py  # Component 4: InterventionalMediation, sensitivity
      reduction/
        __init__.py
        intervals.py       # Component 5a: SplitConformal, coverage_report
        resampling.py      # Component 5b: posterior_bootstrap
        portability.py     # Component 5c: transfer_degradation, mr_triangulation
      magnitudes/
        __init__.py
        ranking.py         # c_index, calibration
        testing.py         # benjamini_hochberg, e_value, refutation, SHD, Wald
      dome/
        __init__.py
        seeds.py           # set_seed, SeedState
        logbook.py         # atomic_write, RunLedger, get_logger
      survey/
        __init__.py
        targets.py         # shifting_effect, population/trajectory effects, pathway decomposition, prescribe
        campaign.py        # run_all orchestration, Report dataclass
      observer/
        __init__.py
        __main__.py        # cyclopts app: synthesize/fit-trajectories/estimate/mediate/validate/run-all
    configs/
      data/{oai,most,ukb,synthetic}.yaml
      model/{gbtm,dml,causal_forest,mediation,conformal}.yaml
      experiment/{main,ablation_metalearner,ablation_ngroups,ablation_pathway,
                  supplementary_mr,supplementary_sensitivity,_smoke}.yaml
    tests/  (recovery, orthogonality, shapes, coverage, mediation, graph, BH/E-value,
             bootstrap, ledger, e2e smoke, style guard)
    docs/{project-context,implementation-map,repo-plan,deviations}.md
    scripts/{prepare_data.sh,run_main.sh,run_validation.sh}
    README.md LICENSE pyproject.toml requirements.txt environment.yml
    Dockerfile Makefile .gitignore .pre-commit-config.yaml .github/workflows/ci.yml
    assets/

## Configuration and command stack

Configuration is a frozen `Settings` dataclass (`almanac/tables.py`) composed of `DataCfg`,
`TrajectoryCfg`, `DmlCfg`, `ForestCfg`, `MediationCfg`, `ConformalCfg`, `ResampleCfg`, `RuntimeCfg`.
`almanac/reader.py` loads `configs/experiment/<name>.yaml` into `Settings` with a small typed
recursive loader and applies dotted `key=value` overrides last; absent keys fall back to the
dataclass defaults, which equal the manuscript's main values. The command console (`observer`) is a
`cyclopts` application; each subcommand takes `--config` and repeatable `--set key=value` flags.
`configs/experiment/_smoke.yaml` is labelled for unit-test use only.

## Dependencies (pinned ranges)

Runtime: python >=3.10,<3.14; numpy>=1.26; scipy>=1.11; pandas>=2.1; scikit-learn>=1.3;
xgboost>=2.0; statsmodels>=0.14; cyclopts>=2.9; PyYAML>=6.0.
Dev: pytest>=8; ruff>=0.5; black>=24.3; isort>=5.13; mypy>=1.10; pre-commit>=3.7.

No torch, no econml, no R bridge: every estimator runs on the scikit-learn / xgboost substrate.

## Test coverage plan

Planted-group recovery and single-block overfit; double-machine-learning unbiasedness and
nuisance-orthogonality; 4x4 CATE and prescription shapes; conformal coverage at nominal level;
mediation pathway-share conservation; graph structural-distance; Benjamini-Hochberg agreement with
`statsmodels`; bootstrap determinism; atomic-checkpoint seed round-trip; end-to-end smoke; and a
source-style guard (no comments, no docstrings, no forbidden phrases or emoji).

## Deviations policy

Faithful mechanism substitutions (honest-split forest standing in for grf; EM mixture standing in for
lcmm; Gaussian working likelihood for the generalized propensity) are logged in `docs/deviations.md`
with the paper section and justification. The file is created up front.
