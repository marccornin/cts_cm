# Implementation Map

Source files carry no inline comments and no docstrings; this table is the single place that
links every paper item to the module that realises it. Columns:
`paper location | equation / figure / table | file | object | notes`.

## Component 1 — Trajectory identification (GBTM)

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §C ¶1-2 | latent finite mixture of polynomial trajectories | `cts_cm/instruments/lightcurves.py` | `TrajectoryMixture` | EM over Gaussian polynomial-in-time means; quadratic/cubic basis |
| §C ¶2 | BIC model selection, 2*loglik - k*ln n | `cts_cm/instruments/lightcurves.py` | `select_group_count` | scans K in 3..6, returns best by BIC |
| §C ¶2 | min group >= 2%, mean posterior >= 0.70 | `cts_cm/instruments/lightcurves.py` | `TrajectoryMixture.admissible` | clinical interpretability gate |
| §B ¶2 / §A | posterior pi_ik = P(G=k | Y_{1:T}) | `cts_cm/instruments/lightcurves.py` | `TrajectoryMixture.responsibilities` | soft assignments propagated downstream |
| Table I | four groups G1..G4 (51.9/27.0/16.0/5.1%) | `cts_cm/aperture/simulator.py` | `SyntheticCohort` | generator reproduces group sizes + baseline moments |

## Component 2 — Causal DAG specification and validation

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §D Eq(2) | mechanical pathway X_BMI -> M1 -> G | `cts_cm/instruments/astrometry.py` | `CausalGraph.mechanical_path` | edge set for M1 mediators |
| §D Eq(3) | metabolic pathway X_BMI -> M2 -> G | `cts_cm/instruments/astrometry.py` | `CausalGraph.metabolic_path` | edge set for M2 mediators |
| §D ¶3 | 32 variables, 5 categories | `cts_cm/instruments/astrometry.py` | `CausalGraph` | nodes typed by category |
| §D ¶3 | bootstrap stability selection >= 80% edges | `cts_cm/instruments/astrometry.py` | `stability_select_edges` | retains edges present in >= 80% resamples |
| §D ¶3 | structural Hamming distance expert vs learned | `cts_cm/magnitudes/testing.py` | `structural_hamming_distance` | SHD between two edge sets |
| §B Eq(1) | back-door adjustment set | `cts_cm/instruments/astrometry.py` | `CausalGraph.adjustment_set` | confounders for each treatment |

## Component 3 — Doubly robust causal effect estimation

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §E Eq(4) | Neyman-orthogonal DML score | `cts_cm/instruments/spectroscopy.py` | `DoubleMLEstimator` | 5-fold cross-fit, influence-function SE |
| §E ¶2 | XGBoost nuisance mu_k, generalized propensity | `cts_cm/instruments/spectroscopy.py` | `GBNuisance` | depth 6, lr 0.1, 500 rounds |
| §E ¶3 | CATE via causal forest, honest splitting | `cts_cm/instruments/spectroscopy.py` | `HonestCausalForest` | sklearn honest-split ensemble |
| §E ¶3 | meta-learners S/T/X/DR/R, X primary | `cts_cm/instruments/spectroscopy.py` | `metalearner` | factory selecting learner by name |
| Table II | ATE per factor + attenuation vs naive | `cts_cm/survey/targets.py` | `population_effects` | naive logistic OR vs DML-adjusted |
| Table III | 4x4 trajectory-specific CATE + ratio G4/G1 | `cts_cm/survey/targets.py` | `trajectory_effects` | per-group CATE matrix |

## Component 4 — Causal mediation decomposition

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §F Eq(5) | interventional indirect effect IIE_j | `cts_cm/instruments/interferometry.py` | `InterventionalMediation.indirect_effect` | Monte-Carlo integration, 200 draws |
| §F ¶3 | proportion mediated per pathway | `cts_cm/instruments/interferometry.py` | `InterventionalMediation.decompose` | IIE_j / (IIE_1+IIE_2+NDE), sums to 1 |
| §F ¶4 | sensitivity to mediator-outcome confounding rho | `cts_cm/instruments/interferometry.py` | `mediation_sensitivity` | rho grid 0..0.5 |
| Table IV | mech/metab/direct % by group; gradient test | `cts_cm/survey/targets.py` | `pathway_decomposition` | per-group pathway share + trend test |

## Component 5 — Uncertainty quantification and external validation

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §G Eq(6) | split-conformal prediction interval | `cts_cm/reduction/intervals.py` | `SplitConformal` | 80/20, nominal 1-alpha |
| §G ¶1 | empirical coverage, interval width | `cts_cm/reduction/intervals.py` | `coverage_report` | per-group coverage + median width |
| §B ¶2 / §H | bootstrap with posterior-weight propagation | `cts_cm/reduction/resampling.py` | `posterior_bootstrap` | 1,000 resamples, pi_ik reweighting |
| §G ¶2 / Table V | OAI->MOST transfer, C-index degradation | `cts_cm/reduction/portability.py` | `transfer_degradation` | internal vs external concordance |
| §G ¶2 / Table V | UK Biobank MR direction concordance | `cts_cm/reduction/portability.py` | `mr_triangulation` | DML OR vs MR OR sign agreement |

## Cross-cutting

| paper | item | file | object | notes |
|-------|------|------|--------|-------|
| §B Eq(1) | trajectory-shifting estimand TE_k(x,x') | `cts_cm/survey/targets.py` | `shifting_effect` | P(G=k|do(x)) - P(G=k|do(x')) |
| Fig 1 | prescription head (dominant-pathway match) | `cts_cm/survey/targets.py` | `prescribe` | argmax pathway share per patient |
| §H | Benjamini-Hochberg FDR <= 0.05 | `cts_cm/magnitudes/testing.py` | `benjamini_hochberg` | matches statsmodels fdr_bh |
| Table VI | E-values for unmeasured confounding | `cts_cm/magnitudes/testing.py` | `e_value` | VanderWeele-Ding formula |
| Table II/VI | refutation tests (placebo, random cause, subset) | `cts_cm/magnitudes/testing.py` | `refutation_tests` | P>0.05 supports identification |
| §H | MICE chained imputation + Rubin pooling | `cts_cm/aperture/infill.py` | `ChainedImputer`, `rubin_pool` | 10 imputations default |
| §G Table V | C-index, calibration slope/intercept | `cts_cm/magnitudes/ranking.py` | `c_index`, `calibration` | concordance over trajectory severity score |
| R4 | set_seed, atomic checkpoint, run manifest | `cts_cm/dome/{seeds,logbook}.py` | `set_seed`, `atomic_write`, `RunLedger` | tmp + os.replace |
| layout | configuration schema + YAML merge | `cts_cm/almanac/{tables,reader}.py` | `Settings`, `load_settings` | frozen dataclasses, typed YAML loader |
| CLI | synthesize/fit/estimate/mediate/validate/run-all | `cts_cm/observer/__main__.py` | `main` | cyclopts subcommand dispatch |
| orchestration | end-to-end run | `cts_cm/survey/campaign.py` | `run_all` | aperture -> instruments -> reduction -> report |

Every numbered equation (1-6), every reported table (I-VI), every component (1-5) and the two
named ablation axes (meta-learner choice, trajectory-group count) are covered above.
