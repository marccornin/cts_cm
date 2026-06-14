# Project Context — CTS-CM

    project_name       : cts_cm                                        [HIGH]
    domain             : population-scale causal ML for knee OA        [HIGH]
                         (modifiable-risk-factor effects on trajectory membership)
    framework          : Python scientific stack, no deep learning     [MED]
                         (numpy / scipy / pandas / scikit-learn / xgboost / statsmodels)
    venue              : IEEE Trans. Biomedical and Health Informatics  [HIGH]
    primary_datasets   : 3 cohorts (see §6)                            [HIGH]
    compute_target     : COMPUTE_NOT_REPORTED -> CPU upper bound (§7)   [LOW]
    hparams_reference  : Methods §B-H + Tables I-VI (see §8)            [HIGH]
    supp_path          : none (no separate supplement file shipped)
    extra_signals      : no neural nets / no checkpoints; verbatim code-availability
                         statement present (kept in §9, not in README); 6 equations;
                         6 tables; 3 figures; 5 modular components

    NEEDS_USER_DECISION: 0
    Layout / config-CLI / README form: observatory sky-survey / cyclopts + typed-YAML loader /
    observing-proposal (confirmed with user before build).

---

## 1. project_name

`cts_cm` — from the method name CTS-CM (Counterfactual Trajectory Shifting via Causal
Mediation), Abstract L7 and Methods §B. The acronym is the framework's own identity and is
used throughout the paper, so it is preferred over a title-derived slug. [HIGH]

## 2. supp_path

`none`. Globbing the sibling locations for this manuscript found only the main PDF; no
`*supp*`, `*_si*`, `appendix*`, or cross-referenced supplementary file accompanies it. All
quantitative detail used here comes from the main text (Methods §A-H, Tables I-VI). [HIGH]

## 3. domain

Population-scale causal machine learning for knee osteoarthritis epidemiology: estimating the
causal effects of modifiable risk factors (BMI, physical activity, occupational loading,
metabolic syndrome) on membership in latent pain-trajectory groups, decomposed through
mechanical and metabolic biological pathways. This is an estimation / inference pipeline over
longitudinal cohort data, not an imaging or generative model. Source: Abstract, Methods §A-B. [HIGH]

## 4. framework

Python scientific stack with no deep-learning component:
numpy, scipy, pandas, scikit-learn, xgboost, statsmodels.

The paper's own analysis was run in R (Methods §H): grf v2.3.0 (causal forests), DoubleML
v0.7.0 (DML), lcmm v2.1.0 (group-based trajectory modelling), mediation v4.5.0, mice (chained
imputation). This release re-expresses those five stages in Python so the canonical layout
(package + pyproject + typed API + pytest) is uniform with the lab's other releases. There is
no torch dependency because every estimator in the paper is tree-based or closed-form
(XGBoost nuisance models, honest-split forests, EM mixture, Monte-Carlo mediation,
split-conformal). Confidence MED only because the language differs from the manuscript's R;
the algorithms are matched exactly. [MED]

## 5. venue

IEEE Transactions on Biomedical and Health Informatics (JBHI). Running header on every page
reads "IEEE TRANSACTIONS ON BIOMEDICAL AND HEALTH INFORMATICS". [HIGH]

## 6. primary_datasets

All three are established public longitudinal cohorts (Methods §A; Data and Code Availability).

| name | role | n (this study) | access | license / governance |
|------|------|----------------|--------|----------------------|
| Osteoarthritis Initiative (OAI) | discovery + GBTM + internal cross-fit | 4,484 analytic (4,796 enrolled; 312 excluded) | https://nda.nih.gov/oai | public; free registration + Data Use Agreement via NDA. IRBs of participating sites + UCSF coordinating centre |
| Multicenter Osteoarthritis Study (MOST) | primary external validation (no retraining) | 3,026 | https://most.ucsf.edu/ | public; IRB UAB + Univ. Iowa |
| UK Biobank | Mendelian-randomization triangulation only | 45,581 OA cases (ICD-10 M17.x + self-report) | https://www.ukbiobank.ac.uk | restricted; approved application required (project access). Research Ethics Committee Ref 21/NW/0157 |

Combined N = 53,091 across cohorts. Each cohort analysed independently; no cross-cohort data
pooling. Because OAI requires a DUA and UK Biobank requires an approved application, real
microdata cannot ship; the release provides a deterministic synthetic cohort generator whose
marginal moments and causal/trajectory structure match Tables I-IV so the pipeline runs
end-to-end offline.

## 7. compute_target

COMPUTE_NOT_REPORTED. The paper states no GPU, wall-clock, or memory figures; all estimators
are CPU-bound. Derived upper bound (arithmetic, single modern CPU core unless parallelised):

- nuisance fit: XGBoost depth 6, 500 rounds, 5-fold cross-fit ~ 5 x (500 trees) per model.
- per risk factor (4 total): DML (outcome model x K=4 classes + generalized propensity) +
  X-learner (2 arms) + causal forest (2,000 trees) + interventional mediation
  (200 Monte-Carlo draws per observation x outcome + mediator models).
- repeated over 1,000 bootstrap resamples, 20 random seeds, 10 multiply-imputed datasets
  (Rubin pooling).

Order-of-magnitude: 4 factors x (≈10 nuisance fits) x 1,000 bootstrap x 20 seeds is ~1e6
gradient-boosted fits at full settings; with embarrassingly-parallel bootstrap this is tens of
CPU-core-hours on the full OAI sample, dominated by causal forests and mediation Monte Carlo.
Memory < 8 GB (tabular, ~4.5k rows x ~1,100 columns subset to the analytic variables). The
`_smoke` experiment collapses bootstrap=8, seeds=1, mice=1, trees=64, mc_draws=16 on a 600-row
synthetic cohort and completes in under 30 s.

## 8. hparams_reference

Scattered across Methods §B-H and Tables I-VI; consolidated here (used verbatim in
configs/experiment/main.yaml).

- Trajectory model (GBTM, §C): K = 4 groups (optimal by BIC: 3 groups -47,856; 4 groups
  -47,891; 5 groups -47,943 — higher value = better under 2*loglik - k*ln n); polynomial basis
  quadratic/cubic evaluated for 3-6 groups; min group size >= 2% (~96); mean posterior >= 0.70;
  500 random starts. Groups: G1 stable-low n=2,328 (51.9%), G2 mild-fluctuating n=1,210
  (27.0%), G3 moderate-progressive n=718 (16.0%), G4 severe-persistent n=228 (5.1%).
- DML (§E): 5-fold cross-fitting; XGBoost nuisance (max_depth 6, learning_rate 0.1, 500 rounds,
  5-fold inner CV); generalized propensity via conditional-density / quantile estimation;
  propensity trimmed at 1st/99th percentile; Neyman-orthogonal score (Eq 4).
- Causal forest / meta-learners (§E): grf honest splitting, ~2,000 trees; meta-learners S, T,
  X, DR, R compared, X-learner selected primary (treatment-prevalence imbalance across groups,
  e.g. BMI>=30 is 31.2% in G1 vs 65.8% in G4).
- Interventional mediation (§F): mechanical mediators M1 = {knee disorders, JSW, muscle
  strength}; metabolic mediators M2 = {CRP, metabolic syndrome, waist circumference, fasting
  glucose}; 200 Monte-Carlo integration draws per observation; XGBoost mediator/outcome models,
  5-fold CV; proportion mediated = IIE_j / (IIE_1 + IIE_2 + NDE); sensitivity parameter rho in
  [0, 0.5].
- Conformal (§G): split conformal, 80% train / 20% calibration, nominal 95% (Eq 6); empirical
  coverage 93.2% OAI, 91.8% MOST.
- Resampling / inference (§H): bootstrap 1,000 resamples (posterior-weighted, propagating GBTM
  classification uncertainty pi_ik); 20 independent random seeds; MICE 10 imputed datasets
  (overall missingness 8.3%) pooled by Rubin's rules; Benjamini-Hochberg FDR <= 0.05; R v4.3.2.

## 9. extra_signals

- No neural networks and no released checkpoints; weights/“available upon request” is not
  applicable — the deliverable is the estimation code plus the synthetic-data generator.
- Verbatim Data-and-Code-Availability statement (manuscript p. 10), retained here for
  provenance and deliberately NOT reproduced in README per release policy:
  > All data analysed in this study are publicly available from established repositories:
  > Osteoarthritis Initiative (OAI) at https://nda.nih.gov/oai, Multicenter Osteoarthritis
  > Study (MOST) at https://most.ucsf.edu/, and UK Biobank at https://www.ukbiobank.ac.uk. No
  > private or proprietary datasets were used. All analysis code will be deposited in a public
  > GitHub repository (with a citable Zenodo DOI) upon acceptance, including GBTM trajectory
  > fitting (lcmm v2.1.0), DML estimation (DoubleML v0.7.0), causal forest and meta-learner
  > comparison (grf v2.3.0), interventional mediation analysis (mediation v4.5.0), and conformal
  > prediction interval construction.
- Equations Eq(1)-(6); Tables I-VI; Figures 1-3; five modular components.
- Ethics: OAI site IRBs + UCSF; MOST IRBs (UAB, Univ. Iowa); UK Biobank REC Ref 21/NW/0157;
  written informed consent in original studies; present study used de-identified public data
  and was exempt from additional review.
- Competing interests: none declared.
