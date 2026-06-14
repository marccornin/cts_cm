# Observing Proposal — CTS-CM Survey of Knee-Osteoarthritis Risk Trajectories

| Field | Entry |
|-------|-------|
| Proposal ID | CTS-CM-OBS / knee-OA |
| Instrument | `cts_cm` (Python analysis package, no deep-learning dependency) |
| Category | Population-scale causal-inference survey |
| Fields observed | Osteoarthritis Initiative, Multicenter Osteoarthritis Study, UK Biobank |
| Primary observable | Trajectory-shifting effect of modifiable risk factors on pain-trajectory membership |
| Version | 0.1.0 |

This README is written as an observing proposal: it states what is measured, how the instruments are
configured, which command produces each expected number, and what time and targets are required. The
package is the lab's own implementation of the CTS-CM framework, organised as an observatory whose
instruments record patient pain trajectories the way a survey records light curves.

---

## 1. Abstract

Group-based trajectory modelling describes which patients follow each knee-osteoarthritis pain
trajectory but not which modifiable factor, if changed, would move a patient to a more favourable one.
CTS-CM treats trajectory-group membership as a causal outcome of four modifiable factors — body mass
index, physical activity, occupational loading, and metabolic syndrome — and decomposes each effect
through a mechanical pathway (joint loading, knee disorders, muscle strength) and a metabolic pathway
(C-reactive protein, metabolic-syndrome status, waist circumference, fasting glucose). Five instruments
record the survey: trajectory identification, causal-graph specification, doubly robust effect
estimation, interventional mediation, and uncertainty quantification with cross-field validation.

## 2. Scientific justification

Population-level osteoarthritis interventions show modest effects despite well-established risk
factors. The working hypothesis is a pathway–trajectory mismatch: weight-management programmes act
mostly through the mechanical pathway, yet in severe-trajectory patients the body-mass-index effect
runs mainly through the metabolic pathway. The survey tests three predictions:

- H1 — effects on trajectory membership are heterogeneous across groups, larger in high-risk subgroups.
- H2 — metabolic-pathway dominance rises monotonically from stable-low to severe-persistent
  trajectories.
- H3 — causal trajectory features transport across cohorts better than predictive features.

## 3. Target list

| Field | Role | n | Access | Source |
|-------|------|---|--------|--------|
| Osteoarthritis Initiative | discovery, internal cross-fit | 4,484 analytic | public, data use agreement | https://nda.nih.gov/oai |
| Multicenter Osteoarthritis Study | external validation, no retraining | 3,026 | public | https://most.ucsf.edu/ |
| UK Biobank | Mendelian-randomization triangulation | 45,581 OA cases | approved application | https://www.ukbiobank.ac.uk |

Eligibility for the analytic sample: a baseline WOMAC pain assessment and at least three follow-up
visits; bilateral total knee replacement or rheumatoid arthritis excluded. The Osteoarthritis
Initiative requires a data use agreement and UK Biobank an approved application, so raw participant
data are not redistributed. A deterministic synthetic-cohort generator (`cts_cm/aperture/simulator.py`)
reproduces the group sizes, marginal moments, and pathway structure so every instrument runs offline;
set `csv_path` in a data config to point at real extracts instead.

## 4. Instrument configuration and setup

Set up the instrument one of three ways:

    pip install -e ".[dev]"

    conda env create -f environment.yml && conda activate cts_cm && pip install -e .

    docker build -t cts_cm:0.1.0 . && docker run --rm cts_cm:0.1.0 --help

## 5. Observing strategy

Each instrument is one analysis stage; the `cts-cm` console (equivalently `python -m cts_cm.observer`)
runs a single stage or the whole campaign.

| Instrument | Module | Records |
|-----------|--------|---------|
| Light curves | `cts_cm/instruments/lightcurves.py` | polynomial-mixture trajectories, EM, BIC selection |
| Astrometry | `cts_cm/instruments/astrometry.py` | causal graph, stability selection, back-door sets |
| Spectroscopy | `cts_cm/instruments/spectroscopy.py` | double machine learning, S/T/X/DR/R learners, honest forest |
| Interferometry | `cts_cm/instruments/interferometry.py` | interventional mediation via Monte-Carlo integration |
| Reduction | `cts_cm/reduction/` | conformal intervals, posterior bootstrap, cross-field transfer |

    cts-cm synthesize       --config configs/experiment/main.yaml
    cts-cm fit-trajectories --config configs/experiment/main.yaml
    cts-cm estimate         --config configs/experiment/main.yaml
    cts-cm mediate          --config configs/experiment/main.yaml
    cts-cm validate         --config configs/experiment/main.yaml
    cts-cm run-all          --config configs/experiment/main.yaml

Any field is overridable: `cts-cm run-all --config configs/experiment/main.yaml --set forest.metalearner=DR --set dml.n_folds=10`.

## 6. Expected results

Each observable, the command that records it, and the value reported in the manuscript for the real
fields. Expected values are the manuscript's cohort results; the offline synthetic run reproduces the
direction, ordering, and instrument behaviour rather than these exact numbers.

| Observable | Command | Manuscript value |
|------------|---------|------------------|
| BMI population effect (Table II) | `cts-cm estimate` | ATE -0.21 per 5 kg/m^2, 95% CI [-0.284, -0.136] |
| Severe/stable CATE ratio for BMI (Table III) | `cts-cm estimate` | 4.6x (G4 -0.46 vs G1 -0.10) |
| BMI metabolic pathway share (Table IV) | `cts-cm mediate` | 72.1% metabolic, 22.4% mechanical |
| Trajectory groups selected (Section III-A) | `cts-cm fit-trajectories` | 4 groups (BIC -47,891) |
| Conformal coverage (Section III) | `cts-cm validate` | 93.2% at nominal 95% |
| Cross-cohort degradation (Table V) | `cts-cm validate` | 12.8% vs 22.4% for the predictive baseline |
| MR direction concordance (Table V) | `cts-cm validate` | all four factors concordant |

Ablations and supplementary runs are
`configs/experiment/{ablation_metalearner,ablation_ngroups,ablation_pathway,supplementary_mr,supplementary_sensitivity}.yaml`.

## 7. Technical justification (time request)

The manuscript reports no GPU, wall-clock, or memory figures; every instrument is CPU-bound (tree
ensembles, EM, Monte-Carlo integration). The bound below is derived; the smoke configuration runs the
whole campaign in under thirty seconds on one core.

| Resource | Main configuration (estimate) | Smoke configuration |
|----------|-------------------------------|---------------------|
| Accelerator | none (CPU only) | none |
| Cores | parallel bootstrap benefits from many cores | 1 |
| Memory | under 8 GB | under 1 GB |
| Wall-clock | tens of CPU-core-hours at full bootstrap and seeds | under 30 s |

## 8. Data reduction and quality control

The test suite covers planted-group recovery, single-block overfit, double-machine-learning
unbiasedness and orthogonality, meta-learner output shapes, conformal coverage, mediation
conservation, graph distance, Benjamini-Hochberg agreement with `statsmodels`, bootstrap determinism,
atomic-checkpoint round-trips, the end-to-end smoke campaign, and a source-style guard. Run the gates:

    make test
    make lint
    make type

## 9. Ethics and data governance

All analyses use de-identified data from established public repositories. Osteoarthritis Initiative
approvals were granted by the participating-site institutional review boards and the coordinating
centre; the Multicenter Osteoarthritis Study by the University of Alabama at Birmingham and the
University of Iowa; UK Biobank under Research Ethics Committee reference 21/NW/0157. Participants in the
original studies provided written informed consent. No private or proprietary data are included here.
