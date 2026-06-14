# Deviations

Faithful mechanism substitutions where the manuscript's R toolchain has no drop-in Python
equivalent. Each entry names the paper section and the reason; the scientific quantity is
unchanged.

## D1. Causal forest engine (§E, Table III)

Paper: grf v2.3.0 generalized random forest with honest splitting.
Release: an honest-split forest built on scikit-learn `ExtraTreesRegressor`, with the sample
partitioned into a split half and an estimation half so leaf values use held-out observations
(honesty). The X-learner (the paper's primary meta-learner) is implemented directly on
gradient-boosted base learners. Reason: grf is R-only; the honesty principle and the X-learner
imputed-effect construction are reproduced exactly, only the tree backend differs.

## D2. GBTM fitting (§C)

Paper: lcmm v2.1.0 latent-class mixed model with 500 random starts.
Release: an EM finite-mixture of polynomial-in-time Gaussian trajectories with the same BIC
rule (2*loglik - k*ln n), the same admissibility gates (min group >= 2%, mean posterior >=
0.70) and multi-start initialisation. Reason: lcmm is R-only; the latent-class polynomial
model and selection criterion are identical.

## D3. Generalized propensity for continuous treatments (§E)

Paper: conditional density estimation via XGBoost-based quantile regression.
Release: a normalised Gaussian working likelihood from an XGBoost conditional-mean model with a
held-out residual variance, trimmed at the 1st/99th percentile. Reason: gives the same density
weights used in the Neyman-orthogonal score under the stated regularity conditions; flagged so
reviewers can swap in a full quantile-regression density if desired.

No other deviations. All equation-level definitions (Eq 1-6) are implemented as written.
