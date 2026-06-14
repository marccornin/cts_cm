from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from cts_cm.aperture.frames import CohortFrame
from cts_cm.instruments.lightcurves import TrajectoryMixture, select_group_count


def _matched_accuracy(true_labels: np.ndarray, predicted: np.ndarray, k: int) -> float:
    confusion = np.zeros((k, k))
    for t, p in zip(true_labels, predicted, strict=False):
        confusion[t, p] += 1
    rows, cols = linear_sum_assignment(-confusion)
    return float(confusion[rows, cols].sum() / true_labels.shape[0])


def test_planted_groups_recovered(recovery_cohort: CohortFrame) -> None:
    assert recovery_cohort.group is not None
    model = TrajectoryMixture(n_groups=4, poly_degree=3, n_starts=8, seed=1).fit(
        recovery_cohort.pain, recovery_cohort.times
    )
    predicted = model.predict(recovery_cohort.pain, recovery_cohort.times)
    accuracy = _matched_accuracy(recovery_cohort.group, predicted, 4)
    assert accuracy > 0.9


def test_bic_selects_planted_count(recovery_cohort: CohortFrame) -> None:
    best, scores = select_group_count(
        recovery_cohort.pain,
        recovery_cohort.times,
        grid=(3, 4, 5),
        poly_degree=3,
        n_starts=8,
        seed=2,
    )
    assert best == 4
    assert set(scores) == {3, 4, 5}


def test_components_ordered_by_severity(recovery_cohort: CohortFrame) -> None:
    model = TrajectoryMixture(n_groups=4, poly_degree=3, n_starts=8, seed=3).fit(
        recovery_cohort.pain, recovery_cohort.times
    )
    assert model.coef_ is not None
    intercepts = model.coef_[:, 0]
    assert np.all(np.diff(intercepts) > 0.5)


def test_overfit_single_group_block() -> None:
    times = np.linspace(0.0, 14.0, 10)
    gen = np.random.default_rng(0)
    pain = 7.0 + 0.2 * times[None, :] + 0.05 * gen.standard_normal((120, 10))
    model = TrajectoryMixture(n_groups=1, poly_degree=2, n_starts=2, seed=0).fit(pain, times)
    assert model.sigma2_ is not None
    assert float(model.sigma2_[0]) < 0.05
