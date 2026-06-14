from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

from cts_cm.magnitudes.testing import benjamini_hochberg, e_value, wald_heterogeneity


def test_bh_matches_statsmodels() -> None:
    gen = np.random.default_rng(0)
    pvalues = gen.uniform(0.0, 1.0, 40)
    rejected, adjusted = benjamini_hochberg(pvalues, alpha=0.05)
    sm_rejected, sm_adjusted, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")
    assert np.array_equal(rejected, sm_rejected)
    assert np.allclose(adjusted, sm_adjusted)


def test_e_value_closed_form() -> None:
    assert abs(e_value(1.91) - (1.91 + np.sqrt(1.91 * 0.91))) < 1e-9
    assert abs(e_value(1.0) - 1.0) < 1e-9
    assert e_value(0.5) == e_value(2.0)


def test_wald_detects_heterogeneity() -> None:
    homogeneous = wald_heterogeneity([0.10, 0.11, 0.09], [0.02, 0.02, 0.02])
    heterogeneous = wald_heterogeneity([-0.10, -0.46], [0.02, 0.04])
    assert homogeneous[1] > 0.05
    assert heterogeneous[1] < 0.05
