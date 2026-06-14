from __future__ import annotations

import numpy as np

from cts_cm.reduction.intervals import SplitConformal, coverage_report


def test_split_conformal_attains_nominal_coverage() -> None:
    gen = np.random.default_rng(0)
    truth = gen.normal(size=4000)
    prediction = truth + gen.normal(scale=0.5, size=4000)
    calib = slice(0, 2000)
    test = slice(2000, 4000)
    conformer = SplitConformal(alpha=0.1)
    radius = conformer.calibrate(prediction[calib], truth[calib])
    report = coverage_report(prediction[test], truth[test], radius)
    assert report["coverage"] >= 0.88
    assert report["median_width"] > 0.0


def test_per_group_coverage_reported() -> None:
    gen = np.random.default_rng(1)
    truth = gen.normal(size=2000)
    prediction = truth + gen.normal(scale=0.4, size=2000)
    groups = gen.integers(0, 3, 2000).astype(np.int64)
    conformer = SplitConformal(alpha=0.05)
    radius = conformer.calibrate(prediction, truth)
    report = coverage_report(prediction, truth, radius, groups)
    assert {"coverage_g0", "coverage_g1", "coverage_g2"} <= set(report)
