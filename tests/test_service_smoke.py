from __future__ import annotations

from cts_cm.almanac.tables import Settings
from cts_cm.survey.campaign import Report, run_all


def test_run_all_produces_complete_report(smoke_settings: Settings) -> None:
    report: Report = run_all(smoke_settings)
    assert report.group_count in smoke_settings.trajectory.group_grid
    assert set(report.population) == set(smoke_settings.risk_factors)
    for shares in report.pathways.values():
        total = shares["mechanical"] + shares["metabolic"] + shares["direct"]
        assert abs(total - 1.0) < 1e-6
    assert 0.0 < report.conformal["coverage"] <= 1.0
    assert 0.0 <= report.external["mr_concordance"] <= 1.0


def test_prescription_matches_dominant_pathway(smoke_settings: Settings) -> None:
    report = run_all(smoke_settings)
    for factor, shares in report.pathways.items():
        dominant = "metabolic" if shares["metabolic"] >= shares["mechanical"] else "mechanical"
        assert report.prescriptions[factor] == dominant
