from __future__ import annotations

from pathlib import Path

import pytest

from cts_cm.almanac.reader import load_settings
from cts_cm.almanac.tables import Settings
from cts_cm.aperture.frames import CohortFrame
from cts_cm.aperture.simulator import SyntheticCohort

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_settings() -> Settings:
    return load_settings(PROJECT_ROOT / "configs" / "experiment" / "_smoke.yaml")


@pytest.fixture(scope="session")
def cohort() -> CohortFrame:
    return SyntheticCohort(seed=11, n=900, n_timepoints=8).build()


@pytest.fixture(scope="session")
def recovery_cohort() -> CohortFrame:
    return SyntheticCohort(seed=11, n=3200, n_timepoints=10).build()
