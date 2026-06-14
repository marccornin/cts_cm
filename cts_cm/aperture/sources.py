from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from cts_cm.aperture.frames import (
    MECHANICAL_MEDIATORS,
    METABOLIC_MEDIATORS,
    CohortFrame,
    FloatArray,
)
from cts_cm.aperture.simulator import SyntheticCohort


@dataclass(frozen=True)
class CohortDescriptor:
    key: str
    name: str
    url: str
    n: int
    role: str
    access: str


COHORT_DESCRIPTORS: dict[str, CohortDescriptor] = {
    "oai": CohortDescriptor(
        key="oai",
        name="Osteoarthritis Initiative",
        url="https://nda.nih.gov/oai",
        n=4484,
        role="discovery and internal cross-fit",
        access="public with data use agreement",
    ),
    "most": CohortDescriptor(
        key="most",
        name="Multicenter Osteoarthritis Study",
        url="https://most.ucsf.edu/",
        n=3026,
        role="primary external validation",
        access="public",
    ),
    "ukb": CohortDescriptor(
        key="ukb",
        name="UK Biobank",
        url="https://www.ukbiobank.ac.uk",
        n=45581,
        role="Mendelian randomization triangulation",
        access="restricted, approved application required",
    ),
}

_TREATMENT_COLUMNS: tuple[str, ...] = (
    "bmi",
    "physical_activity",
    "occupational_loading",
    "metabolic_syndrome",
)


@runtime_checkable
class CohortSource(Protocol):
    def frame(self) -> CohortFrame: ...


@dataclass
class CsvCohort:
    path: str
    cohort: str
    n_timepoints: int
    horizon_years: float

    def frame(self) -> CohortFrame:
        import pandas as pd

        table = pd.read_csv(self.path)
        pain_cols = [f"womac_t{idx}" for idx in range(self.n_timepoints)]
        pain = np.asarray(table[pain_cols].to_numpy(), dtype=np.float64)
        treatments = {
            name: np.asarray(table[name].to_numpy(), dtype=np.float64)
            for name in _TREATMENT_COLUMNS
        }
        mechanical = np.asarray(table[list(MECHANICAL_MEDIATORS)].to_numpy(), dtype=np.float64)
        metabolic = np.asarray(table[list(METABOLIC_MEDIATORS)].to_numpy(), dtype=np.float64)
        covariate_cols = ["age", "sex", "race", "baseline_kl"]
        covariates = np.asarray(table[covariate_cols].to_numpy(), dtype=np.float64)
        times = np.linspace(0.0, self.horizon_years, self.n_timepoints)
        group: FloatArray | None = None
        if "group" in table.columns:
            group = np.asarray(table["group"].to_numpy(), dtype=np.float64)
        return CohortFrame(
            cohort=self.cohort,
            times=times,
            pain=pain,
            treatments=treatments,
            mechanical=mechanical,
            metabolic=metabolic,
            covariates=covariates,
            group=None if group is None else group.astype(np.int64),
        )


def load_cohort(
    cohort: str,
    n: int,
    n_timepoints: int,
    horizon_years: float,
    seed: int,
    csv_path: str = "",
    shift: float = 0.0,
) -> CohortFrame:
    if csv_path:
        return CsvCohort(
            path=str(Path(csv_path)),
            cohort=cohort,
            n_timepoints=n_timepoints,
            horizon_years=horizon_years,
        ).frame()
    return SyntheticCohort(
        seed=seed,
        n=n,
        n_timepoints=n_timepoints,
        horizon_years=horizon_years,
        cohort=cohort,
        shift=shift,
    ).build()
