from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MECHANICAL_MEDIATORS: tuple[str, ...] = ("knee_disorders", "joint_space_width", "muscle_strength")
METABOLIC_MEDIATORS: tuple[str, ...] = (
    "crp",
    "metabolic_syndrome_status",
    "waist_circumference",
    "fasting_glucose",
)


@dataclass
class CohortFrame:
    cohort: str
    times: FloatArray
    pain: FloatArray
    treatments: dict[str, FloatArray]
    mechanical: FloatArray
    metabolic: FloatArray
    covariates: FloatArray
    group: IntArray | None = None

    @property
    def n(self) -> int:
        return int(self.pain.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.pain.shape[1])

    def binary_exposure(self, factor: str, threshold: float) -> FloatArray:
        column = self.treatments[factor]
        return (column >= threshold).astype(np.float64)


@dataclass
class EffectEstimate:
    factor: str
    theta: float
    se: float
    ci_low: float
    ci_high: float
    naive: float = float("nan")
    attenuation: float = float("nan")
    refutation: dict[str, float] = field(default_factory=dict)
