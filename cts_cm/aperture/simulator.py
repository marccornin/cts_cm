from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cts_cm.aperture.frames import CohortFrame, FloatArray, IntArray

_GROUP_FRACTIONS: tuple[float, float, float, float] = (0.519, 0.270, 0.160, 0.051)
_BASELINE = np.array([2.1, 5.3, 7.1, 12.8])
_SLOPE = np.array([0.0, 0.036, 0.160, 0.100])
_CURVE = np.array([0.0, 0.0, 0.006, -0.004])


def _standardize(values: FloatArray) -> FloatArray:
    centered = values - float(values.mean())
    scale = float(values.std()) or 1.0
    return centered / scale


def _sigmoid(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class SyntheticCohort:
    seed: int
    n: int = 4484
    n_timepoints: int = 10
    horizon_years: float = 14.0
    cohort: str = "synthetic_oai"
    shift: float = 0.0

    def build(self) -> CohortFrame:
        gen = np.random.default_rng(self.seed)
        n = self.n

        age = 61.3 + 9.1 * gen.standard_normal(n)
        sex = gen.binomial(1, 0.578, n).astype(np.float64)
        race = gen.binomial(1, 0.18, n).astype(np.float64)
        kl = gen.integers(0, 5, n).astype(np.float64)
        age_z = _standardize(age)
        kl_z = _standardize(kl)
        covariates = np.column_stack([age, sex, race, kl])

        bmi = 28.6 + self.shift + 1.0 * age_z + 0.6 * kl_z + 4.5 * gen.standard_normal(n)
        pa = 142.0 - 12.0 * age_z - 8.0 * kl_z + 90.0 * gen.standard_normal(n)
        occ = gen.binomial(1, _sigmoid(-1.4 + 0.3 * age_z + 0.5 * gen.standard_normal(n)), n)
        mets = gen.binomial(
            1, _sigmoid(-1.0 + 0.4 * age_z + 0.5 * kl_z + 0.5 * gen.standard_normal(n)), n
        )
        occ_f = occ.astype(np.float64)
        mets_f = mets.astype(np.float64)
        bmi_z = _standardize(bmi)
        pa_z = _standardize(pa)

        knee_disorders = 0.55 * bmi_z + 0.45 * occ_f + 0.30 * gen.standard_normal(n)
        joint_space_width = -0.60 * bmi_z - 0.35 * occ_f + 0.30 * gen.standard_normal(n)
        muscle_strength = -0.45 * bmi_z + 0.50 * pa_z + 0.40 * gen.standard_normal(n)
        mechanical = np.column_stack([knee_disorders, joint_space_width, muscle_strength])

        crp = 2.6 + 1.5 * bmi_z + 1.2 * mets_f + 0.6 * gen.standard_normal(n)
        mets_status = 0.6 * bmi_z + 0.7 * mets_f + 0.3 * gen.standard_normal(n)
        waist = 96.0 + 6.0 * bmi_z + 4.0 * mets_f + 4.0 * gen.standard_normal(n)
        glucose = 100.0 + 8.0 * bmi_z + 6.0 * mets_f + 6.0 * gen.standard_normal(n)
        metabolic = np.column_stack([crp, mets_status, waist, glucose])

        mech_signal = (
            _standardize(knee_disorders)
            - _standardize(joint_space_width)
            - _standardize(muscle_strength)
        )
        metab_signal = 0.5 * (
            _standardize(crp)
            + _standardize(mets_status)
            + _standardize(waist)
            + _standardize(glucose)
        )
        severity = (
            0.45 * mech_signal
            + 1.05 * metab_signal
            + 0.10 * bmi_z
            - 0.05 * pa_z
            + 0.05 * occ_f
            + 0.20 * age_z
            + 0.15 * kl_z
            + 0.6 * gen.standard_normal(n)
        )

        group = self._assign_groups(severity)
        pain = self._draw_pain(gen, group)

        treatments: dict[str, FloatArray] = {
            "bmi": bmi,
            "physical_activity": pa,
            "occupational_loading": occ_f,
            "metabolic_syndrome": mets_f,
        }
        times = np.linspace(0.0, self.horizon_years, self.n_timepoints)
        return CohortFrame(
            cohort=self.cohort,
            times=times,
            pain=pain,
            treatments=treatments,
            mechanical=mechanical,
            metabolic=metabolic,
            covariates=covariates,
            group=group,
        )

    def _assign_groups(self, severity: FloatArray) -> IntArray:
        cumulative = np.cumsum(_GROUP_FRACTIONS)[:-1]
        cutpoints = np.quantile(severity, cumulative)
        assigned: IntArray = np.digitize(severity, cutpoints).astype(np.int64)
        return assigned

    def _draw_pain(self, gen: np.random.Generator, group: IntArray) -> FloatArray:
        times = np.linspace(0.0, self.horizon_years, self.n_timepoints)
        base = _BASELINE[group][:, None]
        slope = _SLOPE[group][:, None]
        curve = _CURVE[group][:, None]
        mean = base + slope * times[None, :] + curve * times[None, :] ** 2
        noise = 0.8 * gen.standard_normal((group.shape[0], self.n_timepoints))
        return np.clip(mean + noise, 0.0, 20.0)
