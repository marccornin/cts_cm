from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataCfg:
    cohort: str = "synthetic"
    n: int = 4484
    n_timepoints: int = 10
    horizon_years: float = 14.0
    missing_rate: float = 0.083
    csv_path: str = ""


@dataclass(frozen=True)
class TrajectoryCfg:
    n_groups: int = 4
    group_grid: tuple[int, ...] = (3, 4, 5, 6)
    poly_degree: int = 3
    n_starts: int = 500
    min_group_frac: float = 0.02
    min_posterior: float = 0.70
    max_iter: int = 200
    tol: float = 1e-5


@dataclass(frozen=True)
class DmlCfg:
    n_folds: int = 5
    xgb_max_depth: int = 6
    xgb_lr: float = 0.1
    xgb_rounds: int = 500
    propensity_trim: float = 0.01


@dataclass(frozen=True)
class ForestCfg:
    metalearner: str = "X"
    n_trees: int = 2000
    min_leaf: int = 5
    honest_frac: float = 0.5
    max_features: float = 0.5


@dataclass(frozen=True)
class MediationCfg:
    mc_draws: int = 200
    sensitivity_rho: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass(frozen=True)
class ConformalCfg:
    alpha: float = 0.05
    calib_frac: float = 0.20


@dataclass(frozen=True)
class ResampleCfg:
    n_bootstrap: int = 1000
    n_seeds: int = 20
    n_imputations: int = 10
    fdr_alpha: float = 0.05


@dataclass(frozen=True)
class RuntimeCfg:
    seed: int = 20240501
    output_dir: str = "runs/main"
    log_level: str = "INFO"


@dataclass(frozen=True)
class Settings:
    name: str = "main"
    risk_factors: tuple[str, ...] = (
        "bmi",
        "physical_activity",
        "occupational_loading",
        "metabolic_syndrome",
    )
    data: DataCfg = field(default_factory=DataCfg)
    trajectory: TrajectoryCfg = field(default_factory=TrajectoryCfg)
    dml: DmlCfg = field(default_factory=DmlCfg)
    forest: ForestCfg = field(default_factory=ForestCfg)
    mediation: MediationCfg = field(default_factory=MediationCfg)
    conformal: ConformalCfg = field(default_factory=ConformalCfg)
    resample: ResampleCfg = field(default_factory=ResampleCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
