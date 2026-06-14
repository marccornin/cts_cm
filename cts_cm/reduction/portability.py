from __future__ import annotations

import numpy as np


def transfer_degradation(internal_concordance: float, external_concordance: float) -> float:
    if internal_concordance <= 0.0:
        return float("nan")
    return float((internal_concordance - external_concordance) / internal_concordance * 100.0)


def mr_triangulation(dml_odds_ratio: float, mr_odds_ratio: float) -> bool:
    return bool(np.sign(dml_odds_ratio - 1.0) == np.sign(mr_odds_ratio - 1.0))
