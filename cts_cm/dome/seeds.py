from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int


def set_seed(seed: int) -> SeedState:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return SeedState(seed=seed)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
