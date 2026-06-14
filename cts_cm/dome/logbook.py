from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def atomic_write(path: str | Path, payload: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


@dataclass
class RunLedger:
    output_dir: str
    seed: int
    records: dict[str, Any] = field(default_factory=dict)

    def log_result(self, key: str, value: Any) -> None:
        self.records[key] = value

    def checkpoint(self, name: str) -> Path:
        payload = json.dumps({"seed": self.seed, "records": self.records}, indent=2, default=float)
        return atomic_write(Path(self.output_dir) / name, payload)

    @staticmethod
    def restore(path: str | Path) -> RunLedger:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger = RunLedger(output_dir=str(Path(path).parent), seed=int(data["seed"]))
        ledger.records = dict(data["records"])
        return ledger
