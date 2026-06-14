from __future__ import annotations

from pathlib import Path

from cts_cm.dome.logbook import RunLedger, atomic_write
from cts_cm.dome.seeds import set_seed


def test_atomic_write_persists_payload(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.json"
    written = atomic_write(target, '{"ok": 1}')
    assert written.read_text(encoding="utf-8") == '{"ok": 1}'
    assert not any(p.suffix == ".tmp" for p in target.parent.iterdir())


def test_ledger_checkpoint_round_trips_seed(tmp_path: Path) -> None:
    ledger = RunLedger(output_dir=str(tmp_path), seed=20240501)
    ledger.log_result("theta", 0.21)
    path = ledger.checkpoint("report.json")
    restored = RunLedger.restore(path)
    assert restored.seed == 20240501
    assert restored.records["theta"] == 0.21


def test_set_seed_is_reproducible() -> None:
    import numpy as np

    set_seed(123)
    first = np.random.random(5)
    set_seed(123)
    second = np.random.random(5)
    assert (first == second).all()
