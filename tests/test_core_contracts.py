from __future__ import annotations

import json
from pathlib import Path

from scripts.run_core_benchmark import run


def test_core_contract_benchmark(tmp_path: Path) -> None:
    report = run(tmp_path / "core-contracts.json")
    assert report["failed"] == 0
    assert report["passed"] == 9
    assert json.loads((tmp_path / "core-contracts.json").read_text(encoding="utf-8"))["protocol_version"] == (
        "core-contracts-v1"
    )
