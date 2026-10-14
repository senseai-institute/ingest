# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_bench_smoke() -> None:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        import bench_aligner
    finally:
        sys.path.pop(0)
    r = bench_aligner.run(duration_s=0.5, tick_ms=10, speed=0)
    assert r["ticks"] >= 49 and r["drops"] == 0
    assert r["sync_jitter_p99_ms"] < 1.0
    assert set(r["sources"]) == {"dvs0", "rgb0", "mic0", "imu0"}


def test_example_runs_on_fixture(
    dsec_dir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["replay_dsec.py", str(dsec_dir)])
    runpy.run_path(str(ROOT / "examples" / "replay_dsec.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "tick 0" in out and "tick hash" in out and "events_left" in out
