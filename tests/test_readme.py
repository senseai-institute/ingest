# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Keep README code honest: the adapter example must run, all Python blocks must compile."""

import re
from pathlib import Path

import numpy as np

from senseai_ingest.sync import Aligner, AlignerConfig, run_ticks

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.S)


def test_all_python_blocks_compile() -> None:
    assert len(BLOCKS) >= 3
    for i, b in enumerate(BLOCKS):
        if "StrEnum):\n    EVENT, FRAME" in b:  # the schematic type listing is not real code
            continue
        compile(b, f"README-block-{i}", "exec")


def test_adapter_example_runs_and_is_short(tmp_path: Path) -> None:
    (block,) = [b for b in BLOCKS if "class CsvThermalReplay" in b]
    assert len(block.strip().splitlines()) < 50
    ns: dict[str, object] = {}
    exec(compile(block, "README-adapter", "exec"), ns)
    csv = tmp_path / "thermal.csv"
    rows = [",".join([str(t)] + ["300.0"] * (24 * 32)) for t in (1000, 126_000, 251_000)]
    csv.write_text("\n".join(rows) + "\n")
    adapter = ns["CsvThermalReplay"](str(csv))  # type: ignore[operator]
    ticks = run_ticks(Aligner([adapter], AlignerConfig(tick_ns=100_000_000)))
    got = [s for t in ticks for s in t.samples["thermal0"]]
    assert len(got) == 3 and got[0].array.shape == (24, 32)
    assert np.all(got[0].array == np.float32(300.0))
