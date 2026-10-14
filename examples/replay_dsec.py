# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Replay a DSEC sequence (or a generated synthetic one) through the Aligner.

python examples/replay_dsec.py                      # synthetic DSEC-layout fixture
python examples/replay_dsec.py /data/dsec/zurich_city_04_a
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from senseai_ingest import Aligner, AlignerConfig
from senseai_ingest.adapters.replay.dsec import DsecReplay
from senseai_ingest.schema import hash_ticks


async def replay(path: Path) -> str:
    adapters = DsecReplay.from_sequence(path, speed=0.0)
    ticks = []
    async with Aligner(adapters, AlignerConfig(tick_ns=10_000_000)) as aligner:
        async for tick in aligner:
            ticks.append(tick)
            if tick.index < 3:
                counts = {sid: len(v) for sid, v in tick.samples.items()}
                n_ev = sum(len(s.columns["t"]) for s in tick.samples["events_left"])
                print(
                    f"tick {tick.index} @ {tick.t_master_ns} ns  "
                    f"samples={counts}  events_left={n_ev}"
                )
        st = aligner.stats()
    print(f"{st['ticks']} ticks, sync jitter p99 {st['sync_jitter_p99_ns'] / 1e6:.3f} ms")
    for sid, v in st["sources"].items():
        print(f"  {sid:<13} {v['samples_per_s']:7.1f} samples/s  dropped {int(v['dropped'])}")
    return hash_ticks(ticks)


def main() -> None:
    if len(sys.argv) > 1:
        print(f"tick hash {asyncio.run(replay(Path(sys.argv[1])))}")
        return
    from senseai_ingest.testing import make_dsec_fixture

    with tempfile.TemporaryDirectory() as tmp:
        seq = make_dsec_fixture(Path(tmp) / "synthetic_seq")
        print(f"tick hash {asyncio.run(replay(seq))}")


if __name__ == "__main__":
    main()
