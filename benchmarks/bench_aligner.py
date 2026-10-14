# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Aligner throughput / jitter benchmark on 4 synthetic sources.

Sources: events @ 1e6 ev/s (1 kHz batches), RGB frames @ 60 Hz (480x640x3), audio @ 48 kHz in
1024-frame blocks (4 ch), IMU @ 1 kHz. Each source has a simulated device clock (offset + drift)
and transport latency with jitter, so the online offset estimator is exercised end to end.

Reports ticks/s, sync jitter (aligner metric: |model prediction - arrival|), true alignment
error (|t_master - true time|, known exactly for synthetic sources), drops and late samples.

    python benchmarks/bench_aligner.py [--duration-s 10] [--tick-ms 10] [--speed 0] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

from senseai_ingest.adapters.replay.synthetic import SyntheticReplay
from senseai_ingest.stream import Modality
from senseai_ingest.sync import Aligner, AlignerConfig

MS = 1_000_000
TARGET_TICKS_PER_S = 100.0
TARGET_P99_NS = 1 * MS


def make_sources(duration_s: float, seed: int = 0) -> list[SyntheticReplay]:
    """The four benchmark sources."""
    t0 = 1_700_000_000 * 10**9
    common: dict[str, Any] = {"duration_s": duration_s, "t_start_ns": t0}
    return [
        SyntheticReplay("dvs0", Modality.EVENT, rate_hz=1000, events_per_s=1e6, seed=seed,
                        device_offset_ns=-12 * MS, drift_ppm=30.0, latency_ns=300_000,
                        latency_jitter_ns=200_000, **common),
        SyntheticReplay("rgb0", Modality.FRAME, rate_hz=60, shape=(480, 640, 3), seed=seed + 1,
                        device_offset_ns=40 * MS, drift_ppm=-15.0, latency_ns=4 * MS,
                        latency_jitter_ns=300_000, **common),
        SyntheticReplay("mic0", Modality.AUDIO, sample_rate=48_000, block=1024, channels=4,
                        seed=seed + 2, device_offset_ns=7 * MS, drift_ppm=5.0,
                        latency_ns=500_000, latency_jitter_ns=100_000, **common),
        SyntheticReplay("imu0", Modality.IMU, rate_hz=1000, seed=seed + 3,
                        device_offset_ns=-3 * MS, drift_ppm=50.0, latency_ns=150_000,
                        latency_jitter_ns=100_000, **common),
    ]  # fmt: skip


def run(duration_s: float, tick_ms: float, speed: float) -> dict[str, Any]:
    """Run the benchmark and return a result dict."""
    sources = make_sources(duration_s)
    by_id = {s.source_id: s for s in sources}
    aligner = Aligner(sources, AlignerConfig(tick_ns=round(tick_ms * MS), speed=speed))
    true_err: list[int] = []
    t_wall = time.perf_counter()
    with aligner:
        for tick in aligner:
            for sid, lst in tick.samples.items():
                src = by_id[sid]
                true_err.extend(abs(s.t_master_ns - src.true_time(s.t_source_ns)) for s in lst)
    wall = time.perf_counter() - t_wall
    st = aligner.stats()
    err = np.asarray(true_err, dtype=np.float64)
    drops = sum(int(v["dropped"]) for v in st["sources"].values())
    late = sum(int(v["late"]) for v in st["sources"].values())
    ticks_per_s = st["ticks"] / wall if wall > 0 else float("inf")
    return {
        "ticks": st["ticks"],
        "wall_s": wall,
        "ticks_per_s": ticks_per_s,
        "realtime_factor": st["ticks"] * tick_ms * 1e-3 / wall if wall > 0 else float("inf"),
        "sync_jitter_p50_ms": st["sync_jitter_p50_ns"] / MS,
        "sync_jitter_p99_ms": st["sync_jitter_p99_ns"] / MS,
        "true_align_err_p50_ms": float(np.quantile(err, 0.5)) / MS if err.size else float("nan"),
        "true_align_err_p99_ms": float(np.quantile(err, 0.99)) / MS if err.size else float("nan"),
        "drops": drops,
        "late": late,
        "sources": {
            sid: {
                "samples_per_s": v["samples_per_s"],
                "events": v["events"],
                "dropped": v["dropped"],
                "high_water": v["high_water"],
            }
            for sid, v in st["sources"].items()
        },
        "pass": bool(
            ticks_per_s >= TARGET_TICKS_PER_S
            and st["sync_jitter_p99_ns"] < TARGET_P99_NS
            and drops == 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration-s", type=float, default=10.0)
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = run(args.duration_s, args.tick_ms, args.speed)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"ticks            {r['ticks']}  in {r['wall_s']:.2f} s wall")
        print(
            f"ticks/s          {r['ticks_per_s']:.0f}   (target >= {TARGET_TICKS_PER_S:.0f}; "
            f"{r['realtime_factor']:.1f}x real time)"
        )
        print(
            f"sync jitter      p50 {r['sync_jitter_p50_ms']:.3f} ms   "
            f"p99 {r['sync_jitter_p99_ms']:.3f} ms   (target p99 < 1 ms)"
        )
        print(
            f"true align err   p50 {r['true_align_err_p50_ms']:.3f} ms   "
            f"p99 {r['true_align_err_p99_ms']:.3f} ms"
        )
        print(f"drops {r['drops']}   late {r['late']}")
        for sid, v in r["sources"].items():
            print(
                f"  {sid:<6} {v['samples_per_s']:>9.1f} samples/s  events {int(v['events']):>10}"
                f"  dropped {int(v['dropped'])}  high_water {int(v['high_water'])}"
            )
        print("PASS" if r["pass"] else "FAIL")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
