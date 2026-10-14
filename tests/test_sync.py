# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

import pytest

from senseai_ingest.adapters.replay.synthetic import SyntheticReplay
from senseai_ingest.clock import ReplayClock
from senseai_ingest.schema import hash_ticks
from senseai_ingest.stream import AlignedTick, ClockMode, Modality, Stream
from senseai_ingest.sync import Aligner, AlignerConfig, LatePolicy, run_ticks
from senseai_ingest.throttle import ThrottleConfig
from tests.helpers import ListAdapter, events, imu

MS = 1_000_000


def _ts(tick: AlignedTick, sid: str) -> list[int]:
    return [s.t_master_ns for s in tick.samples[sid]]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        AlignerConfig(tick_ns=0)
    with pytest.raises(ValueError):
        AlignerConfig(speed=-1)
    with pytest.raises(ValueError):
        Aligner([])
    a = ListAdapter("a", [imu("a", 0)])
    with pytest.raises(ValueError, match="duplicate"):
        Aligner([a, ListAdapter("a", [imu("a", 0)])])
    with pytest.raises(ValueError, match="unknown"):
        Aligner([a], AlignerConfig(throttle={"zz": ThrottleConfig()}))
    with pytest.raises(ValueError, match="speed"):
        Aligner([ListAdapter("a", [imu("a", 0)], speed=1.0), ListAdapter("b", [imu("b", 0)])])


def test_tick_windows_and_explicit_empty_sources() -> None:
    a = ListAdapter("a", [imu("a", t) for t in (0, 3, 10, 19, 20, 45)])
    b = ListAdapter("b", [imu("b", 15)])
    ticks = run_ticks(Aligner([a, b], AlignerConfig(tick_ns=10)))
    assert [t.t_master_ns for t in ticks] == [10, 20, 30, 40, 50]
    assert [t.index for t in ticks] == [0, 1, 2, 3, 4]
    assert _ts(ticks[0], "a") == [0, 3]
    assert _ts(ticks[1], "a") == [10, 19] and _ts(ticks[1], "b") == [15]
    assert _ts(ticks[2], "a") == [20]
    assert ticks[3].samples == {"a": [], "b": []}  # explicit empties
    assert _ts(ticks[4], "a") == [45]
    for t in ticks:  # causality: no sample at/after its tick close, none before window
        for lst in t.samples.values():
            assert all(t.window_start_ns <= s.t_master_ns < t.t_master_ns for s in lst)
    assert a.opened and a.closed


def test_events_are_split_at_tick_boundaries() -> None:
    e = ListAdapter("e", [events("e", [1, 5, 9, 12, 25])], modality=Modality.EVENT)
    ticks = run_ticks(Aligner([e], AlignerConfig(tick_ns=10, start_ns=0)))
    got = [[list(s.columns["t"]) for s in t.samples["e"]] for t in ticks]
    assert got == [[[1, 5, 9]], [[12]], [[25]]]
    assert ticks[0].samples["e"][0].t_master_ns == 9


def test_no_split_keeps_batches_whole() -> None:
    e = ListAdapter("e", [events("e", [1, 5, 12])], modality=Modality.EVENT)
    ticks = run_ticks(Aligner([e], AlignerConfig(tick_ns=10, start_ns=0, split_events=False)))
    assert [len(t.samples["e"]) for t in ticks] == [0, 1]


def test_start_end_and_max_ticks() -> None:
    a = ListAdapter("a", [imu("a", t) for t in range(0, 100, 5)])
    al = Aligner([a], AlignerConfig(tick_ns=10, start_ns=20, end_ns=50))
    ticks = run_ticks(al)
    assert [t.t_master_ns for t in ticks] == [30, 40, 50]
    assert sum(len(t.samples["a"]) for t in ticks) == 6
    assert al.snapshot()["a.before_start"] == 4
    e = ListAdapter("e", [events("e", [5, 25, 35, 41, 55])], modality=Modality.EVENT)
    ticks = run_ticks(Aligner([e], AlignerConfig(tick_ns=10, start_ns=20, end_ns=40)))
    flat = [int(x) for t in ticks for s in t.samples["e"] for x in s.columns["t"]]
    assert flat == [25, 35]
    ticks = run_ticks(
        Aligner(
            [ListAdapter("a", [imu("a", t) for t in range(100)])],
            AlignerConfig(tick_ns=10, max_ticks=3),
        )
    )
    assert len(ticks) == 3


def test_out_of_order_samples_are_late_in_replay() -> None:
    src = [imu("a", t) for t in (1, 12, 5, 22)]
    for policy, expect in [
        (LatePolicy.DROP, [[1], [12], [22]]),
        (LatePolicy.ATTACH_NEXT, [[1], [5, 12], [22]]),
    ]:
        al = Aligner(
            [ListAdapter("a", src)], AlignerConfig(tick_ns=10, start_ns=0, late_policy=policy)
        )
        ticks = run_ticks(al)
        assert [_ts(t, "a") for t in ticks] == expect
        assert al.stats()["sources"]["a"]["late"] == 1


def test_replay_is_deterministic_hash() -> None:
    def build() -> Aligner:
        srcs = [
            SyntheticReplay(
                "dvs0", Modality.EVENT, rate_hz=700, events_per_s=50_000, duration_s=0.5, seed=1
            ),
            SyntheticReplay(
                "rgb0", Modality.FRAME, rate_hz=60, shape=(8, 8, 3), duration_s=0.5, seed=2
            ),
            SyntheticReplay("mic0", Modality.AUDIO, block=256, duration_s=0.5, seed=3),
            SyntheticReplay(
                "imu0",
                Modality.IMU,
                rate_hz=1000,
                duration_s=0.5,
                seed=4,
                device_offset_ns=7 * MS,
                latency_ns=200_000,
                latency_jitter_ns=50_000,
            ),
        ]
        return Aligner(
            srcs,
            AlignerConfig(
                tick_ns=10 * MS, throttle={"dvs0": ThrottleConfig(max_events_per_s=30_000)}
            ),
        )

    h1 = hash_ticks(run_ticks(build()))
    h2 = hash_ticks(run_ticks(build()))
    assert h1 == h2
    assert h1 == hash_ticks(run_ticks(build()))


def test_stats_and_jitter_metrics() -> None:
    srcs = [
        SyntheticReplay(
            "imu0",
            Modality.IMU,
            rate_hz=1000,
            duration_s=1.0,
            device_offset_ns=3 * MS,
            latency_ns=100_000,
            latency_jitter_ns=40_000,
        ),
        SyntheticReplay("rgb0", Modality.FRAME, rate_hz=60, shape=(4, 4, 1), duration_s=1.0),
    ]
    al = Aligner(srcs, AlignerConfig(tick_ns=10 * MS))
    ticks = run_ticks(al)
    st = al.stats()
    assert st["ticks"] == len(ticks) >= 99
    assert st["sync_jitter_p99_ns"] < 1 * MS
    assert st["staleness_p99_ns"] <= 10 * MS
    assert st["sources"]["imu0"]["samples_per_s"] == pytest.approx(1000, rel=0.05)
    assert st["sources"]["rgb0"]["dropped"] == 0
    assert al.specs()["imu0"].clock_mode is ClockMode.ESTIMATED


def test_replay_speed_paces_wall_clock() -> None:
    a = ListAdapter("a", [imu("a", t * MS) for t in range(0, 50, 10)], speed=1.0)
    t0 = time.monotonic()
    ticks = run_ticks(Aligner([a], AlignerConfig(tick_ns=10 * MS)))
    assert len(ticks) == 5
    assert time.monotonic() - t0 >= 0.04


def test_async_iteration_and_context_manager() -> None:
    async def go() -> list[int]:
        a = ListAdapter("a", [imu("a", t) for t in range(0, 400, 7)])
        out = []
        async with Aligner([a], AlignerConfig(tick_ns=10, speed=0)) as al:
            async for tick in al:
                out.append(tick.index)
        assert a.closed
        return out

    assert asyncio.run(go()) == list(range(40))


def test_accepts_prebuilt_streams_and_merges_metrics() -> None:
    st = Stream(ListAdapter("a", [imu("a", 1), imu("a", 2)]))
    al = Aligner([st], AlignerConfig(tick_ns=10))
    run_ticks(al)
    assert al.snapshot()["a.samples_in"] == 2


def test_empty_sources_produce_no_ticks() -> None:
    a = ListAdapter("a", [])
    a.time_range = lambda: (0, 0)  # type: ignore[method-assign]
    assert run_ticks(Aligner([a], AlignerConfig(tick_ns=10))) == []


# -- live mode ------------------------------------------------------------------------------


def _live(**cfg: object) -> tuple[Aligner, Stream, Stream]:
    a = Stream(ListAdapter("a", [], live=True))
    b = Stream(ListAdapter("b", [], modality=Modality.EVENT, live=True))
    al = Aligner(
        [a, b],
        AlignerConfig(tick_ns=10, grace_ns=2, start_ns=0, **cfg),  # type: ignore[arg-type]
        clock=ReplayClock(),
    )
    return al, a, b


def test_live_deadline_closes_ticks_and_counts_late() -> None:
    al, a, b = _live()
    assert al.live
    a.push(imu("a", 3))
    b.push(events("b", [4, 8, 13]))
    assert al.step_live(11) == []  # deadline is 12
    (t0,) = al.step_live(12)
    assert _ts(t0, "a") == [3] and [list(s.columns["t"]) for s in t0.samples["b"]] == [[4, 8]]
    a.push(imu("a", 7))  # arrives after tick 0 closed → late
    a.push(imu("a", 15))
    (t1,) = al.step_live(22)
    assert _ts(t1, "a") == [15]
    assert [list(s.columns["t"]) for s in t1.samples["b"]] == [[13]]
    assert al.snapshot()["a.late"] == 1
    ticks = al.step_live(52)
    assert [t.index for t in ticks] == [2, 3, 4]
    assert al.snapshot()["tick_lateness_ns.max"] == 20


def test_live_attach_next() -> None:
    al, a, _ = _live(late_policy=LatePolicy.ATTACH_NEXT)
    al.step_live(12)
    a.push(imu("a", 7))
    (t1,) = al.step_live(22)
    assert _ts(t1, "a") == [7]


def test_live_threaded_end_to_end() -> None:
    now = time.monotonic_ns()
    samples = [imu("a", now + i * MS) for i in range(30)]
    ad = ListAdapter("a", samples, live=True)
    al = Aligner([ad], AlignerConfig(tick_ns=10 * MS, grace_ns=1 * MS, start_ns=now))
    ticks = run_ticks(al)
    assert sum(len(t.samples["a"]) for t in ticks) == 30
    assert ad.closed


def test_live_producer_error_propagates() -> None:
    class Boom(ListAdapter):
        def _read_once(self):  # type: ignore[no-untyped-def]
            yield imu("a", 0)
            raise RuntimeError("device unplugged")

    al = Aligner([Boom("a", [imu("a", 0)], live=True)], AlignerConfig(tick_ns=MS))
    with pytest.raises(RuntimeError, match="unplugged"):
        run_ticks(al)


def test_live_stops_at_max_ticks() -> None:
    al, _, _ = _live(max_ticks=2)
    assert len(al.step_live(1000)) == 2
    assert al.stats()["sync_jitter_p99_ns"] == 0.0


def test_latency_config_reaches_streams_and_is_validated() -> None:
    src = SyntheticReplay("imu0", Modality.IMU, rate_hz=1000, duration_s=0.3, latency_ns=3 * MS)
    plain = Aligner([src], AlignerConfig(tick_ns=10 * MS))
    fixed = Aligner([src], AlignerConfig(tick_ns=10 * MS, latency_ns={"imu0": 3 * MS}))
    run_ticks(plain)
    run_ticks(fixed)
    # with the latency declared, aligned times match the true times (offset ≈ 0)
    assert abs(plain.specs()["imu0"].offset_ns - 3 * MS) < 100_000
    assert abs(fixed.specs()["imu0"].offset_ns) < 100_000
    assert fixed.specs()["imu0"].meta["latency_ns"] == 3 * MS
    with pytest.raises(ValueError, match="latency_ns config for unknown"):
        Aligner([src], AlignerConfig(latency_ns={"nope": 1}))
