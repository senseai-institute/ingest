# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from senseai_ingest.adapters.replay.synthetic import SyntheticReplay
from senseai_ingest.schema import hash_ticks
from senseai_ingest.stream import Modality, Sample, SourceSpec, event_count
from senseai_ingest.sync import Aligner, AlignerConfig, run_ticks
from senseai_ingest.throttle import Throttle, ThrottleConfig, ThrottlePolicy
from tests.helpers import events, imu

MS = 1_000_000
IMU_SPEC = SourceSpec("a", Modality.IMU, nominal_rate_hz=1000)
EV_SPEC = SourceSpec("e", Modality.EVENT)


def _run(th: Throttle, samples: list[Sample]) -> list[Sample]:
    out = [o for s in samples for o in th.apply(s)]
    return out + th.flush()


def test_unlimited_passthrough() -> None:
    th = Throttle(ThrottleConfig(), IMU_SPEC)
    s = imu("a", 1)
    assert th.apply(s) == [s] and th.flush() == [] and th.pressure == 0.0


def test_drop_samples_per_window() -> None:
    th = Throttle(ThrottleConfig(max_samples_per_s=500, budget_window_ns=10 * MS), IMU_SPEC)
    out = _run(th, [imu("a", i * MS) for i in range(100)])  # 1 kHz for 100 ms
    assert len(out) == 50
    assert [s.t_master_ns // MS for s in out[:6]] == [0, 1, 2, 3, 4, 10]
    assert th.dropped_samples == 50
    assert th.metrics.get("a.throttled_samples") == 50


def test_decimate_samples_keeps_every_nth() -> None:
    th = Throttle(ThrottleConfig(max_samples_per_s=250, policy=ThrottlePolicy.DECIMATE), IMU_SPEC)
    out = _run(th, [imu("a", i * MS) for i in range(20)])
    assert [s.t_master_ns // MS for s in out] == [0, 4, 8, 12, 16]


def test_drop_events_truncates_batches() -> None:
    th = Throttle(ThrottleConfig(max_events_per_s=500, budget_window_ns=10 * MS), EV_SPEC)
    batch = events("e", [i * 100_000 for i in range(1, 11)])  # 10 events in 1 ms
    out = _run(th, [batch])
    assert event_count(out[0]) == 5
    assert out[0].t_master_ns == 500_000
    assert th.pressure == 1.0
    assert _run(th, [events("e", [1_200_000])]) == []
    assert th.dropped_events == 6


def test_decimate_events_adapts_stride() -> None:
    th = Throttle(
        ThrottleConfig(
            max_events_per_s=1000, policy=ThrottlePolicy.DECIMATE, budget_window_ns=10 * MS
        ),
        EV_SPEC,
    )
    out = _run(th, [events("e", list(range(i * 100_000, i * 100_000 + 40))) for i in range(10)])
    kept = sum(event_count(s) for s in out)
    assert 0 < kept <= 400 and th.dropped_events == 400 - kept
    th2 = Throttle(ThrottleConfig(max_events_per_s=1e9, policy=ThrottlePolicy.DECIMATE), EV_SPEC)
    one = events("e", [1, 2])
    assert th2.apply(one) == [one]


def test_accumulate_coalesces_and_caps() -> None:
    th = Throttle(
        ThrottleConfig(
            max_events_per_s=3000,
            policy=ThrottlePolicy.ACCUMULATE,
            accumulate_window_ns=1 * MS,
            budget_window_ns=1 * MS,
        ),
        EV_SPEC,
    )
    a = events("e", [10, 20])  # x = 0,1  p = 0,1
    b = Sample(
        "e",
        Modality.EVENT,
        30,
        30,
        {
            "t": np.array([30], np.int64),
            "x": np.array([0], np.uint16),
            "y": np.array([0], np.uint16),
            "p": np.array([0], np.uint8),
        },
        {},
    )  # duplicate of first
    c = events("e", [2 * MS])
    out = _run(th, [a, b, c])
    assert len(out) == 2
    assert list(out[0].columns["t"]) == [20, 30]
    assert out[0].meta["accumulated"] == 2
    th_cap = Throttle(
        ThrottleConfig(
            max_events_per_s=2000,
            policy=ThrottlePolicy.ACCUMULATE,
            accumulate_window_ns=1 * MS,
            budget_window_ns=1 * MS,
        ),
        EV_SPEC,
    )
    out = _run(th_cap, [events("e", [1, 2, 3, 4, 5])])
    assert list(out[0].columns["t"]) == [4, 5]


def test_accumulate_on_non_event_falls_back_to_drop() -> None:
    th = Throttle(ThrottleConfig(max_samples_per_s=100, policy=ThrottlePolicy.ACCUMULATE), IMU_SPEC)
    assert len(_run(th, [imu("a", i * MS) for i in range(10)])) == 1


def test_from_mapping() -> None:
    cfg = ThrottleConfig.from_mapping({"max_events_per_s": 10, "policy": "decimate"})
    assert cfg.policy is ThrottlePolicy.DECIMATE and cfg.max_events_per_s == 10
    with pytest.raises(ValueError):
        ThrottleConfig.from_mapping({"policy": "nope"})


@pytest.mark.parametrize("policy", list(ThrottlePolicy))
def test_throttle_is_deterministic_in_replay(policy: ThrottlePolicy) -> None:
    def h() -> str:
        src = SyntheticReplay(
            "e", Modality.EVENT, rate_hz=1000, events_per_s=200_000, duration_s=0.2, seed=9
        )
        cfg = AlignerConfig(
            tick_ns=10 * MS, throttle={"e": ThrottleConfig(max_events_per_s=50_000, policy=policy)}
        )
        return hash_ticks(run_ticks(Aligner([src], cfg)))

    assert h() == h()
