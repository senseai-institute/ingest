# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from senseai_ingest.clock import ClockSync, MonotonicClock, OffsetEstimator, ReplayClock
from senseai_ingest.stream import ClockMode, Modality, SourceSpec
from tests.helpers import imu


def test_monotonic_clock_increases() -> None:
    c = MonotonicClock()
    a = c.now_ns()
    assert c.now_ns() >= a


def test_replay_clock_never_goes_back() -> None:
    c = ReplayClock(10)
    c.advance_to(20)
    c.advance_to(5)
    assert c.now_ns() == 20


def test_estimator_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        OffsetEstimator(k=0)


def test_offset_median_is_robust_to_latency_spikes() -> None:
    rng = np.random.default_rng(1)
    est = OffsetEstimator(k=200)
    offset = 123_456_789
    for i in range(200):
        t_src = i * 1_000_000
        lat = int(rng.integers(0, 50_000))
        if i % 10 == 0:
            lat += 20_000_000  # 10 % scheduling spikes
        est.observe(t_src, t_src + offset + lat)
    assert est.ready
    st = est.state()
    assert abs(st.offset_ns - offset - 25_000) < 15_000  # median of U(0,50us) ≈ 25us
    assert st.offset_samples == 200
    assert st.jitter_ns > 0


def test_provisional_estimate_before_k() -> None:
    est = OffsetEstimator(k=10)
    est.observe(100, 600)
    assert not est.ready
    assert est.to_master(200) == 700
    assert est.state().offset_samples == 1


def test_drift_is_estimated_from_synthetic_clock() -> None:
    drift_ppm = 50.0  # device clock runs 50 ppm slow vs. host
    offset = 5_000_000
    est = OffsetEstimator(k=200, drift_window_ns=1_000_000_000)
    rng = np.random.default_rng(0)
    period = 1_000_000  # 1 kHz
    for i in range(30_000):  # 30 s
        t_src = i * period
        t_true = t_src + offset + int(drift_ppm * 1e-6 * t_src)
        est.observe(t_src, t_true + int(rng.integers(0, 20_000)))
    st = est.state()
    assert st.drift_ppm == pytest.approx(drift_ppm, rel=0.05)
    # prediction error at the end of the run is well under 1 ms
    t_src = 30_000 * period
    t_true = t_src + offset + int(drift_ppm * 1e-6 * t_src)
    assert abs(est.to_master(t_src) - t_true) < 100_000


def test_clock_sync_fixed_passthrough() -> None:
    cs = ClockSync(SourceSpec("a", Modality.IMU))
    s = imu("a", 10, t_src=3)
    out, err = cs.align(s)
    assert out is s and err == 0
    assert cs.update_spec(SourceSpec("a", Modality.IMU)).clock_mode is ClockMode.FIXED


def test_clock_sync_estimated_reports_residual() -> None:
    cs = ClockSync(SourceSpec("a", Modality.IMU, clock_mode=ClockMode.ESTIMATED), k=3)
    errs = []
    for i, lat in enumerate([0, 10, 20, 10, 0]):
        out, err = cs.align(imu("a", i * 100 + 1000 + lat, t_src=i * 100))
        errs.append(err)
    assert out.t_master_ns == 400 + 1010
    assert errs[-1] == 10
    spec = cs.update_spec(SourceSpec("a", Modality.IMU))
    assert spec.clock_mode is ClockMode.ESTIMATED and spec.offset_ns == 1010


def test_known_latency_is_removed_from_offset() -> None:
    # device clock 0 offset; every sample arrives exactly 4 ms late
    spec = SourceSpec("rgb0", Modality.FRAME, clock_mode=ClockMode.ESTIMATED)
    naive = ClockSync(spec, k=5)
    corrected = ClockSync(spec, k=5, latency_ns=4_000_000)
    for i in range(10):
        t = i * 1_000_000
        out_n, _ = naive.align(imu("rgb0", t + 4_000_000, t_src=t))
        out_c, err = corrected.align(imu("rgb0", t + 4_000_000, t_src=t))
    assert out_n.t_master_ns == t + 4_000_000  # latency absorbed into the offset
    assert out_c.t_master_ns == t and err == 0  # latency removed
    assert corrected.update_spec(spec).meta["latency_ns"] == 4_000_000
    assert ClockSync(SourceSpec("a", Modality.IMU, meta={"latency_ns": 7})).latency_ns == 7
    with pytest.raises(ValueError, match="latency"):
        ClockSync(spec, latency_ns=-1)
