# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic sources for tests, benchmarks and examples.

Optionally simulates a device clock (offset + drift) and transport latency/jitter, in which case
the adapter is ``ClockMode.ESTIMATED`` and stamps ``t_master_ns`` with the simulated host arrival
time, exactly like a live adapter would.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from senseai_ingest.adapters.base import ReplayAdapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["SyntheticReplay"]


class SyntheticReplay(ReplayAdapter):
    """Generate ``EVENT``/``FRAME``/``AUDIO``/``IMU`` (or any modality, as zeros) at a fixed rate.

    Args:
        source_id: source id.
        modality: modality to generate.
        rate_hz: samples per second (EVENT: batches per second; AUDIO: derived from
            ``sample_rate / block``).
        duration_s: recording length.
        t_start_ns: true time of the first sample.
        events_per_s: EVENT only.
        shape: FRAME/THERMAL ``(H, W[, C])``.
        sample_rate, block, channels: AUDIO only.
        device_offset_ns, drift_ppm: simulated device clock (``t_source`` vs true time).
        latency_ns, latency_jitter_ns: simulated arrival = true + latency + U(0, jitter).
        seed: RNG seed.
    """

    def __init__(
        self,
        source_id: str,
        modality: Modality,
        *,
        rate_hz: float = 100.0,
        duration_s: float = 1.0,
        t_start_ns: int = 0,
        events_per_s: float = 1e5,
        shape: tuple[int, ...] = (48, 64, 3),
        sample_rate: int = 48_000,
        block: int = 1024,
        channels: int = 4,
        width: int = 640,
        height: int = 480,
        device_offset_ns: int = 0,
        drift_ppm: float = 0.0,
        latency_ns: int = 0,
        latency_jitter_ns: int = 0,
        seed: int = 0,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self._sid = source_id
        self._mod = Modality(modality)
        if self._mod is Modality.AUDIO:
            rate_hz = sample_rate / block
        self.rate_hz = float(rate_hz)
        self.duration_ns = int(duration_s * 1e9)
        self.t_start_ns = int(t_start_ns)
        self.events_per_s = events_per_s
        self.shape = shape
        self.sample_rate = sample_rate
        self.block = block
        self.channels = channels
        self.width, self.height = width, height
        self.device_offset_ns = int(device_offset_ns)
        self.drift_ppm = drift_ppm
        self.latency_ns = int(latency_ns)
        self.latency_jitter_ns = int(latency_jitter_ns)
        self.seed = seed
        self.estimated = bool(device_offset_ns or drift_ppm or latency_ns or latency_jitter_ns)

    def spec(self) -> SourceSpec:
        """Describe the synthetic source."""
        meta: dict[str, Any] = {"synthetic": True}
        if self._mod is Modality.EVENT:
            meta |= {"width": self.width, "height": self.height, "event_rate_hz": self.events_per_s}
        elif self._mod is Modality.FRAME:
            meta |= {"encoding": "rgb8" if self.shape[-1] == 3 else "mono8"}
        elif self._mod is Modality.AUDIO:
            meta |= {"sample_rate": self.sample_rate, "channels": self.channels}
        elif self._mod is Modality.IMU:
            meta |= {"frame_id": "imu"}
        return SourceSpec(
            source_id=self._sid,
            modality=self._mod,
            nominal_rate_hz=self.rate_hz,
            meta=meta,
            clock_mode=ClockMode.ESTIMATED if self.estimated else ClockMode.FIXED,
        )

    def open(self) -> None:
        """No resources to acquire."""

    def close(self) -> None:
        """No resources to release."""

    def time_range(self) -> tuple[int, int]:
        """Return true-time range of the recording."""
        return self.t_start_ns, self.t_start_ns + self.duration_ns

    def device_time(self, t_true_ns: int) -> int:
        """Map true time to the simulated device clock."""
        rel = t_true_ns - self.t_start_ns
        return int(t_true_ns - self.device_offset_ns + round(self.drift_ppm * 1e-6 * rel))

    def _read_once(self) -> Iterator[Sample]:
        rng = np.random.default_rng(self.seed)
        period = 1e9 / self.rate_hz
        n = int(self.duration_ns // period)
        frames: list[np.ndarray[Any, Any]] = []
        if self._mod in (Modality.FRAME, Modality.THERMAL):
            dt = np.uint8 if self._mod is Modality.FRAME else np.float32
            frames = [rng.integers(0, 255, size=self.shape).astype(dt) for _ in range(4)]
        for i in range(n):
            t_true = self.t_start_ns + round((i + 1) * period)
            t_src = self.device_time(t_true)
            if self.estimated:
                jit = int(rng.integers(0, self.latency_jitter_ns + 1))
                t_master = t_true + self.latency_ns + jit
            else:
                t_master = t_true
            if self.start_ns is not None and t_master < self.start_ns:
                continue
            if self.end_ns is not None and t_master >= self.end_ns:
                break
            data: np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]]
            meta: dict[str, Any] = {}
            if self._mod is Modality.EVENT:
                k = max(1, int(self.events_per_s / self.rate_hz))
                lo = t_true - int(period)
                t_ev = np.sort(rng.integers(lo + 1, t_true + 1, size=k)).astype(np.int64)
                t_ev[-1] = t_true
                # FIXED: t column is master time; ESTIMATED: source time (Stream shifts it).
                t_ev = t_ev + (t_src - t_true) if self.estimated else t_ev
                data = {
                    "t": t_ev,
                    "x": rng.integers(0, self.width, size=k).astype(np.uint16),
                    "y": rng.integers(0, self.height, size=k).astype(np.uint16),
                    "p": rng.integers(0, 2, size=k).astype(np.uint8),
                }
            elif self._mod in (Modality.FRAME, Modality.THERMAL):
                data = frames[i % len(frames)]
            elif self._mod is Modality.AUDIO:
                data = rng.integers(-3000, 3000, size=(self.block, self.channels)).astype(np.int16)
            elif self._mod is Modality.IMU:
                data = np.concatenate([[0.0, 0.0, 9.81], rng.normal(0, 0.01, 3)])
            else:
                data = np.zeros(1, dtype=np.float32)
            yield Sample(self._sid, self._mod, t_src, t_master, data, meta)
