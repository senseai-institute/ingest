# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Master clock and per-source offset/drift estimation.

Model::

    t_master = t_source + offset_ns + drift_ppm * 1e-6 * (t_source - offset_ref_ns)

``offset_ns`` is the median of ``t_arrival - t_source`` over the first ``k`` samples; every
``drift_window_ns`` of source time the offset is re-estimated and ``drift_ppm`` is fitted from
consecutive estimates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = [
    "ClockState",
    "ClockSync",
    "MasterClock",
    "MonotonicClock",
    "OffsetEstimator",
    "ReplayClock",
]


class MasterClock(Protocol):
    """Source of master time in nanoseconds."""

    def now_ns(self) -> int:
        """Return the current master time."""
        ...


class MonotonicClock:
    """Live master clock: ``time.monotonic_ns()``."""

    def now_ns(self) -> int:
        """Return ``time.monotonic_ns()``."""
        return time.monotonic_ns()


class ReplayClock:
    """Replay master clock: a cursor that only moves forward. Also usable as a manual test clock."""

    def __init__(self, t0_ns: int = 0) -> None:
        self._t = int(t0_ns)

    def now_ns(self) -> int:
        """Return the cursor."""
        return self._t

    def advance_to(self, t_ns: int) -> None:
        """Move the cursor to ``t_ns`` (never backwards)."""
        self._t = max(self._t, int(t_ns))


@dataclass(frozen=True, slots=True)
class ClockState:
    """Snapshot of an :class:`OffsetEstimator`."""

    offset_ns: int
    offset_ref_ns: int
    drift_ppm: float
    jitter_ns: int
    offset_samples: int


class OffsetEstimator:
    """Robust offset + drift estimator from ``(t_source, t_arrival)`` pairs.

    Until ``k`` samples have been seen, the median of the samples so far is used (provisional).
    Each subsequent window spanning ``drift_window_ns`` of source time yields a new median offset
    at the window's median source time; drift is the slope between consecutive estimates.
    """

    def __init__(
        self, k: int = 200, drift_window_ns: int = 10_000_000_000, max_window: int = 20_000
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.drift_window_ns = int(drift_window_ns)
        self.max_window = max_window
        self._src: list[int] = []
        self._d: list[int] = []
        self._offset = 0
        self._ref = 0
        self._drift_ppm = 0.0
        self._jitter = 0
        self._n = 0
        self._ready = False
        self._win_start: int | None = None

    @property
    def ready(self) -> bool:
        """``True`` once the initial ``k``-sample estimate is in place."""
        return self._ready

    def _fit(self) -> tuple[int, int]:
        d = np.asarray(self._d, dtype=np.int64)
        src = np.asarray(self._src, dtype=np.int64)
        return int(np.median(d)), int(np.median(src))

    def _residual_p99(self) -> int:
        src = np.asarray(self._src, dtype=np.int64)
        d = np.asarray(self._d, dtype=np.int64)
        pred = self._offset + self._drift_ppm * 1e-6 * (src - self._ref)
        return int(np.quantile(np.abs(d - pred), 0.99))

    def observe(self, t_source_ns: int, t_arrival_ns: int) -> None:
        """Record one ``(t_source, t_arrival)`` pair."""
        self._n += 1
        if len(self._d) < self.max_window:
            self._src.append(int(t_source_ns))
            self._d.append(int(t_arrival_ns) - int(t_source_ns))
        if not self._ready:
            self._offset, self._ref = self._fit()
            if len(self._d) >= self.k:
                self._ready = True
                self._jitter = self._residual_p99()
                self._src, self._d = [], []
                self._win_start = int(t_source_ns)
            return
        assert self._win_start is not None
        if int(t_source_ns) - self._win_start >= self.drift_window_ns and len(self._d) >= 2:
            new_offset, new_ref = self._fit()
            if new_ref != self._ref:
                self._drift_ppm = (new_offset - self._offset) / (new_ref - self._ref) * 1e6
            self._offset, self._ref = new_offset, new_ref
            self._jitter = self._residual_p99()
            self._src, self._d = [], []
            self._win_start = int(t_source_ns)

    def to_master(self, t_source_ns: int) -> int:
        """Map a source timestamp onto the master clock."""
        return int(
            t_source_ns
            + self._offset
            + round(self._drift_ppm * 1e-6 * (int(t_source_ns) - self._ref))
        )

    def state(self) -> ClockState:
        """Return the current estimate."""
        return ClockState(
            offset_ns=self._offset,
            offset_ref_ns=self._ref,
            drift_ppm=self._drift_ppm,
            jitter_ns=self._jitter,
            offset_samples=min(self._n, self.k) if not self._ready else self.k,
        )


class ClockSync:
    """Applies a source's clock model to its samples.

    * ``FIXED``: the adapter already set ``t_master_ns``; samples pass through, error 0.
    * ``ESTIMATED``: the adapter set ``t_master_ns`` to the host *arrival* time. That arrival is
      fed to the :class:`OffsetEstimator` and ``t_master_ns`` is replaced by the model
      prediction. EVENT ``t`` columns (given in source time) are shifted onto the master clock.
      The returned alignment error is ``|prediction - arrival|``.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        mode: ClockMode | None = None,
        k: int = 200,
        drift_window_ns: int = 10_000_000_000,
    ) -> None:
        self.mode = mode if mode is not None else spec.clock_mode
        self.estimator = OffsetEstimator(k=k, drift_window_ns=drift_window_ns)

    def align(self, sample: Sample) -> tuple[Sample, int]:
        """Return ``(aligned_sample, alignment_error_ns)``."""
        if self.mode is ClockMode.FIXED:
            return sample, 0
        arrival = sample.t_master_ns
        self.estimator.observe(sample.t_source_ns, arrival)
        t_master = self.estimator.to_master(sample.t_source_ns)
        delta = t_master - sample.t_source_ns
        data = sample.data
        if sample.modality is Modality.EVENT and isinstance(data, dict) and "t" in data:
            data = {**data, "t": data["t"].astype(np.int64, copy=False) + delta}
        return replace(sample, t_master_ns=t_master, data=data), abs(t_master - arrival)

    def update_spec(self, spec: SourceSpec) -> SourceSpec:
        """Return ``spec`` with the current clock state recorded in it."""
        if self.mode is ClockMode.FIXED:
            return replace(spec, clock_mode=ClockMode.FIXED)
        st = self.estimator.state()
        return replace(
            spec,
            clock_mode=ClockMode.ESTIMATED,
            offset_ns=st.offset_ns,
            offset_ref_ns=st.offset_ref_ns,
            drift_ppm=st.drift_ppm,
            jitter_ns=st.jitter_ns,
            offset_samples=st.offset_samples,
        )
