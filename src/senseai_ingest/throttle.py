# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Sample-rate and event-rate throttling.

All decisions depend only on sample timestamps and sequence numbers — never wall time — so the
output is deterministic in replay. Budget windows are aligned to absolute master time
(``t_master_ns // budget_window_ns``) so results don't depend on where a replay starts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np

from senseai_ingest.metrics import Registry
from senseai_ingest.stream import Modality, Sample, SourceSpec, event_count, slice_events

__all__ = ["Throttle", "ThrottleConfig", "ThrottlePolicy"]


class ThrottlePolicy(StrEnum):
    """What to do with data above the budget."""

    DROP = "drop"  # keep the first samples/events of each budget window
    DECIMATE = "decimate"  # keep every n-th sample/event by sequence number
    ACCUMULATE = "accumulate"  # EVENT: merge into fixed windows, coalescing (x, y, p)


@dataclass(frozen=True, slots=True)
class ThrottleConfig:
    """Throttle settings for one source.

    Attributes:
        max_samples_per_s: cap on samples/s (non-EVENT sources), ``None`` = unlimited.
        max_events_per_s: cap on events/s (EVENT sources), ``None`` = unlimited.
        policy: see :class:`ThrottlePolicy`.
        budget_window_ns: window over which the per-second budget is enforced.
        accumulate_window_ns: batch window for ``ACCUMULATE``.
    """

    max_samples_per_s: float | None = None
    max_events_per_s: float | None = None
    policy: ThrottlePolicy = ThrottlePolicy.DROP
    budget_window_ns: int = 10_000_000
    accumulate_window_ns: int = 1_000_000

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> ThrottleConfig:
        """Build from a plain dict (YAML/CLI)."""
        return cls(
            max_samples_per_s=cfg.get("max_samples_per_s"),
            max_events_per_s=cfg.get("max_events_per_s"),
            policy=ThrottlePolicy(cfg.get("policy", "drop")),
            budget_window_ns=int(cfg.get("budget_window_ns", 10_000_000)),
            accumulate_window_ns=int(cfg.get("accumulate_window_ns", 1_000_000)),
        )


class Throttle:
    """Stateful, deterministic throttle for one source. Call :meth:`flush` at end of stream."""

    def __init__(
        self, config: ThrottleConfig, spec: SourceSpec, *, metrics: Registry | None = None
    ) -> None:
        self.config = config
        self.spec = spec
        self.metrics = metrics if metrics is not None else Registry()
        self._name = spec.source_id
        self._is_event = spec.modality is Modality.EVENT
        self._win = -1
        self._win_used = 0
        self._prev_win_count = 0
        self._win_count = 0
        self._seq = 0  # sample sequence number
        self._eseq = 0  # event sequence number
        self._acc: list[Sample] = []
        self._acc_win = -1
        self.dropped_samples = 0
        self.dropped_events = 0

    # -- helpers ----------------------------------------------------------------------------
    @property
    def _budget(self) -> float | None:
        rate = self.config.max_events_per_s if self._is_event else self.config.max_samples_per_s
        if rate is None:
            return None
        return rate * self.config.budget_window_ns * 1e-9

    def _roll(self, t: int) -> None:
        w = t // self.config.budget_window_ns
        if w != self._win:
            self._prev_win_count = self._win_count if w == self._win + 1 else 0
            self._win = w
            self._win_used = 0
            self._win_count = 0

    def _drop(self, samples: int = 0, events: int = 0) -> None:
        if samples:
            self.dropped_samples += samples
            self.metrics.inc(f"{self._name}.throttled_samples", samples)
        if events:
            self.dropped_events += events
            self.metrics.inc(f"{self._name}.throttled_events", events)

    @property
    def pressure(self) -> float:
        """Fraction of the current window's budget already used (backpressure signal)."""
        b = self._budget
        return 0.0 if not b else self._win_used / b

    # -- policies ---------------------------------------------------------------------------
    def apply(self, sample: Sample) -> list[Sample]:
        """Return the samples to emit for one input sample (possibly none, possibly merged)."""
        budget = self._budget
        if budget is None:
            return [sample]
        policy = self.config.policy
        if policy is ThrottlePolicy.ACCUMULATE and self._is_event:
            return self._accumulate(sample, budget)
        self._roll(sample.t_master_ns)
        if self._is_event:
            return self._events(sample, budget, decimate=policy is ThrottlePolicy.DECIMATE)
        return self._samples(sample, budget, decimate=policy is ThrottlePolicy.DECIMATE)

    def _samples(self, sample: Sample, budget: float, *, decimate: bool) -> list[Sample]:
        seq = self._seq
        self._seq += 1
        if decimate:
            nominal = self.spec.nominal_rate_hz
            cap = self.config.max_samples_per_s or nominal
            n = max(1, math.ceil(nominal / cap)) if nominal > 0 else 1
            if seq % n:
                self._drop(samples=1)
                return []
            return [sample]
        if self._win_used + 1 > budget:
            self._drop(samples=1)
            return []
        self._win_used += 1
        return [sample]

    def _events(self, sample: Sample, budget: float, *, decimate: bool) -> list[Sample]:
        n_ev = event_count(sample)
        self._win_count += n_ev
        if decimate:
            # Stride derived from the previous window's observed rate (data-only → deterministic).
            ref = max(self._prev_win_count, self._win_count)
            stride = max(1, math.ceil(ref / budget)) if budget > 0 else n_ev + 1
            start = self._eseq
            self._eseq += n_ev
            idx = np.arange(start, start + n_ev)
            keep = np.flatnonzero(idx % stride == 0)
            self._drop(events=n_ev - len(keep))
            if len(keep) == 0:
                return []
            if len(keep) == n_ev:
                return [sample]
            return [_take_events(sample, keep)]
        room = max(0, int(budget) - self._win_used)
        k = min(room, n_ev)
        self._win_used += k
        self._drop(events=n_ev - k)
        if k == 0:
            return []
        return [sample] if k == n_ev else [slice_events(sample, 0, k)]

    def _accumulate(self, sample: Sample, budget: float) -> list[Sample]:
        out: list[Sample] = []
        w = sample.t_master_ns // self.config.accumulate_window_ns
        if self._acc and w != self._acc_win:
            out.extend(self._emit_acc(budget))
        self._acc_win = w
        self._acc.append(sample)
        return out

    def _emit_acc(self, budget: float) -> list[Sample]:
        if not self._acc:
            return []
        parts = self._acc
        self._acc = []
        assert all(isinstance(p.data, dict) for p in parts)
        cols = {k: np.concatenate([p.data[k] for p in parts]) for k in parts[0].data}
        n_in = len(cols["t"])
        # Coalesce identical (x, y, p) within the window, keeping the latest event.
        key = (
            cols["x"].astype(np.uint64) << np.uint64(32)
            | cols["y"].astype(np.uint64) << np.uint64(1)
            | cols["p"].astype(np.uint64)
        )
        _, last_rev = np.unique(key[::-1], return_index=True)
        keep = np.sort(n_in - 1 - last_rev)
        # Window budget = per-second cap scaled to the accumulate window; keep the latest.
        cap = int(budget * self.config.accumulate_window_ns / self.config.budget_window_ns)
        if cap >= 0 and len(keep) > max(cap, 1):
            keep = keep[-max(cap, 1) :]
        merged = {k: v[keep] for k, v in cols.items()}
        self._drop(events=n_in - len(keep))
        last = parts[-1]
        t_last = int(merged["t"][-1])
        delta = t_last - last.t_master_ns
        return [
            replace(
                last,
                data=merged,
                t_master_ns=t_last,
                t_source_ns=last.t_source_ns + delta,
                meta={**last.meta, "accumulated": len(parts)},
            )
        ]

    def flush(self) -> list[Sample]:
        """Emit anything still held by ``ACCUMULATE``."""
        budget = self._budget
        if budget is None:
            return []
        return self._emit_acc(budget)


def _take_events(sample: Sample, idx: np.ndarray[Any, Any]) -> Sample:
    assert isinstance(sample.data, dict)
    cols = {k: v[idx] for k, v in sample.data.items()}
    t_last = int(cols["t"][-1])
    delta = t_last - sample.t_master_ns
    return replace(sample, data=cols, t_master_ns=t_last, t_source_ns=sample.t_source_ns + delta)
