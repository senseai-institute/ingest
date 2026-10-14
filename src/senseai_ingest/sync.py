# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Multi-source aligner: emits one :class:`AlignedTick` per ``tick_ns`` on the master clock.

Tick ``k`` covers ``[T_k - tick_ns, T_k)`` with ``T_k = t0 + (k + 1) * tick_ns``.

* **Replay** (watermark): tick ``T_k`` closes once every source has produced a sample with
  ``t_master_ns >= T_k`` or is exhausted. Output depends only on the data → deterministic.
* **Live** (deadline): tick ``T_k`` closes at master time ``T_k + grace_ns``. Samples whose
  tick already closed are *late*: counted, then dropped or attached to the next open tick.

Guarantee: no sample appears in a tick whose ``t_master_ns`` (window end) is ``<=`` the sample's
own ``t_master_ns``. EVENT batches are split at tick boundaries.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Literal, Self

import numpy as np
import structlog

from senseai_ingest.adapters.base import Adapter, ReplayAdapter
from senseai_ingest.clock import MasterClock, MonotonicClock, ReplayClock
from senseai_ingest.metrics import Registry
from senseai_ingest.stream import (
    AlignedTick,
    Modality,
    Sample,
    SourceSpec,
    Stream,
    _Item,
    event_count,
    slice_events,
)
from senseai_ingest.throttle import ThrottleConfig

__all__ = ["Aligner", "AlignerConfig", "LatePolicy"]

log = structlog.get_logger(__name__)

_Step = tuple[Literal["wait"], float] | tuple[Literal["tick"], AlignedTick]


class LatePolicy(StrEnum):
    """What to do with a sample that arrives after its tick closed."""

    DROP = "drop"
    ATTACH_NEXT = "attach_next"


@dataclass(frozen=True, slots=True)
class AlignerConfig:
    """Aligner configuration.

    Attributes:
        tick_ns: tick length.
        live: deadline mode; ``None`` = live iff any adapter ``is_live``.
        late_policy: drop late samples or attach them to the next open tick.
        grace_ns: live only — how long after ``T_k`` to wait before closing tick ``k``.
        start_ns: ``t0``; ``None`` = earliest sample (replay) or now (live).
        end_ns: stop once the tick window start reaches ``end_ns``; later samples are discarded.
        max_ticks: stop after this many ticks.
        speed: replay pacing (1.0 real time, 0 as fast as possible); ``None`` = from adapters.
        split_events: split EVENT batches at tick boundaries.
        estimate_offsets: force clock estimation on (``True``) / off (``False``) for all sources.
        throttle: per-source throttle settings.
        buffer_max_samples / buffer_max_bytes: per-source ring-buffer bounds.
    """

    tick_ns: int = 10_000_000
    live: bool | None = None
    late_policy: LatePolicy = LatePolicy.DROP
    grace_ns: int = 2_000_000
    start_ns: int | None = None
    end_ns: int | None = None
    max_ticks: int | None = None
    speed: float | None = None
    split_events: bool = True
    estimate_offsets: bool | None = None
    throttle: Mapping[str, ThrottleConfig] = field(default_factory=dict)
    buffer_max_samples: int = 65_536
    buffer_max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.tick_ns <= 0:
            raise ValueError("tick_ns must be > 0")
        if self.speed is not None and self.speed < 0:
            raise ValueError("speed must be >= 0")


class Aligner:
    """Aligns N sources onto one tick grid. Iterate it (sync or async) to get ticks."""

    def __init__(
        self,
        sources: Sequence[Adapter | Stream],
        config: AlignerConfig | None = None,
        *,
        clock: MasterClock | None = None,
    ) -> None:
        self.config = cfg = config if config is not None else AlignerConfig()
        if not sources:
            raise ValueError("Aligner needs at least one source")
        self.metrics = Registry()
        streams: list[Stream] = []
        for src in sources:
            if isinstance(src, Stream):
                streams.append(src)
            else:
                streams.append(
                    Stream(
                        src,
                        estimate_offsets=cfg.estimate_offsets,
                        throttle=cfg.throttle.get(src.source_id),
                        max_samples=cfg.buffer_max_samples,
                        max_bytes=cfg.buffer_max_bytes,
                        metrics=self.metrics,
                    )
                )
        ids = [s.source_id for s in streams]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate source ids: {sorted(ids)}")
        unknown = set(cfg.throttle) - set(ids)
        if unknown:
            raise ValueError(f"throttle config for unknown sources: {sorted(unknown)}")
        # Deterministic processing order.
        self.streams: list[Stream] = sorted(streams, key=lambda s: s.source_id)
        self.source_ids: list[str] = [s.source_id for s in self.streams]
        adapters = [s.adapter for s in self.streams]
        self.live = cfg.live if cfg.live is not None else any(a.is_live for a in adapters)
        self.speed = cfg.speed if cfg.speed is not None else self._adapter_speed(adapters)
        self.clock: MasterClock = (
            clock if clock is not None else (MonotonicClock() if self.live else ReplayClock())
        )
        self.t0_ns: int | None = None
        self.ticks_emitted = 0
        self._opened = False
        self._done = False

    @staticmethod
    def _adapter_speed(adapters: Sequence[Adapter]) -> float:
        speeds = {a.speed for a in adapters if isinstance(a, ReplayAdapter)}
        if len(speeds) > 1:
            raise ValueError(f"replay adapters disagree on speed: {sorted(speeds)}")
        return speeds.pop() if speeds else 0.0

    # -- lifecycle --------------------------------------------------------------------------
    def open(self) -> None:
        """Open all streams (and start producer threads in live mode)."""
        if self._opened:
            return
        for s in self.streams:
            s.open()
            if self.live:
                s.start_producer()
        self._opened = True
        log.info(
            "aligner.open", sources=self.source_ids, live=self.live, tick_ns=self.config.tick_ns
        )

    def close(self) -> None:
        """Close all streams."""
        for s in self.streams:
            s.close()
        self._opened = False

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def specs(self) -> dict[str, SourceSpec]:
        """Return each source's spec with its current clock-alignment state."""
        return {s.source_id: s.spec() for s in self.streams}

    # -- iteration --------------------------------------------------------------------------
    def __iter__(self) -> Iterator[AlignedTick]:
        """Yield ticks synchronously (blocking sleeps for pacing / live deadlines)."""
        self.open()
        for step in self._steps():
            if step[0] == "wait":
                time.sleep(step[1])
            else:
                yield step[1]

    async def __aiter__(self) -> AsyncIterator[AlignedTick]:
        """Yield ticks asynchronously."""
        self.open()
        n = 0
        for step in self._steps():
            if step[0] == "wait":
                await asyncio.sleep(step[1])
            else:
                yield step[1]
                n += 1
                if n % 16 == 0:
                    await asyncio.sleep(0)

    def _steps(self) -> Iterator[_Step]:
        return self._live_steps() if self.live else self._replay_steps()

    # -- replay -----------------------------------------------------------------------------
    def _first_t(self, s: Stream) -> int | None:
        while len(s.buffer) == 0:
            if not s.pull():
                break
        head = s.buffer.peek()
        if head is None:
            return None
        if head.sample.modality is Modality.EVENT and isinstance(head.sample.data, dict):
            t = head.sample.data["t"]
            if len(t):
                return int(t[0])
        return head.sample.t_master_ns

    def _replay_steps(self) -> Iterator[_Step]:
        cfg = self.config
        if self.t0_ns is None:
            if cfg.start_ns is not None:
                self.t0_ns = cfg.start_ns
            else:
                firsts = [t for t in (self._first_t(s) for s in self.streams) if t is not None]
                if not firsts:
                    return
                self.t0_ns = min(firsts)
        t0 = self.t0_ns
        wall0 = time.monotonic()
        while not self._finished_by_config():
            k = self.ticks_emitted
            t_close = t0 + (k + 1) * cfg.tick_ns
            if self.speed > 0:
                target = wall0 + (t_close - t0) * 1e-9 / self.speed
                delay = target - time.monotonic()
                if delay > 0:
                    yield ("wait", delay)
            if isinstance(self.clock, ReplayClock):
                self.clock.advance_to(t_close)
            tick = self._collect(k, t_close, pull=True)
            yield ("tick", tick)
            if all(s.exhausted and len(s.buffer) == 0 for s in self.streams):
                self._done = True
                return

    def _finished_by_config(self) -> bool:
        cfg = self.config
        if self._done:
            return True
        if cfg.max_ticks is not None and self.ticks_emitted >= cfg.max_ticks:
            return True
        if cfg.end_ns is not None and self.t0_ns is not None:
            lo = self.t0_ns + self.ticks_emitted * cfg.tick_ns
            if lo >= cfg.end_ns:
                return True
        return False

    # -- live -------------------------------------------------------------------------------
    def step_live(self, now_ns: int) -> list[AlignedTick]:
        """Close every tick whose deadline ``T_k + grace_ns`` is ``<= now_ns``.

        This is the live-mode core, exposed so it can be driven by an external loop or a test.
        """
        cfg = self.config
        if self.t0_ns is None:
            self.t0_ns = cfg.start_ns if cfg.start_ns is not None else now_ns
        out: list[AlignedTick] = []
        while not self._finished_by_config():
            k = self.ticks_emitted
            t_close = self.t0_ns + (k + 1) * cfg.tick_ns
            if now_ns < t_close + cfg.grace_ns:
                break
            self.metrics.summary("tick_lateness_ns").observe(
                max(0, now_ns - (t_close + cfg.grace_ns))
            )
            out.append(self._collect(k, t_close, pull=False))
        return out

    def _live_steps(self) -> Iterator[_Step]:
        cfg = self.config
        while not self._finished_by_config():
            for s in self.streams:
                s.raise_if_failed()
            drained = all(s.exhausted and len(s.buffer) == 0 for s in self.streams)
            for tick in self.step_live(self.clock.now_ns()):
                yield ("tick", tick)
            if drained:
                self._done = True
                return
            assert self.t0_ns is not None
            nxt = self.t0_ns + (self.ticks_emitted + 1) * cfg.tick_ns + cfg.grace_ns
            yield ("wait", max(0.0, min((nxt - self.clock.now_ns()) * 1e-9, 0.005)))

    # -- tick assembly ----------------------------------------------------------------------
    def _collect(self, k: int, t_close: int, *, pull: bool) -> AlignedTick:
        cfg = self.config
        assert self.t0_ns is not None
        lo = t_close - cfg.tick_ns
        samples: dict[str, list[Sample]] = {}
        jitter = 0
        staleness = 0
        for s in self.streams:
            got, err = self._collect_source(s, k, lo, t_close, pull=pull)
            samples[s.source_id] = got
            jitter = max(jitter, err)
            if got:
                staleness = max(staleness, t_close - got[-1].t_master_ns)
                self.metrics.inc(f"{s.source_id}.samples_out", len(got))
                if s.modality is Modality.EVENT:
                    self.metrics.inc(f"{s.source_id}.events_out", sum(event_count(x) for x in got))
        self.metrics.summary("sync_jitter_ns").observe(jitter)
        self.metrics.summary("staleness_ns").observe(staleness)
        self.metrics.inc("ticks")
        self.ticks_emitted += 1
        return AlignedTick(index=k, t_master_ns=t_close, tick_ns=cfg.tick_ns, samples=samples)

    def _collect_source(
        self, s: Stream, k: int, lo: int, hi: int, *, pull: bool
    ) -> tuple[list[Sample], int]:
        out: list[Sample] = []
        err = 0
        buf = s.buffer
        split = self.config.split_events and s.modality is Modality.EVENT
        while True:
            item = buf.peek()
            if item is None:
                if pull and s.pull():
                    continue
                break
            smp = item.sample
            if smp.t_master_ns >= hi:
                if split and isinstance(smp.data, dict) and len(smp.data["t"]):
                    t = smp.data["t"]
                    cut = int(np.searchsorted(t, hi, side="left"))
                    if cut > 0:
                        head = slice_events(smp, 0, cut)
                        buf.replace_head(_Item(slice_events(smp, cut, len(t)), item.align_err_ns))
                        err = max(err, self._admit(s, k, lo, _Item(head, item.align_err_ns), out))
                break
            buf.pop()
            err = max(err, self._admit(s, k, lo, item, out))
        if self.config.late_policy is LatePolicy.ATTACH_NEXT and len(out) > 1:
            out.sort(key=lambda x: x.t_master_ns)  # stable; keeps the per-tick order contract
        return out, err

    def _admit(self, s: Stream, k: int, lo: int, item: _Item, out: list[Sample]) -> int:
        """Route one sample (``t < hi`` guaranteed) into ``out``; return its alignment error."""
        assert self.t0_ns is not None
        cfg = self.config
        smp = item.sample
        sid = s.source_id
        end = cfg.end_ns
        if end is not None and smp.t_master_ns >= end:
            if smp.modality is Modality.EVENT and isinstance(smp.data, dict):
                cut = int(np.searchsorted(smp.data["t"], end, side="left"))
                if cut == 0:
                    self.metrics.inc(f"{sid}.after_end")
                    return 0
                smp = slice_events(smp, 0, cut)
            else:
                self.metrics.inc(f"{sid}.after_end")
                return 0
        # Leading events earlier than the window: before t0 (discard) or late.
        if smp.modality is Modality.EVENT and isinstance(smp.data, dict) and len(smp.data["t"]):
            t = smp.data["t"]
            if int(t[0]) < lo:
                cut = int(np.searchsorted(t, lo, side="left"))
                early = slice_events(smp, 0, cut)
                self._admit_early(s, early, out)
                if cut == len(t):
                    return item.align_err_ns
                smp = slice_events(smp, cut, len(t))
        elif smp.t_master_ns < lo:
            self._admit_early(s, smp, out)
            return item.align_err_ns
        out.append(smp)
        return item.align_err_ns

    def _admit_early(self, s: Stream, smp: Sample, out: list[Sample]) -> None:
        assert self.t0_ns is not None
        if smp.t_master_ns < self.t0_ns:
            self.metrics.inc(f"{s.source_id}.before_start")
            return
        self.metrics.inc(f"{s.source_id}.late")
        if self.config.late_policy is LatePolicy.ATTACH_NEXT:
            out.append(smp)

    # -- reporting --------------------------------------------------------------------------
    def snapshot(self) -> dict[str, float]:
        """Return all metrics (aligner + streams) as a flat dict."""
        snap = self.metrics.snapshot()
        for s in self.streams:
            if s.metrics is not self.metrics:
                snap.update(s.metrics.snapshot())
        return dict(sorted(snap.items()))

    def stats(self) -> dict[str, Any]:
        """Return a summary: tick count, jitter/staleness quantiles, per-source counters."""
        snap = self.snapshot()
        span_s = self.ticks_emitted * self.config.tick_ns * 1e-9
        per: dict[str, dict[str, Any]] = {}
        for s in self.streams:
            sid = s.source_id
            n = snap.get(f"{sid}.samples_out", 0)
            per[sid] = {
                "modality": s.modality.value,
                "samples": n,
                "samples_per_s": n / span_s if span_s else 0.0,
                "events": snap.get(f"{sid}.events_out", 0),
                "dropped": s.buffer.dropped
                + snap.get(f"{sid}.throttled_samples", 0)
                + snap.get(f"{sid}.throttled_events", 0),
                "late": snap.get(f"{sid}.late", 0),
                "high_water": s.buffer.high_water,
            }
        return {
            "ticks": self.ticks_emitted,
            "tick_ns": self.config.tick_ns,
            "t0_ns": self.t0_ns,
            "sync_jitter_p50_ns": snap.get("sync_jitter_ns.p50", float("nan")),
            "sync_jitter_p99_ns": snap.get("sync_jitter_ns.p99", float("nan")),
            "staleness_p50_ns": snap.get("staleness_ns.p50", float("nan")),
            "staleness_p99_ns": snap.get("staleness_ns.p99", float("nan")),
            "sources": per,
        }


def run_ticks(aligner: Aligner) -> list[AlignedTick]:
    """Collect all ticks of a (finite) aligner into a list. Convenience for tests and scripts."""
    with aligner:
        return list(aligner)
