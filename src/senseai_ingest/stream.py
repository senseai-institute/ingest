# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Core data types: :class:`Modality`, :class:`Sample`, :class:`AlignedTick`, :class:`SourceSpec`.

Also :class:`Stream`, the per-adapter pipeline (clock alignment → throttle → ring buffer).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from senseai_ingest.adapters.base import Adapter
    from senseai_ingest.buffer import RingBuffer
    from senseai_ingest.clock import ClockSync
    from senseai_ingest.metrics import Registry
    from senseai_ingest.throttle import Throttle, ThrottleConfig

__all__ = [
    "EVENT_DTYPES",
    "AlignedTick",
    "ClockMode",
    "Modality",
    "Payload",
    "Sample",
    "SourceSpec",
    "Stream",
    "payload_nbytes",
]


class Modality(StrEnum):
    """Kind of sensor data carried by a :class:`Sample`. Payload shapes are documented in README."""

    EVENT = "event"  # dict t,x,y,p — sparse address-events
    FRAME = "frame"  # HxWxC uint8/uint16
    AUDIO = "audio"  # NxC int16/float32 block
    IMU = "imu"  # (6,) accel[3], gyro[3]
    CONTACT = "contact"  # force vector or bumper bits
    CURRENT = "current"  # motor current per joint
    THERMAL = "thermal"  # HxW float32
    GAS = "gas"  # dict name -> 0-d float64
    LIDAR = "lidar"  # Nx4 float32
    OTHER = "other"


class ClockMode(StrEnum):
    """How ``t_master_ns`` is derived from ``t_source_ns`` for a source."""

    FIXED = "fixed"  # offset known a priori (dataset timebase / hardware sync)
    ESTIMATED = "estimated"  # offset estimated online from host arrival times


Payload: TypeAlias = "np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]]"

#: Column dtypes of an EVENT payload.
EVENT_DTYPES: Mapping[str, np.dtype[Any]] = {
    "t": np.dtype(np.int64),
    "x": np.dtype(np.uint16),
    "y": np.dtype(np.uint16),
    "p": np.dtype(np.uint8),
}


@dataclass(frozen=True, slots=True)
class Sample:
    """One reading (or one batch, for EVENT/AUDIO) from one source.

    Timestamps follow the causal convention: they are the capture time of the *last* datum in
    the payload.
    """

    source_id: str
    modality: Modality
    t_source_ns: int
    t_master_ns: int
    data: np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]]
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def array(self) -> np.ndarray[Any, Any]:
        """The payload as an array (raises ``TypeError`` for dict payloads)."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"{self.source_id}: payload is a dict, use .columns")
        return self.data

    @property
    def columns(self) -> dict[str, np.ndarray[Any, Any]]:
        """The payload as a dict of columns, e.g. EVENT ``t,x,y,p`` (``TypeError`` otherwise)."""
        if not isinstance(self.data, dict):
            raise TypeError(f"{self.source_id}: payload is an array, use .array")
        return self.data

    @property
    def nbytes(self) -> int:
        """Payload size in bytes (used for byte-bounded buffers)."""
        return payload_nbytes(self.data)

    def with_master(self, t_master_ns: int) -> Sample:
        """Return a copy with a new master timestamp."""
        return replace(self, t_master_ns=int(t_master_ns))


def payload_nbytes(data: np.ndarray[Any, Any] | Mapping[str, np.ndarray[Any, Any]]) -> int:
    """Return the number of payload bytes in an array or dict of arrays."""
    if isinstance(data, np.ndarray):
        return int(data.nbytes)
    return int(sum(np.asarray(v).nbytes for v in data.values()))


def event_count(sample: Sample) -> int:
    """Return the number of events in an EVENT sample (1 for any other modality)."""
    if sample.modality is Modality.EVENT and isinstance(sample.data, dict):
        return len(sample.data["t"])
    return 1


def slice_events(sample: Sample, lo: int, hi: int) -> Sample:
    """Return the events ``[lo, hi)`` of an EVENT sample, re-stamped to the last kept event.

    ``t_source_ns`` shifts by the same amount as ``t_master_ns`` so the source/master delta of
    the batch is preserved.
    """
    assert isinstance(sample.data, dict)
    cols = {k: v[lo:hi] for k, v in sample.data.items()}
    if hi <= lo:
        return replace(sample, data=cols)
    t_last = int(cols["t"][-1])
    delta = t_last - sample.t_master_ns
    return replace(sample, data=cols, t_master_ns=t_last, t_source_ns=sample.t_source_ns + delta)


@dataclass(frozen=True, slots=True)
class AlignedTick:
    """Everything in the master-clock window ``[t_master_ns - tick_ns, t_master_ns)``.

    Every configured source has a key in ``samples``; a source without data maps to ``[]``.
    """

    index: int
    t_master_ns: int
    tick_ns: int
    samples: Mapping[str, list[Sample]]

    @property
    def window_start_ns(self) -> int:
        """Inclusive start of the tick window."""
        return self.t_master_ns - self.tick_ns


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Static description of a source plus its clock-alignment state.

    ``t_master = t_source + offset_ns + drift_ppm * 1e-6 * (t_source - offset_ref_ns)``.
    """

    source_id: str
    modality: Modality
    nominal_rate_hz: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)
    clock_mode: ClockMode = ClockMode.FIXED
    offset_ns: int = 0
    offset_ref_ns: int = 0
    drift_ppm: float = 0.0
    jitter_ns: int = 0
    offset_samples: int = 0


@dataclass(slots=True)
class _Item:
    """A sample travelling through a :class:`Stream`, with its alignment error."""

    sample: Sample
    align_err_ns: int = 0


class Stream:
    """Per-source pipeline: adapter → clock alignment → throttle → bounded ring buffer.

    A ``Stream`` is an async iterator of aligned :class:`Sample` objects. The :class:`Aligner`
    drives it synchronously in replay (pull, deterministic) and through a producer thread in
    live mode (push).
    """

    def __init__(
        self,
        adapter: Adapter,
        *,
        estimate_offsets: bool | None = None,
        throttle: ThrottleConfig | None = None,
        max_samples: int = 65_536,
        max_bytes: int = 512 * 1024 * 1024,
        metrics: Registry | None = None,
    ) -> None:
        from senseai_ingest.buffer import RingBuffer
        from senseai_ingest.clock import ClockSync
        from senseai_ingest.metrics import Registry
        from senseai_ingest.throttle import Throttle

        self.adapter = adapter
        spec = adapter.spec()
        self.source_id = spec.source_id
        self.modality = spec.modality
        self.metrics = metrics if metrics is not None else Registry()
        mode = spec.clock_mode
        if estimate_offsets is True:
            mode = ClockMode.ESTIMATED
        elif estimate_offsets is False:
            mode = ClockMode.FIXED
        self.clock: ClockSync = ClockSync(spec, mode=mode)
        self.throttle: Throttle | None = (
            Throttle(throttle, spec, metrics=self.metrics) if throttle is not None else None
        )
        self.buffer: RingBuffer[_Item] = RingBuffer(
            max_samples=max_samples,
            max_bytes=max_bytes,
            sizeof=lambda it: it.sample.nbytes,
            name=self.source_id,
            metrics=self.metrics,
        )
        self._it: Iterator[Sample] | None = None
        self._pending: list[_Item] = []
        self.exhausted = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None

    # -- spec -------------------------------------------------------------------------------
    def spec(self) -> SourceSpec:
        """Return the adapter spec updated with the current clock-alignment state."""
        return self.clock.update_spec(self.adapter.spec())

    # -- lifecycle --------------------------------------------------------------------------
    def open(self) -> None:
        """Open the underlying adapter."""
        self.adapter.open()
        self._it = iter(self.adapter.read())

    def close(self) -> None:
        """Stop the producer thread (if any) and close the adapter."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.adapter.close()

    # -- processing -------------------------------------------------------------------------
    def _process(self, raw: Sample) -> list[_Item]:
        sample, err = self.clock.align(raw)
        self.metrics.inc(f"{self.source_id}.samples_in")
        if self.throttle is None:
            return [_Item(sample, err)]
        return [_Item(s, err) for s in self.throttle.apply(sample)]

    def _flush(self) -> list[_Item]:
        if self.throttle is None:
            return []
        return [_Item(s, 0) for s in self.throttle.flush()]

    def pull(self) -> bool:
        """Pull one raw sample from the adapter into the buffer (replay/pull mode).

        Returns ``False`` once the adapter is exhausted and everything has been flushed.
        """
        if self.exhausted:
            return False
        assert self._it is not None, "Stream.open() not called"
        while True:
            try:
                raw = next(self._it)
            except StopIteration:
                for it in self._flush():
                    self.buffer.push(it)
                self.exhausted = True
                return False
            items = self._process(raw)
            for it in items:
                self.buffer.push(it)
            if items:
                return True

    def push(self, raw: Sample) -> None:
        """Feed one raw sample (live/push mode; called by the producer thread or by tests)."""
        for it in self._process(raw):
            self.buffer.push(it)

    def start_producer(self) -> None:
        """Run ``adapter.read()`` in a daemon thread that pushes into the buffer (live mode)."""

        def run() -> None:
            assert self._it is not None
            try:
                for raw in self._it:
                    if self._stop.is_set():
                        break
                    self.push(raw)
                for it in self._flush():
                    self.buffer.push(it)
            except BaseException as exc:  # surfaced to the aligner
                self._error = exc
            finally:
                self.exhausted = True

        self._thread = threading.Thread(target=run, name=f"ingest-{self.source_id}", daemon=True)
        self._thread.start()

    def raise_if_failed(self) -> None:
        """Re-raise an exception that happened in the producer thread."""
        if self._error is not None:
            raise self._error

    @property
    def backpressure(self) -> bool:
        """``True`` when the buffer is ≥ 80 % full; live adapters may slow down or decimate."""
        return self.buffer.fill_ratio >= 0.8

    # -- iteration --------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Sample]:
        """Iterate aligned samples synchronously (pull mode)."""
        if self._it is None:
            self.open()
        while True:
            item = self.buffer.pop()
            if item is not None:
                yield item.sample
                continue
            if not self.pull() and len(self.buffer) == 0:
                return

    async def __aiter__(self) -> AsyncIterator[Sample]:
        """Iterate aligned samples asynchronously, yielding control between samples."""
        for i, s in enumerate(self):
            yield s
            if i % 64 == 63:
                await asyncio.sleep(0)
