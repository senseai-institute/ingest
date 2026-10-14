# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Adapter ABC: ``open()``, ``read() -> Iterator[Sample]``, ``close()``, ``spec()``.

One adapter produces exactly one source. Multi-source recordings expose a factory returning a
list of adapters (e.g. :meth:`DsecReplay.from_sequence`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import replace
from types import TracebackType
from typing import Self

import numpy as np

from senseai_ingest.stream import Modality, Sample, SourceSpec

__all__ = ["Adapter", "AdapterNotAvailable", "ReplayAdapter"]


class AdapterNotAvailable(NotImplementedError):
    """Raised when an adapter's backend is missing or not implemented yet (message has a hint)."""


class Adapter(ABC):
    """Base class for all adapters.

    Timestamps: replay adapters set ``t_master_ns`` from the dataset timebase (``FIXED``); live
    adapters set it to the host arrival time and the :class:`~senseai_ingest.stream.Stream`
    replaces it from the offset estimate (``ESTIMATED``). For estimated EVENT sources the ``t``
    column is in source time; for fixed EVENT sources it is already master time.
    """

    #: ``True`` for adapters reading physical devices.
    is_live: bool = False

    @abstractmethod
    def spec(self) -> SourceSpec:
        """Describe the source (id, modality, rate, meta, clock mode)."""

    @abstractmethod
    def open(self) -> None:
        """Acquire resources (files, devices)."""

    @abstractmethod
    def read(self) -> Iterator[Sample]:
        """Yield samples in non-decreasing ``t_source_ns`` order."""

    @abstractmethod
    def close(self) -> None:
        """Release resources. Must be idempotent."""

    @property
    def source_id(self) -> str:
        """Shortcut for ``spec().source_id``."""
        return self.spec().source_id

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


class ReplayAdapter(Adapter):
    """Base for recordings. Subclasses implement :meth:`_read_once` and :meth:`time_range`.

    Common options (all in master-clock ns):

    * ``start_ns`` / ``end_ns`` — keep samples with ``start_ns <= t_master < end_ns``.
    * ``speed`` — 1.0 real time, 0 as fast as possible. Pacing is done by the
      :class:`~senseai_ingest.sync.Aligner` (which owns the replay cursor), not per adapter,
      so all sources stay in lock-step.
    * ``loop`` — restart at the end; each pass is shifted by the recording span + 1 ns so
      timestamps keep increasing.
    """

    def __init__(
        self,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
        speed: float = 0.0,
        loop: bool = False,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.speed = speed
        self.loop = loop

    @abstractmethod
    def time_range(self) -> tuple[int, int]:
        """Return ``(first, last)`` master timestamps in the recording (cheap; no full scan)."""

    @abstractmethod
    def _read_once(self) -> Iterator[Sample]:
        """Yield one pass over ``[start_ns, end_ns)``."""

    def read(self) -> Iterator[Sample]:
        """Yield samples, looping if configured."""
        if not self.loop:
            yield from self._read_once()
            return
        first, last = self.time_range()
        lo = max(first, self.start_ns) if self.start_ns is not None else first
        hi = min(last, self.end_ns - 1) if self.end_ns is not None else last
        period = max(1, hi - lo + 1)
        k = 0
        while True:
            n = 0
            for s in self._read_once():
                n += 1
                yield shift_sample(s, k * period) if k else s
            if n == 0:
                return
            k += 1


def shift_sample(sample: Sample, dt_ns: int) -> Sample:
    """Return ``sample`` shifted by ``dt_ns`` on both clocks (EVENT ``t`` column too)."""
    data = sample.data
    if sample.modality is Modality.EVENT and isinstance(data, dict):
        data = {**data, "t": data["t"] + np.int64(dt_ns)}
    return replace(
        sample,
        t_source_ns=sample.t_source_ns + dt_ns,
        t_master_ns=sample.t_master_ns + dt_ns,
        data=data,
    )
