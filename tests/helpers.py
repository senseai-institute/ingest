# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Test helpers: an in-memory adapter."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from senseai_ingest.adapters.base import ReplayAdapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec


class ListAdapter(ReplayAdapter):
    """Replays a fixed list of samples."""

    def __init__(
        self,
        source_id: str,
        samples: Sequence[Sample],
        *,
        modality: Modality = Modality.IMU,
        clock_mode: ClockMode = ClockMode.FIXED,
        rate_hz: float = 0.0,
        live: bool = False,
        **kw: object,
    ) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._sid = source_id
        self._samples = list(samples)
        self._mod = modality
        self._mode = clock_mode
        self._rate = rate_hz
        self.is_live = live
        self.opened = False
        self.closed = False

    def spec(self) -> SourceSpec:
        return SourceSpec(self._sid, self._mod, self._rate, {}, self._mode)

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def time_range(self) -> tuple[int, int]:
        return self._samples[0].t_master_ns, self._samples[-1].t_master_ns

    def _read_once(self) -> Iterator[Sample]:
        for s in self._samples:
            if self.start_ns is not None and s.t_master_ns < self.start_ns:
                continue
            if self.end_ns is not None and s.t_master_ns >= self.end_ns:
                break
            yield s


def imu(sid: str, t: int, t_src: int | None = None) -> Sample:
    return Sample(sid, Modality.IMU, t if t_src is None else t_src, t, np.zeros(6), {})


def events(sid: str, ts: Sequence[int]) -> Sample:
    t = np.asarray(ts, dtype=np.int64)
    n = len(t)
    data: dict[str, np.ndarray[Any, Any]] = {
        "t": t,
        "x": np.arange(n, dtype=np.uint16),
        "y": np.zeros(n, dtype=np.uint16),
        "p": (np.arange(n) % 2).astype(np.uint8),
    }
    return Sample(sid, Modality.EVENT, int(t[-1]), int(t[-1]), data, {})
