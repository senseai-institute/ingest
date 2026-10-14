# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""CAN bus sensors (python-can). Frame decoding is implemented; bus reading is a Gate-1 stub.

Each :class:`CanSignal` maps one arbitration id to a source: the payload is unpacked with a
``struct`` format, multiplied by ``scale``, and emitted as a 1-D float array, e.g. motor currents
as int16 mA (``fmt="<4h", scale=1e-3``). Timestamps: python-can's host receive time is the arrival
stamp (``ClockMode.ESTIMATED``); if ``t_field`` is set, that unpacked field (µs) is the device time.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from senseai_ingest.adapters.live import LiveAdapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["CanAdapter", "CanConfig", "CanSignal"]


@dataclass(frozen=True, slots=True)
class CanSignal:
    """Decoding rule for one CAN arbitration id."""

    arbitration_id: int
    source_id: str
    modality: Modality = Modality.CURRENT
    fmt: str = "<4h"
    scale: float = 1.0
    t_field: int | None = None  # index of a µs device timestamp in the unpacked tuple
    rate_hz: float = 0.0

    def __post_init__(self) -> None:
        size = struct.calcsize(self.fmt)
        if size > 64:
            raise ValueError(f"{self.source_id}: fmt {self.fmt!r} is {size} bytes (> CAN FD 64)")
        n = len(struct.unpack(self.fmt, bytes(size)))
        if self.t_field is not None and not 0 <= self.t_field < n:
            raise ValueError(f"{self.source_id}: t_field {self.t_field} out of range")

    def decode(self, payload: bytes, t_arrival_ns: int) -> Sample:
        """Decode one frame payload (raises ``struct.error`` on a short frame)."""
        vals = list(struct.unpack_from(self.fmt, payload))
        t_src = t_arrival_ns
        if self.t_field is not None:
            t_src = int(vals.pop(self.t_field)) * 1000
        data = np.asarray(vals, dtype=np.float64) * self.scale
        out = data.astype(np.float32) if self.modality is Modality.CURRENT else data
        return Sample(self.source_id, self.modality, t_src, t_arrival_ns, out, {})


@dataclass(frozen=True, slots=True)
class CanConfig:
    """Bus configuration."""

    channel: str = "can0"
    interface: str = "socketcan"
    bitrate: int = 1_000_000
    signals: Sequence[CanSignal] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ids = [s.arbitration_id for s in self.signals]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate arbitration ids")


class CanAdapter(LiveAdapter):
    """One :class:`CanSignal` of a bus as a live adapter."""

    backend = ("can", "live")

    def __init__(self, config: CanConfig, source_id: str) -> None:
        self.config = config
        matches = [s for s in config.signals if s.source_id == source_id]
        if not matches:
            raise ValueError(f"no CAN signal with source_id {source_id!r}")
        self.signal = matches[0]

    def spec(self) -> SourceSpec:
        """Describe the signal."""
        s = self.signal
        meta: dict[str, Any] = {
            "channel": self.config.channel,
            "arbitration_id": s.arbitration_id,
            "fmt": s.fmt,
            "scale": s.scale,
        }
        return SourceSpec(s.source_id, s.modality, s.rate_hz, meta, ClockMode.ESTIMATED)

    @staticmethod
    def for_config(config: CanConfig) -> list[CanAdapter]:
        """One adapter per configured signal."""
        return [CanAdapter(config, s.source_id) for s in config.signals]
