# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
r"""IMU + bumper + motor current multiplexed on one serial line (pyserial). Read is a Gate-1 stub.

Line protocol (ASCII, one record per ``\n``; ``t_us`` is the device clock in µs)::

    IMU,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>    m/s^2, rad/s        -> Modality.IMU      (6,) f64
    BUMP,<t_us>,<mask>                           uint32 bitmask      -> Modality.CONTACT  () u32
    CUR,<t_us>,<j0>,<j1>,...,<jN>                A per joint         -> Modality.CURRENT  (J,) f32

``mask`` may be decimal or ``0x``-prefixed hex. Blank lines and lines starting with ``#`` are
ignored; anything else malformed is counted in ``<source_id>.parse_errors`` and skipped — the
parser never raises on bad input. Samples are stamped with the host arrival time as
``t_master_ns`` (``ClockMode.ESTIMATED``); the :class:`Stream` estimates the device offset.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from senseai_ingest.adapters.live import LiveAdapter
from senseai_ingest.metrics import Registry
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["LineParser", "SerialChannel", "SerialConfig", "SerialSensorHub"]


@dataclass(frozen=True, slots=True)
class SerialConfig:
    """Serial hub configuration.

    Attributes:
        port: serial device, e.g. ``/dev/ttyACM0``.
        baud: baud rate.
        imu_id / bump_id / current_id: source ids for the three record types.
        imu_rate_hz / bump_rate_hz / current_rate_hz: nominal rates (for ``spec()``).
        joints: joint names for ``CUR`` records (also fixes the expected count).
    """

    port: str = "/dev/ttyACM0"
    baud: int = 921_600
    imu_id: str = "imu0"
    bump_id: str = "bump0"
    current_id: str = "cur0"
    imu_rate_hz: float = 1000.0
    bump_rate_hz: float = 100.0
    current_rate_hz: float = 500.0
    joints: Sequence[str] = field(default_factory=tuple)


class LineParser:
    """Stateless-per-line parser for the serial protocol, with error counters."""

    def __init__(self, config: SerialConfig | None = None, metrics: Registry | None = None) -> None:
        self.config = config or SerialConfig()
        self.metrics = metrics if metrics is not None else Registry()
        c = self.config
        self._ids = {"IMU": c.imu_id, "BUMP": c.bump_id, "CUR": c.current_id}

    def _error(self, tag: str) -> None:
        sid = self._ids.get(tag, "serial")
        self.metrics.inc(f"{sid}.parse_errors")

    def parse(self, line: str | bytes, t_arrival_ns: int) -> Sample | None:
        """Parse one line into a :class:`Sample`, or ``None`` (ignored or malformed)."""
        if isinstance(line, bytes):
            try:
                line = line.decode("ascii")
            except UnicodeDecodeError:
                self._error("")
                return None
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        parts = line.split(",")
        tag = parts[0].strip().upper()
        if tag not in self._ids:
            self._error("")
            return None
        try:
            t_us = int(parts[1])
            if t_us < 0:
                raise ValueError("negative timestamp")
            t_src = t_us * 1000
            vals = parts[2:]
            sid = self._ids[tag]
            if tag == "IMU":
                if len(vals) != 6:
                    raise ValueError("IMU needs 6 values")
                data: np.ndarray[Any, Any] = np.array([float(v) for v in vals], dtype=np.float64)
                if not np.all(np.isfinite(data)):
                    raise ValueError("non-finite IMU value")
                return Sample(sid, Modality.IMU, t_src, t_arrival_ns, data, {})
            if tag == "BUMP":
                if len(vals) != 1:
                    raise ValueError("BUMP needs 1 value")
                mask = int(vals[0].strip(), 0)
                if not 0 <= mask < 2**32:
                    raise ValueError("mask out of range")
                data = np.array(mask, dtype=np.uint32)
                return Sample(sid, Modality.CONTACT, t_src, t_arrival_ns, data, {"kind": "bumper"})
            joints = self.config.joints
            if not vals or (joints and len(vals) != len(joints)):
                raise ValueError("CUR joint count mismatch")
            cur = [float(v) for v in vals]
            if not all(math.isfinite(x) for x in cur):
                raise ValueError("non-finite current")
            data = np.array(cur, dtype=np.float32)
            return Sample(sid, Modality.CURRENT, t_src, t_arrival_ns, data, {})
        except (ValueError, IndexError):
            self._error(tag)
            return None

    def parse_lines(
        self, lines: Iterable[str | bytes], t_arrival_ns: Iterable[int] | None = None
    ) -> Iterator[Sample]:
        """Parse many lines; ``t_arrival_ns`` defaults to the device time (offline use)."""
        arrivals = iter(t_arrival_ns) if t_arrival_ns is not None else None
        for line in lines:
            arr = next(arrivals) if arrivals is not None else None
            s = self.parse(line, arr if arr is not None else -1)
            if s is None:
                continue
            if arr is None:
                s = s.with_master(s.t_source_ns)
            yield s


class SerialChannel(LiveAdapter):
    """One record type (IMU, BUMP or CUR) of a :class:`SerialSensorHub` as an adapter."""

    backend = ("serial", "live")

    def __init__(self, hub: SerialSensorHub, tag: str) -> None:
        self.hub = hub
        self.tag = tag

    def spec(self) -> SourceSpec:
        """Describe this channel."""
        c = self.hub.config
        if self.tag == "IMU":
            return SourceSpec(
                c.imu_id,
                Modality.IMU,
                c.imu_rate_hz,
                {"frame_id": c.imu_id, "port": c.port},
                ClockMode.ESTIMATED,
            )
        if self.tag == "BUMP":
            return SourceSpec(
                c.bump_id,
                Modality.CONTACT,
                c.bump_rate_hz,
                {"kind": "bumper", "port": c.port},
                ClockMode.ESTIMATED,
            )
        return SourceSpec(
            c.current_id,
            Modality.CURRENT,
            c.current_rate_hz,
            {"joints": list(c.joints), "port": c.port},
            ClockMode.ESTIMATED,
        )


class SerialSensorHub:
    """Owns the serial port and demultiplexes records into one adapter per record type."""

    TAGS = ("IMU", "BUMP", "CUR")

    def __init__(self, config: SerialConfig | None = None) -> None:
        self.config = config or SerialConfig()
        self.parser = LineParser(self.config)

    def adapters(self, tags: Sequence[str] = TAGS) -> list[SerialChannel]:
        """Return the channel adapters (default: all three)."""
        bad = set(tags) - set(self.TAGS)
        if bad:
            raise ValueError(f"unknown record types {sorted(bad)}")
        return [SerialChannel(self, t) for t in tags]
