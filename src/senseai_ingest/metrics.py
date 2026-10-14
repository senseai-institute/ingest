# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Tiny in-process metrics registry: counters, gauges and quantile summaries, exported as a dict.

Deliberately dependency-free; a Prometheus/OpenTelemetry exporter can wrap :meth:`snapshot`.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

__all__ = ["Registry", "Summary"]


class Summary:
    """Bounded reservoir of the most recent ``capacity`` observations with quantiles."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        self._values: deque[float] = deque(maxlen=capacity)
        self.count = 0
        self.max = float("-inf")

    def observe(self, value: float) -> None:
        """Record one observation."""
        self._values.append(float(value))
        self.count += 1
        if value > self.max:
            self.max = float(value)

    def quantile(self, q: float) -> float:
        """Return the ``q``-quantile (0..1) of retained observations, or ``nan`` if empty."""
        if not self._values:
            return float("nan")
        return float(np.quantile(np.fromiter(self._values, dtype=np.float64), q))


class Registry:
    """Thread-safe named counters, gauges and summaries."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}
        self._summaries: dict[str, Summary] = {}

    def inc(self, name: str, n: float = 1) -> None:
        """Increment counter ``name`` by ``n``."""
        with self._lock:
            self._values[name] = self._values.get(name, 0) + n

    def set(self, name: str, value: float) -> None:
        """Set gauge ``name``."""
        with self._lock:
            self._values[name] = value

    def set_max(self, name: str, value: float) -> None:
        """Set gauge ``name`` to ``max(current, value)``."""
        with self._lock:
            if value > self._values.get(name, float("-inf")):
                self._values[name] = value

    def get(self, name: str, default: float = 0) -> float:
        """Return the current value of a counter/gauge."""
        with self._lock:
            return self._values.get(name, default)

    def summary(self, name: str) -> Summary:
        """Return (creating if needed) the summary ``name``."""
        with self._lock:
            s = self._summaries.get(name)
            if s is None:
                s = self._summaries[name] = Summary()
            return s

    def snapshot(self) -> dict[str, float]:
        """Return all metrics as a flat dict; summaries expand to ``.count/.p50/.p99/.max``."""
        with self._lock:
            out: dict[str, float] = {
                k: (int(v) if float(v).is_integer() else v) for k, v in self._values.items()
            }
            summaries = list(self._summaries.items())
        for name, s in summaries:
            out[f"{name}.count"] = s.count
            out[f"{name}.p50"] = s.quantile(0.5)
            out[f"{name}.p99"] = s.quantile(0.99)
            out[f"{name}.max"] = s.max if s.count else float("nan")
        return dict(sorted(out.items()))
