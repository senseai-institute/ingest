# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Bounded per-source ring buffer with a drop-oldest policy.

Single-producer / single-consumer. ``deque.append``/``popleft`` are atomic under the GIL; a
short lock guards only the byte accounting, so contention is negligible ("lock-free-ish").
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar

from senseai_ingest.metrics import Registry

__all__ = ["RingBuffer"]

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """FIFO bounded by item count and total bytes; when full, the oldest items are dropped.

    An item larger than ``max_bytes`` on its own is still accepted (everything older is dropped)
    so a single oversized frame never wedges a source.

    Counters (also mirrored into ``metrics`` as ``<name>.dropped`` / ``<name>.high_water``):
    ``pushed``, ``dropped``, ``high_water``.
    """

    def __init__(
        self,
        max_samples: int = 65_536,
        max_bytes: int | None = None,
        *,
        sizeof: Callable[[T], int] | None = None,
        name: str = "buffer",
        metrics: Registry | None = None,
    ) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self.max_samples = max_samples
        self.max_bytes = max_bytes
        self._sizeof = sizeof
        self.name = name
        self._metrics = metrics
        self._q: deque[tuple[T, int]] = deque()
        self._lock = threading.Lock()
        self.nbytes = 0
        self.pushed = 0
        self.dropped = 0
        self.high_water = 0

    def __len__(self) -> int:
        return len(self._q)

    @property
    def fill_ratio(self) -> float:
        """Fraction of capacity in use (max of the sample and byte bounds)."""
        r = len(self._q) / self.max_samples
        if self.max_bytes:
            r = max(r, self.nbytes / self.max_bytes)
        return r

    def push(self, item: T) -> int:
        """Append ``item``; return how many old items were dropped to make room."""
        size = self._sizeof(item) if self._sizeof is not None else 0
        dropped = 0
        with self._lock:
            while self._q and (
                len(self._q) >= self.max_samples
                or (self.max_bytes is not None and self.nbytes + size > self.max_bytes)
            ):
                _, s = self._q.popleft()
                self.nbytes -= s
                dropped += 1
            self._q.append((item, size))
            self.nbytes += size
            self.pushed += 1
            self.dropped += dropped
            n = len(self._q)
            if n > self.high_water:
                self.high_water = n
        if self._metrics is not None:
            if dropped:
                self._metrics.inc(f"{self.name}.dropped", dropped)
            self._metrics.set_max(f"{self.name}.high_water", n)
        return dropped

    def peek(self) -> T | None:
        """Return the oldest item without removing it."""
        with self._lock:
            return self._q[0][0] if self._q else None

    def pop(self) -> T | None:
        """Remove and return the oldest item, or ``None`` if empty."""
        with self._lock:
            if not self._q:
                return None
            item, s = self._q.popleft()
            self.nbytes -= s
            return item

    def replace_head(self, item: T) -> None:
        """Replace the oldest item in place (used to keep the remainder of a split batch)."""
        size = self._sizeof(item) if self._sizeof is not None else 0
        with self._lock:
            if not self._q:
                self._q.append((item, size))
                self.nbytes += size
                return
            _, old = self._q[0]
            self._q[0] = (item, size)
            self.nbytes += size - old

    def drain(self) -> list[T]:
        """Remove and return all items."""
        with self._lock:
            items = [it for it, _ in self._q]
            self._q.clear()
            self.nbytes = 0
            return items
