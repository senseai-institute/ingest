# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import threading

import pytest

from senseai_ingest.buffer import RingBuffer
from senseai_ingest.metrics import Registry, Summary


def test_drop_oldest_by_count() -> None:
    m = Registry()
    b: RingBuffer[int] = RingBuffer(max_samples=3, name="s", metrics=m)
    for i in range(5):
        b.push(i)
    assert b.drain() == [2, 3, 4]
    assert b.dropped == 2 and b.high_water == 3 and b.pushed == 5
    assert m.get("s.dropped") == 2 and m.get("s.high_water") == 3


def test_drop_oldest_by_bytes_and_oversized_item() -> None:
    b: RingBuffer[bytes] = RingBuffer(max_samples=100, max_bytes=10, sizeof=len)
    assert b.push(b"aaaa") == 0
    b.push(b"bbbb")
    assert b.push(b"cccc") == 1
    assert b.nbytes == 8
    assert b.push(b"x" * 50) == 2  # oversized item evicts everything, is still kept
    assert len(b) == 1 and b.nbytes == 50
    assert b.fill_ratio == pytest.approx(5.0)


def test_peek_pop_replace_head() -> None:
    b: RingBuffer[bytes] = RingBuffer(max_samples=4, sizeof=len)
    assert b.pop() is None and b.peek() is None
    b.replace_head(b"zz")
    assert b.nbytes == 2
    b.push(b"abc")
    b.replace_head(b"z")
    assert b.peek() == b"z" and b.nbytes == 4
    assert b.pop() == b"z" and b.nbytes == 3


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        RingBuffer(max_samples=0)


def test_spsc_threads_lose_nothing_within_capacity() -> None:
    b: RingBuffer[int] = RingBuffer(max_samples=1_000_000)
    n = 50_000

    def prod() -> None:
        for i in range(n):
            b.push(i)

    th = threading.Thread(target=prod)
    th.start()
    got: list[int] = []
    while th.is_alive() or len(b):
        x = b.pop()
        if x is not None:
            got.append(x)
    th.join()
    assert got == list(range(n))


def test_registry_snapshot_and_summary() -> None:
    m = Registry()
    m.inc("a")
    m.inc("a", 2)
    m.set("g", 1.5)
    m.set_max("h", 3)
    m.set_max("h", 1)
    for v in range(101):
        m.summary("lat").observe(v)
    snap = m.snapshot()
    assert snap["a"] == 3 and snap["g"] == 1.5 and snap["h"] == 3
    assert snap["lat.count"] == 101 and snap["lat.p50"] == 50 and snap["lat.max"] == 100
    assert snap["lat.p99"] == pytest.approx(99.0)
    assert Summary().quantile(0.5) != Summary().quantile(0.5)  # nan
