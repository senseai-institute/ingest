# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import asyncio

import numpy as np
import pytest

from senseai_ingest.schema import (
    from_proto,
    hash_ticks,
    pb,
    tensor_from_proto,
    tensor_to_proto,
    to_proto,
    value_from_proto,
    value_to_proto,
)
from senseai_ingest.stream import (
    AlignedTick,
    ClockMode,
    Modality,
    Sample,
    SourceSpec,
    Stream,
    payload_nbytes,
    slice_events,
)
from tests.helpers import ListAdapter, events, imu


def test_modality_values_and_proto_enum_names_match() -> None:
    for m in Modality:
        assert pb.Modality.Value(f"MODALITY_{m.name}") > 0
        assert m.value == m.name.lower()


def test_sample_is_frozen_and_nbytes() -> None:
    s = imu("a", 5)
    with pytest.raises(AttributeError):
        s.t_master_ns = 3  # type: ignore[misc]
    assert s.nbytes == 48
    assert s.with_master(9).t_master_ns == 9
    assert payload_nbytes({"a": np.zeros(3, np.uint8), "b": np.zeros(2, np.int64)}) == 19


def test_slice_events_restamps_to_last_event() -> None:
    s = events("e", [10, 20, 30, 40])
    part = slice_events(s, 0, 2)
    assert part.t_master_ns == 20 and part.t_source_ns == 20
    assert list(part.columns["t"]) == [10, 20]
    empty = slice_events(s, 2, 2)
    assert len(empty.columns["t"]) == 0


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        np.array(3.5, dtype=np.float64),
        np.arange(6, dtype=">i4"),
        np.array([True, False]),
        np.zeros((0, 4), dtype=np.float32),
    ],
)
def test_tensor_roundtrip(arr: np.ndarray) -> None:
    out = tensor_from_proto(tensor_to_proto(arr))
    assert out.shape == arr.shape
    np.testing.assert_array_equal(out, arr)
    assert out.dtype.byteorder in "<=|"


def test_tensor_rejects_unsupported_dtype() -> None:
    with pytest.raises(TypeError):
        tensor_to_proto(np.zeros(2, dtype=np.complex64))


def test_value_roundtrip_preserves_int64() -> None:
    big = 1_700_000_000_123_456_789  # > 2**53
    meta = {
        "t": big,
        "f": 1.5,
        "s": "x",
        "b": True,
        "raw": b"\x00",
        "k": np.eye(3),
        "lst": [1, "a"],
        "d": {"n": 2},
    }
    back = {k: value_from_proto(value_to_proto(v)) for k, v in meta.items()}
    assert back["t"] == big
    np.testing.assert_array_equal(back["k"], np.eye(3))
    assert back["lst"] == [1, "a"] and back["d"] == {"n": 2} and back["raw"] == b"\x00"
    assert value_from_proto(pb.Value()) is None
    with pytest.raises(TypeError):
        value_to_proto(object())


def test_sample_tick_spec_roundtrip() -> None:
    e = events("dvs0", [1, 2, 3])
    f = Sample("rgb0", Modality.FRAME, 7, 8, np.ones((2, 3, 1), np.uint8), {"encoding": "mono8"})
    tick = AlignedTick(0, 10, 10, {"dvs0": [e], "rgb0": [f], "mic0": []})
    back = from_proto(to_proto(tick))
    assert back.index == 0 and back.tick_ns == 10
    assert set(back.samples) == {"dvs0", "rgb0", "mic0"}
    assert back.samples["mic0"] == []
    np.testing.assert_array_equal(back.samples["dvs0"][0].columns["t"], [1, 2, 3])
    assert back.samples["rgb0"][0].meta == {"encoding": "mono8"}
    spec = SourceSpec(
        "a", Modality.IMU, 1000.0, {"frame_id": "imu"}, ClockMode.ESTIMATED, 5, 6, 1.5, 7, 200
    )
    assert from_proto(to_proto(spec)) == spec
    with pytest.raises(TypeError):
        to_proto(object())  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        from_proto(pb.Tensor())  # type: ignore[call-overload]


def test_hash_ticks_is_stable_and_sensitive() -> None:
    t1 = AlignedTick(0, 10, 10, {"a": [imu("a", 5)], "b": []})
    t2 = AlignedTick(0, 10, 10, {"b": [], "a": [imu("a", 5)]})
    assert hash_ticks([t1]) == hash_ticks([t2])
    assert hash_ticks([t1]) != hash_ticks([AlignedTick(0, 10, 10, {"a": [imu("a", 6)], "b": []})])


def test_stream_sync_and_async_iteration() -> None:
    samples = [imu("a", t) for t in range(0, 200, 2)]
    st = Stream(ListAdapter("a", samples))
    assert [s.t_master_ns for s in st] == list(range(0, 200, 2))

    async def collect() -> list[int]:
        st2 = Stream(ListAdapter("a", samples))
        st2.open()
        return [s.t_master_ns async for s in st2]

    assert asyncio.run(collect()) == list(range(0, 200, 2))


def test_stream_estimated_mode_shifts_event_t() -> None:
    # device clock is 1000 ns behind host arrival; t column is in device time
    raw = [events("e", [t - 1000 - 5, t - 1000]) for t in range(10_000, 20_000, 1000)]
    raw = [r.with_master(r.t_master_ns + 1000) for r in raw]
    ad = ListAdapter("e", raw, modality=Modality.EVENT, clock_mode=ClockMode.ESTIMATED)
    st = Stream(ad)
    out = list(st)
    assert all(s.t_master_ns - s.t_source_ns == 1000 for s in out)
    assert all(int(s.columns["t"][-1]) == s.t_master_ns for s in out)
    spec = st.spec()
    assert spec.clock_mode is ClockMode.ESTIMATED and spec.offset_ns == 1000
    assert Stream(ad, estimate_offsets=False).clock.mode is ClockMode.FIXED


def test_stream_backpressure_flag() -> None:
    st = Stream(ListAdapter("a", [imu("a", 1)]), max_samples=10)
    for t in range(9):
        st.push(imu("a", t))
    assert st.backpressure
