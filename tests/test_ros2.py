# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from senseai_ingest.adapters.base import AdapterNotAvailable
from senseai_ingest.ros2.node import load_ros, sample_to_ros, tick_to_ros
from senseai_ingest.schema import pb, tensor_from_proto, value_from_proto
from senseai_ingest.stream import AlignedTick, Modality, Sample
from tests.helpers import events, imu

ROOT = Path(__file__).resolve().parents[1]


class _Msg(SimpleNamespace):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)


FAKE = SimpleNamespace(Tensor=_Msg, Sample=_Msg, SampleList=_Msg, AlignedTick=_Msg)


def _fields(msg_file: str) -> list[str]:
    lines = (ROOT / "ros2" / "senseai_ingest_msgs" / "msg" / msg_file).read_text().splitlines()
    return [ln.split()[1] for ln in lines if ln and not ln.startswith("#") and "=" not in ln]


def test_import_never_requires_ros() -> None:
    assert "rclpy" not in sys.modules
    importlib.import_module("senseai_ingest.ros2.node")
    assert "rclpy" not in sys.modules
    with pytest.raises(AdapterNotAvailable, match="rclpy"):
        load_ros()


def test_msg_files_mirror_proto() -> None:
    assert _fields("AlignedTick.msg") == ["index", "t_master_ns", "tick_ns", "sources"]
    assert _fields("Tensor.msg") == ["dtype", "shape", "buffer"]
    s = _fields("Sample.msg")
    for name in ("source_id", "modality", "t_source_ns", "t_master_ns"):
        assert name in s and name in pb.Sample.DESCRIPTOR.fields_by_name
    consts = [
        ln.split()[1]
        for ln in (ROOT / "ros2/senseai_ingest_msgs/msg/Tensor.msg").read_text().splitlines()
        if "=" in ln
    ]
    assert consts == [f"{n}={v}" for n, v in pb.DType.items()]


def test_conversion_with_fake_messages() -> None:
    frame = Sample("rgb0", Modality.FRAME, 1, 2, np.ones((2, 2, 1), np.uint8), {"t": 2**60})
    tick = AlignedTick(3, 10, 10, {"rgb0": [frame], "dvs0": [events("dvs0", [1, 2])], "imu0": []})
    msg = tick_to_ros(FAKE, tick)
    assert msg.index == 3 and [s.source_id for s in msg.sources] == ["dvs0", "imu0", "rgb0"]
    assert msg.sources[1].samples == []
    rgb = msg.sources[2].samples[0]
    assert rgb.has_tensor and rgb.modality == pb.MODALITY_FRAME
    t = pb.Tensor(dtype=rgb.tensor.dtype, shape=rgb.tensor.shape, buffer=bytes(rgb.tensor.buffer))
    assert tensor_from_proto(t).shape == (2, 2, 1)
    meta = pb.ValueMap.FromString(bytes(rgb.meta_pb))
    assert value_from_proto(meta.entries["t"]) == 2**60
    ev = msg.sources[0].samples[0]
    assert not ev.has_tensor and ev.keys == ["p", "t", "x", "y"]
    assert sample_to_ros(FAKE, imu("a", 1)).has_tensor


def test_launch_file_is_valid_python() -> None:
    src = (ROOT / "launch" / "replay.launch.py").read_text()
    compile(src, "replay.launch.py", "exec")
    assert "senseai_ingest.ros2.node" in src
