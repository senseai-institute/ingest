# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag2 import Writer

from senseai_ingest.adapters.replay.rosbag import (
    RosbagReplay,
    _image,
    topic_to_source_id,
    typestore,
)
from senseai_ingest.cli import main
from senseai_ingest.stream import Modality
from senseai_ingest.sync import Aligner, AlignerConfig, run_ticks

BASE = 1_500_000_000 * 10**9


def _msgs(ts: Any) -> dict[str, Any]:
    T = ts.types  # noqa: N806
    return {
        k.rsplit("/", 1)[-1]: v
        for k, v in T.items()
        if k.split("/")[0]
        in ("std_msgs", "builtin_interfaces", "geometry_msgs", "sensor_msgs", "dvs_msgs")
    }


def _header(m: dict[str, Any], t_ns: int, frame: str = "f") -> Any:
    return m["Header"](stamp=m["Time"](sec=t_ns // 10**9, nanosec=t_ns % 10**9), frame_id=frame)


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ts = typestore()
    m = _msgs(ts)
    path = tmp_path_factory.mktemp("bag") / "rec"
    cov = np.zeros(9)
    with Writer(path, version=9) as w:
        c_imu = w.add_connection("/imu", "sensor_msgs/msg/Imu", typestore=ts)
        c_img = w.add_connection("/cam/image_raw", "sensor_msgs/msg/Image", typestore=ts)
        c_ev = w.add_connection("/dvs/events", "dvs_msgs/msg/EventArray", typestore=ts)
        c_x = w.add_connection("/chatter", "std_msgs/msg/String", typestore=ts)
        for i in range(50):
            t = BASE + i * 2_000_000
            v3 = m["Vector3"]
            imu = m["Imu"](
                header=_header(m, t),
                orientation=m["Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=cov,
                angular_velocity=v3(x=0.1, y=0.2, z=0.3),
                angular_velocity_covariance=cov,
                linear_acceleration=v3(x=0.0, y=0.0, z=9.8),
                linear_acceleration_covariance=cov,
            )
            w.write(c_imu, t, ts.serialize_cdr(imu, "sensor_msgs/msg/Imu"))
            if i % 10 == 0:
                px = np.arange(4 * 5 * 3, dtype=np.uint8)
                img = m["Image"](
                    header=_header(m, t),
                    height=4,
                    width=5,
                    encoding="bgr8",
                    is_bigendian=0,
                    step=15,
                    data=px,
                )
                w.write(c_img, t, ts.serialize_cdr(img, "sensor_msgs/msg/Image"))
            evs = (
                []
                if i == 3
                else [
                    m["Event"](
                        x=j,
                        y=2 * j,
                        ts=m["Time"](sec=(t + j) // 10**9, nanosec=(t + j) % 10**9),
                        polarity=bool(j % 2),
                    )
                    for j in range(4)
                ]
            )
            ea = m["EventArray"](header=_header(m, t), height=260, width=346, events=evs)
            w.write(c_ev, t, ts.serialize_cdr(ea, "dvs_msgs/msg/EventArray"))
        w.write(c_x, BASE, ts.serialize_cdr(m["String"](data="hi"), "std_msgs/msg/String"))
    return path


def test_from_bag_discovers_supported_topics(bag: Path) -> None:
    ads = {a.source_id: a for a in RosbagReplay.from_bag(bag)}
    assert set(ads) == {"imu", "cam_image_raw", "dvs_events"}
    assert ads["imu"].spec().modality is Modality.IMU
    assert ads["imu"].spec().nominal_rate_hz == pytest.approx(500, rel=0.05)
    with ads["imu"] as a:
        s = next(iter(a.read()))
    assert s.array.tolist() == [0.0, 0.0, 9.8, 0.1, 0.2, 0.3] and s.t_master_ns == BASE
    with ads["cam_image_raw"] as a:
        f = next(iter(a.read()))
    assert f.array.shape == (4, 5, 3) and f.array[0, 0].tolist() == [2, 1, 0]  # bgr -> rgb
    with ads["dvs_events"] as a:
        evs = list(a.read())
    assert len(evs) == 49  # the empty EventArray is skipped
    assert evs[0].columns["t"].tolist() == [BASE, BASE + 1, BASE + 2, BASE + 3]
    assert evs[0].columns["p"].tolist() == [0, 1, 0, 1]


def test_topic_mapping_selection_window_and_aligner(bag: Path) -> None:
    ads = RosbagReplay.from_bag(bag, topics={"/imu": "imu0", "/dvs/events": "dvs0"})
    assert sorted(a.source_id for a in ads) == ["dvs0", "imu0"]
    assert [a.source_id for a in RosbagReplay.from_bag(bag, sources=["imu"])] == ["imu"]
    (imu,) = RosbagReplay.from_bag(
        bag, sources=["imu"], start_ns=BASE + 10_000_000, end_ns=BASE + 20_000_000
    )
    assert len(list(imu.read())) == 5
    ticks = run_ticks(Aligner(ads, AlignerConfig(tick_ns=10_000_000)))
    assert sum(len(t.samples["imu0"]) for t in ticks) == 50


def test_errors_and_helpers(bag: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="no topic"):
        RosbagReplay(bag, "/missing").spec()
    with pytest.raises(ValueError, match="unsupported message"):
        RosbagReplay(bag, "/chatter").spec()
    assert topic_to_source_id("/") == "root"
    assert RosbagReplay(bag, "/imu").time_range()[0] == BASE

    class Img:
        height, width, step, is_bigendian = 2, 2, 4, 0
        data = np.arange(8, dtype=np.uint8)
        encoding = "mono16"

    arr, enc = _image(Img())
    assert enc == "mono16" and arr.shape == (2, 2, 1) and arr[0, 0, 0] == 256
    Img.encoding, Img.step = "mono8", 2
    Img.data = np.arange(4, dtype=np.uint8)
    assert _image(Img())[0].shape == (2, 2, 1)
    Img.encoding = "yuv422"
    with pytest.raises(ValueError, match="encoding"):
        _image(Img())
    assert main(["replay", "--dataset", "rosbag", "--path", str(bag), "--json"]) == 0
    assert '"imu"' in capsys.readouterr().out
