# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""rosbag (ROS 1) / rosbag2 replay via the pure-Python ``rosbags`` package (``[ros2]`` extra).

No ROS installation is needed. Supported message types:

==============================  ==========  =====================================================
``sensor_msgs/msg/Imu``         IMU         ``[linear_acceleration, angular_velocity]``
``sensor_msgs/msg/Image``       FRAME       ``mono8``, ``mono16``, ``rgb8``, ``bgr8`` (→ rgb8)
``dvs_msgs/msg/EventArray``     EVENT       rpg_dvs_ros (MVSEC); per-event ``ts`` → ``t`` column
==============================  ==========  =====================================================

Timestamps are the message ``header.stamp`` (ROS time) in ns, ``ClockMode.FIXED``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from senseai_ingest.adapters.base import AdapterNotAvailable, ReplayAdapter
from senseai_ingest.stream import EVENT_DTYPES, ClockMode, Modality, Sample, SourceSpec

__all__ = ["DVS_MSGS", "SUPPORTED", "RosbagReplay", "topic_to_source_id"]

#: Message definitions for rpg_dvs_ros ``dvs_msgs`` (registered so rosbag2 files can be read).
DVS_MSGS = {
    "dvs_msgs/msg/Event": "uint16 x\nuint16 y\nbuiltin_interfaces/Time ts\nbool polarity\n",
    "dvs_msgs/msg/EventArray": (
        "std_msgs/Header header\nuint32 height\nuint32 width\ndvs_msgs/Event[] events\n"
    ),
}

SUPPORTED: Mapping[str, Modality] = {
    "sensor_msgs/msg/Imu": Modality.IMU,
    "sensor_msgs/msg/Image": Modality.FRAME,
    "dvs_msgs/msg/EventArray": Modality.EVENT,
}

_HINT = "rosbag replay needs the rosbags package: pip install senseai-ingest[ros2]"


def _rosbags() -> Any:
    try:
        import rosbags.highlevel
        import rosbags.typesys
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise AdapterNotAvailable(_HINT) from exc
    return rosbags


def typestore() -> Any:
    """Return a ROS 2 Humble typestore with ``dvs_msgs`` registered."""
    rb = _rosbags()
    ts = rb.typesys.get_typestore(rb.typesys.Stores.ROS2_HUMBLE)
    types: dict[str, Any] = {}
    for name, text in DVS_MSGS.items():
        types.update(rb.typesys.get_types_from_msg(text, name))
    ts.register(types)
    return ts


def topic_to_source_id(topic: str) -> str:
    """``/davis/left/imu`` → ``davis_left_imu``."""
    return topic.strip("/").replace("/", "_") or "root"


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _image(msg: Any) -> tuple[np.ndarray[Any, Any], str]:
    enc = str(msg.encoding).lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.asarray(msg.data, dtype=np.uint8)
    if enc in ("mono8", "8uc1"):
        return buf.reshape(h, step)[:, :w, None].copy(), "mono8"
    if enc in ("mono16", "16uc1"):
        dt = np.dtype(">u2" if msg.is_bigendian else "<u2")
        arr = buf.reshape(h, step).view(dt)[:, :w].astype(np.uint16)
        return arr[..., None], "mono16"
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return (arr[..., ::-1] if enc == "bgr8" else arr).copy(), "rgb8"
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


class RosbagReplay(ReplayAdapter):
    """Replays one topic of a rosbag1 file or rosbag2 directory."""

    def __init__(
        self,
        bag: str | Path,
        topic: str,
        *,
        source_id: str | None = None,
        msgtype: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.bag = Path(bag)
        self.topic = topic
        self._sid = source_id or topic_to_source_id(topic)
        self._msgtype = msgtype
        self._range: tuple[int, int] | None = None
        self._count = 0
        self._meta: dict[str, Any] = {}

    def _reader(self) -> Any:
        rb = _rosbags()
        return rb.highlevel.AnyReader([self.bag], default_typestore=typestore())

    def _scan(self) -> None:
        if self._range is not None:
            return
        with self._reader() as r:
            conns = [c for c in r.connections if c.topic == self.topic]
            if not conns:
                raise ValueError(f"{self.bag}: no topic {self.topic!r}")
            self._msgtype = conns[0].msgtype
            if self._msgtype not in SUPPORTED:
                raise ValueError(f"{self.topic}: unsupported message type {self._msgtype}")
            self._count = sum(c.msgcount for c in conns)
            self._range = (int(r.start_time), int(r.end_time))

    def spec(self) -> SourceSpec:
        """Describe the topic (rate from message count over the bag span)."""
        self._scan()
        assert self._range is not None and self._msgtype is not None
        span = (self._range[1] - self._range[0]) * 1e-9
        rate = (self._count - 1) / span if span > 0 and self._count > 1 else 0.0
        return SourceSpec(
            source_id=self._sid,
            modality=SUPPORTED[self._msgtype],
            nominal_rate_hz=rate,
            meta={"topic": self.topic, "msgtype": self._msgtype, "bag": str(self.bag)},
            clock_mode=ClockMode.FIXED,
        )

    def open(self) -> None:
        """Validate the topic."""
        self._scan()

    def close(self) -> None:
        """Nothing held open between passes."""

    def time_range(self) -> tuple[int, int]:
        """Bag start/end (record time, ns) — header stamps usually lie within it."""
        self._scan()
        assert self._range is not None
        return self._range

    def _convert(self, msg: Any) -> Sample | None:
        mod = SUPPORTED[str(self._msgtype)]
        if mod is Modality.EVENT:
            evs = msg.events
            if not len(evs):
                return None
            cols = {
                "t": np.fromiter((_stamp_ns(e.ts) for e in evs), np.int64, len(evs)),
                "x": np.fromiter((e.x for e in evs), EVENT_DTYPES["x"], len(evs)),
                "y": np.fromiter((e.y for e in evs), EVENT_DTYPES["y"], len(evs)),
                "p": np.fromiter((bool(e.polarity) for e in evs), EVENT_DTYPES["p"], len(evs)),
            }
            t = int(cols["t"][-1])
            meta = {"width": int(msg.width), "height": int(msg.height)}
            return Sample(self._sid, mod, t, t, cols, meta)
        t = _stamp_ns(msg.header.stamp)
        if mod is Modality.IMU:
            a, g = msg.linear_acceleration, msg.angular_velocity
            data = np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float64)
            return Sample(self._sid, mod, t, t, data, {"frame_id": msg.header.frame_id})
        arr, enc = _image(msg)
        return Sample(self._sid, mod, t, t, arr, {"encoding": enc})

    def _read_once(self) -> Iterator[Sample]:
        self._scan()
        with self._reader() as r:
            conns = [c for c in r.connections if c.topic == self.topic]
            for conn, _, raw in r.messages(connections=conns):
                s = self._convert(r.deserialize(raw, conn.msgtype))
                if s is None:
                    continue
                if self.start_ns is not None and s.t_master_ns < self.start_ns:
                    continue
                if self.end_ns is not None and s.t_master_ns >= self.end_ns:
                    return
                yield s

    @staticmethod
    def from_bag(
        bag: str | Path,
        *,
        sources: Sequence[str] | None = None,
        topics: Mapping[str, str] | None = None,
        **kw: Any,
    ) -> list[RosbagReplay]:
        """One adapter per supported topic. ``topics`` maps topic → source id (default: auto)."""
        rb = _rosbags()
        with rb.highlevel.AnyReader([Path(bag)], default_typestore=typestore()) as r:
            found = sorted({(c.topic, c.msgtype) for c in r.connections if c.msgtype in SUPPORTED})
        out = []
        for topic, msgtype in found:
            if topics is not None and topic not in topics:
                continue
            sid = topics[topic] if topics is not None else topic_to_source_id(topic)
            if sources is not None and sid not in sources:
                continue
            out.append(RosbagReplay(bag, topic, source_id=sid, msgtype=msgtype, **kw))
        return out
