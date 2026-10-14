# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""ROS 2 node: runs an :class:`~senseai_ingest.sync.Aligner` and publishes its ticks.

Topics:

* ``/senseai/ticks`` — ``senseai_ingest_msgs/msg/AlignedTick`` (every source, every tick)
* ``/senseai/<source_id>`` — ``senseai_ingest_msgs/msg/SampleList`` (that source's slice of a tick)

The messages live in the ament package ``ros2/senseai_ingest_msgs`` (build it in your colcon
workspace). ``rclpy`` and the generated messages are imported lazily; the conversion functions
take the message module as an argument so they can be used and tested without ROS.

Run: ``python -m senseai_ingest.ros2.node --dataset dsec --path <seq> --tick-ms 10 --speed 1``
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from senseai_ingest.adapters.base import AdapterNotAvailable
from senseai_ingest.schema import modality_to_proto, pb, tensor_to_proto, value_to_proto
from senseai_ingest.stream import AlignedTick, Sample
from senseai_ingest.sync import Aligner

__all__ = ["load_ros", "sample_to_ros", "tick_to_ros"]

_HINT = (
    "ROS 2 bridge needs rclpy (from a sourced ROS 2 Humble/Jazzy install) and the "
    "senseai_ingest_msgs package built from ros2/senseai_ingest_msgs"
)


def load_ros() -> tuple[Any, Any]:
    """Import ``rclpy`` and ``senseai_ingest_msgs.msg`` or raise :class:`AdapterNotAvailable`."""
    try:
        import rclpy
        import senseai_ingest_msgs.msg as msgs
    except ImportError as exc:
        raise AdapterNotAvailable(_HINT) from exc
    return rclpy, msgs


def _tensor(msgs: Any, arr: np.ndarray[Any, Any]) -> Any:
    t = tensor_to_proto(arr)
    return msgs.Tensor(dtype=int(t.dtype), shape=list(t.shape), buffer=list(t.buffer))


def sample_to_ros(msgs: Any, s: Sample) -> Any:
    """Convert a :class:`Sample` into ``senseai_ingest_msgs/Sample``."""
    meta = pb.ValueMap(entries={str(k): value_to_proto(v) for k, v in s.meta.items()})
    kw: dict[str, Any] = {
        "source_id": s.source_id,
        "modality": modality_to_proto(s.modality),
        "t_source_ns": s.t_source_ns,
        "t_master_ns": s.t_master_ns,
        "meta_pb": list(meta.SerializeToString(deterministic=True)),
    }
    if isinstance(s.data, np.ndarray):
        kw |= {"has_tensor": True, "tensor": _tensor(msgs, s.data), "keys": [], "tensors": []}
    else:
        keys = sorted(s.data)
        kw |= {
            "has_tensor": False,
            "tensor": msgs.Tensor(dtype=0, shape=[], buffer=[]),
            "keys": keys,
            "tensors": [_tensor(msgs, np.asarray(s.data[k])) for k in keys],
        }
    return msgs.Sample(**kw)


def sample_list_to_ros(msgs: Any, source_id: str, samples: Sequence[Sample]) -> Any:
    """Convert one source's samples into ``senseai_ingest_msgs/SampleList``."""
    return msgs.SampleList(source_id=source_id, samples=[sample_to_ros(msgs, s) for s in samples])


def tick_to_ros(msgs: Any, tick: AlignedTick) -> Any:
    """Convert an :class:`AlignedTick` into ``senseai_ingest_msgs/AlignedTick``."""
    return msgs.AlignedTick(
        index=tick.index,
        t_master_ns=tick.t_master_ns,
        tick_ns=tick.tick_ns,
        sources=[sample_list_to_ros(msgs, sid, tick.samples[sid]) for sid in sorted(tick.samples)],
    )


def run_node(
    aligner: Aligner, *, namespace: str = "/senseai", per_source: bool = True
) -> None:  # pragma: no cover - needs a ROS 2 install
    """Spin a node that publishes every tick of ``aligner`` until it ends or ROS shuts down."""
    rclpy, msgs = load_ros()
    from rclpy.node import Node

    rclpy.init()
    node = Node("senseai_ingest")
    pub_tick = node.create_publisher(msgs.AlignedTick, f"{namespace}/ticks", 10)
    pubs = (
        {
            sid: node.create_publisher(msgs.SampleList, f"{namespace}/{sid}", 10)
            for sid in aligner.source_ids
        }
        if per_source
        else {}
    )
    done = threading.Event()

    def pump() -> None:
        try:
            with aligner:
                for tick in aligner:
                    if not rclpy.ok():
                        break
                    pub_tick.publish(tick_to_ros(msgs, tick))
                    for sid, pub in pubs.items():
                        pub.publish(sample_list_to_ros(msgs, sid, tick.samples[sid]))
            node.get_logger().info(f"replay finished: {aligner.stats()['ticks']} ticks")
        finally:
            done.set()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        while rclpy.ok() and not done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a ROS 2 install
    """``python -m senseai_ingest.ros2.node --dataset ... --path ...``."""
    from senseai_ingest.cli import build_adapters
    from senseai_ingest.sync import AlignerConfig

    ap = argparse.ArgumentParser(prog="senseai-ingest-ros2")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--layout")
    ap.add_argument("--tick-ms", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--namespace", default="/senseai")
    ap.add_argument("--no-per-source", action="store_true")
    args, _ros_args = ap.parse_known_args(argv)
    adapters = build_adapters(
        args.dataset, args.path, layout=args.layout, speed=args.speed, loop=args.loop
    )
    aligner = Aligner(adapters, AlignerConfig(tick_ns=round(args.tick_ms * 1e6)))
    run_node(aligner, namespace=args.namespace, per_source=not args.no_per_source)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
