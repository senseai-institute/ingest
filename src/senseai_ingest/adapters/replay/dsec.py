# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""DSEC replay: events from HDF5 (``layouts/dsec.yaml``) plus frames from PNG sequences.

Quirks handled here:

* event ``t`` is µs relative to the scalar dataset ``/t_offset`` (µs);
* frames are ``images/<cam>/rectified/NNNNNN.png`` with one absolute µs timestamp per line in
  ``images/timestamps.txt`` (shared by both cameras). PNG decoding needs Pillow
  (``pip install senseai-ingest[images]``);
* IMU and LiDAR were released later as ROS bags (``lidar_imu.zip``). Any ``*.bag`` found under
  the sequence directory is scanned: the first ``sensor_msgs/Imu`` topic becomes the ``imu``
  source and the first ``sensor_msgs/PointCloud2`` topic the ``lidar`` source
  (needs ``pip install senseai-ingest[ros2]``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from senseai_ingest.adapters.base import Adapter, AdapterNotAvailable, ReplayAdapter
from senseai_ingest.adapters.replay.hdf5 import LayoutError, open_layout
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["DsecImageReplay", "DsecReplay"]

log = structlog.get_logger(__name__)


class DsecImageReplay(ReplayAdapter):
    """Replays one DSEC camera's PNG frames as ``FRAME`` samples (``rgb8``)."""

    def __init__(
        self,
        images_dir: str | Path,
        camera: str = "left",
        *,
        source_id: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.images_dir = Path(images_dir)
        self.camera = camera
        self._sid = source_id or f"rgb_{camera}"
        sub = self.images_dir / camera
        self.frame_dir = sub / "rectified" if (sub / "rectified").is_dir() else sub
        ts_file = self.images_dir / "timestamps.txt"
        if not ts_file.exists():
            raise LayoutError(f"DSEC: missing {ts_file}")
        self._t_us = np.loadtxt(ts_file, dtype=np.int64, ndmin=1)
        self._files = sorted(self.frame_dir.glob("*.png"))
        if len(self._files) != len(self._t_us):
            raise LayoutError(
                f"DSEC: {len(self._files)} PNGs in {self.frame_dir} but "
                f"{len(self._t_us)} timestamps in {ts_file}"
            )

    def spec(self) -> SourceSpec:
        """Describe the camera."""
        n = len(self._t_us)
        span = (self._t_us[-1] - self._t_us[0]) * 1e-6 if n > 1 else 0.0
        return SourceSpec(
            source_id=self._sid,
            modality=Modality.FRAME,
            nominal_rate_hz=(n - 1) / span if span > 0 else 0.0,
            meta={"encoding": "rgb8", "camera": self.camera, "dir": str(self.frame_dir)},
            clock_mode=ClockMode.FIXED,
        )

    def open(self) -> None:
        """Check that Pillow is importable."""
        try:
            import PIL.Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterNotAvailable(
                "DSEC frames are PNG: pip install senseai-ingest[images]"
            ) from exc

    def close(self) -> None:
        """Nothing to release."""

    def time_range(self) -> tuple[int, int]:
        """First and last frame timestamps (ns)."""
        return int(self._t_us[0]) * 1000, int(self._t_us[-1]) * 1000

    def _read_once(self) -> Iterator[Sample]:
        from PIL import Image

        for path, t_us in zip(self._files, self._t_us, strict=True):
            t = int(t_us) * 1000
            if self.start_ns is not None and t < self.start_ns:
                continue
            if self.end_ns is not None and t >= self.end_ns:
                break
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
            yield Sample(self._sid, Modality.FRAME, t, t, arr, {})


class DsecReplay:
    """Factory for all sources of one DSEC sequence directory."""

    @staticmethod
    def from_sequence(
        path: str | Path,
        *,
        sources: Sequence[str] | None = None,
        frames: bool = True,
        imu: bool = True,
        lidar: bool = True,
        **kw: Any,
    ) -> list[Adapter]:
        """Return adapters for ``events_left/right``, ``rgb_left/right``, ``imu``, ``lidar``.

        ``kw`` (``start_ns``, ``end_ns``, ``speed``, ``loop``) is passed to every adapter.
        """
        root = Path(path)
        wanted = set(sources) if sources is not None else None
        ev_ids = {"events_left", "events_right"}
        ev_sources = None if wanted is None else sorted(wanted & ev_ids)
        adapters: list[Adapter] = []
        if ev_sources is None or ev_sources:
            adapters += open_layout(root, "dsec", sources=ev_sources, **kw)
        if frames:
            hits = sorted(p for p in root.glob("**/images") if (p / "timestamps.txt").exists())
            for images in hits[:1]:
                for cam in ("left", "right"):
                    sid = f"rgb_{cam}"
                    if (images / cam).is_dir() and (wanted is None or sid in wanted):
                        adapters.append(DsecImageReplay(images, cam, source_id=sid, **kw))
        bag_wanted = {
            "imu": imu and (wanted is None or "imu" in wanted),
            "lidar": lidar and (wanted is None or "lidar" in wanted),
        }
        if any(bag_wanted.values()):
            adapters += _from_bags(root, [k for k, v in bag_wanted.items() if v], **kw)
        if wanted is not None:
            missing = wanted - {a.source_id for a in adapters}
            if missing:
                raise LayoutError(f"DSEC: sources not found under {root}: {sorted(missing)}")
        return adapters


def _from_bags(root: Path, sids: Sequence[str], **kw: Any) -> list[Adapter]:
    """Return ``imu`` / ``lidar`` adapters from the first bag under ``root`` carrying each."""
    want = {"imu": Modality.IMU, "lidar": Modality.LIDAR}
    bags = sorted(p for p in root.glob("**/*.bag") if p.is_file())
    bags += sorted(p for p in root.glob("**/metadata.yaml") if p.parent != root)
    if not bags:
        return []
    from senseai_ingest.adapters.replay.rosbag import RosbagReplay

    out: dict[str, Adapter] = {}
    for bag in bags:
        bag_path = bag.parent if bag.name == "metadata.yaml" else bag
        try:
            found = RosbagReplay.from_bag(bag_path, **kw)
        except AdapterNotAvailable:
            log.warning("dsec.bag_skipped", bag=str(bag_path), reason="rosbags not installed")
            return []
        for sid in sids:
            if sid in out:
                continue
            hits = [a for a in found if a.spec().modality is want[sid]]
            if hits:
                out[sid] = RosbagReplay(bag_path, hits[0].topic, source_id=sid, **kw)
        if len(out) == len(sids):
            break
    return [out[s] for s in sids if s in out]
