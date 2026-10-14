# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""RGB/mono cameras via V4L2 (OpenCV backend). Gate-1 stub; ``pip install senseai-ingest[live]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from senseai_ingest.adapters.live import LiveAdapter
from senseai_ingest.stream import ClockMode, Modality, SourceSpec

__all__ = ["V4l2Adapter", "V4l2Config"]

_ENCODINGS = {"rgb8": 3, "mono8": 1, "mono16": 1}


@dataclass(frozen=True, slots=True)
class V4l2Config:
    """Camera configuration (``device`` is a path like ``/dev/video0`` or an index)."""

    source_id: str = "rgb0"
    device: str | int = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    encoding: str = "rgb8"

    def __post_init__(self) -> None:
        if self.encoding not in _ENCODINGS:
            raise ValueError(f"encoding must be one of {sorted(_ENCODINGS)}")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width, height and fps must be > 0")

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> V4l2Config:
        """Build from a plain dict (YAML)."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class V4l2Adapter(LiveAdapter):
    """Live V4L2 camera → ``FRAME`` (H, W, C)."""

    backend = ("cv2", "live")

    def __init__(self, config: V4l2Config | None = None) -> None:
        self.config = config or V4l2Config()

    def spec(self) -> SourceSpec:
        """Describe the camera."""
        c = self.config
        return SourceSpec(
            source_id=c.source_id,
            modality=Modality.FRAME,
            nominal_rate_hz=c.fps,
            meta={
                "encoding": c.encoding,
                "width": c.width,
                "height": c.height,
                "channels": _ENCODINGS[c.encoding],
                "device": str(c.device),
            },
            clock_mode=ClockMode.ESTIMATED,
        )
