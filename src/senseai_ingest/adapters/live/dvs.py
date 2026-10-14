# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Event cameras: Prophesee (Metavision SDK) and iniVation (dv-processing). Stub in Gate 1.

The vendor SDKs are not on PyPI as plain wheels for every platform; install them from the vendor
channels. ``pip install senseai-ingest[dvs]`` documents the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from senseai_ingest.adapters.live import LiveAdapter
from senseai_ingest.stream import ClockMode, Modality, SourceSpec

__all__ = ["SENSORS", "DvsAdapter", "DvsConfig"]

#: Known sensors: name -> (backend, width, height).
SENSORS: dict[str, tuple[str, int, int]] = {
    "evk4": ("metavision", 1280, 720),  # Prophesee EVK4 HD (IMX636)
    "evk3": ("metavision", 640, 480),
    "dvxplorer": ("dv", 640, 480),  # iniVation DVXplorer
    "davis346": ("dv", 346, 260),
}

_BACKENDS = {"metavision": "metavision_core", "dv": "dv_processing"}


@dataclass(frozen=True, slots=True)
class DvsConfig:
    """Event camera configuration.

    Attributes:
        source_id: source id.
        sensor: key of :data:`SENSORS` (sets backend and resolution).
        serial: device serial number (``None`` = first device found).
        batch_us: events are delivered in batches of this length.
        bias_file: optional vendor bias file.
        roi: optional ``(x, y, w, h)`` hardware ROI.
    """

    source_id: str = "dvs0"
    sensor: str = "evk4"
    serial: str | None = None
    batch_us: int = 1000
    bias_file: str | None = None
    roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.sensor not in SENSORS:
            raise ValueError(f"unknown sensor {self.sensor!r}; known: {sorted(SENSORS)}")
        if self.batch_us <= 0:
            raise ValueError("batch_us must be > 0")

    @property
    def backend_name(self) -> Literal["metavision", "dv"]:
        """``metavision`` or ``dv``."""
        b = SENSORS[self.sensor][0]
        return "metavision" if b == "metavision" else "dv"

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> DvsConfig:
        """Build from a plain dict (YAML)."""
        roi = d.get("roi")
        return cls(
            source_id=d.get("source_id", "dvs0"),
            sensor=d.get("sensor", "evk4"),
            serial=d.get("serial"),
            batch_us=int(d.get("batch_us", 1000)),
            bias_file=d.get("bias_file"),
            roi=tuple(roi) if roi is not None else None,
        )


class DvsAdapter(LiveAdapter):
    """Live event camera → ``EVENT`` batches (``t`` column in device time; clock estimated)."""

    def __init__(self, config: DvsConfig | None = None) -> None:
        self.config = config or DvsConfig()
        self.backend = (_BACKENDS[self.config.backend_name], "dvs")

    def spec(self) -> SourceSpec:
        """Describe the camera."""
        _, w, h = SENSORS[self.config.sensor]
        if self.config.roi is not None:
            w, h = self.config.roi[2], self.config.roi[3]
        return SourceSpec(
            source_id=self.config.source_id,
            modality=Modality.EVENT,
            nominal_rate_hz=1e6 / self.config.batch_us,
            meta={
                "width": w,
                "height": h,
                "sensor": self.config.sensor,
                "backend": self.config.backend_name,
            },
            clock_mode=ClockMode.ESTIMATED,
        )
