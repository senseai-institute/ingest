# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Microphone arrays (ReSpeaker 4/6-mic) via sounddevice/ALSA. Stub; ``[audio]`` extra."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from senseai_ingest.adapters.live import LiveAdapter
from senseai_ingest.stream import ClockMode, Modality, SourceSpec

__all__ = ["GEOMETRIES", "MicArrayAdapter", "MicArrayConfig"]


def _circle(n: int, radius_m: float) -> np.ndarray[Any, Any]:
    ang = [2 * math.pi * i / n for i in range(n)]
    return np.array([[radius_m * math.cos(a), radius_m * math.sin(a), 0.0] for a in ang])


#: Approximate mic positions (C x 3, metres, array frame); check your board revision.
GEOMETRIES: dict[str, np.ndarray[Any, Any]] = {
    "respeaker4": _circle(4, 0.0325),  # ReSpeaker 4-mic array (square, 45.7 mm side)
    "respeaker6": _circle(6, 0.0463),  # ReSpeaker 6-mic circular array
}


@dataclass(frozen=True, slots=True)
class MicArrayConfig:
    """Mic array configuration.

    Attributes:
        source_id: source id.
        device: sounddevice device name/index (``None`` = default input).
        geometry: key of :data:`GEOMETRIES` or ``None``.
        channels: channels to capture (defaults to the geometry's mic count).
        sample_rate: Hz.
        block: frames per sample.
        dtype: ``int16`` or ``float32``.
    """

    source_id: str = "mic0"
    device: str | int | None = None
    geometry: str | None = "respeaker4"
    channels: int | None = None
    sample_rate: int = 16_000
    block: int = 1024
    dtype: str = "int16"

    def __post_init__(self) -> None:
        if self.geometry is not None and self.geometry not in GEOMETRIES:
            raise ValueError(f"unknown geometry {self.geometry!r}; known: {sorted(GEOMETRIES)}")
        if self.dtype not in ("int16", "float32"):
            raise ValueError("dtype must be int16 or float32")
        if self.block <= 0 or self.sample_rate <= 0:
            raise ValueError("block and sample_rate must be > 0")

    @property
    def n_channels(self) -> int:
        """Effective channel count."""
        if self.channels is not None:
            return self.channels
        return len(GEOMETRIES[self.geometry]) if self.geometry else 1


class MicArrayAdapter(LiveAdapter):
    """Live microphone array → ``AUDIO`` blocks (N, C)."""

    backend = ("sounddevice", "audio")

    def __init__(self, config: MicArrayConfig | None = None) -> None:
        self.config = config or MicArrayConfig()

    def spec(self) -> SourceSpec:
        """Describe the array."""
        c = self.config
        meta: dict[str, Any] = {
            "sample_rate": c.sample_rate,
            "channels": c.n_channels,
            "block": c.block,
            "dtype": c.dtype,
        }
        if c.geometry is not None:
            meta["channel_positions"] = GEOMETRIES[c.geometry][: c.n_channels]
            meta["geometry"] = c.geometry
        return SourceSpec(
            source_id=c.source_id,
            modality=Modality.AUDIO,
            nominal_rate_hz=c.sample_rate / c.block,
            meta=meta,
            clock_mode=ClockMode.ESTIMATED,
        )
