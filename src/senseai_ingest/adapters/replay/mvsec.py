# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""MVSEC replay (HDF5 export; ROS-time seconds, ±1 polarity): wraps ``layouts/mvsec.yaml``.

For the original rosbags use :mod:`senseai_ingest.adapters.replay.rosbag`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from senseai_ingest.adapters.base import Adapter
from senseai_ingest.adapters.replay.hdf5 import open_layout

__all__ = ["MvsecReplay"]


class MvsecReplay:
    """Factory for all sources of one MVSEC sequence (file or directory)."""

    @staticmethod
    def from_sequence(
        path: str | Path, *, sources: Sequence[str] | None = None, **kw: Any
    ) -> list[Adapter]:
        """Return one adapter per source present in the recording."""
        return list(open_layout(path, "mvsec", sources=sources, **kw))
