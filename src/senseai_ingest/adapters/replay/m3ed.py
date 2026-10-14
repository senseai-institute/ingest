# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""M3ED replay (µs timestamps, one clock): generic HDF5 adapter + ``layouts/m3ed.yaml``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from senseai_ingest.adapters.base import Adapter
from senseai_ingest.adapters.replay.hdf5 import open_layout

__all__ = ["M3edReplay"]


class M3edReplay:
    """Factory for all sources of one M3ED sequence (file or directory)."""

    @staticmethod
    def from_sequence(
        path: str | Path, *, sources: Sequence[str] | None = None, **kw: Any
    ) -> list[Adapter]:
        """Return one adapter per source present in the recording."""
        return list(open_layout(path, "m3ed", sources=sources, **kw))
