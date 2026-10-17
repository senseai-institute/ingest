# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""M3ED replay (µs timestamps, one clock): generic HDF5 adapter + ``layouts/m3ed.yaml``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py

from senseai_ingest.adapters.base import Adapter
from senseai_ingest.adapters.replay.hdf5 import open_layout
from senseai_ingest.adapters.replay.ouster import OusterH5Replay

__all__ = ["M3edReplay"]


class M3edReplay:
    """Factory for all sources of one M3ED sequence (file or directory)."""

    @staticmethod
    def from_sequence(
        path: str | Path, *, sources: Sequence[str] | None = None, lidar: bool = True, **kw: Any
    ) -> list[Adapter]:
        """Return one adapter per source present in the recording.

        ``lidar`` is decoded from the raw Ouster packets in ``/ouster`` (see
        :mod:`senseai_ingest.adapters.replay.ouster`).
        """
        wanted = set(sources) if sources is not None else None
        layout_sources = None if wanted is None else sorted(wanted - {"lidar"})
        out: list[Adapter] = []
        if layout_sources is None or layout_sources:
            out += open_layout(path, "m3ed", sources=layout_sources, **kw)
        if lidar and (wanted is None or "lidar" in wanted):
            p = Path(path)
            files = [p] if p.is_file() else sorted(p.glob("*_data.h5"))
            for f in files[:1]:
                with h5py.File(f, "r") as h:
                    has = "ouster/data" in h and "ouster/metadata" in h
                if has:
                    out.append(OusterH5Replay(f, **kw))
                elif wanted is not None:
                    raise ValueError(f"M3ED: no /ouster data in {f}")
        return out
