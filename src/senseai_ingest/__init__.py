# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""senseai-ingest: sensor streams → one master clock → timestamp-aligned ticks.

The data types import eagerly (numpy only); the aligner and throttle are loaded on first use so
``import senseai_ingest.stream`` / ``senseai_ingest.schema`` stay lightweight for consumers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from senseai_ingest.stream import AlignedTick, ClockMode, Modality, Sample, SourceSpec, Stream

if TYPE_CHECKING:
    from senseai_ingest.sync import Aligner, AlignerConfig, LatePolicy
    from senseai_ingest.throttle import ThrottleConfig, ThrottlePolicy

__version__ = "0.1.0"

__all__ = [
    "AlignedTick",
    "Aligner",
    "AlignerConfig",
    "ClockMode",
    "LatePolicy",
    "Modality",
    "Sample",
    "SourceSpec",
    "Stream",
    "ThrottleConfig",
    "ThrottlePolicy",
    "__version__",
]

_LAZY = {
    "Aligner": "senseai_ingest.sync",
    "AlignerConfig": "senseai_ingest.sync",
    "LatePolicy": "senseai_ingest.sync",
    "ThrottleConfig": "senseai_ingest.throttle",
    "ThrottlePolicy": "senseai_ingest.throttle",
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily imported public names."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'senseai_ingest' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod), name)
