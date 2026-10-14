# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""senseai-ingest: sensor streams → one master clock → timestamp-aligned ticks."""

from senseai_ingest.stream import AlignedTick, ClockMode, Modality, Sample, SourceSpec, Stream
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
