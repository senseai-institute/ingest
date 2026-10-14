# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Multichannel WAV replay (AUDIO). PCM 8/16/32-bit via the stdlib; other codecs need soundfile.

Payloads follow the AUDIO contract: 8/16-bit PCM → int16, 32-bit PCM and float → float32.

Each sample is a block of ``block`` frames, shape ``(N, C)``; its timestamp is that of the block's
last frame: ``t = t_start_ns + round((i_last) * 1e9 / sample_rate)``. The final block may be
shorter than ``block``.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from senseai_ingest.adapters.base import AdapterNotAvailable, ReplayAdapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["WavReplay"]

_PCM = {1: np.uint8, 2: np.int16, 4: np.int32}


class WavReplay(ReplayAdapter):
    """Replays a WAV file in fixed-size blocks.

    Args:
        path: WAV file.
        source_id: source id.
        block: frames per sample.
        t_start_ns: master time of the first frame (WAV has no absolute timestamps).
        channel_positions: optional ``(C, 3)`` mic geometry in metres, stored in ``meta``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str = "mic0",
        block: int = 1024,
        t_start_ns: int = 0,
        channel_positions: Any = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        if block < 1:
            raise ValueError("block must be >= 1")
        self.path = Path(path)
        self._sid = source_id
        self.block = block
        self.t_start_ns = int(t_start_ns)
        self.channel_positions = channel_positions
        try:
            with wave.open(str(self.path), "rb") as w:
                self.sample_rate, self.channels = w.getframerate(), w.getnchannels()
                self.n_frames, self.sampwidth = w.getnframes(), w.getsampwidth()
        except wave.Error:  # float / 24-bit / compressed: needs soundfile
            info = self._soundfile().info(str(self.path))
            self.sample_rate, self.channels = int(info.samplerate), int(info.channels)
            self.n_frames, self.sampwidth = int(info.frames), 0

    def _soundfile(self) -> Any:
        try:
            import soundfile
        except ImportError as exc:
            raise AdapterNotAvailable(
                f"{self.path}: not 8/16/32-bit PCM; pip install senseai-ingest[audio]"
            ) from exc
        return soundfile

    def _frame_t(self, i: int) -> int:
        return self.t_start_ns + round(i * 1e9 / self.sample_rate)

    def spec(self) -> SourceSpec:
        """Describe the audio source."""
        meta: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "block": self.block,
            "file": str(self.path),
        }
        if self.channel_positions is not None:
            meta["channel_positions"] = np.asarray(self.channel_positions, dtype=np.float64)
        return SourceSpec(
            source_id=self._sid,
            modality=Modality.AUDIO,
            nominal_rate_hz=self.sample_rate / self.block,
            meta=meta,
            clock_mode=ClockMode.FIXED,
        )

    def open(self) -> None:
        """Nothing to do (files are opened per pass)."""

    def close(self) -> None:
        """Nothing to release."""

    def time_range(self) -> tuple[int, int]:
        """Timestamps of the first and last frames."""
        return self.t_start_ns, self._frame_t(max(0, self.n_frames - 1))

    def _blocks(self) -> Iterator[np.ndarray[Any, Any]]:
        if self.sampwidth in _PCM:
            dt = np.dtype(_PCM[self.sampwidth]).newbyteorder("<")
            with wave.open(str(self.path), "rb") as w:
                while True:
                    raw = w.readframes(self.block)
                    if not raw:
                        return
                    a = np.frombuffer(raw, dtype=dt).reshape(-1, self.channels)
                    if self.sampwidth == 1:  # unsigned 8-bit -> int16 full scale
                        yield ((a.astype(np.int16) - 128) << 8).astype(np.int16)
                    elif self.sampwidth == 4:  # int32 -> float32 in [-1, 1]
                        yield (a.astype(np.float64) / 2**31).astype(np.float32)
                    else:
                        yield a.astype(np.int16)
        else:
            sf = self._soundfile()
            yield from sf.blocks(
                str(self.path), blocksize=self.block, dtype="float32", always_2d=True
            )

    def _read_once(self) -> Iterator[Sample]:
        i0 = 0
        for blk in self._blocks():
            t = self._frame_t(i0 + len(blk) - 1)
            i0 += len(blk)
            if self.start_ns is not None and t < self.start_ns:
                continue
            if self.end_ns is not None and t >= self.end_ns:
                return
            yield Sample(self._sid, Modality.AUDIO, t, t, blk, {})
