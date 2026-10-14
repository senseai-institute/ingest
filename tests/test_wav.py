# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import wave
from pathlib import Path

import numpy as np
import pytest

from senseai_ingest.adapters.base import AdapterNotAvailable
from senseai_ingest.adapters.replay.wav import WavReplay
from senseai_ingest.stream import Modality


def _wav(path: Path, data: np.ndarray, width: int, sr: int = 16_000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(data.astype(data.dtype.newbyteorder("<")).tobytes())
    return path


def test_int16_blocks_and_timestamps(tmp_path: Path) -> None:
    data = (np.arange(2500 * 4) % 1000).astype(np.int16).reshape(-1, 4)
    ad = WavReplay(
        _wav(tmp_path / "a.wav", data, 2),
        block=1000,
        t_start_ns=1_000_000_000,
        channel_positions=np.zeros((4, 3)),
    )
    spec = ad.spec()
    assert spec.modality is Modality.AUDIO and spec.nominal_rate_hz == 16.0
    assert spec.meta["channel_positions"].shape == (4, 3)
    with ad:
        out = list(ad.read())
    assert [s.array.shape for s in out] == [(1000, 4), (1000, 4), (500, 4)]
    assert out[0].t_master_ns == 1_000_000_000 + round(999 * 1e9 / 16_000)
    np.testing.assert_array_equal(np.concatenate([s.array for s in out]), data)
    assert ad.time_range() == (1_000_000_000, 1_000_000_000 + round(2499 * 1e9 / 16_000))


def test_window_and_widths(tmp_path: Path) -> None:
    data = np.zeros((1600, 1), np.int16)
    ad = WavReplay(
        _wav(tmp_path / "a.wav", data, 2), block=160, start_ns=30_000_000, end_ns=60_000_000
    )
    assert len(list(ad.read())) == 3
    u8 = WavReplay(_wav(tmp_path / "u8.wav", np.full((10, 1), 128, np.uint8), 1), block=4)
    (b, *_) = list(u8.read())
    assert b.array.dtype == np.int16 and b.array.max() == 0
    i32 = WavReplay(_wav(tmp_path / "i32.wav", np.full((10, 2), 2**30, np.int32), 4), block=4)
    (b, *_) = list(i32.read())
    assert b.array.dtype == np.float32 and b.array[0, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        WavReplay(tmp_path / "a.wav", block=0)


def test_non_pcm_needs_soundfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "f.wav"
    p.write_bytes(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x03\x00\x01\x00"
        b"\x80\x3e\x00\x00\x00\xfa\x00\x00\x04\x00\x20\x00data\x00\x00\x00\x00"
    )
    import builtins

    real = builtins.__import__

    def fake(name: str, *a: object, **k: object) -> object:
        if name == "soundfile":
            raise ImportError
        return real(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(AdapterNotAvailable, match="audio"):
        WavReplay(p)
