# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from senseai_ingest.adapters.replay.ouster import (
    BLOCKS_PER_PACKET,
    TICKS_PER_REV,
    OusterFormat,
    OusterH5Replay,
    decode_legacy_scan,
)
from senseai_ingest.stream import Modality

C = 8  # channels in the synthetic sensor


def _meta(**extra: Any) -> dict[str, Any]:
    return {
        "beam_intrinsics": {
            "beam_altitude_angles": list(np.linspace(-15, 15, C)),
            "beam_azimuth_angles": [3.0] * C,
            "lidar_origin_to_beam_origin_mm": 15.806,
        },
        "lidar_data_format": {"pixels_per_column": C, "columns_per_frame": 32},
        "lidar_mode": "32x10",
        **extra,
    }


def _encode(fmt: OusterFormat, ranges_m: np.ndarray, t0: int = 1_000) -> np.ndarray:
    """Build LEGACY packets for one scan: ranges_m is (columns, channels)."""
    cols, ch = ranges_m.shape
    blocks = []
    for v in range(cols):
        enc = int(TICKS_PER_REV * v / cols)
        hdr = np.array([t0 + v], "<u8").tobytes() + np.array([v, 7], "<u2").tobytes()
        hdr += np.array([enc], "<u4").tobytes()
        words = np.zeros((ch, 3), "<u4")
        words[:, 0] = np.round(ranges_m[v] * 1000).astype(np.uint32) & 0xFFFFF
        words[:, 1] = (np.arange(ch) + 100) | (5 << 16)  # reflectivity | signal
        blocks.append(hdr + words.tobytes() + b"\xff\xff\xff\xff")
    while len(blocks) % BLOCKS_PER_PACKET:  # pad to whole packets with invalid blocks
        blocks.append(bytes(fmt.block_bytes - 4) + b"\x00\x00\x00\x00")
    buf = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    return buf.reshape(-1, fmt.packet_bytes)


def test_format_from_metadata_variants() -> None:
    fmt = OusterFormat.from_metadata(json.dumps(_meta()))
    assert fmt.pixels_per_column == C and fmt.columns_per_frame == 32
    assert fmt.packet_bytes == 16 * (16 + C * 12 + 4)
    assert fmt.origin_to_beam_m == pytest.approx(0.015806)
    flat = {**_meta()["beam_intrinsics"], "lidar_mode": "2048x10"}
    assert OusterFormat.from_metadata(flat).columns_per_frame == 2048
    with pytest.raises(ValueError, match="beam angles"):
        OusterFormat.from_metadata(_meta(lidar_data_format={"pixels_per_column": 3}))
    tf = np.eye(4)
    tf[:3, 3] = [1000.0, 0, 0]
    assert (
        OusterFormat.from_metadata(
            _meta(lidar_to_sensor_transform=tf.ravel().tolist())
        ).lidar_to_sensor[0, 3]
        == 1.0
    )


def test_decode_geometry_roundtrip() -> None:
    fmt = OusterFormat.from_metadata(_meta())
    cols = 32
    ranges = np.full((cols, C), 10.0)
    ranges[3, :] = 0.0  # no return
    ranges[5, 2] = 0.1  # too close
    pts, t_last, ts = decode_legacy_scan(_encode(fmt, ranges), fmt, min_range_m=0.3)
    assert pts.dtype == np.float32 and pts.shape == (cols * C - C - 1, 4)
    assert t_last == 1_000 + cols - 1 and len(ts) == cols
    # all kept points lie at ~10 m from the beam origin (range measured from the beam origin)
    n = fmt.origin_to_beam_m
    d = np.linalg.norm(pts[:, :3], axis=1)
    assert np.all(np.abs(d - 10.0) < 0.05)
    # column 0 (encoder 0) -> θ_enc = 2π: points near +x axis, rotated by -3° azimuth
    first = pts[0]
    assert first[0] > 9.5 and abs(np.degrees(np.arctan2(first[1], first[0])) + 3.0) < 0.1
    assert pts[0, 3] == 100.0 and pts[1, 3] == 101.0  # reflectivity as intensity
    assert abs(pts[0, 2] - (10.0 - n) * np.sin(np.deg2rad(-15))) < 1e-3
    with pytest.raises(ValueError, match="no"):
        decode_legacy_scan(np.zeros(10, np.uint8), fmt)


def test_h5_replay_m3ed_layout(tmp_path: Path) -> None:
    fmt = OusterFormat.from_metadata(_meta())
    scans = [_encode(fmt, np.full((32, C), 5.0 + i), t0=i * 100_000) for i in range(3)]
    data = np.stack([np.concatenate([s, np.zeros((len(s), 1), np.uint8)], axis=1) for s in scans])
    path = tmp_path / "seq_data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("ouster/data", data=data)  # (3, packets, packet_bytes + 1 pad)
        f.create_dataset("ouster/metadata", data=json.dumps(_meta()))
        f.create_dataset("ouster/ts_start", data=np.array([0, 100_000, 200_000], np.int64))
        f.create_dataset("ouster/ts_end", data=np.array([90_000, 190_000, 290_000], np.int64))
    ad = OusterH5Replay(path)
    spec = ad.spec()
    assert spec.modality is Modality.LIDAR and spec.nominal_rate_hz == pytest.approx(10.0)
    assert ad.time_range() == (90_000_000, 290_000_000)
    with ad:
        out = list(ad.read())
    assert [s.t_master_ns for s in out] == [90_000_000, 190_000_000, 290_000_000]
    assert out[1].array.shape == (32 * C, 4)
    assert np.allclose(np.linalg.norm(out[1].array[:, :3], axis=1), 6.0, atol=0.05)
    ad2 = OusterH5Replay(path, start_ns=100_000_000, end_ns=250_000_000)
    with ad2:
        assert [s.meta["scan"] for s in ad2.read()] == [1]
    with pytest.raises(ValueError, match="missing"), OusterH5Replay(path, ts="nope"):
        pass
    with h5py.File(path, "a") as f:
        del f["ouster/data"]
        f.create_dataset("ouster/data", data=data[:, :, :100])
    with pytest.raises(ValueError, match="packets are"), OusterH5Replay(path):
        pass
