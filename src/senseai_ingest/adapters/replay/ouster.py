# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Ouster LiDAR: decode raw LEGACY-format packets into ``LIDAR`` point clouds (pure numpy).

Used by M3ED, which stores the sensor's raw UDP packets (``/ouster/data``) plus the sensor
metadata JSON (``/ouster/metadata``) rather than decoded points. No Ouster SDK is needed.

LEGACY packet (firmware 1.x-2.x; deprecated in 3.x): 16 measurement blocks, each::

    u64 timestamp_ns | u16 measurement_id | u16 frame_id | u32 encoder_count
    pixels_per_column x channel block (12 bytes):
        u32 word0: bits 0..19 range_mm
        u32 word1: bits 0..15 reflectivity, bits 16..31 signal photons
        u32 word2: bits 0..15 near-IR photons
    u32 block_status (0xFFFFFFFF = valid)

Range → Cartesian (sensor docs, "lidar frame")::

    θ_enc = 2π · (1 - encoder_count / ticks_per_rev)        # ticks_per_rev = 90112
    θ_az  = -2π · beam_azimuth_angles[u] / 360
    φ     =  2π · beam_altitude_angles[u] / 360
    r     = range_mm - n,  n = lidar_origin_to_beam_origin_mm
    x = r·cos(θ_enc + θ_az)·cos φ + n·cos θ_enc
    y = r·sin(θ_enc + θ_az)·cos φ + n·sin θ_enc
    z = r·sin φ

Output per scan: ``(N, 4) float32`` ``x, y, z`` in **metres** and ``intensity`` = reflectivity,
zero-range returns dropped. The scan timestamp is the last valid block's timestamp.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from senseai_ingest.adapters.base import ReplayAdapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

__all__ = ["OusterFormat", "OusterH5Replay", "decode_legacy_scan"]

TICKS_PER_REV = 90112
BLOCKS_PER_PACKET = 16
_HEADER = 16
_CHANNEL = 12
_STATUS = 4


@dataclass(frozen=True, slots=True)
class OusterFormat:
    """Beam geometry and packet layout taken from the sensor metadata JSON."""

    pixels_per_column: int
    columns_per_frame: int
    beam_altitude_rad: np.ndarray[Any, Any]
    beam_azimuth_rad: np.ndarray[Any, Any]
    origin_to_beam_m: float
    lidar_to_sensor: np.ndarray[Any, Any]  # 4x4, metres

    @property
    def block_bytes(self) -> int:
        """Bytes per measurement block."""
        return _HEADER + self.pixels_per_column * _CHANNEL + _STATUS

    @property
    def packet_bytes(self) -> int:
        """Bytes per packet (16 blocks)."""
        return self.block_bytes * BLOCKS_PER_PACKET

    @classmethod
    def from_metadata(cls, meta: str | bytes | Mapping[str, Any]) -> OusterFormat:
        """Parse the sensor metadata JSON (``get_metadata`` / ``ouster-sdk`` ``SensorInfo``)."""
        d: Mapping[str, Any] = json.loads(meta) if isinstance(meta, str | bytes) else meta
        bi = d.get("beam_intrinsics", d)
        alt = np.deg2rad(np.asarray(bi["beam_altitude_angles"], dtype=np.float64))
        azi = np.deg2rad(np.asarray(bi["beam_azimuth_angles"], dtype=np.float64))
        n_mm = float(bi.get("lidar_origin_to_beam_origin_mm", 0.0))
        fmt = d.get("lidar_data_format", d.get("data_format", {}))
        ppc = int(fmt.get("pixels_per_column", len(alt)))
        if ppc != len(alt):
            raise ValueError(f"pixels_per_column={ppc} but {len(alt)} beam angles")
        cpf = int(fmt.get("columns_per_frame", 0))
        if not cpf:
            mode = str(d.get("lidar_mode", d.get("config_params", {}).get("lidar_mode", "1024x10")))
            cpf = int(mode.split("x")[0])
        tf = d.get("lidar_to_sensor_transform")
        if tf is None:
            tf = d.get("lidar_intrinsics", {}).get("lidar_to_sensor_transform")
        mat = np.eye(4) if tf is None else np.asarray(tf, dtype=np.float64).reshape(4, 4)
        mat = mat.copy()
        mat[:3, 3] /= 1000.0  # mm -> m
        return cls(ppc, cpf, alt, azi, n_mm / 1000.0, mat)


def decode_legacy_scan(
    packets: np.ndarray[Any, Any], fmt: OusterFormat, *, min_range_m: float = 0.0
) -> tuple[np.ndarray[Any, Any], int, np.ndarray[Any, Any]]:
    """Decode the packets of one scan.

    Args:
        packets: ``(n_packets, packet_bytes)`` or flat ``uint8`` buffer (trailing bytes ignored).
        fmt: sensor format.
        min_range_m: drop returns closer than this (also drops zero/no-return pixels).

    Returns:
        ``(points (N, 4) float32 [x, y, z, reflectivity] in the sensor frame, t_last_ns,
        timestamps_ns (n_blocks,) of valid blocks)``.
    """
    buf = np.ascontiguousarray(packets, dtype=np.uint8).reshape(-1)
    bb = fmt.block_bytes
    pb = fmt.packet_bytes
    n_pk = len(buf) // pb
    if n_pk == 0:
        raise ValueError(f"buffer of {len(buf)} bytes holds no {pb}-byte packet")
    blocks = buf[: n_pk * pb].reshape(n_pk * BLOCKS_PER_PACKET, bb)
    ts = blocks[:, 0:8].copy().view("<u8").reshape(-1)
    enc = blocks[:, 12:16].copy().view("<u4").reshape(-1).astype(np.float64)
    status = blocks[:, bb - 4 : bb].copy().view("<u4").reshape(-1)
    valid = status == 0xFFFFFFFF
    ch = blocks[:, _HEADER : _HEADER + fmt.pixels_per_column * _CHANNEL]
    words = ch.copy().view("<u4").reshape(len(blocks), fmt.pixels_per_column, 3)
    rng_m = (words[:, :, 0] & 0xFFFFF).astype(np.float64) / 1000.0
    refl = (words[:, :, 1] & 0xFFFF).astype(np.float32)

    theta_enc = 2 * np.pi * (1.0 - enc / TICKS_PER_REV)  # (B,)
    phi = fmt.beam_altitude_rad[None, :]  # (1, C)
    theta = theta_enc[:, None] + (-fmt.beam_azimuth_rad)[None, :]  # (B, C)
    n = fmt.origin_to_beam_m
    r = rng_m - n
    x = r * np.cos(theta) * np.cos(phi) + n * np.cos(theta_enc)[:, None]
    y = r * np.sin(theta) * np.cos(phi) + n * np.sin(theta_enc)[:, None]
    z = r * np.sin(phi)
    keep = valid[:, None] & (rng_m > max(min_range_m, 0.0))
    xyz = np.stack([x[keep], y[keep], z[keep]], axis=1)
    if not np.allclose(fmt.lidar_to_sensor, np.eye(4)):
        xyz = xyz @ fmt.lidar_to_sensor[:3, :3].T + fmt.lidar_to_sensor[:3, 3]
    pts = np.concatenate([xyz.astype(np.float32), refl[keep][:, None]], axis=1)
    t_valid = ts[valid]
    t_last = int(t_valid.max()) if len(t_valid) else int(ts.max())
    return pts, t_last, t_valid


class OusterH5Replay(ReplayAdapter):
    """Replays raw Ouster packets stored per scan in an HDF5 file (M3ED ``/ouster`` layout).

    Datasets (defaults are M3ED's): ``data`` ``(n_scans, packets_per_scan, packet_bytes)`` uint8,
    ``metadata`` scalar JSON string, ``ts_end`` (or ``ts_start``) per-scan timestamps in µs.
    Timestamps come from the file's per-scan datasets (the common dataset clock), not from the
    packets' internal sensor clock.
    """

    def __init__(
        self,
        file: str | Path,
        *,
        source_id: str = "lidar",
        group: str = "/ouster",
        data: str = "data",
        metadata: str = "metadata",
        ts: str = "ts_end",
        time_unit_ns: int = 1_000,
        min_range_m: float = 0.3,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.file = Path(file)
        self._sid = source_id
        self.group = group.rstrip("/")
        self._data, self._meta, self._ts = data, metadata, ts
        self._unit = int(time_unit_ns)
        self.min_range_m = min_range_m
        self._f: h5py.File | None = None
        self._fmt: OusterFormat | None = None

    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.file, "r")
        return self._f

    def _ds(self, name: str) -> Any:
        return self._file()[f"{self.group}/{name}"]

    def format(self) -> OusterFormat:
        """Parse (once) the sensor metadata stored in the file."""
        if self._fmt is None:
            raw = self._ds(self._meta)[()]
            if isinstance(raw, np.ndarray):
                raw = raw.reshape(-1)[0]
            self._fmt = OusterFormat.from_metadata(raw)
        return self._fmt

    def _t_at(self, i: int) -> int:
        return int(self._ds(self._ts)[i]) * self._unit

    def spec(self) -> SourceSpec:
        """Describe the LiDAR."""
        fmt = self.format()
        n = int(self._ds(self._data).shape[0])
        first, last = self.time_range()
        span = (last - first) * 1e-9
        return SourceSpec(
            source_id=self._sid,
            modality=Modality.LIDAR,
            nominal_rate_hz=(n - 1) / span if span > 0 and n > 1 else 0.0,
            meta={
                "frame_id": "os_sensor",
                "file": str(self.file),
                "pixels_per_column": fmt.pixels_per_column,
                "columns_per_frame": fmt.columns_per_frame,
                "units": "m",
                "intensity": "reflectivity",
            },
            clock_mode=ClockMode.FIXED,
        )

    def open(self) -> None:
        """Open the file and validate the layout."""
        f = self._file()
        for name in (self._data, self._meta, self._ts):
            if f"{self.group}/{name}" not in f:
                raise ValueError(f"{self.file}: missing {self.group}/{name}")
        fmt = self.format()
        pb = int(self._ds(self._data).shape[-1])
        if pb < fmt.packet_bytes:
            raise ValueError(
                f"{self.file}: packets are {pb} bytes, format needs {fmt.packet_bytes}"
            )

    def close(self) -> None:
        """Close the file."""
        if self._f is not None:
            self._f.close()
            self._f = None

    def time_range(self) -> tuple[int, int]:
        """First and last scan timestamps (ns)."""
        n = int(self._ds(self._ts).shape[0])
        return (self._t_at(0), self._t_at(n - 1)) if n else (0, 0)

    def _read_once(self) -> Iterator[Sample]:
        fmt = self.format()
        ds = self._ds(self._data)
        t_all = np.asarray(self._ds(self._ts)).astype(np.int64) * self._unit
        lo = int(np.searchsorted(t_all, self.start_ns)) if self.start_ns is not None else 0
        hi = int(np.searchsorted(t_all, self.end_ns)) if self.end_ns is not None else len(t_all)
        for i in range(lo, hi):
            raw = np.asarray(ds[i])
            if raw.shape[-1] > fmt.packet_bytes:  # tolerate a per-packet pad byte
                raw = raw[..., : fmt.packet_bytes]
            pts, _, _ = decode_legacy_scan(raw, fmt, min_range_m=self.min_range_m)
            t = int(t_all[i])
            yield Sample(self._sid, Modality.LIDAR, t, t, pts, {"scan": i})
