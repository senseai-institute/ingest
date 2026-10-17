# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Tiny synthetic recordings in real dataset layouts (DSEC, M3ED, MVSEC) for tests and examples.

Nothing here downloads data; files are generated deterministically from a seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = ["make_dsec_fixture", "make_m3ed_fixture", "make_mvsec_fixture"]

#: ``t_offset`` used by the DSEC fixture (µs since epoch, large enough to exercise int64 paths).
DSEC_T_OFFSET_US = 1_600_000_000_000_000


def _events(
    rng: np.random.Generator, n: int, t0: int, t1: int, w: int, h: int
) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "t": np.sort(rng.integers(t0, t1, size=n)).astype(np.int64),
        "x": rng.integers(0, w, size=n).astype(np.uint16),
        "y": rng.integers(0, h, size=n).astype(np.uint16),
        "p": rng.integers(0, 2, size=n).astype(np.uint8),
    }


def make_dsec_fixture(
    root: str | Path,
    *,
    duration_s: float = 1.0,
    events_per_s: int = 200_000,
    fps: float = 20.0,
    image_shape: tuple[int, int] = (24, 32),
    seed: int = 0,
) -> Path:
    """Write a DSEC-shaped sequence directory with 4 sources and return its path.

    Layout: ``events/{left,right}/events.h5`` (``/events/{t,x,y,p}``, ``/t_offset``,
    ``/ms_to_idx``) and ``images/{left,right}/rectified/*.png`` + ``images/timestamps.txt``.
    Requires Pillow for the PNGs.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    dur_us = int(duration_s * 1e6)
    n = int(events_per_s * duration_s)
    for cam in ("left", "right"):
        d = root / "events" / cam
        d.mkdir(parents=True, exist_ok=True)
        ev = _events(rng, n, 0, dur_us, 640, 480)
        with h5py.File(d / "events.h5", "w") as f:
            g = f.create_group("events")
            for k, v in ev.items():
                g.create_dataset(k, data=v, chunks=True)
            f.create_dataset("t_offset", data=np.int64(DSEC_T_OFFSET_US))
            ms = np.searchsorted(ev["t"], np.arange(0, dur_us // 1000 + 1) * 1000)
            f.create_dataset("ms_to_idx", data=ms.astype(np.uint64))
    images = root / "images"
    n_img = int(duration_s * fps)
    t_img = DSEC_T_OFFSET_US + (np.arange(n_img) * (1e6 / fps)).astype(np.int64) + 5_000
    for cam in ("left", "right"):
        d = images / cam / "rectified"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_img):
            arr = rng.integers(0, 255, size=(*image_shape, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:06d}.png")
    np.savetxt(images / "timestamps.txt", t_img, fmt="%d")
    return root


def make_m3ed_fixture(root: str | Path, *, duration_s: float = 0.5, seed: int = 0) -> Path:
    """Write a single ``<seq>_data.h5`` in M3ED layout (events, rgb, imu) and return its path."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "synthetic_seq_data.h5"
    dur_us = int(duration_s * 1e6)
    t0 = 1_000_000
    with h5py.File(path, "w") as f:
        ev = _events(rng, 50_000, t0, t0 + dur_us, 1280, 720)
        g = f.create_group("prophesee/left")
        for k, v in ev.items():
            g.create_dataset(k, data=v)
        n_img = int(duration_s * 25)
        f.create_dataset("ovc/ts", data=t0 + np.arange(n_img, dtype=np.int64) * 40_000)
        f.create_dataset(
            "ovc/rgb/data", data=rng.integers(0, 255, (n_img, 12, 16, 3), dtype=np.uint8)
        )
        n_imu = int(duration_s * 400)
        f.create_dataset("ovc/imu/ts", data=t0 + np.arange(n_imu, dtype=np.int64) * 2_500)
        f.create_dataset("ovc/imu/accel", data=rng.normal(0, 1, (n_imu, 3)))
        f.create_dataset("ovc/imu/omega", data=rng.normal(0, 0.1, (n_imu, 3)))
        f.create_dataset(
            "ovc/right/data", data=rng.integers(0, 255, (n_img, 12, 16, 1), dtype=np.uint8)
        )
        n_oi = int(duration_s * 100)
        f.create_dataset("ouster/imu/ts", data=t0 + np.arange(n_oi, dtype=np.int64) * 10_000)
        f.create_dataset("ouster/imu/accel", data=rng.normal(0, 1, (n_oi, 3)))
        f.create_dataset("ouster/imu/omega", data=rng.normal(0, 0.1, (n_oi, 3)))
    return path


def make_mvsec_fixture(root: str | Path, *, duration_s: float = 0.5, seed: int = 0) -> Path:
    """Write a ``<seq>_data.hdf5`` in MVSEC layout (float64 packed events, ±1 polarity)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "synthetic_seq_data.hdf5"
    t0 = 1_506_117_000.0  # ROS time (s)
    with h5py.File(path, "w") as f:
        n = 20_000
        ev = np.stack(
            [
                rng.integers(0, 346, n).astype(np.float64),
                rng.integers(0, 260, n).astype(np.float64),
                np.sort(t0 + rng.random(n) * duration_s),
                rng.choice([-1.0, 1.0], n),
            ],
            axis=1,
        )
        f.create_dataset("davis/left/events", data=ev)
        n_img = int(duration_s * 30)
        f.create_dataset(
            "davis/left/image_raw", data=rng.integers(0, 255, (n_img, 26, 34), dtype=np.uint8)
        )
        f.create_dataset("davis/left/image_raw_ts", data=t0 + np.arange(n_img) / 30.0)
        n_imu = int(duration_s * 1000)
        f.create_dataset("davis/left/imu", data=rng.normal(0, 1, (n_imu, 6)))
        f.create_dataset("davis/left/imu_ts", data=t0 + np.arange(n_imu) / 1000.0)
    return path
