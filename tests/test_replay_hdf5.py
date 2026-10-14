# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from senseai_ingest.adapters.replay.dsec import DsecImageReplay, DsecReplay
from senseai_ingest.adapters.replay.hdf5 import (
    FieldRef,
    Hdf5Replay,
    Layout,
    LayoutError,
    builtin_layouts,
    describe_h5,
    detect_layout,
    load_layout,
    open_layout,
)
from senseai_ingest.adapters.replay.m3ed import M3edReplay
from senseai_ingest.adapters.replay.mvsec import MvsecReplay
from senseai_ingest.schema import hash_ticks
from senseai_ingest.stream import ClockMode, Modality
from senseai_ingest.sync import Aligner, AlignerConfig, run_ticks
from senseai_ingest.testing import DSEC_T_OFFSET_US

MS = 1_000_000


def _layout(**src: Any) -> dict[str, Any]:
    base = {"id": "s", "modality": "imu", "t": "/t", "data": "/d"}
    return {"name": "x", "sources": [base | src]}


def test_builtin_layouts_parse() -> None:
    names = builtin_layouts()
    assert {"dsec", "m3ed", "mvsec"} <= set(names)
    for n in names:
        lay = load_layout(n)
        assert lay.sources and lay.name == n


@pytest.mark.parametrize(
    ("doc", "msg"),
    [
        ([], "sources"),
        ({"sources": [{"id": "a"}]}, "id' and 'modality"),
        (_layout(modality="smell"), "unknown modality"),
        (_layout(time={"unit": "fortnight"}), "time.unit"),
        (_layout(polarity="weird"), "polarity"),
        (_layout(modality="event"), "events"),
        ({"sources": [{"id": "a", "modality": "imu"}]}, "'t' and 'data'"),
        (_layout(t=5), "dataset path"),
        ({"sources": [_layout()["sources"][0], _layout()["sources"][0]]}, "duplicate"),
    ],
)
def test_layout_validation(doc: Any, msg: str) -> None:
    with pytest.raises(LayoutError, match=msg):
        Layout.parse(doc)


def test_load_layout_from_path_mapping_and_missing(tmp_path: Path) -> None:
    p = tmp_path / "l.yaml"
    p.write_text(
        "name: mine\nsources:\n"
        "  - {id: a, modality: imu, t: /t, data: [/d, {path: /e, column: 1}]}\n"
    )
    lay = load_layout(p)
    assert lay.name == "mine" and lay.sources[0].data[1] == FieldRef("/e", 1)
    assert load_layout(lay) is lay
    assert load_layout(_layout()).sources[0].id == "s"
    with pytest.raises(LayoutError, match="no such layout"):
        load_layout(tmp_path / "nope.yaml")


def _write(tmp_path: Path, **ds: Any) -> Path:
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as f:
        for k, v in ds.items():
            f.create_dataset(k, data=v)
    return path


def test_generic_tensor_source_concat_offset_and_window(tmp_path: Path) -> None:
    t = np.arange(10, dtype=np.int64) * 1000  # µs
    path = _write(tmp_path, t=t, a=np.ones((10, 3)), g=np.zeros((10, 3)), off=np.int64(500))
    lay = _layout(data=["/a", "/g"], time={"unit": "us", "offset": {"dataset": "/off"}})
    (ad,) = open_layout(path, lay)
    spec = ad.spec()
    assert spec.offset_ns == 500_000 and spec.clock_mode is ClockMode.FIXED
    assert spec.nominal_rate_hz == pytest.approx(1000.0)
    with ad:
        out = list(ad.read())
    assert len(out) == 10 and out[0].array.shape == (6,)
    assert out[1].t_source_ns == 1_000_000 and out[1].t_master_ns == 1_500_000
    ad2 = Hdf5Replay(path, ad.layout, start_ns=3_500_000, end_ns=6_500_000, chunk_rows=2)
    with ad2:
        assert [s.t_master_ns for s in ad2.read()] == [3_500_000, 4_500_000, 5_500_000]
    assert ad2.time_range() == (500_000, 9_500_000)


def test_offset_from_attr_and_float_seconds(tmp_path: Path) -> None:
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=np.array([0.5, 1.25]))
        f.create_dataset("d", data=np.zeros((2, 4, 5), np.uint8))
        f["t"].attrs["base"] = 10.0
    lay = _layout(modality="frame", time={"unit": "s", "offset": {"attr": "/t@base"}})
    (ad,) = open_layout(path, lay)
    with ad:
        out = list(ad.read())
    assert [s.t_master_ns for s in out] == [10_500_000_000, 11_250_000_000]
    assert out[0].array.shape == (4, 5, 1)  # (H, W) frames gain a channel axis
    bad = Hdf5Replay(path, load_layout(_layout(time={"offset": {"nope": 1}})).sources[0])
    with pytest.raises(LayoutError, match=r"time\.offset"):
        bad.spec()


def test_missing_datasets_and_files(tmp_path: Path) -> None:
    path = _write(tmp_path, t=np.arange(3))
    with pytest.raises(LayoutError, match="not found"):
        open_layout(path, _layout())
    assert open_layout(path, _layout(optional=True)) == []
    with pytest.raises(LayoutError, match="no sources"):
        open_layout(path, _layout(), sources=["zz"])
    with pytest.raises(FileNotFoundError):
        open_layout(tmp_path / "missing", _layout())
    with pytest.raises(LayoutError, match="directory"):
        open_layout(tmp_path, _layout())
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write(tmp_path / "a", t=np.arange(3), d=np.zeros(3))
    _write(tmp_path / "b", t=np.arange(3), d=np.zeros(3))
    with pytest.raises(LayoutError, match="matched 2"):
        open_layout(tmp_path, _layout(file="*/rec.h5"))
    lay = load_layout(_layout()).sources[0]
    with pytest.raises(LayoutError, match="datasets missing"):
        Hdf5Replay(path, lay).open()
    empty = _write(tmp_path / "a", t=np.zeros(0, np.int64), d=np.zeros((0, 3)))
    ad = Hdf5Replay(empty, lay)
    assert ad.time_range() == (0, 0) and ad.spec().nominal_rate_hz == 0.0


def test_dsec_events_offset_batching_and_frames(dsec_dir: Path) -> None:
    ads = DsecReplay.from_sequence(dsec_dir)
    assert sorted(a.source_id for a in ads) == [
        "events_left",
        "events_right",
        "rgb_left",
        "rgb_right",
    ]
    ev = next(a for a in ads if a.source_id == "events_left")
    with ev:
        batches = list(ev.read())
    t_all = np.concatenate([b.columns["t"] for b in batches])
    assert len(t_all) == 200_000 and np.all(np.diff(t_all) >= 0)
    assert t_all[0] >= DSEC_T_OFFSET_US * 1000
    assert all(int(b.columns["t"][-1]) == b.t_master_ns for b in batches)
    assert all(b.t_master_ns - b.t_source_ns == DSEC_T_OFFSET_US * 1000 for b in batches)
    assert len({int(b.columns["t"][0]) // MS for b in batches}) == len(batches)  # 1 ms batches
    assert ev.spec().meta["width"] == 640
    rgb = next(a for a in ads if a.source_id == "rgb_left")
    with rgb:
        frames = list(rgb.read())
    assert len(frames) == 20 and frames[0].array.shape == (24, 32, 3)
    assert rgb.spec().nominal_rate_hz == pytest.approx(20.0)


def test_dsec_source_selection_and_errors(dsec_dir: Path, tmp_path: Path) -> None:
    ads = DsecReplay.from_sequence(dsec_dir, sources=["events_left", "rgb_right"])
    assert sorted(a.source_id for a in ads) == ["events_left", "rgb_right"]
    assert [a.source_id for a in DsecReplay.from_sequence(dsec_dir, sources=["rgb_left"])] == [
        "rgb_left"
    ]
    with pytest.raises(LayoutError, match="not found"):
        DsecReplay.from_sequence(dsec_dir, sources=["imu"])
    with pytest.raises(LayoutError, match="timestamps"):
        DsecImageReplay(tmp_path)
    (tmp_path / "left").mkdir()
    (tmp_path / "timestamps.txt").write_text("1\n2\n")
    with pytest.raises(LayoutError, match="PNGs"):
        DsecImageReplay(tmp_path)


def test_dsec_four_source_replay_is_deterministic(dsec_dir: Path) -> None:
    def run() -> tuple[str, dict[str, Any]]:
        al = Aligner(DsecReplay.from_sequence(dsec_dir), AlignerConfig(tick_ns=10 * MS))
        ticks = run_ticks(al)
        return hash_ticks(ticks), al.stats()

    h1, st = run()
    h2, _ = run()
    assert h1 == h2
    assert st["ticks"] >= 100
    assert st["sync_jitter_p99_ns"] < 1 * MS  # Gate-1 acceptance on DSEC-layout replay
    assert all(v["dropped"] == 0 for v in st["sources"].values())
    assert st["sources"]["events_left"]["events"] == 200_000


def test_dsec_start_end_window(dsec_dir: Path) -> None:
    t0 = DSEC_T_OFFSET_US * 1000
    ads = DsecReplay.from_sequence(dsec_dir, start_ns=t0 + 200 * MS, end_ns=t0 + 300 * MS)
    ticks = run_ticks(Aligner(ads, AlignerConfig(tick_ns=10 * MS, start_ns=t0 + 200 * MS)))
    for t in ticks:
        for lst in t.samples.values():
            assert all(t0 + 200 * MS <= s.t_master_ns < t0 + 300 * MS for s in lst)
    assert sum(len(t.samples["rgb_left"]) for t in ticks) == 2


def test_m3ed_and_mvsec_layouts(m3ed_file: Path, mvsec_file: Path) -> None:
    m3 = {a.source_id: a for a in M3edReplay.from_sequence(m3ed_file)}
    assert set(m3) == {"events_left", "rgb", "imu"}
    with m3["imu"] as imu:
        s = next(iter(imu.read()))
    assert s.array.shape == (6,) and s.modality is Modality.IMU
    mv = {a.source_id: a for a in MvsecReplay.from_sequence(mvsec_file)}
    assert set(mv) == {"events_left", "frames_left", "imu_left"}
    with mv["events_left"] as ev:
        b = next(iter(ev.read()))
    assert set(np.unique(b.columns["p"])) <= {0, 1}
    assert b.columns["x"].dtype == np.uint16
    assert b.t_master_ns > 1_506_117_000 * 10**9
    ticks = run_ticks(Aligner(list(mv.values()), AlignerConfig(tick_ns=10 * MS)))
    assert len(ticks) >= 49


def test_loop_shifts_timestamps(tmp_path: Path) -> None:
    path = _write(tmp_path, t=np.array([0, 10, 20], np.int64), d=np.zeros((3, 1)))
    (ad,) = open_layout(path, _layout(time={"unit": "ns"}), loop=True)
    with ad:
        it = iter(ad.read())
        ts = [next(it).t_master_ns for _ in range(7)]
    assert ts == [0, 10, 20, 21, 31, 41, 42]


def test_describe_and_detect(
    dsec_dir: Path, m3ed_file: Path, mvsec_file: Path, tmp_path: Path
) -> None:
    rows = describe_h5(dsec_dir / "events" / "left" / "events.h5")
    paths = {r["path"] for r in rows}
    assert {"/events/t", "/t_offset", "/ms_to_idx"} <= paths
    assert detect_layout(dsec_dir) == "dsec"
    assert detect_layout(m3ed_file) == "m3ed"
    assert detect_layout(mvsec_file) == "mvsec"
    assert detect_layout(tmp_path) is None
