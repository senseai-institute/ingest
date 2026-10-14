# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import h5py
import numpy as np
import pytest

from senseai_ingest.stream import AlignedTick, Modality, Sample, SourceSpec
from senseai_ingest.writer import TickReader, TickWriter, write_ticks
from tests.helpers import events, imu

SPECS = {
    "a": SourceSpec("a", Modality.IMU, 1000.0),
    "e": SourceSpec("e", Modality.EVENT),
    "m": SourceSpec("m", Modality.AUDIO),
    "g": SourceSpec("g", Modality.GAS),
    "b": SourceSpec("b", Modality.CONTACT),
}


def _ticks() -> list[AlignedTick]:
    aud = [Sample("m", Modality.AUDIO, 5, 5, np.full((n, 2), n, np.int16), {}) for n in (3, 1)]
    gas = Sample("g", Modality.GAS, 6, 6, {"co2": np.array(412.0), "voc": np.array(1.5)}, {})
    bump = Sample("b", Modality.CONTACT, 7, 7, np.array(5, np.uint32), {"kind": "bumper"})
    return [
        AlignedTick(
            0,
            10,
            10,
            {
                "a": [imu("a", 1), imu("a", 2)],
                "e": [events("e", [3, 4])],
                "m": aud,
                "g": [gas],
                "b": [bump],
            },
        ),
        AlignedTick(1, 20, 10, {"a": [], "e": [], "m": [], "g": [], "b": []}),
        AlignedTick(
            2,
            30,
            10,
            {"a": [imu("a", 25)], "e": [events("e", [21, 22, 29])], "m": [], "g": [], "b": []},
        ),
    ]


def test_roundtrip_all_payload_layouts(tmp_path: Path) -> None:
    path = tmp_path / "t.h5"
    ticks = _ticks()
    with TickWriter(path, SPECS, tick_ns=10, flush_every=1) as w:
        for t in ticks:
            w.write(t)
    with TickReader(path) as r:
        back = list(r)
        assert r.t0_ns == 0 and r.counts.tolist()[0] == [2, 1, 1, 1, 2]  # a, b, e, g, m
        assert r.specs()["a"].nominal_rate_hz == 1000.0
    assert [t.index for t in back] == [0, 1, 2]
    assert back[1].samples == {k: [] for k in SPECS}
    assert [s.t_master_ns for s in back[2].samples["a"]] == [25]
    assert back[2].samples["e"][0].columns["t"].tolist() == [21, 22, 29]
    assert [s.array.shape for s in back[0].samples["m"]] == [(3, 2), (1, 2)]
    assert back[0].samples["g"][0].columns["co2"].tolist() == [412.0]
    assert back[0].samples["b"][0].array.tolist() == [5]
    with h5py.File(path) as f:
        assert f["sources/a"].attrs["payload"] == "stacked"
        assert f["sources/m"].attrs["payload"] == "ragged"
        assert f["sources/e"].attrs["payload"] == "columns"
        assert f["sources/e/data_ptr"][:].tolist() == [0, 2, 5]


def test_writer_validation(tmp_path: Path) -> None:
    ticks = _ticks()
    with TickWriter(tmp_path / "x.h5", SPECS, tick_ns=10) as w:
        with pytest.raises(ValueError, match="expected tick 0"):
            w.write(ticks[1])
        with pytest.raises(ValueError, match="sources"):
            w.write(AlignedTick(0, 10, 10, {"a": []}))
        w.write(AlignedTick(0, 10, 10, {k: ([imu("a", 1)] if k == "a" else []) for k in SPECS}))
        bad = Sample("a", Modality.IMU, 12, 12, np.zeros(7), {})
        with pytest.raises(ValueError, match="shape changed"):
            w.write(AlignedTick(1, 20, 10, {k: ([bad] if k == "a" else []) for k in SPECS}))
    w.close()  # idempotent


def test_reader_rejects_foreign_files_and_write_ticks(tmp_path: Path) -> None:
    p = tmp_path / "foreign.h5"
    h5py.File(p, "w").close()
    with pytest.raises(ValueError, match="not a"):
        TickReader(p)
    write_ticks(tmp_path / "w.h5", _ticks(), SPECS, tick_ns=10)
    with TickReader(tmp_path / "w.h5") as r:
        assert len(r) == 3
