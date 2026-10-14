# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from senseai_ingest.cli import build_adapters, main
from senseai_ingest.schema import hash_ticks
from senseai_ingest.stream import ClockMode
from senseai_ingest.writer import TickReader


def _content_hash(path: Path) -> str:
    with TickReader(path) as r:
        return hash_ticks(r)


def test_replay_dsec_writes_deterministic_tick_file(
    dsec_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ticks.h5"
    digests = []
    for _ in range(2):
        rc = main(
            ["replay", "--dataset", "dsec", "--path", str(dsec_dir), "--tick-ms", "10",
             "--speed", "0", "--out", str(out), "--stats"]
        )  # fmt: skip
        assert rc == 0
        digests.append((_content_hash(out), hashlib.sha256(out.read_bytes()).hexdigest()))
    text = capsys.readouterr().out
    assert "sync jitter" in text and "events_left" in text and "rgb_right" in text
    assert digests[0] == digests[1]  # same ticks and byte-identical file
    outs = [out]
    with TickReader(outs[0]) as r:
        assert len(r) >= 100
        assert r.source_ids == ["events_left", "events_right", "rgb_left", "rgb_right"]
        attrs = r.attrs()
        assert attrs["tick_ns"] == 10_000_000 and attrs["config"]["dataset"] == "dsec"
        assert attrs["stats"]["sources"]["events_left"]["events"] == 200_000
        assert r.specs()["events_left"].offset_ns > 0


def test_replay_json_window_and_sources(dsec_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "replay",
            "--dataset",
            "dsec",
            "--path",
            str(dsec_dir),
            "--sources",
            "events_left",
            "--start-s",
            "0.1",
            "--duration-s",
            "0.2",
            "--json",
        ]
    )
    assert rc == 0
    st = json.loads(capsys.readouterr().out)
    assert list(st["sources"]) == ["events_left"]
    assert st["ticks"] == 20
    assert 30_000 < st["sources"]["events_left"]["events"] < 50_000


def test_replay_estimate_offsets_and_plain_output(
    m3ed_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "replay",
                "--dataset",
                "m3ed",
                "--path",
                str(m3ed_file),
                "--max-ticks",
                "5",
                "--estimate-offsets",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "5 ticks"


def test_replay_hdf5_needs_layout(m3ed_file: Path) -> None:
    with pytest.raises(SystemExit, match="layout"):
        main(["replay", "--dataset", "hdf5", "--path", str(m3ed_file)])
    assert (
        main(
            [
                "replay",
                "--dataset",
                "hdf5",
                "--layout",
                "m3ed",
                "--path",
                str(m3ed_file),
                "--max-ticks",
                "3",
            ]
        )
        == 0
    )


def test_replay_loop_guard_and_loop(mvsec_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="loop"):
        main(["replay", "--dataset", "mvsec", "--path", str(mvsec_file), "--loop"])
    assert (
        main(
            [
                "replay",
                "--dataset",
                "mvsec",
                "--path",
                str(mvsec_file),
                "--loop",
                "--max-ticks",
                "120",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ticks"] == 120


def _wav(path: Path, n: int = 4800, ch: int = 2, width: int = 2) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(48_000)
        w.writeframes(np.zeros(n * ch * width, np.uint8).tobytes())
    return path


def test_replay_wav_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _wav(tmp_path / "a.wav")
    _wav(tmp_path / "b.wav")
    ads = build_adapters("wav", tmp_path)
    assert [a.source_id for a in ads] == ["mic0", "mic1"]
    assert [a.source_id for a in build_adapters("wav", tmp_path, sources=["mic1"])] == ["mic1"]
    out = tmp_path / "t.h5"
    assert (
        main(["replay", "--dataset", "wav", "--path", str(tmp_path / "a.wav"), "--out", str(out)])
        == 0
    )
    with TickReader(out) as r:
        frames = sum(s.array.shape[0] for t in r for s in t.samples["mic0"])
    assert frames == 4800


def test_replay_no_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["replay", "--dataset", "wav", "--path", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        build_adapters("nope", tmp_path)


def test_inspect(dsec_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "--path", str(dsec_dir)]) == 0
    out = capsys.readouterr().out
    assert "layout: dsec" in out and "/events/t" in out and "rgb_left" in out
    (tmp_path / "junk.h5").write_bytes(b"not hdf5")
    assert main(["-v", "inspect", "--path", str(tmp_path)]) == 0
    assert "unknown" in capsys.readouterr().out
    assert main(["inspect", "--path", str(tmp_path / "missing")]) == 2


def test_clock_mode_flag_reaches_specs(m3ed_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "t.h5"
    main(
        [
            "replay",
            "--dataset",
            "m3ed",
            "--path",
            str(m3ed_file),
            "--max-ticks",
            "30",
            "--estimate-offsets",
            "--out",
            str(out),
        ]
    )
    with TickReader(out) as r:
        assert r.specs()["imu"].clock_mode is ClockMode.ESTIMATED
