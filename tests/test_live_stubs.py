# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import struct

import numpy as np
import pytest

from senseai_ingest.adapters.base import AdapterNotAvailable
from senseai_ingest.adapters.live import require
from senseai_ingest.adapters.live.can_bus import CanAdapter, CanConfig, CanSignal
from senseai_ingest.adapters.live.dvs import DvsAdapter, DvsConfig
from senseai_ingest.adapters.live.mic_array import MicArrayAdapter, MicArrayConfig
from senseai_ingest.adapters.live.serial_imu import LineParser, SerialConfig, SerialSensorHub
from senseai_ingest.adapters.live.v4l2 import V4l2Adapter, V4l2Config
from senseai_ingest.stream import ClockMode, Modality, Stream


def test_require_names_the_extra() -> None:
    with pytest.raises(AdapterNotAvailable, match=r"senseai-ingest\[dvs\]"):
        require("definitely_not_a_module_xyz", "dvs")
    require("json", "none")


@pytest.mark.parametrize(
    "adapter",
    [
        DvsAdapter(),
        DvsAdapter(DvsConfig(sensor="dvxplorer")),
        V4l2Adapter(),
        MicArrayAdapter(),
        *SerialSensorHub().adapters(),
        *CanAdapter.for_config(CanConfig(signals=[CanSignal(0x101, "cur1")])),
    ],
)
def test_stub_interface(adapter: DvsAdapter) -> None:
    spec = adapter.spec()
    assert adapter.is_live and spec.clock_mode is ClockMode.ESTIMATED
    assert spec.source_id
    with pytest.raises(AdapterNotAvailable) as exc:
        list(adapter.read())
    assert "pip install" in str(exc.value) or "Gate 1" in str(exc.value)
    adapter.close()


def test_open_checks_backend() -> None:
    with pytest.raises(AdapterNotAvailable, match="metavision_core"):
        DvsAdapter().open()


def test_dvs_config() -> None:
    spec = DvsAdapter(DvsConfig(sensor="evk4", batch_us=500)).spec()
    assert spec.meta["width"] == 1280 and spec.nominal_rate_hz == 2000
    assert DvsAdapter(DvsConfig(roi=(0, 0, 100, 50))).spec().meta["height"] == 50
    assert DvsConfig.from_mapping({"sensor": "davis346", "roi": [1, 2, 3, 4]}).roi == (1, 2, 3, 4)
    assert DvsConfig(sensor="dvxplorer").backend_name == "dv"
    for bad in ({"sensor": "nope"}, {"batch_us": 0}):
        with pytest.raises(ValueError):
            DvsConfig(**bad)


def test_v4l2_and_mic_config() -> None:
    assert V4l2Config.from_mapping({"width": 320, "fps": 60, "junk": 1}).width == 320
    assert V4l2Adapter(V4l2Config(encoding="mono8")).spec().meta["channels"] == 1
    for bad in ({"encoding": "yuv"}, {"fps": 0}):
        with pytest.raises(ValueError):
            V4l2Config(**bad)
    spec = MicArrayAdapter(MicArrayConfig(geometry="respeaker6", sample_rate=48_000)).spec()
    assert spec.meta["channel_positions"].shape == (6, 3) and spec.nominal_rate_hz == 46.875
    assert MicArrayConfig(geometry=None).n_channels == 1
    assert MicArrayConfig(channels=2).n_channels == 2
    for bad in ({"geometry": "x"}, {"dtype": "int8"}, {"block": 0}):
        with pytest.raises(ValueError):
            MicArrayConfig(**bad)


def test_serial_parser_records() -> None:
    p = LineParser(SerialConfig(joints=("hip", "knee")))
    imu = p.parse("IMU,1500,0.1,0.2,9.8,0.01,0.02,0.03\n", 99)
    assert imu is not None and imu.modality is Modality.IMU
    assert imu.t_source_ns == 1_500_000 and imu.t_master_ns == 99
    assert imu.array.tolist() == [0.1, 0.2, 9.8, 0.01, 0.02, 0.03]
    bump = p.parse(b"BUMP,2000,0x5", 1)
    assert bump is not None and bump.array.dtype == np.uint32 and int(bump.array) == 5
    assert bump.meta == {"kind": "bumper"}
    cur = p.parse("cur,3000,1.5,-0.25", 1)
    assert cur is not None and cur.array.dtype == np.float32 and cur.source_id == "cur0"


@pytest.mark.parametrize(
    "line",
    [
        "IMU,1,2,3",  # too few values
        "IMU,x,0,0,0,0,0,0",  # bad timestamp
        "IMU,-5,0,0,0,0,0,0",  # negative timestamp
        "IMU,1,nan,0,0,0,0,0",  # non-finite
        "BUMP,1",  # missing mask
        "BUMP,1,0x1ffffffff",  # > 32 bit
        "CUR,1,1.0",  # joint count mismatch (2 joints configured)
        "CUR,1,inf,0",
        "TEMP,1,25",  # unknown tag
        b"\xff\xfe",  # not ascii
    ],
)
def test_serial_parser_never_raises(line: str | bytes) -> None:
    p = LineParser(SerialConfig(joints=("hip", "knee")))
    assert p.parse(line, 0) is None
    assert sum(v for k, v in p.metrics.snapshot().items() if k.endswith("parse_errors")) == 1


def test_serial_parse_lines_and_estimated_stream() -> None:
    p = LineParser()
    lines = ["# header", "", "IMU,0,0,0,0,0,0,0", "garbage", "BUMP,10,1", "CUR,20,1,2,3"]
    out = list(p.parse_lines(lines))
    assert [s.source_id for s in out] == ["imu0", "bump0", "cur0"]
    assert [s.t_master_ns for s in out] == [0, 10_000, 20_000]
    assert p.metrics.get("serial.parse_errors") == 1
    # host arrivals 5 ms after device time: the Stream's estimator recovers the offset
    imu_lines = [f"IMU,{i * 1000},0,0,9.8,0,0,0" for i in range(300)]
    arrivals = [i * 1_000_000 + 5_000_000 + (i % 3) * 10_000 for i in range(300)]
    hub = SerialSensorHub()
    st = Stream(hub.adapters(["IMU"])[0])
    for s in LineParser().parse_lines(imu_lines, arrivals):
        st.push(s)
    assert abs(st.spec().offset_ns - 5_010_000) <= 10_000
    with pytest.raises(ValueError):
        hub.adapters(["XYZ"])


def test_can_signal_decode() -> None:
    sig = CanSignal(0x200, "cur0", fmt="<I3h", scale=1e-3, t_field=0)
    s = sig.decode(struct.pack("<I3h", 1234, 1000, -500, 0), 77)
    assert s.t_source_ns == 1_234_000 and s.t_master_ns == 77
    assert s.array.dtype == np.float32
    np.testing.assert_allclose(s.array, [1.0, -0.5, 0.0])
    imu = CanSignal(0x201, "imu1", modality=Modality.IMU, fmt="<6h", scale=0.01)
    assert imu.decode(bytes(12), 5).array.dtype == np.float64
    with pytest.raises(struct.error):
        sig.decode(b"\x00", 0)
    with pytest.raises(ValueError, match="t_field"):
        CanSignal(1, "x", fmt="<h", t_field=3)
    with pytest.raises(ValueError, match="CAN FD"):
        CanSignal(1, "x", fmt="<100h")
    with pytest.raises(ValueError, match="duplicate"):
        CanConfig(signals=[CanSignal(1, "a"), CanSignal(1, "b")])
    with pytest.raises(ValueError, match="no CAN signal"):
        CanAdapter(CanConfig(), "zz")
    assert CanAdapter(CanConfig(signals=[sig]), "cur0").spec().meta["arbitration_id"] == 0x200
