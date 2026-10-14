# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest

SCHEMA = Path(__file__).resolve().parents[1] / "src" / "senseai_ingest" / "schema"


def test_generated_code_matches_proto(tmp_path: Path) -> None:
    grpc_tools = pytest.importorskip("grpc_tools.protoc")
    rc = grpc_tools.main(
        ["protoc", f"-I{SCHEMA}", f"--python_out={tmp_path}", f"--pyi_out={tmp_path}",
         str(SCHEMA / "sample.proto")]
    )  # fmt: skip
    assert rc == 0
    for name in ("sample_pb2.py", "sample_pb2.pyi"):
        assert (tmp_path / name).read_text() == (SCHEMA / name).read_text(), (
            f"{name} is stale: run `make proto`"
        )


def test_downstream_imports_are_light() -> None:
    code = (
        "import sys, senseai_ingest.stream, senseai_ingest.schema\n"
        "heavy = {'h5py', 'structlog', 'yaml', 'rclpy', 'rosbags', 'torch'} & set(sys.modules)\n"
        "print(sorted(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_lazy_top_level_exports() -> None:
    import senseai_ingest

    assert senseai_ingest.Aligner.__name__ == "Aligner"
    assert senseai_ingest.ThrottlePolicy.DROP.value == "drop"
    with pytest.raises(AttributeError):
        _ = senseai_ingest.nope
