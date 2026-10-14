# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from senseai_ingest.testing import make_dsec_fixture, make_m3ed_fixture, make_mvsec_fixture


@pytest.fixture(scope="session")
def dsec_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dsec_fixture(tmp_path_factory.mktemp("dsec") / "zurich_city_00_a")


@pytest.fixture(scope="session")
def m3ed_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_m3ed_fixture(tmp_path_factory.mktemp("m3ed"))


@pytest.fixture(scope="session")
def mvsec_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_mvsec_fixture(tmp_path_factory.mktemp("mvsec"))
