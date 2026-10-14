# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Live device adapters. Gate 1: interfaces, config and ``spec()`` are final; capture is stubbed.

``read()`` raises :class:`~senseai_ingest.adapters.base.AdapterNotAvailable` with an install hint
when the backend package is missing, and a "not implemented in Gate 1" message otherwise.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import NoReturn

from senseai_ingest.adapters.base import Adapter, AdapterNotAvailable
from senseai_ingest.stream import Sample

__all__ = ["LiveAdapter", "require"]


def require(module: str, extra: str) -> None:
    """Import ``module`` or raise :class:`AdapterNotAvailable` naming the pip extra."""
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise AdapterNotAvailable(
            f"{module} is not installed: pip install senseai-ingest[{extra}]"
        ) from exc


class LiveAdapter(Adapter):
    """Base for live stubs: ``open()`` checks the backend, ``read()`` is not implemented yet."""

    is_live = True
    #: (importable module, pip extra) providing the backend.
    backend: tuple[str, str] = ("", "")

    def open(self) -> None:
        """Check that the backend package is importable."""
        module, extra = self.backend
        if module:
            require(module, extra)

    def read(self) -> Iterator[Sample]:
        """Live capture is not part of Gate 1."""
        self._unavailable()

    def close(self) -> None:
        """Nothing to release in the stub."""

    def _unavailable(self) -> NoReturn:
        module, extra = self.backend
        if module:
            require(module, extra)
        raise AdapterNotAvailable(
            f"{type(self).__name__}: live capture is not implemented in Gate 1 (replay only)"
        )
