# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Tick files: a flat, indexed HDF5 layout for aligned ticks (see README §"Tick file format").

::

    /                       attrs: format, version, tick_ns, t0_ns, source_ids, config, stats
    /ticks/t_master_ns      int64  [K]
    /ticks/counts           uint32 [K, S]       samples per (tick, source); 0 = explicit empty
    /sources/<sid>/         attrs: modality, payload ("stacked" | "ragged" | "columns"), spec
        t_source_ns, t_master_ns  int64 [N]
        tick_index                uint64 [N]
        data            [N, ...]                stacked: constant-shape arrays
        data, data_ptr  [ΣM, ...], [N + 1]      ragged: arrays with a variable first axis
        data/<key>, data_ptr  [ΣM], [N + 1]     columns: dict payloads
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import h5py
import numpy as np

from senseai_ingest.schema import from_proto, pb, to_proto
from senseai_ingest.stream import AlignedTick, Modality, Sample, SourceSpec

__all__ = ["FORMAT", "VERSION", "TickReader", "TickWriter"]

FORMAT = "senseai-ingest/ticks"
VERSION = 1

_RAGGED = {Modality.AUDIO, Modality.LIDAR}


class _SourceSink:
    def __init__(self, grp: h5py.Group, spec: SourceSpec) -> None:
        self.grp = grp
        self.spec = spec
        self.payload: str | None = None
        self.t_src: list[int] = []
        self.t_mst: list[int] = []
        self.tick: list[int] = []
        self.parts: list[Any] = []
        self.ptr_end = 0
        grp.attrs["modality"] = spec.modality.value
        grp.attrs["spec"] = np.void(to_proto(spec).SerializeToString(deterministic=True))

    def add(self, s: Sample, k: int) -> None:
        if self.payload is None:
            self._init_layout(s)
        if self.payload == "stacked":
            arr = s.array
            if arr.shape != self.grp["data"].shape[1:]:
                raise ValueError(
                    f"{self.spec.source_id}: payload shape changed {self.grp['data'].shape[1:]}"
                    f" -> {arr.shape}; constant-shape modalities must not change shape"
                )
        self.t_src.append(s.t_source_ns)
        self.t_mst.append(s.t_master_ns)
        self.tick.append(k)
        self.parts.append(s.data)

    def _init_layout(self, s: Sample) -> None:
        g = self.grp
        kw: dict[str, Any] = {"track_times": False}
        for name in ("t_source_ns", "t_master_ns"):
            g.create_dataset(name, (0,), np.int64, maxshape=(None,), chunks=(4096,), **kw)
        g.create_dataset("tick_index", (0,), np.uint64, maxshape=(None,), chunks=(4096,), **kw)
        if isinstance(s.data, dict):
            self.payload = "columns"
            dg = g.create_group("data")
            for key, v in s.data.items():
                v1 = np.atleast_1d(v)
                dg.create_dataset(
                    key,
                    (0, *v1.shape[1:]),
                    v1.dtype,
                    maxshape=(None, *v1.shape[1:]),
                    chunks=(65536, *v1.shape[1:]),
                    **kw,
                )
        else:
            arr = np.asarray(s.data)
            if s.modality in _RAGGED or arr.ndim == 0:
                self.payload = "ragged"
                a1 = np.atleast_1d(arr)
                g.create_dataset(
                    "data",
                    (0, *a1.shape[1:]),
                    a1.dtype,
                    maxshape=(None, *a1.shape[1:]),
                    chunks=(max(1, 65536 // max(1, a1[0:1].nbytes)), *a1.shape[1:]),
                    **kw,
                )
            else:
                self.payload = "stacked"
                rows = max(1, (1 << 20) // max(1, arr.nbytes))
                g.create_dataset(
                    "data",
                    (0, *arr.shape),
                    arr.dtype,
                    maxshape=(None, *arr.shape),
                    chunks=(rows, *arr.shape),
                    **kw,
                )
        if self.payload != "stacked":
            g.create_dataset(
                "data_ptr", data=np.zeros(1, np.uint64), maxshape=(None,), chunks=(4096,), **kw
            )
        g.attrs["payload"] = self.payload

    @staticmethod
    def _append(ds: h5py.Dataset, arr: np.ndarray[Any, Any]) -> None:
        n = ds.shape[0]
        ds.resize(n + len(arr), axis=0)
        ds[n:] = arr

    def flush(self) -> None:
        if not self.t_src:
            return
        g = self.grp
        self._append(g["t_source_ns"], np.asarray(self.t_src, np.int64))
        self._append(g["t_master_ns"], np.asarray(self.t_mst, np.int64))
        self._append(g["tick_index"], np.asarray(self.tick, np.uint64))
        if self.payload == "stacked":
            self._append(g["data"], np.stack(self.parts))
        else:
            if self.payload == "columns":
                lens = [len(np.atleast_1d(next(iter(p.values())))) for p in self.parts]
                for key in g["data"]:
                    self._append(
                        g["data"][key], np.concatenate([np.atleast_1d(p[key]) for p in self.parts])
                    )
            else:
                arrs = [np.atleast_1d(p) for p in self.parts]
                lens = [len(a) for a in arrs]
                self._append(g["data"], np.concatenate(arrs))
            ends = self.ptr_end + np.cumsum(lens, dtype=np.uint64)
            self._append(g["data_ptr"], ends)
            self.ptr_end = int(ends[-1])
        self.t_src, self.t_mst, self.tick, self.parts = [], [], [], []


class TickWriter:
    """Append :class:`AlignedTick` objects to a tick file. Use as a context manager."""

    def __init__(
        self,
        path: str | Path,
        specs: Mapping[str, SourceSpec],
        *,
        tick_ns: int,
        config: Mapping[str, Any] | None = None,
        flush_every: int = 256,
    ) -> None:
        self.path = Path(path)
        self.source_ids: list[str] = sorted(specs)
        self.f = h5py.File(self.path, "w", track_order=False)
        self.f.attrs["format"] = FORMAT
        self.f.attrs["version"] = VERSION
        self.f.attrs["tick_ns"] = int(tick_ns)
        self.f.attrs["source_ids"] = json.dumps(self.source_ids)
        self.f.attrs["config"] = json.dumps(dict(config or {}), sort_keys=True, default=str)
        tg = self.f.create_group("ticks")
        kw: dict[str, Any] = {"track_times": False}
        tg.create_dataset("t_master_ns", (0,), np.int64, maxshape=(None,), chunks=(4096,), **kw)
        s = len(self.source_ids)
        tg.create_dataset(
            "counts", (0, s), np.uint32, maxshape=(None, s), chunks=(4096, max(1, s)), **kw
        )
        sg = self.f.create_group("sources")
        self._sinks = {
            sid: _SourceSink(sg.create_group(sid), specs[sid]) for sid in self.source_ids
        }
        self._t: list[int] = []
        self._counts: list[list[int]] = []
        self.flush_every = flush_every
        self.n_ticks = 0
        self.t0_ns: int | None = None

    def write(self, tick: AlignedTick) -> None:
        """Append one tick. Ticks must arrive in index order with no gaps."""
        if tick.index != self.n_ticks:
            raise ValueError(f"expected tick {self.n_ticks}, got {tick.index}")
        if set(tick.samples) != set(self.source_ids):
            raise ValueError(f"tick sources {sorted(tick.samples)} != file {self.source_ids}")
        if self.t0_ns is None:
            self.t0_ns = tick.window_start_ns
        row = []
        for sid in self.source_ids:
            lst = tick.samples[sid]
            row.append(len(lst))
            for s in lst:
                self._sinks[sid].add(s, tick.index)
        self._t.append(tick.t_master_ns)
        self._counts.append(row)
        self.n_ticks += 1
        if len(self._t) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered ticks to disk."""
        if self._t:
            tg = self.f["ticks"]
            _SourceSink._append(tg["t_master_ns"], np.asarray(self._t, np.int64))
            counts = np.asarray(self._counts, np.uint32).reshape(-1, len(self.source_ids))
            _SourceSink._append(tg["counts"], counts)
            self._t, self._counts = [], []
        for sink in self._sinks.values():
            sink.flush()

    def update_specs(self, specs: Mapping[str, SourceSpec]) -> None:
        """Overwrite stored specs (e.g. with final clock estimates at the end of a run)."""
        for sid, spec in specs.items():
            blob = to_proto(spec).SerializeToString(deterministic=True)
            self.f["sources"][sid].attrs["spec"] = np.void(blob)

    def close(self, stats: Mapping[str, Any] | None = None) -> None:
        """Flush, record ``t0_ns`` (and optional stats JSON), and close the file."""
        if not self.f.id.valid:
            return
        self.flush()
        self.f.attrs["t0_ns"] = int(self.t0_ns) if self.t0_ns is not None else 0
        if stats is not None:
            self.f.attrs["stats"] = json.dumps(dict(stats), sort_keys=True, default=str)
        self.f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class TickReader:
    """Read a tick file back as :class:`AlignedTick` objects (for downstream packages/tests)."""

    def __init__(self, path: str | Path) -> None:
        self.f = h5py.File(path, "r")
        if self.f.attrs.get("format") != FORMAT:
            self.f.close()
            raise ValueError(f"{path} is not a {FORMAT} file")
        self.tick_ns = int(self.f.attrs["tick_ns"])
        self.t0_ns = int(self.f.attrs["t0_ns"])
        self.source_ids: list[str] = json.loads(self.f.attrs["source_ids"])
        self.t_master_ns = np.asarray(self.f["ticks/t_master_ns"])
        self.counts = np.asarray(self.f["ticks/counts"])

    def __len__(self) -> int:
        return len(self.t_master_ns)

    def specs(self) -> dict[str, SourceSpec]:
        """Return the stored :class:`SourceSpec` of each source."""
        out = {}
        for sid in self.source_ids:
            raw = bytes(self.f["sources"][sid].attrs["spec"].tobytes())
            out[sid] = from_proto(pb.SourceSpec.FromString(raw))
        return out

    def _source(self, sid: str, lo: int, hi: int) -> list[Sample]:
        g = self.f["sources"][sid]
        if "t_master_ns" not in g:
            return []
        mod = Modality(g.attrs["modality"])
        payload = g.attrs["payload"]
        ts = g["t_source_ns"][lo:hi]
        tm = g["t_master_ns"][lo:hi]
        out = []
        if payload == "stacked":
            block = g["data"][lo:hi]
            for i in range(hi - lo):
                out.append(Sample(sid, mod, int(ts[i]), int(tm[i]), block[i], {}))
            return out
        ptr = g["data_ptr"][lo : hi + 1].astype(np.int64)
        for i in range(hi - lo):
            a, b = int(ptr[i]), int(ptr[i + 1])
            data: Any
            if payload == "columns":
                data = {k: g["data"][k][a:b] for k in sorted(g["data"])}
            else:
                data = g["data"][a:b]
            out.append(Sample(sid, mod, int(ts[i]), int(tm[i]), data, {}))
        return out

    def __iter__(self) -> Iterator[AlignedTick]:
        """Iterate all ticks (sample ``meta`` is not stored and comes back empty)."""
        starts = np.vstack(
            [
                np.zeros((1, len(self.source_ids)), np.int64),
                np.cumsum(self.counts, axis=0, dtype=np.int64),
            ]
        )
        for k in range(len(self)):
            samples = {
                sid: self._source(sid, int(starts[k, j]), int(starts[k + 1, j]))
                for j, sid in enumerate(self.source_ids)
            }
            yield AlignedTick(k, int(self.t_master_ns[k]), self.tick_ns, samples)

    def attrs(self) -> dict[str, Any]:
        """Return root attributes (``config``/``stats`` decoded from JSON)."""
        out: dict[str, Any] = {}
        for k, v in self.f.attrs.items():
            out[k] = json.loads(v) if k in ("config", "stats", "source_ids") else _py(v)
        return out

    def close(self) -> None:
        """Close the file."""
        self.f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _py(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def write_ticks(
    path: str | Path,
    ticks: Sequence[AlignedTick],
    specs: Mapping[str, SourceSpec],
    *,
    tick_ns: int,
) -> None:
    """Write a list of ticks in one call."""
    with TickWriter(path, specs, tick_ns=tick_ns) as w:
        for t in ticks:
            w.write(t)
