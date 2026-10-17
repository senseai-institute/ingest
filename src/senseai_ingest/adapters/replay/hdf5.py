# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Generic HDF5 replay driven by a YAML *layout* that maps dataset paths → modality/fields.

Layout format::

    name: my_dataset
    detect: ["**/events.h5"]        # globs (relative to the root dir) identifying this layout
    sources:
      - id: events_left
        modality: event
        file: "**/events/left/events.h5"   # glob relative to root (ignored if root is a file)
        events:                            # EVENT sources: one field per column
          t: /events/t                     # dataset path, or {path: /ds, column: 2}
          x: /events/x
          y: /events/y
          p: /events/p
        polarity: identity                 # identity (0/1 stored) | sign (-1/+1 or 0/1 → 0/1)
        batch_ns: 1000000                  # events are batched into fixed master-time windows
        time:
          unit: us                         # ns | us | ms | s
          offset: {dataset: /t_offset}     # int, {dataset: path} or {attr: "path@name"}; same unit
        meta: {width: 640, height: 480}
      - id: imu
        modality: imu
        file: data.h5
        t: /imu/ts                         # tensor sources: one row per sample
        data: [/imu/accel, /imu/omega]     # several datasets are concatenated on the last axis
        time: {unit: us}
        rate_hz: 1000                      # optional; estimated from timestamps otherwise
        optional: true                     # skip silently if the file/dataset is missing
      - id: lidar
        modality: lidar
        file: data.h5
        t: /lidar/ts                       # one timestamp per scan
        data: /lidar/points                # (sum of points, 4): all scans concatenated ...
        ptr: /lidar/ptr                    # ... (n_scans + 1,) offsets: scan i = ptr[i]:ptr[i+1]
        time: {unit: us}

``t_source_ns = raw * unit`` and ``t_master_ns = (raw + offset) * unit``; the clock mode is
``FIXED`` and ``SourceSpec.offset_ns`` records the dataset offset.

Rows whose raw timestamp is the int64 minimum (``-2**63``, the sentinel M3ED uses for invalid
IMU rows) or is NaN are skipped and counted in ``<source_id>.invalid_rows``.
"""

from __future__ import annotations

import glob
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import structlog
import yaml

from senseai_ingest.adapters.base import ReplayAdapter
from senseai_ingest.stream import EVENT_DTYPES, ClockMode, Modality, Sample, SourceSpec

__all__ = [
    "FieldRef",
    "Hdf5Replay",
    "Layout",
    "LayoutError",
    "SourceLayout",
    "builtin_layouts",
    "describe_h5",
    "detect_layout",
    "load_layout",
    "open_layout",
]

log = structlog.get_logger(__name__)

_UNIT_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
_INT64_MIN = np.iinfo(np.int64).min


def _valid_rows(raw_t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Boolean mask of rows with a usable timestamp (not the int64-min sentinel, not NaN)."""
    if np.issubdtype(raw_t.dtype, np.integer):
        return np.asarray(raw_t != _INT64_MIN)
    return np.asarray(np.isfinite(raw_t))


class LayoutError(ValueError):
    """Invalid layout file, or the data does not match the layout."""


@dataclass(frozen=True, slots=True)
class FieldRef:
    """A dataset path, optionally selecting one column of a 2-D dataset."""

    path: str
    column: int | None = None

    @classmethod
    def parse(cls, v: Any, where: str) -> FieldRef:
        """Parse ``"/path"`` or ``{path: ..., column: ...}``."""
        if isinstance(v, str):
            return cls(v)
        if isinstance(v, Mapping) and isinstance(v.get("path"), str):
            col = v.get("column")
            return cls(v["path"], int(col) if col is not None else None)
        raise LayoutError(f"{where}: expected a dataset path or {{path, column}}, got {v!r}")

    def read(self, f: h5py.File, sl: slice) -> np.ndarray[Any, Any]:
        """Read rows ``sl`` (and the column, if any)."""
        ds = f[self.path]
        if self.column is None:
            return np.asarray(ds[sl])
        return np.asarray(ds[sl, self.column])

    def length(self, f: h5py.File) -> int:
        """Return the number of rows."""
        return int(f[self.path].shape[0])

    def at(self, f: h5py.File, i: int) -> Any:
        """Read a single element (row ``i``)."""
        ds = f[self.path]
        return ds[i] if self.column is None else ds[i, self.column]


@dataclass(frozen=True, slots=True)
class SourceLayout:
    """One source in a :class:`Layout`."""

    id: str
    modality: Modality
    file: str | None = None
    t: FieldRef | None = None
    data: tuple[FieldRef, ...] = ()
    ptr: FieldRef | None = None
    events: Mapping[str, FieldRef] = field(default_factory=dict)
    polarity: str = "identity"
    batch_ns: int = 1_000_000
    unit: str = "us"
    offset: Any = 0
    rate_hz: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    optional: bool = False

    @classmethod
    def parse(cls, d: Mapping[str, Any], where: str) -> SourceLayout:
        """Validate and build from a YAML mapping."""
        if not isinstance(d, Mapping) or "id" not in d or "modality" not in d:
            raise LayoutError(f"{where}: each source needs 'id' and 'modality'")
        try:
            modality = Modality(d["modality"])
        except ValueError:
            raise LayoutError(f"{where}: unknown modality {d['modality']!r}") from None
        time = d.get("time", {}) or {}
        unit = time.get("unit", "us")
        if unit not in _UNIT_NS:
            raise LayoutError(f"{where}: time.unit must be one of {sorted(_UNIT_NS)}")
        polarity = d.get("polarity", "identity")
        if polarity not in ("identity", "sign"):
            raise LayoutError(f"{where}: polarity must be 'identity' or 'sign'")
        events: dict[str, FieldRef] = {}
        t: FieldRef | None = None
        ptr: FieldRef | None = None
        data: tuple[FieldRef, ...] = ()
        if modality is Modality.EVENT:
            ev = d.get("events")
            if not isinstance(ev, Mapping) or set(ev) < {"t", "x", "y", "p"}:
                raise LayoutError(f"{where}: event sources need events: {{t, x, y, p}}")
            events = {k: FieldRef.parse(ev[k], f"{where}.events.{k}") for k in ("t", "x", "y", "p")}
        else:
            if "t" not in d or "data" not in d:
                raise LayoutError(f"{where}: tensor sources need 't' and 'data'")
            t = FieldRef.parse(d["t"], f"{where}.t")
            raw = d["data"] if isinstance(d["data"], list) else [d["data"]]
            data = tuple(FieldRef.parse(x, f"{where}.data") for x in raw)
            if "ptr" in d:
                if len(data) != 1:
                    raise LayoutError(f"{where}: 'ptr' (ragged) sources take exactly one 'data'")
                ptr = FieldRef.parse(d["ptr"], f"{where}.ptr")
        return cls(
            id=str(d["id"]),
            modality=modality,
            file=d.get("file"),
            t=t,
            data=data,
            ptr=ptr,
            events=events,
            polarity=polarity,
            batch_ns=int(d.get("batch_ns", 1_000_000)),
            unit=unit,
            offset=time.get("offset", 0),
            rate_hz=float(d["rate_hz"]) if d.get("rate_hz") is not None else None,
            meta=dict(d.get("meta", {}) or {}),
            optional=bool(d.get("optional", False)),
        )


@dataclass(frozen=True, slots=True)
class Layout:
    """A parsed layout file."""

    name: str
    sources: tuple[SourceLayout, ...]
    detect: tuple[str, ...] = ()
    description: str = ""

    @classmethod
    def parse(cls, d: Any, where: str = "layout") -> Layout:
        """Validate and build from a YAML document."""
        if not isinstance(d, Mapping) or not isinstance(d.get("sources"), list):
            raise LayoutError(f"{where}: expected a mapping with a 'sources' list")
        srcs = tuple(
            SourceLayout.parse(s, f"{where}.sources[{i}]") for i, s in enumerate(d["sources"])
        )
        ids = [s.id for s in srcs]
        if len(set(ids)) != len(ids):
            raise LayoutError(f"{where}: duplicate source ids {ids}")
        return cls(
            name=str(d.get("name", where)),
            sources=srcs,
            detect=tuple(d.get("detect", []) or []),
            description=str(d.get("description", "")),
        )


def builtin_layouts() -> dict[str, Path]:
    """Return ``{name: path}`` for layouts shipped with the package."""
    root = resources.files("senseai_ingest.adapters.replay") / "layouts"
    return {Path(str(p)).stem: Path(str(p)) for p in root.iterdir() if str(p).endswith(".yaml")}


def load_layout(layout: str | Path | Mapping[str, Any] | Layout) -> Layout:
    """Load a layout by builtin name, YAML path, already-parsed mapping, or pass one through."""
    if isinstance(layout, Layout):
        return layout
    if isinstance(layout, Mapping):
        return Layout.parse(layout)
    builtins = builtin_layouts()
    path = builtins.get(str(layout), Path(layout))
    if not path.exists():
        raise LayoutError(f"no such layout {layout!r} (builtins: {sorted(builtins)})")
    with open(path) as fh:
        return Layout.parse(yaml.safe_load(fh), where=path.name)


def _resolve_file(root: Path, pattern: str | None) -> Path | None:
    if root.is_file():
        return root
    if pattern is None:
        raise LayoutError(f"{root} is a directory; the layout source needs a 'file' glob")
    hits = sorted(Path(p) for p in glob.glob(str(root / pattern), recursive=True))
    if not hits:
        return None
    if len(hits) > 1:
        raise LayoutError(f"glob {pattern!r} under {root} matched {len(hits)} files: {hits[:3]}…")
    return hits[0]


class Hdf5Replay(ReplayAdapter):
    """Replays one source of an HDF5 recording according to a :class:`SourceLayout`."""

    def __init__(
        self,
        file: str | Path,
        source: SourceLayout,
        *,
        chunk_rows: int = 1 << 20,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.file = Path(file)
        self.layout = source
        self.chunk_rows = int(chunk_rows)
        self._f: h5py.File | None = None
        self._unit = _UNIT_NS[source.unit]
        self._offset_raw: int | float | None = None
        self._spec: SourceSpec | None = None
        #: rows skipped because their timestamp was the int64-min sentinel / NaN
        self.invalid_rows = 0

    # -- helpers ----------------------------------------------------------------------------
    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.file, "r")
        return self._f

    def _tref(self) -> FieldRef:
        lay = self.layout
        ref = lay.events["t"] if lay.modality is Modality.EVENT else lay.t
        assert ref is not None
        return ref

    def _offset(self) -> int | float:
        if self._offset_raw is None:
            off = self.layout.offset
            f = self._file()
            if isinstance(off, int | float):
                val: Any = off
            elif isinstance(off, Mapping) and "dataset" in off:
                val = f[off["dataset"]][()]
            elif isinstance(off, Mapping) and "attr" in off:
                path, _, name = str(off["attr"]).rpartition("@")
                val = f[path or "/"].attrs[name]
            else:
                raise LayoutError(f"{self.layout.id}: bad time.offset {off!r}")
            arr = np.asarray(val).reshape(-1)[0]
            self._offset_raw = int(arr) if np.issubdtype(arr.dtype, np.integer) else float(arr)
        return self._offset_raw

    def to_ns(self, raw: np.ndarray[Any, Any], *, master: bool = True) -> np.ndarray[Any, Any]:
        """Convert raw dataset timestamps to int64 ns (master: including the dataset offset)."""
        off = self._offset() if master else 0
        if np.issubdtype(raw.dtype, np.integer) and isinstance(off, int):
            return (raw.astype(np.int64) + np.int64(off)) * np.int64(self._unit)
        with np.errstate(invalid="ignore"):  # NaN sentinels are masked out by the callers
            return np.round((raw.astype(np.float64) + off) * self._unit).astype(np.int64)

    def _t_at(self, i: int) -> int:
        raw = np.asarray(self._tref().at(self._file(), i)).reshape(1)
        return int(self.to_ns(raw)[0])

    def _n(self) -> int:
        return self._tref().length(self._file())

    def _bisect(self, t_ns: int) -> int:
        """First row index with master time ``>= t_ns`` (O(log n) single-element reads)."""
        lo, hi = 0, self._n()
        while lo < hi:
            mid = (lo + hi) // 2
            if self._t_at(mid) < t_ns:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # -- Adapter API ------------------------------------------------------------------------
    def spec(self) -> SourceSpec:
        """Describe the source (rate estimated from timestamps when not given)."""
        if self._spec is not None:
            return self._spec
        lay = self.layout
        rate = lay.rate_hz
        meta: dict[str, Any] = {"file": str(self.file), **lay.meta}
        if rate is None:
            n = self._n()
            if n > 1:
                first, last = self.time_range()
                span = (last - first) * 1e-9
                rate = (n - 1) / span if span > 0 else 0.0
                if lay.modality is Modality.EVENT:
                    meta.setdefault("event_rate_hz", rate)
                    rate = 1e9 / lay.batch_ns
            else:
                rate = 0.0
        off = self._offset()
        self._spec = SourceSpec(
            source_id=lay.id,
            modality=lay.modality,
            nominal_rate_hz=float(rate),
            meta=meta,
            clock_mode=ClockMode.FIXED,
            offset_ns=round(off * self._unit),
        )
        return self._spec

    def open(self) -> None:
        """Open the HDF5 file and validate that the layout's datasets exist."""
        f = self._file()
        refs = list(self.layout.events.values()) + list(self.layout.data)
        for extra in (self.layout.t, self.layout.ptr):
            if extra is not None:
                refs.append(extra)
        missing = [r.path for r in refs if r.path not in f]
        if missing:
            raise LayoutError(f"{self.file}: datasets missing for {self.layout.id}: {missing}")

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._f is not None:
            self._f.close()
            self._f = None

    def time_range(self) -> tuple[int, int]:
        """First and last master timestamps."""
        n = self._n()
        if n == 0:
            return (0, 0)
        return self._t_at(0), self._t_at(n - 1)

    def _row_range(self) -> tuple[int, int]:
        n = self._n()
        lo = self._bisect(self.start_ns) if self.start_ns is not None else 0
        hi = self._bisect(self.end_ns) if self.end_ns is not None else n
        return lo, hi

    def _read_once(self) -> Iterator[Sample]:
        if self.layout.modality is Modality.EVENT:
            yield from self._read_events()
        elif self.layout.ptr is not None:
            yield from self._read_ragged()
        else:
            yield from self._read_tensors()

    def _read_events(self) -> Iterator[Sample]:
        lay = self.layout
        f = self._file()
        lo, hi = self._row_range()
        off_ns = self.spec().offset_ns
        carry: dict[str, np.ndarray[Any, Any]] | None = None
        meta: dict[str, Any] = {}
        for a in range(lo, hi, self.chunk_rows):
            sl = slice(a, min(hi, a + self.chunk_rows))
            t = self.to_ns(lay.events["t"].read(f, sl))
            p_raw = lay.events["p"].read(f, sl)
            p = (p_raw > 0) if lay.polarity == "sign" else p_raw
            cols = {
                "t": t,
                "x": lay.events["x"].read(f, sl).astype(EVENT_DTYPES["x"]),
                "y": lay.events["y"].read(f, sl).astype(EVENT_DTYPES["y"]),
                "p": np.asarray(p).astype(EVENT_DTYPES["p"]),
            }
            if carry is not None:
                cols = {k: np.concatenate([carry[k], cols[k]]) for k in cols}
            win = cols["t"] // lay.batch_ns
            cuts = np.flatnonzero(np.diff(win)) + 1
            bounds = [0, *cuts.tolist(), len(win)]
            for i in range(len(bounds) - 2):  # all complete batches
                b0, b1 = bounds[i], bounds[i + 1]
                yield self._event_sample({k: v[b0:b1] for k, v in cols.items()}, off_ns, meta)
            b0 = bounds[-2]
            carry = {k: v[b0:] for k, v in cols.items()}
        if carry is not None and len(carry["t"]):
            yield self._event_sample(carry, off_ns, meta)

    def _event_sample(
        self, cols: dict[str, np.ndarray[Any, Any]], off_ns: int, meta: Mapping[str, Any]
    ) -> Sample:
        t_last = int(cols["t"][-1])
        return Sample(self.layout.id, Modality.EVENT, t_last - off_ns, t_last, cols, meta)

    def _read_ragged(self) -> Iterator[Sample]:
        lay = self.layout
        f = self._file()
        assert lay.t is not None and lay.ptr is not None
        lo, hi = self._row_range()
        n = self._n()
        if lay.ptr.length(f) != n + 1:
            raise LayoutError(f"{lay.id}: ptr must have n_samples + 1 = {n + 1} entries")
        meta: dict[str, Any] = {}
        for a in range(lo, hi, self.chunk_rows):
            b = min(hi, a + self.chunk_rows)
            raw_t = lay.t.read(f, slice(a, b))
            t_master = self.to_ns(raw_t)
            t_source = self.to_ns(raw_t, master=False)
            ok = _valid_rows(raw_t)
            if not ok.all():
                self.invalid_rows += int((~ok).sum())
            ptr = np.asarray(lay.ptr.read(f, slice(a, b + 1))).astype(np.int64)
            block = lay.data[0].read(f, slice(int(ptr[0]), int(ptr[-1])))
            for i in np.flatnonzero(ok).tolist():
                rows = block[int(ptr[i] - ptr[0]) : int(ptr[i + 1] - ptr[0])]
                yield Sample(lay.id, lay.modality, int(t_source[i]), int(t_master[i]), rows, meta)

    def _read_tensors(self) -> Iterator[Sample]:
        lay = self.layout
        f = self._file()
        assert lay.t is not None
        lo, hi = self._row_range()
        row_bytes = sum(
            int(np.prod(f[r.path].shape[1:], dtype=np.int64)) * f[r.path].dtype.itemsize
            for r in lay.data
        )
        step = max(1, min(self.chunk_rows, (64 << 20) // max(1, row_bytes)))
        meta: dict[str, Any] = {}
        for a in range(lo, hi, step):
            sl = slice(a, min(hi, a + step))
            raw_t = lay.t.read(f, sl)
            ok = _valid_rows(raw_t)
            if not ok.all():
                self.invalid_rows += int((~ok).sum())
            t_master = self.to_ns(raw_t)
            t_source = self.to_ns(raw_t, master=False)
            parts = [r.read(f, sl) for r in lay.data]
            block = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
            if lay.modality is Modality.FRAME and block.ndim == 3:
                block = block[..., None]  # (N, H, W) -> (N, H, W, 1)
            for i in np.flatnonzero(ok):
                yield Sample(
                    lay.id, lay.modality, int(t_source[i]), int(t_master[i]), block[i], meta
                )


def open_layout(
    root: str | Path,
    layout: str | Path | Mapping[str, Any] | Layout,
    *,
    sources: Sequence[str] | None = None,
    **kw: Any,
) -> list[Hdf5Replay]:
    """Build one :class:`Hdf5Replay` per layout source found under ``root``.

    Missing *optional* sources are skipped; missing required ones raise :class:`LayoutError`.
    ``kw`` is forwarded to each adapter (``start_ns``, ``end_ns``, ``speed``, ``loop``...).
    """
    lay = load_layout(layout)
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(rootp)
    wanted = set(sources) if sources is not None else None
    if wanted is not None:
        unknown = wanted - {s.id for s in lay.sources}
        if unknown:
            raise LayoutError(f"layout {lay.name!r} has no sources {sorted(unknown)}")
    out: list[Hdf5Replay] = []
    for src in lay.sources:
        if wanted is not None and src.id not in wanted:
            continue
        path = _resolve_file(rootp, src.file)
        ok = path is not None
        if ok:
            assert path is not None
            with h5py.File(path, "r") as f:
                refs = [*src.events.values(), *src.data]
                refs += [r for r in (src.t, src.ptr) if r is not None]
                ok = all(r.path in f for r in refs)
        if not ok:
            if src.optional and wanted is None:
                log.info("layout.source_missing", layout=lay.name, source=src.id)
                continue
            raise LayoutError(f"{lay.name}: source {src.id!r} not found under {rootp}")
        assert path is not None
        out.append(Hdf5Replay(path, src, **kw))
    return out


def describe_h5(path: str | Path) -> list[dict[str, Any]]:
    """List every dataset in an HDF5 file: path, shape, dtype and attributes."""
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                rows.append(
                    {
                        "path": "/" + name,
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "attrs": {k: _plain(v) for k, v in obj.attrs.items()},
                    }
                )

        f.visititems(visit)
    return rows


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 16 else f"<array {v.shape} {v.dtype}>"
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def detect_layout(root: str | Path) -> str | None:
    """Return the name of the first builtin layout whose sources resolve under ``root``."""
    rootp = Path(root)
    for name, path in sorted(builtin_layouts().items()):
        try:
            lay = load_layout(path)
            detected = all(glob.glob(str(rootp / d), recursive=True) for d in lay.detect)
            if rootp.is_dir() and lay.detect and not detected:
                continue
            if open_layout(rootp, lay):
                return name
        except (LayoutError, OSError, ValueError):
            continue
    return None
