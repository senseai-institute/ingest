# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Wire format (``sample.proto``, package ``senseai.ingest.v1``) and dataclass ↔ proto conversion.

Generated code (``sample_pb2.py``/``.pyi``) is committed; regenerate with ``make proto``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, overload

import numpy as np

from senseai_ingest.schema import sample_pb2 as pb
from senseai_ingest.stream import AlignedTick, ClockMode, Modality, Sample, SourceSpec

__all__ = [
    "from_proto",
    "hash_ticks",
    "modality_to_proto",
    "pb",
    "tensor_from_proto",
    "tensor_to_proto",
    "to_proto",
    "value_from_proto",
    "value_to_proto",
]

_MODALITY_TO_PB = {m: pb.Modality.Value(f"MODALITY_{m.name}") for m in Modality}
_MODALITY_FROM_PB = {v: k for k, v in _MODALITY_TO_PB.items()}
_CLOCK_TO_PB = {c: pb.ClockMode.Value(f"CLOCK_MODE_{c.name}") for c in ClockMode}
_CLOCK_FROM_PB = {v: k for k, v in _CLOCK_TO_PB.items()}

_DTYPE_TO_PB: dict[np.dtype[Any], int] = {
    np.dtype(np.bool_): pb.DTYPE_BOOL,
    np.dtype(np.uint8): pb.DTYPE_UINT8,
    np.dtype(np.int8): pb.DTYPE_INT8,
    np.dtype("<u2"): pb.DTYPE_UINT16,
    np.dtype("<i2"): pb.DTYPE_INT16,
    np.dtype("<u4"): pb.DTYPE_UINT32,
    np.dtype("<i4"): pb.DTYPE_INT32,
    np.dtype("<u8"): pb.DTYPE_UINT64,
    np.dtype("<i8"): pb.DTYPE_INT64,
    np.dtype("<f2"): pb.DTYPE_FLOAT16,
    np.dtype("<f4"): pb.DTYPE_FLOAT32,
    np.dtype("<f8"): pb.DTYPE_FLOAT64,
}
_DTYPE_FROM_PB = {v: k for k, v in _DTYPE_TO_PB.items()}


def modality_to_proto(m: Modality) -> int:
    """Return the ``senseai.ingest.v1.Modality`` enum number of ``m``."""
    return int(_MODALITY_TO_PB[m])


def tensor_to_proto(arr: np.ndarray[Any, Any]) -> pb.Tensor:
    """Encode an array as a little-endian, C-ordered :class:`Tensor`."""
    a = np.asarray(arr)
    le = a.dtype.newbyteorder("<") if a.dtype.byteorder == ">" else a.dtype
    try:
        code = _DTYPE_TO_PB[np.dtype(le)]
    except KeyError:
        raise TypeError(f"unsupported tensor dtype {a.dtype}") from None
    buf = np.ascontiguousarray(a, dtype=le).tobytes()
    return pb.Tensor(dtype=code, shape=list(a.shape), buffer=buf)  # type: ignore[arg-type]


def tensor_from_proto(t: pb.Tensor) -> np.ndarray[Any, Any]:
    """Decode a :class:`Tensor` (returns a writable copy)."""
    dt = _DTYPE_FROM_PB[t.dtype]
    return np.frombuffer(t.buffer, dtype=dt).reshape(tuple(t.shape)).copy()


def value_to_proto(v: Any) -> pb.Value:
    """Encode a metadata value (int, float, str, bool, bytes, ndarray, list/tuple, dict)."""
    if isinstance(v, bool | np.bool_):
        return pb.Value(bool_value=bool(v))
    if isinstance(v, int | np.integer):
        return pb.Value(int_value=int(v))
    if isinstance(v, float | np.floating):
        return pb.Value(double_value=float(v))
    if isinstance(v, str):
        return pb.Value(string_value=v)
    if isinstance(v, bytes):
        return pb.Value(bytes_value=v)
    if isinstance(v, np.ndarray):
        return pb.Value(tensor_value=tensor_to_proto(v))
    if isinstance(v, list | tuple):
        return pb.Value(list_value=pb.ValueList(values=[value_to_proto(x) for x in v]))
    if isinstance(v, Mapping):
        return pb.Value(
            map_value=pb.ValueMap(entries={str(k): value_to_proto(x) for k, x in v.items()})
        )
    raise TypeError(f"unsupported meta value type {type(v).__name__}")


def value_from_proto(v: pb.Value) -> Any:
    """Decode a metadata :class:`Value` (lists come back as ``list``)."""
    kind = v.WhichOneof("kind")
    if kind == "int_value":
        return v.int_value
    if kind == "double_value":
        return v.double_value
    if kind == "string_value":
        return v.string_value
    if kind == "bool_value":
        return v.bool_value
    if kind == "bytes_value":
        return v.bytes_value
    if kind == "tensor_value":
        return tensor_from_proto(v.tensor_value)
    if kind == "list_value":
        return [value_from_proto(x) for x in v.list_value.values]
    if kind == "map_value":
        return {k: value_from_proto(x) for k, x in v.map_value.entries.items()}
    return None


def _meta_to_proto(meta: Mapping[str, Any]) -> dict[str, pb.Value]:
    return {str(k): value_to_proto(v) for k, v in meta.items()}


def _meta_from_proto(meta: Mapping[str, pb.Value]) -> dict[str, Any]:
    return {k: value_from_proto(v) for k, v in meta.items()}


def _sample_to_proto(s: Sample) -> pb.Sample:
    msg = pb.Sample(
        source_id=s.source_id,
        modality=_MODALITY_TO_PB[s.modality],  # type: ignore[arg-type]
        t_source_ns=s.t_source_ns,
        t_master_ns=s.t_master_ns,
        meta=_meta_to_proto(s.meta),
    )
    if isinstance(s.data, np.ndarray):
        msg.tensor.CopyFrom(tensor_to_proto(s.data))
    else:
        msg.tensors.CopyFrom(
            pb.TensorMap(entries={k: tensor_to_proto(np.asarray(v)) for k, v in s.data.items()})
        )
    return msg


def _sample_from_proto(m: pb.Sample) -> Sample:
    data: np.ndarray[Any, Any] | dict[str, np.ndarray[Any, Any]]
    if m.WhichOneof("data") == "tensors":
        data = {k: tensor_from_proto(v) for k, v in m.tensors.entries.items()}
    else:
        data = tensor_from_proto(m.tensor)
    return Sample(
        source_id=m.source_id,
        modality=_MODALITY_FROM_PB[m.modality],
        t_source_ns=m.t_source_ns,
        t_master_ns=m.t_master_ns,
        data=data,
        meta=_meta_from_proto(m.meta),
    )


def _tick_to_proto(t: AlignedTick) -> pb.AlignedTick:
    return pb.AlignedTick(
        index=t.index,
        t_master_ns=t.t_master_ns,
        tick_ns=t.tick_ns,
        samples={
            sid: pb.SampleList(samples=[_sample_to_proto(s) for s in lst])
            for sid, lst in t.samples.items()
        },
    )


def _tick_from_proto(m: pb.AlignedTick) -> AlignedTick:
    return AlignedTick(
        index=m.index,
        t_master_ns=m.t_master_ns,
        tick_ns=m.tick_ns,
        samples={
            sid: [_sample_from_proto(s) for s in lst.samples]
            for sid, lst in sorted(m.samples.items())
        },
    )


def _spec_to_proto(s: SourceSpec) -> pb.SourceSpec:
    return pb.SourceSpec(
        source_id=s.source_id,
        modality=_MODALITY_TO_PB[s.modality],  # type: ignore[arg-type]
        nominal_rate_hz=s.nominal_rate_hz,
        meta=_meta_to_proto(s.meta),
        clock_mode=_CLOCK_TO_PB[s.clock_mode],  # type: ignore[arg-type]
        offset_ns=s.offset_ns,
        offset_ref_ns=s.offset_ref_ns,
        drift_ppm=s.drift_ppm,
        jitter_ns=s.jitter_ns,
        offset_samples=s.offset_samples,
    )


def _spec_from_proto(m: pb.SourceSpec) -> SourceSpec:
    return SourceSpec(
        source_id=m.source_id,
        modality=_MODALITY_FROM_PB[m.modality],
        nominal_rate_hz=m.nominal_rate_hz,
        meta=_meta_from_proto(m.meta),
        clock_mode=_CLOCK_FROM_PB.get(m.clock_mode, ClockMode.FIXED),
        offset_ns=m.offset_ns,
        offset_ref_ns=m.offset_ref_ns,
        drift_ppm=m.drift_ppm,
        jitter_ns=m.jitter_ns,
        offset_samples=m.offset_samples,
    )


@overload
def to_proto(obj: Sample) -> pb.Sample: ...
@overload
def to_proto(obj: AlignedTick) -> pb.AlignedTick: ...
@overload
def to_proto(obj: SourceSpec) -> pb.SourceSpec: ...
def to_proto(obj: Sample | AlignedTick | SourceSpec) -> Any:
    """Convert a :class:`Sample`, :class:`AlignedTick` or :class:`SourceSpec` to protobuf."""
    if isinstance(obj, Sample):
        return _sample_to_proto(obj)
    if isinstance(obj, AlignedTick):
        return _tick_to_proto(obj)
    if isinstance(obj, SourceSpec):
        return _spec_to_proto(obj)
    raise TypeError(f"cannot convert {type(obj).__name__} to proto")


@overload
def from_proto(msg: pb.Sample) -> Sample: ...
@overload
def from_proto(msg: pb.AlignedTick) -> AlignedTick: ...
@overload
def from_proto(msg: pb.SourceSpec) -> SourceSpec: ...
def from_proto(msg: pb.Sample | pb.AlignedTick | pb.SourceSpec) -> Any:
    """Convert a protobuf message back to its dataclass."""
    if isinstance(msg, pb.Sample):
        return _sample_from_proto(msg)
    if isinstance(msg, pb.AlignedTick):
        return _tick_from_proto(msg)
    if isinstance(msg, pb.SourceSpec):
        return _spec_from_proto(msg)
    raise TypeError(f"cannot convert {type(msg).__name__} from proto")


def hash_ticks(ticks: Iterable[AlignedTick]) -> str:
    """SHA-256 over the deterministic serialization of a tick sequence."""
    h = hashlib.sha256()
    for t in ticks:
        b = to_proto(t).SerializeToString(deterministic=True)
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()
