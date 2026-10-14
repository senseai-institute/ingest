from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Modality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODALITY_UNSPECIFIED: _ClassVar[Modality]
    MODALITY_EVENT: _ClassVar[Modality]
    MODALITY_FRAME: _ClassVar[Modality]
    MODALITY_AUDIO: _ClassVar[Modality]
    MODALITY_IMU: _ClassVar[Modality]
    MODALITY_CONTACT: _ClassVar[Modality]
    MODALITY_CURRENT: _ClassVar[Modality]
    MODALITY_THERMAL: _ClassVar[Modality]
    MODALITY_GAS: _ClassVar[Modality]
    MODALITY_LIDAR: _ClassVar[Modality]
    MODALITY_OTHER: _ClassVar[Modality]

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[DType]
    DTYPE_BOOL: _ClassVar[DType]
    DTYPE_UINT8: _ClassVar[DType]
    DTYPE_INT8: _ClassVar[DType]
    DTYPE_UINT16: _ClassVar[DType]
    DTYPE_INT16: _ClassVar[DType]
    DTYPE_UINT32: _ClassVar[DType]
    DTYPE_INT32: _ClassVar[DType]
    DTYPE_UINT64: _ClassVar[DType]
    DTYPE_INT64: _ClassVar[DType]
    DTYPE_FLOAT16: _ClassVar[DType]
    DTYPE_FLOAT32: _ClassVar[DType]
    DTYPE_FLOAT64: _ClassVar[DType]

class ClockMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLOCK_MODE_UNSPECIFIED: _ClassVar[ClockMode]
    CLOCK_MODE_FIXED: _ClassVar[ClockMode]
    CLOCK_MODE_ESTIMATED: _ClassVar[ClockMode]
MODALITY_UNSPECIFIED: Modality
MODALITY_EVENT: Modality
MODALITY_FRAME: Modality
MODALITY_AUDIO: Modality
MODALITY_IMU: Modality
MODALITY_CONTACT: Modality
MODALITY_CURRENT: Modality
MODALITY_THERMAL: Modality
MODALITY_GAS: Modality
MODALITY_LIDAR: Modality
MODALITY_OTHER: Modality
DTYPE_UNSPECIFIED: DType
DTYPE_BOOL: DType
DTYPE_UINT8: DType
DTYPE_INT8: DType
DTYPE_UINT16: DType
DTYPE_INT16: DType
DTYPE_UINT32: DType
DTYPE_INT32: DType
DTYPE_UINT64: DType
DTYPE_INT64: DType
DTYPE_FLOAT16: DType
DTYPE_FLOAT32: DType
DTYPE_FLOAT64: DType
CLOCK_MODE_UNSPECIFIED: ClockMode
CLOCK_MODE_FIXED: ClockMode
CLOCK_MODE_ESTIMATED: ClockMode

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "buffer")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BUFFER_FIELD_NUMBER: _ClassVar[int]
    dtype: DType
    shape: _containers.RepeatedScalarFieldContainer[int]
    buffer: bytes
    def __init__(self, dtype: _Optional[_Union[DType, str]] = ..., shape: _Optional[_Iterable[int]] = ..., buffer: _Optional[bytes] = ...) -> None: ...

class TensorMap(_message.Message):
    __slots__ = ("entries",)
    class EntriesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.MessageMap[str, Tensor]
    def __init__(self, entries: _Optional[_Mapping[str, Tensor]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ("int_value", "double_value", "string_value", "bool_value", "bytes_value", "tensor_value", "list_value", "map_value")
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_VALUE_FIELD_NUMBER: _ClassVar[int]
    LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAP_VALUE_FIELD_NUMBER: _ClassVar[int]
    int_value: int
    double_value: float
    string_value: str
    bool_value: bool
    bytes_value: bytes
    tensor_value: Tensor
    list_value: ValueList
    map_value: ValueMap
    def __init__(self, int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., string_value: _Optional[str] = ..., bool_value: bool = ..., bytes_value: _Optional[bytes] = ..., tensor_value: _Optional[_Union[Tensor, _Mapping]] = ..., list_value: _Optional[_Union[ValueList, _Mapping]] = ..., map_value: _Optional[_Union[ValueMap, _Mapping]] = ...) -> None: ...

class ValueList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[Value]
    def __init__(self, values: _Optional[_Iterable[_Union[Value, _Mapping]]] = ...) -> None: ...

class ValueMap(_message.Message):
    __slots__ = ("entries",)
    class EntriesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.MessageMap[str, Value]
    def __init__(self, entries: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class Sample(_message.Message):
    __slots__ = ("source_id", "modality", "t_source_ns", "t_master_ns", "tensor", "tensors", "meta")
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODALITY_FIELD_NUMBER: _ClassVar[int]
    T_SOURCE_NS_FIELD_NUMBER: _ClassVar[int]
    T_MASTER_NS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    modality: Modality
    t_source_ns: int
    t_master_ns: int
    tensor: Tensor
    tensors: TensorMap
    meta: _containers.MessageMap[str, Value]
    def __init__(self, source_id: _Optional[str] = ..., modality: _Optional[_Union[Modality, str]] = ..., t_source_ns: _Optional[int] = ..., t_master_ns: _Optional[int] = ..., tensor: _Optional[_Union[Tensor, _Mapping]] = ..., tensors: _Optional[_Union[TensorMap, _Mapping]] = ..., meta: _Optional[_Mapping[str, Value]] = ...) -> None: ...

class SampleList(_message.Message):
    __slots__ = ("samples",)
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    samples: _containers.RepeatedCompositeFieldContainer[Sample]
    def __init__(self, samples: _Optional[_Iterable[_Union[Sample, _Mapping]]] = ...) -> None: ...

class AlignedTick(_message.Message):
    __slots__ = ("index", "t_master_ns", "tick_ns", "samples")
    class SamplesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: SampleList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[SampleList, _Mapping]] = ...) -> None: ...
    INDEX_FIELD_NUMBER: _ClassVar[int]
    T_MASTER_NS_FIELD_NUMBER: _ClassVar[int]
    TICK_NS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    index: int
    t_master_ns: int
    tick_ns: int
    samples: _containers.MessageMap[str, SampleList]
    def __init__(self, index: _Optional[int] = ..., t_master_ns: _Optional[int] = ..., tick_ns: _Optional[int] = ..., samples: _Optional[_Mapping[str, SampleList]] = ...) -> None: ...

class SourceSpec(_message.Message):
    __slots__ = ("source_id", "modality", "nominal_rate_hz", "meta", "clock_mode", "offset_ns", "offset_ref_ns", "drift_ppm", "jitter_ns", "offset_samples")
    class MetaEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODALITY_FIELD_NUMBER: _ClassVar[int]
    NOMINAL_RATE_HZ_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    CLOCK_MODE_FIELD_NUMBER: _ClassVar[int]
    OFFSET_NS_FIELD_NUMBER: _ClassVar[int]
    OFFSET_REF_NS_FIELD_NUMBER: _ClassVar[int]
    DRIFT_PPM_FIELD_NUMBER: _ClassVar[int]
    JITTER_NS_FIELD_NUMBER: _ClassVar[int]
    OFFSET_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    modality: Modality
    nominal_rate_hz: float
    meta: _containers.MessageMap[str, Value]
    clock_mode: ClockMode
    offset_ns: int
    offset_ref_ns: int
    drift_ppm: float
    jitter_ns: int
    offset_samples: int
    def __init__(self, source_id: _Optional[str] = ..., modality: _Optional[_Union[Modality, str]] = ..., nominal_rate_hz: _Optional[float] = ..., meta: _Optional[_Mapping[str, Value]] = ..., clock_mode: _Optional[_Union[ClockMode, str]] = ..., offset_ns: _Optional[int] = ..., offset_ref_ns: _Optional[int] = ..., drift_ppm: _Optional[float] = ..., jitter_ns: _Optional[int] = ..., offset_samples: _Optional[int] = ...) -> None: ...
