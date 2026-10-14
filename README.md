# senseai-ingest

Sense-agnostic ingress for the SenseAI perception layer: it gets bytes off sensors
(or recordings), puts them on one clock, and hands them to downstream tokenizers
as one uniform stream of timestamp-aligned ticks.

`senseai-ingest` is Apache-2.0 and has **no model code**. Tokenization lives in
`senseai-tokens`, and evaluation lives in `senseai-bench`.

```
 sensors / recordings          senseai-ingest                          downstream
 ┌───────────┐   Sample   ┌───────┐  ┌────────┐  ┌──────────┐  AlignedTick  ┌────────────────┐
 │ Adapter   │──────────▶│ Clock │─▶│ Buffer │─▶│          │──────────────▶│ senseai-tokens │
 │ (replay / │            │ sync  │  │ +      │  │ Aligner  │  every tick   │ reflex head    │
 │  live)    │  × N       └───────┘  │Throttle│  │          │               │ ROS 2 topics   │
 └───────────┘                       └────────┘  └──────────┘               └────────────────┘
```

**Status: Gate 1 (replay only).** You can replay DSEC, M3ED, MVSEC, generic HDF5 and
multichannel WAV recordings. The live adapters (DVS, V4L2, mic array, serial
IMU, CAN) already have their final interface, config parsing and `spec()`, but
`read()` raises `AdapterNotAvailable` for now. The serial line protocol parser
is complete and tested.

---

## Install

```bash
pip install senseai-ingest              # core: numpy, h5py, protobuf, structlog, pyyaml
pip install senseai-ingest[audio]       # + soundfile (WAV replay), sounddevice (live mic)
pip install senseai-ingest[dvs]         # + live event-camera SDK shims (Metavision / dv-processing)
pip install senseai-ingest[ros2]        # + rosbags (rosbag2 replay); rclpy comes from your ROS 2 install
pip install senseai-ingest[dev]         # + pytest, ruff, mypy, grpcio-tools
```

Python ≥ 3.11. Development uses `uv`:

```bash
uv sync --all-extras
make test lint type
```

`import senseai_ingest` never imports ROS, a vendor SDK or torch.

## 30-second replay

```bash
senseai-ingest inspect --path /data/dsec/zurich_city_04_a       # print detected layout
senseai-ingest replay  --dataset dsec --path /data/dsec/zurich_city_04_a \
                       --tick-ms 10 --speed 0 --out /tmp/ticks.h5 --stats
```

The `--stats` summary looks like this (illustrative numbers):

```
ticks: 41 213   wall: 6.8 s   (6 060 ticks/s)
sync jitter     p50 0.00 ms   p99 0.00 ms
source          modality  samples/s   dropped  late  high_water
events_left     event         100.0         0     0          3
rgb_left        frame          20.0         0     0          1
imu             imu          1000.0         0     0         11
```

You can do the same thing from Python:

```python
import asyncio
from senseai_ingest import Aligner, AlignerConfig
from senseai_ingest.adapters.replay.dsec import DsecReplay

async def main() -> None:
    adapters = DsecReplay.from_sequence("/data/dsec/zurich_city_04_a", speed=0.0)
    async with Aligner(adapters, AlignerConfig(tick_ns=10_000_000)) as aligner:
        async for tick in aligner:
            ev = tick.samples["events_left"]   # list[Sample], possibly empty
            print(tick.index, tick.t_master_ns, {k: len(v) for k, v in tick.samples.items()})

asyncio.run(main())
```

`examples/replay_dsec.py` runs the same loop against the synthetic test fixture.
That means it works without downloading DSEC.

---

## Core types (`senseai_ingest.stream`)

```python
class Modality(str, Enum):
    EVENT, FRAME, AUDIO, IMU, CONTACT, CURRENT, THERMAL, GAS, LIDAR, OTHER

@dataclass(frozen=True, slots=True)
class Sample:
    source_id: str                       # "dvs0", "rgb0", "mic0"
    modality: Modality
    t_source_ns: int                     # device timebase
    t_master_ns: int                     # master timebase, after clock alignment
    data: np.ndarray | dict[str, np.ndarray]
    meta: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class AlignedTick:
    index: int                           # 0-based, contiguous
    t_master_ns: int                     # window end (exclusive)
    tick_ns: int
    samples: Mapping[str, list[Sample]]  # every source present; [] if no data
```

- **`Stream`** is an async iterator of `Sample`s from one adapter. It applies clock
  alignment, buffering and throttling.
- **`Aligner`** takes N streams and yields one `AlignedTick` per `tick_ns`.

### Timestamp convention (causal)

`t_*_ns` is the capture time of the **last** datum in a sample's payload. This is
the earliest moment the whole payload could have existed. Because of this rule,
a tick never contains information from its future.

- For an audio block of N frames at rate `sr`, the first frame was captured at
  `t − (N−1)/sr`.
- For a frame, the timestamp is the device's frame timestamp, as end of exposure
  where the dataset provides it. `meta["exposure_ns"]` holds the exposure time
  when it is known.

### Tick window

Tick `k` covers the master-clock window `[T_k − tick_ns, T_k)`, where
`T_k = t0 + (k+1)·tick_ns`. `t0` is the earliest `t_master_ns` across all
sources, or `start` if you give one.

The Aligner **splits EVENT batches at tick boundaries**, so every event in a tick
lies inside that tick's window. It never splits other modalities.

## Modality payloads

These shapes are a contract. Adapters must produce them, and `senseai-tokens` may
rely on them.

| Modality  | `data` type | shape / keys | dtype | units / notes | required `meta` |
|-----------|-------------|--------------|-------|---------------|-----------------|
| `EVENT`   | dict | `t` (N,), `x` (N,), `y` (N,), `p` (N,) | `t` int64, `x`,`y` uint16, `p` uint8 | `t` = **master-clock ns** per event, sorted; `p` ∈ {0 = OFF, 1 = ON} | `width`, `height` |
| `FRAME`   | ndarray | (H, W, C) | uint8 \| uint16 | `C` = 1 (mono / raw Bayer) or 3 (RGB order) | `encoding` (`mono8`, `rgb8`, `bayer_rggb8`, `mono16`, …) |
| `AUDIO`   | ndarray | (N, C) | int16 \| float32 | float32 lies in [−1, 1]; frames × channels | `sample_rate` |
| `IMU`     | ndarray | (6,) | float64 | `[ax, ay, az]` m/s², `[gx, gy, gz]` rad/s, sensor frame | `frame_id` |
| `CONTACT` | ndarray | (K,) or () | float32 \| uint32 | `kind="force"`: N per taxel. `kind="bumper"`: 0-d bitmask | `kind` |
| `CURRENT` | ndarray | (J,) | float32 | A per joint | `joints` (names) |
| `THERMAL` | ndarray | (H, W) | float32 | K, or raw counts | `unit` (`K` \| `raw`) |
| `GAS`     | dict | `{name: ()}` | float64 | one 0-d array per channel | `units` (dict name→unit) |
| `LIDAR`   | ndarray | (N, 4) | float32 | `x, y, z` m, intensity | `frame_id` |
| `OTHER`   | either | free | free | free | — |

Optional `meta` keys with fixed meanings are `intrinsics` (3×3 float64),
`distortion`, `channel_positions` (C×3 m) and `exposure_ns`.

---

## Clock model

There is exactly one **master clock**, measured in int64 nanoseconds.

| Mode   | Master clock | Where a sample's `t_master_ns` comes from |
|--------|--------------|-------------------------------------------|
| live   | `time.monotonic_ns()` on the host | from `t_source_ns` through the per-source estimate below |
| replay | the replay cursor, in dataset time | from the dataset's timebase through a fixed transform in the layout (`ClockMode.FIXED`) |

**Per-source offset (live, `ClockMode.ESTIMATED`).** For each source, the clock
collects `d_i = t_master_arrival_i − t_source_i` over the first K samples
(K = 200 by default) and sets

```
offset_ns = median(d_i)                       # robust to scheduling spikes
t_master  = t_source + offset_ns + drift_ppm·1e-6·(t_source − offset_ref_ns)
```

The median of `d_i` absorbs the constant part of transport latency into the
offset. That's deliberate: the master time then sits on the device's own
timestamps, and host-side jitter doesn't leak into it. Every `drift_window_s`
(10 s by default), the clock re-estimates the offset on a fresh window and fits
`drift_ppm` from successive estimates. `jitter_ns` is the p99 residual
`|arrival − predicted|`. `SourceSpec` records `offset_ns`, `offset_ref_ns`,
`drift_ppm`, `jitter_ns` and `offset_samples`.

**Replay (`ClockMode.FIXED`).** Each dataset layout declares how to convert its
timestamps to ns. Examples:

- DSEC: `t_us · 1000 + t_offset_us · 1000`
- M3ED: `t_us · 1000`
- MVSEC: ROS time → ns

No estimation happens, so replay is exact. To exercise the estimator on recorded
data, set `--estimate-offsets`, which feeds the cursor time as the arrival time.

**When a tick closes.**

- **Replay** uses a watermark. Tick `T_k` closes when every source has produced a
  sample with `t_master_ns ≥ T_k` or is exhausted. Thread and async scheduling
  never affect the output, so **replay is bit-for-bit deterministic**.
- **Live** uses a deadline. Tick `T_k` closes at master time `T_k + grace_ns`.
  A sample that shows up after its tick closed is counted in `late_samples`. It
  is then dropped (`late_policy="drop"`, the default) or attached to the next
  open tick (`late_policy="attach_next"`), keeping its true `t_master_ns`.

**Sync jitter metric.** For each tick, `sync_jitter_ns` is the largest alignment
error of any sample in that tick: `|t_master_ns − t_master_true|`. `t_master_true`
is known exactly in replay and in the benchmark's synthetic sources. In live mode
it is approximated by the clock residual. The CLI and the benchmark log p50 and
p99. The Gate-1 acceptance target is p99 < 1 ms on 4-source DSEC replay.

## Buffers and throttling

- **Buffer.** Each source gets its own ring buffer, bounded by `max_samples` and
  `max_bytes`. When it's full it drops the oldest sample. Its counters are
  `dropped` and `high_water`.
- **Throttle.** You can cap samples/s per source, and events/s for `EVENT`
  sources. There are three policies:
  - `drop`: keep the first samples in each budget window.
  - `decimate`: keep every n-th sample or event, chosen deterministically from
    its sequence number.
  - `accumulate`: merge events into fixed `window_ns` batches.

  All three depend only on sample timestamps and sequence numbers, never on wall
  time, so they're deterministic in replay. In live mode the throttle signals
  backpressure to the adapter before it starts dropping.

Metrics are plain counters with no Prometheus. Read them with
`senseai_ingest.metrics.snapshot() -> dict[str, int | float]`, where keys look
like `"<source_id>.dropped"`. Logging uses `structlog`.

---

## Wire format (`schema/sample.proto`)

`src/senseai_ingest/schema/sample.proto` (package `senseai.ingest.v1`) is the
contract for `senseai-tokens`, the ROS 2 bridge and any non-Python consumer. The
generated `sample_pb2.py` / `.pyi` files are committed. Convert with
`senseai_ingest.schema.to_proto(obj)` and `from_proto(msg)`.

| Python (`senseai_ingest.stream`) | Protobuf (`senseai.ingest.v1`) |
|---|---|
| `Modality.EVENT` (`"event"`) | `Modality.MODALITY_EVENT`, and likewise for each member |
| `Sample.source_id / t_source_ns / t_master_ns` | same-named fields |
| `Sample.data: np.ndarray` | `oneof data { Tensor tensor }` |
| `Sample.data: dict[str, np.ndarray]` | `oneof data { TensorMap tensors }` |
| `np.ndarray` | `Tensor{dtype, shape, buffer}`: little-endian, C-order |
| `Sample.meta: Mapping[str, Any]` | `map<string, Value>`: int, float, str, bool, bytes, ndarray, list and dict map to `int_value` … `map_value` |
| `AlignedTick.samples: Mapping[str, list[Sample]]` | `map<string, SampleList>`: the key exists even when the list is empty |
| `SourceSpec` | `SourceSpec`: same field names |

`meta` uses a typed `Value` rather than `google.protobuf.Struct`. Struct stores
every number as a double, which corrupts int64 ns timestamps above 2⁵³.

Determinism hashes serialize with `SerializeToString(deterministic=True)` to fix
map ordering.

## Tick file format (`replay --out ticks.h5`)

The tick file uses a flat, indexed layout rather than one group per tick, which
keeps 10⁵+ ticks cheap to write and to slice.

```
/                         attrs: format="senseai-ingest/ticks", version=1, tick_ns, t0_ns,
                                 source_ids (ordered), config (JSON)
/ticks/t_master_ns        int64  [K]
/ticks/counts             uint32 [K, S]   samples per (tick, source); 0 = explicit empty
/sources/<sid>/           attrs: modality, spec (serialized SourceSpec bytes)
    t_source_ns           int64  [N]
    t_master_ns           int64  [N]
    tick_index            uint64 [N]      non-decreasing
    data                  [N, ...]        ndarray payloads with a constant shape (stacked)
    data + data_ptr       [ΣM, ...], [N+1] ndarray payloads with a variable first axis (LIDAR)
    data/<key> + data_ptr [ΣM], [N+1]     dict payloads (EVENT columns, GAS)
```

The samples of source `s` in tick `k` are the rows where `tick_index == k`. That
range is contiguous: find it with `np.searchsorted` or by summing
`/ticks/counts[:k, s]`.

## Adding an adapter (< 50 lines)

```python
# Copyright 2026 SenseAI Institute — Apache-2.0
from collections.abc import Iterator
import numpy as np
from senseai_ingest.adapters.base import Adapter
from senseai_ingest.stream import Modality, Sample, SourceSpec, ClockMode

class CsvThermalReplay(Adapter):
    """Replays `t_us,<H*W floats>` lines as THERMAL samples."""

    def __init__(self, path: str, source_id: str = "thermal0", shape: tuple[int, int] = (24, 32)) -> None:
        self._path, self._sid, self._shape = path, source_id, shape
        self._f = None

    def spec(self) -> SourceSpec:
        return SourceSpec(source_id=self._sid, modality=Modality.THERMAL, nominal_rate_hz=8.0,
                          meta={"unit": "K", "height": self._shape[0], "width": self._shape[1]},
                          clock_mode=ClockMode.FIXED)

    def open(self) -> None:
        self._f = open(self._path)

    def read(self) -> Iterator[Sample]:
        assert self._f is not None
        for line in self._f:
            t_us, *vals = line.split(",")
            t_ns = int(t_us) * 1000
            img = np.asarray(vals, dtype=np.float32).reshape(self._shape)
            yield Sample(self._sid, Modality.THERMAL, t_ns, t_ns, img, {"unit": "K"})

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
```

Adapters fill in `t_master_ns` as follows:

- **Replay adapters** set it from the layout transform.
- **Live adapters** set it to the host arrival time (`clock.now_ns()`). The
  `Stream` then replaces it using the estimated offset.

For an HDF5 dataset you usually only need a YAML layout file, not a class. See
`adapters/replay/layouts/*.yaml`.

## Serial line protocol (`adapters/live/serial_imu.py`)

The protocol is ASCII, one record per `\n`-terminated line, with `t_us` on the
device clock:

```
IMU,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>     m/s², rad/s       -> Modality.IMU
BUMP,<t_us>,<mask>                            uint32 bitmask    -> Modality.CONTACT (kind="bumper")
CUR,<t_us>,<j0>,<j1>,...,<jN>                 A                 -> Modality.CURRENT
```

The parser counts malformed lines in `<source_id>.parse_errors` and skips them.
It never raises on bad input.

## ROS 2 (optional)

`senseai_ingest.ros2.node` wraps an `Aligner` in an `rclpy` node:

- It publishes `senseai_ingest/msg/AlignedTick` on `/senseai/ticks`.
- It publishes per-source samples on `/senseai/<source_id>`.
- It offers `launch/replay.launch.py`.

The `.msg` definitions mirror the proto. All ROS imports are guarded.

## Deliberately out of scope

- Tokenization, feature extraction, learned models, and torch → `senseai-tokens`
- Benchmarks of models or tokenizers → `senseai-bench`. The aligner benchmark
  lives here.
- Calibration (intrinsics/extrinsics estimation). Calibration is carried in
  `meta` when the dataset provides it, but never computed.
- Image debayering, rectification and resampling, and audio resampling
- Hardware time sync (PTP, trigger lines). Hardware-synced sources are declared
  `ClockMode.FIXED`.
- Live device drivers beyond stubs in Gate 1
- Prometheus or OpenTelemetry export (the counter dict is the extension point)

## License

Apache-2.0. Copyright 2026 SenseAI Institute.
