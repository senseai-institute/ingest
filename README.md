# senseai-ingest

[![CI](https://github.com/senseai-institute/ingest/actions/workflows/ci.yml/badge.svg)](https://github.com/senseai-institute/ingest/actions/workflows/ci.yml)

Sense-agnostic ingress for the SenseAI perception layer: it gets bytes off sensors
(or recordings), puts them on one clock, and hands them to downstream tokenizers
as one uniform stream of timestamp-aligned ticks.

`senseai-ingest` is Apache-2.0 and has **no model code**. Tokenization lives in
`senseai-tokens`, and evaluation lives in `senseai-bench`. Downstream packages
only need `senseai_ingest.stream` (types) and `senseai_ingest.schema` (wire format).

```
 sensors / recordings          senseai-ingest                          downstream
 ┌───────────┐   Sample   ┌───────┐  ┌────────┐  ┌──────────┐  AlignedTick  ┌────────────────┐
 │ Adapter   │──────────▶│ Clock │─▶│Throttle│─▶│          │──────────────▶│ senseai-tokens │
 │ (replay / │            │ sync  │  │ + ring │  │ Aligner  │  every tick   │ reflex head    │
 │  live)    │  × N       └───────┘  │ buffer │  │          │               │ ROS 2 topics   │
 └───────────┘                       └────────┘  └──────────┘               └────────────────┘
```

**Status: Gate 1 (replay only).** You can replay DSEC, M3ED, MVSEC, generic HDF5 (via a YAML
layout), rosbag1/rosbag2 and multichannel WAV recordings. The live adapters (DVS, V4L2, mic
array, serial IMU/bumper/current, CAN) already have their final interface, config parsing
and `spec()`, but `read()` raises `AdapterNotAvailable` for now. The serial line protocol
parser and the CAN frame decoder are complete and tested.

---

## Install

```bash
pip install senseai-ingest              # core: numpy, h5py, protobuf, structlog, pyyaml
pip install senseai-ingest[images]      # + pillow (DSEC PNG frames)
pip install senseai-ingest[audio]       # + soundfile (non-PCM WAV), sounddevice (live mic)
pip install senseai-ingest[ros2]        # + rosbags (rosbag1/2 replay without ROS); rclpy comes from ROS
pip install senseai-ingest[live]        # + pyserial, python-can, opencv (live adapters, later gates)
pip install senseai-ingest[dvs]         # marker extra: Metavision / dv-processing come from vendor channels
pip install senseai-ingest[dev]         # + pytest, ruff, mypy, grpcio-tools, pillow, rosbags
```

Python ≥ 3.11. Development uses `uv`:

```bash
uv sync --extra dev
make test lint type        # pytest + coverage (≥ 85 %), ruff, mypy --strict
make bench                 # benchmarks/bench_aligner.py
make proto                 # regenerate schema/sample_pb2.py(i) from sample.proto
```

`import senseai_ingest` never imports ROS, a vendor SDK or torch.

## 30-second replay

```bash
senseai-ingest inspect --path /data/dsec/zurich_city_04_a       # print detected layout
senseai-ingest replay  --dataset dsec --path /data/dsec/zurich_city_04_a \
                       --tick-ms 10 --speed 0 --out /tmp/ticks.h5 --stats
```

This is real `--stats` output on a 5 s synthetic DSEC-layout sequence (2 × 200 k ev/s, 2 × 20 Hz
frames), from `senseai_ingest.testing.make_dsec_fixture`:

```
ticks: 500   wall: 0.66 s   (758 ticks/s)
sync jitter     p50 0.000 ms   p99 0.000 ms
staleness       p50 0.008 ms   p99 5.003 ms
source          modality    samples/s      events  dropped  late  high_water
events_left     event          1042.8     1000000        0     0           1
events_right    event          1045.6     1000000        0     0           1
rgb_left        frame            20.0           0        0     0           1
rgb_right       frame            20.0           0        0     0           1
```

Other options:

- `--sources a,b` replays only those sources.
- `--start-s` / `--duration-s` select a time window.
- `--loop --max-ticks N` loops the recording.
- `--late-policy attach_next` attaches late samples to the next tick instead of dropping them.
- `--estimate-offsets` runs the clock estimator on recorded data.
- `--json` prints the stats as JSON.
- `--dataset hdf5 --layout my.yaml` replays any HDF5 file described by a layout.

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
        print(aligner.stats())

asyncio.run(main())
```

`Aligner` is also a plain iterator (`for tick in aligner`), and `run_ticks(aligner)` collects
the ticks into a list. `examples/replay_dsec.py` runs the same loop on a generated fixture,
or on a real sequence if you pass its path.

---

## Core types (`senseai_ingest.stream`)

```python
class Modality(StrEnum):
    EVENT, FRAME, AUDIO, IMU, CONTACT, CURRENT, THERMAL, GAS, LIDAR, OTHER

@dataclass(frozen=True, slots=True)
class Sample:
    source_id: str                       # "dvs0", "rgb0", "mic0"
    modality: Modality
    t_source_ns: int                     # device timebase
    t_master_ns: int                     # master timebase, after clock alignment
    data: np.ndarray | dict[str, np.ndarray]
    meta: Mapping[str, Any]
    # typed accessors: .array (ndarray payloads), .columns (dict payloads), .nbytes

@dataclass(frozen=True, slots=True)
class AlignedTick:
    index: int                           # 0-based, contiguous
    t_master_ns: int                     # window end (exclusive)
    tick_ns: int
    samples: Mapping[str, list[Sample]]  # every source present; [] if no data

@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id, modality, nominal_rate_hz, meta, clock_mode,
    offset_ns, offset_ref_ns, drift_ppm, jitter_ns, offset_samples
```

- **`Stream`** runs one adapter's pipeline: clock alignment, then throttling, then a
  bounded ring buffer. It's iterable both sync and async.
- **`Aligner`** takes N adapters or streams and yields one `AlignedTick` per `tick_ns`.

### Timestamp convention (causal)

`t_*_ns` is the capture time of the **last** datum in a sample's payload. This is the earliest
moment the whole payload could have existed. Because of this rule, a tick never contains
information from its future.

- For an audio block of N frames at rate `sr`, the first frame was captured at
  `t − (N−1)/sr`.
- For a frame, the timestamp is the device's frame timestamp.

### Tick window

Tick `k` covers the master-clock window `[T_k − tick_ns, T_k)`, where
`T_k = t0 + (k+1)·tick_ns`. `t0` is the earliest `t_master_ns` across all sources, or
`AlignerConfig.start_ns` if you give one.

The Aligner **splits EVENT batches at tick boundaries**, so every event in a tick lies inside
that tick's window. It never splits other modalities. Within a tick, each source's samples are
in non-decreasing `t_master_ns` order.

## Modality payloads

These shapes are a contract. Adapters must produce them, and `senseai-tokens` may rely on them.

| Modality  | `data` type | shape / keys | dtype | units / notes | required `meta` |
|-----------|-------------|--------------|-------|---------------|-----------------|
| `EVENT`   | dict | `t` (N,), `x` (N,), `y` (N,), `p` (N,) | `t` int64, `x`,`y` uint16, `p` uint8 | `t` = **master-clock ns** per event, sorted; `p` ∈ {0 = OFF, 1 = ON} | `width`, `height` (spec) |
| `FRAME`   | ndarray | (H, W, C) | uint8 \| uint16 | `C` = 1 (mono / raw Bayer) or 3 (RGB order) | `encoding` (`mono8`, `rgb8`, `bayer_rggb8`, `mono16`, …) |
| `AUDIO`   | ndarray | (N, C) | int16 \| float32 | float32 lies in [−1, 1]; frames × channels; last block may be short | `sample_rate` |
| `IMU`     | ndarray | (6,) | float64 | `[ax, ay, az]` m/s², `[gx, gy, gz]` rad/s, sensor frame | `frame_id` |
| `CONTACT` | ndarray | (K,) or () | float32 \| uint32 | `kind="force"`: N per taxel. `kind="bumper"`: 0-d bitmask | `kind` |
| `CURRENT` | ndarray | (J,) | float32 | A per joint | `joints` (names) |
| `THERMAL` | ndarray | (H, W) | float32 | K, or raw counts | `unit` (`K` \| `raw`) |
| `GAS`     | dict | `{name: ()}` | float64 | one 0-d array per channel | `units` (dict name→unit) |
| `LIDAR`   | ndarray | (N, 4) | float32 | `x, y, z` m, intensity | `frame_id` |
| `OTHER`   | either | free | free | free | — |

"Required meta" means it lives in the `SourceSpec.meta` and/or the per-sample `meta`. Optional
keys with fixed meanings are `intrinsics` (3×3 float64), `distortion`, `channel_positions`
(C×3 m) and `exposure_ns`.

---

## Clock model

There is exactly one **master clock**, measured in int64 nanoseconds.

| Mode   | Master clock | Where a sample's `t_master_ns` comes from |
|--------|--------------|-------------------------------------------|
| live   | `time.monotonic_ns()` on the host | from `t_source_ns` through the per-source estimate below |
| replay | the replay cursor, in dataset time | from the dataset's timebase through a fixed transform in the layout (`ClockMode.FIXED`) |

**Per-source offset (`ClockMode.ESTIMATED`).** A live adapter stamps each sample's
`t_master_ns` with the host *arrival* time. The `Stream` collects
`d_i = t_arrival_i − t_source_i` over the first K samples (K = 200 by default) and sets

```
offset_ns = median(d_i)                       # robust to scheduling spikes
t_master  = t_source + offset_ns + drift_ppm·1e-6·(t_source − offset_ref_ns)
```

The clock then rewrites `t_master_ns` from the model. For EVENT sources, it also shifts the
`t` column, which adapters provide in device time.

Every `drift_window_ns` (10 s by default), the clock re-estimates the offset on a fresh window
and fits `drift_ppm` from successive estimates. `jitter_ns` is the p99 residual
`|arrival − predicted|`. `SourceSpec` records `offset_ns`, `offset_ref_ns`, `drift_ppm`,
`jitter_ns` and `offset_samples`.

> **Limitation (by design):** the median absorbs the *constant* part of transport latency into
> the offset. Arrival times can't distinguish a constant latency from a clock offset. Removing
> that bias requires hardware sync (PTP or trigger lines, which you'd declare as
> `ClockMode.FIXED`) or a measured per-device latency, declared as
> `AlignerConfig(latency_ns={"rgb0": 4_000_000})`, `SourceSpec.meta["latency_ns"]` or
> `--latency-ms rgb0=4.0`. The clock then subtracts it from every arrival time before
> estimating, and records it in the spec.

**Replay (`ClockMode.FIXED`).** Each dataset layout declares how to convert its timestamps to ns.
Examples:

- DSEC: `(t + t_offset) µs`
- M3ED: µs
- MVSEC: ROS-time seconds
- rosbag: `header.stamp`

`SourceSpec.offset_ns` records the dataset offset. No estimation happens, so replay is exact.
`--estimate-offsets` feeds the dataset time as the arrival time, which exercises the estimator.

**When a tick closes.**

- **Replay** uses a watermark. Tick `T_k` closes when every source has produced a sample with
  `t_master_ns ≥ T_k` or is exhausted. Thread and async scheduling never affect the output, so
  **replay is bit-for-bit deterministic**: the same config produces the same `hash_ticks()`
  and a byte-identical tick file.
- **Live** uses a deadline. Tick `T_k` closes at master time `T_k + grace_ns`.
  - A sample that shows up after its tick closed is counted in `<sid>.late`. It is then dropped
    (`LatePolicy.DROP`, the default) or attached to the next open tick
    (`LatePolicy.ATTACH_NEXT`), keeping its true `t_master_ns`.
  - Replay can also produce late samples, but only for out-of-order source timestamps.
  - `Aligner.step_live(now_ns)` exposes the deadline logic so you can drive it from an
    external loop.

**Sync jitter metric (`sync_jitter_ns`).** Each sample's *alignment error* is
`|t_master_ns − t_arrival|` for estimated sources, which is the clock-model residual. For fixed
sources it's 0. A tick's jitter is the maximum over its samples, and the CLI and benchmark log
p50/p99 over ticks.

`staleness_ns` is reported separately. It's `max(T_k − latest sample)` over the sources present
in a tick, and it's bounded by the tick length and the source rates. The benchmark additionally
reports the *true* alignment error against the synthetic ground truth.

## Buffers and throttling

- **Ring buffer.** Each source gets its own buffer, bounded by `buffer_max_samples` and
  `buffer_max_bytes`. When it's full it drops the oldest sample. Its counters are `dropped`,
  `high_water` and `pushed`. `Stream.backpressure` is true at ≥ 80 % fill.
- **Throttle (`ThrottleConfig`).** You can cap `max_samples_per_s` per source, and
  `max_events_per_s` for EVENT sources. Budgets are enforced per `budget_window_ns`, aligned
  to absolute master time. There are three policies:
  - `drop`: keep the first samples or events of each window.
  - `decimate`: keep every n-th sample or event by sequence number. For events, the stride
    comes from the previous window's observed rate.
  - `accumulate`: merge EVENT batches into `accumulate_window_ns` windows, coalescing
    duplicate `(x, y, p)` to the latest event, then cap.

  All three depend only on timestamps and sequence numbers, never wall time, so they're
  deterministic in replay.

Configure throttles per source with `AlignerConfig(throttle={"dvs0": ThrottleConfig(...)})`.

**Metrics.** Plain counters, gauges and quantile summaries, with no Prometheus:

- `aligner.snapshot() -> dict[str, float]` returns raw keys such as `dvs0.dropped`,
  `dvs0.late`, `sync_jitter_ns.p99` and `ticks`.
- `aligner.stats()` returns a structured summary.

Logging uses `structlog`. The CLI logs at WARNING unless you pass `-v`.

---

## Wire format (`schema/sample.proto`)

`src/senseai_ingest/schema/sample.proto` (package `senseai.ingest.v1`) is the contract for
`senseai-tokens`, the ROS 2 bridge and any non-Python consumer.

- The generated `sample_pb2.py` / `.pyi` files are committed. They're produced by
  grpcio-tools 1.66 (protobuf gencode 5.27), so they work with any protobuf runtime from
  5.27 to 7.x.
- Convert with `senseai_ingest.schema.to_proto(obj)` and `from_proto(msg)`.
- `hash_ticks(ticks)` gives a determinism hash. It serializes with
  `SerializeToString(deterministic=True)` to fix map ordering.

| Python (`senseai_ingest.stream`) | Protobuf (`senseai.ingest.v1`) |
|---|---|
| `Modality.EVENT` (`"event"`) | `Modality.MODALITY_EVENT`, and likewise for each member |
| `ClockMode.FIXED` / `ESTIMATED` | `ClockMode.CLOCK_MODE_FIXED` / `CLOCK_MODE_ESTIMATED` |
| `Sample.source_id / t_source_ns / t_master_ns` | same-named fields |
| `Sample.data: np.ndarray` | `oneof data { Tensor tensor }` |
| `Sample.data: dict[str, np.ndarray]` | `oneof data { TensorMap tensors }` |
| `np.ndarray` | `Tensor{dtype, shape, buffer}`: `DType` enum, little-endian, C-order |
| `Sample.meta: Mapping[str, Any]` | `map<string, Value>`: int, float, str, bool, bytes, ndarray, list and dict map to `int_value` … `map_value` |
| `AlignedTick.index / t_master_ns / tick_ns` | same-named fields |
| `AlignedTick.samples: Mapping[str, list[Sample]]` | `map<string, SampleList>`: the key exists even when the list is empty |
| `SourceSpec` | `SourceSpec`: same field names |

`meta` uses a typed `Value` rather than `google.protobuf.Struct`. Struct stores every number
as a double, which corrupts int64 ns timestamps above 2⁵³.

Source specs travel out-of-band: as attributes in tick files, or from `aligner.specs()`.

## Tick file format (`replay --out ticks.h5`)

The tick file uses a flat, indexed layout rather than one group per tick, which keeps 10⁵+
ticks cheap to write and to slice. Write it with `senseai_ingest.writer.TickWriter` and read it
back with `TickReader`.

```
/                         attrs: format="senseai-ingest/ticks", version=1, tick_ns, t0_ns,
                                 source_ids (JSON, sorted), config (JSON), stats (JSON)
/ticks/t_master_ns        int64  [K]
/ticks/counts             uint32 [K, S]   samples per (tick, source); 0 = explicit empty
/sources/<sid>/           attrs: modality, payload, spec (serialized SourceSpec, final clock state)
    t_source_ns           int64  [N]
    t_master_ns           int64  [N]
    tick_index            uint64 [N]      non-decreasing
    data                  [N, ...]        payload="stacked": constant-shape arrays (FRAME, IMU, ...)
    data + data_ptr       [ΣM, ...], [N+1] payload="ragged": AUDIO, LIDAR, 0-d arrays
    data/<key> + data_ptr [ΣM], [N+1]     payload="columns": dict payloads (EVENT t/x/y/p, GAS)
```

- The samples of source `s` in tick `k` are a contiguous row range. Find it by summing
  `/ticks/counts[:k, s]`.
- Ragged and columns payload `i` spans `data[data_ptr[i]:data_ptr[i+1]]`.
- Per-sample `meta` is not stored. Static metadata lives in the spec.
- Datasets are written with `track_times=False`, so identical runs produce byte-identical
  files.

## Datasets

| Dataset | Entry point | Sources | Notes |
|---|---|---|---|
| DSEC | `DsecReplay.from_sequence(seq_dir)` | `events_left/right`, `rgb_left/right`, `imu` | Events: `events/<cam>/events.h5`, `t` µs relative to `/t_offset`. Frames: `images/<cam>/rectified/*.png` plus `images/timestamps.txt` (needs `[images]`). IMU: DSEC ships it as ROS bags (`lidar_imu.zip`); any `*.bag` under the sequence directory is scanned for an IMU topic (needs `[ros2]`). |
| M3ED | `M3edReplay.from_sequence(file_or_dir)` | `events_left/right`, `rgb`, `mono_left`, `imu` | `layouts/m3ed.yaml`, µs. Ouster LiDAR is raw packets in M3ED and isn't decoded in Gate 1. |
| MVSEC | `MvsecReplay.from_sequence(file_or_dir)` | `events_left/right`, `frames_left`, `imu_left` | `layouts/mvsec.yaml`: packed (N,4) float64 events, ROS-time seconds, ±1 polarity. |
| rosbag | `RosbagReplay.from_bag(bag)` | one per topic | rosbag1 or rosbag2 via `rosbags`. Supports `sensor_msgs/Imu`, `sensor_msgs/Image` and `dvs_msgs/EventArray` (MVSEC bags). |
| WAV | `WavReplay(path, block=1024)` | `mic0` | 8/16-bit PCM → int16; 32-bit and float → float32. |
| any HDF5 | `open_layout(path, "my.yaml")` | per layout | See the layout format in `adapters/replay/hdf5.py`. |

> The M3ED and MVSEC layouts follow the published file layouts but haven't been checked against
> real downloads in CI, which only uses synthetic fixtures. Run `senseai-ingest inspect --path …`
> on your copy first. Fixing a path is a one-line YAML change.

A layout maps HDF5 datasets to sources:

```yaml
name: my_dataset
detect: ["**/events.h5"]
sources:
  - id: events_left
    modality: event
    file: "**/events/left/events.h5"
    events: {t: /events/t, x: /events/x, y: /events/y, p: /events/p}   # or {path, column}
    polarity: identity            # or sign (±1 → 0/1)
    batch_ns: 1000000             # event batches = fixed 1 ms master-time windows
    time: {unit: us, offset: {dataset: /t_offset}}
    meta: {width: 640, height: 480}
  - id: imu
    modality: imu
    file: "**/imu.h5"
    t: /imu/ts
    data: [/imu/accel, /imu/gyro]  # concatenated on the last axis
    time: {unit: us}
    optional: true
```

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

The test suite runs this exact snippet (`tests/test_readme.py`).

**Rules for adapters:**
- One adapter is one source. Expose a factory for multi-source recordings.
- `read()` yields samples in non-decreasing time order.
- Replay adapters set `t_master_ns` from the dataset timebase (`FIXED`). Live adapters set it
  to the host arrival time (`ESTIMATED`). For estimated EVENT sources, the `t` column is in
  device time.
- Subclass `ReplayAdapter` and implement `_read_once()` and `time_range()` to get `start_ns`,
  `end_ns`, `loop` and `speed` for free.

## Serial line protocol (`adapters/live/serial_imu.py`)

The protocol is ASCII, one record per `\n`-terminated line, with `t_us` on the device clock:

```
IMU,<t_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>     m/s², rad/s       -> Modality.IMU      (6,) float64
BUMP,<t_us>,<mask>                            uint32 bitmask    -> Modality.CONTACT  () uint32, kind="bumper"
CUR,<t_us>,<j0>,<j1>,...,<jN>                 A                 -> Modality.CURRENT  (J,) float32
```

- `mask` may be decimal or `0x` hex.
- Blank lines and `#` comments are ignored.
- Malformed lines are counted in `<source_id>.parse_errors` and skipped. `LineParser` never
  raises on bad input.
- `SerialSensorHub(config).adapters()` exposes one adapter per record type.

## ROS 2 (optional)

- **Messages.** `ros2/senseai_ingest_msgs/` is an ament interface package (`Tensor`, `Sample`,
  `SampleList`, `AlignedTick`) that mirrors the proto. Build it in your colcon workspace.

  It's named `senseai_ingest_msgs` rather than `senseai_ingest` because the generated Python
  module `senseai_ingest.msg` would shadow this package. `Sample.meta_pb` carries a serialized
  `senseai.ingest.v1.ValueMap`.
- **Node.** `python -m senseai_ingest.ros2.node --dataset dsec --path … --speed 1` runs an
  `Aligner`. It publishes `senseai_ingest_msgs/AlignedTick` on `/senseai/ticks`, and one
  `SampleList` per source on `/senseai/<source_id>`.
- **Launch.** `ros2 launch launch/replay.launch.py dataset:=dsec path:=…`
- **Imports.** All ROS imports are lazy. The conversion functions (`tick_to_ros`,
  `sample_to_ros`) take the message module as an argument, which lets them be tested without
  ROS.

## Benchmark

`make bench` runs `benchmarks/bench_aligner.py`. It uses four synthetic sources: events at
10⁶ ev/s, 480×640 RGB at 60 Hz, 4-channel audio at 48 kHz in 1024-frame blocks, and IMU at
1 kHz. Each source has a simulated device clock (offset plus 5–50 ppm drift) and transport
latency with jitter.

Results from one 10 s run in the development container:

```
ticks            1001  in 1.76 s wall
ticks/s          570   (target >= 100; 5.7x real time)
sync jitter      p50 0.284 ms   p99 0.532 ms   (target p99 < 1 ms)
true align err   p50 0.509 ms   p99 4.119 ms
drops 0   late 0
PASS
```

The true-error p99 is dominated by the simulated RGB camera's 4 ms constant transport latency.
That's the clock-model limitation described above, not an aligner error.

## Deliberately out of scope

- Tokenization, feature extraction, learned models, and torch → `senseai-tokens`
- Benchmarks of models or tokenizers → `senseai-bench`. The aligner benchmark lives here.
- Calibration (intrinsics/extrinsics estimation). Calibration is carried in `meta` when the
  dataset provides it, but never computed.
- Image debayering, rectification and resampling, and audio resampling
- Hardware time sync (PTP, trigger lines), and latency calibration. Hardware-synced sources
  are declared `ClockMode.FIXED`.
- Live device capture: Gate 1 has stubs with final interfaces.
- LiDAR packet decoding (e.g. M3ED Ouster)
- Prometheus or OpenTelemetry export (the counter dict is the extension point)

## License

Apache-2.0. Copyright 2026 SenseAI Institute. See `LICENSE` and `NOTICE`.
