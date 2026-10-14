# Claude Code prompt — `senseai-bench`

Paste everything below the line into Claude Code from an empty directory (a new GitHub repository named `senseai-bench`). Do this **after** `senseai-tokens` exists; until then only the first milestone (ingest-level benchmarks) can be built.

---

You are building `senseai-bench`, the evaluation package of the SenseAI perception layer. The layer has three packages: `senseai-ingest` (sensor streams → one clock → 10 ms aligned ticks; https://github.com/senseai-institute/ingest, done), `senseai-tokens` (ticks → token streams for a predictive world model and a sub-15 ms reflex head), and this one. `senseai-bench` answers one question, repeatably: **how good is a given build of the stack, and is it better than the last one?** It is Apache-2.0, contains no model code and no tokenizer code, and trains nothing. It runs other packages' code against fixed inputs and reports numbers.

## Context you need

- Read `README.md` of `senseai-ingest` (sections "Core types", "Modality payloads", "Tick file format", "Clock model") and treat them as fixed contracts. Depend on it: `pip install "senseai-ingest @ git+https://github.com/senseai-institute/ingest.git"`. Use `senseai_ingest.writer.TickReader` to read tick files and `senseai_ingest.testing` to generate synthetic recordings; never parse tick files yourself.
- Treat `senseai-tokens` as a black box behind a small interface you define here (`bench/contracts.py`): a `Tokenizer` that maps an `AlignedTick` to a token batch, and a `Predictor`/`Reflex` that consume token streams. If `senseai-tokens` is not installed, every benchmark that needs it is skipped with a clear message, not failed.
- The hardware target later is a laptop-class CPU and an embedded board; report numbers per machine and never compare across machines silently.

## What a benchmark is here

A benchmark is a frozen (input recording, procedure, metric) triple with a stable id, e.g. `ingest.sync_jitter@dsec-zurich_city_04_a@v1`. Changing any of the three changes the id. Every run records: benchmark id, git commit of each package under test, machine fingerprint (CPU, RAM, OS, Python), wall time, the metric values, and the ingest tick-file hash (`senseai_ingest.schema.hash_ticks`) of the input, so a result is reproducible to the byte.

## Deliverables

### 1. Package skeleton

```
senseai-bench/
  pyproject.toml          # name senseai-bench; deps: senseai-ingest, numpy, pyarrow, rich, typer; extras: tokens, dev
  src/senseai_bench/
    contracts.py          # Tokenizer / Predictor / Reflex protocols (typing.Protocol), BenchResult dataclass
    registry.py           # @benchmark(id, needs=("ingest",|"tokens",...)) decorator; discovery; stable ids
    machine.py            # machine fingerprint
    runner.py             # run N repeats, warm-up, p50/p99, write results
    store.py              # results as parquet in ./results/<date>/<id>.parquet + a results index
    compare.py            # compare two runs or two commits; flag regressions beyond a tolerance
    report.py             # Markdown + JSON report; one table per suite
    suites/
      ingest.py           # milestone 1 (below)
      tokens.py           # milestone 2
      reflex.py           # milestone 3
    cli.py                # senseai-bench run|list|compare|report
  benchmarks/data/        # tiny synthetic fixtures generated at build time (no real datasets in git)
  tests/
```

### 2. Milestone 1 — ingest suite (buildable now)

Benchmarks that only need `senseai-ingest`:

- `ingest.throughput`: ticks/s and real-time factor of the Aligner on the 4-source synthetic benchmark sources (reuse `benchmarks/bench_aligner.py`'s `make_sources` from senseai-ingest), 10 s, 5 repeats.
- `ingest.sync_jitter`: p50/p99 sync jitter and true alignment error on the same sources; pass if p99 < 1 ms.
- `ingest.determinism`: two replays of a recording give the same tick hash; metric is boolean plus wall time.
- `ingest.throttle_cost`: throughput with event throttling at 10⁵, 10⁶, 10⁷ events/s caps, each policy.
- `ingest.dataset_replay`: for each recording found under `BENCH_DATA_DIR` (DSEC/M3ED/MVSEC layouts, real data on the user's machine, never in CI): ticks/s, jitter, drops, late. Skipped when the directory is absent.

### 3. Milestone 2 — tokens suite (needs `senseai-tokens`)

- `tokens.latency`: per-tick tokenization latency p50/p99 per modality and for a full 4-source tick; pass if p99 fits inside the tick (10 ms) with headroom reported.
- `tokens.rate`: tokens per second per source at the synthetic rates; bytes in → tokens out ratio.
- `tokens.determinism`: same tick file → same token stream hash.
- `tokens.robustness`: token-stream stability under the ingest throttle policies and under dropped/late samples (how much the token stream changes when 1 %, 5 %, 20 % of samples are dropped).

### 4. Milestone 3 — reflex & world-model suite (needs a model behind the `Reflex`/`Predictor` protocols)

- `reflex.reaction`: time from a synthetic stimulus (bumper bit set, looming object in the event stream, loud transient in audio) to the reflex output, end to end through ingest → tokens → reflex, p50/p99; the Gate target is < 15 ms.
- `reflex.false_alarms`: reflex activations per minute on quiet recordings.
- `world.next_tick`: prediction error of the world model on the next tick's tokens, per modality, on held-out synthetic and real recordings.
- `world.compute`: tokens/s the predictor sustains on this machine, and energy per tick where the platform exposes it (RAPL on Linux; otherwise "n/a").

### 5. CLI

```
senseai-bench list                            # all benchmarks, with which are runnable here
senseai-bench run ingest                      # a suite
senseai-bench run ingest.sync_jitter --repeats 10
senseai-bench compare results/2026-10-14 results/2026-10-21 --tolerance 5%
senseai-bench report results/2026-10-21 --format md > REPORT.md
```

`run` prints a rich table and writes parquet + a `summary.json`. `compare` exits non-zero on a regression so it can gate CI.

### 6. Tests and quality

- Python 3.11+, `uv`, `ruff`, `mypy --strict`, `pytest`, coverage ≥ 85 %.
- Tests use synthetic recordings only; a fake `Tokenizer` in `tests/fakes.py` exercises the tokens suite without `senseai-tokens`.
- The registry test asserts every benchmark id is stable (snapshot file) and that changing a benchmark's version bumps the id.
- GitHub Actions: tests on push; a weekly scheduled job runs the ingest suite and uploads the report as an artifact; a `compare` step fails the job on regression against the last `main` run.
- Apache-2.0 licence headers; `LICENSE`; copyright SenseAI Institute.
- README: what a benchmark is, how to add one in < 30 lines, how to read a report, how to run on a real dataset directory, what is out of scope (training, tuning, model selection).

## Constraints

- No training, no tuning, no model code, no torch dependency in the core (an optional `[torch]` extra may be used only inside milestone-3 benchmarks).
- Results are append-only; never overwrite a previous run.
- Any number reported without machine fingerprint, package commits and input hash is a bug.
- Benchmarks must be deterministic given a seed; wall-time metrics are the only allowed source of run-to-run variation, and they are always reported as p50/p99 over repeats, never a single value.

## Working style

1. Write `contracts.py`, `registry.py` and the README first; show them to me before the suites.
2. Then milestone 1 end to end (runner, store, compare, report, CLI, CI), green, before touching milestone 2.
3. After each module run `make test lint type` and paste the summary.
4. Finish milestone 1 with a real `senseai-bench run ingest` report committed as `docs/example-report.md`.

Definition of done for milestone 1: `senseai-bench run ingest` produces a reproducible report on a laptop; `compare` detects an injected 10 % throughput regression in a test; `senseai-tokens` can be benchmarked by implementing the `Tokenizer` protocol and nothing else.
