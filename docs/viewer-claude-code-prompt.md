# Claude Code prompt — `senseai-viewer`

Paste everything below the line into Claude Code from an empty directory (a new GitHub repository named `senseai-viewer`).

---

You are building `senseai-viewer`, a small web application that lets non-engineers browse the output of `senseai-ingest` (https://github.com/senseai-institute/ingest) in a browser. It is deployed on Railway and reads pre-computed summaries from an Amazon S3 bucket. It contains no model code and never reads raw datasets.

## Context you need

`senseai-ingest` replays recorded sensor datasets (event cameras, RGB frames, audio, IMU) and writes **tick files**: HDF5 files where every 10 ms "tick" lists what every sensor produced. Read these two parts of that repository first and treat them as fixed contracts; do not modify them:

- `README.md`, sections "Core types", "Modality payloads", and "Tick file format".
- `src/senseai_ingest/writer.py` (`TickReader`) and `src/senseai_ingest/stream.py`.

Install it as a dependency: `pip install "senseai-ingest @ git+https://github.com/senseai-institute/ingest.git"`. Use `senseai_ingest.writer.TickReader` to read tick files; never parse the HDF5 layout yourself. `senseai_ingest.testing.make_dsec_fixture` plus the `senseai-ingest replay` CLI generate realistic tick files for tests, so no real datasets are needed in CI.

Tick files are tens to hundreds of MB and live on a work computer or in `s3://<bucket>/processed/`. The viewer must not load them at request time. Instead there is an **export step** that turns one tick file into a compact summary (a few MB) uploaded to `s3://<bucket>/viewer/<recording_id>/`, and the web app reads only those summaries.

## Deliverables

### 1. Export CLI: `senseai-viewer export <tick_file.h5> [--id <recording_id>] [--upload]`

Reads a tick file with `TickReader` and writes a summary directory (locally under `./out/<recording_id>/`, and to S3 with `--upload`):

```
manifest.json        recording_id, dataset name, source list with modality/rate/spec, tick_ns, n_ticks,
                     t0_ns, duration_s, stats (copy of the tick file's stats attr), export version, created_at
timeline.parquet     one row per tick: index, t_master_ns, and per-source sample count and event count
traces/<sid>.parquet downsampled numeric traces (max ~5,000 points per source):
                       IMU → 6 channels; AUDIO → RMS per channel per block; EVENT → events per tick
                       and ON/OFF ratio; CURRENT/CONTACT/GAS → raw values; FRAME → none
thumbs/<sid>/<k>.jpg up to 60 evenly spaced JPEG thumbnails per FRAME source (max 320 px wide) with a
                     thumbs/index.json mapping thumbnail → tick index. EVENT sources also get up to 60
                     thumbnails: accumulate the events of that tick into an image (ON = white, OFF = black on grey).
```

Parquet via `pyarrow`; keep the schema documented in `docs/summary-format.md`. The export must be deterministic for the same input. Target: exporting a 1,000-tick, 4-source file takes under 30 s on a laptop.

### 2. Web app (FastAPI + a single-page frontend)

- Backend: FastAPI serving JSON endpoints that read summaries from S3 (`boto3`), cached in memory with a TTL. Endpoints: `GET /api/recordings` (list from `viewer/*/manifest.json`), `GET /api/recordings/{id}` (manifest), `GET /api/recordings/{id}/timeline`, `.../traces/{sid}`, `.../thumbs/{sid}/{k}.jpg` (streamed from S3, cached), `GET /healthz`.
- Frontend: plain HTML/TypeScript with Vite, no heavy framework needed (Preact or Svelte is fine if you prefer; no Next.js). Charts with Plotly.js or uPlot. Pages:
  1. **Recordings list**: name, dataset, duration, number of sources, sync jitter p99 (green if < 1 ms), drops (green if 0), export date.
  2. **Recording view**: top, a stats card (ticks, tick length, jitter p50/p99, staleness, per-source rate/drops/late); middle, a stacked timeline with one lane per source showing activity per tick (events per tick as a bar chart for EVENT sources, dots for frames, RMS for audio, line for IMU); bottom, a scrubber bound to the timeline that shows the nearest frame and event thumbnails and the exact per-source sample counts for the selected tick. Clicking any point in the timeline moves the scrubber.
  3. **Compare** (nice to have): overlay the events-per-tick trace of two recordings.
- Password protection: a single shared password from `VIEWER_PASSWORD`, checked by HTTP Basic auth on everything except `/healthz`. If the variable is unset, refuse to start in production (`RAILWAY_ENVIRONMENT` set) and allow in development.
- Must be usable on a phone (the list page and stats card at least).

### 3. Railway deployment

- `Dockerfile` (multi-stage: build the frontend with Node, serve everything from one Python image with uvicorn). Listen on `$PORT`.
- `railway.json` with the health check on `/healthz`.
- Configuration only through environment variables: `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `VIEWER_PASSWORD`, optional `S3_PREFIX` (default `viewer`), optional `CACHE_TTL_S` (default 60).
- A `README.md` with: what the viewer is, a screenshot placeholder, the three-step deploy on Railway (connect repo → set variables → generate domain), how to run locally against a folder instead of S3 (`LOCAL_SUMMARIES=./out` bypasses S3), and the export command.

### 4. Tests and quality

- Python 3.11+, `uv`, `ruff`, `mypy --strict` on the backend and export code, `pytest` with coverage ≥ 80 %.
- Tests generate a tick file in a fixture using `senseai_ingest.testing.make_dsec_fixture` and the `senseai-ingest replay` CLI (via `subprocess` or `senseai_ingest.cli.main`), run the export, and check the manifest, timeline row count, trace lengths and thumbnail counts. Export determinism test: two exports of the same file produce identical bytes for every non-JPEG file.
- API tests with FastAPI's `TestClient` against the local-folder mode (no S3 in CI); use `moto` for one S3 round-trip test.
- A frontend smoke test with Playwright: the list page loads, clicking a recording shows the stats card and the timeline, the scrubber updates the thumbnails.
- `make test lint type build` all green. GitHub Actions workflow running them on push.
- Apache-2.0 licence header on every source file; `LICENSE` file; copyright SenseAI Institute.

## Constraints

- No model code; no torch; no reading of raw datasets. The only input is tick files, through `TickReader`.
- The web app must never read a tick file; only summaries. If a summary is missing a piece (e.g. no thumbnails), the UI shows "not available", it does not crash.
- Keep the dependency list short. Backend: fastapi, uvicorn, boto3, pyarrow, numpy, pillow, senseai-ingest. No database.
- Every public function has a docstring and type hints.

## Working style

1. Start by writing `docs/summary-format.md` (the summary contract) and the `README.md`; show them to me before implementation.
2. Then: export CLI with tests → backend API with tests → frontend → Dockerfile and Railway config → GitHub Actions.
3. After each step run `make test lint type` and paste the summary.
4. Finish by running the whole thing locally end to end: generate a synthetic DSEC fixture, replay it with `senseai-ingest`, export, start the server in local-folder mode, and take a Playwright screenshot of the recording view; save it as `docs/screenshot.png` and reference it in the README.

Definition of done: a synthetic recording can be exported and browsed locally with one `make demo` command; the Railway deploy works with the five environment variables and nothing else; tests ≥ 80 % coverage; the summary format is documented and versioned.
