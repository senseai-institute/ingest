# Copyright 2026 SenseAI Institute
# SPDX-License-Identifier: Apache-2.0
"""Command line: ``senseai-ingest replay`` and ``senseai-ingest inspect``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from senseai_ingest.adapters.base import Adapter, ReplayAdapter
from senseai_ingest.sync import Aligner, AlignerConfig, LatePolicy

__all__ = ["build_adapters", "main"]

DATASETS = ("dsec", "m3ed", "mvsec", "hdf5", "wav", "rosbag")


def build_adapters(
    dataset: str,
    path: str | Path,
    *,
    layout: str | None = None,
    sources: Sequence[str] | None = None,
    **kw: Any,
) -> list[Adapter]:
    """Create replay adapters for ``dataset`` at ``path`` (``kw`` goes to each adapter)."""
    p = Path(path)
    if dataset == "dsec":
        from senseai_ingest.adapters.replay.dsec import DsecReplay

        return DsecReplay.from_sequence(p, sources=sources, **kw)
    if dataset == "m3ed":
        from senseai_ingest.adapters.replay.m3ed import M3edReplay

        return M3edReplay.from_sequence(p, sources=sources, **kw)
    if dataset == "mvsec":
        from senseai_ingest.adapters.replay.mvsec import MvsecReplay

        return MvsecReplay.from_sequence(p, sources=sources, **kw)
    if dataset == "hdf5":
        from senseai_ingest.adapters.replay.hdf5 import open_layout

        if layout is None:
            raise SystemExit("--dataset hdf5 requires --layout <name|file.yaml>")
        return list(open_layout(p, layout, sources=sources, **kw))
    if dataset == "wav":
        from senseai_ingest.adapters.replay.wav import WavReplay

        files = sorted(p.glob("*.wav")) if p.is_dir() else [p]
        ads: list[Adapter] = [
            WavReplay(f, source_id=f"mic{i}" if len(files) > 1 else "mic0", **kw)
            for i, f in enumerate(files)
        ]
        return [a for a in ads if sources is None or a.source_id in sources]
    if dataset == "rosbag":
        from senseai_ingest.adapters.replay.rosbag import RosbagReplay

        return list(RosbagReplay.from_bag(p, sources=sources, **kw))
    raise SystemExit(f"unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")


def _fmt_ms(ns: float) -> str:
    return "n/a" if ns != ns else f"{ns / 1e6:.3f} ms"


def format_stats(stats: dict[str, Any], wall_s: float) -> str:
    """Render aligner stats as a small text table."""
    n = stats["ticks"]
    lines = [
        f"ticks: {n}   wall: {wall_s:.2f} s   ({n / wall_s if wall_s > 0 else 0:.0f} ticks/s)",
        f"sync jitter     p50 {_fmt_ms(stats['sync_jitter_p50_ns'])}   "
        f"p99 {_fmt_ms(stats['sync_jitter_p99_ns'])}",
        f"staleness       p50 {_fmt_ms(stats['staleness_p50_ns'])}   "
        f"p99 {_fmt_ms(stats['staleness_p99_ns'])}",
        f"{'source':<16}{'modality':<10}{'samples/s':>11}{'events':>12}{'dropped':>9}"
        f"{'late':>6}{'high_water':>12}",
    ]
    for sid, s in stats["sources"].items():
        lines.append(
            f"{sid:<16}{s['modality']:<10}{s['samples_per_s']:>11.1f}{int(s['events']):>12}"
            f"{int(s['dropped']):>9}{int(s['late']):>6}{int(s['high_water']):>12}"
        )
    return "\n".join(lines)


def _replay(args: argparse.Namespace) -> int:
    sources = args.sources.split(",") if args.sources else None
    adapters = build_adapters(
        args.dataset,
        args.path,
        layout=args.layout,
        sources=sources,
        speed=args.speed,
        loop=args.loop,
    )
    if not adapters:
        print(f"no sources found under {args.path}", file=sys.stderr)
        return 2
    start_ns = end_ns = None
    if args.start_s is not None or args.duration_s is not None:
        first = min(a.time_range()[0] for a in adapters if isinstance(a, ReplayAdapter))
        start_ns = first + round((args.start_s or 0.0) * 1e9)
        if args.duration_s is not None:
            end_ns = start_ns + round(args.duration_s * 1e9)
        for a in adapters:
            if isinstance(a, ReplayAdapter):
                a.start_ns, a.end_ns = start_ns, end_ns
    if args.loop and args.max_ticks is None and end_ns is None:
        raise SystemExit("--loop needs --max-ticks or --duration-s")
    cfg = AlignerConfig(
        tick_ns=round(args.tick_ms * 1e6),
        start_ns=start_ns,
        end_ns=end_ns,
        max_ticks=args.max_ticks,
        late_policy=LatePolicy(args.late_policy),
        estimate_offsets=True if args.estimate_offsets else None,
    )
    aligner = Aligner(adapters, cfg)
    writer = None
    t_wall = time.monotonic()
    with aligner:
        if args.out:
            from senseai_ingest.writer import TickWriter

            config = {k: v for k, v in vars(args).items() if k != "func"}
            writer = TickWriter(args.out, aligner.specs(), tick_ns=cfg.tick_ns, config=config)
        try:
            for tick in aligner:
                if writer is not None:
                    writer.write(tick)
        finally:
            stats = aligner.stats()
            if writer is not None:
                writer.update_specs(aligner.specs())  # final clock estimates
                writer.close(stats=stats)
    wall = time.monotonic() - t_wall
    if args.json:
        print(json.dumps({**stats, "wall_s": wall}, indent=2, default=str))
    elif args.stats:
        print(format_stats(stats, wall))
    else:
        dest = f" -> {args.out}" if args.out else ""
        print(f"{stats['ticks']} ticks{dest}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from senseai_ingest.adapters.replay.hdf5 import describe_h5, detect_layout

    p = Path(args.path)
    if not p.exists():
        print(f"no such path: {p}", file=sys.stderr)
        return 2
    name = detect_layout(p)
    print(f"path:   {p}")
    print(f"layout: {name or 'unknown (use --dataset hdf5 --layout <yaml>)'}")
    if name is not None:
        adapters = build_adapters(name, p)
        print("sources:")
        for a in adapters:
            spec = a.spec()
            first, last = a.time_range() if isinstance(a, ReplayAdapter) else (0, 0)
            print(
                f"  {spec.source_id:<16}{spec.modality.value:<9}{spec.nominal_rate_hz:>10.1f} Hz"
                f"  span {(last - first) / 1e9:8.3f} s  offset {spec.offset_ns} ns"
            )
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in (".h5", ".hdf5"))
    for f in files[: args.max_files]:
        try:
            rows = describe_h5(f)
        except OSError:
            continue
        print(f"{f}:")
        for r in rows:
            print(f"  {r['path']:<40}{r['dtype']:<10}{r['shape']!s:<20}{r['attrs'] or ''}")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="senseai-ingest", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true", help="enable info logging")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="replay a recording into aligned ticks")
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--path", required=True)
    r.add_argument("--layout", help="layout name or YAML file (for --dataset hdf5)")
    r.add_argument("--sources", help="comma-separated source ids to include")
    r.add_argument("--tick-ms", type=float, default=10.0)
    r.add_argument("--speed", type=float, default=0.0, help="1 = real time, 0 = max speed")
    r.add_argument("--start-s", type=float, help="skip this many seconds from the first sample")
    r.add_argument("--duration-s", type=float)
    r.add_argument("--max-ticks", type=int)
    r.add_argument("--loop", action="store_true")
    r.add_argument("--late-policy", choices=[p.value for p in LatePolicy], default="drop")
    r.add_argument(
        "--estimate-offsets",
        action="store_true",
        help="exercise the clock estimator on recorded timestamps",
    )
    r.add_argument("--out", help="write ticks to this HDF5 file")
    r.add_argument("--stats", action="store_true", help="print jitter / rate / drop table")
    r.add_argument("--json", action="store_true", help="print stats as JSON")
    r.set_defaults(func=_replay)

    i = sub.add_parser("inspect", help="print a dataset's detected layout and HDF5 contents")
    i.add_argument("--path", required=True)
    i.add_argument("--max-files", type=int, default=8)
    i.set_defaults(func=_inspect)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``senseai-ingest`` console script."""
    args = _parser().parse_args(argv)
    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO if args.verbose else logging.WARNING
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
