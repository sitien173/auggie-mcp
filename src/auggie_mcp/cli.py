"""Command-line interface for index maintenance and telemetry statistics."""

import argparse
import json
import math
import os
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from auggie_mcp import config as config_module
from auggie_mcp import embed, indexer, telemetry
from auggie_mcp.locking import IndexBusyError

EXIT_BUSY = 3


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CliUsageError(Exception):
    """Argument parsing failed."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of printing usage and exiting."""

    def error(self, message):
        raise CliUsageError(message)


def _parse_since(value: str) -> timedelta:
    if not re.fullmatch(r"\d+[dh]", value) or int(value[:-1]) == 0:
        raise argparse.ArgumentTypeError(
            f"invalid --since value {value!r}; use Nd or Nh with N > 0"
        )
    unit = "days" if value[-1] == "d" else "hours"
    try:
        window = timedelta(**{unit: int(value[:-1])})
        _utc_now() - window
    except OverflowError:
        raise argparse.ArgumentTypeError(f"--since value {value!r} is too large")
    return window


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="auggie",
        description="Maintain the SQLite index and inspect local telemetry.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_cmd = subparsers.add_parser("index", help="Build or update the SQLite repository index.")
    index_cmd.add_argument("repository", help="Repository directory to index.")
    index_cmd.add_argument("--rebuild", action="store_true", help="Rebuild from scratch.")
    index_cmd.add_argument("--json", action="store_true", help="Print the structured JSON result.")

    status_cmd = subparsers.add_parser("status", help="Report repository index status.")
    status_cmd.add_argument("repository", help="Repository directory to inspect.")

    stats = subparsers.add_parser("stats", help="Summarize local telemetry files.")
    stats.add_argument("--since", type=_parse_since, default=_parse_since("7d"), metavar="Nd|Nh")
    stats.add_argument("--json", action="store_true")
    return parser


def _read_events(since, telemetry_config=None):
    if telemetry_config is not None and not telemetry_config.enabled:
        return [], 0
    cutoff = _utc_now() - since
    events, skipped = [], 0
    directory = telemetry.telemetry_dir(
        directory=telemetry_config.dir if telemetry_config is not None else None
    )
    for path in sorted(directory.glob("events-*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("v") != 1:
                        raise ValueError
                    ts = datetime.fromisoformat(re.sub(r"Z$", "+00:00", event["ts"]))
                    if ts.tzinfo is None:
                        raise ValueError
                    for key in ("tool", "source", "outcome"):
                        if not isinstance(event.get(key), str):
                            raise TypeError
                except (ValueError, KeyError, TypeError):
                    skipped += 1
                    continue
                if ts >= cutoff:
                    events.append(event)
    return events, skipped


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _percentile(sorted_values, percent):
    rank = math.ceil(percent / 100 * len(sorted_values))
    return sorted_values[rank - 1]


def _summarize(events, skipped):
    counts = {"tool": {}, "source": {}, "outcome": {}}
    durations, ratios, tokens = {}, [], 0
    for event in events:
        for key, table in counts.items():
            name = event.get(key)
            table[name] = table.get(name, 0) + 1
        stages = event.get("durations_ms")
        for stage, value in (stages.items() if isinstance(stages, dict) else ()):
            if _is_number(value):
                durations.setdefault(stage, []).append(value)
        processing = event.get("processing")
        if isinstance(processing, dict):
            total = processing.get("chunks_total")
            retained = processing.get("chunks_retained")
            if _is_number(total) and total > 0 and _is_number(retained):
                ratios.append(retained / total)
            if _is_number(processing.get("estimated_tokens")):
                tokens += processing["estimated_tokens"]
    return {
        "events": len(events), "skipped": skipped,
        "by_tool": counts["tool"], "by_source": counts["source"],
        "outcomes": counts["outcome"],
        "durations_ms": {
            stage: {"p50": _percentile(sorted(values), 50), "p95": _percentile(sorted(values), 95)}
            for stage, values in durations.items()
        },
        "retained_chunk_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "estimated_tokens_total": tokens,
    }


def _format_stats(summary):
    ratio = summary["retained_chunk_ratio_mean"]
    rows = [
        ("events", summary["events"]), ("skipped", summary["skipped"]),
        ("retained_chunk_ratio_mean", "n/a" if ratio is None else f"{ratio:.3f}"),
        ("estimated_tokens_total", summary["estimated_tokens_total"]),
    ]
    for title in ("by_tool", "by_source", "outcomes"):
        rows.extend((f"{title}.{name}", count) for name, count in summary[title].items())
    for stage, pct in summary["durations_ms"].items():
        rows.append((f"durations_ms.{stage}", f"p50 {pct['p50']:.1f}  p95 {pct['p95']:.1f}"))
    width = max(len(label) for label, _ in rows)
    return "".join(f"{label:<{width}}  {value}\n" for label, value in rows)


def _emit_error(message: str) -> None:
    json.dump({"error": message}, sys.stderr)
    sys.stderr.write("\n")


def _index_error(exc: Exception) -> dict:
    if isinstance(exc, embed.MissingApiKeyError):
        code = "MISSING_API_KEY"
    elif isinstance(exc, embed.EmbedError):
        code = "EMBED_FAILED"
    elif isinstance(exc, indexer.ModelMismatchError):
        code = "INDEX_MODEL_MISMATCH"
    elif isinstance(exc, IndexBusyError):
        code = "INDEX_BUSY"
    elif isinstance(exc, TimeoutError):
        code = "TIMEOUT"
    elif isinstance(exc, FileNotFoundError):
        code = "INVALID_PATH"
    elif isinstance(exc, ValueError):
        code = "INVALID_PATH" if "path" in str(exc).lower() else "INVALID_ARGUMENT"
    else:
        code = "INTERNAL_ERROR"
    return {
        "code": code,
        "message": str(exc) or exc.__class__.__name__,
        "http_status": exc.http_status if isinstance(exc, embed.EmbedError) else None,
    }


def _emit_index_error(exc: Exception) -> None:
    json.dump({"error": _index_error(exc)}, sys.stdout)
    sys.stdout.write("\n")


def main(argv=None, *, config=None, embedder=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except CliUsageError as exc:
        _emit_error(str(exc))
        return 2
    cfg = config or config_module.load()
    telemetry.prune_old_files(
        enabled=cfg.telemetry.enabled,
        directory=cfg.telemetry.dir,
        retention=cfg.telemetry.retention_days,
    )
    try:
        if args.command == "stats":
            summary = _summarize(*_read_events(args.since, cfg.telemetry))
            if args.json:
                json.dump(summary, sys.stdout)
                sys.stdout.write("\n")
            else:
                sys.stdout.write(_format_stats(summary))
            return 0
        repo = Path(args.repository)
        if not repo.is_dir():
            raise FileNotFoundError(f"repository directory does not exist: {args.repository}")
        if args.command == "index":
            # CLI-only injection keeps subprocess error-channel coverage offline.
            if os.environ.get("AUGGIE_TEST_INDEX_EMBED_ERROR_STATUS") == "429":
                raise embed.EmbedError("test-injected embedding failure", status=429)
            result = indexer.index_update(repo, config=cfg, embedder=embedder, rebuild=args.rebuild)
        else:
            result = indexer.index_status(repo)
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except IndexBusyError as exc:
        if args.command == "index" and args.json:
            _emit_index_error(exc)
        else:
            _emit_error(str(exc))
        return EXIT_BUSY
    except Exception as exc:  # noqa: BLE001 - CLI errors become stable exit results.
        if args.command == "index" and args.json:
            _emit_index_error(exc)
        else:
            _emit_error(str(exc) or exc.__class__.__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
