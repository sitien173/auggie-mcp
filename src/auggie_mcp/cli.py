"""Command-line interface for the existing retrieval and search runners.

Exposes ``auggie retrieve`` and ``auggie search`` by calling ``pi_runner`` and
``tgrep_runner`` directly. Output is JSON by default and text with ``--human``.
Also registers the semantic index subcommands from ``auggie.cli`` (index, query,
chunk, reset, reindex, clean, cache-report, delete, log, status, health, tune,
watch). MCP registration and transport behavior are unchanged.
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from auggie import cli as semantic_cli
from auggie_mcp import pi_runner, telemetry, tgrep_runner


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CliUsageError(Exception):
    """Argument parsing failed."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of printing usage and exiting."""

    def error(self, message):
        raise CliUsageError(message)


def _parse_since(value: str) -> timedelta:
    """Parse a positive ``Nd`` or ``Nh`` window."""
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
    """Build the ``auggie`` argument parser."""
    parser = _ArgumentParser(
        prog="auggie",
        description="Semantic retrieval and exact repository search.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Semantically retrieve code with the Pi retrieval runner.",
        description="Run the Pi retrieval runner and print its response.",
    )
    retrieve.add_argument("repository", help="Repository directory to search.")
    retrieve.add_argument("query", help="Natural-language retrieval query.")
    retrieve.add_argument(
        "--human",
        action="store_true",
        help="Print the retrieval response as text instead of JSON.",
    )

    search = subparsers.add_parser(
        "search",
        help="Run an exact tgrep search in the invocation directory.",
        description=(
            "Run tgrep in the invocation directory. Remaining arguments are "
            "forwarded to the existing tgrep flag validation."
        ),
    )
    search.add_argument("pattern", help="Search pattern passed to tgrep.")
    search.add_argument(
        "--human",
        action="store_true",
        help="Print the search standard output instead of JSON.",
    )

    stats = subparsers.add_parser(
        "stats",
        help="Summarize local telemetry files.",
        description="Summarize the local telemetry events in a time window.",
    )
    stats.add_argument(
        "--since",
        type=_parse_since,
        default=_parse_since("7d"),
        metavar="Nd|Nh",
        help="Window size in days or hours, for example 7d or 24h (default: 7d).",
    )
    stats.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object instead of text.",
    )

    semantic_cli._add_index_command(subparsers)
    semantic_cli._add_query_command(subparsers)
    semantic_cli._add_chunk_command(subparsers)
    semantic_cli._add_reset_command(subparsers)
    semantic_cli._add_reindex_command(subparsers)
    semantic_cli._add_clean_command(subparsers)
    semantic_cli._add_cache_report_command(subparsers)
    semantic_cli._add_delete_command(subparsers)
    semantic_cli._add_log_command(subparsers)
    semantic_cli._add_status_command(subparsers)
    semantic_cli._add_health_command(subparsers)
    semantic_cli._add_tune_command(subparsers)
    semantic_cli._add_watch_command(subparsers)

    return parser


def _emit_error(message: str) -> None:
    """Write a JSON error object to standard error."""
    json.dump({"error": message}, sys.stderr)
    sys.stderr.write("\n")


def _run_retrieve(args, extra, *, pi_executable):
    record = telemetry.CallRecord(
        "cli",
        "codebase_retrieval",
        {"repository_path": args.repository, "query": args.query, "goal": None},
        client=None,
    )
    outcome, error = "ok", None
    try:
        if extra:
            raise CliUsageError(
                "unrecognized arguments: " + " ".join(extra)
            )
        system_prompt = pi_runner.load_system_prompt()
        try:
            with record.stage("runner"):
                response = asyncio.run(
                    pi_runner.run_pi(
                        args.repository,
                        args.query,
                        system_prompt,
                        executable=pi_executable,
                    )
                )
        except pi_runner.PiExecutionError as exc:
            record.set_runner(
                exit_code=exc.returncode,
                stdout_bytes=0,
                stderr_bytes=len((exc.stderr or "").encode("utf-8")),
            )
            raise
        record.set_runner(
            exit_code=0,
            stdout_bytes=len(response.encode("utf-8")),
            stderr_bytes=0,
        )
        if args.human:
            sys.stdout.write(response + "\n")
        else:
            json.dump({"response": response}, sys.stdout)
            sys.stdout.write("\n")
        return 0
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as exc:
        outcome, error = "error", exc
        raise
    finally:
        record.finish(outcome, error)


def _run_search(args, extra, *, tgrep_executable):
    repository_root = os.getcwd()
    record = telemetry.CallRecord(
        "cli",
        "codebase_search",
        {
            "pattern": args.pattern,
            "path": ".",
            "flags": extra,
            "goal": None,
            "repository_path": repository_root,
        },
    )
    outcome, error = "ok", None
    try:
        with record.stage("runner"):
            result = asyncio.run(
                tgrep_runner.run_tgrep(
                    repository_root,
                    args.pattern,
                    path=".",
                    flags=extra,
                    executable=tgrep_executable,
                )
            )
        exit_code = result["exit_code"]
        record.set_runner(
            exit_code=exit_code,
            stdout_bytes=len(result["stdout"].encode("utf-8")),
            stderr_bytes=len(result["stderr"].encode("utf-8")),
        )
        if exit_code not in (0, 1):
            message = result["stderr"].strip() or f"tgrep exited with status {exit_code}"
            raise RuntimeError(message)
        if exit_code == 1:
            outcome = "no_match"
        if args.human:
            sys.stdout.write(result["stdout"])
        else:
            json.dump(result, sys.stdout)
            sys.stdout.write("\n")
        return exit_code
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as exc:
        outcome, error = "error", exc
        raise
    finally:
        record.finish(outcome, error)


def _percentile(sorted_values, percent):
    # Nearest rank: rank = ceil(p/100 * n), value = sorted[rank - 1].
    rank = math.ceil(percent / 100 * len(sorted_values))
    return sorted_values[rank - 1]


def _read_events(since):
    """Return (events in window, skipped line count) from the telemetry files."""
    cutoff = _utc_now() - since
    events, skipped = [], 0
    for path in sorted(telemetry.telemetry_dir().glob("events-*.jsonl")):
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
                            raise ValueError
                except (ValueError, KeyError, TypeError):
                    skipped += 1
                    continue
                if ts >= cutoff:
                    events.append(event)
    return events, skipped


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
        "events": len(events),
        "skipped": skipped,
        "by_tool": counts["tool"],
        "by_source": counts["source"],
        "outcomes": counts["outcome"],
        "durations_ms": {
            stage: {
                "p50": _percentile(sorted(values), 50),
                "p95": _percentile(sorted(values), 95),
            }
            for stage, values in durations.items()
        },
        "retained_chunk_ratio_mean": (
            sum(ratios) / len(ratios) if ratios else None
        ),
        "estimated_tokens_total": tokens,
    }


def _format_stats(summary):
    ratio = summary["retained_chunk_ratio_mean"]
    rows = [
        ("events", summary["events"]),
        ("skipped", summary["skipped"]),
        (
            "retained_chunk_ratio_mean",
            "n/a" if ratio is None else f"{ratio:.3f}",
        ),
        ("estimated_tokens_total", summary["estimated_tokens_total"]),
    ]
    for title, key in (
        ("by_tool", "by_tool"),
        ("by_source", "by_source"),
        ("outcomes", "outcomes"),
    ):
        rows += [(f"{title}.{name}", n) for name, n in summary[key].items()]
    for stage, pct in summary["durations_ms"].items():
        rows.append(
            (f"durations_ms.{stage}", f"p50 {pct['p50']:.1f}  p95 {pct['p95']:.1f}")
        )
    width = max(len(label) for label, _ in rows)
    return "".join(f"{label:<{width}}  {value}\n" for label, value in rows)


def _run_stats(args, extra):
    if extra:
        raise CliUsageError("unrecognized arguments: " + " ".join(extra))
    summary = _summarize(*_read_events(args.since))
    if args.json:
        json.dump(summary, sys.stdout)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_stats(summary))
    return 0


def main(
    argv=None,
    *,
    pi_executable: str = "pi",
    tgrep_executable: str = "tgrep",
) -> int:
    """Run the ``auggie`` command line and return an exit code."""
    parser = build_parser()
    try:
        args, extra = parser.parse_known_args(argv)
    except CliUsageError as exc:
        _emit_error(str(exc))
        return 2

    telemetry.prune_old_files()
    func = getattr(args, "func", None)
    if func is not None:
        if extra:
            _emit_error("unrecognized arguments: " + " ".join(extra))
            return 2
        return func(args)
    try:
        if args.command == "retrieve":
            return _run_retrieve(args, extra, pi_executable=pi_executable)
        if args.command == "stats":
            return _run_stats(args, extra)
        return _run_search(args, extra, tgrep_executable=tgrep_executable)
    except CliUsageError as exc:
        _emit_error(str(exc))
        return 2
    except (ValueError, FileNotFoundError) as exc:
        _emit_error(str(exc))
        return 2
    except pi_runner.PiRunnerError as exc:
        message = str(exc)
        if isinstance(exc, pi_runner.PiExecutionError) and exc.stderr:
            message = f"{message}: {exc.stderr}"
        _emit_error(message)
        return 2
    except Exception as exc:  # timeout, unexpected runner failure
        _emit_error(str(exc) or exc.__class__.__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
