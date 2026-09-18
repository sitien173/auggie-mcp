"""Command-line interface for the existing retrieval and search runners.

Exposes ``auggie retrieve`` and ``auggie search`` by calling ``pi_runner`` and
``tgrep_runner`` directly. Output is JSON by default and text with ``--human``.
MCP registration and transport behavior are unchanged.
"""

import argparse
import asyncio
import json
import os
import sys

from auggie_mcp import pi_runner, tgrep_runner


class CliUsageError(Exception):
    """Argument parsing failed."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of printing usage and exiting."""

    def error(self, message):
        raise CliUsageError(message)


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

    return parser


def _emit_error(message: str) -> None:
    """Write a JSON error object to standard error."""
    json.dump({"error": message}, sys.stderr)
    sys.stderr.write("\n")


def _run_retrieve(args, extra, *, pi_executable):
    if extra:
        raise CliUsageError(
            "unrecognized arguments: " + " ".join(extra)
        )
    response = asyncio.run(
        pi_runner.run_pi(
            args.repository,
            args.query,
            pi_runner.load_system_prompt(),
            executable=pi_executable,
        )
    )
    if args.human:
        sys.stdout.write(response + "\n")
    else:
        json.dump({"response": response}, sys.stdout)
        sys.stdout.write("\n")
    return 0


def _run_search(args, extra, *, tgrep_executable):
    repository_root = os.getcwd()
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
    if exit_code not in (0, 1):
        message = result["stderr"].strip() or f"tgrep exited with status {exit_code}"
        raise RuntimeError(message)
    if args.human:
        sys.stdout.write(result["stdout"])
    else:
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
    return exit_code


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

    try:
        if args.command == "retrieve":
            return _run_retrieve(args, extra, pi_executable=pi_executable)
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
