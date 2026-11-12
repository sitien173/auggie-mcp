"""Unit tests for auggie_mcp.cli."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from auggie_mcp import cli, pi_runner, rg_runner

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"
FAKE_RG = TESTS_DIR / "fake_rg.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)
if not os.access(FAKE_RG, os.X_OK):
    os.chmod(FAKE_RG, 0o755)


_saved_telemetry_dir = None
_telemetry_tmp = None


def setUpModule():
    global _saved_telemetry_dir, _telemetry_tmp
    _saved_telemetry_dir = os.environ.get("AUGGIE_TELEMETRY_DIR")
    _telemetry_tmp = tempfile.mkdtemp()
    os.environ["AUGGIE_TELEMETRY_DIR"] = _telemetry_tmp


def tearDownModule():
    if _saved_telemetry_dir is None:
        os.environ.pop("AUGGIE_TELEMETRY_DIR", None)
    else:
        os.environ["AUGGIE_TELEMETRY_DIR"] = _saved_telemetry_dir
    shutil.rmtree(_telemetry_tmp, ignore_errors=True)


def parse(argv):
    return cli.build_parser().parse_known_args(argv)


class CliParserTest(unittest.TestCase):
    """Task 1: the ``auggie`` argument parser and console script."""

    def test_parser_exposes_retrieve_and_search(self):
        parser = cli.build_parser()
        subparsers = None
        for action in parser._actions:
            if isinstance(action, cli.argparse._SubParsersAction):
                subparsers = action
                break
        self.assertIsNotNone(subparsers)
        self.assertEqual(
            set(subparsers.choices),
            {"retrieve", "search", "stats", *SEMANTIC_COMMANDS},
        )

    def test_retrieve_parses_repository_and_query(self):
        args, extra = parse(["retrieve", "/tmp/repo", "where is main"])
        self.assertEqual(args.command, "retrieve")
        self.assertEqual(args.repository, "/tmp/repo")
        self.assertEqual(args.query, "where is main")
        self.assertFalse(args.human)
        self.assertEqual(extra, [])

    def test_search_parses_pattern_and_human(self):
        args, extra = parse(["search", "hello", "--human"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.pattern, "hello")
        self.assertTrue(args.human)
        self.assertEqual(extra, [])

    def test_search_collects_remaining_flags(self):
        args, extra = parse(["search", "hello", "-F", "-g", "*.py", "-C", "2"])
        self.assertEqual(args.pattern, "hello")
        self.assertEqual(extra, ["-F", "-g", "*.py", "-C", "2"])

    def test_retrieve_requires_repository_and_query(self):
        for argv in (["retrieve"], ["retrieve", "/tmp/repo"]):
            with self.subTest(argv=argv):
                with self.assertRaises(cli.CliUsageError):
                    parse(argv)

    def test_search_requires_pattern(self):
        with self.assertRaises(cli.CliUsageError):
            parse(["search"])

    def test_unknown_command_is_a_usage_error(self):
        with self.assertRaises(cli.CliUsageError):
            parse(["nope"])

    def test_auggie_console_script_is_registered(self):
        import tomllib

        with open("pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        scripts = data["project"]["scripts"]
        self.assertEqual(scripts["auggie"], "auggie_mcp.cli:main")

    def test_auggie_mcp_console_script_is_unchanged(self):
        import tomllib

        with open("pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        scripts = data["project"]["scripts"]
        self.assertEqual(scripts["auggie-mcp"], "auggie_mcp.server:run")


class CliMainTest(unittest.TestCase):
    """Task 2: direct runner invocation, output modes, and exit codes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")

    def invoke(self, argv, **kwargs):
        """Run ``cli.main`` and capture (exit_code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv, **kwargs)
        return code, out.getvalue(), err.getvalue()

    def fake_executables(self):
        return {
            "pi_executable": str(FAKE_PI),
            "rg_executable": str(FAKE_RG),
        }

    # ---- retrieve -----------------------------------------------------

    def test_retrieve_emits_json_response_by_default(self):
        code, out, err = self.invoke(
            ["retrieve", str(self.repo), "what is here?"],
            **self.fake_executables(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        payload = json.loads(out)
        self.assertEqual(payload["response"], "Fake final answer.")

    def test_retrieve_human_prints_response_only(self):
        code, out, err = self.invoke(
            ["retrieve", str(self.repo), "what is here?", "--human"],
            **self.fake_executables(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "Fake final answer.\n")
        self.assertEqual(err, "")

    def test_retrieve_invokes_existing_runner(self):
        with patch.object(
            cli.pi_runner, "run_pi", AsyncMock(return_value="answer text")
        ) as runner:
            code, out, _ = self.invoke(
                ["retrieve", str(self.repo), "query text"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["response"], "answer text")
        call = runner.await_args
        self.assertEqual(call.args[0], str(self.repo))
        self.assertEqual(call.args[1], "query text")
        # the packaged system prompt is loaded and passed to the runner
        self.assertIn("Retrieval Assistant", call.args[2])

    def test_retrieve_runner_failure_emits_json_error_on_stderr(self):
        error = pi_runner.PiExecutionError(
            "Pi exited with status 3", returncode=3, stderr="fake pi failure"
        )
        with patch.object(cli.pi_runner, "run_pi", AsyncMock(side_effect=error)):
            code, out, err = self.invoke(
                ["retrieve", str(self.repo), "query"]
            )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        payload = json.loads(err)
        self.assertIn("fake pi failure", payload["error"])

    def test_retrieve_invalid_repository_emits_json_error(self):
        code, out, err = self.invoke(
            ["retrieve", str(self.repo) + "-missing", "query"],
            **self.fake_executables(),
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("does not exist", json.loads(err)["error"])

    def test_retrieve_rejects_stray_arguments(self):
        code, out, err = self.invoke(
            ["retrieve", str(self.repo), "query", "--bogus"],
            **self.fake_executables(),
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        json.loads(err)

    # ---- argument validation -----------------------------------------

    def test_missing_arguments_emit_json_error(self):
        for argv in (["retrieve"], ["retrieve", "repo"], ["search"], ["nope"]):
            with self.subTest(argv=argv):
                code, out, err = self.invoke(argv)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                json.loads(err)

    # ---- search -------------------------------------------------------

    def test_search_emits_runner_result_json_by_default(self):
        code, out, err = self.invoke(
            ["search", "hello", "-F"],
            **self.fake_executables(),
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("search warning", payload["stderr"])
        self.assertIn("hello", payload["stdout"])

    def test_search_uses_invocation_directory_and_dot_path(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            with patch.object(
                cli.rg_runner,
                "run_rg",
                AsyncMock(
                    return_value={"stdout": "", "stderr": "", "exit_code": 0}
                ),
            ) as runner:
                code, _, _ = self.invoke(["search", "hello", "-F"])
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        call = runner.await_args
        self.assertEqual(call.args[0], str(self.repo))
        self.assertEqual(call.args[1], "hello")
        self.assertEqual(call.kwargs["path"], ".")
        self.assertEqual(call.kwargs["flags"], ["-F"])

    def test_search_forwards_flags_to_existing_validation(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            # --exclude is not in the runner allowlist; the runner's own
            # validation must reject it.
            code, out, err = self.invoke(
                ["search", "hello", "--exclude", "vendor"],
                **self.fake_executables(),
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        payload = json.loads(err)
        self.assertIn("--exclude", payload["error"])

    def test_search_human_prints_stdout(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            with patch.object(
                cli.rg_runner,
                "run_rg",
                AsyncMock(
                    return_value={
                        "stdout": "hello.txt:1:hello world\n",
                        "stderr": "",
                        "exit_code": 0,
                    }
                ),
            ):
                code, out, err = self.invoke(["search", "hello", "--human"])
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        self.assertEqual(out, "hello.txt:1:hello world\n")
        self.assertEqual(err, "")

    def test_search_no_match_emits_json_and_exit_one(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            code, out, err = self.invoke(
                ["search", "nomatch"],
                **self.fake_executables(),
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual(payload["exit_code"], 1)
        self.assertEqual(err, "")

    def test_search_no_match_human_exits_one(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            code, out, _ = self.invoke(
                ["search", "nomatch", "--human"],
                **self.fake_executables(),
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 1)
        self.assertNotIn("{", out)

    def test_search_runner_error_exit_two_emits_json_error(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            code, out, err = self.invoke(
                ["search", "bad["],
                **self.fake_executables(),
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("regex parse error", json.loads(err)["error"])

    def test_search_missing_executable_emits_json_error(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            code, out, err = self.invoke(
                ["search", "hello"],
                pi_executable=str(FAKE_PI),
                rg_executable="/definitely/missing/rg",
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        json.loads(err)

    def test_search_pattern_is_not_shell_interpreted(self):
        marker = Path(self.tmp.name) / "pwned"
        pattern = f"hello; touch {marker}"
        cwd = os.getcwd()
        try:
            os.chdir(self.repo)
            code, out, _ = self.invoke(
                ["search", pattern, "-F"],
                **self.fake_executables(),
            )
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0)
        payload = json.loads(out)
        argv = json.loads(payload["stdout"])["argv"]
        self.assertIn(pattern, argv)
        self.assertFalse(marker.exists())


class CliTelemetryTest(unittest.TestCase):
    """Task 1: CLI calls append one schema-v1 event and keep output unchanged."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")
        self.tdir = Path(self.tmp.name) / "telemetry"

    def invoke(self, argv, *, enabled=True, cwd=None, **kwargs):
        env = {
            "AUGGIE_TELEMETRY_DIR": str(self.tdir),
            "AUGGIE_TELEMETRY": "1" if enabled else "0",
        }
        out, err = io.StringIO(), io.StringIO()
        old = os.getcwd()
        try:
            if cwd:
                os.chdir(cwd)
            with patch.dict(os.environ, env), contextlib.redirect_stdout(
                out
            ), contextlib.redirect_stderr(err):
                code = cli.main(
                    argv,
                    pi_executable=str(FAKE_PI),
                    rg_executable=str(FAKE_RG),
                    **kwargs,
                )
        finally:
            os.chdir(old)
        return code, out.getvalue(), err.getvalue()

    def events(self):
        out = []
        if self.tdir.is_dir():
            for path in sorted(self.tdir.glob("events-*.jsonl")):
                out += [json.loads(l) for l in path.read_text().splitlines()]
        return out

    def assert_same_with_telemetry_off(self, argv, cwd=None):
        on = self.invoke(argv, enabled=True, cwd=cwd)
        off = self.invoke(argv, enabled=False, cwd=cwd)
        self.assertEqual(on, off)
        return on

    def test_retrieve_emits_one_cli_event(self):
        code, _, _ = self.invoke(["retrieve", str(self.repo), "what?"])
        self.assertEqual(code, 0)
        (event,) = self.events()
        self.assertEqual(event["v"], 1)
        self.assertEqual(event["source"], "cli")
        self.assertEqual(event["tool"], "codebase_retrieval")
        self.assertEqual(event["outcome"], "ok")
        self.assertIsNone(event["client"])
        self.assertEqual(
            event["input"],
            {"repository_path": str(self.repo), "query": "what?", "goal": None},
        )
        self.assertEqual(
            event["runner"],
            {
                "exit_code": 0,
                "stdout_bytes": len("Fake final answer.".encode()),
                "stderr_bytes": 0,
            },
        )
        self.assertIn("runner", event["durations_ms"])
        self.assertIsNone(event["processing"])

    def test_retrieve_output_identical_with_telemetry_off(self):
        code, out, err = self.assert_same_with_telemetry_off(
            ["retrieve", str(self.repo), "what?"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(len(self.events()), 1)

    def test_disabled_telemetry_writes_nothing(self):
        self.invoke(["retrieve", str(self.repo), "what?"], enabled=False)
        self.assertEqual(self.events(), [])

    def test_retrieve_pi_failure_records_error_and_runner_bytes(self):
        error = pi_runner.PiExecutionError(
            "Pi exited with status 3", returncode=3, stderr="bad\u00e9"
        )
        with patch.object(cli.pi_runner, "run_pi", AsyncMock(side_effect=error)):
            code, out, err = self.invoke(["retrieve", str(self.repo), "q"])
        self.assertEqual(code, 2)
        (event,) = self.events()
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error"]["type"], "PiExecutionError")
        self.assertEqual(
            event["runner"],
            {"exit_code": 3, "stdout_bytes": 0, "stderr_bytes": len("bad\u00e9".encode())},
        )

    def test_retrieve_usage_error_exit_two_identical_and_recorded(self):
        argv = ["retrieve", str(self.repo), "q", "--bogus"]
        code, out, err = self.assert_same_with_telemetry_off(argv)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        json.loads(err)
        (event,) = self.events()
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error"]["type"], "CliUsageError")
        self.assertIsNone(event["runner"])

    def test_retrieve_cancelled_is_recorded_and_reraised(self):
        import asyncio

        with patch.object(
            cli.pi_runner, "run_pi", AsyncMock(side_effect=asyncio.CancelledError())
        ):
            with patch.dict(
                os.environ,
                {"AUGGIE_TELEMETRY_DIR": str(self.tdir), "AUGGIE_TELEMETRY": "1"},
            ):
                with self.assertRaises(BaseException):
                    cli._run_retrieve(
                        cli.build_parser().parse_known_args(
                            ["retrieve", str(self.repo), "q"]
                        )[0],
                        [],
                        pi_executable="pi",
                    )
        (event,) = self.events()
        self.assertEqual(event["outcome"], "cancelled")

    def test_search_emits_one_cli_event(self):
        code, out, _ = self.invoke(["search", "hello", "-F"], cwd=self.repo)
        self.assertEqual(code, 0)
        (event,) = self.events()
        self.assertEqual(event["source"], "cli")
        self.assertEqual(event["tool"], "codebase_search")
        self.assertEqual(event["outcome"], "ok")
        self.assertEqual(
            event["input"],
            {
                "pattern": "hello",
                "path": ".",
                "flags": ["-F"],
                "goal": None,
                "repository_path": str(self.repo.resolve()),
            },
        )
        result = json.loads(out)
        self.assertEqual(event["runner"]["exit_code"], 0)
        self.assertEqual(
            event["runner"]["stdout_bytes"], len(result["stdout"].encode())
        )
        self.assertEqual(
            event["runner"]["stderr_bytes"], len(result["stderr"].encode())
        )
        self.assertIn("runner", event["durations_ms"])

    def test_search_output_identical_with_telemetry_off(self):
        code, _, _ = self.assert_same_with_telemetry_off(
            ["search", "hello", "-F"], cwd=self.repo
        )
        self.assertEqual(code, 0)

    def test_search_no_match_identical_and_recorded_no_match(self):
        code, out, err = self.assert_same_with_telemetry_off(
            ["search", "nomatch"], cwd=self.repo
        )
        self.assertEqual(code, 1)
        self.assertEqual(err, "")
        (event,) = self.events()
        self.assertEqual(event["outcome"], "no_match")
        self.assertEqual(event["runner"]["exit_code"], 1)

    def test_search_runner_error_identical_and_recorded_error(self):
        code, out, err = self.assert_same_with_telemetry_off(
            ["search", "bad["], cwd=self.repo
        )
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        (event,) = self.events()
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error"]["type"], "RuntimeError")
        self.assertEqual(event["runner"]["exit_code"], 2)

    def test_search_validation_error_exit_two_identical(self):
        code, out, err = self.assert_same_with_telemetry_off(
            ["search", "hello", "--exclude", "vendor"], cwd=self.repo
        )
        self.assertEqual(code, 2)
        (event,) = self.events()
        self.assertEqual(event["outcome"], "error")

    def test_main_prunes_after_parsing_and_before_dispatch(self):
        calls = []
        with patch.object(
            cli.telemetry, "prune_old_files", lambda: calls.append("prune")
        ):
            self.invoke(["search", "hello"], cwd=self.repo)
            self.invoke(["nope"])
        self.assertEqual(calls, ["prune"])


class CliStatsTest(unittest.TestCase):
    """Task 2: ``auggie stats`` summarizes local telemetry files."""

    NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tdir = Path(self.tmp.name) / "telemetry"
        self.tdir.mkdir()

    @staticmethod
    def event(ts, tool, source, outcome, runner, total, processing=None):
        return {
            "v": 1,
            "ts": ts,
            "tool": tool,
            "source": source,
            "outcome": outcome,
            "durations_ms": {"runner": runner, "total": total},
            "processing": processing,
        }

    def write_fixture(self):
        recent = [
            self.event(
                "2026-11-12T11:30:00.000Z", "codebase_retrieval", "cli", "ok",
                100, 120,
                {"chunks_total": 4, "chunks_retained": 2, "estimated_tokens": 100},
            ),
            self.event(
                "2026-11-12T11:45:00.000Z", "codebase_search", "mcp", "no_match",
                10, 20,
            ),
            self.event(
                "2026-11-12T11:50:00.000Z", "codebase_retrieval", "mcp", "error",
                300, 310,
                {"chunks_total": 0, "chunks_retained": 0, "estimated_tokens": 50},
            ),
        ]
        lines = [json.dumps(e) for e in recent]
        lines += [
            "not json",
            "[1, 2]",
            json.dumps({"v": 2, "ts": "2026-11-12T11:00:00.000Z"}),
            json.dumps({"v": 1, "ts": "yesterday-ish", "tool": "x"}),
        ]
        (self.tdir / "events-2026-11-12.jsonl").write_text("\n".join(lines) + "\n")
        old = self.event(
            "2026-11-10T09:00:00.000Z", "codebase_search", "cli", "ok", 50, 60,
            {"chunks_total": 2, "chunks_retained": 2, "estimated_tokens": 25},
        )
        (self.tdir / "events-2026-11-10.jsonl").write_text(json.dumps(old) + "\n")

    def stats(self, *extra, directory=None):
        env = {"AUGGIE_TELEMETRY_DIR": str(directory or self.tdir)}
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env), patch.object(
            cli, "_utc_now", lambda: self.NOW
        ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["stats", *extra])
        return code, out.getvalue(), err.getvalue()

    def test_stats_json_default_window(self):
        self.write_fixture()
        code, out, err = self.stats("--json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertTrue(out.endswith("\n"))
        data = json.loads(out)
        self.assertEqual(data["events"], 4)
        self.assertEqual(data["skipped"], 4)
        self.assertEqual(
            data["by_tool"], {"codebase_retrieval": 2, "codebase_search": 2}
        )
        self.assertEqual(data["by_source"], {"cli": 2, "mcp": 2})
        self.assertEqual(
            data["outcomes"], {"ok": 2, "no_match": 1, "error": 1}
        )
        self.assertEqual(
            data["durations_ms"],
            {
                "runner": {"p50": 50, "p95": 300},
                "total": {"p50": 60, "p95": 310},
            },
        )
        self.assertAlmostEqual(data["retained_chunk_ratio_mean"], 0.75)
        self.assertEqual(data["estimated_tokens_total"], 175)

    def test_stats_since_hours_excludes_older_events(self):
        self.write_fixture()
        _, out, _ = self.stats("--since", "1h", "--json")
        data = json.loads(out)
        self.assertEqual(data["events"], 3)
        self.assertEqual(data["by_tool"]["codebase_search"], 1)
        self.assertEqual(
            data["durations_ms"]["runner"], {"p50": 100, "p95": 300}
        )
        self.assertAlmostEqual(data["retained_chunk_ratio_mean"], 0.5)
        self.assertEqual(data["estimated_tokens_total"], 150)

    def test_stats_missing_directory_reports_zero(self):
        code, out, err = self.stats(
            "--json", directory=Path(self.tmp.name) / "absent"
        )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        data = json.loads(out)
        self.assertEqual(data["events"], 0)
        self.assertEqual(data["skipped"], 0)
        self.assertIsNone(data["retained_chunk_ratio_mean"])
        self.assertEqual(data["estimated_tokens_total"], 0)

    def test_stats_empty_directory_reports_zero(self):
        code, out, _ = self.stats("--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["events"], 0)

    def test_stats_invalid_since_exits_two_with_json_error(self):
        for value in ("7", "d", "7w", "-1d", "1.5d", "0d", ""):
            with self.subTest(value=value):
                code, out, err = self.stats("--since", value)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("error", json.loads(err))

    def test_stats_accepts_z_suffix_timestamp(self):
        # Python < 3.11 rejects "Z" in fromisoformat; the reader must normalize it.
        event = self.event("2026-11-12T11:30:00Z", "t", "cli", "ok", 1, 2)
        (self.tdir / "events-2026-11-12.jsonl").write_text(json.dumps(event) + "\n")
        real = datetime.fromisoformat
        strict = lambda v: real(v) if not v.endswith("Z") else self.fail("raw Z")

        class StrictDatetime(datetime):
            fromisoformat = staticmethod(strict)

        with patch.object(cli, "datetime", StrictDatetime):
            code, out, _ = self.stats("--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual((data["events"], data["skipped"]), (1, 0))

    def test_stats_skips_malformed_field_types(self):
        good = self.event("2026-11-12T11:30:00+00:00", "t", "cli", "ok", 1, 2)
        bad = [
            {**good, "tool": []},
            {**good, "source": {}},
            {**good, "outcome": [1]},
            {**good, "tool": None},
            {k: v for k, v in good.items() if k != "source"},
        ]
        lines = [json.dumps(e) for e in bad + [good]]
        (self.tdir / "events-2026-11-12.jsonl").write_text("\n".join(lines) + "\n")
        code, out, err = self.stats("--json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        data = json.loads(out)
        self.assertEqual((data["events"], data["skipped"]), (1, 5))

    def test_stats_since_overflow_exits_two_without_traceback(self):
        for value in ("1000000000d", "1000000d"):
            with self.subTest(value=value):
                code, out, err = self.stats("--since", value)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("error", json.loads(err))
                self.assertNotIn("Traceback", err)

    def test_stats_default_output_is_human_text(self):
        self.write_fixture()
        code, out, err = self.stats()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertFalse(out.lstrip().startswith("{"))
        self.assertIn("events", out)
        self.assertIn("codebase_retrieval", out)

    def test_stats_writes_no_telemetry_event(self):
        self.write_fixture()
        before = sorted(
            (p.name, p.read_text()) for p in self.tdir.glob("events-*.jsonl")
        )
        self.stats("--json")
        after = sorted(
            (p.name, p.read_text()) for p in self.tdir.glob("events-*.jsonl")
        )
        self.assertEqual(before, after)

    def test_stats_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "auggie_mcp.cli", "stats", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--since", result.stdout)


SEMANTIC_COMMANDS = (
    "index query chunk reset reindex clean cache-report delete log status "
    "health tune watch"
).split()


class SemanticSubcommandTest(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_help_lists_all_sixteen_subcommands(self):
        text = cli.build_parser().format_help()
        for name in ["retrieve", "search", "stats", *SEMANTIC_COMMANDS]:
            with self.subTest(name=name):
                self.assertIn(name, text)

    def test_status_help_exits_zero(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["status", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("status", out.getvalue())

    def test_status_dispatches_to_cloned_handler(self):
        with patch("auggie.cli.cmd_status", return_value=7) as handler:
            code, _, _ = self.run_main("status")
        self.assertEqual(code, 7)
        handler.assert_called_once()

    def test_cloned_subcommand_unknown_extra_argument_returns_two(self):
        with patch("auggie.cli.cmd_status", return_value=0) as handler:
            code, out, err = self.run_main("status", "--bogus-flag")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unrecognized arguments", json.loads(err)["error"])
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
