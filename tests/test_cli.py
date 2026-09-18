"""Unit tests for auggie_mcp.cli."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from auggie_mcp import cli, pi_runner, tgrep_runner

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"
FAKE_TGREP = TESTS_DIR / "fake_tgrep.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)
if not os.access(FAKE_TGREP, os.X_OK):
    os.chmod(FAKE_TGREP, 0o755)


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
        self.assertEqual(set(subparsers.choices), {"retrieve", "search"})

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
            "tgrep_executable": str(FAKE_TGREP),
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
                cli.tgrep_runner,
                "run_tgrep",
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
                cli.tgrep_runner,
                "run_tgrep",
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
                tgrep_executable="/definitely/missing/tgrep",
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


if __name__ == "__main__":
    unittest.main()
