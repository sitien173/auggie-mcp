"""Unit tests for auggie_mcp.grep."""

import asyncio
import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

from auggie_mcp import grep

TESTS_DIR = Path(__file__).resolve().parent
FAKE_RG = TESTS_DIR / "fake_rg.py"


class RgRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.subdir = self.repo / "src"
        self.subdir.mkdir()
        self.source = self.subdir / "main.py"
        self.source.write_text("print('hello')\n")
        self.pattern_file = self.repo / "patterns.txt"
        self.pattern_file.write_text("hello\n")
        self.ignore_file = self.repo / "extra.ignore"
        self.ignore_file.write_text("ignored\n")

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_relative_path_resolves_beneath_repository_root(self):
        resolved = grep.resolve_search_path(self.repo, "src")
        self.assertEqual(resolved, self.subdir)

    def test_search_path_accepts_single_files(self):
        resolved = grep.resolve_search_path(self.repo, "src/main.py")
        self.assertEqual(resolved, self.source)

    def test_absolute_search_path_accepts_path_outside_repository_root(self):
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()

        resolved = grep.resolve_search_path(self.repo, outside)

        self.assertEqual(resolved, outside)

    def test_search_path_rejects_missing_paths_and_root_escape(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            grep.resolve_search_path(self.repo, "missing")
        with self.assertRaisesRegex(ValueError, "escapes repository root"):
            grep.resolve_search_path(self.repo, "..")

    def test_symlink_escape_is_rejected(self):
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (self.repo / "escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes repository root"):
            grep.resolve_search_path(self.repo, "escape")

    def test_all_documented_search_flags_are_accepted(self):
        no_value_flags = [
            "-i", "--ignore-case", "-s", "--case-sensitive", "-S",
            "--smart-case", "-F", "--fixed-strings", "-w", "--word-regexp",
            "-v", "--invert-match", "-o", "--only-matching", "-U",
            "--multiline", "--multiline-dotall", "-n", "--line-number", "-N",
            "--no-line-number", "-c", "--count", "-l", "--files-with-matches",
            "--files-without-match", "-q", "--quiet", "--glob-case-insensitive",
            "--type-list", "--files", "--heading", "--no-heading", "-H",
            "--with-filename", "-I", "--no-filename", "--json", "--vimgrep",
            "-0", "--null", "--trim", "-.", "--hidden", "--no-ignore", "-a",
            "--text", "--binary", "-u", "-uu", "-uuu", "--no-max-filesize",
            "--no-messages", "--stats", "-x",
            "--line-regexp", "-P", "--pcre2", "--pcre2-version", "--no-unicode",
            "--passthru", "--stop-on-nonmatch", "--column", "--no-column", "-b",
            "--byte-offset", "--max-columns-preview", "--count-matches",
            "--include-zero", "-p", "--pretty", "--no-context-separator",
            "--sort-files", "--line-buffered", "--block-buffered", "--no-encoding",
            "--one-file-system", "--ignore-file-case-insensitive", "--no-ignore-dot",
            "--no-ignore-exclude", "--no-ignore-files", "--no-ignore-global",
            "--no-ignore-parent", "--no-ignore-vcs", "--no-ignore-messages",
            "--no-require-git", "--mmap", "--no-mmap", "--crlf", "--no-crlf",
            "--no-config", "--debug", "--trace",
        ]
        value_flags = {
            "-e": "other", "--regexp": "other", "-f": str(self.pattern_file),
            "--file": str(self.pattern_file), "-m": "5", "--max-count": "5",
            "-g": "*.py", "--glob": "*.py", "--iglob": "*.PY", "-t": "py",
            "--type": "py", "-T": "rust", "--type-not": "rust",
            "--type-add": "web:*.html", "--type-clear": "web", "-A": "2",
            "--after-context": "2", "-B": "2", "--before-context": "2", "-C": "2",
            "--context": "2", "--color": "never", "--max-filesize": "64M",
            "--engine": "auto",
            "--regex-size-limit": "10M", "--dfa-size-limit": "10M", "-r": "x",
            "--replace": "x", "-M": "120", "--max-columns": "120",
            "--context-separator": "--", "--field-match-separator": ":",
            "--field-context-separator": "-", "--path-separator": "/",
            "--sort": "path", "--sortr": "path", "-E": "utf-8",
            "--encoding": "utf-8", "--max-depth": "5",
            "--ignore-file": str(self.ignore_file), "-j": "4", "--threads": "4",
            "--colors": "match:fg:red",
        }

        for flag in no_value_flags:
            with self.subTest(flag=flag):
                self.assertEqual(grep.validate_flags(self.repo, [flag]), [flag])
        for flag, value in value_flags.items():
            with self.subTest(flag=flag):
                validated = grep.validate_flags(self.repo, [flag, value])
                self.assertEqual(validated[0], flag)
                self.assertEqual(len(validated), 2)

    def test_long_equals_forms_are_accepted(self):
        self.assertEqual(
            grep.validate_flags(self.repo, ["--glob=*.py", "--context=2"]),
            ["--glob=*.py", "--context=2"],
        )

    def test_file_valued_flags_cannot_escape_repository(self):
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("x")
        for flag in ("-f", "--file", "--ignore-file"):
                with self.subTest(flag=flag), self.assertRaisesRegex(
                    ValueError, "escapes repository root"
                ):
                    grep.validate_flags(self.repo, [flag, str(outside)])

    def test_unknown_setup_and_positional_flags_are_rejected(self):
        for flags in (
            ["--wat"], ["--exclude", "vendor"], ["-z"], ["--follow"], ["bare"],
            ["--no-index"], ["--index-path", "idx"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                grep.validate_flags(self.repo, flags)

    def test_build_command_places_flags_before_pattern_and_path(self):
        cmd = grep.build_rg_command(
            self.repo, "hello", "src", ["-F", "-g", "*.py"]
        )
        self.assertEqual(
            cmd,
            ["-F", "-g", "*.py", "--", "hello", str(self.subdir)],
        )

    def test_explicit_pattern_flags_keep_the_primary_pattern(self):
        cmd = grep.build_rg_command(
            self.repo, "hello", "src", ["-e", "other"]
        )
        self.assertEqual(
            cmd,
            ["-e", "other", "--", "hello", str(self.subdir)],
        )

    def test_run_returns_stdout_stderr_and_all_exit_codes(self):
        for pattern, expected_code in (("hello", 0), ("nomatch", 1), ("bad[", 2)):
            with self.subTest(pattern=pattern):
                result = self.run_async(
                    grep.run_rg(
                        self.repo,
                        pattern,
                        "src",
                        ["-F"],
                        executable=str(FAKE_RG),
                    )
                )
                self.assertEqual(result["exit_code"], expected_code)
                self.assertIn("stderr", result)
                self.assertIn("stdout", result)

    def test_pattern_and_flags_are_not_interpreted_by_a_shell(self):
        marker = Path(self.tmp.name) / "pwned"
        pattern = f"hello; touch {marker}"
        result = self.run_async(
            grep.run_rg(
                self.repo,
                pattern,
                ".",
                ["--replace", "$(touch nope)"],
                executable=str(FAKE_RG),
            )
        )
        payload = json.loads(result["stdout"])
        self.assertIn(pattern, payload["argv"])
        self.assertFalse(marker.exists())

    def test_cancellation_terminates_rg_process(self):
        async def scenario():
            task = asyncio.create_task(
                grep.run_rg(
                    self.repo, "stall", executable=str(FAKE_RG)
                )
            )
            pid_file = self.repo / "rg.pid"
            deadline = time.monotonic() + 3
            while not pid_file.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            self.assertTrue(pid_file.exists())
            pid = int(pid_file.read_text())

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            deadline = time.monotonic() + 3
            while _pid_alive(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            if _pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            self.assertFalse(_pid_alive(pid))

        self.run_async(scenario())

    def test_missing_executable_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.run_async(
                grep.run_rg(
                    self.repo, "hello", executable="/definitely/missing/rg"
                )
            )


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    unittest.main()


class StructuredSearchTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        (self.repo / "a.py").write_text("alpha\nbeta\nalpha again\n")

    async def test_structured_matches_are_relative_and_max_results_is_honored(self):
        result = await grep.grep_search(self.repo, "alpha", max_results=1)
        self.assertEqual(result["matches"], [{"path": "a.py", "line": 1, "text": "alpha"}])
        self.assertTrue(result["truncated"])

    async def test_no_match_returns_empty_structured_result(self):
        result = await grep.grep_search(self.repo, "missing")
        self.assertEqual(result, {"matches": [], "truncated": False, "archive_path": None})

    async def test_output_over_limit_is_archived_and_truncated(self):
        (self.repo / "large.txt").write_text("x\n" * 40000)
        result = await grep.grep_search(self.repo, "x", path="large.txt", max_results=50000)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["archive_path"])
        self.assertLess(sum(len(m["text"].encode()) for m in result["matches"]), 64 * 1024)
        self.assertTrue(Path(result["archive_path"]).is_file())

    async def test_hung_rg_times_out(self):
        executable = self.repo / "hang"
        executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
        executable.chmod(0o755)
        with self.assertRaises(TimeoutError):
            await grep.grep_search(self.repo, "x", executable=str(executable), timeout=0.05)

    async def test_timeout_covers_process_wait_after_streams_close(self):
        pid_file = self.repo / "sleeping.pid"
        executable = self.repo / "close-streams-and-sleep"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import os, time\n"
            "os.close(1)\n"
            "os.close(2)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(5)\n"
        )
        executable.chmod(0o755)

        with self.assertRaises(TimeoutError):
            await grep.grep_search(
                self.repo, "x", executable=str(executable), timeout=0.1
            )

        pid = int(pid_file.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_invalid_utf8_line_and_path_decode_with_replacement(self):
        invalid_name = os.fsdecode(b"bad-\xff.py")
        file_path = self.repo / invalid_name
        file_path.write_bytes(b"\xffalpha\n")

        result = await grep.grep_search(
            self.repo, "alpha", path=invalid_name, flags=["-a"]
        )

        self.assertEqual(
            result["matches"],
            [{"path": "bad-\ufffd.py", "line": 1, "text": "\ufffdalpha"}],
        )

    async def test_timeout_kills_child_when_leader_exited_but_pipe_is_open(self):
        pid_file = self.repo / "pipe-holder.pid"
        child_code = (
            "import os, time; "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "time.sleep(30)"
        )
        executable = self.repo / "leader-exits-after-spawn"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        )
        executable.chmod(0o755)
        child_pid = None
        try:
            with self.assertRaises(TimeoutError):
                await grep.grep_search(
                    self.repo, "x", executable=str(executable), timeout=0.1
                )
            child_pid = int(pid_file.read_text())
            deadline = time.monotonic() + 1
            while _process_is_running(child_pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            self.assertFalse(_process_is_running(child_pid))
        finally:
            if child_pid is not None and _process_is_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)


    async def test_timeout_kills_sigterm_ignoring_child_after_leader_exits(self):
        pid_file = self.repo / "stubborn.pid"
        child_code = (
            "import os, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "time.sleep(30)"
        )
        executable = self.repo / "leader-exits-stubborn-child"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
            "time.sleep(0.05)\n"
        )
        executable.chmod(0o755)
        child_pid = None
        try:
            with self.assertRaises(TimeoutError):
                await grep.grep_search(
                    self.repo, "x", executable=str(executable), timeout=0.2
                )
            child_pid = int(pid_file.read_text())
            deadline = time.monotonic() + 2
            while _process_is_running(child_pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            self.assertFalse(_process_is_running(child_pid))
        finally:
            if child_pid is not None and _process_is_running(child_pid):
                os.kill(child_pid, signal.SIGKILL)


def _process_is_running(pid):
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        stat = stat_path.read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(") ", 1)[1][0] != "Z"
