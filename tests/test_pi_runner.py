"""Unit tests for auggie_mcp.pi_runner."""

import asyncio
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

from auggie_mcp import pi_runner

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)


class PiRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "hello.txt").write_text("hello world\n")
        # leader PIDs observed across tests, for hard cleanup
        self._leader_pids = []
        self.addCleanup(self._kill_lingering_leaders)

    def _kill_lingering_leaders(self):
        for pid in self._leader_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # defensive: kill any fake that outlived its test
        for child in Path(tempfile.gettempdir()).glob("*auggie*"):
            pass

    def run_async(self, coro):
        return asyncio.run(coro)

    # ---- Task 2: packaged system prompt -------------------------------

    def test_prompt_loads_from_installed_package(self):
        text = pi_runner.load_system_prompt()
        self.assertIn("Retrieval Assistant", text)
        self.assertIn("Evidence:", text)
        self.assertIn("Never modify", text)

    # ---- Task 3: path validation --------------------------------------

    def test_missing_path_fails_without_starting_pi(self):
        missing = Path(self.tmp.name) / "does-not-exist"
        with self.assertRaises(ValueError):
            pi_runner.validate_repository(missing)

    def test_file_path_fails_as_non_directory(self):
        with self.assertRaises(ValueError):
            pi_runner.validate_repository(self.repo / "hello.txt")

    def test_relative_path_resolves_to_absolute(self):
        cwd = Path.cwd()
        try:
            os.chdir(self.tmp.name)
            resolved = pi_runner.validate_repository("repo")
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved, self.repo)
        finally:
            os.chdir(cwd)

    def test_absolute_path_returns_absolute(self):
        resolved = pi_runner.validate_repository(self.repo)
        self.assertEqual(resolved, self.repo)

    def test_symlink_to_directory_resolves_to_target(self):
        target = Path(self.tmp.name) / "real-repo"
        target.mkdir()
        (target / "f.txt").write_text("x")
        link = Path(self.tmp.name) / "link-repo"
        link.symlink_to(target, target_is_directory=True)
        resolved = pi_runner.validate_repository(link)
        self.assertEqual(resolved, target)

    def test_symlink_to_file_fails_as_non_directory(self):
        target = Path(self.tmp.name) / "f.txt"
        target.write_text("x")
        link = Path(self.tmp.name) / "f-link"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            pi_runner.validate_repository(link)

    # ---- Task 3: argument construction --------------------------------

    def test_pi_receives_all_confirmed_flags_and_query(self):
        prompt = pi_runner.load_system_prompt()
        args = pi_runner.build_pi_command(self.repo, "what does hello.txt say", prompt)
        expected_flags = [
            "--model",
            "deepseek/deepseek-v4-flash",
            "--approve",
            "--print",
            "--no-session",
            "--mode",
            "text",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--system-prompt",
            prompt,
            "what does hello.txt say",
        ]
        self.assertEqual(args, expected_flags)

    def test_no_tools_restriction(self):
        args = pi_runner.build_pi_command(self.repo, "q", "prompt")
        self.assertNotIn("--tools", args)
        self.assertNotIn("--exclude-tools", args)
        self.assertNotIn("--no-tools", args)

    def test_no_shell_invocation(self):
        # fake_pi.py is an executable python script; Popen execs it directly,
        # never through a shell.
        cmd = [str(FAKE_PI)] + pi_runner.build_pi_command(self.repo, "q", "prompt")
        self.assertTrue(cmd[0].endswith("fake_pi.py"))
        self.assertNotIn("shell=True", repr(cmd))

    def test_query_passed_unchanged(self):
        # The fake returns exactly the query argument it received as its final
        # answer; a shell would split or interpret the metacharacters.
        query = "list  files  in  src; echo pwned > /tmp/x"
        result = self.run_async(
            pi_runner.run_pi(
                self.repo,
                query,
                "prompt",
                executable=str(FAKE_PI),
                extra_args=["--echo-query"],
            )
        )
        self.assertEqual(result, query)
        self.assertFalse(Path("/tmp/x").exists())

    # ---- Task 3: success path -----------------------------------------

    def test_success_returns_final_assistant_text(self):
        result = self.run_async(
            pi_runner.run_pi(
                self.repo,
                "what does hello.txt say",
                pi_runner.load_system_prompt(),
                executable=str(FAKE_PI),
            )
        )
        self.assertEqual(result, "Fake final answer.")

    def test_oversized_single_line_returns_without_reader_limit(self):
        result = self.run_async(
            pi_runner.run_pi(
                self.repo,
                "q",
                "prompt",
                executable=str(FAKE_PI),
                extra_args=["--stdout-huge"],
            )
        )
        self.assertEqual(result, "x" * 70_000)

    # ---- Task 3: failures ---------------------------------------------

    def test_missing_pi_executable_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable="/definitely/missing/pi",
                )
            )

    def test_nonzero_exit_includes_truncated_stderr(self):
        with self.assertRaises(pi_runner.PiExecutionError) as cm:
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--fail-at", "2"],
                )
            )
        self.assertIn("fake pi failure", cm.exception.stderr)
        self.assertLessEqual(len(cm.exception.stderr), pi_runner.STDERR_LIMIT)

    def test_stderr_retains_exact_bytes(self):
        """Exact stderr bytes survive the failure path (no .strip())."""
        payload = "  padded stderr  \n  \n"
        with self.assertRaises(pi_runner.PiExecutionError) as cm:
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--stderr-payload", payload],
                )
            )
        self.assertEqual(cm.exception.stderr, payload)

    def test_missing_final_response_raises(self):
        with self.assertRaises(pi_runner.PiParseError):
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--no-final"],
                )
            )

    def test_empty_output_raises(self):
        # --empty emits valid JSON events but never an assistant response.
        with self.assertRaises(pi_runner.PiParseError):
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--empty"],
                )
            )

    # ---- Task 3: bounded output ---------------------------------------

    def test_stderr_truncated(self):
        try:
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--fail-at", "2"],
                )
            )
            self.fail("expected PiExecutionError")
        except pi_runner.PiExecutionError as e:
            self.assertLessEqual(len(e.stderr), pi_runner.STDERR_LIMIT)

    def test_stderr_retained_buffer_bounded_after_oversized_input(self):
        """The retained stderr buffer never exceeds STDERR_LIMIT bytes."""
        async def drain():
            reader = asyncio.StreamReader()
            reader.feed_data(b"z" * 32768)  # 4x STDERR_LIMIT
            reader.feed_eof()
            chunks = []
            await pi_runner._drain_bounded(reader, chunks, pi_runner.STDERR_LIMIT)
            return chunks

        chunks = self.run_async(drain())
        retained = sum(len(c) for c in chunks)
        self.assertLessEqual(retained, pi_runner.STDERR_LIMIT)
        self.assertEqual(pi_runner._join_bounded(chunks, pi_runner.STDERR_LIMIT),
                         b"z" * pi_runner.STDERR_LIMIT)

    def test_oversized_stderr_from_pi_is_bounded(self):
        """A Pi that floods stderr must not retain more than STDERR_LIMIT bytes."""
        with self.assertRaises(pi_runner.PiExecutionError) as cm:
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--stderr-huge"],
                )
            )
        self.assertEqual(len(cm.exception.stderr), pi_runner.STDERR_LIMIT)
        self.assertEqual(cm.exception.stderr, "z" * pi_runner.STDERR_LIMIT)

    # ---- Task 3: timeout and cancellation -----------------------------

    def _spawn_stalling_fake(self, mode="--stall"):
        """Start a stalling fake; return (runner_coro, tracked_pids, cleanup).

        ``tracked_pids`` is a mutable list the fake fills with the PIDs of
        processes that must be gone after termination: the fake leader and, in
        ``--child-stall`` mode, the SIGTERM-ignoring grandchild. The fake writes
        these to ``child.pid`` in its cwd (the repo); the leader PID is always
        first, so tests may safely index [0] even if the grandchild line raced.
        """
        tracked = []
        coro = pi_runner.run_pi(
            self.repo,
            "q",
            "prompt",
            executable=str(FAKE_PI),
            extra_args=[mode],
        )

        def fill_tracked():
            pid_file = self.repo / "child.pid"
            try:
                raw = pid_file.read_text().split()
            except OSError:
                return
            tracked.extend(int(p) for p in raw if p.strip().isdigit())
            # leader PID is always written first by the fake
            if tracked and tracked[0] not in self._leader_pids:
                self._leader_pids.append(tracked[0])

        def poll_filled(timeout=3.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                fill_tracked()
                if tracked:
                    return True
                time.sleep(0.02)
            return False

        def cleanup():
            fill_tracked()
            for pid in tracked:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        return coro, tracked, cleanup, poll_filled

    def _poll_gone(self, pids, timeout=6.0):
        """Wait until every pid is gone; return the still-alive subset."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            alive = [p for p in pids if _pid_alive(p)]
            if not alive:
                return []
            time.sleep(0.05)
        return alive

    def test_timeout_terminates_process_group(self):
        coro, tracked, cleanup, poll_filled = self._spawn_stalling_fake()
        self.addCleanup(cleanup)
        start = time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            self.run_async(
                asyncio.wait_for(coro, timeout=1.0)
            )
        self.assertLess(time.monotonic() - start, 5.0)
        # the fake leader wrote its own PID; it must be gone after termination
        tracked.clear()
        self.assertTrue(poll_filled())
        self.assertEqual(self._poll_gone(tracked), [])

    def test_timeout_via_adapter_terminates_process_group(self):
        """The adapter's own timeout= kwarg kills the process group."""
        coro, tracked, cleanup, poll_filled = self._spawn_stalling_fake()
        coro.close()
        self.addCleanup(cleanup)
        start = time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            self.run_async(
                pi_runner.run_pi(
                    self.repo,
                    "q",
                    "prompt",
                    executable=str(FAKE_PI),
                    extra_args=["--stall"],
                    timeout=1.0,
                )
            )
        self.assertLess(time.monotonic() - start, 5.0)
        tracked.clear()
        self.assertTrue(poll_filled())
        self.assertEqual(self._poll_gone(tracked), [])

    def test_timeout_kills_sigterm_ignoring_grandchild(self):
        """A grandchild that ignores SIGTERM must be SIGKILLed with the group."""
        coro, tracked, cleanup, poll_filled = self._spawn_stalling_fake(mode="--child-stall")
        self.addCleanup(cleanup)
        start = time.monotonic()
        with self.assertRaises(asyncio.TimeoutError):
            self.run_async(
                asyncio.wait_for(coro, timeout=1.0)
            )
        self.assertLess(time.monotonic() - start, 5.0)
        # the leader exits on SIGTERM; the ignoring grandchild must still be
        # SIGKILLed. Both must be gone.
        tracked.clear()
        self.assertTrue(poll_filled())
        self.assertEqual(self._poll_gone(tracked), [])

    def test_cancellation_kills_sigterm_ignoring_grandchild(self):
        """Cancellation must SIGKILL the SIGTERM-ignoring grandchild too."""
        coro, tracked, cleanup, poll_filled = self._spawn_stalling_fake(mode="--child-stall")
        self.addCleanup(cleanup)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(coro)

        async def cancel_after():
            await asyncio.sleep(0.5)
            task.cancel()

        loop.run_until_complete(
            asyncio.gather(cancel_after(), task, return_exceptions=True)
        )
        self.assertTrue(task.cancelled())
        loop.close()
        tracked.clear()
        self.assertTrue(poll_filled())
        self.assertEqual(self._poll_gone(tracked), [])

    def test_adapter_default_timeout_is_thirty_minutes(self):
        import inspect
        sig = inspect.signature(pi_runner.run_pi)
        self.assertEqual(sig.parameters["timeout"].default, 1800)
        self.assertEqual(pi_runner.TIMEOUT_SECONDS, 1800)

    def test_unlimited_parallel_calls(self):
        async def main():
            results = await asyncio.gather(
                *[
                    pi_runner.run_pi(
                        self.repo,
                        f"query {i}",
                        "prompt",
                        executable=str(FAKE_PI),
                    )
                    for i in range(10)
                ]
            )
            return results

        results = self.run_async(main())
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r == "Fake final answer." for r in results))

    @unittest.skipUnless(
        os.environ.get("AUGGIE_LIVE_PI") == "1",
        "set AUGGIE_LIVE_PI=1 to run the live-Pi test",
    )
    def test_live_pi_real_command(self):
        """Real pi (0.80.10) accepts the confirmed flags and answers."""
        result = self.run_async(
            pi_runner.run_pi(
                self.repo,
                "What does hello.txt contain? Answer with exactly: FOUND",
                pi_runner.load_system_prompt(),
            )
        )
        self.assertIn("FOUND", result)

    def test_real_pi_would_receive_exact_flags(self):
        """Build the exact command the adapter would run for real pi."""
        prompt = pi_runner.load_system_prompt()
        cmd = ["pi"] + pi_runner.build_pi_command(self.repo, "q", prompt)
        self.assertEqual(cmd[0], "pi")
        self.assertIn("--approve", cmd)
        self.assertIn("--print", cmd)
        self.assertIn("--no-session", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "text")
        self.assertEqual(cmd[cmd.index("--system-prompt") + 1], prompt)
        self.assertEqual(cmd[-1], "q")


def _pid_alive(pid):
    """Return True if the process ``pid`` is still alive."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    unittest.main()
