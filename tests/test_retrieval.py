"""Tests for the coding-agent CLI adapter."""

import asyncio
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

from coding_agent_cli import Outcome, Result

from auggie_mcp import config, retrieval

NO_HITS = {"hits": [], "index": {}, "truncated": False, "archive_path": None}

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)


class RunAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.config = retrieval.AgentConfig(driver="pi")

        patcher = patch.object(retrieval, "semantic_search", return_value=NO_HITS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_agent(self, **kwargs):
        kwargs.setdefault("config", self.config)
        return asyncio.run(retrieval.run_agent(self.repo, "query", **kwargs))

    def test_success_returns_final_text(self):
        result = self.run_agent(executable=str(FAKE_PI))
        self.assertEqual(result["answer"], "Fake final answer.")
        self.assertEqual(result["driver"], "pi")
        self.assertIsInstance(result["duration_ms"], int)
        self.assertFalse(result["repo_modified"])
        self.assertEqual(
            set(result),
            {"answer", "driver", "duration_ms", "repo_modified", "git_check"},
        )
        self.assertEqual(result["git_check"], "skipped")

    def test_missing_executable_raises_agent_error(self):
        with self.assertRaises(retrieval.AgentError) as cm:
            self.run_agent(executable=str(self.repo / "no-such-pi"))
        self.assertEqual(cm.exception.outcome, Outcome.AGENT_FATAL)
        self.assertIn("missing_cli", str(cm.exception))

    def test_unknown_driver_raises_value_error(self):
        config = retrieval.AgentConfig(driver="nope")
        with self.assertRaises(ValueError):
            self.run_agent(config=config)

    def test_missing_driver_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "no agent driver configured"):
            self.run_agent(config=retrieval.AgentConfig())

    def test_escalating_configured_flag_is_rejected_before_agent_starts(self):
        for args in (
            ("--sandbox=danger-full-access",),
            ("--sandbox", "workspace-write"),
            ("-s", "danger-full-access"),
        ):
            with self.subTest(args=args):
                config = retrieval.AgentConfig(driver="codex", args=args)
                with patch.object(retrieval, "run") as run, self.assertRaisesRegex(
                    ValueError, "not allowed"
                ):
                    self.run_agent(config=config)
                run.assert_not_awaited()

    def test_codex_configured_mcp_servers_are_disabled_before_agent_starts(self):
        listed = subprocess.CompletedProcess(
            ["codex", "mcp", "list", "--json"],
            0,
            stdout=b'[{"name":"test-server","enabled":true}]',
            stderr=b"",
        )
        git_not_repo = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")
        requests = []

        async def fake_run(request):
            requests.append(request)
            return Result(outcome=Outcome.SUCCESS, text="ok")

        def fake_subprocess_run(args, **kwargs):
            if args[:3] == ["codex", "mcp", "list"]:
                self.assertEqual(kwargs["cwd"], self.repo)
                return listed
            return git_not_repo

        with (
            patch.object(retrieval, "run", fake_run),
            patch.object(retrieval.subprocess, "run", side_effect=fake_subprocess_run),
        ):
            self.run_agent(config=retrieval.AgentConfig(driver="codex"))

        self.assertIn(
            ("--config", "mcp_servers.test-server.enabled=false"),
            list(zip(requests[0].args, requests[0].args[1:])),
        )

    def test_codex_mcp_listing_failure_refuses_to_start_agent(self):
        failed = subprocess.CompletedProcess(
            ["codex", "mcp", "list", "--json"], 1, stdout=b"", stderr=b"failed"
        )
        with (
            patch.object(retrieval, "run") as run,
            patch.object(retrieval.subprocess, "run", return_value=failed),
            self.assertRaisesRegex(retrieval.AgentError, "MCP server listing"),
        ):
            self.run_agent(config=retrieval.AgentConfig(driver="codex"))
        run.assert_not_awaited()

    def test_codex_mcp_listing_timeout_refuses_to_start_agent(self):
        def hang(args, **kwargs):
            if args[1:3] != ["mcp", "list"]:
                return subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")
            self.assertIsNotNone(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        with (
            patch.object(retrieval, "run") as run,
            patch.object(retrieval.subprocess, "run", side_effect=hang),
            self.assertRaisesRegex(retrieval.AgentError, "timed out"),
        ):
            self.run_agent(config=retrieval.AgentConfig(driver="codex"))
        run.assert_not_awaited()

    def test_codex_mcp_listing_uses_configured_executable_off_event_loop(self):
        listed = subprocess.CompletedProcess([], 0, stdout=b"[]", stderr=b"")
        git_not_repo = subprocess.CompletedProcess([], 1, stdout=b"", stderr=b"")
        calls = []

        async def fake_run(request):
            return Result(outcome=Outcome.SUCCESS, text="ok")

        def fake_subprocess_run(args, **kwargs):
            if args[1:3] == ["mcp", "list"]:
                calls.append((args[0], threading.current_thread() is threading.main_thread()))
                return listed
            return git_not_repo

        with (
            patch.object(retrieval, "run", fake_run),
            patch.object(retrieval.subprocess, "run", side_effect=fake_subprocess_run),
        ):
            self.run_agent(
                config=retrieval.AgentConfig(driver="codex", executable="/opt/codex")
            )

        self.assertEqual(calls, [("/opt/codex", False)])

    def test_rejects_all_codex_config_forms_and_attached_forbidden_short_flags(self):
        for args in (
            ("-c", "model=x"),
            ("-cmodel=x",),
            ("--config", "model=x"),
            ("--config=model=x",),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "not allowed"):
                retrieval._validate_args("codex", args)
        for driver, args in (
            ("codex", ("-sdanger-full-access",)),
            ("codex", ("-anever",)),
            ("codex", ("-ppath",)),
            ("pi", ("-epath",)),
            ("pi", ("-tfoo",)),
        ):
            with self.subTest(driver=driver, args=args), self.assertRaisesRegex(ValueError, "not allowed"):
                retrieval._validate_args(driver, args)

    def test_path_hash_does_not_follow_symlinked_parent(self):
        outside = self.repo / "outside"
        outside.mkdir()
        (outside / "file.txt").write_text("secret target contents")
        (self.repo / "link").symlink_to(outside, target_is_directory=True)

        self.assertEqual(retrieval._path_hash(self.repo, "link/file.txt"), "<parent-symlink>")
        regular_file = self.repo / "not-a-directory"
        regular_file.write_text("file")
        self.assertEqual(
            retrieval._path_hash(self.repo, "not-a-directory/child.txt"),
            "<parent-not-directory>",
        )

    def test_path_hash_handles_symlinks_directories_and_special_files(self):
        target = self.repo / "target.txt"
        target.write_text("target")
        link = self.repo / "link.txt"
        link.symlink_to("target.txt")
        directory = self.repo / "directory"
        directory.mkdir()
        fifo = self.repo / "pipe"
        os.mkfifo(fifo)

        with patch.object(retrieval.os, "open", side_effect=AssertionError("opened non-regular file")):
            self.assertEqual(retrieval._path_hash(self.repo, "link.txt"), "target.txt")
            self.assertEqual(retrieval._path_hash(self.repo, "directory"), "dir")
            self.assertEqual(
                retrieval._path_hash(self.repo, "pipe"),
                f"special:{stat.S_IFMT(os.lstat(fifo).st_mode):#o}",
            )

    def test_path_hash_reads_regular_files_in_fixed_size_chunks(self):
        content = b"abc123" * 20000
        path = self.repo / "large.txt"
        path.write_bytes(content)
        read_sizes = []
        original_read = os.read

        def recording_read(fd, size):
            read_sizes.append(size)
            return original_read(fd, size)

        with (
            patch.object(retrieval.os, "read", side_effect=recording_read),
            patch.object(Path, "read_bytes", side_effect=AssertionError("read whole file")),
        ):
            digest = retrieval._path_hash(self.repo, "large.txt")

        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertGreater(len(read_sizes), 1)
        self.assertLessEqual(max(read_sizes), 65536)

    def test_presets_disable_mcp_or_extensions_and_reject_load_overrides(self):
        self.assertIn("--safe-mode", retrieval._PRESETS["claude"])
        self.assertIn("--no-mcp", retrieval._PRESETS["pi"])
        self.assertIn("--no-extensions", retrieval._PRESETS["pi"])
        with self.assertRaisesRegex(ValueError, "--mcp-config"):
            retrieval._validate_args("claude", ("--mcp-config", "servers.json"))
        with self.assertRaisesRegex(ValueError, "--plugin-dir"):
            retrieval._validate_args("claude", ("--plugin-dir=plugins",))
        with self.assertRaisesRegex(ValueError, "-e"):
            retrieval._validate_args("pi", ("-e", "extension.ts"))
        with self.assertRaisesRegex(ValueError, "not allowed"):
            retrieval._validate_args("codex", ("-c", "sandbox_mode=workspace-write"))
        with self.assertRaisesRegex(ValueError, "not allowed"):
            retrieval._validate_args("codex", ("--config=approval_policy=never",))
        with self.assertRaisesRegex(ValueError, "not allowed"):
            retrieval._validate_args("codex", ("-c", "mcp_servers.local={}"))
        with self.assertRaisesRegex(ValueError, "-p"):
            retrieval._validate_args("codex", ("-p", "untrusted"))
        with self.assertRaisesRegex(ValueError, "--extension"):
            retrieval._validate_args("pi", ("--extension=plugin.ts",))

    def test_nested_call_is_refused_before_spawn(self):
        with (
            patch.dict(os.environ, {retrieval.NESTED_ENV: "1"}),
            patch.object(retrieval, "run") as run,
            self.assertRaisesRegex(ValueError, "nested"),
        ):
            self.run_agent(executable=str(FAKE_PI))
        run.assert_not_called()

    def test_child_env_carries_nested_marker(self):
        requests = []

        async def fake_run(request):
            requests.append(request)
            return Result(outcome=Outcome.SUCCESS, text="ok")

        with patch.object(retrieval, "run", fake_run):
            self.assertEqual(self.run_agent()["answer"], "ok")
        self.assertEqual(requests[0].env[retrieval.NESTED_ENV], "1")
        self.assertIn("--tools", requests[0].args)

    def test_agent_write_in_git_repo_raises_repo_modified_error(self):
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "test"], check=True)
        tracked = self.repo / "tracked.txt"
        tracked.write_text("initial\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "initial"], check=True)

        fake_agent = self.repo / "fake-agent"
        fake_agent.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "from pathlib import Path\n"
            "Path('created.txt').write_text('changed\\n')\n"
            "print(json.dumps({'type':'message_end','message':{'role':'assistant',"
            "'content':[{'type':'text','text':'discard me'}]}}))\n"
        )
        fake_agent.chmod(0o755)
        with self.assertRaisesRegex(retrieval.RepoModifiedError, "created.txt"):
            self.run_agent(executable=str(fake_agent))

    def test_changed_dirty_file_content_is_detected_with_unchanged_status(self):
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "test"], check=True)
        tracked = self.repo / "tracked.txt"
        tracked.write_text("clean\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "initial"], check=True)
        tracked.write_text("dirty before\n")
        untracked = self.repo / "untracked.txt"
        untracked.write_text("untracked before\n")
        status_before = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        ).stdout

        async def change_content(request):
            tracked.write_text("dirty after\n")
            untracked.write_text("untracked after\n")
            status_after = subprocess.run(  # noqa: ASYNC221 - test callback samples Git state synchronously.
                ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                cwd=self.repo,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(status_after, status_before)
            return Result(outcome=Outcome.SUCCESS, text="discard me")

        with patch.object(retrieval, "run", change_content), self.assertRaises(
            retrieval.RepoModifiedError
        ) as cm:
            self.run_agent()
        self.assertIn("tracked.txt", str(cm.exception))
        self.assertIn("untracked.txt", str(cm.exception))

    def test_lost_work_tree_after_run_is_rejected(self):
        states = iter(((True, set(), {}), (False, set(), {})))

        async def fake_run(request):
            return Result(outcome=Outcome.SUCCESS, text="discard me")

        with (
            patch.object(retrieval, "_git_state", side_effect=lambda repo: next(states)),
            patch.object(retrieval, "run", fake_run),
            self.assertRaisesRegex(retrieval.RepoModifiedError, "work tree"),
        ):
            self.run_agent()

    def test_importing_cli_with_invalid_config_env_does_not_load_config(self):
        result = subprocess.run(
            [sys.executable, "-c", "import auggie_mcp.cli"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ | {"AUGGIE_DIMENSIONS": "not-an-integer"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_agent_uses_supplied_concurrency_limit(self):
        with patch.object(retrieval, "_RETRIEVAL_SLOTS", None):
            slots = retrieval._retrieval_slots(3)
            self.assertTrue(all(slots.acquire(blocking=False) for _ in range(3)))
            self.assertFalse(slots.acquire(blocking=False))
            for _ in range(3):
                slots.release()
        retrieval._RETRIEVAL_SLOTS = None

    def test_result_marks_non_git_repository_as_skipped(self):
        async def fake_run(request):
            return Result(outcome=Outcome.SUCCESS, text="ok")

        with patch.object(retrieval, "run", fake_run):
            result = self.run_agent()
        self.assertEqual(
            set(result),
            {"answer", "driver", "duration_ms", "repo_modified", "git_check"},
        )
        self.assertEqual(result["git_check"], "skipped")

    def test_busy_when_default_concurrency_limit_is_full(self):
        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()
            started_count = 0

            async def held_run(request):
                nonlocal started_count
                started_count += 1
                if started_count == 2:
                    started.set()
                await release.wait()
                return Result(outcome=Outcome.SUCCESS, text="ok")

            with patch.object(retrieval, "run", held_run):
                first = asyncio.create_task(retrieval.run_agent(self.repo, "one", config=self.config))
                second = asyncio.create_task(retrieval.run_agent(self.repo, "two", config=self.config))
                await started.wait()
                with self.assertRaises(retrieval.RetrievalBusyError) as cm:
                    await asyncio.wait_for(
                        retrieval.run_agent(self.repo, "three", config=self.config),
                        0.1,
                    )
                self.assertTrue(str(cm.exception).startswith("RETRIEVAL_BUSY:"))
                release.set()
                await asyncio.gather(first, second)

        asyncio.run(scenario())

    def test_cancelled_and_timed_out_runs_release_slots(self):
        async def scenario():
            started = asyncio.Event()

            async def held_run(request):
                started.set()
                await asyncio.Event().wait()

            with patch.object(retrieval, "run", held_run):
                task = asyncio.create_task(retrieval.run_agent(self.repo, "cancel", config=self.config))
                await started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            async def timeout_run(request):
                raise TimeoutError

            with patch.object(retrieval, "run", timeout_run), self.assertRaises(
                asyncio.TimeoutError
            ):
                await retrieval.run_agent(self.repo, "timeout", config=self.config)

            async def success_run(request):
                return Result(outcome=Outcome.SUCCESS, text="released")

            with patch.object(retrieval, "run", success_run):
                result = await retrieval.run_agent(self.repo, "released", config=self.config)
                self.assertEqual(result["answer"], "released")

        asyncio.run(scenario())

    def test_missing_repository_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                retrieval.run_agent(
                    self.repo / "missing", "q", config=self.config
                )
            )


class RagPromptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(os.path.realpath(self.tmp.name))
        self.config = retrieval.AgentConfig(driver="pi")
        self.requests = []

    def run_agent(self, search):
        async def fake_run(request):
            self.requests.append(request)
            return Result(outcome=Outcome.SUCCESS, text="ok")

        with (
            patch.object(retrieval, "run", fake_run),
            patch.object(retrieval, "semantic_search", search) as patched,
        ):
            asyncio.run(
                retrieval.run_agent(self.repo, "query", config=self.config)
            )
        return patched

    def hits(self, *chunks):
        return {"hits": [
            {"path": chunk.split(":", 1)[0],
             "start_line": int(chunk.split(":", 1)[1].split("-", 1)[0]),
             "end_line": int(chunk.split(":", 1)[1].split("-", 1)[1].split(" ", 1)[0]),
             "score": float(chunk.split("score=", 1)[1].split("\n", 1)[0])
             if "score=" in chunk else 0.0}
            for chunk in chunks
        ]}

    def assert_no_hints(self):
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            self.requests[0].prompt, retrieval.build_prompt("query", "")
        )

    def test_prompt_orders_system_hints_question(self):
        chunks = ("a.py:1-2 score=0.900\nalpha", "b.py:3-4 score=0.800\nbeta")
        self.run_agent(lambda *a, **k: self.hits(*chunks))
        prompt = self.requests[0].prompt
        hints = "a.py:1-2 score=0.900\n--\nb.py:3-4 score=0.800"
        self.assertEqual(prompt, retrieval.build_prompt("query", hints))
        self.assertTrue(prompt.startswith(retrieval.SYSTEM_PROMPT))
        self.assertIn("## Index hints\n" + hints, prompt)
        self.assertTrue(prompt.endswith("\n\n## Question\nquery"))

    def test_failures_fall_back_to_prompt_without_hints(self):
        def missing_index(*a, **k):
            raise retrieval.IndexMissingError("No index")

        def generic(*a, **k):
            raise RuntimeError("boom")

        for name, search in (
            ("missing_index", missing_index),
            ("generic", generic),
            ("no_hits", lambda *a, **k: NO_HITS),
        ):
            with self.subTest(name):
                self.requests.clear()
                self.run_agent(search)
                self.assert_no_hints()

    def test_timeout_falls_back_to_prompt_without_hints(self):
        def slow(*a, **k):
            time.sleep(0.5)
            return self.hits("a.py:1-2 score=0.900\nalpha")

        with patch.object(retrieval, "RAG_TIMEOUT_S", 0.05):
            self.run_agent(slow)
        self.assert_no_hints()

    def test_chunks_beyond_cap_are_dropped_whole(self):
        chunks = ("a.py:1-2 score=0.9\naaaa", "b.py:1-2 score=0.8\nbbbb", "c.py:1-2 score=0.7\ncccc")
        cap = len(b"a.py:1-2 score=0.900\n--\nb.py:1-2 score=0.800")
        with patch.object(retrieval, "RAG_MAX_BYTES", cap):
            self.run_agent(lambda *a, **k: self.hits(*chunks))
        prompt = self.requests[0].prompt
        self.assertIn("a.py:1-2 score=0.900\n--\nb.py:1-2 score=0.800", prompt)
        self.assertNotIn("c.py", prompt)

    def test_first_chunk_over_cap_gives_no_hints(self):
        with patch.object(retrieval, "RAG_MAX_BYTES", 5):
            self.run_agent(lambda *a, **k: self.hits("a.py:1-2 score=0.9\naaaa"))
        self.assert_no_hints()

    def test_block_containing_separator_line_is_kept_or_dropped_whole(self):
        first = "a.py:1-3 score=0.9\nx\n--\ny"
        second = "b.py:1-2 score=0.8\nbbbb"
        cap = len(b"a.py:1-3 score=0.900")
        with patch.object(retrieval, "RAG_MAX_BYTES", cap):
            self.run_agent(lambda *a, **k: self.hits(first, second))
        prompt = self.requests[0].prompt
        self.assertIn("a.py:1-3 score=0.900\n\n## Question", prompt)
        self.assertNotIn("b.py", prompt)

    def test_cap_smaller_than_separator_block_gives_no_hints(self):
        first = "a.py:1-3 score=0.9\nx\n--\ny"
        with patch.object(retrieval, "RAG_MAX_BYTES", len("a.py:1-3 score=0.900") - 1):
            self.run_agent(lambda *a, **k: self.hits(first))
        self.assert_no_hints()

    def test_run_semantic_call_arguments(self):
        search = unittest.mock.Mock(return_value=NO_HITS)
        patched = self.run_agent(search)
        patched.assert_called_once_with(
            self.repo, "query", path=".", top_k=5, context_lines=0, config=None
        )

    def test_index_hints_receive_loaded_application_config(self):
        app_config = config.Config()
        with patch.object(retrieval, "semantic_search", return_value=NO_HITS) as search:
            asyncio.run(retrieval._index_hints(self.repo, "query", config=app_config))
        self.assertIs(search.call_args.kwargs["config"], app_config)

    def test_nested_refusal_never_searches(self):
        search = unittest.mock.Mock(return_value=NO_HITS)
        with (
            patch.dict(os.environ, {retrieval.NESTED_ENV: "1"}),
            patch.object(retrieval, "semantic_search", search),
            self.assertRaisesRegex(ValueError, "nested"),
        ):
            asyncio.run(
                retrieval.run_agent(self.repo, "query", config=self.config)
            )
        search.assert_not_called()

    def test_cancelled_error_propagates(self):
        def cancelled(*a, **k):
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            self.run_agent(cancelled)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
