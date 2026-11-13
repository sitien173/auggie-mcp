"""Tests for the coding-agent CLI adapter."""

import asyncio
import os
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

from coding_agent_cli import Outcome, Result
from mcp.server.mcpserver.exceptions import ToolError

from auggie_mcp import agent_runner, semantic_search

NO_HITS = {"stdout": "", "blocks": [], "stderr": "", "exit_code": 1, "index": {}}
INDEX = {"stale_modified": 2, "stale_deleted": 1, "last_indexed": "2026-11-01 10:00:00"}
STALENESS = (
    "Index last updated 2026-11-01 10:00:00; "
    "2 indexed files modified and 1 deleted since then."
)

TESTS_DIR = Path(__file__).resolve().parent
FAKE_PI = TESTS_DIR / "fake_pi.py"

if not os.access(FAKE_PI, os.X_OK):
    os.chmod(FAKE_PI, 0o755)


class LoadConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "config.toml"

    def load(self, env=None):
        return agent_runner.load_config(env=env or {}, path=self.path)

    def test_defaults_without_file_or_env(self):
        config = self.load()
        self.assertIsNone(config.driver)
        self.assertIsNone(config.model)
        self.assertIsNone(config.reasoning)
        self.assertEqual(config.args, ())
        self.assertIsNone(config.executable)
        self.assertEqual(config.timeout_s, agent_runner.TIMEOUT_SECONDS)

    def test_file_values_override_defaults(self):
        self.path.write_text(
            '[agent]\nmodel = "m1"\nargs = ["--x"]\ntimeout_s = 5\n'
        )
        config = self.load()
        self.assertEqual(config.model, "m1")
        self.assertEqual(config.args, ("--x",))
        self.assertEqual(config.timeout_s, 5.0)
        self.assertIsNone(config.reasoning)

    def test_env_overrides_file(self):
        self.path.write_text('[agent]\nmodel = "m1"\n')
        config = self.load(
            {"AUGGIE_AGENT_MODEL": "m2", "AUGGIE_AGENT_ARGS": "--a '--b c'"}
        )
        self.assertEqual(config.model, "m2")
        self.assertEqual(config.args, ("--a", "--b c"))

    def test_pi_defaults_skipped_for_other_drivers(self):
        config = self.load({"AUGGIE_AGENT_DRIVER": "claude"})
        self.assertEqual(config.driver, "claude")
        self.assertIsNone(config.model)
        self.assertIsNone(config.reasoning)
        self.assertEqual(config.args, ())

    def test_unknown_file_key_is_rejected(self):
        self.path.write_text('[agent]\nsystem_prompt = "x"\n')
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_toml_is_rejected(self):
        self.path.write_text("[agent\n")
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"AUGGIE_AGENT_TIMEOUT_S": "soon"})

    def test_string_args_in_file_is_rejected(self):
        self.path.write_text('[agent]\nargs = "--x"\n')
        with self.assertRaises(ValueError):
            self.load()


class RunAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.config = agent_runner.load_config(
            env={"AUGGIE_AGENT_DRIVER": "pi"}, path=self.repo / "missing.toml"
        )

        patcher = patch.object(
            semantic_search, "run_semantic", return_value=NO_HITS
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_agent(self, **kwargs):
        kwargs.setdefault("config", self.config)
        return asyncio.run(agent_runner.run_agent(self.repo, "query", **kwargs))

    def test_success_returns_final_text(self):
        self.assertEqual(
            self.run_agent(executable=str(FAKE_PI)), "Fake final answer."
        )

    def test_missing_executable_raises_agent_error(self):
        with self.assertRaises(agent_runner.AgentError) as cm:
            self.run_agent(executable=str(self.repo / "no-such-pi"))
        self.assertEqual(cm.exception.outcome, Outcome.AGENT_FATAL)
        self.assertIn("missing_cli", str(cm.exception))

    def test_unknown_driver_raises_value_error(self):
        config = agent_runner.AgentConfig(driver="nope")
        with self.assertRaises(ValueError):
            self.run_agent(config=config)

    def test_missing_driver_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "no agent driver configured"):
            self.run_agent(config=agent_runner.AgentConfig())

    def test_nested_call_is_refused_before_spawn(self):
        with (
            patch.dict(os.environ, {agent_runner.NESTED_ENV: "1"}),
            patch.object(agent_runner, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "nested"):
                self.run_agent(executable=str(FAKE_PI))
        run.assert_not_called()

    def test_child_env_carries_nested_marker(self):
        requests = []

        async def fake_run(request):
            requests.append(request)
            return Result(outcome=Outcome.SUCCESS, text="ok")

        with patch.object(agent_runner, "run", fake_run):
            self.assertEqual(self.run_agent(), "ok")
        self.assertEqual(requests[0].env[agent_runner.NESTED_ENV], "1")

    def test_missing_repository_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                agent_runner.run_agent(
                    self.repo / "missing", "q", config=self.config
                )
            )


class RagPromptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(os.path.realpath(self.tmp.name))
        self.config = agent_runner.load_config(
            env={"AUGGIE_AGENT_DRIVER": "pi"}, path=self.repo / "missing.toml"
        )
        self.requests = []

    def run_agent(self, search):
        async def fake_run(request):
            self.requests.append(request)
            return Result(outcome=Outcome.SUCCESS, text="ok")

        with (
            patch.object(agent_runner, "run", fake_run),
            patch.object(semantic_search, "run_semantic", search) as patched,
        ):
            asyncio.run(
                agent_runner.run_agent(self.repo, "query", config=self.config)
            )
        return patched

    def hits(self, *chunks):
        return {
            "stdout": "\n--\n".join(chunks),
            "blocks": list(chunks),
            "stderr": "",
            "exit_code": 0,
            "index": INDEX,
        }

    def assert_no_hints(self):
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            self.requests[0].prompt, agent_runner.build_prompt("query", "")
        )

    def test_prompt_orders_system_hints_question(self):
        chunks = ("a.py:1-2 score=0.900\nalpha", "b.py:3-4 score=0.800\nbeta")
        self.run_agent(lambda *a, **k: self.hits(*chunks))
        prompt = self.requests[0].prompt
        hints = f"{STALENESS}\n" + "\n--\n".join(chunks)
        self.assertEqual(prompt, agent_runner.build_prompt("query", hints))
        self.assertTrue(prompt.startswith(agent_runner.SYSTEM_PROMPT))
        self.assertIn("## Index hints\n" + STALENESS, prompt)
        self.assertTrue(prompt.endswith("\n\n## Question\nquery"))

    def test_failures_fall_back_to_prompt_without_hints(self):
        def tool_error(*a, **k):
            raise ToolError("No index")

        def generic(*a, **k):
            raise RuntimeError("boom")

        for name, search in (
            ("tool_error", tool_error),
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

        with patch.object(agent_runner, "RAG_TIMEOUT_S", 0.05):
            self.run_agent(slow)
        self.assert_no_hints()

    def test_chunks_beyond_cap_are_dropped_whole(self):
        chunks = ("a.py:1-2 score=0.9\naaaa", "b.py:1-2 score=0.8\nbbbb", "c.py:1-2 score=0.7\ncccc")
        cap = len("\n--\n".join(chunks[:2]).encode())
        with patch.object(agent_runner, "RAG_MAX_BYTES", cap):
            self.run_agent(lambda *a, **k: self.hits(*chunks))
        prompt = self.requests[0].prompt
        self.assertIn(chunks[0] + "\n--\n" + chunks[1], prompt)
        self.assertNotIn("c.py", prompt)

    def test_first_chunk_over_cap_gives_no_hints(self):
        with patch.object(agent_runner, "RAG_MAX_BYTES", 5):
            self.run_agent(lambda *a, **k: self.hits("a.py:1-2 score=0.9\naaaa"))
        self.assert_no_hints()

    def test_block_containing_separator_line_is_kept_or_dropped_whole(self):
        first = "a.py:1-3 score=0.9\nx\n--\ny"
        second = "b.py:1-2 score=0.8\nbbbb"
        cap = len(first.encode())
        with patch.object(agent_runner, "RAG_MAX_BYTES", cap):
            self.run_agent(lambda *a, **k: self.hits(first, second))
        prompt = self.requests[0].prompt
        self.assertIn(f"{STALENESS}\n{first}\n\n## Question", prompt)
        self.assertNotIn("b.py", prompt)

    def test_cap_smaller_than_separator_block_gives_no_hints(self):
        first = "a.py:1-3 score=0.9\nx\n--\ny"
        with patch.object(agent_runner, "RAG_MAX_BYTES", len(first.encode()) - 1):
            self.run_agent(lambda *a, **k: self.hits(first))
        self.assert_no_hints()

    def test_run_semantic_call_arguments(self):
        search = unittest.mock.Mock(return_value=NO_HITS)
        patched = self.run_agent(search)
        patched.assert_called_once_with(
            self.repo, "query", ".", 10, 0, display_root=self.repo
        )

    def test_nested_refusal_never_searches(self):
        search = unittest.mock.Mock(return_value=NO_HITS)
        with (
            patch.dict(os.environ, {agent_runner.NESTED_ENV: "1"}),
            patch.object(semantic_search, "run_semantic", search),
            self.assertRaisesRegex(ValueError, "nested"),
        ):
            asyncio.run(
                agent_runner.run_agent(self.repo, "query", config=self.config)
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
