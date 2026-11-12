"""Tests for the coding-agent CLI adapter."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from coding_agent_cli import Outcome

from auggie_mcp import agent_runner

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
        self.assertEqual(config.driver, "pi")
        self.assertEqual(config.model, "gpt-6-luna")
        self.assertEqual(config.reasoning, "high")
        self.assertIn("--approve", config.args)
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
        self.assertEqual(config.reasoning, "high")

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
            env={}, path=self.repo / "missing.toml"
        )

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

    def test_missing_repository_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                agent_runner.run_agent(
                    self.repo / "missing", "q", config=self.config
                )
            )


if __name__ == "__main__":
    unittest.main()
