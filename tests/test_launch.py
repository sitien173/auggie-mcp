"""Launcher tests for single-load config handoff."""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent_cli import Outcome, Result
from mcp import Client
from mcp.types import Implementation

from auggie_mcp import config, launch, retrieval, server

SRC = str(Path(__file__).resolve().parent.parent / "src")


class LaunchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        (self.home / ".auggie").mkdir()

    def run_py(self, code):
        env = {key: value for key, value in os.environ.items() if not key.startswith("AUGGIE_")}
        env.update(HOME=str(self.home), PYTHONPATH=SRC)
        return subprocess.run(
            [sys.executable, "-c", code], cwd=self.home, env=env, check=False,
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
        )

    def test_cli_loads_once_and_passes_config(self):
        proc = self.run_py(
            "import sys,types; from auggie_mcp import launch,config; "
            "count=[0]; orig=launch.config.load; "
            "launch.config.load=lambda: (count.__setitem__(0,count[0]+1) or orig()); "
            "fake=types.ModuleType('auggie_mcp.cli'); "
            "fake.main=lambda config=None: print(config.agent.timeout_s) or 0; "
            "sys.modules['auggie_mcp.cli']=fake; launch.cli(); print(count[0])"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), ["1800.0", "1"])

    def test_server_loads_once_and_passes_config(self):
        proc = self.run_py(
            "import sys,types; from auggie_mcp import launch; "
            "count=[0]; orig=launch.config.load; "
            "launch.config.load=lambda: (count.__setitem__(0,count[0]+1) or orig()); "
            "fake=types.ModuleType('auggie_mcp.server'); "
            "fake.run=lambda cfg=None: print(cfg.agent.timeout_s); "
            "sys.modules['auggie_mcp.server']=fake; launch.server(); print(count[0])"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), ["1800.0", "1"])

    def test_invalid_config_exits_two(self):
        (self.home / ".auggie" / "config.toml").write_text("[bogus]\nx = 1\n")
        proc = self.run_py("from auggie_mcp import launch; launch.cli()")
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(proc.stderr.startswith("auggie: "))

    def test_server_invalid_config_exits_two(self):
        (self.home / ".auggie" / "config.toml").write_text("[bogus]\nx = 1\n")
        proc = self.run_py("from auggie_mcp import launch; launch.server()")
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(proc.stderr.startswith("auggie: "))

    def test_cli_help_through_launcher_lists_supported_commands(self):
        proc = self.run_py(
            "import sys; from auggie_mcp import launch; sys.argv=['auggie','--help']; launch.cli()"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("{index,status,stats}", proc.stdout)
        self.assertNotIn("retrieve", proc.stdout)
        self.assertNotIn("search", proc.stdout)

    def test_real_launcher_config_reaches_multiple_tool_calls(self):
        repository = self.home / "repository"
        repository.mkdir()
        (repository / "hello.txt").write_text("hello world\n")
        telemetry_dir = self.home / "configured-events"
        cfg = config.Config(
            agent=config.AgentConfig(driver="pi"),
            telemetry=config.TelemetryConfig(dir=telemetry_dir),
        )
        load_count = [0]

        def load_config():
            load_count[0] += 1
            return cfg

        def run_server(received_config):
            self.assertIs(received_config, cfg)
            srv = server.create_server(
                app_config=received_config,
                repository_root=repository,
            )

            async def calls():
                async with Client(
                    srv,
                    mode="2026-07-28",
                    client_info=Implementation(name="launcher-test", version="1"),
                ) as client:
                    await client.call_tool("index_status", {
                        "repository_path": str(repository)
                    })
                    await client.call_tool("codebase_retrieval", {
                        "repository_path": str(repository), "query": "q"
                    })

            asyncio.run(calls())

        semantic_configs = []

        def semantic_search(*args, **kwargs):
            semantic_configs.append(kwargs.get("config"))
            return {"hits": [], "index": {}, "truncated": False, "archive_path": None}

        async def agent_run(request):
            return Result(outcome=Outcome.SUCCESS, text="answer")

        with patch.object(launch.config, "load", side_effect=load_config), patch.object(
            server, "run", side_effect=run_server
        ), patch.object(retrieval, "semantic_search", side_effect=semantic_search), patch.object(
            retrieval, "run", side_effect=agent_run
        ):
            launch.server()

        self.assertEqual(load_count, [1])
        self.assertEqual(semantic_configs, [cfg])
        event_files = list(telemetry_dir.glob("events-*.jsonl"))
        self.assertEqual(len(event_files), 1)
        self.assertEqual(len(event_files[0].read_text().splitlines()), 2)

    def test_import_does_not_import_legacy_auggie(self):
        proc = self.run_py(
            "import sys,auggie_mcp.launch; "
            "print([name for name in sys.modules if name == 'auggie' or name.startswith('auggie.')])"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")
