"""Subprocess tests for the console-script launcher."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")

FAKE_CLI = """
import sys, types
fake = types.ModuleType("auggie_mcp.cli")
def main():
    import auggie.search
    print(auggie.search.RRF_K)
    return 0
fake.main = main
sys.modules["auggie_mcp.cli"] = fake
"""

FAKE_SERVER = """
import sys, types
fake = types.ModuleType("auggie_mcp.server")
def run():
    import auggie.search
    print(auggie.search.RRF_K)
fake.run = run
sys.modules["auggie_mcp.server"] = fake
"""


class LaunchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        (self.home / ".auggie").mkdir()
        self.config = self.home / ".auggie" / "config.toml"

    def run_py(self, code, *args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("AUGGIE_")}
        env["HOME"] = str(self.home)
        env["PYTHONPATH"] = SRC
        return subprocess.run(
            [sys.executable, "-c", code, *args],
            env=env,
            cwd=self.home,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_cli_file_value_reaches_rrf_k(self):
        self.config.write_text("[search]\nrrf_k = 7\n")
        proc = self.run_py(FAKE_CLI + "import sys; from auggie_mcp import launch; sys.exit(launch.cli())")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "7")

    def test_server_file_value_reaches_rrf_k(self):
        self.config.write_text("[search]\nrrf_k = 7\n")
        proc = self.run_py(FAKE_SERVER + "from auggie_mcp import launch; launch.server()")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "7")

    def test_cli_unknown_section_exits_2(self):
        self.config.write_text("[bogus]\nx = 1\n")
        proc = self.run_py("import sys; from auggie_mcp import launch; sys.exit(launch.cli())")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertTrue(proc.stderr.startswith("auggie: "), proc.stderr)

    def test_server_unknown_section_exits_2(self):
        self.config.write_text("[bogus]\nx = 1\n")
        proc = self.run_py("from auggie_mcp import launch; launch.server()")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertTrue(proc.stderr.startswith("auggie: "), proc.stderr)

    def test_import_does_not_load_auggie(self):
        proc = self.run_py(
            "import sys, auggie_mcp.launch;"
            "print([m for m in sys.modules if m == 'auggie' or m.startswith('auggie.')])"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "[]")

    def test_cli_help_without_config_exits_0(self):
        proc = self.run_py(
            "import sys; from auggie_mcp import launch; sys.argv = ['auggie', '--help'];"
            "sys.exit(launch.cli())"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
