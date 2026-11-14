from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_versions.py"
VERSION_FILES = (
    "package.json",
    ".claude-plugin/marketplace.json",
    "plugin/.claude-plugin/plugin.json",
    "plugin/.codex-plugin/plugin.json",
    "plugin/.antigravity-plugin/plugin.json",
)


def test_mismatched_manifest_exits_one_and_names_file(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "auggie-mcp"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    for path in VERSION_FILES:
        manifest = tmp_path / path
        manifest.parent.mkdir(parents=True, exist_ok=True)
        data = (
            {"plugins": [{"version": "1.2.3"}]}
            if path == ".claude-plugin/marketplace.json"
            else {"version": "1.2.3"}
        )
        manifest.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / VERSION_FILES[0]).write_text(
        json.dumps({"version": "9.9.9"}), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "package.json" in result.stdout


def test_matching_versions_exit_zero(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "auggie-mcp"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    for path in VERSION_FILES:
        manifest = tmp_path / path
        manifest.parent.mkdir(parents=True, exist_ok=True)
        data = (
            {"plugins": [{"version": "1.2.3"}]}
            if path == ".claude-plugin/marketplace.json"
            else {"version": "1.2.3"}
        )
        manifest.write_text(json.dumps(data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
