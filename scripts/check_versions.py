from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

VERSION_FILES = (
    "package.json",
    ".claude-plugin/marketplace.json",
    "plugin/.claude-plugin/plugin.json",
    "plugin/.codex-plugin/plugin.json",
    "plugin/.antigravity-plugin/plugin.json",
)


def _versions(path: str, manifest: dict[str, Any]) -> list[str]:
    if path == ".claude-plugin/marketplace.json":
        return [plugin["version"] for plugin in manifest["plugins"]]
    return [manifest["version"]]


def check_versions(root: Path) -> list[str]:
    with (root / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    expected = project["project"]["version"]
    mismatches = []
    for relative_path in VERSION_FILES:
        with (root / relative_path).open(encoding="utf-8") as file:
            manifest = json.load(file)
        if any(version != expected for version in _versions(relative_path, manifest)):
            mismatches.append(relative_path)
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    mismatches = check_versions(root)
    for path in mismatches:
        print(f"Version mismatch: {path}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
