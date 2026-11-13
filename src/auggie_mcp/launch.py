"""Console-script launchers that apply ``~/.auggie/config.toml`` before ``auggie`` loads."""

import sys

from auggie_mcp import settings


def _apply() -> None:
    try:
        settings.apply()
    except settings.ConfigError as exc:
        print(f"auggie: {exc}", file=sys.stderr)
        sys.exit(2)


def server() -> None:
    _apply()
    from auggie_mcp.server import run

    run()


def cli() -> int:
    _apply()
    from auggie_mcp.cli import main

    return main()
