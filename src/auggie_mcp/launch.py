"""Console-script launchers that apply ``~/.auggie/config.toml`` before ``auggie`` loads."""

import sys

from auggie_mcp import config


def _load_config():
    try:
        return config.load()
    except config.ConfigError as exc:
        print(f"auggie: {exc}", file=sys.stderr)
        sys.exit(2)


def server() -> None:
    cfg = _load_config()
    from auggie_mcp.server import run

    run(cfg)


def cli() -> int:
    cfg = _load_config()
    from auggie_mcp.cli import main

    return main(config=cfg)
