"""Coding-agent CLI adapter for codebase retrieval.

Runs one noninteractive agent CLI per invocation through ``coding_agent_cli``
against a caller-selected directory and returns the agent's final response.
The query is sent unchanged. Driver, model, reasoning, extra arguments,
executable, and timeout come from ``AUGGIE_AGENT_*`` environment variables,
then the ``[agent]`` table in ``~/.auggie/config.toml``, then built-in defaults.
"""

import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

from coding_agent_cli import Outcome, Request, run

CONFIG_PATH = Path("~/.auggie/config.toml")

# Thirty-minute per-invocation timeout (seconds).
TIMEOUT_SECONDS = 1800.0

DEFAULT_DRIVER = "pi"

# Set in the agent's environment so a retrieval started inside it is refused.
NESTED_ENV = "AUGGIE_RETRIEVAL_ACTIVE"

# Defaults applied only when the resolved driver is ``pi``.
_PI_DEFAULTS = {
    "model": "gpt-6-luna",
    "reasoning": "high",
    "args": (
        "--approve",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
    ),
}

_KEYS = ("driver", "model", "reasoning", "args", "executable", "timeout_s")


class AgentError(Exception):
    """The agent CLI run did not succeed."""

    def __init__(self, message, outcome=None, returncode=None, detail=""):
        super().__init__(message)
        self.outcome = outcome
        self.returncode = returncode
        self.detail = detail


@dataclass(frozen=True)
class AgentConfig:
    driver: str = DEFAULT_DRIVER
    model: str | None = None
    reasoning: str | None = None
    args: tuple[str, ...] = ()
    executable: str | None = None
    timeout_s: float = TIMEOUT_SECONDS


def _read_file(path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            table = tomllib.load(fh).get("agent", {})
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid agent config {path}: {exc}") from exc
    if not isinstance(table, dict):
        raise ValueError(f"invalid agent config {path}: [agent] must be a table")
    unknown = set(table) - set(_KEYS)
    if unknown:
        raise ValueError(
            f"invalid agent config {path}: unknown keys {sorted(unknown)}"
        )
    return table


def _read_env(env) -> dict:
    values = {}
    for key in _KEYS:
        raw = env.get(f"AUGGIE_AGENT_{key.upper()}")
        if raw:
            values[key] = shlex.split(raw) if key == "args" else raw
    return values


def load_config(env=None, path=CONFIG_PATH) -> AgentConfig:
    """Resolve the agent config. Environment overrides file; file overrides defaults."""
    values = _read_file(path) | _read_env(os.environ if env is None else env)
    driver = values.get("driver", DEFAULT_DRIVER)
    if driver == "pi":
        values = _PI_DEFAULTS | values
    args = values.get("args", ())
    if isinstance(args, str) or not all(isinstance(a, str) for a in args):
        raise ValueError("agent args must be a list of strings")
    try:
        timeout_s = float(values.get("timeout_s", TIMEOUT_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid agent timeout_s: {values['timeout_s']!r}") from exc
    return AgentConfig(
        driver=driver,
        model=values.get("model"),
        reasoning=values.get("reasoning"),
        args=tuple(args),
        executable=values.get("executable"),
        timeout_s=timeout_s,
    )


def validate_repository(path) -> Path:
    """Resolve ``path`` to an absolute directory or raise ``ValueError``.

    Symlinks are followed so the returned path is the real target directory.
    Missing paths and non-directories fail before any process creation.
    """
    resolved = Path(os.path.realpath(os.fspath(path)))
    if not resolved.exists():
        raise ValueError(f"repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"repository path is not a directory: {resolved}")
    return resolved


async def run_agent(repository, query, *, config=None, executable=None) -> str:
    """Run the configured agent CLI in ``repository`` with ``query`` and return its answer.

    ``executable`` overrides the configured executable. Cancellation kills the
    agent's process tree before ``CancelledError`` propagates.
    """
    if os.environ.get(NESTED_ENV):
        raise ValueError("nested codebase_retrieval refused: already inside a retrieval agent")
    repo = validate_repository(repository)
    config = config or load_config()
    try:
        result = await run(
            Request(
                driver=config.driver,
                prompt=query,
                cwd=repo,
                model=config.model,
                reasoning=config.reasoning,
                args=config.args,
                timeout_s=config.timeout_s,
                executable=executable or config.executable,
                env={NESTED_ENV: "1"},
            )
        )
    except KeyError as exc:
        raise ValueError(f"unknown agent driver: {config.driver}") from exc
    if result.outcome != Outcome.SUCCESS:
        error = result.error
        detail = f"{error.code}: {error.message}" if error else ""
        message = f"{config.driver} run ended with outcome {result.outcome}"
        raise AgentError(
            f"{message}: {detail}" if detail else message,
            outcome=result.outcome,
            returncode=result.returncode,
            detail=detail,
        )
    return result.text
