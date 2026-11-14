"""Frozen configuration loader for auggie-mcp."""

from __future__ import annotations

import logging
import os
import shlex
import threading
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("~/.auggie/config.toml")

REMOVED_SECTIONS = frozenset({"search", "confidence", "rerank", "cache", "watch", "jev"})
REMOVED_KEYS = frozenset({("voyage", "chars_per_token")})
_WARNED_REMOVED_KEYS: set[tuple[str, str]] = set()
_REMOVED_WARNING_LOCK = threading.Lock()

_TOML_NAMES = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "float",
    list: "array",
    dict: "table",
}


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "voyage-code-3"
    dimensions: int = 1024
    batch_size: int = 128


@dataclass(frozen=True)
class VoyageConfig:
    timeout_s: float = 120.0
    retries: int = 5


@dataclass(frozen=True)
class ChunkingConfig:
    lines: int = 30
    overlap_lines: int = 15


@dataclass(frozen=True)
class IndexConfig:
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    dir: Path | None = None
    retention_days: int = 30


@dataclass(frozen=True)
class RetrievalConfig:
    max_concurrent: int = 2


@dataclass(frozen=True)
class AgentConfig:
    driver: str | None = None
    model: str | None = None
    reasoning: str | None = None
    args: tuple[str, ...] = ()
    executable: str | None = None
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    voyage: VoyageConfig = field(default_factory=VoyageConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# (section, key) -> (env_var, kind, default)
_SCHEMA: dict[tuple[str, str], tuple[str, str, Any]] = {
    ("embedding", "model"): ("AUGGIE_MODEL", "string", "voyage-code-3"),
    ("embedding", "dimensions"): ("AUGGIE_DIMENSIONS", "integer", 1024),
    ("embedding", "batch_size"): ("AUGGIE_BATCH_SIZE", "integer", 128),
    ("voyage", "timeout_s"): ("AUGGIE_VOYAGE_TIMEOUT", "float", 120.0),
    ("voyage", "retries"): ("AUGGIE_VOYAGE_RETRIES", "integer", 5),
    ("chunking", "lines"): ("AUGGIE_CHUNK_LINES", "integer", 30),
    ("chunking", "overlap_lines"): ("AUGGIE_OVERLAP_LINES", "integer", 15),
    ("index", "timeout_s"): ("AUGGIE_INDEX_TIMEOUT_S", "float", 1800.0),
    ("telemetry", "enabled"): ("AUGGIE_TELEMETRY", "boolean", True),
    ("telemetry", "dir"): ("AUGGIE_TELEMETRY_DIR", "path", None),
    ("telemetry", "retention_days"): ("AUGGIE_TELEMETRY_RETENTION_DAYS", "integer", 30),
    ("retrieval", "max_concurrent"): ("AUGGIE_RETRIEVAL_MAX_CONCURRENT", "integer", 2),
    ("agent", "driver"): ("AUGGIE_AGENT_DRIVER", "string", None),
    ("agent", "model"): ("AUGGIE_AGENT_MODEL", "string", None),
    ("agent", "reasoning"): ("AUGGIE_AGENT_REASONING", "string", None),
    ("agent", "args"): ("AUGGIE_AGENT_ARGS", "args", ()),
    ("agent", "executable"): ("AUGGIE_AGENT_EXECUTABLE", "string", None),
    ("agent", "timeout_s"): ("AUGGIE_AGENT_TIMEOUT_S", "float", 1800.0),
}

_SECTIONS = frozenset(section for section, _ in _SCHEMA)


def _toml_type(value: Any) -> str:
    return _TOML_NAMES.get(type(value), type(value).__name__)


def _parse_toml_value(path: Path, section: str, key: str, kind: str, value: Any) -> Any:
    if kind == "string":
        if not isinstance(value, str):
            raise ConfigError(
                f"{path}: [{section}] {key}: expected string, got {_toml_type(value)}"
            )
        return value
    if kind == "path":
        if not isinstance(value, str):
            raise ConfigError(
                f"{path}: [{section}] {key}: expected string, got {_toml_type(value)}"
            )
        return Path(os.path.expanduser(value))
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ConfigError(
                f"{path}: [{section}] {key}: expected boolean, got {_toml_type(value)}"
            )
        return value
    if kind == "integer":
        if type(value) is not int:
            raise ConfigError(
                f"{path}: [{section}] {key}: expected integer, got {_toml_type(value)}"
            )
        return value
    if kind == "float":
        if type(value) not in (int, float):
            raise ConfigError(
                f"{path}: [{section}] {key}: expected float, got {_toml_type(value)}"
            )
        return float(value)
    if kind == "args":
        if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
            raise ConfigError(
                f"{path}: [{section}] {key}: expected array of strings, got {_toml_type(value)}"
            )
        return tuple(value)
    raise AssertionError(f"unknown kind: {kind}")


def _parse_env_value(env_name: str, kind: str, raw: str) -> Any:
    if kind == "string":
        return raw
    if kind == "path":
        return Path(os.path.expanduser(raw))
    if kind == "boolean":
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{env_name}: expected boolean, got {raw!r}")
    if kind == "integer":
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name}: expected integer, got {raw!r}") from exc
    if kind == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"{env_name}: expected float, got {raw!r}") from exc
    if kind == "args":
        try:
            return tuple(shlex.split(raw))
        except ValueError as exc:
            raise ConfigError(f"{env_name}: invalid shell-quoted args {raw!r}: {exc}") from exc
    raise AssertionError(f"unknown kind: {kind}")


def _warn_removed_key(section: str, key: str) -> None:
    removed_key = (section, key)
    with _REMOVED_WARNING_LOCK:
        if removed_key in _WARNED_REMOVED_KEYS:
            return
        _WARNED_REMOVED_KEYS.add(removed_key)
    logger.warning("Ignoring removed config key [%s] %s", section, key)


def _read_toml(path: Path) -> dict[tuple[str, str], Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config: {exc.strerror or exc}") from exc

    values: dict[tuple[str, str], Any] = {}
    for section, table in data.items():
        if not isinstance(table, dict):
            raise ConfigError(
                f"{path}: top-level key {section} must be inside a section"
            )
        if section in REMOVED_SECTIONS:
            for key in table:
                _warn_removed_key(section, key)
            continue
        if section not in _SECTIONS:
            raise ConfigError(f"{path}: unknown section [{section}]")
        for key, raw_value in table.items():
            if (section, key) in REMOVED_KEYS:
                _warn_removed_key(section, key)
                continue
            if (section, key) not in _SCHEMA:
                raise ConfigError(f"{path}: unknown key [{section}] {key}")
            _, kind, _ = _SCHEMA[(section, key)]
            values[(section, key)] = _parse_toml_value(path, section, key, kind, raw_value)
    return values


def load(
    path: Path | str = CONFIG_PATH,
    environ: Mapping[str, str] = os.environ,
) -> Config:
    """Load configuration from environment variables, TOML file, and defaults."""
    resolved_path = Path(path).expanduser()
    toml_values = _read_toml(resolved_path)

    resolved: dict[str, dict[str, Any]] = {section: {} for section in _SECTIONS}
    for (section, key), (env_name, kind, default) in _SCHEMA.items():
        env_raw = environ.get(env_name)
        if env_raw:
            value = _parse_env_value(env_name, kind, env_raw)
        elif (section, key) in toml_values:
            value = toml_values[(section, key)]
        else:
            value = default
        resolved[section][key] = value

    return Config(
        embedding=EmbeddingConfig(**resolved["embedding"]),
        voyage=VoyageConfig(**resolved["voyage"]),
        chunking=ChunkingConfig(**resolved["chunking"]),
        index=IndexConfig(**resolved["index"]),
        telemetry=TelemetryConfig(**resolved["telemetry"]),
        retrieval=RetrievalConfig(**resolved["retrieval"]),
        agent=AgentConfig(**resolved["agent"]),
    )
