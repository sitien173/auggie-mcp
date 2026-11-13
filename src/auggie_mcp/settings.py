"""Validate ``~/.auggie/config.toml`` and apply it to the environment.

Standard library only: this module runs before ``auggie`` is imported.
"""

import os
import tomllib
from pathlib import Path

CONFIG_PATH = Path("~/.auggie/config.toml")

# (section, key) -> (environment variable, type). ``path`` is a string that
# gets ``~`` expanded. ``cache.enabled`` is inverted onto AUGGIE_CACHE_DISABLED.
SCHEMA = {
    ("search", "mode"): ("AUGGIE_SEARCH_MODE", "string"),
    ("search", "fusion_method"): ("AUGGIE_FUSION_METHOD", "string"),
    ("search", "hybrid_alpha"): ("AUGGIE_HYBRID_ALPHA", "float"),
    ("search", "rrf_k"): ("AUGGIE_RRF_K", "integer"),
    ("search", "ann_overfetch"): ("AUGGIE_ANN_OVERFETCH", "integer"),
    ("confidence", "mode"): ("AUGGIE_CONFIDENCE_MODE", "string"),
    ("confidence", "absolute_high"): ("AUGGIE_CONFIDENCE_HIGH", "float"),
    ("confidence", "absolute_medium"): ("AUGGIE_CONFIDENCE_MEDIUM", "float"),
    ("confidence", "absolute_low"): ("AUGGIE_CONFIDENCE_LOW", "float"),
    ("confidence", "relative_high"): ("AUGGIE_RELATIVE_HIGH", "float"),
    ("confidence", "relative_medium"): ("AUGGIE_RELATIVE_MEDIUM", "float"),
    ("confidence", "relative_low"): ("AUGGIE_RELATIVE_LOW", "float"),
    ("confidence", "quality_strong"): ("AUGGIE_ABSOLUTE_STRONG", "float"),
    ("confidence", "quality_expected"): ("AUGGIE_ABSOLUTE_EXPECTED", "float"),
    ("confidence", "quality_weak"): ("AUGGIE_ABSOLUTE_WEAK", "float"),
    ("embedding", "model"): ("AUGGIE_MODEL", "string"),
    ("embedding", "dimensions"): ("AUGGIE_DIMENSIONS", "integer"),
    ("embedding", "batch_size"): ("AUGGIE_BATCH_SIZE", "integer"),
    ("voyage", "timeout_s"): ("AUGGIE_VOYAGE_TIMEOUT", "float"),
    ("voyage", "retries"): ("AUGGIE_VOYAGE_RETRIES", "integer"),
    ("voyage", "chars_per_token"): ("AUGGIE_VOYAGE_CHARS_PER_TOKEN", "float"),
    ("chunking", "lines"): ("AUGGIE_CHUNK_LINES", "integer"),
    ("chunking", "overlap_lines"): ("AUGGIE_OVERLAP_LINES", "integer"),
    ("rerank", "model"): ("AUGGIE_RERANK_MODEL", "string"),
    ("rerank", "top_n"): ("AUGGIE_RERANK_TOPN", "integer"),
    ("cache", "enabled"): ("AUGGIE_CACHE_DISABLED", "boolean"),
    ("cache", "l1_ttl_s"): ("AUGGIE_CACHE_L1_TTL", "integer"),
    ("cache", "l1_max_entries"): ("AUGGIE_CACHE_L1_MAX", "integer"),
    ("cache", "l2_max_entries"): ("AUGGIE_CACHE_L2_MAX", "integer"),
    ("cache", "l3_max_entries"): ("AUGGIE_CACHE_L3_MAX", "integer"),
    ("watch", "debounce_ms"): ("AUGGIE_WATCH_DEBOUNCE_MS", "integer"),
    ("index", "timeout_s"): ("AUGGIE_INDEX_TIMEOUT_S", "float"),
    ("jev", "model"): ("AUGGIE_JEV_MODEL", "string"),
    ("jev", "timeout_s"): ("AUGGIE_JEV_TIMEOUT_SECONDS", "float"),
    ("jev", "max_requests"): ("AUGGIE_JEV_MAX_REQUESTS", "integer"),
    ("jev", "archive_dir"): ("AUGGIE_JEV_ARCHIVE_DIR", "path"),
    ("telemetry", "enabled"): ("AUGGIE_TELEMETRY", "boolean"),
    ("telemetry", "dir"): ("AUGGIE_TELEMETRY_DIR", "path"),
    ("telemetry", "retention_days"): ("AUGGIE_TELEMETRY_RETENTION_DAYS", "integer"),
}

AGENT_KEYS = ("driver", "model", "reasoning", "args", "executable", "timeout_s")

SECTIONS = {section for section, _ in SCHEMA} | {"agent"}

_TOML_NAMES = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "float",
    list: "array",
    dict: "table",
}


class ConfigError(ValueError):
    """The config file is invalid."""


def _toml_type(value):
    return _TOML_NAMES.get(type(value), type(value).__name__)


def _check(path, section, key, kind, value):
    if kind in ("string", "path"):
        ok, expected = isinstance(value, str), "string"
    elif kind == "boolean":
        ok, expected = isinstance(value, bool), "boolean"
    elif kind == "integer":
        ok, expected = type(value) is int, "integer"
    else:
        ok, expected = type(value) in (int, float), "float"
    if not ok:
        raise ConfigError(
            f"{path}: [{section}] {key}: expected {expected}, got {_toml_type(value)}"
        )


def _encode(name, kind, value):
    if kind == "boolean":
        if name == "AUGGIE_CACHE_DISABLED":
            return None if value else "1"
        return "1" if value else "0"
    if kind == "path":
        return os.path.expanduser(value)
    return str(value)


def apply(path=CONFIG_PATH, environ=os.environ):
    """Validate the whole file, then fill unset or empty ``environ`` entries."""
    path = Path(path).expanduser()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read config: {exc.strerror or exc}") from exc

    pending = {}
    for section, table in data.items():
        if not isinstance(table, dict):
            raise ConfigError(
                f"{path}: top-level key {section} must be inside a section"
            )
        if section not in SECTIONS:
            raise ConfigError(f"{path}: unknown section [{section}]")
        for key, value in table.items():
            if section == "agent":
                if key not in AGENT_KEYS:
                    raise ConfigError(f"{path}: unknown key [{section}] {key}")
                continue
            if (section, key) not in SCHEMA:
                raise ConfigError(f"{path}: unknown key [{section}] {key}")
            name, kind = SCHEMA[(section, key)]
            _check(path, section, key, kind, value)
            encoded = _encode(name, kind, value)
            if encoded is not None:
                pending[name] = encoded

    for name, encoded in pending.items():
        if not environ.get(name):
            environ[name] = encoded
