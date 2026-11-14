"""Local, owner-only usage telemetry. Standard library only; never raises."""

import importlib.metadata
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_RETENTION_DAYS = 30
_FILE_RE = re.compile(r"^events-(\d{4}-\d{2}-\d{2})\.jsonl$")


def telemetry_dir(
    environ: Mapping[str, str] | None = None,
    *,
    directory: Path | str | None = None,
) -> Path:
    """Resolve telemetry directory based on environment hierarchy."""
    if directory is not None:
        return Path(directory)
    env = os.environ if environ is None else environ

    explicit = env.get("AUGGIE_TELEMETRY_DIR")
    if explicit:
        return Path(explicit)

    xdg_state = env.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "auggie-mcp" / "telemetry"

    return Path.home() / ".local" / "state" / "auggie-mcp" / "telemetry"


def is_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("AUGGIE_TELEMETRY") != "0"


def retention_days(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        days = int(env.get("AUGGIE_TELEMETRY_RETENTION_DAYS", ""))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return days if days > 0 else DEFAULT_RETENTION_DAYS


def _utc_now() -> datetime:
    return datetime.now(UTC)


def append_event(
    event: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    enabled: bool | None = None,
    directory: Path | str | None = None,
    now: datetime | None = None,
) -> bool:
    """Append one JSON line to today's file. Returns False on disable or error."""
    try:
        if not (is_enabled(environ) if enabled is None else enabled):
            return False
        target_dir = telemetry_dir(environ, directory=directory)
        os.makedirs(target_dir, mode=0o700, exist_ok=True)
        if stat.S_IMODE(os.stat(target_dir).st_mode) != 0o700:
            os.chmod(target_dir, 0o700)
        day = (now or _utc_now()).astimezone(UTC).strftime("%Y-%m-%d")
        data = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        fd = os.open(
            target_dir / f"events-{day}.jsonl",
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) & ~0o600:
                os.fchmod(fd, 0o600)
            written = os.write(fd, data)
        finally:
            os.close(fd)
        return written == len(data)
    except Exception:  # noqa: BLE001 - telemetry must never disrupt tool calls.
        return False


def prune_old_files(
    *,
    environ: Mapping[str, str] | None = None,
    enabled: bool | None = None,
    directory: Path | str | None = None,
    retention: int | None = None,
    now: datetime | None = None,
) -> None:
    """Delete daily files older than the retention window."""
    try:
        if not (is_enabled(environ) if enabled is None else enabled):
            return
        today = (now or _utc_now()).astimezone(UTC).date()
        days = retention_days(environ) if retention is None else retention
        cutoff = today - timedelta(days=days)
        for path in telemetry_dir(environ, directory=directory).iterdir():
            try:
                match = _FILE_RE.match(path.name)
                if match and date.fromisoformat(match.group(1)) < cutoff:
                    path.unlink()
            except Exception:  # noqa: BLE001, S112 - skip malformed old event files.
                continue
    except Exception:  # noqa: BLE001 - telemetry pruning is best effort.
        return


process_id = uuid.uuid4().hex


def _app_version() -> str:
    try:
        return importlib.metadata.version("auggie-mcp")
    except Exception:  # noqa: BLE001 - package metadata is optional at runtime.
        return "unknown"


class CallRecord:
    """Accumulates one tool call's telemetry and writes it on finish()."""

    def __init__(
        self,
        source: str,
        tool: str,
        input: Any,
        *,
        client: str | dict[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
        telemetry_enabled: bool | None = None,
        telemetry_directory: Path | str | None = None,
    ) -> None:
        self.source = source
        self.tool = tool
        self.input = input
        self.client = client
        self.environ = environ
        self.telemetry_enabled = telemetry_enabled
        self.telemetry_directory = telemetry_directory
        self.event_id = uuid.uuid4().hex
        self.process_id = process_id
        self.app_version = _app_version()
        self.durations_ms: dict = {}
        self.runner: dict | None = None
        self.processing: dict | None = None
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        try:
            start = time.perf_counter()
        except Exception:  # noqa: BLE001 - a clock failure cannot mask the wrapped call.
            start = None
        try:
            yield
        finally:
            try:
                if start is not None:
                    elapsed = (time.perf_counter() - start) * 1000
                    self.durations_ms[name] = (
                        self.durations_ms.get(name, 0.0) + elapsed
                    )
            except Exception:  # noqa: BLE001, S110 - timing must not mask call results.
                pass

    def set_runner(
        self, *, exit_code: int | None, stdout_bytes: int, stderr_bytes: int
    ) -> None:
        self.runner = {
            "exit_code": exit_code,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
        }

    def set_processing(
        self, processing: Mapping[str, Any], output: str | None = None
    ) -> None:
        try:
            chunks = processing.get("chunks") or []
            self.processing = {
                "status": processing.get("status"),
                "retention": processing.get("retention"),
                "estimated_tokens": processing.get("estimated_tokens"),
                "chunks_total": len(chunks),
                "chunks_retained": sum(1 for c in chunks if c.get("retained") is True),
                "output_bytes": (
                    len(output.encode("utf-8")) if isinstance(output, str) else None
                ),
            }
        except Exception:  # noqa: BLE001, S110 - telemetry serialization is best effort.
            pass

    def finish(
        self,
        outcome: str,
        error: BaseException | None = None,
        *,
        now: datetime | None = None,
    ) -> bool:
        try:
            moment = (now or _utc_now()).astimezone(UTC)
            self.durations_ms["total"] = (time.perf_counter() - self._start) * 1000
            event = {
                "v": 1,
                "ts": moment.strftime("%Y-%m-%dT%H:%M:%S.")
                + f"{moment.microsecond // 1000:03d}Z",
                "event_id": self.event_id,
                "process_id": self.process_id,
                "source": self.source,
                "tool": self.tool,
                "app_version": self.app_version,
                "client": self.client,
                "input": self.input,
                "outcome": outcome,
                "error": (
                    {"type": type(error).__name__, "message": str(error)}
                    if error is not None
                    else None
                ),
                "durations_ms": self.durations_ms,
                "runner": self.runner,
                "processing": self.processing,
            }
            return append_event(
                event,
                environ=self.environ,
                enabled=self.telemetry_enabled,
                directory=self.telemetry_directory,
                now=moment,
            )
        except Exception:  # noqa: BLE001 - telemetry writes must not disrupt calls.
            return False
