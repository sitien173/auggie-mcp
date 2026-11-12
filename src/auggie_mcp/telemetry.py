"""Local, owner-only usage telemetry. Standard library only; never raises."""

import importlib.metadata
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

DEFAULT_RETENTION_DAYS = 30
_FILE_RE = re.compile(r"^events-(\d{4}-\d{2}-\d{2})\.jsonl$")


def telemetry_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve telemetry directory based on environment hierarchy."""
    env = os.environ if environ is None else environ

    explicit = env.get("AUGGIE_TELEMETRY_DIR")
    if explicit:
        return Path(explicit)

    xdg_state = env.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "auggie-mcp" / "telemetry"

    return Path.home() / ".local" / "state" / "auggie-mcp" / "telemetry"


def is_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("AUGGIE_TELEMETRY") != "0"


def retention_days(environ: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if environ is None else environ
    try:
        days = int(env.get("AUGGIE_TELEMETRY_RETENTION_DAYS", ""))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return days if days > 0 else DEFAULT_RETENTION_DAYS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_event(
    event: Mapping[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Append one JSON line to today's file. Returns False on disable or error."""
    try:
        if not is_enabled(environ):
            return False
        directory = telemetry_dir(environ)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if stat.S_IMODE(os.stat(directory).st_mode) != 0o700:
            os.chmod(directory, 0o700)
        day = (now or _utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%d")
        data = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        fd = os.open(
            directory / f"events-{day}.jsonl",
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
    except Exception:
        return False


def prune_old_files(
    *,
    environ: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> None:
    """Delete daily files older than the retention window."""
    try:
        if not is_enabled(environ):
            return
        today = (now or _utc_now()).astimezone(timezone.utc).date()
        cutoff = today - timedelta(days=retention_days(environ))
        for path in telemetry_dir(environ).iterdir():
            try:
                match = _FILE_RE.match(path.name)
                if match and date.fromisoformat(match.group(1)) < cutoff:
                    path.unlink()
            except Exception:
                continue
    except Exception:
        return


process_id = uuid.uuid4().hex


def _app_version() -> str:
    try:
        return importlib.metadata.version("auggie-mcp")
    except Exception:
        return "unknown"


class CallRecord:
    """Accumulates one tool call's telemetry and writes it on finish()."""

    def __init__(
        self,
        source: str,
        tool: str,
        input: Any,
        *,
        client: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.source = source
        self.tool = tool
        self.input = input
        self.client = client
        self.environ = environ
        self.event_id = uuid.uuid4().hex
        self.process_id = process_id
        self.app_version = _app_version()
        self.durations_ms: dict = {}
        self.runner: Optional[dict] = None
        self.processing: Optional[dict] = None
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        try:
            start = time.perf_counter()
        except Exception:
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
            except Exception:
                pass

    def set_runner(
        self, *, exit_code: Optional[int], stdout_bytes: int, stderr_bytes: int
    ) -> None:
        self.runner = {
            "exit_code": exit_code,
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
        }

    def set_processing(
        self, processing: Mapping[str, Any], output: Optional[str] = None
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
        except Exception:
            pass

    def finish(
        self,
        outcome: str,
        error: Optional[BaseException] = None,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        try:
            moment = (now or _utc_now()).astimezone(timezone.utc)
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
            return append_event(event, environ=self.environ, now=moment)
        except Exception:
            return False
