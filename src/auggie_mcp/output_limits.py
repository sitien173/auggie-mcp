"""Shared output truncation and archive helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

MAX_OUTPUT_BYTES = 64 * 1024
_ARCHIVE_RETENTION = timedelta(days=7)


def _archive_dir(repository_path: Path | str) -> Path:
    return Path(repository_path) / ".auggie" / "archive"


def cleanup_archives(
    repository_path: Path | str, *, now: datetime | None = None
) -> None:
    archive_dir = _archive_dir(repository_path)
    if not archive_dir.is_dir():
        return
    current = now or datetime.now(UTC)
    cutoff = current.timestamp() - _ARCHIVE_RETENTION.total_seconds()
    for archive in archive_dir.iterdir():
        if archive.is_file() and archive.stat().st_mtime < cutoff:
            archive.unlink()


def serialize_json(value: object) -> str:
    """Serialize JSON with standard defaults for conservative output sizing."""
    return json.dumps(value)


def cleanup_archives_on_startup(repository_path: Path | str) -> None:
    """Remove expired archives when the server starts."""
    cleanup_archives(repository_path)


def truncate_output(
    text: str, repository_path: Path | str, *, now: datetime | None = None
) -> tuple[str, str | None]:
    encoded = text.encode("utf-8")
    if len(encoded) < MAX_OUTPUT_BYTES:
        return text, None

    archive_dir = _archive_dir(repository_path)
    archive_dir.mkdir(parents=True, exist_ok=True)
    cleanup_archives(repository_path, now=now)
    archive_path = archive_dir / f"{uuid4().hex}.txt"
    archive_path.write_text(text, encoding="utf-8")

    prefix = encoded[: MAX_OUTPUT_BYTES - 1]
    last_newline = prefix.rfind(b"\n")
    truncated = prefix[: last_newline + 1] if last_newline >= 0 else b""
    return truncated.decode("utf-8"), str(archive_path)
