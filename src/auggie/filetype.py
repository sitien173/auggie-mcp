"""
File type detection module for auggie.

Uses the `file` command for robust MIME-type detection with
fallback to heuristic null-byte detection on unsupported platforms.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Cache file command availability at module load
_FILE_CMD: str | None = shutil.which("file")

# MIME type prefixes that indicate text content
TEXT_MIME_PREFIXES = ("text/",)

# Specific application/* and inode/* types that are actually text-based
TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/xml",
        "application/x-sh",
        "application/x-shellscript",
        "application/x-perl",
        "application/x-ruby",
        "application/x-python",
        "application/x-php",
        "application/x-httpd-php",
        "application/x-awk",
        "application/x-ndjson",
        "application/sql",
        "application/x-empty",  # Empty files are fine
        "inode/x-empty",  # Empty files (Linux)
    }
)

# Types to explicitly block even if they might pass other checks
BLOCKED_MIME_TYPES = frozenset(
    {
        "application/x-sqlite3",
        "application/x-executable",
        "application/x-sharedlib",
        "application/x-mach-binary",
        "application/x-dosexec",
        "application/octet-stream",
        "application/gzip",
        "application/x-tar",
        "application/zip",
        "application/x-bzip2",
        "application/x-7z-compressed",
        "application/x-rar",
        "application/pdf",
        "application/x-object",
        "application/x-archive",
    }
)


@dataclass(frozen=True)
class FileTypeResult:
    """Result of file type detection."""

    path: Path
    mime_type: str | None
    is_text: bool
    detection_method: str  # "file_cmd" or "null_byte"


def has_file_command() -> bool:
    """Check if the file command is available."""
    return _FILE_CMD is not None


def _is_text_mime(mime_type: str) -> bool:
    """
    Check if a MIME type indicates text content.

    Args:
        mime_type: MIME type string (e.g., "text/plain", "application/x-sqlite3").

    Returns:
        True if the MIME type indicates indexable text content.
    """
    # Explicitly blocked types take priority
    if mime_type in BLOCKED_MIME_TYPES:
        return False

    # Binary types by prefix
    if mime_type.startswith(("image/", "audio/", "video/", "font/")):
        return False

    # Text types by prefix
    if mime_type.startswith(TEXT_MIME_PREFIXES):
        return True

    # Known text application types
    if mime_type in TEXT_APPLICATION_TYPES:
        return True

    # Default: unknown types are assumed binary for safety
    return False


def _null_byte_check(content: bytes) -> bool:
    """
    Fast heuristic: text files don't contain null bytes.

    Args:
        content: File content as bytes.

    Returns:
        True if no null bytes found (likely text), False otherwise.
    """
    return content.find(b"\x00") == -1


# Batch size for file command to avoid ARG_MAX limits and timeouts
_BATCH_SIZE = 500


def detect_file_types_batch(
    paths: list[Path],
    progress_callback: callable | None = None,
) -> dict[Path, FileTypeResult]:
    """
    Detect file types for multiple files using batched calls.

    Uses `file --mime-type -b` for efficiency. Falls back to null-byte
    detection if file command is unavailable or fails.

    Args:
        paths: List of file paths to check.
        progress_callback: Optional callback(n) called after each batch with count processed.

    Returns:
        Dict mapping paths to FileTypeResult objects.
    """
    results: dict[Path, FileTypeResult] = {}

    if not paths:
        return results

    if not has_file_command():
        # Fallback: read each file and check for null bytes
        return _fallback_null_byte_detection(paths)

    # Process in batches to avoid ARG_MAX limits and timeouts
    for i in range(0, len(paths), _BATCH_SIZE):
        batch = paths[i : i + _BATCH_SIZE]
        batch_results = _detect_batch(batch)
        results.update(batch_results)
        if progress_callback:
            progress_callback(len(batch))

    # Fill in any paths not covered by file command with null-byte detection
    missing_paths = [p for p in paths if p not in results]
    if missing_paths:
        fallback_results = _fallback_null_byte_detection(missing_paths)
        results.update(fallback_results)
        if progress_callback:
            progress_callback(len(missing_paths))

    return results


def _detect_batch(paths: list[Path]) -> dict[Path, FileTypeResult]:
    """
    Detect file types for a single batch of files.

    Args:
        paths: List of file paths (should be <= _BATCH_SIZE).

    Returns:
        Dict mapping paths to FileTypeResult objects.
    """
    results: dict[Path, FileTypeResult] = {}

    try:
        proc = subprocess.run(
            [_FILE_CMD, "--mime-type", "-b", "--"] + [str(p) for p in paths],
            capture_output=True,
            text=True,
            timeout=60,  # Allow more time per batch
        )
        # Parse output even if return code is non-zero (some files may have failed)
        if proc.stdout:
            mime_types = proc.stdout.strip().split("\n")
            for path, mime in zip(paths, mime_types, strict=False):
                mime = mime.strip()
                # Skip error messages from file command
                if mime.startswith("cannot open") or mime.startswith("ERROR:"):
                    continue
                results[path] = FileTypeResult(
                    path=path,
                    mime_type=mime,
                    is_text=_is_text_mime(mime),
                    detection_method="file_cmd",
                )
    except (subprocess.TimeoutExpired, OSError):
        pass  # Will fall through to null-byte detection for missing paths

    return results


def _fallback_null_byte_detection(paths: list[Path]) -> dict[Path, FileTypeResult]:
    """
    Fallback detection using null-byte check.

    Args:
        paths: List of file paths to check.

    Returns:
        Dict mapping paths to FileTypeResult objects.
    """
    results: dict[Path, FileTypeResult] = {}
    for p in paths:
        try:
            content = p.read_bytes()
            is_text = _null_byte_check(content)
        except Exception:
            is_text = False
        results[p] = FileTypeResult(
            path=p,
            mime_type=None,
            is_text=is_text,
            detection_method="null_byte",
        )
    return results


def is_text_file(path: Path, content: bytes | None = None) -> FileTypeResult:
    """
    Check if a file is a text file.

    Uses file command if available, otherwise falls back to null-byte check.

    Args:
        path: Path to the file.
        content: Optional pre-read content (avoids re-reading for null-byte check).

    Returns:
        FileTypeResult with detection details.
    """
    if has_file_command():
        results = detect_file_types_batch([path])
        if path in results:
            return results[path]

    # Fallback
    if content is None:
        try:
            content = path.read_bytes()
        except Exception:
            return FileTypeResult(
                path=path,
                mime_type=None,
                is_text=False,
                detection_method="null_byte",
            )

    return FileTypeResult(
        path=path,
        mime_type=None,
        is_text=_null_byte_check(content),
        detection_method="null_byte",
    )
