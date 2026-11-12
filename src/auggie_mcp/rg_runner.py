"""Safe ripgrep subprocess adapter for repository searches."""

import asyncio
import os
import signal
from pathlib import Path
from typing import TypedDict

_RG_EXECUTABLE = "rg"
_SHUTDOWN_GRACE_SECONDS = 0.5


class RgResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int

_NO_VALUE_FLAGS = {
    "-i", "--ignore-case", "-s", "--case-sensitive", "-S", "--smart-case",
    "-F", "--fixed-strings", "-w", "--word-regexp", "-v", "--invert-match",
    "-o", "--only-matching", "-U", "--multiline", "--multiline-dotall", "-n",
    "--line-number", "-N", "--no-line-number", "-c", "--count", "-l",
    "--files-with-matches", "--files-without-match", "-q", "--quiet",
    "--glob-case-insensitive", "--type-list", "--files", "--heading",
    "--no-heading", "-H", "--with-filename", "-I", "--no-filename", "--json",
    "--vimgrep", "-0", "--null", "--trim", "-.", "--hidden", "--no-ignore",
    "-a", "--text", "--binary", "-u", "-uu", "-uuu", "--no-max-filesize",
    "--no-messages", "--stats", "-x",
    "--line-regexp", "-P", "--pcre2", "--pcre2-version", "--no-unicode",
    "--passthru", "--stop-on-nonmatch", "--column", "--no-column", "-b",
    "--byte-offset", "--max-columns-preview", "--count-matches",
    "--include-zero", "-p", "--pretty", "--no-context-separator",
    "--sort-files", "--line-buffered", "--block-buffered", "--no-encoding",
    "--one-file-system", "--ignore-file-case-insensitive", "--no-ignore-dot",
    "--no-ignore-exclude", "--no-ignore-files", "--no-ignore-global",
    "--no-ignore-parent", "--no-ignore-vcs", "--no-ignore-messages",
    "--no-require-git", "--mmap", "--no-mmap", "--crlf", "--no-crlf",
    "--no-config", "--debug", "--trace",
}

_VALUE_FLAGS = {
    "-e", "--regexp", "-f", "--file", "-m", "--max-count", "-g", "--glob",
    "--iglob", "-t", "--type", "-T", "--type-not", "--type-add",
    "--type-clear", "-A", "--after-context", "-B", "--before-context", "-C",
    "--context", "--color", "--max-filesize", "--engine",
    "--regex-size-limit", "--dfa-size-limit", "-r", "--replace", "-M",
    "--max-columns", "--context-separator", "--field-match-separator",
    "--field-context-separator", "--path-separator", "--sort", "--sortr", "-E",
    "--encoding", "--max-depth", "--ignore-file", "-j", "--threads", "--colors",
}

_EXISTING_FILE_FLAGS = {"-f", "--file", "--ignore-file"}


def _repository_root(repository_root) -> Path:
    root = Path(os.path.realpath(os.fspath(repository_root)))
    if not root.exists():
        raise ValueError(f"repository root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    return root


def _resolve_beneath(root: Path, path, label: str) -> Path:
    candidate = Path(os.fspath(path))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = Path(os.path.realpath(candidate))
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root: {resolved}") from exc
    return resolved


def resolve_search_path(repository_root, path) -> Path:
    """Resolve an absolute path or a path beneath ``repository_root``."""
    root = _repository_root(repository_root)
    candidate = Path(os.fspath(path))
    resolved = (
        Path(os.path.realpath(candidate))
        if candidate.is_absolute()
        else _resolve_beneath(root, candidate, "search path")
    )
    if not resolved.exists():
        raise ValueError(f"search path does not exist: {resolved}")
    if not resolved.is_dir() and not resolved.is_file():
        raise ValueError(f"search path is not a file or directory: {resolved}")
    return resolved


def _validate_flag_value(root: Path, flag: str, value: str) -> str:
    if flag not in _EXISTING_FILE_FLAGS:
        return value

    resolved = _resolve_beneath(root, value, f"{flag} path")
    if not resolved.exists():
        raise ValueError(f"{flag} path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{flag} path is not a file: {resolved}")
    return str(resolved)


def validate_flags(repository_root, flags=None) -> list[str]:
    """Allow only documented rg search flags and their values."""
    root = _repository_root(repository_root)
    supplied = list(flags or [])
    validated = []
    index = 0

    while index < len(supplied):
        token = supplied[index]
        if not isinstance(token, str):
            raise ValueError("rg flags must be strings")

        if token in _NO_VALUE_FLAGS:
            validated.append(token)
            index += 1
            continue

        if token in _VALUE_FLAGS:
            if index + 1 >= len(supplied):
                raise ValueError(f"rg flag requires a value: {token}")
            value = supplied[index + 1]
            if not isinstance(value, str):
                raise ValueError(f"rg flag value must be a string: {token}")
            validated.extend([token, _validate_flag_value(root, token, value)])
            index += 2
            continue

        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag not in _VALUE_FLAGS:
                raise ValueError(f"unsupported rg flag: {flag}")
            validated.append(f"{flag}={_validate_flag_value(root, flag, value)}")
            index += 1
            continue

        raise ValueError(f"unsupported rg flag or positional argument: {token}")

    return validated


def build_rg_command(repository_root, pattern, path=".", flags=None) -> list[str]:
    """Build rg arguments without the executable."""
    root = _repository_root(repository_root)
    search_path = resolve_search_path(root, path)
    validated_flags = validate_flags(root, flags)
    return validated_flags + ["--", pattern, str(search_path)]


async def _terminate_process_group(process):
    if process is None or process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    try:
        await asyncio.wait_for(
            process.wait(), timeout=_SHUTDOWN_GRACE_SECONDS
        )
        return
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    await process.wait()


async def run_rg(
    repository_root,
    pattern,
    path=".",
    flags=None,
    *,
    executable=_RG_EXECUTABLE,
) -> RgResult:
    """Run rg without a shell and preserve its complete result."""
    root = _repository_root(repository_root)
    command = [executable] + build_rg_command(root, pattern, path, flags)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"rg executable not found: {executable} ({exc})"
        ) from exc

    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        await _terminate_process_group(process)
        raise

    return {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "exit_code": process.returncode,
    }
