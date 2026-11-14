"""Repository file discovery for auggie-mcp."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

DEFAULT_MAX_BYTES = 1024 * 1024  # 1 MiB
_ALWAYS_SKIP_DIRS = frozenset({".git", ".auggie"})


def _compile_gitignore_glob(pattern: str, *, anchored: bool) -> re.Pattern[str]:
    """Translate a gitignore glob pattern into a regex matching POSIX relative paths."""
    if pattern == "**":
        return re.compile("^.*$")
    i = 0
    n = len(pattern)
    parts: list[str] = []
    while i < n:
        ch = pattern[i]
        if ch == "*" and i + 1 < n and pattern[i + 1] == "*":
            at_boundary = i == 0 or pattern[i - 1] == "/"
            if at_boundary and i + 2 < n and pattern[i + 2] == "/":
                parts.append("(?:.+/)?")
                i += 3
            elif i > 0 and pattern[i - 1] == "/" and i + 2 == n:
                parts.append(".*")
                i += 2
            else:
                parts.append("[^/]*")
                i += 2
        elif ch == "*":
            parts.append("[^/]*")
            i += 1
        elif ch == "?":
            parts.append("[^/]")
            i += 1
        elif ch == "[":
            j = i + 1
            if j < n and pattern[j] in ("!", "^"):
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                parts.append(re.escape(ch))
                i += 1
            else:
                inner = pattern[i + 1 : j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                parts.append(f"[{inner}]")
                i = j + 1
        elif ch == "\\":
            if i + 1 < n:
                parts.append(re.escape(pattern[i + 1]))
                i += 2
            else:
                parts.append(re.escape(ch))
                i += 1
        else:
            parts.append(re.escape(ch))
            i += 1

    body = "".join(parts)
    if anchored:
        return re.compile(f"^{body}$")
    return re.compile(f"^(?:.+/)?{body}$")


class _IgnoreRules:
    """Gitignore-style rule evaluator."""

    def __init__(self, rules: list[tuple[bool, bool, re.Pattern[str]]]) -> None:
        # Each entry: (negated, dir_only, compiled_regex)
        self._rules = rules

    @classmethod
    def from_file(cls, path: Path) -> _IgnoreRules:
        if not path.is_file():
            return cls([])
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return cls([])

        rules: list[tuple[bool, bool, re.Pattern[str]]] = []
        for raw in lines:
            line = raw.rstrip("\r")
            if not line or line.startswith("#"):
                continue
            negated = False
            if line.startswith("!"):
                negated = True
                line = line[1:]
            elif line.startswith(("\\#", "\\!")):
                line = line[1:]
            if not line:
                continue
            dir_only = line.endswith("/")
            if dir_only:
                line = line.rstrip("/")
            if not line:
                continue
            anchored = "/" in line
            line = line.removeprefix("/")
            regex = _compile_gitignore_glob(line, anchored=anchored)
            rules.append((negated, dir_only, regex))
        return cls(rules)

    def _is_path_ignored(self, rel_posix: str, *, is_dir: bool) -> bool:
        ignored = False
        for negated, dir_only, regex in self._rules:
            if dir_only and not is_dir:
                continue
            if regex.match(rel_posix):
                ignored = not negated
        return ignored

    def is_ignored(self, rel_posix: str, *, is_dir: bool) -> bool:
        parts = rel_posix.split("/")
        for idx in range(1, len(parts)):
            if self._is_path_ignored("/".join(parts[:idx]), is_dir=True):
                return True
        return self._is_path_ignored(rel_posix, is_dir=is_dir)


def _is_git_work_tree(repo: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _git_candidates(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="surrogateescape")
    return [entry for entry in raw.split("\x00") if entry]


def _plain_candidates(
    repo: Path,
    repo_resolved: Path,
    gitignore: _IgnoreRules,
) -> list[str]:
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo, followlinks=False):
        current = Path(dirpath)
        rel_dir = current.relative_to(repo)
        kept_dirs: list[str] = []
        for dname in sorted(dirnames):
            if dname in _ALWAYS_SKIP_DIRS:
                continue
            dpath = current / dname
            if dpath.is_symlink() and not dpath.resolve(strict=False).is_relative_to(repo_resolved):
                continue
            rel_posix = (rel_dir / dname).as_posix() if rel_dir != Path(".") else dname
            if gitignore.is_ignored(rel_posix, is_dir=True):
                continue
            kept_dirs.append(dname)
        dirnames[:] = kept_dirs

        for fname in sorted(filenames):
            fpath = current / fname
            if fpath.is_symlink() and not fpath.resolve(strict=False).is_relative_to(
                repo_resolved
            ):
                continue
            rel_posix = (rel_dir / fname).as_posix() if rel_dir != Path(".") else fname
            if gitignore.is_ignored(rel_posix, is_dir=False):
                continue
            results.append(rel_posix)
    return results


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(65536):
                if b"\x00" in chunk:
                    return True
    except OSError:
        return False
    return False


def walk_repo(
    repo_root: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    check_binary: bool = True,
) -> Iterator[str]:
    """Yield repository-relative POSIX paths of indexable files."""
    repo = Path(repo_root)
    repo_resolved = repo.resolve()
    auggieignore = _IgnoreRules.from_file(repo / ".auggieignore")

    if _is_git_work_tree(repo):
        candidates = sorted(set(_git_candidates(repo)))
    else:
        gitignore = _IgnoreRules.from_file(repo / ".gitignore")
        candidates = sorted(_plain_candidates(repo, repo_resolved, gitignore))

    for rel_posix in candidates:
        parts = rel_posix.split("/")
        if any(part in _ALWAYS_SKIP_DIRS for part in parts[:-1]) or (
            parts[0] in _ALWAYS_SKIP_DIRS
        ):
            continue
        if auggieignore.is_ignored(rel_posix, is_dir=False):
            continue

        full_path = repo / rel_posix
        resolved_path = full_path.resolve(strict=False)
        if not resolved_path.is_relative_to(repo_resolved):
            continue
        if not resolved_path.is_file():
            continue
        try:
            size = resolved_path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            continue
        if check_binary and _is_binary(resolved_path):
            continue
        yield rel_posix
