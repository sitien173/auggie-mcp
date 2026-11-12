"""
Shared utilities for auggie CLI commands.

This module provides common functionality used across multiple commands,
including database path resolution, branch detection, and argument parsing helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Cache for branch detection (avoid repeated git calls in same session)
_branch_cache: dict[str, str] = {}


def require_embedding_config() -> bool:
    """
    Check if embedding API is configured, print error if not.

    Returns:
        True if configured, False otherwise (with error printed to stderr).
    """
    if os.environ.get("VOYAGE_API_KEY"):
        return True

    print(
        "Error: VOYAGE_API_KEY is not set.\n"
        "Set VOYAGE_API_KEY for Voyage embeddings.",
        file=sys.stderr,
    )
    return False


def _repo_hash(root: Path) -> str:
    """
    Generate a short hash from repository path for global cache keying.

    Args:
        root: The repository root path.

    Returns:
        A 12-character hex string derived from the absolute path.
    """
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def _resolve_git_dir(repo_root: Path) -> Path | None:
    """
    Resolve the git metadata directory for a repository root.

    Supports both standard repositories where `.git` is a directory and
    worktrees/submodules where `.git` is a file containing `gitdir: ...`.
    """
    git_entry = repo_root / ".git"
    if git_entry.is_dir():
        return git_entry
    if not git_entry.is_file():
        return None

    try:
        content = git_entry.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    prefix = "gitdir:"
    if not content.lower().startswith(prefix):
        return None

    git_dir = content[len(prefix) :].strip()
    resolved = Path(git_dir)
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()
    return resolved


def _read_head_ref(repo_root: Path) -> str | None:
    """Read the repository HEAD file and return its raw contents."""
    git_dir = _resolve_git_dir(repo_root)
    if git_dir is None:
        return None

    head_path = git_dir / "HEAD"
    try:
        return head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def resolve_db_path(
    db_arg: str | None,
    profile: str | None,
    global_cache: bool,
    repo_root: Path | None = None,
) -> Path:
    """
    Resolve the database path based on scope flags.

    Determines the appropriate SQLite database location based on the
    provided scope options. This enables multi-repo setups without
    index pollution between projects.

    Priority (highest to lowest):
        1. Explicit --db path
        2. --global-cache: ~/.cache/auggie/<repo_hash>/index.sqlite
        3. --profile: .auggie/<profile>/index.sqlite
        4. Default: .auggie/index.sqlite

    Args:
        db_arg: Explicit database path from --db flag.
        profile: Profile name from --profile flag.
        global_cache: Whether to use global cache from --global-cache flag.
        repo_root: Repository root path (defaults to current directory).

    Returns:
        Resolved Path to the SQLite database file.

    Examples:
        >>> resolve_db_path(None, None, False)
        PosixPath('.auggie/index.sqlite')

        >>> resolve_db_path(None, "dev", False)
        PosixPath('.auggie/dev/index.sqlite')

        >>> resolve_db_path("/custom/path.db", None, False)
        PosixPath('/custom/path.db')
    """
    if db_arg:
        return Path(db_arg)

    root = repo_root or Path.cwd()

    if global_cache:
        cache_dir = Path.home() / ".cache" / "auggie" / _repo_hash(root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "index.sqlite"

    if profile:
        return root / ".auggie" / profile / "index.sqlite"

    return root / ".auggie" / "index.sqlite"


def get_current_branch(repo_root: Path | None = None) -> str:
    """
    Detect the current git branch, with caching.

    Returns 'default' for non-git directories or when git is unavailable.
    For detached HEAD state, returns 'detached-<short-hash>'.

    Args:
        repo_root: Repository root path (defaults to current directory).

    Returns:
        Branch name, or 'default' if not in a git repo.

    Examples:
        >>> get_current_branch()
        'main'
        >>> get_current_branch(Path("/non-git-dir"))
        'default'
    """
    root = repo_root or Path.cwd()
    root_key = str(root.resolve())

    # Return cached result if available
    if root_key in _branch_cache:
        return _branch_cache[root_key]

    branch = _detect_git_branch(root)
    _branch_cache[root_key] = branch
    return branch


def _detect_git_branch(repo_root: Path) -> str:
    """
    Detect git branch without caching.

    Args:
        repo_root: Repository root path.

    Returns:
        Branch name, 'detached-<hash>' for detached HEAD, or 'default'.
    """
    head = _read_head_ref(repo_root)
    if head:
        if head.startswith("ref: "):
            ref = head[5:].strip()
            if ref.startswith("refs/heads/"):
                return ref.removeprefix("refs/heads/")
            return ref.rsplit("/", 1)[-1]
        return f"detached-{head[:7]}"

    return "default"


def find_git_root(start_path: Path) -> Path | None:
    """
    Find the root of the git repository containing the given path.

    Runs `git rev-parse --show-toplevel` to find the repository root.
    This ensures .auggie is created in the repo root, not in subdirectories.

    Args:
        start_path: Path to start searching from.

    Returns:
        Path to the git repository root, or None if not in a git repo.

    Examples:
        >>> find_git_root(Path("/repo/subdir"))
        PosixPath('/repo')
        >>> find_git_root(Path("/not-a-repo"))
        None
    """
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if _resolve_git_dir(candidate) is not None:
            return candidate

    return None


def get_git_branches(repo_root: Path | None = None) -> set[str]:
    """
    Get all git branch names (local branches only).

    Used for pruning orphaned branch data.

    Args:
        repo_root: Repository root path.

    Returns:
        Set of branch names, or empty set if not in a git repo.
    """
    root = repo_root or Path.cwd()
    git_dir = _resolve_git_dir(root)
    if git_dir is None:
        return {"default"}

    branches: set[str] = {"default"}
    heads_dir = git_dir / "refs" / "heads"

    if heads_dir.exists():
        for path in heads_dir.rglob("*"):
            if path.is_file():
                try:
                    branches.add(str(path.relative_to(heads_dir)).replace("\\", "/"))
                except ValueError:
                    continue

    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        try:
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or line.startswith("^") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                    branches.add(parts[1].removeprefix("refs/heads/"))
        except OSError:
            pass
    return branches


def clear_branch_cache() -> None:
    """Clear the branch detection cache."""
    _branch_cache.clear()


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common scope/fencing arguments to an argument parser.

    Adds the standard set of arguments for controlling index scope:
        --db, --profile, --global-cache, --repo-root

    These arguments allow users to manage multiple indexes and prevent
    cross-repository pollution in monorepo or multi-project setups.

    Args:
        parser: The argparse parser or subparser to add arguments to.
    """
    parser.add_argument(
        "--db",
        default=None,
        help="Explicit SQLite DB path (overrides all other scope options)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name for multiple indexes per repo (.auggie/<profile>/index.sqlite)",
    )
    parser.add_argument(
        "--global-cache",
        action="store_true",
        help="Use global cache at ~/.cache/auggie/<repo_hash>/index.sqlite",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Explicit repository root (default: current directory)",
    )


# Extension to programming language mapping
EXTENSION_LANGUAGES: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    # JavaScript/TypeScript
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    # Web
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    # Systems languages
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".zig": "zig",
    # JVM
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".groovy": "groovy",
    # .NET
    ".cs": "csharp",
    ".fs": "fsharp",
    ".vb": "vb",
    # Scripting
    ".rb": "ruby",
    ".php": "php",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    # Shell
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    # Functional
    ".hs": "haskell",
    ".lhs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    # Data/Config (included for completeness, often excluded from indexing)
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    # Swift/Objective-C
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    # Other
    ".vim": "vim",
    ".el": "elisp",
    ".rkt": "racket",
    ".nim": "nim",
    ".d": "d",
    ".dart": "dart",
    ".v": "v",
    ".asm": "assembly",
    ".s": "assembly",
    ".S": "assembly",
    ".wasm": "wasm",
    ".wat": "wasm",
}


def detect_language(path: str) -> str | None:
    """
    Detect programming language from file extension.

    Args:
        path: File path (absolute or relative).

    Returns:
        Language name (lowercase) or None if unknown.

    Example:
        >>> detect_language("/home/user/project/auth.py")
        'python'
        >>> detect_language("main.rs")
        'rust'
        >>> detect_language("Makefile")
        None
    """
    ext = Path(path).suffix.lower()
    # Handle uppercase extensions like .R
    if ext not in EXTENSION_LANGUAGES:
        ext = Path(path).suffix  # Try original case
    return EXTENSION_LANGUAGES.get(ext)
