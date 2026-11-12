"""
Indexer module for auggie.

Handles the core indexing logic: scanning directories, reading files,
chunking text, generating embeddings, and storing everything in the
database. Supports incremental updates by tracking file hashes.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .chunking import chunk_lines as chunk_text
from .db import (
    connect,
    delete_file_record,
    get_file_chunks,
    get_file_states,
    log_history,
    touch_file_state,
)
from .embed import embed_texts
from .filetype import detect_file_types_batch, has_file_command
from .lance_store import LanceStore, get_lance_dir
from .models import get_model, resolve_model


# AST chunking - optional, lazy import
def _get_ast_chunker():
    """Lazy import AST chunker to avoid import errors if not installed."""
    try:
        from .ast_chunking import chunk_ast, is_ast_available

        if is_ast_available():
            return chunk_ast
    except ImportError:
        pass
    return None


#: Directories to skip during indexing (version control, dependencies, caches)
DEFAULT_SKIP_DIRS = {
    ".git",
    ".svn",
    ".hg",  # Mercurial
    ".venv",
    "venv",
    "node_modules",
    "bin",
    "build",
    "target",
    "publish",
    ".auggie",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".githooks",
    "storage",  # Laravel/framework cache directories
}

#: Default exclude patterns for common non-source files
DEFAULT_EXCLUDES = (
    # Binary/compiled
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.egg-info/*",
    "*.egg",
    "*.whl",
    "*.dist-info/*",
    # OS files
    ".DS_Store",
    "Thumbs.db",
    # Git metadata
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".gitkeep",
    # Environment/secrets (never index these!)
    ".env",
    ".env.*",
    "*.env",
    ".envrc",
    "secrets.*",
    "credentials.*",
    # Documentation (index source code, not docs)
    "*.md",
    "*.txt",
    "*.rst",
    "docs/*",
    # Config/data files
    "*.json",
    "*.toml",
    "*.ini",
    "*.cfg",
    "*.conf",
    ".editorconfig",
    # Lock files
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    # Build outputs
    "dist/*",
    "build/*",
    "out/*",
    "target/*",
    # Minified files
    "*.min.js",
    "*.min.css",
    "*.map",
    # Test/coverage
    "coverage/*",
    ".coverage",
    "htmlcov/*",
    ".phpunit.result.cache",
    # Vendor directories
    "vendor/*",
    "third_party/*",
    # Common non-source
    "LICENSE*",
    "LICENCE*",
    "COPYING*",
    "Makefile",
    "Dockerfile",
    "*.dockerfile",
    # Logs and temp files
    "logs/*",
    "*.log",
    "logs/*",
    "*.tmp",
    "*.temp",
    # Backup files
    "*.old",
    "*.bak",
    "*.backup",
    "*.orig",
    "*.swp",
    "*~",
    # Data files
    "*.csv",
    "*.tsv",
    "*.sqlt",
    "*.dat",
    "*.xml",
    # Images (also filtered by binary detection, but skip early)
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.ico",
    "*.svg",
    "*.webp",
    "*.tiff",
    "*.tif",
    "*.psd",
    "*.ai",
    "*.eps",
    # Fonts
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.eot",
    # Audio/video
    "*.mp3",
    "*.mp4",
    "*.wav",
    "*.avi",
    "*.mov",
    "*.webm",
    # Archives
    "*.zip",
    "*.tar",
    "*.gz",
    "*.rar",
    "*.7z",
    # Database files
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.sql",
    "*.dump",
    # Python package metadata
    "*.pth",
    "py.typed",
)


def load_auggieignore(root: Path) -> list[str]:
    """
    Load exclude patterns from .auggieignore file.

    The file format is similar to .gitignore:
    - One pattern per line
    - Lines starting with # are comments
    - Empty lines are ignored
    - Patterns use glob syntax (*.sql, vendor/*, etc.)

    Args:
        root: Directory to look for .auggieignore file.

    Returns:
        List of exclude patterns (empty if file doesn't exist).

    Example .auggieignore file:
        # Exclude SQL files
        *.sql

        # Exclude generated code
        generated/*
    """
    ignore_file = root / ".auggieignore"
    if not ignore_file.is_file():
        return []

    patterns = []
    try:
        for line in ignore_file.read_text().splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    except Exception:
        return []

    return patterns


@dataclass(frozen=True)
class _GitIgnoreRule:
    base_dir: Path
    pattern: str
    negated: bool
    directory_only: bool
    anchored: bool


def _parse_gitignore_file(ignore_file: Path) -> list[_GitIgnoreRule]:
    """Parse one .gitignore file into ordered match rules."""
    try:
        lines = ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    rules: list[_GitIgnoreRule] = []
    base_dir = ignore_file.parent
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        negated = line.startswith("!")
        if negated:
            line = line[1:]

        if not line:
            continue

        anchored = line.startswith("/")
        if anchored:
            line = line[1:]

        directory_only = line.endswith("/")
        if directory_only:
            line = line.rstrip("/")

        if not line:
            continue

        rules.append(
            _GitIgnoreRule(
                base_dir=base_dir,
                pattern=line,
                negated=negated,
                directory_only=directory_only,
                anchored=anchored,
            )
        )

    return rules


def _collect_gitignore_rules(
    root: Path,
    current_dir: Path,
    cache: dict[Path, list[_GitIgnoreRule]],
) -> list[_GitIgnoreRule]:
    """Collect .gitignore rules from root through current_dir."""
    rules: list[_GitIgnoreRule] = []
    rel_dir = current_dir.relative_to(root)
    parents = [root]
    current = root
    for part in rel_dir.parts:
        current = current / part
        parents.append(current)

    for parent in parents:
        if parent in cache:
            parsed = cache[parent]
        else:
            parsed = _parse_gitignore_file(parent / ".gitignore")
            cache[parent] = parsed
        rules.extend(parsed)
    return rules


def _gitignore_rule_matches(rule: _GitIgnoreRule, path: Path, *, is_dir: bool) -> bool:
    """Return True when a parsed .gitignore rule matches a path."""
    if rule.directory_only and not is_dir:
        return False

    try:
        rel_path = path.relative_to(rule.base_dir)
    except ValueError:
        return False

    parts = rel_path.parts
    if not parts:
        return False

    rel_str = rel_path.as_posix()
    pattern = rule.pattern

    if rule.anchored or "/" in pattern:
        return fnmatch.fnmatch(rel_str, pattern)

    return any(fnmatch.fnmatch(part, pattern) for part in parts)


def _is_gitignored(path: Path, rules: Sequence[_GitIgnoreRule], *, is_dir: bool) -> bool:
    """Evaluate ordered .gitignore rules for one path."""
    ignored = False
    for rule in rules:
        if _gitignore_rule_matches(rule, path, is_dir=is_dir):
            ignored = not rule.negated
    return ignored


def _sha256_bytes(b: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.

    Args:
        b: Input bytes.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(b).hexdigest()


def _is_probably_text(b: bytes) -> bool:
    """
    Heuristic check for text files (no null bytes).

    Binary files typically contain null bytes, while text files don't.
    This is a fast heuristic that works well in practice.

    Args:
        b: File contents as bytes.

    Returns:
        True if the content appears to be text, False otherwise.
    """
    return b.find(b"\x00") == -1


def _matches_pattern(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    """
    Check if a path matches any of the exclude patterns.

    Patterns can be:
    - Simple globs: *.md, *.pyc
    - Directory globs: vendor/*, docs/*
    - Full path globs: **/test_*.py

    Args:
        path: File path to check.
        root: Root directory for relative path calculation.
        patterns: Sequence of glob patterns to match against.

    Returns:
        True if the path matches any pattern, False otherwise.
    """
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = path

    rel_str = str(rel_path)
    name = path.name

    for pattern in patterns:
        # Match against filename
        if fnmatch.fnmatch(name, pattern):
            return True
        # Match against relative path
        if fnmatch.fnmatch(rel_str, pattern):
            return True
        # Match with ** prefix for deep matching
        if "**" not in pattern and fnmatch.fnmatch(rel_str, f"**/{pattern}"):
            return True

    return False


def iter_files(
    root: Path,
    exclude: Sequence[str] = (),
    include: Sequence[str] = (),
    skip_dirs: set[str] | None = None,
) -> Iterable[Path]:
    """
    Recursively iterate over files in a directory, with filtering.

    Skips directories like .git, node_modules, .venv, etc. that typically
    contain non-source files. Supports exclude/include patterns for
    fine-grained file filtering.

    Also skips:
    - Empty files (0 bytes)
    - Duplicate symlinks (symlinks pointing to already-seen files)

    Args:
        root: Root directory to scan.
        exclude: Additional glob patterns to exclude (added to defaults).
        include: Glob patterns to include even if they match excludes.
            Use to override defaults, e.g., include=["*.md"] to index markdown.
        skip_dirs: Directory names to skip. Defaults to DEFAULT_SKIP_DIRS.

    Yields:
        Path objects for each file found.

    Example:
        >>> list(iter_files(Path("."), exclude=["test_*"]))
        >>> list(iter_files(Path("."), include=["*.md"]))  # Override default exclude
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_SKIP_DIRS

    all_excludes = list(DEFAULT_EXCLUDES) + list(exclude)

    # Track real paths to avoid duplicate symlinks
    seen_real_paths: set[Path] = set()
    gitignore_cache: dict[Path, list[_GitIgnoreRule]] = {}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current_dir = Path(dirpath)
        gitignore_rules = _collect_gitignore_rules(root, current_dir, gitignore_cache)

        # Modify dirnames in-place to skip certain directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        dirnames[:] = [
            d
            for d in dirnames
            if not _is_gitignored(current_dir / d, gitignore_rules, is_dir=True)
        ]
        for fn in filenames:
            p = current_dir / fn

            if _is_gitignored(p, gitignore_rules, is_dir=False):
                continue

            # Skip symlinks to already-seen files (dedup)
            try:
                real_path = p.resolve()
                if real_path in seen_real_paths:
                    continue
                seen_real_paths.add(real_path)

                # Skip empty files (0 bytes)
                if p.stat().st_size == 0:
                    continue
            except (OSError, FileNotFoundError):
                # Broken symlink or permission error
                continue

            # Check if explicitly included (overrides excludes)
            if include and _matches_pattern(p, root, include):
                yield p
                continue
            # Check excludes
            if all_excludes and _matches_pattern(p, root, all_excludes):
                continue
            yield p


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_total: int = 0
    chunks_reused: int = 0
    chunks_reused_global: int = 0  # Reused from other files
    chunks_reused_local: int = 0  # Reused from same file edit
    chunks_embedded: int = 0
    indexed_files: list[str] | None = None  # Paths of files that were indexed (verbose mode)
    branch: str | None = None  # Git branch that was indexed
    model: str | None = None  # Embedding model used for indexing
    ast_mode: str | None = None  # "enabled", "disabled", or "unavailable"
    ast_hint: str | None = None  # Hint message when AST is not available

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved by reusing embeddings (~100 tokens per chunk)."""
        return self.chunks_reused * 100

    @property
    def dedup_ratio(self) -> float:
        """Percentage of chunks that were deduplicated."""
        if self.chunks_total == 0:
            return 0.0
        return self.chunks_reused / self.chunks_total * 100


def _find_global_embeddings(
    con,
    lance_store: LanceStore,
    chunk_hashes: list[str],
    model: str,
    expected_dim: int,
    *,
    exclude_file_id: int | None = None,
) -> dict[str, tuple[bytes, int]]:
    """
    Find existing embeddings across ALL files for given chunk hashes.

    Performs integrity checks:
    - Model must match
    - Embedding dimension must match expected

    Args:
        con: Database connection.
        chunk_hashes: List of text_sha256 hashes to look up.
        model: Required embedding model name.
        expected_dim: Expected embedding dimensions for this model.

    Returns:
        Dict mapping text_sha256 -> (embedding_bytes, dim).
    """
    if not chunk_hashes:
        return {}

    # Batch query with model filter
    placeholders = ",".join("?" * len(chunk_hashes))
    sql = f"""SELECT DISTINCT id, text_sha256, dim
            FROM chunks
            WHERE text_sha256 IN ({placeholders})
              AND model = ?
        """
    params: list[object] = [*chunk_hashes, model]
    if exclude_file_id is not None:
        sql += " AND file_id != ?"
        params.append(exclude_file_id)
    rows = con.execute(sql, params).fetchall()

    vector_rows = lance_store.fetch_vectors_by_chunk_ids([int(row[0]) for row in rows])
    result = {}
    for chunk_id, text_sha256, dim in rows:
        if dim != expected_dim:
            continue
        if int(chunk_id) not in vector_rows:
            continue
        result[text_sha256] = vector_rows[int(chunk_id)]

    return result


def _check_model_consistency(con, model: str) -> None:
    """
    Verify the requested model matches the index's existing model.

    Args:
        con: Database connection.
        model: Requested embedding model name.

    Raises:
        ValueError: If index uses a different model.
    """
    existing_model_row = con.execute("SELECT DISTINCT model FROM chunks LIMIT 1").fetchone()
    if existing_model_row and existing_model_row[0] != model:
        raise ValueError(
            f"Model mismatch: index uses '{existing_model_row[0]}' "
            f"but requested '{model}'. "
            f"Use --force to reindex with new model."
        )


def _get_expected_dimension(con, model: str, dimensions: int | None) -> int | None:
    """
    Determine expected embedding dimension for the model.

    Priority:
    1. Explicit dimensions argument
    2. Most common dimension in existing chunks for this model
    3. Model definition default
    4. None (learn from first embed)

    Args:
        con: Database connection.
        model: Embedding model name.
        dimensions: Explicit dimensions override (or None).

    Returns:
        Expected dimension, or None if unknown.
    """
    if dimensions is not None:
        return dimensions

    # Check what dimension the majority of existing chunks use
    existing_dim_row = con.execute(
        """SELECT dim, COUNT(*) as cnt FROM chunks
           WHERE model = ?
           GROUP BY dim
           ORDER BY cnt DESC, dim ASC
           LIMIT 1""",
        (model,),
    ).fetchone()
    if existing_dim_row:
        return existing_dim_row[0]

    # Fall back to model definition
    try:
        model_info = get_model(model)
        return model_info.dimensions
    except KeyError:
        # Unknown model - will learn from first embed
        return None


def _read_file_if_indexable(
    path: Path,
    max_bytes: int,
    detection_results: dict,
    stats: IndexStats,
) -> bytes | None:
    """
    Read file contents if the file is indexable.

    Checks:
    - File exists
    - Not too large
    - Not binary (no null bytes)
    - Passes MIME detection (if enabled)

    Args:
        path: File to read.
        max_bytes: Maximum file size.
        detection_results: MIME detection results (may be empty).
        stats: IndexStats to update on skip.

    Returns:
        File contents as bytes, or None if file should be skipped.
    """
    if not path.is_file():
        return None

    try:
        st = path.stat()
    except FileNotFoundError:
        return None

    # Skip large files
    if st.st_size > max_bytes:
        stats.files_skipped += 1
        return None

    # Read file contents
    try:
        b = path.read_bytes()
    except Exception:
        stats.files_skipped += 1
        return None

    # Skip binary files (null-byte check)
    if not _is_probably_text(b):
        stats.files_skipped += 1
        return None

    # Skip files that failed MIME type detection
    if path in detection_results and not detection_results[path].is_text:
        stats.files_skipped += 1
        return None

    return b


def _cache_existing_embeddings(
    con, lance_store: LanceStore, file_id: int
) -> dict[str, tuple[bytes, int]]:
    """
    Cache embeddings from existing file chunks before deletion.

    Args:
        con: Database connection.
        file_id: File ID to cache embeddings for.

    Returns:
        Dict mapping text_sha256 -> (embedding_bytes, dim).
    """
    rows = con.execute(
        "SELECT id, text_sha256, dim FROM chunks WHERE file_id=?",
        (file_id,),
    ).fetchall()
    chunk_ids = [int(row[0]) for row in rows]
    vectors = lance_store.fetch_vectors_by_chunk_ids(chunk_ids)

    embeddings = {}
    for chunk_id, text_sha256, dim in rows:
        vector = vectors.get(int(chunk_id))
        if vector is None:
            continue
        embeddings[str(text_sha256)] = (vector[0], int(dim))
    return embeddings


def _plan_chunk_diff(
    chunks,
    chunk_hashes: list[str],
    existing_chunks: list[tuple[int, int, int, int, str, str, int, str]],
    existing_embeddings: dict[str, tuple[bytes, int]],
    global_embeddings: dict[str, tuple[bytes, int]],
    stats: IndexStats,
) -> tuple[
    dict[int, tuple[int, int]],
    list[tuple[int, bytes, int]],
    list[tuple[int, str]],
    list[int],
]:
    """
    Plan which chunks to keep, reuse, embed, and delete.

    Returns:
        Tuple of (preserved_rows, reusable_new_rows, chunks_to_embed, removed_chunk_ids).
        preserved_rows maps new chunk index -> (existing chunk_id, dim).
    """
    preserved_rows: dict[int, tuple[int, int]] = {}
    reusable_new_rows: list[tuple[int, bytes, int]] = []
    chunks_to_embed: list[tuple[int, str]] = []
    used_old_ids: set[int] = set()

    exact_matches = {
        (chunk_index, text_sha256): (chunk_id, dim)
        for chunk_id, chunk_index, _sl, _el, _text, text_sha256, dim, _model in existing_chunks
        if text_sha256 in existing_embeddings
    }

    for i, c in enumerate(chunks):
        match = exact_matches.get((c.chunk_index, chunk_hashes[i]))
        if match is None:
            continue
        chunk_id, dim = match
        preserved_rows[i] = (chunk_id, dim)
        used_old_ids.add(chunk_id)
        stats.chunks_reused += 1
        stats.chunks_reused_local += 1

    local_candidates: dict[str, list[tuple[int, bytes, int]]] = defaultdict(list)
    for chunk_id, _chunk_index, _sl, _el, _text, text_sha256, dim, _model in existing_chunks:
        if chunk_id in used_old_ids:
            continue
        embedding = existing_embeddings.get(text_sha256)
        if embedding is None:
            continue
        local_candidates[text_sha256].append((chunk_id, embedding[0], dim))

    for i, tsha in enumerate(chunk_hashes):
        if i in preserved_rows:
            continue
        if local_candidates[tsha]:
            old_chunk_id, emb, dim = local_candidates[tsha].pop(0)
            used_old_ids.add(old_chunk_id)
            reusable_new_rows.append((i, emb, dim))
            stats.chunks_reused += 1
            stats.chunks_reused_local += 1
        elif tsha in global_embeddings:
            emb, dim = global_embeddings[tsha]
            reusable_new_rows.append((i, emb, dim))
            stats.chunks_reused += 1
            stats.chunks_reused_global += 1
        else:
            chunks_to_embed.append((i, chunks[i].text.replace("\r\n", "\n")))
            stats.chunks_embedded += 1

    removed_chunk_ids = [chunk_id for chunk_id, *_rest in existing_chunks if chunk_id not in used_old_ids]
    return preserved_rows, reusable_new_rows, chunks_to_embed, removed_chunk_ids


def _upsert_file_record(
    con,
    rel_path: str,
    mtime_ns: int,
    size: int,
    sha: str,
    branch: str,
    existing_row: tuple | None,
) -> int:
    """
    Insert or update file record in database for a specific branch.

    Args:
        con: Database connection.
        rel_path: Resolved file path.
        mtime_ns: Modification time in nanoseconds.
        size: File size in bytes.
        sha: SHA-256 hash of file contents.
        branch: Git branch name for scoping.
        existing_row: Existing row from files table (or None).

    Returns:
        File ID.
    """
    if existing_row:
        file_id = int(existing_row[0])
        con.execute(
            "UPDATE files SET mtime_ns=?, size=?, sha256=? WHERE id=?",
            (mtime_ns, size, sha, file_id),
        )
    else:
        cur = con.execute(
            "INSERT INTO files(path, branch, mtime_ns, size, sha256) VALUES(?,?,?,?,?)",
            (rel_path, branch, mtime_ns, size, sha),
        )
        file_id = int(cur.lastrowid)
    return file_id


def _compute_chunk_hashes(chunks) -> list[str]:
    """
    Compute SHA-256 hashes for all chunks.

    Args:
        chunks: List of Chunk objects.

    Returns:
        Chunk SHA-256 hashes.
    """
    chunk_hashes = []
    for c in chunks:
        normalized_text = c.text.replace("\r\n", "\n")
        tsha = hashlib.sha256(normalized_text.encode("utf-8", errors="ignore")).hexdigest()
        chunk_hashes.append(tsha)
    return chunk_hashes


def _store_chunk_metadata(
    con,
    file_id: int,
    chunks,
    chunk_hashes: list[str],
    preserved_rows: dict[int, tuple[int, int]],
    new_embeddings: dict[int, tuple[bytes, int]],
    reusable_new_rows: list[tuple[int, bytes, int]],
    model: str,
) -> list[tuple[int, bytes, int]]:
    """
    Store chunk metadata in SQLite and return vector rows for Lance.

    Args:
        con: Database connection.
        file_id: File ID to associate chunks with.
        chunks: List of Chunk objects.
        chunk_hashes: List of text_sha256 hashes.
        preserved_rows: Dict mapping index -> (existing chunk_id, dim) to keep in place.
        new_embeddings: Dict mapping index -> (embedding, dim) for new chunks.
        reusable_new_rows: List of (index, embedding, dim) for reused chunks that need new rows.
        model: Embedding model name.
    """
    reusable_lookup = {idx: (emb, dim) for idx, emb, dim in reusable_new_rows}
    vector_rows: list[tuple[int, bytes, int]] = []
    for i, c in enumerate(chunks):
        tsha = chunk_hashes[i]
        if i in preserved_rows:
            chunk_id, dim = preserved_rows[i]
            con.execute(
                """
                UPDATE chunks
                SET chunk_index = ?, start_line = ?, end_line = ?, text = ?, text_sha256 = ?, dim = ?, model = ?
                WHERE id = ? AND file_id = ?
                """,
                (
                    c.chunk_index,
                    c.start_line,
                    c.end_line,
                    c.text,
                    tsha,
                    dim,
                    model,
                    chunk_id,
                    file_id,
                ),
            )
            continue
        if i in new_embeddings:
            emb, dim = new_embeddings[i]
        else:
            emb, dim = reusable_lookup[i]

        cur = con.execute(
            """INSERT INTO chunks(file_id, chunk_index, start_line, end_line,
               text, text_sha256, dim, model)
               VALUES(?,?,?,?,?,?,?,?)""",
            (file_id, c.chunk_index, c.start_line, c.end_line, c.text, tsha, dim, model),
        )
        chunk_id = int(cur.lastrowid)
        vector_rows.append((chunk_id, emb, dim))
    return vector_rows


def index_path(
    root: Path,
    db_path: Path,
    model: str | None = None,
    dimensions: int | None = None,
    chunk_lines: int = 60,
    overlap: int = 10,
    max_bytes: int = 2_000_000,
    exclude: Sequence[str] = (),
    include: Sequence[str] = (),
    detect: bool = True,
    verbose: bool = False,
    ast: bool = False,
    branch: str | None = None,
) -> IndexStats:
    """
    Index a directory for semantic search.

    Scans all files under root, chunks text files, generates embeddings,
    and stores them in the database. Supports incremental updates by
    checking file modification time, size, and content hash.

    Smart embedding reuse: When a file changes, existing chunk embeddings
    are reused if the chunk text hasn't changed (matched by text_sha256).
    This saves API tokens for common edit patterns like appending code.

    By default, excludes common non-source files (docs, config, build outputs).
    Use --include to override specific excludes. Additional patterns can be
    specified in a .auggieignore file in the root directory.

    Args:
        root: Directory to index.
        db_path: Path to the SQLite database file.
        model: Voyage embedding model name or alias (None for default/env).
        dimensions: Embedding dimensions (None for model default).
        chunk_lines: Number of lines per chunk.
        overlap: Number of overlapping lines between chunks.
        max_bytes: Maximum file size to index (larger files are skipped).
        exclude: Additional glob patterns to exclude.
        include: Glob patterns to include (overrides default excludes).
        detect: Use file command for MIME type detection (default True).
        verbose: Track and return paths of indexed files (default False).
        ast: Use AST-aware chunking for semantic boundaries (default False).
            Requires: pip install tree-sitter-<language>  (for example: pip install tree-sitter-python)
        branch: Git branch name for scoping (None for auto-detect).
            Files are indexed per-branch, enabling branch-aware search.

    Returns:
        IndexStats with counts of files/chunks processed and reused.

    Note:
        Files are skipped if:
        - They match an exclude pattern (unless overridden by include)
        - They exceed max_bytes in size
        - They appear to be binary (contain null bytes)
        - They fail MIME type detection (if detect=True)
        - They haven't changed since last indexing (same mtime, size, hash)

    Example:
        >>> stats = index_path(
        ...     root=Path("."),
        ...     db_path=Path(".auggie/index.sqlite"),
        ... )
        >>> print(f"Reused {stats.chunks_reused} chunks")
    """
    # Resolve model from arg, env, or default
    model = resolve_model(model)
    stats = IndexStats(indexed_files=[] if verbose else None, model=model)

    # Resolve branch - auto-detect if not specified
    if branch is None:
        from .commands._common import get_current_branch

        branch = get_current_branch(root)

    # Initialize AST chunker
    # ast=None: auto-detect (use AST when available, silent fallback)
    # ast=False: explicitly disabled via --no-ast
    # ast=True: explicitly requested (legacy, warn if unavailable)
    ast_chunker = None
    if ast is False:
        # Explicitly disabled
        stats.ast_mode = "disabled"
    else:
        # Auto-detect (ast=None) or explicitly requested (ast=True)
        ast_chunker = _get_ast_chunker()
        if ast_chunker is not None:
            stats.ast_mode = "enabled"
        elif ast is True:
            # Explicitly requested but unavailable - warn
            import sys

            print(
                "Warning: AST chunking requested but tree-sitter not available.\n"
                "Install with: pip install tree-sitter-<language>\n"
                "Falling back to line-based chunking.",
                file=sys.stderr,
            )
            stats.ast_mode = "unavailable"
            stats.ast_hint = "Install AST support: pip install tree-sitter-<language>"
        else:
            # Auto-detect but unavailable - silent fallback with hint
            stats.ast_mode = "unavailable"
            stats.ast_hint = "AST chunking not available. Install: pip install tree-sitter-<language>"

    # Load .auggieignore patterns and combine with CLI excludes
    ignore_patterns = load_auggieignore(root)
    all_exclude = list(exclude) + ignore_patterns

    con = connect(db_path)
    lance_store = LanceStore(get_lance_dir(db_path))

    # Model consistency check - prevent mixing models in the same index
    _check_model_consistency(con, model)

    # Store AST mode in metadata (tracks how index was built)
    from .db import set_metadata

    ast_mode_effective = "true" if (ast is not False and ast_chunker is not None) else "false"
    set_metadata(con, "ast_mode", ast_mode_effective)

    # Get expected dimensions
    expected_dim = _get_expected_dimension(con, model, dimensions)
    file_states = get_file_states(con, branch)
    had_existing_index = bool(file_states)
    seen_paths: set[str] = set()
    refresh_touched = 0
    refresh_deleted = 0
    refresh_deleted_chunks = 0

    # ── Discovery phase: scan filesystem for candidate files ──
    files = list(
        tqdm(
            iter_files(root, exclude=all_exclude, include=include),
            desc="Scanning",
            unit=" files",
            leave=False,
        )
    )
    stats.files_scanned = len(files)

    # ── Detection phase: check file types via MIME ──
    detection_results = {}
    if detect and has_file_command() and files:
        with tqdm(total=len(files), desc="Detecting", unit=" files", leave=False) as pbar:
            detection_results = detect_file_types_batch(files, progress_callback=pbar.update)

    for p in tqdm(files, desc="Indexing"):
        st = p.stat()
        rel = str(p.resolve())
        existing_state = file_states.get(rel)

        if (
            existing_state is not None
            and existing_state[1] == st.st_mtime_ns
            and existing_state[2] == st.st_size
        ):
            seen_paths.add(rel)
            stats.files_skipped += 1
            continue

        # ── Phase 1: File validation and confirm gate ──
        b = _read_file_if_indexable(p, max_bytes, detection_results, stats)
        if b is None:
            continue
        seen_paths.add(rel)

        sha = _sha256_bytes(b)
        if existing_state is not None and existing_state[3] == sha:
            try:
                con.execute("BEGIN IMMEDIATE")
                touch_file_state(con, existing_state[0], st.st_mtime_ns, st.st_size)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            refresh_touched += 1
            stats.files_skipped += 1
            continue

        # ── Phase 2: Prepare for indexing (read-only DB ops) ──
        stats.files_indexed += 1
        if stats.indexed_files is not None:
            try:
                stats.indexed_files.append(str(p.relative_to(root)))
            except ValueError:
                stats.indexed_files.append(str(p))

        existing_file_id = existing_state[0] if existing_state is not None else None
        existing_chunks: list[tuple[int, int, int, int, str, str, int, str]] = []
        existing_embeddings: dict[str, tuple[bytes, int]] = {}
        if existing_file_id is not None:
            existing_chunks = get_file_chunks(con, existing_file_id)
            existing_embeddings = _cache_existing_embeddings(con, lance_store, existing_file_id)

        # ── Phase 3: Chunk text and compute hashes ──
        text = b.decode("utf-8", errors="ignore")

        # Use AST chunking if available, with fallback to line-based
        if ast_chunker is not None:
            chunks = ast_chunker(
                text,
                filename=str(p),
                max_chunk_lines=chunk_lines,
            )
            # Fall back to line-based if AST produces no chunks (unsupported language)
            if not chunks:
                chunks = chunk_text(text, chunk_size=chunk_lines, overlap=overlap)
        else:
            chunks = chunk_text(text, chunk_size=chunk_lines, overlap=overlap)

        if not chunks:
            continue

        stats.chunks_total += len(chunks)
        chunk_hashes = _compute_chunk_hashes(chunks)

        # ── Phase 4: Find reusable embeddings (read-only DB ops) ──
        global_embeddings: dict[str, tuple[bytes, int]] = {}
        if expected_dim is not None:
            global_embeddings = _find_global_embeddings(
                con,
                lance_store,
                chunk_hashes,
                model,
                expected_dim,
                exclude_file_id=existing_file_id,
            )

        preserved_rows, reusable_new_rows, chunks_to_embed, removed_chunk_ids = _plan_chunk_diff(
            chunks,
            chunk_hashes,
            existing_chunks,
            existing_embeddings,
            global_embeddings,
            stats,
        )

        # ── Phase 5: Generate new embeddings (API call, outside transaction) ──
        new_embeddings: dict[int, tuple[bytes, int]] = {}
        if chunks_to_embed:
            texts = [t for _, t in chunks_to_embed]
            emb_blobs, dim = embed_texts(texts, model=model, dimensions=dimensions)
            for (idx, _), emb in zip(chunks_to_embed, emb_blobs, strict=True):
                new_embeddings[idx] = (emb, dim)
            if expected_dim is None:
                expected_dim = dim

        # ── Phase 6: Atomic DB write (transaction) ──
        new_chunk_ids: list[int] = []
        try:
            con.execute("BEGIN IMMEDIATE")
            file_id = _upsert_file_record(
                con,
                rel,
                st.st_mtime_ns,
                st.st_size,
                sha,
                branch,
                (existing_file_id,) if existing_file_id is not None else None,
            )
            if removed_chunk_ids:
                placeholders = ",".join("?" * len(removed_chunk_ids))
                con.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", removed_chunk_ids)
            # Vacate the positive chunk_index space for this file before applying the
            # new layout. Preserved rows are renumbered in place and new rows are
            # inserted; doing that without first moving surviving rows out of the
            # target range transiently violates UNIQUE(file_id, chunk_index) when an
            # edit shifts chunk boundaries.
            if existing_file_id is not None:
                con.execute(
                    "UPDATE chunks SET chunk_index = -(chunk_index + 1) WHERE file_id = ?",
                    (file_id,),
                )
            vector_rows = _store_chunk_metadata(
                con,
                file_id,
                chunks,
                chunk_hashes,
                preserved_rows,
                new_embeddings,
                reusable_new_rows,
                model,
            )
            new_chunk_ids = [chunk_id for chunk_id, _emb, _dim in vector_rows]
            # Remove any stale rows left in the vacated negative space. These are
            # old chunks whose vectors were reused under a new chunk_id (reuse-as-
            # new-row), so they are now redundant duplicates.
            stale_chunk_ids: list[int] = []
            if existing_file_id is not None:
                stale_chunk_ids = [
                    int(r[0])
                    for r in con.execute(
                        "SELECT id FROM chunks WHERE file_id = ? AND chunk_index < 0",
                        (file_id,),
                    ).fetchall()
                ]
                if stale_chunk_ids:
                    placeholders = ",".join("?" * len(stale_chunk_ids))
                    con.execute(
                        f"DELETE FROM chunks WHERE id IN ({placeholders})", stale_chunk_ids
                    )
            lance_store.upsert_chunk_vectors(vector_rows, model=model)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            if new_chunk_ids:
                lance_store.delete_chunk_ids(new_chunk_ids)
            raise
        if removed_chunk_ids:
            lance_store.delete_chunk_ids(removed_chunk_ids)
        if stale_chunk_ids:
            lance_store.delete_chunk_ids(stale_chunk_ids)

    deleted_paths = sorted(set(file_states) - seen_paths)
    for deleted_path in deleted_paths:
        file_id, _mtime_ns, _size, _sha = file_states[deleted_path]
        existing_chunks = get_file_chunks(con, file_id)
        removed_chunk_ids = [chunk_id for chunk_id, *_rest in existing_chunks]
        try:
            con.execute("BEGIN IMMEDIATE")
            delete_file_record(con, file_id)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        if removed_chunk_ids:
            lance_store.delete_chunk_ids(removed_chunk_ids)
        refresh_deleted += 1
        refresh_deleted_chunks += len(removed_chunk_ids)

    # ── Phase 7: Log history entry (AI tool integration) ──
    refresh_files_affected = stats.files_indexed + refresh_touched + refresh_deleted
    refresh_chunks_affected = stats.chunks_embedded + stats.chunks_reused + refresh_deleted_chunks
    if had_existing_index:
        log_history(
            con,
            action="refresh",
            files_affected=refresh_files_affected,
            chunks_affected=refresh_chunks_affected,
            details={
                "files_scanned": stats.files_scanned,
                "files_indexed": stats.files_indexed,
                "files_touched": refresh_touched,
                "files_deleted": refresh_deleted,
                "files_skipped": stats.files_skipped,
                "chunks_embedded": stats.chunks_embedded,
                "chunks_reused": stats.chunks_reused,
                "indexed_files": stats.indexed_files,
                "branch": branch,
                "model": model,
            },
        )
    elif stats.files_indexed > 0:
        log_history(
            con,
            action="index",
            files_affected=stats.files_indexed,
            chunks_affected=stats.chunks_total,
            details={
                "files_scanned": stats.files_scanned,
                "files_skipped": stats.files_skipped,
                "chunks_embedded": stats.chunks_embedded,
                "chunks_reused": stats.chunks_reused,
                "indexed_files": stats.indexed_files,
                "branch": branch,
                "model": model,
            },
        )

    if stats.files_indexed > 0 or refresh_deleted > 0:
        # Increment db_version to invalidate L2 cache
        from .cache import increment_db_version

        increment_db_version(con)

    # Store current branch in metadata
    set_metadata(con, "current_branch", branch)

    # Set branch in stats for reporting
    stats.branch = branch

    con.close()
    return stats
